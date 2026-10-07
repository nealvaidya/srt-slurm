# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real local sockets and guards; only the Slurm transport is replaced."""

import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from test_services import _load, _orchestrator

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.core.job_ports import JobPortPlan, PortConflict, PortRequest
from srtctl.core.port_reservation import PortLeaseManager
from srtctl.core.processes import ProcessRegistry
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.mock import FakePopen
from srtctl.ports import HTTP_PORTS, VLLM_SCAN_PORTS
from srtctl.services import get_service_kind
from srtctl.services.implicit import effective_services


@pytest.fixture
def local_slurm(tmp_path):
    children = []

    def launch(**kwargs):
        command = kwargs["command"]
        directory = tmp_path / "node-leases" / kwargs["nodelist"][0]
        directory.parent.mkdir(parents=True, exist_ok=True)
        with Path(kwargs["output"]).open("w") as output:
            child = subprocess.Popen(
                [sys.executable, *command[1:], "--directory", str(directory)],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        children.append(child)
        return child

    with (
        patch("srtctl.core.port_reservation.start_srun_process", side_effect=launch),
        patch("srtctl.core.processes.signal_step", return_value=False),
    ):
        yield children
    for child in children:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def runtime(tmp_path, plan, job="123"):
    logs = tmp_path / job
    logs.mkdir(exist_ok=True)
    return RuntimeContext(
        job_id=job,
        run_name="ports-test",
        nodes=Nodes(head="node0", infra="node0", bench="node0", worker=("node0", "node1")),
        head_node_ip="127.0.0.1",
        infra_node_ip="127.0.0.1",
        log_dir=logs,
        model_path=Path("/model"),
        container_image=Path("/job.sqsh"),
        gpus_per_node=8,
        network_interface=None,
        job_ports=plan,
        frontend_port=plan.fixed("frontend"),
    )


def test_unused_exporter_and_scan_hint_do_not_block_startup(tmp_path, local_slurm):
    plan = JobPortPlan(1)
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", plan.fixed("node-exporter")))
        occupied.listen()
        # A hint is recorded honestly without claiming its engine obeys a bound.
        plan.record(PortRequest("scan", "workers", "node1", occupied.getsockname()[1], enforced=False))
        manager = PortLeaseManager(runtime(tmp_path, plan))
        manager.acquire(ProcessRegistry(job_id="123"))
        assert all(proc.is_running for proc in manager.processes)
        manager.close()


def test_conflict_on_later_node_releases_the_entire_partial_lease(tmp_path, local_slurm):
    plan = JobPortPlan(2)
    port = plan.request("node-exporter", "node1", owner="node-exporter")
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", port))
        occupied.listen()
        manager = PortLeaseManager(runtime(tmp_path, plan))
        with pytest.raises(PortConflict, match="node1.*node-exporter"):
            manager.acquire(ProcessRegistry(job_id="123"))
        assert all(not proc.is_running for proc in manager.processes)
        occupied.getsockname()  # The unrelated listener is still open.
    next_manager = PortLeaseManager(runtime(tmp_path, JobPortPlan(2), "138"))
    next_manager.acquire(ProcessRegistry(job_id="123"))
    next_manager.close()


def test_same_slot_is_leased_on_every_node_until_cleanup(tmp_path, local_slurm):
    first = PortLeaseManager(runtime(tmp_path, JobPortPlan(3)))
    first.acquire(ProcessRegistry(job_id="123"))
    second = PortLeaseManager(runtime(tmp_path, JobPortPlan(3), "138"))
    with pytest.raises(PortConflict, match="job 123"):
        second.acquire(ProcessRegistry(job_id="123"))
    assert all(proc.is_running for proc in first.processes)
    first.close()
    third = PortLeaseManager(runtime(tmp_path, JobPortPlan(3), "153"))
    third.acquire(ProcessRegistry(job_id="123"))
    third.close()


def test_exited_guard_cannot_supply_stale_ready_status(tmp_path, local_slurm):
    manager = PortLeaseManager(runtime(tmp_path, JobPortPlan(3)))
    manager.acquire(ProcessRegistry(job_id="123"))
    manager.processes[0].popen.terminate()
    manager.processes[0].popen.wait(timeout=5)
    with pytest.raises(RuntimeError, match="port guard exited on node0"):
        manager._wait()
    manager.close()


def test_new_service_request_conflict_reassigns_and_updates_manifest(tmp_path, local_slurm):
    plan = JobPortPlan(4)
    manager = PortLeaseManager(runtime(tmp_path, plan))
    manager.acquire(ProcessRegistry(job_id="123"))
    old = plan.fixed("etcd-client")
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", old))
        occupied.listen()
        with pytest.raises(PortConflict, match="etcd"):
            plan.request("etcd-client", "node1", owner="etcd")
        assert plan.reassign("etcd")
        new = plan.request("etcd-client", "node1", owner="etcd")
        assert new != old
        # A running service may keep its port bound during subsequent revisions.
        with socket.socket() as own_listener:
            own_listener.bind(("0.0.0.0", new))
            own_listener.listen()
            plan.request("nats", "node0", owner="nats")
        manifest = json.loads((manager.runtime.log_dir / "port_plan.json").read_text())
        assert manifest["fixed"]["etcd-client"] == new
        assert next(r for r in manifest["requests"] if r["name"] == "etcd-client")["node"] == "node1"
    manager.close()


def test_allocation_retries_slots_and_rebuilds_worker_and_frontend_ports(tmp_path, local_slurm, monkeypatch):
    monkeypatch.delenv("SRTCTL_PORT_SLOT", raising=False)
    config = SrtConfig.Schema().load(
        {
            "schema": 2,
            "name": "ports",
            "job_scoped_ports": True,
            "model": {"path": "/model", "container": "/job.sqsh", "precision": "bf16"},
            "resources": {"gpus_per_node": 8},
            "engine": {"type": "vllm", "connector": None},
            "frontend": {"type": "dynamo"},
            "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 1}},
            "benchmark": {"type": "manual"},
            "observability": {"tachometer": {"enabled": False}},
        }
    )
    first = JobPortPlan.from_job_id("123")
    initial_runtime = runtime(tmp_path, first)
    orchestrator = SweepOrchestrator(config, initial_runtime)
    old_process_ports = [p.sys_port for p in orchestrator.backend_processes]
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", first.fixed("frontend")))
        occupied.listen()
        orchestrator._reserve_job_ports(ProcessRegistry(job_id="123"))
    selected = orchestrator.runtime.job_ports
    assert selected.slot != first.slot
    assert orchestrator.runtime.frontend_port == selected.fixed("frontend")
    assert [p.sys_port for p in orchestrator.backend_processes] != old_process_ports
    orchestrator.port_leases.close()


def test_explicit_slot_and_range_bounds_are_respected(monkeypatch):
    monkeypatch.setenv("SRTCTL_PORT_SLOT", "7")
    assert [plan.slot for plan in JobPortPlan.candidates("123")] == [7]
    plan = JobPortPlan(7)
    allocator = plan.allocator()
    allocator.next(HTTP_PORTS, "node0")
    allocator.next(VLLM_SCAN_PORTS)
    requests = list(plan.requests.values())
    assert requests[0].size == HTTP_PORTS.span
    assert not requests[1].enforced
    for _ in range(9):
        allocator.next(VLLM_SCAN_PORTS)
    with pytest.raises(ValueError, match="range exhausted"):
        allocator.next(VLLM_SCAN_PORTS)


def test_concurrent_jobs_with_same_preference_get_distinct_slots_and_release_them(tmp_path, local_slurm, monkeypatch):
    monkeypatch.delenv("SRTCTL_PORT_SLOT", raising=False)

    def allocate(job):
        for plan in JobPortPlan.candidates(job):
            selected = runtime(tmp_path, plan, job)
            selected = replace(selected, nodes=Nodes(head="node0", infra="node0", bench="node0", worker=("node0",)))
            port = plan.request("frontend", "node0", owner="frontend")
            manager = PortLeaseManager(selected)
            try:
                manager.acquire(ProcessRegistry(job_id=job))
            except PortConflict:
                continue
            listener = socket.socket()
            listener.bind(("0.0.0.0", port))
            listener.listen()
            return manager, listener
        pytest.fail("local allocation exhausted")

    # All five jobs prefer slot 1 and race to acquire it on the same node.
    with ThreadPoolExecutor(max_workers=5) as pool:
        allocated = list(pool.map(allocate, [str(15000 + 15 * i) for i in range(5)]))
    assert len({manager.plan.slot for manager, _ in allocated}) == 5
    assert len({listener.getsockname()[1] for _, listener in allocated}) == 5
    released, listener = allocated.pop()
    listener.close()
    released.close()
    replacement, listener = allocate("16005")
    assert replacement.plan.slot == released.plan.slot
    listener.close()
    replacement.close()
    for manager, listener in allocated:
        listener.close()
        manager.close()


@pytest.mark.parametrize("failure", ["bind: address already in use", "CUDA out of memory", "bind-always"])
def test_service_startup_retries_only_bind_conflicts_and_restarts_whole_fleet(tmp_path, failure):
    config = _load("""
services:
  - name: host-metrics
    type: node-exporter
    placement: {node: workers}
""")
    orchestrator = _orchestrator(config, tmp_path)
    plan = JobPortPlan(8)
    orchestrator.runtime = replace(orchestrator.runtime, job_ports=plan)
    launches = []
    children = []

    def launch(**kwargs):
        proc = FakePopen(cmd=kwargs["command"], output=kwargs["output"], duration_s=3600)
        launches.append(kwargs)
        children.append(proc)
        if len(launches) == 2 or failure == "bind-always":
            proc._returncode = 1
            Path(kwargs["output"]).write_text("bind: address already in use" if failure == "bind-always" else failure)
        return proc

    def wait(proc, service, ctx):
        readiness = get_service_kind(service.type).readiness(service, ctx)
        assert readiness.probe_port == plan.fixed(service.name)
        if proc.exit_code == 1:
            raise RuntimeError("service exited before readiness")

    with (
        patch("srtctl.cli.mixins.service_stage.start_srun_process", side_effect=launch),
        patch("srtctl.cli.mixins.service_stage.get_hostname_ip", return_value="127.0.0.1"),
        patch.object(orchestrator, "_wait_service_ready", side_effect=wait),
    ):
        if "address already in use" in failure:
            result = orchestrator.start_services("after_frontend", ProcessRegistry(job_id="123"))
            assert len(result) == 3
            assert len(launches) == 5  # two failed-attempt instances, then the three-node fleet
            assert children[0].poll() is not None
            old = next(arg for arg in launches[0]["command"] if "listen-address" in arg)
            new = next(arg for arg in launches[2]["command"] if "listen-address" in arg)
            assert old != new
            assert all(new in call["command"] for call in launches[2:])
            effective = next(
                entry.service
                for entry in effective_services(config, orchestrator.runtime)
                if entry.service.name == "host-metrics"
            )
            metrics = get_service_kind(effective.type).metrics(effective)
            assert metrics[0].port == plan.fixed("host-metrics")
            assert list(tmp_path.glob("*.port-attempt-1.out"))
            assert any(proc.is_running for proc in result)
        else:
            with pytest.raises(RuntimeError, match="service exited before readiness"):
                orchestrator.start_services("after_frontend", ProcessRegistry(job_id="123"))
            assert len(launches) == (4 if failure == "bind-always" else 2)

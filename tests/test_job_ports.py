# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from srtctl.core.job_ports import SLOT_COUNT, JobPortPlan
from srtctl.ports import HTTP_PORTS


def test_all_slots_and_listener_families_have_disjoint_reservations():
    reserved = set()
    for slot in range(1, SLOT_COUNT):
        plan = JobPortPlan(slot)
        listeners = set(plan.to_dict()["fixed"].values())
        for base, size in plan.worker_ranges().values():
            block = set(range(base, base + size))
            assert not listeners.intersection(block)
            listeners.update(block)
        assert not reserved.intersection(listeners)
        assert max(listeners) < 49151
        reserved.update(listeners)


def test_neighbor_jobs_separate_ports_and_never_select_legacy_slot():
    with patch.dict("os.environ", {}, clear=True):
        for job in range(100, 160):
            left = JobPortPlan.from_job_id(str(job))
            right = JobPortPlan.from_job_id(str(job + 1))
            assert left.slot != right.slot
            assert left.fixed("frontend") != right.fixed("frontend")


def test_range_exhaustion_fails_instead_of_entering_the_next_jobs_slot():
    allocator = JobPortPlan(1).allocator()
    for _ in range(16):
        allocator.next(HTTP_PORTS, "node0")
    with pytest.raises(ValueError, match="range exhausted"):
        allocator.next(HTTP_PORTS, "node0")


def test_slot_override_and_invalid_ids_are_explicit():
    with patch.dict("os.environ", {"SRTCTL_PORT_SLOT": "3"}):
        assert JobPortPlan.from_job_id("123").slot == 3
    with patch.dict("os.environ", {"SRTCTL_PORT_SLOT": "0"}), pytest.raises(ValueError):
        JobPortPlan.from_job_id("123")
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError):
        JobPortPlan.from_job_id("unknown")


def test_slot_lease_rejects_another_process_and_releases_on_failure(tmp_path):
    directory = tmp_path / "leases"
    plan = JobPortPlan(1)
    code = (
        "from pathlib import Path; from srtctl.core.job_ports import JobPortPlan; "
        f"ctx=JobPortPlan(1).lease('115', directory=Path({str(directory)!r})); ctx.__enter__()"
    )
    with pytest.raises(ValueError, match="test failure"), plan.lease("100", directory=directory):
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        assert child.returncode != 0
        assert "already leased by job 100" in child.stderr
        with JobPortPlan(2).lease("101", directory=directory):
            pass
        raise ValueError("test failure")
    # Retained inode is safe to reuse, with no stale-job ownership after exit.
    with plan.lease("115", directory=directory):
        assert (directory / "slot-1.lock").read_text() == "job 115\n"


def test_slot_lease_rejects_occupied_infrastructure_port_and_releases_lock(tmp_path):
    plan = JobPortPlan(1)
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with (
            patch("srtctl.core.job_ports.FIXED_KINDS", ("frontend",)),
            patch.object(JobPortPlan, "fixed", return_value=port),
            pytest.raises(RuntimeError, match="frontend port .* unavailable"),
            plan.lease("100", directory=tmp_path / "leases"),
        ):
            pytest.fail("occupied listener must fail before startup")
    with plan.lease("100", directory=tmp_path / "leases"):
        pass


def test_slot_lease_rejects_nonprivate_directory(tmp_path):
    directory = tmp_path / "public"
    directory.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="must be private"), JobPortPlan(1).lease("100", directory=directory):
        pass


def test_cli_uses_one_runtime_and_delegates_reservation_to_orchestrator(tmp_path):
    from srtctl.cli.do_sweep import main

    config_path = tmp_path / "recipe.yaml"
    config_path.touch()
    runtime = Mock()
    config = Mock(job_scoped_ports=True)
    with (
        patch.object(sys, "argv", ["do_sweep", str(config_path)]),
        patch("srtctl.cli.do_sweep.load_config", return_value=config),
        patch("srtctl.cli.do_sweep.get_slurm_job_id", return_value="100"),
        patch("srtctl.cli.do_sweep.RuntimeContext.from_config", return_value=runtime) as from_config,
        patch("srtctl.cli.do_sweep.SweepOrchestrator", return_value=Mock(run=lambda: 0)) as orchestrator,
        pytest.raises(SystemExit) as exited,
    ):
        main()
    assert exited.value.code == 0
    from_config.assert_called_once_with(config, "100")
    orchestrator.assert_called_once_with(config=config, runtime=runtime, serve_only=False)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorder topology, graceful lifecycle, and required capture evidence."""

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import tomli
import yaml

from srtctl.core.schema import SrtConfig
from srtctl.core.topology import Process
from srtctl.services.config import ServiceConfig
from srtctl.services.kv_events import KVEventsService, publisher_sources, recorder_toml


def config(dp=1, tp=1):
    return SrtConfig.Schema().load(
        yaml.safe_load(f"""
schema: 2
name: kv-test
model: {{path: /model, container: /worker.sqsh, precision: bf16}}
resources: {{gpu_type: b200, gpus_per_node: 4}}
engine: vllm
roles:
  agg:
    nodes: 1
    workers: 1
    gpus: 4
    args: {{data-parallel-size: {dp}, tensor-parallel-size: {tp}}}
    kv_events: {{enable_kv_cache_events: true, publisher: zmq, topic: kv-events}}
benchmark: {{type: manual}}
observability: {{tachometer: {{enabled: false}}}}
""")
    )


def process(rank=0, gpus=4, port=22000):
    return Process(
        node="n0",
        gpu_indices=frozenset(range(gpus)),
        sys_port=21000,
        http_port=23000,
        endpoint_mode="aggregated",
        endpoint_index=0,
        node_rank=rank,
        kv_events_port=port,
    )


def runtime(tmp_path):
    return SimpleNamespace(
        log_dir=tmp_path,
        container_log_dir=PurePosixPath("/logs"),
        job_id="12345",
        run_name="kv-test",
        network_interface="eth0",
    )


def test_local_dp_sources_cover_actual_reserved_listener_ports(tmp_path):
    with patch("srtctl.services.kv_events.get_hostname_ip", return_value="10.1.2.3"):
        sources = publisher_sources(config(dp=4), [process()], runtime(tmp_path), {"clu_run_id": "run"})
    assert [s["endpoint"] for s in sources] == [f"tcp://10.1.2.3:{22000 + i}" for i in range(4)]
    assert [s["metadata"]["dp_rank"] for s in sources] == ["0", "1", "2", "3"]
    assert len({s["name"] for s in sources}) == 4
    assert all(s["metadata"]["clu_run_id"] == "run" for s in sources)


def test_tp_records_only_the_leader_publisher(tmp_path):
    with patch("srtctl.services.kv_events.get_hostname_ip", return_value="10.1.2.3"):
        sources = publisher_sources(config(tp=4), [process(), process(rank=1)], runtime(tmp_path), {})
    assert len(sources) == 1


def test_kv_only_config_and_command_keep_recorder_alive(tmp_path):
    service = ServiceConfig(name="kv-events", type="kv-events", container="/recorder.sqsh")
    ctx = SimpleNamespace(config=config(), processes=(process(),), runtime=runtime(tmp_path))
    with patch("srtctl.services.kv_events.get_hostname_ip", return_value="10.1.2.3"):
        toml = tomli.loads(recorder_toml(service, ctx))
    assert toml["endpoints"] == []
    assert len(toml["event_streams"]["sources"]) == 1
    kind = KVEventsService()
    cmd = kind.build_command(service, ctx)
    assert cmd[cmd.index("--sync-interval") + 1] == "30"
    assert kind.default_critical and kind.shutdown_tier == 1


def test_readiness_rejects_a_dead_recorder_even_with_marker(tmp_path):
    service = ServiceConfig(name="kv-events", type="kv-events")
    (tmp_path / "telemetry").mkdir()
    (tmp_path / "telemetry/kv_events.ready").write_text('{"ready":true}')
    with pytest.raises(RuntimeError, match="exited before subscriber readiness"):
        KVEventsService().wait_fleet_ready(
            service, runtime(tmp_path), [SimpleNamespace(popen=SimpleNamespace(poll=lambda: 1))]
        )


def valid_capture(tmp_path):
    (tmp_path / "telemetry/kv-events").mkdir(parents=True)
    (tmp_path / "telemetry/kv-events/one.jsonl.gz").write_bytes(b"captured trace")
    (tmp_path / "kv_events_config.toml").write_text('[[event_streams.sources]]\nname="publisher"\n')
    manifest = {
        "complete": True,
        "expected_sources": 1,
        "observed_sources": 1,
        "sources": [
            {
                "name": "publisher",
                "received_batches": 1,
                "received_events": 1,
                "sequence_gaps": 0,
                "invalid_frames": 0,
                "decode_errors": 0,
                "fatal_error": None,
                "trace_files": ["/logs/telemetry/kv-events/one.jsonl.gz"],
            }
        ],
    }
    return manifest


@pytest.mark.parametrize("failure", ["none", "incomplete", "wrong-fleet", "gap", "missing-trace", "no-events"])
def test_shutdown_requires_complete_gapless_fleet_and_all_trace_files(tmp_path, failure):
    data = valid_capture(tmp_path)
    if failure == "incomplete":
        data["complete"] = False
    if failure == "wrong-fleet":
        data["sources"][0]["name"] = "different"
    if failure == "gap":
        data["sources"][0]["sequence_gaps"] = 1
    if failure == "missing-trace":
        (tmp_path / "telemetry/kv-events/one.jsonl.gz").unlink()
    if failure == "no-events":
        data["sources"][0]["received_events"] = 0
    (tmp_path / "telemetry/kv_events_manifest.json").write_text(json.dumps(data))
    service = ServiceConfig(name="kv-events", type="kv-events")
    if failure == "none":
        KVEventsService().finalize(service, runtime(tmp_path))
    else:
        with pytest.raises(RuntimeError):
            KVEventsService().finalize(service, runtime(tmp_path))


def test_vllm_global_dp_offset_binds_the_reserved_listener(tmp_path):
    from unittest.mock import MagicMock

    cfg = config(dp=4)
    leader = process(gpus=2)
    worker = process(rank=2, gpus=2, port=22002)
    rt = MagicMock(model_path=Path("/model"), is_hf_model=False, frontend_port=9010, job_id="12345", run_name="kv-test")
    with patch("srtctl.core.slurm.get_hostname_ip", return_value="10.1.2.3"):
        command = cfg.backend.build_worker_command(worker, [leader, worker], rt)
    publisher = json.loads(command[command.index("--kv-events-config") + 1])
    assert publisher["endpoint"] == "tcp://*:22000"
    # vLLM adds rank 2 back, landing at the listener the recorder subscribes to.
    assert int(publisher["endpoint"].rsplit(":", 1)[1]) + worker.node_rank == worker.kv_events_listener()


def test_worker_health_checks_the_job_frontend_port(tmp_path):
    import threading

    from srtctl.cli.do_sweep import SweepOrchestrator
    from srtctl.core.runtime import Nodes, RuntimeContext

    rt = RuntimeContext(
        job_id="12345",
        run_name="kv-test",
        nodes=Nodes(head="n0", bench="n0", infra="n0", worker=("n0",)),
        head_node_ip="10.1.2.3",
        infra_node_ip="10.1.2.3",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/worker.sqsh"),
        gpus_per_node=4,
        network_interface="eth0",
        frontend_port=9010,
    )
    orch = SweepOrchestrator(config(), rt)
    with patch("srtctl.cli.mixins.benchmark_stage.wait_for_model", return_value=True) as health:
        assert orch._wait_for_service_ready(threading.Event())
    assert health.call_args.kwargs["port"] == 9010

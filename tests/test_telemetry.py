# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for telemetry configuration and startup."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomllib
from marshmallow import ValidationError

from srtctl.backends import VLLMProtocol, VLLMServerConfig
from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin, _trace_producer_ids
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.schema import (
    BenchmarkConfig,
    ForwardPassMetricsTelemetryConfig,
    KvCacheEventsTelemetryConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.telemetry import effective_exporter_port, generate_telemetry_config
from srtctl.core.topology import Process


def _make_config(*, telemetry: TelemetryConfig | None = None, backend=None) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="h100"),
        benchmark=BenchmarkConfig(type="manual"),
        telemetry=telemetry or TelemetryConfig(),
        **({"backend": backend} if backend is not None else {}),
    )


class TestTelemetryConfig:
    """Telemetry schema validation."""

    def test_requires_container_image_when_enabled(self):
        with pytest.raises(ValidationError, match="telemetry.container_image"):
            _make_config(
                telemetry=TelemetryConfig(
                    enabled=True,
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                )
            )

    def test_forward_pass_metrics_requires_telemetry(self):
        with pytest.raises(ValidationError, match="telemetry.enabled=true"):
            _make_config(
                telemetry=TelemetryConfig(forward_pass_metrics=ForwardPassMetricsTelemetryConfig(enabled=True))
            )

    def test_forward_pass_metrics_rejects_invalid_trace_mode(self):
        with pytest.raises(ValidationError, match="mode must be full or sampled"):
            _make_config(
                telemetry=TelemetryConfig(
                    enabled=True,
                    container_image="telemetry:latest",
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                    forward_pass_metrics=ForwardPassMetricsTelemetryConfig(enabled=True, mode="latest"),
                )
            )

    def test_kv_cache_events_requires_vllm_backend(self):
        with pytest.raises(ValidationError, match="backend.type=vllm"):
            _make_config(
                telemetry=TelemetryConfig(
                    enabled=True,
                    container_image="telemetry:latest",
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                    kv_cache_events=KvCacheEventsTelemetryConfig(enabled=True),
                )
            )

    def test_managed_exporter_port_is_job_scoped(self):
        exporter = TelemetryExporterConfig(container_image="dcgm:latest", port=9401)

        assert effective_exporter_port(exporter, 256) == 9657

    def test_opaque_custom_exporter_command_keeps_its_declared_port(self):
        exporter = TelemetryExporterConfig(
            container_image="custom:latest",
            port=9401,
            command="custom-exporter --listen :9401",
        )

        assert effective_exporter_port(exporter, 256) == 9401


class TestTelemetryConfigGeneration:
    """Topology-to-config generation."""

    @patch("srtctl.core.telemetry.get_hostname_ip")
    def test_generate_telemetry_config(self, mock_get_hostname_ip):
        mock_get_hostname_ip.side_effect = lambda host, interface=None: {"node-a": "10.0.0.1", "node-b": "10.0.0.2"}[
            host
        ]

        telemetry = TelemetryConfig(
            enabled=True,
            container_image="telemetry:latest",
            extra_metadata={"cluster": "pdx"},
            dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
            node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
        )
        runtime = MagicMock()
        runtime.job_id = "12345"
        runtime.run_name = "test_12345"
        runtime.network_interface = "eth0"
        processes = [
            Process(
                node="node-a",
                gpu_indices=frozenset({0, 1}),
                sys_port=8081,
                http_port=30000,
                endpoint_mode="prefill",
                endpoint_index=0,
                node_rank=0,
            ),
            Process(
                node="node-b",
                gpu_indices=frozenset({0, 1}),
                sys_port=8082,
                http_port=30000,
                endpoint_mode="decode",
                endpoint_index=0,
                node_rank=0,
            ),
        ]
        topology = FrontendTopology(
            nginx_node=None,
            frontend_nodes=["node-a"],
            frontend_port=8000,
            public_port=8000,
        )

        config_text = generate_telemetry_config(
            processes=processes,
            frontend_topology=topology,
            runtime=runtime,
            telemetry=telemetry,
        )

        assert 'storage = "/logs/telemetry"' in config_text
        assert 'name = "dcgm_node-a"' in config_text
        assert 'url = "http://10.0.0.1:8081/metrics"' in config_text
        assert '"cluster" = "pdx"' in config_text
        assert 'name = "frontend0"' in config_text

    @patch("srtctl.core.telemetry.get_hostname_ip", return_value="10.0.0.1")
    def test_generate_forward_pass_metrics_config(self, _mock_get_hostname_ip):
        telemetry = TelemetryConfig(
            enabled=True,
            container_image="telemetry:latest",
            forward_pass_metrics=ForwardPassMetricsTelemetryConfig(enabled=True),
            extra_metadata={"cluster": "pdx"},
            dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
            node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
        )
        runtime = MagicMock(job_id="12345", run_name="test_12345", network_interface="eth0")
        processes = [
            Process(
                node="node-a",
                gpu_indices=frozenset({0}),
                sys_port=8081,
                http_port=30000,
                endpoint_mode="prefill",
                endpoint_index=0,
                fpm_publisher=True,
            ),
            Process(
                node="node-a",
                gpu_indices=frozenset({1}),
                sys_port=8082,
                http_port=31000,
                endpoint_mode="decode",
                endpoint_index=0,
                fpm_publisher=True,
            ),
        ]
        topology = FrontendTopology(
            nginx_node=None,
            frontend_nodes=["node-a"],
            frontend_port=8000,
            public_port=8000,
        )

        config_text = generate_telemetry_config(
            processes=processes,
            frontend_topology=topology,
            runtime=runtime,
            telemetry=telemetry,
        )

        assert "[fpm]" in config_text
        assert 'trace_path = "/logs/telemetry/fpm/dynamo-fpm"' in config_text
        assert "[fpm.expected_workers]" in config_text
        assert '"prefill" = 1' in config_text
        assert '"backend" = 1' in config_text
        assert "[fpm.component_roles]" in config_text
        assert '"backend" = "decode"' in config_text
        assert '"cluster" = "pdx"' in config_text

    @patch("srtctl.core.telemetry.get_hostname_ip", return_value="10.0.0.1")
    def test_generate_direct_vllm_kv_event_sources(self, _mock_get_hostname_ip):
        telemetry = TelemetryConfig(
            enabled=True,
            container_image="telemetry:latest",
            kv_cache_events=KvCacheEventsTelemetryConfig(
                enabled=True,
                topic="kv-events",
                jsonl_gz_roll_bytes=4_096,
                max_segments=8,
                ready_delay_ms=250,
            ),
            extra_metadata={"cluster": "pdx"},
            dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
            node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
        )
        runtime = MagicMock(job_id="12345", run_name="test_12345", network_interface="eth0")
        processes = [
            Process(
                node="node-a",
                gpu_indices=frozenset({0}),
                sys_port=8081,
                http_port=30000,
                endpoint_mode="agg",
                endpoint_index=0,
                node_rank=0,
                kv_events_port=5550,
                fpm_publisher=True,
                kv_events_publisher=True,
            ),
            Process(
                node="node-a",
                gpu_indices=frozenset({1}),
                sys_port=8082,
                http_port=0,
                endpoint_mode="agg",
                endpoint_index=0,
                node_rank=1,
                kv_events_port=5551,
                fpm_publisher=False,
                kv_events_publisher=False,
            ),
        ]
        topology = FrontendTopology(
            nginx_node=None,
            frontend_nodes=["node-a"],
            frontend_port=8000,
            public_port=8000,
        )

        config_text = generate_telemetry_config(
            processes=processes,
            frontend_topology=topology,
            runtime=runtime,
            telemetry=telemetry,
        )
        parsed = tomllib.loads(config_text)

        streams = parsed["event_streams"]
        assert streams["trace_dir"] == "/logs/telemetry/kv-events"
        assert streams["roll_bytes"] == 4_096
        assert streams["ready_delay_ms"] == 250
        assert len(streams["sources"]) == 1
        assert streams["sources"][0]["endpoint"] == "tcp://10.0.0.1:5550"
        assert streams["sources"][0]["codec"] == "vllm_kv_events_v1"
        assert streams["sources"][0]["metadata"]["cluster"] == "pdx"


class TestTelemetryStageMixin:
    """Telemetry stage startup."""

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.cli.mixins.telemetry_stage.generate_telemetry_config", return_value='storage = "/logs/telemetry"\n')
    def test_start_telemetry_starts_exporters_and_scraper(self, _mock_config, mock_srun, tmp_path):
        class Harness(TelemetryStageMixin):
            def __init__(self):
                self.config = _make_config(
                    telemetry=TelemetryConfig(
                        enabled=True,
                        container_image="telemetry:latest",
                        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                        node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                    )
                )
                self.runtime = MagicMock()
                self.runtime.log_dir = tmp_path
                self.runtime.nodes.head = "node-a"
                self.runtime.srun_options = {}
                self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
                self._backend_processes = [
                    Process(
                        node="node-a",
                        gpu_indices=frozenset({0}),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=0,
                        node_rank=0,
                    )
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

            def _compute_frontend_topology(self):
                return FrontendTopology(
                    nginx_node=None,
                    frontend_nodes=["node-a"],
                    frontend_port=8000,
                    public_port=8000,
                )

        mock_srun.return_value = MagicMock()
        harness = Harness()

        procs = harness.start_telemetry()

        assert len(procs) == 3
        assert (tmp_path / "telemetry_config.toml").exists()
        assert (tmp_path / "telemetry" / "local").exists()
        assert mock_srun.call_count == 3

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch("srtctl.cli.mixins.telemetry_stage.generate_telemetry_config", return_value='storage = "/logs/telemetry"\n')
    def test_start_telemetry_uses_producer_traces_without_sidecar(self, _mock_config, mock_srun, tmp_path):
        class Harness(TelemetryStageMixin):
            def __init__(self):
                self.config = _make_config(
                    telemetry=TelemetryConfig(
                        enabled=True,
                        container_image="telemetry:latest",
                        forward_pass_metrics=ForwardPassMetricsTelemetryConfig(enabled=True),
                        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                        node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                    )
                )
                self.runtime = MagicMock()
                self.runtime.job_id = "12345"
                self.runtime.log_dir = tmp_path
                self.runtime.container_image = "/model-image"
                self.runtime.nodes.head = "node-a"
                self.runtime.nodes.infra = "node-a"
                self.runtime.srun_options = {}
                self.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
                self._backend_processes = [
                    Process(
                        node="node-a",
                        gpu_indices=frozenset({0}),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=0,
                    )
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

            def _compute_frontend_topology(self):
                return FrontendTopology(
                    nginx_node=None,
                    frontend_nodes=["node-a"],
                    frontend_port=8000,
                    public_port=8000,
                )

        mock_srun.return_value = MagicMock()

        procs = Harness().start_telemetry()

        assert len(procs) == 3
        assert mock_srun.call_count == 3
        assert (tmp_path / "telemetry" / "fpm").is_dir()
        telemetry_proc = next(proc for proc in procs if proc.name == "telemetry")
        assert telemetry_proc.shutdown_timeout == 600.0

    def test_trace_producer_ids_deduplicate_rotated_segments(self, tmp_path):
        trace_dir = tmp_path / "fpm"
        trace_dir.mkdir()
        for name in (
            "dynamo-fpm.worker-a.000000.jsonl.gz",
            "dynamo-fpm.worker-a.000001.jsonl.gz",
            "dynamo-fpm.worker_b.000000.jsonl.gz",
            "other.jsonl.gz",
        ):
            (trace_dir / name).touch()

        assert _trace_producer_ids(trace_dir) == {"worker-a", "worker_b"}

    def test_wait_for_telemetry_ready_uses_trace_producer_files(self, tmp_path):
        class Harness(TelemetryStageMixin):
            def __init__(self):
                self.config = _make_config(
                    telemetry=TelemetryConfig(
                        enabled=True,
                        container_image="telemetry:latest",
                        forward_pass_metrics=ForwardPassMetricsTelemetryConfig(
                            enabled=True,
                            ready_timeout_secs=1,
                        ),
                        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                        node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                    )
                )
                self.runtime = MagicMock(log_dir=tmp_path)
                self._backend_processes = [
                    Process(
                        node="node-a",
                        gpu_indices=frozenset({0}),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=0,
                        fpm_publisher=True,
                    )
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

        trace_dir = tmp_path / "telemetry" / "fpm"
        trace_dir.mkdir(parents=True)
        (trace_dir / "dynamo-fpm.worker-a.000000.jsonl.gz").touch()
        registry = MagicMock()

        assert Harness().wait_for_telemetry_ready(registry, threading.Event())
        ready = (tmp_path / "telemetry" / "fpm.ready").read_text()
        assert '"expected_producers": 1' in ready
        assert '"worker-a"' in ready

    def test_wait_for_telemetry_ready_uses_tachometer_marker(self, tmp_path):
        telemetry_dir = tmp_path / "telemetry"
        telemetry_dir.mkdir()
        (telemetry_dir / "kv_events.ready").write_text('{"ready": true}\n')

        class Harness(TelemetryStageMixin):
            def __init__(self):
                telemetry = TelemetryConfig(
                    enabled=True,
                    container_image="telemetry:latest",
                    kv_cache_events=KvCacheEventsTelemetryConfig(enabled=True, ready_timeout_secs=1),
                    dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                    node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
                )
                self.config = _make_config(telemetry=telemetry, backend=VLLMProtocol())
                self.runtime = MagicMock(log_dir=tmp_path)
                self._backend_processes = [
                    Process(
                        node="node-a",
                        gpu_indices=frozenset({0}),
                        sys_port=8081,
                        http_port=30000,
                        endpoint_mode="agg",
                        endpoint_index=0,
                        kv_events_port=5550,
                        fpm_publisher=True,
                        kv_events_publisher=True,
                    )
                ]

            @property
            def backend_processes(self):
                return self._backend_processes

        assert Harness().wait_for_telemetry_ready(MagicMock(), threading.Event())


class TestFpmWorkerEnvironment:
    def test_applies_official_dynamo_trace_settings(self):
        telemetry = TelemetryConfig(
            enabled=True,
            container_image="telemetry:latest",
            storage_subdir="telemetry",
            dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
            node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            forward_pass_metrics=ForwardPassMetricsTelemetryConfig(
                enabled=True,
                mode="full",
                sample_interval_ms=1_000,
                jsonl_gz_roll_bytes=4_096,
                max_segments=12,
            ),
        )
        mixin = WorkerStageMixin()
        mixin.config = _make_config(telemetry=telemetry)
        env = {"DYN_EVENT_PLANE": "nats", "DYN_FPM_MODE": "sampled"}

        mixin._apply_fpm_trace_env(env, MagicMock(fpm_port=20_380))

        assert env["DYN_EVENT_PLANE"] == "zmq"
        assert env["DYN_FORWARDPASS_METRIC_PORT"] == "20380"
        assert env["DYN_FPM_TRACE"] == "1"
        assert env["DYN_FPM_MODE"] == "full"
        assert env["DYN_FPM_SAMPLE_INTERVAL_MS"] == "1000"
        assert env["DYN_FPM_JSONL_GZ_ROLL_BYTES"] == "4096"
        assert env["DYN_FPM_MAX_SEGMENTS"] == "12"
        assert env["DYN_FPM_OUTPUT_PATH"] == "/logs/telemetry/fpm/dynamo-fpm"


class TestKvEventWorkerArguments:
    def test_vllm_dp_rank_uses_allocated_external_port(self):
        backend = VLLMProtocol(
            vllm_config=VLLMServerConfig(aggregated={"data-parallel-size": 2, "tensor-parallel-size": 1})
        )
        process = Process(
            node="node-a",
            gpu_indices=frozenset({1}),
            sys_port=8082,
            http_port=0,
            endpoint_mode="agg",
            endpoint_index=0,
            node_rank=1,
            kv_events_port=5551,
            fpm_publisher=True,
            kv_events_publisher=True,
        )

        args = backend.build_kv_event_publisher_args(process, topic="kv-events")
        payload = json.loads(args[1])

        assert args[0] == "--kv-events-config"
        assert payload["endpoint"] == "tcp://*:5550"
        assert payload["topic"] == "kv-events"
        assert payload["enable_kv_cache_events"] is True

    def test_non_scheduler_process_does_not_publish(self):
        backend = VLLMProtocol()
        process = MagicMock(kv_events_publisher=False)
        assert backend.build_kv_event_publisher_args(process) == []

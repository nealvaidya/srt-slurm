# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for telemetry configuration and startup."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin, _trace_producer_ids
from srtctl.cli.mixins.worker_stage import WorkerStageMixin
from srtctl.core.schema import (
    BenchmarkConfig,
    ForwardPassMetricsTelemetryConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.telemetry import generate_telemetry_config
from srtctl.core.topology import Process


def _make_config(*, telemetry: TelemetryConfig | None = None) -> SrtConfig:
    return SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp4"),
        resources=ResourceConfig(gpu_type="h100"),
        benchmark=BenchmarkConfig(type="manual"),
        telemetry=telemetry or TelemetryConfig(),
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

        assert 'storage = "file:///logs/telemetry"' in config_text
        assert 'name = "dcgm_node-a"' in config_text
        assert 'url = "http://10.0.0.1:8081/metrics"' in config_text
        assert '"cluster" = "pdx"' in config_text
        assert (
            '"0" = { "cluster" = "pdx", "hostname" = "node-a", "job_id" = "12345", '
            '"run_name" = "test_12345", "worker_index" = "0", "worker_process" = "0", '
            '"worker_role" = "prefill" }'
        ) in config_text
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

        assert 'storage = "file:///logs/telemetry"' in config_text
        assert "[fpm]" in config_text
        assert 'trace_path = "/logs/telemetry/fpm/dynamo-fpm"' in config_text
        assert "[fpm.expected_workers]" in config_text
        assert '"prefill" = 1' in config_text
        assert '"backend" = 1' in config_text
        assert "[fpm.component_roles]" in config_text
        assert '"backend" = "decode"' in config_text
        assert '"cluster" = "pdx"' in config_text


class TestTelemetryStageMixin:
    """Telemetry stage startup."""

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.cli.mixins.telemetry_stage.generate_telemetry_config",
        return_value='storage = "file:///logs/telemetry"\n',
    )
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
                self.runtime.nodes.het = False
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
        assert mock_srun.call_args_list[0].kwargs["use_bash_wrapper"] is False
        assert mock_srun.call_args_list[1].kwargs["use_bash_wrapper"] is False
        assert "use_bash_wrapper" not in mock_srun.call_args_list[2].kwargs

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    @patch(
        "srtctl.cli.mixins.telemetry_stage.generate_telemetry_config",
        return_value='storage = "file:///logs/telemetry"\n',
    )
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
                self.runtime.nodes.het = False
                self.runtime.nodes.het_group_for.return_value = None
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
                        fpm_port=20380,
                        fpm_publisher=True,
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
        assert mock_srun.call_args_list[0].kwargs["use_bash_wrapper"] is False
        assert mock_srun.call_args_list[1].kwargs["use_bash_wrapper"] is False
        assert "use_bash_wrapper" not in mock_srun.call_args_list[2].kwargs
        telemetry_proc = next(proc for proc in procs if proc.name == "telemetry")
        assert telemetry_proc.shutdown_timeout == 600.0
        telemetry_call = mock_srun.call_args_list[2]
        assert telemetry_call.kwargs["command"][:2] == ["sh", "-c"]
        assert ".shutdown-requested" in telemetry_call.kwargs["command"][2]
        assert "kill -TERM" in telemetry_call.kwargs["command"][2]

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_finalize_telemetry_gracefully_imports_fpm_before_fallback(self, mock_srun, tmp_path):
        harness = TelemetryStageMixin()
        harness.config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                container_image="telemetry:latest",
                forward_pass_metrics=ForwardPassMetricsTelemetryConfig(enabled=True),
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            )
        )
        harness.runtime = MagicMock()
        harness.runtime.log_dir = tmp_path
        telemetry_dir = tmp_path / "telemetry"
        telemetry_dir.mkdir()

        popen = MagicMock()

        def _finish_gracefully(*, timeout):
            assert timeout == 600
            assert (telemetry_dir / ".shutdown-requested").read_text() == "shutdown\n"
            (telemetry_dir / "final.parquet").write_bytes(b"combined-parquet")
            (telemetry_dir / "fpm_manifest.json").write_text('{"complete": true, "received_events": 42}\n')
            return 0

        popen.wait.side_effect = _finish_gracefully
        telemetry_proc = MagicMock(is_running=True, popen=popen)
        registry = MagicMock()
        registry.get_process.return_value = telemetry_proc

        result = harness.finalize_telemetry(registry)

        assert result == telemetry_dir / "final.parquet"
        assert not (telemetry_dir / ".shutdown-requested").exists()
        mock_srun.assert_not_called()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_finalize_telemetry_fpm_requires_manifest(self, mock_srun, tmp_path):
        harness = TelemetryStageMixin()
        harness.config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                container_image="telemetry:latest",
                forward_pass_metrics=ForwardPassMetricsTelemetryConfig(enabled=True),
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            )
        )
        harness.runtime = MagicMock()
        harness.runtime.log_dir = tmp_path
        telemetry_dir = tmp_path / "telemetry"
        telemetry_dir.mkdir()
        (telemetry_dir / "final.parquet").write_bytes(b"metrics-only")

        assert harness.finalize_telemetry() is None
        mock_srun.assert_not_called()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_finalize_telemetry_compacts_checkpoints(self, mock_srun, tmp_path):
        harness = TelemetryStageMixin()
        harness.config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                container_image="telemetry:latest",
                binary_path="/usr/local/bin/tachometer-scraper",
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            )
        )
        harness.runtime = MagicMock()
        harness.runtime.log_dir = tmp_path
        harness.runtime.nodes.head = "node-a"
        harness.runtime.nodes.het_group_for.return_value = None
        harness.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
        harness.runtime.srun_options = {}

        local_dir = tmp_path / "telemetry" / "local"
        local_dir.mkdir(parents=True)
        (local_dir / "current.arrow").write_bytes(b"checkpoint")

        proc = MagicMock()

        def _finish_compaction(*, timeout):
            assert timeout == 300
            final_path = tmp_path / "telemetry" / "final.parquet"
            final_path.write_bytes(b"parquet")
            return 0

        proc.wait.side_effect = _finish_compaction
        mock_srun.return_value = proc

        result = harness.finalize_telemetry()

        assert result == tmp_path / "telemetry" / "final.parquet"
        call = mock_srun.call_args
        assert call.kwargs["command"] == [
            "/usr/local/bin/tachometer-scraper",
            "compact",
            "/logs/telemetry/local",
            "--output",
            "file:///logs/telemetry",
        ]
        assert call.kwargs["container_image"] == "telemetry:latest"
        assert call.kwargs["use_bash_wrapper"] is False

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_finalize_telemetry_skips_without_checkpoints(self, mock_srun, tmp_path):
        harness = TelemetryStageMixin()
        harness.config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                container_image="telemetry:latest",
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            )
        )
        harness.runtime = MagicMock()
        harness.runtime.log_dir = tmp_path
        (tmp_path / "telemetry" / "local").mkdir(parents=True)

        assert harness.finalize_telemetry() is None
        mock_srun.assert_not_called()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process")
    def test_finalize_telemetry_preserves_existing_final(self, mock_srun, tmp_path):
        harness = TelemetryStageMixin()
        harness.config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                container_image="telemetry:latest",
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            )
        )
        harness.runtime = MagicMock()
        harness.runtime.log_dir = tmp_path
        final_path = tmp_path / "telemetry" / "final.parquet"
        final_path.parent.mkdir(parents=True)
        final_path.write_bytes(b"already-final")

        assert harness.finalize_telemetry() == final_path
        mock_srun.assert_not_called()

    @patch("srtctl.cli.mixins.telemetry_stage.start_srun_process", side_effect=RuntimeError("no step"))
    def test_finalize_telemetry_launch_failure_is_nonfatal(self, mock_srun, tmp_path):
        harness = TelemetryStageMixin()
        harness.config = _make_config(
            telemetry=TelemetryConfig(
                enabled=True,
                container_image="telemetry:latest",
                dcgm_exporter=TelemetryExporterConfig(container_image="dcgm:latest", port=9401),
                node_exporter=TelemetryExporterConfig(container_image="node:latest", port=9101),
            )
        )
        harness.runtime = MagicMock()
        harness.runtime.log_dir = tmp_path
        harness.runtime.nodes.head = "node-a"
        harness.runtime.nodes.het_group_for.return_value = None
        harness.runtime.container_mounts = {Path(tmp_path): Path("/logs")}
        harness.runtime.srun_options = {}
        local_dir = tmp_path / "telemetry" / "local"
        local_dir.mkdir(parents=True)
        (local_dir / "current.arrow").write_bytes(b"checkpoint")

        assert harness.finalize_telemetry() is None
        mock_srun.assert_called_once()

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

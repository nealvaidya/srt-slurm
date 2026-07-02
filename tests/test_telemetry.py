# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for telemetry configuration and startup."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from marshmallow import ValidationError

from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.cli.mixins.telemetry_stage import TelemetryStageMixin
from srtctl.core.schema import (
    BenchmarkConfig,
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

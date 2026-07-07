# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry stage mixin for SweepOrchestrator."""

from __future__ import annotations

import json
import logging
import shlex
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process
from srtctl.core.telemetry import generate_telemetry_config

if TYPE_CHECKING:
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig, TelemetryExporterConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class TelemetryStageMixin:
    """Mixin for telemetry startup stage."""

    config: SrtConfig
    runtime: RuntimeContext

    @property
    def backend_processes(self) -> list[Process]:
        """Backend worker processes."""
        raise NotImplementedError

    def _compute_frontend_topology(self) -> Any:
        """Frontend topology helper provided by FrontendStageMixin."""
        raise NotImplementedError

    def _start_exporter_container(
        self,
        *,
        exporter_config: TelemetryExporterConfig,
        name: str,
        nodelist: list[str],
        log_file: Path,
        default_command_template: str,
    ) -> ManagedProcess:
        """Start one exporter container across the requested nodes."""
        if exporter_config.command is None:
            cmd_str = default_command_template.format(port=exporter_config.port)
        elif "{port}" in exporter_config.command:
            cmd_str = exporter_config.command.format(port=exporter_config.port)
        else:
            cmd_str = exporter_config.command

        proc = start_srun_process(
            command=shlex.split(cmd_str),
            ntasks=len(nodelist),
            nodelist=nodelist,
            output=str(log_file),
            container_image=exporter_config.container_image,
            container_mounts=self.runtime.container_mounts,
            srun_options=self.runtime.srun_options,
        )
        return ManagedProcess(
            name=name,
            popen=proc,
            log_file=log_file,
            node=",".join(nodelist),
        )

    def wait_for_telemetry_ready(
        self,
        registry: ProcessRegistry,
        stop_event: threading.Event,
    ) -> bool:
        """Wait until every expected Dynamo producer has opened a trace file."""
        fpm = self.config.telemetry.forward_pass_metrics
        if not fpm.enabled:
            return True

        telemetry_dir = self.runtime.log_dir / self.config.telemetry.storage_subdir
        trace_dir = telemetry_dir / "fpm"
        ready_path = telemetry_dir / "fpm.ready"
        expected_producers = sum(process.fpm_publisher for process in self.backend_processes)
        deadline = time.monotonic() + fpm.ready_timeout_secs
        logger.info("Waiting for %d Dynamo FPM trace producer(s) under %s", expected_producers, trace_dir)
        while time.monotonic() < deadline and not stop_event.is_set():
            producer_ids = _trace_producer_ids(trace_dir)
            if len(producer_ids) >= expected_producers:
                ready_path.write_text(
                    json.dumps(
                        {
                            "ready": True,
                            "expected_producers": expected_producers,
                            "producer_ids": sorted(producer_ids),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                logger.info("Dynamo FPM tracing is ready for %d producer(s)", len(producer_ids))
                return True
            if registry.check_failures():
                logger.error("A critical process failed while waiting for FPM readiness")
                return False
            time.sleep(1)

        logger.error(
            "Dynamo FPM trace producers did not become ready within %ss",
            fpm.ready_timeout_secs,
        )
        return False

    def start_telemetry(self) -> list[ManagedProcess]:
        """Start the configured telemetry provider."""
        telemetry = self.config.telemetry
        if not telemetry.enabled:
            logger.info("Telemetry disabled")
            return []
        if telemetry.dcgm_exporter is None or telemetry.node_exporter is None or telemetry.container_image is None:
            raise ValueError("Telemetry is enabled but required provider configuration is missing")

        logger.info("Starting telemetry provider: %s", telemetry.provider.value)

        topology = self._compute_frontend_topology()
        config_path = self.runtime.log_dir / "telemetry_config.toml"
        config_path.write_text(
            generate_telemetry_config(
                processes=self.backend_processes,
                frontend_topology=topology,
                runtime=self.runtime,
                telemetry=telemetry,
            )
        )

        telemetry_dir = self.runtime.log_dir / telemetry.storage_subdir
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        local_dir = telemetry_dir / "local"
        local_dir.mkdir(parents=True, exist_ok=True)
        if telemetry.forward_pass_metrics.enabled:
            (telemetry_dir / "fpm").mkdir(parents=True, exist_ok=True)
            for stale_path in (telemetry_dir / "fpm.ready", telemetry_dir / "fpm_manifest.json"):
                stale_path.unlink(missing_ok=True)

        worker_nodes = sorted({process.node for process in self.backend_processes})
        processes: list[ManagedProcess] = []
        processes.append(
            self._start_exporter_container(
                exporter_config=telemetry.dcgm_exporter,
                name="telemetry_dcgm_exporter",
                nodelist=worker_nodes,
                log_file=self.runtime.log_dir / "telemetry_dcgm_exporter.out",
                default_command_template="dcgm-exporter --collect-interval=100 --address :{port}",
            )
        )
        processes.append(
            self._start_exporter_container(
                exporter_config=telemetry.node_exporter,
                name="telemetry_node_exporter",
                nodelist=worker_nodes,
                log_file=self.runtime.log_dir / "telemetry_node_exporter.out",
                default_command_template=(
                    "/bin/node_exporter --web.listen-address=:{port} "
                    "--collector.disable-defaults --collector.cpu --collector.infiniband --collector.meminfo"
                ),
            )
        )

        cmd = [
            telemetry.binary_path,
            "--config",
            "/telemetry_config.toml",
            "--local-dir",
            f"/logs/{telemetry.storage_subdir}/local",
        ]
        if telemetry.sync_interval_secs > 0:
            cmd.extend(["--sync-interval", str(telemetry.sync_interval_secs)])

        env_to_set: dict[str, str] = {}
        if telemetry.compaction_threads > 0:
            env_to_set["POLARS_MAX_THREADS"] = str(telemetry.compaction_threads)

        scraper_mounts = self.runtime.container_mounts | {
            config_path: Path("/telemetry_config.toml"),
        }
        processes.append(
            ManagedProcess(
                name="telemetry",
                popen=start_srun_process(
                    command=cmd,
                    nodelist=[self.runtime.nodes.head],
                    output=str(self.runtime.log_dir / "telemetry.out"),
                    container_image=telemetry.container_image,
                    container_mounts=scraper_mounts,
                    env_to_set=env_to_set,
                    srun_options=self.runtime.srun_options,
                ),
                log_file=self.runtime.log_dir / "telemetry.out",
                node=self.runtime.nodes.head,
                shutdown_timeout=600.0,
            )
        )
        logger.info("Telemetry started with artifacts under %s", telemetry_dir)
        return processes


def _trace_producer_ids(trace_dir: Path) -> set[str]:
    """Return producer IDs represented by Dynamo's trace segment filenames."""
    producer_ids = set()
    for path in trace_dir.glob("dynamo-fpm.*.jsonl.gz"):
        remainder = path.name.removeprefix("dynamo-fpm.").removesuffix(".jsonl.gz")
        producer_id, separator, segment = remainder.rpartition(".")
        if separator and producer_id and segment.isdigit():
            producer_ids.add(producer_id)
    return producer_ids

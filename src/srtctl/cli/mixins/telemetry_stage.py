# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry stage mixin for SweepOrchestrator."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process
from srtctl.core.telemetry import effective_exporter_port, generate_telemetry_config

if TYPE_CHECKING:
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig, TelemetryExporterConfig
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)

TELEMETRY_FINALIZE_TIMEOUT_SECS = 300
TELEMETRY_GRACEFUL_SHUTDOWN_TIMEOUT_SECS = 600
TELEMETRY_SHUTDOWN_REQUEST = ".shutdown-requested"


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
        port = effective_exporter_port(exporter_config, self.runtime.port_plan.offset)
        if exporter_config.command is None:
            cmd_str = default_command_template.format(port=port)
        elif "{port}" in exporter_config.command:
            cmd_str = exporter_config.command.format(port=port)
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
            # Exporter images are commonly distroless or scratch-based and do
            # not contain bash. Their commands do not need shell expansion.
            use_bash_wrapper=False,
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
        """Wait until all enabled benchmark-owned telemetry collectors are ready."""
        fpm = self.config.telemetry.forward_pass_metrics
        kv_events = self.config.telemetry.kv_cache_events
        if not fpm.enabled and not kv_events.enabled:
            return True

        telemetry_dir = self.runtime.log_dir / self.config.telemetry.storage_subdir
        trace_dir = telemetry_dir / "fpm"
        fpm_ready_path = telemetry_dir / "fpm.ready"
        kv_ready_path = telemetry_dir / "kv_events.ready"
        expected_producers = sum(process.fpm_publisher for process in self.backend_processes)
        started_at = time.monotonic()
        fpm_ready = not fpm.enabled
        kv_ready = not kv_events.enabled
        if fpm.enabled:
            logger.info("Waiting for %d Dynamo FPM trace producer(s) under %s", expected_producers, trace_dir)
        if kv_events.enabled:
            logger.info("Waiting for Tachometer KV-event subscribers at %s", kv_ready_path)

        while not stop_event.is_set():
            elapsed = time.monotonic() - started_at
            if not fpm_ready:
                producer_ids = _trace_producer_ids(trace_dir)
                if len(producer_ids) >= expected_producers:
                    fpm_ready_path.write_text(
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
                    fpm_ready = True
                elif elapsed >= fpm.ready_timeout_secs:
                    logger.error(
                        "Dynamo FPM trace producers did not become ready within %ss",
                        fpm.ready_timeout_secs,
                    )
                    return False

            if not kv_ready and kv_ready_path.is_file():
                try:
                    marker = json.loads(kv_ready_path.read_text())
                except (OSError, json.JSONDecodeError):
                    marker = {}
                if marker.get("ready") is True:
                    logger.info("Tachometer KV-event subscribers are ready")
                    kv_ready = True
            if not kv_ready and elapsed >= kv_events.ready_timeout_secs:
                logger.error(
                    "Tachometer KV-event subscribers did not become ready within %ss",
                    kv_events.ready_timeout_secs,
                )
                return False

            if fpm_ready and kv_ready:
                return True
            if registry.check_failures():
                logger.error("A critical process failed while waiting for telemetry readiness")
                return False
            time.sleep(1)

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
        (telemetry_dir / TELEMETRY_SHUTDOWN_REQUEST).unlink(missing_ok=True)
        if telemetry.forward_pass_metrics.enabled:
            (telemetry_dir / "fpm").mkdir(parents=True, exist_ok=True)
            for stale_path in (telemetry_dir / "fpm.ready", telemetry_dir / "fpm_manifest.json"):
                stale_path.unlink(missing_ok=True)
        if telemetry.kv_cache_events.enabled:
            (telemetry_dir / "kv-events").mkdir(parents=True, exist_ok=True)
            for stale_path in (telemetry_dir / "kv_events.ready", telemetry_dir / "kv_events_manifest.json"):
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

        scraper_cmd = [
            telemetry.binary_path,
            "--config",
            "/telemetry_config.toml",
            "--local-dir",
            f"/logs/{telemetry.storage_subdir}/scraper/local",
        ]
        if telemetry.sync_interval_secs > 0:
            scraper_cmd.extend(["--sync-interval", str(telemetry.sync_interval_secs)])

        # The local srun client does not reliably forward SIGTERM to the
        # container task. Run the scraper behind an in-container watcher so
        # finalize_telemetry can request shutdown through the shared log
        # directory. The watcher signals the real Tachometer PID in the same
        # namespace, allowing its FPM import and compaction handler to finish.
        container_shutdown_path = f"/logs/{telemetry.storage_subdir}/{TELEMETRY_SHUTDOWN_REQUEST}"
        scraper_script = "\n".join(
            [
                "set -u",
                f"shutdown_request={shlex.quote(container_shutdown_path)}",
                'rm -f "$shutdown_request"',
                f"{shlex.join(scraper_cmd)} &",
                "scraper_pid=$!",
                "(",
                '  while kill -0 "$scraper_pid" 2>/dev/null; do',
                '    if [ -f "$shutdown_request" ]; then',
                '      kill -TERM "$scraper_pid" 2>/dev/null || true',
                "      exit 0",
                "    fi",
                "    sleep 0.2",
                "  done",
                ") &",
                "watcher_pid=$!",
                'wait "$scraper_pid"',
                "status=$?",
                'kill "$watcher_pid" 2>/dev/null || true',
                'wait "$watcher_pid" 2>/dev/null || true',
                'exit "$status"',
            ]
        )

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
                    command=["sh", "-c", scraper_script],
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

    def finalize_telemetry(self, registry: ProcessRegistry | None = None) -> Path | None:
        """Ensure durable telemetry output exists before post-processing.

        When the scraper is still registered, request graceful shutdown through
        its shared sentinel before falling back to checkpoint compaction. The
        scraper wrapper translates the sentinel into SIGTERM inside the
        container, so Tachometer can import completed FPM trace segments and
        write the FPM manifest without losing its original timestamp origin.
        """
        telemetry = self.config.telemetry
        if not telemetry.enabled or telemetry.container_image is None:
            return None

        telemetry_dir = self.runtime.log_dir / telemetry.storage_subdir
        scraper_dir = telemetry_dir / "scraper"
        final_path = scraper_dir / "final.parquet"
        manifest_path = telemetry_dir / "fpm_manifest.json"
        fpm_enabled = telemetry.forward_pass_metrics.enabled

        telemetry_proc = registry.get_process("telemetry") if registry is not None else None
        if telemetry_proc is not None and telemetry_proc.is_running:
            shutdown_path = telemetry_dir / TELEMETRY_SHUTDOWN_REQUEST
            logger.info("Requesting graceful telemetry shutdown through %s", shutdown_path)
            shutdown_path.write_text("shutdown\n")
            try:
                return_code = telemetry_proc.popen.wait(timeout=TELEMETRY_GRACEFUL_SHUTDOWN_TIMEOUT_SECS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Telemetry graceful shutdown timed out after %ss",
                    TELEMETRY_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
                )
            except Exception as exc:
                logger.warning("Telemetry graceful shutdown failed: %s", exc)
            else:
                if return_code != 0:
                    logger.warning("Telemetry scraper exited with code %s during graceful shutdown", return_code)
                elif self._telemetry_outputs_complete(final_path, manifest_path, fpm_enabled):
                    logger.info("Telemetry graceful finalization complete: %s", final_path)
                    return final_path
            finally:
                shutdown_path.unlink(missing_ok=True)

            if telemetry_proc.is_running:
                logger.warning("Forcing unresponsive telemetry step down before checkpoint recovery")
                try:
                    telemetry_proc.terminate(timeout=10)
                except Exception as exc:
                    logger.warning("Unable to terminate telemetry step before recovery: %s", exc)

        if self._telemetry_outputs_complete(final_path, manifest_path, fpm_enabled):
            logger.info("Telemetry final outputs already exist: %s", final_path)
            return final_path

        local_dir = scraper_dir / "local"
        checkpoint_files = [local_dir / "current.arrow"]
        checkpoint_files.extend(local_dir.glob("out-*.parquet"))
        checkpoint_files.extend(local_dir.glob("incomplete-*.parquet"))
        if not any(path.is_file() for path in checkpoint_files):
            logger.warning("Telemetry finalization skipped: no checkpoints under %s", local_dir)
            return None

        if fpm_enabled:
            logger.warning(
                "Telemetry scraper did not complete graceful FPM finalization; "
                "checkpoint compaction cannot import FPM traces"
            )

        container_local_dir = f"/logs/{telemetry.storage_subdir}/scraper/local"
        container_output = f"file:///logs/{telemetry.storage_subdir}/scraper"
        log_file = self.runtime.log_dir / "telemetry_finalize.out"
        command = [
            telemetry.binary_path,
            "compact",
            container_local_dir,
            "--output",
            container_output,
        ]
        logger.info("Finalizing telemetry checkpoints into %s", final_path)
        try:
            proc = start_srun_process(
                command=command,
                nodelist=[self.runtime.nodes.head],
                output=str(log_file),
                container_image=telemetry.container_image,
                container_mounts=self.runtime.container_mounts,
                srun_options=self.runtime.srun_options,
                het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
                use_bash_wrapper=False,
            )
        except Exception as exc:
            logger.warning("Unable to launch telemetry finalization: %s", exc)
            return None
        try:
            return_code = proc.wait(timeout=TELEMETRY_FINALIZE_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Telemetry finalization timed out after %ss; terminating compact process",
                TELEMETRY_FINALIZE_TIMEOUT_SECS,
            )
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            return None
        except Exception as exc:
            logger.warning("Telemetry finalization process failed: %s", exc)
            return None

        if return_code != 0:
            logger.warning("Telemetry finalization failed with exit code %s; see %s", return_code, log_file)
            return None
        if not final_path.is_file():
            logger.warning("Telemetry compact command completed without producing %s", final_path)
            return None

        logger.info("Telemetry finalization complete: %s", final_path)
        return final_path

    @staticmethod
    def _telemetry_outputs_complete(final_path: Path, manifest_path: Path, fpm_enabled: bool) -> bool:
        """Return whether the expected durable telemetry outputs are present."""
        if not final_path.is_file():
            return False
        if not fpm_enabled:
            return True
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        received_events = manifest.get("received_events")
        return manifest.get("complete") is True and type(received_events) is int and received_events > 0


def _trace_producer_ids(trace_dir: Path) -> set[str]:
    """Return producer IDs represented by Dynamo's trace segment filenames."""
    producer_ids = set()
    for path in trace_dir.glob("dynamo-fpm.*.jsonl.gz"):
        remainder = path.name.removeprefix("dynamo-fpm.").removesuffix(".jsonl.gz")
        producer_id, separator, segment = remainder.rpartition(".")
        if separator and producer_id and segment.isdigit():
            producer_ids.add(producer_id)
    return producer_ids

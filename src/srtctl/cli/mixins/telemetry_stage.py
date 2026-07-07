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
from srtctl.core.telemetry import generate_telemetry_config

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
    ) -> list[ManagedProcess]:
        """Start one exporter container across the requested nodes.

        Under SLURM heterogeneous jobs the nodelist may span both het
        components (prefill on group 0, decode on group 1). A single srun
        cannot target multiple het components, so we split the launch into
        one srun per group when needed.
        """
        if exporter_config.command is None:
            cmd_str = default_command_template.format(port=exporter_config.port)
        elif "{port}" in exporter_config.command:
            cmd_str = exporter_config.command.format(port=exporter_config.port)
        else:
            cmd_str = exporter_config.command

        if self.runtime.nodes.het:
            groups: dict[int, list[str]] = {}
            for node in nodelist:
                g = self.runtime.nodes.het_group_for(node)
                if g is None:
                    raise RuntimeError(f"node {node!r} not in any het component")
                groups.setdefault(g, []).append(node)
            chunks = sorted(groups.items())
        else:
            chunks = [(-1, nodelist)]  # sentinel: no --het-group

        managed: list[ManagedProcess] = []
        for group_id, nodes in chunks:
            het_group = group_id if group_id >= 0 else None
            chunk_log = log_file if len(chunks) == 1 else log_file.with_suffix(f".g{group_id}.out")
            proc = start_srun_process(
                command=shlex.split(cmd_str),
                ntasks=len(nodes),
                nodelist=nodes,
                output=str(chunk_log),
                container_image=exporter_config.container_image,
                container_mounts=self.runtime.container_mounts,
                srun_options=self.runtime.srun_options,
                het_group=het_group,
                # Exporter images are commonly scratch-based and contain no
                # shell.  Their commands need neither environment setup nor a
                # cluster bash preamble, so execute them directly.
                use_bash_wrapper=False,
            )
            chunk_name = name if len(chunks) == 1 else f"{name}_g{group_id}"
            managed.append(
                ManagedProcess(
                    name=chunk_name,
                    popen=proc,
                    log_file=chunk_log,
                    node=",".join(nodes),
                )
            )
        return managed

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
        (telemetry_dir / TELEMETRY_SHUTDOWN_REQUEST).unlink(missing_ok=True)
        if telemetry.forward_pass_metrics.enabled:
            (telemetry_dir / "fpm").mkdir(parents=True, exist_ok=True)
            for stale_path in (telemetry_dir / "fpm.ready", telemetry_dir / "fpm_manifest.json"):
                stale_path.unlink(missing_ok=True)

        worker_nodes = sorted({process.node for process in self.backend_processes})
        processes: list[ManagedProcess] = []
        processes.extend(
            self._start_exporter_container(
                exporter_config=telemetry.dcgm_exporter,
                name="telemetry_dcgm_exporter",
                nodelist=worker_nodes,
                log_file=self.runtime.log_dir / "telemetry_dcgm_exporter.out",
                default_command_template="dcgm-exporter --collect-interval=100 --address :{port}",
            )
        )
        processes.extend(
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
            f"/logs/{telemetry.storage_subdir}/local",
        ]
        if telemetry.sync_interval_secs > 0:
            scraper_cmd.extend(["--sync-interval", str(telemetry.sync_interval_secs)])

        # The local srun client does not reliably forward SIGTERM to the
        # container task.  Run the scraper behind an in-container watcher so
        # finalize_telemetry can request shutdown through the shared log
        # directory.  The watcher signals the real Tachometer PID in the same
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
                    het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
                ),
                log_file=self.runtime.log_dir / "telemetry.out",
                node=self.runtime.nodes.head,
                shutdown_timeout=600.0,
            )
        )
        logger.info("Telemetry started with artifacts under %s", telemetry_dir)
        return processes

    def finalize_telemetry(self, registry: ProcessRegistry | None = None) -> Path | None:
        """Ensure a final telemetry parquet exists before post-processing.

        When the scraper is still registered, request graceful shutdown through
        its shared sentinel before falling back to checkpoint compaction.  The
        scraper wrapper translates the sentinel into SIGTERM inside the
        container, so Tachometer can import completed FPM trace segments, write
        ``fpm_manifest.json``, and preserve its original timestamp origin.

        Generic checkpoint compaction remains a recovery path for telemetry
        runs without FPM, or when the scraper already exited unexpectedly.
        """
        telemetry = self.config.telemetry
        if not telemetry.enabled or telemetry.container_image is None:
            return None

        telemetry_dir = self.runtime.log_dir / telemetry.storage_subdir
        final_path = telemetry_dir / "final.parquet"
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

        local_dir = telemetry_dir / "local"
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

        container_local_dir = f"/logs/{telemetry.storage_subdir}/local"
        container_output = f"file:///logs/{telemetry.storage_subdir}"
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

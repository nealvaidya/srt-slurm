# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry stage mixin for SweepOrchestrator."""

from __future__ import annotations

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

    def _build_dynamo_preamble(self) -> str | None:
        """Build the same setup/install preamble used by Dynamo workers."""
        parts = []
        if self.config.setup_script:
            script_path = f"/configs/{self.config.setup_script}"
            parts.append(
                f"echo 'Running setup script: {script_path}' && "
                f"if [ -f '{script_path}' ]; then bash '{script_path}'; else echo 'WARNING: {script_path} not found'; fi"
            )
        if self.config.dynamo.install:
            parts.append(self.config.dynamo.get_install_commands())
        return " && ".join(parts) if parts else None

    def _fpm_components(self) -> list[str]:
        modes = {process.endpoint_mode for process in self.backend_processes}
        components = []
        if "prefill" in modes:
            components.append("prefill")
        if modes & {"decode", "agg"}:
            components.append("backend")
        return components

    def wait_for_telemetry_ready(
        self,
        registry: ProcessRegistry,
        stop_event: threading.Event,
    ) -> bool:
        """Wait until Tachometer has stored an FPM event from every logical worker."""
        fpm = self.config.telemetry.forward_pass_metrics
        if not fpm.enabled:
            return True

        ready_path = self.runtime.log_dir / self.config.telemetry.storage_subdir / "fpm.ready"
        deadline = time.monotonic() + fpm.ready_timeout_secs
        logger.info("Waiting for Dynamo forward-pass metrics at %s", ready_path)
        while time.monotonic() < deadline and not stop_event.is_set():
            if ready_path.exists():
                logger.info("Dynamo forward-pass metrics are ready")
                return True
            if registry.check_failures():
                logger.error("A critical process failed while waiting for FPM readiness")
                return False
            time.sleep(1)

        logger.error(
            "Dynamo forward-pass metrics did not become ready within %ss",
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
        fpm_socket_dir: Path | None = None
        if telemetry.forward_pass_metrics.enabled:
            fpm_socket_dir = Path(f"/tmp/srtctl-fpm-{self.runtime.job_id}")
            fpm_socket_dir.mkdir(parents=True, exist_ok=True)
            for stale_path in (
                fpm_socket_dir / "fpm.sock",
                telemetry_dir / "fpm.ready",
                telemetry_dir / "fpm_manifest.json",
            ):
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
        if fpm_socket_dir is not None:
            scraper_mounts[fpm_socket_dir] = Path("/fpm")
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
                    het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
                ),
                log_file=self.runtime.log_dir / "telemetry.out",
                node=self.runtime.nodes.head,
            )
        )

        if telemetry.forward_pass_metrics.enabled:
            assert fpm_socket_dir is not None
            fpm = telemetry.forward_pass_metrics
            fpm_cmd = [
                "python3",
                "-m",
                "dynamo.common.export_forward_pass_metrics",
                "--namespace",
                fpm.namespace,
                "--socket",
                "/fpm/fpm.sock",
                "--connect-timeout",
                str(fpm.connect_timeout_secs),
            ]
            for component in self._fpm_components():
                fpm_cmd.extend(["--component", component])
            fpm_mounts = self.runtime.container_mounts | {
                fpm_socket_dir: Path("/fpm"),
            }
            processes.append(
                ManagedProcess(
                    name="telemetry_fpm_exporter",
                    popen=start_srun_process(
                        command=fpm_cmd,
                        nodelist=[self.runtime.nodes.head],
                        output=str(self.runtime.log_dir / "telemetry_fpm_exporter.out"),
                        container_image=str(self.runtime.container_image),
                        container_mounts=fpm_mounts,
                        env_to_set={
                            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.infra}:2379",
                            "DYN_DISCOVERY_BACKEND": "etcd",
                            "DYN_EVENT_PLANE": "zmq",
                            "DYN_REQUEST_PLANE": "tcp",
                            "DYN_SYSTEM_PORT": str(max(process.sys_port for process in self.backend_processes) + 1000),
                        },
                        bash_preamble=self._build_dynamo_preamble(),
                        srun_options=self.runtime.srun_options,
                        het_group=self.runtime.nodes.het_group_for(self.runtime.nodes.head),
                    ),
                    log_file=self.runtime.log_dir / "telemetry_fpm_exporter.out",
                    node=self.runtime.nodes.head,
                    critical=True,
                )
            )
        logger.info("Telemetry started with artifacts under %s", telemetry_dir)
        return processes

    def finalize_telemetry(self) -> Path | None:
        """Ensure a final telemetry parquet exists before post-processing.

        Generic process cleanup terminates the local ``srun`` client, which
        may kill the remote scraper before Tachometer can finish its signal
        handler.  Recover from the durable Arrow/parquet checkpoints with the
        scraper's own compact command before the log directory is uploaded.
        """
        telemetry = self.config.telemetry
        if not telemetry.enabled or telemetry.container_image is None:
            return None

        telemetry_dir = self.runtime.log_dir / telemetry.storage_subdir
        final_path = telemetry_dir / "final.parquet"
        if final_path.exists():
            logger.info("Telemetry final parquet already exists: %s", final_path)
            return final_path

        local_dir = telemetry_dir / "local"
        checkpoint_files = [local_dir / "current.arrow"]
        checkpoint_files.extend(local_dir.glob("out-*.parquet"))
        checkpoint_files.extend(local_dir.glob("incomplete-*.parquet"))
        if not any(path.is_file() for path in checkpoint_files):
            logger.warning("Telemetry finalization skipped: no checkpoints under %s", local_dir)
            return None

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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from srtctl.core.slurm import get_hostname_ip

if TYPE_CHECKING:
    from srtctl.cli.mixins.frontend_stage import FrontendTopology
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import TelemetryConfig, TelemetryExporterConfig
    from srtctl.core.topology import Process


def effective_exporter_port(config: TelemetryExporterConfig, offset: int) -> int:
    """Shift managed exporter commands while preserving opaque custom commands."""
    port = config.port
    command = config.command
    return port + offset if command is None or "{port}" in command else port


@dataclass(frozen=True)
class TelemetryEndpoint:
    """One telemetry endpoint entry in the scraper config."""

    name: str
    url: str
    frequency: float
    filter: str | None = None
    node_metadata: dict[str, str] = field(default_factory=dict)
    gpu_metadata: dict[str, dict[str, str]] = field(default_factory=dict)


def generate_telemetry_config(
    *,
    processes: list[Process],
    frontend_topology: FrontendTopology,
    runtime: RuntimeContext,
    telemetry: TelemetryConfig,
) -> str:
    """Generate telemetry TOML from backend and frontend topology."""
    dcgm_exporter = telemetry.dcgm_exporter
    node_exporter = telemetry.node_exporter
    if dcgm_exporter is None or node_exporter is None:
        raise ValueError("Telemetry exporters must be configured before generating telemetry config")

    endpoints: list[TelemetryEndpoint] = []
    physical_nodes: dict[str, list[Process]] = {}
    for process in processes:
        physical_nodes.setdefault(process.node, []).append(process)

    for node in sorted(physical_nodes):
        node_processes = physical_nodes[node]
        node_metadata = {"hostname": node, "job_id": runtime.job_id, "run_name": runtime.run_name}
        node_metadata.update(telemetry.extra_metadata)

        gpu_metadata: dict[str, dict[str, str]] = {}
        for process in node_processes:
            for gpu_idx in sorted(process.gpu_indices):
                gpu_metadata[str(gpu_idx)] = {
                    "worker_index": str(process.endpoint_index),
                    "worker_process": str(process.node_rank),
                    "worker_role": process.endpoint_mode,
                }

        endpoints.append(
            TelemetryEndpoint(
                name=f"dcgm_{node}",
                url=f"http://{node}:{effective_exporter_port(dcgm_exporter, runtime.port_plan.offset)}/metrics",
                frequency=telemetry.default_frequency,
                filter="dcgm",
                node_metadata=node_metadata,
                gpu_metadata=gpu_metadata,
            )
        )
        endpoints.append(
            TelemetryEndpoint(
                name=f"node_exporter_{node}",
                url=f"http://{node}:{effective_exporter_port(node_exporter, runtime.port_plan.offset)}/metrics",
                frequency=telemetry.default_frequency,
                filter="node_exporter",
                node_metadata=node_metadata,
            )
        )

    for process in sorted(processes, key=lambda p: (p.endpoint_mode, p.endpoint_index, p.node_rank, p.node)):
        node_ip = get_hostname_ip(process.node, runtime.network_interface)
        node_metadata = {
            "hostname": process.node,
            "worker_index": str(process.endpoint_index),
            "worker_process": str(process.node_rank),
            "worker_role": process.endpoint_mode,
        }
        node_metadata.update(telemetry.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=f"backend_{process.endpoint_mode}{process.endpoint_index}_rank{process.node_rank}",
                url=f"http://{node_ip}:{process.sys_port}/metrics",
                frequency=telemetry.default_frequency,
                filter="backend",
                node_metadata=node_metadata,
            )
        )

    for frontend_index, node in enumerate(frontend_topology.frontend_nodes):
        node_ip = get_hostname_ip(node, runtime.network_interface)
        node_metadata = {
            "frontend_index": str(frontend_index),
            "hostname": node,
        }
        node_metadata.update(telemetry.extra_metadata)
        endpoints.append(
            TelemetryEndpoint(
                name=f"frontend{frontend_index}",
                url=f"http://{node_ip}:{frontend_topology.frontend_port}/metrics",
                frequency=telemetry.default_frequency,
                filter="frontend",
                node_metadata=node_metadata,
            )
        )

    fpm_config: dict[str, object] | None = None
    if telemetry.forward_pass_metrics.enabled:
        publishers_by_mode = {
            mode: [process for process in processes if process.endpoint_mode == mode and process.fpm_publisher]
            for mode in ("prefill", "decode", "agg")
        }
        expected_workers: dict[str, int] = {}
        component_roles: dict[str, str] = {}
        if publishers_by_mode["prefill"]:
            expected_workers["prefill"] = len(publishers_by_mode["prefill"])
            component_roles["prefill"] = "prefill"
        backend_publishers = publishers_by_mode["decode"] + publishers_by_mode["agg"]
        if backend_publishers:
            expected_workers["backend"] = len(backend_publishers)
            component_roles["backend"] = "decode" if publishers_by_mode["decode"] else "agg"

        metadata = {"job_id": runtime.job_id, "run_name": runtime.run_name}
        metadata.update(telemetry.extra_metadata)
        fpm_config = {
            "trace_path": f"/logs/{telemetry.storage_subdir}/fpm/dynamo-fpm",
            "manifest_path": f"/logs/{telemetry.storage_subdir}/fpm_manifest.json",
            "expected_workers": expected_workers,
            "component_roles": component_roles,
            "metadata": metadata,
        }

    event_streams_config: dict[str, object] | None = None
    if telemetry.kv_cache_events.enabled:
        sources: list[dict[str, object]] = []
        for process in sorted(
            (process for process in processes if process.kv_events_publisher),
            key=lambda process: (
                process.endpoint_mode,
                process.endpoint_index,
                process.node_rank,
                process.node,
            ),
        ):
            if process.kv_events_port is None:
                raise ValueError("KV-cache event recording enabled but a publisher has no allocated port")
            node_ip = get_hostname_ip(process.node, runtime.network_interface)
            metadata = {
                "hostname": process.node,
                "job_id": runtime.job_id,
                "run_name": runtime.run_name,
                "worker_index": str(process.endpoint_index),
                "worker_process": str(process.node_rank),
                "worker_role": process.endpoint_mode,
                "dp_rank": str(process.node_rank),
            }
            metadata.update(telemetry.extra_metadata)
            sources.append(
                {
                    "name": (
                        f"vllm_{process.endpoint_mode}{process.endpoint_index}_rank{process.node_rank}_{process.node}"
                    ),
                    "transport": "zmq",
                    "codec": "vllm_kv_events_v1",
                    "endpoint": f"tcp://{node_ip}:{process.kv_events_port}",
                    "topic": telemetry.kv_cache_events.topic,
                    "metadata": metadata,
                }
            )
        if not sources:
            raise ValueError("KV-cache event recording enabled but no vLLM publisher processes were found")
        event_streams_config = {
            "trace_dir": f"/logs/{telemetry.storage_subdir}/kv-events",
            "manifest_path": f"/logs/{telemetry.storage_subdir}/kv_events_manifest.json",
            "ready_path": f"/logs/{telemetry.storage_subdir}/kv_events.ready",
            "roll_bytes": telemetry.kv_cache_events.jsonl_gz_roll_bytes,
            "max_segments": telemetry.kv_cache_events.max_segments,
            "ready_delay_ms": telemetry.kv_cache_events.ready_delay_ms,
            "sources": sources,
        }

    return _dump_toml(
        endpoints=endpoints,
        # Tachometer owns this new nested directory. The parent is created by
        # srtctl for FPM traces and readiness markers, while current scraper
        # builds reject an already-existing storage directory.
        storage=f"/logs/{telemetry.storage_subdir}/scraper",
        fpm=fpm_config,
        event_streams=event_streams_config,
    )


def _dump_toml(
    *,
    endpoints: list[TelemetryEndpoint],
    storage: str,
    fpm: dict[str, object] | None,
    event_streams: dict[str, object] | None,
) -> str:
    """Render a compact TOML document without extra dependencies."""
    lines = [f"storage = {json.dumps(storage)}", ""]
    for endpoint in endpoints:
        lines.append("[[endpoints]]")
        lines.append(f"name = {json.dumps(endpoint.name)}")
        lines.append(f"url = {json.dumps(endpoint.url)}")
        lines.append(f"frequency = {endpoint.frequency}")
        if endpoint.filter is not None:
            lines.append(f"filter = {json.dumps(endpoint.filter)}")
        if endpoint.node_metadata:
            lines.append("[endpoints.node_metadata]")
            for key, value in sorted(endpoint.node_metadata.items()):
                lines.append(f"{json.dumps(key)} = {json.dumps(value)}")
        if endpoint.gpu_metadata:
            lines.append("[endpoints.gpu_metadata]")
            for gpu_idx, metadata in sorted(endpoint.gpu_metadata.items(), key=lambda item: int(item[0])):
                fields = ", ".join(f"{json.dumps(k)} = {json.dumps(v)}" for k, v in sorted(metadata.items()))
                lines.append(f"{json.dumps(gpu_idx)} = {{ {fields} }}")
        lines.append("")

    if fpm is not None:
        lines.append("[fpm]")
        lines.append(f"trace_path = {json.dumps(fpm['trace_path'])}")
        lines.append(f"manifest_path = {json.dumps(fpm['manifest_path'])}")
        for table in ("expected_workers", "component_roles", "metadata"):
            values = fpm[table]
            if not isinstance(values, dict) or not values:
                continue
            lines.append(f"[fpm.{table}]")
            for key, value in sorted(values.items()):
                lines.append(f"{json.dumps(key)} = {json.dumps(value)}")
        lines.append("")

    if event_streams is not None:
        lines.append("[event_streams]")
        for key in (
            "trace_dir",
            "manifest_path",
            "ready_path",
            "roll_bytes",
            "max_segments",
            "ready_delay_ms",
        ):
            lines.append(f"{key} = {json.dumps(event_streams[key])}")
        lines.append("")
        sources = event_streams["sources"]
        if not isinstance(sources, list):
            raise TypeError("event_streams.sources must be a list")
        for source in sources:
            if not isinstance(source, dict):
                raise TypeError("event_streams source must be a dictionary")
            source_values = cast(dict[str, object], source)
            lines.append("[[event_streams.sources]]")
            for key in ("name", "transport", "codec", "endpoint", "topic"):
                lines.append(f"{key} = {json.dumps(source_values[key])}")
            metadata = source_values.get("metadata")
            if isinstance(metadata, dict) and metadata:
                lines.append("[event_streams.sources.metadata]")
                for key, value in sorted(metadata.items()):
                    lines.append(f"{json.dumps(key)} = {json.dumps(value)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"

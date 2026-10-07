# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required KV-event capture, independent of metrics capture and its load window."""

from __future__ import annotations

import base64
import json
import shlex
import time
from pathlib import PurePosixPath

import tomli
from marshmallow import ValidationError

from srtctl.backends.vllm import VLLMProtocol
from srtctl.core.slurm import get_hostname_ip
from srtctl.services.registry import ServiceKind, register_service


def publisher_sources(config, processes, runtime, metadata):
    """Subscribe to backend-resolved bindings without inferring the DP topology."""
    backend = config.backend
    sources = []
    for process in sorted(processes, key=lambda p: (p.endpoint_mode, p.endpoint_index, p.node_rank, p.node)):
        if process.engine_id:
            continue
        for publisher in backend.kv_event_publishers(process):
            rank = publisher.dp_rank
            host = get_hostname_ip(publisher.node, runtime.network_interface)
            sources.append(
                {
                    "name": f"vllm_{process.endpoint_mode}{process.endpoint_index}_rank{rank}_{publisher.node}",
                    "transport": "zmq",
                    "codec": "vllm_kv_events_v1",
                    "endpoint": f"tcp://{host}:{publisher.port}",
                    "topic": (backend.get_kv_events_config_for_mode(process.endpoint_mode) or {}).get(
                        "topic", "kv-events"
                    ),
                    "metadata": {
                        **metadata,
                        "hostname": publisher.node,
                        "job_id": runtime.job_id,
                        "run_name": runtime.run_name,
                        "worker_index": str(process.endpoint_index),
                        "worker_process": str(process.node_rank),
                        "worker_role": process.endpoint_mode,
                        "dp_rank": str(rank),
                    },
                }
            )
    if not sources:
        raise ValueError("KV-event recording requested without any publishers")
    return sources


def recorder_toml(service, ctx):
    options = service.options
    subdir = options.get("storage_subdir", "telemetry")
    root = str(ctx.runtime.container_log_dir / subdir)
    sources = publisher_sources(ctx.config, ctx.processes, ctx.runtime, options.get("extra_metadata", {}))
    lines = [f"storage = {json.dumps(root + '/kv-recorder')}", "endpoints = []", "", "[event_streams]"]
    values = {
        "trace_dir": root + "/kv-events",
        "manifest_path": root + "/kv_events_manifest.json",
        "ready_path": root + "/kv_events.ready",
        "roll_bytes": options.get("jsonl_gz_roll_bytes", 268435456),
        "max_segments": options.get("max_segments", 64),
        "ready_delay_ms": options.get("ready_delay_ms", 1000),
    }
    lines.extend(f"{key} = {json.dumps(value)}" for key, value in values.items())
    for source in sources:
        lines.extend(("", "[[event_streams.sources]]"))
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in source.items() if key != "metadata")
        lines.append("[event_streams.sources.metadata]")
        lines.extend(f"{json.dumps(key)} = {json.dumps(value)}" for key, value in sorted(source["metadata"].items()))
    return "\n".join(lines) + "\n"


@register_service("kv-events")
class KVEventsService(ServiceKind):
    builds_command = True
    default_critical = True
    shutdown_tier = 1  # workers publish their last evictions before subscribers drain
    terminate_timeout = 120.0
    option_keys = (
        "binary_path",
        "storage_subdir",
        "extra_metadata",
        "topic",
        "jsonl_gz_roll_bytes",
        "max_segments",
        "ready_delay_ms",
        "ready_timeout_secs",
    )

    def validate(self, service, config):
        from srtctl.frontends import get_frontend

        if not isinstance(config.backend, VLLMProtocol) or get_frontend(config.frontend.type).worker_launch != "dynamo":
            raise ValidationError("kv-events requires a Dynamo vLLM deployment")
        if service.effective_placement != "head" or service.effective_per != "node" or service.nodes is not None:
            raise ValidationError("kv-events runs one recorder on the head node")
        if service.effective_start != "after_frontend" or not service.effective_critical:
            raise ValidationError("kv-events must start after_frontend and be critical")
        subdir = PurePosixPath(service.options.get("storage_subdir", "telemetry"))
        if subdir.is_absolute() or ".." in subdir.parts or str(subdir) == ".":
            raise ValidationError("kv-events storage_subdir must be a nonempty relative directory")
        for key, default in (("ready_timeout_secs", 600), ("jsonl_gz_roll_bytes", 268435456), ("max_segments", 64)):
            value = service.options.get(key, default)
            if type(value) is not int or value <= 0:
                raise ValidationError(f"kv-events {key} must be a positive integer")
        for role in config.roles.values():
            if not role.kv_events:
                raise ValidationError("kv-events requires every worker role to enable kv_events")
        if config.backend.failover is not None or config.dynamo.sidecar:
            raise ValidationError("kv-events does not support shadow engines or sidecars")

    def build_command(self, service, ctx):
        subdir = service.options.get("storage_subdir", "telemetry")
        log_dir = getattr(ctx.runtime, "container_log_dir", PurePosixPath("/logs"))
        return [
            service.options.get("binary_path", "/usr/local/bin/tachometer-scraper"),
            "--config",
            str(log_dir / "kv_events_config.toml"),
            "--local-dir",
            str(log_dir / subdir / "kv-recorder" / "local"),
            # The pinned recorder waits on scrape/sync tasks. Keep its ordinary
            # sync task alive when there are no Prometheus endpoints; otherwise
            # it exits as soon as subscribers report ready.
            "--sync-interval",
            "30",
        ]

    def preamble(self, service, ctx):
        if ctx.config is None:  # preview has no allocated topology
            return None
        encoded = base64.b64encode(recorder_toml(service, ctx).encode()).decode()
        root = ctx.runtime.container_log_dir
        subdir = service.options.get("storage_subdir", "telemetry")
        return (
            f"mkdir -p {shlex.quote(str(root / subdir))}; "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(str(root / 'kv_events_config.toml'))}"
        )

    def wait_fleet_ready(self, service, runtime, procs):
        ready = runtime.log_dir / service.options.get("storage_subdir", "telemetry") / "kv_events.ready"
        deadline = time.monotonic() + service.options.get("ready_timeout_secs", 600)
        while time.monotonic() < deadline:
            if any(proc.popen.poll() is not None for proc in procs):
                raise RuntimeError("KV-event recorder exited before subscriber readiness")
            try:
                marker = json.loads(ready.read_text())
                if marker.get("ready") is True:
                    with (runtime.log_dir / "kv_events_config.toml").open("rb") as stream:
                        expected = {s["name"] for s in tomli.load(stream)["event_streams"]["sources"]}
                    observed = marker.get("sources", [])
                    if not isinstance(observed, list) or len(observed) != len(expected) or set(observed) != expected:
                        raise RuntimeError("KV-event readiness does not cover the allocated publisher fleet")
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.2)
        raise RuntimeError("KV-event subscribers did not become ready")

    def finalize(self, service, runtime):
        manifest = runtime.log_dir / service.options.get("storage_subdir", "telemetry") / "kv_events_manifest.json"
        data = json.loads(manifest.read_text())
        with (runtime.log_dir / "kv_events_config.toml").open("rb") as stream:
            expected = {source["name"] for source in tomli.load(stream)["event_streams"]["sources"]}
        sources = data.get("sources", [])
        if (
            data.get("complete") is not True
            or not sources
            or data.get("expected_sources") != len(sources)
            or data.get("observed_sources") != len(sources)
        ):
            raise RuntimeError("KV-event manifest is incomplete")
        names = [source.get("name") for source in sources]
        if len(set(names)) != len(names) or set(names) != expected:
            raise RuntimeError("KV-event manifest does not cover the allocated publisher fleet")
        for source in sources:
            if source.get("fatal_error") or any(
                source.get(key) != 0 for key in ("sequence_gaps", "invalid_frames", "decode_errors")
            ):
                raise RuntimeError(f"KV-event source failed: {source.get('name')}")
            if any(
                type(source.get(key)) is not int or source[key] <= 0 for key in ("received_events", "received_batches")
            ) or not source.get("trace_files"):
                raise RuntimeError(f"KV-event source has no captured events: {source.get('name')}")
            for trace_file in source["trace_files"]:
                relative = PurePosixPath(trace_file).relative_to(runtime.container_log_dir)
                trace = runtime.log_dir / relative
                if ".." in relative.parts or not trace.is_file() or trace.stat().st_size == 0:
                    raise RuntimeError("KV-event trace file is missing or outside the job output directory")

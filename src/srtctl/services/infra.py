# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``type: etcd`` and ``type: nats``: the Dynamo discovery plane as services.

Both are implied by ``frontend.type: dynamo`` (see :mod:`srtctl.services.implicit`)
and placed on the infra node, which is the head node unless one of them asks for
``placement.node: dedicated``. Declaring either by name takes over the default:
change the container, the placement, NATS's payload limit, or point at an
already-running instance with ``external``.

The binaries come from the job container's ``/configs`` mount, as before
(``/configs/etcd``, ``/configs/nats-server``); data lives on the node-local
``/tmp`` so Raft and JetStream never touch network storage.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from marshmallow import ValidationError

from srtctl.core.job_ports import runtime_port
from srtctl.ports import ETCD_CLIENT_PORT, ETCD_PEER_PORT, NATS_PORT
from srtctl.services.config import ServiceReadinessConfig, TcpProbe
from srtctl.services.registry import ServiceKind, ServiceLaunchContext, register_service

if TYPE_CHECKING:
    from srtctl.core.schema import SrtConfig
    from srtctl.services.config import ServiceConfig

ETCD_BINARY = "/configs/etcd"
NATS_BINARY = "/configs/nats-server"
ETCD_DATA_DIR = "/tmp/etcd"
NATS_STORE_DIR = "/tmp/nats"
NATS_CONFIG_PATH = "/tmp/nats.conf"
INFRA_READINESS_TIMEOUT = 300  # slow container imports on first use


class _InfraKind(ServiceKind):
    builds_command = True
    default_start = "infra"
    default_critical = True
    default_placement = "infra"
    default_readiness_timeout = INFRA_READINESS_TIMEOUT
    supports_dedicated = True
    supports_external = True

    def readiness(self, service, ctx):
        if getattr(ctx.runtime, "job_ports", None) is None:
            return None
        name = "etcd-client" if self.type_name == "etcd" else "nats"
        return ServiceReadinessConfig(
            tcp=TcpProbe(port=ctx.runtime.job_ports.fixed(name)), timeout_seconds=INFRA_READINESS_TIMEOUT
        )

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        if service.effective_placement not in ("head", "infra", "dedicated"):
            raise ValidationError(
                f"services[{service.name}] (type {self.type_name}) must be placed on head, infra, or dedicated; "
                f"got {service.effective_placement!r}. The discovery plane lives on the infra node."
            )


@register_service("etcd")
class EtcdService(_InfraKind):
    """etcd for Dynamo worker discovery; advertises the node's own IP so every other node can reach it."""

    default_readiness_ports = (ETCD_CLIENT_PORT,)

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        client_port = runtime_port(ctx.runtime, "etcd-client", ETCD_CLIENT_PORT)
        data_dir = _job_path(ctx, ETCD_DATA_DIR)
        peer_args = []
        if getattr(ctx.runtime, "job_ports", None) is not None:
            peer = runtime_port(ctx.runtime, "etcd-peer", ETCD_PEER_PORT)
            peer_args = [
                "--listen-peer-urls",
                f"http://0.0.0.0:{peer}",
                "--initial-advertise-peer-urls",
                f"http://{ctx.node_ip}:{peer}",
                "--initial-cluster",
                f"default=http://{ctx.node_ip}:{peer}",
            ]
        return [
            ETCD_BINARY,
            "--data-dir",
            data_dir,
            "--listen-client-urls",
            f"http://0.0.0.0:{client_port}",
            "--advertise-client-urls",
            f"http://{ctx.node_ip}:{client_port}",  # a reachable IP, never 0.0.0.0
            *peer_args,
            *service.args,
        ]

    def preamble(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> str | None:
        # Fresh data dir on node-local disk: stale Raft state from an earlier job breaks startup.
        data_dir = shlex.quote(_job_path(ctx, ETCD_DATA_DIR))
        return f"rm -rf {data_dir} && mkdir -p {data_dir}"


@register_service("nats")
class NatsService(_InfraKind):
    """NATS with JetStream for the Dynamo request and event planes.

    ``options.max_payload_mb`` raises the message size limit (needed for long
    prompts on the NATS request plane).
    """

    default_readiness_ports = (NATS_PORT,)
    option_keys = ("max_payload_mb",)

    def build_command(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> list[str]:
        if service.command is not None:
            return list(service.effective_command)
        port_args = (
            ["-p", str(runtime_port(ctx.runtime, "nats", NATS_PORT))]
            if getattr(ctx.runtime, "job_ports", None) is not None
            else []
        )
        if service.options.get("max_payload_mb") is not None:
            return [NATS_BINARY, "-c", _job_path(ctx, NATS_CONFIG_PATH), *port_args, *service.args]
        return [NATS_BINARY, "-js", "-sd", _job_path(ctx, NATS_STORE_DIR), *port_args, *service.args]

    def preamble(self, service: ServiceConfig, ctx: ServiceLaunchContext) -> str | None:
        store_dir = _job_path(ctx, NATS_STORE_DIR)
        parts = [f"rm -rf {shlex.quote(store_dir)} && mkdir -p {shlex.quote(store_dir)}"]
        max_payload_mb = service.options.get("max_payload_mb")
        if max_payload_mb is not None:
            max_payload_bytes = int(max_payload_mb) * 1024 * 1024
            conf = f'max_payload: {max_payload_bytes}\njetstream {{ store_dir: "{store_dir}" }}\n'
            parts.append(f"printf %s {shlex.quote(conf)} > {shlex.quote(_job_path(ctx, NATS_CONFIG_PATH))}")
        return " && ".join(parts)

    def validate(self, service: ServiceConfig, config: SrtConfig) -> None:
        super().validate(service, config)
        max_payload_mb = service.options.get("max_payload_mb")
        if max_payload_mb is not None and (not isinstance(max_payload_mb, int) or max_payload_mb <= 0):
            raise ValidationError(f"services[{service.name}].options.max_payload_mb must be a positive integer")


def _job_path(ctx, default):
    if getattr(ctx.runtime, "job_ports", None) is not None:
        from pathlib import PurePosixPath

        return f"/tmp/srtctl-{ctx.runtime.job_id}/{PurePosixPath(default).name}"
    return default

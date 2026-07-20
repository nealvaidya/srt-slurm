# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-scoped runtime port allocation.

Slurm can place several non-exclusive jobs on the same host.  Pyxis uses host
networking, so using the same fixed listener ports in every job makes otherwise
independent allocations contend.  ``PortPlan`` shifts all runtime port families
by one deterministic job slot while keeping every producer and consumer on the
same plan.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.core.topology import NodePortAllocator


# FPM reserves 1,024 ports per slot starting at 20,380. Twenty-eight slots
# keep the complete highest-slot reservation within Dynamo's registered
# user-port ceiling (49,151).
PORT_SLOT_COUNT = 28
PORT_SLOT_STRIDE = 128


@dataclass(frozen=True)
class PortPlan:
    """Deterministic listener ports for one Slurm job.

    Numeric Slurm job IDs are sequential, so modulo slotting prevents the
    common case of concurrent neighboring jobs selecting the same ports.  The
    slot can be overridden with ``SRTCTL_PORT_SLOT`` when a site needs to break
    a rare modulo collision deliberately.
    """

    slot: int

    def __post_init__(self) -> None:
        if not 0 <= self.slot < PORT_SLOT_COUNT:
            raise ValueError(f"Port slot must be in [0, {PORT_SLOT_COUNT}), got {self.slot}")

    @classmethod
    def default(cls) -> PortPlan:
        """Return the legacy fixed-port plan used outside a Slurm job."""
        return cls(slot=0)

    @classmethod
    def from_job_id(cls, job_id: str) -> PortPlan:
        """Build a non-legacy plan from a Slurm job ID.

        Slot zero preserves the fixed ports used before job-scoped allocation
        and remains available through :meth:`default` or an explicit override.
        Slurm jobs avoid that slot because those conventional ports are more
        likely to be occupied by workloads that do not use ``PortPlan``.
        """
        override = os.environ.get("SRTCTL_PORT_SLOT")
        if override is not None:
            try:
                slot = int(override)
            except ValueError as exc:
                raise ValueError(f"Invalid SRTCTL_PORT_SLOT={override!r}") from exc
            return cls(slot=slot)

        match = re.search(r"\d+", job_id)
        if match is None:
            raise ValueError(f"Cannot derive a port slot from job ID {job_id!r}")
        nonlegacy_slot_count = PORT_SLOT_COUNT - 1
        return cls(slot=(int(match.group()) % nonlegacy_slot_count) + 1)

    @property
    def offset(self) -> int:
        return self.slot * PORT_SLOT_STRIDE

    @property
    def etcd_client_port(self) -> int:
        return 2379 + self.offset

    @property
    def etcd_peer_port(self) -> int:
        return 2380 + self.offset

    @property
    def nats_port(self) -> int:
        return 4222 + self.offset

    @property
    def kv_events_port_base(self) -> int:
        return 5550 + self.offset

    @property
    def nixl_port_base(self) -> int:
        return 6550 + self.offset

    @property
    def frontend_public_port(self) -> int:
        return 8000 + self.offset

    @property
    def dyn_system_port_base(self) -> int:
        return 8081 + self.offset

    @property
    def frontend_internal_port(self) -> int:
        return 8180 + self.offset

    @property
    def kvbm_zmq_port_base(self) -> int:
        return 56001 + (self.slot * 32)

    @property
    def vllm_data_parallel_rpc_port_base(self) -> int:
        return 13345 + self.offset

    @property
    def sglang_dist_init_port_base(self) -> int:
        return 29500 + self.offset

    @property
    def http_port_base(self) -> int:
        return 30000 + self.offset

    @property
    def bootstrap_port_base(self) -> int:
        return 31000 + self.offset

    @property
    def fpm_port_base(self) -> int:
        # Each colocated process reserves 128 ports for Dynamo DP-rank offsets.
        return 20380 + (self.slot * 1024)

    def node_port_allocator(self) -> NodePortAllocator:
        """Create the per-node allocator whose bases belong to this plan."""
        from srtctl.core.topology import NodePortAllocator

        return NodePortAllocator(
            base_http_port=self.http_port_base,
            base_bootstrap_port=self.bootstrap_port_base,
            base_kv_events_port=self.kv_events_port_base,
            base_nixl_port=self.nixl_port_base,
            base_fpm_port=self.fpm_port_base,
        )

    def to_dict(self) -> dict[str, int]:
        """Return a durable, human-readable representation of the plan."""
        values = asdict(self)
        for name in (
            "offset",
            "etcd_client_port",
            "etcd_peer_port",
            "nats_port",
            "kv_events_port_base",
            "nixl_port_base",
            "frontend_public_port",
            "dyn_system_port_base",
            "frontend_internal_port",
            "kvbm_zmq_port_base",
            "vllm_data_parallel_rpc_port_base",
            "sglang_dist_init_port_base",
            "http_port_base",
            "bootstrap_port_base",
            "fpm_port_base",
        ):
            values[name] = getattr(self, name)
        return values

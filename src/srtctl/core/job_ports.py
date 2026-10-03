# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded port slots for co-located CLU Dynamo/vLLM jobs.

Slots are deterministic, not an inter-job lease. Explicit SRTCTL_PORT_SLOT can
separate concurrent jobs whose numeric IDs select the same slot.
"""

import os
import re
from dataclasses import dataclass

from srtctl import ports
from srtctl.core.topology import NodePortAllocator

SLOT_COUNT = 16
FIXED_BASE = 9000
WORKER_BASE = 12000
FIXED_KINDS = (
    "etcd-client",
    "etcd-peer",
    "nats",
    "frontend",
    "frontend-internal",
    "dcgm-exporter",
    "node-exporter",
    "process-exporter",
)
WORKER_KINDS = (
    ports.SYS_PORTS,
    ports.HTTP_PORTS,
    ports.BOOTSTRAP_PORTS,
    ports.KV_EVENTS_PORTS,
    ports.NIXL_PORTS,
    ports.DP_RPC_PORTS,
    ports.KVBM_ZMQ_PORTS,
    ports.SIDECAR_GRPC_PORTS,
    ports.NCCL_PORTS,
    ports.DIST_INIT_PORTS,
    ports.VLLM_SCAN_PORTS,
    ports.MORIIO_HANDSHAKE_PORTS,
    ports.MORIIO_NOTIFY_PORTS,
    ports.TRTLLM_DIST_INIT_PORTS,
)


@dataclass(frozen=True)
class JobPortPlan:
    slot: int

    def __post_init__(self):
        if type(self.slot) is not int or not 1 <= self.slot < SLOT_COUNT:
            raise ValueError(f"port slot must be an integer in 1..{SLOT_COUNT - 1}")

    @classmethod
    def from_job_id(cls, job_id):
        override = os.environ.get("SRTCTL_PORT_SLOT")
        if override is not None:
            return cls(int(override))
        match = re.fullmatch(r"\d+", str(job_id))
        if match is None:
            raise ValueError("job-scoped ports require a numeric Slurm job ID")
        return cls(int(job_id) % (SLOT_COUNT - 1) + 1)

    def fixed(self, name):
        return FIXED_BASE + FIXED_KINDS.index(name) * SLOT_COUNT + self.slot

    def worker_ranges(self):
        ranges = {}
        base = WORKER_BASE
        for kind in WORKER_KINDS:
            width = 512 if kind in (ports.HTTP_PORTS, ports.VLLM_SCAN_PORTS) else 64
            ranges[kind.name] = (base + self.slot * width, width)
            base += SLOT_COUNT * width
        if base > 49151:
            raise ValueError("job port reservation exceeds the user-port ceiling")
        return ranges

    def allocator(self):
        ranges = self.worker_ranges()
        return NodePortAllocator(
            bases={key: value[0] for key, value in ranges.items()},
            limits={key: value[0] + value[1] for key, value in ranges.items()},
        )

    def to_dict(self):
        return {
            "slot": self.slot,
            "fixed": {name: self.fixed(name) for name in FIXED_KINDS},
            "worker_ranges": {key: {"base": base, "size": size} for key, (base, size) in self.worker_ranges().items()},
        }


def runtime_port(runtime, name, default):
    plan = getattr(runtime, "job_ports", None)
    return plan.fixed(name) if plan is not None else default

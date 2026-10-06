# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded port slots for co-located CLU Dynamo/vLLM jobs.

Slots are deterministic. A node-local lease rejects overlapping slots owned by
the same user before infrastructure starts; it does not reassign locked ports.
"""

import fcntl
import os
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from srtctl import ports
from srtctl.core.topology import NodePortAllocator

SLOT_COUNT = 16
FIXED_BASE = 10000
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

    @contextmanager
    def lease(self, job_id: str, *, directory: Path | None = None) -> Iterator[None]:
        """Hold the slot through process cleanup; kernel exit also releases it.

        The directory must be node-local, never the shared job output directory.
        Files are retained to avoid unlinking a lock another process has opened.
        """
        directory = directory or Path(f"/tmp/srtctl-port-slots-{os.getuid()}")
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not directory.is_dir() or directory.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError(f"port-slot lease directory must be private and owned by this user: {directory}")
        fd = os.open(directory / f"slot-{self.slot}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                owner = lock.read().strip() or "unknown job"
                raise RuntimeError(
                    f"port slot {self.slot} is already leased by {owner}; choose SRTCTL_PORT_SLOT"
                ) from exc
            try:
                # Avoid attaching to another user's discovery plane already
                # listening in this slot. Other users do not share our lease.
                for name in FIXED_KINDS:
                    with socket.socket() as probe:
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        try:
                            probe.bind(("0.0.0.0", self.fixed(name)))
                            probe.listen(1)
                        except OSError as exc:
                            raise RuntimeError(
                                f"port slot {self.slot}: {name} port {self.fixed(name)} unavailable"
                            ) from exc
                lock.seek(0)
                lock.truncate()
                lock.write(f"job {job_id}\n")
                lock.flush()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def runtime_port(runtime, name, default):
    plan = getattr(runtime, "job_ports", None)
    return plan.fixed(name) if plan is not None else default

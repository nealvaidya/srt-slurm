# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded port slots for co-located CLU Dynamo/vLLM jobs.

The job ID supplies a preferred slot. Node-local guards coordinate the selected
slot, and launchers request the managed listeners they actually start.
"""

import fcntl
import os
import re
import socket
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from srtctl import ports
from srtctl.core.topology import NodePortAllocator

SLOT_COUNT = 16
FIXED_BASE = 10128
SERVICE_SLOT_WIDTH = 120
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
class PortRequest:
    """A managed listener, or a best-effort engine scan hint."""

    name: str
    owner: str
    node: str | None
    port: int
    size: int = 1
    transport: str = "tcp"
    bind: str = "0.0.0.0"
    enforced: bool = True


class PortConflict(RuntimeError):
    """An allocation was rejected before its consumer could start."""


@dataclass(frozen=True)
class JobPortPlan:
    slot: int
    assignments: dict[str, int] = field(default_factory=dict, compare=False)
    requests: dict[tuple[str, str | None], PortRequest] = field(default_factory=dict, compare=False)
    rejected: set[int] = field(default_factory=set, compare=False)
    # Installed by the controller after the node-local slot guards are ready.
    check: Callable[[], None] | None = field(default=None, compare=False, repr=False)

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

    @classmethod
    def candidates(cls, job_id: str) -> Iterator["JobPortPlan"]:
        first = cls.from_job_id(job_id)
        yield first
        if "SRTCTL_PORT_SLOT" not in os.environ:
            for offset in range(1, SLOT_COUNT - 1):
                yield cls((first.slot - 1 + offset) % (SLOT_COUNT - 1) + 1)

    def fixed(self, name: str) -> int:
        if name not in self.assignments:
            start = FIXED_BASE + (self.slot - 1) * SERVICE_SLOT_WIDTH
            preferred = start + FIXED_KINDS.index(name) if name in FIXED_KINDS else start
            used = set(self.assignments.values()) | self.rejected
            candidates = [preferred, *range(start, start + SERVICE_SLOT_WIDTH)]
            available = next((port for port in candidates if port not in used), None)
            if available is None:
                raise PortConflict(f"service port pool exhausted in slot {self.slot}")
            self.assignments[name] = available
        return self.assignments[name]

    def request(self, name: str, node: str, *, owner: str) -> int:
        port = self.fixed(name)
        self.record(PortRequest(name=name, owner=owner, node=node, port=port))
        return port

    def record(self, request: PortRequest) -> None:
        key = (request.name, request.node)
        previous = self.requests.get(key)
        if previous is not None and previous.owner != request.owner:
            raise ValueError(f"port request {request.name!r} on {request.node} has two owners")
        if previous == request:
            if self.check is not None:
                self.check()
            return
        self.requests[key] = request
        if self.check is not None:
            self.check()

    def reassign(self, owner: str) -> bool:
        """Replace a failed service's allocations; callers stop its whole fleet first."""
        names = {request.name for request in self.requests.values() if request.owner == owner}
        if not names:
            return False
        for name in names:
            self.rejected.add(self.assignments.pop(name))
        for key, request in list(self.requests.items()):
            if request.owner == owner:
                del self.requests[key]
        for name in sorted(names):
            self.fixed(name)
        return True

    def record_worker(self, kind: ports.PortKind, node: str | None, base: int, span: int) -> None:
        self.record(
            PortRequest(
                name=f"{kind.name}:{base}",
                owner="workers",
                node=node,
                port=base,
                size=span,
                enforced=kind.bounded,
            )
        )

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
            on_allocate=self.record_worker,
        )

    def to_dict(self):
        for name in FIXED_KINDS:
            self.fixed(name)
        return {
            "schema_version": 2,
            "slot": self.slot,
            "fixed": dict(self.assignments),
            "requests": [asdict(request) for request in self.requests.values()],
            "worker_ranges": {key: {"base": base, "size": size} for key, (base, size) in self.worker_ranges().items()},
        }

    @contextmanager
    def lease(self, job_id: str, *, directory: Path | None = None) -> Generator[None]:
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


def runtime_port(runtime, name, default, *, node: str | None = None, owner: str | None = None):
    plan = getattr(runtime, "job_ports", None)
    if plan is None:
        return default
    return plan.request(name, node, owner=owner or name) if node is not None else plan.fixed(name)

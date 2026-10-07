# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate node-local port leases through the existing Slurm launch path."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.core.job_ports import PortConflict
from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process
from srtctl.runtime_scripts.port_guard import write_json

if TYPE_CHECKING:
    from srtctl.core.processes import ProcessRegistry
    from srtctl.core.runtime import RuntimeContext

logger = logging.getLogger(__name__)
GUARD_TIMEOUT_SECONDS = 60


class PortLeaseManager:
    """A job's helpers keep cooperating jobs out of its slot until cleanup."""

    def __init__(self, runtime: RuntimeContext):
        assert runtime.job_ports is not None
        self.runtime = runtime
        self.plan = runtime.job_ports
        nodes = runtime.nodes
        self.nodes = list(dict.fromkeys((nodes.head, nodes.infra, nodes.bench, *nodes.compute)))
        self.directory = runtime.log_dir / "port-reservations" / f"slot-{self.plan.slot}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.processes: list[ManagedProcess] = []
        self.revision = 0

    def _paths(self, index: int) -> tuple[Path, Path]:
        return self.directory / f"node-{index}.control.json", self.directory / f"node-{index}.status.json"

    def _publish(self) -> None:
        for index, node in enumerate(self.nodes):
            control, _ = self._paths(index)
            requests = [
                asdict(r)
                for r in self.plan.requests.values()
                if r.node == node or (r.node is None and node in self.runtime.nodes.worker)
            ]
            write_json(control, {"revision": self.revision, "requests": requests})
        payload = self.plan.to_dict()
        payload.update(job_id=self.runtime.job_id, nodes=self.nodes, revision=self.revision)
        write_json(self.runtime.log_dir / "port_plan.json", payload)

    def _wait(self) -> None:
        deadline = time.monotonic() + GUARD_TIMEOUT_SECONDS
        pending = set(range(len(self.nodes)))
        while pending:
            for index in list(pending):
                _, status = self._paths(index)
                if status.exists():
                    result = json.loads(status.read_text())
                    if result["revision"] in (-1, self.revision):
                        if result["status"] != "ready":
                            message = f"port slot {self.plan.slot} on {self.nodes[index]}: {result['message']}"
                            with (self.runtime.log_dir / "port_allocation_attempts.jsonl").open("a") as stream:
                                stream.write(
                                    json.dumps({"node": self.nodes[index], "slot": self.plan.slot, **result}) + "\n"
                                )
                            if result["status"] == "conflict":
                                raise PortConflict(message)
                            raise RuntimeError(message)
                        if self.processes[index].is_running:
                            pending.remove(index)
                            continue
                if not self.processes[index].is_running:
                    raise RuntimeError(
                        f"port guard exited on {self.nodes[index]}; see {self.processes[index].log_file}"
                    )
            if time.monotonic() >= deadline:
                raise RuntimeError(f"port guards timed out on {[self.nodes[index] for index in pending]}")
            if pending:
                time.sleep(0.05)

    def acquire(self, registry: ProcessRegistry) -> None:
        self._publish()
        script = Path(__file__).resolve().parents[1] / "runtime_scripts" / "port_guard.py"
        try:
            for index, node in enumerate(self.nodes):
                control, status = self._paths(index)
                status.unlink(missing_ok=True)
                name = f"port_guard_{node}_{self.plan.slot}"
                log = self.directory / f"node-{index}.out"
                popen = start_srun_process(
                    command=[
                        "python3",
                        str(script),
                        "--slot",
                        str(self.plan.slot),
                        "--job",
                        self.runtime.job_id,
                        "--control",
                        str(control),
                        "--status",
                        str(status),
                    ],
                    nodelist=[node],
                    output=str(log),
                    container_image=None,
                    cpus_per_task=1,
                    srun_options=self.runtime.srun_options,
                    het_group=self.runtime.nodes.het_group_for(node),
                    step_name=name,
                )
                process = ManagedProcess(
                    name=name, popen=popen, log_file=log, node=node, step_name=name, critical=False, shutdown_tier=3
                )
                self.processes.append(process)
                registry.add_process(process)
            self._wait()
        except BaseException:
            self.close()
            raise
        for process in self.processes:
            process.critical = True
        object.__setattr__(self.plan, "check", self.check)

    def check(self) -> None:
        self.revision += 1
        self._publish()
        self._wait()

    def close(self) -> None:
        object.__setattr__(self.plan, "check", None)
        for process in self.processes:
            process.critical = False
        for index in range(len(self.processes)):
            control, _ = self._paths(index)
            write_json(control, {"stop": True})
        for process in self.processes:
            process.terminate(timeout=5)


def service_bind_conflict(processes: list[ManagedProcess]) -> bool:
    """Recognize bind failures only in exited processes, never from a live warning."""
    import re

    pattern = re.compile(r"address already in use|EADDRINUSE|(?:errno|os error)\s*[\[:]?\s*98", re.IGNORECASE)
    for process in processes:
        if process.exit_code in (None, 0) or process.log_file is None:
            continue
        try:
            with process.log_file.open("rb") as stream:
                stream.seek(max(0, process.log_file.stat().st_size - 65536))
                if pattern.search(stream.read().decode(errors="replace")):
                    return True
        except FileNotFoundError:
            continue
    return False

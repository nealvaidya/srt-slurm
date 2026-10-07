# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare-host slot guard. Uses only the standard library; no host srtctl install."""

import argparse
import errno
import fcntl
import json
import os
import socket
import time
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value) + "\n")
    temporary.replace(path)


def probe(request: dict) -> None:
    family = socket.AF_INET6 if ":" in request["bind"] else socket.AF_INET
    transport = socket.SOCK_DGRAM if request["transport"] == "udp" else socket.SOCK_STREAM
    for port in range(request["port"], request["port"] + request["size"]):
        with socket.socket(family, transport) as sock:
            sock.bind((request["bind"], port))
            if transport == socket.SOCK_STREAM:
                sock.listen(1)


def guard(slot: int, job: str, control: Path, status: Path, directory: Path | None = None) -> None:
    directory = directory or Path(f"/tmp/srtctl-port-slots-{os.getuid()}")
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not directory.is_dir() or directory.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"port-slot lease directory must be private and owned by this user: {directory}")
    fd = os.open(directory / f"slot-{slot}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write_json(status, {"revision": -1, "status": "conflict", "message": lock.read().strip() or "slot leased"})
            return
        lock.seek(0)
        lock.truncate()
        lock.write(f"job {job}\n")
        lock.flush()
        seen = set()
        revision = -1
        while True:
            if control.exists():
                update = json.loads(control.read_text())
                if update.get("stop"):
                    return
                if update["revision"] != revision:
                    revision = update["revision"]
                    result = {"revision": revision, "status": "ready"}
                    for request in update["requests"]:
                        identity = json.dumps(request, sort_keys=True)
                        if identity in seen or not request["enforced"]:
                            continue
                        try:
                            probe(request)
                        except OSError as exc:
                            result.update(
                                status="conflict" if exc.errno == errno.EADDRINUSE else "error",
                                message=f"{request['owner']}: {request['bind']}:{request['port']}.."
                                f"{request['port'] + request['size'] - 1} unavailable: {exc}",
                                request=request,
                            )
                            break
                        seen.add(identity)
                    write_json(status, result)
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    try:
        guard(args.slot, args.job, args.control, args.status, args.directory)
    except Exception as exc:
        write_json(args.status, {"revision": -1, "status": "error", "message": str(exc)})
        raise


if __name__ == "__main__":
    main()

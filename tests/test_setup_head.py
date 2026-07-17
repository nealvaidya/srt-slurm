# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for job-scoped head infrastructure startup."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.setup_head import ensure_ports_available, start_etcd, start_nats


def test_ensure_ports_available_rejects_an_existing_listener():
    probe = MagicMock()
    probe.bind.side_effect = OSError("occupied")
    with (
        patch("srtctl.cli.setup_head.socket.socket", return_value=probe),
        pytest.raises(RuntimeError, match="6010"),
    ):
        ensure_ports_available([6010])

    probe.close.assert_called_once()


def test_start_nats_uses_job_port_and_state_directory(tmp_path: Path):
    binary = tmp_path / "nats-server"
    binary.touch()
    state_dir = tmp_path / "job-42"

    with patch("srtctl.cli.setup_head.subprocess.Popen", return_value=MagicMock(pid=7)) as popen:
        start_nats(str(binary), port=6010, state_dir=state_dir)

    command = popen.call_args.args[0]
    assert command == [str(binary), "-p", "6010", "-js", "-sd", str(state_dir / "nats")]
    assert (state_dir / "nats").is_dir()


def test_start_etcd_uses_job_ports_and_state_directory(tmp_path: Path):
    binary = tmp_path / "etcd"
    binary.touch()
    state_dir = tmp_path / "job-42"

    with patch("srtctl.cli.setup_head.subprocess.Popen", return_value=MagicMock(pid=8)) as popen:
        start_etcd(
            "10.0.0.1",
            str(binary),
            client_port=7010,
            peer_port=7011,
            state_dir=state_dir,
        )

    command = popen.call_args.args[0]
    assert command[command.index("--listen-client-urls") + 1] == "http://0.0.0.0:7010"
    assert command[command.index("--listen-peer-urls") + 1] == "http://0.0.0.0:7011"
    assert command[command.index("--data-dir") + 1] == str(state_dir / "etcd")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts needed by CLU's immutable runs."""

import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin
from srtctl.core.schema import ReportingConfig, S3Config
from srtctl.core.slurm import _command_for_log


class CluCompatibilityTests(unittest.TestCase):
    def test_upload_uses_job_recipe_despite_shared_cluster_config(self):
        job_s3 = S3Config(bucket="job-bucket", prefix="immutable-run", exclude=[], archive=[])
        runner = PostProcessStageMixin()
        runner.config = SimpleNamespace(reporting=ReportingConfig(s3=job_s3))
        with patch(
            "srtctl.cli.mixins.postprocess_stage.load_cluster_config",
            return_value={"reporting": {"s3": {"bucket": "other-job", "prefix": "other-run"}}},
        ) as shared:
            self.assertIs(runner._get_s3_config(), job_s3)
            shared.assert_not_called()

    def test_upload_retains_cluster_fallback(self):
        runner = PostProcessStageMixin()
        runner.config = SimpleNamespace(reporting=None)
        with patch(
            "srtctl.cli.mixins.postprocess_stage.load_cluster_config",
            return_value={"reporting": {"s3": {"bucket": "cluster-default"}}},
        ):
            self.assertEqual(runner._get_s3_config().bucket, "cluster-default")

    def test_logging_redacts_shell_quoted_credentials_without_changing_command(self):
        secret = "value with spaces and a 'quote'"
        command = ["bash", "-c", f"export AWS_SECRET_ACCESS_KEY={shlex.quote(secret)}; worker"]
        original = command.copy()
        logged = _command_for_log(command, {"AWS_SECRET_ACCESS_KEY": secret, "MODEL_ID": "model/name"})
        self.assertNotIn(secret, logged)
        self.assertNotIn("value with spaces", logged)
        self.assertIn("redacted", logged)
        self.assertEqual(command, original)

    def test_warmup_cap_is_absent_from_measured_replay(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binaries = root / "bin"
            binaries.mkdir()
            calls = root / "calls.jsonl"
            uv = binaries / "uv"
            uv.write_text("#!/bin/sh\nexit 0\n")
            uv.chmod(0o755)
            aiperf = binaries / "aiperf"
            aiperf.write_text(
                "#!/usr/bin/env python3\nimport json,os,sys\n"
                "if sys.argv[1] != '--version':\n"
                " with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            )
            aiperf.chmod(0o755)
            trace = root / "trace.jsonl"
            trace.write_text("{}\n")
            script = Path(__file__).parents[1] / "src/srtctl/benchmarks/scripts/trace-replay/bench.sh"
            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "http://localhost:8000",
                    "model",
                    str(trace),
                    "1",
                    "2000",
                    "25",
                    "/model",
                    "--endpoint-type",
                    "chat",
                ],
                env={
                    **os.environ,
                    "PATH": f"{binaries}:{os.environ['PATH']}",
                    "BASE_DIR": str(root / "logs"),
                    "CALLS": str(calls),
                    "SLURM_JOB_ID": "clu-warmup-test",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            warmup, replay = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(json.loads(warmup[-1]), {"ignore_eos": True, "max_tokens": 512})
            self.assertIn("chat", warmup)
            self.assertIn("--input-file", replay)
            self.assertFalse(any("max_tokens" in arg for arg in replay))

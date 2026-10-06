# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise OSMO submission and workflow commands without a cluster or GPU."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "osmo/run_osmo.py"


class OsmoRendererTests(unittest.TestCase):
    """Check renderer selection through the launcher and generated Bash entrypoint."""

    def launch(self, subcommand, flags, credentials=True):
        command = [
            sys.executable,
            str(LAUNCHER), subcommand, "--experiment-name", "renderer-test", "--wandb-project",
            "test", "--dry-run"
        ]
        if subcommand == "eval":
            command += ["--checkpoint", "test/model:latest"]
        if credentials:
            command += ["--image", "example/compass:test"]
        # Never use or print the developer's credentials in dry-run output.
        environment = {"PATH": os.environ["PATH"]}
        if credentials:
            environment.update(WANDB_API_KEY="test-key", HF_TOKEN="test-token")
        return subprocess.run(command + flags,
                              env=environment,
                              capture_output=True,
                              text=True,
                              check=False)

    def workflow_commands(self, submission, use_defaults=False):
        tokens = shlex.split(submission.stdout.strip().removeprefix("+ "))
        template = Path(tokens[3]).read_text()
        values = dict(
            re.findall(r"^  (\w+): (.*)$",
                       template.split("default-values:\n")[1], re.MULTILINE))
        values = {key: value.strip('"') for key, value in values.items()}
        overrides = dict(item.split("=", 1) for item in tokens[tokens.index("--set") + 1:])
        if use_defaults:
            overrides.pop("camera_renderer")
            overrides.pop("physics_backend")
        values.update(overrides)
        values["output"] = "/unused-output"
        entry = template.split("    - contents: |\n", 1)[1].split("      path: /tmp/entry.sh", 1)[0]
        entry = textwrap.dedent(entry)
        entry = re.sub(r"{{(\w+)}}", lambda match: values[match[1]], entry)
        subprocess.run(["bash", "-n"], input=entry, text=True, check=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Stub downloads, filesystem changes, and runtime launches; execute
            # the actual workflow's Bash argument construction and branching.
            for name in ("apt", "mv", "unzip", "rm", "mkdir", "ls"):
                stub = directory / name
                stub.write_text("#!/bin/sh\n" +
                                ("echo model_1.pt\n" if name == "ls" else "exit 0\n"))
                stub.chmod(0o755)
            python_stub = directory / "python"
            python_stub.write_text(f"#!{sys.executable}\nimport json, sys\n"
                                   "if 'run.py' in sys.argv:\n"
                                   "    print('RUN_ARGS=' + json.dumps(sys.argv[1:]))\n")
            python_stub.chmod(0o755)
            isaaclab_stub = directory / "isaaclab.sh"
            isaaclab_stub.write_text('#!/bin/sh\nshift\nexec python "$@"\n')
            isaaclab_stub.chmod(0o755)
            result = subprocess.run(["bash"],
                                    input=entry,
                                    text=True,
                                    capture_output=True,
                                    check=True,
                                    env={
                                        "PATH": f"{directory}:{os.environ['PATH']}",
                                        "ISAACLAB_PATH": temporary
                                    },
                                    cwd=temporary)
        return [
            json.loads(line.removeprefix("RUN_ARGS="))
            for line in result.stdout.splitlines()
            if line.startswith("RUN_ARGS=")
        ]

    def test_selection_reaches_every_runtime_command(self):
        cases = [([], "isaac_rtx", "physx"), (["--physics-backend", "physx"], "isaac_rtx", "physx"),
                 (["--camera-renderer", "ovrtx"], "ovrtx", "newton"),
                 (["--camera-renderer", "ovrtx", "--physics-backend", "newton"], "ovrtx", "newton"),
                 (["--camera-renderer", "ovrtx", "--physics-backend",
                   "ovphysx"], "ovrtx", "ovphysx")]
        for subcommand in ("train", "eval"):
            for flags, renderer, backend in cases:
                with self.subTest(subcommand=subcommand, renderer=renderer, backend=backend):
                    submission = self.launch(subcommand,
                                             flags + ["--nurec-scene", "nova_carter-galileo"])
                    self.assertEqual(submission.returncode, 0, submission.stderr)
                    commands = self.workflow_commands(submission)
                    self.assertEqual(len(commands), 2 if subcommand == "train" else 1)
                    for command in commands:
                        self.assertEqual(command[command.index("--camera-renderer") + 1], renderer)
                        self.assertEqual(command[command.index("--physics-backend") + 1], backend)
                        self.assertEqual(command[command.index("--nurec-scene") + 1],
                                         "nova_carter-galileo")
                        if renderer == "ovrtx":
                            self.assertEqual(command[command.index("--visualizer") + 1], "none")
                        else:
                            self.assertNotIn("--visualizer", command)

    def test_invalid_pairs_rejected_before_credentials_or_build(self):
        for subcommand in ("train", "eval"):
            for renderer, backend in (("isaac_rtx", "newton"), ("isaac_rtx", "ovphysx"), ("ovrtx",
                                                                                          "physx")):
                with self.subTest(subcommand=subcommand, renderer=renderer, backend=backend):
                    result = self.launch(
                        subcommand, ["--camera-renderer", renderer, "--physics-backend", backend],
                        credentials=False)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("does not support", result.stderr)
                    self.assertNotIn("WANDB_API_KEY", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_workflow_defaults_allow_existing_submitters(self):
        for subcommand in ("train", "eval"):
            with self.subTest(subcommand=subcommand):
                submission = self.launch(subcommand, [])
                for command in self.workflow_commands(submission, use_defaults=True):
                    self.assertEqual(command[command.index("--camera-renderer") + 1], "isaac_rtx")
                    self.assertNotIn("--physics-backend", command)
                    self.assertNotIn("--visualizer", command)


if __name__ == "__main__":
    unittest.main()

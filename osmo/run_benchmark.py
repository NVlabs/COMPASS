#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Submit COMPASS no-regression benchmark jobs to OSMO.

Fires one ``rl_es_eval_workflow.yaml`` submission per ``--environments`` entry,
all using the same ``--embodiment``. Re-run with different ``--embodiment``
values to cover the full embodiment x environment matrix.

Results land in W&B at ``<wandb-project-name>/bm_<embodiment>_<environment>_<experiment-name>``.
Each run logs ``eval/goal_reached_rate``, ``eval/fall_down_rate``,
``eval/total_travel_time``, and ``eval/weighted_travel_time`` -- pull those out
of W&B for the regression assessment.

Usage:
    export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>
    export WANDB_API_KEY=<your-wandb-key>
    export HF_TOKEN=<your-hf-token>

    # Default 5-scene sweep for one embodiment.
    python osmo/run_benchmark.py \\
        --experiment-name release_1.6 \\
        --wandb-project-name compass_release_1.6_benchmark \\
        --checkpoint-artifact <residual-wandb-artifact> \\
        --embodiment g1

    # Single-cell smoke (skip build/push by reusing a pre-built image).
    python osmo/run_benchmark.py \\
        --experiment-name release_1.6_smoke \\
        --wandb-project-name compass_release_1.6_benchmark \\
        --checkpoint-artifact <residual-wandb-artifact> \\
        --embodiment g1 --environments combined_single_rack \\
        --image-name nvcr.io/<org>/<team>/compass_release_1.6:<tag>
"""

import argparse
import getpass
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = Path(__file__).resolve().parent / "workflows" / "rl_es_eval_workflow.yaml"
DOCKERFILE = "docker/Dockerfile.rl"

DEFAULT_ENVIRONMENTS = [
    "simple_office",
    "warehouse_single_rack",
    "warehouse_multi_rack",
    "combined_single_rack",
    "combined_multi_rack",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a COMPASS no-regression benchmark sweep to OSMO. "
        "One rl_es_eval_workflow.yaml job is fired per --environments entry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-name",
                        "-m",
                        default="",
                        help="Pre-built docker image. If empty, build+push using "
                        "--registry-prefix.")
    parser.add_argument("--registry-prefix",
                        default=os.environ.get("COMPASS_OSMO_REGISTRY", ""),
                        help="Registry prefix for build+push (e.g. nvcr.io/<org>/<team>); "
                        "defaults to $COMPASS_OSMO_REGISTRY.")
    parser.add_argument("--experiment-name",
                        "-e",
                        type=str,
                        required=True,
                        help="Short identifier; used in workflow_name, wandb run name, image tag.")
    parser.add_argument("--checkpoint-artifact",
                        "-p",
                        type=str,
                        required=True,
                        help="wandb artifact for the residual checkpoint to benchmark.")
    parser.add_argument("--distillation-ckpt-artifact",
                        "-d",
                        type=str,
                        default=None,
                        help="Optional wandb artifact for a distillation checkpoint.")
    parser.add_argument("--wandb-project-name",
                        "-n",
                        type=str,
                        required=True,
                        help="wandb project to log into.")
    parser.add_argument("--embodiment",
                        type=str,
                        default="h1",
                        help="Embodiment to benchmark; re-run with different values for "
                        "full matrix.")
    parser.add_argument("--environments",
                        nargs="+",
                        type=str,
                        default=DEFAULT_ENVIRONMENTS,
                        help="Scenes to iterate over; one OSMO eval job per entry.")
    parser.add_argument("--pool",
                        default="",
                        help="OSMO pool to target (e.g. isaac-dev-l40-03); "
                        "if empty, uses the profile's default pool.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print the osmo workflow submit commands without executing them.")
    parser.add_argument("--prompt",
                        action="store_true",
                        help="Interactively prompt for missing credentials.")
    return parser.parse_args()


def get_credential(name: str, prompt: bool) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    if prompt:
        return getpass.getpass(f"Enter {name}: ")
    sys.stderr.write(f"ERROR: ${name} is not set. Either export it or pass --prompt.\n")
    sys.exit(2)


def build_and_push_image(experiment: str, registry_prefix: str, dry_run: bool) -> str:
    if not registry_prefix:
        sys.stderr.write("ERROR: --image-name not given and --registry-prefix is empty "
                         "(also $COMPASS_OSMO_REGISTRY).\n")
        sys.exit(2)
    image = f"{registry_prefix.rstrip('/')}/compass_{experiment}:{uuid.uuid4().hex[:8]}"
    cmds = [
        ["docker", "build", "--network=host", "-t", image, "-f", DOCKERFILE,
         str(REPO_ROOT)],
        ["docker", "push", image],
    ]
    for cmd in cmds:
        print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
        if not dry_run:
            subprocess.check_call(cmd)
    return image


def submit_workflow(set_args: dict, pool: str, dry_run: bool) -> None:
    cmd = ["osmo", "workflow", "submit", str(WORKFLOW_PATH)]
    if pool:
        cmd += ["--pool", pool]
    cmd += ["--set"] + [f"{k}={v}" for k, v in set_args.items()]
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
    if not dry_run:
        subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()

    wandb_key = get_credential("WANDB_API_KEY", args.prompt)
    hf_token = get_credential("HF_TOKEN", args.prompt)

    if args.image_name:
        image = args.image_name
    else:
        image = build_and_push_image(args.experiment_name, args.registry_prefix, args.dry_run)

    for environment in args.environments:
        run_name = f"bm_{args.embodiment}_{environment}_{args.experiment_name}"
        set_args = {
            "workflow_name": run_name,
            "image": image,
            "wandb_api_key": wandb_key,
            "wandb_project_name": args.wandb_project_name,
            "wandb_run_name": run_name,
            "hf_token": hf_token,
            "checkpoint_artifact": args.checkpoint_artifact,
            "embodiment": args.embodiment,
            "environment": environment,
        }
        if args.distillation_ckpt_artifact:
            set_args["distillation_ckpt_artifact"] = args.distillation_ckpt_artifact
        submit_workflow(set_args, args.pool, args.dry_run)


if __name__ == "__main__":
    main()

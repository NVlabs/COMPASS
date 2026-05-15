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

# COMPASS_HOST_SIDE: true -- do not run inside the docker container.
# The `python` shim from `source ./docker/activate` scans for this marker and
# auto-routes the script to host Python (which has docker / osmo CLIs).
"""Submit COMPASS training/eval/record/distill workflows to OSMO.

Replaces the interactive shell launcher used in the internal repo with a
non-interactive Python CLI. Reads credentials from environment variables
(``WANDB_API_KEY``, ``HF_TOKEN``) with an opt-in ``--prompt`` fallback.

Run host-side — this script shells out to ``docker build``, ``docker push``,
and ``osmo workflow submit``, none of which exist inside the COMPASS runtime
container. The ``# COMPASS_HOST_SIDE: true`` marker above causes the
``source ./docker/activate`` shim to auto-route this launcher to host Python,
so plain ``python osmo/run_osmo.py …`` works from the activated shell. On
a stale shim, fall back to ``/usr/bin/python3 osmo/run_osmo.py …`` or
refresh with ``deactivate && source ./docker/activate``.

Usage:
    # Pre-built image, env vars already exported. The X-Mobility base
    # checkpoint and the COMPASS USDs are downloaded inside the workflow
    # from huggingface.co/nvidia/X-Mobility and huggingface.co/nvidia/COMPASS
    # respectively, so no --base-policy-ckpt flag is needed.
    python osmo/run_osmo.py train \\
        --experiment-name pilot \\
        --wandb-project compass-rl \\
        --image nvcr.io/<org>/compass_pilot:<tag>

    # Build+push the image automatically
    export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>
    python osmo/run_osmo.py train --experiment-name pilot --wandb-project compass-rl

    # Inspect the would-be submit command without executing it
    python osmo/run_osmo.py train --experiment-name pilot --wandb-project compass-rl \\
        --image nvcr.io/<org>/img:tag --dry-run

See https://nvlabs.github.io/COMPASS/docs/osmo.html for the full prerequisites
and per-subcommand examples.
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
WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"

# subcommand -> (workflow YAML filename, Dockerfile path relative to REPO_ROOT)
SUBCOMMAND_CONFIG = {
    "train": ("rl_es_train_workflow.yaml", "docker/Dockerfile.rl"),
    "eval": ("rl_es_eval_workflow.yaml", "docker/Dockerfile.rl"),
    "record": ("rl_es_record_workflow.yaml", "docker/Dockerfile.rl"),
    "distill": ("distillation_train_workflow.yaml", "docker/Dockerfile.distillation"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a COMPASS workflow to OSMO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(sp):
        sp.add_argument("--experiment-name",
                        required=True,
                        help="Short identifier; used in workflow_name and image tag.")
        sp.add_argument(
            "--image",
            default="",
            help="Pre-built docker image. If empty, build+push using --registry-prefix.")
        sp.add_argument("--registry-prefix",
                        default=os.environ.get("COMPASS_OSMO_REGISTRY", ""),
                        help="Registry prefix for build+push (e.g. nvcr.io/<org>/<team>); "
                        "defaults to $COMPASS_OSMO_REGISTRY.")
        sp.add_argument("--dry-run",
                        action="store_true",
                        help="Print the osmo workflow submit command without executing it.")
        sp.add_argument("--prompt",
                        action="store_true",
                        help="Interactively prompt for missing credentials.")

    train = sub.add_parser("train", help="Submit residual RL training.")
    add_common(train)
    train.add_argument("--wandb-project",
                       required=True,
                       help="wandb project to log into (e.g. compass-rl).")
    train.add_argument("--resume-ckpt",
                       default="",
                       help="wandb artifact for resuming from a prior residual checkpoint.")
    train.add_argument("--no-residual",
                       action="store_true",
                       help="Skip the residual head (use base policy only).")
    train.add_argument("--embodiment", default="", help="Override the gin-config embodiment.")
    train.add_argument("--environment", default="", help="Override the gin-config environment.")
    train.add_argument("--num-gpus",
                       type=int,
                       default=8,
                       help="Number of GPUs (= torchrun ranks). All counts use the same "
                       "distributed workflow YAML; the trainer's distributed code paths are "
                       "world_size-aware so num_gpus=1 also works as a single-rank run.")

    evl = sub.add_parser("eval", help="Submit residual RL evaluation.")
    add_common(evl)
    evl.add_argument("--wandb-project",
                     required=True,
                     help="wandb project to log into (e.g. compass-rl).")
    evl.add_argument("--checkpoint",
                     required=True,
                     help="wandb artifact for the residual checkpoint to evaluate.")
    evl.add_argument("--distillation-ckpt",
                     default="",
                     help="wandb artifact for an optional distillation checkpoint.")
    evl.add_argument("--no-residual",
                     action="store_true",
                     help="Evaluate without the residual head (base policy only).")
    evl.add_argument("--embodiment", default="", help="Override the gin-config embodiment.")
    evl.add_argument("--environment", default="", help="Override the gin-config environment.")

    rec = sub.add_parser("record", help="Submit distillation-data recording.")
    add_common(rec)
    rec.add_argument("--dataset-name",
                     required=True,
                     help="OSMO output dataset name for recorded specialist data.")

    dis = sub.add_parser("distill", help="Submit generalist distillation training.")
    add_common(dis)
    dis.add_argument("--wandb-project",
                     required=True,
                     help="wandb project to log into (e.g. compass-distill).")
    dis.add_argument("--dataset-name",
                     required=True,
                     help="OSMO input dataset name (specialist rollouts).")
    dis.add_argument("--checkpoint",
                     default="",
                     help="wandb artifact for an optional resume checkpoint.")
    dis.add_argument("--train-config",
                     default="distillation_config",
                     help="Gin config name without .gin (default: distillation_config).")

    return parser.parse_args()


def get_credential(name: str, prompt: bool) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    if prompt:
        return getpass.getpass(f"Enter {name}: ")
    sys.stderr.write(f"ERROR: ${name} is not set. Either export it or pass --prompt.\n")
    sys.exit(2)


def build_and_push_image(experiment: str, registry_prefix: str, dockerfile: str,
                         dry_run: bool) -> str:
    if not registry_prefix:
        sys.stderr.write("ERROR: --image not given and --registry-prefix is empty "
                         "(also $COMPASS_OSMO_REGISTRY).\n")
        sys.exit(2)
    image = f"{registry_prefix.rstrip('/')}/compass_{experiment}:{uuid.uuid4().hex[:8]}"
    cmds = [
        ["docker", "build", "--network=host", "-t", image, "-f", dockerfile,
         str(REPO_ROOT)],
        ["docker", "push", image],
    ]
    for cmd in cmds:
        print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
        if not dry_run:
            subprocess.check_call(cmd)
    return image


def submit_workflow(yaml_path: Path, set_args: dict, dry_run: bool) -> None:
    cmd = ["osmo", "workflow", "submit", str(yaml_path), "--set"]
    cmd += [f"{k}={v}" for k, v in set_args.items()]
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
    if not dry_run:
        subprocess.check_call(cmd)


def cmd_train(args, image: str, wandb_key: str, hf_token: str) -> None:
    if args.num_gpus < 1:
        sys.stderr.write(f"ERROR: --num-gpus must be >= 1 (got {args.num_gpus}).\n")
        sys.exit(2)
    yaml_path = WORKFLOWS_DIR / SUBCOMMAND_CONFIG["train"][0]
    set_args = {
        "workflow_name": f"compass_rl_es_{args.experiment_name}",
        "image": image,
        "num_gpus": args.num_gpus,
        "wandb_api_key": wandb_key,
        "wandb_project_name": args.wandb_project,
        "wandb_run_name": args.experiment_name,
        "hf_token": hf_token,
        "resume_ckpt_artifact": args.resume_ckpt,
        "no_residual": "1" if args.no_residual else "",
        "embodiment": args.embodiment,
        "environment": args.environment,
    }
    submit_workflow(yaml_path, set_args, args.dry_run)


def cmd_eval(args, image: str, wandb_key: str, hf_token: str) -> None:
    yaml_path = WORKFLOWS_DIR / SUBCOMMAND_CONFIG["eval"][0]
    set_args = {
        "workflow_name": f"compass_rl_es_{args.experiment_name}",
        "image": image,
        "wandb_api_key": wandb_key,
        "wandb_project_name": args.wandb_project,
        "wandb_run_name": args.experiment_name,
        "hf_token": hf_token,
        "checkpoint_artifact": args.checkpoint,
        "distillation_ckpt_artifact": args.distillation_ckpt,
        "no_residual": "1" if args.no_residual else "",
        "embodiment": args.embodiment,
        "environment": args.environment,
    }
    submit_workflow(yaml_path, set_args, args.dry_run)


def cmd_record(args, image: str, wandb_key: str, hf_token: str) -> None:
    yaml_path = WORKFLOWS_DIR / SUBCOMMAND_CONFIG["record"][0]
    set_args = {
        "workflow_name": f"compass_rl_es_{args.experiment_name}",
        "image": image,
        "wandb_api_key": wandb_key,
        "hf_token": hf_token,
        "dataset_name": args.dataset_name,
    }
    submit_workflow(yaml_path, set_args, args.dry_run)


def cmd_distill(args, image: str, wandb_key: str, hf_token: str) -> None:
    del hf_token    # distillation workflow doesn't need HF auth
    yaml_path = WORKFLOWS_DIR / SUBCOMMAND_CONFIG["distill"][0]
    set_args = {
        "workflow_name": f"compass_distillation_{args.experiment_name}",
        "image": image,
        "wandb_api_key": wandb_key,
        "wandb_project_name": args.wandb_project,
        "wandb_run_name": args.experiment_name,
        "dataset_name": args.dataset_name,
        "checkpoint_artifact": args.checkpoint,
        "train_config": args.train_config,
    }
    submit_workflow(yaml_path, set_args, args.dry_run)


SUBCOMMAND_DISPATCH = {
    "train": cmd_train,
    "eval": cmd_eval,
    "record": cmd_record,
    "distill": cmd_distill,
}


def main() -> None:
    args = parse_args()

    wandb_key = get_credential("WANDB_API_KEY", args.prompt)
    # HF_TOKEN is consumed by RL workflows; distillation doesn't need it.
    hf_token = ""
    if args.subcommand in ("train", "eval", "record"):
        hf_token = get_credential("HF_TOKEN", args.prompt)

    if args.image:
        image = args.image
    else:
        dockerfile = SUBCOMMAND_CONFIG[args.subcommand][1]
        image = build_and_push_image(args.experiment_name, args.registry_prefix, dockerfile,
                                     args.dry_run)

    SUBCOMMAND_DISPATCH[args.subcommand](args, image, wandb_key, hf_token)


if __name__ == "__main__":
    main()

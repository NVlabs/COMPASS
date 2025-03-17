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

import argparse
import os
import subprocess
import random
import shutil
import yaml

import wandb


def run(command: str):
    print(f'Running: {" ".join(command)}')
    subprocess.run(command, check=True)


def collect_data(dataset_config: dict, environments: list, base_policy_path: str, output_dir: str,
                 wandb_project_name: str, wandb_entity_name: str):
    """
    Collects data for each environment and embodiment combination.

    Args:
        environments (List[str]): A list of environments to collect data for.
        base_policy_path (str): The path to the base policy.
        output_dir (str): The directory to store the collected data.
    """
    # Initialize wandb if using artifact.
    if dataset_config['policy_ckpt_type'] == 'wandb_artifact':
        wandb.init(project=wandb_project_name, entity=wandb_entity_name)

    # Record data for each embodiment and environment.
    isaaclab_path = os.getenv("ISAACLAB_PATH")
    for embodiment, residual_policy in dataset_config['embodiment_policies'].items():
        if dataset_config['policy_ckpt_type'] == 'wandb_artifact':
            print(f'Collecting data for {embodiment} with {residual_policy}...')
            artifact = wandb.use_artifact(residual_policy, type='model')
            artifact_dir = artifact.download(root=f"/tmp/{embodiment}")
            residual_policy_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
            print(f'Downloaded embodiment residual policy: {residual_policy_path}')
        elif dataset_config['policy_ckpt_type'] == 'local':
            residual_policy_path = residual_policy
        else:
            raise ValueError(
                f'Invalid policy checkpoint type: {dataset_config["policy_ckpt_type"]}')
        for environment in environments:
            print(f'Collecting environment {environment}')
            dataset_dir = os.path.join(output_dir, embodiment, environment)
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            command = [
                f"{isaaclab_path}/isaaclab.sh", "-p", 'run.py', '-c', 'configs/record_config.gin',
                '--enable_cameras', '-b', base_policy_path, '-p', residual_policy_path, '-o',
                dataset_dir, '--embodiment', embodiment, '--environment', environment, '--headless'
            ]
            run(command)

    # Finish wandb if using artifact.
    if dataset_config['policy_ckpt_type'] == 'wandb_artifact':
        wandb.finish()


def compose_dataset(dataset_config: dict, output_dir: str, dataset_name: str):
    """
    Compose a dataset by splitting and organizing files in the output directory.

    Args:
        output_dir (str): The directory containing the files to be split.
        dataset_name (str): The name of the dataset.
    """
    # Process each embodiment separately
    for embodiment in os.listdir(output_dir):
        embodiment_path = os.path.join(output_dir, embodiment)
        if not os.path.isdir(embodiment_path):
            continue
        embodiment_split_files = {"train": [], "val": [], "test": []}
        # Process each environment
        for env in os.listdir(embodiment_path):
            env_path = os.path.join(embodiment_path, env, 'data')
            if not os.path.isdir(env_path):
                continue
            # Split per environment
            files = [os.path.join(env_path, f) for f in os.listdir(env_path)]
            random.shuffle(files)
            num_files = len(files)
            train_end = int(num_files * dataset_config['data_split_ratios']['train'])
            val_end = train_end + int(num_files * dataset_config['data_split_ratios']['val'])
            env_splits = {
                "train": files[:train_end],
                "val": files[train_end:val_end],
                "test": files[val_end:]
            }
            # Collect split files per robot (flattening env layer)
            for split, split_files in env_splits.items():
                embodiment_split_files[split].extend(split_files)

        # Move files to corresponding split directories (grouped by robot)
        for split, split_files in embodiment_split_files.items():
            split_dir = os.path.join(output_dir, dataset_name, 'data', split, embodiment)
            os.makedirs(split_dir, exist_ok=True)
            for file_path in split_files:
                target_file_path = os.path.join(split_dir, file_path.replace("/", "_"))
                shutil.move(file_path, target_file_path)


def main():
    # Add argparse arguments
    parser = argparse.ArgumentParser(description="Script to record datasets for RL distillation.")
    parser.add_argument('--dataset-config-file',
                        '-c',
                        type=str,
                        required=True,
                        help='The path to the dataset config yaml file.')
    parser.add_argument('--base_policy_path',
                        '-b',
                        type=str,
                        required=True,
                        help='The path to the base policy checkpoint artifact.')
    parser.add_argument('--wandb-project-name',
                        '-n',
                        type=str,
                        default='afm_rl_enhance_record',
                        help='The project name of W&B.')
    parser.add_argument('--wandb-entity-name',
                        '-e',
                        type=str,
                        default='nvidia-isaac',
                        help='The entity name of W&B.')
    parser.add_argument('--environments',
                        nargs='+',
                        type=str,
                        default=['combined_single_rack'],
                        help='A list of environments to benchmark.')
    parser.add_argument('--dataset-name', type=str, help='Name of the dataset in OSMO.')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory for the dataset.')
    args = parser.parse_args()
    with open(args.dataset_config_file, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    # Record dataset.
    collect_data(dataset_config=dataset_config,
                 environments=args.environments,
                 base_policy_path=args.base_policy_path,
                 output_dir=args.output_dir,
                 wandb_project_name=args.wandb_project_name,
                 wandb_entity_name=args.wandb_entity_name)
    compose_dataset(dataset_config=dataset_config,
                    output_dir=args.output_dir,
                    dataset_name=args.dataset_name)


if __name__ == '__main__':
    # Run the main function.
    main()

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

# pylint: skip-file

import argparse
import os
import gymnasium as gym

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="COMPASS Mobility Generalist.")
parser.add_argument('--config-files',
                    '-c',
                    nargs='+',
                    required=True,
                    help='The list of the config files.')
parser.add_argument('--base-policy-path',
                    '-b',
                    type=str,
                    default=None,
                    help='The path to the base policy checkpoint.')
parser.add_argument('--distillation-policy-path',
                    '-d',
                    type=str,
                    default=None,
                    help='The path to the distillation policy checkpoint.')
parser.add_argument('--checkpoint-path',
                    '-p',
                    type=str,
                    default=None,
                    help='The path to the checkpoint.')
parser.add_argument('--logger',
                    type=str,
                    choices=['wandb', 'tensorboard'],
                    default='tensorboard',
                    help='Logger to use: wandb or tensorboard')
parser.add_argument('--wandb-project-name',
                    '-n',
                    type=str,
                    default='afm_rl_enhance',
                    help='The project name of W&B.')
parser.add_argument('--wandb-run-name',
                    '-r',
                    type=str,
                    default='train_run',
                    help='The run name of W&B.')
parser.add_argument('--wandb-entity-name',
                    '-e',
                    type=str,
                    default='nvidia-isaac',
                    help='The entity name of W&B.')
parser.add_argument('--output-dir',
                    '-o',
                    type=str,
                    required=True,
                    help='The path to the output dir.')
parser.add_argument("--video",
                    action="store_true",
                    default=False,
                    help="Record videos during training.")
parser.add_argument("--video_interval",
                    type=int,
                    default=10,
                    help="Interval between video recordings (in iterations).")
# Optional parameters to override gin config.
parser.add_argument('--embodiment', type=str, help='Embodiment type')
parser.add_argument('--environment', type=str, help='Environment type')
parser.add_argument('--num_envs', type=int, help='Number of environments')

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gin
import torch
import wandb

from mobility_es.config import environments
from mobility_es.config.carter_env_cfg import CarterGoalReachingEnvCfg
from mobility_es.config.h1_env_cfg import H1GoalReachingEnvCfg
from mobility_es.config.spot_env_cfg import SpotGoalReachingEnvCfg
from mobility_es.config.g1_env_cfg import G1GoalReachingEnvCfg
from mobility_es.config.digit_env_cfg import DigitGoalReachingEnvCfg
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper

from compass.residual_rl.x_mobility_rl import XMobilityBasePolicy
from compass.distillation.distillation import ESDistillationPolicyWrapper
from compass.residual_rl.residual_ppo_trainer import ResidualPPOTrainer
from compass.utils.logger import Logger

# Map from the embedding type to the RL env config.
EmbodimentEnvCfgMap = {
    'h1': H1GoalReachingEnvCfg,
    'spot': SpotGoalReachingEnvCfg,
    'carter': CarterGoalReachingEnvCfg,
    'g1': G1GoalReachingEnvCfg,
    'digit': DigitGoalReachingEnvCfg
}

# Map from the environment type to the env scene asset config.
EnvSceneAssetCfgMap = {
    'warehouse_single_rack': environments.warehouse_single_rack,
    'galileo_lab': environments.galileo_lab,
    'simple_office': environments.simple_office,
    'combined_single_rack': environments.combined_single_rack,
    'combined_multi_rack': environments.combined_multi_rack,
    'random_envs': environments.random_envs,
    'hospital': environments.hospital,
    'warehouse_multi_rack': environments.warehouse_multi_rack
}


def gin_config_to_dictionary(gin_config):
    """
    Parses the gin configuration to a dictionary.
    """
    config_dict = {}
    for (scope, selector), value in gin_config.items():
        # Construct a key from scope and selector
        key = f"{scope}:{selector}" if scope else selector
        config_dict[key] = value
    return config_dict


@gin.configurable
def run(run_mode,
        embodiment,
        environment,
        num_envs,
        num_iterations,
        num_steps_per_iteration,
        seed,
        enable_curriculum=False):

    # Setup logger.
    logger = Logger(log_dir=args_cli.output_dir,
                    backend=args_cli.logger,
                    experiment_name=args_cli.wandb_run_name,
                    project_name=args_cli.wandb_project_name,
                    entity=args_cli.wandb_entity_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Setup base policy.
    base_policy = XMobilityBasePolicy(args_cli.base_policy_path)
    base_policy = torch.nn.DataParallel(base_policy)
    base_policy.to(device)
    base_policy.eval()

    # Setup distillated policy.
    if args_cli.distillation_policy_path is not None:
        distillation_policy = ESDistillationPolicyWrapper(args_cli.distillation_policy_path,
                                                          embodiment)
        distillation_policy = torch.nn.DataParallel(distillation_policy)
        distillation_policy.to(device)
        distillation_policy.eval()
    else:
        distillation_policy = None

    # Setup embodiment type.
    if embodiment in EmbodimentEnvCfgMap:
        env_cfg = EmbodimentEnvCfgMap[embodiment]()
    else:
        raise ValueError(f'Unsupported embodiment type: {embodiment}')

    # Setup environment scene.
    if environment in EnvSceneAssetCfgMap:
        env_cfg.scene.environment = EnvSceneAssetCfgMap[environment]
    else:
        raise ValueError(f'Unsupported environment type: {environment}')
    env_cfg.scene.replicate_physics = env_cfg.scene.environment.replicate_physics
    env_cfg.scene.env_spacing = env_cfg.scene.environment.env_spacing
    env_cfg.scene.num_envs = num_envs
    env_cfg.events.reset_base.params["pose_range"] = env_cfg.scene.environment.pose_sample_range

    # Setup the curriculum
    if enable_curriculum:
        env_cfg.curriculum.command_min_distance_prob.params[
            "num_steps_per_iteration"] = num_steps_per_iteration
        env_cfg.curriculum.command_min_distance_prob.params["total_iterations"] = num_iterations
    else:
        env_cfg.curriculum = None

    # Setup viewer
    env_cfg.viewer.origin_type = 'asset_root'
    env_cfg.viewer.asset_name = 'robot'
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-2.5, -0.5, 1.5)

    # Setup seed
    env_cfg.seed = seed

    # Disable rewards, termination and curriculum for eval.
    if run_mode == 'eval' or run_mode == 'record':
        env_cfg.rewards = None
        env_cfg.terminations = None
        env_cfg.curriculum = None
    env = RLESEnvWrapper(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Setup video if enabled.
    if args_cli.video:
        video_kwargs = {
            "video_folder":
                os.path.join(args_cli.output_dir, "videos"),
            "step_trigger":
                lambda step: step % (num_steps_per_iteration * args_cli.video_interval) == 0,
            "video_length":
                num_steps_per_iteration,
            "disable_logger":
                True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Setup the agent.
    rl_trainer = ResidualPPOTrainer(env=env,
                                    base_policy=base_policy,
                                    output_dir=args_cli.output_dir,
                                    logger=logger,
                                    device=device)

    if run_mode == 'train':
        if args_cli.checkpoint_path:
            rl_trainer.load(path=args_cli.checkpoint_path)
        rl_trainer.learn(num_iterations)
    elif run_mode == 'eval':
        if args_cli.checkpoint_path:
            rl_trainer.load(path=args_cli.checkpoint_path, load_optimizer=False)
        rl_trainer.eval(num_iterations, distillation_policy)
    elif run_mode == 'record':
        metadata = {
            'embodiment': embodiment,
            'environment': environment,
            'batch_size': num_envs,
            'sequence_length': num_steps_per_iteration,
            'seed': seed,
            'checkpoint_path': args_cli.checkpoint_path
        }
        rl_trainer.load(path=args_cli.checkpoint_path, load_optimizer=False)
        rl_trainer.record(num_iterations, metadata, os.path.join(args_cli.output_dir, 'data'))
    else:
        raise ValueError('Unsupported run mode.')

    # Log configs.
    logger.log_config(gin_config_to_dictionary(gin.config._OPERATIVE_CONFIG))

    logger.close()


def main():
    # Load parameters from gin-config.
    for config_file in args_cli.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    # Override gin-configurable parameters with command line arguments.
    if args_cli.embodiment is not None:
        gin.bind_parameter('run.embodiment', args_cli.embodiment)
    if args_cli.environment is not None:
        gin.bind_parameter('run.environment', args_cli.environment)
    if args_cli.num_envs is not None:
        gin.bind_parameter('run.num_envs', args_cli.num_envs)

    # Run the training/evaluation/recording.
    run()


if __name__ == '__main__':
    # Run the main function.
    main()
    # Close the sim app.
    simulation_app.close()

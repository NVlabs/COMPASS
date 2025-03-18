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

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Embodiment specialist to enhance navigation policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from mobility_es.config.carter_env_cfg import CarterGoalReachingEnvCfg
from mobility_es.config.h1_env_cfg import H1GoalReachingEnvCfg
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper


def main():
    """Main function."""
    # Parse the arguments
    env_cfg = H1GoalReachingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Setup base environment
    env = RLESEnvWrapper(cfg=env_cfg)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # Sample random actions
            action_commands = torch.randn_like(env.action_manager.action)
            # Step the environment
            obs_dict, rewards, dones, truncateds, infos = env.step(action_commands)
            print(obs_dict)
            # Update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()

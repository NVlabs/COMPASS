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

import gin
import wandb
import torch
import torch.nn.functional as F
from torch import nn

from model.x_mobility.utils import pack_sequence_dim
from model.x_mobility.x_mobility import XMobility

from compass.residual_rl.constants import INPUT_IMAGE_SIZE
from compass.residual_rl.actor_critic import ActorCriticXMobility
from compass.distillation.distillation import ESDistillationPolicyWrapper


class PolicyInference(nn.Module):
    ''' Wrapper of policy for Onnx conversion.
    '''

    def __init__(self, base_checkpoint_path: str):
        super().__init__()
        # Load the state dict for X-Mobility.
        state_dict = torch.load(base_checkpoint_path, weights_only=False)['state_dict']
        state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
        self.x_mobility = XMobility()
        self.x_mobility.load_state_dict(state_dict)
        self.goal_heading_dim = 2
        self.policy_state_dim = self.x_mobility.action_policy.poly_state_dim + self.goal_heading_dim

    def base_policy_forward(self, image, route, speed, action_input, history_input, sample_input):
        inputs = {}
        # Resize the input image to desired size.
        image = image.squeeze(0)
        image = F.interpolate(image, size=INPUT_IMAGE_SIZE, mode='bilinear', align_corners=False)
        inputs['image'] = image.unsqueeze(0)
        inputs['route'] = route
        inputs['speed'] = speed
        inputs['action'] = action_input
        inputs['history'] = history_input
        inputs['sample'] = sample_input

        # Run base policy.
        base_action, _, history, sample, _, _, _ = self.x_mobility.inference(
            inputs, False, False, False)
        latent_state = torch.cat([history, sample], dim=-1)
        route_feat = self.x_mobility.action_policy.route_encoder(pack_sequence_dim(route))
        base_policy_state = self.x_mobility.action_policy.policy_state_fusion(
            latent_state, route_feat)
        return base_action, base_policy_state, history, sample

    def compose_policy_state(self, base_policy_state, goal_heading):
        goal_heading_encoding = self._encode_goal_heading(goal_heading)
        return torch.cat([base_policy_state, goal_heading_encoding], dim=1)

    def _encode_goal_heading(self, goal_heading):
        return torch.cat([torch.cos(goal_heading), torch.sin(goal_heading)], dim=1)


class SpecialistPolicyInference(PolicyInference):
    ''' Wrapper of embodiment specialist policy for Onnx conversion.
    '''

    def __init__(self, base_checkpoint_path: str, residual_checkpoint_path: str):
        super().__init__(base_checkpoint_path)
        # Load the state dict for the residual policy.
        self.residual_policy = ActorCriticXMobility(self.x_mobility.action_policy,
                                                    action_dim=6,
                                                    critic_state_dim=self.policy_state_dim,
                                                    actor_state_dim=self.policy_state_dim)
        self.residual_policy.load_state_dict(
            torch.load(residual_checkpoint_path, weights_only=False)['model_state_dict'])

    def forward(self, image, route, goal_heading, speed, action_input, history_input, sample_input):
        # Run base policy.
        base_action, base_policy_state, history, sample = self.base_policy_forward(
            image, route, speed, action_input, history_input, sample_input)
        # Add additional encodings to policy state.
        policy_state = self.compose_policy_state(base_policy_state, goal_heading)
        # Run residual policy.
        residual_action = self.residual_policy.act_inference(policy_state)
        final_action = base_action + residual_action

        # Outputs: [action_output, history_output, sample_output]
        return final_action, history, sample


class GeneralistPolicyInference(PolicyInference):
    ''' Wrapper of generalist policy for Onnx conversion.
    '''

    def __init__(self, base_checkpoint_path: str, generalist_policy_ckpt_path: str,
                 embodiment_type: str):
        super().__init__(base_checkpoint_path)

        # Load the state dict for the generalist policy.
        self.generalist_policy = ESDistillationPolicyWrapper(generalist_policy_ckpt_path,
                                                             embodiment_type)

    def forward(self, image, route, goal_heading, speed, action_input, history_input, sample_input):
        # Run base policy.
        _, base_policy_state, history, sample = self.base_policy_forward(
            image, route, goal_heading, speed, action_input, history_input, sample_input)
        # Add additional encodings to policy state.
        policy_state = self.compose_policy_state(base_policy_state, goal_heading)

        # Run generalist policy.
        generalist_action = self.generalist_policy(policy_state)

        # Outputs: [action_output, history_output, sample_output]
        return generalist_action, history, sample


def convert(base_checkpoint_path: str,
            residual_checkpoint_path: str,
            generalist_checkpoint_path: str,
            embodiment_type: str,
            onnx_path: str,
            jit_path: str,
            image_size: list = [1200, 1920]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if residual_checkpoint_path:
        model = SpecialistPolicyInference(base_checkpoint_path, residual_checkpoint_path)
    elif generalist_checkpoint_path:
        model = GeneralistPolicyInference(base_checkpoint_path, generalist_checkpoint_path,
                                          embodiment_type)
    else:
        raise ValueError('Either residual or generalist checkpoint path must be provided.')

    model.to(device)
    model.eval()

    # Input tensors.
    image = torch.randn((1, 1, 3, image_size[0], image_size[1]), dtype=torch.float32).to(device)
    speed = torch.randn((1, 1, 1), dtype=torch.float32).to(device)
    action = torch.randn((1, 6), dtype=torch.float32).to(device)
    goal_heading = torch.randn((1, 1), dtype=torch.float32).to(device)
    history = torch.zeros((1, 1024), dtype=torch.float32).to(device)
    sample = torch.zeros((1, 512), dtype=torch.float32).to(device)
    route = torch.randn((1, 1, 10, 4), dtype=torch.float32).to(device)

    # Output jit file.
    if jit_path:
        traced_model = torch.jit.trace(model,
                                       (image, route, goal_heading, speed, action, history, sample))
        traced_model.save(jit_path)

    # Output names.
    output_names = ['action_output', 'history_output', 'sample_output']

    # ONNX conversion.
    torch.onnx.export(model, (image, route, goal_heading, speed, action, history, sample),
                      onnx_path,
                      verbose=True,
                      input_names=[
                          'image', 'route', 'goal_heading', 'speed', 'action_input',
                          'history_input', 'sample_input'
                      ],
                      output_names=output_names)


def main():
    # Parse the arguments.
    parser = argparse.ArgumentParser(description='Convert the E2E Nav model to onnx.')
    parser.add_argument('--base-ckpt-path',
                        '-b',
                        type=str,
                        required=True,
                        help='The path to the base policy checkpoint.')
    parser.add_argument('--residual-ckpt-path',
                        '-r',
                        type=str,
                        required=False,
                        help='The path to the residual policy checkpoint.')
    parser.add_argument('--generalist-ckpt-path',
                        '-g',
                        type=str,
                        required=False,
                        help='The path to the generalist policy checkpoint.')
    parser.add_argument('--embodiment-type',
                        '-e',
                        type=str,
                        required=False,
                        help='The embodiment type to use for the generalist policy.')
    parser.add_argument('--onnx-file',
                        '-o',
                        type=str,
                        required=True,
                        help='The path to the output onnx file.')
    parser.add_argument('--jit-file',
                        '-j',
                        type=str,
                        required=False,
                        help='The path to the output JIT file.')
    parser.add_argument('--onnx-artifact',
                        '-a',
                        type=str,
                        required=False,
                        help='The wandb onnx artifact to upload.')
    parser.add_argument('--image-size',
                        '-i',
                        type=int,
                        nargs=2,
                        default=[1200, 1920],
                        help='The input image size as [height, width].')

    args = parser.parse_args()

    # Load hyperparameters.
    gin.parse_config_file('configs/eval_config.gin', skip_unknown=True)

    # Run the convert.
    print("Converting ONNX.")
    convert(args.base_ckpt_path, args.residual_ckpt_path, args.generalist_ckpt_path,
            args.embodiment_type, args.onnx_file, args.jit_file, args.image_size)

    # Upload onnx to wandb if onnx_artifact is provided.
    if args.onnx_artifact:
        print(f'Uploading to WANDB: {args.onnx_artifact.split("/")}.')
        wandb_project = args.onnx_artifact.split('/')[1]
        wandb_run_id = args.onnx_artifact.split('/')[2].split(':')[0]
        wandb.init(dir='/tmp', project=wandb_project, id=wandb_run_id)
        version = args.onnx_artifact.split('/')[2].split(':')[1]
        onnx_artifact = wandb.Artifact(f'onnx-{wandb_run_id}-{version}', type='onnx')
        onnx_artifact.add_file(args.onnx_file)
        wandb.log_artifact(onnx_artifact)
        wandb.finish()


if __name__ == '__main__':
    main()

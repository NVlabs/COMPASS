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

import torch
from torch import nn

from model.x_mobility.x_mobility import XMobility
from model.x_mobility.utils import pack_sequence_dim

from compass.residual_rl.constants import INPUT_IMAGE_SIZE


class XMobilityBasePolicy(nn.Module):
    '''Wrapper of the X-Mobility to enable parallel inference during RL training.'''

    def __init__(self, base_policy_ckpt_path):
        super().__init__()
        # Load the checkpoint and remove the prefix if any.
        state_dict = torch.load(base_policy_ckpt_path, weights_only=False)['state_dict']
        state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
        # Load the state dict.
        self.model = XMobility()
        self.model.load_state_dict(state_dict)

    def forward(self, obs_dict, history=None, sample=None, action=None):
        b = obs_dict["policy"]["camera_rgb_img"].shape[0]
        batch = {}
        batch['speed'] = obs_dict["policy"]["base_speed"].reshape(b, 1, -1).float()
        batch['image'] = obs_dict["policy"]["camera_rgb_img"].reshape(b, 1, INPUT_IMAGE_SIZE[0],
                                                                      INPUT_IMAGE_SIZE[1],
                                                                      3).float()
        batch['image'] = batch['image'].permute(0, 1, 4, 2, 3) / 255.0
        batch['history'] = batch['speed'].new_zeros(
            (b, self.model.rssm.hidden_state_dim)).float() if history is None else history
        batch['sample'] = batch['speed'].new_zeros(
            (b, self.model.rssm.state_dim)).float() if sample is None else sample
        batch['action'] = batch['speed'].new_zeros(
            (b, 6)).float() if action is None else (action.reshape(b, -1).float())
        batch['route'] = obs_dict["policy"]['route'].unsqueeze(1).float()

        action, _, history, sample, _, _, _ = self.model.inference(batch, False, False, False)
        action = pack_sequence_dim(action)
        latent_state = torch.cat([history, sample], dim=-1).reshape(b, -1)
        route_feat = self.model.action_policy.route_encoder(pack_sequence_dim(batch['route']))
        speed_feat = self.model.observation_encoder.speed_encoder(pack_sequence_dim(batch['speed']))
        policy_state = self.model.action_policy.policy_state_fusion(latent_state, route_feat)

        extras = {}
        extras["route_feat"] = route_feat
        extras["speed_feat"] = speed_feat
        return policy_state, action, history, sample, extras

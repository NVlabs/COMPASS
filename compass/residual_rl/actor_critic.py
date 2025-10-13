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

import copy

import gin
import torch
from torch import nn
from torch.distributions import Normal

from compass.residual_rl.mlp import build_mlp_network, get_activation


@gin.configurable
class ActorCriticBase(nn.Module):
    """Base class of the ActorCritic.

    Note: The inherited class needs to provide the definitions of the actor and critic network.

    """

    def __init__(self, action_dim: int, init_noise_std: float = 1.0):

        super().__init__()

        self.actor = None
        self.critic = None

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, policy_state):
        mean = self.act_inference(policy_state)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, policy_state):
        self.update_distribution(policy_state)
        action = self.distribution.sample()
        return action

    def act_inference(self, policy_state):
        if self.actor is None:
            raise ValueError("User needs to provide the definition of actor network")
        # pylint: disable=not-callable
        return self.actor(policy_state)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_state):
        if self.critic is None:
            raise ValueError("User needs to provide the definition of critic network")
        # Compute the value.
        # pylint: disable=not-callable
        value = self.critic(critic_state)
        return value


@gin.configurable
class ActorCriticXMobility(ActorCriticBase):
    """ActorCritic for residual policy to enhance X-Mobility.

    The actor network is initialized with the X-Mobility's policy head with the last linear layer
    reset to zero.
    """

    def __init__(self,
                 base_policy,
                 action_dim,
                 critic_state_dim,
                 actor_state_dim=None,
                 critic_hidden_dims=256,
                 init_noise_std=1.0):
        super().__init__(action_dim=action_dim, init_noise_std=init_noise_std)
        # Load the base_policy and reset the last linear layer to 0.0
        x_mobility_actor = copy.deepcopy(base_policy.policy_mlp.command_fc)
        nn.init.zeros_(x_mobility_actor[-2].weight)    # Set weights to zero
        if x_mobility_actor[-2].bias is not None:
            nn.init.zeros_(x_mobility_actor[-2].bias)    # Set bias to zero
        # Add a linear projection layer if actor_state_dim is provided to match the input dimension.
        if actor_state_dim is not None:
            x_mobility_input_dim = x_mobility_actor[0].in_features
            self.actor = nn.Sequential(nn.Linear(actor_state_dim, x_mobility_input_dim),
                                       nn.ReLU(True), *x_mobility_actor)
        else:
            self.actor = x_mobility_actor
        # Enable the gradients.
        for param in self.actor.parameters():
            param.requires_grad = True

        # Define critic
        activation = get_activation('relu', inplace=True)
        self.critic = build_mlp_network(
            input_dim=critic_state_dim,
            hidden_dims=[critic_hidden_dims, critic_hidden_dims, critic_hidden_dims // 2],
            output_dim=1,
            activation_fn=activation)


@gin.configurable
class ActorCriticMLP(ActorCriticBase):
    """ ActorCritic with both actor and critic built as MLP modules.
    """

    def __init__(self,
                 actor_state_dim,
                 critic_state_dim,
                 actor_hidden_dims,
                 critic_hidden_dims,
                 action_dim,
                 init_noise_std=1.0):
        super().__init__(action_dim, init_noise_std)

        activation = get_activation('relu', inplace=True)
        self.actor = build_mlp_network(input_dim=actor_state_dim,
                                       hidden_dims=actor_hidden_dims,
                                       output_dim=action_dim,
                                       activation_fn=activation)

        self.critic = build_mlp_network(input_dim=critic_state_dim,
                                        hidden_dims=critic_hidden_dims,
                                        output_dim=1,
                                        activation_fn=activation)

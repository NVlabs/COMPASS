# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gin
import torch
from torch import nn, optim

from compass.residual_rl.rollout_storage import RolloutStorage


@gin.configurable
class PPO:
    """PPO class to train a A2C policy"""

    def __init__(self,
                 actor_critic,
                 device='cpu',
                 value_loss_coef=1.0,
                 use_clipped_value_loss=True,
                 clip_param=0.2,
                 entropy_coef=0.01,
                 num_learning_epochs=5,
                 num_mini_batches=4,
                 learning_rate=1.e-4,
                 schedule='adaptive',
                 gamma=0.99,
                 lam=0.95,
                 desired_kl=0.01,
                 max_grad_norm=1.0):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Parameters
        self.learning_rate = learning_rate
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.schedule = schedule
        self.gamma = gamma
        self.lam = lam
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm

    def init_storage(self, num_envs, num_transitions_per_env, policy_states_shape,
                     critic_states_shape, actions_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, policy_states_shape,
                                      critic_states_shape, actions_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, states):
        policy_states = self._get_policy_states(states)
        critic_states = self._get_critic_states(states)

        # Compute the actions and values
        self.transition.actions = (self.actor_critic.act(policy_states)).detach()
        self.transition.values = self.actor_critic.evaluate(critic_states).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = (self.actor_critic.action_mean).detach()
        self.transition.action_sigma = (self.actor_critic.action_std).detach()

        # Need to record policy state before env.step()
        self.transition.policy_states = policy_states
        self.transition.critic_states = critic_states
        return self.transition.actions

    def act_inference(self, states):
        policy_states = self._get_policy_states(states)
        return self.actor_critic.act_inference(policy_states)

    def process_env_step(self, rewards, dones):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_states):
        critic_states = self._get_critic_states(last_states)
        last_values = self.actor_critic.evaluate(critic_states).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches,
                                                      self.num_learning_epochs)
        for states_batch, actions_batch, target_values_batch, advantages_batch, \
                returns_batch, old_actions_log_prob_batch, old_mu_batch, \
                old_sigma_batch, _ in generator:
            policy_states_batch = self._get_policy_states(states_batch)
            critic_states_batch = self._get_critic_states(states_batch)
            self.actor_critic.act(policy_states_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_states_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif (self.desired_kl / 2.0) > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                          self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - \
                self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

    def _get_policy_states(self, states):
        return states[0]

    def _get_critic_states(self, states):
        return states[1]

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

import os
import time
from contextlib import contextmanager

import numpy as np
import h5py
import gin
import torch
import torch.distributed as dist
from tqdm import tqdm

from compass.residual_rl.constants import INPUT_IMAGE_SIZE
from compass.residual_rl.actor_critic import ActorCriticXMobility
from compass.residual_rl.critic_state_assembler import CriticObservationEncoder
from compass.residual_rl.ppo import PPO
from compass.residual_rl.gr00t_service import ExternalRobotInferenceClient


@gin.configurable
class ResidualPPOTrainer:
    """Trainer for residual policy."""

    def __init__(self,
                 env,
                 base_policy,
                 output_dir,
                 critic_state_assembler,
                 logger,
                 device='cpu',
                 num_steps_per_env=100,
                 ckpt_save_interval=50,
                 debug_viz=False):
        # Prepare log directory. exist_ok=True avoids a TOCTOU race when several
        # torchrun ranks check + create concurrently.
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.logger = logger
        # Init environment and base policy.
        self.env = env
        self.base_policy = base_policy
        for param in self.base_policy.parameters():
            param.requires_grad = False
        self.device = device
        # Multi-rank state. WORLD_SIZE / RANK are set by torchrun; default to 1 / 0 in
        # single-process runs. is_rank_zero gates writes (checkpoints, video upload,
        # episode-log aggregation) so only rank 0 produces side-effects on disk.
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.is_rank_zero = self.global_rank == 0

        # Get policy and critic state dimensions.
        self.goal_heading_dim = 2
        self.policy_state_dim = self.base_policy.module.model.action_policy.poly_state_dim + self.goal_heading_dim    #pylint: disable=line-too-long
        route_feat_dim = self.base_policy.module.model.action_policy.route_encoder.out_channels
        speed_feat_dim = self.base_policy.module.model.observation_encoder.speed_encoder.out_channels    #pylint: disable=line-too-long
        critic_enc = CriticObservationEncoder(route_feat_dim=route_feat_dim,
                                              speed_feat_dim=speed_feat_dim).to(device)
        self.critic_state_assembler = critic_state_assembler(env=self.env,
                                                             encoder=critic_enc,
                                                             policy_state_dim=self.policy_state_dim)
        self.critic_state_dim = self.critic_state_assembler.state_dim

        self.gr00t_client = ExternalRobotInferenceClient(host="0.0.0.0", port=8888)

        # Init actor critic.
        self.action_dim = 6
        actor_critic = ActorCriticXMobility(self.base_policy.module.model.action_policy,
                                            actor_state_dim=self.policy_state_dim,
                                            action_dim=self.action_dim,
                                            critic_state_dim=self.critic_state_dim).to(self.device)
        # Broadcast initial actor-critic params from rank 0 so all ranks start identical.
        # Combined with PPO's gradient all-reduce, this keeps params synchronized across ranks.
        if self.world_size > 1:
            for p in actor_critic.parameters():
                dist.broadcast(p.data, src=0)

        # Init storage and model
        self.num_steps_per_env = num_steps_per_env
        self.alg = PPO(actor_critic, device=self.device)
        self.alg.init_storage(self.env.unwrapped.num_envs,
                              self.num_steps_per_env,
                              policy_states_shape=[self.policy_state_dim],
                              critic_states_shape=[self.critic_state_dim],
                              actions_shape=[self.action_dim])

        # Init logging.
        self.ckpt_save_interval = ckpt_save_interval
        self.current_learning_iteration = 0
        self.debug_viz = debug_viz

        self.env.reset()

        self._install_env_timers()

        print(self)

    def __str__(self) -> str:
        msg = "ResidualPPOTrainer: \n"
        msg += f"\tCritic state assembler: {self.critic_state_assembler}\n"
        msg += f"\tActor state dim:: {self.policy_state_dim}\n"
        msg += f"\tCritic state dim: {self.critic_state_dim}\n"
        return msg

    def base_policy_process(self, obs_dict, history=None, sample=None, action=None):
        policy_state, base_actions, history, sample, extras = self.base_policy(
            obs_dict, history, sample, action)
        # Extend policy state with goal heading
        policy_state = torch.cat([policy_state, obs_dict['policy']['goal_heading']], dim=1)
        return policy_state, base_actions, history, sample, extras

    def gr00t_policy_process(self, obs_dict):
        b = obs_dict['policy']['camera_rgb_img'].shape[0]
        device = obs_dict['policy']['camera_rgb_img'].device
        rgb = obs_dict['policy']['camera_rgb_img'].reshape(b, 1, INPUT_IMAGE_SIZE[0],
                                                           INPUT_IMAGE_SIZE[1], 3).cpu().numpy()
        observations = {
            "annotation.human.action.task_description":
                np.repeat(np.array([["Robot navigation task"]]), b, axis=0),
            "video.ego_view":
                rgb,
            "state.speed":
                obs_dict['policy']['base_speed'].reshape(b, 1, 1).cpu().numpy(),
            "state.route":
                obs_dict['policy']['route'].reshape(b, 1, 40).cpu().numpy(),
            "state.goal_heading":
                obs_dict['policy']['goal_heading'].reshape(b, 1, 2).cpu().numpy(),
        }
        outputs = self.gr00t_client.get_action(observations)
        action_tensor = torch.from_numpy(outputs['action.vel_cmd'][:, 0]).to(device=device,
                                                                             dtype=torch.float32)
        # Create zero tensor
        new_tensor = torch.zeros(b, 6, device=action_tensor.device, dtype=action_tensor.dtype)
        # Fill in the specific indices
        new_tensor[:, (0, 1, 5)] = action_tensor
        return new_tensor

    def compute_states(self, policy_state, obs_dict, extras):
        return [
            policy_state,
            self.critic_state_assembler.compute_critic_state(policy_state, obs_dict, extras)
        ]

    def _install_env_timers(self):
        # Monkey-patch a handful of Isaac Lab manager methods + sim.step / sim.render
        # so each call adds its wall-clock time into self._env_breakdown. The trainer
        # reads (and clears) this dict around every env.step() call to surface a
        # per-stage breakdown of where env_step time goes.
        # ~3% wall-clock overhead from extra cuda.synchronize() calls.
        self._env_breakdown = {}
        cuda_active = self.device != 'cpu' and torch.cuda.is_available()
        env = self.env.unwrapped
        targets = [
            ("action_pre", getattr(env, "action_manager", None), "process_action"),
            ("termination", getattr(env, "termination_manager", None), "compute"),
            ("reward", getattr(env, "reward_manager", None), "compute"),
            ("command", getattr(env, "command_manager", None), "compute"),
            ("event", getattr(env, "event_manager", None), "apply"),
            ("obs", getattr(env, "observation_manager", None), "compute"),
            ("reset_idx", env, "_reset_idx"),
            ("sim_step", env.sim, "step"),
            ("sim_render", env.sim, "render"),
        ]
        breakdown = self._env_breakdown
        for key, obj, method_name in targets:
            if obj is None or not hasattr(obj, method_name):
                continue
            original = getattr(obj, method_name)

            def _make_wrapper(orig, k):
                def wrapper(*args, **kwargs):
                    if cuda_active:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    try:
                        return orig(*args, **kwargs)
                    finally:
                        if cuda_active:
                            torch.cuda.synchronize()
                        breakdown[k] = breakdown.get(k, 0.0) + (time.perf_counter() - t0)
                return wrapper

            setattr(obj, method_name, _make_wrapper(original, key))

        # Drill one level deeper into observation_manager.compute by also wrapping
        # each ObsTerm's func — surfaces which obs (camera_rgb_img, camera_depth_img,
        # camera_to_world, route, ...) actually eats the time.
        obs_mgr = getattr(env, "observation_manager", None)
        if obs_mgr is not None:
            for group_name, term_cfgs in obs_mgr._group_obs_term_cfgs.items():
                term_names = obs_mgr._group_obs_term_names[group_name]
                for name, cfg in zip(term_names, term_cfgs):
                    if not callable(getattr(cfg, "func", None)):
                        continue
                    original = cfg.func
                    key = f"obs_term/{group_name}/{name}"

                    def _make_term_wrapper(orig, k):
                        def wrapper(*args, **kwargs):
                            if cuda_active:
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            try:
                                return orig(*args, **kwargs)
                            finally:
                                if cuda_active:
                                    torch.cuda.synchronize()
                                breakdown[k] = breakdown.get(k, 0.0) + (time.perf_counter() - t0)
                        return wrapper

                    cfg.func = _make_term_wrapper(original, key)

    @contextmanager
    def _timer(self, key, accumulator):
        # CUDA ops are async; sync so the timer covers the actual GPU work.
        # Cumulative add lets the same key be used across rollout sub-steps.
        cuda_active = self.device != 'cpu' and torch.cuda.is_available()
        if cuda_active:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda_active:
                torch.cuda.synchronize()
            accumulator[key] = accumulator.get(key, 0.0) + (time.perf_counter() - t0)

    def learn(self, num_learning_iterations):
        self.train_mode()

        tot_iter = self.current_learning_iteration + num_learning_iterations
        ep_logs = []

        for it in tqdm(range(self.current_learning_iteration, tot_iter)):
            '''
                env.reset()
                obs_dict = env.get_observation()
                policy_state, base_action = base_policy.tokenize_and_plan(obs_dict)
                states = states_process(obs_dict, policy_state)
                for step in range(num_steps_per_env):
                    residual_action = alg.compute_residual_action(states)
                    next_obs_dict, reward, done = env.step(base_action + residual_action)
                    alg.log_transitions()
                    policy_state, base_action = base_policy.tokenize_and_plan(next_obs_dict)
                    states = states_process(next_obs_dict, policy_state)
                alg.optimize_actor_critic_network()
            '''
            print(10 * "_")
            print(f"Learning iteration {it}")
            print("Rollout:")
            times = {"checkpoint": 0.0}
            with torch.inference_mode():
                with self._timer("env_reset_and_init", times):
                    self.env.reset()
                    # Get obs ready for policy state and base action from base_policy
                    obs_dict = self.env.unwrapped.observation_manager.compute()
                    policy_state, base_actions, history, sample, extras = self.base_policy_process(
                        obs_dict)
                    states = self.compute_states(policy_state, obs_dict, extras)

                # The four rollout/* sub-step timers below sum to slightly less than
                # rollout total; the gap is small bookkeeping ops (dones reshape/cat,
                # history zeroing, viz) which we don't time individually.
                with self._timer("rollout", times):
                    for _ in tqdm(range(self.num_steps_per_env)):
                        # Get actions.
                        with self._timer("rollout/act", times):
                            residual_actions = self.alg.act(states)
                            final_actions = base_actions + residual_actions

                        # Run env step.
                        self._env_breakdown.clear()
                        with self._timer("rollout/env_step", times):
                            obs_dict, rewards, dones, truncateds, infos = self.env.step(
                                final_actions.float())
                        for _k, _v in self._env_breakdown.items():
                            times[f"rollout/env_step/{_k}"] = times.get(
                                f"rollout/env_step/{_k}", 0.0) + _v

                        # Move time out information to the extras dict
                        # this is only needed for infinite horizon tasks
                        if not self.env.unwrapped.cfg.is_finite_horizon:
                            infos["time_outs"] = truncateds
                        dones = torch.cat([
                            dones.reshape(self.env.unwrapped.num_envs, 1),
                            truncateds.reshape(self.env.unwrapped.num_envs, 1)
                        ],
                                          dim=1)
                        dones = torch.any(dones, dim=1)

                        # Process dones for ppo.
                        with self._timer("rollout/process_env_step", times):
                            self.alg.process_env_step(rewards, dones)

                        # Process dones for base policy and policy state.
                        history[dones] = 0
                        sample[dones] = 0
                        final_actions[dones] = 0

                        if "log" in infos:
                            ep_logs.append(infos["log"])

                        # Visualize actions.
                        if self.debug_viz:
                            self.env.unwrapped.action_manager.get_term('drive_joints').visualize(
                                base_actions, residual_actions)

                        # Process obs for next step
                        with self._timer("rollout/base_policy_process", times):
                            policy_state, base_actions, history, sample, extras = self.base_policy_process(
                                obs_dict, history, sample, final_actions)
                            states = self.compute_states(policy_state, obs_dict, extras)

                # Learning step
                with self._timer("compute_returns", times):
                    self.alg.compute_returns(states)

            print("Update")
            with self._timer("update", times):
                mean_value_loss, mean_surrogate_loss, ppo_diagnostics = self.alg.update()
            self.logger.log_dict(
                {
                    "Mean_value_loss": mean_value_loss,
                    "Mean_surrogate_loss": mean_surrogate_loss,
                    **ppo_diagnostics,
                },
                step=it)

            with self._timer("logging", times):
                self._save_episode_logs(ep_logs, it)
                self._upload_video(it)
                ep_logs.clear()

            # Save checkpoint.
            if it % self.ckpt_save_interval == 0:
                with self._timer("checkpoint", times):
                    self._save_ckpt(os.path.join(self.output_dir, f"model_{it}.pt"))

            self.logger.log_dict({f"time/{k}": v for k, v in times.items()}, step=it)

        self.current_learning_iteration += num_learning_iterations
        self._save_ckpt(os.path.join(self.output_dir,
                                     f"model_{self.current_learning_iteration}.pt"))

    def eval(self, num_eval_iterations, distillation_policy=None, gr00t_policy=False):
        goal_reached_sum = 0
        fall_down_sum = 0
        travel_step_sum = 0
        self.eval_mode()
        for it in range(num_eval_iterations):
            print(10 * "_")
            print(f"Eval iteration: {it}:")
            print("Rollout")
            with torch.inference_mode():
                # Rollout
                self.env.reset()
                # Get obs ready for latent state of world model
                obs_dict = self.env.unwrapped.observation_manager.compute()
                policy_state, base_actions, history, sample, _ = self.base_policy_process(obs_dict)
                goal_reached = torch.zeros_like(obs_dict['eval']['goal_reached'], dtype=torch.bool)
                fall_down = torch.zeros_like(obs_dict['eval']['goal_reached'], dtype=torch.bool)
                travel_step = torch.zeros_like(obs_dict['eval']['goal_reached'], dtype=torch.int32)
                for step in tqdm(range(self.num_steps_per_env)):
                    # Get final actions.
                    if gr00t_policy:
                        final_actions = self.gr00t_policy_process(obs_dict)
                    elif distillation_policy:
                        final_actions = distillation_policy(policy_state)
                    else:
                        residual_actions = self.alg.act_inference([policy_state, None])
                        final_actions = base_actions + residual_actions

                    obs_dict, _, _, _, _ = self.env.step(final_actions)

                    # Update metrics.
                    if 'fall_down' in obs_dict['eval']:
                        fall_down = torch.logical_or(fall_down,
                                                     obs_dict['eval']['fall_down'] & ~goal_reached)
                    goal_reached_cur_step = obs_dict['eval'][
                        'goal_reached'] & ~fall_down & ~goal_reached
                    goal_reached = torch.logical_or(goal_reached, goal_reached_cur_step)
                    travel_step[torch.nonzero(goal_reached_cur_step).squeeze()] = step

                    # Visualize actions.
                    if self.debug_viz:
                        if distillation_policy or gr00t_policy:
                            self.env.unwrapped.action_manager.get_term('drive_joints').visualize(
                                final_actions, torch.zeros_like(final_actions))
                        else:
                            self.env.unwrapped.action_manager.get_term('drive_joints').visualize(
                                base_actions, residual_actions)

                    # Base action and token for next step
                    policy_state, base_actions, history, sample, _ = self.base_policy_process(
                        obs_dict, history, sample, final_actions)

                goal_reached_sum += torch.sum(goal_reached)
                fall_down_sum += torch.sum(fall_down)
                travel_step_sum += torch.sum(travel_step)

                print(
                    f'Iter: {it}. Collision: {fall_down_sum}, GoalReached: {goal_reached_sum},  TravelStep: {travel_step_sum}'    #pylint: disable=line-too-long
                )
                self._upload_video(it)

        total_commands = num_eval_iterations * self.env.unwrapped.num_envs
        total_travel_time = travel_step_sum * self.env.unwrapped.cfg.sim.dt * self.env.unwrapped.cfg.decimation    #pylint: disable=line-too-long
        self.logger.log_dict({
            'eval/total_commands': total_commands,
            'eval/goal_reached_rate': goal_reached_sum / total_commands,
            'eval/fall_down_rate': fall_down_sum / total_commands,
            'eval/total_travel_time': total_travel_time,
            'eval/weighted_travel_time': total_travel_time / (goal_reached_sum / total_commands)
        })

    def record(self, num_eval_iterations, metadata, data_dir):
        self.eval_mode()
        os.makedirs(data_dir, exist_ok=True)
        data_elements = ['image', 'speed', 'route', 'action', 'policy_state', 'goal_heading']
        for it in range(num_eval_iterations):
            print(10 * "_")
            print(f"Record iteration: {it}:")
            data_to_record = {element: [] for element in data_elements}
            with torch.inference_mode():
                # Rollout
                self.env.reset()
                # Get obs ready for latent state of world model
                obs_dict = self.env.unwrapped.observation_manager.compute()
                policy_state, base_action, history, sample, _ = self.base_policy_process(obs_dict)
                goal_reached = torch.zeros_like(obs_dict['eval']['goal_reached'], dtype=torch.bool)
                fall_down = torch.zeros_like(obs_dict['eval']['goal_reached'], dtype=torch.bool)
                for _ in tqdm(range(self.num_steps_per_env)):
                    residual_actions = self.alg.act_inference([policy_state, None])
                    final_actions = base_action + residual_actions
                    self._record_data(obs_dict, policy_state, final_actions, data_to_record)
                    obs_dict, _, _, _, _ = self.env.step(final_actions)
                    # Update task status.
                    if 'fall_down' in obs_dict['eval']:
                        fall_down = torch.logical_or(fall_down,
                                                     obs_dict['eval']['fall_down'] & ~goal_reached)
                    goal_reached_cur_step = obs_dict['eval'][
                        'goal_reached'] & ~fall_down & ~goal_reached
                    goal_reached = torch.logical_or(goal_reached, goal_reached_cur_step)
                    # Base action and token for next step
                    policy_state, base_action, history, sample, _ = self.base_policy_process(
                        obs_dict, history, sample, final_actions)
                # Dump the data.
                with h5py.File(os.path.join(data_dir, f'es_{it}.h5'), 'w') as f:
                    for key, value in metadata.items():
                        f.attrs[key] = value
                    for k, v in data_to_record.items():
                        f.create_dataset(k, data=torch.stack(v, dim=1).cpu())
                    f.create_dataset('action_sigma', data=self.alg.actor_critic.std.cpu())
                    f.create_dataset('goal_reached', data=goal_reached.cpu())
                    f.create_dataset('fall_down', data=fall_down.cpu())

    def _record_data(self, obs_dict, policy_state, actions, data_to_record):
        b = obs_dict["policy"]["camera_rgb_img"].shape[0]
        data_to_record['image'].append(obs_dict["policy"]["camera_rgb_img"].reshape(
            b, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3).float())
        data_to_record['speed'].append(obs_dict["policy"]["base_speed"])
        data_to_record['route'].append(obs_dict["policy"]['route'])
        data_to_record['action'].append(actions)
        data_to_record['policy_state'].append(policy_state)
        data_to_record['goal_heading'].append(obs_dict["policy"]['goal_heading'])

    def _save_episode_logs(self, ep_logs, iteration):
        # Each rank contributes the latest episode-log dict from its rollout
        # (or None if no episodes terminated this iter). Rank 0 then computes
        # a per-key element-weighted mean across all ranks: total elements
        # summed / total elements counted. This is a true cross-rank
        # aggregation, weighted by the number of envs that contributed on
        # each rank (no equal-weight-per-rank skew, no dropped logs when
        # some ranks happen to have empty ep_logs).
        # all_gather_object is structurally symmetric (every rank sends one
        # object, every rank gets the full list back) so it can't deadlock
        # on rank-asymmetric data shapes the way per-key all_reduce can.
        # Move tensor values to CPU before the gather so we don't round-trip
        # cuda tensors through pickle and so the rank-0 aggregation is pure
        # CPU work.
        local = None
        if ep_logs:
            local = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                     for k, v in ep_logs[-1].items()}
        if self.world_size > 1:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if not self.is_rank_zero:
            return
        keys = set()
        for d in gathered:
            if d is not None:
                keys.update(d.keys())
        for key in sorted(keys):
            total = 0.0
            count = 0
            for d in gathered:
                if d is None or key not in d:
                    continue
                v = d[key]
                if not isinstance(v, torch.Tensor):
                    v = torch.Tensor([v])
                if v.ndim == 0:
                    v = v.unsqueeze(0)
                v = v.float()
                total += float(v.sum().item())
                count += int(v.numel())
            if count > 0:
                self.logger.log_dict({key: total / count}, step=iteration)

    def _save_ckpt(self, path, infos=None):
        if not self.is_rank_zero:
            return
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.log_artifact(path, "rl_es_checkpoints", "model")

    def _upload_video(self, iteration):
        if not self.is_rank_zero:
            return
        # Video generation can be slower than this function call, so add one iteration
        # delay for video upload.
        target_iteration = iteration - 1
        target_video_path = os.path.join(
            self.output_dir, 'videos',
            f'rl-video-step-{target_iteration*self.num_steps_per_env}.mp4')
        print(target_video_path)
        if os.path.exists(target_video_path):
            self.logger.log_video(name="episode_video",
                                  video_path=target_video_path,
                                  fps=16,
                                  step=target_iteration)

    def load(self, path, load_optimizer=True):
        # map_location pins both model and optimizer tensors directly onto this
        # rank's device. Without it every rank deserializes onto the original
        # save device (cuda:0 from rank-0's checkpoint) before copying, which
        # spikes GPU 0's memory by N x model size on resume.
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        self.alg.actor_critic.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()

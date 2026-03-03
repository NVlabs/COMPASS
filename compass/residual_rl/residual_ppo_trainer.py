# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from datetime import datetime

import numpy as np
import h5py
import gin
import matplotlib.pyplot as plt
import torch
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
                 debug_viz=False,
                 max_debug_images=2,
                 debug_image_interval=10):
        # Prepare log directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.logger = logger
        # Init environment and base policy.
        self.env = env
        self.base_policy = base_policy
        for param in self.base_policy.parameters():
            param.requires_grad = False
        self.device = device

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

        # Init debug images directory and counter
        self.debug_images_dir = os.path.join(output_dir, 'debug_images')
        if not os.path.exists(self.debug_images_dir):
            os.makedirs(self.debug_images_dir)
        self.image_counter = 0
        # Maximum number of grid images to save (None/0 = Skip saving images)
        # each grid is comprised of 8 images, and so for 64 envs, there would be
        # 64 / 8 = 8 grids. But we can save less than 8 grids if we want to.
        self.max_debug_images = max_debug_images
        # Save images every `debug_image_interval` iterations
        self.debug_image_interval = debug_image_interval

        self.env.reset()

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
            with torch.inference_mode():
                self.env.reset()
                # Get obs ready for policy state and base action from base_policy
                obs_dict = self.env.unwrapped.observation_manager.compute()
                policy_state, base_actions, history, sample, extras = self.base_policy_process(
                    obs_dict)
                states = self.compute_states(policy_state, obs_dict, extras)

                for _ in tqdm(range(self.num_steps_per_env)):
                    # Get actions.
                    residual_actions = self.alg.act(states)
                    final_actions = base_actions + residual_actions

                    # Run env step.
                    obs_dict, rewards, dones, truncateds, infos = self.env.step(
                        final_actions.float())

                    self._save_debug_images(obs_dict, it, _)

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
                    policy_state, base_actions, history, sample, extras = self.base_policy_process(
                        obs_dict, history, sample, final_actions)
                    states = self.compute_states(policy_state, obs_dict, extras)

                # Learning step
                self.alg.compute_returns(states)

            print("Update")
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            self.logger.log_dict(
                {
                    "Mean_value_loss": mean_value_loss,
                    "Mean_surrogate_loss": mean_surrogate_loss,
                },
                step=it)

            self._save_episode_logs(ep_logs, it)
            self._upload_video(it)
            ep_logs.clear()

            # Save checkpoint.
            if it % self.ckpt_save_interval == 0:
                self._save_ckpt(os.path.join(self.output_dir, f"model_{it}.pt"))

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
        for key in ep_logs[0]:
            info_tensor = torch.tensor([], device=self.device)
            ep_info = ep_logs[-1]
            # handle scalar and zero dimensional tensor infos
            if key not in ep_info:
                continue
            if not isinstance(ep_info[key], torch.Tensor):
                ep_info[key] = torch.Tensor([ep_info[key]])
            if len(ep_info[key].shape) == 0:
                ep_info[key] = ep_info[key].unsqueeze(0)
            info_tensor = torch.cat((info_tensor, ep_info[key].to(self.device)))
            value = torch.mean(info_tensor)
            self.logger.log_dict({key: value}, step=iteration)

    def _save_ckpt(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.log_artifact(path, "rl_es_checkpoints", "model")

    def _upload_video(self, iteration):
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
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self):
        self.alg.actor_critic.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()

    def _save_debug_images(self, obs_dict, iteration, step):
        """Save debug images from all cameras as multiple grids for 1 step during training."""
        try:
            if self.max_debug_images is None or self.max_debug_images == 0:
                return

            if "policy" not in obs_dict or "camera_rgb_img" not in obs_dict["policy"]:
                return

            # Only save images for the first step to avoid too many files
            if step != 0:
                return

            # Only save images at specified iteration intervals
            if iteration % self.debug_image_interval != 0:
                return

            camera_rgb = obs_dict["policy"]["camera_rgb_img"]
            batch_size = camera_rgb.shape[0]

            # Convert tensors to numpy
            rgb_images_np = camera_rgb.detach().cpu().numpy()

            # Get depth images if available
            depth_images_np = None
            if "privileged" in obs_dict and "camera_depth_img" in obs_dict["privileged"]:
                depth_images = obs_dict["privileged"]["camera_depth_img"]
                depth_images_np = depth_images.detach().cpu().numpy()

            # Process environments in batches of 8 to create multiple grids
            envs_per_grid = 8
            num_grids = (batch_size + envs_per_grid - 1) // envs_per_grid    # Ceiling division

            # Limit the number of grids if max_debug_images is set
            if self.max_debug_images is not None:
                num_grids = min(num_grids, self.max_debug_images)

            for grid_idx in range(num_grids):
                start_env = grid_idx * envs_per_grid
                end_env = min(start_env + envs_per_grid, batch_size)

                # Prepare images for this grid
                grid_images = []
                subtitles = []

                for env_idx in range(start_env, end_env):
                    # Process RGB image
                    rgb_img = rgb_images_np[env_idx]

                    # Reshape if flattened
                    if len(rgb_img.shape) == 1:
                        height, width = INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]
                        channels = 3
                        rgb_img = rgb_img.reshape(height, width, channels)

                    # Denormalize RGB from [0, 1] to [0, 255] if needed
                    if rgb_img.max() <= 1.0:
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                    else:
                        rgb_img = rgb_img.astype(np.uint8)

                    grid_images.append(rgb_img)
                    subtitles.append(f"RGB Env {env_idx}")

                    # Process depth image if available
                    if depth_images_np is not None:
                        depth_img = depth_images_np[env_idx]

                        # Reshape if flattened
                        if len(depth_img.shape) == 1:
                            height, width = INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]
                            depth_img = depth_img.reshape(height, width)

                        # Normalize depth for visualization (0-255)
                        # Handle NaN and infinity values
                        depth_img_clean = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                        depth_max = depth_img_clean.max()
                        depth_min = depth_img_clean.min()

                        if depth_max > depth_min and depth_max > 0:
                            # Normalize to 0-255 range
                            depth_img_norm = ((depth_img_clean - depth_min) /
                                              (depth_max - depth_min) * 255).astype(np.uint8)
                        else:
                            depth_img_norm = np.zeros_like(depth_img_clean, dtype=np.uint8)

                        # Convert to 3-channel for consistency
                        depth_img_3ch = np.stack([depth_img_norm] * 3, axis=-1)

                        grid_images.append(depth_img_3ch)
                        subtitles.append(f"Depth Env {env_idx}")

                # Create grid layout and log to wandb
                if len(grid_images) > 0:
                    grid_image_path = self._create_image_grid(grid_images, subtitles, iteration,
                                                              step, grid_idx)

                    # Log the image to wandb using the logger with grid index in name
                    if grid_image_path is not None:
                        self.logger.log_image(name=f"debug/camera_grid_{grid_idx}",
                                              image_path=grid_image_path,
                                              step=iteration)

        except (KeyError, IOError, OSError, ValueError, RuntimeError, AttributeError) as e:
            print(f"Warning: Failed to save debug images: {e}")

    def _create_image_grid(self, images, subtitles, iteration, step, grid_idx=0):
        """Create and save a grid of images. Returns the filepath of the saved image."""

        num_images = len(images)
        if num_images == 0:
            return None

        # Calculate grid dimensions (prefer wider grids)
        if num_images <= 4:
            rows, cols = 1, num_images
        elif num_images <= 8:
            rows, cols = 2, 4
        elif num_images <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot images
        for i, (img, subtitle) in enumerate(zip(images, subtitles)):
            if i < len(axes):
                axes[i].imshow(img)
                axes[i].set_title(subtitle, fontsize=10)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')

        # Add overall title
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fig.suptitle(f"Camera Views Grid {grid_idx} - Iter {iteration:04d} Step {step:04d}",
                     fontsize=14)

        # Save the grid
        filename = f"camera_grid_{grid_idx}_iter_{iteration:04d}_step_{step:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.debug_images_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath
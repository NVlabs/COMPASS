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

from isaaclab.app import AppLauncher

from compass.utils.nurec_utils import PARTICLE_SPG_RUNTIME_USD_FILE
from compass.utils.renderer_utils import (
    apply_camera_renderer_settings,
    configure_renderer_runtime,
    configure_nurec_isaacsim_rtx_viewport,
)
from compass.utils.visualizer_utils import (
    configure_kit_scene_partition,
    configure_visualizers,
    requested_visualizers,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="COMPASS Mobility Generalist.")
parser.add_argument(
    "--camera-renderer",
    choices=["isaac_rtx", "ovrtx"],
    default="isaac_rtx",
    help="Camera renderer. OVRTX runs without Kit; select physics with --physics-backend.",
)
parser.add_argument(
    "--physics-backend",
    choices=["physx", "newton", "ovphysx"],
    default=None,
    help="Physics backend. Defaults to physx for Isaac RTX or newton (MJWarp) for OVRTX. "
    "Standalone ovphysx requires --camera-renderer ovrtx.",
)
parser.add_argument(
    "--config-files",
    "-c",
    nargs="+",
    required=True,
    help="The list of the config files.",
)
parser.add_argument(
    "--base-policy-path",
    "-b",
    type=str,
    default=None,
    help="The path to the base policy checkpoint.",
)
parser.add_argument(
    "--distillation-policy-path",
    "-d",
    type=str,
    default=None,
    help="The path to the distillation policy checkpoint.",
)
parser.add_argument(
    "--checkpoint-path",
    "-p",
    type=str,
    default=None,
    help="The path to the checkpoint.",
)
parser.add_argument(
    "--gr00t-policy",
    action="store_true",
    default=False,
    help="Use gr00t policy for evaluation.",
)
parser.add_argument(
    "--logger",
    type=str,
    choices=["wandb", "tensorboard"],
    default="tensorboard",
    help="Logger to use: wandb or tensorboard",
)
parser.add_argument(
    "--wandb-project-name",
    "-n",
    type=str,
    default="compass",
    help="The project name of W&B (only consulted when --logger wandb).",
)
parser.add_argument("--wandb-run-name",
                    "-r",
                    type=str,
                    default="train_run",
                    help="The run name of W&B.")
parser.add_argument(
    "--wandb-entity-name",
    "-e",
    type=str,
    default="nvidia-isaac",
    help="The entity name of W&B.",
)
parser.add_argument("--output-dir",
                    "-o",
                    type=str,
                    required=True,
                    help="The path to the output dir.")
parser.add_argument("--video",
                    action="store_true",
                    default=False,
                    help="Record videos during training.")
parser.add_argument(
    "--video_interval",
    type=int,
    default=10,
    help="Interval between video recordings (in iterations).",
)
parser.add_argument(
    "--camera_sensor_name",
    type=str,
    default="camera",
    help="Name of the onboard camera sensor in env.scene.sensors "
    "used for robot-camera video recording (default: 'camera').",
)
# Optional parameters to override gin config.
parser.add_argument("--embodiment", type=str, help="Embodiment type")
parser.add_argument("--environment", type=str, help="Environment type")
parser.add_argument(
    "--nurec-scene",
    type=str,
    help="NuRec scene to run. Alias for --environment; assets must "
    "already be installed under the mobility_es usd directory.",
)
parser.add_argument(
    "--nurec-usd-file",
    type=str,
    default=PARTICLE_SPG_RUNTIME_USD_FILE,
    help="NuRec USD filename under the selected environment folder.",
)
parser.add_argument(
    "--nurec-omap-file",
    type=str,
    default=None,
    help="NuRec occupancy-map YAML filename under the selected environment folder.",
)
parser.add_argument(
    "--spg-runtime",
    action="store_true",
    default=False,
    help="Force SPG runtime renderer settings before simulation starts. "
    f"This is automatic for {PARTICLE_SPG_RUNTIME_USD_FILE} "
    "when --nurec-scene is set.",
)
parser.add_argument("--num_envs", type=int, help="Number of environments")
parser.add_argument(
    "--precompute_valid_poses",
    action="store_true",
    default=False,
    help="Precompute valid pose locations for faster sampling",
)
parser.add_argument(
    "--precompute_valid_orientations",
    action="store_true",
    default=False,
    help="Precompute valid orientations for each pose location. "
    "If False, uses randomly generated orientations.",
)
parser.add_argument(
    "--disable_terrain",
    action="store_true",
    default=False,
    help="Disable terrain (set terrain to None).",
)

# Multi-GPU training. Pair with `torchrun --nproc_per_node N run.py --distributed ...`;
# AppLauncher consumes this to bind each rank to its own GPU.
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training across multiple GPUs (one process per GPU via torchrun).",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()
if args_cli.nurec_scene is not None:
    if args_cli.environment is not None:
        parser.error("Pass either --nurec-scene or --environment, not both.")
    args_cli.environment = args_cli.nurec_scene
elif args_cli.nurec_omap_file is not None:
    parser.error("--nurec-omap-file requires --nurec-scene.")
try:
    configure_renderer_runtime(args_cli)
except ValueError as exc:
    parser.error(str(exc))

if args_cli.video:
    # Load the FFmpeg-enabled wheel before Kit prepends its bundled OpenCV to sys.path.
    import cv2

    if not cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG):
        parser.error(
            "--video requires FFmpeg-enabled OpenCV. Install requirements.txt and recreate "
            f"the Docker container after rebuilding the image. Loaded OpenCV: {cv2.__file__}")

# OVRTX must never construct AppLauncher, even for headless runs.
app_launcher = None
simulation_app = None
if args_cli.camera_renderer == "isaac_rtx":
    app_launcher = AppLauncher(args_cli, enable_cameras=True)
    simulation_app = app_launcher.app
else:
    try:
        import ovrtx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OVRTX requires Isaac Lab's optional ovrtx dependencies. "
            "Install the ovrtx extra and run with the Kit-less Python environment.") from exc
    ovrtx.register_schema_paths()
    AppLauncher.sync_visualizer_cli_settings_to_carb({
        **vars(args_cli),
        "visualizer_disable_all":
            getattr(args_cli, "visualizer_explicit", False) and not args_cli.visualizer,
    })

import gin
import torch
import torch.distributed as dist
import wandb

from mobility_es.config import environments
from mobility_es.config.carter_env_cfg import CarterGoalReachingEnvCfg
from mobility_es.config.h1_env_cfg import H1GoalReachingEnvCfg
from mobility_es.config.spot_env_cfg import SpotGoalReachingEnvCfg
from mobility_es.config.g1_env_cfg import G1GoalReachingEnvCfg
from mobility_es.config.digit_env_cfg import DigitGoalReachingEnvCfg
from mobility_es.config.nurec_scenes import make_nurec_scene_asset_cfg
from mobility_es.wrapper.env_wrapper import RLESEnvWrapper

from compass.residual_rl.x_mobility_rl import XMobilityBasePolicy
from compass.distillation.distillation import ESDistillationPolicyWrapper
from compass.residual_rl.residual_ppo_trainer import ResidualPPOTrainer
from compass.utils.logger import Logger
from compass.utils.multi_camera_video_recorder import MultiCameraVideoRecorder


class _NoOpLogger:
    """Discards everything. Used on non-rank-0 processes in multi-GPU runs so
    only rank 0 produces TensorBoard / W&B / artifact writes."""

    def log_dict(self, *args, **kwargs):
        pass

    def log_video(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def log_config(self, *args, **kwargs):
        pass

    def close(self):
        pass


# Map from the embedding type to the RL env config.
EmbodimentEnvCfgMap = {
    "h1": H1GoalReachingEnvCfg,
    "spot": SpotGoalReachingEnvCfg,
    "carter": CarterGoalReachingEnvCfg,
    "g1": G1GoalReachingEnvCfg,
    "digit": DigitGoalReachingEnvCfg,
}

# Map from the environment type to the env scene asset config.
EnvSceneAssetCfgMap = {
    "warehouse_single_rack": environments.warehouse_single_rack,
    "galileo_lab": environments.galileo_lab,
    "simple_office": environments.simple_office,
    "combined_single_rack": environments.combined_single_rack,
    "combined_multi_rack": environments.combined_multi_rack,
    "random_envs": environments.random_envs,
    "hospital": environments.hospital,
    "warehouse_multi_rack": environments.warehouse_multi_rack,
}
if args_cli.nurec_scene is not None:
    EnvSceneAssetCfgMap[args_cli.nurec_scene] = make_nurec_scene_asset_cfg(
        args_cli.nurec_scene,
        args_cli.nurec_usd_file,
        args_cli.nurec_omap_file,
    )


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
def run(
    run_mode,
    embodiment,
    environment,
    num_envs,
    num_iterations,
    num_steps_per_iteration,
    seed,
    enable_curriculum=False,
    goal_pose_collision_distance=0.5,
    start_pose_collision_distance=0.75,
    precompute_valid_poses=False,
    precompute_valid_orientations=False,
    disable_terrain=False,
    render_interval=None,
    num_rerenders_on_reset=None,
):

    # Kit-less runs read torchrun ranks directly because AppLauncher is skipped.
    if args_cli.distributed:
        local_rank = (app_launcher.local_rank if app_launcher is not None else int(
            os.environ.get("LOCAL_RANK", "0")))
        global_rank = (app_launcher.global_rank if app_launcher is not None else int(
            os.environ.get("RANK", "0")))
        # Pin PyTorch's current CUDA device to this rank's GPU BEFORE
        # init_process_group / any object-collective. NCCL's object
        # collectives (dist.all_gather_object in _save_episode_logs)
        # serialize through tensors built on torch.cuda.current_device().
        # Without this call current_device() defaults to 0 on every rank
        # and object-collective traffic routes through GPU 0 instead of
        # the rank's GPU. (Tensor all-reduces are unaffected because their
        # tensors carry an explicit device.)
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
        is_rank_zero = global_rank == 0
    else:
        local_rank = 0
        global_rank = 0
        device = (args_cli.device if args_cli.camera_renderer == "ovrtx" else
                  "cuda" if torch.cuda.is_available() else "cpu")
        is_rank_zero = True

    # Setup logger. Only rank 0 writes TensorBoard / W&B / artifacts; other ranks get
    # a no-op logger that discards everything.
    if is_rank_zero:
        logger = Logger(
            log_dir=args_cli.output_dir,
            backend=args_cli.logger,
            experiment_name=args_cli.wandb_run_name,
            project_name=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity_name,
        )
    else:
        logger = _NoOpLogger()

    # Keep policy inference on this run's GPU. Implicit DataParallel across every
    # visible GPU can stall the first rollout on multi-GPU simulation hosts.
    # Retain the wrapper's .module interface; torchrun assigns one GPU per rank.
    policy_device = torch.device(device)
    policy_device_ids = ([
        policy_device.index if policy_device.index is not None else torch.cuda.current_device()
    ] if policy_device.type == "cuda" else None)
    base_policy = XMobilityBasePolicy(args_cli.base_policy_path)
    base_policy = torch.nn.DataParallel(base_policy, device_ids=policy_device_ids)
    base_policy.to(device)
    base_policy.eval()

    # Setup distillated policy.
    if args_cli.distillation_policy_path is not None:
        distillation_policy = ESDistillationPolicyWrapper(args_cli.distillation_policy_path,
                                                          embodiment)
        distillation_policy = torch.nn.DataParallel(distillation_policy,
                                                    device_ids=policy_device_ids)
        distillation_policy.to(device)
        distillation_policy.eval()
    else:
        distillation_policy = None

    # Setup embodiment type.
    if embodiment in EmbodimentEnvCfgMap:
        env_cfg = EmbodimentEnvCfgMap[embodiment]()
    else:
        raise ValueError(f"Unsupported embodiment type: {embodiment}")

    if environment not in EnvSceneAssetCfgMap:
        raise ValueError(f"Unsupported environment type: {environment}")
    env_cfg.scene.environment = EnvSceneAssetCfgMap[environment]
    if render_interval is not None:
        env_cfg.sim.render_interval = render_interval
    if num_rerenders_on_reset is not None:
        env_cfg.num_rerenders_on_reset = num_rerenders_on_reset
    env_cfg.scene.replicate_physics = env_cfg.scene.environment.replicate_physics
    env_cfg.scene.env_spacing = env_cfg.scene.environment.env_spacing
    env_cfg.scene.num_envs = num_envs
    env_cfg.events.reset_base.params["pose_range"] = (env_cfg.scene.environment.pose_sample_range)

    # Setup terrain (disable if requested)
    if disable_terrain or args_cli.disable_terrain:
        env_cfg.scene.terrain = None

    # Setup the curriculum
    if enable_curriculum:
        env_cfg.curriculum.command_min_distance_prob.params[
            "num_steps_per_iteration"] = num_steps_per_iteration
        env_cfg.curriculum.command_min_distance_prob.params["total_iterations"] = (num_iterations)
    else:
        env_cfg.curriculum = None

    requested_viz = requested_visualizers(args_cli)
    configure_visualizers(env_cfg, requested_viz)
    # Set the device before configuring OVRTX's native renderer initialization.
    if args_cli.distributed or args_cli.camera_renderer == "ovrtx":
        env_cfg.sim.device = device
    apply_camera_renderer_settings(env_cfg, args_cli)

    # Setup seed. Per-rank offset diversifies env initial conditions across GPUs so
    # rollouts collected by each rank explore different states (matches Isaac Lab's
    # rsl_rl reference pattern).
    env_cfg.seed = seed + global_rank

    # Set collision distances and max resample trial from gin config
    env_cfg.commands.goal_pose.collision_distance = goal_pose_collision_distance
    env_cfg.events.reset_base.params["collision_distance"] = (start_pose_collision_distance)

    # Set collision distances and max resample trial from gin config
    env_cfg.commands.goal_pose.collision_distance = goal_pose_collision_distance
    env_cfg.events.reset_base.params["collision_distance"] = (start_pose_collision_distance)

    # Disable rewards, termination and curriculum for eval.
    if run_mode == "eval" or run_mode == "record":
        env_cfg.rewards = None
        env_cfg.terminations = None
        env_cfg.curriculum = None
    # Only rank 0 records video — non-rank-0 ranks would compete for the same files
    # under output_dir/videos/ and produce duplicates.
    record_video = args_cli.video and is_rank_zero
    # Use CLI flag if provided, otherwise use gin config
    precompute_flag = args_cli.precompute_valid_poses or precompute_valid_poses
    precompute_orientations_flag = (args_cli.precompute_valid_orientations
                                    or precompute_valid_orientations)
    env = RLESEnvWrapper(
        cfg=env_cfg,
        render_mode=None,
        precompute_valid_poses=precompute_flag,
        precompute_valid_orientations=precompute_orientations_flag,
    )
    if "kit" in requested_viz:
        if args_cli.nurec_scene is not None:
            configure_nurec_isaacsim_rtx_viewport(
                nurec_usd_path=env_cfg.scene.environment.spawn.usd_path,
                spg_runtime=args_cli.spg_runtime,
            )
        else:
            configure_kit_scene_partition()

    # Precompute valid pose locations if requested
    if precompute_flag and env.collision_checker.is_initialized():
        print("Precomputing valid pose locations...")
        env.collision_checker.precompute_valid_poses(
            start_collision_distance=start_pose_collision_distance,
            goal_collision_distance=goal_pose_collision_distance,
            precompute_valid_orientations=precompute_orientations_flag,
        )

    # Setup video if enabled.
    if record_video:
        video_kwargs = {
            "video_folder":
                os.path.join(args_cli.output_dir, "videos"),
            "step_trigger":
                lambda step: step % (num_steps_per_iteration * args_cli.video_interval) == 0,
            "video_length":
                num_steps_per_iteration,
            "camera_sensor_name":
                args_cli.camera_sensor_name,
            "record_kit_viewport":
                "kit" in requested_viz,
        }
        # MultiCameraVideoRecorder reads explicit camera streams and avoids the
        # deprecated IsaacLab render_mode="rgb_array" / Gym RecordVideo path.
        env = MultiCameraVideoRecorder(env, **video_kwargs)

    # Setup the agent.
    rl_trainer = ResidualPPOTrainer(
        env=env,
        base_policy=base_policy,
        output_dir=args_cli.output_dir,
        logger=logger,
        device=device,
        save_debug_viewport_images="kit" in requested_viz,
    )

    if run_mode == "train":
        if args_cli.checkpoint_path:
            rl_trainer.load(path=args_cli.checkpoint_path)
        rl_trainer.learn(num_iterations)
    elif run_mode == "eval":
        if args_cli.checkpoint_path:
            rl_trainer.load(path=args_cli.checkpoint_path, load_optimizer=False)
        rl_trainer.eval(num_iterations, distillation_policy, args_cli.gr00t_policy)
    elif run_mode == "record":
        metadata = {
            "embodiment": embodiment,
            "environment": environment,
            "batch_size": num_envs,
            "sequence_length": num_steps_per_iteration,
            "seed": seed,
            "checkpoint_path": args_cli.checkpoint_path,
        }
        rl_trainer.load(path=args_cli.checkpoint_path, load_optimizer=False)
        rl_trainer.record(num_iterations, metadata, os.path.join(args_cli.output_dir, "data"))
    else:
        raise ValueError("Unsupported run mode.")

    # Log configs.
    logger.log_config(gin_config_to_dictionary(gin.config._OPERATIVE_CONFIG))

    logger.close()
    env.close()


def main():
    # Load parameters from gin-config.
    for config_file in args_cli.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    # Override gin-configurable parameters with command line arguments.
    if args_cli.embodiment is not None:
        gin.bind_parameter("run.embodiment", args_cli.embodiment)
    if args_cli.environment is not None:
        gin.bind_parameter("run.environment", args_cli.environment)
    if args_cli.num_envs is not None:
        gin.bind_parameter("run.num_envs", args_cli.num_envs)
    if args_cli.precompute_valid_poses:
        gin.bind_parameter("run.precompute_valid_poses", True)
    if args_cli.precompute_valid_orientations:
        gin.bind_parameter("run.precompute_valid_orientations", True)
    if args_cli.disable_terrain:
        gin.bind_parameter("run.disable_terrain", True)

    # Run the training/evaluation/recording.
    run()


if __name__ == "__main__":
    # Run the main function.
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()

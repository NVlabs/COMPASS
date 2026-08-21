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
parser.add_argument('--gr00t-policy',
                    action='store_true',
                    default=False,
                    help='Use gr00t policy for evaluation.')
parser.add_argument('--logger',
                    type=str,
                    choices=['wandb', 'tensorboard'],
                    default='tensorboard',
                    help='Logger to use: wandb or tensorboard')
parser.add_argument('--wandb-project-name',
                    '-n',
                    type=str,
                    default='compass',
                    help='The project name of W&B (only consulted when --logger wandb).')
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
parser.add_argument("--camera_sensor_name",
                    type=str,
                    default="camera",
                    help="Name of the onboard camera sensor in env.scene.sensors "
                    "used for robot-camera video recording (default: 'camera').")
# Optional parameters to override gin config.
parser.add_argument('--embodiment', type=str, help='Embodiment type')
parser.add_argument('--environment', type=str, help='Environment type')
parser.add_argument('--num_envs', type=int, help='Number of environments')
parser.add_argument('--precompute_valid_poses',
                    action='store_true',
                    default=False,
                    help='Precompute valid pose locations for faster sampling')
parser.add_argument('--precompute_valid_orientations',
                    action='store_true',
                    default=False,
                    help='Precompute valid orientations for each pose location. '
                    'If False, uses randomly generated orientations.')
parser.add_argument('--disable_terrain',
                    action='store_true',
                    default=False,
                    help='Disable terrain (set terrain to None).')

# Multi-GPU training. Pair with `torchrun --nproc_per_node N run.py --distributed ...`;
# AppLauncher consumes this to bind each rank to its own GPU.
parser.add_argument('--distributed',
                    action='store_true',
                    default=False,
                    help='Run training across multiple GPUs (one process per GPU via torchrun).')

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

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
    'warehouse_multi_rack': environments.warehouse_multi_rack,
}
# Register all NuRec Real2Sim scenes (keyed by their ``--environment`` alias).
EnvSceneAssetCfgMap.update(environments.nurec_envs)

KIT_PERSPECTIVE_CAMERA_PATH = "/OmniverseKit_Persp"
KIT_PPISP_RENDER_PRODUCT_PATH = "/Render/COMPASS_KitPerspective_PPISP"
NUREC_VIEWPORT_SOURCE_RENDER_PRODUCT_PATH = "/Render/front_stereo_camera_left__0"
KIT_SCENE_PARTITION = "env_0"
KIT_VIEWER_EYE = (-2.5, -0.5, 1.5)
KIT_VIEWER_LOOKAT = (0.0, 0.0, 0.35)


def _requested_visualizers():
    requested_viz = getattr(args_cli, 'visualizer', None) or []
    if isinstance(requested_viz, str):
        requested_viz = requested_viz.split(',')
    return [v.strip().lower() for v in requested_viz]


def _append_visualizer_cfg(env_cfg, visualizer_cfg):
    existing_cfgs = env_cfg.sim.visualizer_cfgs
    if existing_cfgs is None:
        env_cfg.sim.visualizer_cfgs = [visualizer_cfg]
    elif isinstance(existing_cfgs, list):
        env_cfg.sim.visualizer_cfgs.append(visualizer_cfg)
    else:
        env_cfg.sim.visualizer_cfgs = [existing_cfgs, visualizer_cfg]


def _configure_kit_visualizer(env_cfg):
    from isaaclab_visualizers.kit import KitVisualizerCfg

    kit_viz_cfg = KitVisualizerCfg()
    kit_viz_cfg.eye = KIT_VIEWER_EYE
    kit_viz_cfg.lookat = KIT_VIEWER_LOOKAT
    kit_viz_cfg.origin_type = "asset"
    kit_viz_cfg.origin_track_path = "robot"
    kit_viz_cfg.origin_env_index = 0
    _append_visualizer_cfg(env_cfg, kit_viz_cfg)


def _package_qualified_asset(package_path, asset_path):
    if not asset_path:
        return asset_path
    if "[" in asset_path:
        return asset_path
    return f"{package_path}[{os.path.basename(asset_path)}]"


def _format_render_product_choices(stage):
    choices = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "RenderProduct":
            continue
        targets = prim.GetRelationship("camera").GetTargets()
        camera_path = str(targets[0]) if targets else "<no camera target>"
        choices.append(f"{prim.GetPath()} -> {camera_path}")
    return "\n".join(f"  {choice}" for choice in sorted(choices)) or "  <none>"


def _copy_source_ppisp_render_product(stage, source_usd_path):
    from pxr import Sdf, Usd

    source_stage = Usd.Stage.Open(source_usd_path)
    if source_stage is None:
        raise RuntimeError(f"Could not open NuRec USD: {source_usd_path}")

    source_rp_path = NUREC_VIEWPORT_SOURCE_RENDER_PRODUCT_PATH
    source_rp_prim = source_stage.GetPrimAtPath(source_rp_path)
    if not source_rp_prim or not source_rp_prim.IsValid():
        raise RuntimeError(
            f"NuRec viewport RenderProduct does not exist in {source_usd_path}: "
            f"{source_rp_path}\nAvailable RenderProducts:\n"
            f"{_format_render_product_choices(source_stage)}")
    if source_rp_prim.GetTypeName() != "RenderProduct":
        raise RuntimeError(
            f"NuRec viewport path is not a RenderProduct in {source_usd_path}: "
            f"{source_rp_path} (type={source_rp_prim.GetTypeName()!r})")

    prim_stack = source_rp_prim.GetPrimStack()
    if not prim_stack:
        raise RuntimeError(f"PPISP RenderProduct has empty prim stack: {source_rp_path}")

    source_layer = prim_stack[0].layer
    session = stage.GetSessionLayer()
    src_path = Sdf.Path(source_rp_path)
    dst_path = Sdf.Path(KIT_PPISP_RENDER_PRODUCT_PATH)
    package_path = source_stage.GetRootLayer().identifier

    with Usd.EditContext(stage, session):
        if stage.GetPrimAtPath(KIT_PPISP_RENDER_PRODUCT_PATH).IsValid():
            stage.RemovePrim(dst_path)

    # Clone the NuRec camera PPISP graph for the GUI viewport. Binding the
    # authored robot-camera RenderProduct directly would switch the viewport to
    # that camera, or mutate the sensor RenderProduct used by observations.
    Sdf.CreatePrimInLayer(session, dst_path)
    Sdf.CopySpec(source_layer, src_path, session, dst_path)

    with Usd.EditContext(stage, session):
        dst_prim = stage.GetPrimAtPath(KIT_PPISP_RENDER_PRODUCT_PATH)
        for prim in Usd.PrimRange(dst_prim):
            for attr in prim.GetAttributes():
                connections = attr.GetConnections()
                remapped = [conn.ReplacePrefix(src_path, dst_path) for conn in connections]
                if remapped != connections:
                    attr.SetConnections(remapped)
                value = attr.Get()
                if isinstance(value, Sdf.AssetPath):
                    attr.Set(Sdf.AssetPath(_package_qualified_asset(package_path, value.path)))
            for rel in prim.GetRelationships():
                targets = rel.GetTargets()
                remapped = [target.ReplacePrefix(src_path, dst_path) for target in targets]
                if remapped != targets:
                    rel.SetTargets(remapped)

            spec = session.GetPrimAtPath(prim.GetPath())
            if spec and spec.referenceList.prependedItems:
                spec.referenceList.prependedItems = [
                    Sdf.Reference(_package_qualified_asset(package_path, ref.assetPath))
                    for ref in spec.referenceList.prependedItems
                ]

        dst_prim.GetRelationship("camera").SetTargets([Sdf.Path(KIT_PERSPECTIVE_CAMERA_PATH)])

    return KIT_PPISP_RENDER_PRODUCT_PATH, source_stage


def _apply_render_settings_from_stage(source_stage):
    import carb.settings

    layer_data = source_stage.GetRootLayer().customLayerData or {}
    render_settings = layer_data.get("renderSettings") or {}
    if not render_settings:
        return

    settings = carb.settings.get_settings()
    for key, value in render_settings.items():
        settings.set("/" + str(key).replace(":", "/"), value)


def _apply_nurec_identity_exposure_to_camera(stage, camera_path):
    from pxr import Sdf, Usd

    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsValid() or camera_prim.GetTypeName() != "Camera":
        return

    identity_exposure = {
        "exposure": 0.0,
        "exposure:fStop": 1.0,
        "exposure:iso": 0.0,
        "exposure:responsivity": 1.0,
        "exposure:time": 1.0,
    }

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera_prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
        camera_prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
        for name, value in identity_exposure.items():
            camera_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        camera_prim.CreateAttribute(
            "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(False)


def _bind_nurec_ppisp_render_product(stage, viewport, nurec_usd_path, quiet=False):
    if not nurec_usd_path:
        return None

    try:
        render_product_path, source_stage = _copy_source_ppisp_render_product(stage, nurec_usd_path)
        _apply_nurec_identity_exposure_to_camera(stage, KIT_PERSPECTIVE_CAMERA_PATH)
        viewport.render_product_path = render_product_path
        if str(getattr(viewport, "render_product_path", "")) != render_product_path:
            if not quiet:
                print(
                    f"[WARN] Kit ignored NuRec PPISP RenderProduct "
                    f"{render_product_path}; keeping default viewport render product.")
            return None

        _apply_render_settings_from_stage(source_stage)
        return render_product_path
    except Exception as exc:    # pylint: disable=broad-except
        if not quiet:
            print(f"[WARN] Could not bind NuRec PPISP RenderProduct: {type(exc).__name__}: {exc}")
        return None


def _set_kit_camera_scene_partition(nurec_usd_path=None, quiet=False):
    try:
        import omni.kit.viewport.utility as viewport_utils
        import omni.usd
        from pxr import Sdf
    except ImportError as exc:
        if not quiet:
            print(f"[WARN] Could not import USD utilities for Kit camera setup: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        if not quiet:
            print("[WARN] Could not configure Kit camera: no active USD stage.")
        return

    camera_prim = stage.GetPrimAtPath(KIT_PERSPECTIVE_CAMERA_PATH)
    if not camera_prim or not camera_prim.IsValid():
        if not quiet:
            print(
                f"[WARN] Could not configure Kit camera: "
                f"{KIT_PERSPECTIVE_CAMERA_PATH} does not exist.")
        return

    attr = camera_prim.GetAttribute("omni:scenePartition")
    if not attr.IsValid():
        attr = camera_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token)
    attr.Set(KIT_SCENE_PARTITION)

    render_product_path = None
    viewport = viewport_utils.get_active_viewport()
    if viewport is not None:
        try:
            viewport.camera_path = Sdf.Path(KIT_PERSPECTIVE_CAMERA_PATH)
        except Exception:    # pylint: disable=broad-except
            pass
        bound_nurec_rp_path = _bind_nurec_ppisp_render_product(
            stage, viewport, nurec_usd_path, quiet=quiet)
        render_product_path = getattr(viewport, "render_product_path", None)
    else:
        bound_nurec_rp_path = None

    if not quiet:
        message = (
            f"[INFO] Set {KIT_PERSPECTIVE_CAMERA_PATH} scene partition to "
            f"{KIT_SCENE_PARTITION}.")
        if bound_nurec_rp_path:
            message += f" Bound NuRec PPISP RenderProduct: {bound_nurec_rp_path}."
        if render_product_path:
            message += f" Active Kit render product: {render_product_path}."
        print(message)


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
        enable_curriculum=False,
        goal_pose_collision_distance=0.5,
        start_pose_collision_distance=0.75,
        precompute_valid_poses=False,
        precompute_valid_orientations=False,
        disable_terrain=False):

    # Multi-GPU distributed setup. With `--distributed`, AppLauncher (already invoked
    # at module load) reads LOCAL_RANK / RANK / WORLD_SIZE from torchrun's env, sets
    # physics/active GPU per rank, and limits CPU threads. We still need to call
    # init_process_group ourselves before any cross-rank op (param broadcast in
    # ResidualPPOTrainer.__init__, gradient all-reduce in PPO.update).
    if args_cli.distributed:
        local_rank = app_launcher.local_rank
        global_rank = app_launcher.global_rank
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        is_rank_zero = True

    # Setup logger. Only rank 0 writes TensorBoard / W&B / artifacts; other ranks get
    # a no-op logger that discards everything.
    if is_rank_zero:
        logger = Logger(log_dir=args_cli.output_dir,
                        backend=args_cli.logger,
                        experiment_name=args_cli.wandb_run_name,
                        project_name=args_cli.wandb_project_name,
                        entity=args_cli.wandb_entity_name)
    else:
        logger = _NoOpLogger()

    # Setup base policy. Pin DataParallel to the rank's GPU when distributed; let it
    # span all visible GPUs in the single-process / single-GPU path (the legacy default).
    base_policy = XMobilityBasePolicy(args_cli.base_policy_path)
    if args_cli.distributed:
        base_policy = torch.nn.DataParallel(base_policy, device_ids=[local_rank])
    else:
        base_policy = torch.nn.DataParallel(base_policy)
    base_policy.to(device)
    base_policy.eval()

    # Setup distillated policy.
    if args_cli.distillation_policy_path is not None:
        distillation_policy = ESDistillationPolicyWrapper(args_cli.distillation_policy_path,
                                                          embodiment)
        if args_cli.distributed:
            distillation_policy = torch.nn.DataParallel(distillation_policy,
                                                        device_ids=[local_rank])
        else:
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
    nurec_usd_path = None
    if environment in environments.nurec_envs:
        nurec_usd_path = getattr(env_cfg.scene.environment.spawn, "usd_path", None)
    env_cfg.scene.replicate_physics = env_cfg.scene.environment.replicate_physics
    env_cfg.scene.env_spacing = env_cfg.scene.environment.env_spacing
    env_cfg.scene.num_envs = num_envs
    env_cfg.events.reset_base.params["pose_range"] = env_cfg.scene.environment.pose_sample_range

    # Setup terrain (disable if requested)
    if disable_terrain or args_cli.disable_terrain:
        env_cfg.scene.terrain = None

    # Setup the curriculum
    if enable_curriculum:
        env_cfg.curriculum.command_min_distance_prob.params[
            "num_steps_per_iteration"] = num_steps_per_iteration
        env_cfg.curriculum.command_min_distance_prob.params["total_iterations"] = num_iterations
    else:
        env_cfg.curriculum = None

    # Keep the legacy ViewerCfg aligned with the Kit visualizer camera. Isaac Lab 3
    # drives Kit through KitVisualizerCfg, but some env UI code still reads ViewerCfg.
    env_cfg.viewer.origin_type = 'asset_root'
    env_cfg.viewer.asset_name = 'robot'
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.cam_prim_path = KIT_PERSPECTIVE_CAMERA_PATH
    env_cfg.viewer.eye = KIT_VIEWER_EYE
    env_cfg.viewer.lookat = KIT_VIEWER_LOOKAT

    # Newton visualizer camera (new Visualizers API). Newton reads its camera from
    # sim.visualizer_cfgs, not ViewerCfg. Kit also uses visualizer_cfgs in Isaac Lab 3,
    # which keeps the GUI perspective camera independent from the onboard robot camera.
    _requested_viz = _requested_visualizers()
    if 'newton' in _requested_viz or 'newton_gl' in _requested_viz:
        from isaaclab_visualizers.newton import NewtonVisualizerCfg
        newton_viz_cfg = NewtonVisualizerCfg()
        newton_viz_cfg.eye = KIT_VIEWER_EYE    # initial interactive framing
        newton_viz_cfg.lookat = KIT_VIEWER_LOOKAT
        newton_viz_cfg.tiled_cam_view = True    # follow-cam panel (Newton's follow path)
        newton_viz_cfg.tiled_cam_num = 1
        newton_viz_cfg.tiled_cam_target_prim_path = "/World/envs/*/Robot"
        newton_viz_cfg.tiled_cam_eye = KIT_VIEWER_EYE
        _append_visualizer_cfg(env_cfg, newton_viz_cfg)
    if 'kit' in _requested_viz:
        _configure_kit_visualizer(env_cfg)

    # Setup seed. Per-rank offset diversifies env initial conditions across GPUs so
    # rollouts collected by each rank explore different states (matches Isaac Lab's
    # rsl_rl reference pattern).
    env_cfg.seed = seed + global_rank

    # Pin PhysX + Isaac Sim's render device to this rank's GPU. Without this every
    # rank's env_cfg.sim.device defaults to cuda:0 and all 8 sims pile onto a single
    # GPU (caught with a Vulkan OOM during material loading on the first attempt).
    if args_cli.distributed:
        env_cfg.sim.device = device

    # Set collision distances and max resample trial from gin config
    env_cfg.commands.goal_pose.collision_distance = goal_pose_collision_distance
    env_cfg.events.reset_base.params["collision_distance"] = start_pose_collision_distance

    # Set collision distances and max resample trial from gin config
    env_cfg.commands.goal_pose.collision_distance = goal_pose_collision_distance
    env_cfg.events.reset_base.params["collision_distance"] = start_pose_collision_distance

    # Disable rewards, termination and curriculum for eval.
    if run_mode == 'eval' or run_mode == 'record':
        env_cfg.rewards = None
        env_cfg.terminations = None
        env_cfg.curriculum = None
    # Only rank 0 records video — non-rank-0 ranks would compete for the same files
    # under output_dir/videos/ and produce duplicates.
    record_video = args_cli.video and is_rank_zero
    # Use CLI flag if provided, otherwise use gin config
    precompute_flag = args_cli.precompute_valid_poses or precompute_valid_poses
    precompute_orientations_flag = args_cli.precompute_valid_orientations or precompute_valid_orientations
    env = RLESEnvWrapper(cfg=env_cfg,
                         render_mode="rgb_array" if record_video else None,
                         precompute_valid_poses=precompute_flag,
                         precompute_valid_orientations=precompute_orientations_flag)
    if 'kit' in _requested_viz:
        _set_kit_camera_scene_partition(nurec_usd_path=nurec_usd_path)

    # Precompute valid pose locations if requested
    if precompute_flag and env.collision_checker.is_initialized():
        print("Precomputing valid pose locations...")
        env.collision_checker.precompute_valid_poses(
            start_collision_distance=start_pose_collision_distance,
            goal_collision_distance=goal_pose_collision_distance,
            precompute_valid_orientations=precompute_orientations_flag)

    # Setup video if enabled.
    if record_video:
        video_kwargs = {
            "video_folder":
                os.path.join(args_cli.output_dir, "videos"),
            "step_trigger":
                lambda step: step % (num_steps_per_iteration * args_cli.video_interval) == 0,
            "video_length":
                num_steps_per_iteration,
            "disable_logger":
                True,
            "camera_sensor_name":
                args_cli.camera_sensor_name,
        }
        # MultiCameraVideoRecorder wraps the viewport stream with gymnasium's
        # RecordVideo internally and additionally records the onboard robot
        # camera sensor to a separate "robot_camera/" sub-folder.
        env = MultiCameraVideoRecorder(env, **video_kwargs)

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
        rl_trainer.eval(num_iterations, distillation_policy, args_cli.gr00t_policy)
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
    if args_cli.precompute_valid_poses:
        gin.bind_parameter('run.precompute_valid_poses', True)
    if args_cli.precompute_valid_orientations:
        gin.bind_parameter('run.precompute_valid_orientations', True)
    if args_cli.disable_terrain:
        gin.bind_parameter('run.disable_terrain', True)

    # Run the training/evaluation/recording.
    run()


if __name__ == '__main__':
    # Run the main function.
    main()
    # Close the sim app.
    simulation_app.close()

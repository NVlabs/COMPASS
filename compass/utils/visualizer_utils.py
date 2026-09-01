# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# pylint: disable=import-outside-toplevel

# Public Exports
__all__ = [
    "KIT_PERSPECTIVE_CAMERA_PATH",
    "KIT_SCENE_PARTITION",
    "configure_kit_scene_partition",
    "configure_visualizers",
    "requested_visualizers",
]

# Kit/Newton Visualizer Defaults
KIT_PERSPECTIVE_CAMERA_PATH = "/OmniverseKit_Persp"
KIT_SCENE_PARTITION = "env_0"
KIT_VIEWER_EYE = (1.5, -1.5, 2.2)
KIT_VIEWER_LOOKAT = (0.0, 0.0, 0.35)


# Public API
def requested_visualizers(args):
    """Return normalized visualizer names from CLI arguments."""
    requested_viz = getattr(args, "visualizer", None) or []
    if isinstance(requested_viz, str):
        requested_viz = requested_viz.split(",")
    return [v.strip().lower() for v in requested_viz]


def configure_visualizers(env_cfg, requested_viz):
    """Configure requested Isaac Lab visualizers."""
    if "newton" in requested_viz or "newton_gl" in requested_viz:
        _configure_newton_visualizer(env_cfg)
    if "kit" in requested_viz:
        _configure_kit_visualizer(env_cfg)


# Private Helpers
def _append_visualizer_cfg(env_cfg, visualizer_cfg):
    """Append a visualizer config while preserving existing config shape."""
    existing_cfgs = env_cfg.sim.visualizer_cfgs
    if existing_cfgs is None:
        env_cfg.sim.visualizer_cfgs = [visualizer_cfg]
    elif isinstance(existing_cfgs, list):
        env_cfg.sim.visualizer_cfgs.append(visualizer_cfg)
    else:
        env_cfg.sim.visualizer_cfgs = [existing_cfgs, visualizer_cfg]


################################
# Kit Visualizer Setup
################################
def configure_kit_scene_partition(quiet=False):
    """Restrict the Kit perspective camera to the first environment clone."""
    try:
        import omni.usd    # pylint: disable=import-outside-toplevel
        from pxr import Sdf    # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        if not quiet:
            print(f"[WARN] Could not import USD utilities for Kit viewport setup: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        if not quiet:
            print("[WARN] Could not configure Kit viewport: no active USD stage.")
        return

    camera_prim = stage.GetPrimAtPath(KIT_PERSPECTIVE_CAMERA_PATH)
    if not camera_prim or not camera_prim.IsValid():
        if not quiet:
            print(f"[WARN] Could not configure Kit viewport: "
                  f"{KIT_PERSPECTIVE_CAMERA_PATH} does not exist.")
        return

    attr = camera_prim.GetAttribute("omni:scenePartition")
    if not attr.IsValid():
        attr = camera_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token)
    attr.Set(KIT_SCENE_PARTITION)
    if not quiet:
        print(f"[INFO] Set {KIT_PERSPECTIVE_CAMERA_PATH} scene partition to "
              f"{KIT_SCENE_PARTITION}.")


def _configure_kit_visualizer(env_cfg):
    """Configure the Kit visualizer follow camera."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    kit_viz_cfg = KitVisualizerCfg()
    kit_viz_cfg.eye = KIT_VIEWER_EYE
    kit_viz_cfg.lookat = KIT_VIEWER_LOOKAT
    kit_viz_cfg.origin_type = "asset"
    kit_viz_cfg.origin_track_path = "robot"
    kit_viz_cfg.origin_env_index = 0
    _append_visualizer_cfg(env_cfg, kit_viz_cfg)


################################
# Newton Visualizer Setup
################################
def _configure_newton_visualizer(env_cfg):
    """Configure Newton visualizer framing and follow-camera panel."""
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    newton_viz_cfg = NewtonVisualizerCfg()
    newton_viz_cfg.eye = KIT_VIEWER_EYE
    newton_viz_cfg.lookat = KIT_VIEWER_LOOKAT
    newton_viz_cfg.tiled_cam_view = True
    newton_viz_cfg.tiled_cam_num = 1
    newton_viz_cfg.tiled_cam_target_prim_path = "/World/envs/*/Robot"
    newton_viz_cfg.tiled_cam_eye = KIT_VIEWER_EYE
    _append_visualizer_cfg(env_cfg, newton_viz_cfg)

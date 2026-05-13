#!/usr/bin/env python3
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
"""Generate a 2D occupancy map (PNG + ROS YAML) from a USD scene.

Wraps Isaac Sim's ``isaacsim.asset.gen.omap.bindings._omap.Generator`` and emits
files compatible with ``OccupancyMapCollisionChecker``.

Output convention matches Isaac Sim's ROS occupancy-map export: the saved image
is rotated 180 degrees from the raw generator buffer and the YAML ``origin`` is
the world-coordinate of the bottom-left map cell.

Run via the Isaac Lab Python wrapper:

    ${ISAACLAB_PATH}/isaaclab.sh -p scripts/generate_omap_from_usd.py \
        compass/rl_env/exts/mobility_es/mobility_es/usd/office/office.usd

By default the PNG + YAML land at ``<usd_dir>/omap/`` next to the USD, which is
where ``OccupancyMapCollisionChecker``'s sibling-fallback looks first.
"""

import argparse
import sys
from pathlib import Path

# Initialize Isaac Sim before importing pxr / omni.physx.
# Match the headless-boot pattern used by .claude/skills/compass/scripts/sage10k_to_usd.py.
from isaacsim import SimulationApp    # pylint: disable=import-error

_app = SimulationApp({"headless": True})

# pylint: disable=wrong-import-position
import numpy as np
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
import PIL.Image

# NOTE: `isaacsim.asset.gen.omap` is intentionally enabled later, AFTER the
# USD stage has finished loading. Enabling it at module import time
# transitively loads `omni.graph.image.core`, which on current Isaac Sim
# (kit 110.0.0 / Isaac Lab 3.0.0-beta1) installs a fabric callback that
# corrupts a std::vector during the next `ctx.open_stage()` and crashes the
# process with `std::out_of_range: no null terminator at count` inside
# `libomni.graph.core.plugin.so`. Deferring the extension enable to
# `_enable_omap_extension()` (called from `main()` post-load) sidesteps the
# bug entirely; the `_omap` binding is only consulted at `generate2d()`
# call time so the lazy import is safe.
_OMAP_EXT_NAME = "isaacsim.asset.gen.omap"

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

# Buffer values written by Generator.update_settings(...) below.
# Generator.get_buffer() returns these integer codes per cell.
OCCUPIED_VALUE = 4
FREE_VALUE = 5
UNKNOWN_VALUE = 6

# Grayscale pixel values written into the output PNG. ROS map_server expects
# a near-white free pixel, near-black occupied pixel, and a mid-gray unknown.
PNG_OCCUPIED = 0
PNG_FREE = 254
PNG_UNKNOWN = 205

# ROS thresholds (matching the existing maps and Isaac Sim defaults).
ROS_OCCUPIED_THRESH = 0.65
ROS_FREE_THRESH = 0.196


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("usd_path", type=str, help="Path to the input USD scene.")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: <usd_dir>/omap/ (next to the USD).",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default=None,
        help="Stem for the output PNG and YAML. Default: <usd_basename>.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.05,
        help="Grid cell size in meters (matches existing bundled maps).",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.1,
        help="Lower height (meters) of the slab used for 2D rasterization.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=0.62,
        help="Upper height (meters) of the slab used for 2D rasterization.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=None,
        help="World-space (x,y) bounds in meters. Default: derived from USD bbox + 1m padding.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Padding (meters) added around the auto-derived bbox. Ignored when --bounds is given.",
    )
    parser.add_argument(
        "--seed-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(0.0, 0.0),
        help="World-space (x,y) seed for the omap generator's reachability flood-fill. "
        "Cells reachable from this point become FREE, others become UNKNOWN. "
        "Default: (0, 0) — the USD's world origin, which is inside the main "
        "scene for the bundled COMPASS USDs. Override (a) when the scene has "
        "multiple disconnected reachable regions and you want a specific one "
        "to be FREE, or (b) when you pass a --bounds that doesn't contain the "
        "USD origin (the flood-fill needs to start inside the slab).",
    )
    return parser.parse_args()


def compute_xy_bounds_from_stage(stage, padding: float):
    """Compute world-space (xmin, xmax, ymin, ymax) bounds from the stage's bbox.

    Mirrors the policy used by Isaac Sim's omap UI "Auto Bound" button
    (isaacsim.asset.gen.omap.ui/.../extension.py around line 350): only
    Tokens.default_ purpose, no extent-hint short-circuit, fresh cache.
    Some USDs ship stale extent hints that mislead BBoxCache into returning
    garbage bounds, so we don't trust them.
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox_cache.Clear()
    # Prefer /World if it exists (USD assembly convention); fall back to root.
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim or not world_prim.IsValid():
        world_prim = stage.GetPseudoRoot()
    world_bbox = bbox_cache.ComputeWorldBound(world_prim)
    aligned = world_bbox.ComputeAlignedRange()
    if aligned.IsEmpty():
        raise RuntimeError("USD bbox is empty; pass --bounds explicitly.")
    minp = aligned.GetMin()
    maxp = aligned.GetMax()
    return (
        float(minp[0]) - padding,
        float(maxp[0]) + padding,
        float(minp[1]) - padding,
        float(maxp[1]) + padding,
    )


def ensure_physics_scene(stage) -> None:
    """The occupancy generator queries PhysX. Ensure a UsdPhysics.Scene exists."""
    physics_path = Sdf.Path("/World/physicsScene")
    if not stage.GetPrimAtPath(physics_path):
        UsdPhysics.Scene.Define(stage, physics_path)


def buffer_to_grayscale_png(buffer, dims) -> np.ndarray:
    """Convert the Generator's flat int buffer into a (H, W) uint8 grayscale image.

    Isaac Sim's ROS occupancy-map UI uses a 180 degree image rotation before
    saving. Convert the raw buffer so the emitted PNG has:

    * columns increasing with world +X
    * rows increasing with world -Y

    The YAML ``origin`` is written separately using the standard ROS bottom-left
    map origin.
    """
    arr = np.array(buffer, dtype=np.int32)
    expected = dims[0] * dims[1]
    if arr.size != expected:
        raise RuntimeError(
            f"Buffer size {arr.size} != dims[0]*dims[1] = {expected} ({dims[0]}x{dims[1]})")
    grid = arr.reshape((dims[1], dims[0]))    # buffer is row-major (H, W) = (y, x)
    img = np.full_like(grid, PNG_UNKNOWN, dtype=np.uint8)
    img[grid == OCCUPIED_VALUE] = PNG_OCCUPIED
    img[grid == FREE_VALUE] = PNG_FREE
    # Buffer convention: row 0 = y_min (bottom), col 0 = x_max (right of world).
    # PNG convention: row 0 = y_max (top), col 0 = x_min (left).
    # Need both flipud (rows) and fliplr (cols) — equivalent to a 180° rotate.
    return np.flipud(np.fliplr(img))


def write_yaml(yaml_path: Path, image_filename: str, resolution: float, origin_xy):
    """Write a ROS-compatible map_server YAML.

    ``origin`` is the world (x, y) of the bottom-left map cell, matching ROS
    map_server and Isaac Sim's "ROS Occupancy Map Parameters File" convention.
    """
    text = (f"image: {image_filename}\n"
            f"resolution: {resolution}\n"
            f"origin: [{origin_xy[0]:.6f}, {origin_xy[1]:.6f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: {ROS_OCCUPIED_THRESH}\n"
            f"free_thresh: {ROS_FREE_THRESH}\n")
    yaml_path.write_text(text)


def main() -> int:
    args = parse_args()

    usd_path = Path(args.usd_path).resolve()
    if not usd_path.exists():
        print(f"ERROR: USD not found: {usd_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else usd_path.parent / "omap"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    map_name = args.map_name or usd_path.stem
    png_path = out_dir / f"{map_name}.png"
    yaml_path = out_dir / "occupancy_map.yaml"

    print(f"[generate_omap] Loading USD: {usd_path}")
    ctx = omni.usd.get_context()
    ctx.open_stage(str(usd_path))
    # Wait for all references / textures / MDL shaders to finish loading before
    # we rasterize. The `get_stage_loading_status()` tuple is (assets_loaded,
    # assets_loading, assets_to_load); we block until the third element drops
    # to zero, capped at ~5 minutes so we don't hang forever on a broken USD.
    # Without this guard, generate2d() can crash kit when it queries fabric
    # for prims whose references are still resolving (mirrors the
    # `tearDown`-side wait in isaacsim.asset.gen.omap's own tests).
    max_load_ticks = 6000
    for tick in range(max_load_ticks):
        omni.kit.app.get_app().update()
        loaded, loading, to_load = ctx.get_stage_loading_status()
        if to_load == 0 and loading == 0:
            print(f"[generate_omap] Stage fully loaded after {tick + 1} ticks "
                  f"(loaded={loaded}, loading={loading}, to_load={to_load})")
            break
    else:
        print(f"[generate_omap] WARNING: stage still loading after {max_load_ticks} ticks; "
              "proceeding anyway — generate2d() may rasterize a partially loaded scene.")

    stage = ctx.get_stage()
    ensure_physics_scene(stage)
    for _ in range(5):
        omni.kit.app.get_app().update()

    if args.bounds is not None:
        xmin, xmax, ymin, ymax = args.bounds
    else:
        print(f"[generate_omap] Computing bounds from USD bbox (padding {args.padding} m)")
        xmin, xmax, ymin, ymax = compute_xy_bounds_from_stage(stage, args.padding)

    print(f"[generate_omap] Bounds (m): x=[{xmin:.3f}, {xmax:.3f}] "
          f"y=[{ymin:.3f}, {ymax:.3f}]   slab z=[{args.z_min:.3f}, {args.z_max:.3f}]")
    print(f"[generate_omap] Cell size: {args.cell_size} m")

    # Drive the generator. The "origin" arg to set_transform is the flood-fill
    # seed: cells reachable from this point become FREE, others UNKNOWN.
    # Defaults to the USD world origin (0, 0) which sits inside the main
    # scene for all bundled COMPASS USDs. Pass --seed-xy when your --bounds
    # clip doesn't include (0, 0) or the scene has multiple disconnected
    # reachable regions.
    z_mid = 0.5 * (args.z_min + args.z_max)
    seed_x, seed_y = args.seed_xy
    print(f"[generate_omap] Flood-fill seed (x, y, z): ({seed_x:.3f}, {seed_y:.3f}, {z_mid:.3f})")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    omni.kit.app.get_app().update()

    # Late-enable the omap extension (see note at module top about the
    # graph.image.core crash if this is done before open_stage()).
    print(f"[generate_omap] Enabling {_OMAP_EXT_NAME} extension")
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate(_OMAP_EXT_NAME, True)
    from isaacsim.asset.gen.omap.bindings import _omap    # pylint: disable=import-outside-toplevel

    physx = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    generator = _omap.Generator(physx, stage_id)
    generator.update_settings(args.cell_size, OCCUPIED_VALUE, FREE_VALUE, UNKNOWN_VALUE)
    # The C++ MapGenerator API documents minPoint/maxPoint as RELATIVE to the
    # inputOrigin (see MapGenerator.h:setTransform). Convert from absolute
    # world bounds to seed-relative offsets here, otherwise the rasterized
    # volume shifts by `seed` and the YAML origin no longer matches the
    # image content (manifested as a constant world-coord offset).
    generator.set_transform(
        (seed_x, seed_y, z_mid),
        (xmin - seed_x, ymin - seed_y, args.z_min),
        (xmax - seed_x, ymax - seed_y, args.z_max),
    )

    # The C++ generator needs at least one app-update tick before generate2d.
    omni.kit.app.get_app().update()
    print("[generate_omap] Running generate2d() ...")
    generator.generate2d()
    omni.kit.app.get_app().update()

    timeline.stop()

    buffer = generator.get_buffer()
    dims = generator.get_dimensions()
    print(f"[generate_omap] Output dims (W x H): {dims[0]} x {dims[1]}  "
          f"(buffer size {len(buffer)})")

    img = buffer_to_grayscale_png(buffer, dims)
    # Save as RGBA to match the existing bundled PNGs — OccupancyMapCollisionChecker
    # accesses img[:, :, 0] regardless of the grayscale value.
    rgba = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
    PIL.Image.fromarray(rgba).save(png_path)
    # ROS occupancy-map convention: YAML origin is the world-space lower-left
    # corner of the map, while image row 0 still stores the top row.
    write_yaml(yaml_path, png_path.name, args.cell_size, (xmin, ymin))

    print(f"[generate_omap] Wrote PNG : {png_path}")
    print(f"[generate_omap] Wrote YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    rc = main()
    _app.close()
    sys.exit(rc)

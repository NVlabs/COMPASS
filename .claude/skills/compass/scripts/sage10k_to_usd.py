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
"""Convert a SAGE-10k scene to a single USD file for COMPASS training.

Uses only Isaac Sim native APIs (pxr) and numpy — no extra dependencies.
Reads the layout JSON + PLY objects and creates a simulation-ready USD scene.

Usage:
    # Must run inside Isaac Sim environment
    conda run -n env_isaaclab python scripts/sage10k_to_usd.py \
        <layout_json_path> <output_usd_path>

Example:
    conda run -n env_isaaclab python scripts/sage10k_to_usd.py \
        ./sage_10k_scenes/teen_bedroom/layout_ed24dd11.json \
        ./compass/rl_env/exts/mobility_es/mobility_es/usd/teen_bedroom/teen_bedroom.usd
"""

import argparse
import json
import math
import os
import struct

import numpy as np

# Initialize Isaac Sim before importing pxr
from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})


def parse_ply(ply_path):
    """Parse a SAGE-10k PLY file.

    These PLYs have separate vertex and texcoord elements, and faces with
    both vertex_indices and texcoord_indices lists.

    Returns: (verts, faces, uvs, face_uvs)
        verts: (N, 3) float32 vertex positions
        faces: (F, 3) int32 vertex indices per face
        uvs: (M, 2) float32 texture coordinates (separate element)
        face_uvs: (F, 3) int32 texcoord indices per face, or None
    """
    with open(ply_path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = 0
        n_texcoords = 0
        n_faces = 0
        is_binary = any("binary_little_endian" in l for l in header_lines)

        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                if parts[1] == "vertex":
                    n_verts = int(parts[2])
                elif parts[1] == "texcoord":
                    n_texcoords = int(parts[2])
                elif parts[1] == "face":
                    n_faces = int(parts[2])

        # Count face property lists to know how many index lists per face
        face_list_count = sum(1 for l in header_lines
                              if l.startswith("property list") and header_lines.index(l) > next(
                                  i for i, x in enumerate(header_lines) if "element face" in x))

        if is_binary:
            # Read vertices (x, y, z as float32)
            verts = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(n_verts, 3).copy()

            # Read texcoords (s, t as float32)
            uvs = None
            if n_texcoords > 0:
                uvs = np.frombuffer(f.read(n_texcoords * 8),
                                    dtype=np.float32).reshape(n_texcoords, 2).copy()

            # Read faces
            faces = []
            face_uvs = []
            for i in range(n_faces):
                # First list: vertex_indices
                n = struct.unpack("<B", f.read(1))[0]
                vi = struct.unpack(f"<{n}i", f.read(4 * n))

                # Second list (if present): texcoord_indices
                ti = None
                if face_list_count >= 2:
                    n2 = struct.unpack("<B", f.read(1))[0]
                    ti = struct.unpack(f"<{n2}i", f.read(4 * n2))

                if n == 3:
                    faces.append(vi)
                    if ti and len(ti) == 3:
                        face_uvs.append(ti)
                elif n >= 4:
                    # Triangulate quads
                    faces.append((vi[0], vi[1], vi[2]))
                    faces.append((vi[0], vi[2], vi[3]))
                    if ti and len(ti) >= 4:
                        face_uvs.append((ti[0], ti[1], ti[2]))
                        face_uvs.append((ti[0], ti[2], ti[3]))

            faces = np.array(faces, dtype=np.int32)
            face_uvs = np.array(face_uvs, dtype=np.int32) if face_uvs else None
        else:
            raise NotImplementedError("Only binary PLY supported")

    return verts, faces, uvs, face_uvs


def load_layout(json_path):
    """Load and parse a SAGE-10k layout JSON."""
    with open(json_path) as f:
        return json.load(f)


def create_usd_scene(layout_data, layout_dir, output_usd_path):
    """Create a USD scene from SAGE layout data using pxr APIs."""
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, Sdf, UsdShade    # pylint: disable=import-outside-toplevel

    os.makedirs(os.path.dirname(output_usd_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(output_usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # NOTE: No GroundPlane or PhysicsScene here — COMPASS provides its own
    # terrain plane and physics scene. Adding them causes duplicate collision
    # surfaces that trigger illegal_contact termination on wheeled robots.

    obj_count = 0
    objects_dir = os.path.join(layout_dir, "objects")

    for room in layout_data.get("rooms", []):
        room_id = room.get("id", "room")
        room_path = f"/World/Room_{room_id[:8]}"
        UsdGeom.Xform.Define(stage, room_path)

        # Process walls. Each SAGE wall has a unique id, but the first 8 chars
        # often collide across walls in the same room (e.g.
        # "wall_room_702bc9b8_north_..." and "..._south_..." both truncate to
        # "wall_roo"). Append the enumeration index to guarantee unique prim
        # paths so each Define creates a distinct prim instead of overwriting.
        wall_x_coords = []
        wall_y_coords = []
        for wi, wall in enumerate(room.get("walls", [])):
            wall_id = wall.get("id", "wall")[:8]
            sp = wall.get("start_point", {})
            ep = wall.get("end_point", {})
            h = wall.get("height", 2.7)
            t = wall.get("thickness", 0.1)

            sx, sy = sp.get("x", 0), sp.get("y", 0)
            ex, ey = ep.get("x", 0), ep.get("y", 0)
            wall_x_coords.extend([sx, ex])
            wall_y_coords.extend([sy, ey])

            dx, dy = ex - sx, ey - sy
            length = math.sqrt(dx * dx + dy * dy)
            if length < 0.01:
                continue

            # Create wall as a box mesh
            nx, ny = -dy / length, dx / length
            hw = t / 2
            pts = [
                Gf.Vec3f(sx - nx * hw, sy - ny * hw, 0),
                Gf.Vec3f(ex - nx * hw, ey - ny * hw, 0),
                Gf.Vec3f(ex + nx * hw, ey + ny * hw, 0),
                Gf.Vec3f(sx + nx * hw, sy + ny * hw, 0),
                Gf.Vec3f(sx - nx * hw, sy - ny * hw, h),
                Gf.Vec3f(ex - nx * hw, ey - ny * hw, h),
                Gf.Vec3f(ex + nx * hw, ey + ny * hw, h),
                Gf.Vec3f(sx + nx * hw, sy + ny * hw, h),
            ]
            faces_idx = [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 2, 6, 7, 3, 0, 3, 7, 4, 1, 5, 6, 2]
            face_counts = [4, 4, 4, 4, 4, 4]

            wall_mesh = UsdGeom.Mesh.Define(stage, f"{room_path}/wall_{wall_id}_{wi}")
            wall_mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
            wall_mesh.CreateFaceVertexCountsAttr(Vt.IntArray(face_counts))
            wall_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces_idx))
            wall_prim = wall_mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(wall_prim)
            PhysxSchema.PhysxCollisionAPI.Apply(wall_prim)

        # Process floor. SAGE walls use absolute layout coordinates and
        # `room["position"]` is the room's corner/origin (not center). Derive
        # the floor footprint from the wall bounds so it always aligns with
        # the walls, regardless of which convention room["position"] follows.
        dims = room.get("dimensions", {})
        rw = dims.get("width", 5)
        rl = dims.get("length", 5)
        rpos = room.get("position", {})
        rx, ry = rpos.get("x", 0), rpos.get("y", 0)

        if wall_x_coords:
            fx0, fx1 = min(wall_x_coords), max(wall_x_coords)
            fy0, fy1 = min(wall_y_coords), max(wall_y_coords)
        else:
            fx0, fx1 = rx - rw / 2, rx + rw / 2
            fy0, fy1 = ry - rl / 2, ry + rl / 2

        floor_mesh = UsdGeom.Mesh.Define(stage, f"{room_path}/floor")
        floor_mesh.CreatePointsAttr(
            Vt.Vec3fArray([
                Gf.Vec3f(fx0, fy0, 0.001),
                Gf.Vec3f(fx1, fy0, 0.001),
                Gf.Vec3f(fx1, fy1, 0.001),
                Gf.Vec3f(fx0, fy1, 0.001),
            ]))
        floor_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        floor_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        # NOTE: No collision on floor — robot drives on COMPASS terrain plane.
        # Adding floor collision causes duplicate ground surfaces and resets.

        # Process objects from PLY files
        for obj in room.get("objects", []):
            obj_id = obj.get("source_id", obj.get("id", ""))[:8]
            if not obj_id:
                continue

            ply_path = os.path.join(objects_dir, f"{obj_id}.ply")
            if not os.path.exists(ply_path):
                continue

            try:
                verts, faces, uvs, face_uvs = parse_ply(ply_path)
            except Exception as e:    # pylint: disable=broad-exception-caught
                print(f"Warning: failed to parse {ply_path}: {e}")
                continue

            # Scale PLY vertices to match the object's target dimensions
            obj_dims = obj.get("dimensions", {})
            tw = obj_dims.get("width", 1.0)
            tl = obj_dims.get("length", 1.0)
            th = obj_dims.get("height", 1.0)

            # Compute current bounding box of PLY mesh
            bbox_min = verts.min(axis=0)
            bbox_max = verts.max(axis=0)
            bbox_size = bbox_max - bbox_min
            bbox_size = np.maximum(bbox_size, 1e-6)    # avoid div by zero

            # Center the mesh at origin, then scale to target dimensions
            center = (bbox_min + bbox_max) / 2
            verts = verts - center
            scale = np.array([tw / bbox_size[0], tl / bbox_size[1], th / bbox_size[2]],
                             dtype=np.float32)
            verts = verts * scale

            # Shift so bottom of mesh is at z=0
            verts[:, 2] -= verts[:, 2].min()

            pos = obj.get("position", {})
            rot = obj.get("rotation", {})
            ox = pos.get("x", 0)
            oy = pos.get("y", 0)
            oz = pos.get("z", 0)
            rx_deg = rot.get("x", 0)
            ry_deg = rot.get("y", 0)
            rz_deg = rot.get("z", 0)

            obj_path = f"{room_path}/obj_{obj_id}_{obj_count}"
            obj_count += 1

            obj_mesh = UsdGeom.Mesh.Define(stage, obj_path)
            obj_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
            face_counts = np.full(len(faces), 3, dtype=np.int32)
            obj_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(face_counts))
            obj_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten()))

            # Apply transform
            xformable = UsdGeom.Xformable(obj_mesh.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
            if rx_deg != 0:
                xformable.AddRotateXOp().Set(float(rx_deg))
            if ry_deg != 0:
                xformable.AddRotateYOp().Set(float(ry_deg))
            if rz_deg != 0:
                xformable.AddRotateZOp().Set(float(rz_deg))

            # Apply texture if available
            tex_path = os.path.join(objects_dir, f"{obj_id}_texture.png")
            if os.path.exists(tex_path) and uvs is not None:
                try:
                    # Use face_uvs indices if available, otherwise fall back to vertex indices
                    if face_uvs is not None and len(face_uvs) == len(faces):
                        uv_indices = np.clip(face_uvs.flatten(), 0, len(uvs) - 1)
                    else:
                        uv_indices = np.clip(faces.flatten(), 0, len(uvs) - 1)
                    tex_coords_arr = uvs[uv_indices].reshape(-1, 2)

                    mat_path = f"{obj_path}_mat"
                    material = UsdShade.Material.Define(stage, mat_path)
                    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),
                                                                   "surface")

                    st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
                    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
                    st_input = material.CreateInput("frame:stPrimvarName", Sdf.ValueTypeNames.Token)
                    st_input.Set("st")
                    st_reader.CreateInput("varname",
                                          Sdf.ValueTypeNames.Token).ConnectToSource(st_input)

                    tex_sampler = UsdShade.Shader.Define(stage, f"{mat_path}/diffuseTexture")
                    tex_sampler.CreateIdAttr("UsdUVTexture")
                    tex_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
                    tex_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                        st_reader.ConnectableAPI(), "result")
                    tex_sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        tex_sampler.ConnectableAPI(), "rgb")

                    primvar = UsdGeom.PrimvarsAPI(obj_mesh).CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
                    primvar.Set(Vt.Vec2fArray.FromNumpy(tex_coords_arr))

                    obj_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
                    UsdShade.MaterialBindingAPI(obj_mesh).Bind(material)
                except Exception as e:    # pylint: disable=broad-exception-caught
                    print(f"Warning: texture failed for {obj_id}: {e}")

            # Add collision — all objects are static for navigation training
            obj_prim = obj_mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(obj_prim)
            ps_collision = PhysxSchema.PhysxCollisionAPI.Apply(obj_prim)
            ps_collision.CreateContactOffsetAttr(0.005)
            ps_collision.CreateRestOffsetAttr(0.001)

    stage.GetRootLayer().Save()
    print(f"USD scene saved to: {output_usd_path}")
    print(f"  Rooms: {len(layout_data.get('rooms', []))}")
    print(f"  Objects loaded: {obj_count}")
    return output_usd_path


def main():
    parser = argparse.ArgumentParser(description="Convert SAGE-10k scene to USD")
    parser.add_argument("layout_json", help="Path to layout JSON file")
    parser.add_argument("output_usd", help="Output USD file path")
    args = parser.parse_args()

    layout_data = load_layout(args.layout_json)
    layout_dir = os.path.dirname(os.path.abspath(args.layout_json))
    create_usd_scene(layout_data, layout_dir, args.output_usd)


if __name__ == "__main__":
    main()
    _app.close()

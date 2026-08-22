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

# pylint: disable=import-outside-toplevel

NUREC_SCENE_NAMES = [
    "nova_carter-galileo",
    "nova_carter-cafe",
    "hand_hold-endeavor-andoria",
    "hand_hold-endeavor-livingroom",
    "hand_hold-endeavor-wormhole",
    "hand_hold-endeavor-wormhole-table",
    "hand_hold-voyager-babyboom",
    "xgrid-wormhole",
]
PARTICLE_SPG_RUNTIME_USD_FILE = "particle_spg-runtime.usdz"
NUREC_SPG_RUNTIME_KIT_ARGS = (
    "--/rtx/spg/enabled=true",
    "--/omni/rtx/nre/compositing/disableNuRecPostProcessings=true",
    "--/rtx/rtpt/gaussian/skipTonemapping/enabled=false",
    "--enable omni.rtx.spg",
)
KIT_PERSPECTIVE_CAMERA_PATH = "/OmniverseKit_Persp"
KIT_SCENE_PARTITION = "env_0"
KIT_PPISP_RENDER_PRODUCT_PATH = "/Render/COMPASS_KitPerspective_PPISP"
NUREC_IDENTITY_EXPOSURE = {
    "exposure": 0.0,
    "exposure:fStop": 1.0,
    "exposure:iso": 0.0,
    "exposure:responsivity": 1.0,
    "exposure:time": 1.0,
}


def apply_nurec_spg_kit_args(args):
    """Prepend the SPG runtime Kit args when --spg-runtime is enabled."""
    if not args.spg_runtime:
        return

    existing_kit_args = getattr(args, "kit_args", None) or ""
    if isinstance(existing_kit_args, list):
        existing_kit_args = " ".join(existing_kit_args)
    additions = [arg for arg in NUREC_SPG_RUNTIME_KIT_ARGS if arg not in existing_kit_args]
    args.kit_args = " ".join(additions + ([existing_kit_args] if existing_kit_args else []))


def _render_product_camera_path(render_product_prim):
    """Return the first camera target from a RenderProduct relationship."""
    from pxr import UsdRender

    product = UsdRender.Product(render_product_prim)
    camera_rel = render_product_prim.GetRelationship("camera")
    targets = camera_rel.GetTargets() if camera_rel else []
    if not targets:
        targets = product.GetCameraRel().GetForwardedTargets()
    return targets[0] if targets else None


def _is_render_product_prim(prim):
    """Return whether a prim is a USD RenderProduct."""
    from pxr import UsdRender

    return prim.GetTypeName() == "RenderProduct" or prim.IsA(UsdRender.Product)


def _iter_render_products(stage):
    """Yield RenderProduct prims in stage traversal order."""
    for prim in stage.Traverse():
        if _is_render_product_prim(prim):
            yield prim


def _format_render_product_choices(stage):
    """Return RenderProduct choices in the same path-to-camera form used in warnings."""
    choices = []
    for prim in _iter_render_products(stage):
        camera_path = _render_product_camera_path(prim) or "<no camera target>"
        choices.append(f"{prim.GetPath()} -> {camera_path}")
    return "\n".join(f"  {choice}" for choice in sorted(choices)) or "  <none>"


def _find_source_render_product(stage):
    """Return the first RenderProduct with a camera target, if any."""
    for prim in _iter_render_products(stage):
        camera_path = _render_product_camera_path(prim)
        if camera_path is not None:
            return prim, camera_path
    return None


def _get_source_render_product(stage, stage_label):
    """Return the source RenderProduct or raise with available choices."""
    discovered = _find_source_render_product(stage)
    if discovered is not None:
        return discovered

    raise RuntimeError(f"Could not find NuRec PPISP RenderProduct in {stage_label}. "
                       f"Available RenderProducts:\n{_format_render_product_choices(stage)}")


def _package_qualified_asset(package_path, asset_path):
    """Qualify package-internal asset references with their USD package."""
    if not asset_path or "[" in asset_path:
        return asset_path
    return f"{package_path}[{os.path.basename(asset_path)}]"


def _copy_source_ppisp_render_product(stage, source_stage, source_usd_path, kit_camera_path):
    """Copy the source PPISP RenderProduct and retarget it to the Kit camera.

    This mirrors show-renderproduct.py's CopyPrim flow, but uses Sdf.CopySpec so
    the RenderProduct can be copied from the source USD package even when it is
    not present in the composed GUI stage.
    """
    from pxr import Sdf, Usd, UsdRender

    source_rp_prim, source_camera_path = _get_source_render_product(source_stage, source_usd_path)
    prim_stack = source_rp_prim.GetPrimStack()
    if not prim_stack:
        raise RuntimeError(f"PPISP RenderProduct has empty prim stack: {source_rp_prim.GetPath()}")

    source_layer = prim_stack[0].layer
    session = stage.GetSessionLayer()
    src_path = Sdf.Path(str(source_rp_prim.GetPath()))
    dst_path = Sdf.Path(KIT_PPISP_RENDER_PRODUCT_PATH)
    package_path = source_stage.GetRootLayer().identifier

    with Usd.EditContext(stage, session):
        existing = stage.GetPrimAtPath(dst_path)
        if existing and existing.IsValid():
            stage.RemovePrim(dst_path)

    Sdf.CreatePrimInLayer(session, dst_path)
    Sdf.CopySpec(source_layer, src_path, session, dst_path)

    with Usd.EditContext(stage, session):
        dst_prim = stage.GetPrimAtPath(dst_path)
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

        UsdRender.Product(dst_prim).GetCameraRel().SetTargets([Sdf.Path(kit_camera_path)])

    return KIT_PPISP_RENDER_PRODUCT_PATH, source_camera_path


def _apply_render_settings_from_stage(stage):
    """Apply render settings authored in the source stage custom layer data."""
    import carb.settings

    layer_data = stage.GetRootLayer().customLayerData or {}
    render_settings = layer_data.get("renderSettings") or {}
    if not render_settings:
        return

    settings = carb.settings.get_settings()
    for key, value in render_settings.items():
        settings.set("/" + str(key).replace(":", "/"), value)


def _copy_camera_metadata_to_chase_camera(stage, source_stage, source_camera_path, kit_camera_path):
    """Copy NuRec PPISP camera metadata while preserving the Kit chase-camera pose."""
    from pxr import Sdf, Usd

    source_camera_prim = source_stage.GetPrimAtPath(source_camera_path)
    if (not source_camera_prim or not source_camera_prim.IsValid()
            or source_camera_prim.GetTypeName() != "Camera"):
        raise RuntimeError(f"Viewport source camera does not exist: {source_camera_path}")

    chase_camera_prim = stage.GetPrimAtPath(kit_camera_path)
    if (not chase_camera_prim or not chase_camera_prim.IsValid()
            or chase_camera_prim.GetTypeName() != "Camera"):
        raise RuntimeError(f"Kit chase camera does not exist: {kit_camera_path}")

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        api_schemas = source_camera_prim.GetMetadata("apiSchemas")
        if api_schemas:
            chase_camera_prim.SetMetadata("apiSchemas", api_schemas)

        for src_attr in source_camera_prim.GetAuthoredAttributes():
            name = src_attr.GetName()
            if not name.startswith("ppisp:"):
                continue

            value = src_attr.Get()
            if value is None:
                continue

            dst_attr = chase_camera_prim.GetAttribute(name)
            if not dst_attr:
                dst_attr = chase_camera_prim.CreateAttribute(name, src_attr.GetTypeName())
            dst_attr.Set(value)

        chase_camera_prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
        chase_camera_prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
        for name, value in NUREC_IDENTITY_EXPOSURE.items():
            chase_camera_prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        chase_camera_prim.CreateAttribute("omni:rtx:autoExposure:enabled",
                                          Sdf.ValueTypeNames.Bool).Set(False)


def _bind_nurec_ppisp_render_product(stage, viewport, nurec_usd_path, kit_camera_path, quiet=False):
    """Bind the Kit viewport to a copied NuRec PPISP RenderProduct."""
    if not nurec_usd_path:
        return None

    try:
        from pxr import Sdf, Usd

        source_stage = Usd.Stage.Open(nurec_usd_path)
        if source_stage is None:
            raise RuntimeError(f"Could not open NuRec USD: {nurec_usd_path}")

        render_product_path, source_camera_path = _copy_source_ppisp_render_product(
            stage, source_stage, nurec_usd_path, kit_camera_path)
        _copy_camera_metadata_to_chase_camera(stage, source_stage, source_camera_path,
                                              kit_camera_path)
        _apply_render_settings_from_stage(source_stage)

        viewport.camera_path = Sdf.Path(kit_camera_path)
        viewport.render_product_path = render_product_path
        if str(getattr(viewport, "render_product_path", "")) != render_product_path:
            if not quiet:
                print(f"[WARN] Kit ignored viewport RenderProduct "
                      f"{render_product_path}; keeping default viewport render product.")
            return None

        return render_product_path
    except Exception as exc:    # pylint: disable=broad-except
        if not quiet:
            print(f"[WARN] Could not bind Kit viewport RenderProduct: "
                  f"{type(exc).__name__}: {exc}")
        return None


def configure_nurec_kit_viewport(nurec_usd_path, spg_runtime, quiet=False):
    """Configure the active Kit viewport for NuRec scene visualization.

    The viewport keeps COMPASS's external chase camera. For SPG runtime assets,
    copy the source PPISP RenderProduct and bind it to that camera so the
    viewport uses the same sensor-processing path as show-renderproduct.py.
    """
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
            print(f"[WARN] Could not configure Kit camera: "
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
        if spg_runtime:
            bound_nurec_rp_path = _bind_nurec_ppisp_render_product(stage,
                                                                   viewport,
                                                                   nurec_usd_path,
                                                                   KIT_PERSPECTIVE_CAMERA_PATH,
                                                                   quiet=quiet)
        else:
            bound_nurec_rp_path = None
        render_product_path = getattr(viewport, "render_product_path", None)
    else:
        bound_nurec_rp_path = None

    if not quiet:
        message = (f"[INFO] Set {KIT_PERSPECTIVE_CAMERA_PATH} scene partition to "
                   f"{KIT_SCENE_PARTITION}.")
        if bound_nurec_rp_path:
            message += f" Bound Kit viewport RenderProduct: {bound_nurec_rp_path}."
        if render_product_path:
            message += f" Active Kit render product: {render_product_path}."
        print(message)

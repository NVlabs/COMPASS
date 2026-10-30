# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Binary scene transfer for NuRec assets with large Gaussian arrays."""

import atexit
import os
import tempfile
from functools import lru_cache


@lru_cache(maxsize=1)
def _export_directory():
    # OVRTX may read the sublayer after initial import. Keep it until process exit,
    # including across multiple renderer instances, and isolate distributed ranks.
    directory = tempfile.TemporaryDirectory(prefix="compass_ovrtx_")
    atexit.register(directory.cleanup)
    return directory


def export_stage_to_binary_sublayer(stage, num_envs, source_paths, keep_env_roots=True):
    """Return a small USDA layer referencing a flattened binary scene.

    Isaac Lab's text export produces an empty default value for Galileo's
    56-million-entry SH array with the bundled USD 25.05 runtime. USDC preserves
    the array and its metadata without converting the coefficients to text.
    Match Isaac Lab's source pruning for both legacy and ovstage cloning, using
    an export-only session so the simulation stage remains unchanged.
    """
    from isaaclab_ov.renderers.ovrtx_usd import _collect_prims_to_deactivate
    from pxr import Sdf, Usd

    export_stage = stage
    if num_envs > 1:
        export_session = Sdf.Layer.CreateAnonymous()
        export_session.subLayerPaths = [stage.GetSessionLayer().identifier]
        export_stage = Usd.Stage.Open(stage.GetRootLayer(), export_session)
        envs_prim = export_stage.GetPrimAtPath("/World/envs")
        if not envs_prim.IsValid():
            raise RuntimeError("Failed to get prim at path: /World/envs")
        source_path_set = frozenset(map(Sdf.Path, source_paths))
        prim_paths = []
        if keep_env_roots:
            for child in envs_prim.GetChildren():
                if child.GetPath() not in source_path_set:
                    prim_paths.extend(_collect_prims_to_deactivate(child, source_path_set))
        else:
            prim_paths = _collect_prims_to_deactivate(envs_prim, source_path_set)
        with Sdf.ChangeBlock():
            for prim_path in prim_paths:
                Sdf.CreatePrimInLayer(export_session, prim_path).active = False

    descriptor, path = tempfile.mkstemp(suffix=".usdc", dir=_export_directory().name)
    os.close(descriptor)
    flattened = export_stage.Flatten()
    if not flattened.Export(path):
        raise RuntimeError(f"Failed to export OVRTX binary scene: {path}")
    root = Sdf.Layer.CreateAnonymous("ovrtx.usda")
    # Stage metadata (e.g. upAxis and defaultPrim) is read from the root/session
    # layers rather than composed from sublayers.
    for key in flattened.pseudoRoot.ListInfoKeys():
        root.pseudoRoot.SetInfo(key, flattened.pseudoRoot.GetInfo(key))
    root.subLayerPaths = [path]
    return root.ExportToString()

# SPDX-License-Identifier: Apache-2.0
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

from functools import wraps

# Import simulation dependencies only after renderer startup is configured.
# pylint: disable=import-outside-toplevel


def configure_gaussian_hdr(enabled):
    """Configure each NuRec SPG render product for the PPISP HDR input.

    Author the required Gaussian setting directly in USD without debug-only
    environment variables. The newer integration builds products separately.
    Keep the adapter local to Isaac Lab and leave the native OVRTX API unchanged.
    """
    from isaaclab_ov.renderers import ovrtx_renderer

    key = '_compass_original_build_render_product_as_string'
    if not hasattr(ovrtx_renderer, key):
        setattr(ovrtx_renderer, key, ovrtx_renderer.build_render_product_as_string)
    original = getattr(ovrtx_renderer, key)
    if not enabled:
        ovrtx_renderer.build_render_product_as_string = original
        return

    @wraps(original)
    def build_with_gaussian_hdr(*args, **kwargs):
        result = original(*args, **kwargs)
        # Older Isaac Lab builders return the layer and product path together;
        # newer builders return only the layer. Preserve either interface.
        if isinstance(result, tuple):
            usd_text, product_path = result
            return apply_gaussian_hdr(usd_text), product_path
        return apply_gaussian_hdr(result)

    ovrtx_renderer.build_render_product_as_string = build_with_gaussian_hdr


def apply_gaussian_hdr(usd_text):
    """Author a typed per-product setting while preserving the layer metadata."""
    from pxr import Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous('compass-ovrtx-products.usda')
    if not layer.ImportFromString(usd_text):
        raise RuntimeError('Could not parse OVRTX render-product layer')
    stage = Usd.Stage.Open(layer)
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'RenderProduct':
            prim.CreateAttribute('omni:rtx:rtpt:gaussian:skipTonemapping:enabled',
                                 Sdf.ValueTypeNames.Bool,
                                 custom=False).Set(False)
    return layer.ExportToString()

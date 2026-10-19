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
import os

# Import simulation dependencies only after renderer startup is configured.
# pylint: disable=import-outside-toplevel


def configure_gaussian_hdr(enabled):
    """Mirror the Gaussian tonemapping environment override into each product.

    Isaac Lab's OVRTX 0.5 path creates render products separately from the scene.
    The environment override alone does not supply their Gaussian HDR setting.
    Keep the adapter local to Isaac Lab and leave the native OVRTX API unchanged.
    """
    from isaaclab_ov.renderers import ovrtx_renderer

    # The backend owner and separate render-product layers arrived with the
    # new integration. Older integrations retain their existing startup path.
    if not hasattr(ovrtx_renderer, 'OVRTXBackend'):
        return
    key = '_compass_original_build_render_product_as_string'
    if not hasattr(ovrtx_renderer, key):
        setattr(ovrtx_renderer, key, ovrtx_renderer.build_render_product_as_string)
    original = getattr(ovrtx_renderer, key)
    if not enabled:
        ovrtx_renderer.build_render_product_as_string = original
        return

    @wraps(original)
    def build_with_gaussian_hdr(*args, **kwargs):
        return apply_gaussian_hdr(original(*args, **kwargs))

    ovrtx_renderer.build_render_product_as_string = build_with_gaussian_hdr


def apply_gaussian_hdr(usd_text):
    """Author a typed per-product setting while preserving the layer metadata."""
    from pxr import Sdf, Usd

    layer = Sdf.Layer.CreateAnonymous('compass-ovrtx-products.usda')
    if not layer.ImportFromString(usd_text):
        raise RuntimeError('Could not parse OVRTX render-product layer')
    stage = Usd.Stage.Open(layer)
    skip = os.environ.get('OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled',
                          '0').lower() in {'1', 'true', 'yes', 'on'}
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'RenderProduct':
            prim.CreateAttribute('omni:rtx:rtpt:gaussian:skipTonemapping:enabled',
                                 Sdf.ValueTypeNames.Bool,
                                 custom=False).Set(skip)
    return layer.ExportToString()

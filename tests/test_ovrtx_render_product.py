# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import patch

try:
    from pxr import Sdf, Usd
except ImportError:
    Usd = None

from compass.utils.ovrtx_render_product import apply_gaussian_hdr


@unittest.skipIf(Usd is None, 'Requires OpenUSD')
class RenderProductHDRTests(unittest.TestCase):
    def test_hdr_setting_preserves_layer_and_camera_relationships(self):
        text = '''#usda 1.0
(defaultPrim = "RenderCamera_0")
def Scope "RenderCamera_0" {
    def RenderProduct "RenderProduct" {
        rel camera = </World/envs/env_0/Camera>
        uniform int2 resolution = (512, 320)
    }
}
'''
        for override, expected in [('0', False), ('1', True)]:
            with self.subTest(override=override), patch.dict(os.environ, {
                    'OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled': override}):
                layer = Sdf.Layer.CreateAnonymous()
                layer.ImportFromString(apply_gaussian_hdr(text))
                stage = Usd.Stage.Open(layer)
                self.assertEqual(str(stage.GetDefaultPrim().GetPath()), '/RenderCamera_0')
                product = stage.GetPrimAtPath('/RenderCamera_0/RenderProduct')
                self.assertEqual(product.GetRelationship('camera').GetTargets(),
                                 [Sdf.Path('/World/envs/env_0/Camera')])
                self.assertEqual(tuple(product.GetAttribute('resolution').Get()), (512, 320))
                attr = product.GetAttribute('omni:rtx:rtpt:gaussian:skipTonemapping:enabled')
                self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Bool)
                self.assertEqual(attr.Get(), expected)

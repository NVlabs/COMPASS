# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from pxr import Sdf, Usd
except ImportError:
    Usd = None

from compass.utils.ovrtx_render_product import apply_gaussian_hdr, configure_gaussian_hdr


@unittest.skipIf(Usd is None, 'Requires OpenUSD')
class RenderProductHDRTests(unittest.TestCase):
    """Verify the PPISP HDR setting and both render-product builder interfaces."""

    def setUp(self):
        self.usd_text = '''#usda 1.0
(defaultPrim = "RenderCamera_0")
def Scope "RenderCamera_0" {
    def RenderProduct "RenderProduct" {
        rel camera = </World/envs/env_0/Camera>
        uniform int2 resolution = (512, 320)
    }
}
'''

    def assert_hdr_layer(self, text):
        layer = Sdf.Layer.CreateAnonymous()
        self.assertTrue(layer.ImportFromString(text))
        stage = Usd.Stage.Open(layer)
        self.assertEqual(str(stage.GetDefaultPrim().GetPath()), '/RenderCamera_0')
        product = stage.GetPrimAtPath('/RenderCamera_0/RenderProduct')
        self.assertEqual(
            product.GetRelationship('camera').GetTargets(), [Sdf.Path('/World/envs/env_0/Camera')])
        self.assertEqual(tuple(product.GetAttribute('resolution').Get()), (512, 320))
        attr = product.GetAttribute('omni:rtx:rtpt:gaussian:skipTonemapping:enabled')
        self.assertEqual(attr.GetTypeName(), Sdf.ValueTypeNames.Bool)
        self.assertFalse(attr.Get())

    def test_hdr_setting_preserves_layer_and_camera_relationships(self):
        self.assert_hdr_layer(apply_gaussian_hdr(self.usd_text))

    def test_hdr_setting_replaces_incompatible_authored_value(self):
        text = self.usd_text.replace(
            'rel camera',
            'bool omni:rtx:rtpt:gaussian:skipTonemapping:enabled = true\n        rel camera')
        self.assert_hdr_layer(apply_gaussian_hdr(text))

    def test_adapter_preserves_builder_interfaces_and_restores_original(self):
        for returns_path in (False, True):
            with self.subTest(returns_path=returns_path):
                result = (self.usd_text,
                          '/RenderCamera_0/RenderProduct') if returns_path else self.usd_text
                original = Mock(return_value=result)
                adapter = SimpleNamespace(build_render_product_as_string=original)
                with patch.dict(sys.modules,
                                {'isaaclab_ov.renderers': SimpleNamespace(ovrtx_renderer=adapter)}):
                    configure_gaussian_hdr(True)
                    configure_gaussian_hdr(True)
                    actual = adapter.build_render_product_as_string('spec', device_id=0)
                    original.assert_called_once_with('spec', device_id=0)
                    if returns_path:
                        text, path = actual
                        self.assertEqual(path, '/RenderCamera_0/RenderProduct')
                    else:
                        text = actual
                    self.assert_hdr_layer(text)
                    configure_gaussian_hdr(False)
                    self.assertIs(adapter.build_render_product_as_string, original)

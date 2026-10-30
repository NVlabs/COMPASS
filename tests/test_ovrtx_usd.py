# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check binary scene transfer against Isaac Lab's export with real USD layers."""

import unittest

try:
    from isaaclab_ov.renderers.ovrtx_usd import export_stage_to_string
    from pxr import Gf, Sdf, Usd, Vt
except ImportError:
    Usd = None

from compass.utils.ovrtx_usd import export_stage_to_binary_sublayer


@unittest.skipIf(Usd is None, "Requires the Isaac Lab USD runtime")
class BinarySceneExportTests(unittest.TestCase):

    def make_stage(self):
        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)
        stage.SetMetadata("upAxis", "Z")
        for index in range(3):
            prim = stage.DefinePrim(f"/World/envs/env_{index}/Gaussians", "Xform")
            attr = prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients",
                                        Sdf.ValueTypeNames.Float3Array)
            attr.Set(Vt.Vec3fArray([Gf.Vec3f(1, 2, 3)] * 16))
            attr.SetMetadata("elementSize", 16)
        # Include session opinions and a nested clone source.
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            prim.CreateAttribute("sessionValue", Sdf.ValueTypeNames.Int).Set(42)
        return stage

    def test_roundtrip_matches_text_export_and_preserves_source(self):
        for count in (1, 3):
            for keep_roots in (True, False):
                for source in ("/World/envs/env_0", "/World/envs/env_0/Gaussians"):
                    with self.subTest(count=count, keep_roots=keep_roots, source=source):
                        stage = self.make_stage()
                        before = stage.ExportToString()
                        expected = export_stage_to_string(stage, count, (source,), keep_roots)
                        text = export_stage_to_binary_sublayer(stage, count, (source,), keep_roots)
                        # Coefficients must stay in binary, out of the inline USDA.
                        self.assertNotIn("sphericalHarmonicsCoefficients", text)
                        root = Sdf.Layer.CreateAnonymous("test.usda")
                        self.assertTrue(root.ImportFromString(text))
                        self.assertTrue(root.subLayerPaths[0].endswith(".usdc"))
                        result = Usd.Stage.Open(root)
                        # Export comments contain anonymous source identifiers.
                        self.assertEqual(result.Flatten(False).ExportToString(),
                                         self._flatten_text(expected))
                        self.assertEqual(stage.ExportToString(), before)
                        attr = result.GetPrimAtPath("/World/envs/env_0/Gaussians").GetAttribute(
                            "radiance:sphericalHarmonicsCoefficients")
                        self.assertEqual(len(attr.Get()), 16)
                        self.assertEqual(attr.GetMetadata("elementSize"), 16)

    def _flatten_text(self, text):
        layer = Sdf.Layer.CreateAnonymous("test.usda")
        layer.ImportFromString(text)
        return Usd.Stage.Open(layer).Flatten(False).ExportToString()

    def test_missing_envs_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Failed to get prim"):
            export_stage_to_binary_sublayer(Usd.Stage.CreateInMemory(), 2, ())


if __name__ == "__main__":
    unittest.main()

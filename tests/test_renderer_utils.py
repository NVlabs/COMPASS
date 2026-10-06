# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer startup contracts that can be checked without Isaac Sim or a GPU."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from compass.utils.nurec_utils import PARTICLE_SPG_RUNTIME_USD_FILE
from compass.utils.renderer_utils import _configure_ovrtx_device, configure_renderer_runtime


class RendererRuntimeTests(unittest.TestCase):
    """Verify renderer startup configuration without loading simulation runtimes."""

    def make_args(self, **overrides):
        args = {
            "camera_renderer": "ovrtx",
            "physics_backend": None,
            "nurec_scene": "nova_carter-galileo",
            "nurec_usd_file": PARTICLE_SPG_RUNTIME_USD_FILE,
            "spg_runtime": False,
            "visualizer": None,
            "livestream": -1,
            "kit_args": "",
            "experience": "",
            "xr": False,
        }
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_default_physics_backends(self):
        for renderer, expected in (("isaac_rtx", "physx"), ("ovrtx", "newton")):
            with self.subTest(renderer=renderer), patch.dict(os.environ, {}, clear=True):
                args = self.make_args(camera_renderer=renderer)
                configure_renderer_runtime(args)
                self.assertEqual(args.physics_backend, expected)

    def test_explicit_ovphysx_runs_without_kit(self):
        args = self.make_args(physics_backend="ovphysx")
        with patch.dict(os.environ, {}, clear=True):
            configure_renderer_runtime(args)
            self.assertEqual(args.physics_backend, "ovphysx")
            self.assertTrue(args.spg_runtime)
            self.assertEqual(args.kit_args, "")
            self.assertEqual(os.environ["OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled"], "0")

    def test_incompatible_physics_rejected_before_runtime_configuration(self):
        for renderer, backend in (("isaac_rtx", "ovphysx"), ("isaac_rtx", "newton"), ("ovrtx",
                                                                                      "physx")):
            with self.subTest(renderer=renderer, backend=backend), patch.dict(os.environ, {},
                                                                              clear=True):
                args = self.make_args(camera_renderer=renderer, physics_backend=backend)
                with self.assertRaisesRegex(ValueError, "does not support --physics-backend"):
                    configure_renderer_runtime(args)
                self.assertEqual(args.kit_args, "")
                self.assertFalse(args.spg_runtime)
                self.assertNotIn("OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled", os.environ)

    def test_ovrtx_spg_uses_environment_without_kit_args(self):
        args = self.make_args()
        with patch.dict(os.environ, {}, clear=True):
            configure_renderer_runtime(args)
            self.assertEqual(os.environ["OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled"], "0")
        self.assertTrue(args.spg_runtime)
        self.assertEqual(args.kit_args, "")

    def test_ovrtx_preserves_explicit_tonemapping(self):
        key = "OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled"
        with patch.dict(os.environ, {key: "1"}, clear=True):
            configure_renderer_runtime(self.make_args())
            self.assertEqual(os.environ[key], "1")

    def test_synthetic_scene_does_not_enable_spg(self):
        args = self.make_args(nurec_scene=None)
        with patch.dict(os.environ, {}, clear=True):
            configure_renderer_runtime(args)
        self.assertFalse(args.spg_runtime)
        self.assertEqual(args.kit_args, "")

    def test_ovrtx_rejects_kit_options_before_changing_environment(self):
        for overrides in ({
                "visualizer": ["kit"]
        }, {
                "visualizer": "newton_gl,kit"
        }, {
                "livestream": 1
        }, {
                "xr": True
        }, {
                "kit_args": "--enable omni.rtx.spg"
        }, {
                "experience": "custom.kit"
        }):
            with self.subTest(overrides=overrides), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    configure_renderer_runtime(self.make_args(**overrides))
                self.assertNotIn("OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled", os.environ)

    def test_livestream_environment_and_cli_precedence(self):
        with patch.dict(os.environ, {"LIVESTREAM": "1"}, clear=True):
            with self.assertRaises(ValueError):
                configure_renderer_runtime(self.make_args())
            configure_renderer_runtime(self.make_args(livestream=0))

    def test_isaac_rtx_keeps_spg_kit_configuration(self):
        args = self.make_args(camera_renderer="isaac_rtx",
                              visualizer=["kit"],
                              kit_args="--custom=value")
        with patch.dict(os.environ, {}, clear=True):
            configure_renderer_runtime(args)
            self.assertNotIn("OVRTX_rtx_rtpt_gaussian_skipTonemapping_enabled", os.environ)
        self.assertIn("--enable omni.rtx.spg", args.kit_args)
        self.assertIn("--custom=value", args.kit_args)
        configured = args.kit_args
        configure_renderer_runtime(args)
        self.assertEqual(args.kit_args, configured)


class OVRTXDeviceTests(unittest.TestCase):
    """Check the native config boundary without importing GPU libraries."""

    def setUp(self):
        self.native_config = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        self.adapter = SimpleNamespace(RendererConfig=self.native_config)
        self.ovrtx = SimpleNamespace(RendererConfig=self.native_config)
        self.torch = SimpleNamespace(device=Mock(),
                                     cuda=SimpleNamespace(current_device=Mock(return_value=3)))
        self.modules = patch.dict(
            sys.modules, {
                "torch": self.torch,
                "ovrtx": self.ovrtx,
                "isaaclab_ov.renderers": SimpleNamespace(ovrtx_renderer=self.adapter),
            })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_each_rank_pins_native_initialization(self):
        # LOCAL_RANK is already resolved into sim.device by run.py. Do not remap
        # it through CUDA_VISIBLE_DEVICES or use the multi-node global rank.
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7", "RANK": "11"}):
            for index in range(4):
                with self.subTest(index=index):
                    self.torch.device.return_value = SimpleNamespace(type="cuda", index=index)
                    _configure_ovrtx_device(f"cuda:{index}")
                    config = self.adapter.RendererConfig(log_file_path="renderer.log",
                                                         keep_system_alive=True)
                    self.assertEqual(config.active_cuda_gpus, str(index))
                    self.assertEqual(config.log_file_path, "renderer.log")
                    self.assertTrue(config.keep_system_alive)
        self.torch.cuda.current_device.assert_not_called()
        self.assertIs(self.ovrtx.RendererConfig, self.native_config)

    def test_bare_cuda_uses_current_device(self):
        self.torch.device.return_value = SimpleNamespace(type="cuda", index=None)
        _configure_ovrtx_device("cuda")
        self.assertEqual(self.adapter.RendererConfig().active_cuda_gpus, "3")

    def test_cpu_rejected_before_changing_adapter(self):
        self.torch.device.return_value = SimpleNamespace(type="cpu", index=None)
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            _configure_ovrtx_device("cpu")
        self.assertIs(self.adapter.RendererConfig, self.native_config)
        self.native_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()

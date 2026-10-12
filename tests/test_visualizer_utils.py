# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualizer selection must retain its meaning across Isaac Lab launcher versions."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from compass.utils.visualizer_utils import (
    configure_camera_marker_updates,
    sync_visualizer_cli_settings,
)


class VisualizerSettingsTests(unittest.TestCase):
    """Check legacy and current launcher selection semantics without starting Kit."""

    def test_kitless_selection_and_visibility(self):
        values = {}
        settings = SimpleNamespace(set=values.__setitem__)
        modules = {
            "isaaclab.app": SimpleNamespace(AppLauncher=SimpleNamespace()),
            "isaaclab.app.settings_manager": SimpleNamespace(get_settings_manager=lambda: settings),
        }
        with patch.dict(sys.modules, modules):
            # Include repeated calls so stale settings cannot leak between selections.
            for selection, limit, expected in [
                (["newton_gl", "rerun"], 2, ("newton_gl rerun", True, False, 2)),
                ([], None, ("", True, True, -1)),
                (None, None, ("", False, False, -1)),
            ]:
                with self.subTest(selection=selection):
                    sync_visualizer_cli_settings(
                        SimpleNamespace(visualizer=selection, max_visible_envs=limit))
                    actual = tuple(values["/isaaclab/visualizer/" + key]
                                   for key in ("types", "explicit", "disable_all",
                                               "max_visible_envs"))
                    self.assertEqual(actual, expected)
            with self.assertRaises(ValueError):
                sync_visualizer_cli_settings(SimpleNamespace(visualizer=None, max_visible_envs=-1))

    def test_legacy_launcher_keeps_explicit_disable_flag(self):
        legacy_sync = Mock()
        launcher = SimpleNamespace(sync_visualizer_cli_settings_to_carb=legacy_sync)
        with patch.dict(sys.modules, {"isaaclab.app": SimpleNamespace(AppLauncher=launcher)}):
            for explicit in (False, True):
                args = SimpleNamespace(visualizer=[], visualizer_explicit=explicit)
                sync_visualizer_cli_settings(args)
                legacy_sync.assert_called_with({
                    **vars(args),
                    "visualizer_disable_all": explicit,
                })


class CameraMarkerUpdatesTests(unittest.TestCase):
    """Camera-only frames must use the current goal, including after reset."""

    def test_camera_capture_sees_updated_goal_after_each_reset(self):
        goal = [2.0, -3.0, 0.0]
        marker = [0.0, 0.0, 0.0]
        frames = []
        callbacks = {}
        registry = SimpleNamespace(
            dispatch_callbacks=lambda event: marker.__setitem__(slice(None), goal))
        sim = SimpleNamespace(
            get_setting=lambda key: True,
            visualizers=[],
            vis_marker_registry=registry,
            add_render_callback=lambda name, fn, order: callbacks.__setitem__(name, (order, fn)),
        )
        # Match the runtime camera-pump order, registering it before COMPASS.
        sim.add_render_callback("physx_headless_video_pump",
                                lambda event: frames.append(tuple(marker)), order=-10)
        configure_camera_marker_updates(sim)
        configure_camera_marker_updates(sim)
        for position in [(2.0, -3.0, 0.0), (-4.0, 1.5, 0.0)]:
            goal[:] = position
            for _, callback in sorted(callbacks.values(), key=lambda item: item[0]):
                callback(None)
        self.assertEqual(frames, [(2.0, -3.0, 0.0), (-4.0, 1.5, 0.0)])
        self.assertEqual(len(callbacks), 2)

    def test_existing_visualizer_dispatch_is_not_duplicated(self):
        for markers, plots, plots_enabled, expected_calls in [
            (True, False, False, 0),
            (False, True, True, 0),
            (False, True, False, 1),
            (False, False, False, 1),
        ]:
            with self.subTest(markers=markers, plots=plots, plots_enabled=plots_enabled):
                viz = SimpleNamespace(
                    supports_markers=lambda: markers,
                    supports_live_plots=lambda: plots,
                    cfg=SimpleNamespace(enable_live_plots=plots_enabled),
                )
                sim = SimpleNamespace(get_setting=lambda key: True,
                                      visualizers=[viz],
                                      vis_marker_registry=Mock(),
                                      add_render_callback=Mock())
                configure_camera_marker_updates(sim)
                callback = sim.add_render_callback.call_args.args[1]
                callback(None)
                self.assertEqual(sim.vis_marker_registry.dispatch_callbacks.call_count,
                                 expected_calls)

    def test_other_renderers_do_not_install_kit_marker_updates(self):
        sim = SimpleNamespace(get_setting=lambda key: False, add_render_callback=Mock())
        configure_camera_marker_updates(sim)
        sim.add_render_callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()

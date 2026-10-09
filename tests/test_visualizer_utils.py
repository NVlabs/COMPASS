# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualizer selection must retain its meaning across Isaac Lab launcher versions."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from compass.utils.visualizer_utils import sync_visualizer_cli_settings


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


if __name__ == "__main__":
    unittest.main()

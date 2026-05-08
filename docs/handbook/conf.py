# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the COMPASS documentation site."""

from __future__ import annotations

import os
import sys

# Make the repo root importable so future autodoc / autosummary can find compass.*.
ROOT = os.path.abspath("../..")
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

project = "COMPASS"
copyright = "2025-2026, NVIDIA"
author = "NVIDIA"

# Bump alongside CHANGELOG.md / pyproject.toml when we cut a release.
COMPASS_VERSION = "2.0.0"
version = COMPASS_VERSION
release = COMPASS_VERSION

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "venv"]

# ---------------------------------------------------------------------------
# MyST (markdown) configuration
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",    # ::: directive blocks
    "deflist",    # definition lists
    "fieldlist",    # field lists
    "tasklist",    # GitHub-style task lists
    "attrs_inline",    # {.class} inline attribute syntax
]
myst_fence_as_directive = {"mermaid"}
myst_heading_anchors = 3

# The handbook is self-contained — every page owns its content directly.
# We deliberately do NOT suppress `myst.xref_missing` or `image.not_readable`
# so `-W` catches broken intra-handbook links and missing images. If a future
# {include} of an external README needs those suppressions back, scope them
# to that page rather than globally.

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# ---------------------------------------------------------------------------
# Theme: nvidia-sphinx-theme
# https://github.com/NVIDIA/nvidia-sphinx-theme
# ---------------------------------------------------------------------------

html_theme = "nvidia_sphinx_theme"
html_title = f"COMPASS Handbook"
html_show_sphinx = False
html_theme_options = {
    "github_url": "https://github.com/NVlabs/COMPASS",
    "copyright_override": {
        "start": 2025
    },
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# ---------------------------------------------------------------------------
# Mermaid (sequence/flow diagrams in markdown)
# ---------------------------------------------------------------------------

mermaid_version = "11"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

todo_include_todos = False

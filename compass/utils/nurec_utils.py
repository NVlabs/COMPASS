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

import os

# NuRec Asset Selection
PARTICLE_SPG_RUNTIME_USD_FILE = "particle_spg-runtime.usdz"

__all__ = ["PARTICLE_SPG_RUNTIME_USD_FILE", "uses_nurec_spg_runtime"]


# Public API
def uses_nurec_spg_runtime(args):
    """Return whether the CLI selection should enable NuRec SPG runtime."""
    if getattr(args, "spg_runtime", False):
        return True

    if getattr(args, "nurec_scene", None) is None:
        return False

    usd_file = getattr(args, "nurec_usd_file", "") or ""
    return os.path.basename(usd_file) == PARTICLE_SPG_RUNTIME_USD_FILE

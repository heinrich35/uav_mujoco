# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Direct workflow environments.
"""

import gymnasium as gym

# Explicitly import g1_locomotion to ensure it's registered
# This is needed because automatic discovery may fail in some cases
try:
    from . import g1_locomotion  # noqa: F401
except Exception as e:  # noqa: BLE001
    # If import fails, log the error but continue
    # The import might succeed later when dependencies are available
    import warnings
    warnings.warn(f"Failed to import g1_locomotion: {e}", ImportWarning)

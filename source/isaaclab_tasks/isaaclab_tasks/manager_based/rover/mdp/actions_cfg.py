# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Action term configuration for differential drive."""

from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Two-axis differential drive: [forward, yaw_rate] in [-1,1] before scaling.

    Set ``class_type`` to :class:`DifferentialDriveAction` in the environment config.
    """

    # Meters per second (forward) and rad/s body yaw rate; mapped to wheel ω ≈ v/r.
    # URDF (`rover_isaacsim.urdf`): wheel radius 0.08 m, |y| spacing 0.24 m → track 0.48 m;
    # joint velocity limit 30 rad/s caps straight-line speed (~2.4 m/s) and turn+forward combos.
    max_forward_vel: float = 2.0
    max_turn_diff_vel: float = 4.0
    max_target_slew_rad_s2: float = 200.0
    wheel_radius_m: float = 0.08
    track_width_m: float = 0.48

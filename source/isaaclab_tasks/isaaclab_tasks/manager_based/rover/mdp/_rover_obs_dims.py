# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Observation slice sizes (concatenated policy obs must match prior checkpoint layout: 113)."""

# Visual / pseudo-vision stack (total 84; must stay aligned with policy concat + proprio = 113).
# Packed tracks use columns [0::4],[1::4],[2::4],[3::4]; widths must be multiples of 4 or strides disagree.
OBS_VIS_LANDMARKS = 32
OBS_VIS_OBJECT_TRACKS = 24
OBS_VIS_LANDMARK_COUNTS = 4
OBS_VIS_CONFIDENCE = 4
OBS_VIS_BEARING_DIST = 20

# Proprio / dynamics remainder (total 29) -> 84 + 29 = 113
OBS_IMU_YAW = 1
OBS_IMU_ACCEL = 3
OBS_OPTICAL_FW = 1
OBS_OPTICAL_YAW = 1
OBS_WHEEL_VEL = 4
OBS_WHEEL_ODO = 4
OBS_BASE_ANG_YAW = 1
OBS_BASE_LIN_CYL = 3
OBS_HEIGHT = 1
OBS_TILT = 1
OBS_THETA_SPAWN = 1
OBS_CUM_PROGRESS = 1
OBS_RADIAL_NORM = 1
OBS_HEADING_TAN = 2
OBS_LAST_ACTION = 2
OBS_LAST_CMD = 2

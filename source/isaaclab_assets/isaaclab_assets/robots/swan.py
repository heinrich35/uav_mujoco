# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Swan water robot (collision sphere 0.4 m)."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "isaaclab.sh").is_file():
            return parent
    raise RuntimeError("Could not locate Isaac Lab repo root (isaaclab.sh).")


SWAN_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Swan",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_repo_root() / "assets" / "pond" / "glb" / "usd" / "swan" / "swan_articulation.usd"),
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
"""Swan: single-link floating robot, 8 kg, collision sphere 0.4 m.

Actuation via external fluid-medium forces (buoyancy, drag, paddle force/torque).
"""

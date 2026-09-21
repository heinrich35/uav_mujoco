# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pond task — fluid dynamics environment with floating gosling robot."""

from __future__ import annotations

import gymnasium as gym

#
# Register the Pond environment
#

gym.register(
    id="Isaac-Pond-v0",
    entry_point=f"{__name__}.pond_rl_env:PondRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pond_env_cfg:PondEnvCfg",
    },
)

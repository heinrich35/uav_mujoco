# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rover tasks: disk-world obstacle avoidance and flat-world goal navigation (custom; not upstream Isaac Lab)."""

from __future__ import annotations

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Rover-DiskWorld-ObstacleAvoidance-v0",
    entry_point=f"{__name__}.rover_disk_world_obstacle_rl_env:RoverDiskWorldObstacleRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_disk_world_obstacle_env_cfg:RoverDiskWorldObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverDiskWorldObstaclePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-DiskWorld-ObstacleAvoidance-Play-v0",
    entry_point=f"{__name__}.rover_disk_world_obstacle_rl_env:RoverDiskWorldObstacleRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_disk_world_obstacle_env_cfg:RoverDiskWorldObstacleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverDiskWorldObstaclePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-FlatWorld-GoalNav-v0",
    entry_point=f"{__name__}.rover_flat_goal_rl_env:RoverFlatWorldGoalNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_flat_goal_env_cfg:RoverFlatWorldGoalNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverFlatWorldGoalNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-FlatWorld-GoalNav-Play-v0",
    entry_point=f"{__name__}.rover_flat_goal_rl_env:RoverFlatWorldGoalNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_flat_goal_env_cfg:RoverFlatWorldGoalNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverFlatWorldGoalNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-FlatWorld-GoalNav-Continuous-Play-v0",
    entry_point=f"{__name__}.rover_flat_goal_rl_env:RoverFlatWorldGoalNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_flat_goal_env_cfg:RoverFlatWorldGoalNavEnvCfg_CONTINUOUS_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverFlatWorldGoalNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-OAFlat-v0",
    entry_point=f"{__name__}.rover_oa_flat_rl_env:RoverOAFlatNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_oa_flat_env_cfg:RoverOAFlatNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverOAFlatNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-OAFlat-Play-v0",
    entry_point=f"{__name__}.rover_oa_flat_rl_env:RoverOAFlatNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_oa_flat_env_cfg:RoverOAFlatNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverOAFlatNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-OAFlat-Continuous-Play-v0",
    entry_point=f"{__name__}.rover_oa_flat_rl_env:RoverOAFlatNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_oa_flat_env_cfg:RoverOAFlatNavEnvCfg_CONTINUOUS_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverOAFlatNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-OAFlat-Physparam-v0",
    entry_point=f"{__name__}.rover_oa_flat_physparam_rl_env:RoverOAFlatPhysparamNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_oa_flat_physparam_env_cfg:RoverOAFlatPhysparamNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverOAFlatPhysparamNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-OAFlat-Physparam-Play-v0",
    entry_point=f"{__name__}.rover_oa_flat_physparam_rl_env:RoverOAFlatPhysparamNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_oa_flat_physparam_env_cfg:RoverOAFlatPhysparamNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverOAFlatPhysparamNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-OAFlat-Physparam-Continuous-Play-v0",
    entry_point=f"{__name__}.rover_oa_flat_physparam_rl_env:RoverOAFlatPhysparamNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_oa_flat_physparam_env_cfg:RoverOAFlatPhysparamNavEnvCfg_CONTINUOUS_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverOAFlatPhysparamNavPPORunnerCfg",
    },
)

# --- Rover Localization task ---
gym.register(
    id="Isaac-Rover-Localization-v0",
    entry_point=f"{__name__}.rover_localization_rl_env:RoverLocalizationRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_localization_env_cfg:RoverLocalizationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverLocalizationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-Localization-Play-v0",
    entry_point=f"{__name__}.rover_localization_rl_env:RoverLocalizationRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_localization_env_cfg:RoverLocalizationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverLocalizationPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rover-Localization-Continuous-Play-v0",
    entry_point=f"{__name__}.rover_localization_rl_env:RoverLocalizationRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rover_localization_env_cfg:RoverLocalizationEnvCfg_CONTINUOUS_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverLocalizationPPORunnerCfg",
    },
)

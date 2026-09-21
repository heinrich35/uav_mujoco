# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""OA-flat navigation like :class:`RoverOAFlatNavEnvCfg` plus episode physparam latch + success-triggered relocate."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.rover.mdp.flat_goal_rewards as fg_r
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_events as oa_ev
from isaaclab_tasks.manager_based.rover.mdp.actions_physparam_latch_drive import PhysparamLatchDifferentialDriveActionCfg
from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import (
    RewardsCfg,
    RoverOAFlatNavEnvCfg,
    RoverOAFlatSceneCfg,
    TerminationsCfg,
    _oa_flat_boost_rtx_for_gui_play,
)


@configclass
class ActionsCfgPhysparam:
    drive = PhysparamLatchDifferentialDriveActionCfg()


@configclass
class EventCfgPhysparam:
    reset_oa_flat = EventTerm(
        func=oa_ev.reset_oa_flat_episode,
        mode="reset",
        params={
            "spawn_half_size_m": 15.0,
            "target_half_size_m": 23.0,
            "target_exclusion_half_size_m": 15.5,
            "target_min_spawn_dist_m": 5.0,
            "spawn_z_m": 0.2,
            "spawn_target_marker": True,
            "marker_asset_name": "goal_marker",
            "oa_flat_box_count": 70,
            "oa_flat_box_arena_half_m": 25.0,
            "oa_flat_box_grid_step_m": 4.0,
            "oa_flat_box_min_spawn_clear_m": 4.0,
            "oa_flat_box_min_pair_clear_m": 1.5,
            "oa_flat_box_half_height_m": 0.5,
            "oa_flat_target_min_blue_box_clearance_m": 3.0,
        },
    )
    relocate_goal_after_success = EventTerm(
        func=oa_ev.oa_flat_relocate_goal_if_pending,
        mode="interval",
        interval_range_s=(0.1, 0.18),
        is_global_time=True,
        params={
            "target_half_size_m": 23.0,
            "target_exclusion_half_size_m": 15.5,
            "target_min_spawn_dist_m": 5.0,
            "spawn_target_marker": True,
            "marker_asset_name": "goal_marker",
            "oa_flat_box_count": 70,
            "oa_flat_target_min_blue_box_clearance_m": 3.0,
            "max_target_samples": 64,
        },
    )


@configclass
class RewardsCfgPhysparam(RewardsCfg):
    close_to_target_bonus = RewTerm(
        func=fg_r.close_to_target_bonus,
        weight=1.0,
        params={"threshold_m": 5.0, "bonus_max_dist_m": 2.75},
    )
    goal_reached_bonus = RewTerm(
        func=fg_r.goal_reached_bonus,
        weight=70.0,
        params={"threshold_m": 1.5, "set_relocate_pending": True},
    )
    target_approach_time_decay = RewTerm(
        func=fg_r.target_approach_time_decay_shaping,
        weight=2.5,
        params={"tau_s": 18.0, "goal_threshold_m": 1.5},
    )


@configclass
class RoverOAFlatPhysparamNavEnvCfg(RoverOAFlatNavEnvCfg):
    """12-D actions (10 latched physparam + 2 drive); relocate after :term:`goal_reached_bonus` only."""

    scene: RoverOAFlatSceneCfg = RoverOAFlatSceneCfg(
        num_envs=2048, env_spacing=64.0, replicate_physics=True, clone_in_fabric=False
    )
    actions: ActionsCfgPhysparam = ActionsCfgPhysparam()
    events: EventCfgPhysparam = EventCfgPhysparam()
    rewards: RewardsCfgPhysparam = RewardsCfgPhysparam()
    terminations: TerminationsCfg = TerminationsCfg()

    # ``close_to_target_bonus`` max radius ≈ ``flat_goal_reached_threshold_m + margin`` (anti ring exploit).
    flat_close_bonus_goal_margin_m: float = 1.25
    flat_approach_time_decay_tau_s: float = 18.0
    flat_approach_time_decay_weight: float = 2.5

    def __post_init__(self) -> None:
        super().__post_init__()
        thr = float(self.flat_goal_reached_threshold_m)
        cap = thr + float(self.flat_close_bonus_goal_margin_m)
        self.rewards.close_to_target_bonus.params["bonus_max_dist_m"] = cap
        self.rewards.goal_reached_bonus.params["set_relocate_pending"] = True
        self.rewards.target_approach_time_decay.weight = float(self.flat_approach_time_decay_weight)
        self.rewards.target_approach_time_decay.params["tau_s"] = float(self.flat_approach_time_decay_tau_s)
        self.rewards.target_approach_time_decay.params["goal_threshold_m"] = thr


@configclass
class RoverOAFlatPhysparamNavEnvCfg_PLAY(RoverOAFlatPhysparamNavEnvCfg):
    """Single-env play preset."""

    oa_flat_camera_width: int = 384
    oa_flat_camera_height: int = 304

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        super().__post_init__()
        _oa_flat_boost_rtx_for_gui_play(self)


@configclass
class RoverOAFlatPhysparamNavEnvCfg_CONTINUOUS_PLAY(RoverOAFlatPhysparamNavEnvCfg):
    """Same arena / spawn / goals / blue-box layout as training; huge episode_length_s for open-ended play."""

    oa_flat_camera_width: int = 384
    oa_flat_camera_height: int = 304

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        self.target_half_size_m = 20.0
        self.target_exclusion_half_size_m = 15.5
        super().__post_init__()
        self.episode_length_s = 1.0e9
        _oa_flat_boost_rtx_for_gui_play(self)

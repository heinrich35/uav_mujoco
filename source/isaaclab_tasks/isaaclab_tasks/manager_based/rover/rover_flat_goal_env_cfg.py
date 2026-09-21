# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Flat ground plane goal navigation for the differential-drive rover (XY plane, +Z up)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as env_mdp
from isaaclab.envs.mdp import rewards as mdp_r
from isaaclab.envs.mdp import terminations as mdp_term
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.rover.mdp.flat_goal_events as fg_ev
import isaaclab_tasks.manager_based.rover.mdp.flat_goal_observations as fg_obs
import isaaclab_tasks.manager_based.rover.mdp.flat_goal_rewards as fg_r
import isaaclab_tasks.manager_based.rover.mdp.flat_goal_terminations as fg_term
from isaaclab_tasks.manager_based.rover.mdp.actions import DifferentialDriveAction
from isaaclab_tasks.manager_based.rover.mdp.actions_cfg import DifferentialDriveActionCfg
import isaaclab_tasks.manager_based.rover.mdp.observations as rover_obs
from isaaclab_tasks.manager_based.rover.rover_disk_assets import ensure_rover_usd, make_rover_cfg


def _rover_flat() -> ArticulationCfg:
    return make_rover_cfg(usd_path=ensure_rover_usd(), prim_path="{ENV_REGEX_NS}/Rover").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class RoverFlatWorldGoalSceneCfg(InteractiveSceneCfg):
    """Infinite plane + optional kinematic walls + yellow goal marker."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
    )
    rover = _rover_flat()
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.9, 0.55),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.92, 0.15)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
    )
    wall_px = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallPx",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 56.0, 2.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(26.0, 0.0, 1.4)),
    )
    wall_nx = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallNx",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 56.0, 2.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-26.0, 0.0, 1.4)),
    )
    wall_py = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallPy",
        spawn=sim_utils.CuboidCfg(
            size=(56.0, 1.2, 2.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 26.0, 1.4)),
    )
    wall_ny = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallNy",
        spawn=sim_utils.CuboidCfg(
            size=(56.0, 1.2, 2.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -26.0, 1.4)),
    )


@configclass
class ActionsCfg:
    drive = DifferentialDriveActionCfg(
        class_type=DifferentialDriveAction,
        asset_name="rover",
        max_forward_vel=2.25,
        max_turn_diff_vel=9.0,
        max_target_slew_rad_s2=720.0,
        wheel_radius_m=0.08,
        track_width_m=0.48,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        goal_vec_xy = ObsTerm(func=fg_obs.goal_vector_xy_scaled, params={"scale_m": 50.0})
        dist_goal = ObsTerm(func=fg_obs.distance_to_goal_normalized, params={"scale_m": 50.0})
        projected_gravity = ObsTerm(func=env_mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("rover")})
        base_lin_vel_xy = ObsTerm(func=fg_obs.base_lin_vel_xy)
        base_ang_vel_z = ObsTerm(func=fg_obs.base_ang_vel_z)
        wheel_velocities_normalized = ObsTerm(func=rover_obs.wheel_velocities_normalized)
        last_action = ObsTerm(func=rover_obs.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_flat_goal = EventTerm(
        func=fg_ev.reset_flat_goal_episode,
        mode="reset",
        params={
            "spawn_half_size_m": 15.0,
            "target_half_size_m": 23.0,
            "target_exclusion_half_size_m": 15.5,
            "target_min_spawn_dist_m": 5.0,
            "spawn_z_m": 0.2,
            "spawn_target_marker": True,
            "marker_asset_name": "goal_marker",
        },
    )
    # When within goal_threshold_m, sample a new target (episode continues; see terminations.goal_reached).
    relocate_goal_interval = EventTerm(
        func=fg_ev.continuous_relocate_goal_if_close,
        mode="interval",
        interval_range_s=(0.1, 0.2),
        params={
            "goal_threshold_m": 3.0,
            "target_half_size_m": 23.0,
            "target_exclusion_half_size_m": 15.5,
            "target_min_spawn_dist_m": 5.0,
            "spawn_half_size_m": 15.0,
            "spawn_target_marker": True,
            "marker_asset_name": "goal_marker",
        },
    )


@configclass
class RewardsCfg:
    distance_to_target = RewTerm(func=fg_r.distance_to_target_reward, weight=1.0, params={"scale_m": 50.0})
    distance_progress = RewTerm(func=fg_r.distance_progress_reward, weight=7.0)
    distance_increase_penalty = RewTerm(func=fg_r.distance_increase_penalty, weight=2.0)
    heading_to_target = RewTerm(func=fg_r.heading_to_target_reward, weight=0.2)
    heading_goal_closure_speed = RewTerm(
        func=fg_r.heading_goal_closure_speed_reward,
        weight=0.9,
        params={"cap_ms": 3.0},
    )
    yaw_without_goal_closure = RewTerm(
        func=fg_r.yaw_rate_without_goal_closure_penalty,
        weight=-0.14,
        params={
            "min_dist_m": 3.5,
            "closure_speed_threshold_m_s": 0.12,
            "yaw_thresh_rad_s": 0.55,
            "yaw_ref_rad_s": 2.25,
        },
    )
    forward_velocity_to_target = RewTerm(
        func=fg_r.forward_velocity_to_target_reward, weight=1.0, params={"cap_ms": 3.0}
    )
    close_to_target_bonus = RewTerm(
        func=fg_r.close_to_target_bonus, weight=1.0, params={"threshold_m": 5.0}
    )
    goal_reached_bonus = RewTerm(
        func=fg_r.goal_reached_bonus, weight=50.0, params={"threshold_m": 1.5}
    )
    # func returns a positive penalty magnitude; weight must be negative (same pattern as tilt / boundary).
    action_rate_penalty = RewTerm(func=fg_r.action_rate_penalty_flat, weight=-0.05)
    tilt_penalty = RewTerm(func=fg_r.tilt_penalty_flat, weight=-0.2)
    boundary_penalty = RewTerm(
        func=fg_r.boundary_penalty_flat, weight=-0.5, params={"plane_half_size_m": 25.0, "margin_m": 3.0}
    )
    four_wheels_contact = RewTerm(func=fg_r.four_wheels_ground_contact_reward, weight=0.2, params={"threshold_n": 1.0})
    front_wheels_contact = RewTerm(func=fg_r.front_wheels_ground_contact_reward, weight=0.5, params={"threshold_n": 1.0})
    rear_wheels_contact = RewTerm(func=fg_r.rear_wheels_ground_contact_reward, weight=0.5, params={"threshold_n": 1.0})
    not_four_wheels_contact_penalty = RewTerm(
        func=fg_r.not_four_wheels_ground_contact_penalty, weight=-0.5, params={"threshold_n": 1.0}
    )
    alive_bonus = RewTerm(func=mdp_r.is_alive, weight=0.1)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_term.time_out, time_out=True)
    goal_reached = DoneTerm(func=fg_term.flat_goal_reached, params={"threshold_m": 1.5})
    upside_down = DoneTerm(func=fg_term.upside_down_flat, params={"min_up_dot": 0.2})
    wheelie = DoneTerm(func=fg_term.wheelie_pitch_termination, params={"min_pitch_sin": 0.2})


@configclass
class RoverFlatWorldGoalNavEnvCfg(ManagerBasedRLEnvCfg):
    """Train a rover to reach random planar goals on flat ground.

    Reaching the goal does **not** end the episode: the target is resampled in-place (interval event).
    Episodes end on time-out, upside-down, or wheelie termination.
    """

    scene: RoverFlatWorldGoalSceneCfg = RoverFlatWorldGoalSceneCfg(
        num_envs=2048, env_spacing=64.0, replicate_physics=True, clone_in_fabric=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # --- train.py / shell mirrors (scalar fields; __post_init__ copies into RewTerm weights & params) ---
    flat_distance_to_target_weight: float = 1.0
    flat_distance_to_target_scale_m: float = 50.0
    flat_distance_progress_weight: float = 7.0
    flat_distance_increase_penalty_weight: float = 2.0
    flat_heading_to_target_weight: float = 0.2
    flat_heading_goal_closure_weight: float = 0.9
    flat_yaw_no_closure_penalty_weight: float = 0.14
    flat_forward_velocity_to_target_weight: float = 1.0
    flat_forward_velocity_to_target_cap_ms: float = 3.0
    flat_close_to_target_bonus_weight: float = 1.0
    flat_close_to_target_threshold_m: float = 2.5
    flat_goal_reached_bonus_weight: float = 50.0
    flat_goal_reached_threshold_m: float = 1.5
    # When planar dist to goal < this (m), interval event resamples goal (wider than success threshold).
    flat_goal_relocate_threshold_m: float = 3.50
    flat_action_rate_weight: float = 0.05
    flat_tilt_weight: float = 0.2
    flat_boundary_weight: float = 0.5
    flat_boundary_margin_m: float = 3.0
    flat_four_wheels_ground_contact_weight: float = 0.2
    flat_front_wheels_ground_contact_weight: float = 0.5
    flat_rear_wheels_ground_contact_weight: float = 0.5
    flat_not_four_wheels_ground_contact_penalty_weight: float = 0.5
    flat_wheel_ground_contact_threshold_n: float = 1.0
    flat_alive_weight: float = 0.1
    flat_wheel_effort_limit_sim: float = 2500.0
    flat_wheel_target_slew_rad_s2: float = 720.0
    flat_drive_max_forward_vel_m_s: float = 2.25
    flat_drive_max_turn_yaw_rate_rad_s: float = 9.0
    flat_drive_track_width_m: float = 0.48
    flat_actuator_wheel_damping: float = 150.0
    flat_wheel_velocity_limit_sim: float = 165.0
    plane_half_size_m: float = 25.0
    spawn_half_size_m: float = 15.0
    target_half_size_m: float = 20.0
    target_exclusion_half_size_m: float = 15.5
    target_min_spawn_dist_m: float = 5.0
    spawn_z_m: float = 0.2
    spawn_target_marker: bool = True
    flat_upside_down_min_dot: float = 0.2
    flat_wheelie_min_pitch_sin: float = 0.2
    flat_goal_train_stage: int = 0
    flat_goal_train_stage_note: str = ""

    def __post_init__(self) -> None:
        H = float(self.plane_half_size_m)
        t = 1.2
        hz = 2.8
        L = 2.0 * H + 4.0 * t
        self.scene.wall_px.init_state.pos = (H + t * 0.5, 0.0, hz * 0.5)
        self.scene.wall_nx.init_state.pos = (-(H + t * 0.5), 0.0, hz * 0.5)
        self.scene.wall_py.init_state.pos = (0.0, H + t * 0.5, hz * 0.5)
        self.scene.wall_ny.init_state.pos = (0.0, -(H + t * 0.5), hz * 0.5)
        self.scene.wall_px.spawn.size = (t, L, hz)
        self.scene.wall_nx.spawn.size = (t, L, hz)
        self.scene.wall_py.spawn.size = (L, t, hz)
        self.scene.wall_ny.spawn.size = (L, t, hz)

        spacing = max(70.0, 2.0 * H + 15.0)
        self.scene.env_spacing = spacing

        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.viewer.eye = (H * 1.4, H * 1.1, 18.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.scene.rover.init_state.pos = (0.0, 0.0, float(self.spawn_z_m))

        wcfg = self.scene.rover.actuators["wheels"]
        wcfg.effort_limit_sim = float(self.flat_wheel_effort_limit_sim)
        wcfg.damping = float(self.flat_actuator_wheel_damping)
        wcfg.velocity_limit_sim = float(self.flat_wheel_velocity_limit_sim)
        self.actions.drive.max_target_slew_rad_s2 = float(self.flat_wheel_target_slew_rad_s2)
        self.actions.drive.max_forward_vel = float(self.flat_drive_max_forward_vel_m_s)
        self.actions.drive.max_turn_diff_vel = float(self.flat_drive_max_turn_yaw_rate_rad_s)
        self.actions.drive.track_width_m = float(self.flat_drive_track_width_m)

        self.events.reset_flat_goal.params["spawn_half_size_m"] = float(self.spawn_half_size_m)
        self.events.reset_flat_goal.params["target_half_size_m"] = float(self.target_half_size_m)
        self.events.reset_flat_goal.params["target_exclusion_half_size_m"] = float(self.target_exclusion_half_size_m)
        self.events.reset_flat_goal.params["target_min_spawn_dist_m"] = float(self.target_min_spawn_dist_m)
        self.events.reset_flat_goal.params["spawn_z_m"] = float(self.spawn_z_m)
        self.events.reset_flat_goal.params["spawn_target_marker"] = bool(self.spawn_target_marker)

        thr_g = float(self.flat_goal_reached_threshold_m)
        rg = self.events.relocate_goal_interval.params
        rg["goal_threshold_m"] = float(self.flat_goal_relocate_threshold_m)
        rg["target_half_size_m"] = float(self.target_half_size_m)
        rg["target_exclusion_half_size_m"] = float(self.target_exclusion_half_size_m)
        rg["target_min_spawn_dist_m"] = float(self.target_min_spawn_dist_m)
        rg["spawn_half_size_m"] = float(self.spawn_half_size_m)
        rg["spawn_target_marker"] = bool(self.spawn_target_marker)

        # Success = relocate target, not episode reset (bonus still uses threshold_m above).
        self.terminations.goal_reached = DoneTerm(func=fg_term.never_terminate)

        sc = float(self.flat_distance_to_target_scale_m)
        self.observations.policy.goal_vec_xy.params["scale_m"] = sc
        self.observations.policy.dist_goal.params["scale_m"] = sc
        self.rewards.distance_to_target.params["scale_m"] = sc
        self.rewards.distance_to_target.weight = float(self.flat_distance_to_target_weight)
        self.rewards.distance_progress.weight = float(self.flat_distance_progress_weight)
        self.rewards.distance_increase_penalty.weight = float(self.flat_distance_increase_penalty_weight)
        self.rewards.heading_to_target.weight = float(self.flat_heading_to_target_weight)
        self.rewards.heading_goal_closure_speed.weight = float(self.flat_heading_goal_closure_weight)
        self.rewards.heading_goal_closure_speed.params["cap_ms"] = float(self.flat_forward_velocity_to_target_cap_ms)
        self.rewards.yaw_without_goal_closure.weight = -abs(float(self.flat_yaw_no_closure_penalty_weight))
        self.rewards.forward_velocity_to_target.weight = float(self.flat_forward_velocity_to_target_weight)
        self.rewards.forward_velocity_to_target.params["cap_ms"] = float(self.flat_forward_velocity_to_target_cap_ms)
        self.rewards.close_to_target_bonus.weight = float(self.flat_close_to_target_bonus_weight)
        self.rewards.close_to_target_bonus.params["threshold_m"] = float(self.flat_close_to_target_threshold_m)
        self.rewards.goal_reached_bonus.weight = float(self.flat_goal_reached_bonus_weight)
        self.rewards.goal_reached_bonus.params["threshold_m"] = float(self.flat_goal_reached_threshold_m)
        self.rewards.action_rate_penalty.weight = -abs(float(self.flat_action_rate_weight))
        self.rewards.tilt_penalty.weight = -abs(float(self.flat_tilt_weight))
        self.rewards.boundary_penalty.weight = -abs(float(self.flat_boundary_weight))
        self.rewards.boundary_penalty.params["plane_half_size_m"] = float(self.plane_half_size_m)
        self.rewards.boundary_penalty.params["margin_m"] = float(self.flat_boundary_margin_m)
        self.rewards.four_wheels_contact.weight = float(self.flat_four_wheels_ground_contact_weight)
        self.rewards.front_wheels_contact.weight = float(self.flat_front_wheels_ground_contact_weight)
        self.rewards.rear_wheels_contact.weight = float(self.flat_rear_wheels_ground_contact_weight)
        self.rewards.not_four_wheels_contact_penalty.weight = -abs(float(self.flat_not_four_wheels_ground_contact_penalty_weight))
        thr_c = float(self.flat_wheel_ground_contact_threshold_n)
        for name in (
            "four_wheels_contact",
            "front_wheels_contact",
            "rear_wheels_contact",
            "not_four_wheels_contact_penalty",
        ):
            getattr(self.rewards, name).params["threshold_n"] = thr_c
        self.rewards.alive_bonus.weight = float(self.flat_alive_weight)
        self.terminations.upside_down.params["min_up_dot"] = float(self.flat_upside_down_min_dot)
        self.terminations.wheelie.params["min_pitch_sin"] = float(self.flat_wheelie_min_pitch_sin)


@configclass
class RoverFlatWorldGoalNavEnvCfg_PLAY(RoverFlatWorldGoalNavEnvCfg):
    """Single-env GUI play defaults."""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        super().__post_init__()


@configclass
class EventCfgContinuousPlay(EventCfg):
    """Same ``reset_flat_goal`` as base; faster relocate cadence so the marker moves soon after the rover is close."""

    relocate_goal_interval = EventTerm(
        func=fg_ev.continuous_relocate_goal_if_close,
        mode="interval",
        interval_range_s=(0.08, 0.14),
        params={
            "goal_threshold_m": 3.0,
            "target_half_size_m": 20.0,
            "target_exclusion_half_size_m": 15.5,
            "target_min_spawn_dist_m": 3.0,
            "spawn_half_size_m": 10.0,
            "spawn_target_marker": True,
            "marker_asset_name": "goal_marker",
        },
    )


@configclass
class RoverFlatWorldGoalNavEnvCfg_CONTINUOUS_PLAY(RoverFlatWorldGoalNavEnvCfg):
    """60×60 m arena; very long episode; goal marker relocates when close (no goal episode reset)."""

    events: EventCfgContinuousPlay = EventCfgContinuousPlay()

    def __post_init__(self) -> None:
        # 60 m × 60 m ground plane (half-extent 30 m); walls/scene follow ``plane_half_size_m`` in super().
        self.plane_half_size_m = 30.0
        self.spawn_half_size_m = 25.0
        self.target_half_size_m = 20.0
        self.target_exclusion_half_size_m = 15.5
        self.target_min_spawn_dist_m = 5.0
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        super().__post_init__()
        self.episode_length_s = 1.0e9

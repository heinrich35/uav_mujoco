# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Manager-based RL: rover drives on the disk world with pseudo-visual obstacle avoidance."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
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
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.rover.mdp.localization_observations as loc_obs
import isaaclab_tasks.manager_based.rover.mdp.observations as obs
import isaaclab_tasks.manager_based.rover.mdp.rewards as rrew
import isaaclab_tasks.manager_based.rover.mdp.rover_events as rev
import isaaclab_tasks.manager_based.rover.mdp.rover_obstacle_avoidance_rewards as oa
import isaaclab_tasks.manager_based.rover.mdp.rover_physparam_rewards as sm
import isaaclab_tasks.manager_based.rover.mdp.terminations as rterm

from isaaclab_tasks.manager_based.rover.mdp.actions import DifferentialDriveAction
from isaaclab_tasks.manager_based.rover.mdp.actions_cfg import DifferentialDriveActionCfg
from isaaclab_tasks.manager_based.rover.rover_disk_assets import (
    OA_DEFAULT_ROVER_SPAWN_LOCAL_XYZ,
    OA_DISK_CYLINDER_HALF_WIDTH_Y_M,
    build_oa_obstacle_rigid_objects_dict,
    default_goal_mesh_path,
    ensure_disk_world_usd,
    ensure_rover_usd,
    make_rover_cfg,
)


def _disk_usd() -> str:
    return ensure_disk_world_usd()


def _rover_usd() -> str:
    return ensure_rover_usd()


@configclass
class RoverDiskWorldObstacleSceneCfg(InteractiveSceneCfg):
    """Per-env disk arena (cylinder mesh, axis || world +Y) + rover + lights + obstacle USDs (no ground plane).

    The disk GLB/USD is spawned **per environment** under ``{ENV_REGEX_NS}/DiskWorldArena`` so replicated
    training matches ``scripts/rover/launch_rover_stage.py`` (global ``/World/DiskWorld`` alone would only
    exist once while env origins are on a grid). Blue boxes and goals are added in ``RoverDiskWorldObstacleEnvCfg.__post_init__``.
    """

    # Disk mesh per env (ramps + surface). Use ``AssetBaseCfg``: USD root is not a single rigid body with
    # ``RigidBodyAPI`` on ``DiskWorldArena`` (colliders live under the referenced mesh), so ``RigidObject`` fails.
    disk_arena = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DiskWorldArena",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_disk_usd(),
            copy_from_source=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        collision_group=-1,
    )
    rover = make_rover_cfg(usd_path=_rover_usd(), prim_path="{ENV_REGEX_NS}/Rover")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85)),
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
        visible_landmarks = ObsTerm(func=loc_obs.visible_landmarks)
        visual_object_tracks = ObsTerm(func=loc_obs.visual_object_tracks)
        visual_landmark_counts = ObsTerm(func=loc_obs.visual_landmark_counts)
        visual_confidence_stats = ObsTerm(func=loc_obs.visual_confidence_stats)
        visual_bearing_distribution = ObsTerm(func=loc_obs.visual_bearing_distribution)
        imu_yaw_rate_noisy = ObsTerm(func=loc_obs.imu_yaw_rate_noisy)
        imu_lin_accel_noisy = ObsTerm(func=loc_obs.imu_lin_accel_noisy)
        optical_flow_forward_speed = ObsTerm(func=loc_obs.optical_flow_forward_speed)
        optical_flow_yaw_rate = ObsTerm(func=loc_obs.optical_flow_yaw_rate)
        wheel_velocities_normalized = ObsTerm(func=obs.wheel_velocities_normalized)
        wheel_odometry_estimate = ObsTerm(func=loc_obs.wheel_odometry_estimate)
        base_ang_vel_yaw = ObsTerm(func=obs.base_ang_vel_yaw)
        base_lin_vel_cylindrical = ObsTerm(func=obs.base_lin_vel_cylindrical)
        height_offset = ObsTerm(func=obs.height_offset)
        body_tilt_quality = ObsTerm(func=obs.body_tilt_quality)
        theta_from_spawn = ObsTerm(func=obs.theta_from_spawn)
        cumulative_angular_progress = ObsTerm(func=obs.cumulative_angular_progress)
        radial_distance_normalized = ObsTerm(func=obs.radial_distance_normalized)
        heading_tangent_cos_sin = ObsTerm(func=obs.heading_tangent_cos_sin)
        last_action = ObsTerm(func=obs.last_action)
        last_drive_command = ObsTerm(func=loc_obs.last_drive_command)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene_to_default = EventTerm(func=env_mdp.reset_scene_to_default, mode="reset")
    randomize_obstacles = EventTerm(
        func=rev.randomize_disk_ring_obstacles,
        mode="reset",
        params={
            "goal_count": 30,
            "box_count": 80,
            "goal_radial_distance": 100.3,
            "box_radial_distance_m": 100.0,
            "min_object_spacing_m": 4.0,
            "cylinder_y_half_width_m": OA_DISK_CYLINDER_HALF_WIDTH_Y_M,
            "rover_spawn_exclusion_radius_m": 7.5,
            "exclude_spawn_local_xyz": OA_DEFAULT_ROVER_SPAWN_LOCAL_XYZ,
            "max_sample_attempts": 512,
        },
    )


@configclass
class RewardsCfg:
    forward_angular_progress = RewTerm(func=rrew.forward_angular_progress, weight=30.0)
    forward_speed_tracking_penalty = RewTerm(
        func=rrew.forward_speed_tracking,
        weight=10.0,
        params={"target_tangential_speed": 1.5},
    )
    blue_box_avoidance_penalty = RewTerm(
        func=oa.blue_box_avoidance_penalty,
        weight=2.0,
        params={"danger_radius_m": 5.0},
    )
    next_goal_pass_through_reward = RewTerm(func=oa.next_goal_pass_through_reward, weight=20.0)
    disk_tangent_heading_alignment = RewTerm(
        func=oa.disk_tangent_heading_alignment_reward,
        weight=1.0,
        params={"tolerance_rad": 1.1},
    )
    radial_z_alignment_reward = RewTerm(func=rrew.radial_z_alignment_reward, weight=1.0)
    radial_z_alignment_penalty = RewTerm(
        func=rrew.radial_z_alignment_penalty,
        weight=1.0,
        params={"max_abs_z_error_before_clip_m": 2.5},
    )
    near_fall_radial_penalty = RewTerm(
        func=oa.near_fall_radial_penalty,
        weight=1.0,
        params={"fall_radius_m": 99.5, "danger_band_m": 1.15},
    )
    visual_blue_box_ahead_penalty = RewTerm(
        func=oa.visual_blue_box_ahead_penalty,
        weight=1.0,
        params={"bearing_gate_rad": 0.4, "range_gate_m": 6.0},
    )
    visual_goal_ahead_alignment_reward = RewTerm(
        func=oa.visual_goal_ahead_alignment_reward,
        weight=1.0,
        params={"bearing_gate_rad": 0.4, "range_gate_m": 12.0},
    )
    next_goal_approach_shaping = RewTerm(
        func=oa.next_goal_approach_shaping,
        weight=1.0,
        params={"sigma_m": 5.0},
    )
    upside_down_termination_penalty = RewTerm(
        func=mdp_r.is_terminated_term,
        weight=-2500.0,
        params={"term_keys": "upside_down"},
    )
    fell_off_disk_termination_penalty = RewTerm(
        func=mdp_r.is_terminated_term,
        weight=-12000.0,
        params={"term_keys": "fell_off_disk"},
    )
    action_rate_penalty = RewTerm(func=rrew.action_rate_penalty, weight=0.04)
    motion_jerk_penalty = RewTerm(func=sm.motion_jerk_penalty, weight=0.01)
    heading_oscillation_penalty = RewTerm(func=sm.heading_oscillation_penalty, weight=0.05)
    alive_bonus = RewTerm(func=rrew.alive_bonus, weight=0.05)
    teacher_action_imitation = RewTerm(
        func=oa.teacher_action_imitation_reward,
        weight=0.0,
        params={"decay_env_steps": 20000, "match_sigma": 0.25},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_term.time_out, time_out=True)
    fell_off_disk = DoneTerm(
        func=rterm.fell_off_disk,
        params={"min_radius": 99.5},
    )
    upside_down = DoneTerm(
        func=rterm.upside_down_disk_world,
        params={"min_radial_dot": 0.0},
    )


@configclass
class RoverDiskWorldObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Rover obstacle avoidance on the cylindrical disk world."""

    scene: RoverDiskWorldObstacleSceneCfg = RoverDiskWorldObstacleSceneCfg(
        num_envs=1, env_spacing=300.0, replicate_physics=True, clone_in_fabric=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # CLI / train.py mirrors (Rover obstacle-avoidance block)
    oa_forward_progress_weight: float = 30.0
    oa_forward_speed_tracking_penalty_weight: float = 10.0
    oa_forward_speed_target_ms: float = 1.5
    oa_blue_box_penalty_weight: float = 2.0
    oa_blue_box_danger_radius_m: float = 5.0
    oa_goal_pass_through_weight: float = 20.0
    oa_heading_alignment_weight: float = 1.0
    oa_heading_alignment_tolerance_rad: float = 1.1
    oa_radial_z_alignment_reward_weight: float = 1.0
    oa_radial_z_alignment_penalty_weight: float = 1.0
    oa_radial_z_penalty_max_abs_m: float = 2.5
    oa_upside_down_penalty_weight: float = 2500.0
    oa_fell_off_disk_penalty_weight: float = 12000.0
    oa_near_fall_penalty_weight: float = 1.0
    oa_near_fall_danger_band_m: float = 1.15
    oa_visual_box_ahead_penalty_weight: float = 1.0
    oa_visual_box_ahead_bearing_gate_rad: float = 0.4
    oa_visual_box_ahead_range_gate_m: float = 6.0
    oa_visual_goal_ahead_reward_weight: float = 1.0
    oa_visual_goal_ahead_bearing_gate_rad: float = 0.4
    oa_visual_goal_ahead_range_gate_m: float = 12.0
    oa_goal_approach_shaping_weight: float = 1.0
    oa_goal_approach_sigma_m: float = 5.0
    oa_upside_down_min_radial_dot: float = 0.0
    oa_action_rate_penalty_weight: float = 0.04
    oa_motion_jerk_penalty_weight: float = 0.01
    oa_heading_oscillation_penalty_weight: float = 0.05
    oa_alive_bonus_weight: float = 0.05
    oa_teacher_imitation_weight: float = 0.0
    oa_teacher_imitation_decay_env_steps: int = 20000
    oa_teacher_imitation_match_sigma: float = 0.25
    oa_teacher_action_blend_weight: float = 0.0
    oa_teacher_action_blend_decay_env_steps: int = 12000
    oa_teacher_policy_enable: bool = False
    oa_teacher_policy_path: str = ""
    oa_fall_min_radius_m: float = 99.5
    drive_policy_path: str = ""

    goal_pass_threshold_m: float = 2.0
    goal_radial_distance: float = 100.3
    oa_box_radial_distance_m: float = 100.0
    oa_cylinder_y_half_width_m: float = OA_DISK_CYLINDER_HALF_WIDTH_Y_M
    oa_rover_spawn_exclusion_radius_m: float = 7.5
    oa_cylindrical_gravity_m_s2: float = 9.81
    goal_count: int = 30
    box_count: int = 80
    visual_obstacle_max_spawned_envs: int = 16
    visual_obstacle_usd_all_envs_cap: int = 0
    goal_collision_enabled: bool = False
    box_collision_enabled: bool = False
    goal_asset_path: str = ""

    def __post_init__(self) -> None:
        if not str(self.goal_asset_path or "").strip():
            self.goal_asset_path = default_goal_mesh_path()
        # Default USD layout (play.py). train.py rebuilds after ``--rover_localization_goal_asset_path`` etc.
        self.scene.oa_obstacles = RigidObjectCollectionCfg(
            rigid_objects=build_oa_obstacle_rigid_objects_dict(
                goal_mesh_path=self.goal_asset_path,
                box_collision_enabled=self.box_collision_enabled,
                goal_collision_enabled=self.goal_collision_enabled,
            )
        )
        self.decimation = 4
        self.episode_length_s = 120.0
        self.viewer.eye = (-6.0, 0.0, 104.0)
        self.viewer.lookat = (0.0, 0.0, 100.3)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, 0.0)
        if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
            self.sim.physx.enable_external_forces_every_iteration = True
        self.events.randomize_obstacles.params["goal_count"] = self.goal_count
        self.events.randomize_obstacles.params["box_count"] = self.box_count
        self.events.randomize_obstacles.params["goal_radial_distance"] = self.goal_radial_distance
        self.events.randomize_obstacles.params["box_radial_distance_m"] = self.oa_box_radial_distance_m
        self.events.randomize_obstacles.params["cylinder_y_half_width_m"] = self.oa_cylinder_y_half_width_m
        self.events.randomize_obstacles.params["rover_spawn_exclusion_radius_m"] = self.oa_rover_spawn_exclusion_radius_m
        self.rewards.forward_angular_progress.weight = self.oa_forward_progress_weight
        self.rewards.forward_speed_tracking_penalty.weight = self.oa_forward_speed_tracking_penalty_weight
        self.rewards.forward_speed_tracking_penalty.params["target_tangential_speed"] = self.oa_forward_speed_target_ms
        self.rewards.blue_box_avoidance_penalty.weight = self.oa_blue_box_penalty_weight
        self.rewards.blue_box_avoidance_penalty.params["danger_radius_m"] = self.oa_blue_box_danger_radius_m
        self.rewards.next_goal_pass_through_reward.weight = self.oa_goal_pass_through_weight
        self.rewards.disk_tangent_heading_alignment.weight = self.oa_heading_alignment_weight
        self.rewards.disk_tangent_heading_alignment.params["tolerance_rad"] = self.oa_heading_alignment_tolerance_rad
        self.rewards.radial_z_alignment_reward.weight = self.oa_radial_z_alignment_reward_weight
        self.rewards.radial_z_alignment_penalty.weight = self.oa_radial_z_alignment_penalty_weight
        self.rewards.radial_z_alignment_penalty.params["max_abs_z_error_before_clip_m"] = float(
            self.oa_radial_z_penalty_max_abs_m
        )
        self.rewards.upside_down_termination_penalty.weight = -abs(self.oa_upside_down_penalty_weight)
        self.rewards.fell_off_disk_termination_penalty.weight = -abs(self.oa_fell_off_disk_penalty_weight)
        self.rewards.near_fall_radial_penalty.weight = abs(self.oa_near_fall_penalty_weight)
        self.rewards.near_fall_radial_penalty.params["fall_radius_m"] = self.oa_fall_min_radius_m
        self.rewards.near_fall_radial_penalty.params["danger_band_m"] = self.oa_near_fall_danger_band_m
        self.rewards.visual_blue_box_ahead_penalty.weight = abs(self.oa_visual_box_ahead_penalty_weight)
        self.rewards.visual_blue_box_ahead_penalty.params["bearing_gate_rad"] = self.oa_visual_box_ahead_bearing_gate_rad
        self.rewards.visual_blue_box_ahead_penalty.params["range_gate_m"] = self.oa_visual_box_ahead_range_gate_m
        self.rewards.visual_goal_ahead_alignment_reward.weight = self.oa_visual_goal_ahead_reward_weight
        self.rewards.visual_goal_ahead_alignment_reward.params["bearing_gate_rad"] = (
            self.oa_visual_goal_ahead_bearing_gate_rad
        )
        self.rewards.visual_goal_ahead_alignment_reward.params["range_gate_m"] = self.oa_visual_goal_ahead_range_gate_m
        self.rewards.next_goal_approach_shaping.weight = self.oa_goal_approach_shaping_weight
        self.rewards.next_goal_approach_shaping.params["sigma_m"] = self.oa_goal_approach_sigma_m
        self.rewards.action_rate_penalty.weight = self.oa_action_rate_penalty_weight
        self.rewards.motion_jerk_penalty.weight = self.oa_motion_jerk_penalty_weight
        self.rewards.heading_oscillation_penalty.weight = self.oa_heading_oscillation_penalty_weight
        self.rewards.alive_bonus.weight = self.oa_alive_bonus_weight
        self.rewards.teacher_action_imitation.weight = self.oa_teacher_imitation_weight
        self.rewards.teacher_action_imitation.params["decay_env_steps"] = int(max(1, self.oa_teacher_imitation_decay_env_steps))
        self.rewards.teacher_action_imitation.params["match_sigma"] = self.oa_teacher_imitation_match_sigma
        self.terminations.fell_off_disk.params["min_radius"] = self.oa_fall_min_radius_m
        self.terminations.upside_down.params["min_radial_dot"] = self.oa_upside_down_min_radial_dot


@configclass
class RoverDiskWorldObstacleEnvCfg_PLAY(RoverDiskWorldObstacleEnvCfg):
    """Single-env defaults for ``play.py``."""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 300.0
        super().__post_init__()

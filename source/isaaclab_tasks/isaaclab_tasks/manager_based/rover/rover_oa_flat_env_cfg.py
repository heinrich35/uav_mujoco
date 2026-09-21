# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Flat goal navigation + obstacles; policy uses sim depth scan; critic uses privileged obstacle layout (+ aux distance)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg, RigidObjectCfg
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
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.rover.mdp.flat_goal_observations as fg_obs
import isaaclab_tasks.manager_based.rover.mdp.flat_goal_rewards as fg_r
import isaaclab_tasks.manager_based.rover.mdp.flat_goal_terminations as fg_term
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_events as oa_ev
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_rewards as oa_flat_r
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_observations as oa_flat_obs
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_vision as oa_vis
from isaaclab_tasks.manager_based.rover.mdp.actions import DifferentialDriveAction
from isaaclab_tasks.manager_based.rover.mdp.actions_cfg import DifferentialDriveActionCfg
import isaaclab_tasks.manager_based.rover.mdp.observations as rover_obs
from isaaclab_tasks.manager_based.rover.rover_disk_assets import (
    build_oa_obstacle_rigid_objects_dict,
    default_goal_mesh_path,
    ensure_rover_usd,
    make_rover_cfg,
)


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
class RoverOAFlatSceneCfg(InteractiveSceneCfg):
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
    rover_oa_rgb_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Rover/base_link/rover_oa_rgb",
        update_period=0.0,
        # Per-env Replicator: very small products (e.g. 128×96) can break depth annotator attach on some
        # Isaac Sim / RTX builds (``TypeError: unknown dtype, kind=f, size=0``). Keep min span ≥~300px
        # for stable RTX, or use NumPy 1.26.x (see Isaac Lab issue #3312). Width/height overridden in
        # ``RoverOAFlatNavEnvCfg.__post_init__`` from ``oa_flat_camera_{width,height}``.
        # Template matches training defaults; PLAY subclasses restore full-res RGB for GUI.
        width=320,
        height=304,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.28, 0.0, 0.12),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.CuboidCfg(
            size=(1.05, 1.05, 0.62),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.98, 0.06),
                emissive_color=(0.55, 0.48, 0.04),
                roughness=0.38,
                metallic=0.0,
            ),
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
        depth_scan = ObsTerm(
            func=oa_flat_obs.oa_flat_depth_scan_normalized,
            params={
                "sensor_name": None,
                "num_bins": 32,
                "clip_near_m": 0.05,
                "clip_far_m": 28.0,
                "row_center_frac": 0.58,
                "vertical_half_window_px": 7,
            },
        )
        visual_yellow_goal_tracks = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_tracks,
            params={"max_detections": 6},
        )
        # Parallels blue `_oa_flat_visual_*` aggregates + compares goal.blob vs obstacle.blob at slot 0.
        visual_yellow_goal_detection_stats = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_detection_stats,
            params={"max_detections": 6},
        )
        visual_yellow_goal_bearing_distribution = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_bearing_distribution,
            params={"max_detections": 6},
        )
        visual_yellow_goal_vs_blue_slot0 = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_vs_blue_slot0,
            params={"max_detections": 6},
        )
        imu_yaw_rate_noisy = ObsTerm(func=oa_vis.oa_flat_imu_yaw_rate_noisy)
        imu_lin_accel_noisy = ObsTerm(func=oa_vis.oa_flat_imu_lin_accel_noisy)
        optical_flow_forward_speed = ObsTerm(func=oa_vis.oa_flat_optical_flow_forward_speed)
        optical_flow_yaw_rate = ObsTerm(func=oa_vis.oa_flat_optical_flow_yaw_rate)
        wheel_odometry_estimate = ObsTerm(func=oa_vis.oa_flat_wheel_odometry_estimate_flat)
        projected_gravity = ObsTerm(func=env_mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("rover")})
        base_lin_vel_xy = ObsTerm(func=fg_obs.base_lin_vel_xy)
        base_ang_vel_z = ObsTerm(func=fg_obs.base_ang_vel_z)
        wheel_velocities_normalized = ObsTerm(func=rover_obs.wheel_velocities_normalized)
        last_action = ObsTerm(func=rover_obs.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        nearest_obstacles = ObsTerm(
            func=oa_flat_obs.oa_flat_privileged_nearest_obstacles,
            params={"k_nearest": 4, "dist_scale_m": 50.0},
        )
        nearest_obstacle_dist = ObsTerm(
            func=oa_flat_obs.oa_flat_privileged_nearest_obstacle_distance,
            params={"scale_m": 50.0},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AuxTrainingCfg(ObsGroup):
        nearest_obstacle_distance_target = ObsTerm(
            func=oa_flat_obs.oa_flat_aux_nearest_obstacle_distance_target,
            params={"scale_m": 50.0},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    aux_training: AuxTrainingCfg = AuxTrainingCfg()


def strip_oa_flat_policy_yellow_visual_terms(policy_cfg: ObservationsCfg.PolicyCfg) -> None:
    """Remove yellow RGB+D blob terms from the policy observation group (narrow vec baseline)."""
    policy_cfg.visual_yellow_goal_tracks = None
    policy_cfg.visual_yellow_goal_detection_stats = None
    policy_cfg.visual_yellow_goal_bearing_distribution = None
    policy_cfg.visual_yellow_goal_vs_blue_slot0 = None


def ensure_rover_oa_flat_policy_yellow_visual_terms(policy_cfg: ObservationsCfg.PolicyCfg, *, max_detections: int) -> None:
    """Recreate yellow ObsTerms after ``strip_oa_flat_policy_yellow_visual_terms`` (e.g. checkpoint play sync)."""
    kd = int(max_detections)
    if getattr(policy_cfg, "visual_yellow_goal_tracks", None) is None:
        policy_cfg.visual_yellow_goal_tracks = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_tracks,
            params={"max_detections": kd},
        )
    elif isinstance(getattr(policy_cfg.visual_yellow_goal_tracks, "params", None), dict):
        policy_cfg.visual_yellow_goal_tracks.params["max_detections"] = kd
    if getattr(policy_cfg, "visual_yellow_goal_detection_stats", None) is None:
        policy_cfg.visual_yellow_goal_detection_stats = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_detection_stats,
            params={"max_detections": kd},
        )
    elif isinstance(getattr(policy_cfg.visual_yellow_goal_detection_stats, "params", None), dict):
        policy_cfg.visual_yellow_goal_detection_stats.params["max_detections"] = kd
    if getattr(policy_cfg, "visual_yellow_goal_bearing_distribution", None) is None:
        policy_cfg.visual_yellow_goal_bearing_distribution = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_bearing_distribution,
            params={"max_detections": kd},
        )
    elif isinstance(getattr(policy_cfg.visual_yellow_goal_bearing_distribution, "params", None), dict):
        policy_cfg.visual_yellow_goal_bearing_distribution.params["max_detections"] = kd
    if getattr(policy_cfg, "visual_yellow_goal_vs_blue_slot0", None) is None:
        policy_cfg.visual_yellow_goal_vs_blue_slot0 = ObsTerm(
            func=oa_vis.oa_flat_visual_yellow_goal_vs_blue_slot0,
            params={"max_detections": kd},
        )
    elif isinstance(getattr(policy_cfg.visual_yellow_goal_vs_blue_slot0, "params", None), dict):
        policy_cfg.visual_yellow_goal_vs_blue_slot0.params["max_detections"] = kd


def sync_oa_flat_policy_yellow_visual_obs_terms(env_cfg) -> None:
    """Apply ``oa_flat_policy_include_yellow_goal_visual`` and slot count to policy ObsTerms (runs after CLI tweaks)."""
    obs_parent = getattr(env_cfg, "observations", None)
    pol = getattr(obs_parent, "policy", None)
    if pol is None:
        return
    if bool(getattr(env_cfg, "oa_flat_policy_include_yellow_goal_visual", False)):
        kd = int(getattr(env_cfg, "oa_flat_visual_yellow_goal_max_detections", 6))
        ensure_rover_oa_flat_policy_yellow_visual_terms(pol, max_detections=kd)
    else:
        strip_oa_flat_policy_yellow_visual_terms(pol)


@configclass
class EventCfg:
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
    # Relocate only after :func:`flat_goal_rewards.goal_reached_bonus` sets ``_oa_flat_relocate_goal_pending``
    # (same step credits success reward before interval events run — see ``RoverOAFlatNavRLEnv.step``).
    relocate_goal_interval = EventTerm(
        func=oa_ev.oa_flat_relocate_goal_if_pending,
        mode="interval",
        # Pending relocate only needs to run shortly after success; very tight cadence (e.g. 20–50 Hz) wastes wall time
        # on EventManager + GPU sync at large NUM_ENVS × cameras (~2× train slowdown vs ~0.1–0.2 s cadence).
        interval_range_s=(0.1, 0.18),
        is_global_time=True,
        params={
            "target_half_size_m": 23.0,
            "target_exclusion_half_size_m": 15.5,
            "target_min_spawn_dist_m": 5.0,
            "spawn_half_size_m": 15.0,
            "spawn_target_marker": True,
            "marker_asset_name": "goal_marker",
            "oa_flat_box_count": 70,
            "oa_flat_target_min_blue_box_clearance_m": 3.0,
            "max_target_samples": 64,
        },
    )


@configclass
class RewardsCfg:
    distance_to_target = RewTerm(func=fg_r.distance_to_target_reward, weight=1.45, params={"scale_m": 45.0})
    distance_progress = RewTerm(func=fg_r.distance_progress_reward, weight=14.0)
    distance_increase_penalty = RewTerm(func=fg_r.distance_increase_penalty, weight=1.2)
    heading_to_target = RewTerm(func=fg_r.heading_to_target_reward, weight=0.32)
    heading_goal_closure_speed = RewTerm(
        func=fg_r.heading_goal_closure_speed_reward,
        weight=2.2,
        params={"cap_ms": 3.25},
    )
    yaw_without_goal_closure = RewTerm(
        func=fg_r.yaw_rate_without_goal_closure_penalty,
        weight=-0.35,
        params={
            "min_dist_m": 3.5,
            "closure_speed_threshold_m_s": 0.22,
            "yaw_thresh_rad_s": 0.55,
            "yaw_ref_rad_s": 2.25,
        },
    )
    forward_velocity_to_target = RewTerm(
        func=fg_r.forward_velocity_to_target_reward, weight=2.3, params={"cap_ms": 3.25}
    )
    close_to_target_bonus = RewTerm(
        func=fg_r.close_to_target_bonus,
        weight=1.0,
        params={"threshold_m": 5.0, "bonus_max_dist_m": 2.75},
    )
    goal_reached_bonus = RewTerm(
        func=fg_r.goal_reached_bonus, weight=70.0, params={"threshold_m": 1.5, "set_relocate_pending": True}
    )
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
    blue_box_avoidance_penalty = RewTerm(
        func=oa_flat_r.oa_flat_blue_box_avoidance_penalty,
        weight=-9.0,
        params={
            "danger_radius_m": 2.85,
            "near_ramp_weight": 3.5,
            "gaussian_sigma_scale": 0.42,
        },
    )
    blue_box_collision_penalty = RewTerm(
        func=oa_flat_r.oa_flat_blue_box_collision_penalty,
        weight=-18.0,
        params={"collision_radius_m": 1.82},
    )
    body_reverse_vel_penalty = RewTerm(
        func=fg_r.body_reverse_linear_velocity_penalty,
        weight=-0.42,
        params={
            "scale_ms": 2.25,
            "far_boost_start_m": 5.0,
            "far_boost_max_mul": 2.2,
            "far_boost_ramp_m": 14.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_term.time_out, time_out=True)
    goal_reached = DoneTerm(func=fg_term.flat_goal_reached, params={"threshold_m": 1.5})
    upside_down = DoneTerm(func=fg_term.upside_down_flat, params={"min_up_dot": 0.2})
    wheelie = DoneTerm(func=fg_term.wheelie_pitch_termination, params={"min_pitch_sin": 0.2})


@configclass
class RoverOAFlatNavEnvCfg(ManagerBasedRLEnvCfg):
    """Flat goal + obstacles: depth scan policy obs, privileged obstacle features for critic, aux distance target."""

    # Fabric clone can leave RigidObjectCollection with one PhysX instance → reset() CUDA OOB; use USD clone path.
    scene: RoverOAFlatSceneCfg = RoverOAFlatSceneCfg(
        num_envs=2048, env_spacing=64.0, replicate_physics=True, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    oa_flat_camera_sensor_name: str = "rover_oa_rgb_cam"
    # Onboard RGB+depth resolution; small values can trigger Replicator/SyntheticData init failures.
    # Keep each side ≥~300px for stable Replicator annotators (#3312). Training uses a modest size; GUI PLAY
    # configs override to full-res for nicer previews while keeping train VRAM/descriptor load lower.
    oa_flat_camera_width: int = 320
    oa_flat_camera_height: int = 304
    oa_flat_depth_scan_num_bins: int = 32
    oa_flat_privileged_k_nearest: int = 4
    oa_flat_box_count: int = 70
    # 1 m cube obstacles (see ``build_oa_obstacle_rigid_objects_dict(..., box_extent_m=...)``).
    oa_flat_box_extent_m: float = 1.0
    # Defaults match a 50 m × 50 m plane (``plane_half_size_m``); overwritten in ``__post_init__``.
    oa_flat_box_arena_half_m: float = 25.0
    oa_flat_box_grid_step_m: float = 6.0
    oa_flat_box_min_spawn_clear_m: float = 4.0
    oa_flat_box_min_pair_clear_m: float = 1.5
    oa_flat_box_half_height_m: float = 0.5
    # Blue boxes are reward-only (no physics); strengthen shaping so paths through boxes lose to detours.
    oa_flat_blue_avoidance_weight: float = 20.0
    oa_flat_blue_avoidance_danger_radius_m: float = 1.7
    oa_flat_blue_avoidance_near_ramp_weight: float = 3.5
    oa_flat_blue_avoidance_gaussian_sigma_scale: float = 0.42
    oa_flat_blue_collision_weight: float = 60.0
    oa_flat_blue_collision_radius_m: float = 1.2
    # Global scalar for all OA-flat obstacle-avoidance rewards (blue avoid + blue collision).
    # Set to 0.0 to disable obstacle-avoidance shaping without touching goal-reaching rewards.
    oa_flat_obstacle_avoidance_reward_scale: float = 1.0
    # Per-environment RNG for goal/reset sampling when True (flat_goal_tasks without this flag use global torch.rand).
    oa_flat_per_env_seed_streams: bool = True
    # If True (default), ``oa_flat_box_arena_half_m`` tracks ``plane_half_size_m``. Set False via CLI when overriding arena.
    oa_flat_box_arena_use_plane: bool = True
    # Goal samples only after blue obstacles are placed; planar dist(goal, any blue center) >= this (m).
    oa_flat_target_min_blue_box_clearance_m: float = 3.0
    # Z lift of the kinematic goal cuboid above terrain + ``spawn_z_m`` (helps RGB yellow-blob cue).
    oa_flat_goal_marker_z_above_spawn_m: float = 0.42
    # Policy obs: ``max_detections × 4`` (sin, cos, tanh(range), conf); must match vision ``_peak_slot_tracks`` slots.
    oa_flat_visual_yellow_goal_max_detections: int = 6
    # When True: add (+OA_FLAT_VISUAL_YELLOW_GOAL_ENRICH_DIM) policy floats (stats+histogram+yellow-vs-blue slot0).
    oa_flat_visual_yellow_goal_enrichment_enabled: bool = True
    # Expected extra width added by enrichment ObsTerms (kept aligned with ``oa_flat_vision.OA_FLAT_VISUAL_YELLOW_GOAL_ENRICH_DIM``).
    oa_flat_visual_yellow_goal_enrichment_extra_dim: int = oa_vis.OA_FLAT_VISUAL_YELLOW_GOAL_ENRICH_DIM
    # Yellow-blob vision: soft track scaling by default (see ``oa_flat_vision.update_oa_flat_blue_detections_from_camera``).
    oa_flat_visual_yellow_excess_threshold: float = 0.045
    oa_flat_visual_yellow_conf_sigmoid_slope: float = 28.0
    # Legacy hard gate (multiply tracks by 1[c>gate]×c); prefer ``oa_flat_visual_yellow_track_use_hard_gate=False``.
    oa_flat_visual_yellow_track_conf_gate: float = 0.018
    oa_flat_visual_yellow_track_use_hard_gate: bool = False
    # When normalized yellow histogram peak exceeds this, add ``hist_peak_boost_floor`` under sqrt(conf) scaling.
    oa_flat_visual_yellow_hist_peak_boost_thr: float = 0.085
    oa_flat_visual_yellow_hist_peak_boost_floor: float = 0.22
    oa_flat_visual_yellow_score_relu: bool = True
    oa_flat_visual_yellow_score_boost: float = 1.35
    # Rare Isaac Lab #3312 workaround only: force carb ``/rtx/post/aa/op = Off``. Default False — globally disabling
    # post-AA breaks the balanced RTX stack (DLSS/DLAA + DL denoiser), producing heavy viewport/Replicator grain.
    oa_flat_rtx_force_post_aa_off: bool = False
    # When False (default): omit yellow goal-marker vision from policy obs (goal_vec / depth_scan / IMU baseline).
    # Set True for checkpoints trained with yellow slots + enrichment; play sync restores terms automatically when needed.
    oa_flat_policy_include_yellow_goal_visual: bool = False

    flat_body_reverse_vel_penalty_weight: float = 0.42
    flat_body_reverse_vel_scale_ms: float = 2.25
    # When farther than ``start_m`` from the goal, scale reverse penalty up toward ``max_mul`` over ``ramp_m``.
    flat_body_reverse_far_boost_start_m: float = 5.0
    flat_body_reverse_far_boost_max_mul: float = 2.2
    flat_body_reverse_far_boost_ramp_m: float = 14.0
    flat_reset_max_target_samples: int = 64

    flat_distance_to_target_weight: float = 1.45
    flat_distance_to_target_scale_m: float = 45.0
    flat_distance_progress_weight: float = 14.0
    flat_distance_increase_penalty_weight: float = 1.2
    flat_heading_to_target_weight: float = 0.32
    flat_heading_goal_closure_weight: float = 2.2
    flat_yaw_no_closure_penalty_weight: float = 0.35
    flat_yaw_closure_speed_threshold_m_s: float = 0.22
    flat_forward_velocity_to_target_weight: float = 2.3
    flat_forward_velocity_to_target_cap_ms: float = 3.25
    flat_close_to_target_bonus_weight: float = 1.2
    flat_close_to_target_threshold_m: float = 5.0
    flat_goal_reached_bonus_weight: float = 120.0
    flat_goal_reached_threshold_m: float = 1.5
    # Outer radius for quadratic ``close_to_target_bonus`` is ``flat_goal_reached_threshold_m + margin`` (see
    # ``__post_init__``). Relocation is **not** distance-triggered; it follows ``goal_reached_bonus`` only.
    flat_close_bonus_goal_margin_m: float = 1.25
    # Legacy CLI/scripts only: OA-flat uses ``oa_flat_relocate_goal_if_pending`` (no distance threshold).
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
    target_half_size_m: float = 23.0
    target_exclusion_half_size_m: float = 15.5
    target_min_spawn_dist_m: float = 5.0
    spawn_z_m: float = 0.2
    spawn_target_marker: bool = True
    flat_upside_down_min_dot: float = 0.2
    flat_wheelie_min_pitch_sin: float = 0.2
    flat_goal_train_stage: int = 0
    flat_goal_train_stage_note: str = ""

    def __post_init__(self) -> None:
        self.scene.rover_oa_rgb_cam.width = int(self.oa_flat_camera_width)
        self.scene.rover_oa_rgb_cam.height = int(self.oa_flat_camera_height)

        # Blue-box XY grid half-extent: match plane by default; CLI can set ``oa_flat_box_arena_use_plane=False``.
        if bool(self.oa_flat_box_arena_use_plane):
            self.oa_flat_box_arena_half_m = float(self.plane_half_size_m)

        refresh_oa_flat_blue_box_assets_from_env_fields(self)

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
        # RTX / Replicator (multi-env train vs 1-env GUI play):
        # - ``--rendering_mode`` from AppLauncher (train scripts default ``performance``) loads the matching preset.
        # - Do **not** set ``enable_dl_denoiser`` / ``samples_per_pixel`` here: forcing them overrides ``performance``
        #   and can exhaust GPU descriptor pools (ParameterBlock / descriptor sets) at NUM_ENVS × onboard cameras.
        # - GUI play tasks (# ``RoverOAFlatNavEnvCfg_PLAY``, ``CONTINUOUS_PLAY``) call ``_oa_flat_boost_rtx_for_gui_play``.
        # - Fallback ``rendering_mode`` when carb string is empty (cameras off / rare paths).
        # Training defaults to ``performance``; AppLauncher ``--rendering_mode`` still overrides when set (e.g. play).
        self.sim.render.rendering_mode = "performance"
        self.sim.render.enable_dl_denoiser = None
        self.sim.render.samples_per_pixel = None
        _rtx_cs = dict(self.sim.render.carb_settings) if self.sim.render.carb_settings else {}
        if bool(self.oa_flat_rtx_force_post_aa_off):
            self.sim.render.antialiasing_mode = "Off"
            _rtx_cs["/rtx/post/aa/op"] = 0  # 0=Off (3=DLSS, 4=DLAA per Isaac Lab render tests)
        else:
            # Leave ``antialiasing_mode`` unset so ``balanced`` preset + DLSS/TAA stack stays intact.
            self.sim.render.antialiasing_mode = None
        self.sim.render.carb_settings = _rtx_cs if _rtx_cs else None
        self.viewer.eye = (H * 1.4, H * 1.1, 18.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.num_rerenders_on_reset = 1

        self.scene.rover.init_state.pos = (0.0, 0.0, float(self.spawn_z_m))

        wcfg = self.scene.rover.actuators["wheels"]
        wcfg.effort_limit_sim = float(self.flat_wheel_effort_limit_sim)
        wcfg.damping = float(self.flat_actuator_wheel_damping)
        wcfg.velocity_limit_sim = float(self.flat_wheel_velocity_limit_sim)
        self.actions.drive.max_target_slew_rad_s2 = float(self.flat_wheel_target_slew_rad_s2)
        self.actions.drive.max_forward_vel = float(self.flat_drive_max_forward_vel_m_s)
        self.actions.drive.max_turn_diff_vel = float(self.flat_drive_max_turn_yaw_rate_rad_s)
        self.actions.drive.track_width_m = float(self.flat_drive_track_width_m)

        rp = self.events.reset_oa_flat.params
        rp["spawn_half_size_m"] = float(self.spawn_half_size_m)
        rp["target_half_size_m"] = float(self.target_half_size_m)
        rp["target_exclusion_half_size_m"] = float(self.target_exclusion_half_size_m)
        rp["target_min_spawn_dist_m"] = float(self.target_min_spawn_dist_m)
        rp["spawn_z_m"] = float(self.spawn_z_m)
        rp["spawn_target_marker"] = bool(self.spawn_target_marker)
        rp["max_target_samples"] = int(self.flat_reset_max_target_samples)

        rg = _oa_flat_relocate_interval_event_params(self)
        if rg is not None:
            rg["target_half_size_m"] = float(self.target_half_size_m)
            rg["target_exclusion_half_size_m"] = float(self.target_exclusion_half_size_m)
            rg["target_min_spawn_dist_m"] = float(self.target_min_spawn_dist_m)
            rg["spawn_half_size_m"] = float(self.spawn_half_size_m)
            rg["spawn_target_marker"] = bool(self.spawn_target_marker)
            rg["marker_asset_name"] = "goal_marker"
            rg["max_target_samples"] = int(self.flat_reset_max_target_samples)

        self.terminations.goal_reached = DoneTerm(func=fg_term.never_terminate)

        sc = float(self.flat_distance_to_target_scale_m)
        self.observations.policy.goal_vec_xy.params["scale_m"] = sc
        self.observations.policy.dist_goal.params["scale_m"] = sc
        self.observations.policy.depth_scan.params["num_bins"] = int(self.oa_flat_depth_scan_num_bins)
        sync_oa_flat_policy_yellow_visual_obs_terms(self)
        self.observations.privileged.nearest_obstacles.params["k_nearest"] = int(self.oa_flat_privileged_k_nearest)
        self.observations.privileged.nearest_obstacles.params["dist_scale_m"] = sc
        self.observations.privileged.nearest_obstacle_dist.params["scale_m"] = sc
        self.observations.aux_training.nearest_obstacle_distance_target.params["scale_m"] = sc
        self.rewards.distance_to_target.params["scale_m"] = sc
        self.rewards.distance_to_target.weight = float(self.flat_distance_to_target_weight)
        self.rewards.distance_progress.weight = float(self.flat_distance_progress_weight)
        self.rewards.distance_increase_penalty.weight = float(self.flat_distance_increase_penalty_weight)
        self.rewards.heading_to_target.weight = float(self.flat_heading_to_target_weight)
        self.rewards.heading_goal_closure_speed.weight = float(self.flat_heading_goal_closure_weight)
        self.rewards.heading_goal_closure_speed.params["cap_ms"] = float(self.flat_forward_velocity_to_target_cap_ms)
        self.rewards.yaw_without_goal_closure.weight = -abs(float(self.flat_yaw_no_closure_penalty_weight))
        self.rewards.yaw_without_goal_closure.params["closure_speed_threshold_m_s"] = float(
            self.flat_yaw_closure_speed_threshold_m_s
        )
        self.rewards.forward_velocity_to_target.weight = float(self.flat_forward_velocity_to_target_weight)
        self.rewards.forward_velocity_to_target.params["cap_ms"] = float(self.flat_forward_velocity_to_target_cap_ms)
        self.rewards.close_to_target_bonus.weight = float(self.flat_close_to_target_bonus_weight)
        self.rewards.close_to_target_bonus.params["threshold_m"] = float(self.flat_close_to_target_threshold_m)
        thr_goal = float(self.flat_goal_reached_threshold_m)
        cap_close = thr_goal + float(self.flat_close_bonus_goal_margin_m)
        self.rewards.close_to_target_bonus.params["bonus_max_dist_m"] = cap_close
        self.rewards.goal_reached_bonus.weight = float(self.flat_goal_reached_bonus_weight)
        self.rewards.goal_reached_bonus.params["threshold_m"] = thr_goal
        self.rewards.goal_reached_bonus.params["set_relocate_pending"] = True
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
        _oa_obs_scale = max(0.0, float(self.oa_flat_obstacle_avoidance_reward_scale))
        self.rewards.blue_box_avoidance_penalty.weight = -abs(float(self.oa_flat_blue_avoidance_weight)) * _oa_obs_scale
        self.rewards.blue_box_avoidance_penalty.params["danger_radius_m"] = float(self.oa_flat_blue_avoidance_danger_radius_m)
        self.rewards.blue_box_avoidance_penalty.params["near_ramp_weight"] = float(self.oa_flat_blue_avoidance_near_ramp_weight)
        self.rewards.blue_box_avoidance_penalty.params["gaussian_sigma_scale"] = float(
            self.oa_flat_blue_avoidance_gaussian_sigma_scale
        )
        self.rewards.blue_box_collision_penalty.weight = -abs(float(self.oa_flat_blue_collision_weight)) * _oa_obs_scale
        self.rewards.blue_box_collision_penalty.params["collision_radius_m"] = float(self.oa_flat_blue_collision_radius_m)
        self.rewards.body_reverse_vel_penalty.weight = -abs(float(self.flat_body_reverse_vel_penalty_weight))
        self.rewards.body_reverse_vel_penalty.params["scale_ms"] = float(self.flat_body_reverse_vel_scale_ms)
        self.rewards.body_reverse_vel_penalty.params["far_boost_start_m"] = float(self.flat_body_reverse_far_boost_start_m)
        self.rewards.body_reverse_vel_penalty.params["far_boost_max_mul"] = float(self.flat_body_reverse_far_boost_max_mul)
        self.rewards.body_reverse_vel_penalty.params["far_boost_ramp_m"] = float(self.flat_body_reverse_far_boost_ramp_m)
        self.terminations.upside_down.params["min_up_dot"] = float(self.flat_upside_down_min_dot)
        self.terminations.wheelie.params["min_pitch_sin"] = float(self.flat_wheelie_min_pitch_sin)


def _oa_flat_boost_rtx_for_gui_play(env_cfg: RoverOAFlatNavEnvCfg) -> None:
    """Tighter onboard-camera RTX after ``--rendering_mode quality/balanced``; use only for single-env GUI play."""
    env_cfg.sim.render.enable_dl_denoiser = True
    env_cfg.sim.render.samples_per_pixel = 8


@configclass
class RoverOAFlatNavEnvCfg_PLAY(RoverOAFlatNavEnvCfg):
    """Single-env GUI play: full-resolution onboard camera + RTX denoise/SPP (see ``_oa_flat_boost_rtx_for_gui_play``)."""

    oa_flat_camera_width: int = 384
    oa_flat_camera_height: int = 304

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        super().__post_init__()
        _oa_flat_boost_rtx_for_gui_play(self)


@configclass
class RoverOAFlatNavEnvCfg_CONTINUOUS_PLAY(RoverOAFlatNavEnvCfg):
    """Same arena / spawn / goals / blue-box layout as training; only ``episode_length_s`` is huge for open-ended play."""

    oa_flat_camera_width: int = 384
    oa_flat_camera_height: int = 304

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        # Match flat-goal continuous play / train: slightly tighter target half-extent than base OA-Flat (23 m).
        self.target_half_size_m = 20.0
        self.target_exclusion_half_size_m = 15.5
        super().__post_init__()
        self.episode_length_s = 1.0e9
        _oa_flat_boost_rtx_for_gui_play(self)


def _oa_flat_relocate_interval_event_params(env_cfg: RoverOAFlatNavEnvCfg) -> dict | None:
    """Mutable ``params`` dict for interval goal relocate (standard close-based relocate or pending-after-success).

    Physparam envs use :func:`oa_flat_relocate_goal_if_pending` — same kwargs / params shape as relocate interval.
    """
    ev = env_cfg.events
    if hasattr(ev, "relocate_goal_interval"):
        return ev.relocate_goal_interval.params
    if hasattr(ev, "relocate_goal_after_success"):
        return ev.relocate_goal_after_success.params
    return None


def refresh_oa_flat_blue_box_assets_from_env_fields(env_cfg: RoverOAFlatNavEnvCfg) -> None:
    """Rebuild ``scene.oa_blue_boxes`` and mirror blue-box scalars into reset/relocate event params.

    Use after mutating ``oa_flat_box_*`` / goal–box clearance (CLI, Hydra, or play flags) so USD prims and
    ``EventTerm`` kwargs stay consistent with ``oa_flat_box_count``.
    """
    env_cfg.scene.oa_blue_boxes = RigidObjectCollectionCfg(
        rigid_objects=build_oa_obstacle_rigid_objects_dict(
            max_boxes=int(env_cfg.oa_flat_box_count),
            max_goals=0,
            goal_mesh_path=default_goal_mesh_path(),
            box_collision_enabled=False,
            goal_collision_enabled=False,
            box_extent_m=float(env_cfg.oa_flat_box_extent_m),
        )
    )
    rp = env_cfg.events.reset_oa_flat.params
    rg = _oa_flat_relocate_interval_event_params(env_cfg)
    rp["oa_flat_box_count"] = int(env_cfg.oa_flat_box_count)
    rp["oa_flat_box_arena_half_m"] = float(env_cfg.oa_flat_box_arena_half_m)
    rp["oa_flat_box_grid_step_m"] = float(env_cfg.oa_flat_box_grid_step_m)
    rp["oa_flat_box_min_spawn_clear_m"] = float(env_cfg.oa_flat_box_min_spawn_clear_m)
    rp["oa_flat_box_min_pair_clear_m"] = float(env_cfg.oa_flat_box_min_pair_clear_m)
    rp["oa_flat_box_half_height_m"] = float(env_cfg.oa_flat_box_half_height_m)
    rp["oa_flat_target_min_blue_box_clearance_m"] = float(env_cfg.oa_flat_target_min_blue_box_clearance_m)
    if rg is not None:
        rg["oa_flat_box_count"] = int(env_cfg.oa_flat_box_count)
        rg["oa_flat_target_min_blue_box_clearance_m"] = float(env_cfg.oa_flat_target_min_blue_box_clearance_m)


def sync_oa_flat_play_obstacle_reward_weights(env_cfg: RoverOAFlatNavEnvCfg) -> None:
    """Reapply blue obstacle reward weights after ``oa_flat_obstacle_avoidance_reward_scale`` changes (e.g. play CLI).

    ``RoverOAFlatNavEnvCfg.__post_init__`` already sets these once; call this before ``gym.make`` when overriding
    the scale so play matches a checkpoint trained with ``oa_flat_obstacle_avoidance_reward_scale=0``.
    """
    _oa_obs_scale = max(0.0, float(env_cfg.oa_flat_obstacle_avoidance_reward_scale))
    env_cfg.rewards.blue_box_avoidance_penalty.weight = -abs(float(env_cfg.oa_flat_blue_avoidance_weight)) * _oa_obs_scale
    env_cfg.rewards.blue_box_collision_penalty.weight = -abs(float(env_cfg.oa_flat_blue_collision_weight)) * _oa_obs_scale

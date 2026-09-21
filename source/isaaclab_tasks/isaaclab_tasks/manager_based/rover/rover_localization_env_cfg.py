# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rover localization task: frozen expert drives, agent estimates 3-D position.

Environment layout matches ``RoverOAFlatNavEnvCfg`` (50 m × 50 m plane, 70 blue
boxes, yellow goal marker with relocate). The agent observes the same sensory
stream as the OA-Flat policy but does **not** control the rover. Instead the
agent outputs a 3-D position estimate which is scored against the ground-truth
rover position via the reward function.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as env_mdp
from isaaclab.envs.mdp import rewards as mdp_r
from isaaclab.envs.mdp import terminations as mdp_term
from isaaclab.managers import ActionTermCfg
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
import isaaclab_tasks.manager_based.rover.mdp.localization_observations as loc_obs
import isaaclab_tasks.manager_based.rover.mdp.localization_rewards as loc_r
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_events as oa_ev
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_observations as oa_flat_obs
import isaaclab_tasks.manager_based.rover.mdp.oa_flat_vision as oa_vis
from isaaclab_tasks.manager_based.rover.mdp.actions_localization import RoverPositionEstimationActionCfg
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
class RoverLocalizationSceneCfg(InteractiveSceneCfg):
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
    estimated_target_visual: RigidObjectCfg | None = None


@configclass
class ActionsCfg:
    """3-D position estimate action: (x_est, y_est, z_est)."""
    estimate: RoverPositionEstimationActionCfg = RoverPositionEstimationActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: same OA-Flat policy stream (vision + goal + proprio).

        The ``last_action`` slot returns the *expert's* last drive command so that
        the frozen OA-Flat policy receives the exact observation layout it was
        trained on.
        """
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
        last_action = ObsTerm(func=loc_obs.expert_last_drive_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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
    relocate_goal_interval = EventTerm(
        func=oa_ev.oa_flat_continuous_relocate_goal_if_close,
        mode="interval",
        interval_range_s=(0.1, 0.18),
        is_global_time=True,
        params={
            "goal_threshold_m": 5.0,
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
    """Reward for accurate 3-D position estimation."""
    position_estimation_reward = RewTerm(
        func=loc_r.position_estimation_reward,
        weight=1.0,
        params={
            "position_error_scale_m": 50.0,
            "position_error_exp_scale": 0.1,
            "position_error_linear_weight": 0.5,
            "position_error_exp_weight": 0.5,
        },
    )
    position_estimation_bonus = RewTerm(
        func=loc_r.position_estimation_bonus,
        weight=0.0,
        params={
            "threshold_m": 1.0,
            "bonus_weight": 5.0,
        },
    )
    goal_reached_bonus = RewTerm(
        func=fg_r.goal_reached_bonus,
        weight=0.0,
        params={"threshold_m": 1.5, "set_relocate_pending": True},
    )
    alive_bonus = RewTerm(func=mdp_r.is_alive, weight=0.05)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_term.time_out, time_out=True)
    upside_down = DoneTerm(func=fg_term.upside_down_flat, params={"min_up_dot": 0.2})
    wheelie = DoneTerm(func=fg_term.wheelie_pitch_termination, params={"min_pitch_sin": 0.2})


@configclass
class RoverLocalizationEnvCfg(ManagerBasedRLEnvCfg):
    """Rover localization: estimate 3-D position while an expert drives toward goals."""

    scene: RoverLocalizationSceneCfg = RoverLocalizationSceneCfg(
        num_envs=2048, env_spacing=64.0, replicate_physics=True, clone_in_fabric=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Expert policy path (OA-Flat goal navigation checkpoint)
    expert_policy_path: str = (
        "/home/heinz/isaaclab_rover/logs/rsl_rl/rover_oa_flat_env/2026-05-10_16-52-12/final_model.pt"
    )

    # Reward parameters (override via CLI --rover_localization_*)
    localization_position_error_scale_m: float = 50.0
    localization_position_error_exp_scale: float = 0.1
    localization_position_error_linear_weight: float = 0.5
    localization_position_error_exp_weight: float = 0.5
    localization_position_bonus_threshold_m: float = 1.0
    localization_position_bonus_weight: float = 5.0
    localization_alive_weight: float = 0.05

    # Goal-reached bonus (weight=0: no reward to localization agent, but triggers
    # the _oa_flat_relocate_goal_pending flag so the interval event moves the goal)
    flat_goal_reached_bonus_weight: float = 1.0e-8
    flat_goal_reached_threshold_m: float = 1.5
    flat_goal_relocate_threshold_m: float = 5.0

    # Debug visualization: spawn a red sphere at the agent's estimated position
    debug_mode: bool = False

    # Camera parameters
    oa_flat_camera_sensor_name: str = "rover_oa_rgb_cam"
    oa_flat_camera_width: int = 320
    oa_flat_camera_height: int = 304
    oa_flat_depth_scan_num_bins: int = 32

    # Blue box parameters
    oa_flat_box_count: int = 70
    oa_flat_box_extent_m: float = 1.0
    oa_flat_box_arena_half_m: float = 25.0
    oa_flat_box_grid_step_m: float = 6.0
    oa_flat_box_min_spawn_clear_m: float = 4.0
    oa_flat_box_min_pair_clear_m: float = 1.5
    oa_flat_box_half_height_m: float = 0.5
    oa_flat_box_arena_use_plane: bool = True
    oa_flat_target_min_blue_box_clearance_m: float = 3.0

    # Goal marker
    oa_flat_goal_marker_z_above_spawn_m: float = 0.42
    oa_flat_visual_yellow_goal_max_detections: int = 6
    oa_flat_visual_yellow_goal_enrichment_enabled: bool = True
    oa_flat_visual_yellow_goal_enrichment_extra_dim: int = oa_vis.OA_FLAT_VISUAL_YELLOW_GOAL_ENRICH_DIM

    # Yellow blob vision parameters
    oa_flat_visual_yellow_excess_threshold: float = 0.045
    oa_flat_visual_yellow_conf_sigmoid_slope: float = 28.0
    oa_flat_visual_yellow_track_conf_gate: float = 0.018
    oa_flat_visual_yellow_track_use_hard_gate: bool = False
    oa_flat_visual_yellow_hist_peak_boost_thr: float = 0.085
    oa_flat_visual_yellow_hist_peak_boost_floor: float = 0.22
    oa_flat_visual_yellow_score_relu: bool = True
    oa_flat_visual_yellow_score_boost: float = 1.35

    oa_flat_per_env_seed_streams: bool = True

    # Geometry
    plane_half_size_m: float = 25.0
    spawn_half_size_m: float = 15.0
    target_half_size_m: float = 23.0
    target_exclusion_half_size_m: float = 15.5
    target_min_spawn_dist_m: float = 5.0
    spawn_z_m: float = 0.2
    spawn_target_marker: bool = True

    # Drive physics (mirrors OA-Flat defaults for the expert)
    flat_wheel_effort_limit_sim: float = 2500.0
    flat_actuator_wheel_damping: float = 150.0
    flat_wheel_velocity_limit_sim: float = 165.0
    flat_wheel_target_slew_rad_s2: float = 720.0
    flat_drive_max_forward_vel_m_s: float = 2.25
    flat_drive_max_turn_yaw_rate_rad_s: float = 9.0
    flat_drive_track_width_m: float = 0.48

    # Visual toggle (False = strip yellow goal ObsTerms; True = include).
    # Must stay False for the 2026-05-10 checkpoint (expects 57-dim obs). Setting True causes dim mismatch (109 vs 57).
    oa_flat_policy_include_yellow_goal_visual: bool = False

    # Distance scale for goal observations
    localization_distance_scale_m: float = 50.0

    # Episode
    episode_length_s: float = 30.0

    def __post_init__(self) -> None:
        self.scene.rover_oa_rgb_cam.width = int(self.oa_flat_camera_width)
        self.scene.rover_oa_rgb_cam.height = int(self.oa_flat_camera_height)

        # Blue-box XY grid half-extent: match plane by default
        if bool(self.oa_flat_box_arena_use_plane):
            self.oa_flat_box_arena_half_m = float(self.plane_half_size_m)

        # Rebuild blue box assets
        self.scene.oa_blue_boxes = RigidObjectCollectionCfg(
            rigid_objects=build_oa_obstacle_rigid_objects_dict(
                max_boxes=int(self.oa_flat_box_count),
                max_goals=0,
                goal_mesh_path=default_goal_mesh_path(),
                box_collision_enabled=False,
                goal_collision_enabled=False,
                box_extent_m=float(self.oa_flat_box_extent_m),
            )
        )

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
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.render.rendering_mode = "performance"
        self.sim.render.enable_dl_denoiser = None
        self.sim.render.samples_per_pixel = None
        self.sim.render.antialiasing_mode = None
        self.sim.render.carb_settings = None
        self.viewer.eye = (H * 1.4, H * 1.1, 18.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.num_rerenders_on_reset = 1

        self.scene.rover.init_state.pos = (0.0, 0.0, float(self.spawn_z_m))

        # Update event params
        rp = self.events.reset_oa_flat.params
        rp["spawn_half_size_m"] = float(self.spawn_half_size_m)
        rp["target_half_size_m"] = float(self.target_half_size_m)
        rp["target_exclusion_half_size_m"] = float(self.target_exclusion_half_size_m)
        rp["target_min_spawn_dist_m"] = float(self.target_min_spawn_dist_m)
        rp["spawn_z_m"] = float(self.spawn_z_m)
        rp["spawn_target_marker"] = bool(self.spawn_target_marker)
        rp["max_target_samples"] = 64
        rp["oa_flat_box_count"] = int(self.oa_flat_box_count)
        rp["oa_flat_box_arena_half_m"] = float(self.oa_flat_box_arena_half_m)
        rp["oa_flat_box_grid_step_m"] = float(self.oa_flat_box_grid_step_m)
        rp["oa_flat_box_min_spawn_clear_m"] = float(self.oa_flat_box_min_spawn_clear_m)
        rp["oa_flat_box_min_pair_clear_m"] = float(self.oa_flat_box_min_pair_clear_m)
        rp["oa_flat_box_half_height_m"] = float(self.oa_flat_box_half_height_m)
        rp["oa_flat_target_min_blue_box_clearance_m"] = float(self.oa_flat_target_min_blue_box_clearance_m)

        rg = self.events.relocate_goal_interval.params
        rg["goal_threshold_m"] = float(self.flat_goal_relocate_threshold_m)
        rg["target_half_size_m"] = float(self.target_half_size_m)
        rg["target_exclusion_half_size_m"] = float(self.target_exclusion_half_size_m)
        rg["target_min_spawn_dist_m"] = float(self.target_min_spawn_dist_m)
        rg["spawn_half_size_m"] = float(self.spawn_half_size_m)
        rg["spawn_target_marker"] = bool(self.spawn_target_marker)
        rg["marker_asset_name"] = "goal_marker"
        rg["max_target_samples"] = 64
        rg["oa_flat_box_count"] = int(self.oa_flat_box_count)
        rg["oa_flat_target_min_blue_box_clearance_m"] = float(self.oa_flat_target_min_blue_box_clearance_m)

        # Update observation params
        self.observations.policy.depth_scan.params["num_bins"] = int(self.oa_flat_depth_scan_num_bins)
        sc = float(self.localization_distance_scale_m)
        self.observations.policy.goal_vec_xy.params["scale_m"] = sc
        self.observations.policy.dist_goal.params["scale_m"] = sc

        # Update reward params
        self.rewards.position_estimation_reward.weight = 1.0
        self.rewards.position_estimation_reward.params["position_error_scale_m"] = float(
            self.localization_position_error_scale_m
        )
        self.rewards.position_estimation_reward.params["position_error_exp_scale"] = float(
            self.localization_position_error_exp_scale
        )
        self.rewards.position_estimation_reward.params["position_error_linear_weight"] = float(
            self.localization_position_error_linear_weight
        )
        self.rewards.position_estimation_reward.params["position_error_exp_weight"] = float(
            self.localization_position_error_exp_weight
        )
        self.rewards.position_estimation_bonus.weight = float(self.localization_position_bonus_weight)
        self.rewards.position_estimation_bonus.params["threshold_m"] = float(
            self.localization_position_bonus_threshold_m
        )
        self.rewards.alive_bonus.weight = float(self.localization_alive_weight)

        # Goal-reached bonus: tiny weight so the reward manager actually calls the
        # function (it skips terms with weight==0).  The bonus itself is negligible
        # to the localization reward, but the side-effect of setting
        # _oa_flat_relocate_goal_pending triggers interval goal relocation.
        self.rewards.goal_reached_bonus.weight = float(self.flat_goal_reached_bonus_weight)
        self.rewards.goal_reached_bonus.params["threshold_m"] = float(self.flat_goal_reached_threshold_m)
        self.rewards.goal_reached_bonus.params["set_relocate_pending"] = True

        # Spawn debug red sphere at estimated position when debug_mode is enabled
        if getattr(self, "debug_mode", False):
            from isaaclab.sim.spawners.shapes import SphereCfg
            self.scene.estimated_target_visual = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/EstimatedTargetVisual",
                spawn=SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        emissive_color=(0.5, 0.0, 0.0),
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
        else:
            self.scene.estimated_target_visual = None

        # Configure yellow visual obs terms based on flag
        if bool(getattr(self, "oa_flat_policy_include_yellow_goal_visual", False)):
            kd = int(getattr(self, "oa_flat_visual_yellow_goal_max_detections", 6))
            # Ensure terms exist with correct max_detections
            for term_name in (
                "visual_yellow_goal_tracks",
                "visual_yellow_goal_detection_stats",
                "visual_yellow_goal_bearing_distribution",
                "visual_yellow_goal_vs_blue_slot0",
            ):
                term = getattr(self.observations.policy, term_name, None)
                if term is not None and isinstance(getattr(term, "params", None), dict):
                    term.params["max_detections"] = kd
        else:
            # Strip yellow visual terms to match the default OA-Flat narrow baseline
            self.observations.policy.visual_yellow_goal_tracks = None
            self.observations.policy.visual_yellow_goal_detection_stats = None
            self.observations.policy.visual_yellow_goal_bearing_distribution = None
            self.observations.policy.visual_yellow_goal_vs_blue_slot0 = None


@configclass
class RoverLocalizationEnvCfg_PLAY(RoverLocalizationEnvCfg):
    """Single-env GUI play: full-resolution onboard camera + RTX denoise/SPP + debug sphere."""

    debug_mode: bool = True
    episode_length_s: float = 180.0
    oa_flat_camera_width: int = 384
    oa_flat_camera_height: int = 304

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        super().__post_init__()
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 8


@configclass
class RoverLocalizationEnvCfg_CONTINUOUS_PLAY(RoverLocalizationEnvCfg):
    """Continuous open-ended GUI play: same arena as training; episode_length_s is huge.

    Matches ``RoverOAFlatNavEnvCfg_CONTINUOUS_PLAY``: 50 m plane, 70 blue boxes,
    goal relocates after expert reaches it (via ``goal_reached_bonus`` pending flag).
    The agent's position-estimate action runs on top of this open-ended behaviour.
    """

    debug_mode: bool = True
    episode_length_s: float = 1.0e9
    oa_flat_camera_width: int = 384
    oa_flat_camera_height: int = 304

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.env_spacing = 8.0
        # Match OA-Flat continuous play: slightly tighter target half-extent
        self.target_half_size_m = 20.0
        self.target_exclusion_half_size_m = 15.5
        super().__post_init__()
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 8

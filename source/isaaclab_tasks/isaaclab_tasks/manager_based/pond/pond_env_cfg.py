# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pond environment configuration — fluid dynamics simulation with floating robot."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp import rewards as mdp_r
from isaaclab.envs.mdp import terminations as mdp_term
from isaaclab_tasks.manager_based.pond import pond_mdp

from isaaclab_assets.robots.duck import DUCK_CFG
from isaaclab_assets.robots.gosling import GOSLING_CFG
from isaaclab_assets.robots.swan import SWAN_CFG
from isaaclab_tasks.manager_based.pond.pond_ui_window import PondEnvWindow

# Path to distractor USD meshes
USD_DIR = "/home/heinz/isaaclab_uav/assets/pond/glb/usd"


def _make_distractor(name: str, idx: int) -> RigidObjectCfg:
    """Factory for a kinematic distractor asset loaded from USD."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}_{idx}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DIR}/{name}/{name}.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Fluid dynamics parameters
# ---------------------------------------------------------------------------

@configclass
class FluidDynamicsCfg:
    """Configuration for pond fluid dynamics simulation.

    The water surface is at world z = 0. Forces are applied to simulate
    buoyancy, hydrodynamic drag, and user-controlled paddling actuation.
    """

    water_density: float = 1000.0
    """Water density in kg/m³."""

    gravity_mag: float = 9.81
    """Gravitational acceleration magnitude in m/s² (used for buoyancy)."""

    # --- Buoyancy ---
    displaced_volume: float = 0.008
    """Displaced water volume at full submersion in m³ (~8 liters for a small water bird)."""

    buoyancy_ramp_depth: float = 0.05
    """Depth range (m) over which buoyancy force ramps from 0 to full.

    Below z=0, buoyancy increases linearly from 0 at z=0 to full at z=-buoyancy_ramp_depth.
    Below z=-buoyancy_ramp_depth, full buoyancy is applied.
    """

    # --- Linear drag (hydrodynamic) ---
    #drag_linear_air: tuple[float, float, float] = (0.5, 2.0, 2.0)
    drag_linear_air: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Linear drag coefficients (body-frame x, y, z) when in air (above water)."""

    #drag_linear_water: tuple[float, float, float] = (8.0, 40.0, 25.0)
    drag_linear_water: tuple[float, float, float] = (16.0, 80.0, 50.0)
    """Linear drag coefficients (body-frame x, y, z) when in water.

    Higher sideways (y) drag for directional stability like a boat hull.
    """

    # --- Angular drag (rotational damping) ---
    drag_angular_air: tuple[float, float, float] = (1.0, 1.0, 1.5)
    """Angular drag coefficients (body-frame x, y, z) when in air."""

    drag_angular_water: tuple[float, float, float] = (5.0, 5.0, 12.0)
    """Angular drag coefficients (body-frame x, y, z) when in water."""

    # --- Actuation (paddling forces) ---
    max_forward_force: float = 25.0
    """Maximum forward paddle force in body +X (N)."""

    max_turn_torque: float = 8.0
    """Maximum turning torque around body +Z (N·m)."""


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class PondSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the pond environment.

    Contains:
    - A ground/pond-bed plane below the water
    - A visual water surface at z=0
    - The gosling robot floating on the water
    - Dome light for illumination
    """

    # Pond bed — solid bottom at z=-2.0 (below water surface at z=0)
    pond_bed = AssetBaseCfg(
        prim_path="/World/pondBed",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.20, 0.15),
                roughness=1.0,
                metallic=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.7,
                restitution=0.05,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )

    # Visual water surface at z=0
    water_surface = AssetBaseCfg(
        prim_path="/World/waterSurface",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.5, 0.8),
                emissive_color=(0.02, 0.05, 0.08),
                roughness=0.3,
                metallic=0.1,
                opacity=0.6,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Dome light
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(0.85, 0.90, 1.0),
        ),
    )

    # Distant light (sun)
    sun_light = AssetBaseCfg(
        prim_path="/World/sunLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 0.95, 0.85),
            angle=1.5,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(20.0, 10.0, 30.0),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
    )

    # Gosling robot — single-link floating articulation
    gosling: ArticulationCfg = GOSLING_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Gosling",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )

    # 3 Duck instances — single-link floating articulation (collision sphere 0.6m)
    duck_0: ArticulationCfg = DUCK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/duck_0",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.0, 1.5, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )
    duck_1: ArticulationCfg = DUCK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/duck_1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-2.0, 2.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )
    duck_2: ArticulationCfg = DUCK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/duck_2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -2.5, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )

    # 3 Swan instances — single-link floating articulation (collision sphere 0.8m)
    swan_0: ArticulationCfg = SWAN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/swan_0",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-3.0, -2.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )
    swan_1: ArticulationCfg = SWAN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/swan_1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(3.0, -1.5, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )
    swan_2: ArticulationCfg = SWAN_CFG.replace(
        prim_path="{ENV_REGEX_NS}/swan_2",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.5, 3.5, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
    )

    # -------------------------------------------------------------------
    # Distractor objects — static kinematic props with collision
    # 3× rock, 3× turtle, 3× lilypad, 3× log
    # Positions are randomized at reset (see pond_mdp.reset_pond_scene).
    # -------------------------------------------------------------------

    rock_0 = _make_distractor("rock", 0)
    rock_1 = _make_distractor("rock", 1)
    rock_2 = _make_distractor("rock", 2)
    turtle_0 = _make_distractor("turtle", 0)
    turtle_1 = _make_distractor("turtle", 1)
    turtle_2 = _make_distractor("turtle", 2)
    lilypad_0 = _make_distractor("lilypad", 0)
    lilypad_1 = _make_distractor("lilypad", 1)
    lilypad_2 = _make_distractor("lilypad", 2)
    log_0 = _make_distractor("log", 0)
    log_1 = _make_distractor("log", 1)
    log_2 = _make_distractor("log", 2)

    # Forward-facing camera mounted on gosling (rotated 90°, portrait aspect)
    gosling_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Gosling/gosling_cam",
        update_period=0.0,
        width=480,
        height=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.25, 0.0, 0.25),
            rot=(0.5, -0.5, 0.5, -0.5),  # look +X, -90° roll
            convention="ros",
        ),
    )


# ---------------------------------------------------------------------------
# Actions (paddle commands)
# ---------------------------------------------------------------------------

@configclass
class PondActionsCfg:
    """Action configuration for gosling paddle control.

    Actions are continuous 2D:
      - action[0]: forward paddle force (0.0 to 1.0, maps to 0..max_forward_force)
      - action[1]: yaw turn torque (-1.0 to 1.0, maps to -max_turn_torque..+max_turn_torque)
    """

    pass  # Action processing handled directly in the env step


# ---------------------------------------------------------------------------
# Observations (minimal — for teleop we don't need RL observations)
# ---------------------------------------------------------------------------

@configclass
class PondObservationsCfg:
    """Minimal observation configuration for teleoperation."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_pos_z = ObsTerm(func=mdp.base_pos_z, params={"asset_cfg": SceneEntityCfg("gosling")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("gosling")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("gosling")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@configclass
class PondEventCfg:
    """Randomize bird positions with clearance > 2m, random Z rotations."""

    reset_scene = EventTerm(func=pond_mdp.reset_pond_scene, mode="reset")


# ---------------------------------------------------------------------------
# Rewards (minimal — teleop doesn't need RL rewards)
# ---------------------------------------------------------------------------

@configclass
class PondRewardsCfg:
    """Minimal reward config (required by RL env)."""

    alive_bonus = RewTerm(func=mdp_r.is_alive, weight=0.0)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class PondTerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp_term.time_out, time_out=True)


# ---------------------------------------------------------------------------
# Full environment configuration
# ---------------------------------------------------------------------------

@configclass
class PondEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Pond fluid dynamics environment."""

    scene: PondSceneCfg = PondSceneCfg(
        num_envs=1,
        env_spacing=20.0,
        replicate_physics=True,
    )
    observations: PondObservationsCfg = PondObservationsCfg()
    actions: PondActionsCfg = PondActionsCfg()
    events: PondEventCfg = PondEventCfg()
    rewards: PondRewardsCfg = PondRewardsCfg()
    terminations: PondTerminationsCfg = PondTerminationsCfg()

    # Fluid dynamics parameters
    fluid: FluidDynamicsCfg = FluidDynamicsCfg()

    def __post_init__(self) -> None:
        """Configure simulation and viewer defaults for the pond environment."""
        # Use custom UI window with camera preview
        self.ui_window_class_type = PondEnvWindow

        # Simulation settings
        self.decimation = 2
        self.episode_length_s = 1.0e9  # Essentially infinite for teleop
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -self.fluid.gravity_mag)

        # Rendering
        self.sim.render.rendering_mode = "balanced"
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 4

        # Isometric view: camera 10m from robot, looking at origin
        d = 10.0 * 0.5774  # isometric at 10m distance
        self.viewer.eye = (d, d, d)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # Seed
        self.seed = 42

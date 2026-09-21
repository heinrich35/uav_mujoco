# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""USD paths and articulation config for rover + disk world (no dependency on ``scripts/`` on PYTHONPATH)."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg, UrdfConverter, UrdfConverterCfg
from isaaclab.sim.schemas import schemas_cfg


def isaaclab_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "isaaclab.sh").is_file():
            return parent
    raise RuntimeError("Could not locate Isaac Lab repo root (isaaclab.sh).")


def rover_urdf_path() -> Path:
    return isaaclab_repo_root() / "assets" / "rover_description" / "urdf" / "rover_isaacsim.urdf"


def rover_usd_path() -> Path:
    return isaaclab_repo_root() / "assets" / "rover_description" / "rover.usd"


def disk_world_glb_path() -> Path:
    return isaaclab_repo_root() / "assets" / "disk_world" / "disk_world.glb"


def disk_world_usd_path() -> Path:
    return isaaclab_repo_root() / "assets" / "disk_world" / "disk_world.usd"


WHEEL_JOINT_NAMES = [
    "wheel_fr_joint",
    "wheel_fl_joint",
    "wheel_rr_joint",
    "wheel_rl_joint",
]


def ensure_rover_usd(force_conversion: bool = False) -> str:
    urdf = rover_urdf_path()
    if not urdf.is_file():
        raise FileNotFoundError(f"Rover URDF not found: {urdf}")
    out_dir = rover_usd_path().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    usd = rover_usd_path()
    if not force_conversion and usd.is_file():
        force_conversion = usd.stat().st_mtime < urdf.stat().st_mtime
    # If the USD is already there and at least as new as the URDF, skip UrdfConverter entirely.
    # UrdfConverter always enables ``isaacsim.asset.importer.urdf`` (native plugin); skipping avoids that
    # path when a pre-generated ``assets/rover_description/rover.usd`` is sufficient (e.g. pip Sim / OV ABI issues).
    if not force_conversion and usd.is_file():
        return str(usd.resolve())
    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(out_dir),
        usd_file_name=usd.name,
        force_usd_conversion=force_conversion,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        collision_from_visuals=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=40.0, damping=0.0),
        ),
    )
    return UrdfConverter(cfg).usd_path


def ensure_disk_world_usd(force_conversion: bool = False) -> str:
    glb = disk_world_glb_path()
    if not glb.is_file():
        raise FileNotFoundError(f"disk_world GLB not found: {glb}")
    usd = disk_world_usd_path()
    usd.parent.mkdir(parents=True, exist_ok=True)
    if not force_conversion and usd.exists():
        force_conversion = usd.stat().st_mtime < glb.stat().st_mtime
    cfg = MeshConverterCfg(
        asset_path=str(glb),
        usd_dir=str(usd.parent),
        usd_file_name=usd.name,
        force_usd_conversion=force_conversion,
        make_instanceable=False,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.TriangleMeshPropertiesCfg(),
    )
    return MeshConverter(cfg).usd_path


def ensure_goal_usd(
    mesh_path: str | Path,
    *,
    collision_enabled: bool = False,
    force_conversion: bool = False,
) -> str:
    goal_mesh_path = Path(mesh_path).expanduser().resolve()
    if not goal_mesh_path.is_file():
        raise FileNotFoundError(f"Goal mesh not found: {goal_mesh_path}")
    # Distinct names from older ``*_visual.usd`` exports that omitted RigidBodyAPI (RigidObjectCollection needs it).
    usd_name = (
        f"{goal_mesh_path.stem}_rigid_collision.usd"
        if collision_enabled
        else f"{goal_mesh_path.stem}_rigid_visual.usd"
    )
    goal_usd_path = goal_mesh_path.parent / usd_name
    goal_usd_path.parent.mkdir(parents=True, exist_ok=True)
    if not force_conversion and goal_usd_path.exists():
        force_conversion = goal_usd_path.stat().st_mtime < goal_mesh_path.stat().st_mtime
    # MeshConverter applies RigidBodyAPI on the root only when ``rigid_props`` is set.
    cfg = MeshConverterCfg(
        asset_path=str(goal_mesh_path),
        usd_dir=str(goal_usd_path.parent),
        usd_file_name=goal_usd_path.name,
        force_usd_conversion=force_conversion,
        make_instanceable=False,
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=1.0,
        ),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=0.1),
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)
        if collision_enabled
        else None,
        mesh_collision_props=schemas_cfg.TriangleMeshPropertiesCfg() if collision_enabled else None,
    )
    return MeshConverter(cfg).usd_path


def make_rover_cfg(
    usd_path: str | None = None,
    prim_path: str = "{ENV_REGEX_NS}/Rover",
    *,
    activate_contact_sensors: bool = False,
) -> ArticulationCfg:
    if usd_path is None:
        usd_path = ensure_rover_usd()
    spawn_z = 0.6
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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
            activate_contact_sensors=activate_contact_sensors,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(100.3, 0.0, spawn_z),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=WHEEL_JOINT_NAMES,
                effort_limit_sim=6250.0,
                velocity_limit_sim=150.0,
                stiffness=0.0,
                damping=225.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
            ),
        },
    )


# --- Disk-world obstacle visuals (aligned with ``scripts/rover/launch_rover_stage.py`` spawn pattern) ---

OA_MAX_BLUE_BOXES: int = 80
OA_MAX_GOALS: int = 30
OA_BOX_SIZE: float = 0.5
OA_GOAL_USD_SCALE: tuple[float, float, float] = (0.6, 0.6, 0.6)
# Cylinder: radius 100 m, total width 5 m along world +Y (``launch_rover_stage`` / design doc).
OA_DISK_CYLINDER_RADIUS_M: float = 100.0
OA_DISK_CYLINDER_HALF_WIDTH_Y_M: float = 2.5
OA_DEFAULT_ROVER_SPAWN_LOCAL_XYZ: tuple[float, float, float] = (100.3, 0.0, 0.6)


def default_goal_mesh_path() -> str:
    return str(isaaclab_repo_root() / "assets" / "disk_world" / "goal.glb")


def _oa_kinematic_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        max_depenetration_velocity=1.0,
    )


def quat_cylindrical_tangent_frame(position_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """World (wxyz) for blue box: cylindrical frame about **world +Y** (same as ``launch_rover_stage``).

    ``position_xyz`` should be **relative** to the cylinder axis through the env origin (axis || +Y).
    Body +Z is outward radial in XZ; body +X is clockwise tangent when viewed from +Y.
    """
    x, _y, z = position_xyz
    r = math.hypot(x, z)
    if r < 1.0e-6:
        return (1.0, 0.0, 0.0, 0.0)
    x_hat, z_hat = x / r, z / r
    z_body_w = (x_hat, 0.0, z_hat)
    x_body_w = (z_hat, 0.0, -x_hat)
    y_body_w = (
        z_body_w[1] * x_body_w[2] - z_body_w[2] * x_body_w[1],
        z_body_w[2] * x_body_w[0] - z_body_w[0] * x_body_w[2],
        z_body_w[0] * x_body_w[1] - z_body_w[1] * x_body_w[0],
    )
    rot = torch.tensor(
        [
            [x_body_w[0], y_body_w[0], z_body_w[0]],
            [x_body_w[1], y_body_w[1], z_body_w[1]],
            [x_body_w[2], y_body_w[2], z_body_w[2]],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    q = math_utils.quat_from_matrix(rot)[0]
    return (float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item()))


def quat_goal_with_local_x_roll(position_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Goal USD orientation (``launch_rover_stage.quat_goal_with_local_x_roll``) in the cylindrical frame."""
    q_cyl = torch.tensor(quat_cylindrical_tangent_frame(position_xyz), dtype=torch.float32).unsqueeze(0)
    q_rx = math_utils.quat_from_angle_axis(
        torch.tensor([0.5 * math.pi], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
    )
    q_rz1 = math_utils.quat_from_angle_axis(
        torch.tensor([math.pi], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    q_ry = math_utils.quat_from_angle_axis(
        torch.tensor([math.pi], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32),
    )
    q_rz2 = math_utils.quat_from_angle_axis(
        torch.tensor([math.pi], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    q = math_utils.quat_mul(
        math_utils.quat_mul(math_utils.quat_mul(math_utils.quat_mul(q_cyl, q_rx), q_rz1), q_ry),
        q_rz2,
    )[0]
    return (float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item()))


def quat_cylindrical_tangent_frame_batched(rel: torch.Tensor) -> torch.Tensor:
    """Batched ``quat_cylindrical_tangent_frame``. ``rel`` shape ``(N, 3)``, returns quat ``(N, 4)`` wxyz."""
    x = rel[:, 0]
    z = rel[:, 2]
    r = torch.sqrt(x * x + z * z).clamp(min=1.0e-6)
    xh, zh = x / r, z / r
    x_body_w = torch.stack([zh, torch.zeros_like(x), -xh], dim=-1)
    z_body_w = torch.stack([xh, torch.zeros_like(x), zh], dim=-1)
    y_body_w = torch.linalg.cross(z_body_w, x_body_w, dim=-1)
    rot = torch.stack([x_body_w, y_body_w, z_body_w], dim=-1)
    return math_utils.quat_from_matrix(rot)


def quat_goal_with_local_x_roll_batched(rel: torch.Tensor) -> torch.Tensor:
    """Batched goal orientation; ``rel`` shape ``(N, 3)``."""
    q_cyl = quat_cylindrical_tangent_frame_batched(rel)
    q_rx = math_utils.quat_from_angle_axis(
        torch.full((rel.shape[0],), 0.5 * math.pi, device=rel.device, dtype=rel.dtype),
        torch.tensor([[1.0, 0.0, 0.0]], device=rel.device, dtype=rel.dtype).expand(rel.shape[0], -1),
    )
    q_rz1 = math_utils.quat_from_angle_axis(
        torch.full((rel.shape[0],), math.pi, device=rel.device, dtype=rel.dtype),
        torch.tensor([[0.0, 0.0, 1.0]], device=rel.device, dtype=rel.dtype).expand(rel.shape[0], -1),
    )
    q_ry = math_utils.quat_from_angle_axis(
        torch.full((rel.shape[0],), math.pi, device=rel.device, dtype=rel.dtype),
        torch.tensor([[0.0, 1.0, 0.0]], device=rel.device, dtype=rel.dtype).expand(rel.shape[0], -1),
    )
    q_rz2 = math_utils.quat_from_angle_axis(
        torch.full((rel.shape[0],), math.pi, device=rel.device, dtype=rel.dtype),
        torch.tensor([[0.0, 0.0, 1.0]], device=rel.device, dtype=rel.dtype).expand(rel.shape[0], -1),
    )
    q = math_utils.quat_mul(
        math_utils.quat_mul(math_utils.quat_mul(math_utils.quat_mul(q_cyl, q_rx), q_rz1), q_ry),
        q_rz2,
    )
    return q


def build_oa_obstacle_rigid_objects_dict(
    *,
    max_boxes: int = OA_MAX_BLUE_BOXES,
    max_goals: int = OA_MAX_GOALS,
    goal_mesh_path: str,
    box_collision_enabled: bool,
    goal_collision_enabled: bool,
    box_extent_m: float | None = None,
) -> dict[str, RigidObjectCfg]:
    """Rigid bodies for blue boxes (cuboids) and goals (USD), cloned per env like ``launch_rover_stage``."""
    ext = float(OA_BOX_SIZE if box_extent_m is None else box_extent_m)
    goal_usd = ensure_goal_usd(goal_mesh_path, collision_enabled=goal_collision_enabled)
    hide = (0.0, 0.0, -2000.0)
    out: dict[str, RigidObjectCfg] = {}
    box_collision = (
        sim_utils.CollisionPropertiesCfg(collision_enabled=True) if box_collision_enabled else None
    )
    for i in range(max_boxes):
        box_spawn = sim_utils.CuboidCfg(
            size=(ext, ext, ext),
            collision_props=box_collision,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9), roughness=0.6),
            rigid_props=_oa_kinematic_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        )
        # Parent must be ``{ENV_REGEX_NS}`` (existing env Xform). ``.../Obstacles/...`` breaks @clone:
        # root becomes ``.../env_.*/Obstacles`` which is not spawned.
        out[f"blue_box_{i:03d}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/OA_BlueBox_{i:03d}",
            spawn=box_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=hide, rot=(1.0, 0.0, 0.0, 0.0)),
            collision_group=0,
        )
    goal_spawn = sim_utils.UsdFileCfg(
        usd_path=goal_usd,
        copy_from_source=False,
        scale=OA_GOAL_USD_SCALE,
        rigid_props=_oa_kinematic_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    )
    for i in range(max_goals):
        out[f"goal_{i:03d}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/OA_Goal_{i:03d}",
            spawn=goal_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=hide, rot=(1.0, 0.0, 0.0, 0.0)),
            collision_group=0,
        )
    return out

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Reset-time layout of goals and blue boxes on the disk cylinder; cylindrical gravity on the rover."""

from __future__ import annotations

import math

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv

from isaaclab_tasks.manager_based.rover.rover_disk_assets import (
    OA_MAX_BLUE_BOXES,
    OA_MAX_GOALS,
    quat_cylindrical_tangent_frame_batched,
    quat_goal_with_local_x_roll_batched,
)

if TYPE_CHECKING:
    pass


def apply_rover_cylindrical_gravity(env: ManagerBasedEnv) -> None:
    """Constant-magnitude acceleration toward each env's cylinder axis (world +Y through ``env_origins``).

    Matches ``launch_rover_stage.apply_cylindrical_gravity`` (body-frame forces, ``is_global=False``).
    """
    g = float(getattr(env.cfg, "oa_cylindrical_gravity_m_s2", 9.81))
    if g <= 0.0:
        return
    rover: Articulation = env.scene["rover"]
    root_pos = rover.data.root_pos_w
    origins = env.scene.env_origins[:, :3]
    rel_x = root_pos[:, 0] - origins[:, 0]
    rel_z = root_pos[:, 2] - origins[:, 2]
    radial_dir = torch.stack((-rel_x, torch.zeros_like(rel_x), -rel_z), dim=-1)
    radial_norm = torch.linalg.norm(radial_dir[:, [0, 2]], dim=-1, keepdim=True).clamp(min=1.0e-6)
    accel = g * radial_dir / radial_norm
    body_mass = rover.data.default_mass
    global_forces = body_mass.unsqueeze(-1) * accel.unsqueeze(1)
    body_quat_w = rover.data.body_link_quat_w
    forces = math_utils.quat_apply_inverse(body_quat_w, global_forces)
    torques = torch.zeros_like(forces)
    rover.set_external_force_and_torque(forces=forces, torques=torques, is_global=False)


def _normalize_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids.dtype != torch.long:
        return env_ids.to(dtype=torch.long)
    return env_ids


def _ensure_buffers(
    env: ManagerBasedEnv,
    n_goals: int,
    n_boxes: int,
) -> None:
    u = env.unwrapped
    if not hasattr(u, "_goal_positions_w") or u._goal_positions_w.shape[1] != n_goals:
        u._goal_positions_w = torch.zeros(env.num_envs, n_goals, 3, device=env.device)
    if not hasattr(u, "_box_positions_w") or u._box_positions_w.shape[1] != n_boxes:
        u._box_positions_w = torch.zeros(env.num_envs, n_boxes, 3, device=env.device)
    if not hasattr(u, "_goal_idx"):
        u._goal_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(u, "_goal_bonus_given") or u._goal_bonus_given.shape[1] != n_goals:
        u._goal_bonus_given = torch.zeros(env.num_envs, n_goals, dtype=torch.bool, device=env.device)


def _sync_oa_obstacle_prims(
    env: ManagerBasedEnv,
    eids: torch.Tensor,
    goal_count: int,
    box_count: int,
) -> None:
    """Move spawned USD cuboids / goal meshes to match ``_goal_positions_w`` / ``_box_positions_w``."""
    try:
        assets = env.scene["oa_obstacles"]
    except KeyError:
        return
    u = env.unwrapped
    device = env.device
    ne = int(eids.shape[0])
    n_obj = int(assets.num_objects)
    if ne == 0 or n_obj < 1:
        return
    if box_count > OA_MAX_BLUE_BOXES or goal_count > OA_MAX_GOALS:
        raise ValueError(
            f"box_count={box_count} / goal_count={goal_count} exceed OA_MAX_BLUE_BOXES={OA_MAX_BLUE_BOXES} "
            f"/ OA_MAX_GOALS={OA_MAX_GOALS}; increase caps in rover_disk_assets.py."
        )
    poses = torch.zeros(ne, n_obj, 7, device=device)
    poses[..., 3] = 1.0
    poses[..., 2] = -2000.0
    vel = torch.zeros(ne, n_obj, 6, device=device)

    origins = env.scene.env_origins[eids, :3]
    boxes = u._box_positions_w[eids, :box_count]
    goals = u._goal_positions_w[eids, :goal_count]

    if box_count > 0:
        rel_b = boxes - origins.unsqueeze(1)
        qb = quat_cylindrical_tangent_frame_batched(rel_b.reshape(-1, 3)).reshape(ne, box_count, 4)
        poses[:, :box_count, 0:3] = boxes
        poses[:, :box_count, 3:7] = qb
    if goal_count > 0:
        rel_g = goals - origins.unsqueeze(1)
        qg = quat_goal_with_local_x_roll_batched(rel_g.reshape(-1, 3)).reshape(ne, goal_count, 4)
        poses[:, box_count : box_count + goal_count, 0:3] = goals
        poses[:, box_count : box_count + goal_count, 3:7] = qg

    assets.write_object_pose_to_sim(poses, env_ids=eids)
    assets.write_object_velocity_to_sim(vel, env_ids=eids)


def _clear_reward_state(env: ManagerBasedEnv, eids: torch.Tensor) -> None:
    u = env.unwrapped
    for name in (
        "_prev_theta_for_reward",
        "_prev_theta_prog",
        "_cumulative_progress_rad",
        "_prev_drive_action",
        "_prev_root_vel_jerk",
        "_prev_root_dv",
        "_wheel_odom",
        "_prev_lin_vel_imu",
    ):
        buf = getattr(u, name, None)
        if buf is None:
            continue
        if buf.ndim == 1:
            buf[eids] = 0.0
        elif buf.ndim == 2:
            buf[eids] = 0.0
        elif buf.ndim == 3:
            buf[eids] = 0.0


def randomize_disk_ring_obstacles(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    goal_count: int,
    box_count: int,
    goal_radial_distance: float,
    min_object_spacing_m: float,
    box_radial_distance_m: float = 100.0,
    cylinder_y_half_width_m: float = 2.5,
    rover_spawn_exclusion_radius_m: float = 7.5,
    exclude_spawn_local_xyz: tuple[float, float, float] = (100.3, 0.0, 0.6),
    max_sample_attempts: int = 512,
    **kwargs,
) -> None:
    """Sample boxes then goals on ``|x|^2+|z|^2=R^2`` (axis || world +Y), ``y`` uniform in band (launch script).

    Each angle is **uniform** on ``[0, 2π)`` (no sorting onto an arc). ``min_object_spacing_m`` is enforced
    sequentially like ``launch_rover_stage.sample_non_overlapping_positions`` (boxes, then goals).
    """
    _ = kwargs.pop("height_above_origin_z", None)  # legacy key from older cfgs
    eids = _normalize_env_ids(env, env_ids)
    if eids.numel() == 0:
        return
    _ensure_buffers(env, goal_count, box_count)
    u = env.unwrapped
    origins = env.scene.env_origins[eids, :3]
    n = int(eids.shape[0])
    device = env.device
    dtype = origins.dtype
    y_lo = float(-abs(cylinder_y_half_width_m))
    y_hi = float(abs(cylinder_y_half_width_m))
    ex = torch.tensor(exclude_spawn_local_xyz, device=device, dtype=dtype).view(1, 3).expand(n, 3)
    exclude_world = origins + ex
    min_d = float(min_object_spacing_m)
    excl_r = float(rover_spawn_exclusion_radius_m)

    # --- boxes first (same order as ``design_scene`` obstacle pass) ---
    placed_boxes = torch.zeros(n, 0, 3, device=device, dtype=dtype)
    for _bi in range(box_count):
        newcol = torch.zeros(n, 1, 3, device=device, dtype=dtype)
        for i in range(n):
            ok = False
            for _ in range(max_sample_attempts):
                ang = float(torch.rand(1, device=device).item()) * (2.0 * math.pi)
                y = y_lo + float(torch.rand(1, device=device).item()) * (y_hi - y_lo)
                lx = float(box_radial_distance_m) * math.cos(ang)
                lz = float(box_radial_distance_m) * math.sin(ang)
                p = torch.tensor(
                    [origins[i, 0] + lx, origins[i, 1] + y, origins[i, 2] + lz],
                    device=device,
                    dtype=dtype,
                )
                if torch.sum((p - exclude_world[i]) ** 2).item() < excl_r * excl_r:
                    continue
                if placed_boxes.shape[1] > 0:
                    d2 = torch.sum((placed_boxes[i, :, :] - p.view(1, 3)) ** 2, dim=-1)
                    if torch.any(d2 < min_d * min_d):
                        continue
                newcol[i, 0, :] = p
                ok = True
                break
            if not ok:
                ang = float(torch.rand(1, device=device).item()) * (2.0 * math.pi)
                y = y_lo + float(torch.rand(1, device=device).item()) * (y_hi - y_lo)
                newcol[i, 0, 0] = origins[i, 0] + float(box_radial_distance_m) * math.cos(ang)
                newcol[i, 0, 1] = origins[i, 1] + y
                newcol[i, 0, 2] = origins[i, 2] + float(box_radial_distance_m) * math.sin(ang)
        placed_boxes = torch.cat([placed_boxes, newcol], dim=1)

    u._box_positions_w[eids, :box_count] = placed_boxes

    # --- goals (respect spacing to boxes + prior goals) ---
    placed_goals = torch.zeros(n, 0, 3, device=device, dtype=dtype)
    for _gi in range(goal_count):
        newcol = torch.zeros(n, 1, 3, device=device, dtype=dtype)
        for i in range(n):
            ok = False
            for _ in range(max_sample_attempts):
                ang = float(torch.rand(1, device=device).item()) * (2.0 * math.pi)
                y = y_lo + float(torch.rand(1, device=device).item()) * (y_hi - y_lo)
                lx = float(goal_radial_distance) * math.cos(ang)
                lz = float(goal_radial_distance) * math.sin(ang)
                p = torch.tensor(
                    [origins[i, 0] + lx, origins[i, 1] + y, origins[i, 2] + lz],
                    device=device,
                    dtype=dtype,
                )
                if torch.sum((p - exclude_world[i]) ** 2).item() < excl_r * excl_r:
                    continue
                if placed_boxes.shape[1] > 0:
                    d2b = torch.sum((placed_boxes[i, :, :] - p.view(1, 3)) ** 2, dim=-1)
                    if torch.any(d2b < min_d * min_d):
                        continue
                if placed_goals.shape[1] > 0:
                    d2g = torch.sum((placed_goals[i, :, :] - p.view(1, 3)) ** 2, dim=-1)
                    if torch.any(d2g < min_d * min_d):
                        continue
                newcol[i, 0, :] = p
                ok = True
                break
            if not ok:
                ang = float(torch.rand(1, device=device).item()) * (2.0 * math.pi)
                y = y_lo + float(torch.rand(1, device=device).item()) * (y_hi - y_lo)
                newcol[i, 0, 0] = origins[i, 0] + float(goal_radial_distance) * math.cos(ang)
                newcol[i, 0, 1] = origins[i, 1] + y
                newcol[i, 0, 2] = origins[i, 2] + float(goal_radial_distance) * math.sin(ang)
        placed_goals = torch.cat([placed_goals, newcol], dim=1)

    u._goal_positions_w[eids, :goal_count] = placed_goals
    u._goal_idx[eids] = 0
    u._goal_bonus_given[eids, :] = False

    y_nom = origins[:, 1] + 0.5 * (y_lo + y_hi)
    if not hasattr(u, "_y_nominal") or u._y_nominal.shape[0] != env.num_envs:
        u._y_nominal = torch.zeros(env.num_envs, device=device, dtype=dtype)
    u._y_nominal[eids] = y_nom

    _clear_reward_state(env, eids)
    _sync_oa_obstacle_prims(env, eids, goal_count, box_count)

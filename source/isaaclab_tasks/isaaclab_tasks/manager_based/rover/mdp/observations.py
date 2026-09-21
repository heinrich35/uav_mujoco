# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Proprioceptive / kinematic observations."""

from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _asset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def wheel_velocities_normalized(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    ids, _ = asset.find_joints(
        ["wheel_fr_joint", "wheel_fl_joint", "wheel_rr_joint", "wheel_rl_joint"], preserve_order=True
    )
    wv = asset.data.joint_vel[:, ids]
    return torch.tanh(wv * 0.02)


def base_ang_vel_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Angular rate about world +Y (cylinder axis)."""
    asset = _asset(env, asset_cfg)
    return asset.data.root_ang_vel_w[:, 1:2]


def base_lin_vel_cylindrical(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Radial / tangential / axial components for cylinder axis **world +Y** (XZ radial, ``v_y`` axial)."""
    asset = _asset(env, asset_cfg)
    v = asset.data.root_lin_vel_w[:, :3]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    tang = torch.stack([pos[:, 2], torch.zeros_like(pos[:, 0]), -pos[:, 0]], dim=-1)
    tang = torch.nn.functional.normalize(tang + 1.0e-8, dim=-1)
    radial = torch.stack([pos[:, 0], torch.zeros_like(pos[:, 0]), pos[:, 2]], dim=-1)
    radial = torch.nn.functional.normalize(radial + 1.0e-8, dim=-1)
    vt = torch.sum(v * tang, dim=-1, keepdim=True)
    vr = torch.sum(v * radial, dim=-1, keepdim=True)
    vy = v[:, 1:2]
    return torch.cat([vr, vt, vy], dim=-1)


def height_offset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Offset along cylinder axis (world +Y) from nominal mid-band ``_y_nominal``."""
    asset = _asset(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    y0 = getattr(env.unwrapped, "_y_nominal", pos[:, 1:2].detach())
    return (pos[:, 1:2] - y0) * 0.1


def body_tilt_quality(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Cosine between body +Z and outward radial in XZ (upright on the cylindrical rim)."""
    asset = _asset(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    rad = torch.stack([pos[:, 0], torch.zeros_like(pos[:, 0]), pos[:, 2]], dim=-1)
    rad = torch.nn.functional.normalize(rad + 1.0e-8, dim=-1)
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    z_w = torch.stack(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dim=-1,
    )
    c = torch.sum(z_w * rad, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    return c


def theta_from_spawn(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    th = torch.atan2(pos[:, 2:3], pos[:, 0:1])
    return th / 3.14159265


def cumulative_angular_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    v = getattr(env.unwrapped, "_cumulative_progress_rad", torch.zeros(env.num_envs, device=env.device))
    return v.unsqueeze(-1) * 0.01


def radial_distance_normalized(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    r = torch.linalg.norm(torch.stack([pos[:, 0], pos[:, 2]], dim=-1), dim=-1, keepdim=True)
    r0 = getattr(env.cfg, "goal_radial_distance", 100.3)
    return (r / (r0 + 1.0e-6)) - 1.0


def heading_tangent_cos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Heading vs clockwise tangent in the world XZ plane (cylinder about +Y)."""
    asset = _asset(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    tang_w = torch.stack([pos[:, 2], torch.zeros_like(pos[:, 0]), -pos[:, 0]], dim=-1)
    tang_w = torch.nn.functional.normalize(tang_w + 1.0e-8, dim=-1)
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    fxz = torch.stack([fx, fz], dim=-1)
    fxz = torch.nn.functional.normalize(fxz + 1.0e-8, dim=-1)
    txz = torch.nn.functional.normalize(torch.stack([tang_w[:, 0], tang_w[:, 2]], dim=-1) + 1.0e-8, dim=-1)
    c = torch.sum(fxz * txz, dim=-1, keepdim=True)
    s = (fxz[:, 0:1] * txz[:, 1:2] - fxz[:, 1:2] * txz[:, 0:1])
    return torch.cat([c, s], dim=-1)


def last_action(env: ManagerBasedRLEnv, term_name: str = "drive") -> torch.Tensor:
    return env.action_manager.get_term(term_name).raw_actions


def last_drive_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.get_term("drive").processed_actions

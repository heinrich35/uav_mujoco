# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Obstacle-avoidance reward terms."""

from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _rover(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def forward_circumference_progress_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")
) -> torch.Tensor:
    """Clockwise arc length increment (m) approximated as R * |Δθ| with R from cfg (XZ cylinder)."""
    asset = _rover(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    r = torch.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2 + 1.0e-12).clamp(min=1.0)
    theta = torch.atan2(pos[:, 2], pos[:, 0])
    if not hasattr(env.unwrapped, "_prev_theta_prog"):
        env.unwrapped._prev_theta_prog = theta.clone()
    d = theta - env.unwrapped._prev_theta_prog
    d = torch.atan2(torch.sin(d), torch.cos(d))
    env.unwrapped._prev_theta_prog = theta.clone()
    # accumulate for logging / optional terminations
    if not hasattr(env.unwrapped, "_cumulative_progress_rad"):
        env.unwrapped._cumulative_progress_rad = torch.zeros(env.num_envs, device=env.device)
    env.unwrapped._cumulative_progress_rad += d.abs()
    return r * (-d)  # positive reward for clockwise motion


def blue_box_avoidance_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"), danger_radius_m: float = 1.5
) -> torch.Tensor:
    """Gaussian penalty from nearest blue box (negative reward contribution via negative weight in cfg)."""
    asset = _rover(env, asset_cfg)
    root = asset.data.root_pos_w[:, :3]
    boxes = getattr(env.unwrapped, "_box_positions_w", None)
    if boxes is None:
        return torch.zeros(env.num_envs, device=env.device)
    d = torch.linalg.norm(root[:, None, :3] - boxes, dim=-1)
    dmin, _ = d.min(dim=1)
    sigma = danger_radius_m * 0.5 + 1.0e-6
    return torch.exp(-0.5 * torch.square(dmin / sigma))


def next_goal_pass_through_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")
) -> torch.Tensor:
    """Sparse bonus when entering goal threshold (latched per goal index)."""
    _ = asset_cfg
    thr = float(getattr(env.cfg, "goal_pass_threshold_m", 2.0))
    goals = getattr(env.unwrapped, "_goal_positions_w", None)
    idx = getattr(env.unwrapped, "_goal_idx", None)
    given = getattr(env.unwrapped, "_goal_bonus_given", None)
    if goals is None or idx is None or given is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["rover"]
    root = asset.data.root_pos_w[:, :3]
    n_g = goals.shape[1]
    gi = idx.long().clamp(0, n_g - 1)
    g = goals[torch.arange(env.num_envs, device=env.device), gi]
    dist = torch.linalg.norm(root - g, dim=-1)
    ar = torch.arange(env.num_envs, device=env.device)
    hit = (dist < thr) & (~given[ar, gi])
    out = hit.float()
    given[ar, gi] = True
    new_idx = gi + hit.long()
    env.unwrapped._goal_idx = torch.clamp(new_idx, max=n_g - 1)
    return out


def disk_tangent_heading_alignment_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"), tolerance_rad: float = 1.1
) -> torch.Tensor:
    """Reward heading (forward x) alignment with clockwise tangent in world XZ (cylinder about +Y)."""
    asset = _rover(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    tang = torch.stack([pos[:, 2], torch.zeros_like(pos[:, 0]), -pos[:, 0]], dim=-1)
    tang = torch.nn.functional.normalize(tang + 1.0e-8, dim=-1)
    # forward vector from quaternion (x axis body in world)
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    f = torch.stack([fx, fy, fz], dim=-1)
    f = torch.nn.functional.normalize(f + 1.0e-8, dim=-1)
    c = torch.sum(f * tang, dim=-1).clamp(-1.0, 1.0)
    ang = torch.acos(c)
    return ((ang < tolerance_rad).float()) * torch.exp(-ang)


def near_fall_radial_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    fall_radius_m: float = 99.5,
    danger_band_m: float = 1.15,
) -> torch.Tensor:
    """Penalty when radial distance is just outside ``fall_radius_m`` (danger strip)."""
    asset = _rover(env, asset_cfg)
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    r = torch.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2 + 1.0e-12)
    edge = r - (fall_radius_m - danger_band_m)
    active = torch.clamp(edge, min=0.0) / (danger_band_m + 1.0e-6)
    return torch.square(active)


def visual_blue_box_ahead_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    bearing_gate_rad: float = 0.4,
    range_gate_m: float = 6.0,
    range_norm_m: float = 30.0,
) -> torch.Tensor:
    """Proxy visual penalty using geometry (bearing/range to nearest box)."""
    _ = bearing_gate_rad
    asset = _rover(env, asset_cfg)
    root = asset.data.root_pos_w[:, :3]
    boxes = getattr(env.unwrapped, "_box_positions_w", None)
    if boxes is None:
        return torch.zeros(env.num_envs, device=env.device)
    rel = boxes - root[:, None, :]
    dist = torch.linalg.norm(rel, dim=-1)
    dmin, j = dist.min(dim=1)
    relm = rel[torch.arange(env.num_envs, device=env.device), j]
    bearing = torch.atan2(relm[:, 2], relm[:, 0]).abs()
    gate = (bearing < bearing_gate_rad).float() * torch.exp(-dmin / range_gate_m)
    return gate * torch.exp(-dmin / range_norm_m)


def visual_goal_ahead_alignment_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    bearing_gate_rad: float = 0.4,
    range_gate_m: float = 12.0,
    range_norm_m: float = 30.0,
) -> torch.Tensor:
    """Proxy reward for facing current goal (bearing in XZ; heading projected to XZ)."""
    asset = _rover(env, asset_cfg)
    root = asset.data.root_pos_w[:, :3]
    goals = getattr(env.unwrapped, "_goal_positions_w", None)
    idx = getattr(env.unwrapped, "_goal_idx", None)
    if goals is None or idx is None:
        return torch.zeros(env.num_envs, device=env.device)
    gi = idx.long().clamp(0, goals.shape[1] - 1)
    g = goals[torch.arange(env.num_envs, device=env.device), gi]
    rel = g - root
    dist = torch.linalg.norm(rel, dim=-1)
    bearing = torch.atan2(rel[:, 2], rel[:, 0]).abs()
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    f_xz = torch.nn.functional.normalize(torch.stack([fx, fz], dim=-1) + 1.0e-8, dim=-1)
    rel_xz = torch.nn.functional.normalize(torch.stack([rel[:, 0], rel[:, 2]], dim=-1) + 1.0e-8, dim=-1)
    c = torch.sum(f_xz * rel_xz, dim=-1).clamp(-1.0, 1.0)
    align = torch.acos(c)
    return ((bearing < bearing_gate_rad).float()) * torch.exp(-dist / range_gate_m) * torch.exp(-align) * torch.exp(
        -dist / range_norm_m
    )


def next_goal_approach_shaping(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"), sigma_m: float = 5.0
) -> torch.Tensor:
    """Shaped distance to current goal."""
    asset = _rover(env, asset_cfg)
    root = asset.data.root_pos_w[:, :3]
    goals = getattr(env.unwrapped, "_goal_positions_w", None)
    idx = getattr(env.unwrapped, "_goal_idx", None)
    if goals is None or idx is None:
        return torch.zeros(env.num_envs, device=env.device)
    gi = idx.long().clamp(0, goals.shape[1] - 1)
    g = goals[torch.arange(env.num_envs, device=env.device), gi]
    dist = torch.linalg.norm(root - g, dim=-1)
    return torch.exp(-torch.square(dist / (sigma_m + 1.0e-6)))


def teacher_action_imitation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    decay_env_steps: int = 20000,
    match_sigma: float = 0.25,
) -> torch.Tensor:
    """Optional teacher matching (disabled when teacher weight is 0)."""
    _ = asset_cfg, decay_env_steps, match_sigma
    return torch.zeros(env.num_envs, device=env.device)

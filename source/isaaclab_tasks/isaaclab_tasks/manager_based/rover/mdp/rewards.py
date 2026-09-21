# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared rover disk-world reward helpers."""

from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def forward_angular_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Clockwise angular progress using ``atan2(z, x)`` (cylinder about world +Y)."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    theta = torch.atan2(pos[:, 2], pos[:, 0])
    if not hasattr(env.unwrapped, "_prev_theta_for_reward"):
        env.unwrapped._prev_theta_for_reward = theta.clone()
    d = theta - env.unwrapped._prev_theta_for_reward
    d = torch.atan2(torch.sin(d), torch.cos(d))
    env.unwrapped._prev_theta_for_reward = theta.clone()
    if not hasattr(env.unwrapped, "_cumulative_progress_rad"):
        env.unwrapped._cumulative_progress_rad = torch.zeros(env.num_envs, device=env.device)
    env.unwrapped._cumulative_progress_rad += d.abs()
    # positive reward for clockwise motion when viewed from +Y
    return -d


def forward_speed_tracking(
    env: ManagerBasedRLEnv, target_tangential_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")
) -> torch.Tensor:
    """Penalty for mismatch to target tangential speed (m/s) in the world XZ plane (about +Y)."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    tang = torch.stack([pos[:, 2], torch.zeros_like(pos[:, 0]), -pos[:, 0]], dim=-1)
    tang = torch.nn.functional.normalize(tang + 1.0e-8, dim=-1)
    v = asset.data.root_lin_vel_w[:, :3]
    v_t = torch.sum(v * tang, dim=-1)
    return -torch.square(v_t - target_tangential_speed)


def radial_z_alignment_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Reward staying near nominal **Y** (cylinder axis / disk mid-plane) from reset."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    y0 = getattr(env.unwrapped, "_y_nominal", None)
    if y0 is None:
        y0 = pos[:, 1].detach()
    return torch.exp(-torch.abs(pos[:, 1] - y0))


def radial_z_alignment_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    max_abs_z_error_before_clip_m: float = 2.5,
) -> torch.Tensor:
    """Penalty for vertical offset from nominal height (squared error, clipped before squaring).

    Without a clip, long episodes on a curved mesh or unstable vertical motion can produce
    enormous squared errors and blow up the value function.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    y0 = getattr(env.unwrapped, "_y_nominal", None)
    if y0 is None:
        y0 = pos[:, 1].detach()
    dy = torch.clamp(pos[:, 1] - y0, -max_abs_z_error_before_clip_m, max_abs_z_error_before_clip_m)
    return torch.square(dy)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 squared action delta on drive command."""
    term = env.action_manager.get_term("drive")
    a = term.raw_actions
    if not hasattr(env.unwrapped, "_prev_drive_action"):
        env.unwrapped._prev_drive_action = torch.zeros_like(a)
    d = a - env.unwrapped._prev_drive_action
    env.unwrapped._prev_drive_action = a.clone()
    return torch.sum(torch.square(d), dim=-1)


def alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

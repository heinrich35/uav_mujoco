# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Smoothness rewards shared with physparam / obstacle tasks."""

from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def motion_jerk_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Finite-difference jerk proxy on root linear velocity (second difference)."""
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_w[:, :3]
    if not hasattr(env.unwrapped, "_prev_root_vel_jerk"):
        env.unwrapped._prev_root_vel_jerk = v.clone()
        env.unwrapped._prev_root_dv = torch.zeros_like(v)
    dv = v - env.unwrapped._prev_root_vel_jerk
    j = dv - env.unwrapped._prev_root_dv
    env.unwrapped._prev_root_vel_jerk = v.clone()
    env.unwrapped._prev_root_dv = dv.clone()
    return torch.sum(torch.square(j), dim=-1)


def heading_oscillation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Penalty on yaw-rate magnitude (reduce shimmy)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_w[:, 2]
    return torch.square(wz)


def heading_stability_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.abs(asset.data.root_ang_vel_w[:, 2]))


def motion_jerk_smoothness_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    return -motion_jerk_penalty(env, asset_cfg)

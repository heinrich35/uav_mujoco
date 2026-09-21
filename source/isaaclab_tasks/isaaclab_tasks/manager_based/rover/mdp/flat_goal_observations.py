# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Observations for flat XY goal navigation."""

from __future__ import annotations

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def goal_vector_xy_scaled(
    env: ManagerBasedRLEnv,
    scale_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    u = env.unwrapped
    if not hasattr(u, "_flat_goal_target_xy"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    g = u._flat_goal_target_xy - rel
    s = max(float(scale_m), 1.0e-6)
    return (g / s).clamp(-1.0, 1.0)


def distance_to_goal_normalized(
    env: ManagerBasedRLEnv,
    scale_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    u = env.unwrapped
    if not hasattr(u, "_flat_goal_target_xy"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    rel = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d = torch.linalg.norm(u._flat_goal_target_xy - rel, dim=-1, keepdim=True)
    return d / max(float(scale_m), 1.0e-6)


def base_lin_vel_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, :2]


def base_ang_vel_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2:3]

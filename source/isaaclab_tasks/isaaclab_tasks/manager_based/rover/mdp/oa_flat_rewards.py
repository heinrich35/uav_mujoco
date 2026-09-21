# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Flat OA rewards: GT box positions for penalties only (not in policy observations)."""

from __future__ import annotations

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _rover(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def oa_flat_blue_box_avoidance_penalty(
    env: ManagerBasedRLEnv,
    danger_radius_m: float,
    near_ramp_weight: float = 3.0,
    gaussian_sigma_scale: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """OA-flat soft repulsion vs nearest blue box (GT centers): Gaussian + squared ramp inside ``danger_radius_m``.

    The ramp adds gradient when the policy would otherwise skim just outside contact; combine with negative weight
    in cfg. Uses ``_oa_flat_box_positions_w`` / ``_box_positions_w`` (same layout as collision term).
    """
    asset = _rover(env, asset_cfg)
    root = asset.data.root_pos_w[:, :3]
    u = env.unwrapped
    boxes = getattr(u, "_oa_flat_box_positions_w", None)
    if boxes is None:
        boxes = getattr(u, "_box_positions_w", None)
    if boxes is None:
        return torch.zeros(env.num_envs, device=env.device)
    d = torch.linalg.norm(root[:, None, :3] - boxes, dim=-1)
    dmin, _ = d.min(dim=1)
    dr = max(float(danger_radius_m), 1.0e-6)
    sigma = dr * max(float(gaussian_sigma_scale), 0.05) + 1.0e-6
    g = torch.exp(-0.5 * torch.square(dmin / sigma))
    inv = torch.clamp(dmin / (dr + 1.0e-6), max=1.0)
    ramp = torch.square(1.0 - inv)
    w = max(float(near_ramp_weight), 0.0)
    return g + w * ramp


def oa_flat_blue_box_collision_penalty(
    env: ManagerBasedRLEnv,
    collision_radius_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """Positive magnitude inside ``collision_radius_m`` (planar) to nearest blue box center (weight < 0 in cfg).

    Stronger than plain ``inside^2`` near contact so cutting through boxes dominates short path rewards.
    """
    asset = _rover(env, asset_cfg)
    root = asset.data.root_pos_w[:, :3]
    boxes = getattr(env.unwrapped, "_oa_flat_box_positions_w", None)
    if boxes is None:
        return torch.zeros(env.num_envs, device=env.device)
    d = torch.linalg.norm(root[:, None, :2] - boxes[:, :, :2], dim=-1)
    dmin, _ = d.min(dim=1)
    r = float(collision_radius_m)
    inside = torch.clamp(r - dmin, min=0.0) / (r + 1.0e-6)
    return inside * inside * (1.0 + 2.5 * inside)

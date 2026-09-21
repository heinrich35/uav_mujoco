# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Disk-world termination functions."""

from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def fell_off_disk(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    min_radius: float = 99.5,
) -> torch.Tensor:
    """Terminate when cylindrical radius in world XZ drops below ``min_radius`` (m)."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    r_xz = torch.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2 + 1.0e-12)
    return r_xz < min_radius


def upside_down_disk_world(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    min_radial_dot: float = 0.13,
) -> torch.Tensor:
    """Terminate when body +Z tips away from outward radial in XZ (cylinder about world +Y).

    ``min_radial_dot`` is the minimum acceptable :math:`\\hat{z}_\\text{body} \\cdot \\hat{r}_\\text{xz}`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    rad = torch.stack([pos[:, 0], torch.zeros_like(pos[:, 0]), pos[:, 2]], dim=-1)
    rad = torch.nn.functional.normalize(rad + 1.0e-8, dim=-1)
    qw, qx, qy, qz = asset.data.root_quat_w[:, 0], asset.data.root_quat_w[:, 1], asset.data.root_quat_w[:, 2], asset.data.root_quat_w[:, 3]
    z_w = torch.stack(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dim=-1,
    )
    align = torch.sum(z_w * rad, dim=-1)
    return align < min_radial_dot


def root_z_offset_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    max_abs_offset_m: float = 8.0,
) -> torch.Tensor:
    """Terminate when |root z − nominal z| is too large (sink / launch).

    Optional for custom envs; default obstacle-avoidance cfg relies on clipped ``radial_z_alignment_penalty``
    instead, because a flat ``_z_nominal`` does not match a curved shared disk mesh for all trajectories.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    y0 = getattr(env.unwrapped, "_y_nominal", None)
    if y0 is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    dy = torch.abs(pos[:, 1] - y0)
    return dy > max_abs_offset_m


def forward_progress_goal_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    progress_threshold_rad: float = 200.0,
) -> torch.Tensor:
    """Optional termination when cumulative angular progress exceeds threshold (other tasks)."""
    _ = asset_cfg
    if not hasattr(env.unwrapped, "_cumulative_progress_rad"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.unwrapped._cumulative_progress_rad.abs() > progress_threshold_rad

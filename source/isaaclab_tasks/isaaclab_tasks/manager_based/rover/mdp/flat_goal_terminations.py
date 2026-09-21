# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Terminations for flat XY goal navigation."""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.rover.mdp.flat_goal_rewards as fg_r


def never_terminate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def flat_goal_reached(
    env: ManagerBasedRLEnv,
    threshold_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    u = env.unwrapped
    if not hasattr(u, "_flat_goal_target_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    rel = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    d = torch.linalg.norm(u._flat_goal_target_xy - rel, dim=-1)
    return d < float(threshold_m)


def upside_down_flat(
    env: ManagerBasedRLEnv,
    min_up_dot: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """Terminate when body +Z · world +Z falls below ``min_up_dot`` (cos of tilt from upright)."""
    asset: Articulation = env.scene[asset_cfg.name]
    qw, qx, qy, qz = asset.data.root_quat_w[:, 0], asset.data.root_quat_w[:, 1], asset.data.root_quat_w[:, 2], asset.data.root_quat_w[:, 3]
    z_w = torch.stack(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dim=-1,
    )
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(z_w)
    c = torch.sum(z_w * up, dim=-1)
    return c < float(min_up_dot)


def wheelie_pitch_termination(
    env: ManagerBasedRLEnv,
    min_pitch_sin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """Terminate on large :math:`|\\hat{x}_\\text{body} \\cdot \\hat{z}_\\text{world}|` (wheelie / nose-up pose)."""
    if float(min_pitch_sin) < -0.5:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    ex_b = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=q.dtype).unsqueeze(0).expand(q.shape[0], -1)
    x_w = math_utils.quat_apply(q, ex_b)
    pitch_gate = torch.abs(x_w[:, 2])
    bad_pose = pitch_gate > float(min_pitch_sin)
    thr = float(getattr(env.cfg, "flat_wheel_ground_contact_threshold_n", 1.0))
    c = fg_r.wheel_contact_proxy_mask(env, thr, asset_cfg)
    front_air = ~((c[:, 0]) & (c[:, 1]))
    return front_air & bad_pose

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Localization reward functions."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def position_estimation_reward(
    env: ManagerBasedRLEnv,
    position_error_scale_m: float = 50.0,
    position_error_exp_scale: float = 0.1,
    position_error_linear_weight: float = 0.5,
    position_error_exp_weight: float = 0.5,
) -> torch.Tensor:
    """Reward for accurate 3-D position estimation.

    Combines a linear penalty (encourages steady improvement) and an exponential
    penalty (heavily rewards very accurate guesses). Both are negative so the
    RL agent maximizes the sum toward zero.

    Args:
        env: The environment instance.
        position_error_scale_m: Scale divisor for linear error term (m).
        position_error_exp_scale: Inverse scale for exponential error term (1/m).
        position_error_linear_weight: Weight of the linear penalty component.
        position_error_exp_weight: Weight of the exponential penalty component.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    rover = env.scene["rover"]
    # Use env-local position so the reward is consistent across parallel envs
    # (the policy only sees local observations, so it naturally outputs local coords).
    true_pos = rover.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]

    # Agent's estimate (stored by RoverPositionEstimationAction)
    est = getattr(env.unwrapped, "_rover_estimated_position", None)
    if est is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 3-D position error
    pos_err = torch.linalg.norm(true_pos - est[:, :3], dim=-1)

    # Linear component: smooth gradient everywhere
    linear_penalty = -(pos_err / max(float(position_error_scale_m), 1e-6))

    # Exponential component: strong shaping near the true position
    exp_penalty = -torch.expm1(-pos_err * max(float(position_error_exp_scale), 1e-6))

    reward = (
        float(position_error_linear_weight) * linear_penalty
        + float(position_error_exp_weight) * exp_penalty
    )

    return reward


def position_estimation_bonus(
    env: ManagerBasedRLEnv,
    threshold_m: float = 1.0,
    bonus_weight: float = 1.0,
) -> torch.Tensor:
    """Sparse bonus when the estimate is very close to the true position.

    Args:
        env: The environment instance.
        threshold_m: Distance threshold (m) under which the bonus is awarded.
        bonus_weight: Scalar weight for the bonus.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    rover = env.scene["rover"]
    # Use env-local position so the bonus is consistent across parallel envs.
    true_pos = rover.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    est = getattr(env.unwrapped, "_rover_estimated_position", None)
    if est is None:
        return torch.zeros(env.num_envs, device=env.device)

    pos_err = torch.linalg.norm(true_pos - est[:, :3], dim=-1)
    return float(bonus_weight) * (pos_err < float(threshold_m)).float()

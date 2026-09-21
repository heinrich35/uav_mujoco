# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rewards for flat XY goal navigation (differential-drive rover)."""

from __future__ import annotations

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.rover.mdp.rewards as rrew
from isaaclab_tasks.manager_based.rover.mdp.flat_goal_events import materialize_planar_goal_dist_buffer


def _rover_rel_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def _target_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    u = env.unwrapped
    if not hasattr(u, "_flat_goal_target_xy"):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return u._flat_goal_target_xy


def _to_goal_vec(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _target_xy(env) - _rover_rel_xy(env, asset_cfg)


def _planar_dist(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    v = _to_goal_vec(env, asset_cfg)
    return torch.linalg.norm(v, dim=-1)


def distance_to_target_reward(
    env: ManagerBasedRLEnv,
    scale_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    d = _planar_dist(env, asset_cfg)
    return -(d / max(float(scale_m), 1.0e-6))


def distance_progress_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Uses ``_flat_goal_dist_snapshot`` set in :class:`RoverFlatWorldGoalNavRLEnv` before reward aggregation."""
    u = env.unwrapped
    d = _planar_dist(env, asset_cfg)
    # Play runs ``env.step`` in ``torch.inference_mode()``; whole-buffer ``detach().clone()`` stays an inference
    # tensor. :func:`materialize_planar_goal_dist_buffer` keeps buffers slice-assignable and avoids spurious clears.
    if not hasattr(u, "_flat_goal_dist_snapshot"):
        u._flat_goal_prev_dist = materialize_planar_goal_dist_buffer(d)
        return torch.zeros(env.num_envs, device=env.device)
    old = u._flat_goal_dist_snapshot
    prog = torch.clamp(old - d, min=0.0)
    u._flat_goal_prev_dist = materialize_planar_goal_dist_buffer(d)
    return prog


def distance_increase_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    u = env.unwrapped
    d = _planar_dist(env, asset_cfg)
    if not hasattr(u, "_flat_goal_dist_snapshot"):
        return torch.zeros(env.num_envs, device=env.device)
    old = u._flat_goal_dist_snapshot
    # Ignore tiny step-level jitter so this term penalizes genuine "moving away" behavior.
    away = torch.clamp(d - old - 0.01, min=0.0)
    return -(away + 0.5 * away * away)


def heading_to_target_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Cosine alignment between body forward (+X projected on world XY) and goal direction in the XY plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    g = _to_goal_vec(env, asset_cfg)
    gn = torch.nn.functional.normalize(g[:, :2] + 1.0e-8, dim=-1)
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fxy = torch.stack([fx, fy], dim=-1)
    fxy = torch.nn.functional.normalize(fxy + 1.0e-8, dim=-1)
    # Strongly reward facing toward the target; give no credit when facing away.
    h = torch.sum(fxy * gn, dim=-1)
    return torch.square(torch.clamp(h, min=0.0, max=1.0))


def heading_goal_closure_speed_reward(
    env: ManagerBasedRLEnv,
    cap_ms: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """Rewards facing the goal **only while** moving toward it (discourages spin / orbit without closure).

    Returns ``relu(cos heading) * clip(v·ĝ / cap, 0, 1)`` using the same heading convention as
    :func:`heading_to_target_reward` and world-frame velocity projected on the planar goal unit
    vector ``ĝ`` (positive = closing distance). Pure in-place rotation yields zero; tangential
    motion with ``cos heading ≈ 0`` also yields near-zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g = _to_goal_vec(env, asset_cfg)
    gn = torch.nn.functional.normalize(g[:, :2] + 1.0e-8, dim=-1)
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fxy = torch.stack([fx, fy], dim=-1)
    fxy = torch.nn.functional.normalize(fxy + 1.0e-8, dim=-1)
    h = torch.sum(fxy * gn, dim=-1)
    h_pos = torch.clamp(h, min=0.0, max=1.0)
    v = asset.data.root_lin_vel_w[:, :2]
    v_along = torch.sum(v * gn, dim=-1)
    cap = max(float(cap_ms), 1.0e-6)
    closure = torch.clamp(v_along / cap, min=0.0, max=1.0)
    # Emphasize "point at goal first, then translate toward it" over tangential motion.
    return (h_pos * h_pos) * closure


def yaw_rate_without_goal_closure_penalty(
    env: ManagerBasedRLEnv,
    min_dist_m: float,
    closure_speed_threshold_m_s: float,
    yaw_thresh_rad_s: float,
    yaw_ref_rad_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """Penalty magnitude when turning fast without translating toward the goal while still far away.

    Reduces orbit / pinwheel exploitation where ``distance_progress`` stays small per step but
    ``heading_to_target`` can still intermittently reward re-pointing toward the marker.
    """
    d = _planar_dist(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    gn = torch.nn.functional.normalize(_to_goal_vec(env, asset_cfg)[:, :2] + 1.0e-8, dim=-1)
    v = asset.data.root_lin_vel_w[:, :2]
    v_along = torch.sum(v * gn, dim=-1)
    wz = torch.abs(asset.data.root_ang_vel_w[:, 2])
    far = (d > float(min_dist_m)).float()
    low_close = (v_along < float(closure_speed_threshold_m_s)).float()
    thresh = float(yaw_thresh_rad_s)
    ref = max(float(yaw_ref_rad_s), 1.0e-6)
    yaw_excess = torch.relu(wz - thresh)
    shaped = torch.clamp(yaw_excess / ref, 0.0, 2.5)
    return far * low_close * shaped


def body_reverse_linear_velocity_penalty(
    env: ManagerBasedRLEnv,
    scale_ms: float = 2.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    far_boost_start_m: float = 0.0,
    far_boost_max_mul: float = 1.0,
    far_boost_ramp_m: float = 14.0,
) -> torch.Tensor:
    """Positive magnitude when the rover moves opposite body +X (reverse / backing up).

    Uses root linear velocity in the body frame: penalizes ``v_x^b < 0``. Normalized by
    ``scale_ms`` and squared so small reverse speeds are discouraged smoothly.

    Optional OA-flat shaping: when ``far_boost_start_m > 0`` and ``far_boost_max_mul > 1``, scale the
    penalty up with planar goal distance so backing away from a distant goal is clearly worse than
    small reverse corrections near the marker.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vx_b = asset.data.root_lin_vel_b[:, 0]
    rev = torch.relu(-vx_b)
    cap = max(float(scale_ms), 1.0e-6)
    mag = (rev / cap) ** 2
    d0 = float(far_boost_start_m)
    mmax = float(far_boost_max_mul)
    if d0 > 0.0 and mmax > 1.0 + 1.0e-6:
        d = _planar_dist(env, asset_cfg)
        ramp = max(float(far_boost_ramp_m), 1.0e-6)
        t = torch.clamp((d - d0) / ramp, 0.0, 1.0)
        mag = mag * (1.0 + t * (mmax - 1.0))
    return mag


def forward_velocity_to_target_reward(
    env: ManagerBasedRLEnv,
    cap_ms: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    g = _to_goal_vec(env, asset_cfg)
    gn = torch.nn.functional.normalize(g[:, :2] + 1.0e-8, dim=-1)
    v = asset.data.root_lin_vel_w[:, :2]
    v_along = torch.sum(v * gn, dim=-1)
    cap = float(cap_ms)
    v_along = torch.clamp(v_along, -cap, cap) / max(cap, 1.0e-6)
    return v_along


def close_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    bonus_max_dist_m: float | None = None,
) -> torch.Tensor:
    """Shaping for proximity to the planar goal marker.

    **Legacy** (``bonus_max_dist_m`` is None): sparse ``1.0`` when ``dist < threshold_m``. Wide thresholds invite
    orbiting at the edge to defer success while farming dense progress/other terms.

    **OA-flat / continuous goals** (``bonus_max_dist_m`` set): dense bonus in ``[0, 1]`` with maximum at the marker
    and quadratic decay to zero at ``bonus_max_dist_m``:

    ``((1 - dist / bonus_max_dist_m)_+)^2``

    Set ``bonus_max_dist_m ≈ goal_reached_threshold_m + margin`` so credit concentrates inside the real success
    radius instead of a thin avoidance ring just outside relocation distance.
    """
    d = _planar_dist(env, asset_cfg)
    cap_opt = bonus_max_dist_m
    if cap_opt is None:
        thr = float(threshold_m)
        return (d < thr).float()
    cap = max(float(cap_opt), 1.0e-6)
    # Peak at d=0; zero at and beyond cap; discourage hovering at the outer edge vs driving through to goal success.
    t = torch.clamp(1.0 - d / cap, min=0.0, max=1.0)
    return t * t


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    set_relocate_pending: bool = False,
) -> torch.Tensor:
    u = env.unwrapped
    d = _planar_dist(env, asset_cfg)
    thr = float(threshold_m)
    reached = (d < thr) & (~u._flat_goal_bonus_given)
    u._flat_goal_bonus_given = materialize_planar_goal_dist_buffer(u._flat_goal_bonus_given | reached)
    if set_relocate_pending and bool(torch.any(reached)):
        if not hasattr(u, "_oa_flat_relocate_goal_pending"):
            u._oa_flat_relocate_goal_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        u._oa_flat_relocate_goal_pending = materialize_planar_goal_dist_buffer(
            u._oa_flat_relocate_goal_pending | reached
        )
    return reached.float()


def target_approach_time_decay_shaping(
    env: ManagerBasedRLEnv,
    tau_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    goal_threshold_m: float = 1.5,
) -> torch.Tensor:
    """Per-step shaping that is largest right after a target is placed and decays with time.

    ``exp(- age_s / tau_s)`` where ``age_s`` is episode time since the current target segment started (reset on
    full episode reset and whenever the goal marker is relocated). While inside ``goal_threshold_m`` the term is
    zero so it does not fight the one-shot success bonus.
    """
    u = env.unwrapped
    if not hasattr(u, "_flat_goal_segment_start_episode_steps"):
        return torch.zeros(env.num_envs, device=env.device)
    d = _planar_dist(env, asset_cfg)
    thr = float(goal_threshold_m)
    inside = d < thr
    age_steps = env.episode_length_buf - u._flat_goal_segment_start_episode_steps
    age_s = age_steps.clamp(min=0).float() * float(env.step_dt)
    tau = max(float(tau_s), 1.0e-6)
    return torch.where(inside, torch.zeros_like(age_s), torch.exp(-age_s / tau))


def tilt_penalty_flat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
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
    c = torch.sum(z_w * up, dim=-1).clamp(-1.0, 1.0)
    return 1.0 - c


def boundary_penalty_flat(
    env: ManagerBasedRLEnv,
    plane_half_size_m: float,
    margin_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    p = _rover_rel_xy(env, asset_cfg)
    H = float(plane_half_size_m)
    m = float(margin_m)
    dx = H - torch.abs(p[:, 0])
    dy = H - torch.abs(p[:, 1])
    dmin = torch.minimum(dx, dy)
    return torch.clamp(m - dmin, min=0.0)


_WHEEL_BODY_NAMES: tuple[str, str, str, str] = ("wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl")


def _flat_goal_wheel_body_indices(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cached body indices for FR, FL, RR, RL (see :data:`_WHEEL_BODY_NAMES`)."""
    u = env.unwrapped
    if hasattr(u, "_flat_goal_wheel_body_idx"):
        return u._flat_goal_wheel_body_idx
    asset: Articulation = env.scene[asset_cfg.name]
    ids, _ = asset.find_bodies(list(_WHEEL_BODY_NAMES), preserve_order=True)
    u._flat_goal_wheel_body_idx = torch.tensor(ids, device=env.device, dtype=torch.long)
    return u._flat_goal_wheel_body_idx


def wheel_contact_proxy_mask(
    env: ManagerBasedRLEnv,
    threshold_n: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    """Approximate wheel–ground contact from link incoming joint linear force magnitude (no ContactSensor).

    Uses :attr:`ArticulationData.body_incoming_joint_wrench_b` linear part (N, 4, 3) → norm per wheel.
    ``threshold_n`` is a Newton-scale cutoff (tune with ``--rover_flat_wheel_ground_contact_threshold_n``).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    idx = _flat_goal_wheel_body_indices(env, asset_cfg)
    w = asset.data.body_incoming_joint_wrench_b[:, idx, :3]
    n = torch.linalg.norm(w + 1.0e-9, dim=-1)
    return n > float(threshold_n)


def four_wheels_ground_contact_reward(
    env: ManagerBasedRLEnv,
    threshold_n: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    c = wheel_contact_proxy_mask(env, threshold_n, asset_cfg)
    return torch.all(c, dim=-1).float()


def front_wheels_ground_contact_reward(
    env: ManagerBasedRLEnv,
    threshold_n: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    c = wheel_contact_proxy_mask(env, threshold_n, asset_cfg)
    return (c[:, 0] & c[:, 1]).float()


def rear_wheels_ground_contact_reward(
    env: ManagerBasedRLEnv,
    threshold_n: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    c = wheel_contact_proxy_mask(env, threshold_n, asset_cfg)
    return (c[:, 2] & c[:, 3]).float()


def not_four_wheels_ground_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold_n: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
) -> torch.Tensor:
    c = wheel_contact_proxy_mask(env, threshold_n, asset_cfg)
    return (~torch.all(c, dim=-1)).float()


def action_rate_penalty_flat(env: ManagerBasedRLEnv) -> torch.Tensor:
    return rrew.action_rate_penalty(env)

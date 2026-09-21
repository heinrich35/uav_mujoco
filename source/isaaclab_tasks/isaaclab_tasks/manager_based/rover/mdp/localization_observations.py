# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Pseudo-visual and localization-style observations (fixed dims for PPO)."""

from __future__ import annotations

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._rover_obs_dims import (
    OBS_VIS_BEARING_DIST,
    OBS_VIS_CONFIDENCE,
    OBS_VIS_LANDMARKS,
    OBS_VIS_LANDMARK_COUNTS,
    OBS_VIS_OBJECT_TRACKS,
)


def _fill_visual_buffers(env: ManagerBasedRLEnv) -> None:
    """Populate cached geometry-based pseudo-vision from goals/boxes."""
    asset: Articulation = env.scene["rover"]
    root = asset.data.root_pos_w[:, :3]
    goals = getattr(env.unwrapped, "_goal_positions_w", None)
    boxes = getattr(env.unwrapped, "_box_positions_w", None)
    n_vis_lm = OBS_VIS_LANDMARKS // 4
    lm = torch.zeros(env.num_envs, OBS_VIS_LANDMARKS, device=env.device)
    if goals is not None:
        k_lm = min(n_vis_lm, int(goals.shape[1]))
        bearing = torch.zeros(env.num_envs, n_vis_lm, device=env.device)
        dist = torch.zeros(env.num_envs, n_vis_lm, device=env.device)
        if k_lm > 0:
            g = goals[:, :k_lm, :]
            rel = g - root[:, None, :]
            bearing[:, :k_lm] = torch.atan2(rel[:, :, 2], rel[:, :, 0])
            dist[:, :k_lm] = torch.linalg.norm(rel, dim=-1)
        lm[:, 0::4] = torch.sin(bearing)
        lm[:, 1::4] = torch.cos(bearing)
        lm[:, 2::4] = bearing * 0.0
        lm[:, 3::4] = torch.tanh(dist / 30.0)
    env.unwrapped._obs_visible_landmarks = lm

    n_tr = OBS_VIS_OBJECT_TRACKS // 4
    tr = torch.zeros(env.num_envs, OBS_VIS_OBJECT_TRACKS, device=env.device)
    if boxes is not None:
        k_tr = min(n_tr, int(boxes.shape[1]))
        bearing = torch.zeros(env.num_envs, n_tr, device=env.device)
        dist = torch.zeros(env.num_envs, n_tr, device=env.device)
        if k_tr > 0:
            b = boxes[:, :k_tr, :]
            rel = b - root[:, None, :]
            bearing[:, :k_tr] = torch.atan2(rel[:, :, 2], rel[:, :, 0])
            dist[:, :k_tr] = torch.linalg.norm(rel, dim=-1)
        tr[:, 0::4] = torch.sin(bearing)
        tr[:, 1::4] = torch.cos(bearing)
        tr[:, 2::4] = torch.tanh(dist / 20.0)
        tr[:, 3::4] = torch.tanh(dist / 30.0)
    env.unwrapped._obs_visual_tracks = tr

    counts = torch.zeros(env.num_envs, OBS_VIS_LANDMARK_COUNTS, device=env.device)
    if goals is not None:
        counts[:, 0] = float(goals.shape[1])
    if boxes is not None:
        counts[:, 1] = float(boxes.shape[1])
    env.unwrapped._obs_landmark_counts = counts * 0.1

    conf = torch.zeros(env.num_envs, OBS_VIS_CONFIDENCE, device=env.device)
    conf[:, :] = 0.5
    env.unwrapped._obs_confidence = conf

    bd = torch.zeros(env.num_envs, OBS_VIS_BEARING_DIST, device=env.device)
    if boxes is not None:
        rel = boxes - root[:, None, :]
        bearing = torch.atan2(rel[:, :, 2], rel[:, :, 0])
        bd[:, : min(bearing.shape[1], OBS_VIS_BEARING_DIST)] = bearing[:, : OBS_VIS_BEARING_DIST] * 0.1
    env.unwrapped._obs_bearing_dist = bd


def visible_landmarks(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    _fill_visual_buffers(env)
    return env.unwrapped._obs_visible_landmarks


def visual_object_tracks(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    _fill_visual_buffers(env)
    return env.unwrapped._obs_visual_tracks


def visual_landmark_counts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    _fill_visual_buffers(env)
    return env.unwrapped._obs_landmark_counts


def visual_confidence_stats(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    _fill_visual_buffers(env)
    return env.unwrapped._obs_confidence


def visual_bearing_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    _fill_visual_buffers(env)
    return env.unwrapped._obs_bearing_dist


def imu_yaw_rate_noisy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 1:2] + 0.01 * torch.randn_like(asset.data.root_ang_vel_w[:, 1:2])


def imu_lin_accel_noisy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_w[:, :3]
    if not hasattr(env.unwrapped, "_prev_lin_vel_imu"):
        env.unwrapped._prev_lin_vel_imu = v.clone()
    a = (v - env.unwrapped._prev_lin_vel_imu) / (env.step_dt + 1.0e-8)
    env.unwrapped._prev_lin_vel_imu = v.clone()
    return a + 0.02 * torch.randn_like(a)


def optical_flow_forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    tang = torch.stack([pos[:, 2], torch.zeros_like(pos[:, 0]), -pos[:, 0]], dim=-1)
    tang = torch.nn.functional.normalize(tang + 1.0e-8, dim=-1)
    v = asset.data.root_lin_vel_w[:, :3]
    return torch.sum(v * tang, dim=-1, keepdim=True) * 0.05


def optical_flow_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 1:2] * 0.05


def wheel_odometry_estimate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ids, _ = asset.find_joints(
        ["wheel_fr_joint", "wheel_fl_joint", "wheel_rr_joint", "wheel_rl_joint"], preserve_order=True
    )
    wv = asset.data.joint_vel[:, ids]
    if not hasattr(env.unwrapped, "_wheel_odom"):
        env.unwrapped._wheel_odom = torch.zeros(env.num_envs, 3, device=env.device)
    # crude integrate forward / lateral from differential model (match DifferentialDriveActionCfg / URDF wheel r)
    r = 0.08
    tw = 0.48
    # Keep wheel kinematics as [num_envs, 1] so in-place adds match _wheel_odom slices (avoid [N]+[N,1] -> [N,N]).
    wl = ((wv[:, 1] + wv[:, 3]) * 0.5).unsqueeze(-1)
    wr = ((wv[:, 0] + wv[:, 2]) * 0.5).unsqueeze(-1)
    v = r * (wl + wr) * 0.5
    wz = r * (wr - wl) / (tw + 1.0e-8)
    th = env.unwrapped._wheel_odom[:, 2:3] + wz * env.step_dt
    # Integrate in world XZ (cylinder about +Y): slot0 ≈ Δx, slot1 ≈ Δz, slot2 = heading in XZ.
    env.unwrapped._wheel_odom[:, 0:1] += v * torch.cos(th) * env.step_dt
    env.unwrapped._wheel_odom[:, 1:2] += v * torch.sin(th) * env.step_dt
    env.unwrapped._wheel_odom[:, 2:3] = th
    o = env.unwrapped._wheel_odom
    return torch.tanh(torch.cat([o, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1) * 0.02)


def last_drive_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.get_term("drive").processed_actions


def expert_last_drive_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the last expert drive action from the private expert drive term."""
    expert_drive = getattr(env.unwrapped, "_expert_drive", None)
    if expert_drive is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return expert_drive.raw_actions

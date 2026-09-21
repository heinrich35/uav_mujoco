# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Privileged obstacle geometry + depth-based scans (sim ``distance_to_image_plane``) for OA-Flat."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _align_depth_to_cam_batch(dep: torch.Tensor, ne: int) -> torch.Tensor:
    """Match depth leading dim to ``ne`` (handles batch-1 annotators)."""
    nd = int(dep.shape[0])
    if nd == ne:
        return dep
    if nd == 1 and ne > 1:
        return dep.expand(ne, -1, -1).contiguous()
    if nd > ne:
        return dep[:ne].contiguous()
    reps = (ne + nd - 1) // nd
    return dep.repeat(reps, 1, 1)[:ne].contiguous()


def _match_depth_batch(dep: torch.Tensor, ne: int) -> torch.Tensor:
    nd = int(dep.shape[0])
    if nd == ne:
        return dep
    return _align_depth_to_cam_batch(dep, ne)


def oa_flat_depth_scan_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    sensor_name: str | None = None,
    num_bins: int = 32,
    clip_near_m: float = 0.05,
    clip_far_m: float = 28.0,
    row_center_frac: float = 0.58,
    vertical_half_window_px: int = 7,
) -> torch.Tensor:
    """Horizon-band min depth samples (geometry from sim), normalized to [0, 1].

    Each column uses the minimum ``distance_to_image_plane`` over a small vertical window
    at a fixed image row (pseudo-lidar stripe), matching the Replicator depth buffer.
    """
    _ = asset_cfg
    ne = env.num_envs
    device = env.device
    k = max(4, int(num_bins))
    sensor = str(sensor_name or getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    if sensor not in env.scene.sensors:
        return torch.zeros(ne, k, device=device)
    cam = env.scene.sensors[sensor]
    dep = cam.data.output.get("distance_to_image_plane", None)
    if dep is None:
        return torch.zeros(ne, k, device=device)
    if dep.dim() == 4:
        dep = dep.squeeze(-1)
    dep = dep.float()
    dep = _match_depth_batch(dep, ne)
    _, h, w = dep.shape
    if w < 2 or h < 4:
        return torch.zeros(ne, k, device=device)
    row = max(0, min(h - 1, int(round(float(row_center_frac) * float(h - 1)))))
    vhw = max(0, int(vertical_half_window_px))
    r0 = max(0, row - vhw)
    r1 = min(h, row + vhw + 1)
    cols = torch.linspace(0, w - 1, k, device=device).long().clamp(0, w - 1)
    strip = dep[:, r0:r1, :][:, :, cols]
    min_d = strip.min(dim=1).values
    near = max(float(clip_near_m), 1.0e-3)
    span = max(float(clip_far_m) - near, 1.0e-3)
    out = ((min_d.clamp(near, near + span) - near) / span).clamp(0.0, 1.0)
    return out


def _box_exterior_dist_xy(
    rover_xy: torch.Tensor,
    boxes_xy: torch.Tensor,
    half_extent_m: float,
) -> torch.Tensor:
    """Planar distance from points to axis-aligned square footprints (vectorized). Shape (N, B)."""
    h = max(float(half_extent_m), 1.0e-6)
    dx = torch.abs(boxes_xy[..., 0] - rover_xy[:, None, 0]) - h
    dy = torch.abs(boxes_xy[..., 1] - rover_xy[:, None, 1]) - h
    dx = torch.relu(dx)
    dy = torch.relu(dy)
    return torch.sqrt(dx * dx + dy * dy + 1.0e-12)


def _nearest_obstacle_distance_and_order(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
    """Returns (d_min combined, rover_xy env frame, boxes_xy env frame, nb, half_extent, H)."""
    asset: Articulation = env.scene[asset_cfg.name]
    boxes_w = getattr(env.unwrapped, "_oa_flat_box_positions_w", None)
    if boxes_w is None:
        boxes_w = getattr(env.unwrapped, "_box_positions_w", None)
    rover_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    H = float(getattr(env.cfg, "plane_half_size_m", 25.0))
    dx = rover_xy[:, 0].abs()
    dy = rover_xy[:, 1].abs()
    d_wall_only = torch.minimum(H - dx, H - dy).clamp(min=0.0)

    nb_cfg = int(getattr(env.cfg, "oa_flat_box_count", boxes_w.shape[1] if boxes_w is not None else 0))
    nb = min(nb_cfg, int(boxes_w.shape[1])) if boxes_w is not None else 0
    if boxes_w is None or nb < 1:
        return (
            d_wall_only,
            rover_xy,
            torch.zeros(env.num_envs, max(1, nb_cfg), 2, device=env.device, dtype=rover_xy.dtype),
            0,
            float(getattr(env.cfg, "oa_flat_box_extent_m", 1.0)) * 0.5,
            H,
        )
    boxes_xy = boxes_w[:, :nb, :2] - env.scene.env_origins[:, None, :2]
    half_extent = float(getattr(env.cfg, "oa_flat_box_extent_m", 1.0)) * 0.5
    d_boxes = _box_exterior_dist_xy(rover_xy, boxes_xy, half_extent)
    d_min_boxes, _ = torch.sort(d_boxes, dim=1)
    d_obstacle = torch.minimum(d_min_boxes[:, 0], d_wall_only)
    return d_obstacle, rover_xy, boxes_xy, nb, half_extent, H


def oa_flat_privileged_nearest_obstacles(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    k_nearest: int = 4,
    dist_scale_m: float = 25.0,
) -> torch.Tensor:
    """K nearest obstacle centers in rover body XY: sin/cos bearing + bounded distance."""
    kk = max(1, int(k_nearest))
    feat = 3  # sin, cos, tanh(dist/scale_like)
    out = torch.zeros(env.num_envs, kk * feat, device=env.device)
    _, rover_xy, boxes_xy, nb, half_extent, _ = _nearest_obstacle_distance_and_order(env, asset_cfg)
    if nb < 1:
        return out
    d_sorted, order = torch.sort(
        _box_exterior_dist_xy(rover_xy, boxes_xy, half_extent),
        dim=1,
    )
    picks = order[:, : min(kk, nb)]
    b_idx = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand_as(picks)
    sel = boxes_xy[b_idx, picks]
    delta_w = torch.zeros(env.num_envs, picks.shape[1], 3, device=env.device)
    delta_w[:, :, 0] = sel[:, :, 0] - rover_xy[:, None, 0]
    delta_w[:, :, 1] = sel[:, :, 1] - rover_xy[:, None, 1]
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    q_exp = quat[:, None, :].expand(-1, delta_w.shape[1], -1)
    rel_b = math_utils.quat_apply_inverse(q_exp.reshape(-1, 4), delta_w.reshape(-1, 3)).reshape(
        env.num_envs, -1, 3
    )
    dists = d_sorted[:, : picks.shape[1]]
    s = max(float(dist_scale_m), 1.0e-6)
    for i in range(picks.shape[1]):
        bx = rel_b[:, i, 0]
        by = rel_b[:, i, 1]
        bearing = torch.atan2(by, bx)
        out[:, i * feat + 0] = torch.sin(bearing)
        out[:, i * feat + 1] = torch.cos(bearing)
        out[:, i * feat + 2] = torch.tanh(dists[:, i] / s)
    return out


def oa_flat_privileged_nearest_obstacle_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    scale_m: float = 25.0,
) -> torch.Tensor:
    """Scalar normalized nearest planar distance to blue boxes or arena boundary (privileged)."""
    d_obstacle, _, _, _, _, _ = _nearest_obstacle_distance_and_order(env, asset_cfg)
    s = max(float(scale_m), 1.0e-6)
    return (d_obstacle / s).clamp(0.0, 4.0).unsqueeze(-1)


def oa_flat_aux_nearest_obstacle_distance_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    scale_m: float = 25.0,
) -> torch.Tensor:
    """Auxiliary regression target from GT layout (matches privileged distance scale). Must stay in rollout obs."""
    return oa_flat_privileged_nearest_obstacle_distance(env, asset_cfg=asset_cfg, scale_m=scale_m)

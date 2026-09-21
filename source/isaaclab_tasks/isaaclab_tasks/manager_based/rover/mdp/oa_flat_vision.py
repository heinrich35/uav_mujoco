# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RGB + depth blob detection (blue obstacles, yellow goal) → robot-frame bearing/range estimates."""

from __future__ import annotations

# Extra yellow policy obs dims (beyond ``k × 4`` tracks): aggregates + histogram + vs-blue (see env cfg toggle).
OA_FLAT_VISUAL_YELLOW_GOAL_ENRICH_DIM: int = 4 + 20 + 4

import torch
import torch.nn.functional as F
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def _quat_body_forward_xy_w(asset: Articulation) -> torch.Tensor:
    q = asset.data.root_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fy = 2.0 * (qx * qy + qw * qz)
    fxy = torch.stack([fx, fy], dim=-1)
    return torch.nn.functional.normalize(fxy + 1.0e-8, dim=-1)


def _peak_slot_tracks(
    sm: torch.Tensor,
    sm_dep: torch.Tensor,
    *,
    n_env: int,
    ne: int,
    k: int,
    hh: int,
    ww: int,
    w: int,
    h: int,
    env_i: torch.Tensor,
    intr: torch.Tensor | None,
    asset: Articulation,
    fxy_w: torch.Tensor,
    excess_threshold: float,
    max_range_m: float,
    device: torch.device,
    conf_sigmoid_slope: float = 40.0,
) -> torch.Tensor:
    """Vectorized top-``k`` blob peaks → (n_env,k,4) sin/cos/tanh(depth/scale)/conf."""
    flat0 = sm.reshape(n_env, -1)
    tracks = torch.zeros(n_env, k, 4, device=device)
    ncol = int(flat0.shape[1])
    work = flat0.clone()
    for slot in range(k):
        peak = torch.argmax(work, dim=1).clamp(min=0, max=max(ncol - 1, 0))
        work.scatter_(1, peak.unsqueeze(1), torch.full_like(work[:, :1], -1.0e3))
        ww_i = int(ww)
        hh_i = int(hh)
        pv = torch.div(peak, ww_i, rounding_mode="floor").clamp(0, max(hh_i - 1, 0))
        pu = (peak - pv * ww_i).clamp(0, max(ww_i - 1, 0))
        u_n = (pu.float() + 0.5) / float(ww_i) * 2.0 - 1.0
        d_pix = sm_dep[env_i, pv.long(), pu.long()].clamp(0.05, max_range_m)

        if intr is not None and intr.shape[-1] >= 3:
            fx = intr[:, 0, 0].clamp(min=1.0e-3)
            fy_i = intr[:, 1, 1].clamp(min=1.0e-3)
            cx = intr[:, 0, 2]
            cy = intr[:, 1, 2]
            scale_x = float(w) / float(ww_i)
            scale_y = float(h) / float(hh_i)
            u_pix = (pu.float() + 0.5) * scale_x
            v_pix = (pv.float() + 0.5) * scale_y
            x_cam = (u_pix - cx) * d_pix / fx
            z_cam = (v_pix - cy) * d_pix / fy_i
            ray_cam = torch.stack([x_cam, z_cam, torch.ones_like(x_cam)], dim=-1)
            ray_cam = torch.nn.functional.normalize(ray_cam + 1.0e-6, dim=-1)
            q = asset.data.root_quat_w[:n_env]
            ray_w = math_utils.quat_apply(q, ray_cam)
        else:
            ray_w = torch.stack(
                [u_n * d_pix * 0.35, torch.zeros_like(u_n), torch.ones_like(u_n) * d_pix.clamp(0.5, max_range_m)],
                dim=-1,
            )
            ray_w = torch.nn.functional.normalize(ray_w + 1.0e-6, dim=-1)

        rel_xy = torch.nn.functional.normalize(ray_w[:, :2] + 1.0e-6, dim=-1)
        c = torch.sum(rel_xy * fxy_w, dim=-1).clamp(-1.0, 1.0)
        s = torch.sum(torch.stack([-rel_xy[:, 1], rel_xy[:, 0]], dim=-1) * fxy_w, dim=-1).clamp(-1.0, 1.0)
        bearing = torch.atan2(s, c)
        mass = sm[env_i, pv.long(), pu.long()]
        slope = float(conf_sigmoid_slope)
        conf = torch.sigmoid((mass - excess_threshold) * slope) * (mass > (excess_threshold * 0.5)).float()
        conf = conf * (d_pix < max_range_m).float()
        tracks[:, slot, 0] = torch.sin(bearing)
        tracks[:, slot, 1] = torch.cos(bearing)
        tracks[:, slot, 2] = torch.tanh(d_pix / 12.0)
        tracks[:, slot, 3] = conf
    if n_env < ne:
        reps = (ne + n_env - 1) // n_env
        tracks = tracks.repeat(reps, 1, 1)[:ne].contiguous()
    return tracks


def update_oa_flat_blue_detections_from_camera(
    env: ManagerBasedRLEnv,
    sensor_name: str = "rover_oa_rgb_cam",
    max_detections: int = 6,
    blue_excess_threshold: float = 0.18,
    min_blob_pixels: int = 8,
    max_range_m: float = 28.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    compute_blue: bool = True,
    compute_yellow_goal: bool = True,
    yellow_goal_excess_threshold: float = 0.06,
    yellow_goal_max_range_m: float = 40.0,
    yellow_goal_relu_score: bool = True,
    yellow_goal_score_boost: float = 1.35,
) -> None:
    """Populate blue and/or yellow-goal blob tracks from onboard RGB+D (camera-only).

    Yellow channel targets the saturated yellow goal cuboid (:math:`\\min(R,G)\\!-\\!B` excess).
    """
    u = env.unwrapped
    n = env.num_envs
    device = env.device
    k = int(max_detections)
    feat = 4  # sin, cos, tanh(range), conf

    if sensor_name not in env.scene.sensors:
        if not hasattr(u, "_oa_flat_visual_tracks"):
            u._oa_flat_visual_tracks = torch.zeros(n, k * feat, device=device)
            u._oa_flat_visual_counts = torch.zeros(n, 4, device=device)
            u._oa_flat_visual_confidence = torch.zeros(n, 4, device=device)
            u._oa_flat_visual_bearing_hist = torch.zeros(n, 20, device=device)
        if not hasattr(u, "_oa_flat_visual_yellow_goal_tracks"):
            u._oa_flat_visual_yellow_goal_tracks = torch.zeros(n, k * feat, device=device)
            u._oa_flat_visual_yellow_goal_confidence = torch.zeros(n, 4, device=device)
            u._oa_flat_visual_yellow_goal_bearing_hist = torch.zeros(n, 20, device=device)
            u._oa_flat_visual_yellow_goal_counts = torch.zeros(n, 4, device=device)
        return

    cam = env.scene.sensors[sensor_name]
    rgb = cam.data.output.get("rgb", None)
    depth = cam.data.output.get("distance_to_image_plane", None)
    if rgb is None or depth is None:
        return

    # rgb: [N,H,W,C] uint8 or float; depth: [N,H,W] or [N,H,W,1]
    if rgb.dtype != torch.float32:
        rgb_f = rgb.float() / 255.0
    else:
        rgb_f = rgb
    if rgb_f.dim() == 3:
        rgb_f = rgb_f.unsqueeze(0)
    if depth.dim() == 3:
        dep = depth
    else:
        dep = depth.squeeze(-1)

    # Never expand camera tensors to env.num_envs: GPU camera buffers stay at ``view.count`` (often 1 on
    # first frames); expanding here desyncs from ``sensor.reset(env_ids)`` and causes CUDA index asserts.
    ne = int(env.num_envs)
    n_cam = int(rgb_f.shape[0])
    nd = int(dep.shape[0])
    if nd != n_cam:
        if nd == 1:
            dep = dep.expand(n_cam, -1, -1).contiguous()
        elif nd > n_cam:
            dep = dep[:n_cam].contiguous()
        else:
            reps = (n_cam + nd - 1) // nd
            dep = dep.repeat(reps, 1, 1)[:n_cam].contiguous()
    if n_cam > ne:
        rgb_f = rgb_f[:ne].contiguous()
        dep = dep[:ne].contiguous()
        n_env = ne
    else:
        n_env = n_cam

    _, h, w, _ = rgb_f.shape
    # RGB vs depth can differ in H×W (annotator / buffer layout). Mismatch breaks ``torch.gather`` on ``peak``.
    if int(dep.shape[-2]) != int(h) or int(dep.shape[-1]) != int(w):
        dep = F.interpolate(
            dep.unsqueeze(1).float(),
            size=(int(h), int(w)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    r = rgb_f[..., 0]
    gch = rgb_f[..., 1]
    b = rgb_f[..., 2]
    blue_score = b - torch.maximum(r, gch)
    yellow_raw = torch.minimum(r, gch) - b
    y_relu = bool(getattr(env.cfg, "oa_flat_visual_yellow_score_relu", yellow_goal_relu_score))
    if y_relu:
        yellow_score = torch.relu(yellow_raw)
    else:
        yellow_score = yellow_raw
    boost_y = float(getattr(env.cfg, "oa_flat_visual_yellow_score_boost", yellow_goal_score_boost))
    if boost_y != 1.0:
        yellow_score = yellow_score * boost_y

    sm = F.avg_pool2d(blue_score.unsqueeze(1), kernel_size=5, stride=2, padding=2).squeeze(1)
    sm_y = F.avg_pool2d(yellow_score.unsqueeze(1), kernel_size=5, stride=2, padding=2).squeeze(1)
    sm_dep = F.avg_pool2d(dep.unsqueeze(1).float(), kernel_size=5, stride=2, padding=2).squeeze(1)
    _, hh, ww = sm.shape
    sm_dep = F.interpolate(sm_dep.unsqueeze(1), size=(hh, ww), mode="nearest").squeeze(1).contiguous()
    bs = int(sm.shape[0])
    bsd = int(sm_dep.shape[0])
    if bs != bsd:
        if bsd == 1:
            sm_dep = sm_dep.expand(bs, -1, -1).contiguous()
        elif bs == 1:
            sm = sm.expand(bsd, -1, -1).contiguous()
            sm_y = sm_y.expand(bsd, -1, -1).contiguous()
            bs = bsd
        else:
            m = min(bs, bsd)
            sm = sm[:m].contiguous()
            sm_y = sm_y[:m].contiguous()
            sm_dep = sm_dep[:m].contiguous()
            bs = m
    bs_y = int(sm_y.shape[0])
    if bs_y != bs:
        if bs_y == 1:
            sm_y = sm_y.expand(bs, -1, -1).contiguous()
        else:
            sm_y = sm_y[:bs].contiguous()
    n_env = min(bs, ne)
    if bs > ne:
        sm = sm[:ne].contiguous()
        sm_y = sm_y[:ne].contiguous()
        sm_dep = sm_dep[:ne].contiguous()
        n_env = ne

    env_i = torch.arange(n_env, device=device, dtype=torch.long)

    def _bearing_hist(score_map: torch.Tensor) -> torch.Tensor:
        col_score = score_map[:n_env].mean(dim=1)
        wc_bins = int(col_score.shape[1])
        hist = torch.zeros(n_env, 20, device=device)
        for bi in range(20):
            lo = int(bi * wc_bins / 20) if wc_bins > 0 else 0
            hi = int((bi + 1) * wc_bins / 20) if wc_bins > 0 else 0
            lo = max(0, min(lo, wc_bins))
            hi = max(0, min(hi, wc_bins))
            if hi <= lo:
                hist[:, bi] = 0.0
            else:
                hist[:, bi] = col_score[:, lo:hi].mean(dim=1)
        return torch.nn.functional.normalize(hist + 1.0e-4, dim=-1)

    hist = _bearing_hist(blue_score) if compute_blue else torch.zeros(n_env, 20, device=device)
    hist_y = _bearing_hist(yellow_score) if compute_yellow_goal else torch.zeros(n_env, 20, device=device)

    intr = getattr(cam.data, "intrinsic_matrices", None)
    if intr is not None:
        ni = int(intr.shape[0])
        if ni == 1 and n_env > 1:
            intr = intr.expand(n_env, -1, -1).contiguous()
        elif ni > n_env:
            intr = intr[:n_env].contiguous()
        elif ni < n_env:
            reps = (n_env + ni - 1) // ni
            intr = intr.repeat(reps, 1, 1)[:n_env].contiguous()

    asset: Articulation = env.scene[asset_cfg.name]
    fxy_w = _quat_body_forward_xy_w(asset)[:n_env]

    if compute_blue:
        tracks = _peak_slot_tracks(
            sm,
            sm_dep,
            n_env=n_env,
            ne=ne,
            k=k,
            hh=hh,
            ww=ww,
            w=int(w),
            h=int(h),
            env_i=env_i,
            intr=intr,
            asset=asset,
            fxy_w=fxy_w,
            excess_threshold=float(blue_excess_threshold),
            max_range_m=float(max_range_m),
            device=device,
        )
    else:
        tracks = torch.zeros(ne, k, feat, device=device)

    if compute_yellow_goal:
        y_thresh = float(
            getattr(env.cfg, "oa_flat_visual_yellow_excess_threshold", yellow_goal_excess_threshold)
        )
        y_slope = float(getattr(env.cfg, "oa_flat_visual_yellow_conf_sigmoid_slope", 28.0))
        tracks_y = _peak_slot_tracks(
            sm_y,
            sm_dep,
            n_env=n_env,
            ne=ne,
            k=k,
            hh=hh,
            ww=ww,
            w=int(w),
            h=int(h),
            env_i=env_i,
            intr=intr,
            asset=asset,
            fxy_w=fxy_w,
            excess_threshold=y_thresh,
            max_range_m=float(yellow_goal_max_range_m),
            device=device,
            conf_sigmoid_slope=y_slope,
        )
        cval = tracks_y[..., 3:4]
        tracks_y = tracks_y.clone()
        if bool(getattr(env.cfg, "oa_flat_visual_yellow_track_use_hard_gate", False)):
            cg = float(getattr(env.cfg, "oa_flat_visual_yellow_track_conf_gate", 0.02))
            gated = (cval > cg).to(dtype=tracks_y.dtype)
            tracks_y[..., 0:3] = tracks_y[..., 0:3] * gated * cval
        else:
            # Peak-level confidence can be ~0 while the column histogram still shows yellow (pooling vs argmax).
            # Hard zeroing (old default) removed vision-aligned bearing while hist_* stayed hot — confusing for the policy.
            hp_thr = float(getattr(env.cfg, "oa_flat_visual_yellow_hist_peak_boost_thr", 0.085))
            cfloor = float(getattr(env.cfg, "oa_flat_visual_yellow_hist_peak_boost_floor", 0.22))
            hist_peak_env = hist_y.max(dim=1).values
            hb = max(int(hist_peak_env.shape[0]), 1)
            rhy = (ne + hb - 1) // hb
            hist_peak_ne = hist_peak_env.repeat(rhy)[:ne].view(ne, 1, 1).expand(ne, k, 1).to(dtype=cval.dtype)
            boost = (hist_peak_ne > hp_thr).to(dtype=cval.dtype) * cfloor
            scale = torch.sqrt((cval + boost).clamp(min=0.0, max=1.0))
            tracks_y[..., 0:3] = tracks_y[..., 0:3] * scale
    else:
        tracks_y = torch.zeros(ne, k, feat, device=device)

    if int(hist.shape[0]) != ne:
        hb = max(int(hist.shape[0]), 1)
        r = (ne + hb - 1) // hb
        hist = hist.repeat(r, 1)[:ne].contiguous()
    if int(hist_y.shape[0]) != ne:
        hy = max(int(hist_y.shape[0]), 1)
        ry = (ne + hy - 1) // hy
        hist_y = hist_y.repeat(ry, 1)[:ne].contiguous()

    u._oa_flat_visual_tracks = tracks.reshape(ne, -1)
    det_count = (tracks[:, :, 3] > 0.25).sum(dim=1, keepdim=True).float()
    u._oa_flat_visual_counts = torch.cat(
        [
            det_count * 0.1,
            tracks[:, :, 3].mean(dim=1, keepdim=True),
            tracks[:, :, 3].max(dim=1, keepdim=True).values,
            hist[:, :1].mean(dim=1, keepdim=True),
        ],
        dim=1,
    )
    u._oa_flat_visual_confidence = tracks[:, :4, 3].clamp(0.0, 1.0)
    u._oa_flat_visual_bearing_hist = hist

    u._oa_flat_visual_yellow_goal_tracks = tracks_y.reshape(ne, -1)
    u._oa_flat_visual_yellow_goal_confidence = tracks_y[:, :4, 3].clamp(0.0, 1.0)
    u._oa_flat_visual_yellow_goal_bearing_hist = hist_y
    det_count_y = (tracks_y[:, :, 3] > 0.25).sum(dim=1, keepdim=True).float()
    u._oa_flat_visual_yellow_goal_counts = torch.cat(
        [
            det_count_y * 0.1,
            tracks_y[:, :, 3].mean(dim=1, keepdim=True),
            tracks_y[:, :, 3].max(dim=1, keepdim=True).values,
            hist_y[:, :1].mean(dim=1, keepdim=True),
        ],
        dim=1,
    )


def _oa_flat_visual_yellow_enrichment_on(env_cfg) -> bool:
    return bool(getattr(env_cfg, "oa_flat_visual_yellow_goal_enrichment_enabled", True))


def oa_flat_visual_yellow_goal_detection_stats(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    max_detections: int | None = None,
) -> torch.Tensor:
    """Yellow-blob aggregate stats parallel to ``_oa_flat_visual_counts`` (blue); 4 scalars."""
    _ = asset_cfg
    if not _oa_flat_visual_yellow_enrichment_on(env.cfg):
        return torch.zeros(env.num_envs, 0, device=env.device)
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    kd = int(max_detections) if max_detections is not None else int(
        getattr(env.cfg, "oa_flat_visual_yellow_goal_max_detections", 6)
    )
    if kd <= 0:
        return torch.zeros(env.num_envs, 0, device=env.device)
    update_oa_flat_blue_detections_from_camera(
        env,
        sensor_name=sensor_name,
        max_detections=kd,
        compute_blue=True,
        compute_yellow_goal=True,
    )
    u = env.unwrapped
    return getattr(u, "_oa_flat_visual_yellow_goal_counts", torch.zeros(env.num_envs, 4, device=env.device))


def oa_flat_visual_yellow_goal_bearing_distribution(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    max_detections: int | None = None,
) -> torch.Tensor:
    """Normalized 20-bin yellow bearing histogram parallel to ``_oa_flat_visual_bearing_hist`` (blue)."""
    _ = asset_cfg
    if not _oa_flat_visual_yellow_enrichment_on(env.cfg):
        return torch.zeros(env.num_envs, 0, device=env.device)
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    kd = int(max_detections) if max_detections is not None else int(
        getattr(env.cfg, "oa_flat_visual_yellow_goal_max_detections", 6)
    )
    if kd <= 0:
        return torch.zeros(env.num_envs, 0, device=env.device)
    update_oa_flat_blue_detections_from_camera(
        env,
        sensor_name=sensor_name,
        max_detections=kd,
        compute_blue=True,
        compute_yellow_goal=True,
    )
    u = env.unwrapped
    return getattr(u, "_oa_flat_visual_yellow_goal_bearing_hist", torch.zeros(env.num_envs, 20, device=env.device))


def oa_flat_visual_yellow_goal_vs_blue_slot0(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    max_detections: int | None = None,
) -> torch.Tensor:
    """Compare strongest yellow blob vs strongest blue blob: sinΔ, cosΔ, tanh(conf diff), tanh(range diff).

    Gives a camera-relative cue analogous to stacked blue aggregates, but anchored on goal-vs-obstacle
    disagreement in bearing and detector confidence."""
    _ = asset_cfg
    if not _oa_flat_visual_yellow_enrichment_on(env.cfg):
        return torch.zeros(env.num_envs, 0, device=env.device)
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    kd = int(max_detections) if max_detections is not None else int(
        getattr(env.cfg, "oa_flat_visual_yellow_goal_max_detections", 6)
    )
    if kd <= 0:
        return torch.zeros(env.num_envs, 0, device=env.device)
    update_oa_flat_blue_detections_from_camera(
        env,
        sensor_name=sensor_name,
        max_detections=kd,
        compute_blue=True,
        compute_yellow_goal=True,
    )
    ne = env.num_envs
    u = env.unwrapped
    ksz = kd
    y_flat = getattr(u, "_oa_flat_visual_yellow_goal_tracks", torch.zeros(ne, ksz * 4, device=env.device))
    b_flat = getattr(u, "_oa_flat_visual_tracks", torch.zeros(ne, ksz * 4, device=env.device))
    yt = y_flat.reshape(ne, ksz, 4)
    bt = b_flat.reshape(ne, ksz, 4)
    sy, cy = yt[:, 0, 0], yt[:, 0, 1]
    ry, fy = yt[:, 0, 2], yt[:, 0, 3]
    sb, cb = bt[:, 0, 0], bt[:, 0, 1]
    rb, fb = bt[:, 0, 2], bt[:, 0, 3]
    sin_d = (sy * cb - cy * sb).clamp(-1.0, 1.0)
    cos_d = (cy * cb + sy * sb).clamp(-1.0, 1.0)
    cdf = torch.tanh((fy - fb) * 3.0)
    rdf = torch.tanh((ry - rb) * 3.0)
    return torch.stack([sin_d, cos_d, cdf, rdf], dim=-1)


def oa_flat_visual_yellow_goal_tracks(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    max_detections: int | None = None,
) -> torch.Tensor:
    """RGB+D yellow-blob (goal marker) cues: ``k`` × (sin, cos, tanh(range), confidence) flattened."""
    _ = asset_cfg
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    kd = int(max_detections) if max_detections is not None else int(getattr(env.cfg, "oa_flat_visual_yellow_goal_max_detections", 6))
    if kd <= 0:
        return torch.zeros(env.num_envs, 0, device=env.device)
    update_oa_flat_blue_detections_from_camera(
        env,
        sensor_name=sensor_name,
        max_detections=kd,
        compute_blue=True,
        compute_yellow_goal=True,
    )
    return env.unwrapped._oa_flat_visual_yellow_goal_tracks


def oa_flat_visual_detection_counts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    update_oa_flat_blue_detections_from_camera(env, sensor_name=sensor_name)
    return env.unwrapped._oa_flat_visual_counts


def oa_flat_visual_confidence_stats(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    update_oa_flat_blue_detections_from_camera(env, sensor_name=sensor_name)
    return env.unwrapped._oa_flat_visual_confidence


def oa_flat_visual_bearing_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    _ = asset_cfg
    sensor_name = str(getattr(env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
    update_oa_flat_blue_detections_from_camera(env, sensor_name=sensor_name)
    return env.unwrapped._oa_flat_visual_bearing_hist


def oa_flat_imu_yaw_rate_noisy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2:3] + 0.01 * torch.randn_like(asset.data.root_ang_vel_w[:, 2:3])


def oa_flat_imu_lin_accel_noisy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_w[:, :3]
    u = env.unwrapped
    if not hasattr(u, "_oa_flat_prev_lin_vel_imu"):
        u._oa_flat_prev_lin_vel_imu = v.clone()
    a = (v - u._oa_flat_prev_lin_vel_imu) / (env.step_dt + 1.0e-8)
    u._oa_flat_prev_lin_vel_imu = v.clone()
    return a + 0.02 * torch.randn_like(a)


def oa_flat_optical_flow_forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    g = _quat_body_forward_xy_w(asset)
    v = asset.data.root_lin_vel_w[:, :2]
    return torch.sum(v * g, dim=-1, keepdim=True) * 0.05


def oa_flat_optical_flow_yaw_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, 2:3] * 0.05


def oa_flat_wheel_odometry_estimate_flat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("rover")) -> torch.Tensor:
    """Planar wheel odom (world XY) — same idea as localization_observations but +Z world up."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids, _ = asset.find_joints(
        ["wheel_fr_joint", "wheel_fl_joint", "wheel_rr_joint", "wheel_rl_joint"], preserve_order=True
    )
    wv = asset.data.joint_vel[:, ids]
    u = env.unwrapped
    if not hasattr(u, "_oa_flat_wheel_odom"):
        u._oa_flat_wheel_odom = torch.zeros(env.num_envs, 3, device=env.device)
    tw = float(getattr(env.cfg, "flat_drive_track_width_m", 0.48))
    r = 0.08
    wl = ((wv[:, 1] + wv[:, 3]) * 0.5).unsqueeze(-1)
    wr = ((wv[:, 0] + wv[:, 2]) * 0.5).unsqueeze(-1)
    v = r * (wl + wr) * 0.5
    wz = r * (wr - wl) / (tw + 1.0e-8)
    th = u._oa_flat_wheel_odom[:, 2:3] + wz * env.step_dt
    u._oa_flat_wheel_odom[:, 0:1] += v * torch.cos(th) * env.step_dt
    u._oa_flat_wheel_odom[:, 1:2] += v * torch.sin(th) * env.step_dt
    u._oa_flat_wheel_odom[:, 2:3] = th
    o = u._oa_flat_wheel_odom
    return torch.tanh(torch.cat([o, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1) * 0.02)

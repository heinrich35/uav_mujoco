# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Reset: rover spawn + blue boxes first, then goal sample with clearance to blue boxes."""

from __future__ import annotations

import math
import random

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.rover.mdp.flat_goal_events import (
    _bump_per_env_episode_idx,
    _ensure_goal_buffers,
    _per_env_rand_draws,
)

_OA_RELOC_SALT = 7_025_913


def _sync_flat_blue_boxes(
    env: ManagerBasedEnv,
    eids: torch.Tensor,
    box_count: int,
) -> None:
    """Write world poses for kinematic blue boxes (identity orientation, +Z up)."""
    try:
        assets = env.scene["oa_blue_boxes"]
    except KeyError:
        return
    u = env.unwrapped
    device = env.device
    ne = int(eids.shape[0])
    n_obj = int(assets.num_objects)
    if ne == 0 or n_obj < 1 or box_count <= 0:
        return
    poses = torch.zeros(ne, n_obj, 7, device=device)
    poses[..., 3] = 1.0
    poses[..., 2] = -2000.0
    vel = torch.zeros(ne, n_obj, 6, device=device)
    boxes = u._oa_flat_box_positions_w[eids, :box_count]
    poses[:, :box_count, 0:3] = boxes
    poses[:, :box_count, 3] = 1.0
    poses[:, :box_count, 4:7] = 0.0
    assets.write_object_pose_to_sim(poses, env_ids=eids)
    assets.write_object_velocity_to_sim(vel, env_ids=eids)


def _sample_grid_boxes_xy(
    n: int,
    spawn_xy: torch.Tensor,
    *,
    arena_half_m: float,
    grid_step_m: float,
    n_boxes: int,
    min_spawn_clear_m: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-env ``n_boxes`` XY offsets in env-local frame on a jittered grid (vectorized)."""
    n_per_side = max(2, int(2.0 * arena_half_m / grid_step_m) + 1)
    xs = torch.linspace(-arena_half_m, arena_half_m, n_per_side, device=device, dtype=dtype)
    ys = torch.linspace(-arena_half_m, arena_half_m, n_per_side, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    flat = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    g = flat.shape[0]
    jitter = (torch.rand(n, g, 2, device=device, dtype=dtype) - 0.5) * (0.35 * grid_step_m)
    pts = flat.unsqueeze(0).expand(n, -1, -1) + jitter
    d_spawn = torch.linalg.norm(pts - spawn_xy[:, None, :], dim=-1)
    valid = d_spawn >= float(min_spawn_clear_m)
    rnd = torch.rand(n, g, device=device, dtype=dtype)
    rnd = torch.where(valid, rnd, torch.full_like(rnd, -1.0))
    order = torch.argsort(rnd, dim=1, descending=True)
    n_take = min(int(n_boxes), int(g))
    if n_take < 1:
        return torch.zeros(n, int(n_boxes), 2, device=device, dtype=dtype)
    ar = torch.arange(n, device=device).unsqueeze(1).expand(n, n_take)
    pick = order[:, :n_take]
    out = pts[ar, pick]
    if n_take < int(n_boxes):
        pad = int(n_boxes) - n_take
        out = torch.cat([out, out[:, -1:, :].expand(n, pad, 2)], dim=1)
    return out


def _target_clear_of_boxes(
    cand_x: torch.Tensor,
    cand_y: torch.Tensor,
    boxes_rel: torch.Tensor,
    clearance_m: float,
) -> torch.Tensor:
    """``boxes_rel`` is (n, nb, 2) env-local centers; candidates (n,)."""
    dx = cand_x[:, None] - boxes_rel[:, :, 0]
    dy = cand_y[:, None] - boxes_rel[:, :, 1]
    d = torch.sqrt(dx * dx + dy * dy + 1.0e-12)
    return torch.all(d >= float(clearance_m), dim=1)


def _repair_bad_targets(
    tx: torch.Tensor,
    ty: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    *,
    th: float,
    ex: float,
    md: float,
    boxes_rel: torch.Tensor,
    clear_m: float,
    device: torch.device,
    dtype: torch.dtype,
    iters: int = 288,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch.rand fallback matching ``continuous_relocate`` edge-case handler + box clearance."""
    outside_ok = (torch.abs(tx) > ex) | (torch.abs(ty) > ex)
    dist_ok = torch.sqrt((tx - sx) ** 2 + (ty - sy) ** 2 + 1.0e-12) >= md
    box_ok = _target_clear_of_boxes(tx, ty, boxes_rel, clear_m)
    bad = ~(outside_ok & dist_ok & box_ok)
    if not bool(torch.any(bad)):
        return tx, ty
    bi = bad.nonzero(as_tuple=False).flatten()
    lb = int(bi.numel())
    fsx = sx[bi]
    fsy = sy[bi]
    ftx = tx[bi].clone()
    fty = ty[bi].clone()
    for _ in range(iters):
        cx = (torch.rand(lb, device=device, dtype=dtype) * 2.0 - 1.0) * th
        cy = (torch.rand(lb, device=device, dtype=dtype) * 2.0 - 1.0) * th
        outside = (torch.abs(cx) > ex) | (torch.abs(cy) > ex)
        dd = torch.sqrt((cx - fsx) ** 2 + (cy - fsy) ** 2 + 1.0e-12)
        br = boxes_rel[bi]
        ok = outside & (dd >= md) & _target_clear_of_boxes(cx, cy, br, clear_m)
        ftx = torch.where(ok, cx, ftx)
        fty = torch.where(ok, cy, fty)
        vo = (torch.abs(ftx) > ex) | (torch.abs(fty) > ex)
        vd = torch.sqrt((ftx - fsx) ** 2 + (fty - fsy) ** 2 + 1.0e-12) >= md
        vb = _target_clear_of_boxes(ftx, fty, br, clear_m)
        if bool(torch.all(vo & vd & vb)):
            break
    still_bad = ~(
        ((torch.abs(ftx) > ex) | (torch.abs(fty) > ex))
        & (torch.sqrt((ftx - fsx) ** 2 + (fty - fsy) ** 2 + 1.0e-12) >= md)
        & _target_clear_of_boxes(ftx, fty, br, clear_m)
    )
    if bool(torch.any(still_bad)):
        sb = still_bad.nonzero(as_tuple=False).flatten()
        ang = torch.rand(sb.numel(), device=device, dtype=dtype) * (2.0 * math.pi)
        rad = torch.full((sb.numel(),), float(th) * 0.92, device=device, dtype=dtype)
        rtx = torch.cos(ang) * rad
        rty = torch.sin(ang) * rad
        inner = (torch.abs(rtx) <= ex) & (torch.abs(rty) <= ex)
        push = float(ex) + max(float(md), 0.5)
        sgx = torch.where(torch.rand(sb.numel(), device=device, dtype=dtype) > 0.5, 1.0, -1.0)
        sgy = torch.where(torch.rand(sb.numel(), device=device, dtype=dtype) > 0.5, 1.0, -1.0)
        rtx = torch.where(inner, sgx * push, rtx)
        rty = torch.where(inner, sgy * push, rty)
        ftx[sb] = rtx
        fty[sb] = rty
    tx2 = tx.clone()
    ty2 = ty.clone()
    tx2[bi] = ftx
    ty2[bi] = fty
    return tx2, ty2


def _oa_flat_relocate_goal_at_rover(
    env: ManagerBasedEnv,
    move_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    *,
    target_half_size_m: float,
    target_exclusion_half_size_m: float,
    target_min_spawn_dist_m: float,
    spawn_target_marker: bool,
    marker_asset_name: str,
    max_target_samples: int,
    oa_flat_box_count: int,
    oa_flat_target_min_blue_box_clearance_m: float,
) -> None:
    """Sample a new planar goal for ``move_ids`` (already filtered); updates target buffers + marker pose."""
    if move_ids.numel() == 0:
        return
    u = env.unwrapped
    device = env.device
    m = int(move_ids.numel())
    rover: Articulation = env.scene[asset_cfg.name]
    origins_m = env.scene.env_origins[move_ids, :3]
    rel = rover.data.root_pos_w[move_ids, :2] - origins_m[:, :2]
    sx = rel[:, 0]
    sy = rel[:, 1]

    th = float(target_half_size_m)
    ex = float(target_exclusion_half_size_m)
    md = float(target_min_spawn_dist_m)
    ms = int(max_target_samples)
    nb_cfg = int(oa_flat_box_count)
    nb_buf = int(u._oa_flat_box_positions_w.shape[1]) if hasattr(u, "_oa_flat_box_positions_w") else 0
    nb = min(nb_cfg, nb_buf) if nb_buf > 0 else nb_cfg
    dtype = origins_m.dtype
    boxes_rel = torch.zeros(m, nb, 2, device=device, dtype=dtype)
    if hasattr(u, "_oa_flat_box_positions_w") and nb > 0:
        boxes_rel = u._oa_flat_box_positions_w[move_ids, :nb, :2] - origins_m[:, None, :2]
    clear_m = float(oa_flat_target_min_blue_box_clearance_m)

    num_d = 2 * ms
    dr = _per_env_rand_draws(env, move_ids, m, device, num_d, salt=_OA_RELOC_SALT)
    tx = torch.zeros(m, device=device, dtype=dtype)
    ty = torch.zeros(m, device=device, dtype=dtype)
    for si in range(ms):
        cand_x = (dr[:, 2 * si] * 2.0 - 1.0) * th
        cand_y = (dr[:, 2 * si + 1] * 2.0 - 1.0) * th
        outside = (torch.abs(cand_x) > ex) | (torch.abs(cand_y) > ex)
        dplan = torch.sqrt((cand_x - sx) ** 2 + (cand_y - sy) ** 2 + 1.0e-12)
        ok_box = _target_clear_of_boxes(cand_x, cand_y, boxes_rel, clear_m)
        ok = outside & (dplan >= md) & ok_box
        tx = torch.where(ok, cand_x, tx)
        ty = torch.where(ok, cand_y, ty)
        if bool(torch.all(ok)):
            break

    tx, ty = _repair_bad_targets(tx, ty, sx, sy, th=th, ex=ex, md=md, boxes_rel=boxes_rel, clear_m=clear_m, device=device, dtype=dtype)

    u._flat_goal_target_xy[move_ids, 0] = tx
    u._flat_goal_target_xy[move_ids, 1] = ty
    u._flat_goal_prev_dist[move_ids] = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + 1.0e-12)
    u._flat_goal_bonus_given[move_ids] = False

    u._flat_goal_segment_start_episode_steps[move_ids] = env.episode_length_buf[move_ids].clone()

    if spawn_target_marker and marker_asset_name in env.scene.keys():
        marker: RigidObject = env.scene[marker_asset_name]
        o = env.scene.env_origins[move_ids, :3]
        _sz = float(getattr(env.cfg, "spawn_z_m", 0.2))
        _lift = float(getattr(env.cfg, "oa_flat_goal_marker_z_above_spawn_m", 0.35))
        mpos = o.clone()
        mpos[:, 0] += tx
        mpos[:, 1] += ty
        mpos[:, 2] = o[:, 2] + _sz + _lift
        mq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(m, -1)
        with torch.inference_mode():
            marker.write_root_pose_to_sim(torch.cat([mpos, mq], dim=-1), env_ids=move_ids)

            # Force the internal goal buffer to exactly match the pose we just wrote
            # (prevents goal_rel vs marker_w mismatch caused by sampling / origin bugs).
            with torch.inference_mode():
                actual = marker.data.root_pos_w[move_ids, :2] - origins_m[:, :2]
                u._flat_goal_target_xy[move_ids] = actual


def oa_flat_relocate_goal_if_pending(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    spawn_half_size_m: float = 15.0,
    target_half_size_m: float = 23.0,
    target_exclusion_half_size_m: float = 15.5,
    target_min_spawn_dist_m: float = 5.0,
    spawn_target_marker: bool = True,
    marker_asset_name: str = "goal_marker",
    max_target_samples: int = 64,
    oa_flat_box_count: int = 70,
    oa_flat_target_min_blue_box_clearance_m: float = 3.0,
):
    """Interval event: relocate the goal when ``goal_reached_bonus`` raised ``_oa_flat_relocate_goal_pending``."""
    if env_ids is None:
        eids_all = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        eids_all = env_ids.to(dtype=torch.long)
    if eids_all.numel() == 0:
        return
    device = env.device
    _ensure_goal_buffers(env, device)
    u = env.unwrapped
    pending = u._oa_flat_relocate_goal_pending[eids_all]
    if not bool(torch.any(pending)):
        return
    move_ids = eids_all[pending]
    u._oa_flat_relocate_goal_pending[move_ids] = False
    _oa_flat_relocate_goal_at_rover(
        env,
        move_ids,
        asset_cfg,
        target_half_size_m=target_half_size_m,
        target_exclusion_half_size_m=target_exclusion_half_size_m,
        target_min_spawn_dist_m=target_min_spawn_dist_m,
        spawn_target_marker=spawn_target_marker,
        marker_asset_name=marker_asset_name,
        max_target_samples=max_target_samples,
        oa_flat_box_count=oa_flat_box_count,
        oa_flat_target_min_blue_box_clearance_m=oa_flat_target_min_blue_box_clearance_m,
    )


def reset_oa_flat_episode(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    spawn_half_size_m: float = 15.0,
    target_half_size_m: float = 23.0,
    target_exclusion_half_size_m: float = 15.5,
    target_min_spawn_dist_m: float = 5.0,
    spawn_z_m: float = 0.2,
    spawn_target_marker: bool = True,
    marker_asset_name: str = "goal_marker",
    max_target_samples: int = 64,
    oa_flat_box_count: int = 70,
    oa_flat_box_arena_half_m: float = 15.0,
    oa_flat_box_grid_step_m: float = 4.0,
    oa_flat_box_min_spawn_clear_m: float = 4.0,
    oa_flat_box_min_pair_clear_m: float = 1.5,
    oa_flat_box_half_height_m: float = 0.25,
    oa_flat_target_min_blue_box_clearance_m: float = 3.0,
):
    """Sample rover pose, lay out blue obstacles, **then** pick a planar goal honoring box clearance."""
    u0 = env.unwrapped
    play_seed = getattr(u0, "_oa_flat_play_layout_seed", None)
    if play_seed is not None:
        s = int(play_seed)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        random.seed(s)
        try:
            delattr(u0, "_oa_flat_play_layout_seed")
        except AttributeError:
            pass

    if isinstance(env_ids, slice):
        eids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        eids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if eids.numel() == 0:
        return

    device = env.device
    _ensure_goal_buffers(env, device)
    u = env.unwrapped
    rover: Articulation = env.scene[asset_cfg.name]
    n = int(eids.shape[0])
    origins = env.scene.env_origins[eids, :3]
    if origins.dim() != 2 or origins.shape[-1] != 3:
        origins = origins.reshape(n, 3)

    sh = float(spawn_half_size_m)
    ex_sz = float(target_exclusion_half_size_m)
    th_t = float(target_half_size_m)
    md = float(target_min_spawn_dist_m)
    ms = int(max_target_samples)
    nb = int(oa_flat_box_count)
    dtype = origins.dtype

    # --- 1) Spawn only (reuse per-env RNG stream salt=0 pattern from flat_goal) ---
    d_spawn = _per_env_rand_draws(env, eids, n, device, 3, salt=0)
    sx = (d_spawn[:, 0] * 2.0 - 1.0) * sh
    sy = (d_spawn[:, 1] * 2.0 - 1.0) * sh
    yaw = (d_spawn[:, 2] * 2.0 - 1.0) * math.pi
    spawn_xy = torch.stack([sx, sy], dim=-1)

    # --- 2) Blue boxes (after spawn XY is fixed) ---
    xy_local = _sample_grid_boxes_xy(
        n,
        spawn_xy,
        arena_half_m=float(oa_flat_box_arena_half_m),
        grid_step_m=float(oa_flat_box_grid_step_m),
        n_boxes=nb,
        min_spawn_clear_m=float(oa_flat_box_min_spawn_clear_m),
        device=device,
        dtype=dtype,
    )
    z = origins[:, 2:3] + float(spawn_z_m) + float(oa_flat_box_half_height_m)
    boxes_w = torch.zeros(n, nb, 3, device=device, dtype=dtype)
    boxes_w[:, :, 0:2] = origins[:, None, 0:2] + xy_local
    boxes_w[:, :, 2:3] = z.unsqueeze(1).expand(-1, nb, -1)

    for _ in range(4):
        dmat = torch.linalg.norm(boxes_w[:, :, None, :2] - boxes_w[:, None, :, :2], dim=-1)
        eye = torch.eye(nb, device=device, dtype=torch.bool).unsqueeze(0).expand(n, -1, -1)
        dmat = dmat.masked_fill(eye, 1.0e9)
        viol = (dmat < float(oa_flat_box_min_pair_clear_m)).any(dim=-1).any(dim=-1)
        if not bool(viol.any()):
            break
        dirs = torch.randn(n, nb, 2, device=device, dtype=dtype)
        dirs = torch.nn.functional.normalize(dirs + 1.0e-6, dim=-1)
        boxes_w[:, :, 0:2] = boxes_w[:, :, 0:2] + viol[:, None, None] * dirs * 0.5

    if not hasattr(u, "_oa_flat_box_positions_w") or u._oa_flat_box_positions_w.shape[1] != nb:
        u._oa_flat_box_positions_w = torch.zeros(env.num_envs, nb, 3, device=device, dtype=dtype)
    u._oa_flat_box_positions_w[eids] = boxes_w
    u._box_positions_w = u._oa_flat_box_positions_w

    boxes_rel = boxes_w[:, :, :2] - origins[:, None, :2]

    clear_m = float(oa_flat_target_min_blue_box_clearance_m)

    # --- 3) Goal planar sample (knows obstacle layout; salt distinct from spawn) ---
    d_tgt = _per_env_rand_draws(env, eids, n, device, 2 * ms, salt=901_337)
    tx = torch.zeros(n, device=device, dtype=dtype)
    ty = torch.zeros(n, device=device, dtype=dtype)
    for si in range(ms):
        cand_x = (d_tgt[:, 2 * si] * 2.0 - 1.0) * th_t
        cand_y = (d_tgt[:, 2 * si + 1] * 2.0 - 1.0) * th_t
        outside = (torch.abs(cand_x) > ex_sz) | (torch.abs(cand_y) > ex_sz)
        dplan = torch.sqrt((cand_x - sx) ** 2 + (cand_y - sy) ** 2 + 1.0e-12)
        ok_box = _target_clear_of_boxes(cand_x, cand_y, boxes_rel, clear_m)
        ok = outside & (dplan >= md) & ok_box
        tx = torch.where(ok, cand_x, tx)
        ty = torch.where(ok, cand_y, ty)
        if bool(torch.all(ok)):
            break

    tx, ty = _repair_bad_targets(tx, ty, sx, sy, th=th_t, ex=ex_sz, md=md, boxes_rel=boxes_rel, clear_m=clear_m, device=device, dtype=dtype)

    u._flat_goal_target_xy[eids, 0] = tx
    u._flat_goal_target_xy[eids, 1] = ty

    pos_w = origins.clone()
    pos_w[:, 0] += sx
    pos_w[:, 1] += sy
    pos_w[:, 2] = float(spawn_z_m)
    quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root_pose = torch.cat([pos_w, quat], dim=-1)
    vel = torch.zeros(n, 6, device=device, dtype=dtype)
    _jp = rover.data.default_joint_pos[eids].clone()
    _jv = rover.data.default_joint_vel[eids].clone()
    with torch.inference_mode():
        rover.write_root_pose_to_sim(root_pose, env_ids=eids)
        rover.write_root_velocity_to_sim(vel, env_ids=eids)
        rover.write_joint_state_to_sim(_jp, _jv, env_ids=eids)

        if spawn_target_marker and marker_asset_name in env.scene.keys():
            marker: RigidObject = env.scene[marker_asset_name]
            mpos = origins.clone()
            mpos[:, 0] += tx
            mpos[:, 1] += ty
            _lift = float(getattr(env.cfg, "oa_flat_goal_marker_z_above_spawn_m", 0.35))
            mpos[:, 2] = origins[:, 2] + float(spawn_z_m) + _lift
            mq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
            marker.write_root_pose_to_sim(torch.cat([mpos, mq], dim=-1), env_ids=eids)

    dist0 = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + 1.0e-12)
    u._flat_goal_prev_dist[eids] = dist0
    u._flat_goal_bonus_given[eids] = False
    u._oa_flat_relocate_goal_pending[eids] = False
    # Time-to-goal shaping segment: full episode reset → age measured from post-reset step 0.
    u._flat_goal_segment_start_episode_steps[eids] = 0

    if hasattr(u, "_oa_flat_wheel_odom"):
        u._oa_flat_wheel_odom[eids] = 0.0

    _bump_per_env_episode_idx(env, eids)
    _sync_flat_blue_boxes(env, eids, nb)


def oa_flat_continuous_relocate_goal_if_close(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    goal_threshold_m: float = 5.0,
    target_half_size_m: float = 23.0,
    target_exclusion_half_size_m: float = 15.5,
    target_min_spawn_dist_m: float = 5.0,
    spawn_half_size_m: float = 15.0,
    spawn_target_marker: bool = True,
    marker_asset_name: str = "goal_marker",
    max_target_samples: int = 64,
    oa_flat_box_count: int = 70,
    oa_flat_target_min_blue_box_clearance_m: float = 3.0,
):
    """Same as ``continuous_relocate_goal_if_close`` plus minimum planar clearance to blue box centers."""
    if env_ids is None:
        eids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        eids = env_ids.to(dtype=torch.long)
    if eids.numel() == 0:
        return

    device = env.device
    _ensure_goal_buffers(env, device)
    u = env.unwrapped

    rover: Articulation = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[eids, :3]
    rel = rover.data.root_pos_w[eids, :2] - origins[:, :2]
    tgt = u._flat_goal_target_xy[eids]
    dist = torch.linalg.norm(rel - tgt, dim=-1)
    close = dist < float(goal_threshold_m)
    if not bool(torch.any(close)):
        return

    move_ids = eids[close]
    _oa_flat_relocate_goal_at_rover(
        env,
        move_ids,
        asset_cfg,
        target_half_size_m=float(target_half_size_m),
        target_exclusion_half_size_m=float(target_exclusion_half_size_m),
        target_min_spawn_dist_m=float(target_min_spawn_dist_m),
        spawn_target_marker=spawn_target_marker,
        marker_asset_name=marker_asset_name,
        max_target_samples=int(max_target_samples),
        oa_flat_box_count=int(oa_flat_box_count),
        oa_flat_target_min_blue_box_clearance_m=float(oa_flat_target_min_blue_box_clearance_m),
    )

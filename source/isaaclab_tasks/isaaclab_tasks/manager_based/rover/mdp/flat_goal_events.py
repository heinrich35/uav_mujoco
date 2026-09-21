# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Reset-time sampling for flat XY goal navigation (per-env target + rover spawn)."""

from __future__ import annotations

import math

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

_PRIME_E = 1_000_003
_PRIME_K = 48_271
_RELOC_SALT = 7_000_000
_PLAY_RESPAWN_SALT = 9_000_000


def materialize_planar_goal_dist_buffer(t: torch.Tensor) -> torch.Tensor:
    """Return a plain (non–inference-mode) copy suitable for persistent env buffers.

    ``play.py`` wraps stepping in ``torch.inference_mode()``; assignments like
    ``buf = distance.detach().clone()`` replace the **whole** buffer with an inference tensor.
    Slice writes into an existing buffer stay plain tensors; whole-tensor replacement does not.

    Downstream :func:`_ensure_goal_buffers` must not wipe valid distances when it sees
    ``torch.is_inference(buf)`` — materialize instead (same numeric values).
    """
    fn = getattr(torch, "is_inference", None)
    td = t.detach()
    if fn is None:
        return td.clone()
    if fn(td):
        return torch.tensor(td.tolist(), device=td.device, dtype=td.dtype)
    return td.clone()


def _per_env_streams_on(env: ManagerBasedEnv) -> bool:
    return bool(getattr(env.cfg, "oa_flat_per_env_seed_streams", False))


def _ensure_per_env_episode_idx(env: ManagerBasedEnv, device: torch.device) -> None:
    u = env.unwrapped
    if not hasattr(u, "_oa_flat_per_env_episode_idx") or u._oa_flat_per_env_episode_idx.shape[0] != env.num_envs:
        u._oa_flat_per_env_episode_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)


def _bump_per_env_episode_idx(env: ManagerBasedEnv, eids: torch.Tensor) -> None:
    if not _per_env_streams_on(env):
        return
    _ensure_per_env_episode_idx(env, env.device)
    env.unwrapped._oa_flat_per_env_episode_idx[eids] += 1


def _per_env_rand_draws(
    env: ManagerBasedEnv,
    eids: torch.Tensor,
    n: int,
    device: torch.device,
    num_floats: int,
    *,
    salt: int = 0,
) -> torch.Tensor:
    """Shape ``(n, num_floats)`` U(0,1). Per-env streams when ``oa_flat_per_env_seed_streams`` is set on cfg."""
    if not _per_env_streams_on(env):
        return torch.rand(n, num_floats, device=device)
    base = int(getattr(env.cfg, "seed", 0) or 0)
    _ensure_per_env_episode_idx(env, device)
    u = env.unwrapped
    ep = u._oa_flat_per_env_episode_idx
    out = torch.empty(n, num_floats, device=device)
    e_list = eids.reshape(-1).tolist()
    g = torch.Generator(device=device)
    for i in range(n):
        ei = int(e_list[i])
        g.manual_seed(base + ei * _PRIME_E + int(ep[ei].item()) * _PRIME_K + int(salt))
        out[i] = torch.rand(num_floats, generator=g, device=device)
    return out


def _ensure_goal_buffers(env: ManagerBasedEnv, device: torch.device) -> None:
    u = env.unwrapped
    n = env.num_envs
    if not hasattr(u, "_flat_goal_target_xy") or u._flat_goal_target_xy.shape != (n, 2):
        u._flat_goal_target_xy = torch.zeros(n, 2, device=device)
    _prev = getattr(u, "_flat_goal_prev_dist", None)
    _infer_fn = getattr(torch, "is_inference", None)
    _prev_bad = _infer_fn is not None and _prev is not None and _infer_fn(_prev)
    if not hasattr(u, "_flat_goal_prev_dist") or u._flat_goal_prev_dist.shape != (n,):
        u._flat_goal_prev_dist = torch.zeros(n, device=device)
    elif _prev_bad:
        # Was: zeros — destroyed real distances after play-mode inference tensors hit this path.
        u._flat_goal_prev_dist = materialize_planar_goal_dist_buffer(_prev)
    if not hasattr(u, "_flat_goal_bonus_given") or u._flat_goal_bonus_given.shape != (n,):
        u._flat_goal_bonus_given = torch.zeros(n, dtype=torch.bool, device=device)
    elif _infer_fn is not None and _infer_fn(u._flat_goal_bonus_given):
        u._flat_goal_bonus_given = materialize_planar_goal_dist_buffer(u._flat_goal_bonus_given)
    if not hasattr(u, "_flat_goal_segment_start_episode_steps") or u._flat_goal_segment_start_episode_steps.shape != (n,):
        u._flat_goal_segment_start_episode_steps = torch.zeros(n, device=device, dtype=torch.long)
    if not hasattr(u, "_oa_flat_relocate_goal_pending") or u._oa_flat_relocate_goal_pending.shape != (n,):
        u._oa_flat_relocate_goal_pending = torch.zeros(n, dtype=torch.bool, device=device)
    elif _infer_fn is not None and _infer_fn(u._oa_flat_relocate_goal_pending):
        u._oa_flat_relocate_goal_pending = materialize_planar_goal_dist_buffer(u._oa_flat_relocate_goal_pending)


def reset_flat_goal_episode(
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
):
    """Sample rover XY/yaw in the spawn square and a target outside the exclusion square; optional marker sync."""
    if isinstance(env_ids, slice):
        eids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        eids = env_ids.to(dtype=torch.long)

    if eids.numel() == 0:
        return

    device = env.device
    _ensure_goal_buffers(env, device)
    u = env.unwrapped

    rover: Articulation = env.scene[asset_cfg.name]
    n = eids.shape[0]
    origins = env.scene.env_origins[eids, :3]

    sh = float(spawn_half_size_m)
    ex_sz = float(target_exclusion_half_size_m)
    th = float(target_half_size_m)
    md = float(target_min_spawn_dist_m)
    ms = int(max_target_samples)
    num_d = 3 + 2 * ms
    d = _per_env_rand_draws(env, eids, n, device, num_d, salt=0)
    sx = (d[:, 0] * 2.0 - 1.0) * sh
    sy = (d[:, 1] * 2.0 - 1.0) * sh
    yaw = (d[:, 2] * 2.0 - 1.0) * math.pi

    tx = torch.zeros(n, device=device)
    ty = torch.zeros(n, device=device)
    for si in range(ms):
        cand_x = (d[:, 3 + 2 * si] * 2.0 - 1.0) * th
        cand_y = (d[:, 4 + 2 * si] * 2.0 - 1.0) * th
        outside = (torch.abs(cand_x) > ex_sz) | (torch.abs(cand_y) > ex_sz)
        dplan = torch.sqrt((cand_x - sx) ** 2 + (cand_y - sy) ** 2 + 1.0e-12)
        ok = outside & (dplan >= md)
        tx = torch.where(ok, cand_x, tx)
        ty = torch.where(ok, cand_y, ty)
        if bool(torch.all(ok)):
            break

    u._flat_goal_target_xy[eids, 0] = tx
    u._flat_goal_target_xy[eids, 1] = ty

    pos_w = origins.clone()
    pos_w[:, 0] += sx
    pos_w[:, 1] += sy
    pos_w[:, 2] = float(spawn_z_m)

    quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root_pose = torch.cat([pos_w, quat], dim=-1)
    vel = torch.zeros(n, 6, device=device)
    _jp = rover.data.default_joint_pos[eids].clone()
    _jv = rover.data.default_joint_vel[eids].clone()
    # Play / input callbacks can run outside ``torch.inference_mode()`` while articulation
    # buffers are inference tensors from the last policy step; sim writes must use the same mode.
    with torch.inference_mode():
        rover.write_root_pose_to_sim(root_pose, env_ids=eids)
        rover.write_root_velocity_to_sim(vel, env_ids=eids)
        rover.write_joint_state_to_sim(_jp, _jv, env_ids=eids)

        if spawn_target_marker and marker_asset_name in env.scene.keys():
            marker: RigidObject = env.scene[marker_asset_name]
            mpos = origins.clone()
            mpos[:, 0] += tx
            mpos[:, 1] += ty
            mpos[:, 2] = float(spawn_z_m) + 0.35
            mq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n, -1)
            marker.write_root_pose_to_sim(torch.cat([mpos, mq], dim=-1), env_ids=eids)

    dist0 = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + 1.0e-12)
    u._flat_goal_prev_dist[eids] = dist0
    u._flat_goal_bonus_given[eids] = False

    _bump_per_env_episode_idx(env, eids)


def randomize_rover_spawn_pose_only(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    spawn_half_size_m: float = 15.0,
    spawn_z_m: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
):
    """New random XY/yaw in the spawn square only; does not move goal marker or cached target XY."""
    if isinstance(env_ids, slice):
        eids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        eids = env_ids.to(dtype=torch.long)
    if eids.numel() == 0:
        return

    device = env.device
    rover: Articulation = env.scene[asset_cfg.name]
    n = eids.shape[0]
    origins = env.scene.env_origins[eids, :3]
    sh = float(spawn_half_size_m)
    d3 = _per_env_rand_draws(env, eids, n, device, 3, salt=_PLAY_RESPAWN_SALT)
    sx = (d3[:, 0] * 2.0 - 1.0) * sh
    sy = (d3[:, 1] * 2.0 - 1.0) * sh
    yaw = (d3[:, 2] * 2.0 - 1.0) * math.pi
    pos_w = origins.clone()
    pos_w[:, 0] += sx
    pos_w[:, 1] += sy
    pos_w[:, 2] = float(spawn_z_m)
    quat = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    root_pose = torch.cat([pos_w, quat], dim=-1)
    vel = torch.zeros(n, 6, device=device)
    _jp = rover.data.default_joint_pos[eids].clone()
    _jv = rover.data.default_joint_vel[eids].clone()
    with torch.inference_mode():
        rover.write_root_pose_to_sim(root_pose, env_ids=eids)
        rover.write_root_velocity_to_sim(vel, env_ids=eids)
        rover.write_joint_state_to_sim(_jp, _jv, env_ids=eids)


def continuous_relocate_goal_if_close(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rover"),
    goal_threshold_m: float = 1.0,
    target_half_size_m: float = 12.5,
    target_exclusion_half_size_m: float = 10.0,
    target_min_spawn_dist_m: float = 3.0,
    spawn_half_size_m: float = 10.0,
    spawn_target_marker: bool = True,
    marker_asset_name: str = "goal_marker",
    max_target_samples: int = 64,
):
    """Interval-mode: move the kinematic goal marker when the rover is within ``goal_threshold_m`` (play mode)."""
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
    m = move_ids.shape[0]
    sx = rel[close, 0]
    sy = rel[close, 1]

    th = float(target_half_size_m)
    ex = float(target_exclusion_half_size_m)
    md = float(target_min_spawn_dist_m)
    ms = int(max_target_samples)

    num_d = 2 * ms
    dr = _per_env_rand_draws(env, move_ids, m, device, num_d, salt=_RELOC_SALT)
    tx = torch.zeros(m, device=device)
    ty = torch.zeros(m, device=device)
    for si in range(ms):
        cand_x = (dr[:, 2 * si] * 2.0 - 1.0) * th
        cand_y = (dr[:, 2 * si + 1] * 2.0 - 1.0) * th
        outside = (torch.abs(cand_x) > ex) | (torch.abs(cand_y) > ex)
        dplan = torch.sqrt((cand_x - sx) ** 2 + (cand_y - sy) ** 2 + 1.0e-12)
        ok = outside & (dplan >= md)
        tx = torch.where(ok, cand_x, tx)
        ty = torch.where(ok, cand_y, ty)
        if bool(torch.all(ok)):
            break

    # Rejection can fail (e.g. rover deep inside spawn): fix any env with invalid (tx,ty) or still inside exclusion.
    outside_ok = (torch.abs(tx) > ex) | (torch.abs(ty) > ex)
    dist_ok = torch.sqrt((tx - sx) ** 2 + (ty - sy) ** 2 + 1.0e-12) >= md
    bad = ~(outside_ok & dist_ok)
    if bool(torch.any(bad)):
        bi = bad.nonzero(as_tuple=False).flatten()
        lb = int(bi.numel())
        fsx = sx[bi]
        fsy = sy[bi]
        ftx = tx[bi].clone()
        fty = ty[bi].clone()
        for _ in range(256):
            cx = (torch.rand(lb, device=device) * 2.0 - 1.0) * th
            cy = (torch.rand(lb, device=device) * 2.0 - 1.0) * th
            outside = (torch.abs(cx) > ex) | (torch.abs(cy) > ex)
            dd = torch.sqrt((cx - fsx) ** 2 + (cy - fsy) ** 2 + 1.0e-12)
            ok = outside & (dd >= md)
            ftx = torch.where(ok, cx, ftx)
            fty = torch.where(ok, cy, fty)
            vo = (torch.abs(ftx) > ex) | (torch.abs(fty) > ex)
            vd = torch.sqrt((ftx - fsx) ** 2 + (fty - fsy) ** 2 + 1.0e-12) >= md
            if bool(torch.all(vo & vd)):
                break
        still_bad = ~((torch.abs(ftx) > ex) | (torch.abs(fty) > ex)) | (
            torch.sqrt((ftx - fsx) ** 2 + (fty - fsy) ** 2 + 1.0e-12) < md
        )
        if bool(torch.any(still_bad)):
            sb = still_bad.nonzero(as_tuple=False).flatten()
            ang = torch.rand(sb.numel(), device=device) * (2.0 * math.pi)
            rad = torch.full((sb.numel(),), float(th) * 0.92, device=device)
            rtx = torch.cos(ang) * rad
            rty = torch.sin(ang) * rad
            inner = (torch.abs(rtx) <= ex) & (torch.abs(rty) <= ex)
            push = float(ex) + max(float(md), 0.5)
            sgx = torch.where(torch.rand(sb.numel(), device=device) > 0.5, 1.0, -1.0)
            sgy = torch.where(torch.rand(sb.numel(), device=device) > 0.5, 1.0, -1.0)
            rtx = torch.where(inner, sgx * push, rtx)
            rty = torch.where(inner, sgy * push, rty)
            ftx[sb] = rtx
            fty[sb] = rty
        tx[bi] = ftx
        ty[bi] = fty

    u._flat_goal_target_xy[move_ids, 0] = tx
    u._flat_goal_target_xy[move_ids, 1] = ty
    u._flat_goal_prev_dist[move_ids] = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + 1.0e-12)
    u._flat_goal_bonus_given[move_ids] = False

    if spawn_target_marker and marker_asset_name in env.scene.keys():
        marker: RigidObject = env.scene[marker_asset_name]
        o = env.scene.env_origins[move_ids, :3]
        mpos = o.clone()
        mpos[:, 0] += tx
        mpos[:, 1] += ty
        mpos[:, 2] = o[:, 2] + 0.35
        mq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(m, -1)
        with torch.inference_mode():
            marker.write_root_pose_to_sim(torch.cat([mpos, mq], dim=-1), env_ids=move_ids)

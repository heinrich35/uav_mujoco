# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Apply Disk-World–style physparam latch (p0–p9) to the OA-flat rover.

Uses ``scripts/rover/rover_utils.py`` for mass/friction mapping (see physparam latch doc).
p4 (cylindrical gravity scale in disk-world) is logged as ``_cylindrical_gravity_scale`` but does not retarget
global gravity at runtime (single shared gravity in Isaac Lab; changing it would affect all envs unevenly).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _rover_utils():
    sp = str(_repo_root() / "scripts" / "rover")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import rover_utils as ru  # noqa: PLC0415

    return ru


def _ensure_physparam_cache(env: ManagerBasedEnv, rover: Articulation) -> None:
    u = env.unwrapped
    if getattr(u, "_oa_physparam_cache_ready", False):
        return
    base_ids, _ = rover.find_bodies(["base_link"])
    u._phy_base_body_idx = int(base_ids[0])
    wh = ("wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl")
    wheel_ids, _ = rover.find_bodies(list(wh))
    u._phy_wheel_body_indices = [int(x) for x in wheel_ids]
    jids, _ = rover.find_joints(
        ["wheel_fr_joint", "wheel_fl_joint", "wheel_rr_joint", "wheel_rl_joint"], preserve_order=True
    )
    u._phy_wheel_joint_indices = [int(x) for x in jids]

    masses = rover.root_physx_view.get_masses().clone().to(env.device)
    inertias = rover.root_physx_view.get_inertias().clone().to(env.device)
    bi = u._phy_base_body_idx
    u._phy_base_mass_ref = masses[:, bi].clone()
    u._phy_base_inertia_ref = inertias[:, bi, :].clone()
    wm = u._phy_wheel_body_indices
    u._phy_wheel_mass_ref = masses[:, wm].clone()
    u._phy_wheel_inertia_ref = inertias[:, wm, :].clone()

    u._phy_default_joint_damping = rover.data.default_joint_damping[:, u._phy_wheel_joint_indices].clone().to(env.device)
    if hasattr(rover.data, "joint_viscous_friction_coeff") and rover.data.joint_viscous_friction_coeff is not None:
        u._phy_wheel_viscous_ref = rover.data.joint_viscous_friction_coeff[:, u._phy_wheel_joint_indices].clone().to(env.device)
    else:
        u._phy_wheel_viscous_ref = torch.zeros(
            env.num_envs, len(u._phy_wheel_joint_indices), device=env.device, dtype=torch.float32
        )
    u._oa_physparam_cache_ready = True


def _lerp_map(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = 0.5 * (x + 1.0)
    return lo + t.clamp(0.0, 1.0) * (hi - lo)


def apply_oa_flat_physparam_latch(
    env: ManagerBasedEnv,
    rover: Articulation,
    env_ids: torch.Tensor,
    p: torch.Tensor,
) -> None:
    """``p`` shape ``(n, 10)`` in [-1, 1] (use ``tanh`` on raw actions)."""
    ru = _rover_utils()
    _ensure_physparam_cache(env, rover)
    u = env.unwrapped
    dev = env.device
    n = int(env_ids.shape[0])
    if n < 1:
        return
    if p.shape != (n, 10):
        raise ValueError(f"expected p shape (n, 10), got {tuple(p.shape)}")

    bidx = u._phy_base_body_idx
    masses = rover.root_physx_view.get_masses().clone().to(dev)
    ref_b = masses[env_ids, bidx]
    bmax = torch.tensor([ru.base_mass_max_scale_for_ref_kg(ref_b[i : i + 1]) for i in range(n)], device=dev, dtype=ref_b.dtype)
    smin, smax = float(ru.BASE_MASS_SCALE_MIN), float(ru.BASE_MASS_SCALE_MAX)
    b_scale = _lerp_map(p[:, 0], smin, smax)
    b_scale = torch.minimum(b_scale, bmax)

    wm_ref = u._phy_wheel_mass_ref[env_ids]
    wmax = torch.tensor(
        [ru.wheel_link_mass_max_scale_for_ref_kg(wm_ref[i : i + 1]) for i in range(n)],
        device=dev,
        dtype=wm_ref.dtype,
    )
    wmin, wbig = float(ru.WHEEL_LINK_MASS_SCALE_MIN), float(ru.WHEEL_LINK_MASS_SCALE_MAX)
    w_scale = _lerp_map(p[:, 1], wmin, wbig)
    w_scale = torch.minimum(w_scale, wmax)

    mu_s = _lerp_map(p[:, 2], 0.4, 9.0)
    mu_d = _lerp_map(p[:, 3], 0.35, 8.5)
    mu_d = torch.minimum(mu_d, mu_s - 0.05).clamp(min=0.05)

    g_scale = _lerp_map(p[:, 4], 0.55, 1.25)

    fmin, fmax = float(ru.WHEEL_CMD_SCALE_MIN), float(ru.WHEEL_CMD_SCALE_MAX)
    u.episode_forward_cmd_scale[env_ids] = _lerp_map(p[:, 5], fmin, fmax)
    u.episode_turn_cmd_scale[env_ids] = _lerp_map(p[:, 6], fmin, fmax)
    smin_s, smax_s = float(ru.WHEEL_VEL_TARGET_SLEW_MIN), float(ru.WHEEL_VEL_TARGET_SLEW_MAX)
    u.episode_wheel_target_max_slew[env_ids] = _lerp_map(p[:, 7], smin_s, smax_s)

    jids = u._phy_wheel_joint_indices
    nw = len(jids)
    ms = mu_s.unsqueeze(-1).expand(n, nw)
    md = mu_d.unsqueeze(-1).expand(n, nw)
    visc_mul = _lerp_map(p[:, 8], 0.1, 5.0)
    cv = u._phy_wheel_viscous_ref[env_ids] * visc_mul.unsqueeze(-1)
    if ru.isaac_sim_has_joint_friction_triplet():
        rover.write_joint_friction_coefficient_to_sim(ms, md, cv, joint_ids=jids, env_ids=env_ids)
    else:
        rover.write_joint_friction_coefficient_to_sim(ms, joint_ids=jids, env_ids=env_ids)

    for i in range(n):
        ei = int(env_ids[i].item())
        ru.apply_rover_base_mass_scale(
            rover,
            bidx,
            float(b_scale[i].item()),
            u._phy_base_mass_ref[ei : ei + 1],
            u._phy_base_inertia_ref[ei : ei + 1],
        )
        ru.apply_wheel_link_mass_scale(
            rover,
            u._phy_wheel_body_indices,
            float(w_scale[i].item()),
            u._phy_wheel_mass_ref[ei : ei + 1],
            u._phy_wheel_inertia_ref[ei : ei + 1],
        )

    damp_mul = _lerp_map(p[:, 9], 0.25, 4.0)
    damp_rows = u._phy_default_joint_damping[env_ids] * damp_mul.unsqueeze(-1)
    rover.write_joint_damping_to_sim(damp_rows, joint_ids=jids, env_ids=env_ids)

    u._applied_base_mass_scale = torch.ones(env.num_envs, device=dev)
    u._applied_wheel_mass_scale = torch.ones(env.num_envs, device=dev)
    u._applied_mu_s = torch.zeros(env.num_envs, device=dev)
    u._applied_mu_d = torch.zeros(env.num_envs, device=dev)
    u._cylindrical_gravity_scale = torch.ones(env.num_envs, device=dev)
    u._applied_wheel_viscous_scale = torch.ones(env.num_envs, device=dev)
    u._applied_wheel_damping_scale = torch.ones(env.num_envs, device=dev)
    u._applied_base_mass_scale[env_ids] = b_scale
    u._applied_wheel_mass_scale[env_ids] = w_scale
    u._applied_mu_s[env_ids] = mu_s
    u._applied_mu_d[env_ids] = mu_d
    u._cylindrical_gravity_scale[env_ids] = g_scale
    u._applied_wheel_viscous_scale[env_ids] = visc_mul
    u._applied_wheel_damping_scale[env_ids] = damp_mul

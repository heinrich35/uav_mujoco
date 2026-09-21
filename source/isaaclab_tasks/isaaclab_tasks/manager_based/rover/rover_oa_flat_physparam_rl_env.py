# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""OA-flat + episode physparam latch buffers (12-D policy actions)."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.rover.rover_oa_flat_rl_env import RoverOAFlatNavRLEnv


class RoverOAFlatPhysparamNavRLEnv(RoverOAFlatNavRLEnv):
    """Like :class:`RoverOAFlatNavRLEnv` but initializes physparam latch + per-episode drive scale state."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        d = self.device
        n = self.num_envs
        u = self.unwrapped
        u._oa_physparam_latch_pending = torch.ones(n, dtype=torch.bool, device=d)
        u.episode_forward_cmd_scale = torch.ones(n, device=d)
        u.episode_turn_cmd_scale = torch.ones(n, device=d)
        u.episode_wheel_target_max_slew = torch.full(
            (n,), float(self.cfg.actions.drive.max_target_slew_rad_s2), device=d
        )
        # Physparam metric buffers — normally set by apply_oa_flat_physparam_latch.
        # Ensure they exist so logging always works (even before first latch).
        self._init_physparam_metric_buffers(n)

    def _init_physparam_metric_buffers(self, n: int) -> None:
        """Ensure metric tensors exist (p0-p9 mean values)."""
        u = self.unwrapped
        for attr in (
            "_applied_base_mass_scale",
            "_applied_wheel_mass_scale",
            "_applied_mu_s",
            "_applied_mu_d",
            "_cylindrical_gravity_scale",
            "_applied_wheel_viscous_scale",
            "_applied_wheel_damping_scale",
        ):
            if not hasattr(u, attr):
                setattr(u, attr, torch.ones(n, device=self.device))

    def _log_physparam_metrics(self, env_ids: torch.Tensor) -> None:
        """Write Episode_RoverPhysparam/* metrics to extras['log'] for training_progress_log."""
        u = self.unwrapped
        log = self.extras.setdefault("log", {})
        metrics = {
            "Episode_RoverPhysparam/mean_base_mass_scale": u._applied_base_mass_scale,
            "Episode_RoverPhysparam/mean_wheel_mass_scale": u._applied_wheel_mass_scale,
            "Episode_RoverPhysparam/mean_mu_s": u._applied_mu_s,
            "Episode_RoverPhysparam/mean_mu_d": u._applied_mu_d,
            "Episode_RoverPhysparam/mean_gravity_scale": u._cylindrical_gravity_scale,
            "Episode_RoverPhysparam/mean_forward_cmd_scale": u.episode_forward_cmd_scale,
            "Episode_RoverPhysparam/mean_turn_cmd_scale": u.episode_turn_cmd_scale,
            "Episode_RoverPhysparam/mean_wheel_target_slew": u.episode_wheel_target_max_slew,
            "Episode_RoverPhysparam/mean_wheel_viscous_scale": u._applied_wheel_viscous_scale,
            "Episode_RoverPhysparam/mean_wheel_damping_scale": u._applied_wheel_damping_scale,
        }
        for key, tensor in metrics.items():
            if tensor is not None and tensor.numel() > 0:
                log[key] = float(tensor.detach().float().mean().item())

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        e = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if e.numel() < 1:
            return
        u = self.unwrapped
        u._oa_physparam_latch_pending[e] = True
        u.episode_forward_cmd_scale[e] = 1.0
        u.episode_turn_cmd_scale[e] = 1.0
        u.episode_wheel_target_max_slew[e] = float(self.cfg.actions.drive.max_target_slew_rad_s2)

        # Log physparam metrics to extras["log"] for training_progress_log.txt.
        # The latched values for newly-reset envs will show defaults (1.0) until
        # the next process_actions applies the new latch. This is deliberate:
        # a mean across all envs gives a representative picture of the policy output.
        self._log_physparam_metrics(e)

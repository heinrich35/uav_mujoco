# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""12-D action: physparam latch (p0–p9) then differential drive."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.rover.mdp.oa_flat_physparam_apply import apply_oa_flat_physparam_latch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


class PhysparamLatchDifferentialDriveAction(ActionTerm):
    """First 10 controls latched once per episode; last 2 = differential drive."""

    cfg: "PhysparamLatchDifferentialDriveActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "PhysparamLatchDifferentialDriveActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        with torch.inference_mode(False):
            self._raw_actions = torch.zeros(self.num_envs, 12, device=self.device)
            self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)
            self._prev_cmd = torch.zeros(self.num_envs, 2, device=self.device)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            ["wheel_fr_joint", "wheel_fl_joint", "wheel_rr_joint", "wheel_rl_joint"], preserve_order=True
        )
        with torch.inference_mode(False):
            self._wheel_omega = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self):
        from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

        self._IO_descriptor.shape = (12,)
        self._IO_descriptor.dtype = str(self._raw_actions.dtype)
        self._IO_descriptor.action_type = "PhysparamLatchDifferentialDriveAction"
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        u = self._env.unwrapped
        pending = u._oa_physparam_latch_pending
        need = pending.nonzero(as_tuple=False).squeeze(-1)
        if need.numel() > 0:
            apply_oa_flat_physparam_latch(self._env, self._asset, need, torch.tanh(self._raw_actions[need, :10]))
            pending[need] = False

        fwd_s = u.episode_forward_cmd_scale
        turn_s = u.episode_turn_cmd_scale
        slew = u.episode_wheel_target_max_slew

        cmd = torch.clamp(self._raw_actions[:, 10:12], -1.0, 1.0)
        max_fwd = cmd[:, 0:1] * (self.cfg.max_forward_vel * fwd_s.unsqueeze(-1))
        yaw = cmd[:, 1:2] * (self.cfg.max_turn_diff_vel * turn_s.unsqueeze(-1))
        max_d = slew.unsqueeze(-1) * self._env.step_dt
        dy = yaw - self._prev_cmd[:, 1:2]
        dy = torch.clamp(dy, -max_d, max_d)
        yaw_l = self._prev_cmd[:, 1:2] + dy
        cmd_xy_inf = torch.cat([max_fwd, yaw_l], dim=-1)
        with torch.inference_mode(False):
            cmd_xy = cmd_xy_inf.clone()
            self._prev_cmd = cmd_xy
            self._processed_actions = cmd_xy.clone()

        v = self._processed_actions[:, 0:1]
        w = self._processed_actions[:, 1:2]
        tw = self.cfg.track_width_m
        rr = self.cfg.wheel_radius_m
        v_l = v - 0.5 * tw * w
        v_r = v + 0.5 * tw * w
        omega_fr = (v_r / rr).squeeze(-1)
        omega_fl = (v_l / rr).squeeze(-1)
        omega_rr = (v_r / rr).squeeze(-1)
        omega_rl = (v_l / rr).squeeze(-1)
        om_inf = torch.stack([omega_fr, omega_fl, omega_rr, omega_rl], dim=-1)
        with torch.inference_mode(False):
            self._wheel_omega = om_inf.clone()

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._wheel_omega, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_cmd[env_ids] = 0.0
        self._wheel_omega[env_ids] = 0.0


@configclass
class PhysparamLatchDifferentialDriveActionCfg(ActionTermCfg):
    """See :class:`PhysparamLatchDifferentialDriveAction`."""

    class_type = PhysparamLatchDifferentialDriveAction
    asset_name: str = "rover"

    max_forward_vel: float = 2.25
    max_turn_diff_vel: float = 9.0
    max_target_slew_rad_s2: float = 720.0
    wheel_radius_m: float = 0.08
    track_width_m: float = 0.48

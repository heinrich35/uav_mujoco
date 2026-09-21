# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Differential drive action for four-wheel skid-steer style rover."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from .actions_cfg import DifferentialDriveActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


class DifferentialDriveAction(ActionTerm):
    """Maps 2-D normalized actions to wheel joint velocity targets (rad/s)."""

    cfg: DifferentialDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        # Env ctor / play can run under ``torch.inference_mode()``; those tensors are not
        # in-place writable on reset unless we materialize outside inference mode.
        with torch.inference_mode(False):
            self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self._prev_cmd = torch.zeros(self.num_envs, 2, device=self.device)
        # wheel_fr, wheel_fl, wheel_rr, wheel_rl
        self._joint_ids, self._joint_names = self._asset.find_joints(
            ["wheel_fr_joint", "wheel_fl_joint", "wheel_rr_joint", "wheel_rl_joint"], preserve_order=True
        )
        with torch.inference_mode(False):
            self._wheel_omega = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

        self._IO_descriptor.shape = (2,)
        self._IO_descriptor.dtype = str(self._raw_actions.dtype)
        self._IO_descriptor.action_type = "DifferentialDriveAction"
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        # clamp normalized command
        cmd = torch.clamp(self._raw_actions, -1.0, 1.0)
        fwd = cmd[:, 0:1] * self.cfg.max_forward_vel
        yaw = cmd[:, 1:2] * self.cfg.max_turn_diff_vel
        # slew limit on yaw command (per policy step)
        max_d = self.cfg.max_target_slew_rad_s2 * self._env.step_dt
        dy = yaw - self._prev_cmd[:, 1:2]
        dy = torch.clamp(dy, -max_d, max_d)
        yaw_l = self._prev_cmd[:, 1:2] + dy
        # Under inference mode, ``.clone()`` still yields inference tensors; nested
        # ``inference_mode(False)`` produces buffers that reset can slice-assign.
        cmd_xy_inf = torch.cat([fwd, yaw_l], dim=-1)
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
        # FL/RL left, FR/RR right (matches joint order fl, fr, rl, rr — remap)
        # joint order: fr, fl, rr, rl
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

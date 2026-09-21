# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Position estimation action for rover localization task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RoverPositionEstimationAction(ActionTerm):
    """Agent action = estimated rover 3-D position (world frame)."""

    cfg: "RoverPositionEstimationActionCfg"

    def __init__(self, cfg: "RoverPositionEstimationActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._env.unwrapped._rover_estimated_position = actions.clone()

    def apply_actions(self):
        pass  # estimation only; does not move the rover

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # Initialize guess near origin (reasonable default since rover spawns near centre)
        self._raw_actions[env_ids] = 0.0


@configclass
class RoverPositionEstimationActionCfg(ActionTermCfg):
    """Configuration for :class:`RoverPositionEstimationAction`."""

    class_type = RoverPositionEstimationAction
    asset_name: str = "rover"

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


class ActorCriticAuxObstacleDistance(ActorCritic):
    """Actor-critic with an auxiliary head predicting normalized nearest-obstacle distance from actor observations."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        aux_obstacle_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        **kwargs: dict[str, Any],
    ) -> None:
        aux_hid = list(aux_obstacle_hidden_dims)
        kwargs.pop("aux_obstacle_hidden_dims", None)
        activation = str(kwargs.get("activation", "elu"))
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        num_actor_obs = 0
        for grp in obs_groups["policy"]:
            num_actor_obs += int(obs[grp].shape[-1])
        self.aux_obstacle_distance_head = MLP(num_actor_obs, 1, aux_hid, activation)

    def predict_aux_obstacle_distance(self, obs: TensorDict) -> torch.Tensor:
        x = self.get_actor_obs(obs)
        x = self.actor_obs_normalizer(x)
        return self.aux_obstacle_distance_head(x).squeeze(-1)

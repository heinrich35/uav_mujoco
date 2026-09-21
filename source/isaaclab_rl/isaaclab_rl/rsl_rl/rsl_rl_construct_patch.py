# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Patches ``rsl_rl`` OnPolicyRunner to construct Isaac Lab obstacle-aux PPO without ``eval()`` on custom classes."""

from __future__ import annotations

import warnings
from typing import Any

from tensordict import TensorDict

from rsl_rl.algorithms import PPO


_ORIGINAL_CONSTRUCT = None


def _construct_aux_obstacle_algorithm(self: Any, obs: TensorDict) -> PPO:
    # rsl_rl: these live in ``rsl_rl.modules`` (not ``rsl_rl.utils``) — match OnPolicyRunner imports.
    from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config

    from isaaclab_rl.rsl_rl.aux_obstacle_actor_critic import ActorCriticAuxObstacleDistance
    from isaaclab_rl.rsl_rl.aux_obstacle_ppo import PPOAuxObstacleDistance

    self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
    self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

    if self.cfg.get("empirical_normalization") is not None:
        warnings.warn(
            "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
            "`critic_obs_normalization` as part of the `policy` configuration instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.policy_cfg.get("actor_obs_normalization") is None:
            self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
        if self.policy_cfg.get("critic_obs_normalization") is None:
            self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

    self.policy_cfg.pop("class_name")
    actor_critic = ActorCriticAuxObstacleDistance(
        obs,
        self.cfg["obs_groups"],
        self.env.num_actions,
        **self.policy_cfg,
    ).to(self.device)

    self.alg_cfg.pop("class_name")
    alg = PPOAuxObstacleDistance(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

    alg.init_storage(
        "rl",
        self.env.num_envs,
        self.num_steps_per_env,
        obs,
        [self.env.num_actions],
    )
    return alg


def _patched_construct_algorithm(self: Any, obs: TensorDict) -> PPO:
    global _ORIGINAL_CONSTRUCT
    assert _ORIGINAL_CONSTRUCT is not None
    pname = self.policy_cfg.get("class_name")
    aname = self.alg_cfg.get("class_name")
    if pname == "ActorCriticAuxObstacleDistance" and aname == "PPOAuxObstacleDistance":
        return _construct_aux_obstacle_algorithm(self, obs)
    # RslRlPpoAlgorithmCfg / RslRlPpoActorCriticCfg always serialize aux obstacle fields with defaults.
    # Upstream rsl_rl.PPO rejects unknown kwargs; ActorCritic only warns. Strip when not using aux classes.
    self.alg_cfg.pop("aux_obstacle_distance_coef", None)
    if pname != "ActorCriticAuxObstacleDistance":
        self.policy_cfg.pop("aux_obstacle_hidden_dims", None)
    return _ORIGINAL_CONSTRUCT(self, obs)


def apply_rsl_rl_aux_obstacle_construct_patch() -> None:
    """Idempotent: swaps ``OnPolicyRunner._construct_algorithm`` to route aux obstacle classes."""
    global _ORIGINAL_CONSTRUCT
    from rsl_rl.runners import on_policy_runner as m

    if getattr(m.OnPolicyRunner, "_isaaclab_aux_obstacle_construct_v1", False):
        return
    if _ORIGINAL_CONSTRUCT is None:
        _ORIGINAL_CONSTRUCT = m.OnPolicyRunner._construct_algorithm
    m.OnPolicyRunner._construct_algorithm = _patched_construct_algorithm
    m.OnPolicyRunner._isaaclab_aux_obstacle_construct_v1 = True

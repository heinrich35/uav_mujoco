# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RoverDiskWorldObstaclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 4000
    save_interval = 200
    experiment_name = "rover_disk_world_obstacle_avoidance"
    empirical_normalization = False
    # Explicit keys silence rsl_rl resolve_obs_groups deprecation warnings (critic = policy obs for this task).
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0015,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoverOAFlatNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for OA-flat: asymmetric critic (privileged obstacle geometry), depth scan for policy, aux distance head."""

    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 100
    experiment_name = "rover_oa_flat_env"
    empirical_normalization = False
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy", "privileged"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticAuxObstacleDistance",
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        aux_obstacle_hidden_dims=(128, 64),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOAuxObstacleDistance",
        aux_obstacle_distance_coef=0.05,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoverOAFlatPhysparamNavPPORunnerCfg(RoverOAFlatNavPPORunnerCfg):
    """Same network + aux head as OA-flat; 12-D action space (episode physparam latch + differential drive)."""

    experiment_name = "rover_oa_flat_physparam_env"


@configclass
class RoverFlatWorldGoalNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO defaults for flat-world goal navigation (overridden by ``train.py`` / shell scripts)."""

    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 200
    experiment_name = "rover_flat_world_goal_nav"
    empirical_normalization = False
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoverLocalizationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for rover localization: estimate 3-D position from camera + wheel odometry."""

    num_steps_per_env = 32
    max_iterations = 4000
    save_interval = 100
    experiment_name = "rover_localization_env"
    empirical_normalization = False
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rover localization RL env: frozen expert OA-Flat policy drives rover; agent estimates 3-D position."""

from __future__ import annotations

import os
import re
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.rover.mdp.actions import DifferentialDriveAction
from isaaclab_tasks.manager_based.rover.mdp.actions_cfg import DifferentialDriveActionCfg


class _ExpertOADrivePolicy(nn.Module):
    """Lightweight actor-only wrapper loaded from an OA-Flat RSL-RL checkpoint.

    Reconstructs the actor MLP and observation normalizer directly from the saved
    ``model_state_dict`` so we do not need the full ``ActorCriticAuxObstacleDistance``
    class or a TensorDict observation layout at inference time.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)

        # Build actor layers by inspecting saved weight shapes
        layer_indices = sorted({int(m.group(1)) for k in sd if (m := re.match(r"actor\.(\d+)\.weight", k))})
        layers: list[nn.Module] = []
        for i, idx in enumerate(layer_indices):
            w = sd[f"actor.{idx}.weight"]
            b = sd[f"actor.{idx}.bias"]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=True)
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers.append(lin)
            if i < len(layer_indices) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

        # Observation normalizer (same logic as rsl_rl RunningMeanStd)
        self.register_buffer("obs_mean", sd["actor_obs_normalizer._mean"].squeeze(0))
        self.register_buffer("obs_var", sd["actor_obs_normalizer._var"].squeeze(0))
        self.register_buffer("obs_std", sd["actor_obs_normalizer._std"].squeeze(0))

        # Action std (not used for deterministic inference but kept for completeness)
        self.register_buffer("action_std", sd.get("std", torch.zeros(2, device=device)))

        self.to(device)
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return self.net(norm)

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.forward(obs)


class RoverLocalizationRLEnv(ManagerBasedRLEnv):
    """Localization environment: frozen OA-Flat expert drives, agent estimates 3-D position.

    The agent's action space is 3-D: ``(x_est, y_est, z_est)`` — a continuous guess of the
    rover's world-frame position. The true position is **never** exposed in observations;
    it is used only for the reward function.

    A frozen expert OA-Flat policy runs in the background and produces the differential-drive
    commands that actually move the rover toward the goal. The expert's drive action is kept
    entirely separate from the agent's estimate action.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Private differential-drive action term for the frozen expert policy
        drive_cfg = DifferentialDriveActionCfg(
            class_type=DifferentialDriveAction,
            asset_name="rover",
            max_forward_vel=float(getattr(cfg, "flat_drive_max_forward_vel_m_s", 2.25)),
            max_turn_diff_vel=float(getattr(cfg, "flat_drive_max_turn_yaw_rate_rad_s", 9.0)),
            max_target_slew_rad_s2=float(getattr(cfg, "flat_wheel_target_slew_rad_s2", 720.0)),
            wheel_radius_m=0.08,
            track_width_m=0.48,
        )
        self._expert_drive = DifferentialDriveAction(drive_cfg, self)
        self._expert_drive.reset()

        # Load frozen expert policy
        self._expert_policy = None
        self._load_expert_policy()

        # Cache for the expert's last action (used by observations)
        self._expert_last_action = torch.zeros(self.num_envs, 2, device=self.device)

        # Debug visualization: red sphere at the agent's estimated position
        if getattr(self.cfg, "debug_mode", False):
            self._estimated_target_visual = self.scene["estimated_target_visual"]
        else:
            self._estimated_target_visual = None

    def _load_expert_policy(self):
        expert_path = getattr(self.cfg, "expert_policy_path", "")
        if not expert_path or not os.path.isfile(expert_path):
            print(f"[WARNING] Expert policy not found at: {expert_path}. Rover will not move.")
            return
        try:
            self._expert_policy = _ExpertOADrivePolicy(expert_path, self.device)
            print(f"[INFO] Expert OA-Flat policy loaded from: {expert_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load expert policy: {e}")
            self._expert_policy = None

    def _get_expert_observations(self) -> torch.Tensor:
        """Build the flat observation vector expected by the expert OA-Flat policy.

        Because the environment's single observation group already mirrors the OA-Flat
        policy layout (including the expert's last drive action), we can simply compute
        the ``policy`` group.
        """
        obs = self.observation_manager.compute_group("policy", update_history=False)
        if isinstance(obs, dict):
            obs = obs.get("policy", obs)
        return obs

    def step(self, action: torch.Tensor):
        """Execute one environment step.

        Args:
            action: Agent's 3-D position estimate ``(x_est, y_est, z_est)``.

        Returns:
            obs, reward, terminated, truncated, info
        """
        # ---- 1. Process the agent's estimate action ----
        self.action_manager.process_action(action.to(self.device))

        # Update debug sphere to show the agent's guess (policy outputs env-local
        # coordinates, so we add env_origins to place the sphere in world frame).
        if self._estimated_target_visual is not None:
            est_local = self.unwrapped._rover_estimated_position
            est_world = est_local + self.scene.env_origins[:, :3]
            mq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4)
            self._estimated_target_visual.write_root_pose_to_sim(torch.cat([est_world, mq], dim=-1))

        # ---- 2. Compute expert observations & run expert policy ----
        if self._expert_policy is not None:
            # Ensure sensors and scene state are fresh before the expert reads observations
            self.scene.update(dt=self.physics_dt)
            expert_obs = self._get_expert_observations()
            expert_action = self._expert_policy.act(expert_obs, deterministic=True)
            # Scale raw policy outputs (large values from this checkpoint) into the
            # normalized [-1, 1] range expected by DifferentialDriveAction.
            expert_action = torch.clamp(expert_action / 60.0, -1.0, 1.0)
            self._expert_drive.process_actions(expert_action)
            # Store the drive term's raw_actions (clamped to action-space bounds by the
            # action manager in OA-Flat) so the ``last_action`` observation matches the
            # training distribution exactly.
            self._expert_last_action[:] = self._expert_drive.raw_actions.clone()

            # Lightweight diagnostic (every ~30 steps when running with few envs).
            # Helps see whether the expert actually receives a reasonable goal distance.
            if self.num_envs <= 4:
                self._expert_dbg_counter = getattr(self, "_expert_dbg_counter", 0) + 1
                if self._expert_dbg_counter % 30 == 0 and hasattr(self.unwrapped, "_flat_goal_target_xy"):
                    u = self.unwrapped
                    tgt = u._flat_goal_target_xy
                    rover = self.scene["rover"]
                    rel = rover.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                    gvec = tgt - rel
                    gdist = torch.linalg.norm(gvec, dim=-1)
                    for i in range(min(self.num_envs, 2)):
                        marker_w = (0.0, 0.0)
                        if "goal_marker" in self.scene.keys():
                            m = self.scene["goal_marker"].data.root_pos_w[i, :2]
                            marker_w = (float(m[0]), float(m[1]))
                        print(
                            f"[EXPERT] env{i}: goal_rel=({tgt[i,0]:.1f},{tgt[i,1]:.1f}) "
                            f"marker_w=({marker_w[0]:.1f},{marker_w[1]:.1f}) "
                            f"rover=({rel[i,0]:.1f},{rel[i,1]:.1f}) dist={gdist[i]:.1f}m "
                            f"act=({expert_action[i,0]:.2f},{expert_action[i,1]:.2f})"
                        )
                    self._expert_dbg_counter = 0

            # Debug: print expert goal observation and action only when something
            # looks off (very large goal distance or near-zero action when far).
            # Only in debug_mode (GUI play / continuous play); disabled in multi-env training.
            if getattr(self.cfg, "debug_mode", False) and hasattr(self.unwrapped, "_flat_goal_target_xy"):
                u = self.unwrapped
                tgt = u._flat_goal_target_xy
                rover = self.scene["rover"]
                rel = rover.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                gvec = tgt - rel
                gdist = torch.linalg.norm(gvec, dim=-1)
                for i in range(self.num_envs):
                    if gdist[i] > 15.0 or (gdist[i] > 2.0 and expert_action[i].abs().max() < 0.05):
                        print(
                            f"[INFO] Expert env{i}: goal=({tgt[i,0]:.2f},{tgt[i,1]:.2f}) "
                            f"rover=({rel[i,0]:.2f},{rel[i,1]:.2f}) dist={gdist[i]:.2f}m "
                            f"act=({expert_action[i,0]:.3f},{expert_action[i,1]:.3f})"
                        )
        else:
            self._expert_last_action[:] = 0.0

        # ---- 3. Standard pre-step recording ----
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # ---- 4. Physics stepping ----
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # Apply agent's estimate (no-op) and expert drive
            self.action_manager.apply_action()
            self._expert_drive.apply_actions()

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

        # ---- 5. Post-physics bookkeeping ----
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # Snapshot distance for progress rewards (if any downstream terms need it)
        u = self.unwrapped
        if hasattr(u, "_flat_goal_prev_dist"):
            pd = u._flat_goal_prev_dist
            _infer = getattr(torch, "is_inference", None)
            if _infer is not None and _infer(pd):
                from isaaclab_tasks.manager_based.rover.mdp.flat_goal_events import materialize_planar_goal_dist_buffer
                pd = materialize_planar_goal_dist_buffer(pd)
                u._flat_goal_prev_dist = pd
            u._flat_goal_dist_snapshot = pd.clone()

        # ---- 6. Reward computation ----
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # ---- 7. Reset terminated envs ----
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # ---- 8. Commands & interval events ----
        self.command_manager.compute(dt=self.step_dt)
        _goal_relocated = False
        _old_goal = None
        _new_goal = None
        if "interval" in self.event_manager.available_modes:
            u = self.unwrapped
            if hasattr(u, "_flat_goal_target_xy"):
                _old_goal = u._flat_goal_target_xy.clone()
            self.event_manager.apply(mode="interval", dt=self.step_dt)
            if _old_goal is not None and hasattr(u, "_flat_goal_target_xy"):
                _new_goal = u._flat_goal_target_xy
                _relocated_mask = (_old_goal != _new_goal).any(dim=-1)
                if _relocated_mask.any():
                    _goal_relocated = True
                    # Keep the distance snapshot in sync so downstream OA-Flat rewards
                    # (e.g. distance_progress) see the new distance on the next step.
                    if hasattr(u, "_flat_goal_prev_dist") and hasattr(u, "_flat_goal_dist_snapshot"):
                        u._flat_goal_dist_snapshot[_relocated_mask] = u._flat_goal_prev_dist[_relocated_mask].clone()
                    if getattr(self.cfg, "debug_mode", False):
                        for i in range(self.num_envs):
                            if _relocated_mask[i]:
                                print(
                                    f"[INFO] Goal relocated env{i}: old=({_old_goal[i,0]:.2f},{_old_goal[i,1]:.2f}) "
                                    f"new=({_new_goal[i,0]:.2f},{_new_goal[i,1]:.2f})"
                                )

        # If goal relocated, log expert goal observation on the next step's expert run
        if _goal_relocated and getattr(self.cfg, "debug_mode", False):
            # Expert observations were computed at the top of step() using the OLD goal.
            # On the next step they will use the NEW goal — nothing more to do here.
            pass

        # ---- 9. Final observations for the agent ----
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        """Reset environments and also reset expert-drive state."""
        super()._reset_idx(env_ids)
        if hasattr(self._expert_drive, "reset"):
            self._expert_drive.reset(env_ids=env_ids)
        self._expert_last_action[env_ids] = 0.0

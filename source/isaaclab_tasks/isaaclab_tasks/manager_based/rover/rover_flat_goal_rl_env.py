# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Flat goal navigation RL env: snapshots distance buffer before reward aggregation."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.rover.mdp.flat_goal_events import materialize_planar_goal_dist_buffer


class RoverFlatWorldGoalNavRLEnv(ManagerBasedRLEnv):
    """Standard :class:`ManagerBasedRLEnv` with a per-step snapshot for distance-shaping rewards."""

    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        u = self.unwrapped
        if hasattr(u, "_flat_goal_prev_dist"):
            pd = u._flat_goal_prev_dist
            _infer = getattr(torch, "is_inference", None)
            if _infer is not None and _infer(pd):
                pd = materialize_planar_goal_dist_buffer(pd)
                u._flat_goal_prev_dist = pd
            u._flat_goal_dist_snapshot = pd.clone()

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

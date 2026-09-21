# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner with optional divergence checks (non-finite losses, mean return band).

The training loop mirrors ``rsl_rl.runners.on_policy_runner.OnPolicyRunner.learn`` so we can
inject checks after each policy update. If upstream ``learn`` changes, sync this file.
"""

from __future__ import annotations

import math
import os
import statistics
import time
from collections import deque

import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state


class DivergenceAwareOnPolicyRunner(OnPolicyRunner):
    """Same as ``OnPolicyRunner`` but aborts training on NaN/Inf losses or out-of-band mean return.

    ``train.py`` may set ``REWARD_MIN`` / ``REWARD_MAX`` on the instance to bound mean episode
    return (from the last-100 rolling buffer). ``None`` means no bound on that side.
    """

    REWARD_MIN: float | None = None
    REWARD_MAX: float | None = None

    def _check_post_update(self, it: int, loss_dict: dict, rewbuffer: deque) -> None:
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        f"DivergenceAwareOnPolicyRunner: non-finite loss '{key}' at iteration {it}."
                    )
            elif isinstance(value, (float, int)):
                if not math.isfinite(float(value)):
                    raise RuntimeError(
                        f"DivergenceAwareOnPolicyRunner: non-finite loss '{key}' at iteration {it}."
                    )
        rmin = getattr(self, "REWARD_MIN", None)
        rmax = getattr(self, "REWARD_MAX", None)
        if len(rewbuffer) >= 1 and (rmin is not None or rmax is not None):
            m = float(statistics.mean(rewbuffer))
            if rmin is not None and m < float(rmin):
                raise RuntimeError(
                    f"DivergenceAwareOnPolicyRunner: mean episode return {m:g} < REWARD_MIN {float(rmin):g} "
                    f"at iteration {it}."
                )
            if rmax is not None and m > float(rmax):
                raise RuntimeError(
                    f"DivergenceAwareOnPolicyRunner: mean episode return {m:g} > REWARD_MAX {float(rmax):g} "
                    f"at iteration {it}."
                )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # --- Begin sync with rsl_rl OnPolicyRunner.learn (keep aligned on upgrades) ---
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer: deque = deque(maxlen=100)
            irewbuffer: deque = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            self._check_post_update(it, loss_dict, rewbuffer)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                locs = {
                    "it": it,
                    "num_learning_iterations": num_learning_iterations,
                    "tot_iter": tot_iter,
                    "start_iter": start_iter,
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "loss_dict": loss_dict,
                    "rewbuffer": rewbuffer,
                    "lenbuffer": lenbuffer,
                    "ep_infos": ep_infos,
                }
                if self.alg.rnd:
                    locs["erewbuffer"] = erewbuffer
                    locs["irewbuffer"] = irewbuffer
                self.log(locs)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        # --- End sync ---

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to control RSL RL logging frequency."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner


class RewardLoggerThread(threading.Thread):
    """Background thread for logging reward information periodically."""
    
    def __init__(self, runner: "OnPolicyRunner", log_interval: int = 100):
        """Initialize the reward logger thread.
        
        Args:
            runner: The RSL RL runner instance
            log_interval: Number of iterations between reward logs. Defaults to 100.
        """
        super().__init__(daemon=True)
        self.runner = runner
        self.log_interval = log_interval
        self.last_logged_iteration = -1
        self.stop_event = threading.Event()
        self.last_iteration = -1
    
    def should_log(self, current_iteration: int) -> bool:
        """Check if we should log at this iteration."""
        return (current_iteration == 0 or current_iteration % self.log_interval == 0) and \
               current_iteration != self.last_logged_iteration
    
    def log_env_rewards(self, current_iteration: int):
        """Log reward information from the environment if available."""
        if not self.should_log(current_iteration):
            return False
        
        try:
            # Get the environment from the runner
            env = self.runner.env
            
            # Check if the environment has our custom logging method
            if hasattr(env, 'log_reward_debug_info'):
                # Try to get the unwrapped environment if it's wrapped
                unwrapped_env = env
                while hasattr(unwrapped_env, 'unwrapped'):
                    unwrapped_env = unwrapped_env.unwrapped
                
                if hasattr(unwrapped_env, 'log_reward_debug_info'):
                    unwrapped_env.log_reward_debug_info(current_iteration)
                    self.last_logged_iteration = current_iteration
                    return True
        except Exception as e:
            # Silently skip if reward logging fails
            pass
        
        return False
    
    def run(self):
        """Background thread main loop - monitors and logs rewards periodically."""
        # This runs in background but doesn't actively participate in training
        # Actual logging is triggered at key moments via try_log_rewards()
        pass


class RewardLogger:
    """Helper class to log task-specific reward information."""
    
    def __init__(self, log_interval: int = 100, debug_log_file: str = None):
        """Initialize the reward logger.
        
        Args:
            log_interval: Number of iterations between reward logs. Defaults to 100.
            debug_log_file: Optional path to write reward logs to a file.
        """
        self.log_interval = log_interval
        self.last_logged_iteration = -1
        self.debug_log_file = debug_log_file
    
    def should_log(self, current_iteration: int) -> bool:
        """Check if we should log at this iteration."""
        # Log at iteration 50 (early debug) and every log_interval after
        return (current_iteration == 50 or (current_iteration > 0 and current_iteration % self.log_interval == 0)) and \
               current_iteration != self.last_logged_iteration
    
    def try_log_rewards(self, runner: "OnPolicyRunner", current_iteration: int, **extra_stats):
        """Attempt to log reward information from the environment.

        Args:
            runner: The RSL RL runner instance
            current_iteration: Current training iteration
            **extra_stats: Optional training stats (e.g. mean_entropy_loss, mean_action_std) to pass to the env logger.

        Returns:
            True if logging was performed, False otherwise
        """
        if not self.should_log(current_iteration):
            return False

        try:
            # Get the environment from the runner
            env = runner.env

            # Try to unwrap to the base environment
            unwrapped_env = env
            while hasattr(unwrapped_env, 'unwrapped') and unwrapped_env != unwrapped_env.unwrapped:
                unwrapped_env = unwrapped_env.unwrapped

            # Also try the .env attribute if available
            if not hasattr(unwrapped_env, 'log_reward_debug_info'):
                temp = env
                for _ in range(10):
                    if hasattr(temp, 'log_reward_debug_info'):
                        unwrapped_env = temp
                        break
                    if hasattr(temp, 'env'):
                        temp = temp.env
                    else:
                        break

            # Call the logging method if it exists
            if hasattr(unwrapped_env, 'log_reward_debug_info'):
                # This will print to console and also write to file if configured
                unwrapped_env.log_reward_debug_info(current_iteration, **extra_stats)
                self.last_logged_iteration = current_iteration
                return True

        except Exception as e:
            import traceback
            traceback.print_exc()

        return False


class LoggingFilter:
    """Filter to control RSL RL logging frequency and trigger reward logging.
    Can wrap either stdout or stderr; use shared_state when using both so block detection works.
    """

    def __init__(
        self,
        log_interval: int = 100,
        original_stream=None,
        reward_logger=None,
        runner=None,
        shared_state=None,
        progress_log_dir: str = None,
        progress_log_interval: int = 100,
        progress_plot_script: str = None,
    ):
        """Initialize the logging filter.

        Args:
            log_interval: Number of iterations between logs. Defaults to 100.
            original_stream: The stream to forward to (stdout or stderr).
            reward_logger: RewardLogger instance for logging rewards.
            runner: The RSL RL runner instance.
            shared_state: Optional dict shared with another filter (for stdout+stderr). If None, use instance state.
            progress_log_dir: If set, append full iteration block to progress_log_dir/training_progress_log.txt every progress_log_interval.
            progress_log_interval: Write to training_progress_log.txt every N iterations. Defaults to 100.
            progress_plot_script: If set, path to print_training_progress_log.py; run with --plot-only after each write to update training_progress_analysis.png.
        """
        self.log_interval = log_interval
        self.original_stream = original_stream or sys.stdout
        self.reward_logger = reward_logger
        self.runner = runner
        self._shared = shared_state if shared_state is not None else {}
        self.progress_log_dir = progress_log_dir
        self.progress_log_interval = progress_log_interval or 100
        self.progress_plot_script = progress_plot_script

    def write(self, text: str):
        """Filter writes: forward to stream immediately; accumulate in shared state for parsing when block ends."""
        iteration_match = re.search(r"Learning iteration (\d+)/", text)
        if iteration_match:
            self._shared["current_iteration"] = int(iteration_match.group(1))
            self._shared["should_log_current_block"] = (
                self._shared["current_iteration"] == 50
                or (
                    self._shared["current_iteration"] > 0
                    and self._shared["current_iteration"] % self.log_interval == 0
                )
            )
            self._shared["in_log_block"] = True
            self._shared["all_lines"] = [text]
            self.original_stream.write(text)
            # RSL-RL prints the entire iteration block in one print() call, so ETA: is in this same chunk
            if "ETA:" in text:
                self._process_block_end()
            return

        if self._shared.get("in_log_block"):
            self._shared.setdefault("all_lines", []).append(text)
            if "ETA:" in text:
                self._process_block_end()
            self.original_stream.write(text)
            return

        self.original_stream.write(text)

    def _process_block_end(self):
        """Parse block, call reward logger if needed, write to training_progress_log.txt every N iters, reset shared state."""
        block_text = "".join(self._shared.get("all_lines", []))
        current_iter = self._shared.get("current_iteration", -1)
        # Write full iteration block to log path every progress_log_interval (e.g. 100)
        if (
            self.progress_log_dir
            and current_iter >= 0
            and (current_iter == 0 or current_iter % self.progress_log_interval == 0)
        ):
            try:
                path = os.path.join(self.progress_log_dir, "training_progress_log.txt")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(block_text)
                    if not block_text.endswith("\n"):
                        f.write("\n")
            except Exception:
                pass
            # Update training_progress_analysis.png every progress_log_interval (e.g. 100) iterations
            if self.progress_plot_script and os.path.isfile(self.progress_plot_script):
                try:
                    import subprocess
                    _workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(self.progress_plot_script))))
                    subprocess.run(
                        [sys.executable, self.progress_plot_script, self.progress_log_dir, "--plot-only"],
                        cwd=_workspace_root,
                        timeout=60,
                        capture_output=True,
                        check=False,
                    )
                except Exception:
                    pass
        if (
            self._shared.get("should_log_current_block")
            and self.reward_logger
            and self.runner
        ):
            # RSL-RL format: "Mean entropy loss:" and "Mean action noise std:" (right-padded)
            entropy_m = re.search(
                r"Mean entropy loss\s*:\s*([\d.]+)", block_text, re.IGNORECASE
            )
            action_std_m = re.search(
                r"Mean action noise std\s*:\s*([\d.]+)", block_text, re.IGNORECASE
            )
            extra = {}
            if entropy_m:
                extra["mean_entropy_loss"] = float(entropy_m.group(1))
            if action_std_m:
                extra["mean_action_std"] = float(action_std_m.group(1))
            self.reward_logger.try_log_rewards(
                self.runner, self._shared["current_iteration"], **extra
            )
        self._shared["in_log_block"] = False
        self._shared["all_lines"] = []

    def flush(self):
        """Flush the original stream."""
        self.original_stream.flush()


class LoggingWrapper:
    """Wrapper for RSL RL runner to control logging frequency and add custom reward logging."""

    def __init__(
        self,
        runner: "OnPolicyRunner",
        log_interval: int = 100,
        debug_log_file: str = None,
        progress_log_dir: str = None,
        progress_log_interval: int = 100,
        progress_plot_script: str = None,
    ):
        """Initialize the logging wrapper.

        Args:
            runner: The RSL RL runner instance.
            log_interval: Number of iterations between logs. Defaults to 100.
            debug_log_file: Optional path to write reward debug logs to a file.
            progress_log_dir: If set, append full iteration block to progress_log_dir/training_progress_log.txt every progress_log_interval.
            progress_log_interval: Write to training_progress_log.txt every N iterations. Defaults to 100.
            progress_plot_script: If set, path to print_training_progress_log.py; run with --plot-only every progress_log_interval to update training_progress_analysis.png.
        """
        self.runner = runner
        self.log_interval = log_interval
        self.log_filter = None
        self.original_stdout = None
        self.reward_logger = RewardLogger(log_interval, debug_log_file)
        self.debug_log_file = debug_log_file
        self.progress_log_dir = progress_log_dir
        self.progress_log_interval = progress_log_interval
        self.progress_plot_script = progress_plot_script

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Run training with filtered logging.
        Patches both stdout and stderr so reward logging works regardless of which stream RSL-RL uses.
        """
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        shared_state = {}
        self.log_filter_stdout = LoggingFilter(
            self.log_interval,
            original_stream=sys.stdout,
            reward_logger=self.reward_logger,
            runner=self.runner,
            shared_state=shared_state,
            progress_log_dir=self.progress_log_dir,
            progress_log_interval=self.progress_log_interval,
            progress_plot_script=self.progress_plot_script,
        )
        self.log_filter_stderr = LoggingFilter(
            self.log_interval,
            original_stream=sys.stderr,
            reward_logger=self.reward_logger,
            runner=self.runner,
            shared_state=shared_state,
            progress_log_dir=self.progress_log_dir,
            progress_log_interval=self.progress_log_interval,
            progress_plot_script=self.progress_plot_script,
        )
        sys.stdout = self.log_filter_stdout
        sys.stderr = self.log_filter_stderr

        try:
            self.runner.learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            if num_learning_iterations > 0:
                self.reward_logger.try_log_rewards(self.runner, num_learning_iterations - 1)

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped runner."""
        return getattr(self.runner, name)


"""
Reward Logging Callback System for RSL-RL Training

This module provides a simple callback mechanism to log rewards during training without
needing to deeply integrate with RSL-RL internals.
"""

import os
import threading
import time
from typing import Optional, Callable


class RewardLoggingCallback:
    """Callback system that periodically logs reward information during training."""
    
    def __init__(self, env, log_file: str, log_interval: int = 100):
        """
        Initialize reward logging callback.
        
        Args:
            env: The environment (unwrapped) with log_reward_debug_info method
            log_file: Path to the log file
            log_interval: Log every N iterations
        """
        self.env = env
        self.log_file = log_file
        self.log_interval = log_interval
        self.current_iteration = 0
        self.last_logged_iteration = -1
        self.running = True
        self.lock = threading.Lock()
        
    def on_iteration_start(self, iteration: int):
        """Called at the start of each training iteration."""
        with self.lock:
            self.current_iteration = iteration
            
    def on_iteration_end(self, iteration: int):
        """Called at the end of each training iteration."""
        with self.lock:
            self.current_iteration = iteration
            # Check if we should log
            if iteration == 50 or (iteration > 0 and iteration % self.log_interval == 0):
                if iteration != self.last_logged_iteration:
                    self.last_logged_iteration = iteration
                    self._log_rewards(iteration)
    
    def _log_rewards(self, iteration: int):
        """Internal method to log rewards."""
        try:
            if hasattr(self.env, 'log_reward_debug_info'):
                # Redirect output to file
                import sys
                from io import StringIO
                
                # Capture print output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    # Call the environment's logging method
                    self.env.log_reward_debug_info(iteration)
                    output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                # Write to file
                if output:
                    with open(self.log_file, 'a') as f:
                        f.write(output)
                        f.flush()
                    print(f"[LOGGED] Reward info at iteration {iteration}")
        except Exception as e:
            print(f"[ERROR] Failed to log rewards: {e}")


def create_reward_logging_wrapper(runner, env, log_file: str, log_interval: int = 100):
    """
    Create a wrapped runner that logs rewards periodically.
    
    Args:
        runner: The OnPolicyRunner instance
        env: The unwrapped environment
        log_file: Path to the log file
        log_interval: Log every N iterations
        
    Returns:
        The runner with wrapped learn method
    """
    callback = RewardLoggingCallback(env, log_file, log_interval)
    
    # Store original learn method
    original_learn = runner.learn
    
    def learn_with_logging(num_learning_iterations, init_at_random_ep_len=False):
        """Wrapped learn method that calls callbacks."""
        # We'll try to hook into the training loop via monkey-patching
        # This is a bit hacky but works without modifying RSL-RL
        
        # Try to access the runner's algorithm  
        if hasattr(runner, 'alg') and hasattr(runner.alg, 'update'):
            original_update = runner.alg.update
            update_count = [0]
            
            def update_with_callback():
                """Wrapper around update that triggers callbacks."""
                result = original_update()
                
                # Get iteration from runner if available
                if hasattr(runner.alg, 'current_learning_iteration'):
                    iteration = runner.alg.current_learning_iteration
                    callback.on_iteration_end(iteration)
                
                update_count[0] += 1
                # Also check based on update count (rough estimate)
                # Each iteration is roughly num_envs/batch_size updates
                if update_count[0] % 32 == 0:  # Rough check every 32 updates
                    estimated_iter = update_count[0] // 32
                    callback.on_iteration_start(estimated_iter)
                
                return result
            
            # Replace the update method
            runner.alg.update = update_with_callback
            
            try:
                # Call original learn
                return original_learn(num_learning_iterations, init_at_random_ep_len)
            finally:
                # Restore original
                runner.alg.update = original_update
        else:
            # Fallback: just call original learn
            return original_learn(num_learning_iterations, init_at_random_ep_len)
    
    # Replace learn method
    runner.learn = learn_with_logging
    return runner

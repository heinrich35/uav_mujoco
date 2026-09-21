# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play with G1 robot and switch between standing, walking, and squat models using keyboard input."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import threading
import select
import termios
import tty

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with G1 robot and switch between models using keyboard.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-G1-General-Imitation-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--standing_model", type=str, required=True, help="Path to standing model checkpoint.")
parser.add_argument("--walking_model", type=str, required=True, help="Path to walking model checkpoint.")
parser.add_argument("--squat_model", type=str, required=True, help="Path to squat model checkpoint.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Global variables for model switching
current_model = "standing"  # 'standing', 'walking', 'squat'
model_paths = {}
policy = None
policy_nn = None
runner = None
policy_lock = threading.Lock()
should_stop = False
old_settings = None


def setup_terminal():
    """Set up terminal for non-blocking input."""
    global old_settings
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())


def restore_terminal():
    """Restore terminal settings."""
    global old_settings
    if old_settings is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def get_key():
    """Get a single keypress from stdin (non-blocking)."""
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        return sys.stdin.read(1)
    return None


def keyboard_listener():
    """Listen for keyboard input in a separate thread."""
    global current_model, should_stop
    setup_terminal()
    
    print("\n[INFO] Keyboard controls enabled:")
    print("  - Press 'w' to switch to walking model")
    print("  - Press 'q' to switch to squat model")
    print("  - Press 's' to switch back to standing model")
    print("  - Press 'x' to exit\n")
    
    try:
        while not should_stop:
            key = get_key()
            if key:
                key = key.lower()
                if key == 'w':
                    with policy_lock:
                        if current_model != "walking":
                            current_model = "walking"
                            print("\n[INFO] Switching to WALKING model...")
                elif key == 'q':
                    with policy_lock:
                        if current_model != "squat":
                            current_model = "squat"
                            print("\n[INFO] Switching to SQUAT model...")
                elif key == 's':
                    with policy_lock:
                        if current_model != "standing":
                            current_model = "standing"
                            print("\n[INFO] Switching to STANDING model...")
                elif key == 'x':
                    print("\n[INFO] 'x' key pressed. Stopping simulation...")
                    should_stop = True
                    simulation_app.close()
                    break
            time.sleep(0.01)  # Small delay to avoid busy waiting
    finally:
        restore_terminal()


def normalize_checkpoint_path(checkpoint_path):
    """Normalize checkpoint path to absolute path."""
    if os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.normpath(os.path.abspath(checkpoint_path))
        # Fix common path issues like /h/home -> /home
        if checkpoint_path.startswith('/h/home'):
            checkpoint_path = '/home' + checkpoint_path[7:]
    elif os.sep in checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
    else:
        checkpoint_path = retrieve_file_path(checkpoint_path)
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    return checkpoint_path


def load_policy_from_checkpoint(checkpoint_path, env, agent_cfg):
    """Load a policy from a checkpoint file.
    
    Note: The runner is created with the current environment's observation space,
    which must match the checkpoint's observation space. If there's a mismatch,
    ensure all models are trained on the same task with the same observation space.
    """
    checkpoint_path = normalize_checkpoint_path(checkpoint_path)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    
    # Use the current agent_cfg which matches the environment's observation space
    # The environment observation space is determined by the --task argument
    if agent_cfg.class_name == "OnPolicyRunner":
        runner_instance = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner_instance = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    try:
        runner_instance.load(checkpoint_path)
    except RuntimeError as e:
        if "size mismatch" in str(e) and "observation" in str(e).lower():
            error_msg = "\n" + "="*80 + "\n"
            error_msg += "OBSERVATION SPACE MISMATCH ERROR\n"
            error_msg += "="*80 + "\n"
            error_msg += "The checkpoint was trained with a different observation space than the current environment.\n\n"
            error_msg += "This typically happens when:\n"
            error_msg += "1. Models were trained on different tasks with different observation spaces\n"
            error_msg += "2. The environment configuration changed between training and inference\n\n"
            error_msg += "SOLUTION:\n"
            error_msg += "All models (standing, walking, squat) must be trained on the same task\n"
            error_msg += "with the same observation space. Use the --task argument to specify\n"
            error_msg += "the correct task that matches your checkpoints.\n\n"
            error_msg += "Example:\n"
            error_msg += "  If your models were trained on 'Isaac-G1-Standing-Revised-Direct-v0',\n"
            error_msg += "  use: --task Isaac-G1-Standing-Revised-Direct-v0\n\n"
            error_msg += "Original error:\n"
            error_msg += str(e) + "\n"
            error_msg += "="*80 + "\n"
            raise RuntimeError(error_msg) from e
        raise
    
    # obtain the trained policy for inference
    policy_instance = runner_instance.get_inference_policy(device=env.unwrapped.device)
    
    # extract the neural network module
    try:
        # version 2.3 onwards
        policy_nn_instance = runner_instance.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn_instance = runner_instance.alg.actor_critic
    
    return policy_instance, policy_nn_instance, runner_instance


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with G1 robot and switch between models."""
    global current_model, model_paths, policy, policy_nn, runner, should_stop
    
    # Normalize model paths
    model_paths = {
        "standing": normalize_checkpoint_path(args_cli.standing_model),
        "walking": normalize_checkpoint_path(args_cli.walking_model),
        "squat": normalize_checkpoint_path(args_cli.squat_model),
    }
    
    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Set log directory from standing model path
    log_dir = os.path.dirname(model_paths["standing"])
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "game"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load initial standing policy
    print(f"\n[INFO] Starting with STANDING model...")
    policy, policy_nn, runner = load_policy_from_checkpoint(model_paths["standing"], env, agent_cfg)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    last_model = "standing"

    print("\n[INFO] Simulation started. Use keyboard controls to switch models.")
    print("[INFO] Make sure the terminal window has focus for keyboard input to work.\n")

    # simulate environment
    while simulation_app.is_running() and not should_stop:
        start_time = time.time()
        
        # Check for model switch request
        with policy_lock:
            if current_model != last_model:
                try:
                    print(f"[INFO] Loading {current_model.upper()} model...")
                    policy, policy_nn, runner = load_policy_from_checkpoint(
                        model_paths[current_model], env, agent_cfg
                    )
                    last_model = current_model
                    print(f"[INFO] {current_model.upper()} model loaded successfully.")
                except Exception as e:
                    print(f"[ERROR] Failed to switch to {current_model} model: {e}")
                    current_model = last_model  # Revert on error
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        
        timestep += 1
        
        # Exit the play loop after recording one video (if video mode)
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    should_stop = True


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close sim app
        restore_terminal()
        simulation_app.close()

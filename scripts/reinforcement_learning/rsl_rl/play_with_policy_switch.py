# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play an RL agent with the ability to switch between policies at runtime."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import threading
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and policy switching capability.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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

import carb
import omni.appwindow

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

# Global variables for policy switching
current_policy_path = None
policy_switch_requested = None
policy_lock = threading.Lock()
should_stop = False


def load_policy_from_checkpoint(checkpoint_path, env, agent_cfg):
    """Load a policy from a checkpoint file."""
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(checkpoint_path)
    
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # extract the neural network module
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    
    return policy, policy_nn, runner


def watch_for_policy_switch():
    """Watch for policy switch requests from external script."""
    global policy_switch_requested, should_stop
    
    # Create a named pipe or file to watch
    switch_file = "/tmp/isaac_policy_switch"
    
    # Create the file if it doesn't exist
    if not os.path.exists(switch_file):
        os.makedirs(os.path.dirname(switch_file), exist_ok=True)
        with open(switch_file, "w") as f:
            f.write("")
    
    last_mtime = 0
    
    while not should_stop:
        try:
            if os.path.exists(switch_file):
                mtime = os.path.getmtime(switch_file)
                if mtime > last_mtime:
                    last_mtime = mtime
                    # Read the new policy path
                    with open(switch_file, "r") as f:
                        new_path = f.read().strip()
                    if new_path and os.path.isfile(new_path):
                        with policy_lock:
                            policy_switch_requested = new_path
                        print(f"\n[INFO] Policy switch requested: {new_path}")
            time.sleep(0.1)  # Check every 100ms
        except Exception as e:
            print(f"[WARNING] Error watching for policy switch: {e}")
            time.sleep(0.5)


# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent with policy switching capability."""
    global current_policy_path, policy_switch_requested, should_stop
    
    def on_keyboard_event(event, *args, **kwargs):
        """Handle keyboard events."""
        global should_stop
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "X":
                print("\n[INFO] 'X' key pressed. Stopping simulation...")
                should_stop = True
                simulation_app.close()
        return True

    # Set up keyboard input
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # Start thread to watch for policy switches
    switch_thread = threading.Thread(target=watch_for_policy_switch, daemon=True)
    switch_thread.start()

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # Determine initial checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        # Check if checkpoint is a full path (absolute or contains path separators)
        if os.path.isabs(args_cli.checkpoint):
            # Treat checkpoint as absolute path - use directly if it exists
            resume_path = os.path.abspath(args_cli.checkpoint)
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        elif os.sep in args_cli.checkpoint:
            # Relative path with separators - try to resolve it
            resume_path = os.path.abspath(args_cli.checkpoint)
            if not os.path.isfile(resume_path):
                # Fallback to retrieve_file_path for Nucleus paths
                resume_path = retrieve_file_path(args_cli.checkpoint)
        elif agent_cfg.load_run:
            # Treat checkpoint as filename and use load_run
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
        else:
            # Treat checkpoint as full path (fallback)
            resume_path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg.load_run and agent_cfg.load_checkpoint:
        # Use load_run and load_checkpoint to construct path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    current_policy_path = resume_path
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load initial policy
    policy, policy_nn, runner = load_policy_from_checkpoint(resume_path, env, agent_cfg)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    print("\n[INFO] Policy loaded and running. Press 'X' to stop the simulation.")
    print("[INFO] Make sure the Isaac Sim window has focus for keyboard input to work.")
    print("[INFO] Use switch_policy.sh to switch policies at runtime.\n")

    # simulate environment
    while simulation_app.is_running() and not should_stop:
        start_time = time.time()
        
        # Check for policy switch request
        with policy_lock:
            if policy_switch_requested and policy_switch_requested != current_policy_path:
                try:
                    print(f"[INFO] Switching policy to: {policy_switch_requested}")
                    policy, policy_nn, runner = load_policy_from_checkpoint(policy_switch_requested, env, agent_cfg)
                    current_policy_path = policy_switch_requested
                    policy_switch_requested = None
                    print("[INFO] Policy switched successfully.")
                except Exception as e:
                    print(f"[ERROR] Failed to switch policy: {e}")
                    policy_switch_requested = None
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Unsubscribe from keyboard events
    if keyboard_sub is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard_sub)

    # close the simulator
    env.close()
    print("[INFO] Simulation stopped.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


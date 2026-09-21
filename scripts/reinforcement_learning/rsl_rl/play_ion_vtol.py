# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play script for ion VTOL: 1 env, same as single training env.
# Load RSL-RL .pt checkpoint. Press key 5 to run PPO policy.
"""Play ion VTOL with RSL-RL checkpoint. Press 5 to run policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play ion VTOL with RSL-RL checkpoint. 1 env, same as training. Press 5 to run policy."
)
parser.add_argument("checkpoint", type=str, nargs="?", default=None, help="Path to .pt model file (required)")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
parser.add_argument("--video", action="store_true", default=False, help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Video length (steps)")
parser.add_argument("--seed", type=int, default=42, help="Environment seed")
parser.add_argument("--width", type=int, default=2560, help="Viewport width (pixels)")
parser.add_argument("--height", type=int, default=1440, help="Viewport height (pixels)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time
import torch

import gymnasium as gym

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.divergence_aware_runner import DivergenceAwareOnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Ion task id and config
ION_TASK = "Isaac-Ion-Vertical-Takeoff-Direct-v0"

# Neutral action: thrust ~25N each, yaw neutral. Actions in [-1,1], (raw*0.5+0.5) maps to [0,1]
# raw=0 -> thrust 25N, yaw π rad. Keeps drone roughly hover.
NEUTRAL_ACTION = torch.zeros(1, 8)


def _load_checkpoint_with_std_compat(runner, checkpoint_path: str, device: str):
    """Load checkpoint; convert legacy 'std' to 'log_std' if needed for policy compatibility."""
    loaded = torch.load(checkpoint_path, weights_only=False, map_location=device)
    sd = loaded["model_state_dict"]
    # Checkpoints saved with noise_std_type='scalar' have "std"; current config uses "log" -> "log_std"
    if "std" in sd and "log_std" not in sd:
        sd = dict(sd)
        sd["log_std"] = torch.log(sd["std"] + 1e-7)
        del sd["std"]
    runner.alg.policy.load_state_dict(sd, strict=True)

# Global: True when user presses 5
policy_active = [False]  # list for mutability in closure


def stdin_key_listener():
    """Background thread: read stdin. '5'+enter -> enable policy."""
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            if line.strip() == "5":
                policy_active[0] = True
                print("\n[INFO] Key 5 pressed - PPO policy ENABLED\n")
    except Exception:
        pass


def main():
    global policy_active

    if args_cli.checkpoint is None or not os.path.isfile(args_cli.checkpoint):
        print("Usage: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_ion_vtol.py <path_to_model.pt>")
        print("  Example: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_ion_vtol.py logs/rsl_rl/ion_vertical_takeoff_direct/2026-03-08_17-48-18/model_500.pt")
        print("  Press 5 and Enter to run the PPO policy.")
        simulation_app.close()
        return

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)

    # Load env and agent config (same as training)
    env_cfg: DirectRLEnvCfg = load_cfg_from_registry(ION_TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(ION_TASK, "rsl_rl_cfg_entry_point")

    # 1 env, same properties as single training env
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = log_dir
    device = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg.sim.device = device
    agent_cfg.device = device

    # Viewport rendering: DLSS quality, high resolution
    env_cfg.sim.render.antialiasing_mode = "DLSS"
    env_cfg.sim.render.dlss_mode = 2  # 0=Performance, 1=Balanced, 2=Quality

    env = gym.make(ION_TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model: {checkpoint_path}")
    runner = DivergenceAwareOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    _load_checkpoint_with_std_compat(runner, checkpoint_path, agent_cfg.device)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    # Keyboard: try carb (GUI) or stdin (headless). AppLauncher adds --headless
    headless = getattr(args_cli, "headless", False)
    use_carb = not headless
    keyboard_sub = None
    input_interface = None

    if use_carb:
        try:
            import carb
            import omni.appwindow

            def on_keyboard(event, *args, **kwargs):
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if event.input.name == "NUMPAD_5" or event.input.name == "5":
                        policy_active[0] = True
                        print("\n[INFO] Key 5 pressed - PPO policy ENABLED\n")
                return True

            input_interface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard)
        except Exception:
            use_carb = False

    if not use_carb:
        t = threading.Thread(target=stdin_key_listener, daemon=True)
        t.start()
        print("\n[INFO] Headless mode: type 5 and Enter to enable PPO policy\n")
    else:
        print("\n[INFO] Press key 5 to enable PPO policy (Isaac Sim window must have focus)\n")

    obs = env.get_observations()
    timestep = 0
    last_debug_time = time.time()
    DEBUG_INTERVAL_S = 0.1  # Log pose every 100 ms

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            if policy_active[0]:
                actions = policy(obs)
            else:
                num_envs = obs["policy"].shape[0]
                actions = NEUTRAL_ACTION.to(env.unwrapped.device).expand(num_envs, -1).clone()

            obs, _, dones, _ = env.step(actions)
            if policy_active[0]:
                policy_nn.reset(dones)

        # Debug: log pose, location, orientation every 100 ms
        now = time.time()
        if now - last_debug_time >= DEBUG_INTERVAL_S:
            last_debug_time = now
            try:
                robot = env.unwrapped._robot
                pos = robot.data.root_pos_w[0].cpu().numpy()
                quat = robot.data.root_quat_w[0].cpu().numpy()
                vel = robot.data.root_lin_vel_w[0].cpu().numpy()
                # Quat (w,x,y,z) to euler (deg) for readability
                w, x, y, z = quat
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw_rad = math.atan2(siny_cosp, cosy_cosp)
                sinp = 2 * (w * y - z * x)
                pitch_rad = math.asin(min(1, max(-1, sinp)))
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll_rad = math.atan2(sinr_cosp, cosr_cosp)
                yaw_deg = math.degrees(yaw_rad)
                pitch_deg = math.degrees(pitch_rad)
                roll_deg = math.degrees(roll_rad)
                print(
                    f"[DEBUG t={now:.2f}s] pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m  "
                    f"vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}) m/s  "
                    f"orient r/p/y=({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f}) deg"
                )
            except Exception as e:
                print(f"[DEBUG] pose log error: {e}")

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if keyboard_sub is not None and input_interface is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard_sub)

    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()

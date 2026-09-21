# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G1 Teleoperation (trained policy): run a policy trained with Isaac-G1-Teleoperation-Direct-v0.
# The policy observes [state(103), hand_target(3)] and outputs full-body actions. You drive the
# right-hand target with a gaming controller or keyboard arrow keys; the policy keeps the robot
# stable and tracks the target.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/play_g1_teleop_policy.py <checkpoint.pt>
#   python scripts/reinforcement_learning/rsl_rl/play_g1_teleop_policy.py logs/rsl_rl/g1_teleoperation/.../policy_checkpoints/model_5000.pt
#
# Controller: move right pad / sticks to move hand target. Button A (or --reset_button): reset target to current hand.
# Keyboard: Arrow keys move target (Up/Down=Z, Left/Right=Y, W/S=X when --keyboard_x enabled).

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="G1 teleoperation: trained policy follows hand target from controller or keyboard."
)
parser.add_argument("checkpoint", type=str, help="Path to teleoperation policy .pt (106-dim obs).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--joy_id", type=int, default=0)
parser.add_argument("--pos_scale", type=float, default=0.02, help="Controller to target delta (m).")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument("--axis_layout", type=str, default="right_first", choices=("default", "right_first", "right_first_swap"))
parser.add_argument("--x_scale", type=float, default=1.0)
parser.add_argument("--y_scale", type=float, default=1.0)
parser.add_argument("--z_scale", type=float, default=1.0)
parser.add_argument("--reset_button", type=int, default=0)
parser.add_argument("--keyboard", action="store_true", help="Use keyboard arrows for target (Up/Down=Z, Left/Right=Y).")
parser.add_argument("--keyboard_scale", type=float, default=0.01, help="Target step per key press (m).")
parser.add_argument("--keyboard_x", action="store_true", help="Use W/S for X axis.")
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--debug_interval", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

try:
    import carb
    import omni.appwindow
    HAS_CARB = True
except Exception:
    HAS_CARB = False

try:
    import pygame
    pygame.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.assets import retrieve_file_path
from rsl_rl.runners import OnPolicyRunner
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.g1_locomotion_v1.agents.rsl_rl_ppo_cfg import G1LocomotionV1PPORunnerCfg

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Default target (pelvis frame), same as waypoint initial
DEFAULT_TARGET_B = [0.12, -0.2, -0.08]


def _safe_axis(joy, axis_id: int) -> float:
    if not joy:
        return 0.0
    try:
        if joy.get_numaxes() > axis_id:
            return float(joy.get_axis(axis_id))
    except Exception:
        pass
    return 0.0


def read_gamepad_delta(joy, pos_scale: float, sensitivity: float, axis_layout: str,
                      x_scale: float, y_scale: float, z_scale: float):
    if not HAS_PYGAME or joy is None:
        return 0.0, 0.0, 0.0
    import pygame
    pygame.event.pump()
    scale = pos_scale * sensitivity
    a0, a1, a2, a3 = _safe_axis(joy, 0), _safe_axis(joy, 1), _safe_axis(joy, 2), _safe_axis(joy, 3)
    if axis_layout == "right_first":
        dy, dz = a0 * scale, a1 * scale
        dx = (a2 + 2.0 * a3) * scale
        dy += a2 * scale
    elif axis_layout == "right_first_swap":
        dz, dy = a0 * scale, a1 * scale
        dx = (a2 + 2.0 * a3) * scale
        dy += a2 * scale
    else:
        dx = (a0 + 2.0 * a1) * scale
        dy = a0 * scale + a2 * scale
        dz = a3 * scale
    return dx * x_scale, dy * y_scale, dz * z_scale


def main():
    checkpoint_path = args_cli.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.normpath(os.path.join(_ROOT, checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_env_cfg import G1TeleoperationEnvCfg
    env_cfg = G1TeleoperationEnvCfg()
    env_cfg.log_dir = os.path.dirname(checkpoint_path)
    env = gym.make("Isaac-G1-Teleoperation-Direct-v0", cfg=env_cfg)
    agent_cfg = G1LocomotionV1PPORunnerCfg()
    clip_val = getattr(agent_cfg, "clip_actions", 1.0)
    if isinstance(clip_val, bool):
        clip_val = 1.0 if clip_val else None
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=clip_val)

    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    if not hasattr(unwrapped, "set_teleop_target"):
        print("[ERROR] Environment does not support set_teleop_target (use Isaac-G1-Teleoperation-Direct-v0).")
        return

    device = unwrapped.device
    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    if hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    else:
        policy_nn = runner.alg.actor_critic
    print(f"[INFO] Loaded teleoperation policy from {checkpoint_path} (106-dim obs).")

    joy = None
    if HAS_PYGAME:
        try:
            import pygame
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > args_cli.joy_id:
                joy = pygame.joystick.Joystick(args_cli.joy_id)
                joy.init()
                print(f"[INFO] Gamepad: {joy.get_name()}. Button {args_cli.reset_button} = reset target to hand.")
        except Exception as e:
            print(f"[WARNING] Gamepad: {e}")
    if not joy and not args_cli.keyboard:
        print("[INFO] No gamepad and --keyboard not set. Target will follow default waypoint path.")

    # Teleop target in pelvis frame (single env)
    ik_target_b = torch.tensor(DEFAULT_TARGET_B, dtype=torch.float32, device=device).unsqueeze(0)
    reset_requested = [False]
    keyboard_dx, keyboard_dy, keyboard_dz = [0.0], [0.0], [0.0]

    if HAS_CARB and args_cli.keyboard:
        def _on_key(event):
            try:
                k = event.input.name
                s = getattr(args_cli, "keyboard_scale", 0.01)
                if k == "Up":
                    keyboard_dz[0] += s
                elif k == "Down":
                    keyboard_dz[0] -= s
                elif k == "Left":
                    keyboard_dy[0] += s
                elif k == "Right":
                    keyboard_dy[0] -= s
                elif args_cli.keyboard_x and k == "W":
                    keyboard_dx[0] += s
                elif args_cli.keyboard_x and k == "S":
                    keyboard_dx[0] -= s
                elif k == "A":
                    reset_requested[0] = True
            except Exception:
                pass
            return True
        try:
            inp = carb.input.acquire_input_interface()
            win = omni.appwindow.get_default_app_window()
            if win and win.get_keyboard():
                carb.input.acquire_input_interface().subscribe_to_keyboard_events(win.get_keyboard(), _on_key)
                print("[INFO] Keyboard: Arrows = target Z/Y, A = reset target to hand.")
        except Exception:
            pass

    obs, _ = env.reset()
    prev_buttons = [False] * 4

    while simulation_app.is_running():
        start = time.time()

        # Update target from controller or keyboard
        dx, dy, dz = read_gamepad_delta(
            joy, getattr(args_cli, "pos_scale", 0.02), getattr(args_cli, "sensitivity", 1.0),
            getattr(args_cli, "axis_layout", "right_first"),
            getattr(args_cli, "x_scale", 1.0), getattr(args_cli, "y_scale", 1.0), getattr(args_cli, "z_scale", 1.0),
        )
        dx += keyboard_dx[0]
        dy += keyboard_dy[0]
        dz += keyboard_dz[0]
        keyboard_dx[0] = keyboard_dy[0] = keyboard_dz[0] = 0.0

        if HAS_PYGAME and joy:
            import pygame
            pygame.event.pump()
            for i in range(min(4, joy.get_numbuttons())):
                b = bool(joy.get_button(i))
                if b and not prev_buttons[i]:
                    if i == getattr(args_cli, "reset_button", 0):
                        reset_requested[0] = True
                prev_buttons[i] = b

        if reset_requested[0]:
            reset_requested[0] = False
            root_pos = unwrapped._robot.data.root_pos_w[:1]
            root_quat = unwrapped._robot.data.root_quat_w[:1]
            hand_pos_w = unwrapped._robot.data.body_pos_w[:1, unwrapped._right_hand_body_idx, :]
            hand_quat_w = unwrapped._robot.data.body_quat_w[:1, unwrapped._right_hand_body_idx, :]
            hand_b, _ = subtract_frame_transforms(root_pos, root_quat, hand_pos_w, hand_quat_w)
            ik_target_b = hand_b.clone()
            print("[INFO] Target reset to current hand position.")

        ik_target_b = ik_target_b + torch.tensor([[dx, dy, dz]], device=device, dtype=ik_target_b.dtype)
        unwrapped.set_teleop_target(ik_target_b)

        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        if policy_nn is not None:
            policy_nn.reset(torch.zeros(num_envs, dtype=torch.bool, device=device))

        if getattr(args_cli, "debug_interval", 0) > 0:
            pass  # could log target and hand pos every N steps

        if getattr(args_cli, "real_time", False):
            elapsed = time.time() - start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

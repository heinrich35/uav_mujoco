# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G1 teleoperation driven by user input (gamepad/keyboard) with inverse kinematics
# from KDL/PyKDL (Orocos). Uses PyKDL ChainIkSolverPos_LMA in a subprocess—no Drake,
# DifferentialIK, or workspace chain IK. See G1_IK_ONLINE_SOURCES_IMPLEMENTATION.md.
#
# Usage:
#   Terminal 1 (optional): start KDL IK solver (stdin/stdout):
#     python scripts/kdl_ik/kdl_g1_ik_solver.py
#   Terminal 2: run this script (it will spawn the solver if not piped).
#
# Run: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_g1_teleop_kdl_ik.py
"""Play G1 teleoperation with user input; arm controlled by KDL IK (ChainIkSolverPos_LMA)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="G1 teleoperation: user input drives hand target; KDL IK subprocess returns joint angles; PD control to arm."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--robot_usd", type=str, default=None, help="Robot USD (default: assets/G1_minimal/G1_minimal.usd).")
parser.add_argument("--kdl_ik_script", type=str, default=None,
                    help="Path to kdl_g1_ik_solver.py (default: scripts/kdl_ik/kdl_g1_ik_solver.py).")
parser.add_argument("--joy_id", type=int, default=0)
parser.add_argument("--pos_scale", type=float, default=0.02, help="Hand target delta per stick unit (m).")
parser.add_argument("--axis_layout", type=str, default="right_first", choices=("default", "right_first", "right_first_swap"))
parser.add_argument("--reset_button", type=int, default=0)
parser.add_argument("--debug_interval", type=int, default=30)
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--ik_kp", type=float, default=80.0)
parser.add_argument("--ik_kd", type=float, default=12.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

try:
    import carb
    HAS_CARB = True
except Exception:
    HAS_CARB = False
try:
    import pygame
    pygame.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import subtract_frame_transforms

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ROBOT_USD = os.path.join(_ROOT, "assets", "G1_minimal", "G1_minimal.usd")
DEFAULT_KDL_IK_SCRIPT = os.path.join(_ROOT, "scripts", "kdl_ik", "kdl_g1_ik_solver.py")

# Right arm joint names (must match G1_right_arm_only.urdf order used by KDL solver)
RIGHT_ARM_JOINT_EXPR = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]


def _safe_axis(joy, axis_id: int) -> float:
    try:
        if joy.get_numaxes() > axis_id:
            return float(joy.get_axis(axis_id))
    except Exception:
        pass
    return 0.0


class KDLIKSubprocess:
    """Communicate with kdl_g1_ik_solver.py via stdin/stdout."""

    def __init__(self, script_path: str):
        self.script_path = script_path
        self.proc = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        if not os.path.isfile(self.script_path):
            raise FileNotFoundError(f"KDL IK solver not found: {self.script_path}")
        self.proc = subprocess.Popen(
            [sys.executable, self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        def drain_stderr():
            for line in iter(self.proc.stderr.readline, ""):
                if line:
                    sys.stderr.write(f"[kdl_ik] {line}")
        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()

    def solve(self, x: float, y: float, z: float) -> list[float] | None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return None
            try:
                self.proc.stdin.write(f"{x} {y} {z}\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    return None
                line = line.strip()
                if line == "FAIL":
                    return None
                return [float(v) for v in line.split()]
            except Exception:
                return None

    def close(self):
        with self._lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.stdin.close()
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
                self.proc = None


def main():
    robot_usd = args_cli.robot_usd or DEFAULT_ROBOT_USD
    kdl_ik_script = args_cli.kdl_ik_script or DEFAULT_KDL_IK_SCRIPT

    env_cfg = None
    for _ in range(1):
        try:
            from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_kdl_ik_env_cfg import G1TeleoperationKDLIKEnvCfg
            env_cfg = G1TeleoperationKDLIKEnvCfg()
            break
        except Exception:
            pass
        try:
            from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_fixed_base_env_cfg import G1TeleoperationFixedBaseEnvCfg
            env_cfg = G1TeleoperationFixedBaseEnvCfg()
            env_cfg.robot.spawn.usd_path = robot_usd
            break
        except Exception:
            pass
    if env_cfg is None:
        print("[ERROR] Could not load G1 teleoperation KDL IK or fixed-base env config.")
        return
    env_cfg.scene.num_envs = 1
    if getattr(env_cfg.robot.spawn, "usd_path", None) != robot_usd:
        env_cfg.robot.spawn.usd_path = robot_usd

    env = gym.make("Isaac-G1-Teleoperation-KDL-IK-Direct-v0", cfg=env_cfg)
    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    robot = unwrapped._robot
    device = unwrapped.device
    dt = unwrapped.step_dt

    ee_body_name = "right_wrist_yaw_link"
    try:
        body_ids, _ = robot.find_bodies(ee_body_name, preserve_order=True)
        ee_body_idx = int(body_ids[0]) if body_ids else None
    except Exception:
        ee_body_idx = None
    right_arm_joint_ids, right_arm_joint_names = robot.find_joints(RIGHT_ARM_JOINT_EXPR, preserve_order=True)
    if len(right_arm_joint_ids) < 5:
        print("[ERROR] Need at least 5 right arm joints for KDL IK.")
        env.close()
        return
    right_arm_joint_ids = [int(x) for x in right_arm_joint_ids[:5]]
    all_joint_ids, _ = robot.find_joints(".*", preserve_order=True)
    action_space = getattr(env_cfg, "action_space", 29)
    if len(all_joint_ids) > action_space:
        all_joint_ids = list(all_joint_ids)[:action_space]
    right_arm_action_indices = []
    for jid in right_arm_joint_ids:
        try:
            idx = list(all_joint_ids).index(jid)
            right_arm_action_indices.append(idx)
        except ValueError:
            pass
    right_arm_action_indices = sorted(right_arm_action_indices)
    joint_gears = torch.tensor(env_cfg.joint_gears[:action_space], dtype=torch.float32, device=device)
    action_scale = env_cfg.action_scale

    try:
        kdl_ik = KDLIKSubprocess(kdl_ik_script)
        print("[INFO] KDL IK solver started (ChainIkSolverPos_LMA, base frame target).")
    except Exception as e:
        print(f"[ERROR] Failed to start KDL IK solver: {e}")
        print("        Install PyKDL and ensure scripts/kdl_ik/kdl_g1_ik_solver.py exists.")
        env.close()
        return

    obs, _ = env.reset()
    with torch.inference_mode():
        robot.update(dt=dt)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        if ee_body_idx is not None:
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
            ik_target_pos_b, _ = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )
            ik_target_pos_b = ik_target_pos_b[0].cpu().tolist()
        else:
            ik_target_pos_b = [0.12, -0.2, -0.08]

    joy = None
    if HAS_PYGAME:
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > args_cli.joy_id:
                joy = pygame.joystick.Joystick(args_cli.joy_id)
                joy.init()
                print(f"[INFO] Gamepad: {joy.get_name()}. Key A or button {args_cli.reset_button} = reset target to hand.")
        except Exception as e:
            print(f"[WARNING] Gamepad: {e}")
    reset_requested = [False]
    if HAS_CARB:
        try:
            def _on_key(event):
                if getattr(event, "input", None) and getattr(event.input, "name", None) == "A":
                    reset_requested[0] = True
                return True
            carb.input.acquire_input_interface().subscribe_to_keyboard_events(_on_key)
        except Exception:
            pass
    prev_buttons = [False, False, False]
    timestep = 0

    try:
        while True:
            start_time = time.time()
            dx, dy, dz = 0.0, 0.0, 0.0
            if HAS_PYGAME and joy is not None:
                pygame.event.pump()
                layout = getattr(args_cli, "axis_layout", "right_first")
                if layout == "right_first":
                    dy, dz = _safe_axis(joy, 0), _safe_axis(joy, 1)
                    dx = _safe_axis(joy, 2) if joy.get_numaxes() > 2 else 0.0
                elif layout == "right_first_swap":
                    dz, dy = _safe_axis(joy, 0), _safe_axis(joy, 1)
                    dx = _safe_axis(joy, 2) if joy.get_numaxes() > 2 else 0.0
                else:
                    dx, dy = _safe_axis(joy, 0), _safe_axis(joy, 1)
                    dz = _safe_axis(joy, 2) if joy.get_numaxes() > 2 else 0.0
                scale = args_cli.pos_scale
                dx *= scale
                dy *= scale
                dz *= scale
                for i in range(min(3, joy.get_numbuttons())):
                    b = bool(joy.get_button(i))
                    if b and not prev_buttons[i]:
                        reset_requested[0] = True
                    prev_buttons[i] = b

            if reset_requested[0]:
                reset_requested[0] = False
                with torch.inference_mode():
                    robot.update(dt=dt)
                    root_pos_w = robot.data.root_pos_w
                    root_quat_w = robot.data.root_quat_w
                    if ee_body_idx is not None:
                        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                        ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                        ik_target_pos_b, _ = subtract_frame_transforms(
                            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
                        )
                        ik_target_pos_b = ik_target_pos_b[0].cpu().tolist()
                print("[INFO] IK target reset to hand position.")

            ik_target_pos_b[0] += dx
            ik_target_pos_b[1] += dy
            ik_target_pos_b[2] += dz

            q_arm = kdl_ik.solve(ik_target_pos_b[0], ik_target_pos_b[1], ik_target_pos_b[2])

            actions = torch.zeros(1, action_space, device=device)
            if q_arm is not None and len(right_arm_action_indices) >= 5:
                joint_pos_arm = robot.data.joint_pos[:, right_arm_joint_ids]
                joint_vel_arm = robot.data.joint_vel[:, right_arm_joint_ids]
                q_des = torch.tensor(q_arm, dtype=torch.float32, device=device).unsqueeze(0)
                pos_error = q_des - joint_pos_arm
                effort_arm = args_cli.ik_kp * pos_error - args_cli.ik_kd * joint_vel_arm
                for i, act_idx in enumerate(right_arm_action_indices):
                    if i < effort_arm.shape[1]:
                        gear = joint_gears[act_idx]
                        actions[0, act_idx] = (effort_arm[0, i] / (action_scale * gear)).clamp(-1.0, 1.0)

            obs, _, _, _, _ = env.step(actions)

            if args_cli.debug_interval > 0 and timestep % args_cli.debug_interval == 0:
                jp = robot.data.joint_pos[0, right_arm_joint_ids].cpu().numpy()
                print(f"[DEBUG] step={timestep} target_b=({ik_target_pos_b[0]:.3f},{ik_target_pos_b[1]:.3f},{ik_target_pos_b[2]:.3f}) "
                      f"q_arm={[round(x, 3) for x in jp]}")

            timestep += 1
            if args_cli.real_time:
                elapsed = time.time() - start_time
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        kdl_ik.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

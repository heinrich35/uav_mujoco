# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G1 teleoperation driven by user input (gamepad/keyboard) with inverse kinematics
# from Drake (Robot Locomotion Group). Uses Drake's InverseKinematics with
# AddPositionConstraint in a subprocess—no DifferentialIK or workspace chain IK.
# See G1_IK_ONLINE_SOURCES_IMPLEMENTATION.md and HUMANOID_IK_SOURCES.md.
#
# Usage:
#   Terminal 1 (optional): start Drake IK solver as server (stdin/stdout):
#     python scripts/drake_ik/drake_g1_ik_solver.py
#   Terminal 2: run this script (it will spawn the solver if not piped).
#   Or: echo "0.12 -0.2 -0.08" | python scripts/drake_ik/drake_g1_ik_solver.py
#
# Run: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_g1_teleop_drake_ik.py
"""Play G1 teleoperation with user input; arm controlled by Drake IK (position constraint)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="G1 teleoperation: user input drives hand target; Drake IK subprocess returns joint angles; PD control to arm."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--robot_usd", type=str, default=None, help="Robot USD (default: assets/G1_minimal/G1_minimal.usd).")
parser.add_argument("--drake_ik_script", type=str, default=None,
                    help="Path to drake_g1_ik_solver.py (default: scripts/drake_ik/drake_g1_ik_solver.py).")
parser.add_argument("--joy_id", type=int, default=0)
parser.add_argument("--pos_scale", type=float, default=0.02, help="Hand target delta per stick unit (m).")
parser.add_argument("--axis_layout", type=str, default="right_first", choices=("default", "right_first", "right_first_swap"))
parser.add_argument("--reset_button", type=int, default=0)
parser.add_argument("--debug_interval", type=int, default=30)
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--ik_kp", type=float, default=80.0)
parser.add_argument("--ik_kd", type=float, default=12.0)
parser.add_argument("--fixed_base", action="store_true",
                    help="Use fixed-base G1 spawn (Isaac-G1-Teleoperation-Fixed-Base-Direct-v0). Base does not move.")
parser.add_argument("--no_ik_target_vis", action="store_true",
                    help="Do not add the red IK target sphere to the stage.")
parser.add_argument("--starting_policy_path", type=str, default=None,
                    help="Path to policy checkpoint (.pt); policy drives base/legs, Drake IK overwrites right arm. "
                    "For full-body standing-IK policy (no arm overwrite, same as Isaac-G1-Standing-IK training), use play_g1_standing_ik_policy.py.")
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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms
from rsl_rl.runners import OnPolicyRunner

try:
    from pxr import Gf, UsdGeom
except Exception:
    Gf = None
    UsdGeom = None

import isaaclab.sim.utils.prims as prim_utils

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Red sphere at IK target (radius 0.05 m, no physics)
IK_TARGET_PRIM_PATH = "/World/ik_target"
IK_TARGET_RADIUS = 0.05
DEFAULT_ROBOT_USD = os.path.join(_ROOT, "assets", "G1_minimal", "G1_minimal.usd")
DEFAULT_DRAKE_IK_SCRIPT = os.path.join(_ROOT, "scripts", "drake_ik", "drake_g1_ik_solver.py")

# Right arm joint names (must match G1_right_arm_only.urdf order used by Drake solver)
RIGHT_ARM_JOINT_EXPR = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]


def _create_ik_target_sphere():
    """Add a red sphere at /World/ik_target, radius 0.05, no physics. Returns XFormPrim for pose updates or None."""
    try:
        if not prim_utils.is_prim_path_valid(IK_TARGET_PRIM_PATH):
            prim_utils.create_prim(
                IK_TARGET_PRIM_PATH,
                "Sphere",
                translation=(0.0, 0.0, 0.0),
                attributes={"radius": IK_TARGET_RADIUS},
            )
            if UsdGeom is not None and Gf is not None:
                prim = prim_utils.get_prim_at_path(IK_TARGET_PRIM_PATH)
                if prim:
                    sphere = UsdGeom.Sphere(prim)
                    sphere.CreateDisplayColorAttr()
                    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        from isaacsim.core.prims import XFormPrim
        return XFormPrim(IK_TARGET_PRIM_PATH, reset_xform_properties=False)
    except Exception:
        return None


def _safe_axis(joy, axis_id: int) -> float:
    try:
        if joy.get_numaxes() > axis_id:
            return float(joy.get_axis(axis_id))
    except Exception:
        pass
    return 0.0


class DrakeIKSubprocess:
    """Communicate with drake_g1_ik_solver.py via stdin/stdout."""

    def __init__(self, script_path: str):
        self.script_path = script_path
        self.proc = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        if not os.path.isfile(self.script_path):
            raise FileNotFoundError(f"Drake IK solver not found: {self.script_path}")
        self.proc = subprocess.Popen(
            [sys.executable, self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Consume stderr in background so it doesn't block
        def drain_stderr():
            for line in iter(self.proc.stderr.readline, ""):
                if line:
                    sys.stderr.write(f"[drake_ik] {line}")
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
    drake_ik_script = args_cli.drake_ik_script or DEFAULT_DRAKE_IK_SCRIPT

    use_fixed_base = getattr(args_cli, "fixed_base", False)
    env_cfg = None
    task_id = "Isaac-G1-Teleoperation-Drake-IK-Direct-v0"
    for _ in range(1):
        if use_fixed_base:
            try:
                from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_fixed_base_env_cfg import G1TeleoperationFixedBaseEnvCfg
                env_cfg = G1TeleoperationFixedBaseEnvCfg()
                env_cfg.robot.spawn.usd_path = robot_usd
                task_id = "Isaac-G1-Teleoperation-Fixed-Base-Direct-v0"
                break
            except Exception:
                pass
        try:
            from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_drake_ik_env_cfg import G1TeleoperationDrakeIKEnvCfg
            env_cfg = G1TeleoperationDrakeIKEnvCfg()
            break
        except Exception:
            pass
        try:
            from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_fixed_base_env_cfg import G1TeleoperationFixedBaseEnvCfg
            env_cfg = G1TeleoperationFixedBaseEnvCfg()
            env_cfg.robot.spawn.usd_path = robot_usd
            task_id = "Isaac-G1-Teleoperation-Fixed-Base-Direct-v0"
            break
        except Exception:
            pass
    if env_cfg is None:
        print("[ERROR] Could not load G1 teleoperation Drake IK or fixed-base env config.")
        return
    env_cfg.scene.num_envs = 1
    if getattr(env_cfg.robot.spawn, "usd_path", None) != robot_usd:
        env_cfg.robot.spawn.usd_path = robot_usd

    env = gym.make(task_id, cfg=env_cfg)

    # Optional: wrap for policy and load starting policy (runs from sim start)
    starting_policy_path = getattr(args_cli, "starting_policy_path", None) or None
    policy = None
    policy_nn = None
    if starting_policy_path and os.path.isfile(starting_policy_path):
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from isaaclab_tasks.direct.g1_locomotion_v1.agents.rsl_rl_ppo_cfg import G1LocomotionV1PPORunnerCfg
            agent_cfg = G1LocomotionV1PPORunnerCfg()
            clip_val = getattr(agent_cfg, "clip_actions", 1.0)
            if isinstance(clip_val, bool):
                clip_val = 1.0 if clip_val else None
            env = RslRlVecEnvWrapper(env, clip_actions=clip_val)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(starting_policy_path)
            policy = runner.get_inference_policy(device=agent_cfg.device)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = getattr(runner.alg, "actor_critic", None)
            print(f"[INFO] Starting policy loaded from {starting_policy_path} (runs each step; IK overwrites right arm).")
        except Exception as e:
            print(f"[WARNING] Could not load starting policy from {starting_policy_path}: {e}")
            policy = None
            policy_nn = None

    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    robot = unwrapped._robot
    device = unwrapped.device
    dt = unwrapped.step_dt

    # Right hand body and arm joint indices (for applying Drake IK / differential IK)
    ee_body_idx = None
    ee_body_name = None
    for name in ("right_palm_link", "right_wrist_yaw_link", "right_hand"):
        try:
            body_ids, _ = robot.find_bodies(name, preserve_order=True)
            if body_ids:
                ee_body_idx = int(body_ids[0])
                ee_body_name = name
                break
        except Exception:
            continue
    if ee_body_idx is None:
        try:
            body_ids, _ = robot.find_bodies("right_wrist_yaw_link", preserve_order=True)
            ee_body_idx = int(body_ids[0]) if body_ids else None
            ee_body_name = "right_wrist_yaw_link"
        except Exception:
            pass
    right_arm_joint_ids, right_arm_joint_names = robot.find_joints(RIGHT_ARM_JOINT_EXPR, preserve_order=True)
    if len(right_arm_joint_ids) < 5:
        print("[ERROR] Need at least 5 right arm joints for Drake IK.")
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

    # Differential IK fallback when Drake/pydrake is unavailable (hand follows target)
    diff_ik_controller = None
    jacobi_body_idx = None
    jacobi_joint_col_slice = None
    joint_limits = None
    if ee_body_idx is not None and len(right_arm_joint_ids) >= 5:
        try:
            jacobi_body_idx = ee_body_idx - 1 if getattr(robot, "is_fixed_base", True) else ee_body_idx
            jacobi_joint_col_slice = list(right_arm_joint_ids)
            diff_ik_cfg = DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.01},
            )
            diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)
            joint_limits = getattr(robot.data, "joint_pos_limits", None)
            print(f"[INFO] Differential IK fallback enabled (EE={ee_body_name or 'right_wrist_yaw_link'}). Hand will follow target when pydrake is not available.")
        except Exception as e:
            diff_ik_controller = None
            print(f"[WARNING] Differential IK fallback not available: {e}")

    # Start Drake IK subprocess (Drake position-constraint IK, see Drake docs)
    try:
        drake_ik = DrakeIKSubprocess(drake_ik_script)
        print("[INFO] Drake IK solver started (position constraint, base frame target).")
    except Exception as e:
        print(f"[ERROR] Failed to start Drake IK solver: {e}")
        print("        Install pydrake and ensure scripts/drake_ik/drake_g1_ik_solver.py exists.")
        env.close()
        return

    # Initial target = current hand position in base frame (reset to get valid robot state)
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

    # Red sphere at IK target (radius 0.05 m), unless disabled
    show_ik_target_vis = not getattr(args_cli, "no_ik_target_vis", False)
    ik_target_xform = _create_ik_target_sphere() if show_ik_target_vis else None
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    if ik_target_xform is not None:
        print(f"[INFO] Red sphere at {IK_TARGET_PRIM_PATH} (radius {IK_TARGET_RADIUS} m) shows IK target position.")

    # Gamepad
    joy = None
    if HAS_PYGAME:
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > args_cli.joy_id:
                joy = pygame.joystick.Joystick(args_cli.joy_id)
                joy.init()
                print(f"[INFO] Gamepad: {joy.get_name()}. Button {args_cli.reset_button} = reset target to hand.")
                print("[INFO] IK target control: right stick axes move target (axis_layout=right_first: axis0=Y, axis1=Z, axis2=X).")
        except Exception as e:
            print(f"[WARNING] Gamepad: {e}")
    # Keyboard: W/X = ±X, A/D = ±Y, Q/E = ±Z, S = reset (Isaac Sim window must have focus)
    reset_requested = [False]
    key_dx, key_dy, key_dz = [0.0], [0.0], [0.0]
    keyboard_sub = None
    keyboard_input_interface = None
    if HAS_CARB:
        try:
            import omni.appwindow
            def _on_key(event):
                if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                    return True
                k = getattr(event.input, "name", "") or ""
                s = args_cli.pos_scale
                if k == "W":
                    key_dx[0] += s
                elif k == "X":
                    key_dx[0] -= s
                elif k == "A":
                    key_dy[0] -= s   # -Y
                elif k == "D":
                    key_dy[0] += s   # +Y
                elif k == "Q":
                    key_dz[0] += s   # +Z
                elif k == "E":
                    key_dz[0] -= s   # -Z
                elif k == "S":
                    reset_requested[0] = True
                return True
            keyboard_input_interface = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            if app_window and app_window.get_keyboard():
                keyboard_sub = keyboard_input_interface.subscribe_to_keyboard_events(app_window.get_keyboard(), _on_key)
                print("[INFO] Keyboard (focus Isaac Sim window): W/X = ±X, A/D = ±Y, Q/E = ±Z, S = reset target to hand.")
        except Exception:
            pass
    prev_buttons = [False, False, False]
    timestep = 0

    try:
        while True:
            start_time = time.time()
            # User input -> target delta (no DifferentialIK; we use Drake only)
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

            # Keyboard deltas (W/X=±X, A/D=±Y, Q/E=±Z, S=reset)
            dx += key_dx[0]
            dy += key_dy[0]
            dz += key_dz[0]
            key_dx[0] = key_dy[0] = key_dz[0] = 0.0

            # Reset target to current hand
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

            # Current end-effector state (for differential IK fallback and logging)
            with torch.inference_mode():
                robot.update(dt=dt)
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                if ee_body_idx is not None:
                    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
                else:
                    ee_pos_b = torch.zeros(1, 3, device=device)
                    ee_quat_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

            # Drake IK: target position (base frame) -> joint angles; fallback to differential IK when pydrake unavailable
            q_arm = drake_ik.solve(ik_target_pos_b[0], ik_target_pos_b[1], ik_target_pos_b[2])
            joint_pos_des = None
            if q_arm is not None:
                joint_pos_des = torch.tensor(q_arm, dtype=torch.float32, device=device).unsqueeze(0)
            elif diff_ik_controller is not None and jacobi_body_idx is not None:
                try:
                    ik_target_t = torch.tensor([ik_target_pos_b], dtype=torch.float32, device=device)
                    delta_cart = ik_target_t - ee_pos_b
                    diff_ik_controller.set_command(delta_cart, ee_pos_b, ee_quat_b)
                    jacobian_w = robot.root_physx_view.get_jacobians()[
                        :, jacobi_body_idx, :, jacobi_joint_col_slice
                    ]
                    base_rot_mat = matrix_from_quat(quat_inv(root_quat_w))
                    jacobian_b = jacobian_w.clone()
                    jacobian_b[:, :3, :] = torch.bmm(base_rot_mat, jacobian_w[:, :3, :])
                    jacobian_b[:, 3:, :] = torch.bmm(base_rot_mat, jacobian_w[:, 3:, :])
                    joint_pos_arm = robot.data.joint_pos[:, right_arm_joint_ids]
                    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos_arm)
                    if joint_limits is not None and joint_limits.shape[1] > max(right_arm_joint_ids):
                        low = joint_limits[:, right_arm_joint_ids, 0]
                        high = joint_limits[:, right_arm_joint_ids, 1]
                        joint_pos_des = joint_pos_des.clamp(min=low, max=high)
                except Exception:
                    joint_pos_des = None

            # Base actions: from starting policy (if set) or zeros
            if policy is not None:
                with torch.inference_mode():
                    actions = policy(obs)
                    if isinstance(actions, dict) and "policy" in actions:
                        actions = actions["policy"]
                # Policy output is an inference tensor; clone before in-place arm overwrite below.
                actions = actions.clone()
            else:
                actions = torch.zeros(1, action_space, device=device)

            # Overwrite right arm with PD to IK solution
            if joint_pos_des is not None and len(right_arm_action_indices) >= 5:
                joint_pos_arm = robot.data.joint_pos[:, right_arm_joint_ids]
                joint_vel_arm = robot.data.joint_vel[:, right_arm_joint_ids]
                pos_error = joint_pos_des - joint_pos_arm
                effort_arm = args_cli.ik_kp * pos_error - args_cli.ik_kd * joint_vel_arm
                for i, act_idx in enumerate(right_arm_action_indices):
                    if i < effort_arm.shape[1]:
                        gear = joint_gears[act_idx]
                        actions[0, act_idx] = (effort_arm[0, i] / (action_scale * gear)).clamp(-1.0, 1.0)

            step_out = env.step(actions)
            # Bare DirectRLEnv: 5-tuple (obs, rew, terminated, truncated, info).
            # RslRlVecEnvWrapper: 4-tuple (obs, rew, dones, extras) for rsl_rl.
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
                dones_for_reset = terminated | truncated
            else:
                obs, _, dones, _ = step_out
                dones_for_reset = dones.bool() if dones.dtype != torch.bool else dones
            if policy_nn is not None:
                policy_nn.reset(dones_for_reset)

            # Refresh robot state and update IK target sphere position (base -> world)
            with torch.inference_mode():
                robot.update(dt=dt)
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                if ik_target_xform is not None:
                    ik_t = torch.tensor([ik_target_pos_b], dtype=torch.float32, device=device)
                    ik_target_pos_w = root_pos_w[0:1] + quat_apply(root_quat_w[0:1], ik_t[0:1])
                    ik_target_xform.set_world_poses(positions=ik_target_pos_w, orientations=identity_quat)

            if args_cli.debug_interval > 0 and timestep % args_cli.debug_interval == 0:
                jp = robot.data.joint_pos[0, right_arm_joint_ids].cpu().numpy()
                print(f"[DEBUG] step={timestep} target_b=({ik_target_pos_b[0]:.3f},{ik_target_pos_b[1]:.3f},{ik_target_pos_b[2]:.3f}) "
                      f"q_arm={[round(x, 3) for x in jp]}")
                if ee_body_idx is not None:
                    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                    ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
                    ee_pos_b_list = ee_pos_b[0].cpu().tolist()
                else:
                    ee_pos_b_list = [0.0, 0.0, 0.0]
                print(f"[INFO] ik_target_pos_b=({ik_target_pos_b[0]:.4f}, {ik_target_pos_b[1]:.4f}, {ik_target_pos_b[2]:.4f}) "
                      f"ee_pos_b=({ee_pos_b_list[0]:.4f}, {ee_pos_b_list[1]:.4f}, {ee_pos_b_list[2]:.4f})")

            timestep += 1
            if args_cli.real_time:
                elapsed = time.time() - start_time
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        if keyboard_sub is not None and keyboard_input_interface is not None:
            try:
                keyboard_input_interface.unsubscribe_to_keyboard_events(keyboard_sub)
            except Exception:
                pass
        drake_ik.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

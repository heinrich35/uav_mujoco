# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G1 teleoperation with visible IK target: same as play_g1_teleop plus:
# - Optional --base fixed | moveable for G1 spawn.
# - Red sphere "ik_target" in stage (radius 0.05, no physics) driven by same gamepad/keys as IK target.
"""Launch Isaac Sim with one G1 (fixed or moveable base), run standing policy, teleoperate right-hand IK from gamepad, and show a red sphere at the IK target."""

from __future__ import annotations

import argparse
import os
import struct
import time

from isaaclab.app import AppLauncher

# Default standing-for-teleoperation policy (106-dim)
DEFAULT_CHECKPOINT = "logs/rsl_rl/g1_locomotion_v1_standing/2026-03-15_21-39-22/final_model.pt"

parser = argparse.ArgumentParser(
    description="G1 teleoperation with visible IK target sphere (red, 0.05m). Same gamepad/keys move the sphere and the IK target."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default 1).")
parser.add_argument("--robot_usd", type=str, default=None,
                    help="Robot USD path (default: assets/G1/G1_minimal_1.usd).")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                    help=f"Path to policy checkpoint .pt (default: {DEFAULT_CHECKPOINT}).")
parser.add_argument("--base", type=str, default="moveable", choices=("fixed", "moveable"),
                    help="G1 base: moveable (default) or fixed.")
parser.add_argument("--no_policy", action="store_true", help="Disable policy; use IK only for right arm.")
parser.add_argument("--right_hand_body", type=str, default="right_wrist_yaw_link",
                    help="Body name for right hand IK.")
parser.add_argument("--debug_interval", type=int, default=30, help="Print debug info every N steps (0 = disable).")
parser.add_argument("--joy_id", type=int, default=0, help="Joystick device index (default 0).")
parser.add_argument("--pos_scale", type=float, default=0.02, help="Scale for stick/touchpad to hand delta (m per unit).")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Multiplier for position command from controller.")
parser.add_argument("--axis_layout", type=str, default="right_first", choices=("default", "right_first", "right_first_swap"))
parser.add_argument("--x_scale", type=float, default=1.0)
parser.add_argument("--y_scale", type=float, default=1.0)
parser.add_argument("--z_scale", type=float, default=1.0)
parser.add_argument("--debug_axes", action="store_true")
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--obs_99", action="store_true", help="Use 99-dim observation (legacy).")
parser.add_argument("--reset_button", type=int, default=0, help="Gamepad button to reset IK target to hand.")
parser.add_argument("--no_ik_target_vis", action="store_true", help="Do not add the red ik_target sphere to the stage.")
parser.add_argument("--ik_kp", type=float, default=80.0, help="IK joint position gain Kp (default 80).")
parser.add_argument("--ik_kd", type=float, default=12.0, help="IK joint damping Kd (default 12).")
parser.add_argument("--ik_iterations", type=int, default=3, help="IK iterations per step for better convergence (default 3).")
parser.add_argument("--ik_max_cartesian_step", type=float, default=None, help="Max Cartesian error step per frame in m (default None). e.g. 0.03 for stability.")
parser.add_argument("--ik_smooth_alpha", type=float, default=0.0, help="Smooth IK joint target: goal += alpha*(ik_des - goal). 0=off, 0.2-0.4=slower/smoother (default 0).")
parser.add_argument("--ik_max_joint_vel", type=float, default=None, help="Max joint goal velocity in rad/s (default None). Caps per-frame change of smoothed goal.")
parser.add_argument("--ik_ee_jump_thresh", type=float, default=0.15, help="EE position jump threshold in m; reset smoothed goal if exceeded (default 0.15). 0=disable.")
parser.add_argument("--ik_moveit_socket", type=str, default="", help="Use MoveIt IK: host:port (e.g. 127.0.0.1:9999). Empty=use built-in differential IK.")
parser.add_argument("--ik_moveit_max_jump", type=float, default=1.5, help="Reject MoveIt solution if any joint jumps > this (rad). Default 1.5.")
parser.add_argument("--ik_moveit_recv_timeout", type=float, default=5.0, help="Socket recv timeout (s) for MoveIt IK reply. Default 5.0 (server may retry 3x ~1.2s each).")
parser.add_argument("--ik_moveit_fail_streak", type=int, default=30, help="Drop MoveIt connection after this many consecutive request failures. Default 30 so initial pose can fail while you move target.")
parser.add_argument("--ik_moveit_skip_radius", type=float, default=0.02, help="Skip MoveIt when target is within this distance (m) of initial/reset pose. Avoids burning connection on one unreachable pose. Default 0.02.")
parser.add_argument("--ik_moveit_same_pose_tol", type=float, default=0.01, help="Treat two target poses as 'same' if within this distance (m). Only one failure per pose counts toward disconnect streak. Default 0.01.")
parser.add_argument("--ik_chain_urdf", type=str, nargs="?", default=None, const="default", metavar="PATH",
                    help="Use in-process chain IK (no server). PATH = URDF for right arm (default: assets/G1_minimal/G1_right_arm_only.urdf). Requires ikpy. Overrides MoveIt when both set.")
parser.add_argument("--ik_chain_max_jump", type=float, default=1.5, help="Reject chain IK solution if any joint jumps > this (rad). Default 1.5.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest after app launch."""

import gymnasium as gym
import numpy as np
import torch

try:
    import carb
    import omni.appwindow
    HAS_CARB_KEYBOARD = True
except Exception:
    HAS_CARB_KEYBOARD = False

try:
    from pxr import Gf, UsdGeom
except Exception:
    UsdGeom = None
    Gf = None

import isaaclab.sim.utils.prims as prim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_apply, subtract_frame_transforms

from rsl_rl.runners import OnPolicyRunner

# In-process chain IK (no server): load g1_chain_ik from same directory as this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_g1_chain_ik = None
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("g1_chain_ik", os.path.join(_script_dir, "g1_chain_ik.py"))
    _g1_chain_ik = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_g1_chain_ik)
except Exception:
    pass
if _g1_chain_ik is not None:
    load_chain = getattr(_g1_chain_ik, "load_chain", None)
    compute_ik_position_only = getattr(_g1_chain_ik, "compute_ik_position_only", None)
    HAS_CHAIN_IK = load_chain is not None and compute_ik_position_only is not None
else:
    HAS_CHAIN_IK = False
    load_chain = None
    compute_ik_position_only = None

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v1_env_cfg import G1LocomotionV1EnvCfg
from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_env_cfg import G1TeleoperationEnvCfg
from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_fixed_base_env_cfg import G1TeleoperationFixedBaseEnvCfg
from isaaclab_tasks.direct.g1_locomotion_v1.agents.rsl_rl_ppo_cfg import G1LocomotionV1PPORunnerCfg

try:
    import pygame
    pygame.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ROBOT_USD = os.path.join(_ROOT, "assets", "G1", "G1_minimal_1.usd")

IK_TARGET_PRIM_PATH = "/World/ik_target"
IK_TARGET_RADIUS = 0.05

# IK chain: shoulder -> elbow -> wrist (no right_five_joint). EE = right_palm_link.
RIGHT_ARM_JOINT_EXPR = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]
# Default PD gains for IK joint tracking (overridable via --ik_kp, --ik_kd).
IK_POSITION_KP_DEFAULT = 80.0
IK_POSITION_KD_DEFAULT = 12.0


def _moveit_ik_socket_connect(host: str, port: int, timeout: float = 2.0, recv_timeout: float = 0.5):
    """Return a connected socket or None. recv_timeout: max wait for one IK reply (MoveIt can be slow)."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.settimeout(recv_timeout)
        return s
    except Exception:
        return None


def _moveit_ik_socket_request(sock, pos_b, quat_b, seed_joints, device):
    """Send pose (3) + quat (4) + seed (5) = 12 floats; receive 5 floats.
    Returns (tensor or None, reason_str or None). reason_str only set when result is None (e.g. 'timeout', 'nan_solution')."""
    try:
        import socket
        p = pos_b[0].cpu().tolist()
        q = quat_b[0].cpu().tolist()  # qw,qx,qy,qz
        seed = seed_joints[0].cpu().tolist()
        buf = struct.pack("12f", p[0], p[1], p[2], q[0], q[1], q[2], q[3], *seed)
        sock.sendall(buf)
        data = sock.recv(5 * 4)
        if len(data) != 5 * 4:
            return None, "short_recv"
        joints = struct.unpack("5f", data)
        if any(x != x for x in joints):  # any NaN
            return None, "nan_solution"
        return torch.tensor([joints], dtype=torch.float32, device=device), None
    except Exception as e:
        import socket as _sock
        if isinstance(e, _sock.timeout):
            return None, "timeout"
        return None, "error"


def _safe_axis(joy, axis_id: int) -> float:
    try:
        if joy.get_numaxes() > axis_id:
            return float(joy.get_axis(axis_id))
    except Exception:
        pass
    return 0.0


def read_gamepad(joy, pos_scale: float, sensitivity: float = 1.0, axis_layout: str = "right_first",
                 x_scale: float = 1.0, y_scale: float = 1.0, z_scale: float = 1.0):
    if not HAS_PYGAME or joy is None:
        return 0.0, 0.0, 0.0
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


def _create_ik_target_sphere():
    """Add a red sphere at /World/ik_target, radius 0.05, no physics. Returns XFormPrim for pose updates or None."""
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
    try:
        from isaacsim.core.prims import XFormPrim
        return XFormPrim(IK_TARGET_PRIM_PATH, reset_xform_properties=False)
    except Exception:
        return None


def main():
    robot_usd = args_cli.robot_usd or DEFAULT_ROBOT_USD
    if not os.path.isabs(robot_usd):
        robot_usd = os.path.normpath(os.path.join(_ROOT, robot_usd))
    if not os.path.isfile(robot_usd):
        print(f"[ERROR] Robot USD not found: {robot_usd}")
        return

    checkpoint_path = args_cli.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.normpath(os.path.join(_ROOT, checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    use_base_fixed = getattr(args_cli, "base", "moveable").lower() == "fixed"
    use_obs_99 = args_cli.obs_99
    use_teleop_env = False
    if use_base_fixed:
        use_teleop_env = True
        print("[INFO] Base=fixed: using Isaac-G1-Teleoperation-Fixed-Base-Direct-v0.")
    elif not args_cli.no_policy:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            if "actor.0.weight" in state:
                obs_dim = state["actor.0.weight"].shape[1]
                if obs_dim == 99:
                    use_obs_99 = True
                elif obs_dim == 103:
                    use_obs_99 = False
                elif obs_dim == 106:
                    use_teleop_env = True
                    print("[INFO] Checkpoint 106-dim; using teleoperation env.")
        except Exception:
            pass

    if use_teleop_env:
        env_cfg = G1TeleoperationFixedBaseEnvCfg() if use_base_fixed else G1TeleoperationEnvCfg()
    else:
        env_cfg = G1LocomotionV1EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.robot.spawn.usd_path = robot_usd
    if not use_teleop_env and use_obs_99:
        env_cfg.include_external_torque_obs = False
        env_cfg.observation_space = 99

    if env_cfg.scene.num_envs == 1 and getattr(env_cfg.sim, "physx", None):
        p = env_cfg.sim.physx
        p.gpu_max_rigid_contact_count = min(p.gpu_max_rigid_contact_count, 2**14)
        p.gpu_max_rigid_patch_count = min(p.gpu_max_rigid_patch_count, 2**10)
        p.gpu_found_lost_pairs_capacity = min(p.gpu_found_lost_pairs_capacity, 2**14)
        p.gpu_found_lost_aggregate_pairs_capacity = min(p.gpu_found_lost_aggregate_pairs_capacity, 2**16)
        p.gpu_total_aggregate_pairs_capacity = min(p.gpu_total_aggregate_pairs_capacity, 2**14)

    task_id = (
        "Isaac-G1-Teleoperation-Fixed-Base-Direct-v0" if use_base_fixed
        else "Isaac-G1-Teleoperation-Direct-v0" if use_teleop_env
        else "Isaac-G1-Locomotion-V1-Direct-v0"
    )
    env = gym.make(task_id, cfg=env_cfg)
    env_cfg.log_dir = os.path.dirname(checkpoint_path)

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    agent_cfg = G1LocomotionV1PPORunnerCfg()
    clip_val = getattr(agent_cfg, "clip_actions", 1.0)
    if isinstance(clip_val, bool):
        clip_val = 1.0 if clip_val else None
    env = RslRlVecEnvWrapper(env, clip_actions=clip_val)

    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    robot = unwrapped._robot
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt

    body_candidates = ["right_palm_link"]
    ee_body_name = None
    for name in body_candidates:
        try:
            body_ids, _ = robot.find_bodies(name, preserve_order=True)
            if len(body_ids) >= 1:
                ee_body_name = name
                break
        except Exception:
            continue
    if ee_body_name is None:
        ik_controller = None
        right_arm_joint_ids = None
        ee_body_idx = None
        jacobi_body_idx = None
        jacobi_joint_col_slice = None
    else:
        ee_body_idx = robot.find_bodies(ee_body_name, preserve_order=True)[0][0]
        jacobi_body_idx = ee_body_idx - 1 if robot.is_fixed_base else ee_body_idx
        right_arm_joint_ids, right_arm_joint_names = robot.find_joints(RIGHT_ARM_JOINT_EXPR, preserve_order=True)
        if len(right_arm_joint_ids) < 3:
            ik_controller = None
            right_arm_joint_ids = None
            jacobi_joint_col_slice = None
        else:
            diff_ik_cfg = DifferentialIKControllerCfg(
                command_type="position", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.01}
            )
            ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=device)
            offset = 6 if not robot.is_fixed_base else 0
            jacobi_joint_col_slice = [offset + j for j in right_arm_joint_ids]
            print(f"[INFO] IK enabled: EE=right_palm_link, arm joints={right_arm_joint_names}")

    all_joint_ids, _ = robot.find_joints(".*", preserve_order=True)
    action_space = getattr(env_cfg, "action_space", 29)
    if len(all_joint_ids) > action_space:
        all_joint_ids = all_joint_ids[:action_space]
    right_arm_action_indices = []
    if right_arm_joint_ids is not None:
        for jid in right_arm_joint_ids:
            try:
                right_arm_action_indices.append(all_joint_ids.index(jid))
            except ValueError:
                pass
    right_arm_action_indices = sorted(right_arm_action_indices)
    joint_gears = torch.tensor(env_cfg.joint_gears[:action_space], dtype=torch.float32, device=device)
    action_scale = env_cfg.action_scale

    policy = None
    policy_nn = None
    if not args_cli.no_policy:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        print(f"[INFO] Loaded policy from {checkpoint_path}")

    joy = None
    if HAS_PYGAME:
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > args_cli.joy_id:
                joy = pygame.joystick.Joystick(args_cli.joy_id)
                joy.init()
                print(f"[INFO] Gamepad: {joy.get_name()}. Key A or button {args_cli.reset_button} = reset IK target.")
        except Exception as e:
            print(f"[WARNING] Gamepad init failed: {e}")

    ik_target_pos_b = None
    if use_teleop_env and hasattr(unwrapped, "set_teleop_target") and ee_body_idx is not None:
        with torch.inference_mode():
            robot.update(dt=dt)
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            ik_target_pos_b = ee_pos_b.clone()
            unwrapped.set_teleop_target(ik_target_pos_b)
    obs = env.get_observations()

    show_ik_target_vis = not getattr(args_cli, "no_ik_target_vis", False)
    ik_target_xform = None
    identity_quat = None
    if show_ik_target_vis:
        ik_target_xform = _create_ik_target_sphere()
        if ik_target_xform is not None:
            identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
            print("[INFO] Red sphere 'ik_target' added at /World/ik_target (radius 0.05, no physics). Same keys/gamepad move it.")

    reset_requested_keyboard = [False]
    reset_requested_controller = [False]
    prev_buttons = [False, False, False]
    current_ee_pos_b = None
    current_ee_quat_b = None

    def _on_keyboard(event):
        try:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name == "A":
                reset_requested_keyboard[0] = True
        except Exception:
            pass
        return True

    keyboard_sub = None
    if HAS_CARB_KEYBOARD:
        try:
            input_interface = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            if app_window and app_window.get_keyboard():
                keyboard_sub = input_interface.subscribe_to_keyboard_events(app_window.get_keyboard(), _on_keyboard)
        except Exception:
            pass

    joint_goal_smoothed = None
    reset_just_happened = [False]
    ee_pos_b_prev = None

    moveit_sock = None
    moveit_fail_streak = 0
    moveit_ref_pos = None
    moveit_last_failed_pose = None
    if getattr(args_cli, "ik_moveit_socket", "").strip():
        parts = args_cli.ik_moveit_socket.strip().rsplit(":", 1)
        if len(parts) == 2:
            try:
                moveit_host = parts[0]
                moveit_port = int(parts[1])
                recv_timeout = getattr(args_cli, "ik_moveit_recv_timeout", 5.0)
                moveit_sock = _moveit_ik_socket_connect(moveit_host, moveit_port, recv_timeout=recv_timeout)
                if moveit_sock is not None:
                    print(f"[INFO] MoveIt IK: connected to {moveit_host}:{moveit_port} (recv_timeout={recv_timeout}s)")
                else:
                    print(f"[WARNING] MoveIt IK: could not connect to {moveit_host}:{moveit_port}; using built-in IK.")
            except Exception as e:
                print(f"[WARNING] MoveIt IK: invalid --ik_moveit_socket or connect failed: {e}")

    chain_ik = None
    if getattr(args_cli, "ik_chain_urdf", None) is not None and HAS_CHAIN_IK and load_chain is not None:
        chain_urdf_path = getattr(args_cli, "ik_chain_urdf", "")
        if chain_urdf_path == "default" or chain_urdf_path == "":
            chain_urdf_path = os.path.join(_ROOT, "assets", "G1_minimal", "G1_right_arm_only.urdf")
        if not os.path.isabs(chain_urdf_path):
            chain_urdf_path = os.path.normpath(os.path.join(_ROOT, chain_urdf_path))
        chain_ik = load_chain(chain_urdf_path)
        if chain_ik is not None:
            print(f"[INFO] Chain IK: in-process IK enabled (ikpy, URDF={chain_urdf_path})")
        else:
            print(f"[WARNING] Chain IK: could not load chain from {chain_urdf_path} (install ikpy?); using differential IK.")

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        dx, dy, dz = read_gamepad(
            joy, args_cli.pos_scale, getattr(args_cli, "sensitivity", 1.0),
            getattr(args_cli, "axis_layout", "right_first"),
            getattr(args_cli, "x_scale", 1.0), getattr(args_cli, "y_scale", 1.0), getattr(args_cli, "z_scale", 1.0),
        ) if HAS_PYGAME else (0.0, 0.0, 0.0)

        if HAS_PYGAME and joy is not None:
            pygame.event.pump()
            for i in range(min(3, joy.get_numbuttons())):
                b = bool(joy.get_button(i))
                if b and not prev_buttons[i]:
                    reset_requested_controller[0] = True
                prev_buttons[i] = b

        has_user_input = (dx != 0 or dy != 0 or dz != 0) or reset_requested_keyboard[0] or reset_requested_controller[0]

        if use_teleop_env and hasattr(unwrapped, "set_teleop_target"):
            with torch.inference_mode():
                robot.update(dt=dt)
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
                current_ee_pos_b, current_ee_quat_b = ee_pos_b, ee_quat_b
                if reset_requested_keyboard[0] or reset_requested_controller[0]:
                    reset_requested_keyboard[0] = False
                    reset_requested_controller[0] = False
                    ik_target_pos_b = ee_pos_b.clone()
                    reset_just_happened[0] = True
                    print("[INFO] IK target reset to hand.")
                if ik_target_pos_b is None:
                    ik_target_pos_b = ee_pos_b.clone()
                ik_target_pos_b = ik_target_pos_b + torch.tensor([[dx, dy, dz]], device=device, dtype=ee_pos_b.dtype)
                unwrapped.set_teleop_target(ik_target_pos_b)
                obs = env.get_observations()

        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs)
        else:
            actions = torch.zeros(num_envs, action_space, device=device)

        with torch.inference_mode():
            if ik_controller is not None and right_arm_joint_ids is not None and len(right_arm_action_indices) > 0:
                actions = actions.clone()
                if use_teleop_env:
                    robot.update(dt=dt)
                    root_quat_w = robot.data.root_quat_w
                    ee_pos_b, ee_quat_b = current_ee_pos_b, current_ee_quat_b
                else:
                    robot.update(dt=dt)
                    root_pos_w = robot.data.root_pos_w
                    root_quat_w = robot.data.root_quat_w
                    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
                    if reset_requested_keyboard[0] or reset_requested_controller[0]:
                        reset_requested_keyboard[0] = reset_requested_controller[0] = False
                        ik_target_pos_b = ee_pos_b.clone()
                        reset_just_happened[0] = True
                    if ik_target_pos_b is None:
                        ik_target_pos_b = ee_pos_b.clone()
                    ik_target_pos_b = ik_target_pos_b + torch.tensor([[dx, dy, dz]], device=device, dtype=ee_pos_b.dtype)
                jacobian_w = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_col_slice]
                base_rot_mat = matrix_from_quat(quat_inv(root_quat_w))
                jacobian_b = jacobian_w.clone()
                jacobian_b[:, :3, :] = torch.bmm(base_rot_mat, jacobian_w[:, :3, :])
                jacobian_b[:, 3:, :] = torch.bmm(base_rot_mat, jacobian_w[:, 3:, :])
                joint_pos_arm = robot.data.joint_pos[:, right_arm_joint_ids]
                ee_pos_iter = ee_pos_b.clone()
                joint_pos_iter = joint_pos_arm.clone()
                joint_limits = robot.data.joint_pos_limits
                has_limits = joint_limits is not None and joint_limits.shape[1] > max(right_arm_joint_ids)

                joint_pos_des = None
                use_moveit_ik = False
                ik_source = "differential"
                if chain_ik is not None:
                    try:
                        pos_np = ik_target_pos_b[0].cpu().numpy()
                        seed_np = joint_pos_arm[0].cpu().numpy()
                        sol = compute_ik_position_only(chain_ik, pos_np, seed_np)
                        if sol is not None:
                            sol_arr = np.asarray(sol, dtype=np.float32).reshape(1, -1)
                            joint_pos_des = torch.from_numpy(sol_arr).to(device=device)
                            max_jump = getattr(args_cli, "ik_chain_max_jump", 1.5)
                            if (joint_pos_des - joint_pos_arm).abs().max().item() > max_jump:
                                joint_pos_des = None
                            else:
                                ik_source = "chain"
                    except Exception:
                        pass
                if joint_pos_des is None and moveit_sock is not None:
                    if reset_just_happened[0]:
                        moveit_ref_pos = None
                    if moveit_ref_pos is None:
                        moveit_ref_pos = ik_target_pos_b.clone()
                    skip_radius = getattr(args_cli, "ik_moveit_skip_radius", 0.02)
                    if (ik_target_pos_b - moveit_ref_pos).norm().item() < skip_radius:
                        pass
                    else:
                        same_pose_tol = getattr(args_cli, "ik_moveit_same_pose_tol", 0.01)
                        if moveit_last_failed_pose is not None and (
                            ik_target_pos_b - moveit_last_failed_pose
                        ).norm().item() <= same_pose_tol:
                            pass
                        else:
                            identity_quat_b = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=ee_pos_b.dtype)
                            joint_pos_des, moveit_fail_reason = _moveit_ik_socket_request(
                                moveit_sock, ik_target_pos_b, identity_quat_b, joint_pos_arm, device
                            )
                            if joint_pos_des is None:
                                fail_pose_new = moveit_last_failed_pose is None or (
                                    ik_target_pos_b - moveit_last_failed_pose
                                ).norm().item() > getattr(args_cli, "ik_moveit_same_pose_tol", 0.01)
                                if fail_pose_new:
                                    moveit_last_failed_pose = ik_target_pos_b.clone()
                                    if moveit_fail_streak == 0 and moveit_fail_reason:
                                        print(f"[INFO] MoveIt IK first failure: {moveit_fail_reason}. Will retry (disconnect after {getattr(args_cli, 'ik_moveit_fail_streak', 30)} different poses failing).")
                                    moveit_fail_streak += 1
                                max_streak = max(1, getattr(args_cli, "ik_moveit_fail_streak", 30))
                                if moveit_fail_streak >= max_streak:
                                    try:
                                        moveit_sock.close()
                                    except Exception:
                                        pass
                                    moveit_sock = None
                                    moveit_fail_streak = 0
                                    moveit_last_failed_pose = None
                                    print(f"[WARNING] MoveIt IK: disconnected after {max_streak} different poses failed.")
                            else:
                                moveit_fail_streak = 0
                                moveit_last_failed_pose = None
                                max_jump = getattr(args_cli, "ik_moveit_max_jump", 1.5)
                                if (joint_pos_des - joint_pos_arm).abs().max().item() > max_jump:
                                    joint_pos_des = None
                                else:
                                    use_moveit_ik = True
                                    ik_source = "MoveIt"

                if joint_pos_des is None:
                    ik_source = "differential"
                    ik_iterations = max(1, getattr(args_cli, "ik_iterations", 3))
                    max_cart_step = getattr(args_cli, "ik_max_cartesian_step", None)
                    for _ in range(ik_iterations):
                        delta_cart = ik_target_pos_b - ee_pos_iter
                        if max_cart_step is not None and max_cart_step > 0:
                            norm = delta_cart.norm(dim=1, keepdim=True).clamp(min=1e-6)
                            scale = (max_cart_step / norm).clamp(max=1.0)
                            delta_cart = delta_cart * scale
                        ik_controller.set_command(delta_cart, ee_pos_iter, ee_quat_b)
                        joint_pos_des = ik_controller.compute(ee_pos_iter, ee_quat_b, jacobian_b, joint_pos_iter)
                        if has_limits:
                            low = joint_limits[:, right_arm_joint_ids, 0]
                            high = joint_limits[:, right_arm_joint_ids, 1]
                            joint_pos_des = joint_pos_des.clamp(min=low, max=high)
                        if ik_iterations > 1:
                            delta_joint = (joint_pos_des - joint_pos_iter).unsqueeze(-1)
                            ee_pos_iter = ee_pos_iter + (jacobian_b[:, :3, :] @ delta_joint).squeeze(-1)
                            joint_pos_iter = joint_pos_des

                if joint_pos_des is not None and has_limits:
                    low = joint_limits[:, right_arm_joint_ids, 0]
                    high = joint_limits[:, right_arm_joint_ids, 1]
                    joint_pos_des = joint_pos_des.clamp(min=low, max=high)
                joint_vel_arm = robot.data.joint_vel[:, right_arm_joint_ids]
                if reset_just_happened[0]:
                    joint_goal_smoothed = None
                    reset_just_happened[0] = False
                ee_jump_thresh = getattr(args_cli, "ik_ee_jump_thresh", 0.15)
                if ee_jump_thresh > 0 and ee_pos_b_prev is not None:
                    ee_jump = (ee_pos_b - ee_pos_b_prev).norm(dim=1)
                    if ee_jump.max().item() > ee_jump_thresh:
                        joint_goal_smoothed = None
                ee_pos_b_prev = ee_pos_b.clone()
                smooth_alpha = max(0.0, getattr(args_cli, "ik_smooth_alpha", 0.0))
                max_joint_vel = getattr(args_cli, "ik_max_joint_vel", None)
                if smooth_alpha > 0:
                    if joint_goal_smoothed is None:
                        joint_goal_smoothed = joint_pos_arm.clone()
                    prev_goal = joint_goal_smoothed.clone()
                    joint_goal_smoothed = joint_goal_smoothed + smooth_alpha * (joint_pos_des - joint_goal_smoothed)
                    if max_joint_vel is not None and max_joint_vel > 0:
                        max_delta = max_joint_vel * dt
                        delta = (joint_goal_smoothed - prev_goal).clamp(-max_delta, max_delta)
                        joint_goal_smoothed = prev_goal + delta
                    if has_limits:
                        low = joint_limits[:, right_arm_joint_ids, 0]
                        high = joint_limits[:, right_arm_joint_ids, 1]
                        joint_goal_smoothed = joint_goal_smoothed.clamp(min=low, max=high)
                    joint_pos_des_for_pd = joint_goal_smoothed
                else:
                    joint_pos_des_for_pd = joint_pos_des
                if has_user_input:
                    print("[DEBUG] IK source:", ik_source)
                    print("[DEBUG] ik_target (body):", [round(x, 5) for x in ik_target_pos_b[0].cpu().tolist()])
                    print("[DEBUG] EE right_palm_link (body):", [round(x, 5) for x in ee_pos_b[0].cpu().tolist()])
                    print("[DEBUG] IK desired:", dict(zip(right_arm_joint_names, [round(x, 5) for x in joint_pos_des[0].cpu().tolist()])))
                    if smooth_alpha > 0:
                        print("[DEBUG] IK smoothed (PD):", dict(zip(right_arm_joint_names, [round(x, 5) for x in joint_pos_des_for_pd[0].cpu().tolist()])))
                    print("[DEBUG] IK actual: ", dict(zip(right_arm_joint_names, [round(x, 5) for x in joint_pos_arm[0].cpu().tolist()])))
                ik_kp = getattr(args_cli, "ik_kp", IK_POSITION_KP_DEFAULT)
                ik_kd = getattr(args_cli, "ik_kd", IK_POSITION_KD_DEFAULT)
                joint_pos_des_raw = joint_pos_des.clone()
                if has_limits and smooth_alpha == 0:
                    low = joint_limits[:, right_arm_joint_ids, 0]
                    high = joint_limits[:, right_arm_joint_ids, 1]
                    joint_pos_des = joint_pos_des.clamp(min=low, max=high)
                effort_arm = ik_kp * (joint_pos_des_for_pd - joint_pos_arm) - ik_kd * joint_vel_arm
                max_torque = action_scale * joint_gears[right_arm_action_indices]
                actions_clamped = 0
                for i, act_idx in enumerate(right_arm_action_indices):
                    if i < effort_arm.shape[1]:
                        raw_action = effort_arm[0, i] / max_torque[i]
                        ac = raw_action.clamp(-1.0, 1.0)
                        if abs(ac) >= 1.0 and abs(raw_action) > 1.0:
                            actions_clamped += 1
                        actions[0, act_idx] = ac
                if has_user_input:
                    cart_error = (ik_target_pos_b - ee_pos_b)[0]
                    cart_norm = float(cart_error.norm().item())
                    print("[DEBUG] cart_error [m] norm:", round(cart_norm, 5))
                    if has_limits:
                        low = joint_limits[:, right_arm_joint_ids, 0]
                        high = joint_limits[:, right_arm_joint_ids, 1]
                        at_low = (joint_pos_des_raw < low).squeeze(0)
                        at_high = (joint_pos_des_raw > high).squeeze(0)
                        limit_joints = [right_arm_joint_names[j] + "(low)" for j in range(len(right_arm_joint_names)) if at_low[j].item()]
                        limit_joints += [right_arm_joint_names[j] + "(high)" for j in range(len(right_arm_joint_names)) if at_high[j].item()]
                        if limit_joints:
                            print("[DEBUG] at_limit:", limit_joints)
                    print("[DEBUG] saturated:", actions_clamped, "/", len(right_arm_action_indices))

            obs, _, _, _ = env.step(actions)
            if policy_nn is not None:
                policy_nn.reset(torch.zeros(num_envs, dtype=torch.bool, device=device))

            if ik_target_xform is not None and ik_target_pos_b is not None and identity_quat is not None:
                robot.update(dt=dt)
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                ik_target_pos_w = root_pos_w[0:1] + quat_apply(root_quat_w[0:1], ik_target_pos_b[0:1])
                ik_target_xform.set_world_poses(positions=ik_target_pos_w, orientations=identity_quat)

        timestep += 1
        if args_cli.real_time:
            elapsed = time.time() - start_time
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    env.close()
    if keyboard_sub is not None and HAS_CARB_KEYBOARD:
        try:
            carb.input.acquire_input_interface().unsubscribe_to_keyboard_events(keyboard_sub)
        except Exception:
            pass


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G1 teleoperation: 1 env, trained standing policy, right-hand IK from gamepad (Steam Controller).
# Uses Isaac Sim native Differential IK (DLS) for right hand link; gamepad moves hand in space.
"""Launch Isaac Sim with one G1, run standing policy, and teleoperate right hand via gamepad + IK."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 teleoperation: standing policy + right-hand IK from gamepad.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default 1).")
parser.add_argument("--robot_usd", type=str, default=None,
                    help="Robot USD path (default: assets/G1/G1_minimal_1.usd).")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to policy checkpoint .pt (e.g. logs/rsl_rl/g1_locomotion_v1_standing/2026-03-12_19-58-55/policy_checkpoints/model_2700.pt).")
parser.add_argument("--no_policy", action="store_true", help="Disable policy; use IK only for right arm (rest holds default pose).")
parser.add_argument("--right_hand_body", type=str, default="right_wrist_yaw_link",
                    help="Body name for right hand IK (default: right_wrist_yaw_link). Fallback: right_six_link.")
parser.add_argument("--debug_interval", type=int, default=30,
                    help="Print debug info every N steps (0 = disable). Default 30.")
parser.add_argument("--joy_id", type=int, default=0, help="Joystick device index (default 0).")
parser.add_argument("--pos_scale", type=float, default=0.02,
                    help="Scale for stick/touchpad to hand delta position (m per unit). Default 0.02.")
parser.add_argument("--sensitivity", type=float, default=1.0,
                    help="Multiplier for position command from controller (1.0 = default). Tune for Steam Controller.")
parser.add_argument("--axis_layout", type=str, default="right_first", choices=("default", "right_first", "right_first_swap"),
                    help="Steam Controller: right_first=0,1 right pad (Y,Z); right_first_swap=0,1 right pad (Z,Y) if Y doesn't move; default=0,1 left/thumb 2,3 right.")
parser.add_argument("--x_scale", type=float, default=1.0,
                    help="Extra multiplier for X (forward/back) hand motion.")
parser.add_argument("--y_scale", type=float, default=1.0,
                    help="Extra multiplier for Y (left/right) hand motion. Use 1.5 or 2.0 if Y barely moves.")
parser.add_argument("--z_scale", type=float, default=1.0,
                    help="Extra multiplier for Z (up/down) hand motion.")
parser.add_argument("--debug_axes", action="store_true",
                    help="Print raw controller axis values at debug_interval so you can see which axis is which.")
parser.add_argument("--real_time", action="store_true", help="Run in real-time.")
parser.add_argument("--obs_99", action="store_true",
                    help="Use 99-dim observation (legacy checkpoints trained without external-torque obs).")
parser.add_argument("--fixed_base", action="store_true",
                    help="Use fixed-base G1 teleoperation env (Isaac-G1-Teleoperation-Fixed-Base-Direct-v0). Checkpoint must be 106-dim from g1_teleoperation_fixed_base.")
parser.add_argument("--reset_button", type=int, default=0,
                    help="Gamepad button index for resetting IK target to hand (0=A on Xbox layout). Key A (keyboard) also resets when Isaac Sim window has focus.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest after app launch."""

import gymnasium as gym
import torch

try:
    import carb
    import omni.appwindow
    HAS_CARB_KEYBOARD = True
except Exception:
    HAS_CARB_KEYBOARD = False

import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.g1_locomotion_v1 import G1LocomotionV1Env
from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v1_env_cfg import G1LocomotionV1EnvCfg
from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_env_cfg import G1TeleoperationEnvCfg
from isaaclab_tasks.direct.g1_locomotion_v1.g1_teleoperation_fixed_base_env_cfg import G1TeleoperationFixedBaseEnvCfg
from isaaclab_tasks.direct.g1_locomotion_v1.agents.rsl_rl_ppo_cfg import G1LocomotionV1PPORunnerCfg

# Optional gamepad (pygame)
try:
    import pygame
    pygame.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# Default paths
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ROBOT_USD = os.path.join(_ROOT, "assets", "G1", "G1_minimal_1.usd")

# Right arm IK chain from torso to wrist (G1 tree: torso_link -> shoulder_pitch/roll/yaw -> elbow_pitch/roll -> palm -> five/six_link)
# 6 DOF: shoulder 3 + elbow 2 + wrist 1 (right_five_joint)
RIGHT_ARM_JOINT_EXPR = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_five_joint",
]

# PD gains for converting IK position target to effort (stiffness / damping similar to G1 arm config)
IK_POSITION_KP = 40.0
IK_POSITION_KD = 10.0


def _safe_axis(joy, axis_id: int) -> float:
    """Return axis value in [-1, 1] or 0 if not available."""
    try:
        if joy.get_numaxes() > axis_id:
            return float(joy.get_axis(axis_id))
    except Exception:
        pass
    return 0.0


def read_gamepad(joy, pos_scale: float, sensitivity: float = 1.0, axis_layout: str = "right_first", x_scale: float = 1.0, y_scale: float = 1.0, z_scale: float = 1.0):
    """Read Steam Controller and return hand delta in root frame (X, Y, Z).

    right_first: right pad 0,1 -> Y,Z; right_first_swap: right pad 0,1 -> Z,Y (use if Y doesn't move).
    x_scale, y_scale, z_scale: extra multipliers per axis.
    """
    if not HAS_PYGAME or joy is None:
        return 0.0, 0.0, 0.0
    pygame.event.pump()
    scale = pos_scale * sensitivity
    a0 = _safe_axis(joy, 0)
    a1 = _safe_axis(joy, 1)
    a2 = _safe_axis(joy, 2)
    a3 = _safe_axis(joy, 3)

    if axis_layout == "right_first":
        dy = a0 * scale
        dz = a1 * scale
        dx = (a2 + 2.0 * a3) * scale
        dy += a2 * scale
    elif axis_layout == "right_first_swap":
        dz = a0 * scale
        dy = a1 * scale
        dx = (a2 + 2.0 * a3) * scale
        dy += a2 * scale
    else:
        dx = (a0 + 2.0 * a1) * scale
        dy = a0 * scale
        dy += a2 * scale
        dz = a3 * scale

    dx *= x_scale
    dy *= y_scale
    dz *= z_scale
    return dx, dy, dz


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

    # Detect obs dim from checkpoint so we match policy input size (99 vs 103 vs 106)
    use_obs_99 = args_cli.obs_99
    use_teleop_env = False  # 106-dim: use teleoperation env and set_teleop_target each step
    use_fixed_base = getattr(args_cli, "fixed_base", False)
    if use_fixed_base:
        use_teleop_env = True
        print("[INFO] Fixed-base mode: using Isaac-G1-Teleoperation-Fixed-Base-Direct-v0 (robot root fixed).")
    elif not args_cli.no_policy:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            if "actor.0.weight" in state:
                obs_dim = state["actor.0.weight"].shape[1]
                if obs_dim == 99:
                    use_obs_99 = True
                    print("[INFO] Checkpoint uses 99-dim obs; using legacy observation (no external-torque).")
                elif obs_dim == 103:
                    use_obs_99 = False
                elif obs_dim == 106:
                    use_teleop_env = True
                    print("[INFO] Checkpoint uses 106-dim obs; using teleoperation env and set_teleop_target each step.")
        except Exception:
            pass

    # Build env config: 1 env, G1_minimal_1.usd. Use teleoperation (or fixed-base) env for 106-dim checkpoints.
    if use_teleop_env:
        env_cfg = G1TeleoperationFixedBaseEnvCfg() if use_fixed_base else G1TeleoperationEnvCfg()
    else:
        env_cfg = G1LocomotionV1EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.robot.spawn.usd_path = robot_usd
    if not use_teleop_env and use_obs_99:
        env_cfg.include_external_torque_obs = False
        env_cfg.observation_space = 99

    # Reduce PhysX buffers for single env
    if env_cfg.scene.num_envs == 1 and getattr(env_cfg.sim, "physx", None):
        p = env_cfg.sim.physx
        p.gpu_max_rigid_contact_count = min(p.gpu_max_rigid_contact_count, 2**14)
        p.gpu_max_rigid_patch_count = min(p.gpu_max_rigid_patch_count, 2**10)
        p.gpu_found_lost_pairs_capacity = min(p.gpu_found_lost_pairs_capacity, 2**14)
        p.gpu_found_lost_aggregate_pairs_capacity = min(p.gpu_found_lost_aggregate_pairs_capacity, 2**16)
        p.gpu_total_aggregate_pairs_capacity = min(p.gpu_total_aggregate_pairs_capacity, 2**14)

    if use_teleop_env:
        task_id = "Isaac-G1-Teleoperation-Fixed-Base-Direct-v0" if use_fixed_base else "Isaac-G1-Teleoperation-Direct-v0"
    else:
        task_id = "Isaac-G1-Locomotion-V1-Direct-v0"
    env = gym.make(task_id, cfg=env_cfg)
    env_cfg.log_dir = os.path.dirname(checkpoint_path)

    agent_cfg = G1LocomotionV1PPORunnerCfg()
    clip_val = getattr(agent_cfg, "clip_actions", 1.0)
    if isinstance(clip_val, bool):
        clip_val = 1.0 if clip_val else None
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=clip_val)

    unwrapped = env
    while hasattr(unwrapped, "env"):
        unwrapped = unwrapped.env
    robot = unwrapped._robot
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt

    # Resolve right-hand body and arm joints
    body_candidates = [args_cli.right_hand_body, "right_six_link", "right_two_link", "right_four_link"]
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
        print("[WARNING] Right hand body not found. Available body names (first 30):",
              getattr(robot, "body_names", [])[:30])
        print("[INFO] Continuing without IK; only policy will run.")
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
            print("[WARNING] Could not find enough right arm joints:", right_arm_joint_names)
            ik_controller = None
            right_arm_joint_ids = None
            jacobi_joint_col_slice = None
        else:
            diff_ik_cfg = DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.01},
            )
            ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=num_envs, device=device)
            # Jacobian columns: for floating base, first 6 are base
            offset = 6 if not robot.is_fixed_base else 0
            jacobi_joint_col_slice = [offset + j for j in right_arm_joint_ids]
            print(f"[INFO] IK enabled: body={ee_body_name}, arm joints={right_arm_joint_names}")

    # Map right-arm joint indices into the 29-DOF action vector
    all_joint_ids, all_joint_names = robot.find_joints(".*", preserve_order=True)
    action_space = getattr(env_cfg, "action_space", 29)
    if len(all_joint_ids) > action_space:
        all_joint_ids = all_joint_ids[:action_space]
        all_joint_names = all_joint_names[:action_space]
    right_arm_action_indices = []
    if right_arm_joint_ids is not None:
        for jid in right_arm_joint_ids:
            try:
                idx = all_joint_ids.index(jid)
                right_arm_action_indices.append(idx)
            except ValueError:
                pass
    right_arm_action_indices = sorted(right_arm_action_indices)
    joint_gears = torch.tensor(env_cfg.joint_gears[:action_space], dtype=torch.float32, device=device)
    action_scale = env_cfg.action_scale

    # Load policy (unless --no_policy)
    policy = None
    policy_nn = None
    if not args_cli.no_policy:
        agent_cfg = G1LocomotionV1PPORunnerCfg()
        agent_cfg.experiment_name = "g1_locomotion_v1"
        agent_cfg.load_run = None
        agent_cfg.load_checkpoint = None
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        print(f"[INFO] Loaded policy from {checkpoint_path}")
    else:
        print("[INFO] No policy loaded; using default pose for non-right-arm joints.")

    # Gamepad
    joy = None
    if HAS_PYGAME:
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > args_cli.joy_id:
                joy = pygame.joystick.Joystick(args_cli.joy_id)
                joy.init()
                layout = getattr(args_cli, "axis_layout", "right_first")
                rb = getattr(args_cli, "reset_button", 0)
                print(f"[INFO] Gamepad: {joy.get_name()} (id={args_cli.joy_id}). axis_layout={layout}. Key A or button {rb} = reset IK target to hand.")
            else:
                print(f"[WARNING] No joystick at index {args_cli.joy_id}. Connect Steam Controller and re-run.")
        except Exception as e:
            print(f"[WARNING] Gamepad init failed: {e}")
    else:
        print("[WARNING] pygame not available. Install with: pip install pygame")

    # 3D IK target in base frame (position method: target stays in space, hand follows; key A or controller A resets)
    ik_target_pos_b = None
    if use_teleop_env and hasattr(unwrapped, "set_teleop_target") and ee_body_idx is not None:
        with torch.inference_mode():
            robot.update(dt=dt)
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )
            ik_target_pos_b = ee_pos_b.clone()
            unwrapped.set_teleop_target(ik_target_pos_b)
    obs = env.get_observations()
    timestep = 0
    # Reset: keyboard A (carb) or controller button (any of buttons 0,1,2 on rising edge)
    reset_requested_keyboard = [False]
    reset_requested_controller = [False]
    prev_buttons = [False, False, False]  # track 0, 1, 2 for edge detection

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
            if app_window is not None:
                keyboard = app_window.get_keyboard()
                if keyboard is not None:
                    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, _on_keyboard)
                    print("[INFO] Keyboard: press A (Isaac Sim window focused) to reset IK target to hand.")
        except Exception as e:
            print(f"[WARNING] Keyboard (carb) not available: {e}")

    reset_button_idx = getattr(args_cli, "reset_button", 0)
    # For 106-dim teleop: we update target and get obs at start of loop so policy sees the target
    current_ee_pos_b = None
    current_ee_quat_b = None

    while simulation_app.is_running():
        start_time = time.time()

        # Read gamepad (for IK and debug)
        dx, dy, dz = read_gamepad(
            joy, args_cli.pos_scale,
            getattr(args_cli, "sensitivity", 1.0),
            getattr(args_cli, "axis_layout", "right_first"),
            getattr(args_cli, "x_scale", 1.0),
            getattr(args_cli, "y_scale", 1.0),
            getattr(args_cli, "z_scale", 1.0),
        ) if HAS_PYGAME else (0.0, 0.0, 0.0)

        # Controller reset: check buttons 0,1,2 every frame (rising edge) so we don't miss presses
        if HAS_PYGAME and joy is not None:
            pygame.event.pump()
            try:
                nbt = joy.get_numbuttons()
                for i in range(min(3, nbt)):
                    b = bool(joy.get_button(i))
                    if b and not prev_buttons[i]:
                        reset_requested_controller[0] = True
                    prev_buttons[i] = b
            except Exception:
                pass

        # 106-dim teleop: update target and get obs before policy so policy sees the hand target
        if use_teleop_env and hasattr(unwrapped, "set_teleop_target"):
            with torch.inference_mode():
                robot.update(dt=dt)
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
                )
                current_ee_pos_b, current_ee_quat_b = ee_pos_b, ee_quat_b
                reset_target = reset_requested_keyboard[0] or reset_requested_controller[0]
                if reset_requested_keyboard[0]:
                    reset_requested_keyboard[0] = False
                if reset_requested_controller[0]:
                    reset_requested_controller[0] = False
                if reset_target:
                    ik_target_pos_b = ee_pos_b.clone()
                    print("[INFO] IK target reset to hand position (key A or controller).")
                if ik_target_pos_b is None:
                    ik_target_pos_b = ee_pos_b.clone()
                ik_target_pos_b = ik_target_pos_b + torch.tensor(
                    [[dx, dy, dz]], device=device, dtype=ee_pos_b.dtype
                )
                unwrapped.set_teleop_target(ik_target_pos_b)
                obs = env.get_observations()

        # Policy action (or zeros)
        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs)
        else:
            actions = torch.zeros(num_envs, action_space, device=device)

        # Position method: 3D target in base frame; user input moves target; hand follows target; key A or controller resets target.
        with torch.inference_mode():
            if ik_controller is not None and (right_arm_joint_ids is not None) and len(right_arm_action_indices) > 0:
                actions = actions.clone()
                if use_teleop_env:
                    robot.update(dt=dt)  # refresh for jacobian/root_quat_w
                    root_quat_w = robot.data.root_quat_w
                    ee_pos_b, ee_quat_b = current_ee_pos_b, current_ee_quat_b
                else:
                    robot.update(dt=dt)
                    root_pos_w = robot.data.root_pos_w
                    root_quat_w = robot.data.root_quat_w
                    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
                    )
                    # Reset 3D target to current hand position (joints controlled by policy)
                    reset_target = reset_requested_keyboard[0] or reset_requested_controller[0]
                    if reset_requested_keyboard[0]:
                        reset_requested_keyboard[0] = False
                    if reset_requested_controller[0]:
                        reset_requested_controller[0] = False
                    if reset_target:
                        ik_target_pos_b = ee_pos_b.clone()
                        print("[INFO] IK target reset to hand position (key A or controller).")
                    if ik_target_pos_b is None:
                        ik_target_pos_b = ee_pos_b.clone()
                    # User input moves the 3D target position (stays in base frame)
                    ik_target_pos_b = ik_target_pos_b + torch.tensor(
                        [[dx, dy, dz]], device=device, dtype=ee_pos_b.dtype
                    )
                # Delta from current hand to target so hand follows and stays at target
                delta = ik_target_pos_b - ee_pos_b
                ik_controller.set_command(delta, ee_pos_b, ee_quat_b)
                jacobian_w = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_col_slice]
                base_rot_mat_batch = matrix_from_quat(quat_inv(root_quat_w))
                jacobian_b = jacobian_w.clone()
                jacobian_b[:, :3, :] = torch.bmm(base_rot_mat_batch, jacobian_w[:, :3, :])
                jacobian_b[:, 3:, :] = torch.bmm(base_rot_mat_batch, jacobian_w[:, 3:, :])
                joint_pos_arm = robot.data.joint_pos[:, right_arm_joint_ids]
                joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos_arm)
                joint_vel_arm = robot.data.joint_vel[:, right_arm_joint_ids]
                pos_error = joint_pos_des - joint_pos_arm
                effort_arm = IK_POSITION_KP * pos_error - IK_POSITION_KD * joint_vel_arm
                for i, act_idx in enumerate(right_arm_action_indices):
                    if i < effort_arm.shape[1]:
                        gear = joint_gears[act_idx]
                        actions[0, act_idx] = (effort_arm[0, i] / (action_scale * gear)).clamp(-1.0, 1.0)

            obs, _, _, _ = env.step(actions)
            if policy_nn is not None:
                policy_nn.reset(torch.zeros(num_envs, dtype=torch.bool, device=device))

        # Debug output
        if args_cli.debug_interval > 0 and timestep % args_cli.debug_interval == 0:
            try:
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                user_input_str = f"stick_delta=({dx:.3f},{dy:.3f},{dz:.3f}) target_mode" if ik_controller else "no_IK"
                if ee_body_name and hasattr(robot.data, "body_pos_w"):
                    hand_w = robot.data.body_pos_w[0, ee_body_idx].cpu().numpy()
                    root_pos = robot.data.root_pos_w[0].cpu().numpy()
                    root_quat = robot.data.root_quat_w[0].cpu().numpy()
                    hand_b, _ = subtract_frame_transforms(
                        torch.tensor(root_pos, device=device).unsqueeze(0),
                        torch.tensor(root_quat, device=device).unsqueeze(0),
                        torch.tensor(hand_w, device=device).unsqueeze(0),
                        robot.data.body_quat_w[0:1, ee_body_idx],
                    )
                    hand_b_np = hand_b[0].cpu().numpy()
                    print(f"[DEBUG] step={timestep} {user_input_str} | hand_pos_local=({hand_b_np[0]:.3f},{hand_b_np[1]:.3f},{hand_b_np[2]:.3f}) | joint_pos[21:29]={joint_pos[21:29].round(3).tolist()}")
                else:
                    print(f"[DEBUG] step={timestep} {user_input_str} | joint_pos[21:29]={joint_pos[21:29].round(3).tolist()}")
                if getattr(args_cli, "debug_axes", False) and HAS_PYGAME and joy is not None:
                    pygame.event.pump()
                    nax = joy.get_numaxes()
                    raw = [round(_safe_axis(joy, i), 3) for i in range(min(6, nax))]
                    print(f"[DEBUG] raw_axes[0..5]={raw}  (move pads/stick to see which axis is which)")
            except Exception as e:
                print(f"[DEBUG] step={timestep} error: {e}")

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

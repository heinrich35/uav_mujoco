# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# G1 Standing IK — run a policy trained on Isaac-G1-Standing-IK-Direct-v0 (106-dim obs).
# The right-hand target in pelvis frame is driven by gamepad/keyboard; the policy controls
# the full body (same as training). Optional red sphere at /World/ik_target.
#
# Usage:
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_g1_standing_ik_policy.py \
#     --starting_policy_path logs/rsl_rl/g1_standing_ik/.../final_model.pt
#
"""Play G1 standing IK policy: user moves IK target; policy tracks (no external arm IK)."""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="G1 standing IK: trained policy (106-dim obs); user moves IK target like training."
)
parser.add_argument(
    "--starting_policy_path",
    "--checkpoint",
    type=str,
    default=None,
    dest="starting_policy_path",
    help="Policy checkpoint (.pt) from g1_standing_ik training.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--robot_usd", type=str, default=None, help="Override robot USD (default: G1_minimal from env cfg).")
parser.add_argument("--joy_id", type=int, default=0)
parser.add_argument("--pos_scale", type=float, default=0.02, help="Target delta per stick unit or key step (m).")
parser.add_argument("--axis_layout", type=str, default="right_first", choices=("default", "right_first", "right_first_swap"))
parser.add_argument("--reset_button", type=int, default=0)
parser.add_argument("--debug_interval", type=int, default=30)
parser.add_argument("--real_time", action="store_true")
parser.add_argument(
    "--fixed_base",
    action="store_true",
    help="Fix robot root (fix_root_link); use if you trained / prefer fixed base.",
)
parser.add_argument("--no_ik_target_vis", action="store_true", help="Do not spawn red ik_target sphere.")
parser.add_argument(
    "--ik_target_obs_scale",
    type=float,
    default=None,
    help="Must match training (default: from G1StandingIKEnvCfg).",
)
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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab_tasks.direct.g1_locomotion_v1.agents.rsl_rl_ppo_cfg import G1LocomotionV1PPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

try:
    from pxr import Gf, UsdGeom
except Exception:
    Gf = None
    UsdGeom = None

import isaaclab.sim.utils.prims as prim_utils

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

IK_TARGET_PRIM_PATH = "/World/ik_target"
IK_TARGET_RADIUS = 0.05
DEFAULT_ROBOT_USD = os.path.join(_ROOT, "assets", "G1_minimal", "G1_minimal.usd")


def _create_ik_target_sphere():
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


def _policy_obs_after_teleop(env, unwrapped, ik_obs_scale: float):
    """Match training rollout: refresh derived obs (vel, angles, …) then policy command (last 3 dims)."""
    unwrapped._compute_intermediate_values()
    obs = env.get_observations()
    pol = obs["policy"]
    cmd = unwrapped._hand_target_pos_b * ik_obs_scale
    pol[:, -3:].copy_(cmd)
    return obs


def _safe_axis(joy, axis_id: int) -> float:
    try:
        if joy.get_numaxes() > axis_id:
            return float(joy.get_axis(axis_id))
    except Exception:
        pass
    return 0.0


def main():
    ckpt = getattr(args_cli, "starting_policy_path", None) or None
    if not ckpt or not os.path.isfile(ckpt):
        if not ckpt:
            print("[ERROR] Pass --starting_policy_path / --checkpoint to a g1_standing_ik .pt file.")
        else:
            print(f"[ERROR] Checkpoint not found: {ckpt}")
        return

    from isaaclab_tasks.direct.g1_locomotion_v1.g1_standing_ik_env_cfg import G1StandingIKEnvCfg

    env_cfg = G1StandingIKEnvCfg()
    robot_usd = args_cli.robot_usd or getattr(env_cfg.robot.spawn, "usd_path", None) or DEFAULT_ROBOT_USD
    env_cfg.robot.spawn.usd_path = robot_usd
    if getattr(args_cli, "fixed_base", False):
        env_cfg.spawn_fixed_base = True
        env_cfg.robot.spawn.articulation_props.fix_root_link = True
    if args_cli.ik_target_obs_scale is not None:
        env_cfg.ik_target_obs_scale = float(args_cli.ik_target_obs_scale)

    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Isaac-G1-Standing-IK-Direct-v0", cfg=env_cfg)

    agent_cfg = G1LocomotionV1PPORunnerCfg()
    clip_val = getattr(agent_cfg, "clip_actions", 1.0)
    if isinstance(clip_val, bool):
        clip_val = 1.0 if clip_val else None
    env = RslRlVecEnvWrapper(env, clip_actions=clip_val)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=agent_cfg.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = getattr(runner.alg, "actor_critic", None)

    print(f"[INFO] Loaded standing IK policy from {ckpt} (Isaac-G1-Standing-IK-Direct-v0, 106-dim obs).")
    print("[INFO] Full-body actions from policy — no Drake / external arm IK overwrite.")

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "set_teleop_target"):
        print("[ERROR] Environment has no set_teleop_target.")
        env.close()
        return

    robot = unwrapped._robot
    device = unwrapped.device
    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt
    ik_obs_scale = float(getattr(unwrapped.cfg, "ik_target_obs_scale", 1.0))

    ee_body_idx = getattr(unwrapped, "_right_hand_body_idx", None)
    if ee_body_idx is None:
        print(
            "[ERROR] _right_hand_body_idx is not set on the env (same link as training rewards / obs). "
            "Check right_hand_body_name in cfg and G1_minimal.usd body names."
        )
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
            ik_target_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            ik_target_pos_b = ik_target_pos_b[0].cpu().tolist()
        else:
            ik_target_pos_b = [0.12, -0.2, -0.08]

    ik_target_tensor = torch.tensor([ik_target_pos_b], dtype=torch.float32, device=device)
    unwrapped.set_teleop_target(ik_target_tensor)
    obs = _policy_obs_after_teleop(env, unwrapped, ik_obs_scale)

    show_vis = not getattr(args_cli, "no_ik_target_vis", False)
    ik_target_xform = _create_ik_target_sphere() if show_vis else None
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    if ik_target_xform is not None:
        print(f"[INFO] Red sphere at {IK_TARGET_PRIM_PATH} (radius {IK_TARGET_RADIUS} m) shows IK target.")

    joy = None
    if HAS_PYGAME:
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > args_cli.joy_id:
                joy = pygame.joystick.Joystick(args_cli.joy_id)
                joy.init()
                print(f"[INFO] Gamepad: {joy.get_name()}. Button {args_cli.reset_button} = reset target to hand.")
        except Exception as e:
            print(f"[WARNING] Gamepad: {e}")

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
                    key_dy[0] -= s
                elif k == "D":
                    key_dy[0] += s
                elif k == "Q":
                    key_dz[0] += s
                elif k == "E":
                    key_dz[0] -= s
                elif k == "S":
                    reset_requested[0] = True
                return True
            keyboard_input_interface = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            if app_window and app_window.get_keyboard():
                keyboard_sub = keyboard_input_interface.subscribe_to_keyboard_events(app_window.get_keyboard(), _on_key)
                print("[INFO] Keyboard: W/X = ±X, A/D = ±Y, Q/E = ±Z, S = reset target to hand.")
        except Exception:
            pass

    prev_buttons = [False, False, False]
    timestep = 0

    try:
        while simulation_app.is_running():
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

            dx += key_dx[0]
            dy += key_dy[0]
            dz += key_dz[0]
            key_dx[0] = key_dy[0] = key_dz[0] = 0.0

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

            ik_target_tensor = torch.tensor([ik_target_pos_b], dtype=torch.float32, device=device)
            unwrapped.set_teleop_target(ik_target_tensor)
            obs = _policy_obs_after_teleop(env, unwrapped, ik_obs_scale)

            with torch.inference_mode():
                actions = policy(obs)

            _, _, dones, _ = env.step(actions)
            dones_b = dones.bool() if dones.dtype != torch.bool else dones
            if policy_nn is not None:
                policy_nn.reset(dones_b)

            with torch.inference_mode():
                robot.update(dt=dt)
                root_pos_w = robot.data.root_pos_w
                root_quat_w = robot.data.root_quat_w
                if ik_target_xform is not None:
                    # Same transform as combine_frame_transforms / EE math (root frame point -> world).
                    ik_t = torch.tensor([ik_target_pos_b], dtype=torch.float32, device=device)
                    ik_target_pos_w, _ = combine_frame_transforms(
                        root_pos_w[0:1], root_quat_w[0:1], ik_t, None
                    )
                    ik_target_xform.set_world_poses(positions=ik_target_pos_w, orientations=identity_quat)

            if args_cli.debug_interval > 0 and timestep % args_cli.debug_interval == 0:
                if ee_body_idx is not None:
                    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
                    ee_quat_w = robot.data.body_quat_w[:, ee_body_idx]
                    ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
                    ee_list = ee_pos_b[0].cpu().tolist()
                else:
                    ee_list = [0.0, 0.0, 0.0]
                print(
                    f"[DEBUG] step={timestep} target_b=({ik_target_pos_b[0]:.3f},{ik_target_pos_b[1]:.3f},{ik_target_pos_b[2]:.3f}) "
                    f"ee_b=({ee_list[0]:.4f},{ee_list[1]:.4f},{ee_list[2]:.4f})"
                )

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
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

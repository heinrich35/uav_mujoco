# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play ion VTOL with switchable policies: vertical takeoff and horizontal cruise.
# Key 0: reset to (0, 0, 0.04), zero velocity and force.
# Key 5: run vertical takeoff model.
# Key 8: run horizontal cruise model.
"""Play ion VTOL — switch between vertical takeoff and horizontal cruise policies."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play ion VTOL: Key 0=reset, 5=vertical takeoff policy, 8=horizontal cruise policy."
)
parser.add_argument(
    "--vertical_checkpoint",
    type=str,
    default="logs/rsl_rl/ion_vertical_takeoff_v1/2026-03-12_15-09-23/model_3000.pt",
    help="Path to vertical takeoff .pt model (key 5).",
)
parser.add_argument(
    "--horizontal_checkpoint",
    type=str,
    default="logs/rsl_rl/ion_horizontal_cruise_v1/2026-03-12_20-35-06/model_2200.pt",
    help="Path to horizontal cruise .pt model (key 8).",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
parser.add_argument("--video", action="store_true", default=False, help="Record video")
parser.add_argument("--seed", type=int, default=42, help="Environment seed")
parser.add_argument(
    "--width",
    type=int,
    default=1280,
    help="Viewport width (default 1280 to reduce VRAM vs 1440p; use 1920/2560 if you have headroom).",
)
parser.add_argument(
    "--height",
    type=int,
    default=720,
    help="Viewport height (default 720; raise with --width if GPU memory allows).",
)
parser.add_argument(
    "--dlss",
    action="store_true",
    default=False,
    help=(
        "Use DLSS for RTX antialiasing (kit default is often DLSS). Without this flag, FXAA is forced to avoid "
        "omni.gpucompute-cuda memcpy errors on some GPUs/drivers."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
import torch

import gymnasium as gym

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.divergence_aware_runner import DivergenceAwareOnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK_VERTICAL_V1 = "Isaac-Ion-Vertical-Takeoff-V1-Direct-v0"
TASK_VERTICAL_V2 = "Isaac-Ion-Vertical-Takeoff-V2-Direct-v0"
TASK_HORIZONTAL = "Isaac-Ion-Horizontal-Cruise-V1-Direct-v0"

# Horizontal cruise checkpoints are often 18-dim (legacy) or 20-dim (V1-aligned).
OBS_DIM_HORIZONTAL_LEGACY = 18
# V1 vertical / horizontal base policy obs (IMU + aero + pose + vz setpoint channels).
OBS_DIM_VERTICAL_V1 = 20
# V2 default: V1 base + normalized target vz + reward-profile phase scalar (see ion_vertical_takeoff_v2_env_cfg).
OBS_DIM_VERTICAL_V2_DEFAULT = 22


def _policy_obs_in_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """Read actor first-layer input size from an RSL-RL .pt checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        w = sd.get("actor.0.weight")
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            return int(w.shape[1])
    except Exception:
        return None
    return None


def _shrink_physx_for_single_env_play(env_cfg: DirectRLEnvCfg) -> None:
    """Cap PhysX GPU/pinned reservations (training cfg targets 1k+ envs) so GUI+RTX fits in VRAM."""
    p = env_cfg.sim.physx
    env_cfg.sim.physx = p.replace(
        gpu_max_rigid_patch_count=min(p.gpu_max_rigid_patch_count, 5 * 2**15),
        gpu_max_rigid_contact_count=min(p.gpu_max_rigid_contact_count, 2**20),
        gpu_found_lost_pairs_capacity=min(p.gpu_found_lost_pairs_capacity, 2**19),
        gpu_found_lost_aggregate_pairs_capacity=min(p.gpu_found_lost_aggregate_pairs_capacity, 2**22),
        gpu_total_aggregate_pairs_capacity=min(p.gpu_total_aggregate_pairs_capacity, 2**19),
        gpu_collision_stack_size=min(p.gpu_collision_stack_size, 2**23),
        gpu_heap_capacity=min(p.gpu_heap_capacity, 2**24),
        gpu_temp_buffer_capacity=min(p.gpu_temp_buffer_capacity, 2**22),
    )


def _adapt_state_dict_pad_policy_obs(state_dict: dict, from_n: int, to_n: int, device=None) -> dict:
    """Pad policy/critic input dim from ``from_n`` to ``to_n`` (first Linear in_* and obs normalizers).

    New input dims use zero weights and normalizer mean 0 / var 1 / std 1 so existing behavior is unchanged.
    """
    if from_n >= to_n:
        return state_dict
    delta = to_n - from_n
    state_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    if device is not None:
        state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}

    def pad_cols(t: torch.Tensor, fill: float) -> torch.Tensor:
        if t.dim() != 2 or t.shape[1] != from_n:
            return t
        extra = torch.full((t.shape[0], delta), fill, dtype=t.dtype, device=t.device)
        return torch.cat([t, extra], dim=1)

    for key in list(state_dict.keys()):
        t = state_dict[key]
        if not isinstance(t, torch.Tensor):
            continue
        if key in ("actor.0.weight", "critic.0.weight"):
            state_dict[key] = pad_cols(t, 0.0)
        elif key in ("actor_obs_normalizer._mean", "critic_obs_normalizer._mean"):
            state_dict[key] = pad_cols(t, 0.0)
        elif key in ("actor_obs_normalizer._var", "critic_obs_normalizer._var"):
            state_dict[key] = pad_cols(t, 1.0)
        elif key in ("actor_obs_normalizer._std", "critic_obs_normalizer._std"):
            state_dict[key] = pad_cols(t, 1.0)
    return state_dict


# Reset position (key 0)
RESET_POSITION = [0.0, 0.0, 0.04]

# Neutral action: thrust ~25N each, yaw neutral
NEUTRAL_ACTION = torch.zeros(1, 8)

# Current policy: "vertical" | "horizontal" | None (neutral)
current_policy = [None]  # list for mutability in closures
request_reset = [False]


def stdin_key_listener():
    """Background thread: read stdin. 0=reset, 5=vertical, 8=horizontal."""
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            key = line.strip()
            if key == "0":
                request_reset[0] = True
                print("\n[INFO] Key 0 pressed - RESET to (0, 0, 0.04), zero velocity/force\n")
            elif key == "5":
                current_policy[0] = "vertical"
                print("\n[INFO] Key 5 pressed - VERTICAL TAKEOFF policy\n")
            elif key == "8":
                current_policy[0] = "horizontal"
                print("\n[INFO] Key 8 pressed - HORIZONTAL CRUISE policy\n")
    except Exception:
        pass


def reset_robot_to_position(env, x=0.0, y=0.0, z=0.04):
    """Set robot root to (x,y,z), zero velocity and zero thrust (env_id 0)."""
    unwrapped = env.unwrapped
    device = unwrapped.device
    env_id = torch.zeros(1, dtype=torch.long, device=device)
    root = unwrapped._robot.data.default_root_state[0:1].clone()
    root[:, 0] = x
    root[:, 1] = y
    root[:, 2] = z
    # identity quat (w,x,y,z); assign tensor, not tuple
    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=root.dtype)
    root[:, 7:10] = 0.0
    root[:, 10:13] = 0.0
    unwrapped._robot.write_root_pose_to_sim(root[:, :7], env_id)
    unwrapped._robot.write_root_velocity_to_sim(root[:, 7:13], env_id)
    unwrapped._spawn_quat = unwrapped._spawn_quat.clone()
    unwrapped._spawn_quat[0] = root[:, 3:7].clone()
    unwrapped._prev_lin_vel = unwrapped._prev_lin_vel.clone()
    unwrapped._prev_lin_vel[0] = 0.0
    unwrapped._thrust = unwrapped._thrust.clone()
    unwrapped._thrust[0] = 0.0
    unwrapped._yaw_target = unwrapped._yaw_target.clone()
    unwrapped._yaw_target[0] = 0.0
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()


def main():
    global current_policy, request_reset

    vertical_path = os.path.abspath(args_cli.vertical_checkpoint) if args_cli.vertical_checkpoint else None
    horizontal_path = os.path.abspath(args_cli.horizontal_checkpoint) if args_cli.horizontal_checkpoint else None

    if (vertical_path and not os.path.isfile(vertical_path)) or (horizontal_path and not os.path.isfile(horizontal_path)):
        print("Usage: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_ion_vtol_switch.py [options]")
        print("  Key 0: reset to (0, 0, 0.04), zero velocity/force")
        print("  Key 5: run vertical takeoff model (--vertical_checkpoint)")
        print("  Key 8: run horizontal cruise model (--horizontal_checkpoint)")
        print("  Example:")
        print("    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_ion_vtol_switch.py \\")
        print("      --vertical_checkpoint logs/rsl_rl/ion_vertical_takeoff_v1/2026-03-12_15-09-23/model_3000.pt \\")
        print("      --horizontal_checkpoint logs/rsl_rl/ion_horizontal_cruise_v1/2026-03-12_20-35-06/model_2200.pt")
        if vertical_path and not os.path.isfile(vertical_path):
            print(f"  Missing: {vertical_path}")
        if horizontal_path and not os.path.isfile(horizontal_path):
            print(f"  Missing: {horizontal_path}")
        simulation_app.close()
        return

    # Match vertical checkpoint obs width: V1=20, V2 default=22 (extra command + phase channels).
    vertical_in_dim = _policy_obs_in_dim_from_checkpoint(vertical_path) if vertical_path else None
    if vertical_in_dim == OBS_DIM_VERTICAL_V2_DEFAULT:
        task_vertical = TASK_VERTICAL_V2
    elif vertical_in_dim in (OBS_DIM_VERTICAL_V1, None):
        task_vertical = TASK_VERTICAL_V1
    else:
        print(
            f"[ERROR] Vertical checkpoint policy expects {vertical_in_dim}-dim observations; "
            f"supported sizes are {OBS_DIM_VERTICAL_V1} (V1) or {OBS_DIM_VERTICAL_V2_DEFAULT} (V2 default). "
            "Train with default V2 obs flags or use a V1 checkpoint."
        )
        simulation_app.close()
        return

    # One env: vertical takeoff task (physics/actions align with horizontal cruise; obs width may be V2).
    env_cfg: DirectRLEnvCfg = load_cfg_from_registry(task_vertical, "env_cfg_entry_point")
    agent_cfg_vertical = load_cfg_from_registry(task_vertical, "rsl_rl_cfg_entry_point")
    agent_cfg_horizontal = load_cfg_from_registry(TASK_HORIZONTAL, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = os.path.dirname(vertical_path) if vertical_path else ""
    device = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg.sim.device = device
    _shrink_physx_for_single_env_play(env_cfg)

    # Experience files often default RTX to DLSS; leaving render.antialiasing_mode as None does not disable that.
    # Forcing FXAA avoids GpuCompute-Cuda cudaMemcpy from (nil) on several driver/GPU combinations.
    if args_cli.dlss:
        env_cfg.sim.render.antialiasing_mode = "DLSS"
        env_cfg.sim.render.dlss_mode = 2
    else:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.enable_dl_denoiser = False

    print(
        "[INFO] Play VRAM tips: PhysX buffers capped for num_envs=1; viewport "
        f"{args_cli.width}x{args_cli.height}. If cudaErrorMemoryAllocation persists, use "
        "`--rendering_mode performance` (see PLAY_ION_VTOL_SWITCH.sh) or `--device cpu`."
    )

    env = gym.make(task_vertical, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_vertical.clip_actions)

    # Match RSL-RL / torch to the device SimulationContext resolved (e.g. "cuda" -> "cuda:0").
    rl_device = str(env.unwrapped.device)
    agent_cfg_vertical.device = rl_device
    agent_cfg_horizontal.device = rl_device

    policy_vertical = None
    policy_horizontal = None
    policy_nn_vertical = None
    policy_nn_horizontal = None

    if vertical_path and os.path.isfile(vertical_path):
        print(f"[INFO] Loading vertical takeoff model: {vertical_path}")
        runner_v = DivergenceAwareOnPolicyRunner(env, agent_cfg_vertical.to_dict(), log_dir=None, device=rl_device)
        runner_v.load(vertical_path)
        policy_vertical = runner_v.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn_vertical = runner_v.alg.policy
        except AttributeError:
            policy_nn_vertical = runner_v.alg.actor_critic
    else:
        print("[WARN] No vertical checkpoint; key 5 will use neutral action.")

    if horizontal_path and os.path.isfile(horizontal_path):
        print(f"[INFO] Loading horizontal cruise model: {horizontal_path}")
        runner_h = DivergenceAwareOnPolicyRunner(env, agent_cfg_horizontal.to_dict(), log_dir=None, device=rl_device)
        env_policy_obs_dim = int(env.unwrapped.cfg.observation_space)
        ckpt = torch.load(horizontal_path, map_location=rl_device, weights_only=False)
        model_sd = ckpt.get("model_state_dict", ckpt)
        ckpt_in = None
        w = model_sd.get("actor.0.weight")
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            ckpt_in = int(w.shape[1])
        if ckpt_in is not None and ckpt_in < env_policy_obs_dim:
            print(
                f"[INFO] Padding horizontal checkpoint obs {ckpt_in} -> {env_policy_obs_dim} "
                f"(extra channels unused by pretrained weights)."
            )
            model_sd = _adapt_state_dict_pad_policy_obs(model_sd, ckpt_in, env_policy_obs_dim, device=rl_device)
            policy_module = getattr(runner_h.alg, "policy", None) or getattr(runner_h.alg, "actor_critic")
            policy_module.load_state_dict(model_sd, strict=True)
        else:
            runner_h.load(horizontal_path)
        policy_horizontal = runner_h.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn_horizontal = runner_h.alg.policy
        except AttributeError:
            policy_nn_horizontal = runner_h.alg.actor_critic
    else:
        print("[WARN] No horizontal checkpoint; key 8 will use neutral action.")

    dt = env.unwrapped.step_dt

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
                    name = getattr(event.input, "name", "") or ""
                    if name in ("NUMPAD_0", "0"):
                        request_reset[0] = True
                        print("\n[INFO] Key 0 pressed - RESET to (0, 0, 0.04)\n")
                    elif name in ("NUMPAD_5", "5"):
                        current_policy[0] = "vertical"
                        print("\n[INFO] Key 5 pressed - VERTICAL TAKEOFF policy\n")
                    elif name in ("NUMPAD_8", "8"):
                        current_policy[0] = "horizontal"
                        print("\n[INFO] Key 8 pressed - HORIZONTAL CRUISE policy\n")
                return True

            input_interface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard)
        except Exception:
            use_carb = False

    if not use_carb:
        t = threading.Thread(target=stdin_key_listener, daemon=True)
        t.start()
        print("\n[INFO] Headless: type 0 / 5 / 8 and Enter (0=reset, 5=vertical, 8=horizontal)\n")
    else:
        print("\n[INFO] Keys: 0=reset to (0,0,0.04), 5=vertical takeoff, 8=horizontal cruise (window focus)\n")

    obs = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        if request_reset[0]:
            request_reset[0] = False
            reset_robot_to_position(env, *RESET_POSITION)

        with torch.inference_mode():
            sel = current_policy[0]
            if sel == "vertical" and policy_vertical is not None:
                actions = policy_vertical(obs)
                policy_nn = policy_nn_vertical
            elif sel == "horizontal" and policy_horizontal is not None:
                actions = policy_horizontal(obs)
                policy_nn = policy_nn_horizontal
            else:
                num_envs = obs["policy"].shape[0]
                actions = NEUTRAL_ACTION.to(env.unwrapped.device).expand(num_envs, -1).clone()
                policy_nn = None

            obs, _, dones, _ = env.step(actions)
            if policy_nn is not None:
                policy_nn.reset(dones)

        timestep += 1
        if args_cli.video and timestep >= getattr(args_cli, "video_length", 200):
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

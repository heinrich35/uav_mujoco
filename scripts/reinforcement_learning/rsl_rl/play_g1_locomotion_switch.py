# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play G1 locomotion with switchable policies: standing and walking.
# Key 0: reset env (position/velocity to default) and run standing model.
# Key 8: run walking model.
"""Play G1 — switch between standing and walking policies."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play G1: Key 0=reset + standing policy, 8=walking policy."
)
parser.add_argument(
    "--standing_checkpoint",
    type=str,
    default="logs/rsl_rl/g1_locomotion_v1_standing/2026-03-12_19-58-55/model_2700.pt",
    help="Path to standing .pt model (key 0 resets and runs this policy).",
)
parser.add_argument(
    "--walking_checkpoint",
    type=str,
    default="logs/rsl_rl/g1_locomotion_v1_walking/2026-03-13_14-49-30/model_2300.pt",
    help="Path to walking .pt model (key 8).",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
parser.add_argument("--video", action="store_true", default=False, help="Record video")
parser.add_argument("--seed", type=int, default=42, help="Environment seed")
parser.add_argument("--width", type=int, default=1920, help="Viewport width (pixels)")
parser.add_argument("--height", type=int, default=1080, help="Viewport height (pixels)")
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
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.divergence_aware_runner import DivergenceAwareOnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Standing task = base G1 locomotion V1 (no forward target velocity)
TASK_STANDING = "Isaac-G1-Locomotion-V1-Direct-v0"
# Walking task = G1 locomotion V1 walking (target +x velocity)
TASK_WALKING = "Isaac-G1-Locomotion-V1-Walking-Direct-v0"

# G1 action space: 29 DOF joint efforts
NEUTRAL_ACTION = torch.zeros(1, 29)

# Current policy: "standing" | "walking" | None (neutral)
current_policy = [None]  # list for mutability in closures
request_reset = [False]


def stdin_key_listener():
    """Background thread: read stdin. 0=reset+standing, 8=walking."""
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            key = line.strip()
            if key == "0":
                request_reset[0] = True
                print("\n[INFO] Key 0 pressed - RESET and STANDING policy\n")
            elif key == "8":
                current_policy[0] = "walking"
                print("\n[INFO] Key 8 pressed - WALKING policy\n")
    except Exception:
        pass


def get_obs_dim_from_checkpoint(path: str) -> int | None:
    """Read policy input dim from checkpoint (actor.0.weight shape [out, in]). Returns None on error."""
    try:
        kwargs = {"map_location": "cpu"}
        if hasattr(torch, "weights_only"):
            kwargs["weights_only"] = True
        ckpt = torch.load(path, **kwargs)
        state = ckpt.get("model_state_dict") or ckpt
        key = "actor.0.weight"
        if key not in state:
            return None
        return int(state[key].shape[1])
    except Exception:
        return None


class ObsSliceWrapper:
    """Wraps env so get_observations() returns obs sliced to obs_dim (for checkpoints trained with smaller obs)."""

    def __init__(self, env, obs_dim: int):
        self.env = env
        self.obs_dim = obs_dim

    def get_observations(self):
        o = self.env.get_observations()
        if "policy" in o:
            sliced = o["policy"][:, : self.obs_dim].clone()
            return TensorDict({"policy": sliced}, batch_size=o.batch_size)
        return o

    def __getattr__(self, name):
        return getattr(self.env, name)


def main():
    global current_policy, request_reset

    standing_path = os.path.abspath(args_cli.standing_checkpoint) if args_cli.standing_checkpoint else None
    walking_path = os.path.abspath(args_cli.walking_checkpoint) if args_cli.walking_checkpoint else None

    if (standing_path and not os.path.isfile(standing_path)) or (walking_path and not os.path.isfile(walking_path)):
        print("Usage: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_g1_locomotion_switch.py [options]")
        print("  Key 0: reset position/velocity to default and run standing model (--standing_checkpoint)")
        print("  Key 8: run walking model (--walking_checkpoint)")
        print("  Example:")
        print("    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_g1_locomotion_switch.py \\")
        print("      --standing_checkpoint logs/rsl_rl/g1_locomotion_v1_standing/2026-03-12_19-58-55/model_2700.pt \\")
        print("      --walking_checkpoint logs/rsl_rl/g1_locomotion_v1_walking/2026-03-13_14-49-30/model_2300.pt")
        if standing_path and not os.path.isfile(standing_path):
            print(f"  Missing: {standing_path}")
        if walking_path and not os.path.isfile(walking_path):
            print(f"  Missing: {walking_path}")
        simulation_app.close()
        return

    # One env: use standing task (same obs/action as walking)
    env_cfg: DirectRLEnvCfg = load_cfg_from_registry(TASK_STANDING, "env_cfg_entry_point")
    agent_cfg_standing = load_cfg_from_registry(TASK_STANDING, "rsl_rl_cfg_entry_point")
    agent_cfg_walking = load_cfg_from_registry(TASK_WALKING, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = os.path.dirname(standing_path) if standing_path else ""
    device = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg.sim.device = device
    agent_cfg_standing.device = device
    agent_cfg_walking.device = device

    # Reduce PhysX GPU buffer sizes for single-env play (avoid CUDA OOM, e.g. mGpuContactPairsDev 640MB)
    if hasattr(env_cfg, "sim") and getattr(env_cfg.sim, "physx", None) is not None:
        physx = env_cfg.sim.physx
        physx.gpu_max_rigid_contact_count = min(physx.gpu_max_rigid_contact_count, 2**14)
        physx.gpu_max_rigid_patch_count = min(physx.gpu_max_rigid_patch_count, 2**10)
        physx.gpu_found_lost_pairs_capacity = min(physx.gpu_found_lost_pairs_capacity, 2**14)
        physx.gpu_found_lost_aggregate_pairs_capacity = min(physx.gpu_found_lost_aggregate_pairs_capacity, 2**16)
        physx.gpu_total_aggregate_pairs_capacity = min(physx.gpu_total_aggregate_pairs_capacity, 2**14)
        physx.gpu_collision_stack_size = min(physx.gpu_collision_stack_size, 2**18)
        physx.gpu_heap_capacity = min(physx.gpu_heap_capacity, 2**20)
        physx.gpu_temp_buffer_capacity = min(physx.gpu_temp_buffer_capacity, 2**18)
        print("[INFO] Reduced PhysX GPU buffer sizes for play (avoid OOM)")

    env_cfg.sim.render.antialiasing_mode = "DLSS"
    env_cfg.sim.render.dlss_mode = 2

    env = gym.make(TASK_STANDING, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_standing.clip_actions)

    policy_standing = None
    policy_walking = None
    policy_nn_standing = None
    policy_nn_walking = None
    standing_obs_dim = None  # obs dim expected by standing policy (e.g. 99 or 103)
    walking_obs_dim = None   # obs dim expected by walking policy

    env_obs_dim = env_cfg.observation_space

    if standing_path and os.path.isfile(standing_path):
        standing_obs_dim = get_obs_dim_from_checkpoint(standing_path)
        if standing_obs_dim is None:
            standing_obs_dim = env_obs_dim
        print(f"[INFO] Loading standing model: {standing_path} (obs_dim={standing_obs_dim})")
        env_for_standing = ObsSliceWrapper(env, standing_obs_dim) if standing_obs_dim != env_obs_dim else env
        runner_s = DivergenceAwareOnPolicyRunner(
            env_for_standing, agent_cfg_standing.to_dict(), log_dir=None, device=agent_cfg_standing.device
        )
        runner_s.load(standing_path)
        policy_standing = runner_s.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn_standing = runner_s.alg.policy
        except AttributeError:
            policy_nn_standing = runner_s.alg.actor_critic
    else:
        print("[WARN] No standing checkpoint; key 0 will use neutral action.")

    if walking_path and os.path.isfile(walking_path):
        walking_obs_dim = get_obs_dim_from_checkpoint(walking_path)
        if walking_obs_dim is None:
            walking_obs_dim = env_obs_dim
        print(f"[INFO] Loading walking model: {walking_path} (obs_dim={walking_obs_dim})")
        env_for_walking = ObsSliceWrapper(env, walking_obs_dim) if walking_obs_dim != env_obs_dim else env
        runner_w = DivergenceAwareOnPolicyRunner(
            env_for_walking, agent_cfg_walking.to_dict(), log_dir=None, device=agent_cfg_walking.device
        )
        runner_w.load(walking_path)
        policy_walking = runner_w.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn_walking = runner_w.alg.policy
        except AttributeError:
            policy_nn_walking = runner_w.alg.actor_critic
    else:
        print("[WARN] No walking checkpoint; key 8 will use neutral action.")

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
                        current_policy[0] = "standing"
                        print("\n[INFO] Key 0 pressed - RESET and STANDING policy\n")
                    elif name in ("NUMPAD_8", "8"):
                        current_policy[0] = "walking"
                        print("\n[INFO] Key 8 pressed - WALKING policy\n")
                return True

            input_interface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard)
        except Exception:
            use_carb = False

    if not use_carb:
        t = threading.Thread(target=stdin_key_listener, daemon=True)
        t.start()
        print("\n[INFO] Headless: type 0 / 8 and Enter (0=reset+standing, 8=walking)\n")
    else:
        print("\n[INFO] Keys: 0=reset + standing, 8=walking (window focus)\n")

    # Start with standing policy after first reset
    current_policy[0] = "standing"
    obs, _ = env.reset()

    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        if request_reset[0]:
            request_reset[0] = False
            obs, _ = env.reset()
            current_policy[0] = "standing"

        with torch.inference_mode():
            sel = current_policy[0]
            if sel == "standing" and policy_standing is not None:
                obs_standing = {"policy": obs["policy"][:, :standing_obs_dim]} if standing_obs_dim is not None else obs
                actions = policy_standing(obs_standing)
                policy_nn = policy_nn_standing
            elif sel == "walking" and policy_walking is not None:
                obs_walking = {"policy": obs["policy"][:, :walking_obs_dim]} if walking_obs_dim is not None else obs
                actions = policy_walking(obs_walking)
                policy_nn = policy_nn_walking
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

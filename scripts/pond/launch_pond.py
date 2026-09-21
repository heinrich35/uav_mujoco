#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the Pond environment with keyboard teleoperation of the Gosling robot.

The gosling robot floats on a water surface (z=0) and can be teleoperated:

.. list-table:: Keyboard controls
   :header-rows: 1

   * - Key
     - Action
   * - Arrow Up
     - Move forward (paddle force in body +X)
   * - Arrow Right
     - Turn left (negative yaw torque)
   * - Arrow Left
     - Turn right (positive yaw torque)
   * - X / Escape
     - Stop simulation and exit

Usage:
    ./isaaclab.sh -p scripts/pond/launch_pond.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Launch the Pond environment with keyboard teleoperation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--vision", action="store_true", default=False,
                    help="Enable real-time duck/swan detection on gosling camera")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras for the gosling forward camera
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch

import carb
import omni.appwindow

import gymnasium as gym

# Import tasks to register the Pond environment
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main():
    """Run the pond simulation with keyboard teleoperation."""
    # ------------------------------------------------------------------
    # Keyboard state tracking
    # ------------------------------------------------------------------
    should_stop = False

    # Key states: True = pressed, False = released
    key_state = {
        "forward": False,   # Arrow Up
        "turn_right": False,  # Arrow Right
        "turn_left": False,   # Arrow Left
    }

    # ------------------------------------------------------------------
    # Keyboard event callback
    # ------------------------------------------------------------------
    def on_keyboard_event(event, *args, **kwargs):
        """Handle keyboard events for teleoperation."""
        nonlocal should_stop, vision, active_model_key

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name

            if name == "UP":
                key_state["forward"] = True
            elif name == "RIGHT":
                key_state["turn_left"] = True
            elif name == "LEFT":
                key_state["turn_right"] = True
            elif name in ("X", "ESCAPE"):
                print("\n[INFO] Exiting simulation...")
                should_stop = True
                simulation_app.close()
            # Model switching: number keys 1-9 (vision mode only)
            elif name in ("1", "2", "3", "4", "5", "6", "7", "8", "9") \
                    or name in ("NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4",
                                "NUMPAD_5", "NUMPAD_6", "NUMPAD_7", "NUMPAD_8",
                                "NUMPAD_9"):
                if vision is not None:
                    k = int(name[-1])
                    if k in MODEL_PRESETS:
                        active_model_key = k
                        vision = _build_vision(k)
            # SAHI toggle: S key
            elif name in ("S", "s", "NUMPAD_S"):
                if vision is not None:
                    sahi_enabled[0] = not sahi_enabled[0]
                    vision = _build_vision(active_model_key)
            # Ensemble toggle: E key
            elif name in ("E", "e", "NUMPAD_E"):
                if vision is not None:
                    ensemble_enabled[0] = not ensemble_enabled[0]
                    vision = _build_vision(active_model_key)
            # Confidence threshold: - / =  (±0.05)
            elif name in ("MINUS", "-", "NUMPAD_MINUS", "NUMPAD_SUBTRACT"):
                if vision is not None and vision_conf[0] > 0.05:
                    vision_conf[0] = max(0.05, vision_conf[0] - 0.05)
                    vision.conf_threshold = vision_conf[0]
                    print(f"  [Vision] conf≥{vision_conf[0]:.2f}  iou≥{vision_iou[0]:.2f}")
            elif name in ("EQUALS", "=", "NUMPAD_PLUS", "NUMPAD_ADD"):
                if vision is not None and vision_conf[0] < 0.95:
                    vision_conf[0] = min(0.95, vision_conf[0] + 0.05)
                    vision.conf_threshold = vision_conf[0]
                    print(f"  [Vision] conf≥{vision_conf[0]:.2f}  iou≥{vision_iou[0]:.2f}")
            # IoU threshold: [ / ]  (±0.05)
            elif name in ("LEFT_BRACKET", "[", "NUMPAD_LEFT_BRACKET"):
                if vision is not None and vision_iou[0] > 0.10:
                    vision_iou[0] = max(0.10, vision_iou[0] - 0.05)
                    vision.iou_threshold = vision_iou[0]
                    print(f"  [Vision] conf≥{vision_conf[0]:.2f}  iou≥{vision_iou[0]:.2f}")
            elif name in ("RIGHT_BRACKET", "]", "NUMPAD_RIGHT_BRACKET"):
                if vision is not None and vision_iou[0] < 0.95:
                    vision_iou[0] = min(0.95, vision_iou[0] + 0.05)
                    vision.iou_threshold = vision_iou[0]
                    print(f"  [Vision] conf≥{vision_conf[0]:.2f}  iou≥{vision_iou[0]:.2f}")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "UP":
                key_state["forward"] = False
            elif event.input.name == "RIGHT":
                key_state["turn_left"] = False
            elif event.input.name == "LEFT":
                key_state["turn_right"] = False

        return True

    # ------------------------------------------------------------------
    # Set up keyboard input
    # ------------------------------------------------------------------
    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    task_name = "Isaac-Pond-v0"

    # Load the environment configuration from the task registry
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    # Override config from CLI args if needed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Create the environment via gym
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # ── Model presets for key-switching ─────────────────────────────────
    # Each preset is a (stage1_weights, stage2_weights, label, conf_threshold) tuple.
    # Press number keys 1-9 to hot-swap between models at runtime.
    S2_LATEST = "runs/pond_stage2/pond_s2_v3/best.pt"
    S2_OLD    = "runs/pond_stage2/pond_s2_v1/best.pt"
    S1_PREFIX = "runs/detect/runs/pond_stage1"
    MODEL_PRESETS = {
        1: (f"{S1_PREFIX}/pond_v2_phase2/weights/best.pt",
            S2_LATEST, "pond_v2  (new train arch, S2 v3)", 0.20),
        2: (f"{S1_PREFIX}/pond_v1_phase2-4/weights/best.pt",
            S2_LATEST, "pond_v1_phase2-4  (prev best + S2 v3)", 0.25),
        3: (f"{S1_PREFIX}/pond_v1_phase2-5_phase2-2/weights/best.pt",
            S2_LATEST, "pond_v1_phase2-5  (extended ds, S2 v3)", 0.10),
        4: (f"{S1_PREFIX}/pond_v1_phase2-3/weights/best.pt",
            S2_OLD,    "pond_v1_phase2-3  (1000 B frames + S2 v1)", 0.25),
        5: (f"{S1_PREFIX}/pond_v1_phase2-2/weights/best.pt",
            S2_OLD,    "pond_v1_phase2-2  (400 B, orig zones + S2 v1)", 0.25),
    }

    # ── Vision pipeline (optional) ──────────────────────────────────────
    vision = None
    active_model_key = 1  # mutable so keyboard callback can change it
    # Runtime-adjustable inference params (mutable containers for callback access)
    vision_conf = [0.15]   # confidence threshold
    vision_iou  = [0.45]   # NMS IoU threshold
    # SAHI + Ensemble toggles
    sahi_enabled   = [True]   # SAHI tiled inference
    ensemble_enabled = [True] # WBF ensemble (use with SAHI)
    if args_cli.vision:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pond.vision_inference import VisionPipeline, EnsemblePipeline
        import cv2

        def _make_pipeline(s1_path, s2_path, name):
            """Create a single VisionPipeline (used individually or in ensemble)."""
            p = VisionPipeline(
                stage1_weights=s1_path, stage2_weights=s2_path,
                device=args_cli.device if args_cli.device else "cuda:0",
                conf_threshold=vision_conf[0],
                name=name,
            )
            return p

        def _build_vision(key):
            """Build pipeline(s) for a preset key: ensemble of key-1 + key-2, or solo."""
            preset = MODEL_PRESETS[key]
            s1, s2, label, default_conf = preset[0], preset[1], preset[2], preset[3]

            # Primary pipeline for this key
            primary = _make_pipeline(s1, s2, f"k{key}")

            if ensemble_enabled[0] and key == 1:
                # Ensemble: phase2-4 (key 1) + phase2-5 (key 2)
                s1b, s2b, _, _ = MODEL_PRESETS[2]
                secondary = _make_pipeline(s1b, s2b, "k2")
                ensemble = EnsemblePipeline(
                    [primary, secondary],
                    sahi=sahi_enabled[0], sahi_rows=2, sahi_overlap=0.35,
                )
                # Propagate conf/iou
                ensemble.conf_threshold = vision_conf[0]
                ensemble.iou_threshold = vision_iou[0]
                mode = "SAHI+WBF" if sahi_enabled[0] else "WBF"
                print(f"  [Vision] Key {key}: {label}  [{mode}]  (conf≥{vision_conf[0]:.2f})")
                return ensemble
            else:
                primary.sahi = sahi_enabled[0]
                primary.sahi_rows = 2
                primary.sahi_overlap = 0.35
                if sahi_enabled[0]:
                    primary.conf_threshold = vision_conf[0]  # lower conf for SAHI
                mode = "SAHI" if sahi_enabled[0] else "full"
                print(f"  [Vision] Key {key}: {label}  [{mode}]  (conf≥{vision_conf[0]:.2f})")
                return primary

        vision = _build_vision(1)
        cv2.namedWindow("Gosling Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gosling Vision", 960, 540)
        print("  [Vision] Real-time 6-class pond object detection ENABLED")
        print("  [Vision] 1-5:model  -/=:conf  [/]:iou  S:SAHI  E:ensemble")

    # Extract fluid config for actuation limits
    fluid_cfg = env.unwrapped._fluid_cfg
    max_forward = fluid_cfg.max_forward_force
    max_turn = fluid_cfg.max_turn_torque

    # ------------------------------------------------------------------
    # Reset and get initial state
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    step_dt = env.unwrapped.step_dt

    print("\n" + "=" * 60)
    print("  POND ENVIRONMENT — Gosling Water Robot Teleoperation")
    print("=" * 60)
    print()
    print("  Controls:")
    print(f"    Arrow Up      — Move forward  ({max_forward:.0f} N paddle force)")
    print(f"    Arrow Right   — Turn left      ({max_turn:.0f} N·m yaw torque)")
    print(f"    Arrow Left    — Turn right     ({max_turn:.0f} N·m yaw torque)")
    print("    X / Escape    — Stop simulation")
    print()
    print("  Fluid dynamics:")
    print(f"    Water density:      {fluid_cfg.water_density:.0f} kg/m³")
    print(f"    Displaced volume:   {fluid_cfg.displaced_volume * 1000:.1f} liters")
    print(f"    Buoyancy ramp:       {fluid_cfg.buoyancy_ramp_depth * 100:.0f} cm")
    print(f"    Max forward force:  {max_forward:.0f} N")
    print(f"    Max turn torque:    {max_turn:.0f} N·m")
    print()
    print("  Make sure the Isaac Sim window has focus for keyboard input.")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    step_count = 0
    print_interval = 120  # Print status every N steps

    while simulation_app.is_running() and not should_stop:
        start_time = time.time()

        # --- Compute paddle command from key state ---
        forward_force = max_forward if key_state["forward"] else 0.0
        turn_torque = 0.0
        if key_state["turn_right"]:
            turn_torque = max_turn
        elif key_state["turn_left"]:
            turn_torque = -max_turn

        # Send paddle command to environment
        env.unwrapped.set_paddle_command(forward_force, turn_torque)

        # --- Step the environment ---
        # Use a zero-dimension action tensor (paddle commands are set via set_paddle_command)
        action = torch.zeros(env.unwrapped.num_envs, 0, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Vision inference ---
        if vision is not None:
            try:
                rgb = env.unwrapped.scene["gosling_cam"].data.output["rgb"][0]
                # Camera: (H=640,W=480,3) portrait with -90° roll quaternion
                # Model expects landscape (W=640,H=480): k=3 gives (480,640,3)
                rgb_rot = torch.rot90(rgb, k=3, dims=(0, 1))  # 90° CCW
                detections = vision.infer(rgb_rot.clone())
                img_np = vision.preprocess_image(rgb_rot.clone())

                if step_count == 1:
                    print(f"[Vision] Camera: {rgb.shape}→{rgb_rot.shape}")

                vision.log_detections(detections)

                # Show live overlay — annotate, rotate 90° CCW, resize to 640×480
                annotated = vision.annotate_frame(img_np, detections)
                display = cv2.rotate(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                     cv2.ROTATE_90_COUNTERCLOCKWISE)
                display = cv2.resize(display, (640, 480))
                # Status bar: model + inference params + mode indicators
                mode_bits = []
                if sahi_enabled[0]:
                    mode_bits.append("SAHI")
                if ensemble_enabled[0]:
                    mode_bits.append("WBF")
                mode_str = "+".join(mode_bits) if mode_bits else "full"
                bar = (f"M{active_model_key}  {mode_str}  "
                       f"conf>{vision_conf[0]:.2f}  iou>{vision_iou[0]:.2f}")
                cv2.putText(display, bar, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                cv2.imshow("Gosling Vision", display)
                cv2.waitKey(1)
            except Exception as e:
                if step_count <= 2:
                    import traceback; traceback.print_exc()

        # --- Status print ---
        step_count += 1
        if step_count % print_interval == 0:
            pos = env.unwrapped.get_robot_position()
            vel = env.unwrapped.get_robot_velocity()
            depth = env.unwrapped.get_submersion_depth()
            print(
                f"[Step {step_count:6d}] "
                f"pos=({pos[0, 0]:+6.2f}, {pos[0, 1]:+6.2f}, {pos[0, 2]:+6.3f}) m | "
                f"speed={torch.norm(vel[0]):5.2f} m/s | "
                f"depth={depth[0]:5.3f} m | "
                f"fwd={'ON ' if key_state['forward'] else 'off'} "
                f"R={'ON ' if key_state['turn_right'] else 'off'} "
                f"L={'ON ' if key_state['turn_left'] else 'off'}"
            )

        # --- Reset if terminated ---
        if terminated.any() or truncated.any():
            print("[INFO] Episode terminated. Resetting...")
            obs, _ = env.reset()

        # --- Real-time delay ---
        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if keyboard_sub is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard_sub)

    if vision is not None:
        cv2.destroyAllWindows()

    env.close()
    print("[INFO] Pond simulation stopped.")


if __name__ == "__main__":
    main()
    simulation_app.close()

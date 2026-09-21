#!/usr/bin/env python3
"""Launch the Pond environment with GROUND-TRUTH object labels overlaid on camera.

Instead of a vision model, this reads the camera's bounding-box annotator
(the same one used during data generation) and draws ground-truth boxes
directly onto the camera feed.

Usage:
    ./isaaclab.sh -p scripts/pond/launch_pond_gt.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pond environment — ground-truth camera overlay.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import numpy as np
import torch

import carb
import omni.appwindow

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# ── Distinct BGR colours (same palette as vision_inference.py) ───────────
LABEL_COLORS_BGR = {
    "duck":    (0, 215, 255),    # gold
    "swan":    (255, 255, 255),  # white
    "turtle":  (255, 160, 60),   # bright blue
    "lilypad": (120, 255, 60),   # lime green
    "log":     (180, 110, 0),    # deep sky blue
    "rock":    (255, 100, 160),  # violet
    "UNKNOWN": (0, 0, 255),      # red
}

# ── Compact icons ────────────────────────────────────────────────────────
LABEL_ICONS = {
    "duck": "🦆", "swan": "🦢", "turtle": "🐢",
    "lilypad": "🍀", "log": "🪵", "rock": "🪨",
}


def _get_camera_projection():
    """Build camera intrinsics + extrinsics from the gosling camera prim."""
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/gosling/camera")
    if not cam_prim or not cam_prim.IsValid():
        return None

    cam = UsdGeom.Camera(cam_prim)
    focal_mm = cam.GetFocalLengthAttr().Get() or 24.0
    aperture_w = cam.GetHorizontalApertureAttr().Get() or 20.955
    aperture_h = cam.GetVerticalApertureAttr().Get() or 15.291
    W, H = 640, 480
    fx = (focal_mm / aperture_w) * W
    fy = (focal_mm / aperture_h) * H
    K = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=np.float64)

    xf = UsdGeom.Xformable(cam_prim)
    ts = xf.GetTimeSamples()
    mat = xf.ComputeLocalToWorldTransform(ts[0] if ts else Usd.TimeCode.Default())
    world_T = np.array(mat.GetTranspose(), dtype=np.float64)
    # world_T is column-major: world_T[:3, 3] = position, world_T[:3, :3] = rotation matrix columns

    # Camera local axes in world frame (columns of rotation matrix)
    # Extract camera position and orientation
    t_world = world_T[:3, 3].copy()
    R_wc = world_T[:3, :3].copy()  # world-from-camera rotation

    # Build camera-from-world: R_cw = R_wc^T
    R_cam = R_wc.T
    t_cam = -R_cam @ t_world
    return K, R_cam, t_cam


def _get_object_positions(env):
    """Get (label, world_xyz) for pond objects visible in the scene.

    Isaac Lab stores rigid objects in ``env.unwrapped.scene`` as named
    entries (e.g. ``duck_0``, ``rock_1``).  We collect their root positions
    and infer the label from the key name.
    """
    objects = []
    valid = {"duck", "swan", "turtle", "lilypad", "log", "rock"}
    scene = env.unwrapped.scene
    for key in dir(scene):
        obj = getattr(scene, key, None)
        if obj is None:
            continue
        if not hasattr(obj, "data") or not hasattr(obj.data, "root_pos_w"):
            continue
        lbl = ""
        for v in valid:
            if v in key.lower():
                lbl = v
                break
        if not lbl:
            continue
        pos = obj.data.root_pos_w[0].cpu().numpy()
        objects.append((lbl, pos))
    return objects


def project_objects_to_screen(proj, objects, img_w=640, img_h=480):
    """Project 3D world points to 2D pixel coordinates via camera matrix.

    Returns list of {x1,y1,x2,y2,label} for each object in front of the camera.
    """
    if proj is None:
        return []
    K, R_cam, t_cam = proj
    boxes = []
    for label, p_w in objects:
        p_c = R_cam @ p_w + t_cam    # world → camera space (camera looks -Z... or +Y?)
        # Try both camera conventions
        if p_c[2] < 0.1:  # behind camera in Z-forward convention
            # Try Y-forward convention (USD camera default)
            p_c = np.array([p_c[0], p_c[2], -p_c[1]], dtype=np.float64)
        if p_c[2] < 0.05:
            continue
        u = K[0, 0] * p_c[0] / p_c[2] + K[0, 2]
        v = K[1, 1] * p_c[1] / p_c[2] + K[1, 2]
        dist = abs(p_c[2])
        half = max(10, int(50.0 / max(dist, 1.0)))
        x1, y1 = int(u) - half, int(v) - half
        x2, y2 = int(u) + half, int(v) + half
        if x2 > 0 and y2 > 0 and x1 < img_w and y1 < img_h:
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
    return boxes


def draw_boxes(img_bgr, boxes):
    """Draw ground-truth bounding boxes onto a BGR OpenCV image."""
    import cv2
    H, W = img_bgr.shape[:2]
    cv2.line(img_bgr, (0, H // 2), (W, H // 2), (0, 255, 255), 1)
    TARGETS = {"duck", "swan"}
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    font_thickness = 1

    for box in boxes:
        x1, y1 = max(0, box["x1"]), max(0, box["y1"])
        x2, y2 = min(W, box["x2"]), min(H, box["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        label = box.get("label", "UNKNOWN")
        color = LABEL_COLORS_BGR.get(label, LABEL_COLORS_BGR["UNKNOWN"])
        icon = LABEL_ICONS.get(label, "?")
        thickness = 3 if label in TARGETS else 2

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        text = f"{label}  gt"
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 4),
                      (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - baseline - 2),
                    font, font_scale, (0, 0, 0), font_thickness)
    return img_bgr


def main():
    should_stop = False
    key_state = {"forward": False, "turn_right": False, "turn_left": False}

    def on_keyboard_event(event, *args, **kwargs):
        nonlocal should_stop
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "UP":
                key_state["forward"] = True
            elif event.input.name == "RIGHT":
                key_state["turn_left"] = True
            elif event.input.name == "LEFT":
                key_state["turn_right"] = True
            elif event.input.name in ("X", "ESCAPE"):
                should_stop = True
                simulation_app.close()
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name == "UP":
                key_state["forward"] = False
            elif event.input.name == "RIGHT":
                key_state["turn_left"] = False
            elif event.input.name == "LEFT":
                key_state["turn_right"] = False
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    task_name = "Isaac-Pond-v0"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # ── Ground-truth camera window ───────────────────────────────────────
    import cv2
    cv2.namedWindow("Gosling Camera (GT)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gosling Camera (GT)", 960, 540)
    print("  [GT] Ground-truth bounding box overlay ENABLED")

    fluid_cfg = env.unwrapped._fluid_cfg
    max_forward = fluid_cfg.max_forward_force
    max_turn = fluid_cfg.max_turn_torque

    obs, _ = env.reset()
    step_dt = env.unwrapped.step_dt

    print("\n" + "=" * 60)
    print("  POND ENVIRONMENT — Ground-Truth Camera Overlay")
    print("=" * 60)
    print()
    print("  Controls:")
    print(f"    Arrow Up      — Move forward  ({max_forward:.0f} N)")
    print(f"    Arrow Right   — Turn left      ({max_turn:.0f} N·m)")
    print(f"    Arrow Left    — Turn right     ({max_turn:.0f} N·m)")
    print("    X / Escape    — Stop simulation")
    print()
    print("  GT Overlay: bounding boxes from camera annotator")
    print("    Thick border = duck / swan (targets)")
    print("    Thin border  = distractor (turtle, lilypad, log, rock)")
    print("=" * 60)
    print()

    step_count = 0
    print_interval = 120
    camera = env.unwrapped.scene["gosling_cam"]

    while simulation_app.is_running() and not should_stop:
        start_time = time.time()

        forward_force = max_forward if key_state["forward"] else 0.0
        turn_torque = 0.0
        if key_state["turn_right"]:
            turn_torque = max_turn
        elif key_state["turn_left"]:
            turn_torque = -max_turn
        env.unwrapped.set_paddle_command(forward_force, turn_torque)

        action = torch.zeros(env.unwrapped.num_envs, 0, device=env.unwrapped.device)
        obs, reward, terminated, truncated, info = env.step(action)

        # ── Ground-truth overlay ─────────────────────────────────────────
        try:
            if step_count == 0:
                # Diagnostic: camera output keys + scene objects
                keys = list(camera.data.output.keys())
                print(f"[GT] Camera keys: {keys}")
                objs = _get_object_positions(env)
                print(f"[GT] Found {len(objs)} objects: {[l for l,_ in objs[:10]]}")

            rgb = camera.data.output["rgb"][0]
            if rgb.shape[0] == 640 and rgb.shape[1] == 480:
                rgb = torch.rot90(rgb, k=3, dims=(0, 1))
            img = rgb.cpu().numpy().copy()
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Project objects to screen every 10 steps (camera moves slowly)
            if step_count % 10 == 0 or step_count == 0:
                proj = _get_camera_projection()
                objs = _get_object_positions(env)

            boxes = project_objects_to_screen(proj, objs)
            if boxes:
                img_bgr = draw_boxes(img_bgr, boxes)

            display = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            display = cv2.resize(display, (640, 480))
            cv2.imshow("Gosling Camera (GT)", display)
            cv2.waitKey(1)

            if step_count <= 2:
                print(f"[GT] Step {step_count}: {len(boxes)} boxes")
        except Exception as e:
            if step_count <= 2:
                import traceback
                traceback.print_exc()

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

        if terminated.any() or truncated.any():
            print("[INFO] Episode terminated. Resetting...")
            obs, _ = env.reset()

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if keyboard_sub is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard_sub)
    cv2.destroyAllWindows()
    env.close()
    print("[INFO] Pond simulation stopped.")


if __name__ == "__main__":
    main()
    simulation_app.close()

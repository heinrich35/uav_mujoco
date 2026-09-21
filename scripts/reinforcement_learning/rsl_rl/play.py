# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--viewer_env_index",
    type=int,
    default=None,
    help="If set, viewport camera uses this environment index (0-based) as origin (viewer.origin_type=env). Requires num_envs > index.",
)
parser.add_argument(
    "--debug_training_fixed_base",
    action="store_true",
    help="G1 Locomotion V3: fix robot root at env-local (0,0,z); joints still move.",
)
parser.add_argument(
    "--debug_training_fixed_base_spawn_z",
    type=float,
    default=None,
    help="Root spawn z (m) with --debug_training_fixed_base; default 0.75 from env cfg.",
)
parser.add_argument(
    "--joints_data",
    type=str,
    default=None,
    help="G1 locomotion: motion file path (e.g. hdf5/walk_action.hdf5 under joints_data/). Match training for soft limits / motion loaders.",
)
parser.add_argument(
    "--frames_per_second",
    "--motion_clip_fps",
    "--hdf5_fps",
    type=int,
    default=None,
    dest="frames_per_second",
    metavar="FPS",
    help="G1 locomotion: motion clip FPS override (hdf5_fps_override). Must match training when loading imitation.",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="G1 locomotion: physics steps per policy step; match training.",
)
parser.add_argument(
    "--sim_dt",
    type=float,
    default=None,
    help="G1 locomotion: override sim.dt (seconds); match training.",
)
parser.add_argument(
    "--include_root_xy_error_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="G1 walking: override root (x,y) drift in policy obs. Must match training (default V3: True → 105-dim with ankle torque).",
)
parser.add_argument(
    "--root_xy_error_obs_scale",
    type=float,
    default=None,
    help="G1 walking: override scale for root XY obs channel. Match training.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--width", type=int, default=1920, help="Viewport width (passed to SimulationApp). Use with --rendering_mode quality for high-res."
)
parser.add_argument(
    "--height", type=int, default=1080, help="Viewport height (passed to SimulationApp)."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log_joints", action="store_true", default=False, help="Log joint positions over time for analysis.")
parser.add_argument("--log_interval", type=int, default=10, help="Interval (in steps) for logging joint data to console.")
parser.add_argument("--log_file", type=str, default=None, help="Path to CSV file for saving joint data. If None, saves to log_dir/joint_log.csv")
parser.add_argument("--debug_joint_positions", action="store_true", default=False, help="Print joint positions to console for debugging.")
parser.add_argument("--debug_joint_interval", type=int, default=15, help="Interval (in frames/steps) when --debug_joint_interval_ms is not set. Default: 15")
parser.add_argument(
    "--debug_joint_interval_ms",
    type=int,
    default=None,
    metavar="MS",
    help="If set with --debug_joint_positions, print on sim time every MS milliseconds (overrides --debug_joint_interval).",
)
parser.add_argument(
    "--debug_joint_filter",
    type=str,
    default=None,
    metavar="NAME1,NAME2,...",
    help="With --debug_joint_positions: print only these comma-separated joint names (e.g. right_shoulder_pitch_joint,left_shoulder_roll_joint).",
)
parser.add_argument(
    "--debug_ankle_contact",
    action="store_true",
    default=False,
    help="Print ankle contact (in_contact boolean) and external torque from ankle joints for foot-step phase. Uses right/left ankle pitch and roll.",
)
parser.add_argument(
    "--debug_ankle_interval_ms",
    type=int,
    default=25,
    help="Interval in milliseconds for ankle contact debug output. Default: 25",
)
parser.add_argument(
    "--debug_g1_link_positions",
    action="store_true",
    default=False,
    help="Print world-frame body origin positions (m) for right_palm_link, left_palm_link, right_ankle_roll_link, left_ankle_roll_link (env 0).",
)
parser.add_argument(
    "--debug_g1_link_interval_ms",
    type=int,
    default=None,
    metavar="MS",
    help="Sim-time interval for --debug_g1_link_positions. Default: same as --debug_joint_interval_ms, else 100.",
)
parser.add_argument(
    "--debug_g1_link_vs_hdf5",
    action="store_true",
    default=False,
    help="G1 Locomotion V3: print pelvis-frame link positions vs interpolated HDF5 targets (same clock as training). "
    "Requires --joints_data. Compare to [DEBUG G1 LINK POS world] only after mentally transforming world→pelvis.",
)
parser.add_argument(
    "--g1_play_ckpt_key0",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 0 / NUMPAD_0 is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key1",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 1 / NUMPAD_1 is pressed (digit hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key2",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 2 / NUMPAD_2 is pressed (digit hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key3",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 3 / NUMPAD_3 is pressed (digit hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_q",
    type=str,
    default=None,
    help="G1 Locomotion play: load when Q is pressed (e.g. latest forward-progress run). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_w",
    type=str,
    default=None,
    help="G1 Locomotion play: load when W is pressed (e.g. latest forward-imitation run). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_e",
    type=str,
    default=None,
    help="G1 Locomotion play: load when E is pressed (e.g. latest imitation-symmetry run). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key4",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 4 / NUMPAD_4 is pressed. Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key5",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 5 / NUMPAD_5 is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key6",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 6 / NUMPAD_6 is pressed. Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key7",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 7 / NUMPAD_7 is pressed. Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key8",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 8 / NUMPAD_8 is pressed. Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key9",
    type=str,
    default=None,
    help="G1 Locomotion play: load when key 9 / NUMPAD_9 is pressed. Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_t",
    type=str,
    default=None,
    help="G1 Locomotion play: load when T is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_y",
    type=str,
    default=None,
    help="G1 Locomotion play: load when Y is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_u",
    type=str,
    default=None,
    help="G1 Locomotion play: load when U is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_i",
    type=str,
    default=None,
    help="G1 Locomotion play: load when I is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_o",
    type=str,
    default=None,
    help="G1 Locomotion play: load when O is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_p",
    type=str,
    default=None,
    help="G1 Locomotion play: load when P is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_z",
    type=str,
    default=None,
    help="G1 Locomotion play: load when Z is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_x",
    type=str,
    default=None,
    help="G1 Locomotion play: load when X is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_c",
    type=str,
    default=None,
    help="G1 Locomotion play: load when C is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_v",
    type=str,
    default=None,
    help="G1 Locomotion play: load when V is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_b",
    type=str,
    default=None,
    help="G1 Locomotion play: load when B is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_n",
    type=str,
    default=None,
    help="G1 Locomotion play: load when N is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_m",
    type=str,
    default=None,
    help="G1 Locomotion play: load when M is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_f",
    type=str,
    default=None,
    help="G1 Locomotion play: load when F is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_g",
    type=str,
    default=None,
    help="G1 Locomotion play: load when G is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_h",
    type=str,
    default=None,
    help="G1 Locomotion play: load when H is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_j",
    type=str,
    default=None,
    help="G1 Locomotion play: load when J is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_k",
    type=str,
    default=None,
    help="G1 Locomotion play: load when K is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_ckpt_key_l",
    type=str,
    default=None,
    help="G1 Locomotion play: load when L is pressed (hot-swap). Same path rules as --checkpoint.",
)
parser.add_argument(
    "--g1_play_walking_checkpoint",
    type=str,
    default=None,
    help="Deprecated: use --g1_play_ckpt_key8. If set and key8 is unset, key 8 loads this .pt. "
    "Key S: reload primary --checkpoint only (standing). Key R: reset_robot_to_rest + --checkpoint. "
    "Same path rules as --checkpoint.",
)
parser.add_argument(
    "--rover_play_log_dir",
    type=str,
    default=None,
    help=(
        "Rover disk-world play: directory containing timestamped training run folders "
        "(default: logs/rsl_rl/<experiment_name> for the task). Key 1 / NUMPAD_1 loads the newest "
        "model_*.pt from the latest run under this path."
    ),
)
parser.add_argument(
    "--rover_start_without_policy",
    action="store_true",
    default=False,
    help=(
        "Rover disk-world play: start with policy disabled (zero actions). "
        "Key 1 / NUMPAD_1 enables latest policy from --rover_play_log_dir; "
        "key 0 / NUMPAD_0 resets and disables policy again."
    ),
)
parser.add_argument(
    "--rover_localization_goal_count",
    type=int,
    default=None,
    help="Rover localization play: number of goal objects to spawn/use in landmark pool.",
)
parser.add_argument(
    "--rover_localization_box_count",
    type=int,
    default=None,
    help="Rover localization play: number of blue-box objects to spawn/use in landmark pool.",
)
parser.add_argument(
    "--rover_localization_debug_pose",
    type=int,
    choices=(0, 1),
    default=None,
    metavar="0|1",
    help=(
        "Rover localization: 1 = every policy step print true vs estimated rover world (x,z,y) and yaw; "
        "0 = off. Default: on for Isaac-Rover-DiskWorld-Localization tasks when this flag is omitted."
    ),
)
parser.add_argument(
    "--rover_localization_show_estimate_marker",
    action="store_true",
    default=False,
    help=(
        "Rover localization play: spawn a red visual sphere (no collision) at the estimated position "
        "and update its pose every step using the agent's estimated (x, y) from the action output."
    ),
)
parser.add_argument(
    "--rover_localization_log_pose_csv",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Rover localization play: log true rover position and estimated position to a CSV every 100 ms. "
        "If PATH is a directory, writes rover_localization_pose_log.csv inside it. "
        "If omitted, no CSV logging occurs."
    ),
)
parser.add_argument(
    "--rover_oa_cam_preview",
    action="store_true",
    default=False,
    help=(
        "Rover OA-Flat play: OpenCV window for env0 onboard RGB (shown 3× upscaled for readability). "
        "Needs a GUI OpenCV build (not opencv-python-headless) and DISPLAY + GTK on Linux. "
        "If imshow is unavailable, preview is skipped once; visual logs still run."
    ),
)
parser.add_argument(
    "--rover_oa_visual_log_interval",
    type=int,
    default=30,
    metavar="N",
    help=(
        "Rover OA-Flat play: print blue-box visual inference (tracks, counts, conf, bearing histogram) "
        "every N policy steps for env 0. Use 0 to disable. Default: 30."
    ),
)
parser.add_argument(
    "--rover_oa_policy_debug_interval",
    type=int,
    default=0,
    metavar="N",
    help=(
        "Rover OA-Flat play: every N policy steps print goal-centric kinematics, raw env0 actions, step reward, "
        "distance delta vs reward snapshot (when available), optional yellow_goal slot0 from vision buffers. "
        "0 disables."
    ),
)
parser.add_argument(
    "--rover_oa_policy_debug_obs",
    action="store_true",
    default=False,
    help=(
        "Rover OA-Flat play: when --rover_oa_policy_debug_interval > 0, also print compact stats on env0 "
        "policy observations (mean/std/L2 norm, first dims)."
    ),
)
parser.add_argument(
    "--rover_flat_goal_relocate_threshold_m",
    type=float,
    default=None,
    help="Rover OA-Flat play: planar distance (m) under which interval event moves goal (default from env cfg, typically 5).",
)
parser.add_argument(
    "--rover_flat_goal_reached_threshold_m",
    type=float,
    default=None,
    help="Rover OA-Flat play: distance (m) for goal_reached_bonus (default from env cfg, typically 1.5).",
)
parser.add_argument(
    "--rover_flat_target_half_size_m",
    type=float,
    default=None,
    help="Rover FlatGoal / OA-Flat play: half-extent (m) when sampling goal XY (default from env cfg).",
)
parser.add_argument(
    "--rover_flat_target_exclusion_half_size_m",
    type=float,
    default=None,
    help="Rover FlatGoal / OA-Flat play: goal sampled outside this central square half-extent (m).",
)
parser.add_argument(
    "--rover_localization_goal_reached_bonus_weight",
    type=float,
    default=None,
    help="Rover Localization play: weight for goal_reached_bonus reward term (default from env cfg).",
)
parser.add_argument(
    "--rover_localization_goal_reached_threshold_m",
    type=float,
    default=None,
    help="Rover Localization play: distance (m) for goal_reached_bonus (default from env cfg, typically 1.5).",
)
parser.add_argument(
    "--rover_oa_flat_box_count",
    type=int,
    default=None,
    help="Rover OA-Flat play: blue obstacle count (rebuilds scene + event params; match training).",
)
parser.add_argument(
    "--rover_oa_flat_box_min_pair_clear_m",
    type=float,
    default=None,
    help="Rover OA-Flat play: min planar center–center spacing between blue boxes (m).",
)
parser.add_argument(
    "--rover_oa_flat_obstacle_avoidance_reward_scale",
    type=float,
    default=None,
    metavar="S",
    help=(
        "Rover OA-Flat play: global scale for blue avoidance + collision reward terms (0=off, matches many "
        "staged-train snapshots; 1=full weights). Default: use env cfg (typically 1.0)."
    ),
)
parser.add_argument(
    "--rover_oa_flat_policy_include_yellow_goal_visual",
    type=int,
    choices=(0, 1),
    default=None,
    help=(
        "Rover OA-Flat play: 1=force yellow goal vision ObsTerms on policy obs; 0=strip them. "
        "Default None=infer from checkpoint (wide actor_in restores yellow terms)."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# Cameras: enable for video; otherwise respect explicit --enable_cameras (localization
# and other tasks spawn CameraCfg sensors and require the app flag).
if args_cli.video:
    args_cli.enable_cameras = True
    print("[INFO] Video recording enabled - cameras will be active")
elif getattr(args_cli, "enable_cameras", False):
    print("[INFO] Cameras explicitly enabled via CLI flag")
elif getattr(args_cli, "task", None) and ("OAFlat" in args_cli.task or "Localization" in args_cli.task):
    args_cli.enable_cameras = True
    print("[INFO] OA-Flat / Localization task uses onboard RGB camera; enabling cameras.")
else:
    args_cli.enable_cameras = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import time
import torch
import csv
from collections import deque
from collections.abc import Mapping

import carb
import omni.appwindow

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _g1_v1_apply_debug_fixed_base_if_available(env_cfg) -> None:
    """Call G1 V3 fixed-base debug helper when ``g1_locomotion_v1`` task sources are present; no-op otherwise."""
    try:
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v3_env_cfg import (
            apply_g1_locomotion_v3_debug_training_fixed_base,
        )
    except ModuleNotFoundError:
        return
    apply_g1_locomotion_v3_debug_training_fixed_base(env_cfg)


def _import_g1_locomotion_v1_policy_obs_dim():
    """Return ``compute_g1_locomotion_policy_obs_dim`` or raise if the G1 V1 task tree was removed."""
    try:
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v1_env_cfg import (
            compute_g1_locomotion_policy_obs_dim,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing isaaclab_tasks.direct.g1_locomotion_v1 (removed or not installed). "
            "Restore that package from upstream Isaac Lab to play G1 locomotion V1/V2/V3 tasks."
        ) from exc
    return compute_g1_locomotion_policy_obs_dim


def _import_g1_locomotion_v4_policy_obs_dim():
    try:
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v4_env_cfg import (
            compute_g1_locomotion_v4_policy_obs_dim,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v4_env_cfg. "
            "Restore g1_locomotion_v1 from upstream Isaac Lab to play G1 locomotion V4."
        ) from exc
    return compute_g1_locomotion_v4_policy_obs_dim


def _configure_play_cuda_linalg_preference() -> None:
    """Set ``torch.backends.cuda.preferred_linalg_library`` before env spawn when useful.

    Some Isaac Sim + PyTorch builds hit ``RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR``
    on ``cusolverDnCreate`` during USD robot spawn (``torch.linalg.solve`` inside ``XFormPrim``).
    Preferring MAGMA for CUDA linalg avoids that path when MAGMA is linked into PyTorch.

    Override with env ``ISAACLAB_TORCH_LINALG_BACKEND`` set to ``magma``, ``cusolver``, or ``default``.
    """
    if not torch.cuda.is_available():
        return
    override = os.environ.get("ISAACLAB_TORCH_LINALG_BACKEND", "").strip().lower()
    candidates = [override] if override else ["magma", "default"]
    for backend in candidates:
        if not backend:
            continue
        try:
            torch.backends.cuda.preferred_linalg_library(backend)
            print(f"[INFO] torch.backends.cuda.preferred_linalg_library({backend!r})", flush=True)
            return
        except Exception:
            continue


def _infer_rsl_rl_actor_obs_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """Return actor MLP input size from an RSL-RL ``.pt`` checkpoint (``actor.0.weight``), or None."""
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    sd = blob.get("model_state_dict", blob)
    if not isinstance(sd, dict):
        return None
    w = sd.get("actor.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) < 2:
        return None
    return int(w.shape[1])


def _rover_oa_flat_last_action_dim(env_cfg) -> int:
    """Width of the ``last_action`` observation (2 = diff drive, 12 = OA-flat physparam latch)."""
    try:
        from isaaclab_tasks.manager_based.rover.mdp.actions_physparam_latch_drive import (
            PhysparamLatchDifferentialDriveAction,
        )

        act = getattr(env_cfg, "actions", None)
        drive = getattr(act, "drive", None) if act is not None else None
        ct = getattr(drive, "class_type", None) if drive is not None else None
        if isinstance(ct, type) and issubclass(ct, PhysparamLatchDifferentialDriveAction):
            return 12
    except Exception:
        pass
    return 2


# Rover OA-Flat policy obs width: infer ``depth_scan`` bins + optional yellow-blob slots from actor MLP input.
def _rover_oa_flat_infer_depth_bins_yellow_enrich(actor_in: int, env_cfg) -> tuple[int, int, bool] | None:
    """Infer ``depth_bins``, yellow slots in {0, y_cfg}, and yellow enrichment on/off.

    ``actor_in = (23 + last_action_dim) + (enrich_extra if on) + 4·yellow_slots + depth_bins`` where the
    default diff-drive layout uses ``last_action_dim=2`` → 25 + … (legacy). OA-flat **physparam** uses
    ``last_action_dim=12`` → 35 + ….
    """
    try:
        from isaaclab_tasks.manager_based.rover.mdp.oa_flat_vision import (
            OA_FLAT_VISUAL_YELLOW_GOAL_ENRICH_DIM as _OA_YELLOW_VIS_ENRICH_DIM,
        )
    except ModuleNotFoundError:
        _OA_YELLOW_VIS_ENRICH_DIM = 28
    nb_pref = int(getattr(env_cfg, "oa_flat_depth_scan_num_bins", 32))
    y_cfg = int(getattr(env_cfg, "oa_flat_visual_yellow_goal_max_detections", 0) or 0)
    enrich_dim_cfg = getattr(env_cfg, "oa_flat_visual_yellow_goal_enrichment_extra_dim", _OA_YELLOW_VIS_ENRICH_DIM)
    enrich_dim = int(enrich_dim_cfg) if enrich_dim_cfg is not None else int(_OA_YELLOW_VIS_ENRICH_DIM)

    last_ad = _rover_oa_flat_last_action_dim(env_cfg)
    fixed_prefix = 23 + last_ad  # 25 for standard 2-D action; 35 for physparam 12-D latch

    best: tuple[int, int, bool] | None = None
    best_key: tuple[float, ...] | None = None
    # Prefer layouts without enrichment when multiple (actor_in, bins, slots) factorizations fit — avoids
    # mis-parsing legacy checkpoints whose width also satisfies an alternate depth_bin count with enrichment on.
    for enrich_on in (False, True):
        e_extra = enrich_dim if enrich_on else 0
        for y_slots in sorted({0, y_cfg}):
            bins = actor_in - fixed_prefix - e_extra - max(0, y_slots) * 4
            if 4 <= bins <= 160:
                key = (
                    1.0 if enrich_on else 0.0,
                    abs(float(bins - nb_pref)),
                    0 if y_slots == y_cfg else 1,
                    float(y_slots),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (bins, y_slots, enrich_on)
    return best


def _rover_oa_flat_apply_depth_scan_bins(env_cfg, num_bins: int) -> None:
    """Keep ``depth_scan`` ObsTerm params in sync with ``oa_flat_depth_scan_num_bins``.

    Hydra constructs env cfg before play adjusts checkpoints; ``RoverOAFlatNavEnvCfg.__post_init__`` already copied the
    default bin count into ``observations.policy.depth_scan.params``, so mutating only ``oa_flat_depth_scan_num_bins``
    leaves stale ``num_bins`` (typically 32 vs checkpoint 52).
    """
    nb = int(num_bins)
    obs = getattr(env_cfg, "observations", None)
    pol = getattr(obs, "policy", None) if obs is not None else None
    ds = getattr(pol, "depth_scan", None) if pol is not None else None
    params = getattr(ds, "params", None) if ds is not None else None
    if isinstance(params, Mapping):
        params["num_bins"] = nb


def _rover_oa_flat_checkpoint_has_aux_obstacle_head(model_state_dict: dict) -> bool:
    return any(str(k).startswith("aux_obstacle_distance_head") for k in model_state_dict.keys())


def _sync_rover_oa_flat_play_cfg_from_checkpoint(
    agent_cfg: RslRlBaseRunnerCfg,
    env_cfg,
    resume_path: str,
    task_str: str,
) -> None:
    """Match env obs layout + RSL agent cfg to an OA-flat checkpoint (legacy vs aux obstacle models)."""
    if not task_str or "OAFlat" not in task_str:
        return
    if not resume_path or not os.path.isfile(resume_path):
        return
    if not hasattr(env_cfg, "oa_flat_depth_scan_num_bins"):
        return
    try:
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[WARNING] Rover OA-Flat play: could not read checkpoint for layout sync ({exc}).")
        return
    sd = blob.get("model_state_dict")
    if not isinstance(sd, dict):
        return
    aw = sd.get("actor.0.weight")
    cw = sd.get("critic.0.weight")
    if aw is None or cw is None or not hasattr(aw, "shape") or len(aw.shape) < 2:
        return
    if not hasattr(cw, "shape") or len(cw.shape) < 2:
        return
    actor_in = int(aw.shape[1])
    critic_in = int(cw.shape[1])
    has_aux = _rover_oa_flat_checkpoint_has_aux_obstacle_head(sd)

    inferred = _rover_oa_flat_infer_depth_bins_yellow_enrich(actor_in, env_cfg)
    if inferred is not None:
        inferred_bins, y_use, enrich_on = inferred
        needs_yellow_visual = int(y_use) > 0 or bool(enrich_on)
        obs_parent = getattr(env_cfg, "observations", None)
        pol_obs = getattr(obs_parent, "policy", None) if obs_parent is not None else None

        slot_for_terms = int(y_use)
        if slot_for_terms <= 0 and bool(enrich_on):
            slot_for_terms = int(getattr(env_cfg, "oa_flat_visual_yellow_goal_max_detections", 6) or 6)

        from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import (
            ensure_rover_oa_flat_policy_yellow_visual_terms,
            strip_oa_flat_policy_yellow_visual_terms,
        )

        if needs_yellow_visual:
            if hasattr(env_cfg, "oa_flat_policy_include_yellow_goal_visual"):
                env_cfg.oa_flat_policy_include_yellow_goal_visual = True
            if pol_obs is not None:
                ensure_rover_oa_flat_policy_yellow_visual_terms(pol_obs, max_detections=slot_for_terms)
        else:
            if hasattr(env_cfg, "oa_flat_policy_include_yellow_goal_visual"):
                env_cfg.oa_flat_policy_include_yellow_goal_visual = False
            if pol_obs is not None:
                strip_oa_flat_policy_yellow_visual_terms(pol_obs)

        if hasattr(env_cfg, "oa_flat_visual_yellow_goal_enrichment_enabled"):
            env_cfg.oa_flat_visual_yellow_goal_enrichment_enabled = bool(enrich_on)

        if needs_yellow_visual:
            vy = getattr(pol_obs, "visual_yellow_goal_tracks", None) if pol_obs is not None else None
            vy_params = getattr(vy, "params", None) if vy is not None else None
            if isinstance(vy_params, dict):
                vy_params["max_detections"] = int(slot_for_terms)
            env_cfg.oa_flat_visual_yellow_goal_max_detections = int(slot_for_terms)
        else:
            env_cfg.oa_flat_visual_yellow_goal_max_detections = int(y_use)
        env_cfg.oa_flat_depth_scan_num_bins = inferred_bins
        _rover_oa_flat_apply_depth_scan_bins(env_cfg, inferred_bins)
        print(
            f"[INFO] Rover OA-Flat play: depth_bins={inferred_bins} yellow_goal_slots={int(y_use)} "
            f"yellow_goal_enrichment={bool(enrich_on)} "
            f"(checkpoint actor/critic MLP in_features {actor_in}/{critic_in})."
        )
    else:
        print(
            f"[WARNING] Rover OA-Flat play: could not reconcile actor_in={actor_in} with "
            f"(23+last_action_dim)(+optional enrichment)+4·yellow_slots+depth_bins layout; leaving cfg depth_bins / yellow / enrichment unchanged."
        )

    if has_aux:
        return

    agent_cfg.policy.class_name = "ActorCritic"
    agent_cfg.algorithm.class_name = "PPO"
    # Hydra ``RslRlPpoAlgorithmCfg`` carries aux-only kwargs; vanilla ``rsl_rl.PPO`` rejects them.
    _algo_d = getattr(agent_cfg.algorithm, "__dict__", None)
    if isinstance(_algo_d, dict):
        _algo_d.pop("aux_obstacle_distance_coef", None)
    if critic_in == actor_in:
        agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["policy"]}
        print(
            "[INFO] Rover OA-Flat play: checkpoint uses standard ActorCritic+PPO without aux obstacle head; "
            "critic observes policy obs only (symmetric layout)."
        )
    else:
        print(
            "[INFO] Rover OA-Flat play: checkpoint uses standard ActorCritic+PPO without aux head; "
            "keeping asymmetric critic obs_groups from Hydra."
        )


def _rover_oa_flat_extract_policy_obs_tensor(obs_in_for_policy) -> torch.Tensor | None:
    """Resolve the batched policy observation tensor from dict / TensorDict / plain tensor."""

    if obs_in_for_policy is None:
        return None
    if torch.is_tensor(obs_in_for_policy):
        return obs_in_for_policy
    # RSL-RL vec env uses ``tensordict.TensorDict`` — it is Mapping-like but not a ``dict``.
    if isinstance(obs_in_for_policy, Mapping):
        raw = obs_in_for_policy.get("policy")
        return raw if torch.is_tensor(raw) else None
    get_fn = getattr(obs_in_for_policy, "get", None)
    if callable(get_fn):
        raw = get_fn("policy", None)
        if torch.is_tensor(raw):
            return raw
    if hasattr(obs_in_for_policy, "__getitem__") and hasattr(obs_in_for_policy, "keys"):
        try:
            if "policy" in obs_in_for_policy:
                raw = obs_in_for_policy["policy"]
                return raw if torch.is_tensor(raw) else None
        except Exception:
            pass
    return None


def _rover_oa_flat_print_policy_obs_snapshot(obs_in_for_policy, env0: int = 0, max_prefix: int = 48) -> None:
    t = _rover_oa_flat_extract_policy_obs_tensor(obs_in_for_policy)
    if t is None:
        _keys = ""
        _keys_fn = getattr(obs_in_for_policy, "keys", None)
        if callable(_keys_fn):
            try:
                _keys = f" keys={list(_keys_fn())}"
            except Exception:
                pass
        print(
            f"[INFO] rover_oa_policy_dbg obs: could not resolve 'policy' tensor from "
            f"{type(obs_in_for_policy).__name__}{_keys}"
        )
        return
    if t.dim() >= 2:
        v = t[env0].detach().float().cpu().reshape(-1)
    else:
        v = t.detach().float().cpu().reshape(-1)
    n = int(v.numel())
    if n == 0:
        print("[INFO] rover_oa_policy_dbg obs: empty vector")
        return
    mean = float(v.mean().item())
    std = float(v.std(unbiased=False).item())
    norm = float(torch.linalg.vector_norm(v).item())
    take = min(max_prefix, n)
    head_str = ", ".join(f"{float(v[i].item()):.4f}" for i in range(take))
    more = f" … (+{n - take} dims)" if n > take else ""
    print(
        f"[INFO] rover_oa_policy_dbg obs env{env0}: dim={n} mean={mean:.5f} std={std:.5f} "
        f"L2(norm)={norm:.5f}"
    )
    print(f"  prefix[{take}]: {head_str}{more}")


def _rover_oa_flat_print_policy_dbg(
    *,
    u,
    rover,
    timestep: int,
    policy_active: bool,
    actions: torch.Tensor | None,
    rewards: torch.Tensor,
    obs_in_for_policy,
    dbg_obs: bool,
) -> None:
    """One-line OA-flat policy behavior trace (goal frame matches flat_goal_observations / rewards XY)."""

    ei = 0
    tgt = getattr(u, "_flat_goal_target_xy", None)

    gx = gy = rx = ry = float("nan")
    prev_d = getattr(u, "_flat_goal_prev_dist", None)
    snap = getattr(u, "_flat_goal_dist_snapshot", None)

    geom_d_txt = "(n/a)"
    d_buf_txt = "(n/a)"
    hcos_txt = "(n/a)"
    d_delta_txt = "(n/a)"

    if tgt is not None and hasattr(u.scene, "env_origins"):
        gx = float(tgt[ei, 0].item())
        gy = float(tgt[ei, 1].item())
        rel = rover.data.root_pos_w[ei : ei + 1, :2] - u.scene.env_origins[ei : ei + 1, :2]
        rx = float(rel[0, 0].item())
        ry = float(rel[0, 1].item())
        g_geom = float(torch.linalg.norm(tgt[ei : ei + 1, :2] - rel, dim=-1)[0].item())
        geom_d_txt = f"geom_dist_m={g_geom:.3f}"
        gvec = tgt[ei : ei + 1, :2] - rel
        gn = torch.nn.functional.normalize(gvec + 1.0e-8, dim=-1)
        q = rover.data.root_quat_w[ei : ei + 1]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        fxy = torch.stack([fx, fy], dim=-1)
        fxy = torch.nn.functional.normalize(fxy + 1.0e-8, dim=-1)
        hcos_txt = f"heading_cos={float(torch.sum(fxy * gn, dim=-1)[0].item()):.4f}"

    goal_txt = f"goal_xy=({gx:.3f},{gy:.3f}) rover_xy=({rx:.3f},{ry:.3f})"

    if isinstance(prev_d, torch.Tensor) and prev_d.numel() > ei:
        d_buf_txt = f"dist_buf_m={float(prev_d[ei].item()):.3f}"

    if torch.is_tensor(snap) and isinstance(prev_d, torch.Tensor):
        if snap.numel() > ei and prev_d.numel() > ei:
            d_delta_txt = f"dist_snap_minus_buf={float(snap[ei].item() - prev_d[ei].item()):+.5f}"

    vxy_w = rover.data.root_lin_vel_w[ei].detach().cpu()
    vx = float(vxy_w[0].item())
    vy = float(vxy_w[1].item())
    wz = float(rover.data.root_ang_vel_w[ei, 2].item())

    rew = (
        float(rewards[ei].item())
        if torch.is_tensor(rewards) and rewards.numel() > ei
        else float("nan")
    )

    pol_st = "active" if policy_active else "disabled_zero_actions"

    act_s = "n/a"
    if torch.is_tensor(actions) and actions.shape[0] > ei:
        a0 = actions[ei].detach().float().flatten().tolist()
        nmax = min(len(a0), 12)
        rest = len(a0) - nmax
        act_s = "[" + ",".join(f"{float(x):.4f}" for x in a0[:nmax]) + (f",…(+{rest})]" if rest > 0 else "]")

    y_line = ""
    if hasattr(u, "_oa_flat_visual_yellow_goal_tracks"):
        ytr_f = u._oa_flat_visual_yellow_goal_tracks[ei].detach().float().flatten()
        if int(ytr_f.numel()) >= 4:
            ys, yc, ytrr, yco = (float(ytr_f[j].item()) for j in range(4))
            y_line = (
                f" | yellow_goal slot0 sin={ys:.4f} cos={yc:.4f} tanh_r={ytrr:.4f} conf={yco:.4f}"
            )

    bonus = getattr(u, "_flat_goal_bonus_given", None)
    bonus_txt = ""
    if torch.is_tensor(bonus) and bonus.numel() > ei:
        bonus_txt = f" goal_bonus_given={bool(bonus[ei].item())}"

    print(
        f"[INFO] rover_oa_policy_dbg step={timestep} policy={pol_st} {goal_txt} | {geom_d_txt} {d_buf_txt} {d_delta_txt} "
        f"{hcos_txt} | vel_w_xy=({vx:.3f},{vy:.3f}) wz={wz:.4f} | act={act_s} | reward={rew:.5f}{bonus_txt}{y_line}"
    )

    if dbg_obs:
        _rover_oa_flat_print_policy_obs_snapshot(obs_in_for_policy, env0=ei, max_prefix=48)


def _resolve_rsl_play_checkpoint_path(checkpoint_arg: str, log_root_path: str, agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve a checkpoint path for play (same rules as main ``--checkpoint``)."""
    checkpoint_arg = checkpoint_arg.strip()
    if os.path.isabs(checkpoint_arg):
        original_path = checkpoint_arg
        resume_path = os.path.normpath(os.path.abspath(original_path))
        if resume_path.startswith("/h/home"):
            resume_path = "/home" + resume_path[7:]
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path} (from {original_path})")
        return resume_path
    if os.sep in checkpoint_arg:
        resume_path = os.path.abspath(checkpoint_arg)
        if not os.path.isfile(resume_path):
            resume_path = retrieve_file_path(checkpoint_arg)
        return resume_path
    if agent_cfg.load_run:
        return get_checkpoint_path(log_root_path, agent_cfg.load_run, checkpoint_arg)
    return retrieve_file_path(checkpoint_arg)


def _resolve_rover_latest_play_checkpoint(log_dir: str) -> str:
    """Resolve the newest ``model_*.pt`` under the latest run directory in ``log_dir``."""
    log_dir = os.path.abspath(os.path.expanduser(log_dir))
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Rover play: checkpoint log directory not found: {log_dir}")
    try:
        return get_checkpoint_path(log_dir, run_dir=".*", checkpoint=r"model_.*\.pt")
    except (ValueError, IndexError, FileNotFoundError):
        # e.g. rover_oa_flat_env stores checkpoints under ``<run>/checkpoints/``
        return get_checkpoint_path(log_dir, run_dir=".*", checkpoint=r"model_.*\.pt", other_dirs=["checkpoints"])


def _log_rover_physparam_values_once(unwrapped_env) -> None:
    """Print mean applied rover physparam values (p0..p9) for current env state."""

    def _mean_attr(name: str):
        val = getattr(unwrapped_env, name, None)
        if val is None:
            return None
        if torch.is_tensor(val):
            if val.numel() == 0:
                return None
            return float(val.detach().float().mean().item())
        if isinstance(val, (int, float)):
            return float(val)
        return None

    items = [
        ("p0 base_mass_scale (G/T)", _mean_attr("_applied_base_mass_scale")),
        ("p1 wheel_mass_scale (Y/H)", _mean_attr("_applied_wheel_mass_scale")),
        ("p2 mu_s (1/2, J/K)", _mean_attr("_applied_mu_s")),
        ("p3 mu_d (3/4, N/M)", _mean_attr("_applied_mu_d")),
        ("p4 gravity_scale", _mean_attr("_cylindrical_gravity_scale")),
        ("p5 forward_cmd_scale (Q/E)", _mean_attr("episode_forward_cmd_scale")),
        ("p6 turn_cmd_scale (F/R)", _mean_attr("episode_turn_cmd_scale")),
        ("p7 wheel_target_max_slew (V/B) [rad/s^2]", _mean_attr("episode_wheel_target_max_slew")),
        ("p8 wheel_viscous_scale (extra)", _mean_attr("_applied_wheel_viscous_scale")),
        ("p9 wheel_damping_scale (extra)", _mean_attr("_applied_wheel_damping_scale")),
    ]
    if all(v is None for _, v in items):
        print("[INFO] Rover disk-world play: physparam state unavailable on this env.")
        return
    print("[INFO] Rover disk-world play: current applied physparams (mean over envs):")
    for k, v in items:
        if v is None:
            print(f"  {k:<46s}: n/a")
        else:
            print(f"  {k:<46s}: {v:.6g}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    _pose_dbg_cli = getattr(args_cli, "rover_localization_debug_pose", None)
    if _pose_dbg_cli is None:
        rover_loc_pose_debug = "Localization" in train_task_name
    else:
        rover_loc_pose_debug = bool(int(_pose_dbg_cli))

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Default to 1 env for play (saves GPU memory and triggers PhysX buffer reduction)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # Reduce PhysX GPU buffer sizes for play with few envs (avoid CUDA OOM on limited GPUs)
    num_envs = getattr(env_cfg.scene, "num_envs", 0)
    if num_envs <= 16 and hasattr(env_cfg, "sim") and getattr(env_cfg.sim, "physx", None) is not None:
        physx = env_cfg.sim.physx
        # Single-env play: use minimal buffers to avoid 640MB+ mGpuContactPairsDev allocation
        if num_envs == 1:
            physx.gpu_max_rigid_contact_count = min(physx.gpu_max_rigid_contact_count, 2**14)
            physx.gpu_max_rigid_patch_count = min(physx.gpu_max_rigid_patch_count, 2**10)
            physx.gpu_found_lost_pairs_capacity = min(physx.gpu_found_lost_pairs_capacity, 2**14)
            physx.gpu_found_lost_aggregate_pairs_capacity = min(physx.gpu_found_lost_aggregate_pairs_capacity, 2**16)
            physx.gpu_total_aggregate_pairs_capacity = min(physx.gpu_total_aggregate_pairs_capacity, 2**14)
            physx.gpu_collision_stack_size = min(physx.gpu_collision_stack_size, 2**18)
            physx.gpu_heap_capacity = min(physx.gpu_heap_capacity, 2**20)
            physx.gpu_temp_buffer_capacity = min(physx.gpu_temp_buffer_capacity, 2**18)
        else:
            physx.gpu_max_rigid_contact_count = min(physx.gpu_max_rigid_contact_count, 2**16)
            physx.gpu_max_rigid_patch_count = min(physx.gpu_max_rigid_patch_count, 2**12)
            physx.gpu_found_lost_pairs_capacity = min(physx.gpu_found_lost_pairs_capacity, 2**16)
            physx.gpu_found_lost_aggregate_pairs_capacity = min(physx.gpu_found_lost_aggregate_pairs_capacity, 2**18)
            physx.gpu_total_aggregate_pairs_capacity = min(physx.gpu_total_aggregate_pairs_capacity, 2**16)
            physx.gpu_collision_stack_size = min(physx.gpu_collision_stack_size, 2**20)
            physx.gpu_heap_capacity = min(physx.gpu_heap_capacity, 2**22)
            physx.gpu_temp_buffer_capacity = min(physx.gpu_temp_buffer_capacity, 2**20)
        print("[INFO] Reduced PhysX GPU buffer sizes for low-env play (avoid OOM)")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    _rover_play_log_dir_arg = getattr(args_cli, "rover_play_log_dir", None)
    rover_play_ckpt_root = (
        os.path.abspath(os.path.expanduser(_rover_play_log_dir_arg))
        if _rover_play_log_dir_arg not in (None, "")
        else log_root_path
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        # Check if checkpoint is a full path (absolute or contains path separators)
        if os.path.isabs(args_cli.checkpoint):
            # Treat checkpoint as absolute path - normalize it first to fix any double slashes or /h/ issues
            original_path = args_cli.checkpoint
            resume_path = os.path.normpath(os.path.abspath(original_path))
            # Fix common path issues like /h/home -> /home
            if resume_path.startswith('/h/home'):
                resume_path = '/home' + resume_path[7:]  # Remove /h from /h/home
            if not os.path.isfile(resume_path):
                # Provide helpful error message with suggestions
                error_msg = f"Checkpoint file not found: {resume_path}\n"
                error_msg += f"Original path: {original_path}\n"
                error_msg += f"Normalized path: {resume_path}\n"
                # Check if parent directory exists
                parent_dir = os.path.dirname(resume_path)
                if os.path.exists(parent_dir):
                    error_msg += f"Parent directory exists: {parent_dir}\n"
                    # List available files
                    try:
                        files = [f for f in os.listdir(parent_dir) if f.endswith('.pt')]
                        if files:
                            error_msg += f"Available .pt files in directory: {', '.join(files[:10])}\n"
                    except Exception:
                        pass
                else:
                    error_msg += f"Parent directory does not exist: {parent_dir}\n"
                raise FileNotFoundError(error_msg)
        elif os.sep in args_cli.checkpoint:
            # Relative path with separators - try to resolve it
            resume_path = os.path.abspath(args_cli.checkpoint)
            if not os.path.isfile(resume_path):
                # Fallback to retrieve_file_path for Nucleus paths
                resume_path = retrieve_file_path(args_cli.checkpoint)
        elif agent_cfg.load_run:
            # Treat checkpoint as filename and use load_run
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
        else:
            # Treat checkpoint as full path (fallback)
            resume_path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg.load_run and agent_cfg.load_checkpoint:
        # Use load_run and load_checkpoint to construct path
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    if train_task_name in (
        "Isaac-Rover-DiskWorld-Forward-v0",
        "Isaac-Rover-DiskWorld-Forward-Play-v0",
        "Isaac-Rover-DiskWorld-Physparam-v0",
        "Isaac-Rover-DiskWorld-Physparam-Play-v0",
        "Isaac-Rover-DiskWorld-Localization-v0",
        "Isaac-Rover-DiskWorld-Localization-Play-v0",
    ):
        print(f"[INFO] Rover disk-world play: latest-checkpoint scan root = {rover_play_ckpt_root}")
    if "Isaac-Rover-DiskWorld-Localization" in train_task_name:
        if getattr(args_cli, "rover_localization_goal_count", None) is not None and hasattr(env_cfg, "goal_count"):
            env_cfg.goal_count = int(args_cli.rover_localization_goal_count)
            print(f"[INFO] Rover localization play: goal_count = {env_cfg.goal_count}")
        if getattr(args_cli, "rover_localization_box_count", None) is not None and hasattr(env_cfg, "box_count"):
            env_cfg.box_count = int(args_cli.rover_localization_box_count)
            print(f"[INFO] Rover localization play: box_count = {env_cfg.box_count}")
        if rover_loc_pose_debug:
            print(
                "[INFO] Rover localization play: per-step pose debug enabled "
                "(true vs estimated x,z,y,yaw). Disable with --rover_localization_debug_pose 0."
            )
        # Localization play: optional goal-reached bonus weight / threshold (match train.py).
        _loc_bonus_w = getattr(args_cli, "rover_localization_goal_reached_bonus_weight", None)
        _loc_thr_m = getattr(args_cli, "rover_localization_goal_reached_threshold_m", None)
        if _loc_bonus_w is not None and hasattr(env_cfg, "flat_goal_reached_bonus_weight"):
            env_cfg.flat_goal_reached_bonus_weight = float(_loc_bonus_w)
            print(f"[INFO] Rover localization play: flat_goal_reached_bonus_weight = {env_cfg.flat_goal_reached_bonus_weight}")
        if _loc_thr_m is not None and hasattr(env_cfg, "flat_goal_reached_threshold_m"):
            env_cfg.flat_goal_reached_threshold_m = float(_loc_thr_m)
            print(f"[INFO] Rover localization play: flat_goal_reached_threshold_m = {env_cfg.flat_goal_reached_threshold_m}")
        if _loc_bonus_w is not None or _loc_thr_m is not None:
            if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "goal_reached_bonus"):
                env_cfg.rewards.goal_reached_bonus.params["threshold_m"] = float(
                    getattr(env_cfg, "flat_goal_reached_threshold_m", 1.5)
                )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    if getattr(args_cli, "joints_data", None) and hasattr(env_cfg, "joints_data"):
        env_cfg.joints_data = args_cli.joints_data
        print(f"[INFO] joints_data set to: {env_cfg.joints_data}")

    if getattr(args_cli, "frames_per_second", None) is not None and hasattr(env_cfg, "hdf5_fps_override"):
        env_cfg.hdf5_fps_override = int(args_cli.frames_per_second)
        if hasattr(env_cfg, "npz_fps_override"):
            env_cfg.npz_fps_override = int(args_cli.frames_per_second)
        print(f"[INFO] G1 play: motion clip FPS (hdf5_fps_override) set to: {env_cfg.hdf5_fps_override}")
    if getattr(args_cli, "decimation", None) is not None and hasattr(env_cfg, "decimation"):
        env_cfg.decimation = int(args_cli.decimation)
        print(f"[INFO] G1 play: decimation set to: {env_cfg.decimation}")
    if getattr(args_cli, "sim_dt", None) is not None and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "dt"):
        env_cfg.sim.dt = float(args_cli.sim_dt)
        print(f"[INFO] G1 play: sim.dt set to: {env_cfg.sim.dt}")
    _dec_play = getattr(args_cli, "decimation", None)
    if _dec_play is not None and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = int(_dec_play)
        print(f"[INFO] G1 play: sim.render_interval set to {env_cfg.sim.render_interval} (match decimation)")

    if getattr(args_cli, "debug_training_fixed_base", False) and hasattr(env_cfg, "debug_training_fixed_base"):
        env_cfg.debug_training_fixed_base = True
        print("[INFO] G1 Locomotion V3: debug_training_fixed_base=True (play)")
    if getattr(args_cli, "debug_training_fixed_base_spawn_z", None) is not None and hasattr(
        env_cfg, "debug_training_fixed_base_spawn_z"
    ):
        env_cfg.debug_training_fixed_base_spawn_z = float(args_cli.debug_training_fixed_base_spawn_z)
        print(f"[INFO] G1 Locomotion V3: debug_training_fixed_base_spawn_z={env_cfg.debug_training_fixed_base_spawn_z}")

    if args_cli.viewer_env_index is not None:
        if hasattr(env_cfg, "viewer"):
            vidx = int(args_cli.viewer_env_index)
            ne = int(getattr(env_cfg.scene, "num_envs", 0))
            if vidx < 0 or vidx >= ne:
                raise ValueError(f"--viewer_env_index {vidx} invalid for num_envs={ne}")
            env_cfg.viewer.env_index = vidx
            env_cfg.viewer.origin_type = "env"
            print(f"[INFO] Viewer: origin_type=env, env_index={vidx}")
        else:
            print("[WARNING] env_cfg has no 'viewer'; ignoring --viewer_env_index")

    _g1_v1_apply_debug_fixed_base_if_available(env_cfg)

    if getattr(args_cli, "debug_g1_link_vs_hdf5", False) and hasattr(
        env_cfg, "load_hdf5_link_reference_when_joints_data"
    ):
        if args_cli.joints_data:
            env_cfg.load_hdf5_link_reference_when_joints_data = True
            print("[INFO] load_hdf5_link_reference_when_joints_data=True (for --debug_g1_link_vs_hdf5)")
        else:
            print(
                "[WARNING] --debug_g1_link_vs_hdf5 without --joints_data: HDF5 link reference will not load "
                "(set --joints_data to the same motion as training, e.g. hdf5/walk_action.hdf5)."
            )

    if getattr(args_cli, "include_root_xy_error_obs", None) is not None and hasattr(
        env_cfg, "include_root_xy_error_obs"
    ):
        env_cfg.include_root_xy_error_obs = bool(args_cli.include_root_xy_error_obs)
        print(f"[INFO] include_root_xy_error_obs (play override): {env_cfg.include_root_xy_error_obs}")
    if getattr(args_cli, "root_xy_error_obs_scale", None) is not None and hasattr(
        env_cfg, "root_xy_error_obs_scale"
    ):
        env_cfg.root_xy_error_obs_scale = float(args_cli.root_xy_error_obs_scale)
        print(f"[INFO] root_xy_error_obs_scale (play override): {env_cfg.root_xy_error_obs_scale}")

    # Hydra env_cfg.from_dict() often drops subclass-only fields; parent V1 default sets
    # include_root_xy_error_obs=False so compute_g1_locomotion_policy_obs_dim yields 103 while checkpoints
    # trained with V3 defaults use 105. Infer actor in_features from the checkpoint and fix toggles.
    _g1_walk_obs_sync_tasks = (
        "Isaac-G1-Locomotion-V1-Walking-Direct-v0",
        "Isaac-G1-Locomotion-Walking-V2-Direct-v0",
        "Isaac-G1-Locomotion-V3-Direct-v0",
    )
    _play_task_id = args_cli.task.split(":")[-1] if args_cli.task else ""
    if _play_task_id in _g1_walk_obs_sync_tasks and hasattr(env_cfg, "observation_space"):
        compute_g1_locomotion_policy_obs_dim = _import_g1_locomotion_v1_policy_obs_dim()
        _ckpt_obs = None
        if resume_path and os.path.isfile(resume_path):
            _ckpt_obs = _infer_rsl_rl_actor_obs_dim_from_checkpoint(resume_path)
        _comp = compute_g1_locomotion_policy_obs_dim(env_cfg)
        if _ckpt_obs is not None and _ckpt_obs != _comp and hasattr(env_cfg, "include_root_xy_error_obs"):
            if _ckpt_obs == _comp + 2:
                env_cfg.include_root_xy_error_obs = True
                print(
                    f"[INFO] G1 locomotion play: checkpoint actor in_features={_ckpt_obs} vs cfg toggles → {_comp}; "
                    "setting include_root_xy_error_obs=True (+2) to match training."
                )
            elif _ckpt_obs == _comp - 2:
                env_cfg.include_root_xy_error_obs = False
                print(
                    f"[INFO] G1 locomotion play: checkpoint actor in_features={_ckpt_obs} vs cfg toggles → {_comp}; "
                    "setting include_root_xy_error_obs=False (-2) to match training."
                )
            else:
                print(
                    f"[WARNING] G1 locomotion play: checkpoint in_features={_ckpt_obs} vs computed obs_dim={_comp} "
                    f"(not ±2 from root_xy). Match training flags: imitation_phase_in_obs (+2), "
                    "imitation_pelvis_link_error_in_obs (+12), imitation_root_lin_vel_in_obs (+3)."
                )
        env_cfg.observation_space = compute_g1_locomotion_policy_obs_dim(env_cfg)
        if _ckpt_obs is not None and int(env_cfg.observation_space) != _ckpt_obs:
            raise RuntimeError(
                f"G1 locomotion play: observation_space={env_cfg.observation_space} does not match checkpoint "
                f"actor in_features={_ckpt_obs}. Pass the same obs CLI flags as training (see train.py / run log header)."
            )
        print(
            f"[INFO] G1 locomotion play: observation_space={env_cfg.observation_space} "
            f"(aligned with checkpoint / obs toggles)"
        )
    elif _play_task_id == "Isaac-G1-Locomotion-V4-Direct-v0" and hasattr(env_cfg, "observation_space"):
        compute_g1_locomotion_v4_policy_obs_dim = _import_g1_locomotion_v4_policy_obs_dim()

        _ckpt_obs = None
        if resume_path and os.path.isfile(resume_path):
            _ckpt_obs = _infer_rsl_rl_actor_obs_dim_from_checkpoint(resume_path)

        def _recompute_v4_obs_space() -> int:
            return int(compute_g1_locomotion_v4_policy_obs_dim(env_cfg))

        env_cfg.observation_space = _recompute_v4_obs_space()
        if _ckpt_obs is not None and int(env_cfg.observation_space) != _ckpt_obs:
            comp = int(env_cfg.observation_space)
            diff = comp - _ckpt_obs
            # Single-dim mismatches: reward-profile phase channel and/or velocity command channel (±1 each).
            if diff == 1:
                if getattr(env_cfg, "include_reward_profile_phase_obs", False):
                    env_cfg.include_reward_profile_phase_obs = False
                    print(
                        "[INFO] G1 locomotion V4 play: checkpoint has one fewer obs dim → "
                        "include_reward_profile_phase_obs=False (e.g. standing vs phase-1 walking)."
                    )
                elif getattr(env_cfg, "velocity_command_obs_enabled", True):
                    env_cfg.velocity_command_obs_enabled = False
                    print(
                        "[INFO] G1 locomotion V4 play: checkpoint has one fewer obs dim → "
                        "velocity_command_obs_enabled=False"
                    )
            elif diff == -1:
                if not getattr(env_cfg, "include_reward_profile_phase_obs", False):
                    env_cfg.include_reward_profile_phase_obs = True
                    print(
                        "[INFO] G1 locomotion V4 play: checkpoint has one more obs dim → "
                        "include_reward_profile_phase_obs=True (e.g. phase-1 walking)."
                    )
                elif not getattr(env_cfg, "velocity_command_obs_enabled", True):
                    env_cfg.velocity_command_obs_enabled = True
                    print(
                        "[INFO] G1 locomotion V4 play: checkpoint has one more obs dim → "
                        "velocity_command_obs_enabled=True"
                    )
            env_cfg.observation_space = _recompute_v4_obs_space()

        if _ckpt_obs is not None and int(env_cfg.observation_space) != _ckpt_obs:
            comp = int(env_cfg.observation_space)
            diff = comp - _ckpt_obs
            # Two-dim mismatch: sin/cos target yaw command obs.
            if diff == 2 and getattr(env_cfg, "target_yaw_command_obs_enabled", False):
                env_cfg.target_yaw_command_obs_enabled = False
                print(
                    "[INFO] G1 locomotion V4 play: checkpoint has two fewer obs dims → "
                    "target_yaw_command_obs_enabled=False"
                )
            elif diff == -2 and not getattr(env_cfg, "target_yaw_command_obs_enabled", False):
                env_cfg.target_yaw_command_obs_enabled = True
                print(
                    "[INFO] G1 locomotion V4 play: checkpoint has two more obs dims → "
                    "target_yaw_command_obs_enabled=True"
                )
            env_cfg.observation_space = _recompute_v4_obs_space()

        if _ckpt_obs is not None and int(env_cfg.observation_space) != _ckpt_obs:
            comp = int(env_cfg.observation_space)
            diff = comp - _ckpt_obs
            # Three-dim mismatch: constant hand target in pelvis frame (teleop-style), appended last.
            if diff == 3 and getattr(env_cfg, "append_teleop_hand_target_obs", False):
                env_cfg.append_teleop_hand_target_obs = False
                print(
                    "[INFO] G1 locomotion V4 play: checkpoint has three fewer obs dims → "
                    "append_teleop_hand_target_obs=False (train without teleop hand-target obs)."
                )
                env_cfg.observation_space = _recompute_v4_obs_space()
            elif diff == -3 and not getattr(env_cfg, "append_teleop_hand_target_obs", False):
                env_cfg.append_teleop_hand_target_obs = True
                print(
                    "[INFO] G1 locomotion V4 play: checkpoint has three more obs dims → "
                    "append_teleop_hand_target_obs=True (e.g. standing + teleop hand-target obs)."
                )
                env_cfg.observation_space = _recompute_v4_obs_space()

        if _ckpt_obs is not None and int(env_cfg.observation_space) != _ckpt_obs:
            raise RuntimeError(
                f"G1 locomotion V4 play: observation_space={env_cfg.observation_space} does not match checkpoint "
                f"actor in_features={_ckpt_obs}. Match training flags: velocity_command_obs_enabled, "
                "include_reward_profile_phase_obs, target_yaw_command_obs_enabled, append_teleop_hand_target_obs "
                "(see g1_locomotion_v4_env_cfg)."
            )
        if _ckpt_obs is not None:
            print(
                f"[INFO] G1 locomotion V4 play: observation_space={env_cfg.observation_space} "
                "(aligned with checkpoint / V4 obs toggles). If behavior is off, also match "
                "env.velocity_command_obs_scale to training."
            )
        else:
            print(
                f"[INFO] G1 locomotion V4 play: observation_space={env_cfg.observation_space} "
                "(aligned with checkpoint / V4 obs toggles)"
            )

    # Optional hot-swap: digits 0–9; Q/W/E; letter keys per --g1_play_ckpt_key_* (see PLAY_G1_LOCOM_POLICY_MANAGER_60fps.sh).
    # Key 8 may fall back to deprecated --g1_play_walking_checkpoint.
    _g1_ckpt_key8_arg = args_cli.g1_play_ckpt_key8 or args_cli.g1_play_walking_checkpoint

    def _g1_validate_and_resolve_hotpath(_raw: str, _flag: str) -> str:
        _rp = _resolve_rsl_play_checkpoint_path(_raw, log_root_path, agent_cfg)
        _wobs = _infer_rsl_rl_actor_obs_dim_from_checkpoint(_rp)
        if (
            _wobs is not None
            and hasattr(env_cfg, "observation_space")
            and int(env_cfg.observation_space) != _wobs
        ):
            _p = int(env_cfg.observation_space)
            _hint = ""
            if _p == 109 and _wobs == 103:
                _hint = (
                    " Typical fix: alternate is a legacy walk policy (103 = base+torque only). "
                    "Use the same obs layout for both .pt files: set Hydra "
                    "env.velocity_command_obs_enabled=false env.include_root_xy_error_obs=false "
                    "env.append_teleop_hand_target_obs=false and a 103-dim --checkpoint, "
                    "or use PLAY_G1_LOCOMOTION_SWITCH.sh with PLAY_LEGACY_WALK_OBS_103=1. "
                    "You cannot mix a 109-dim standing checkpoint with a 103-dim walk checkpoint in one env."
                )
            raise RuntimeError(
                f"{_flag} actor in_features={_wobs} does not match "
                f"observation_space={env_cfg.observation_space} (env is sized for initial --checkpoint).{_hint}"
            )
        return _rp

    g1_hotkey_paths: dict[int, str] = {}
    for _kn, _raw, _flag in (
        (0, args_cli.g1_play_ckpt_key0, "--g1_play_ckpt_key0"),
        (1, args_cli.g1_play_ckpt_key1, "--g1_play_ckpt_key1"),
        (2, args_cli.g1_play_ckpt_key2, "--g1_play_ckpt_key2"),
        (3, args_cli.g1_play_ckpt_key3, "--g1_play_ckpt_key3"),
        (4, args_cli.g1_play_ckpt_key4, "--g1_play_ckpt_key4"),
        (5, args_cli.g1_play_ckpt_key5, "--g1_play_ckpt_key5"),
        (6, args_cli.g1_play_ckpt_key6, "--g1_play_ckpt_key6"),
        (7, args_cli.g1_play_ckpt_key7, "--g1_play_ckpt_key7"),
        (8, _g1_ckpt_key8_arg, "--g1_play_ckpt_key8"),
        (9, args_cli.g1_play_ckpt_key9, "--g1_play_ckpt_key9"),
    ):
        if not _raw:
            continue
        g1_hotkey_paths[_kn] = _g1_validate_and_resolve_hotpath(_raw, _flag)

    g1_letter_hotkey_paths: dict[str, tuple[str, str]] = {}
    for _letter, _raw, _flag, _desc in (
        ("Q", args_cli.g1_play_ckpt_key_q, "--g1_play_ckpt_key_q", "forward progress"),
        ("W", args_cli.g1_play_ckpt_key_w, "--g1_play_ckpt_key_w", "forward imitation"),
        ("E", args_cli.g1_play_ckpt_key_e, "--g1_play_ckpt_key_e", "imitation symmetry"),
        ("T", args_cli.g1_play_ckpt_key_t, "--g1_play_ckpt_key_t", "hot-swap T"),
        ("Y", args_cli.g1_play_ckpt_key_y, "--g1_play_ckpt_key_y", "hot-swap Y"),
        ("U", args_cli.g1_play_ckpt_key_u, "--g1_play_ckpt_key_u", "hot-swap U"),
        ("I", args_cli.g1_play_ckpt_key_i, "--g1_play_ckpt_key_i", "hot-swap I"),
        ("O", args_cli.g1_play_ckpt_key_o, "--g1_play_ckpt_key_o", "hot-swap O"),
        ("P", args_cli.g1_play_ckpt_key_p, "--g1_play_ckpt_key_p", "hot-swap P"),
        ("G", args_cli.g1_play_ckpt_key_g, "--g1_play_ckpt_key_g", "hot-swap G"),
        ("H", args_cli.g1_play_ckpt_key_h, "--g1_play_ckpt_key_h", "hot-swap H"),
        ("J", args_cli.g1_play_ckpt_key_j, "--g1_play_ckpt_key_j", "hot-swap J"),
        ("K", args_cli.g1_play_ckpt_key_k, "--g1_play_ckpt_key_k", "hot-swap K"),
        ("L", args_cli.g1_play_ckpt_key_l, "--g1_play_ckpt_key_l", "hot-swap L"),
        ("Z", args_cli.g1_play_ckpt_key_z, "--g1_play_ckpt_key_z", "hot-swap Z"),
        ("X", args_cli.g1_play_ckpt_key_x, "--g1_play_ckpt_key_x", "hot-swap X"),
        ("C", args_cli.g1_play_ckpt_key_c, "--g1_play_ckpt_key_c", "hot-swap C"),
        ("V", args_cli.g1_play_ckpt_key_v, "--g1_play_ckpt_key_v", "hot-swap V"),
        ("B", args_cli.g1_play_ckpt_key_b, "--g1_play_ckpt_key_b", "hot-swap B"),
        ("N", args_cli.g1_play_ckpt_key_n, "--g1_play_ckpt_key_n", "hot-swap N"),
        ("M", args_cli.g1_play_ckpt_key_m, "--g1_play_ckpt_key_m", "hot-swap M"),
        ("F", args_cli.g1_play_ckpt_key_f, "--g1_play_ckpt_key_f", "hot-swap F"),
    ):
        if not _raw:
            continue
        g1_letter_hotkey_paths[_letter] = (_g1_validate_and_resolve_hotpath(_raw, _flag), _desc)

    if g1_hotkey_paths:
        print("[INFO] G1 Locomotion play digit hot-swap (0–9):")
        for _kn in sorted(g1_hotkey_paths):
            print(f"  key {_kn}: {g1_hotkey_paths[_kn]}")
    if g1_letter_hotkey_paths:
        print("[INFO] G1 Locomotion play letter hot-swap:")
        for _letter in sorted(g1_letter_hotkey_paths):
            print(f"  {_letter}: {g1_letter_hotkey_paths[_letter][0]}")

    # OA-Flat play: optional goal thresholds (match train / PLAY_* env vars; cfg __post_init__ may already set these).
    if getattr(args_cli, "task", None) and ("OAFlat" in args_cli.task or "Localization" in args_cli.task):
        _rloc = getattr(args_cli, "rover_flat_goal_relocate_threshold_m", None)
        _gthr = getattr(args_cli, "rover_flat_goal_reached_threshold_m", None)
        if _rloc is not None and hasattr(env_cfg, "flat_goal_relocate_threshold_m"):
            env_cfg.flat_goal_relocate_threshold_m = float(_rloc)
            print(f"[INFO] Rover play: flat_goal_relocate_threshold_m = {env_cfg.flat_goal_relocate_threshold_m}")
        if _gthr is not None and hasattr(env_cfg, "flat_goal_reached_threshold_m"):
            env_cfg.flat_goal_reached_threshold_m = float(_gthr)
            print(f"[INFO] Rover play: flat_goal_reached_threshold_m = {env_cfg.flat_goal_reached_threshold_m}")
        if _rloc is not None or _gthr is not None:
            if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "relocate_goal_interval"):
                env_cfg.events.relocate_goal_interval.params["goal_threshold_m"] = float(
                    getattr(env_cfg, "flat_goal_relocate_threshold_m", 5.0)
                )
            if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "goal_reached_bonus"):
                env_cfg.rewards.goal_reached_bonus.params["threshold_m"] = float(
                    getattr(env_cfg, "flat_goal_reached_threshold_m", 1.5)
                )

    # Flat-goal / OA-Flat play: optional target sampling region (match train.py / env cfg fields).
    _tname = getattr(args_cli, "task", None) or ""
    if "OAFlat" in _tname or "FlatWorld-GoalNav" in _tname:
        _th_cli = getattr(args_cli, "rover_flat_target_half_size_m", None)
        _tex_cli = getattr(args_cli, "rover_flat_target_exclusion_half_size_m", None)
        if _th_cli is not None and hasattr(env_cfg, "target_half_size_m"):
            env_cfg.target_half_size_m = float(_th_cli)
            print(f"[INFO] Rover flat-goal play: target_half_size_m = {env_cfg.target_half_size_m}")
        if _tex_cli is not None and hasattr(env_cfg, "target_exclusion_half_size_m"):
            env_cfg.target_exclusion_half_size_m = float(_tex_cli)
            print(f"[INFO] Rover flat-goal play: target_exclusion_half_size_m = {env_cfg.target_exclusion_half_size_m}")
        if _th_cli is not None or _tex_cli is not None:
            _th = float(getattr(env_cfg, "target_half_size_m", 23.0))
            _tex = float(getattr(env_cfg, "target_exclusion_half_size_m", 15.5))
            _ev = getattr(env_cfg, "events", None)
            if _ev is not None:
                if hasattr(_ev, "relocate_goal_interval"):
                    _ev.relocate_goal_interval.params["target_half_size_m"] = _th
                    _ev.relocate_goal_interval.params["target_exclusion_half_size_m"] = _tex
                if hasattr(_ev, "reset_oa_flat"):
                    _ev.reset_oa_flat.params["target_half_size_m"] = _th
                    _ev.reset_oa_flat.params["target_exclusion_half_size_m"] = _tex
                if hasattr(_ev, "reset_flat_goal"):
                    _ev.reset_flat_goal.params["target_half_size_m"] = _th
                    _ev.reset_flat_goal.params["target_exclusion_half_size_m"] = _tex

    if getattr(args_cli, "task", None) and "OAFlat" in args_cli.task:
        _bc = getattr(args_cli, "rover_oa_flat_box_count", None)
        _mpc = getattr(args_cli, "rover_oa_flat_box_min_pair_clear_m", None)
        if _bc is not None and hasattr(env_cfg, "oa_flat_box_count"):
            env_cfg.oa_flat_box_count = int(_bc)
            print(f"[INFO] OA-Flat play: oa_flat_box_count = {env_cfg.oa_flat_box_count}")
        if _mpc is not None and hasattr(env_cfg, "oa_flat_box_min_pair_clear_m"):
            env_cfg.oa_flat_box_min_pair_clear_m = float(_mpc)
            print(f"[INFO] OA-Flat play: oa_flat_box_min_pair_clear_m = {env_cfg.oa_flat_box_min_pair_clear_m}")
        if _bc is not None or _mpc is not None:
            from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import (
                refresh_oa_flat_blue_box_assets_from_env_fields,
            )

            refresh_oa_flat_blue_box_assets_from_env_fields(env_cfg)

    _sync_rover_oa_flat_play_cfg_from_checkpoint(agent_cfg, env_cfg, resume_path, getattr(args_cli, "task", "") or "")

    _cli_policy_yellow = getattr(args_cli, "rover_oa_flat_policy_include_yellow_goal_visual", None)
    if _cli_policy_yellow is not None and getattr(args_cli, "task", None) and "OAFlat" in args_cli.task:
        from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import sync_oa_flat_policy_yellow_visual_obs_terms

        if hasattr(env_cfg, "oa_flat_policy_include_yellow_goal_visual"):
            env_cfg.oa_flat_policy_include_yellow_goal_visual = bool(int(_cli_policy_yellow))
        sync_oa_flat_policy_yellow_visual_obs_terms(env_cfg)
        print(
            "[INFO] OA-Flat play: CLI policy yellow goal visual = "
            f"{bool(int(_cli_policy_yellow))} (ObsTerms synced from env_cfg)."
        )

    _oa_obs_scale_cli = getattr(args_cli, "rover_oa_flat_obstacle_avoidance_reward_scale", None)
    if getattr(args_cli, "task", None) and "OAFlat" in args_cli.task and _oa_obs_scale_cli is not None:
        if hasattr(env_cfg, "oa_flat_obstacle_avoidance_reward_scale"):
            env_cfg.oa_flat_obstacle_avoidance_reward_scale = float(_oa_obs_scale_cli)
            from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import sync_oa_flat_play_obstacle_reward_weights

            sync_oa_flat_play_obstacle_reward_weights(env_cfg)
            print(f"[INFO] OA-Flat play: oa_flat_obstacle_avoidance_reward_scale = {env_cfg.oa_flat_obstacle_avoidance_reward_scale}")

    # create isaac environment
    _configure_play_cuda_linalg_preference()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # OA-flat / aux obstacle policies use custom actor-critic + PPO classes that are not in rsl_rl's eval() scope.
    try:
        from isaaclab_rl.rsl_rl.rsl_rl_construct_patch import apply_rsl_rl_aux_obstacle_construct_patch

        apply_rsl_rl_aux_obstacle_construct_patch()
    except Exception:
        pass

    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # Rover play option: keep simulation running with zero actions until user loads policy (key 1).
    rover_policy_active = [True]

    def _zero_policy(obs_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (obs_tensor.shape[0], env.action_space.shape[-1]),
            device=obs_tensor.device,
            dtype=obs_tensor.dtype,
        )

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Get unwrapped environment for joint logging
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env

    # -------------------------------------------------------------------------
    # Rover localization: CSV pose logger (true vs estimated, every 100 ms)
    # -------------------------------------------------------------------------
    _rover_loc_log_csv_path: str | None = None
    _rover_loc_log_csv_file = None
    _rover_loc_log_csv_writer = None
    _rover_loc_log_last_time = -1.0e9
    _rover_loc_console_last_time = -1.0e9
    if "Localization" in train_task_name:
        _csv_arg = getattr(args_cli, "rover_localization_log_pose_csv", None)
        if _csv_arg is not None:
            if os.path.isdir(_csv_arg):
                _rover_loc_log_csv_path = os.path.join(_csv_arg, "rover_localization_pose_log.csv")
            else:
                _rover_loc_log_csv_path = _csv_arg
            os.makedirs(os.path.dirname(_rover_loc_log_csv_path), exist_ok=True)
            _rover_loc_log_csv_file = open(_rover_loc_log_csv_path, "w", newline="")
            _rover_loc_log_csv_writer = csv.writer(_rover_loc_log_csv_file)
            _rover_loc_log_csv_writer.writerow([
                "time_s", "step",
                "true_x", "true_y", "true_z",
                "est_x", "est_y", "est_z",
                "goal_x", "goal_y", "goal_z",
                "error_m",
            ])
            print(f"[INFO] Rover localization pose CSV logging enabled: {_rover_loc_log_csv_path}")

    # Key S: reload primary --checkpoint only (no env.reset); keys 0–9: hot-swap if path set; key R: spawn + reload
    # Rover disk-world: key 0 = full env reset; key 1 = load latest model from rover_play_ckpt_root
    # Subscribe to all keyboards (None) so keys work when viewport has focus
    _rover_play_mode = train_task_name in (
        "Isaac-Rover-DiskWorld-Forward-v0",
        "Isaac-Rover-DiskWorld-Forward-Play-v0",
        "Isaac-Rover-DiskWorld-Physparam-v0",
        "Isaac-Rover-DiskWorld-Physparam-Play-v0",
        "Isaac-Rover-DiskWorld-Localization-v0",
        "Isaac-Rover-DiskWorld-Localization-Play-v0",
        "Isaac-Rover-DiskWorld-ObstacleAvoidance-v0",
        "Isaac-Rover-DiskWorld-ObstacleAvoidance-Play-v0",
        "Isaac-Rover-OAFlat-Physparam-v0",
        "Isaac-Rover-OAFlat-Physparam-Play-v0",
        "Isaac-Rover-OAFlat-Physparam-Continuous-Play-v0",
    )
    # Rover localization: optional red estimate marker (visual sphere, no collision)
    _rover_loc_estimate_marker_prim = None
    _rover_loc_estimate_marker_stage = None
    if (
        "Isaac-Rover-DiskWorld-Localization" in train_task_name
        and getattr(args_cli, "rover_localization_show_estimate_marker", False)
    ):
        try:
            from pxr import UsdGeom, Gf, Sdf
            _rover_loc_estimate_marker_stage = omni.usd.get_context().get_stage()
            if _rover_loc_estimate_marker_stage is not None:
                marker_path = Sdf.Path("/World/EstimateMarker")
                if not _rover_loc_estimate_marker_stage.GetPrimAtPath(marker_path):
                    sphere = UsdGeom.Sphere.Define(_rover_loc_estimate_marker_stage, marker_path)
                    sphere.CreateRadiusAttr(0.3)
                    # Red emissive material
                    mat_path = Sdf.Path("/World/EstimateMarker/RedMaterial")
                    if not _rover_loc_estimate_marker_stage.GetPrimAtPath(mat_path):
                        from pxr import UsdShade, UsdGeom
                        mat = UsdShade.Material.Define(_rover_loc_estimate_marker_stage, mat_path)
                        pbr = UsdShade.Shader.Define(_rover_loc_estimate_marker_stage, mat_path.AppendChild("PBRShader"))
                        pbr.CreateIdAttr("UsdPreviewSurface")
                        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
                        pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.0, 0.0))
                        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
                        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                        mat.CreateSurfaceOutput().ConnectToSource(pbr, "surface")
                        sphere.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
                        UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)
                _rover_loc_estimate_marker_prim = _rover_loc_estimate_marker_stage.GetPrimAtPath(marker_path)
                print("[INFO] Rover localization: red estimate marker sphere created at /World/EstimateMarker")
        except Exception as e:
            print(f"[WARNING] Could not create estimate marker sphere: {e}")
            _rover_loc_estimate_marker_prim = None
            _rover_loc_estimate_marker_stage = None
    rover_physparam_log_pending = [False]
    # Track episode-start logging for OA-Flat-Physparam: set after any done, cleared after log
    rover_physparam_log_episode_start = [False]
    if _rover_play_mode and bool(getattr(args_cli, "rover_start_without_policy", False)):
        rover_policy_active[0] = False
        policy = _zero_policy
    rover_play_reset_requested = [False]
    rover_play_load_latest_requested = [False]
    rover_play_randomize_requested = [False]
    rover_oa_play_rover_respawn = [False]
    rover_oa_play_layout_reset = [False]
    reset_requested = [False]  # mutable so callback can set it
    g1_pending_hotkey_load = [None]  # (resolved_path: str, label: str) or None
    robot_rest_requested = [False]
    _keyboard_sub = None
    _keyboard_for_unsub = None
    _g1_digit_to_hotkey_num = {str(i): i for i in range(10)}
    for _i in range(10):
        _g1_digit_to_hotkey_num[f"NUMPAD_{_i}"] = _i
    try:
        _input = carb.input.acquire_input_interface()

        def _on_key(event, *args):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True
            key_name = getattr(event.input, "name", None)
            if "OAFlat" in train_task_name:
                if key_name == "R":
                    rover_oa_play_layout_reset[0] = True
                    return True
                if key_name in ("0", "NUMPAD_0"):
                    rover_oa_play_rover_respawn[0] = True
                    return True
            if key_name == "S":
                reset_requested[0] = True
            if key_name == "R":
                if _rover_play_mode:
                    rover_play_randomize_requested[0] = True
                else:
                    robot_rest_requested[0] = True
            if key_name in g1_letter_hotkey_paths:
                _qp, _qd = g1_letter_hotkey_paths[key_name]
                g1_pending_hotkey_load[0] = (_qp, f"key {key_name} ({_qd})")
            _hk = _g1_digit_to_hotkey_num.get(key_name)
            if _hk is not None:
                if _rover_play_mode and _hk == 0:
                    rover_play_reset_requested[0] = True
                elif _rover_play_mode and _hk == 1:
                    rover_play_load_latest_requested[0] = True
                elif _hk in g1_hotkey_paths:
                    g1_pending_hotkey_load[0] = (g1_hotkey_paths[_hk], f"key {_hk}")
            return True

        try:
            _keyboard_sub = _input.subscribe_to_keyboard_events(None, _on_key)
            _keyboard_for_unsub = None  # subscribed to all keyboards
        except Exception:
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            _keyboard_sub = _input.subscribe_to_keyboard_events(keyboard, _on_key)
            _keyboard_for_unsub = keyboard
        if _rover_play_mode:
            print("[INFO] Rover disk-world play: key 0 or NUMPAD_0 = reset environment to initial spawn.")
            print(
                f"[INFO] Rover disk-world play: key 1 or NUMPAD_1 = load latest policy from: {rover_play_ckpt_root}"
            )
            print("[INFO] Rover disk-world play: key R = reset + randomize obstacles with a new seed.")
            if bool(getattr(args_cli, "rover_start_without_policy", False)):
                print(
                    "[INFO] Rover disk-world play: starting with policy DISABLED (zero actions). "
                    "Press key 1 to enable latest policy."
                )
            print(
                "[INFO] (Also) Press S to reload primary --checkpoint without env.reset; "
                "R to reset pose + reload primary checkpoint (if env supports it)."
            )
        elif "OAFlat" in train_task_name:
            print(
                "[INFO] Rover OA-Flat play: key 0 or NUMPAD_0 = random rover spawn only "
                "(goal marker and blue boxes unchanged)."
            )
            print(
                "[INFO] Rover OA-Flat play: key R = full environment reset with a new random blue-box layout."
            )
            print(
                "[INFO] (Also) Press S to reload primary --checkpoint without env.reset "
                "(robot pose unchanged; viewport focused)."
            )
        else:
            print(
                "[INFO] Press S to reload primary --checkpoint (standing policy) without env.reset "
                "(robot pose unchanged; viewport focused)."
            )
            print("[INFO] Press R to reset robot pose to spawn and reload --checkpoint.")
        for _letter in (
            "Q",
            "W",
            "E",
            "T",
            "Y",
            "U",
            "I",
            "O",
            "P",
            "G",
            "H",
            "J",
            "K",
            "L",
            "Z",
            "X",
            "C",
            "V",
            "B",
            "N",
            "M",
            "F",
        ):
            if _letter in g1_letter_hotkey_paths:
                _qd = g1_letter_hotkey_paths[_letter][1]
                print(f"[INFO] Press {_letter} to load --g1_play_ckpt_key_{_letter.lower()} ({_qd}).")
        if not _rover_play_mode:
            for _kn in sorted(g1_hotkey_paths):
                print(f"[INFO] Press key {_kn} (or numpad {_kn}) to load --g1_play_ckpt_key{_kn}.")
    except Exception as e:
        print(f"[WARNING] Could not bind G1 play hotkeys: {e}")

    # Setup joint logging if requested
    log_joints = args_cli.log_joints
    joint_log_file = None
    joint_log_writer = None
    joint_log_headers = None
    joint_log_file_handle = None
    has_keyframe_data = False
    
    if log_joints:
        # Determine log file path
        if args_cli.log_file:
            joint_log_file = os.path.abspath(args_cli.log_file)
        else:
            joint_log_file = os.path.join(log_dir, "joint_log.csv")
        
        # Create log directory if needed
        os.makedirs(os.path.dirname(joint_log_file), exist_ok=True)
        
        # Check if environment has keyframe joint data
        has_keyframe_data = (
            hasattr(unwrapped_env, 'bvh_reference_keyframe') and
            hasattr(unwrapped_env, 'current_frame_indices') and
            hasattr(unwrapped_env, 'cfg') and
            hasattr(unwrapped_env.cfg, 'keyframe_joint_names')
        )
        
        if has_keyframe_data:
            print(f"[INFO] Joint logging enabled. Logging to: {joint_log_file}")
            print(f"[INFO] Logging interval: every {args_cli.log_interval} steps")
            joint_log_file_handle = open(joint_log_file, 'w', newline='')
            joint_log_writer = csv.writer(joint_log_file_handle)
        else:
            print("[WARNING] Environment does not support keyframe joint logging. Disabling joint logging.")
            log_joints = False
            has_keyframe_data = False

    # Setup ankle contact debug (G1 locomotion: external torque from ankle joints)
    debug_ankle_contact = args_cli.debug_ankle_contact
    debug_ankle_interval_s = args_cli.debug_ankle_interval_ms / 1000.0
    last_ankle_debug_time = 0.0
    last_joint_debug_time = -1.0e9
    debug_joint_interval_s = (
        max(int(args_cli.debug_joint_interval_ms), 1) / 1000.0
        if args_cli.debug_joint_interval_ms is not None
        else None
    )
    ankle_joint_names_out = ["right_ankle_pitch_joint", "right_ankle_roll_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"]
    # Phase of feet alternation: prev contact for event detection, last foot that had contact event, gait_phase [0,1)
    debug_prev_left_contact = False
    debug_prev_right_contact = False
    debug_last_contact_foot = 0  # 0=none, 1=left, 2=right
    debug_gait_phase = 0.0
    ankle_joint_indices_debug = None
    if debug_ankle_contact:
        robot_debug = getattr(unwrapped_env, "_robot", None) or getattr(unwrapped_env, "robot", None)
        if robot_debug is not None:
            try:
                joint_ids, _ = robot_debug.find_joints(ankle_joint_names_out, preserve_order=True)
                if len(joint_ids) >= 4:
                    ankle_joint_indices_debug = joint_ids[:4]
                    print(f"[INFO] Ankle contact debug enabled. Output every {args_cli.debug_ankle_interval_ms} ms.")
                else:
                    print("[WARNING] Could not find all 4 ankle joints. Disabling ankle contact debug.")
                    debug_ankle_contact = False
            except Exception as e:
                print(f"[WARNING] Ankle contact debug setup failed: {e}. Disabling.")
                debug_ankle_contact = False
        else:
            print("[WARNING] No robot found for ankle contact debug. Disabling.")
            debug_ankle_contact = False

    # G1 link origins in world frame (palms + ankle roll links); optional pelvis-frame vs HDF5 (training clock)
    _G1_HDF5_LINK_SLOT_LABELS = (
        "right_ankle_roll_link",
        "left_ankle_roll_link",
        "right_palm_link",
        "left_palm_link",
    )
    debug_g1_link_positions = args_cli.debug_g1_link_positions
    debug_g1_link_vs_hdf5 = bool(getattr(args_cli, "debug_g1_link_vs_hdf5", False))
    g1_link_body_indices: list[int] | None = None
    g1_link_body_names_resolved: list[str] = []
    _g1_link_ms = args_cli.debug_g1_link_interval_ms
    if _g1_link_ms is None:
        _g1_link_ms = args_cli.debug_joint_interval_ms
    if _g1_link_ms is None:
        _g1_link_ms = 100
    debug_g1_link_interval_s = max(int(_g1_link_ms), 1) / 1000.0
    last_g1_link_debug_time = -1.0e9
    _g1_link_names = ("right_palm_link", "left_palm_link", "right_ankle_roll_link", "left_ankle_roll_link")
    if debug_g1_link_vs_hdf5:
        seq = getattr(unwrapped_env, "_hdf5_link_ref_sequence", None)
        if seq is None:
            print(
                "[WARNING] --debug_g1_link_vs_hdf5: no HDF5 link reference on env "
                "(enable link imitation weights in cfg, or use --joints_data with load flag). Disabling."
            )
            debug_g1_link_vs_hdf5 = False
        else:
            print(
                f"[INFO] G1 link vs HDF5 (pelvis frame): every {_g1_link_ms} ms sim | "
                f"slots={', '.join(_G1_HDF5_LINK_SLOT_LABELS)}"
            )
    if debug_g1_link_positions:
        robot_links = getattr(unwrapped_env, "_robot", None) or getattr(unwrapped_env, "robot", None)
        if robot_links is not None:
            try:
                bids, bn = robot_links.find_bodies(list(_g1_link_names), preserve_order=True)
                g1_link_body_indices = list(bids)
                g1_link_body_names_resolved = list(bn)
                if len(g1_link_body_indices) != len(_g1_link_names):
                    print(
                        f"[WARNING] debug_g1_link_positions: matched {len(g1_link_body_indices)}/{len(_g1_link_names)} "
                        f"bodies (expected {_g1_link_names})."
                    )
                print(
                    f"[INFO] G1 link position debug: every {_g1_link_ms} ms sim | "
                    f"{', '.join(g1_link_body_names_resolved)}"
                )
            except Exception as e:
                print(f"[WARNING] debug_g1_link_positions setup failed: {e}. Disabling.")
                debug_g1_link_positions = False
                g1_link_body_indices = None
        else:
            print("[WARNING] No robot for debug_g1_link_positions. Disabling.")
            debug_g1_link_positions = False

    g1_link_debug_tick = debug_g1_link_positions or debug_g1_link_vs_hdf5

    _rover_oa_log_period = int(getattr(args_cli, "rover_oa_visual_log_interval", 0) or 0)
    _rover_oa_policy_dbg_period = int(getattr(args_cli, "rover_oa_policy_debug_interval", 0) or 0)
    _rover_oa_policy_dbg_obs = bool(getattr(args_cli, "rover_oa_policy_debug_obs", False))
    rover_oa_flat_play_features = "OAFlat" in train_task_name and (
        bool(getattr(args_cli, "rover_oa_cam_preview", False))
        or _rover_oa_log_period > 0
        or _rover_oa_policy_dbg_period > 0
    )
    rover_oa_cv2_holder: list = [None]  # None = lazy import not tried; False = import failed; else cv2 module
    rover_oa_cv2_gui_disabled: list = [False]  # True after first cv2.imshow failure (headless / no GTK)
    if rover_oa_flat_play_features:
        print(
            "[INFO] Rover OA-Flat play extras: "
            f"cam_preview={bool(getattr(args_cli, 'rover_oa_cam_preview', False))} "
            f"visual_log_every={_rover_oa_log_period} steps (0=off) "
            f"policy_dbg_every={_rover_oa_policy_dbg_period} (0=off) "
            f"policy_dbg_obs={_rover_oa_policy_dbg_obs}"
        )

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        # Key S — reload primary --checkpoint (standing) only; keep current sim pose (no env.reset / no teleport).
        if reset_requested[0]:
            print(f"[INFO] Reloading primary checkpoint (key S): {resume_path}")
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic
            policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
            obs = env.get_observations()
            rover_policy_active[0] = True
            reset_requested[0] = False
            timestep = 0
            last_ankle_debug_time = 0.0
            last_joint_debug_time = -1.0e9
            last_g1_link_debug_time = -1.0e9
            debug_prev_left_contact = False
            debug_prev_right_contact = False
            debug_last_contact_foot = 0
            debug_gait_phase = 0.0
            print("[INFO] Primary (standing) policy active (key S); robot pose unchanged.")

        # Rover OA-Flat: key 0 = respawn rover in spawn square only; key R = full reset + new box layout (handled before generic R).
        if "OAFlat" in train_task_name and rover_oa_play_rover_respawn[0]:
            rover_oa_play_rover_respawn[0] = False
            from isaaclab_tasks.manager_based.rover.mdp.flat_goal_events import randomize_rover_spawn_pose_only

            e0 = torch.tensor([0], device=unwrapped_env.device, dtype=torch.long)
            sh = float(getattr(unwrapped_env.cfg, "spawn_half_size_m", 15.0))
            sz = float(getattr(unwrapped_env.cfg, "spawn_z_m", 0.2))
            randomize_rover_spawn_pose_only(unwrapped_env, e0, spawn_half_size_m=sh, spawn_z_m=sz)
            obs = env.get_observations()
            print("[INFO] OA-Flat play: new random rover spawn (key 0); goal and blue boxes unchanged.")

        if "OAFlat" in train_task_name and rover_oa_play_layout_reset[0]:
            rover_oa_play_layout_reset[0] = False
            unwrapped_env._oa_flat_play_layout_seed = random.randint(0, 2**31 - 1)
            obs, _ = env.reset()
            try:
                policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
            except Exception:
                pass
            timestep = 0
            last_ankle_debug_time = 0.0
            last_joint_debug_time = -1.0e9
            last_g1_link_debug_time = -1.0e9
            debug_prev_left_contact = False
            debug_prev_right_contact = False
            debug_last_contact_foot = 0
            debug_gait_phase = 0.0
            print("[INFO] OA-Flat play: full reset with new random layout (key R).")

        # Key R: spawn pose + reload primary --checkpoint (same as launch; use after hot-swap if you want spawn pose)
        if robot_rest_requested[0]:
            if hasattr(unwrapped_env, "reset_robot_to_rest"):
                print(f"[INFO] Reloading primary checkpoint (key R): {resume_path}")
                runner.load(resume_path)
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                try:
                    policy_nn = runner.alg.policy
                except AttributeError:
                    policy_nn = runner.alg.actor_critic
                unwrapped_env.reset_robot_to_rest()
                obs = env.get_observations()
                policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
                rover_policy_active[0] = True
                timestep = 0
                last_ankle_debug_time = 0.0
                last_joint_debug_time = -1.0e9
                last_g1_link_debug_time = -1.0e9
                debug_prev_left_contact = False
                debug_prev_right_contact = False
                debug_last_contact_foot = 0
                debug_gait_phase = 0.0
                print("[INFO] Robot pose reset to spawn; primary policy active (key R).")
            else:
                print("[WARNING] Key R: environment has no reset_robot_to_rest(); ignoring.")
            robot_rest_requested[0] = False

        # Rover disk-world: key 0 = full gym reset (initial episode); key 1 = reload latest trained policy.
        if _rover_play_mode and rover_play_reset_requested[0]:
            rover_play_reset_requested[0] = False
            obs, _ = env.reset()
            if bool(getattr(args_cli, "rover_start_without_policy", False)):
                policy = _zero_policy
                rover_policy_active[0] = False
            try:
                policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
            except Exception:
                pass
            timestep = 0
            last_ankle_debug_time = 0.0
            last_joint_debug_time = -1.0e9
            last_g1_link_debug_time = -1.0e9
            debug_prev_left_contact = False
            debug_prev_right_contact = False
            debug_last_contact_foot = 0
            debug_gait_phase = 0.0
            if rover_policy_active[0]:
                print("[INFO] Rover disk-world play: environment reset to initial (key 0 / NUMPAD 0).")
            else:
                print(
                    "[INFO] Rover disk-world play: environment reset to initial and policy disabled "
                    "(key 0 / NUMPAD 0)."
                )

        if _rover_play_mode and rover_play_load_latest_requested[0]:
            rover_play_load_latest_requested[0] = False
            try:
                _latest_ckpt = _resolve_rover_latest_play_checkpoint(rover_play_ckpt_root)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[WARNING] Rover disk-world play: could not resolve latest checkpoint: {exc}")
            else:
                print(f"[INFO] Rover disk-world play: loading latest checkpoint (key 1): {_latest_ckpt}")
                runner.load(_latest_ckpt)
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                try:
                    policy_nn = runner.alg.policy
                except AttributeError:
                    policy_nn = runner.alg.actor_critic
                policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
                obs = env.get_observations()
                rover_policy_active[0] = True
                rover_physparam_log_pending[0] = True
                print("[INFO] Rover disk-world play: latest policy active (key 1 / NUMPAD 1).")

        if _rover_play_mode and rover_play_randomize_requested[0]:
            rover_play_randomize_requested[0] = False
            used_seed = None
            if hasattr(unwrapped_env, "reseed_and_respawn_visual_obstacles"):
                try:
                    used_seed = int(unwrapped_env.reseed_and_respawn_visual_obstacles())
                except Exception as exc:
                    print(f"[WARNING] Rover disk-world play: randomize obstacles failed: {exc}")
            else:
                print(
                    "[WARNING] Rover disk-world play: env has no reseed_and_respawn_visual_obstacles(); "
                    "performing normal reset only."
                )
            obs, _ = env.reset()
            try:
                policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
            except Exception:
                pass
            timestep = 0
            last_ankle_debug_time = 0.0
            last_joint_debug_time = -1.0e9
            last_g1_link_debug_time = -1.0e9
            debug_prev_left_contact = False
            debug_prev_right_contact = False
            debug_last_contact_foot = 0
            debug_gait_phase = 0.0
            if used_seed is None:
                print("[INFO] Rover disk-world play: environment reset (key R).")
            else:
                print(
                    f"[INFO] Rover disk-world play: environment reset and obstacles randomized "
                    f"(key R, seed={used_seed})."
                )

        # Keys 0–9: hot-swap weights; refresh obs for current pose (no env.reset).
        if g1_pending_hotkey_load[0] is not None:
            _swap_path, _swap_label = g1_pending_hotkey_load[0]
            g1_pending_hotkey_load[0] = None
            print(f"[INFO] Loading checkpoint ({_swap_label}): {_swap_path}")
            runner.load(_swap_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic
            policy_nn.reset(torch.ones(unwrapped_env.num_envs, dtype=torch.bool, device=unwrapped_env.device))
            obs = env.get_observations()
            print(f"[INFO] Hot-swap active ({_swap_label}).")

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            obs_in_for_policy = obs
            actions = policy(obs_in_for_policy)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            if _rover_play_mode and rover_physparam_log_pending[0]:
                rover_physparam_log_pending[0] = False
                _log_rover_physparam_values_once(unwrapped_env)

            # Log physparam values on each new episode for OA-Flat-Physparam task
            # (delayed by 1 step: new latch values are applied in process_actions the step after reset)
            if _rover_play_mode and ("Physparam" in train_task_name):
                if dones.any():
                    rover_physparam_log_episode_start[0] = True
                elif rover_physparam_log_episode_start[0]:
                    rover_physparam_log_episode_start[0] = False
                    _log_rover_physparam_values_once(unwrapped_env)

            if rover_oa_flat_play_features:
                _oa_update_err: Exception | None = None
                try:
                    from isaaclab_tasks.manager_based.rover.mdp.oa_flat_vision import (
                        update_oa_flat_blue_detections_from_camera,
                    )

                    _cam_sn = str(getattr(unwrapped_env.cfg, "oa_flat_camera_sensor_name", "rover_oa_rgb_cam"))
                    update_oa_flat_blue_detections_from_camera(unwrapped_env, sensor_name=_cam_sn)
                except Exception as e:
                    _oa_update_err = e

                if _oa_update_err is None:
                    if (
                        bool(getattr(args_cli, "rover_oa_cam_preview", False))
                        and not rover_oa_cv2_gui_disabled[0]
                    ):
                        _cv = rover_oa_cv2_holder
                        if _cv[0] is not False:
                            if _cv[0] is None:
                                try:
                                    import cv2

                                    _cv[0] = cv2
                                except ImportError:
                                    _cv[0] = False
                                    print("[WARNING] opencv-python not installed; --rover_oa_cam_preview has no effect.")
                            cv2_mod = _cv[0]
                            if cv2_mod is not False:
                                _scene = getattr(unwrapped_env, "scene", None)
                                _cam = _scene.sensors.get(_cam_sn) if _scene is not None else None
                                if _cam is not None:
                                    _rgb = _cam.data.output.get("rgb")
                                    if _rgb is not None and _rgb.shape[0] > 0:
                                        _im = _rgb[0]
                                        if _im.dtype != torch.uint8:
                                            _im = (_im * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                                        _arr = _im.detach().cpu().numpy()
                                        _bgr = _arr[..., ::-1].copy()
                                        _hz = int(_bgr.shape[0]) * 2
                                        _wz = int(_bgr.shape[1]) * 2
                                        _bgr_view = cv2_mod.resize(
                                            _bgr, (_wz, _hz), interpolation=cv2_mod.INTER_LINEAR
                                        )
                                        try:
                                            cv2_mod.imshow("Rover OA RGB (env0)", _bgr_view)
                                            cv2_mod.waitKey(1)
                                        except Exception as gui_exc:
                                            rover_oa_cv2_gui_disabled[0] = True
                                            print(
                                                "[WARNING] OpenCV highgui unavailable (opencv-python-headless has no GUI; "
                                                "missing DISPLAY; or no GTK). Camera preview disabled. Inference logging still runs.\n"
                                                "  Fix: run tools/rover/install_opencv_gui_for_cam_preview.sh in this venv, then "
                                                "sudo apt install -y libgtk-3-dev pkg-config libcanberra-gtk3-module\n"
                                                f"  Detail: {gui_exc}"
                                            )

                    if _rover_oa_log_period > 0 and timestep % _rover_oa_log_period == 0:
                        u = unwrapped_env
                        if hasattr(u, "_oa_flat_visual_tracks"):
                            _tr = u._oa_flat_visual_tracks[0].detach().cpu().float().numpy()
                            _nfeat = 4
                            _k = max(1, _tr.size // _nfeat)
                            _tr2 = _tr.reshape(_k, _nfeat)
                            _counts = (
                                u._oa_flat_visual_counts[0].detach().cpu().numpy()
                                if hasattr(u, "_oa_flat_visual_counts")
                                else None
                            )
                            _conf4 = (
                                u._oa_flat_visual_confidence[0].detach().cpu().numpy()
                                if hasattr(u, "_oa_flat_visual_confidence")
                                else None
                            )
                            _hist = (
                                u._oa_flat_visual_bearing_hist[0].detach().cpu().numpy()
                                if hasattr(u, "_oa_flat_visual_bearing_hist")
                                else None
                            )
                            _hist_prefix = _hist[:8].tolist() if _hist is not None else None
                            print(
                                f"[INFO] rover_oa_visual step={timestep} env0 blue_box: "
                                f"counts={_counts} conf_top4={_conf4} hist[:8]={_hist_prefix}"
                            )
                            for _si in range(_k):
                                _s, _c, _trr, _co = _tr2[_si]
                                print(
                                    f"  slot{_si}: sin_bearing={_s:.4f} cos_bearing={_c:.4f} "
                                    f"tanh_range={_trr:.4f} conf={_co:.4f}"
                                )
                            if hasattr(u, "_oa_flat_visual_yellow_goal_tracks"):
                                _ytr = u._oa_flat_visual_yellow_goal_tracks[0].detach().cpu().float().numpy()
                                _ky = max(1, _ytr.size // _nfeat)
                                _y2 = _ytr.reshape(_ky, _nfeat)
                                _ycf = (
                                    u._oa_flat_visual_yellow_goal_confidence[0].detach().cpu().numpy()
                                    if hasattr(u, "_oa_flat_visual_yellow_goal_confidence")
                                    else None
                                )
                                _ycounts = (
                                    u._oa_flat_visual_yellow_goal_counts[0].detach().cpu().numpy()
                                    if hasattr(u, "_oa_flat_visual_yellow_goal_counts")
                                    else None
                                )
                                _yhist_prefix = (
                                    u._oa_flat_visual_yellow_goal_bearing_hist[0].detach().cpu().numpy()[:8].tolist()
                                    if hasattr(u, "_oa_flat_visual_yellow_goal_bearing_hist")
                                    else None
                                )
                                print(
                                    f"[INFO] rover_oa_visual step={timestep} env0 yellow_goal: "
                                    f"counts={_ycounts} conf_top4={_ycf} hist[:8]={_yhist_prefix}"
                                )
                                for _gi in range(_ky):
                                    _s, _c, _trr, _co = _y2[_gi]
                                    print(
                                        f"  goal_slot{_gi}: sin_bearing={_s:.4f} cos_bearing={_c:.4f} "
                                        f"tanh_range={_trr:.4f} conf={_co:.4f}"
                                    )
                elif _rover_oa_log_period > 0 and timestep % max(_rover_oa_log_period, 120) == 0:
                    print(f"[WARNING] rover_oa_flat vision update failed at step {timestep}: {_oa_update_err}")

            if (
                "OAFlat" in train_task_name
                and _rover_oa_policy_dbg_period > 0
                and timestep % _rover_oa_policy_dbg_period == 0
            ):
                try:
                    _scene_dbg = getattr(unwrapped_env, "scene", None)
                    _rover_dbg = _scene_dbg["rover"] if _scene_dbg is not None else None
                    if _rover_dbg is not None:
                        _rover_oa_flat_print_policy_dbg(
                            u=unwrapped_env,
                            rover=_rover_dbg,
                            timestep=timestep,
                            policy_active=bool(rover_policy_active[0]),
                            actions=actions,
                            rewards=rewards,
                            obs_in_for_policy=obs_in_for_policy,
                            dbg_obs=_rover_oa_policy_dbg_obs,
                        )
                except Exception as e:
                    print(f"[WARNING] rover_oa_policy_dbg failed at step {timestep}: {e}")

            # Rover localization: CSV pose logger (true vs estimated, every 100 ms)
            if _rover_loc_log_csv_writer is not None and "Localization" in train_task_name:
                try:
                    sim_time = timestep * dt
                    if sim_time - _rover_loc_log_last_time >= 0.1:  # 100 ms
                        _rover_loc_log_last_time = sim_time
                        _scene_dbg = getattr(unwrapped_env, "scene", None)
                        _rover_dbg = _scene_dbg["rover"] if _scene_dbg is not None else None
                        if _rover_dbg is not None and hasattr(_rover_dbg, "data"):
                            true_pos = _rover_dbg.data.root_pos_w[0, :3].detach().cpu()
                            # Estimated position: stored by action term on unwrapped env
                            est_pos = getattr(unwrapped_env, "_rover_estimated_position", None)
                            if est_pos is not None:
                                est_pos = est_pos[0, :3].detach().cpu()
                            else:
                                est_pos = torch.zeros(3)
                            err = torch.linalg.norm(true_pos - est_pos).item()
                            # Goal marker position
                            try:
                                _goal_marker = _scene_dbg["goal_marker"] if _scene_dbg is not None else None
                            except Exception:
                                _goal_marker = None
                            if _goal_marker is not None and hasattr(_goal_marker, "data"):
                                goal_pos = _goal_marker.data.root_pos_w[0, :3].detach().cpu()
                            else:
                                goal_pos = torch.zeros(3)
                            _rover_loc_log_csv_writer.writerow([
                                f"{sim_time:.6f}", timestep,
                                f"{true_pos[0].item():.6f}", f"{true_pos[1].item():.6f}", f"{true_pos[2].item():.6f}",
                                f"{est_pos[0].item():.6f}", f"{est_pos[1].item():.6f}", f"{est_pos[2].item():.6f}",
                                f"{goal_pos[0].item():.6f}", f"{goal_pos[1].item():.6f}", f"{goal_pos[2].item():.6f}",
                                f"{err:.6f}",
                            ])
                except Exception as e:
                    if timestep % 100 == 0:
                        print(f"[WARNING] rover_loc_pose CSV logging failed at step {timestep}: {e}")

            # Console pose debug for Isaac-Rover-Localization (non-DiskWorld)
            if rover_loc_pose_debug and "Isaac-Rover-Localization" in train_task_name and "DiskWorld" not in train_task_name:
                try:
                    sim_time = timestep * dt
                    if sim_time - _rover_loc_console_last_time >= 0.1 or timestep == 0:
                        _rover_loc_console_last_time = sim_time
                        _scene_dbg = getattr(unwrapped_env, "scene", None)
                        _rover_dbg = _scene_dbg["rover"] if _scene_dbg is not None else None
                        if _rover_dbg is not None and hasattr(_rover_dbg, "data"):
                            pw = _rover_dbg.data.root_pos_w[0]
                            x_gt = float(pw[0].item())
                            y_gt = float(pw[1].item())
                            z_gt = float(pw[2].item())
                            est_pos = getattr(unwrapped_env, "_rover_estimated_position", None)
                            if est_pos is not None:
                                est = est_pos[0, :3].detach().cpu()
                                x_est = float(est[0].item())
                                y_est = float(est[1].item())
                                z_est = float(est[2].item())
                            else:
                                x_est = y_est = z_est = 0.0
                            err = ((x_gt - x_est) ** 2 + (y_gt - y_est) ** 2 + (z_gt - z_est) ** 2) ** 0.5
                            # Goal marker position
                            try:
                                _goal_marker = _scene_dbg["goal_marker"] if _scene_dbg is not None else None
                            except Exception:
                                _goal_marker = None
                            if _goal_marker is not None and hasattr(_goal_marker, "data"):
                                gw = _goal_marker.data.root_pos_w[0]
                                x_gm = float(gw[0].item())
                                y_gm = float(gw[1].item())
                                z_gm = float(gw[2].item())
                            else:
                                x_gm = y_gm = z_gm = 0.0
                            policy_state = "active" if rover_policy_active[0] else "disabled_zero_actions"
                            print(
                                f"[rover_loc_pose] step={timestep} t={sim_time:.3f}s "
                                f"policy={policy_state} | "
                                f"true=({x_gt:.4f}, {y_gt:.4f}, {z_gt:.4f}) | "
                                f"est=({x_est:.4f}, {y_est:.4f}, {z_est:.4f}) | "
                                f"goal=({x_gm:.4f}, {y_gm:.4f}, {z_gm:.4f}) | "
                                f"err={err:.4f}m"
                            )
                except Exception as e:
                    if timestep % 100 == 0:
                        print(f"[WARNING] rover_loc_pose debug failed at step {timestep}: {e}")

            if rover_loc_pose_debug and "Isaac-Rover-DiskWorld-Localization" in train_task_name:
                try:
                    _scene_dbg = getattr(unwrapped_env, "scene", None)
                    _rover_dbg = None
                    if _scene_dbg is not None:
                        try:
                            _rover_dbg = _scene_dbg["rover"]
                        except KeyError:
                            _rover_dbg = None
                    if (
                        _rover_dbg is not None
                        and hasattr(_rover_dbg, "data")
                        and torch.is_tensor(actions)
                        and actions.shape[-1] >= 4
                    ):
                        # Use the exact policy output passed into env.step(actions).
                        act = actions[0, -4:].detach()
                        # Match reward / ``ground_truth_pose``: horizontal plane is world X and Z (Y-up).
                        pos_norm = float(getattr(unwrapped_env.cfg, "estimate_pos_norm_m", 110.0))
                        x_est = float(act[0].item() * pos_norm)
                        z_est = float(act[1].item() * pos_norm)
                        c, s = float(act[2].item()), float(act[3].item())
                        n = max(1e-9, (c * c + s * s) ** 0.5)
                        yaw_est = math.atan2(s / n, c / n)
                        pw = _rover_dbg.data.root_pos_w[0]
                        x_gt = float(pw[0].item())
                        y_gt = float(pw[1].item())
                        z_gt = float(pw[2].item())
                        qw = _rover_dbg.data.root_quat_w[0:1]
                        fwd_b = torch.tensor(
                            [[1.0, 0.0, 0.0]], device=qw.device, dtype=qw.dtype
                        )
                        fwd_w = quat_apply(qw, fwd_b)[0]
                        yaw_gt = math.atan2(float(fwd_w[2].item()), float(fwd_w[0].item()))
                        policy_state = "active" if rover_policy_active[0] else "disabled_zero_actions"
                        print(
                            f"[rover_loc_pose] step={timestep} "
                            f"policy={policy_state} "
                            f"estimated_xz=({x_est:.4f}, {z_est:.4f}) yaw_est={yaw_est:.4f} rad | "
                            f"true_xz=({x_gt:.4f}, {z_gt:.4f}) yaw_true={yaw_gt:.4f} rad "
                            f"y_up={y_gt:.4f} (horizontal=X,Z per localization reward; not X,Y)"
                        )
                        # Update estimate marker position if enabled
                        if _rover_loc_estimate_marker_prim is not None:
                            try:
                                from pxr import UsdGeom, Gf
                                xform = UsdGeom.Xformable(_rover_loc_estimate_marker_prim)
                                # Place marker at estimated (x, y) in world frame, at same height as rover
                                marker_z = float(y_gt) + 0.5  # slightly above ground
                                xform.AddTranslateOp().Set(Gf.Vec3d(float(x_est), float(marker_z), float(z_est)))
                            except Exception as e:
                                print(f"[WARNING] Could not update estimate marker pose: {e}")
                except Exception as e:
                    print(f"[WARNING] rover_loc_pose debug failed at step {timestep}: {e}")

            # Debug ankle contact: in_contact (bool) and external torque from ankle joints, every N ms
            if debug_ankle_contact and ankle_joint_indices_debug is not None:
                sim_time = timestep * dt
                if sim_time - last_ankle_debug_time >= debug_ankle_interval_s:
                    last_ankle_debug_time = sim_time
                    try:
                        robot = getattr(unwrapped_env, "_robot", None) or getattr(unwrapped_env, "robot", None)
                        if robot is not None:
                            applied_torques = robot.data.applied_torque[0, ankle_joint_indices_debug]
                            joint_vels = robot.data.joint_vel[0, ankle_joint_indices_debug]
                            contact_threshold = 0.1
                            is_low_vel = torch.abs(joint_vels) < contact_threshold
                            torque_vel_opposite = (applied_torques * joint_vels) < 0.0
                            in_contact = is_low_vel | torque_vel_opposite
                            external_torque = applied_torques * in_contact.float()
                            in_contact_np = in_contact.cpu().numpy()
                            external_torque_np = external_torque.cpu().numpy()
                            left_foot_in_contact = bool(in_contact_np[2] or in_contact_np[3])
                            right_foot_in_contact = bool(in_contact_np[0] or in_contact_np[1])
                            # Feet alternation phase: detect contact events and update phase
                            left_event = left_foot_in_contact and not debug_prev_left_contact
                            right_event = right_foot_in_contact and not debug_prev_right_contact
                            if left_event:
                                debug_gait_phase = (debug_gait_phase + 0.5) % 1.0
                                debug_last_contact_foot = 1
                            if right_event:
                                debug_gait_phase = (debug_gait_phase + 0.5) % 1.0
                                debug_last_contact_foot = 2
                            debug_prev_left_contact = left_foot_in_contact
                            debug_prev_right_contact = right_foot_in_contact
                            # Phase label from current contact state
                            if left_foot_in_contact and right_foot_in_contact:
                                phase_label = "double_support"
                            elif left_foot_in_contact:
                                phase_label = "left_stance"
                            elif right_foot_in_contact:
                                phase_label = "right_stance"
                            else:
                                phase_label = "flight"
                            last_foot_str = {0: "none", 1: "left", 2: "right"}.get(debug_last_contact_foot, "?")
                            print(f"\n[ANKLE CONTACT DEBUG] t={sim_time:.3f}s")
                            print(f"  feet_alternation_phase: {debug_gait_phase:.3f}  phase_label: {phase_label}  last_contact_foot: {last_foot_str}")
                            print("  in_contact (bool):")
                            for i, name in enumerate(ankle_joint_names_out):
                                print(f"    {name}: {bool(in_contact_np[i])}")
                            print(f"  left_foot_in_contact: {left_foot_in_contact}, right_foot_in_contact: {right_foot_in_contact}")
                            print("  external_torque (Nm):")
                            for i, name in enumerate(ankle_joint_names_out):
                                print(f"    {name}: {external_torque_np[i]:.4f}")
                            if hasattr(robot.data, "body_incoming_joint_wrench_b"):
                                try:
                                    wrench = robot.data.body_incoming_joint_wrench_b
                                    left_foot_ids, _ = robot.find_bodies("left_ankle_roll_link", preserve_order=True)
                                    right_foot_ids, _ = robot.find_bodies("right_ankle_roll_link", preserve_order=True)
                                    if left_foot_ids and right_foot_ids:
                                        w_left = wrench[0, left_foot_ids[0], :].cpu().numpy()
                                        w_right = wrench[0, right_foot_ids[0], :].cpu().numpy()
                                        print("  body_incoming_joint_wrench (left_ankle_roll_link): force=[%.4f, %.4f, %.4f] N, torque=[%.4f, %.4f, %.4f] Nm" % (*w_left[:3], *w_left[3:6]))
                                        print("  body_incoming_joint_wrench (right_ankle_roll_link): force=[%.4f, %.4f, %.4f] N, torque=[%.4f, %.4f, %.4f] Nm" % (*w_right[:3], *w_right[3:6]))
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"[WARNING] Ankle contact debug failed: {e}")
            
            # Debug joint positions: every N steps, or every MS of sim time (G1: use --debug_joint_interval_ms 100)
            if args_cli.debug_joint_positions:
                sim_time_j = timestep * dt
                use_ms = debug_joint_interval_s is not None
                should_print_joints = False
                if use_ms:
                    if sim_time_j - last_joint_debug_time >= debug_joint_interval_s:
                        should_print_joints = True
                        last_joint_debug_time = sim_time_j
                elif timestep % args_cli.debug_joint_interval == 0:
                    should_print_joints = True
                try:
                    if should_print_joints:
                        robot = getattr(unwrapped_env, "_robot", None) or getattr(unwrapped_env, "robot", None)
                        if robot is not None:
                            joint_pos = robot.data.joint_pos[0].cpu().numpy()
                            if hasattr(robot, "joint_names") and robot.joint_names:
                                joint_names = list(robot.joint_names)
                            else:
                                try:
                                    _, jn = robot.find_joints(".*")
                                    joint_names = list(jn)
                                except Exception:
                                    joint_names = [f"joint_{i}" for i in range(len(joint_pos))]

                            joint_pos_deg = torch.rad2deg(torch.tensor(joint_pos)).numpy()
                            mode = f"every {args_cli.debug_joint_interval_ms} ms sim" if use_ms else f"every {args_cli.debug_joint_interval} steps"
                            jf_raw = getattr(args_cli, "debug_joint_filter", None)
                            name_filter: set[str] | None = None
                            if jf_raw and str(jf_raw).strip():
                                name_filter = {x.strip() for x in str(jf_raw).split(",") if x.strip()}
                            print(f"\n[DEBUG JOINT POSITIONS] {mode} | step {timestep}, t={sim_time_j:.3f}s")
                            if name_filter:
                                print(f"  filter: {', '.join(sorted(name_filter))}")
                            else:
                                print(f"  Total joints: {len(joint_pos)}")
                            printed = 0
                            for i in range(len(joint_pos)):
                                joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                                if name_filter is not None and joint_name not in name_filter:
                                    continue
                                print(f"    {joint_name:35s}: {joint_pos_deg[i]:7.2f}° ({joint_pos[i]:.6f} rad)")
                                printed += 1
                            if name_filter and printed == 0:
                                print(f"  [WARNING] No joints matched filter (check names vs robot.joint_names).")
                            if name_filter is None:
                                print(f"  Joint position range: [{joint_pos_deg.min():.2f}°, {joint_pos_deg.max():.2f}°]")
                                print(f"  Mean |q|: {abs(joint_pos_deg).mean():.2f}°")
                                if hasattr(robot.data, "joint_vel"):
                                    joint_vel = robot.data.joint_vel[0].cpu().numpy()
                                    joint_vel_deg_per_s = torch.rad2deg(torch.tensor(joint_vel)).numpy()
                                    print(f"  Mean |qd|: {abs(joint_vel_deg_per_s).mean():.2f}°/s")
                            elif printed and hasattr(robot.data, "joint_vel"):
                                joint_vel = robot.data.joint_vel[0].cpu().numpy()
                                joint_vel_deg_per_s = torch.rad2deg(torch.tensor(joint_vel)).numpy()
                                for i in range(len(joint_pos)):
                                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                                    if joint_name not in name_filter:
                                        continue
                                    print(f"    {joint_name:35s} qd: {joint_vel_deg_per_s[i]:7.2f}°/s ({joint_vel[i]:.6f} rad/s)")
                        elif timestep == 0:
                            print("[WARNING] No robot on environment; cannot debug joint positions.")
                except Exception as e:
                    if should_print_joints:
                        print(f"[WARNING] Failed to debug joint positions at step {timestep}: {e}")

            if g1_link_debug_tick:
                sim_tl = timestep * dt
                if sim_tl - last_g1_link_debug_time >= debug_g1_link_interval_s:
                    last_g1_link_debug_time = sim_tl
                    try:
                        robot = getattr(unwrapped_env, "_robot", None) or getattr(unwrapped_env, "robot", None)
                        if robot is None:
                            raise RuntimeError("no robot")
                        if debug_g1_link_positions and g1_link_body_indices:
                            pw = robot.data.body_pos_w[0, g1_link_body_indices, :].cpu().numpy()
                            print(f"\n[DEBUG G1 LINK POS world] t={sim_tl:.3f}s step={timestep}")
                            for i, name in enumerate(g1_link_body_names_resolved):
                                if i < pw.shape[0]:
                                    x, y, z = float(pw[i, 0]), float(pw[i, 1]), float(pw[i, 2])
                                    print(f"  {name:28s}  x={x:8.4f}  y={y:8.4f}  z={z:8.4f}  (m)")
                        if debug_g1_link_vs_hdf5:
                            ue = unwrapped_env
                            tgt = ue._get_hdf5_link_local_target_pose() if hasattr(ue, "_get_hdf5_link_local_target_pose") else None
                            body_ids = getattr(ue, "_hdf5_link_body_ids", None)
                            slot_ok = getattr(ue, "_hdf5_link_slot_valid", None)
                            if tgt is not None and body_ids is not None and slot_ok is not None:
                                root_pos_w = robot.data.root_pos_w
                                root_quat_w = robot.data.root_quat_w
                                interp = ue._hdf5_link_motion_interp() if hasattr(ue, "_hdf5_link_motion_interp") else None
                                print(f"\n[DEBUG G1 LINK pelvis vs HDF5] t={sim_tl:.3f}s step={timestep}")
                                if interp is not None:
                                    idx_lo, idx_hi, ww = interp
                                    ep = ue.episode_length_buf[0].item()
                                    print(
                                        f"  motion_clock ep_len={ep} | idx_lo/hi/w[0]={idx_lo[0].item()} "
                                        f"{idx_hi[0].item()} {float(ww[0].item()):.4f}"
                                    )
                                t0 = tgt[0].cpu().numpy()
                                for s in range(4):
                                    label = _G1_HDF5_LINK_SLOT_LABELS[s]
                                    ok = bool(slot_ok[s].item()) and int(body_ids[s].item()) >= 0
                                    if not ok:
                                        print(f"  {label:22s}  (slot invalid or body missing)")
                                        continue
                                    bid = int(body_ids[s].item())
                                    bp = robot.data.body_pos_w[:, bid, :]
                                    bq = robot.data.body_quat_w[:, bid, :]
                                    pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, bp, bq)
                                    act = pos_b[0].cpu().numpy()
                                    ref = t0[s]
                                    err = act - ref
                                    dist = float((err * err).sum() ** 0.5)
                                    print(
                                        f"  {label:22s}  actual=[{act[0]:7.4f},{act[1]:7.4f},{act[2]:7.4f}]  "
                                        f"target=[{ref[0]:7.4f},{ref[1]:7.4f},{ref[2]:7.4f}]  "
                                        f"err=[{err[0]:7.4f},{err[1]:7.4f},{err[2]:7.4f}]  |e|={dist:.4f} m"
                                    )
                            else:
                                print(f"[WARNING] debug_g1_link_vs_hdf5: missing target/body_ids at step {timestep}")
                    except Exception as e:
                        print(f"[WARNING] G1 link debug failed at step {timestep}: {e}")
            
            # Log joint data if enabled
            if log_joints and has_keyframe_data:
                # Get joint data from environment
                try:
                    # Get current frame index
                    current_frame = unwrapped_env.current_frame_indices[0].item()
                    
                    # Get reference joint angles (in radians)
                    ref_joints_rad = unwrapped_env.bvh_reference_keyframe[0].cpu().numpy()
                    
                    # Get actual joint angles (in radians) for keyframe-controlled joints
                    bvh_action_indices = unwrapped_env._bvh_action_indices
                    actual_joints_rad = unwrapped_env.robot.data.joint_pos[0, bvh_action_indices].cpu().numpy()
                    
                    # Get joint names
                    joint_names = unwrapped_env.cfg.keyframe_joint_names
                    
                    # Convert to degrees for logging
                    ref_joints_deg = torch.rad2deg(torch.tensor(ref_joints_rad)).numpy()
                    actual_joints_deg = torch.rad2deg(torch.tensor(actual_joints_rad)).numpy()
                    
                    # Compute errors (in degrees)
                    errors_deg = actual_joints_deg - ref_joints_deg
                    
                    # Write header on first step
                    if joint_log_headers is None:
                        headers = ["timestep", "frame_index", "time_s"]
                        for name in joint_names:
                            headers.extend([f"{name}_ref_deg", f"{name}_actual_deg", f"{name}_error_deg"])
                        joint_log_headers = headers
                        joint_log_writer.writerow(joint_log_headers)
                    
                    # Write data row
                    row = [timestep, current_frame, timestep * dt]
                    for i in range(len(joint_names)):
                        row.extend([ref_joints_deg[i], actual_joints_deg[i], errors_deg[i]])
                    joint_log_writer.writerow(row)
                    
                    # Periodic console output
                    if timestep % args_cli.log_interval == 0:
                        print(f"\n[STEP {timestep}] Frame: {current_frame:.2f}, Time: {timestep * dt:.3f}s")
                        print(f"  Joint Errors (deg):")
                        for i, name in enumerate(joint_names):
                            print(f"    {name:30s}: ref={ref_joints_deg[i]:7.2f}, actual={actual_joints_deg[i]:7.2f}, error={errors_deg[i]:7.2f}")
                        
                        # Compute mean absolute error
                        mean_abs_error = abs(errors_deg).mean()
                        print(f"  Mean Absolute Error: {mean_abs_error:.2f} deg")
                        
                        # Get reward components if available
                        if hasattr(unwrapped_env, 'get_average_reward_components'):
                            reward_components = unwrapped_env.get_average_reward_components()
                            if reward_components:
                                print(f"  Imitation Reward: {reward_components.get('imitation_reward', 0.0):.4f}")
                                print(f"  Total Reward: {reward_components.get('total_reward', 0.0):.4f}")
                
                except Exception as e:
                    if timestep % args_cli.log_interval == 0:
                        print(f"[WARNING] Failed to log joint data at step {timestep}: {e}")
        
        timestep += 1
        
        # Exit the play loop after recording one video (if video mode)
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Close joint log file if opened
    if joint_log_file_handle is not None:
        joint_log_file_handle.close()
        if log_joints:
            print(f"\n[INFO] Joint log saved to: {joint_log_file}")

    # Close rover localization pose CSV log if opened
    if _rover_loc_log_csv_file is not None:
        _rover_loc_log_csv_file.close()
        if _rover_loc_log_csv_path is not None:
            print(f"[INFO] Rover localization pose CSV log saved to: {_rover_loc_log_csv_path}")

    # Unsubscribe G1 play keyboard binding
    if _keyboard_sub is not None:
        try:
            _input = carb.input.acquire_input_interface()
            _input.unsubscribe_to_keyboard_events(_keyboard_for_unsub, _keyboard_sub)
        except Exception:
            pass

    if rover_oa_flat_play_features and rover_oa_cv2_holder[0] not in (None, False):
        try:
            rover_oa_cv2_holder[0].destroyAllWindows()
        except Exception:
            pass

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

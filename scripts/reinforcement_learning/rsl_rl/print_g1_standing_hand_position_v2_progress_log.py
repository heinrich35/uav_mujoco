#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress analysis for G1 Standing Hand Position V2.
# Parses training_progress_log.txt (including Episode_Reward/* and Episode_Termination/*)
# and generates training_progress_analysis.png with reward categories including separate X/Y drift penalties.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_g1_standing_hand_position_v2_progress_log.py <log_dir>
#   python scripts/reinforcement_learning/rsl_rl/print_g1_standing_hand_position_v2_progress_log.py logs/rsl_rl/g1_standing_hand_position_v2/2026-03-15_12-00-00
#
# Options:
#   --last N       Print only the last N iteration blocks (default: all)
#   --no-plot      Do not generate the visual analysis PNG
#   --output NAME  Output filename for the plot (default: training_progress_analysis.png)
#   --plot-only    Only generate the visual analysis PNG

from __future__ import annotations

import argparse
import os
import re
import sys


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _parse_block(block: str) -> tuple[int | None, dict[str, float]]:
    """Parse one iteration block. Returns (iteration_number, {metric_name: value})."""
    block = _strip_ansi(block)
    out = {}
    it_match = re.search(r"Learning iteration\s+(\d+)/", block)
    iteration = int(it_match.group(1)) if it_match else None
    for m in re.finditer(r"^([^:]+?):\s*(-?[\d.]+)\s*$", block, re.MULTILINE):
        key = m.group(1).strip()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return iteration, out


def parse_training_progress_log(path: str) -> tuple[list[int], dict[str, list[float]]]:
    """Parse full training_progress_log.txt. Returns (iterations, {metric: [values]})."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"(?=^################################################################################\n)", content, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]

    iterations = []
    all_keys = set()
    block_data = []
    for block in blocks:
        it, data = _parse_block(block)
        if it is not None and data:
            iterations.append(it)
            block_data.append(data)
            all_keys |= set(data.keys())

    if not block_data:
        return [], {}

    series = {k: [] for k in all_keys}
    for data in block_data:
        for k in all_keys:
            series[k].append(data.get(k))

    for k in list(series.keys()):
        prev = None
        for i, v in enumerate(series[k]):
            if v is not None:
                prev = v
            elif prev is not None:
                series[k][i] = prev
            else:
                series[k][i] = 0.0

    return iterations, series


# G1 Standing Hand Position V2: reward/termination keys from g1_standing_hand_position_v2_env.
#
# Reward components:
#   Positive: up_reward, heading_reward, alive_reward, progress_reward, smooth_reward, hand_position_reward,
#     ik_joint_reward (V2: arm joints following IK for moving target)
#   Penalties: actions_cost, electricity_cost, dof_at_limit_cost, death_cost, joint_limit_cost, jerk_cost,
#     xy_drift_penalty (V1 combined), x_drift_penalty (V2 ±X), y_drift_penalty (V2 ±Y),
#     ik_joint_penalty (V2: joints outside IK solution)
#   Termination: Episode_Termination/fall, Episode_Termination/time_out
#
CATEGORY_MAIN = [
    "Mean reward",
]
CATEGORY_LOSSES = [
    "Mean value_function loss",
    "Mean surrogate loss",
    "Mean entropy loss",
]
CATEGORY_REWARDS_POSITIVE = [
    "Episode_Reward/up_reward",
    "Episode_Reward/heading_reward",
    "Episode_Reward/alive_reward",
    "Episode_Reward/progress_reward",
    "Episode_Reward/smooth_reward",
    "Episode_Reward/hand_position_reward",
    "Episode_Reward/ik_joint_reward",   # V2: arm joints following IK for moving target
]
CATEGORY_REWARDS_PENALTIES = [
    "Episode_Reward/actions_cost",
    "Episode_Reward/electricity_cost",
    "Episode_Reward/dof_at_limit_cost",
    "Episode_Reward/death_cost",
    "Episode_Reward/joint_limit_cost",
    "Episode_Reward/jerk_cost",
    "Episode_Reward/xy_drift_penalty",
    "Episode_Reward/x_drift_penalty",   # V2: drift on ±X axis
    "Episode_Reward/y_drift_penalty",   # V2: drift on ±Y axis
    "Episode_Reward/ik_joint_penalty",  # V2: joints outside IK solution
]
CATEGORY_TERMINATION = [
    "Episode_Termination/fall",
    "Episode_Termination/time_out",
]

# Keys from G1 standing hand position V2 run (order for display)
STANDING_HAND_POSITION_V2_RUN_KEYS = [
    "task",
    "num_envs",
    "max_iterations",
    "seed",
    "headless",
    "device",
    "experiment_name",
    "reward_log_interval",
    "progress_log_interval",
    "progress_reward_multiplier",
    "up_weight",
    "heading_weight",
    "alive_reward_scale",
    "actions_cost_scale",
    "energy_cost_scale",
    "termination_height",
    "joints_data",
    "npz_only_for_limits",
    "joint_limit_penalty_weight",
    "npz_joint_limit_extend_deg",
    "npz_joint_limit_extend_deg_non_leg",
    "action_rate_penalty_scale",
    "joint_acceleration_penalty_scale",
    "smooth_motion_reward_weight",
    "smooth_motion_sigma",
    "decimation",
    "hand_target_pos_b",
    "hand_reward_weight",
    "hand_reward_sigma",
    "xy_drift_tolerance",
    "xy_drift_penalty_weight",
    "x_drift_tolerance",
    "x_drift_penalty_weight",
    "y_drift_tolerance",
    "y_drift_penalty_weight",
    "ik_target_motion_speed",
    "ik_target_motion_period_base",
    "ik_joint_reward_weight",
    "ik_joint_reward_sigma",
    "ik_joint_penalty_weight",
    "ik_joint_penalty_threshold",
    "reward_clip_min",
    "reward_clip_max",
]

RUN_ARG_ABBREV: dict[str, str] = {
    "task": "task",
    "num_envs": "num_envs",
    "max_iterations": "max_iter",
    "seed": "seed",
    "headless": "headless",
    "device": "device",
    "experiment_name": "exp_name",
    "reward_log_interval": "reward_log_int",
    "progress_log_interval": "prog_log_int",
    "progress_reward_multiplier": "prog_rew_mult",
    "up_weight": "up_weight",
    "heading_weight": "heading_weight",
    "alive_reward_scale": "alive_rew_scale",
    "actions_cost_scale": "actions_cost",
    "energy_cost_scale": "energy_cost",
    "termination_height": "term_height",
    "joints_data": "joints_data",
    "npz_only_for_limits": "npz_limits_only",
    "joint_limit_penalty_weight": "joint_lim_pen",
    "npz_joint_limit_extend_deg": "npz_ext_deg",
    "npz_joint_limit_extend_deg_non_leg": "npz_ext_non_leg",
    "action_rate_penalty_scale": "act_rate_pen",
    "joint_acceleration_penalty_scale": "joint_acc_pen",
    "smooth_motion_reward_weight": "smooth_rew",
    "smooth_motion_sigma": "smooth_sigma",
    "decimation": "decimation",
    "hand_target_pos_b": "hand_target_b",
    "hand_reward_weight": "hand_rew",
    "hand_reward_sigma": "hand_sigma",
    "xy_drift_tolerance": "xy_drift_tol",
    "xy_drift_penalty_weight": "xy_drift_pen",
    "x_drift_tolerance": "x_drift_tol",
    "x_drift_penalty_weight": "x_drift_pen",
    "y_drift_tolerance": "y_drift_tol",
    "y_drift_penalty_weight": "y_drift_pen",
    "ik_target_motion_speed": "ik_motion_speed",
    "ik_target_motion_period_base": "ik_motion_period",
    "ik_joint_reward_weight": "ik_rew",
    "ik_joint_reward_sigma": "ik_rew_sigma",
    "ik_joint_penalty_weight": "ik_pen",
    "ik_joint_penalty_threshold": "ik_pen_thr",
    "reward_clip_min": "rew_clip_lo",
    "reward_clip_max": "rew_clip_hi",
}


def parse_header_run_args(path: str) -> dict[str, str]:
    """Parse 'Run arguments (this run)' section from training_progress_log.txt. Returns {key: value}."""
    out: dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    in_run_args = False
    for line in content.splitlines():
        if "Run arguments (this run)" in line:
            in_run_args = True
            continue
        if in_run_args:
            if line.strip().startswith("######") and len(out) > 0:
                break
            if line.startswith("  ") and ":" in line:
                part = line.strip()
                idx = part.find(":")
                if idx >= 0:
                    key = part[:idx].strip()
                    val = part[idx + 1 :].strip()
                    out[key] = val
    return out


def generate_visual_analysis(log_dir: str, output_name: str = "training_progress_analysis.png") -> str | None:
    """Generate training_progress_analysis.png in log_dir. Returns path or None on failure."""
    path = os.path.join(log_dir, "training_progress_log.txt")
    if not os.path.isfile(path):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    iterations, series = parse_training_progress_log(path)
    if not iterations or not series:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Training progress — g1_standing_hand_position_v2 (from training_progress_log.txt)", fontsize=12)

    # Main
    ax = axes[0, 0]
    for k in CATEGORY_MAIN:
        if k in series and series[k]:
            ax.plot(iterations, series[k], label=k, marker="o", markersize=3)
    ax.set_title("Main metrics")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")

    # Losses
    ax = axes[0, 1]
    for k in CATEGORY_LOSSES:
        if k in series and series[k]:
            ax.plot(iterations, series[k], label=k.replace("Mean ", ""), marker="o", markersize=3)
    ax.set_title("Losses")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")

    # Rewards (positive)
    ax = axes[1, 0]
    for k in CATEGORY_REWARDS_POSITIVE:
        if k in series and series[k]:
            ax.plot(iterations, series[k], label=k.replace("Episode_Reward/", ""), marker="o", markersize=3)
    ax.set_title("Rewards (positive)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")

    # Rewards (penalties / costs)
    ax = axes[1, 1]
    neg_min = 0
    for k in CATEGORY_REWARDS_PENALTIES:
        if k in series and series[k]:
            label = k.replace("Episode_Reward/", "")
            if label == "joint_limit_cost":
                label = "joint_limit (NPZ)"
            elif label == "xy_drift_penalty":
                label = "xy_drift"
            elif label == "x_drift_penalty":
                label = "x_drift"
            elif label == "y_drift_penalty":
                label = "y_drift"
            elif label == "ik_joint_penalty":
                label = "ik_joint"
            neg_vals = [-v for v in series[k]]
            ax.plot(iterations, neg_vals, label=label, marker="o", markersize=3)
            neg_min = min(neg_min, min(neg_vals))
    ax.set_ylim(min(-1, neg_min), 0)
    ax.set_ylabel("Contribution to total")
    ax.set_title("Rewards (penalties / costs)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")

    # Termination
    ax = axes[2, 0]
    for k in CATEGORY_TERMINATION:
        if k in series and series[k] is not None and len(series[k]) > 0:
            ax.plot(iterations, series[k], label=k.replace("Episode_Termination/", ""), marker="o", markersize=3)
    ax.set_title("Termination (fall vs time_out)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")

    # Run arguments box
    ax_box = axes[2, 1]
    ax_box.set_axis_off()
    header_args = parse_header_run_args(path)
    items = []
    for key in STANDING_HAND_POSITION_V2_RUN_KEYS:
        if key in header_args:
            short = RUN_ARG_ABBREV.get(key, key)
            val = header_args[key]
            if len(val) > 26:
                val = val[:23] + "…"
            items.append((short, val))
    n = len(items)
    mid = (n + 1) // 2
    left_lines = [f"{s}: {v}" for s, v in items[:mid]]
    right_lines = [f"{s}: {v}" for s, v in items[mid:]]
    col_w = 36
    lines = ["Run arguments (from header):"]
    for i in range(max(len(left_lines), len(right_lines))):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        lines.append(f"{left:<{col_w}}  {right}")
    text = "\n".join(lines)
    ax_box.text(0.02, 0.98, text, transform=ax_box.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9))

    plt.tight_layout()
    out_path = os.path.join(log_dir, output_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Print training_progress_log.txt for g1_standing_hand_position_v2 and generate visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/g1_standing_hand_position_v2/2026-03-15_12-00-00",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Print only the last N iteration blocks (default: all)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate the visual analysis PNG",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_progress_analysis.png",
        metavar="NAME",
        help="Output filename for the plot (default: training_progress_analysis.png)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate the visual analysis PNG; do not print the log.",
    )
    args = parser.parse_args()

    path = os.path.join(args.log_dir, "training_progress_log.txt")
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if not args.plot_only:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"(?=^################################################################################\n)", content, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]
        if args.last is not None and args.last > 0:
            blocks = blocks[-args.last :]
        for block in blocks:
            print(_strip_ansi(block))
            print()

    if not args.no_plot:
        out_path = generate_visual_analysis(args.log_dir, args.output)
        if out_path:
            print(f"[INFO] Visual analysis saved to: {out_path}", file=sys.stderr)
        else:
            print("[WARNING] Could not generate visual analysis (missing data or matplotlib).", file=sys.stderr)


if __name__ == "__main__":
    main()

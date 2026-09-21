#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress analysis for g1_locomotion_v1_walking.
# Parses training_progress_log.txt (including Episode_Reward/* and Episode_Termination/*)
# and generates training_progress_analysis.png with G1 walking reward categories.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_g1_locomotion_v1_walking_progress_log.py <log_dir>
#   python scripts/reinforcement_learning/rsl_rl/print_g1_locomotion_v1_walking_progress_log.py logs/rsl_rl/g1_locomotion_v1_walking/2026-03-12_20-00-00
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
    """Parse one iteration block. Returns (iteration_number, {metric_name: value}).
    Matches lines like '  Episode_Reward/actions_cost: 2.3049' or 'death_cost: -0.0065'.
    """
    block = _strip_ansi(block)
    out = {}
    it_match = re.search(r"Learning iteration\s+(\d+)/", block)
    iteration = int(it_match.group(1)) if it_match else None
    # Match key: value (value can be int, float, or neg; optional exponent e.g. 1e-5)
    for m in re.finditer(r"^([^:]+?):\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*", block, re.MULTILINE):
        key = m.group(1).strip()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return iteration, out


def parse_header_smooth_jerk(path: str) -> dict[str, float] | None:
    """Parse log header for smooth/jerk args. Returns dict with keys action_rate_penalty_scale, etc., or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Header is before first "Learning iteration"; get lines like "  action_rate_penalty_scale: 0.01"
        head = content.split("Learning iteration")[0]
        out = {}
        for name in (
            "action_rate_penalty_scale",
            "joint_acceleration_penalty_scale",
            "smooth_motion_reward_weight",
            "smooth_motion_sigma",
        ):
            m = re.search(rf"^\s*{re.escape(name)}\s*:\s*([\d.eE+-]+)", head, re.MULTILINE)
            if m:
                try:
                    out[name] = float(m.group(1))
                except ValueError:
                    pass
        return out if out else None
    except Exception:
        return None


def parse_header_drift_z_orientation(path: str) -> dict[str, float] | None:
    """Parse log header for drift and z_orientation args. Returns dict or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        head = content.split("Learning iteration")[0]
        out = {}
        for name in (
            "y_drift_tolerance",
            "y_drift_penalty_weight",
            "target_z_orientation",
            "z_orientation_tolerance",
            "z_orientation_penalty_weight",
            "alternating_foot_reward_weight",
            "same_foot_tap_penalty_weight",
            "jump_penalty_weight",
            "symmetry_reward_weight", "symmetry_penalty_weight", "symmetry_contact_sharpness",
            "symmetry_touchdown_sharpness", "symmetry_torque_sharpness",
            "step_length_min_m", "step_length_reward_weight", "step_length_penalty_weight",
            "foot_crossing_reward_weight", "foot_crossing_penalty_weight",
        ):
            m = re.search(rf"^\s*{re.escape(name)}\s*:\s*([\d.eE+-]+)", head, re.MULTILINE)
            if m:
                try:
                    out[name] = float(m.group(1))
                except ValueError:
                    pass
        return out if out else None
    except Exception:
        return None


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


# G1 locomotion V1 walking: reward/termination keys from g1_locomotion_v1_walking_env.
# Config args: alternating_foot_reward_weight, same_foot_tap_penalty_weight, jump_penalty_weight,
#   symmetry_reward_weight, symmetry_penalty_weight, symmetry_contact_sharpness,
#   symmetry_touchdown_sharpness, symmetry_torque_sharpness.
# Note: symmetry reward/penalty are independent—they work even when alternating_foot and
#   same_foot_tap weights are 0 (as long as symmetry_reward_weight or symmetry_penalty_weight > 0).
#
# Rewards (positive): up_reward, heading_reward, alive_reward, progress_reward, velocity_reward,
#   smooth_reward, alternating_foot_reward (L-R-L-R swing), symmetry_reward (L-R contact/touchdown/torque).
# Penalties: actions_cost, electricity_cost, dof_at_limit_cost, death_cost, joint_limit_cost, jerk_cost,
#   drift_penalty, z_orientation_penalty, same_foot_double_non_contact_penalty (brief tap F→T→F),
#   both_feet_non_contact_penalty (jump), symmetry_penalty (L-R asymmetry).
# Termination: Episode_Termination/fall, Episode_Termination/time_out.
#
# Main metrics: outcome + often-neglected ML inspection (no episode length, no duplicates)
CATEGORY_MAIN_LEFT = ["Mean reward"]  # primary outcome
# Right axis: iter time (s), throughput (steps/s ×1e-5) — similar scale, both vary
CATEGORY_LOSSES = [
    "Mean value_function loss",
    "Mean surrogate loss",
    "Mean entropy loss",
    "Mean action noise std",
]
CATEGORY_REWARDS_POSITIVE = [
    "Episode_Reward/up_reward",
    "Episode_Reward/heading_reward",
    "Episode_Reward/alive_reward",
    "Episode_Reward/progress_reward",
    "Episode_Reward/velocity_reward",  # reward for maintaining target +x velocity
    "Episode_Reward/smooth_reward",   # reward for smooth human-like joint motion
    "Episode_Reward/alternating_foot_reward",  # reward for L-R-L-R foot swing
    "Episode_Reward/symmetry_reward",  # reward for L-R symmetry (contact, touchdown, torque)
    "Episode_Reward/step_length_reward",  # reward when foot x step >= step_length_min_m
    "Episode_Reward/foot_crossing_reward",  # reward when foot crosses from behind to ahead (rel x neg→pos)
]
CATEGORY_REWARDS_PENALTIES = [
    "Episode_Reward/actions_cost",
    "Episode_Reward/electricity_cost",
    "Episode_Reward/dof_at_limit_cost",
    "Episode_Reward/death_cost",
    "Episode_Reward/joint_limit_cost",
    "Episode_Reward/jerk_cost",       # penalty for jerky/flickering joint motion
    "Episode_Reward/drift_penalty",   # penalty for +/- y drift beyond tolerance
    "Episode_Reward/z_orientation_penalty",  # penalty for yaw deviation from target
    "Episode_Reward/same_foot_double_non_contact_penalty",  # penalty for brief tap (liftoff→touchdown→liftoff) while other stays in contact
    "Episode_Reward/both_feet_non_contact_penalty",  # penalty when both feet off (jump)
    "Episode_Reward/symmetry_penalty",  # penalty for L-R asymmetry (contact, touchdown, torque)
    "Episode_Reward/step_length_penalty",  # penalty when foot x step < step_length_min_m
    "Episode_Reward/foot_crossing_penalty",  # penalty when step but foot did not cross (rel x neg→pos)
]


def generate_visual_analysis(log_dir: str, output_name: str = "training_progress_analysis.png") -> str | None:
    """Generate training_progress_analysis.png in log_dir. Returns path or None on failure."""
    path = os.path.join(log_dir, "training_progress_log.txt")
    if not os.path.isfile(path):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        return None

    iterations, series = parse_training_progress_log(path)
    if not iterations or not series:
        return None

    # Layout: top row = overview + losses, bottom = rewards; no termination plot; config on right
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], width_ratios=[1, 1, 0.7],
                  hspace=0.35, wspace=0.3)
    fig.suptitle("Training progress — g1_locomotion_v1_walking", fontsize=14, fontweight="bold")

    _plot_style = {"marker": "o", "markersize": 2.5, "linewidth": 1.5}
    _line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]

    def _ax_style(ax):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

    # --- Row 0: Main metrics (outcome + sample efficiency / throughput) + Losses ---
    ax_main = fig.add_subplot(gs[0, 0])
    for k in CATEGORY_MAIN_LEFT:
        if k in series and series[k]:
            ax_main.plot(iterations, series[k], label=k, color="C0", **_plot_style)
    ax_main.set_ylabel("Mean reward", fontsize=9, color="C0")
    ax_main.tick_params(axis="y", labelcolor="C0")
    ax_main.set_title("Main metrics (reward + sample efficiency)", fontsize=10, fontweight="bold")
    _ax_style(ax_main)
    ax_main.set_xlabel("Iteration", fontsize=9)
    ax_main2 = ax_main.twinx()
    has_right = False
    if "Iteration time" in series and series["Iteration time"]:
        ax_main2.plot(iterations, series["Iteration time"], label="iter time (s)", color="C1",
                      linestyle="--", **_plot_style)
        has_right = True
    if "Computation" in series and series["Computation"]:
        comp = [x * 1e-5 for x in series["Computation"]]  # 100k–300k -> 1–3, comparable to iter time
        ax_main2.plot(iterations, comp, label="throughput (×10⁵ steps/s)", color="C2", linestyle=":", **_plot_style)
        has_right = True
    if has_right:
        ax_main2.set_ylabel("iter time (s) / throughput", fontsize=8, color="gray")
        ax_main2.tick_params(axis="y", labelcolor="gray")
        ax_main2.legend(loc="upper right", fontsize=6)
    ax_main.legend(loc="upper left", fontsize=8)

    ax_loss = fig.add_subplot(gs[0, 1])
    for idx, k in enumerate(CATEGORY_LOSSES):
        if k in series and series[k]:
            ls = _line_styles[idx % len(_line_styles)]
            ax_loss.plot(iterations, series[k], label=k.replace("Mean ", ""), linestyle=ls, **_plot_style)
    ax_loss.set_title("Losses", fontsize=10, fontweight="bold")
    ax_loss.legend(loc="upper right", fontsize=8)
    _ax_style(ax_loss)
    ax_loss.set_xlabel("Iteration", fontsize=9)

    # --- Row 1: Rewards (positive) + Rewards (penalties) ---
    ax_pos = fig.add_subplot(gs[1, 0])
    for idx, k in enumerate(CATEGORY_REWARDS_POSITIVE):
        if k in series and series[k]:
            label = k.replace("Episode_Reward/", "")
            if label == "velocity_reward":
                label = "velocity (target +x)"
            elif label == "alternating_foot_reward":
                label = "alternating (L-R)"
            elif label == "symmetry_reward":
                label = "symmetry (L-R contact/TD/torque)"
            elif label == "step_length_reward":
                label = "step_length (x ≥ min)"
            elif label == "foot_crossing_reward":
                label = "foot_crossing (neg→pos)"
            ls = _line_styles[idx % len(_line_styles)]
            ax_pos.plot(iterations, series[k], label=label, linestyle=ls, **_plot_style)
    ax_pos.set_title("Rewards (positive)", fontsize=10, fontweight="bold")
    ax_pos.legend(loc="upper left", fontsize=8, ncol=2)
    _ax_style(ax_pos)
    ax_pos.set_ylabel("Reward", fontsize=9)
    ax_pos.set_xlabel("Iteration", fontsize=9)

    ax_pen = fig.add_subplot(gs[1, 1])
    for idx, k in enumerate(CATEGORY_REWARDS_PENALTIES):
        if k in series and series[k]:
            vals = series[k]
            if "death_cost" in k:
                plot_vals = vals
            else:
                plot_vals = [-abs(v) for v in vals]
            label = k.replace("Episode_Reward/", "")
            if label == "joint_limit_cost":
                label = "joint_limit (NPZ)"
            elif label == "drift_penalty":
                label = "drift (y)"
            elif label == "z_orientation_penalty":
                label = "z_orientation (yaw)"
            elif label == "same_foot_double_non_contact_penalty":
                label = "same_foot_tap (F-T-F)"
            elif label == "both_feet_non_contact_penalty":
                label = "jump (both off)"
            elif label == "symmetry_penalty":
                label = "symmetry (L-R asymmetry)"
            elif label == "step_length_penalty":
                label = "step_length (x < min)"
            elif label == "foot_crossing_penalty":
                label = "foot_crossing (no cross)"
            ls = _line_styles[idx % len(_line_styles)]
            ax_pen.plot(iterations, plot_vals, label=label, linestyle=ls, **_plot_style)
    ax_pen.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_pen.set_title("Penalties / costs (shown negative)", fontsize=10, fontweight="bold")
    ax_pen.legend(loc="lower left", fontsize=7, ncol=2)
    _ax_style(ax_pen)
    ax_pen.set_ylabel("Penalty", fontsize=9)
    ax_pen.set_xlabel("Iteration", fontsize=9)

    # --- Config panel (right column, spans rows 0–2) ---
    ax_cfg = fig.add_subplot(gs[0:2, 2])
    smooth_jerk = parse_header_smooth_jerk(path)
    drift_z = parse_header_drift_z_orientation(path)
    lines = ["Config (from log header)", ""]
    if smooth_jerk:
        lines.extend([
            "Smooth / jerk",
            f"  action_rate_scale: {smooth_jerk.get('action_rate_penalty_scale', '—')}",
            f"  joint_accel_scale: {smooth_jerk.get('joint_acceleration_penalty_scale', '—')}",
            f"  smooth_weight: {smooth_jerk.get('smooth_motion_reward_weight', '—')}",
            f"  smooth_sigma: {smooth_jerk.get('smooth_motion_sigma', '—')}",
            "",
        ])
    if drift_z:
        lines.extend([
            "Drift / yaw",
            f"  y_drift_tol: {drift_z.get('y_drift_tolerance', '—')}",
            f"  y_drift_weight: {drift_z.get('y_drift_penalty_weight', '—')}",
            f"  z_orient_tol: {drift_z.get('z_orientation_tolerance', '—')}",
            f"  z_orient_weight: {drift_z.get('z_orientation_penalty_weight', '—')}",
            "",
            "Alternating foot",
            f"  alt_reward_weight: {drift_z.get('alternating_foot_reward_weight', '—')}",
            f"  same_foot_tap_pen_weight: {drift_z.get('same_foot_tap_penalty_weight', '—')}",
            f"  jump_pen_weight: {drift_z.get('jump_penalty_weight', '—')}",
            "",
            "Symmetry (L-R)",
            f"  sym_reward_weight: {drift_z.get('symmetry_reward_weight', '—')}",
            f"  sym_penalty_weight: {drift_z.get('symmetry_penalty_weight', '—')}",
            f"  sym_contact_sharp: {drift_z.get('symmetry_contact_sharpness', '—')}",
            f"  sym_td_sharp: {drift_z.get('symmetry_touchdown_sharpness', '—')}",
            f"  sym_torque_sharp: {drift_z.get('symmetry_torque_sharpness', '—')}",
            "",
            "Step length",
            f"  step_length_min_m: {drift_z.get('step_length_min_m', '—')}",
            f"  step_length_reward: {drift_z.get('step_length_reward_weight', '—')}",
            f"  step_length_penalty: {drift_z.get('step_length_penalty_weight', '—')}",
            "",
            "Foot crossing",
            f"  foot_crossing_reward: {drift_z.get('foot_crossing_reward_weight', '—')}",
            f"  foot_crossing_penalty: {drift_z.get('foot_crossing_penalty_weight', '—')}",
        ])
    ax_cfg.text(0.05, 0.98, "\n".join(lines), transform=ax_cfg.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, pad=0.8))
    ax_cfg.set_axis_off()

    out_path = os.path.join(log_dir, output_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Print training_progress_log.txt for g1_locomotion_v1_walking and generate G1 walking visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/g1_locomotion_v1_walking/2026-03-12_20-00-00",
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

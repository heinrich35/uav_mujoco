#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress analysis for G1 Locomotion Walking V2 (g1_locomotion_walking_v2).
# Parses training_progress_log.txt (Episode_Reward/* and Episode_Termination/*)
# and generates training_progress_analysis.png with reward component categories.
# All reward components are tunable via CLI; this script reflects the same set as V1 walking env.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_g1_locomotion_walking_v2_progress_log.py <log_dir>
#   python scripts/reinforcement_learning/rsl_rl/print_g1_locomotion_walking_v2_progress_log.py logs/rsl_rl/g1_locomotion_walking_v2/2026-03-15_12-00-00
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
    for m in re.finditer(r"^([^:]+?):\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*", block, re.MULTILINE):
        key = m.group(1).strip()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return iteration, out


def parse_header_smooth_jerk(path: str) -> dict[str, float] | None:
    """Parse log header for smooth/jerk args."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
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


def parse_header_reward_config(path: str) -> dict[str, float] | None:
    """Parse log header for reward config (velocity, drift, z_orientation, alternating, symmetry, step, foot_crossing)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        head = content.split("Learning iteration")[0]
        out = {}
        for name in (
            "target_velocity",
            "velocity_reward_weight",
            "velocity_reward_sharpness",
            "x_drift_tolerance",
            "x_drift_penalty_weight",
            "y_drift_tolerance",
            "y_drift_penalty_weight",
            "target_z_orientation",
            "z_orientation_tolerance",
            "z_orientation_penalty_weight",
            "imitation_joints_position_reward_weight",
            "imitation_joints_position_penalty_weight",
            "imitation_position_reward_weight",
            "imitation_position_penalty_weight",
            "imitation_reward_weight",
            "imitation_penalty_weight",
            "npz_imitation_loop_count",
            "npz_imitation_time_factor",
            "npz_imitation_position_tolerance_pct",
            "npz_imitation_velocity_tolerance_pct",
            "alternating_foot_reward_weight",
            "same_foot_tap_penalty_weight",
            "jump_penalty_weight",
            "symmetry_reward_weight",
            "symmetry_penalty_weight",
            "symmetry_contact_sharpness",
            "symmetry_touchdown_sharpness",
            "symmetry_torque_sharpness",
            "step_length_min_m",
            "step_length_reward_weight",
            "step_length_penalty_weight",
            "foot_crossing_reward_weight",
            "foot_crossing_penalty_weight",
            "lin_vel_z_penalty_weight",
            "lin_vel_z_tolerance",
            "ang_vel_xy_penalty_weight",
            "flat_orientation_penalty_weight",
            "flat_orientation_tolerance",
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


def _joint_pos_imitation_header_weight(reward_cfg: dict[str, float], *, reward: bool) -> float | None:
    keys = (
        ("imitation_joints_position_reward_weight", "imitation_position_reward_weight", "imitation_reward_weight")
        if reward
        else ("imitation_joints_position_penalty_weight", "imitation_position_penalty_weight", "imitation_penalty_weight")
    )
    for k in keys:
        if k in reward_cfg and reward_cfg[k] is not None:
            try:
                return float(reward_cfg[k])
            except (TypeError, ValueError):
                continue
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


# G1 Locomotion Walking V2: same reward/termination keys as V1 walking env (shared implementation).
# Reward components (all tunable from CLI): velocity, drift, z_orientation, alternating_foot,
# same_foot_tap, jump, symmetry (optional), step_length (optional), foot_crossing (optional),
# plus base up/heading/alive/progress/smooth/jerk.
CATEGORY_MAIN_LEFT = ["Mean reward"]
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
    "Episode_Reward/velocity_reward",
    "Episode_Reward/smooth_reward",
    "Episode_Reward/imitation_reward",
    "Episode_Reward/alternating_foot_reward",
    "Episode_Reward/symmetry_reward",
    "Episode_Reward/step_length_reward",
    "Episode_Reward/foot_crossing_reward",
]
CATEGORY_REWARDS_PENALTIES = [
    "Episode_Reward/actions_cost",
    "Episode_Reward/electricity_cost",
    "Episode_Reward/dof_at_limit_cost",
    "Episode_Reward/death_cost",
    "Episode_Reward/joint_limit_cost",
    "Episode_Reward/imitation_penalty",
    "Episode_Reward/jerk_cost",
    "Episode_Reward/drift_penalty",  # legacy (old runs): same as y_drift; removed in code
    "Episode_Reward/x_drift_penalty",
    "Episode_Reward/y_drift_penalty",
    "Episode_Reward/z_orientation_penalty",
    "Episode_Reward/same_foot_double_non_contact_penalty",
    "Episode_Reward/both_feet_non_contact_penalty",
    "Episode_Reward/symmetry_penalty",
    "Episode_Reward/step_length_penalty",
    "Episode_Reward/foot_crossing_penalty",
    "Episode_Reward/lin_vel_z_penalty",
    "Episode_Reward/ang_vel_xy_penalty",
    "Episode_Reward/flat_orientation_penalty",
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

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], width_ratios=[1, 1, 0.7],
                  hspace=0.35, wspace=0.3)
    fig.suptitle("Training progress — g1_locomotion_walking_v2", fontsize=14, fontweight="bold")

    _plot_style = {"marker": "o", "markersize": 2.5, "linewidth": 1.5}
    _line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    # Distinct colors so (color, linestyle) pairs are unique across all series
    _colors = list(plt.get_cmap("tab10").colors)

    def _ax_style(ax):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

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
        comp = [x * 1e-5 for x in series["Computation"]]
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
            c = _colors[idx % len(_colors)]
            ls = _line_styles[idx % len(_line_styles)]
            ax_loss.plot(iterations, series[k], label=k.replace("Mean ", ""), color=c, linestyle=ls, **_plot_style)
    ax_loss.set_title("Losses", fontsize=10, fontweight="bold")
    ax_loss.legend(loc="upper right", fontsize=8)
    _ax_style(ax_loss)
    ax_loss.set_xlabel("Iteration", fontsize=9)

    ax_pos = fig.add_subplot(gs[1, 0])
    for idx, k in enumerate(CATEGORY_REWARDS_POSITIVE):
        if k in series and series[k]:
            label = k.replace("Episode_Reward/", "")
            if label == "velocity_reward":
                label = "velocity (target +x)"
            elif label == "alternating_foot_reward":
                label = "alternating (L-R)"
            elif label == "symmetry_reward":
                label = "symmetry (L-R)"
            elif label == "step_length_reward":
                label = "step_length (x ≥ min)"
            elif label == "foot_crossing_reward":
                label = "foot_crossing (neg→pos)"
            elif label == "imitation_reward":
                label = "imitation_joints (NPZ)"
            c = _colors[idx % len(_colors)]
            ls = _line_styles[idx % len(_line_styles)]
            style = dict(**_plot_style)
            if label == "imitation_joints (NPZ)":
                style["linewidth"] = 2.0
                style["markersize"] = 4.0
            ax_pos.plot(iterations, series[k], label=label, color=c, linestyle=ls, **style)
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
                label = "drift (y, legacy)"
            elif label == "x_drift_penalty":
                label = "drift (x)"
            elif label == "y_drift_penalty":
                label = "drift (y)"
            elif label == "z_orientation_penalty":
                label = "z_orientation (yaw)"
            elif label == "same_foot_double_non_contact_penalty":
                label = "same_foot_tap (F-T-F)"
            elif label == "both_feet_non_contact_penalty":
                label = "jump (both off)"
            elif label == "symmetry_penalty":
                label = "symmetry (L-R)"
            elif label == "step_length_penalty":
                label = "step_length (x < min)"
            elif label == "foot_crossing_penalty":
                label = "foot_crossing (no cross)"
            elif label == "imitation_penalty":
                label = "imitation_penalty (NPZ)"
            elif label == "lin_vel_z_penalty":
                label = "lin_vel_z (vertical vel)"
            elif label == "ang_vel_xy_penalty":
                label = "ang_vel_xy (roll/pitch rate)"
            elif label == "flat_orientation_penalty":
                label = "flat_orientation (tilt)"
            c = _colors[idx % len(_colors)]
            ls = _line_styles[idx % len(_line_styles)]
            ax_pen.plot(iterations, plot_vals, label=label, color=c, linestyle=ls, **_plot_style)
    ax_pen.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_pen.set_title("Penalties / costs (shown negative)", fontsize=10, fontweight="bold")
    ax_pen.legend(loc="lower left", fontsize=7, ncol=2)
    _ax_style(ax_pen)
    ax_pen.set_ylabel("Penalty", fontsize=9)
    ax_pen.set_xlabel("Iteration", fontsize=9)

    ax_cfg = fig.add_subplot(gs[0:2, 2])
    smooth_jerk = parse_header_smooth_jerk(path)
    reward_cfg = parse_header_reward_config(path)
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
    if reward_cfg:
        _hdr_ijr = _joint_pos_imitation_header_weight(reward_cfg, reward=True)
        _hdr_ijp = _joint_pos_imitation_header_weight(reward_cfg, reward=False)
        lines.extend([
            "Velocity",
            f"  target_velocity: {reward_cfg.get('target_velocity', '—')}",
            f"  velocity_reward_weight: {reward_cfg.get('velocity_reward_weight', '—')}",
            f"  velocity_reward_sharpness: {reward_cfg.get('velocity_reward_sharpness', '—')}",
            "",
            "Drift / yaw",
            f"  x_drift_tol: {reward_cfg.get('x_drift_tolerance', '—')}",
            f"  x_drift_weight: {reward_cfg.get('x_drift_penalty_weight', '—')}",
            f"  y_drift_tol: {reward_cfg.get('y_drift_tolerance', '—')}",
            f"  y_drift_weight: {reward_cfg.get('y_drift_penalty_weight', '—')}",
            f"  z_orient_tol: {reward_cfg.get('z_orientation_tolerance', '—')}",
            f"  z_orient_weight: {reward_cfg.get('z_orientation_penalty_weight', '—')}",
            "",
            "Alternating foot",
            f"  alt_reward_weight: {reward_cfg.get('alternating_foot_reward_weight', '—')}",
            f"  same_foot_tap_pen_weight: {reward_cfg.get('same_foot_tap_penalty_weight', '—')}",
            f"  jump_pen_weight: {reward_cfg.get('jump_penalty_weight', '—')}",
            "",
            "Symmetry (L-R)",
            f"  sym_reward_weight: {reward_cfg.get('symmetry_reward_weight', '—')}",
            f"  sym_penalty_weight: {reward_cfg.get('symmetry_penalty_weight', '—')}",
            "",
            "Step length",
            f"  step_length_min_m: {reward_cfg.get('step_length_min_m', '—')}",
            f"  step_length_reward: {reward_cfg.get('step_length_reward_weight', '—')}",
            f"  step_length_penalty: {reward_cfg.get('step_length_penalty_weight', '—')}",
            "",
            "Foot crossing",
            f"  foot_crossing_reward: {reward_cfg.get('foot_crossing_reward_weight', '—')}",
            f"  foot_crossing_penalty: {reward_cfg.get('foot_crossing_penalty_weight', '—')}",
            "",
            "Stability (Isaac-Lab style)",
            f"  lin_vel_z_penalty: {reward_cfg.get('lin_vel_z_penalty_weight', '—')}",
            f"  lin_vel_z_tolerance: {reward_cfg.get('lin_vel_z_tolerance', '—')}",
            f"  ang_vel_xy_penalty: {reward_cfg.get('ang_vel_xy_penalty_weight', '—')}",
            f"  flat_orientation_penalty: {reward_cfg.get('flat_orientation_penalty_weight', '—')}",
            f"  flat_orientation_tolerance: {reward_cfg.get('flat_orientation_tolerance', '—')}",
            "",
            "Imitation — joint angles (NPZ frame-by-frame)",
            f"  imitation_joints_position_reward_weight: {_hdr_ijr if _hdr_ijr is not None else '—'}",
            f"  imitation_joints_position_penalty_weight: {_hdr_ijp if _hdr_ijp is not None else '—'}",
            f"  npz_imitation_loop_count: {reward_cfg.get('npz_imitation_loop_count', '—')}",
            f"  npz_imitation_time_factor: {reward_cfg.get('npz_imitation_time_factor', '—')}",
            f"  npz_imitation_position_tolerance_pct: {reward_cfg.get('npz_imitation_position_tolerance_pct', '—')}",
            f"  npz_imitation_velocity_tolerance_pct: {reward_cfg.get('npz_imitation_velocity_tolerance_pct', '—')}",
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
        description="Print training_progress_log.txt for g1_locomotion_walking_v2 and generate G1 Walking V2 visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/g1_locomotion_walking_v2/2026-03-15_12-00-00",
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

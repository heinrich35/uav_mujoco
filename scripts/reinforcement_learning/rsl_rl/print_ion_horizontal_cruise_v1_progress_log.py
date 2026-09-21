#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Print training_progress_log.txt from an RSL-RL run directory for Ion-Horizontal-Cruise-V1
# and optionally generate a visual analysis (training_progress_analysis.png) grouping metrics by category.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_ion_horizontal_cruise_v1_progress_log.py <log_dir>
#   python scripts/reinforcement_learning/rsl_rl/print_ion_horizontal_cruise_v1_progress_log.py logs/rsl_rl/ion_horizontal_cruise_v1/2026-03-12_10-00-00
#
# Options:
#   --last N       Print only the last N iteration blocks (default: all)
#   --no-plot      Do not generate the visual analysis PNG
#   --output NAME  Output filename for the plot (default: training_progress_analysis.png)

from __future__ import annotations

import argparse
import os
import re
import sys


# Strip ANSI escape codes (e.g. [1m, [0m)
def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _parse_block(block: str) -> tuple[int | None, dict[str, float]]:
    """Parse one iteration block. Returns (iteration_number, {metric_name: value})."""
    block = _strip_ansi(block)
    out = {}
    it_match = re.search(r"Learning iteration\s+(\d+)/", block)
    iteration = int(it_match.group(1)) if it_match else None
    # Match "Label: number" (allow negative and decimals)
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

    # Build series per key (use None for missing)
    series = {k: [] for k in all_keys}
    for data in block_data:
        for k in all_keys:
            series[k].append(data.get(k))

    # Convert None to last known or 0 for numeric continuity
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


# Metric groups for the visual analysis (Ion-Horizontal-Cruise-V1: same as ion_vertical_takeoff_v1)
CATEGORY_MAIN = [
    "Mean reward",
    "Mean episode length",
]
CATEGORY_LOSSES = [
    "Mean value_function loss",
    "Mean surrogate loss",
    "Mean entropy loss",
]
CATEGORY_REWARDS_POSITIVE = [
    "Episode_Reward/positive_z_velocity",
    "Episode_Reward/upward_shaping",
    "Episode_Reward/orientation_reward",
    "Episode_Reward/airborne",
    "Episode_Reward/z_position",
    "Episode_Reward/positive_x_velocity",
    "Episode_Reward/forward_shaping",
]
CATEGORY_REWARDS_PENALTIES = [
    "Episode_Reward/off_band_z_velocity",
    "Episode_Reward/off_band_x_velocity",
    "Episode_Reward/idle",
    "Episode_Reward/x_drift",
    "Episode_Reward/y_drift",
    "Episode_Reward/jerky",
    "Episode_Reward/electricity",
    "Episode_Reward/orientation_penalty_xy",
    "Episode_Reward/orientation_penalty_z",
    "Episode_Reward/low_altitude",
]
CATEGORY_TERMINATION = [
    "Episode_Termination/fall",
    "Episode_Termination/time_out",
]


def generate_visual_analysis(
    log_dir: str,
    output_name: str = "training_progress_analysis.png",
    *,
    plot_title: str | None = None,
) -> str | None:
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
    fig.suptitle(
        plot_title or "Ion-Horizontal-Cruise-V1 — Training progress (from training_progress_log.txt)",
        fontsize=12,
    )

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

    # Rewards (penalties)
    ax = axes[1, 1]
    for k in CATEGORY_REWARDS_PENALTIES:
        if k in series and series[k]:
            ax.plot(iterations, series[k], label=k.replace("Episode_Reward/", ""), marker="o", markersize=3)
    ax.set_title("Rewards (penalties)")
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

    # Hide unused subplot
    axes[2, 1].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(log_dir, output_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Print training_progress_log.txt for Ion-Horizontal-Cruise-V1 and optionally generate visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/ion_horizontal_cruise_v1/2026-03-12_10-00-00",
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
        print("Training will write it every --progress_log_interval iterations (default 100).", file=sys.stderr)
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

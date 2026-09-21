#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Print training_progress_log.txt from an RSL-RL run directory and optionally
# generate a visual analysis (training_progress_analysis.png) grouping metrics by category.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_training_progress_log.py <log_dir>
#   python scripts/reinforcement_learning/rsl_rl/print_training_progress_log.py logs/rsl_rl/ion_vertical_takeoff_v1/2026-03-12_09-19-25
#
# Options:
#   --last N       Print only the last N iteration blocks (default: all)
#   --no-plot      Do not generate the visual analysis PNG
#   --output NAME  Output filename for the plot (default: training_progress_analysis.png)

from __future__ import annotations

import argparse
import os
import re
import subprocess
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


# Mapping from Episode_Reward/name to run argument key in header (ion_vertical_takeoff_v1)
REWARD_TO_HEADER_KEY = {
    "positive_z_velocity": "positive_z_velocity_reward_weight",
    "upward_shaping": "upward_velocity_shaping_weight",
    "orientation_reward": "orientation_reward_weight",
    "airborne": "airborne_reward_weight",
    "off_band_z_velocity": "off_band_z_velocity_penalty_weight",
    "idle": "idle_velocity_penalty_weight",
    "x_drift": "x_drift_penalty",
    "y_drift": "y_drift_penalty",
    "jerky": "jerky_penalty",
    "electricity": "electricity_cost_penalty",
    "orientation_penalty_xy": "xy_orientation_penalty_weight",
    "orientation_penalty_z": "z_orientation_penalty_weight",
    "low_altitude": "low_altitude_penalty_weight",
}


def parse_run_arguments(path: str) -> dict[str, str | float]:
    """Parse Run arguments block from training_progress_log.txt. Returns {key: value}."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    out = {}
    in_header = False
    for line in content.split("\n"):
        line = _strip_ansi(line)
        if "Run arguments (this run)" in line:
            in_header = True
            continue
        if in_header:
            if line.strip().startswith("#####"):
                if len(out) > 0:
                    break
                continue
            m = re.match(r"^\s*([a-z0-9_]+)\s*:\s*(.+)$", line)
            if m:
                key = m.group(1).strip()
                raw = m.group(2).strip()
                try:
                    if raw.lower() in ("true", "false"):
                        out[key] = raw
                    elif "." in raw and raw.replace(".", "").replace("-", "").isdigit():
                        out[key] = float(raw)
                    elif raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
                        out[key] = int(raw)
                    else:
                        out[key] = raw
                except ValueError:
                    out[key] = raw
    return out


# Metric groups for the visual analysis (same path as log)
CATEGORY_MAIN = [
    "Mean reward",
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
]
CATEGORY_REWARDS_PENALTIES = [
    "Episode_Reward/off_band_z_velocity",
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


def _read_training_log_head(log_txt_path: str, nbytes: int = 32768) -> str:
    try:
        with open(log_txt_path, encoding="utf-8", errors="replace") as f:
            return f.read(nbytes)
    except OSError:
        return ""


def _training_log_is_rover_idle_to_forward(log_txt_path: str, head: str | None = None) -> bool:
    """Detect Idle → Forward rover runs for the dedicated plotter."""
    if head is None:
        head = _read_training_log_head(log_txt_path)
    if not head:
        return False
    if "Isaac-Rover-DiskWorld-IdleToForward" in head:
        return True
    if re.search(r"experiment_name:\s*rover_disk_world_idle_to_forward", head):
        return True
    return False


def _training_log_is_rover_localization(log_txt_path: str, head: str | None = None) -> bool:
    """Detect Rover Disk-World Localization runs for the dedicated plotter."""
    if head is None:
        head = _read_training_log_head(log_txt_path)
    if not head:
        return False
    if "Isaac-Rover-DiskWorld-Localization" in head:
        return True
    if re.search(r"experiment_name:\s*rover_disk_world_localization", head):
        return True
    return False


def _training_log_is_rover_flat_goal(log_txt_path: str, head: str | None = None) -> bool:
    """Detect Rover FlatWorld-GoalNav runs for the dedicated plotter."""
    if head is None:
        head = _read_training_log_head(log_txt_path)
    if not head:
        return False
    if "Isaac-Rover-FlatWorld-GoalNav" in head:
        return True
    if re.search(r"experiment_name:\s*rover_(flat_world|70_70)_goal_nav", head):
        return True
    if "Episode_Reward/four_wheels_ground_contact_reward" in head:
        return True
    return False


def _training_log_is_rover_disk_world(log_txt_path: str) -> bool:
    """Detect Rover Disk-World runs so we delegate to the specialized plotter (incl. physparam merge)."""
    head = _read_training_log_head(log_txt_path)
    if not head:
        return False
    if _training_log_is_rover_idle_to_forward(log_txt_path, head=head):
        return False
    if _training_log_is_rover_localization(log_txt_path, head=head):
        return False
    if "Isaac-Rover-DiskWorld" in head:
        return True
    if re.search(r"experiment_name:\s*rover_disk_world", head):
        return True
    if "Episode_RoverPhysparam/" in head:
        return True
    return False


def _run_rover_idle_to_forward_progress_plot(log_dir: str, output_name: str) -> str | None:
    """Invoke print_rover_idle_to_forward_progress_log.py --plot-only."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print_rover_idle_to_forward_progress_log.py")
    if not os.path.isfile(script):
        return None
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script))))
    proc = subprocess.run(
        [sys.executable, script, log_dir, "--plot-only", "--output", output_name],
        cwd=workspace_root,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    out_path = os.path.join(log_dir, output_name)
    if proc.returncode == 0 and os.path.isfile(out_path):
        return out_path
    return None


def _run_rover_localization_progress_plot(log_dir: str, output_name: str) -> str | None:
    """Invoke print_rover_localization_progress_log.py --plot-only."""
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "print_rover_localization_progress_log.py"
    )
    if not os.path.isfile(script):
        return None
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script))))
    proc = subprocess.run(
        [sys.executable, script, log_dir, "--plot-only", "--output", output_name],
        cwd=workspace_root,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    out_path = os.path.join(log_dir, output_name)
    if proc.returncode == 0 and os.path.isfile(out_path):
        return out_path
    return None


def _run_rover_flat_goal_progress_plot(log_dir: str, output_name: str) -> str | None:
    """Invoke print_rover_flat_goal_progress_log.py --plot-only."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print_rover_flat_goal_progress_log.py")
    if not os.path.isfile(script):
        return None
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script))))
    proc = subprocess.run(
        [sys.executable, script, log_dir, "--plot-only", "--output", output_name],
        cwd=workspace_root,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    out_path = os.path.join(log_dir, output_name)
    if proc.returncode == 0 and os.path.isfile(out_path):
        return out_path
    return None


def _run_rover_disk_world_progress_plot(log_dir: str, output_name: str) -> str | None:
    """Invoke print_rover_disk_world_progress_log.py --plot-only. Returns output path or None."""
    rover_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print_rover_disk_world_progress_log.py")
    if not os.path.isfile(rover_script):
        return None
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(rover_script))))
    proc = subprocess.run(
        [sys.executable, rover_script, log_dir, "--plot-only", "--output", output_name],
        cwd=workspace_root,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    out_path = os.path.join(log_dir, output_name)
    if proc.returncode == 0 and os.path.isfile(out_path):
        return out_path
    return None


def generate_visual_analysis(log_dir: str, output_name: str = "training_progress_analysis.png") -> str | None:
    """Generate training_progress_analysis.png in log_dir. Returns path or None on failure."""
    path = os.path.join(log_dir, "training_progress_log.txt")
    if not os.path.isfile(path):
        return None
    if _training_log_is_rover_idle_to_forward(path):
        idle_out = _run_rover_idle_to_forward_progress_plot(log_dir, output_name)
        if idle_out:
            return idle_out
    if _training_log_is_rover_localization(path):
        loc_out = _run_rover_localization_progress_plot(log_dir, output_name)
        if loc_out:
            return loc_out
    if _training_log_is_rover_flat_goal(path):
        flat_out = _run_rover_flat_goal_progress_plot(log_dir, output_name)
        if flat_out:
            return flat_out
    if _training_log_is_rover_disk_world(path):
        rover_out = _run_rover_disk_world_progress_plot(log_dir, output_name)
        if rover_out:
            return rover_out
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
    fig.suptitle("Training progress (from training_progress_log.txt)", fontsize=12)

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

    # Reward arguments box (bottom-right) with weights from run header
    ax = axes[2, 1]
    ax.axis("off")
    run_args = parse_run_arguments(path)

    def _fmt(name: str) -> str:
        header_key = REWARD_TO_HEADER_KEY.get(name)
        if header_key and header_key in run_args:
            v = run_args[header_key]
            if isinstance(v, float) and v == int(v):
                v = int(v)
            return f"{name}: {v}"
        return name

    positive = [k.replace("Episode_Reward/", "") for k in CATEGORY_REWARDS_POSITIVE if k in series]
    penalties = [k.replace("Episode_Reward/", "") for k in CATEGORY_REWARDS_PENALTIES if k in series]
    termination = [k.replace("Episode_Termination/", "") for k in CATEGORY_TERMINATION if k in series]
    lines = ["Reward arguments (weights from run)", ""]
    if positive:
        lines.append("Positive:")
        lines.extend("  " + _fmt(n) for n in positive)
    if penalties:
        lines.append("Penalties:")
        lines.extend("  " + _fmt(n) for n in penalties)
    if termination:
        lines.append("Termination: " + ", ".join(termination))
    ax.text(
        0.5, 0.5, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="center",
        horizontalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        wrap=True,
    )

    plt.tight_layout()
    out_path = os.path.join(log_dir, output_name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Print training_progress_log.txt and optionally generate visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/ion_vertical_takeoff_v1/2026-03-12_09-19-25",
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

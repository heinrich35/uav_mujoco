#!/usr/bin/env python3
"""Training-progress analysis for the Rover Localization task.

Task: Isaac-Rover-Localization-v0

Parses ``training_progress_log.txt`` written by RSL-RL / Isaac Lab wrappers and
generates a PNG with localization-themed panels.

Usage
-----
  python scripts/reinforcement_learning/rsl_rl/print_rover_localization_progress_log.py <log_dir>
  python scripts/reinforcement_learning/rsl_rl/print_rover_localization_progress_log.py \
      logs/rsl_rl/rover_localization_env/2026-05-12_12-00-00

Options
-------
  --last N            Print only the last N iteration blocks (default: all)
  --no-plot           Do not generate the visual analysis PNG
  --plot-only         Only generate the visual analysis PNG (skip text print)
  --output NAME       Output filename (default: training_progress_analysis.png)
  --reward-ylim-after ITER
                      Focus y-scale on iterations >= ITER for reward panels.
                      Default 100. Use -1 for full autoscale.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence


# ---------------------------------------------------------------------------
# ANSI / log parsing helpers
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _pre_iteration_content(log_path: str) -> str:
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    return content.split("Learning iteration", 1)[0]


def _parse_block(block: str) -> tuple[int | None, dict[str, float]]:
    """Parse one RSL-RL log iteration block -> (iteration, {key: value})."""
    block = _strip_ansi(block)
    out: dict[str, float] = {}
    it_match = re.search(r"Learning iteration\s+(\d+)/", block)
    iteration = int(it_match.group(1)) if it_match else None
    for m in re.finditer(r"^([^:]+?):\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*$", block, re.MULTILINE):
        key = m.group(1).strip()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    _merge_rsl_rl_metric_aliases(out)
    return iteration, out


_RSL_RL_TO_PLOT: tuple[tuple[str, str], ...] = (
    ("Mean reward", "Train/mean_reward"),
    ("Mean episode length", "Train/mean_episode_length"),
    ("Mean value_function loss", "Loss/value_function"),
    ("Mean surrogate loss", "Loss/surrogate"),
    ("Mean entropy loss", "Loss/entropy"),
    ("Mean action noise std", "Policy/mean_noise_std"),
)


def _merge_rsl_rl_metric_aliases(d: dict[str, float]) -> None:
    for src, dst in _RSL_RL_TO_PLOT:
        if src in d and dst not in d:
            d[dst] = d[src]


def parse_log(log_path: str) -> list[tuple[int, dict[str, float]]]:
    """Return list of (iteration, metrics_dict) from training_progress_log.txt."""
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    raw_blocks = re.split(r"(?=Learning iteration\s+\d+/)", content)
    results: list[tuple[int, dict[str, float]]] = []
    for block in raw_blocks:
        if "Learning iteration" not in block:
            continue
        it, data = _parse_block(block)
        if it is not None:
            results.append((it, data))
    results.sort(key=lambda x: x[0])
    return results


def parse_header_config(log_path: str) -> dict[str, str]:
    """Extract rover localization config from the log header."""
    try:
        head = _pre_iteration_content(log_path)
    except OSError:
        return {}
    cfg: dict[str, str] = {}
    keys = [
        # Run
        "task",
        "experiment_name",
        "num_envs",
        "max_iterations",
        "seed",
        "device",
        # Scene geometry
        "plane_half_size_m",
        "spawn_half_size_m",
        "target_half_size_m",
        "target_exclusion_half_size_m",
        "target_min_spawn_dist_m",
        "spawn_z_m",
        "episode_length_s",
        # Localization reward params
        "localization_position_error_scale_m",
        "localization_position_error_exp_scale",
        "localization_position_error_linear_weight",
        "localization_position_error_exp_weight",
        "localization_position_bonus_threshold_m",
        "localization_position_bonus_weight",
        "localization_alive_weight",
        "localization_distance_scale_m",
        # Obstacles
        "oa_flat_box_count",
        "oa_flat_box_min_pair_clear_m",
        # PPO
        "ppo_learning_rate",
        "ppo_entropy_coef",
        "ppo_init_noise_std",
        "ppo_clip_param",
        "ppo_desired_kl",
        "ppo_num_steps_per_env",
        "ppo_num_learning_epochs",
        "ppo_num_mini_batches",
        "ppo_gamma",
        "ppo_lam",
    ]
    for k in keys:
        m = re.search(rf"^\s*{re.escape(k)}\s*:\s*(.+)$", head, re.MULTILINE)
        if m:
            cfg[k] = m.group(1).strip()
    return cfg


# ---------------------------------------------------------------------------
# Text printing
# ---------------------------------------------------------------------------

_REW_PREFIX = "Episode_Reward/"
_TERM_PREFIX = "Episode_Termination/"


def print_blocks(blocks: list[tuple[int, dict[str, float]]], last: int | None = None) -> None:
    if last is not None:
        blocks = blocks[-last:]
    for it, data in blocks:
        print(f"\n{'-'*60}")
        print(f"  Iteration {it}")
        print(f"{'-'*60}")

        for k in ("Train/mean_reward", "Train/mean_episode_length", "Perf/FPS"):
            if k in data:
                print(f"  {k:<40s} {data[k]:>10.4f}")

        train_keys = sorted(
            k for k in data
            if k.startswith("Train/") and k not in {"Train/mean_reward", "Train/mean_episode_length"}
        )
        if train_keys:
            print("  -- Train --")
        for k in train_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        rew_keys = sorted(k for k in data if k.startswith(_REW_PREFIX))
        if rew_keys:
            print("  -- Rewards --")
        for k in rew_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        term_keys = sorted(k for k in data if k.startswith(_TERM_PREFIX))
        if term_keys:
            print("  -- Terminations --")
        for k in term_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        loss_keys = sorted(k for k in data if k.startswith("Loss/"))
        if loss_keys:
            print("  -- Losses --")
        for k in loss_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        policy_keys = sorted(k for k in data if k.startswith("Policy/"))
        if policy_keys:
            print("  -- Policy --")
        for k in policy_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        perf_keys = sorted(k for k in data if k.startswith("Perf/"))
        if perf_keys:
            print("  -- Performance --")
        for k in perf_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _series(
    blocks: list[tuple[int, dict[str, float]]],
    key: str,
) -> tuple[list[int], list[float]]:
    its, vals = [], []
    for it, data in blocks:
        if key in data:
            its.append(it)
            vals.append(data[key])
    return its, vals


def _plot_line(ax, blocks, key: str, label: str, color=None, linestyle="-", alpha=1.0):
    its, vals = _series(blocks, key)
    if its:
        kw = dict(label=label, linestyle=linestyle, alpha=alpha)
        if color:
            kw["color"] = color
        ax.plot(its, vals, **kw)
    return bool(its)


def _legend_if_any(ax, **kwargs) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _focus_ylim(ax, blocks, keys: Sequence[str], after_iter: int):
    vals = []
    for it, data in blocks:
        if it >= after_iter:
            for k in keys:
                if k in data:
                    vals.append(data[k])
    if vals:
        lo, hi = min(vals), max(vals)
        pad = max(abs(hi - lo) * 0.15, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)


def generate_plot(
    blocks: list[tuple[int, dict[str, float]]],
    cfg: dict[str, str],
    output_path: str,
    reward_ylim_after: int = 100,
) -> None:
    """Write one combined PNG: 4 themed panels + config sidebar."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed - skipping plot generation.")
        return

    if not blocks:
        print("[WARN] No iteration data found - cannot generate plot.")
        return

    max_iter = blocks[-1][0]
    last_data = blocks[-1][1]
    n_main = 4
    fig_h = 16.0

    fig = plt.figure(figsize=(20, fig_h))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4.2, 1.0], wspace=0.04)
    gs_panels = gridspec.GridSpecFromSubplotSpec(n_main, 1, subplot_spec=gs_outer[0], hspace=0.40)

    ax1 = fig.add_subplot(gs_panels[0])
    ax2 = fig.add_subplot(gs_panels[1])
    ax3 = fig.add_subplot(gs_panels[2])
    ax4 = fig.add_subplot(gs_panels[3])

    title = (
        f"Rover Localization (expert drives, agent estimates 3-D position) — Progress  "
        f"(iter {blocks[0][0]}-{max_iter})"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    # ---- Panel 1: OVERVIEW ----
    ax1.set_title("Overview", fontsize=10, loc="left")
    _plot_line(ax1, blocks, "Train/mean_reward", "Mean reward", color="#1f77b4")
    ax1.set_ylabel("Mean reward", color="#1f77b4")
    ax1_r = ax1.twinx()
    _plot_line(
        ax1_r, blocks, "Train/mean_episode_length", "Episode length (steps)",
        color="#ff7f0e", linestyle="--",
    )
    ax1_r.set_ylabel("Episode length (steps)", color="#ff7f0e")
    _legend_if_any(ax1, loc="upper left", fontsize=8)
    _legend_if_any(ax1_r, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: LOCALIZATION REWARDS ----
    ax2.set_title("Localization rewards (position estimate accuracy)", fontsize=10, loc="left")
    _R = _REW_PREFIX
    _plot_line(ax2, blocks, _R + "position_estimation_reward", "estimation_reward (smooth)", color="#1f77b4")
    _plot_line(ax2, blocks, _R + "position_estimation_bonus", "accuracy_bonus (sparse)", color="#2ca02c", linestyle="--")
    _plot_line(ax2, blocks, _R + "alive_bonus", "alive_bonus", color="#ff7f0e", linestyle="-.")
    ax2.set_ylabel("Episode reward contribution")
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    _legend_if_any(ax2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    if reward_ylim_after >= 0:
        _focus_ylim(
            ax2, blocks,
            [_R + k for k in (
                "position_estimation_reward",
                "position_estimation_bonus",
                "alive_bonus",
            )],
            reward_ylim_after,
        )

    # ---- Panel 3: TERMINATIONS ----
    ax3.set_title("Episode Termination Rates", fontsize=10, loc="left")
    _T = _TERM_PREFIX
    _plot_line(ax3, blocks, _T + "time_out", "time_out (truncated)", color="#7f7f7f", linestyle=":")
    _plot_line(ax3, blocks, _T + "upside_down", "upside_down (tilt failure)", color="#d62728", linestyle="--")
    _plot_line(ax3, blocks, _T + "wheelie", "wheelie (airborne + nose-up)", color="#9467bd", linestyle="-")
    ax3.set_ylabel("Rate [0-1]")
    ax3.set_ylim(-0.02, 1.05)
    ax3.axhline(0, color="gray", linewidth=0.5)
    _legend_if_any(ax3, fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: POLICY / LOSSES ----
    ax4.set_title("Policy losses", fontsize=10, loc="left")
    _plot_line(ax4, blocks, "Loss/value_function", "value_function loss", color="#1f77b4")
    _plot_line(ax4, blocks, "Loss/surrogate", "surrogate loss", color="#ff7f0e", linestyle="--")
    _plot_line(ax4, blocks, "Loss/entropy", "entropy", color="#2ca02c", linestyle="-.")
    ax4.set_ylabel("Loss value")
    ax4.set_xlabel("Training iteration")
    _legend_if_any(ax4, fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4_r = ax4.twinx()
    _plot_line(ax4_r, blocks, "Policy/mean_noise_std", "mean action noise std", color="#9467bd", linestyle=":")
    ax4_r.set_ylabel("Noise std", color="#9467bd")
    _legend_if_any(ax4_r, fontsize=8, loc="center right")

    # ---- Config sidebar ----
    ax_cfg = fig.add_subplot(gs_outer[1])
    ax_cfg.axis("off")

    def _v(k: str, default: str = "-") -> str:
        return cfg.get(k, default)

    def _fmt_latest(key: str, fmt: str = ".4f") -> str:
        if key not in last_data:
            return "-"
        return format(last_data[key], fmt)

    sidebar_lines = [
        "== Run ==",
        f"  experiment   : {_v('experiment_name')}",
        f"  task         : {_v('task')}",
        "",
        "== Scene / Episode ==",
        f"  plane_half_m : {_v('plane_half_size_m')}",
        f"  spawn_half_m : {_v('spawn_half_size_m')}",
        f"  target_half_m: {_v('target_half_size_m')}",
        f"  target_excl_h: {_v('target_exclusion_half_size_m')}",
        f"  min_spawn_d  : {_v('target_min_spawn_dist_m')} m",
        f"  spawn_z_m    : {_v('spawn_z_m')}  (rover root Z)",
        f"  episode_s    : {_v('episode_length_s')} s",
        f"  blue_boxes   : {_v('oa_flat_box_count')}  (clearance {_v('oa_flat_box_min_pair_clear_m')} m)",
        "",
        "== Localization Reward ==",
        f"  error_scale_m: {_v('localization_position_error_scale_m')}",
        f"  exp_scale    : {_v('localization_position_error_exp_scale')}",
        f"  linear_wt    : {_v('localization_position_error_linear_weight')}",
        f"  exp_wt       : {_v('localization_position_error_exp_weight')}",
        f"  bonus_thr_m  : {_v('localization_position_bonus_threshold_m')}",
        f"  bonus_wt     : {_v('localization_position_bonus_weight')}",
        f"  alive_wt     : {_v('localization_alive_weight')}",
        f"  dist_scale_m : {_v('localization_distance_scale_m')}",
        "",
        "== Training ==",
        f"  num_envs     : {_v('num_envs')}",
        f"  max_iter     : {_v('max_iterations')}",
        f"  seed         : {_v('seed')}",
        f"  device       : {_v('device')}",
        "",
        "== PPO ==",
        f"  lr           : {_v('ppo_learning_rate')}",
        f"  entropy      : {_v('ppo_entropy_coef')}",
        f"  noise_std    : {_v('ppo_init_noise_std')}",
        f"  clip         : {_v('ppo_clip_param')}",
        f"  kl_target    : {_v('ppo_desired_kl')}",
        f"  steps/env    : {_v('ppo_num_steps_per_env')}",
        f"  epochs       : {_v('ppo_num_learning_epochs')}",
        f"  mini_batches : {_v('ppo_num_mini_batches')}",
        f"  gamma        : {_v('ppo_gamma')}",
        f"  lam          : {_v('ppo_lam')}",
        "",
        "== Latest Train Values ==",
        f"  mean_reward  : {_fmt_latest('Train/mean_reward', '.2f')}",
        f"  ep_len       : {_fmt_latest('Train/mean_episode_length', '.2f')}",
        f"  vf_loss      : {_fmt_latest('Loss/value_function', '.4f')}",
        f"  surr_loss    : {_fmt_latest('Loss/surrogate', '.4f')}",
        f"  entropy_loss : {_fmt_latest('Loss/entropy', '.4f')}",
        f"  noise_std    : {_fmt_latest('Policy/mean_noise_std', '.4f')}",
        f"  steps_per_s  : {_fmt_latest('Perf/env_steps_per_sec', '.1f')}",
        f"  iter_wall_s  : {_fmt_latest('Perf/iteration_wall_s', '.3f')}",
        "",
        f"== Iterations logged: {len(blocks)} ==",
        f"  {blocks[0][0]}  ->  {max_iter}",
    ]

    ax_cfg.text(
        0.04, 0.97, "\n".join(sidebar_lines),
        transform=ax_cfg.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse Rover Localization training progress from training_progress_log.txt.",
    )
    parser.add_argument("log_dir", help="Path to the RSL-RL log directory.")
    parser.add_argument("--last", type=int, default=None, metavar="N",
                        help="Print only the last N iteration blocks.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not generate the visual analysis PNG.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate the PNG (skip text print).")
    parser.add_argument("--output", type=str, default="training_progress_analysis.png",
                        help="Output PNG filename.")
    parser.add_argument("--reward-ylim-after", type=int, default=100, metavar="ITER",
                        help="Focus reward y-axis on iterations >= ITER (default 100; -1 = full auto).")
    args = parser.parse_args()

    log_path = os.path.join(args.log_dir, "training_progress_log.txt")
    if not os.path.isfile(log_path):
        print(f"[ERROR] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    blocks = parse_log(log_path)
    if not blocks:
        print("[WARN] No iteration blocks parsed from log.", file=sys.stderr)
        sys.exit(1)

    if not args.plot_only:
        print_blocks(blocks, last=args.last)

    if not args.no_plot:
        cfg = parse_header_config(log_path)
        output_path = os.path.join(args.log_dir, args.output)
        generate_plot(
            blocks,
            cfg,
            output_path,
            reward_ylim_after=args.reward_ylim_after,
        )


if __name__ == "__main__":
    main()

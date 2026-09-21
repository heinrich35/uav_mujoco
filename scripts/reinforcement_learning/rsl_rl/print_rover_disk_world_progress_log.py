#!/usr/bin/env python3
"""Training-progress analysis for the Rover Disk-World tasks.

Tasks: Isaac-Rover-DiskWorld-Forward-v0, Isaac-Rover-DiskWorld-Physparam-v0

Parses ``training_progress_log.txt`` written by RSL-RL and generates:

* ``training_progress_analysis.png`` — five themed panels (overview, rewards,
  penalties + jerk/heading smoothness bonus pairs, terminations, policy losses)
  plus a **config sidebar** (parses ``rover_*_weight`` CLI lines from the log header).

* When the log contains ``Episode_RoverPhysparam/*`` (physparam task), **two
  additional panels** (latched physics means) are appended to the same PNG
  so training and physics curves ship in one file.

* Optional ``--split-physparam`` also writes a standalone physparam-only PNG
  (legacy layout).

A config sidebar summarises all reward weights and termination parameters
extracted from the log header so you can cross-reference hyperparameter
choices with the resulting curves.

Usage
-----
  python scripts/reinforcement_learning/rsl_rl/print_rover_disk_world_progress_log.py <log_dir>
  python scripts/reinforcement_learning/rsl_rl/print_rover_disk_world_progress_log.py logs/rsl_rl/rover_disk_world_forward/2026-04-22_12-00-00

Options
-------
  --last N            Print only the last N iteration blocks (default: all)
  --no-plot           Do not generate the visual analysis PNG
  --plot-only         Only generate the visual analysis PNG (skip text print)
  --output NAME       Output filename (default: training_progress_analysis.png)
  --split-physparam   Also write a separate physparam-only PNG (see
                      --physparam-output).
  --physparam-output NAME
                      Filename for the split physparam PNG when --split-physparam
                      is set (default: training_progress_physparam.png).
  --reward-ylim-after ITER
                      Focus y-scale on iterations ≥ ITER for reward panels.
                      Default 200. Use -1 for full autoscale.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence


def _physparam_latch_sidebar_lines() -> list[str]:
    """Sidebar text uses mirrored limits (no ``rover_utils`` import — avoids Isaac/pxr in plot-only runs)."""
    from rover_physparam_log_docs import physparam_latch_doc_lines

    return ["", "══ Physparam latch p0–p9 ══", *physparam_latch_doc_lines(None)]


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
    """Parse one RSL-RL log iteration block → (iteration, {key: value})."""
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


# RSL-RL human-readable lines use "Mean …"; rover plots use tensorboard-style keys.
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
    """Extract rover reward/termination config from the log header."""
    try:
        head = _pre_iteration_content(log_path)
    except OSError:
        return {}
    cfg: dict[str, str] = {}
    keys = [
        "task",
        "experiment_name",
        "rover_forward_progress_weight",
        "rover_speed_tracking_weight",
        "rover_speed_target",
        "rover_heading_weight",
        "rover_alive_weight",
        "rover_tilt_weight",
        "rover_action_rate_weight",
        "rover_motion_jerk_weight",
        "rover_heading_oscillation_weight",
        "rover_motion_jerk_bonus_weight",
        "rover_heading_stability_bonus_weight",
        "rover_motion_jerk_bonus_exp_scale",
        "rover_heading_stability_bonus_exp_scale",
        "rover_goal_bonus_weight",
        "rover_radial_z_alignment_reward_weight",
        "rover_radial_z_alignment_penalty_weight",
        "rover_goal_rad",
        "rover_fall_radius",
        "rover_episode_length_s",
        "num_envs",
        "max_iterations",
        "seed",
        "device",
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

def print_blocks(blocks: list[tuple[int, dict[str, float]]], last: int | None = None) -> None:
    if last is not None:
        blocks = blocks[-last:]
    for it, data in blocks:
        print(f"\n{'─'*60}")
        print(f"  Iteration {it}")
        print(f"{'─'*60}")

        # Overview
        for k in ("Train/mean_reward", "Train/mean_episode_length", "Perf/FPS"):
            if k in data:
                print(f"  {k:<40s} {data[k]:>10.4f}")

        # Reward terms
        rew_keys = sorted(k for k in data if k.startswith("Episode_Reward/"))
        if rew_keys:
            print("  -- Rewards --")
        for k in rew_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        # Termination rates
        term_keys = sorted(k for k in data if k.startswith("Episode_Termination/"))
        if term_keys:
            print("  -- Terminations --")
        for k in term_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        # Latched physics (physparam task)
        phys_keys = sorted(k for k in data if k.startswith("Episode_RoverPhysparam/"))
        if phys_keys:
            print("  -- Latched physics (means) --")
        for k in phys_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")

        # Losses
        loss_keys = sorted(k for k in data if k.startswith("Loss/"))
        if loss_keys:
            print("  -- Losses --")
        for k in loss_keys:
            print(f"  {k:<40s} {data[k]:>10.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _series(
    blocks: list[tuple[int, dict[str, float]]],
    key: str,
) -> tuple[list[int], list[float]]:
    """Extract (iterations, values) for a metric key."""
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
    """Set y-limits focused on data after `after_iter`."""
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


_PHY_PREFIX = "Episode_RoverPhysparam/"


def _physparam_series_specs() -> tuple[tuple[str, str, str, str], ...]:
    """(log suffix after prefix, legend label, color, matplotlib linestyle)."""
    return (
        ("mean_base_mass_scale", "p0 base_mass (G/T)", "#1f77b4", "-"),
        ("mean_wheel_mass_scale", "p1 wheel_mass (Y/H)", "#ff7f0e", "--"),
        ("mean_mu_s", "p2 μ_s (1/2 J/K)", "#2ca02c", "-."),
        ("mean_mu_d", "p3 μ_d (3/4 N/M)", "#d62728", ":"),
        ("mean_gravity_scale", "p4 gravity_scale", "#9467bd", "-"),
        ("mean_forward_cmd_scale", "p5 fwd_cmd (Q/E)", "#8c564b", "--"),
        ("mean_turn_cmd_scale", "p6 turn_cmd (F/R)", "#e377c2", "-."),
        ("mean_wheel_target_slew", "p7 ω slew (V/B)", "#17becf", ":"),
        ("mean_wheel_viscous_scale", "p8 viscous ×USD", "#7f7f7f", "-"),
        ("mean_wheel_damping_scale", "p9 damping ×init", "#bcbd22", "--"),
    )


def blocks_have_physparam_metrics(blocks: list[tuple[int, dict[str, float]]]) -> bool:
    for _, data in blocks:
        if any(k.startswith(_PHY_PREFIX) for k in data):
            return True
    return False


def _draw_physparam_panels(
    ax_top,
    ax_bot,
    blocks: list[tuple[int, dict[str, float]]],
) -> None:
    """Draw latched-physics curves on two axes (body/contact vs drive/actuator)."""
    specs = _physparam_series_specs()
    body_suffixes = frozenset(s[0] for s in specs[:5])
    for suf, lab, col, ls in specs:
        key = _PHY_PREFIX + suf
        ax = ax_top if suf in body_suffixes else ax_bot
        _plot_line(ax, blocks, key, lab, color=col, linestyle=ls)
    ax_top.set_title("Latched physics — body mass, friction & gravity", fontsize=10, loc="left")
    ax_top.set_ylabel("Mean value")
    _legend_if_any(ax_top, fontsize=7, ncol=2, loc="upper right")
    ax_top.grid(True, alpha=0.3)
    ax_bot.set_title("Latched physics — drive scales, ω slew & wheel extras", fontsize=10, loc="left")
    ax_bot.set_ylabel("Mean value")
    ax_bot.set_xlabel("Training iteration")
    _legend_if_any(ax_bot, fontsize=7, ncol=2, loc="upper right")
    ax_bot.grid(True, alpha=0.3)


def generate_physparam_training_plot(
    blocks: list[tuple[int, dict[str, float]]],
    output_path: str,
) -> bool:
    """Write a standalone two-panel physparam figure (--split-physparam). Returns True if saved."""
    if not blocks or not blocks_have_physparam_metrics(blocks):
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed – skipping physparam plot.")
        return False

    max_iter = blocks[-1][0]
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.28}
    )
    fig.suptitle(
        f"Latched policy physics — training means  (iter {blocks[0][0]}–{max_iter})",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    _draw_physparam_panels(ax_top, ax_bot, blocks)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Split physparam plot saved to: {output_path}")
    return True


def generate_plot(
    blocks: list[tuple[int, dict[str, float]]],
    cfg: dict[str, str],
    output_path: str,
    reward_ylim_after: int = 200,
    physparam_split_output_path: str | None = None,
) -> None:
    """Write one combined PNG: 5 training panels (+ 2 physparam panels when data exist) + sidebar."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not installed – skipping plot generation.")
        return

    if not blocks:
        print("[WARN] No iteration data found – cannot generate plot.")
        return

    max_iter = blocks[-1][0]
    has_phy = blocks_have_physparam_metrics(blocks)
    n_main = 7 if has_phy else 5
    fig_h = 26.0 if has_phy else 19.0
    h_ratios = ([1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.92] if has_phy else None)

    # ------------------------------------------------------------------
    # Layout: N panels (left) + config sidebar (right)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(23, fig_h) if has_phy else (22, fig_h))
    _wr_left, _wr_side = (4.0, 1.22) if has_phy else (4.2, 1.0)
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[_wr_left, _wr_side], wspace=0.04)
    gs_kw: dict = {"hspace": 0.40}
    if h_ratios is not None:
        gs_kw["height_ratios"] = h_ratios
    gs_panels = gridspec.GridSpecFromSubplotSpec(n_main, 1, subplot_spec=gs_outer[0], **gs_kw)

    ax1 = fig.add_subplot(gs_panels[0])  # Overview
    ax2 = fig.add_subplot(gs_panels[1])  # Progress rewards
    ax3 = fig.add_subplot(gs_panels[2])  # Penalties / bonus
    ax4 = fig.add_subplot(gs_panels[3])  # Terminations
    ax5 = fig.add_subplot(gs_panels[4])  # Policy / losses

    title = f"Rover Disk-World — Training Progress  (iter {blocks[0][0]}–{max_iter})"
    if has_phy:
        title += "  ·  latched physics appended below"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    # ── Panel 1: OVERVIEW ─────────────────────────────────────────────
    ax1.set_title("Overview", fontsize=10, loc="left")
    _plot_line(ax1, blocks, "Train/mean_reward", "Mean reward", color="#1f77b4")
    ax1.set_ylabel("Mean reward", color="#1f77b4")
    ax1_r = ax1.twinx()
    _plot_line(ax1_r, blocks, "Train/mean_episode_length", "Episode length (steps)",
               color="#ff7f0e", linestyle="--")
    ax1_r.set_ylabel("Episode length (steps)", color="#ff7f0e")
    _legend_if_any(ax1, loc="upper left", fontsize=8)
    _legend_if_any(ax1_r, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: PROGRESS REWARDS ──────────────────────────────────────
    ax2.set_title("Progress Rewards", fontsize=10, loc="left")
    _REW = "Episode_Reward/"
    _plot_line(ax2, blocks, _REW + "forward_angular_progress", "forward_angular_progress", color="#2ca02c")
    _plot_line(ax2, blocks, _REW + "forward_speed_tracking", "speed_tracking", color="#17becf", linestyle="--")
    _plot_line(ax2, blocks, _REW + "forward_clockwise_speed_bonus", "cw_speed_bonus", color="#ff9896", linestyle="-")
    _plot_line(ax2, blocks, _REW + "heading_alignment", "heading_alignment", color="#9467bd", linestyle="-.")
    _plot_line(ax2, blocks, _REW + "goal_bonus", "goal_bonus (on success)", color="#d62728", linestyle=":")
    ax2.set_ylabel("Episode reward contribution")
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    _legend_if_any(ax2, fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    if reward_ylim_after >= 0:
        _focus_ylim(ax2, blocks,
                    [_REW + k for k in ("forward_angular_progress", "forward_speed_tracking",
                                        "forward_clockwise_speed_bonus", "heading_alignment", "goal_bonus")],
                    reward_ylim_after)

    # ── Panel 3: PENALTIES / BONUS ─────────────────────────────────────
    ax3.set_title("Penalties, jerk/heading pairs, radial-Z align & Alive", fontsize=10, loc="left")
    _plot_line(ax3, blocks, _REW + "action_rate_penalty", "action_rate_penalty", color="#e377c2")
    _plot_line(ax3, blocks, _REW + "tilt_penalty", "tilt_penalty", color="#8c564b", linestyle="--")
    _plot_line(ax3, blocks, _REW + "y_drift_penalty", "y_drift_penalty (Y from 0)", color="#1f77b4", linestyle="-.")
    _plot_line(ax3, blocks, _REW + "alive_bonus", "alive_bonus", color="#bcbd22", linestyle="-.")
    _plot_line(ax3, blocks, _REW + "motion_jerk_penalty", "motion_jerk_penalty", color="#17becf", linestyle=":")
    _plot_line(
        ax3, blocks, _REW + "motion_jerk_smoothness_bonus", "motion_jerk_smoothness_bonus", color="#17becf", linestyle="-"
    )
    _plot_line(ax3, blocks, _REW + "heading_oscillation_penalty", "heading_oscillation_penalty", color="#7f7f7f", linestyle="-.")
    _plot_line(
        ax3, blocks, _REW + "heading_stability_bonus", "heading_stability_bonus", color="#7f7f7f", linestyle="-"
    )
    _plot_line(
        ax3, blocks, _REW + "radial_z_alignment_reward", "radial_z_align_reward (body+Z·radial)", color="#98df8a", linestyle="-"
    )
    _plot_line(
        ax3, blocks, _REW + "radial_z_alignment_penalty", "radial_z_align_penalty (-(1-dot)²)", color="#98df8a", linestyle=":"
    )
    ax3.set_ylabel("Episode reward contribution")
    ax3.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    _legend_if_any(ax3, fontsize=8)
    ax3.grid(True, alpha=0.3)
    if reward_ylim_after >= 0:
        _focus_ylim(
            ax3,
            blocks,
            [
                _REW + k
                for k in (
                    "action_rate_penalty",
                    "tilt_penalty",
                    "y_drift_penalty",
                    "alive_bonus",
                    "motion_jerk_penalty",
                    "motion_jerk_smoothness_bonus",
                    "heading_oscillation_penalty",
                    "heading_stability_bonus",
                    "radial_z_alignment_reward",
                    "radial_z_alignment_penalty",
                )
            ],
            reward_ylim_after,
        )

    # ── Panel 4: TERMINATIONS ──────────────────────────────────────────
    ax4.set_title("Episode Termination Rates  (fraction of resets)", fontsize=10, loc="left")
    _TERM = "Episode_Termination/"
    _plot_line(ax4, blocks, _TERM + "fell_off_disk", "fell_off_disk (failure)", color="#d62728")
    _plot_line(ax4, blocks, _TERM + "forward_progress_goal", "goal_reached (success)", color="#2ca02c", linestyle="--")
    _plot_line(ax4, blocks, _TERM + "time_out", "time_out (truncated)", color="#7f7f7f", linestyle=":")
    ax4.set_ylabel("Rate [0–1]")
    ax4.set_ylim(-0.02, 1.05)
    ax4.axhline(0, color="gray", linewidth=0.5)
    _legend_if_any(ax4, fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: POLICY / LOSSES ───────────────────────────────────────
    ax5.set_title("Policy Losses & Noise", fontsize=10, loc="left")
    _plot_line(ax5, blocks, "Loss/value_function", "value_function loss", color="#1f77b4")
    _plot_line(ax5, blocks, "Loss/surrogate", "surrogate loss", color="#ff7f0e", linestyle="--")
    _plot_line(ax5, blocks, "Loss/entropy", "entropy", color="#2ca02c", linestyle="-.")
    ax5.set_ylabel("Loss value")
    _legend_if_any(ax5, fontsize=8, loc="upper right")
    ax5.grid(True, alpha=0.3)
    ax5_r = ax5.twinx()
    _plot_line(ax5_r, blocks, "Policy/mean_noise_std", "mean action noise std", color="#9467bd", linestyle=":")
    ax5_r.set_ylabel("Noise std", color="#9467bd")
    _legend_if_any(ax5_r, fontsize=8, loc="center right")
    if not has_phy:
        ax5.set_xlabel("Training iteration")

    if has_phy:
        ax_phy_top = fig.add_subplot(gs_panels[5])
        ax_phy_bot = fig.add_subplot(gs_panels[6])
        ax_phy_bot.sharex(ax5)
        ax_phy_top.sharex(ax_phy_bot)
        _draw_physparam_panels(ax_phy_top, ax_phy_bot, blocks)

    # ── Config sidebar ─────────────────────────────────────────────────
    ax_cfg = fig.add_subplot(gs_outer[1])
    ax_cfg.axis("off")

    def _cfgv(k: str, default: str = "—") -> str:
        return cfg.get(k, default)

    sidebar_lines = [
        "══ Run ══",
        f"  experiment   : {_cfgv('experiment_name')}",
        f"  task           : {_cfgv('task')}",
        "",
        "══ Reward Weights ══",
        f"  fwd_progress : {_cfgv('rover_forward_progress_weight')}",
        f"  speed_track  : {_cfgv('rover_speed_tracking_weight')}",
        f"    target_spd : {_cfgv('rover_speed_target')} m/s",
        f"  heading      : {_cfgv('rover_heading_weight')}",
        f"  alive        : {_cfgv('rover_alive_weight')}",
        f"  tilt_pen     : {_cfgv('rover_tilt_weight')}",
        f"  radial_z_rew : {_cfgv('rover_radial_z_alignment_reward_weight')}",
        f"  radial_z_pen : {_cfgv('rover_radial_z_alignment_penalty_weight')}",
        f"  act_rate_pen : {_cfgv('rover_action_rate_weight')}",
        f"  motion_jerk  : {_cfgv('rover_motion_jerk_weight')}",
        f"  jerk_bonus   : {_cfgv('rover_motion_jerk_bonus_weight')}  exp={_cfgv('rover_motion_jerk_bonus_exp_scale')}",
        f"  head_oscill  : {_cfgv('rover_heading_oscillation_weight')}",
        f"  head_stab_bon: {_cfgv('rover_heading_stability_bonus_weight')}  exp={_cfgv('rover_heading_stability_bonus_exp_scale')}",
        f"  goal_bonus   : {_cfgv('rover_goal_bonus_weight')}",
        "",
        "══ Termination ══",
        f"  goal_rad     : {_cfgv('rover_goal_rad')} rad",
        f"  fall_r_min   : {_cfgv('rover_fall_radius')} m",
        f"  episode_s    : {_cfgv('rover_episode_length_s')} s",
        "",
        "══ Training ══",
        f"  num_envs     : {_cfgv('num_envs')}",
        f"  max_iter     : {_cfgv('max_iterations')}",
        f"  seed         : {_cfgv('seed')}",
        f"  device       : {_cfgv('device')}",
        "",
        "══ PPO ══",
        f"  lr           : {_cfgv('ppo_learning_rate')}",
        f"  entropy      : {_cfgv('ppo_entropy_coef')}",
        f"  noise_std    : {_cfgv('ppo_init_noise_std')}",
        f"  clip         : {_cfgv('ppo_clip_param')}",
        f"  kl_target    : {_cfgv('ppo_desired_kl')}",
        f"  steps/env    : {_cfgv('ppo_num_steps_per_env')}",
        f"  epochs       : {_cfgv('ppo_num_learning_epochs')}",
        f"  mini_batches : {_cfgv('ppo_num_mini_batches')}",
        f"  gamma        : {_cfgv('ppo_gamma')}",
        f"  lam          : {_cfgv('ppo_lam')}",
        "",
    ]
    if has_phy:
        sidebar_lines.extend(_physparam_latch_sidebar_lines())
    sidebar_lines.extend(
        [
            f"══ Iterations logged: {len(blocks)} ══",
            f"  {blocks[0][0]}  →  {max_iter}",
        ]
    )

    _sidebar_fs = 6.15 if has_phy else 7.5
    ax_cfg.text(
        0.04, 0.97, "\n".join(sidebar_lines),
        transform=ax_cfg.transAxes,
        fontsize=_sidebar_fs,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to: {output_path}")

    if physparam_split_output_path and blocks_have_physparam_metrics(blocks):
        generate_physparam_training_plot(blocks, physparam_split_output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse training progress for Rover Disk-World (Forward / Physparam).",
    )
    parser.add_argument("log_dir", help="Path to the RSL-RL log directory.")
    parser.add_argument("--last", type=int, default=None, metavar="N",
                        help="Print only the last N iteration blocks.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not generate the visual analysis PNG.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate the PNG (skip text print).")
    parser.add_argument("--output", type=str, default="training_progress_analysis.png",
                        help="Output PNG filename (default: training_progress_analysis.png).")
    parser.add_argument(
        "--split-physparam",
        action="store_true",
        help="Also write a standalone physparam-only PNG (--physparam-output).",
    )
    parser.add_argument(
        "--physparam-output",
        type=str,
        default="training_progress_physparam.png",
        help="Filename for split physparam PNG (only with --split-physparam).",
    )
    parser.add_argument("--reward-ylim-after", type=int, default=200, metavar="ITER",
                        help="Focus reward y-axis on iterations ≥ ITER (default 200; -1 = full auto).")
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
        split_phy = (
            os.path.join(args.log_dir, args.physparam_output) if args.split_physparam else None
        )
        generate_plot(
            blocks,
            cfg,
            output_path,
            reward_ylim_after=args.reward_ylim_after,
            physparam_split_output_path=split_phy,
        )


if __name__ == "__main__":
    main()

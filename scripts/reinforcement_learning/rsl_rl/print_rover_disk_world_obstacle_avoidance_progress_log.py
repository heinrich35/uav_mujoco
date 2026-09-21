#!/usr/bin/env python3
"""Training-progress analysis for Rover Disk-World Obstacle-Avoidance.

Task:
    Isaac-Rover-DiskWorld-ObstacleAvoidance-v0

Reads ``training_progress_log.txt`` and generates:
    - ``training_progress_analysis.png`` (default)

When the header records ``rover_oa_objective_ramp_blend_mode: full_lerp`` and a
ramp iteration count, the overview panel draws a vertical line at that horizon
(learning iterations), matching the G1-style staged ramp concept.

The figure includes:
    1) overview (mean reward, episode length, env steps/s)
    2) primary OA rewards + twin axis for forward-speed tracking penalty
    3) penalties + smoothness (blue box, visual box, fall, termination, motion)
    4) terminations
    5) policy/loss metrics
    6) a config sidebar with OA reward / teacher / in-process ramp / scene keys from log header
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _pre_iteration_content(log_path: str) -> str:
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    return content.split("Learning iteration", 1)[0]


_RSL_RL_TO_PLOT: tuple[tuple[str, str], ...] = (
    ("Mean reward", "Train/mean_reward"),
    ("Mean episode length", "Train/mean_episode_length"),
    ("Mean value_function loss", "Loss/value_function"),
    ("Mean surrogate loss", "Loss/surrogate"),
    ("Mean entropy loss", "Loss/entropy"),
    ("Mean action noise std", "Policy/mean_noise_std"),
)


_PERF_ALIASES: tuple[tuple[str, str], ...] = (
    ("Perf/collection_wall_s", "Perf/collection time"),
    ("Perf/learning_wall_s", "Perf/learning_time"),
    # logging_wrapper appends env_steps_per_sec; plot historically used Perf/FPS
    ("Perf/env_steps_per_sec", "Perf/FPS"),
)

# Rover OA progress log may duplicate mean reward under Loc_progress/*
_OA_MEAN_REWARD_ALIASES: tuple[tuple[str, str], ...] = (
    ("Loc_progress/rsl_mean_reward", "Train/mean_reward"),
)


def _merge_rsl_rl_metric_aliases(d: dict[str, float]) -> None:
    for src, dst in _RSL_RL_TO_PLOT:
        if src in d and dst not in d:
            d[dst] = d[src]
    for src, dst in _PERF_ALIASES:
        if src in d and dst not in d:
            d[dst] = d[src]
    for src, dst in _OA_MEAN_REWARD_ALIASES:
        if src in d and dst not in d:
            d[dst] = d[src]


def _parse_block(block: str) -> tuple[int | None, dict[str, float]]:
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


def parse_log(log_path: str) -> list[tuple[int, dict[str, float]]]:
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
    try:
        head = _pre_iteration_content(log_path)
    except OSError:
        return {}
    cfg: dict[str, str] = {}
    keys = [
        "task",
        "experiment_name",
        "rover_oa_forward_progress_weight",
        "rover_oa_forward_speed_tracking_penalty_weight",
        "rover_oa_forward_speed_target_ms",
        "rover_oa_blue_box_penalty_weight",
        "rover_oa_blue_box_danger_radius_m",
        "rover_oa_goal_pass_through_weight",
        "rover_oa_goal_pass_threshold_m",
        "rover_oa_heading_alignment_weight",
        "rover_oa_heading_alignment_tolerance_rad",
        "rover_oa_radial_z_alignment_reward_weight",
        "rover_oa_radial_z_alignment_penalty_weight",
        "rover_oa_radial_z_penalty_max_abs_m",
        "rover_oa_upside_down_penalty_weight",
        "rover_oa_fell_off_disk_penalty_weight",
        "rover_oa_upside_down_min_radial_dot",
        "rover_oa_upside_down_min_dot_z",  # legacy alias
        "rover_oa_action_rate_penalty_weight",
        "rover_oa_motion_jerk_penalty_weight",
        "rover_oa_heading_oscillation_penalty_weight",
        "rover_oa_alive_bonus_weight",
        "rover_oa_teacher_imitation_weight",
        "rover_oa_teacher_imitation_decay_env_steps",
        "rover_oa_teacher_imitation_match_sigma",
        "rover_oa_teacher_policy_enable",
        "rover_oa_teacher_policy_path",
        "rover_oa_teacher_action_blend_weight",
        "rover_oa_teacher_action_blend_decay_env_steps",
        "rover_oa_objective_ramp_enable",
        "rover_oa_objective_ramp_iters",
        "rover_oa_objective_ramp_start_scale",
        "rover_oa_objective_ramp_forward_boost",
        "rover_oa_objective_ramp_blend_mode",
        "rover_oa_objective_ramp_anchor_preset",
        "rover_oa_init_random_episode_length",
        "rover_oa_fall_min_radius_m",
        "rover_oa_near_fall_penalty_weight",
        "rover_oa_near_fall_danger_band_m",
        "rover_oa_visual_box_ahead_penalty_weight",
        "rover_oa_visual_box_ahead_bearing_gate_rad",
        "rover_oa_visual_box_ahead_range_gate_m",
        "rover_oa_visual_goal_ahead_reward_weight",
        "rover_oa_visual_goal_ahead_bearing_gate_rad",
        "rover_oa_visual_goal_ahead_range_gate_m",
        "rover_oa_goal_approach_shaping_weight",
        "rover_oa_goal_approach_sigma_m",
        "rover_episode_length_s",
        "rover_localization_goal_count",
        "rover_localization_box_count",
        "rover_localization_goal_collision",
        "rover_localization_box_collision",
        "rover_localization_goal_asset_path",
        "rover_localization_visual_obstacle_max_envs",
        "num_envs",
        "max_iterations",
        "seed",
        "device",
        # PPO
        "ppo_learning_rate",
        "ppo_entropy_coef",
        "ppo_entropy_coef_final",
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


def print_blocks(blocks: list[tuple[int, dict[str, float]]], last: int | None = None) -> None:
    if last is not None:
        blocks = blocks[-last:]
    for it, data in blocks:
        print(f"\n{'─'*60}")
        print(f"  Iteration {it}")
        print(f"{'─'*60}")
        for k in ("Train/mean_reward", "Train/mean_episode_length", "Perf/FPS"):
            if k in data:
                print(f"  {k:<45s} {data[k]:>10.4f}")

        rew_keys = sorted(k for k in data if k.startswith("Episode_Reward/"))
        if rew_keys:
            print("  -- Rewards --")
            for k in rew_keys:
                print(f"  {k:<45s} {data[k]:>10.4f}")

        term_keys = sorted(k for k in data if k.startswith("Episode_Termination/"))
        if term_keys:
            print("  -- Terminations --")
            for k in term_keys:
                print(f"  {k:<45s} {data[k]:>10.4f}")

        loss_keys = sorted(k for k in data if k.startswith("Loss/"))
        if loss_keys:
            print("  -- Losses --")
            for k in loss_keys:
                print(f"  {k:<45s} {data[k]:>10.4f}")


def _series(blocks: list[tuple[int, dict[str, float]]], key: str) -> tuple[list[int], list[float]]:
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


def _label_with_cfg(label: str, cfg: dict[str, str], cfg_key: str | None = None, prefix: str = "w") -> str:
    """Append a config value to a legend label.

    Important: values like ``0`` / ``0.0`` are valid settings and must be shown,
    not treated as missing. Therefore we check key membership explicitly instead
    of relying on truthiness.
    """
    if cfg_key is None or cfg_key not in cfg:
        return label
    return f"{label} ({prefix}={cfg[cfg_key]})"


def _legend_if_any(ax, **kwargs) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def _focus_ylim(ax, blocks: list[tuple[int, dict[str, float]]], keys: Sequence[str], after_iter: int):
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
    reward_ylim_after: int = 50,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed – skipping plot generation.")
        return

    if not blocks:
        print("[WARN] No iteration data found – cannot generate plot.")
        return

    max_iter = blocks[-1][0]
    ramp_line_x: int | None = None
    try:
        _re = int(float(cfg.get("rover_oa_objective_ramp_iters", "") or 0))
        _en = int(float(cfg.get("rover_oa_objective_ramp_enable", "") or 0))
        _bm = (cfg.get("rover_oa_objective_ramp_blend_mode") or "").strip().lower()
        if _en == 1 and _re > 0 and _bm in ("full_lerp", "full", "lerp"):
            ramp_line_x = _re
    except (TypeError, ValueError):
        ramp_line_x = None

    fig = plt.figure(figsize=(22, 20))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4.2, 1.05], wspace=0.04)
    gs_panels = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_outer[0], hspace=0.42)

    ax1 = fig.add_subplot(gs_panels[0])  # Overview
    ax2 = fig.add_subplot(gs_panels[1])  # Primary rewards
    ax3 = fig.add_subplot(gs_panels[2])  # Penalties
    ax4 = fig.add_subplot(gs_panels[3])  # Terminations
    ax5 = fig.add_subplot(gs_panels[4])  # Losses/policy

    fig.suptitle(
        f"Rover Disk-World Obstacle-Avoidance — Training Progress  (iter {blocks[0][0]}–{max_iter})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # Panel 1: overview
    ax1.set_title("Overview", fontsize=10, loc="left")
    _plot_line(ax1, blocks, "Train/mean_reward", "Mean reward", color="#1f77b4")
    ax1.set_ylabel("Mean reward", color="#1f77b4")
    ax1_r = ax1.twinx()
    _plot_line(ax1_r, blocks, "Train/mean_episode_length", "Episode length (steps)", color="#ff7f0e", linestyle="--")
    _plot_line(ax1_r, blocks, "Perf/FPS", "FPS", color="#2ca02c", linestyle=":")
    ax1_r.set_ylabel("Length / FPS")
    _legend_if_any(ax1, loc="upper left", fontsize=8)
    _legend_if_any(ax1_r, loc="upper right", fontsize=8)
    if ramp_line_x is not None and blocks[0][0] <= ramp_line_x <= max_iter:
        ax1.axvline(ramp_line_x, color="#6c3483", linestyle="--", linewidth=1.2, alpha=0.75)
    ax1.grid(True, alpha=0.3)

    # Panel 2: primary OA rewards (forward, goals, vision shaping, alignment) + speed tracking on twin axis
    _REW = "Episode_Reward/"
    ax2.set_title(
        "Primary rewards: forward, goals & vision shaping, heading — "
        "(right axis: forward_speed_tracking_penalty)",
        fontsize=10,
        loc="left",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "forward_angular_progress",
        _label_with_cfg("forward_angular_progress", cfg, "rover_oa_forward_progress_weight"),
        color="#2ca02c",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "radial_z_alignment_reward",
        _label_with_cfg("radial_z_alignment_reward", cfg, "rover_oa_radial_z_alignment_reward_weight"),
        color="#1f77b4",
        linestyle=":",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "next_goal_pass_through_reward",
        _label_with_cfg("next_goal_pass_through_reward", cfg, "rover_oa_goal_pass_through_weight"),
        color="#ff7f0e",
        linestyle="--",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "visual_goal_ahead_alignment_reward",
        _label_with_cfg("visual_goal_ahead_alignment_reward", cfg, "rover_oa_visual_goal_ahead_reward_weight"),
        color="#f2a900",
        linestyle="-.",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "next_goal_approach_shaping",
        _label_with_cfg("next_goal_approach_shaping", cfg, "rover_oa_goal_approach_shaping_weight"),
        color="#a6761d",
        linestyle=":",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "disk_tangent_heading_alignment",
        _label_with_cfg("disk_tangent_heading_alignment", cfg, "rover_oa_heading_alignment_weight"),
        color="#9467bd",
        linestyle="-.",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "teacher_action_imitation",
        _label_with_cfg("teacher_action_imitation", cfg, "rover_oa_teacher_imitation_weight"),
        color="#17becf",
    )
    _plot_line(
        ax2,
        blocks,
        _REW + "alive_bonus",
        _label_with_cfg("alive_bonus", cfg, "rover_oa_alive_bonus_weight"),
        color="#bcbd22",
        linestyle=":",
    )
    ax2.set_ylabel("Episode reward contribution (positive terms)")
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax2_r = ax2.twinx()
    _plot_line(
        ax2_r,
        blocks,
        _REW + "forward_speed_tracking_penalty",
        _label_with_cfg(
            "forward_speed_tracking_penalty (≤0)",
            cfg,
            "rover_oa_forward_speed_tracking_penalty_weight",
        ),
        color="#e377c2",
        linestyle="-",
        alpha=0.85,
    )
    ax2_r.set_ylabel("Speed tracking penalty (typically ≤0)")
    ax2_r.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    _legend_if_any(ax2, fontsize=8, ncol=2, loc="upper left")
    _legend_if_any(ax2_r, fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)
    if reward_ylim_after >= 0:
        _focus_ylim(
            ax2,
            blocks,
            [
                _REW + "forward_angular_progress",
                _REW + "radial_z_alignment_reward",
                _REW + "next_goal_pass_through_reward",
                _REW + "visual_goal_ahead_alignment_reward",
                _REW + "next_goal_approach_shaping",
                _REW + "disk_tangent_heading_alignment",
                _REW + "teacher_action_imitation",
                _REW + "alive_bonus",
            ],
            reward_ylim_after,
        )

    # Panel 3: penalties / smoothness (including dense blue-box + near-fall + motion)
    ax3.set_title(
        "Penalties & smoothness: blue box, visual box, edge safety, terminations, motion",
        fontsize=10,
        loc="left",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "blue_box_avoidance_penalty",
        _label_with_cfg("blue_box_avoidance_penalty", cfg, "rover_oa_blue_box_penalty_weight"),
        color="#d62728",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "visual_blue_box_ahead_penalty",
        _label_with_cfg("visual_blue_box_ahead_penalty", cfg, "rover_oa_visual_box_ahead_penalty_weight"),
        color="#ff7f7f",
        linestyle="--",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "near_fall_radial_penalty",
        _label_with_cfg("near_fall_radial_penalty", cfg, "rover_oa_near_fall_penalty_weight"),
        color="#4d0000",
        linestyle="-.",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "radial_z_alignment_penalty",
        _label_with_cfg("radial_z_alignment_penalty", cfg, "rover_oa_radial_z_alignment_penalty_weight"),
        color="#ff9896",
        linestyle="--",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "upside_down_termination_penalty",
        _label_with_cfg("upside_down_termination_penalty", cfg, "rover_oa_upside_down_penalty_weight"),
        color="#000000",
        linestyle="-",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "fell_off_disk_termination_penalty",
        _label_with_cfg("fell_off_disk_termination_penalty", cfg, "rover_oa_fell_off_disk_penalty_weight"),
        color="#8b0000",
        linestyle=":",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "action_rate_penalty",
        _label_with_cfg("action_rate_penalty", cfg, "rover_oa_action_rate_penalty_weight"),
        color="#e377c2",
        linestyle="--",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "motion_jerk_penalty",
        _label_with_cfg("motion_jerk_penalty", cfg, "rover_oa_motion_jerk_penalty_weight"),
        color="#8c564b",
        linestyle=":",
    )
    _plot_line(
        ax3,
        blocks,
        _REW + "heading_oscillation_penalty",
        _label_with_cfg("heading_oscillation_penalty", cfg, "rover_oa_heading_oscillation_penalty_weight"),
        color="#7f7f7f",
        linestyle="-.",
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
                _REW + "blue_box_avoidance_penalty",
                _REW + "visual_blue_box_ahead_penalty",
                _REW + "near_fall_radial_penalty",
                _REW + "radial_z_alignment_penalty",
                _REW + "upside_down_termination_penalty",
                _REW + "fell_off_disk_termination_penalty",
                _REW + "action_rate_penalty",
                _REW + "motion_jerk_penalty",
                _REW + "heading_oscillation_penalty",
            ],
            reward_ylim_after,
        )

    # Panel 4: terminations
    _TERM = "Episode_Termination/"
    ax4.set_title("Episode Termination Rates (fraction of resets)", fontsize=10, loc="left")
    _plot_line(ax4, blocks, _TERM + "fell_off_disk", "fell_off_disk (failure)", color="#d62728")
    _plot_line(ax4, blocks, _TERM + "upside_down", "upside_down (failure)", color="#000000", linestyle="-.")
    _plot_line(ax4, blocks, _TERM + "time_out", "time_out (truncated)", color="#7f7f7f", linestyle=":")
    _plot_line(ax4, blocks, _TERM + "forward_progress_goal", "forward_progress_goal", color="#2ca02c", linestyle="--")
    ax4.set_ylabel("Rate [0–1]")
    ax4.set_ylim(-0.02, 1.05)
    ax4.axhline(0, color="gray", linewidth=0.5)
    _legend_if_any(ax4, fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: losses/policy
    ax5.set_title("Policy / Optimisation Metrics", fontsize=10, loc="left")
    _plot_line(ax5, blocks, "Loss/value_function", "value_function loss", color="#1f77b4")
    _plot_line(ax5, blocks, "Loss/surrogate", "surrogate loss", color="#ff7f0e", linestyle="--")
    _plot_line(ax5, blocks, "Loss/entropy", "entropy", color="#2ca02c", linestyle="-.")
    _plot_line(ax5, blocks, "Perf/collection time", "collection time", color="#8c564b", linestyle=":")
    _plot_line(ax5, blocks, "Perf/learning_time", "learning time", color="#7f7f7f", linestyle="-")
    ax5.set_ylabel("Loss / time")
    ax5.set_xlabel("Training iteration")
    _legend_if_any(ax5, fontsize=8, ncol=2, loc="upper right")
    ax5.grid(True, alpha=0.3)
    ax5_r = ax5.twinx()
    _plot_line(ax5_r, blocks, "Policy/mean_noise_std", "mean action noise std", color="#9467bd", linestyle=":")
    _plot_line(ax5_r, blocks, "Policy/learning_rate", "learning rate", color="#17becf", linestyle="--")
    ax5_r.set_ylabel("Noise / LR")
    _legend_if_any(ax5_r, fontsize=8, loc="center right")

    # Config sidebar
    ax_cfg = fig.add_subplot(gs_outer[1])
    ax_cfg.axis("off")

    def _cfgv(k: str, default: str = "—") -> str:
        return cfg.get(k, default)

    sidebar_lines = [
        "══ Run ══",
        f"  experiment : {_cfgv('experiment_name')}",
        f"  task       : {_cfgv('task')}",
        "",
        "══ OA Rewards ══",
        f"  fwd_progress       : {_cfgv('rover_oa_forward_progress_weight')}",
        f"  fwd_speed_pen      : {_cfgv('rover_oa_forward_speed_tracking_penalty_weight')}  target={_cfgv('rover_oa_forward_speed_target_ms')}",
        f"  blue_box_pen       : {_cfgv('rover_oa_blue_box_penalty_weight')}  sigma={_cfgv('rover_oa_blue_box_danger_radius_m')}",
        f"  goal_pass_bonus    : {_cfgv('rover_oa_goal_pass_through_weight')}  thr={_cfgv('rover_oa_goal_pass_threshold_m')}",
        f"  heading_align      : {_cfgv('rover_oa_heading_alignment_weight')}  tol={_cfgv('rover_oa_heading_alignment_tolerance_rad')}",
        f"  radial_z_reward    : {_cfgv('rover_oa_radial_z_alignment_reward_weight')}",
        f"  radial_z_penalty   : {_cfgv('rover_oa_radial_z_alignment_penalty_weight')}  "
        f"clip_m={_cfgv('rover_oa_radial_z_penalty_max_abs_m')}",
        f"  upside_down_pen    : {_cfgv('rover_oa_upside_down_penalty_weight')}  "
        f"min_radial_dot={_cfgv('rover_oa_upside_down_min_radial_dot', _cfgv('rover_oa_upside_down_min_dot_z'))}",
        f"  fell_off_disk_pen  : {_cfgv('rover_oa_fell_off_disk_penalty_weight')}",
        f"  near_fall_pen      : {_cfgv('rover_oa_near_fall_penalty_weight')}  "
        f"band={_cfgv('rover_oa_near_fall_danger_band_m')}",
        f"  vis_box_ahead_pen  : {_cfgv('rover_oa_visual_box_ahead_penalty_weight')}  "
        f"bear={_cfgv('rover_oa_visual_box_ahead_bearing_gate_rad')} rng={_cfgv('rover_oa_visual_box_ahead_range_gate_m')}",
        f"  vis_goal_ahead_rew : {_cfgv('rover_oa_visual_goal_ahead_reward_weight')}  "
        f"bear={_cfgv('rover_oa_visual_goal_ahead_bearing_gate_rad')} rng={_cfgv('rover_oa_visual_goal_ahead_range_gate_m')}",
        f"  goal_approach_shp  : {_cfgv('rover_oa_goal_approach_shaping_weight')}  "
        f"sigma={_cfgv('rover_oa_goal_approach_sigma_m')}",
        f"  action_rate_pen    : {_cfgv('rover_oa_action_rate_penalty_weight')}",
        f"  motion_jerk_pen    : {_cfgv('rover_oa_motion_jerk_penalty_weight')}",
        f"  heading_osc_pen    : {_cfgv('rover_oa_heading_oscillation_penalty_weight')}",
        f"  alive_bonus        : {_cfgv('rover_oa_alive_bonus_weight')}",
        "",
        "══ Teacher Bootstrap ══",
        f"  imitation_weight   : {_cfgv('rover_oa_teacher_imitation_weight')}",
        f"  decay_env_steps    : {_cfgv('rover_oa_teacher_imitation_decay_env_steps')}",
        f"  match_sigma        : {_cfgv('rover_oa_teacher_imitation_match_sigma')}",
        f"  teacher_enable     : {_cfgv('rover_oa_teacher_policy_enable')}",
        f"  teacher_ckpt       : {_cfgv('rover_oa_teacher_policy_path')}",
        f"  action_blend_w     : {_cfgv('rover_oa_teacher_action_blend_weight')}",
        f"  action_blend_decay : {_cfgv('rover_oa_teacher_action_blend_decay_env_steps')}",
        "",
        "══ In-process objective ramp (train.py) ══",
        f"  ramp_enable        : {_cfgv('rover_oa_objective_ramp_enable')}",
        f"  ramp_blend_mode    : {_cfgv('rover_oa_objective_ramp_blend_mode')}",
        f"  ramp_anchor_preset : {_cfgv('rover_oa_objective_ramp_anchor_preset')}",
        f"  ramp_iters         : {_cfgv('rover_oa_objective_ramp_iters')}",
        f"  ramp_start_scale   : {_cfgv('rover_oa_objective_ramp_start_scale')}  (legacy; scale mode)",
        f"  ramp_fwd_boost     : {_cfgv('rover_oa_objective_ramp_forward_boost')}  (legacy; scale mode)",
        f"  init_rand_ep_len   : {_cfgv('rover_oa_init_random_episode_length')}",
        "",
        "══ Termination / Scene ══",
        f"  fall_r_min         : {_cfgv('rover_oa_fall_min_radius_m')}",
        f"  episode_s          : {_cfgv('rover_episode_length_s')}",
        f"  goals              : {_cfgv('rover_localization_goal_count')}",
        f"  blue_boxes         : {_cfgv('rover_localization_box_count')}",
        f"  usd_visual_cap    : {_cfgv('rover_localization_visual_obstacle_max_envs')}",
        f"  goal_asset         : {_cfgv('rover_localization_goal_asset_path')}",
        f"  goal_collision     : {_cfgv('rover_localization_goal_collision')}",
        f"  box_collision      : {_cfgv('rover_localization_box_collision')}",
        "",
        "══ Training ══",
        f"  num_envs           : {_cfgv('num_envs')}",
        f"  max_iter           : {_cfgv('max_iterations')}",
        f"  seed               : {_cfgv('seed')}",
        f"  device             : {_cfgv('device')}",
        "",
        "══ PPO ══",
        f"  lr                 : {_cfgv('ppo_learning_rate')}",
        f"  entropy            : {_cfgv('ppo_entropy_coef')}",
        f"  entropy_final      : {_cfgv('ppo_entropy_coef_final')}",
        f"  noise_std          : {_cfgv('ppo_init_noise_std')}",
        f"  clip               : {_cfgv('ppo_clip_param')}",
        f"  kl_target          : {_cfgv('ppo_desired_kl')}",
        f"  steps/env          : {_cfgv('ppo_num_steps_per_env')}",
        f"  epochs             : {_cfgv('ppo_num_learning_epochs')}",
        f"  mini_batches       : {_cfgv('ppo_num_mini_batches')}",
        f"  gamma              : {_cfgv('ppo_gamma')}",
        f"  lam                : {_cfgv('ppo_lam')}",
        "",
        f"══ Iterations logged: {len(blocks)} ══",
        f"  {blocks[0][0]}  →  {max_iter}",
    ]
    ax_cfg.text(
        0.04,
        0.97,
        "\n".join(sidebar_lines),
        transform=ax_cfg.transAxes,
        fontsize=7.0,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse training progress for Rover Disk-World Obstacle-Avoidance.",
    )
    parser.add_argument("log_dir", help="Path to the RSL-RL log directory.")
    parser.add_argument("--last", type=int, default=None, metavar="N", help="Print only the last N iteration blocks.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate the visual analysis PNG.")
    parser.add_argument("--plot-only", action="store_true", help="Only generate the PNG (skip text print).")
    parser.add_argument("--output", type=str, default="training_progress_analysis.png", help="Output PNG filename.")
    parser.add_argument(
        "--reward-ylim-after",
        type=int,
        default=50,
        metavar="ITER",
        help="Focus reward y-axis on iterations >= ITER (default 50; -1 = full auto).",
    )
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
        generate_plot(blocks, cfg, output_path, reward_ylim_after=args.reward_ylim_after)


if __name__ == "__main__":
    main()

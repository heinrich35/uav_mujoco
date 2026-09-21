#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress for Isaac-Ion-Vertical-Takeoff-Direct-v0 (ion_vertical_takeoff_env.py).
# Layout: wide figure, GridSpec — main, losses, rewards (positive), penalties, config sidebar.
#
# Uses a line-based block parser (RSL-RL pads metric lines; v3 finditer can skip Episode_Reward / Mean lines).
# Merges any Episode_Reward/* keys from the log that match Ion Direct tails.
# Omits plots and printed Episode_Reward lines when the matching run-arg weight is 0.0 (see ION_VERTICAL_TAKEOFF_TRAIN.sh).
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_ion_vertical_takeoff_direct_progress_log.py <log_dir>
#
# Options:
#   --last N, --no-plot, --output NAME, --plot-only
#   --reward-penalty-ylim-focus-after ITER   (default 300; -1 = full autoscale)
#   --reward-penalty-ylim-late-frac FRAC    (default 0.22)

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys


def _load_v3_progress_module():
    """Sibling module; avoids package path issues when run as a script."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "print_g1_locomotion_v3_progress_log.py")
    spec = importlib.util.spec_from_file_location("g1_locomotion_v3_progress_log", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v3 = _load_v3_progress_module()

_strip_ansi = _v3._strip_ansi
_episode_component_ylim_from_tail = _v3._episode_component_ylim_from_tail


def _parse_block_ion(block: str) -> tuple[int | None, dict[str, float]]:
    """Parse one iteration block line-by-line (RSL-RL pads Episode_Reward lines with spaces; v3 regex can skip lines)."""
    block = _strip_ansi(block)
    out: dict[str, float] = {}
    it_match = re.search(r"Learning iteration\s+(\d+)/", block)
    iteration = int(it_match.group(1)) if it_match else None
    for line in block.splitlines():
        line = line.rstrip()
        m = re.match(
            r"^\s*(Episode_Reward/[^\s:]+|Episode_Termination/[^\s:]+|Mean\s+[^\n:]*?)\s*:\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*$",
            line,
        )
        if m:
            key = m.group(1).strip()
            try:
                out[key] = float(m.group(2))
            except ValueError:
                pass
            continue
        m2 = re.match(r"^\s*Computation:\s*([\d.]+)\s+steps/s", line)
        if m2:
            try:
                out["Computation"] = float(m2.group(1))
            except ValueError:
                pass
            continue
        m3 = re.match(r"^\s*Iteration time:\s*([\d.]+)s\s*$", line)
        if m3:
            try:
                out["Iteration time"] = float(m3.group(1))
            except ValueError:
                pass
    return iteration, out


def parse_training_progress_log(path: str) -> tuple[list[int], dict[str, list[float]]]:
    """Parse full training_progress_log.txt (Ion-safe line parser)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"(?=^################################################################################\n)", content, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]

    iterations: list[int] = []
    all_keys: set[str] = set()
    block_data: list[dict[str, float]] = []
    for block in blocks:
        it, data = _parse_block_ion(block)
        if it is not None and data:
            iterations.append(it)
            block_data.append(data)
            all_keys |= set(data.keys())

    if not block_data:
        return [], {}

    series: dict[str, list[float | None]] = {k: [] for k in all_keys}
    for data in block_data:
        for k in all_keys:
            series[k].append(data.get(k))

    series_f: dict[str, list[float]] = {}
    for k in list(series.keys()):
        prev: float | None = None
        filled: list[float] = []
        for v in series[k]:
            if v is not None:
                prev = v
                filled.append(v)
            elif prev is not None:
                filled.append(prev)
            else:
                filled.append(0.0)
        series_f[k] = filled

    return iterations, series_f
_filter_config_sidebar_lines = _v3._filter_config_sidebar_lines

# Ion Direct Episode_Reward/* tails (ion_vertical_takeoff_env.py) — v3's merge uses _reward/_penalty
# suffixes only, so most Ion keys were never merged from series.
ION_DIRECT_POSITIVE_TAILS = frozenset(
    {
        "positive_z_velocity",
        "upward_shaping",
        "idle_velocity_reward",
        "x_drift_free",
        "y_drift_free",
        "smooth_motion",
        "electricity_efficiency",
        "orientation_reward",
        "airborne",
        "takeoff_height",
        "z_position",
        "xy_position_reward",
    }
)
ION_DIRECT_PENALTY_TAILS = frozenset(
    {
        "upward_shaping_miss_penalty",
        "off_band_z_velocity",
        "idle",
        "x_drift",
        "y_drift",
        "jerky",
        "electricity",
        "orientation_penalty",
        "low_altitude",
        "xy_position_penalty",
    }
)


def _ordered_unique(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def merge_ion_direct_episode_reward_keys_from_series(
    series: dict[str, list],
    positive: list[str],
    penalty: list[str],
) -> tuple[list[str], list[str]]:
    """Append Episode_Reward/* present in *series* but missing from plot lists (Ion Direct naming)."""
    known = set(positive) | set(penalty)
    for k in series:
        if not k.startswith("Episode_Reward/"):
            continue
        if k in known:
            continue
        tail = k[len("Episode_Reward/") :]
        if tail in ION_DIRECT_POSITIVE_TAILS:
            positive.append(k)
            known.add(k)
        elif tail in ION_DIRECT_PENALTY_TAILS:
            penalty.append(k)
            known.add(k)
    return _ordered_unique(positive), _ordered_unique(penalty)


# Config key -> Episode_Reward (seed plot lists; zero weights filtered later)
ION_DIRECT_REWARD_POSITIVE_WEIGHTS = (
    ("positive_z_velocity_reward_weight", "Episode_Reward/positive_z_velocity"),
    ("upward_velocity_shaping_weight", "Episode_Reward/upward_shaping"),
    ("idle_velocity_reward_weight", "Episode_Reward/idle_velocity_reward"),
    ("x_drift_free_reward_weight", "Episode_Reward/x_drift_free"),
    ("y_drift_free_reward_weight", "Episode_Reward/y_drift_free"),
    ("smooth_motion_reward_weight", "Episode_Reward/smooth_motion"),
    ("electricity_efficiency_reward_weight", "Episode_Reward/electricity_efficiency"),
    ("orientation_reward_weight", "Episode_Reward/orientation_reward"),
    ("airborne_reward_weight", "Episode_Reward/airborne"),
    ("takeoff_height_shaping_weight", "Episode_Reward/takeoff_height"),
    ("z_position_reward_weight", "Episode_Reward/z_position"),
    ("xy_position_reward_weight", "Episode_Reward/xy_position_reward"),
)

ION_DIRECT_REWARD_PENALTY_WEIGHTS = (
    ("upward_velocity_shaping_miss_penalty_weight", "Episode_Reward/upward_shaping_miss_penalty"),
    ("off_band_z_velocity_penalty_weight", "Episode_Reward/off_band_z_velocity"),
    ("idle_velocity_penalty_weight", "Episode_Reward/idle"),
    ("x_drift_penalty", "Episode_Reward/x_drift"),
    ("y_drift_penalty", "Episode_Reward/y_drift"),
    ("jerky_penalty", "Episode_Reward/jerky"),
    ("electricity_cost_penalty", "Episode_Reward/electricity"),
    ("orientation_penalty_weight", "Episode_Reward/orientation_penalty"),
    ("low_altitude_penalty_weight", "Episode_Reward/low_altitude"),
    ("xy_position_penalty_weight", "Episode_Reward/xy_position_penalty"),
)

_EPISODE_REWARD_CFG_GATE: dict[str, tuple[str, ...]] = {
    "Episode_Reward/positive_z_velocity": ("positive_z_velocity_reward_weight",),
    "Episode_Reward/upward_shaping": ("upward_velocity_shaping_weight",),
    "Episode_Reward/upward_shaping_miss_penalty": ("upward_velocity_shaping_miss_penalty_weight",),
    "Episode_Reward/off_band_z_velocity": ("off_band_z_velocity_penalty_weight",),
    "Episode_Reward/idle": ("idle_velocity_penalty_weight",),
    "Episode_Reward/idle_velocity_reward": ("idle_velocity_reward_weight",),
    "Episode_Reward/x_drift": ("x_drift_penalty",),
    "Episode_Reward/y_drift": ("y_drift_penalty",),
    "Episode_Reward/x_drift_free": ("x_drift_free_reward_weight",),
    "Episode_Reward/y_drift_free": ("y_drift_free_reward_weight",),
    "Episode_Reward/jerky": ("jerky_penalty",),
    "Episode_Reward/smooth_motion": ("smooth_motion_reward_weight",),
    "Episode_Reward/electricity": ("electricity_cost_penalty",),
    "Episode_Reward/electricity_efficiency": ("electricity_efficiency_reward_weight",),
    "Episode_Reward/orientation_reward": ("orientation_reward_weight",),
    "Episode_Reward/orientation_penalty": ("orientation_penalty_weight",),
    "Episode_Reward/low_altitude": ("low_altitude_penalty_weight",),
    "Episode_Reward/airborne": ("airborne_reward_weight",),
    "Episode_Reward/takeoff_height": ("takeoff_height_shaping_weight",),
    "Episode_Reward/z_position": ("z_position_reward_weight",),
    "Episode_Reward/xy_position_reward": ("xy_position_reward_weight",),
    "Episode_Reward/xy_position_penalty": ("xy_position_penalty_weight",),
}

ION_DIRECT_EXTRA_LABELS = {
    "positive_z_velocity": "vz target",
    "upward_shaping": "upward shape",
    "upward_shaping_miss_penalty": "down miss",
    "off_band_z_velocity": "vz off band",
    "idle": "idle pen",
    "idle_velocity_reward": "active vz",
    "x_drift": "x drift",
    "y_drift": "y drift",
    "x_drift_free": "x drift free",
    "y_drift_free": "y drift free",
    "jerky": "jerk",
    "smooth_motion": "smooth",
    "electricity": "electricity",
    "electricity_efficiency": "efficiency",
    "orientation_reward": "orient r",
    "orientation_penalty": "orient pen",
    "low_altitude": "low z",
    "airborne": "airborne",
    "takeoff_height": "AGL",
    "z_position": "z hold",
    "xy_position_reward": "xy hold r",
    "xy_position_penalty": "xy hold pen",
}


def _cfg_float(reward_cfg: dict[str, float | str | None], name: str) -> float | None:
    v = reward_cfg.get(name)
    if v is None or isinstance(v, str):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _episode_reward_plot_enabled(ep_key: str, reward_cfg: dict[str, float | str | None] | None) -> bool:
    """Plot/print Episode_Reward only when the run-arg weight for that term is not 0.0 (ION_VERTICAL_TAKEOFF_TRAIN.sh).

    If no gate mapping exists, keep the series. If the gate's weight key is missing from the header (old logs),
    keep the series. If all gate keys present and every numeric weight is 0.0, omit.
    """
    if reward_cfg is None:
        return True
    gate = _EPISODE_REWARD_CFG_GATE.get(ep_key)
    if gate is None:
        return True
    present = [name for name in gate if name in reward_cfg]
    if not present:
        return True
    for name in present:
        w = _cfg_float(reward_cfg, name)
        if w is None:
            return True
        if w != 0.0:
            return True
    return False


def _filter_episode_reward_keys_by_cfg(
    keys: list[str],
    reward_cfg: dict[str, float | str | None] | None,
) -> list[str]:
    if reward_cfg is None:
        return keys
    return [k for k in keys if _episode_reward_plot_enabled(k, reward_cfg)]


def _filter_block_text_for_zero_weight_episode_rewards(
    block: str,
    reward_cfg: dict[str, float | str | None] | None,
) -> str:
    """Drop Episode_Reward/* lines from printed log blocks when run-arg weight is 0.0."""
    if reward_cfg is None:
        return block
    lines_out: list[str] = []
    for line in block.splitlines():
        m = re.match(r"^\s*(Episode_Reward/[^\s:]+)\s*:\s*", line)
        if m:
            ep_key = m.group(1).strip()
            if not _episode_reward_plot_enabled(ep_key, reward_cfg):
                continue
        lines_out.append(line)
    out = "\n".join(lines_out)
    if block.endswith("\n") and out and not out.endswith("\n"):
        out += "\n"
    return out


def parse_header_ion_direct_reward_config(path: str) -> dict[str, float | str | None] | None:
    """Parse Run arguments block in training_progress_log.txt for Ion Vertical Takeoff Direct."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        head = content.split("Learning iteration")[0]
        out: dict[str, float | str | None] = {}
        for name in (
            "positive_z_velocity_reward_weight",
            "z_velocity_sharpness",
            "target_takeoff_speed",
            "target_velocity_tolerance",
            "upward_velocity_shaping_weight",
            "upward_velocity_shaping_miss_penalty_weight",
            "off_band_z_velocity_penalty_weight",
            "off_band_z_velocity_penalty_sharpness",
            "idle_velocity_penalty_weight",
            "idle_velocity_threshold",
            "idle_velocity_reward_weight",
            "drift_tolerance",
            "drift_penalty_cap_per_step",
            "x_drift_penalty",
            "y_drift_penalty",
            "x_drift_penalty_sharpness",
            "y_drift_penalty_sharpness",
            "x_drift_free_reward_weight",
            "y_drift_free_reward_weight",
            "jerky_penalty",
            "smooth_motion_reward_weight",
            "smooth_motion_reward_sharpness",
            "electricity_cost_penalty",
            "electricity_efficiency_reward_weight",
            "electricity_efficiency_sigma",
            "orientation_reward_weight",
            "orientation_penalty_weight",
            "orientation_penalty_sharpness",
            "orientation_penalty_cap_per_step",
            "low_altitude_penalty_threshold",
            "low_altitude_penalty_weight",
            "low_altitude_penalty_cap_per_step",
            "low_altitude_penalty_ramp",
            "airborne_reward_weight",
            "takeoff_height_shaping_weight",
            "takeoff_height_shaping_cap_m",
            "spawn_height",
            "z_position_reward_weight",
            "z_position_target",
            "z_position_sharpness",
            "xy_position_reward_weight",
            "xy_position_reward_sharpness",
            "xy_position_penalty_weight",
            "xy_position_penalty_tolerance",
            "xy_position_penalty_sharpness",
            "xy_position_penalty_cap_per_step",
            "drift_free_requires_airborne",
            "thrust_output_bias",
            "reward_scale",
            "decimation",
            "num_envs",
        ):
            m = re.search(rf"^\s*{re.escape(name)}\s*:\s*([\d.eE+-]+|None|True|False)\s*", head, re.MULTILINE)
            if m:
                g = m.group(1).strip()
                if g == "None":
                    out[name] = None
                elif g in ("True", "False"):
                    out[name] = g
                else:
                    try:
                        out[name] = float(g)
                    except ValueError:
                        out[name] = g
        return out if out else None
    except OSError:
        return None


def _reward_lists_from_ion_direct_config(
    reward_cfg: dict[str, float | str | None] | None,
) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    penalty: list[str] = []
    if reward_cfg is not None:
        for cfg_key, ep_key in ION_DIRECT_REWARD_POSITIVE_WEIGHTS:
            w = reward_cfg.get(cfg_key)
            if w is not None and not isinstance(w, str) and float(w) != 0.0:
                positive.append(ep_key)
        for cfg_key, ep_key in ION_DIRECT_REWARD_PENALTY_WEIGHTS:
            w = reward_cfg.get(cfg_key)
            if w is not None and not isinstance(w, str) and float(w) != 0.0:
                penalty.append(ep_key)
    return _ordered_unique(positive), _ordered_unique(penalty)


CATEGORY_MAIN_LEFT = ["Mean reward", "Mean episode length"]
CATEGORY_LOSSES = [
    "Mean value_function loss",
    "Mean surrogate loss",
    "Mean entropy loss",
    "Mean action noise std",
]


def generate_visual_analysis(
    log_dir: str,
    output_name: str = "training_progress_analysis.png",
    *,
    reward_penalty_ylim_focus_after: int | None = 300,
    reward_penalty_late_phase_frac: float = 0.22,
    plot_title: str | None = None,
    header_reward_config_parser=None,
    sidebar_extra_lines_fn=None,
) -> str | None:
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

    _parse_hdr = header_reward_config_parser or parse_header_ion_direct_reward_config
    reward_cfg = _parse_hdr(path)
    if reward_cfg is None:
        category_rewards_positive = [ep for _, ep in ION_DIRECT_REWARD_POSITIVE_WEIGHTS]
        category_rewards_penalties = [ep for _, ep in ION_DIRECT_REWARD_PENALTY_WEIGHTS]
    else:
        category_rewards_positive, category_rewards_penalties = _reward_lists_from_ion_direct_config(
            reward_cfg
        )
    category_rewards_positive, category_rewards_penalties = merge_ion_direct_episode_reward_keys_from_series(
        series, category_rewards_positive, category_rewards_penalties
    )
    category_rewards_positive = _filter_episode_reward_keys_by_cfg(category_rewards_positive, reward_cfg)
    category_rewards_penalties = _filter_episode_reward_keys_by_cfg(category_rewards_penalties, reward_cfg)

    def _keys_with_series_data(keys: list[str]) -> list[str]:
        """Only plot Episode_Reward keys that appear in this log (avoids empty legends when env did not log terms yet)."""
        out: list[str] = []
        for k in keys:
            if k not in series:
                continue
            vals = series[k]
            if vals and any(v is not None for v in vals):
                out.append(k)
        return out

    category_rewards_positive = _keys_with_series_data(category_rewards_positive)
    category_rewards_penalties = _keys_with_series_data(category_rewards_penalties)

    label_map = {**getattr(_v3, "EPISODE_REWARD_SHORT_LABEL", {}), **ION_DIRECT_EXTRA_LABELS}

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[0.9, 1.15, 1.15],
        width_ratios=[1.35, 1.35, 0.72],
        hspace=0.36,
        wspace=0.28,
    )
    fig.suptitle(
        plot_title or "Ion Vertical Takeoff (Direct) — training progress",
        fontsize=14,
        fontweight="bold",
    )

    _plot_style = {"marker": "o", "markersize": 2.5, "linewidth": 1.5}
    _cmap_colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    _line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    _palette = []
    for i in range(144):
        _palette.append((_cmap_colors[i % len(_cmap_colors)], _line_styles[i % len(_line_styles)]))

    def _ax_style(ax):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

    ax_main = fig.add_subplot(gs[0, 0])
    _idx = 0
    for k in CATEGORY_MAIN_LEFT:
        if k in series and series[k]:
            c, ls = _palette[_idx]
            ax_main.plot(iterations, series[k], label=k, color=c, linestyle=ls, **_plot_style)
            _idx += 1
    ax_main.set_title("Main metrics", fontsize=10, fontweight="bold")
    _ax_style(ax_main)
    ax_main.set_xlabel("Iteration", fontsize=9)
    ax_main2 = ax_main.twinx()
    _has_r = False
    if "Iteration time" in series and series["Iteration time"]:
        c, ls = _palette[_idx]
        _idx += 1
        ax_main2.plot(
            iterations, series["Iteration time"], label="iter time (s)", color=c, linestyle=ls, **_plot_style
        )
        _has_r = True
    if "Computation" in series and series["Computation"]:
        c, ls = _palette[_idx]
        comp = [x * 1e-5 for x in series["Computation"]]
        ax_main2.plot(
            iterations, comp, label="throughput (×10⁵ steps/s)", color=c, linestyle=ls, **_plot_style
        )
        _has_r = True
    if _has_r:
        ax_main2.set_ylabel("iter time / throughput", fontsize=8, color="gray")
        ax_main2.tick_params(axis="y", labelcolor="gray")
        ax_main2.legend(loc="upper right", fontsize=7, framealpha=0.92)
    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.92)

    ax_loss = fig.add_subplot(gs[0, 1])
    for idx, k in enumerate(CATEGORY_LOSSES):
        if k in series and series[k]:
            c, ls = _palette[idx]
            ax_loss.plot(iterations, series[k], label=k.replace("Mean ", ""), color=c, linestyle=ls, **_plot_style)
    ax_loss.set_title("Losses", fontsize=10, fontweight="bold")
    ax_loss.legend(loc="upper right", fontsize=8, framealpha=0.92)
    _ax_style(ax_loss)
    ax_loss.set_xlabel("Iteration", fontsize=9)

    def _plot_pos(ax, *, with_legend: bool) -> None:
        for idx, k in enumerate(category_rewards_positive):
            if k in series and series[k]:
                tail = k.replace("Episode_Reward/", "")
                label = label_map.get(tail, tail.replace("_", " "))
                c, ls = _palette[idx]
                ax.plot(iterations, series[k], label=label, color=c, linestyle=ls, **_plot_style)
        if with_legend and category_rewards_positive:
            ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.92)
        _ax_style(ax)
        ax.set_ylabel("Reward", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=9)

    ax_pos = fig.add_subplot(gs[1, 0:2])
    _plot_pos(ax_pos, with_legend=True)
    _fp = reward_penalty_ylim_focus_after
    _note = ""
    if _fp is not None and _fp >= 0:
        _lim = _episode_component_ylim_from_tail(
            iterations,
            series,
            category_rewards_positive,
            focus_after_iteration=_fp,
            late_phase_frac=reward_penalty_late_phase_frac,
            include_last_values=True,
        )
        if _lim is not None:
            ax_pos.set_ylim(_lim)
            _note = " — y: late window + quantiles + last points"
    ax_pos.set_title("Rewards (positive)" + _note, fontsize=10, fontweight="bold")

    ax_pen = fig.add_subplot(gs[2, 0:2])
    for idx, k in enumerate(category_rewards_penalties):
        if k in series and series[k]:
            vals = series[k]
            plot_vals = [-abs(v) for v in vals]
            tail = k.replace("Episode_Reward/", "")
            label = label_map.get(tail, tail.replace("_", " "))
            c, ls = _palette[idx]
            ax_pen.plot(iterations, plot_vals, label=label, color=c, linestyle=ls, **_plot_style)
    ax_pen.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    _note_p = ""
    if _fp is not None and _fp >= 0:
        _limp = _episode_component_ylim_from_tail(
            iterations,
            series,
            category_rewards_penalties,
            focus_after_iteration=_fp,
            late_phase_frac=reward_penalty_late_phase_frac,
            penalty_negative_display=True,
            include_last_values=True,
        )
        if _limp is not None:
            lo, hi = _limp
            ax_pen.set_ylim(min(lo, 0.0), max(hi, 0.0))
            _note_p = " — y: late window + quantiles + last points"
    ax_pen.set_title("Penalties / costs (negative)" + _note_p, fontsize=10, fontweight="bold")
    if category_rewards_penalties:
        ax_pen.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.92)
    _ax_style(ax_pen)
    ax_pen.set_ylabel("Penalty", fontsize=9)
    ax_pen.set_xlabel("Iteration", fontsize=9)

    ax_cfg = fig.add_subplot(gs[0:3, 2])
    lines = ["Config (run args)", ""]
    if reward_cfg:
        lines.extend(
            [
                "Vz target / shaping",
                f"  target_vz: {reward_cfg.get('target_takeoff_speed', '—')}",
                f"  vz_tol: {reward_cfg.get('target_velocity_tolerance', '—')}",
                f"  pos_z_rew_w: {reward_cfg.get('positive_z_velocity_reward_weight', '—')}",
                f"  z_vel_sharp: {reward_cfg.get('z_velocity_sharpness', '—')}",
                f"  upward_shape_w: {reward_cfg.get('upward_velocity_shaping_weight', '—')}",
                f"  upward_miss_pen_w: {reward_cfg.get('upward_velocity_shaping_miss_penalty_weight', '—')}",
                f"  off_band_z_pen_w: {reward_cfg.get('off_band_z_velocity_penalty_weight', '—')}",
                "",
                "Idle / drift",
                f"  idle_pen_w: {reward_cfg.get('idle_velocity_penalty_weight', '—')}",
                f"  idle_thresh: {reward_cfg.get('idle_velocity_threshold', '—')}",
                f"  idle_rew_w: {reward_cfg.get('idle_velocity_reward_weight', '—')}",
                f"  drift_tol: {reward_cfg.get('drift_tolerance', '—')}",
                f"  x_pen: {reward_cfg.get('x_drift_penalty', '—')}",
                f"  y_pen: {reward_cfg.get('y_drift_penalty', '—')}",
                f"  x_drift_free: {reward_cfg.get('x_drift_free_reward_weight', '—')}",
                f"  y_drift_free: {reward_cfg.get('y_drift_free_reward_weight', '—')}",
                "",
                "Smooth / power / orient",
                f"  jerky: {reward_cfg.get('jerky_penalty', '—')}",
                f"  smooth_rew: {reward_cfg.get('smooth_motion_reward_weight', '—')}",
                f"  elec_cost: {reward_cfg.get('electricity_cost_penalty', '—')}",
                f"  elec_eff: {reward_cfg.get('electricity_efficiency_reward_weight', '—')}",
                f"  orient_rew: {reward_cfg.get('orientation_reward_weight', '—')}",
                f"  orient_pen: {reward_cfg.get('orientation_penalty_weight', '—')}",
                "",
                "Altitude",
                f"  low_z_thresh: {reward_cfg.get('low_altitude_penalty_threshold', '—')}",
                f"  low_z_pen: {reward_cfg.get('low_altitude_penalty_weight', '—')}",
                f"  low_z_ramp: {reward_cfg.get('low_altitude_penalty_ramp', '—')}",
                f"  airborne: {reward_cfg.get('airborne_reward_weight', '—')}",
                f"  thrust_bias: {reward_cfg.get('thrust_output_bias', '—')}",
                f"  reward_scale: {reward_cfg.get('reward_scale', '—')}",
            ]
        )
    if reward_cfg and sidebar_extra_lines_fn is not None:
        extra = sidebar_extra_lines_fn(reward_cfg)
        if extra:
            lines.extend(extra)
    lines = _filter_config_sidebar_lines(lines)
    ax_cfg.text(
        0.04,
        0.98,
        "\n".join(lines),
        transform=ax_cfg.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, pad=0.75),
    )
    ax_cfg.set_axis_off()

    out_path = os.path.join(log_dir, output_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Print training_progress_log.txt for Isaac-Ion-Vertical-Takeoff-Direct-v0 and generate visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/ion_vertical_takeoff_direct/YYYY-MM-DD_HH-MM-SS",
    )
    parser.add_argument("--last", type=int, default=None, metavar="N", help="Print only the last N iteration blocks")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate the visual analysis PNG")
    parser.add_argument("--output", type=str, default="training_progress_analysis.png", help="Output PNG filename")
    parser.add_argument("--plot-only", action="store_true", help="Only generate the visual analysis PNG")
    parser.add_argument(
        "--reward-penalty-ylim-focus-after",
        type=int,
        default=300,
        metavar="ITER",
        help="Rewards/penalties y-limits: iter ≥ max(ITER, late-phase start). Default 300. Use -1 for full autoscale.",
    )
    parser.add_argument(
        "--reward-penalty-ylim-late-frac",
        type=float,
        default=0.22,
        metavar="FRAC",
        help="Late-phase fraction of logged iteration span for y scaling (default 0.22).",
    )
    args = parser.parse_args()

    path = os.path.join(args.log_dir, "training_progress_log.txt")
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    reward_cfg_for_print = parse_header_ion_direct_reward_config(path)

    if not args.plot_only:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"(?=^################################################################################\n)", content, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]
        if args.last is not None and args.last > 0:
            blocks = blocks[-args.last :]
        for block in blocks:
            filtered = _filter_block_text_for_zero_weight_episode_rewards(block, reward_cfg_for_print)
            print(_strip_ansi(filtered))
            print()

    if not args.no_plot:
        _focus = None if args.reward_penalty_ylim_focus_after < 0 else args.reward_penalty_ylim_focus_after
        out_path = generate_visual_analysis(
            args.log_dir,
            args.output,
            reward_penalty_ylim_focus_after=_focus,
            reward_penalty_late_phase_frac=float(args.reward_penalty_ylim_late_frac),
        )
        if out_path:
            print(f"[INFO] Visual analysis saved to: {out_path}", file=sys.stderr)
        else:
            print("[WARNING] Could not generate visual analysis.", file=sys.stderr)


if __name__ == "__main__":
    main()

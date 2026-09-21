#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress analysis for G1 Standing IK (Isaac-G1-Standing-IK-Direct-v0).
# Layout aligned with print_g1_locomotion_v3_progress_log.py: wide figure, GridSpec,
# positive rewards (full + zoomed y), penalties, main/loss, config sidebar.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_g1_standing_ik_progress_log.py <log_dir>
#
# Options:
#   --last N, --no-plot, --output NAME, --plot-only
#   --reward-penalty-ylim-focus-after ITER   (default 300; -1 = full autoscale)
#   --reward-penalty-ylim-late-frac FRAC    (default 0.22)
#
# Plots omit Episode_Reward/* terms whose controlling weight/scale in the log header is 0.0
# (and jerk/smooth when both relevant scales are 0). Unknown keys from merge still plot.

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
parse_training_progress_log = _v3.parse_training_progress_log
merge_episode_reward_keys_from_series = _v3.merge_episode_reward_keys_from_series
_episode_component_ylim_from_tail = _v3._episode_component_ylim_from_tail
_filter_config_sidebar_lines = _v3._filter_config_sidebar_lines


# Config key -> Episode_Reward (used to seed plot lists; zero weights filtered later)
STANDING_REWARD_POSITIVE_WEIGHTS = (
    ("up_weight", "Episode_Reward/up_reward"),
    ("heading_weight", "Episode_Reward/heading_reward"),
    ("alive_reward_scale", "Episode_Reward/alive_reward"),
    ("progress_reward_multiplier", "Episode_Reward/progress_reward"),
    ("hand_reward_weight", "Episode_Reward/hand_position_reward"),
    ("smooth_motion_reward_weight", "Episode_Reward/smooth_reward"),
    ("ik_joint_reward_weight", "Episode_Reward/ik_joint_reward"),
    ("ee_position_reward_weight", "Episode_Reward/ee_position_reward"),
    ("x_drift_reward_weight", "Episode_Reward/x_drift_reward"),
    ("y_drift_reward_weight", "Episode_Reward/y_drift_reward"),
)

STANDING_REWARD_PENALTY_WEIGHTS = (
    ("actions_cost_scale", "Episode_Reward/actions_cost"),
    ("energy_cost_scale", "Episode_Reward/electricity_cost"),
    ("joint_limit_penalty_weight", "Episode_Reward/joint_limit_cost"),
    ("xy_drift_penalty_weight", "Episode_Reward/xy_drift_penalty"),
    ("x_drift_penalty_weight", "Episode_Reward/x_drift_penalty"),
    ("y_drift_penalty_weight", "Episode_Reward/y_drift_penalty"),
    ("ik_joint_penalty_weight", "Episode_Reward/ik_joint_penalty"),
    ("ee_position_penalty_weight", "Episode_Reward/ee_position_penalty"),
    ("ik_ee_smoothness_penalty_weight", "Episode_Reward/ik_ee_smoothness_penalty"),
    # Walking / shared V1 cfg (often 0 on standing IK; omitted from plots when 0)
    ("z_orientation_penalty_weight", "Episode_Reward/z_orientation_penalty"),
    ("lin_vel_z_penalty_weight", "Episode_Reward/lin_vel_z_penalty"),
    ("ang_vel_xy_penalty_weight", "Episode_Reward/ang_vel_xy_penalty"),
    ("flat_orientation_penalty_weight", "Episode_Reward/flat_orientation_penalty"),
)

# Episode_Reward/* → cfg name(s); episode is shown if any listed weight is non-zero.
# Special: __jerk__, __smooth_reward__ handled in code.
_EPISODE_REWARD_CFG_GATE: dict[str, tuple[str, ...]] = {
    "Episode_Reward/up_reward": ("up_weight",),
    "Episode_Reward/heading_reward": ("heading_weight",),
    "Episode_Reward/alive_reward": ("alive_reward_scale",),
    "Episode_Reward/progress_reward": ("progress_reward_multiplier",),
    "Episode_Reward/hand_position_reward": ("hand_reward_weight",),
    "Episode_Reward/smooth_reward": ("__smooth_reward__",),
    "Episode_Reward/ik_joint_reward": ("ik_joint_reward_weight",),
    "Episode_Reward/ee_position_reward": ("ee_position_reward_weight",),
    "Episode_Reward/x_drift_reward": ("x_drift_reward_weight",),
    "Episode_Reward/y_drift_reward": ("y_drift_reward_weight",),
    "Episode_Reward/actions_cost": ("actions_cost_scale",),
    "Episode_Reward/electricity_cost": ("energy_cost_scale",),
    "Episode_Reward/joint_limit_cost": ("joint_limit_penalty_weight",),
    "Episode_Reward/xy_drift_penalty": ("xy_drift_penalty_weight",),
    "Episode_Reward/x_drift_penalty": ("x_drift_penalty_weight",),
    "Episode_Reward/y_drift_penalty": ("y_drift_penalty_weight",),
    "Episode_Reward/ik_joint_penalty": ("ik_joint_penalty_weight",),
    "Episode_Reward/ee_position_penalty": ("ee_position_penalty_weight",),
    "Episode_Reward/ik_ee_smoothness_penalty": ("ik_ee_smoothness_penalty_weight",),
    "Episode_Reward/jerk_cost": ("__jerk__",),
    "Episode_Reward/dof_at_limit_cost": ("dof_at_limit_cost_scale",),
    "Episode_Reward/death_cost": ("death_cost",),
    "Episode_Reward/z_orientation_penalty": ("z_orientation_penalty_weight",),
    "Episode_Reward/lin_vel_z_penalty": ("lin_vel_z_penalty_weight",),
    "Episode_Reward/ang_vel_xy_penalty": ("ang_vel_xy_penalty_weight",),
    "Episode_Reward/flat_orientation_penalty": ("flat_orientation_penalty_weight",),
}

STANDING_EXTRA_LABELS = {
    "ik_joint_reward": "IK joints",
    "ik_joint_penalty": "IK joints pen",
    "ee_position_reward": "EE target",
    "ee_position_penalty": "EE pen",
    "x_drift_reward": "x drift r",
    "y_drift_reward": "y drift r",
    "ik_ee_smoothness_penalty": "EE motion pen",
    "hand_position_reward": "hand default tgt",
    "z_orientation_penalty": "yaw pen",
    "lin_vel_z_penalty": "lin vz pen",
    "ang_vel_xy_penalty": "ang vxy pen",
    "flat_orientation_penalty": "tilt pen",
    "dof_at_limit_cost": "dof @ limit",
    "death_cost": "death",
}


def _ordered_unique(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _cfg_float(reward_cfg: dict[str, float | str | None], name: str) -> float | None:
    v = reward_cfg.get(name)
    if v is None or isinstance(v, str):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _episode_reward_plot_enabled(ep_key: str, reward_cfg: dict[str, float | str | None] | None) -> bool:
    """Hide Episode_Reward rows whose controlling cfg weight/scale is 0.0 (or missing numeric)."""
    if reward_cfg is None:
        return True
    gate = _EPISODE_REWARD_CFG_GATE.get(ep_key)
    if gate is None:
        return True
    if gate == ("__jerk__",):
        if (
            "action_rate_penalty_scale" not in reward_cfg
            and "joint_acceleration_penalty_scale" not in reward_cfg
        ):
            return True
        ar = _cfg_float(reward_cfg, "action_rate_penalty_scale")
        ja = _cfg_float(reward_cfg, "joint_acceleration_penalty_scale")
        ar = 0.0 if ar is None else ar
        ja = 0.0 if ja is None else ja
        return ar != 0.0 or ja != 0.0
    if gate == ("__smooth_reward__",):
        if "smooth_motion_reward_weight" not in reward_cfg and "smooth_motion_joint_accel_reward_weight" not in reward_cfg:
            return True
        sm = _cfg_float(reward_cfg, "smooth_motion_reward_weight")
        sja = _cfg_float(reward_cfg, "smooth_motion_joint_accel_reward_weight")
        sm = 0.0 if sm is None else sm
        sja = 0.0 if sja is None else sja
        return sm != 0.0 or sja != 0.0
    for name in gate:
        if name not in reward_cfg:
            return True
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


def parse_header_standing_ik_reward_config(path: str) -> dict[str, float | str | None] | None:
    """Parse log header for G1 standing / standing IK reward-related scalars."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        head = content.split("Learning iteration")[0]
        out: dict[str, float | str | None] = {}
        for name in (
            "up_weight",
            "heading_weight",
            "alive_reward_scale",
            "progress_reward_multiplier",
            "actions_cost_scale",
            "energy_cost_scale",
            "termination_height",
            "decimation",
            "action_rate_penalty_scale",
            "joint_acceleration_penalty_scale",
            "smooth_motion_reward_weight",
            "smooth_motion_sigma",
            "smooth_motion_joint_accel_reward_weight",
            "smooth_motion_joint_accel_sigma",
            "smoothness_skip_first_episode_step",
            "joint_limit_penalty_weight",
            "hand_reward_weight",
            "hand_reward_sigma",
            "xy_drift_tolerance",
            "xy_drift_penalty_weight",
            "x_drift_tolerance",
            "x_drift_penalty_weight",
            "y_drift_tolerance",
            "y_drift_penalty_weight",
            "ik_joint_reward_weight",
            "ik_joint_reward_sigma",
            "ik_joint_penalty_weight",
            "ik_joint_penalty_threshold",
            "ik_joint_tolerance_percent",
            "ee_position_reward_weight",
            "ee_position_penalty_weight",
            "ee_position_reward_sigma",
            "ik_ee_smoothness_penalty_weight",
            "ik_target_obs_scale",
            "death_cost",
            "dof_at_limit_cost_scale",
            "z_orientation_penalty_weight",
            "lin_vel_z_penalty_weight",
            "ang_vel_xy_penalty_weight",
            "flat_orientation_penalty_weight",
            "target_z_orientation",
            "z_orientation_tolerance",
            "lin_vel_z_tolerance",
            "flat_orientation_tolerance",
            "x_drift_reward_weight",
            "y_drift_reward_weight",
            "x_drift_reward_sigma",
            "y_drift_reward_sigma",
            "joints_data",
            "npz_only_for_limits",
            "npz_joint_limit_extend_deg",
            "npz_joint_limit_extend_deg_non_leg",
            "npz_limits_exclude_four_ankles",
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
        m_rp = re.search(r"^\s*reward_profile\s*:\s*(\S+)\s*", head, re.MULTILINE)
        if m_rp:
            out["reward_profile"] = m_rp.group(1).strip()
        m_ph = re.search(r"^\s*include_reward_profile_phase_obs\s*:\s*(True|False)\s*", head, re.MULTILINE)
        if m_ph:
            out["include_reward_profile_phase_obs"] = m_ph.group(1).strip()
        for _ik_st_key in ("staged_ik_start_env_steps", "staged_ik_ramp_env_steps"):
            m_ik = re.search(rf"^\s*{re.escape(_ik_st_key)}\s*:\s*(\d+)\s*", head, re.MULTILINE)
            if m_ik:
                out[_ik_st_key] = float(m_ik.group(1))

        m_box_min = re.search(r"^\s*ik_target_box_min\s*:\s*(\(.+\))\s*", head, re.MULTILINE)
        if m_box_min:
            out["ik_target_box_min"] = m_box_min.group(1).strip()
        m_box_max = re.search(r"^\s*ik_target_box_max\s*:\s*(\(.+\))\s*", head, re.MULTILINE)
        if m_box_max:
            out["ik_target_box_max"] = m_box_max.group(1).strip()
        m_jd = re.search(r"^\s*joints_data\s*:\s*(.+)$", head, re.MULTILINE)
        if m_jd:
            out["joints_data"] = m_jd.group(1).strip()
        return out if out else None
    except OSError:
        return None


def _reward_lists_from_standing_ik_config(
    reward_cfg: dict[str, float | str | None] | None,
) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    penalty: list[str] = []
    if reward_cfg is not None:
        for cfg_key, ep_key in STANDING_REWARD_POSITIVE_WEIGHTS:
            w = reward_cfg.get(cfg_key)
            if w is not None and not isinstance(w, str) and float(w) != 0.0:
                positive.append(ep_key)
        for cfg_key, ep_key in STANDING_REWARD_PENALTY_WEIGHTS:
            w = reward_cfg.get(cfg_key)
            if w is not None and not isinstance(w, str) and float(w) != 0.0:
                penalty.append(ep_key)
        ar = float(reward_cfg.get("action_rate_penalty_scale") or 0)
        ja = float(reward_cfg.get("joint_acceleration_penalty_scale") or 0)
        if ar != 0 or ja != 0:
            penalty.append("Episode_Reward/jerk_cost")
        sm = float(reward_cfg.get("smooth_motion_reward_weight") or 0)
        sja = float(reward_cfg.get("smooth_motion_joint_accel_reward_weight") or 0)
        if (sm != 0 or sja != 0) and "Episode_Reward/smooth_reward" not in positive:
            positive.append("Episode_Reward/smooth_reward")
        dlim = float(reward_cfg.get("dof_at_limit_cost_scale") or 0)
        if dlim != 0:
            penalty.append("Episode_Reward/dof_at_limit_cost")
        dc = reward_cfg.get("death_cost")
        if dc is not None and not isinstance(dc, str) and float(dc) != 0.0:
            penalty.append("Episode_Reward/death_cost")
    return _ordered_unique(positive), _ordered_unique(penalty)


CATEGORY_MAIN_LEFT = ["Mean reward"]
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
    plot_title: str = "G1 Standing IK — training progress",
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

    reward_cfg = parse_header_standing_ik_reward_config(path)
    if reward_cfg is None:
        category_rewards_positive = [ep for _, ep in STANDING_REWARD_POSITIVE_WEIGHTS]
        category_rewards_penalties = (
            [ep for _, ep in STANDING_REWARD_PENALTY_WEIGHTS]
            + [
                "Episode_Reward/jerk_cost",
                "Episode_Reward/dof_at_limit_cost",
                "Episode_Reward/death_cost",
            ]
        )
    else:
        category_rewards_positive, category_rewards_penalties = _reward_lists_from_standing_ik_config(
            reward_cfg
        )
    category_rewards_positive, category_rewards_penalties = merge_episode_reward_keys_from_series(
        series, category_rewards_positive, category_rewards_penalties
    )
    category_rewards_positive = _filter_episode_reward_keys_by_cfg(category_rewards_positive, reward_cfg)
    category_rewards_penalties = _filter_episode_reward_keys_by_cfg(category_rewards_penalties, reward_cfg)

    label_map = {**getattr(_v3, "EPISODE_REWARD_SHORT_LABEL", {}), **STANDING_EXTRA_LABELS}

    # Wider canvas than legacy 12×10 (match V3 spirit: 18×12 → standing 20×12)
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[0.85, 1.0, 1.0, 1.0],
        width_ratios=[1.35, 1.35, 0.72],
        hspace=0.38,
        wspace=0.28,
    )
    fig.suptitle(plot_title, fontsize=14, fontweight="bold")

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
        if with_legend:
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

    ax_pos_z = fig.add_subplot(gs[2, 0:2])
    _plot_pos(ax_pos_z, with_legend=False)
    _note_z = ""
    if _fp is not None and _fp >= 0:
        _limz = _episode_component_ylim_from_tail(
            iterations,
            series,
            category_rewards_positive,
            focus_after_iteration=_fp,
            late_phase_frac=reward_penalty_late_phase_frac,
            include_last_values=False,
        )
        if _limz is not None:
            ax_pos_z.set_ylim(_limz)
            _note_z = " — y: late window + quantiles only (zoomed)"
    ax_pos_z.set_title("Rewards (positive) — zoomed" + _note_z + " (legend ↑)", fontsize=10, fontweight="bold")

    ax_pen = fig.add_subplot(gs[3, 0:2])
    for idx, k in enumerate(category_rewards_penalties):
        if k in series and series[k]:
            vals = series[k]
            if "death_cost" in k:
                plot_vals = vals
            else:
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
    ax_pen.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.92)
    _ax_style(ax_pen)
    ax_pen.set_ylabel("Penalty", fontsize=9)
    ax_pen.set_xlabel("Iteration", fontsize=9)

    ax_cfg = fig.add_subplot(gs[0:4, 2])
    lines = ["Config (log header)", ""]
    if reward_cfg:
        lines.extend(
            [
                "Smooth / jerk",
                f"  action_rate: {reward_cfg.get('action_rate_penalty_scale', '—')}",
                f"  joint_accel: {reward_cfg.get('joint_acceleration_penalty_scale', '—')}",
                f"  smooth_rew: {reward_cfg.get('smooth_motion_reward_weight', '—')}",
                f"  smooth_sigma: {reward_cfg.get('smooth_motion_sigma', '—')}",
                f"  smooth_joint_Δvel_rew: {reward_cfg.get('smooth_motion_joint_accel_reward_weight', '—')}",
                f"  smooth_joint_Δvel_sigma: {reward_cfg.get('smooth_motion_joint_accel_sigma', '—')}",
                f"  skip_first_step: {reward_cfg.get('smoothness_skip_first_episode_step', '—')}",
                "",
                "IK / EE",
                f"  ik_joint_rew: {reward_cfg.get('ik_joint_reward_weight', '—')}",
                f"  ik_joint_pen: {reward_cfg.get('ik_joint_penalty_weight', '—')}",
                f"  ik_tol_pct: {reward_cfg.get('ik_joint_tolerance_percent', '—')}",
                f"  ik_sigma: {reward_cfg.get('ik_joint_reward_sigma', '—')}",
                f"  ee_rew: {reward_cfg.get('ee_position_reward_weight', '—')}",
                f"  ee_pen: {reward_cfg.get('ee_position_penalty_weight', '—')}",
                f"  ee_sigma: {reward_cfg.get('ee_position_reward_sigma', '—')}",
                f"  ee_motion_pen: {reward_cfg.get('ik_ee_smoothness_penalty_weight', '—')}",
                "",
                "Hand (default target)",
                f"  hand_rew: {reward_cfg.get('hand_reward_weight', '—')}",
                f"  hand_sigma: {reward_cfg.get('hand_reward_sigma', '—')}",
                "",
                "Drift",
                f"  x_tol: {reward_cfg.get('x_drift_tolerance', '—')}",
                f"  x_pen: {reward_cfg.get('x_drift_penalty_weight', '—')}",
                f"  y_tol: {reward_cfg.get('y_drift_tolerance', '—')}",
                f"  y_pen: {reward_cfg.get('y_drift_penalty_weight', '—')}",
                f"  xy_pen: {reward_cfg.get('xy_drift_penalty_weight', '—')}",
                f"  x_drift_rew: {reward_cfg.get('x_drift_reward_weight', '—')}",
                f"  y_drift_rew: {reward_cfg.get('y_drift_reward_weight', '—')}",
                "",
                "Standing IK V2 (if present)",
                f"  reward_profile: {reward_cfg.get('reward_profile', '—')}",
                f"  staged_ik_start_env_steps: {reward_cfg.get('staged_ik_start_env_steps', '—')}",
                f"  staged_ik_ramp_env_steps: {reward_cfg.get('staged_ik_ramp_env_steps', '—')}",
                f"  include_reward_profile_phase_obs: {reward_cfg.get('include_reward_profile_phase_obs', '—')}",
                "",
                "Base / env",
                f"  up: {reward_cfg.get('up_weight', '—')}",
                f"  heading: {reward_cfg.get('heading_weight', '—')}",
                f"  alive: {reward_cfg.get('alive_reward_scale', '—')}",
                f"  progress_mult: {reward_cfg.get('progress_reward_multiplier', '—')}",
                f"  actions_cost: {reward_cfg.get('actions_cost_scale', '—')}",
                f"  energy: {reward_cfg.get('energy_cost_scale', '—')}",
                f"  joint_limit_pen: {reward_cfg.get('joint_limit_penalty_weight', '—')}",
                f"  term_height: {reward_cfg.get('termination_height', '—')}",
                f"  decimation: {reward_cfg.get('decimation', '—')}",
                f"  ik_obs_scale: {reward_cfg.get('ik_target_obs_scale', '—')}",
                f"  ik_box_min: {reward_cfg.get('ik_target_box_min', '—')}",
                f"  ik_box_max: {reward_cfg.get('ik_target_box_max', '—')}",
                f"  joints_data: {reward_cfg.get('joints_data', '—')}",
            ]
        )
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
        description="Print training_progress_log.txt for g1_standing_ik and generate wide-layout visual analysis."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/g1_standing_ik/YYYY-MM-DD_HH-MM-SS",
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
    parser.add_argument(
        "--plot-title",
        type=str,
        default="G1 Standing IK — training progress",
        help="Figure suptitle for the progress PNG.",
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
        _focus = None if args.reward_penalty_ylim_focus_after < 0 else args.reward_penalty_ylim_focus_after
        out_path = generate_visual_analysis(
            args.log_dir,
            args.output,
            reward_penalty_ylim_focus_after=_focus,
            reward_penalty_late_phase_frac=float(args.reward_penalty_ylim_late_frac),
            plot_title=args.plot_title,
        )
        if out_path:
            print(f"[INFO] Visual analysis saved to: {out_path}", file=sys.stderr)
        else:
            print("[WARNING] Could not generate visual analysis.", file=sys.stderr)


if __name__ == "__main__":
    main()

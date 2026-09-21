#!/usr/bin/env python3
"""Training-progress analysis for Rover Idle → Forward transition task.

Task: ``Isaac-Rover-DiskWorld-IdleToForward-v0`` (experiment ``rover_disk_world_idle_to_forward``)

Based on ``print_rover_disk_world_progress_log.py`` but tuned for staged rewards:

* ``idle_phase_stability`` — hold near idle at episode start
* ``forward_angular_progress`` / ``forward_speed_tracking`` / ``heading_alignment`` — ramped in cfg
* Penalties / paired smoothness bonuses: ``action_rate_penalty``,
  ``motion_jerk_penalty`` / ``motion_jerk_smoothness_bonus``,
  ``heading_oscillation_penalty`` / ``heading_stability_bonus``,
  ``tilt_penalty``, ``y_drift_penalty``, ``alive_bonus``, ``goal_bonus``

Parses ``training_progress_log.txt`` and writes ``training_progress_analysis.png`` (five panels + sidebar).
No physparam merge panels (this task does not log ``Episode_RoverPhysparam/*``).

Usage
-----
  python scripts/reinforcement_learning/rsl_rl/print_rover_idle_to_forward_progress_log.py <log_dir>
  python scripts/reinforcement_learning/rsl_rl/print_rover_idle_to_forward_progress_log.py \\
      logs/rsl_rl/rover_disk_world_idle_to_forward/2026-04-23_14-00-00 --plot-only
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Same-directory helper module (rsl_rl/).
import print_rover_disk_world_progress_log as _rover_base


def parse_header_config(log_path: str) -> dict[str, str]:
    """Header fields for sidebar (extends generic rover header keys)."""
    cfg = _rover_base.parse_header_config(log_path)
    try:
        head = _rover_base._pre_iteration_content(log_path)
    except OSError:
        return cfg
    for k in ("starting_policy_path",):
        m = re.search(rf"^\s*{re.escape(k)}\s*:\s*(.+)$", head, re.MULTILINE)
        if m:
            cfg[k] = m.group(1).strip()
    return cfg


def _idle_to_forward_sidebar_doc() -> list[str]:
    return [
        "",
        "══ Idle → Forward (env cfg) ══",
        "  Phase A: idle_phase_stability (first ~40 policy steps / ep)",
        "  Phase B: ramp forward_angular_progress,",
        "           forward_speed_tracking, heading (~160 steps)",
        "  (exact idle_steps / ramp_steps in rover_idle_to_forward_env_cfg)",
    ]


def generate_plot(
    blocks: list[tuple[int, dict[str, float]]],
    cfg: dict[str, str],
    output_path: str,
    reward_ylim_after: int = 200,
) -> None:
    """Five-panel PNG + sidebar (no physparam extension)."""
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
    n_main = 5
    fig_h = 19.0

    fig = plt.figure(figsize=(22, fig_h))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4.2, 1.0], wspace=0.04)
    gs_panels = gridspec.GridSpecFromSubplotSpec(n_main, 1, subplot_spec=gs_outer[0], hspace=0.40)

    ax1 = fig.add_subplot(gs_panels[0])
    ax2 = fig.add_subplot(gs_panels[1])
    ax3 = fig.add_subplot(gs_panels[2])
    ax4 = fig.add_subplot(gs_panels[3])
    ax5 = fig.add_subplot(gs_panels[4])

    fig.suptitle(
        f"Rover Idle → Forward — Training Progress  (iter {blocks[0][0]}–{max_iter})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # ── Panel 1: OVERVIEW ─────────────────────────────────────────────
    ax1.set_title("Overview", fontsize=10, loc="left")
    _rover_base._plot_line(ax1, blocks, "Train/mean_reward", "Mean reward", color="#1f77b4")
    ax1.set_ylabel("Mean reward", color="#1f77b4")
    ax1_r = ax1.twinx()
    _rover_base._plot_line(
        ax1_r,
        blocks,
        "Train/mean_episode_length",
        "Episode length (steps)",
        color="#ff7f0e",
        linestyle="--",
    )
    ax1_r.set_ylabel("Episode length (steps)", color="#ff7f0e")
    _rover_base._legend_if_any(ax1, loc="upper left", fontsize=8)
    _rover_base._legend_if_any(ax1_r, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: STARTUP + FORWARD REWARDS ─────────────────────────────
    ax2.set_title("Startup & Forward Rewards", fontsize=10, loc="left")
    _REW = "Episode_Reward/"
    _rover_base._plot_line(ax2, blocks, _REW + "idle_phase_stability", "idle_phase_stability", color="#8c564b")
    _rover_base._plot_line(
        ax2, blocks, _REW + "forward_angular_progress", "forward_angular_progress (ramped)", color="#2ca02c", linestyle="-"
    )
    _rover_base._plot_line(
        ax2, blocks, _REW + "forward_speed_tracking", "speed_tracking (ramped)", color="#17becf", linestyle="--"
    )
    _rover_base._plot_line(
        ax2, blocks, _REW + "heading_alignment", "heading_alignment (ramped)", color="#9467bd", linestyle="-."
    )
    _rover_base._plot_line(ax2, blocks, _REW + "goal_bonus", "goal_bonus (on success)", color="#d62728", linestyle=":")
    ax2.set_ylabel("Episode reward contribution")
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    _rover_base._legend_if_any(ax2, fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    if reward_ylim_after >= 0:
        _rover_base._focus_ylim(
            ax2,
            blocks,
            [
                _REW + k
                for k in (
                    "idle_phase_stability",
                    "forward_angular_progress",
                    "forward_speed_tracking",
                    "heading_alignment",
                    "goal_bonus",
                )
            ],
            reward_ylim_after,
        )

    # ── Panel 3: PENALTIES / ALIVE ─────────────────────────────────────
    ax3.set_title("Penalties, jerk/heading pairs, radial-Z align & Alive", fontsize=10, loc="left")
    _rover_base._plot_line(ax3, blocks, _REW + "action_rate_penalty", "action_rate_penalty", color="#e377c2")
    _rover_base._plot_line(ax3, blocks, _REW + "motion_jerk_penalty", "motion_jerk_penalty", color="#17becf", linestyle=":")
    _rover_base._plot_line(
        ax3, blocks, _REW + "motion_jerk_smoothness_bonus", "motion_jerk_smoothness_bonus", color="#17becf", linestyle="-"
    )
    _rover_base._plot_line(
        ax3, blocks, _REW + "heading_oscillation_penalty", "heading_oscillation_penalty", color="#7f7f7f", linestyle="-."
    )
    _rover_base._plot_line(
        ax3, blocks, _REW + "heading_stability_bonus", "heading_stability_bonus", color="#7f7f7f", linestyle="-"
    )
    _rover_base._plot_line(ax3, blocks, _REW + "tilt_penalty", "tilt_penalty", color="#8c564b", linestyle="--")
    _rover_base._plot_line(ax3, blocks, _REW + "y_drift_penalty", "y_drift_penalty (Y from 0)", color="#1f77b4", linestyle="-.")
    _rover_base._plot_line(ax3, blocks, _REW + "alive_bonus", "alive_bonus", color="#bcbd22", linestyle="-.")
    _rover_base._plot_line(
        ax3, blocks, _REW + "radial_z_alignment_reward", "radial_z_align_reward (body+Z·radial)", color="#98df8a", linestyle="-"
    )
    _rover_base._plot_line(
        ax3, blocks, _REW + "radial_z_alignment_penalty", "radial_z_align_penalty (-(1-dot)²)", color="#98df8a", linestyle=":"
    )
    ax3.set_ylabel("Episode reward contribution")
    ax3.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    _rover_base._legend_if_any(ax3, fontsize=8)
    ax3.grid(True, alpha=0.3)
    if reward_ylim_after >= 0:
        _rover_base._focus_ylim(
            ax3,
            blocks,
            [
                _REW + k
                for k in (
                    "action_rate_penalty",
                    "motion_jerk_penalty",
                    "motion_jerk_smoothness_bonus",
                    "heading_oscillation_penalty",
                    "heading_stability_bonus",
                    "tilt_penalty",
                    "y_drift_penalty",
                    "alive_bonus",
                    "radial_z_alignment_reward",
                    "radial_z_alignment_penalty",
                )
            ],
            reward_ylim_after,
        )

    # ── Panel 4: TERMINATIONS ──────────────────────────────────────────
    ax4.set_title("Episode Termination Rates  (fraction of resets)", fontsize=10, loc="left")
    _TERM = "Episode_Termination/"
    _rover_base._plot_line(ax4, blocks, _TERM + "fell_off_disk", "fell_off_disk (failure)", color="#d62728")
    _rover_base._plot_line(
        ax4, blocks, _TERM + "forward_progress_goal", "goal_reached (success)", color="#2ca02c", linestyle="--"
    )
    _rover_base._plot_line(ax4, blocks, _TERM + "time_out", "time_out (truncated)", color="#7f7f7f", linestyle=":")
    ax4.set_ylabel("Rate [0–1]")
    ax4.set_ylim(-0.02, 1.05)
    ax4.axhline(0, color="gray", linewidth=0.5)
    _rover_base._legend_if_any(ax4, fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: POLICY / LOSSES ───────────────────────────────────────
    ax5.set_title("Policy Losses & Noise", fontsize=10, loc="left")
    _rover_base._plot_line(ax5, blocks, "Loss/value_function", "value_function loss", color="#1f77b4")
    _rover_base._plot_line(ax5, blocks, "Loss/surrogate", "surrogate loss", color="#ff7f0e", linestyle="--")
    _rover_base._plot_line(ax5, blocks, "Loss/entropy", "entropy", color="#2ca02c", linestyle="-.")
    ax5.set_ylabel("Loss value")
    _rover_base._legend_if_any(ax5, fontsize=8, loc="upper right")
    ax5.grid(True, alpha=0.3)
    ax5_r = ax5.twinx()
    _rover_base._plot_line(ax5_r, blocks, "Policy/mean_noise_std", "mean action noise std", color="#9467bd", linestyle=":")
    ax5_r.set_ylabel("Noise std", color="#9467bd")
    _rover_base._legend_if_any(ax5_r, fontsize=8, loc="center right")
    ax5.set_xlabel("Training iteration")

    # ── Config sidebar ─────────────────────────────────────────────────
    ax_cfg = fig.add_subplot(gs_outer[1])
    ax_cfg.axis("off")

    def _cfgv(k: str, default: str = "—") -> str:
        return cfg.get(k, default)

    sidebar_lines = [
        "══ Run ══",
        f"  experiment   : {_cfgv('experiment_name')}",
        f"  task           : {_cfgv('task')}",
        f"  start_policy   : {_cfgv('starting_policy_path')}",
        "",
        "══ CLI rover overrides ══",
        f"  fwd_progress : {_cfgv('rover_forward_progress_weight')}",
        f"  speed_track  : {_cfgv('rover_speed_tracking_weight')}",
        f"    target_spd : {_cfgv('rover_speed_target')} m/s",
        f"  heading        : {_cfgv('rover_heading_weight')}",
        f"  alive          : {_cfgv('rover_alive_weight')}",
        f"  tilt_pen       : {_cfgv('rover_tilt_weight')}",
        f"  radial_z_rew   : {_cfgv('rover_radial_z_alignment_reward_weight')}",
        f"  radial_z_pen   : {_cfgv('rover_radial_z_alignment_penalty_weight')}",
        f"  act_rate_pen   : {_cfgv('rover_action_rate_weight')}",
        f"  motion_jerk    : {_cfgv('rover_motion_jerk_weight')}",
        f"  jerk_bonus     : {_cfgv('rover_motion_jerk_bonus_weight')}  exp={_cfgv('rover_motion_jerk_bonus_exp_scale')}",
        f"  head_oscill    : {_cfgv('rover_heading_oscillation_weight')}",
        f"  head_stab_bon  : {_cfgv('rover_heading_stability_bonus_weight')}  exp={_cfgv('rover_heading_stability_bonus_exp_scale')}",
        f"  goal_bonus     : {_cfgv('rover_goal_bonus_weight')}",
        "",
        "══ Termination ══",
        f"  goal_rad       : {_cfgv('rover_goal_rad')} rad",
        f"  fall_r_min     : {_cfgv('rover_fall_radius')} m",
        f"  episode_s      : {_cfgv('rover_episode_length_s')} s",
        "",
        "══ Training ══",
        f"  num_envs       : {_cfgv('num_envs')}",
        f"  max_iter       : {_cfgv('max_iterations')}",
        f"  seed           : {_cfgv('seed')}",
        f"  device         : {_cfgv('device')}",
        "",
        "══ PPO ══",
        f"  lr             : {_cfgv('ppo_learning_rate')}",
        f"  entropy        : {_cfgv('ppo_entropy_coef')}",
        f"  noise_std      : {_cfgv('ppo_init_noise_std')}",
        f"  clip           : {_cfgv('ppo_clip_param')}",
        f"  kl_target      : {_cfgv('ppo_desired_kl')}",
        f"  steps/env      : {_cfgv('ppo_num_steps_per_env')}",
        f"  epochs         : {_cfgv('ppo_num_learning_epochs')}",
        f"  mini_batches   : {_cfgv('ppo_num_mini_batches')}",
        f"  gamma          : {_cfgv('ppo_gamma')}",
        f"  lam            : {_cfgv('ppo_lam')}",
    ]
    sidebar_lines.extend(_idle_to_forward_sidebar_doc())
    sidebar_lines.extend(
        [
            "",
            f"══ Iterations logged: {len(blocks)} ══",
            f"  {blocks[0][0]}  →  {max_iter}",
        ]
    )

    ax_cfg.text(
        0.04,
        0.97,
        "\n".join(sidebar_lines),
        transform=ax_cfg.transAxes,
        fontsize=7.2,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse training progress for Rover Idle → Forward transition task.",
    )
    parser.add_argument("log_dir", help="Path to the RSL-RL log directory.")
    parser.add_argument("--last", type=int, default=None, metavar="N", help="Print only the last N iteration blocks.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate the visual analysis PNG.")
    parser.add_argument("--plot-only", action="store_true", help="Only generate the PNG (skip text print).")
    parser.add_argument(
        "--output",
        type=str,
        default="training_progress_analysis.png",
        help="Output PNG filename (default: training_progress_analysis.png).",
    )
    parser.add_argument(
        "--reward-ylim-after",
        type=int,
        default=200,
        metavar="ITER",
        help="Focus reward y-axis on iterations ≥ ITER (default 200; -1 = full auto).",
    )
    args = parser.parse_args()

    log_path = os.path.join(args.log_dir, "training_progress_log.txt")
    if not os.path.isfile(log_path):
        print(f"[ERROR] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    blocks = _rover_base.parse_log(log_path)
    if not blocks:
        print("[WARN] No iteration blocks parsed from log.", file=sys.stderr)
        sys.exit(1)

    if not args.plot_only:
        _rover_base.print_blocks(blocks, last=args.last)

    if not args.no_plot:
        cfg = parse_header_config(log_path)
        output_path = os.path.join(args.log_dir, args.output)
        generate_plot(blocks, cfg, output_path, reward_ylim_after=args.reward_ylim_after)


if __name__ == "__main__":
    main()

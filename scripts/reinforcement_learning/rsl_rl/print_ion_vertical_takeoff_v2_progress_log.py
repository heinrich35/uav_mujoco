#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress for Isaac-Ion-Vertical-Takeoff-V2-Direct-v0 (takeoff vs stability profiles + extra obs).
# Reuses plot layout from print_ion_vertical_takeoff_direct_progress_log.py with V2 header parsing and sidebar.
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_ion_vertical_takeoff_v2_progress_log.py <log_dir>
#
# Options: same as Direct script (--last, --no-plot, --output, --plot-only, ylim flags).

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys


def _load_ion_direct_progress_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "print_ion_vertical_takeoff_direct_progress_log.py")
    spec = importlib.util.spec_from_file_location("ion_vertical_takeoff_direct_progress_log", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ion = _load_ion_direct_progress_module()

V2_HEADER_NAMES = (
    "reward_profile",
    "staged_stability_start_env_steps",
    "staged_stability_ramp_env_steps",
    "target_vz_command_obs_enabled",
    "vz_command_obs_scale",
    "include_reward_profile_phase_obs",
)


def parse_header_ion_v2_reward_config(path: str) -> dict | None:
    """Run-arg weights from Direct parser plus Ion V2 profile / obs fields from the log header."""
    base = _ion.parse_header_ion_direct_reward_config(path)
    if base is None:
        base = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read().split("Learning iteration")[0]
    except OSError:
        return base if base else None
    for name in V2_HEADER_NAMES:
        m = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+)\s*$", head, re.MULTILINE)
        if not m:
            continue
        raw = m.group(1).strip()
        if name == "reward_profile":
            base[name] = raw
            continue
        if raw in ("True", "False"):
            base[name] = raw
            continue
        try:
            base[name] = float(raw)
        except ValueError:
            base[name] = raw
    return base if base else None


def _v2_sidebar_extra_lines(reward_cfg: dict) -> list[str] | None:
    if not any(k in reward_cfg for k in V2_HEADER_NAMES):
        return None
    lines = [
        "",
        "Ion V2 (cfg)",
        f"  reward_profile: {reward_cfg.get('reward_profile', '—')}",
        f"  staged_stability_start_env_steps: {reward_cfg.get('staged_stability_start_env_steps', '—')}",
        f"  staged_stability_ramp_env_steps: {reward_cfg.get('staged_stability_ramp_env_steps', '—')}",
        f"  target_vz_command_obs: {reward_cfg.get('target_vz_command_obs_enabled', '—')}",
        f"  vz_command_obs_scale: {reward_cfg.get('vz_command_obs_scale', '—')}",
        f"  include_reward_profile_phase_obs: {reward_cfg.get('include_reward_profile_phase_obs', '—')}",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Print / plot training_progress_log.txt for Isaac-Ion-Vertical-Takeoff-V2-Direct-v0."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/ion_vertical_takeoff_v2/YYYY-MM-DD_HH-MM-SS",
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
        help="Rewards/penalties y-limits: iter ≥ max(ITER, late-phase start). -1 = full autoscale.",
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

    reward_cfg_for_print = parse_header_ion_v2_reward_config(path)

    if not args.plot_only:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = re.split(r"(?=^################################################################################\n)", content, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]
        if args.last is not None and args.last > 0:
            blocks = blocks[-args.last :]
        for block in blocks:
            filtered = _ion._filter_block_text_for_zero_weight_episode_rewards(block, reward_cfg_for_print)
            print(_ion._strip_ansi(filtered))
            print()

    if not args.no_plot:
        _focus = None if args.reward_penalty_ylim_focus_after < 0 else args.reward_penalty_ylim_focus_after
        out_path = _ion.generate_visual_analysis(
            args.log_dir,
            args.output,
            reward_penalty_ylim_focus_after=_focus,
            reward_penalty_late_phase_frac=float(args.reward_penalty_ylim_late_frac),
            plot_title="Ion Vertical Takeoff V2 — training progress",
            header_reward_config_parser=parse_header_ion_v2_reward_config,
            sidebar_extra_lines_fn=_v2_sidebar_extra_lines,
        )
        if out_path:
            print(f"[INFO] Visual analysis saved to: {out_path}", file=sys.stderr)
        else:
            print("[WARNING] Could not generate visual analysis.", file=sys.stderr)


if __name__ == "__main__":
    main()

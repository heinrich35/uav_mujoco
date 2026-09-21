#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Progress log + plot for Isaac-Ion-Horizontal-Cruise-V2-Direct-v0 (reuses V1 plot layout).
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_ion_horizontal_cruise_v2_progress_log.py <log_dir>
# Options: --last N, --no-plot, --output NAME, --plot-only

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys


def _load_v1_print_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "print_ion_horizontal_cruise_v1_progress_log.py")
    spec = importlib.util.spec_from_file_location("ion_horizontal_cruise_v1_progress_log", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v1 = _load_v1_print_module()

V2_HEADER_NAMES = (
    "reward_profile",
    "staged_stability_start_env_steps",
    "staged_stability_ramp_env_steps",
    "target_vx_command_obs_enabled",
    "vx_command_obs_scale",
    "target_vz_command_obs_enabled",
    "vz_command_obs_scale",
    "include_reward_profile_phase_obs",
)


def _parse_header_cruise_v2(path: str) -> dict[str, object] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read().split("Learning iteration")[0]
    except OSError:
        return None
    out: dict[str, object] = {}
    for name in V2_HEADER_NAMES:
        m = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+)\s*$", head, re.MULTILINE)
        if not m:
            continue
        raw = m.group(1).strip()
        if name == "reward_profile":
            out[name] = raw
            continue
        if raw in ("True", "False"):
            out[name] = raw
            continue
        try:
            out[name] = float(raw)
        except ValueError:
            out[name] = raw
    return out if out else None


def main():
    parser = argparse.ArgumentParser(
        description="Print / plot training_progress_log.txt for Isaac-Ion-Horizontal-Cruise-V2-Direct-v0."
    )
    parser.add_argument("log_dir", type=str, help="Run directory under logs/rsl_rl/ion_horizontal_cruise_v2/")
    parser.add_argument("--last", type=int, default=None, metavar="N", help="Print only the last N blocks")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG")
    parser.add_argument("--output", type=str, default="training_progress_analysis.png")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate PNG")
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
            print(_v1._strip_ansi(block))
            print()

    if not args.no_plot:
        out_path = _v1.generate_visual_analysis(
            args.log_dir,
            args.output,
            plot_title="Ion-Horizontal-Cruise-V2 — Training progress (from training_progress_log.txt)",
        )
        hdr = _parse_header_cruise_v2(path)
        if hdr and out_path:
            print("[INFO] V2 header fields in log:", file=sys.stderr)
            for k in V2_HEADER_NAMES:
                if k in hdr:
                    print(f"       {k}: {hdr[k]}", file=sys.stderr)
        if out_path:
            print(f"[INFO] Visual analysis saved to: {out_path}", file=sys.stderr)
        else:
            print("[WARNING] Could not generate visual analysis.", file=sys.stderr)


if __name__ == "__main__":
    main()

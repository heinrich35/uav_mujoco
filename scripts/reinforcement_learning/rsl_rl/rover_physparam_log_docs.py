# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Physparam latch (p0–p9) documentation for ``training_progress_log.txt`` and rover progress plots.

Numeric limits mirror ``scripts/rover/rover_utils.py`` — keep in sync when those constants change.
Plotting runs outside Isaac Sim must not import ``rover_utils`` (it pulls ``pxr``); use ``ru=None``.
"""

from __future__ import annotations

import types
from typing import Any

# Defaults copied from scripts/rover/rover_utils.py (avoid importing rover_utils in headless plot jobs).
_PHY_DEFAULT = types.SimpleNamespace(
    WHEEL_CMD_SCALE_MIN=0.05,
    WHEEL_CMD_SCALE_MAX=10.0,
    WHEEL_VEL_TARGET_SLEW_MIN=20.0,
    WHEEL_VEL_TARGET_SLEW_MAX=2000.0,
    BASE_MASS_SCALE_MIN=0.1,
    BASE_MASS_SCALE_MAX=2000.0,
    BASE_MASS_MAX_KG=2000.0,
    WHEEL_LINK_MASS_SCALE_MIN=0.1,
    WHEEL_LINK_MASS_SCALE_MAX=500.0,
)


def _limits(ru: Any | None) -> Any:
    return ru if ru is not None else _PHY_DEFAULT


def physparam_latch_doc_lines(ru: Any | None = None) -> list[str]:
    """Indented lines for log header / plot sidebar (two leading spaces each)."""
    r = _limits(ru)
    fmin, fmax = float(r.WHEEL_CMD_SCALE_MIN), float(r.WHEEL_CMD_SCALE_MAX)
    smin, smax = float(r.WHEEL_VEL_TARGET_SLEW_MIN), float(r.WHEEL_VEL_TARGET_SLEW_MAX)
    bmin = float(r.BASE_MASS_SCALE_MIN)
    bmax = float(r.BASE_MASS_SCALE_MAX)
    bcap = float(r.BASE_MASS_MAX_KG)
    wmin = float(r.WHEEL_LINK_MASS_SCALE_MIN)
    wmax = float(r.WHEEL_LINK_MASS_SCALE_MAX)
    return [
        "  p0  G/T …        Base mass scale (norm → physical); range includes per-env cap vs ref mass",
        f"                   effective scale span ~[{bmin:g}, min({bmax:g}, {bcap:g}/ref_kg)]",
        "  p1  Y/H …        Wheel link mass scale (same idea; wheel ref masses + link cap)",
        f"                   scale span ~[{wmin:g}, min({wmax:g}, wheel_link_mass_max_scale_for_ref)]",
        "  p2  1/2, J/K     Wheel μ_s (static) → mapped to about [0.4, 9.0] in sim",
        "  p3  3/4, N/M     Wheel μ_d (dynamic) → about [0.35, 8.5], enforced < μ_s",
        "  p4  —            Cylindrical gravity scale (unchanged law; scales inward-radial g)",
        "                   mapped to about [0.55, 1.25] × cylindrical g magnitude",
        f"  p5  Q/E          Forward wheel cmd scale → episode_forward_cmd_scale in [{fmin:g}, {fmax:g}]",
        f"  p6  F/R          Turn cmd scale → episode_turn_cmd_scale in [{fmin:g}, {fmax:g}]",
        f"  p7  V/B          ω-target max slew (rad/s²) → episode_wheel_target_max_slew in [{smin:g}, {smax:g}]",
        "  p8  (extra)       Viscous friction multiplier vs initial USD / PhysX value → [0.1, 5.0];",
        "                   written to friction triplet only if PhysX exposes viscous on wheel DOFs.",
        "  p9  (extra)       Wheel joint damping multiplier vs default damping at init → [0.25, 4.0];",
        "                   write_joint_damping_to_sim on the four wheel joints.",
        "",
        "  Episode_RoverPhysparam/mean_* lines: mean applied physical values over envs that",
        "  latched this logging interval (not the raw normalized p values).",
        "  Action tail (dims 10–11): differential drive every step (WASD-style), not latched.",
    ]


def append_physparam_latch_doc_to_header(header_lines: list[str], ru: Any | None) -> None:
    """Append a fenced block to ``training_progress_log.txt`` header (physparam task only)."""
    header_lines.append("################################################################################")
    header_lines.append(" Physparam latch — p0..p9 (first action head; latched once per episode)")
    header_lines.append("################################################################################")
    header_lines.extend(physparam_latch_doc_lines(ru))
    header_lines.append("################################################################################")

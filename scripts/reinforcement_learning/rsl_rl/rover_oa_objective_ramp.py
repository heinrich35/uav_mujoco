# Copyright (c) 2025, IsaacLab fork / workspace scripts.
# SPDX-License-Identifier: BSD-3-Clause
"""Linear in-process ramps for Rover obstacle-avoidance reward weights.

Two modes (``--rover_oa_objective_ramp_blend_mode``):

* ``scale`` (default): blue / goal / visual / approach terms scale from
  ``ramp_start_scale`` → 1.0 over ``ramp_iters`` **learning iterations**; optional
  ``forward_boost`` on ``forward_angular_progress`` (legacy behaviour).

* ``full_lerp``: same iteration horizon as G1-style staged ramps: **every** term
  in ``_FULL_LERP_TERM_NAMES`` present in the reward manager linearly interpolates
  from a fixed **anchor preset** (phase-1 style) to the weights this run was
  constructed with (phase-2/3 CLI targets). ``ramp_start_scale`` / ``forward_boost``
  are ignored in this mode.
"""

from __future__ import annotations

_OBJECTIVE_TERM_NAMES: tuple[str, ...] = (
    "blue_box_avoidance_penalty",
    "next_goal_pass_through_reward",
    "visual_blue_box_ahead_penalty",
    "visual_goal_ahead_alignment_reward",
    "next_goal_approach_shaping",
)

_FORWARD_COAST_TERM: str = "forward_angular_progress"

# Reward-term names on ``reward_manager`` (must match env RewTerm keys).
_FULL_LERP_TERM_NAMES: tuple[str, ...] = (
    "forward_angular_progress",
    "forward_speed_tracking_penalty",
    "blue_box_avoidance_penalty",
    "next_goal_pass_through_reward",
    "visual_blue_box_ahead_penalty",
    "visual_goal_ahead_alignment_reward",
    "next_goal_approach_shaping",
    "disk_tangent_heading_alignment",
    "radial_z_alignment_reward",
    "radial_z_alignment_penalty",
    "near_fall_radial_penalty",
    "upside_down_termination_penalty",
    "fell_off_disk_termination_penalty",
    "action_rate_penalty",
    "motion_jerk_penalty",
    "heading_oscillation_penalty",
    "alive_bonus",
)

# Anchor presets: values match signed ``RewardTermCfg.weight`` after train.py mirroring
# (termination penalties negative; near_fall / visual_blue positive magnitudes).
_PRESET_MODERATE: dict[str, float] = {
    "forward_angular_progress": 60.0,
    "forward_speed_tracking_penalty": 10.0,
    "blue_box_avoidance_penalty": 180.0,
    "next_goal_pass_through_reward": 500.0,
    "visual_blue_box_ahead_penalty": 5.0,
    "visual_goal_ahead_alignment_reward": 4.0,
    "next_goal_approach_shaping": 2.5,
    "disk_tangent_heading_alignment": 10.0,
    "radial_z_alignment_reward": 25.0,
    "radial_z_alignment_penalty": 35.0,
    "near_fall_radial_penalty": 20.0,
    "upside_down_termination_penalty": -2500.0,
    "fell_off_disk_termination_penalty": -12000.0,
    "action_rate_penalty": 0.10,
    "motion_jerk_penalty": 0.04,
    "heading_oscillation_penalty": 0.10,
    "alive_bonus": 0.15,
}

_PRESET_RECOVERY: dict[str, float] = {
    "forward_angular_progress": 35.0,
    "forward_speed_tracking_penalty": 6.0,
    "blue_box_avoidance_penalty": 60.0,
    "next_goal_pass_through_reward": 120.0,
    "visual_blue_box_ahead_penalty": 1.5,
    "visual_goal_ahead_alignment_reward": 1.0,
    "next_goal_approach_shaping": 0.8,
    "disk_tangent_heading_alignment": 8.0,
    "radial_z_alignment_reward": 25.0,
    "radial_z_alignment_penalty": 35.0,
    "near_fall_radial_penalty": 20.0,
    "upside_down_termination_penalty": -2500.0,
    "fell_off_disk_termination_penalty": -12000.0,
    "action_rate_penalty": 0.10,
    "motion_jerk_penalty": 0.04,
    "heading_oscillation_penalty": 0.10,
    "alive_bonus": 0.15,
}


def install_rover_obstacle_avoidance_objective_ramp(runner, env, args_cli) -> bool:
    """Patch ``runner.alg.update`` to refresh reward weights each iteration.

    Returns:
        True if the patch was installed, False otherwise.
    """
    if int(getattr(args_cli, "rover_oa_objective_ramp_enable", 0) or 0) != 1:
        return False

    ramp_iters = max(1, int(getattr(args_cli, "rover_oa_objective_ramp_iters", 1200)))
    blend_mode = str(getattr(args_cli, "rover_oa_objective_ramp_blend_mode", "scale") or "scale").strip().lower()
    if blend_mode in ("full_lerp", "full", "phase_lerp", "lerp"):
        blend_mode = "full_lerp"
    elif blend_mode in ("scale", "objective_scale", "legacy"):
        blend_mode = "scale"
    else:
        print(f"[WARN] rover OA ramp: unknown blend_mode {blend_mode!r} — using 'scale'.")
        blend_mode = "scale"

    ramp_start = float(getattr(args_cli, "rover_oa_objective_ramp_start_scale", 0.0))
    ramp_start = max(0.0, min(1.0, ramp_start))
    forward_boost = float(getattr(args_cli, "rover_oa_objective_ramp_forward_boost", 0.12))
    forward_boost = max(0.0, forward_boost)

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "reward_manager"):
        print("[WARN] rover OA objective ramp: env.unwrapped has no reward_manager; skipping ramp install.")
        return False

    rm = unwrapped.reward_manager
    anchor_iter = int(getattr(runner, "current_learning_iteration", 0))
    _next_rollout_iter = [anchor_iter]

    def _alpha(rollout_iter: int) -> float:
        t = float(rollout_iter - anchor_iter)
        return min(1.0, max(0.0, t / float(ramp_iters)))

    if blend_mode == "full_lerp":
        preset_name = str(getattr(args_cli, "rover_oa_objective_ramp_anchor_preset", "moderate") or "moderate").strip().lower()
        if preset_name in ("recovery", "curriculum_v2_phase1", "v2_phase1"):
            preset = _PRESET_RECOVERY
            preset_label = "recovery (curriculum v2 phase-1 style)"
        else:
            preset = _PRESET_MODERATE
            preset_label = "moderate (pre-ramp / TRAIN.sh style)"

        ends: dict[str, float] = {}
        starts: dict[str, float] = {}
        for name in _FULL_LERP_TERM_NAMES:
            if name not in rm._term_names:
                continue
            idx = rm._term_names.index(name)
            w_end = float(rm._term_cfgs[idx].weight)
            ends[name] = w_end
            w_anchor = preset.get(name, w_end)
            starts[name] = float(w_anchor)

        def _apply_weights(rollout_iter: int) -> None:
            a = _alpha(rollout_iter)
            for name, w_end in ends.items():
                w_start = starts[name]
                w = (1.0 - a) * w_start + a * w_end
                idx = rm._term_names.index(name)
                rm._term_cfgs[idx].weight = float(w)

        _apply_weights(_next_rollout_iter[0])

        _orig_update = runner.alg.update

        def _update_with_ramp():
            out = _orig_update()
            _next_rollout_iter[0] += 1
            _apply_weights(_next_rollout_iter[0])
            return out

        runner.alg.update = _update_with_ramp
        print(
            "[INFO] Rover obstacle-avoidance: in-process **full_lerp** ramp — "
            f"anchor_learning_iter={anchor_iter}, ramp_iters={ramp_iters}, "
            f"preset={preset_label}, terms={sorted(ends.keys())}."
        )
        return True

    # --- legacy objective_scale + forward coast ---
    bases_obj: dict[str, float] = {}
    for name in _OBJECTIVE_TERM_NAMES:
        if name not in rm._term_names:
            continue
        idx = rm._term_names.index(name)
        bases_obj[name] = float(rm._term_cfgs[idx].weight)

    base_forward: float | None = None
    if _FORWARD_COAST_TERM in rm._term_names:
        idx = rm._term_names.index(_FORWARD_COAST_TERM)
        base_forward = float(rm._term_cfgs[idx].weight)

    def _objective_scale(rollout_iter: int) -> float:
        s = _alpha(rollout_iter)
        return ramp_start + (1.0 - ramp_start) * s

    def _apply_weights(rollout_iter: int) -> None:
        s = _objective_scale(rollout_iter)
        for name, w0 in bases_obj.items():
            idx = rm._term_names.index(name)
            rm._term_cfgs[idx].weight = float(w0) * s
        if base_forward is not None:
            coast = 1.0 + forward_boost * (1.0 - s)
            idx = rm._term_names.index(_FORWARD_COAST_TERM)
            rm._term_cfgs[idx].weight = float(base_forward) * coast

    _apply_weights(_next_rollout_iter[0])

    _orig_update = runner.alg.update

    def _update_with_ramp():
        out = _orig_update()
        _next_rollout_iter[0] += 1
        _apply_weights(_next_rollout_iter[0])
        return out

    runner.alg.update = _update_with_ramp
    print(
        "[INFO] Rover obstacle-avoidance: in-process objective ramp enabled — "
        f"anchor_learning_iter={anchor_iter}, ramp_iters={ramp_iters}, "
        f"start_scale={ramp_start:g}→1, forward_boost={forward_boost:g} on '{_FORWARD_COAST_TERM}'."
    )
    return True

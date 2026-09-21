#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Training progress analysis for G1 Locomotion V4 (g1_locomotion_v4).
# Parses training_progress_log.txt (Episode_Reward/* and Episode_Termination/*)
# and generates training_progress_analysis.png aligned with V4: **locomotion** vs **imitation** positive
# reward panels (plus penalties, main/loss), config sidebar (V4 reward_profile + command obs).
# Task: Isaac-G1-Locomotion-V4-Direct-v0; underlying env is V3 + separated reward groups + command in obs.
# V3 content below still applies to logged metrics; plots split positive components by task vs motion clip.
# Imitation: joint angles (imitation_joints_position_* → Episode_Reward/imitation_*), velocity
# (optional leg/arm weights via imitation_joint_velocity_use_leg_arm_weights), strength ramp
# (imitation_strength_ramp_steps), periodic joint vel FD (hdf5_imitation_periodic_joint_vel_fd),
# and pelvis-frame link targets (hdf5_link_local_imitation_*; .hdf5 / .h5; legacy npz_* keys),
# plus optional ankle roll orientation vs clip (hdf5_ankle_roll_y_rot_imitation_* reward/sigma/tolerance_rad
#   and optional squared-error penalty hdf5_ankle_roll_y_rot_imitation_penalty_weight / train --ankle_roll_pen),
# hip/knee link Y rotation vs clip (hdf5_hip_knee_y_rot_imitation_* incl. sigma / tolerance_rad / penalty_weight;
#   train.py short: --hip_knee_y_rot_rew / _sigma / _tol_rad / _pen),
# and foot xyz reward/penalty (hdf5_ankle_roll_link_pos_imitation_*_weight / _sigma_m / _tolerance_m).
# Lateral: x_drift_reward_* (exp) + x_drift_penalty_*; y_drift_reward_* alongside y_drift_penalty_*.
# Palms vs pelvis z (V4): palm_below_pelvis_z_* — left/right palm origins in pelvis frame stay at or below z cap.
# Forward / progress (world +x): progress_reward_multiplier (potential), progress_reward_world_vx_blend (+vx·dt),
# progress_reward_forward_displacement_weight / _threshold_m (Δroot_x per step); Episode_Reward/progress_reward sums all.
# progress_goal_world_offset_xy_m (V4): world xy offset from env origin for V1-style potential/heading goal (or None).
# Policy vs joint smoothness: Episode_Reward/policy_action_{smoothness_reward,roughness_penalty},
# joint_velocity_{smoothness_reward,roughness_penalty} (pairs on ‖Δaction‖² and ‖Δdof_vel‖²). Sidebar: root_lin_vel_jerk_penalty_weight.
# Includes V3 optimizations: base_height_reward, curriculum, velocity_command_range, push.
# Gait refinements in log header: same_foot_tap_min_air_steps, symmetry_use_geometric_mean,
# step_length_use_smooth_reward, gait_single_stance_reward, hip_weight_shift_reward, symmetry_include_hip_torque, etc.
# Dynamic balance (V4): root CoM velocity tracking, forward linear momentum projection,
# base-of-support center penalty, and DCM / capture-point reward + lateral penalty.
# Episode-level symmetry (end of episode): step / joint ROM / link ROM blends plus
# episode_symmetry_single_stance_balance_* (L-only vs R-only support time; same XOR idea as gait_single_stance_reward)
# and episode_symmetry_hdf5_ankle_roll_y_track_* (symmetry of mean |ankle y-rot vs clip| L vs R).
#
# Usage:
#   python scripts/reinforcement_learning/rsl_rl/print_g1_locomotion_v4_progress_log.py <log_dir>
#   python scripts/reinforcement_learning/rsl_rl/print_g1_locomotion_v4_progress_log.py logs/rsl_rl/g1_locomotion_v4_p1/2026-03-27_12-00-00
#
# Options:
#   --last N       Print only the last N iteration blocks (default: all)
#   --no-plot      Do not generate the visual analysis PNG
#   --output NAME  Output filename for the plot (default: training_progress_analysis.png)
#   --plot-only    Only generate the visual analysis PNG
#   --reward-penalty-ylim-focus-after ITER
#                  Rewards/penalties: y-scale uses iter ≥ max(ITER, start of last late-phase window).
#                  Default 300. Use -1 for full autoscale.
#   --reward-penalty-ylim-late-frac FRAC
#                  Late-phase fraction of the logged iteration span (default 0.22 = last 22%).

from __future__ import annotations

import argparse
import math
import os
import re
import sys

from collections.abc import Sequence


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _read_training_progress_pre_iteration(path: str) -> str:
    """File content before the first 'Learning iteration' block (header + env + joint tables, …)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content.split("Learning iteration", 1)[0]


def _extract_kv_section_body(content: str, title_substring: str) -> str | None:
    """Body of a `training_progress_log.txt` section: lines `  key: value` after title until closing `######`.

    Mirrors the state machine in ``parse_header_run_args`` (v1 standing): skip the fence line after the
    title, then collect ``  *:`` lines until the next ``######`` once at least one key was seen.
    """
    lines = content.splitlines()
    in_title = False
    past_open_fence = False
    body: list[str] = []
    seen_kv = False
    for line in lines:
        if not in_title:
            if title_substring in line:
                in_title = True
            continue
        # in_title: skip until first ##### after title
        if not past_open_fence:
            if line.strip().startswith("######"):
                past_open_fence = True
            continue
        if line.strip().startswith("######") and seen_kv:
            break
        if line.startswith("  ") and ":" in line:
            seen_kv = True
            body.append(line)
    if not body:
        return None
    return "\n".join(body)


def _header_text_for_reward_config_parse(path: str) -> str:
    """Text used to regex-extract config: **run arguments first**, then env block (run wins on duplicates).

    Falls back to the full pre-iteration header for older logs without named sections.
    """
    try:
        content = _read_training_progress_pre_iteration(path)
    except OSError:
        return ""
    run = _extract_kv_section_body(content, "Run arguments (this run)")
    env = _extract_kv_section_body(
        content, "Env config (reward weights, velocity, smooth/jerk motion scales)"
    )
    parts = []
    if run:
        parts.append(run)
    if env:
        parts.append(env)
    if parts:
        return "\n".join(parts) + "\n"
    return content


def _parse_block(block: str) -> tuple[int | None, dict[str, float]]:
    """Parse one iteration block. Returns (iteration_number, {metric_name: value})."""
    block = _strip_ansi(block)
    out = {}
    it_match = re.search(r"Learning iteration\s+(\d+)/", block)
    iteration = int(it_match.group(1)) if it_match else None
    for m in re.finditer(r"^([^:]+?):\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*", block, re.MULTILINE):
        key = m.group(1).strip()
        try:
            out[key] = float(m.group(2))
        except ValueError:
            continue
    return iteration, out


def parse_header_smooth_jerk(path: str) -> dict[str, float] | None:
    """Parse log header for smooth/jerk args (run-args section preferred via _header_text_for_reward_config_parse)."""
    try:
        head = _header_text_for_reward_config_parse(path)
        out = {}
        for name in (
            "action_rate_penalty_scale",
            "policy_action_roughness_penalty_scale",
            "policy_action_roughness_huber_delta",
            "joint_velocity_roughness_penalty_scale",
            "joint_velocity_roughness_huber_delta_rad_s",
            "joint_velocity_smoothness_reward_weight",
            "joint_velocity_smoothness_sigma",
            "joint_acceleration_penalty_scale",
            "smooth_motion_reward_weight",
            "smooth_motion_sigma",
            "smooth_motion_joint_accel_reward_weight",
            "smooth_motion_joint_accel_sigma",
            "root_lin_vel_jerk_penalty_weight",
        ):
            m = re.search(rf"^\s*{re.escape(name)}\s*:\s*([\d.eE+-]+)", head, re.MULTILINE)
            if m:
                try:
                    out[name] = float(m.group(1))
                except ValueError:
                    pass
        return out if out else None
    except Exception:
        return None


def parse_header_reward_config(path: str) -> dict[str, float | str | None] | None:
    """Parse log header for reward config (velocity, drift, stability, termination, V3 optimizations).

    Prefers **Run arguments (this run)** over the following **Env config** block so CLI values match
    ``train.py`` header order (duplicated keys in the env snapshot do not override the run).
    """
    try:
        head = _header_text_for_reward_config_parse(path)
        out = {}
        for name in (
            "up_weight",
            "heading_weight",
            "alive_reward_scale",
            "progress_reward_multiplier",
            "progress_reward_world_vx_blend",
            "progress_reward_forward_displacement_weight",
            "progress_reward_forward_displacement_threshold_m",
            "actions_cost_scale",
            "energy_cost_scale",
            "target_velocity",
            "velocity_reward_weight",
            "velocity_reward_sharpness",
            "target_x_velocity_penalty_weight",
            "num_envs",
            "max_iterations",
            "frames_per_second",
            "gravity_scale",
            "seed",
            "x_drift_tolerance",
            "x_drift_penalty_weight",
            "x_drift_reward_weight",
            "x_drift_reward_sigma",
            "y_drift_tolerance",
            "y_drift_reward_weight",
            "y_drift_reward_sigma",
            "xy_drift_tolerance",
            "xy_drift_penalty_weight",
            "y_drift_penalty_weight",
            "target_z_orientation",
            "z_orientation_tolerance",
            "z_orientation_penalty_weight",
            "z_orientation_reward_weight",
            "z_orientation_reward_sharpness",
            "imitation_joints_position_reward_weight",
            "imitation_joints_position_penalty_weight",
            "imitation_joints_position_reward_sharpness",
            "imitation_position_reward_weight",
            "imitation_position_penalty_weight",
            "imitation_velocity_reward_weight",
            "imitation_velocity_penalty_weight",
            "imitation_reward_weight",  # legacy; fallback for position reward
            "imitation_penalty_weight",  # legacy; fallback for position penalty
            "hdf5_imitation_loop_count",
            "hdf5_imitation_time_factor",
            "hdf5_imitation_position_tolerance_pct",
            "hdf5_imitation_velocity_tolerance_pct",
            "hdf5_imitation_max_frames",
            "hdf5_imitation_reward_episode_time_scale_enabled",
            "hdf5_joint_position_data_scale",
            "hdf5_joint_position_data_scale_ramp_delay_steps",
            "hdf5_joint_position_data_scale_ramp_enabled",
            "hdf5_joint_position_data_scale_ramp_end",
            "hdf5_joint_position_data_scale_ramp_in_ppo_iterations",
            "hdf5_joint_position_data_scale_ramp_start",
            "hdf5_joint_position_data_scale_ramp_steps",
            "hdf5_joint_position_data_scale_exclude_ankles",
            "npz_imitation_loop_count",
            "npz_imitation_time_factor",
            "npz_imitation_position_tolerance_pct",
            "npz_imitation_velocity_tolerance_pct",
            "hdf5_link_local_imitation_reward_weight",
            "hdf5_link_local_imitation_penalty_weight",
            "hdf5_link_local_imitation_tolerance_m",
            "hdf5_link_local_imitation_velocity_reward_weight",
            "hdf5_link_local_imitation_velocity_penalty_weight",
            "hdf5_link_local_imitation_velocity_sigma_m_per_s",
            "hdf5_link_local_imitation_err_derivative_penalty_weight",
            "hdf5_link_local_imitation_huber_delta_m",
            "hdf5_link_local_imitation_distance_cap_m",
            "hdf5_link_local_contact_tolerance_scale",
            "hdf5_link_local_imitation_pos_scale_x",
            "hdf5_ankle_roll_y_rot_imitation_reward_weight",
            "hdf5_ankle_roll_y_rot_imitation_penalty_weight",
            "hdf5_ankle_roll_y_rot_imitation_sigma_rad",
            "hdf5_ankle_roll_y_rot_imitation_tolerance_rad",
            "hdf5_hip_knee_y_rot_imitation_reward_weight",
            "hdf5_hip_knee_y_rot_imitation_sigma_rad",
            "hdf5_hip_knee_y_rot_imitation_tolerance_rad",
            "hdf5_hip_knee_y_rot_imitation_penalty_weight",
            "hdf5_ankle_roll_link_pos_imitation_reward_weight",
            "hdf5_ankle_roll_link_pos_imitation_sigma_m",
            "hdf5_ankle_roll_link_pos_imitation_penalty_weight",
            "hdf5_ankle_roll_link_pos_imitation_tolerance_m",
            "hdf5_hip_yaw_link_pos_imitation_reward_weight",
            "hdf5_hip_yaw_link_pos_imitation_sigma_m",
            "hdf5_hip_yaw_link_pos_imitation_penalty_weight",
            "hdf5_hip_yaw_link_pos_imitation_tolerance_m",
            "hdf5_hip_yaw_link_pos_imitation_reward_tolerance_m",
            "hdf5_hip_yaw_link_pos_pair_penalty_weight",
            "hdf5_hip_yaw_link_pos_pair_penalty_tolerance_m",
            "npz_link_local_imitation_reward_weight",
            "npz_link_local_imitation_penalty_weight",
            "npz_link_local_imitation_tolerance_m",
            "npz_link_local_imitation_velocity_reward_weight",
            "npz_link_local_imitation_velocity_penalty_weight",
            "npz_link_local_imitation_velocity_sigma_m_per_s",
            "npz_link_local_imitation_err_derivative_penalty_weight",
            "npz_link_local_imitation_huber_delta_m",
            "npz_link_local_imitation_distance_cap_m",
            "npz_link_local_contact_tolerance_scale",
            "npz_link_local_imitation_pos_scale_x",
            "alternating_foot_reward_weight",
            "same_foot_tap_penalty_weight",
            "jump_penalty_weight",
            "at_least_one_foot_contact_reward_weight",
            "ankle_roll_feet_target_separation_m",
            "ankle_roll_feet_separation_tolerance_m",
            "ankle_roll_feet_separation_reward_weight",
            "ankle_roll_feet_separation_penalty_weight",
            "ankle_roll_foot_height_reward_target_m",
            "ankle_roll_foot_height_reward_weight",
            "symmetry_reward_weight",
            "symmetry_penalty_weight",
            "symmetry_contact_sharpness",
            "episode_symmetry_min_episode_steps",
            "episode_symmetry_reward_weight",
            "episode_symmetry_step_length_blend",
            "episode_symmetry_joint_rom_blend",
            "episode_symmetry_link_rom_blend",
            "episode_symmetry_step_length_reward_weight",
            "episode_symmetry_step_length_penalty_weight",
            "episode_symmetry_joint_rom_reward_weight",
            "episode_symmetry_joint_rom_penalty_weight",
            "episode_symmetry_link_rom_reward_weight",
            "episode_symmetry_link_rom_penalty_weight",
            "episode_symmetry_step_length_sigma_m",
            "episode_symmetry_joint_rom_sigma_rad",
            "episode_symmetry_link_rom_sigma_m",
            "episode_symmetry_single_stance_balance_reward_weight",
            "episode_symmetry_single_stance_balance_penalty_weight",
            "episode_symmetry_single_stance_balance_sigma",
            "episode_symmetry_hdf5_ankle_roll_y_track_reward_weight",
            "episode_symmetry_hdf5_ankle_roll_y_track_penalty_weight",
            "episode_symmetry_hdf5_ankle_roll_y_track_sigma_rad",
            "torso_z_axis_world_align_reward_weight",
            "torso_z_axis_world_align_reward_sigma",
            "torso_z_axis_world_align_penalty_weight",
            "torso_z_axis_world_align_penalty_tolerance",
            "palm_below_pelvis_z_max_pelvis_frame_m",
            "palm_below_pelvis_z_reward_weight",
            "palm_below_pelvis_z_reward_sigma_m",
            "palm_below_pelvis_z_penalty_weight",
            "palm_below_pelvis_z_penalty_tolerance_m",
            "com_velocity_tracking_reward_weight",
            "com_velocity_tracking_sigma_mps",
            "linear_momentum_projection_reward_weight",
            "linear_momentum_projection_scale",
            "base_of_support_center_penalty_weight",
            "base_of_support_forward_lead_min_m",
            "dcm_capture_point_reward_weight",
            "dcm_capture_point_target_ahead_m",
            "dcm_capture_point_sigma_m",
            "dcm_capture_point_lateral_penalty_weight",
            "dcm_capture_point_lateral_tolerance_m",
            "dcm_com_height_min_m",
            "early_termination_time_penalty_weight",
            "symmetry_touchdown_sharpness",
            "symmetry_torque_sharpness",
            "step_length_min_m",
            "step_length_reward_weight",
            "step_length_penalty_weight",
            "step_length_smooth_sigma_m",
            "same_foot_tap_min_air_steps",
            "gait_single_stance_reward_weight",
            "hip_weight_shift_reward_weight",
            "hip_weight_shift_torque_scale",
            "alternating_foot_vel_scale_ref",
            "foot_crossing_reward_weight",
            "foot_crossing_penalty_weight",
            "lin_vel_z_penalty_weight",
            "lin_vel_z_tolerance",
            "ang_vel_xy_penalty_weight",
            "flat_orientation_penalty_weight",
            "flat_orientation_tolerance",
            "termination_height",
            "decimation",
            "action_rate_penalty_scale",
            "joint_acceleration_penalty_scale",
            "smooth_motion_reward_weight",
            "smooth_motion_sigma",
            "smooth_motion_joint_accel_reward_weight",
            "smooth_motion_joint_accel_sigma",
            "root_lin_vel_jerk_penalty_weight",
            "joint_limit_penalty_weight",
            "joint_limit_reward_weight",
            "joint_limit_reward_sigma",
            "hand_reward_weight",
            "hand_reward_sigma",
            # V3 optimizations (base height, curriculum, push)
            "base_height_reference",
            "base_height_reward_weight",
            "base_height_sigma",
            "base_height_penalty_weight",
            "base_height_penalty_tolerance_m",
            "curriculum_stage1_steps",
            "curriculum_stage2_steps",
            "num_steps_per_env",
            "max_push_vel_xy",
            "imitation_strength_ramp_steps",
            "adapter_phase_duration_steps",
            "adapter_phase_start_scale",
            "imitation_legs_weight",
            "imitation_arms_weight",
            "imitation_root_xy_reward_weight",
            "imitation_root_xy_penalty_weight",
            "imitation_root_xy_sigma_m",
            "swing_foot_clearance_reward_weight",
            "swing_foot_clearance_penalty_weight",
            "swing_foot_clearance_sigma_m",
            "swing_foot_clearance_margin_m",
            "stance_foot_slip_penalty_weight",
            "motion_cycle_episode_margin_s",
            "imitation_velocity_sigma_rad_s",
            "hdf5_derived_foot_contact_beta",
            "hdf5_link_local_stance_tolerance_scale",
            "hdf5_link_local_stance_ref_contact_threshold",
            "hdf5_link_local_imitation_swing_ref_contact_threshold",
            "imitation_joint_ref_contact_leg_scale",
            "staged_imitation_start_optimizer_steps",
            "staged_imitation_ramp_optimizer_steps",
            "staged_ramp_start_scale",
            "staged_dybl_start_optimizer_steps",
            "staged_dybl_ramp_optimizer_steps",
            "staged_symmetry_start_optimizer_steps",
            "staged_symmetry_ramp_optimizer_steps",
            "staged_symmetry_ramp_start_scale",
            "teleop_target_obs_scale",
            "velocity_command_obs_scale",
            "root_xy_error_obs_scale",
            "ppo_learning_rate",
            "ppo_max_grad_norm",
            "ppo_value_loss_coef",
            "ppo_entropy_coef",
            "reward_clip_min",
            "reward_clip_max",
            "divergence_mean_reward_min",
        ):
            m = re.search(rf"^\s*{re.escape(name)}\s*:\s*([\d.eE+-]+|None)\s*", head, re.MULTILINE)
            if m:
                try:
                    g = m.group(1)
                    out[name] = float(g) if g != "None" else None
                except ValueError:
                    pass
        # velocity_command_range can be tuple (0.2, 0.8) or None
        m_vel = re.search(r"^\s*velocity_command_range\s*:\s*(\([\d., ]+\)|None)\s*", head, re.MULTILINE)
        if m_vel:
            out["velocity_command_range"] = m_vel.group(1).strip()
        m_pg = re.search(r"^\s*progress_goal_world_offset_xy_m\s*:\s*(.+)$", head, re.MULTILINE)
        if m_pg:
            out["progress_goal_world_offset_xy_m"] = m_pg.group(1).strip()
        m_tz = re.search(r"^\s*torso_z_align_link_body_name\s*:\s*(.+)$", head, re.MULTILINE)
        if m_tz:
            out["torso_z_align_link_body_name"] = m_tz.group(1).strip()
        m_pr = re.search(r"^\s*palm_below_pelvis_right_body_name\s*:\s*(.+)$", head, re.MULTILINE)
        if m_pr:
            out["palm_below_pelvis_right_body_name"] = m_pr.group(1).strip()
        m_pl = re.search(r"^\s*palm_below_pelvis_left_body_name\s*:\s*(.+)$", head, re.MULTILINE)
        if m_pl:
            out["palm_below_pelvis_left_body_name"] = m_pl.group(1).strip()
        m_rp = re.search(r"^\s*reward_profile\s*:\s*(\S+)\s*", head, re.MULTILINE)
        if m_rp:
            out["reward_profile"] = m_rp.group(1).strip()
        for name in ("experiment_name", "device", "starting_policy_path"):
            m_s = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+)$", head, re.MULTILINE)
            if m_s:
                out[name] = m_s.group(1).strip()
        # Booleans: curriculum, push, gait / symmetry / step-length shaping
        for name in (
            "enable_curriculum",
            "enable_push_perturbation",
            "symmetry_use_geometric_mean",
            "step_length_use_smooth_reward",
            "step_length_penalty_squared",
            "alternating_foot_scale_with_forward_vel",
            "symmetry_include_hip_torque",
            "imitation_phase_in_obs",
            "imitation_pelvis_link_error_in_obs",
            "imitation_root_lin_vel_in_obs",
            "imitation_joint_position_use_leg_arm_weights",
            "imitation_joint_velocity_use_leg_arm_weights",
            "hdf5_imitation_periodic_joint_vel_fd",
            "hdf5_derive_foot_contact_from_link_z",
            "hdf5_link_local_imitation_swing_penalty_mask",
            "symmetry_torque_terms_double_support_only",
            "hdf5_link_use_joint_motion_clock_when_aligned",
            "hdf5_link_local_imitation_use_body_lin_vel",
            "adapter_phase_enabled",
            "velocity_command_obs_enabled",
            "include_reward_profile_phase_obs",
            "target_yaw_command_obs_enabled",
            "append_teleop_hand_target_obs",
            "include_root_xy_error_obs",
            "hdf5_link_local_imitation_periodic_fd",
            "hdf5_link_local_imitation_position_reward_penalty_xy_only",
        ):
            m_b = re.search(rf"^\s*{re.escape(name)}\s*:\s*(True|False)\s*", head, re.MULTILINE)
            if m_b:
                out[name] = m_b.group(1)
        for name in ("imitation_strength_ramp_easing", "adapter_phase_easing"):
            m_s = re.search(rf"^\s*{re.escape(name)}\s*:\s*(\S+)\s*", head, re.MULTILINE)
            if m_s:
                out[name] = m_s.group(1).strip()
        m_sw = re.search(
            r"^\s*(?:hdf5|npz)_link_local_imitation_slot_weights\s*:\s*(.+)$", head, re.MULTILINE
        )
        if m_sw:
            out["hdf5_link_local_imitation_slot_weights"] = m_sw.group(1).strip()
        m_jd = re.search(r"^\s*joints_data\s*:\s*(.+)$", head, re.MULTILINE)
        if m_jd:
            out["joints_data"] = m_jd.group(1).strip()
        return out if out else None
    except Exception:
        return None


def _episode_metric_forward_fill_policy(key: str) -> bool:
    """True for keys where a missing line must not inherit the previous block's value.

    Otherwise resume / shorter log blocks carry stale positives (e.g. dropped HDF5 terms when weight=0).
    """
    return key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/")


def parse_training_progress_log(path: str) -> tuple[list[int], dict[str, list[float]]]:
    """Parse full training_progress_log.txt. Returns (iterations, {metric: [values]}).

    Non-episode metrics forward-fill missing steps. ``Episode_Reward/*`` and ``Episode_Termination/*``
    do not: absent keys become NaN after any prior numeric value (matplotlib gaps); leading missing
    steps stay 0.0 until the first logged value.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"(?=^################################################################################\n)", content, flags=re.MULTILINE)
    blocks = [b.strip() for b in blocks if b.strip()]

    iterations = []
    all_keys = set()
    block_data = []
    for block in blocks:
        it, data = _parse_block(block)
        if it is not None and data:
            iterations.append(it)
            block_data.append(data)
            all_keys |= set(data.keys())

    if not block_data:
        return [], {}

    series = {k: [] for k in all_keys}
    for data in block_data:
        for k in all_keys:
            series[k].append(data.get(k))

    # Forward-fill non-episode metrics only. Episode_* absent in a block is NaN (not last value).
    for k in list(series.keys()):
        prev = None
        for i, v in enumerate(series[k]):
            if _episode_metric_forward_fill_policy(k):
                if v is not None:
                    prev = float(v)
                    series[k][i] = prev
                elif prev is not None:
                    series[k][i] = float("nan")
                else:
                    series[k][i] = 0.0
            else:
                if v is not None:
                    prev = v
                elif prev is not None:
                    series[k][i] = prev
                else:
                    series[k][i] = 0.0

    return iterations, series


# Config key -> Episode_Reward key: only plot when parsed weight/scale is non-zero.
REWARD_POSITIVE_WEIGHTS = (
    ("up_weight", "Episode_Reward/up_reward"),
    ("heading_weight", "Episode_Reward/heading_reward"),
    ("alive_reward_scale", "Episode_Reward/alive_reward"),
    ("progress_reward_multiplier", "Episode_Reward/progress_reward"),
    ("progress_reward_world_vx_blend", "Episode_Reward/progress_reward"),
    ("progress_reward_forward_displacement_weight", "Episode_Reward/progress_reward"),
    ("velocity_reward_weight", "Episode_Reward/velocity_reward"),
    ("x_drift_reward_weight", "Episode_Reward/x_drift_reward"),
    ("y_drift_reward_weight", "Episode_Reward/y_drift_reward"),
    ("z_orientation_reward_weight", "Episode_Reward/z_orientation_reward"),
    ("torso_z_axis_world_align_reward_weight", "Episode_Reward/torso_z_axis_world_align_reward"),
    ("palm_below_pelvis_z_reward_weight", "Episode_Reward/palm_below_pelvis_z_reward"),
    ("com_velocity_tracking_reward_weight", "Episode_Reward/com_velocity_tracking_reward"),
    ("linear_momentum_projection_reward_weight", "Episode_Reward/linear_momentum_projection_reward"),
    ("dcm_capture_point_reward_weight", "Episode_Reward/dcm_capture_point_reward"),
    ("base_height_reward_weight", "Episode_Reward/base_height_reward"),
    ("base_height_penalty_weight", "Episode_Reward/base_height_penalty"),
    ("smooth_motion_reward_weight", "Episode_Reward/policy_action_smoothness_reward"),
    ("smooth_motion_joint_accel_reward_weight", "Episode_Reward/joint_velocity_smoothness_reward"),
    ("joint_limit_reward_weight", "Episode_Reward/joint_limit_reward"),
    ("alternating_foot_reward_weight", "Episode_Reward/alternating_foot_reward"),
    ("symmetry_reward_weight", "Episode_Reward/symmetry_reward"),
    ("episode_symmetry_reward_weight", "Episode_Reward/episode_symmetry_reward"),
    ("episode_symmetry_step_length_reward_weight", "Episode_Reward/episode_symmetry_step_length_reward"),
    ("episode_symmetry_joint_rom_reward_weight", "Episode_Reward/episode_symmetry_joint_rom_reward"),
    ("episode_symmetry_link_rom_reward_weight", "Episode_Reward/episode_symmetry_link_rom_reward"),
    (
        "episode_symmetry_single_stance_balance_reward_weight",
        "Episode_Reward/episode_symmetry_single_stance_balance_reward",
    ),
    (
        "episode_symmetry_hdf5_ankle_roll_y_track_reward_weight",
        "Episode_Reward/episode_symmetry_hdf5_ankle_roll_y_track_reward",
    ),
    ("step_length_reward_weight", "Episode_Reward/step_length_reward"),
    ("foot_crossing_reward_weight", "Episode_Reward/foot_crossing_reward"),
    ("gait_single_stance_reward_weight", "Episode_Reward/gait_single_stance_reward"),
    ("at_least_one_foot_contact_reward_weight", "Episode_Reward/at_least_one_foot_contact_reward"),
    ("ankle_roll_feet_separation_reward_weight", "Episode_Reward/ankle_roll_feet_separation_reward"),
    ("hip_weight_shift_reward_weight", "Episode_Reward/hip_weight_shift_reward"),
    ("imitation_joints_position_reward_weight", "Episode_Reward/imitation_reward"),
    ("imitation_velocity_reward_weight", "Episode_Reward/imitation_velocity_reward"),
    ("hdf5_link_local_imitation_reward_weight", "Episode_Reward/hdf5_link_local_imitation_reward"),
    ("hdf5_link_local_imitation_reward_weight", "Episode_Reward/npz_link_local_imitation_reward"),
    ("hdf5_link_local_imitation_velocity_reward_weight", "Episode_Reward/hdf5_link_local_imitation_velocity_reward"),
    ("hdf5_link_local_imitation_velocity_reward_weight", "Episode_Reward/npz_link_local_imitation_velocity_reward"),
    ("hdf5_ankle_roll_y_rot_imitation_reward_weight", "Episode_Reward/hdf5_ankle_roll_y_rot_imitation_reward"),
    ("hdf5_ankle_roll_link_pos_imitation_reward_weight", "Episode_Reward/hdf5_ankle_roll_link_pos_imitation_reward"),
    ("hdf5_hip_yaw_link_pos_imitation_reward_weight", "Episode_Reward/hdf5_hip_yaw_link_pos_imitation_reward"),
    ("hdf5_hip_knee_y_rot_imitation_reward_weight", "Episode_Reward/hdf5_hip_knee_y_rot_imitation_reward"),
    ("hand_reward_weight", "Episode_Reward/hand_position_reward"),
    ("imitation_root_xy_reward_weight", "Episode_Reward/imitation_root_xy_reward"),
    ("swing_foot_clearance_reward_weight", "Episode_Reward/swing_foot_clearance_reward"),
)
REWARD_PENALTY_WEIGHTS = (
    ("actions_cost_scale", "Episode_Reward/actions_cost"),
    ("energy_cost_scale", "Episode_Reward/electricity_cost"),
    ("joint_limit_penalty_weight", "Episode_Reward/joint_limit_cost"),
    ("y_drift_penalty_weight", "Episode_Reward/y_drift_penalty"),
    ("z_orientation_penalty_weight", "Episode_Reward/z_orientation_penalty"),
    ("torso_z_axis_world_align_penalty_weight", "Episode_Reward/torso_z_axis_world_align_penalty"),
    ("palm_below_pelvis_z_penalty_weight", "Episode_Reward/palm_below_pelvis_z_penalty"),
    ("base_of_support_center_penalty_weight", "Episode_Reward/base_of_support_center_penalty"),
    ("dcm_capture_point_lateral_penalty_weight", "Episode_Reward/dcm_capture_point_lateral_penalty"),
    ("early_termination_time_penalty_weight", "Episode_Reward/early_termination_time_penalty"),
    ("x_drift_penalty_weight", "Episode_Reward/x_drift_penalty"),
    ("same_foot_tap_penalty_weight", "Episode_Reward/same_foot_double_non_contact_penalty"),
    ("ankle_roll_feet_separation_penalty_weight", "Episode_Reward/ankle_roll_feet_separation_penalty"),
    ("ankle_roll_foot_height_reward_weight", "Episode_Reward/ankle_roll_left_foot_height_reward"),
    ("ankle_roll_foot_height_reward_weight", "Episode_Reward/ankle_roll_right_foot_height_reward"),
    ("jump_penalty_weight", "Episode_Reward/both_feet_non_contact_penalty"),
    ("swing_foot_clearance_penalty_weight", "Episode_Reward/swing_foot_clearance_penalty"),
    ("lin_vel_z_penalty_weight", "Episode_Reward/lin_vel_z_penalty"),
    ("ang_vel_xy_penalty_weight", "Episode_Reward/ang_vel_xy_penalty"),
    ("flat_orientation_penalty_weight", "Episode_Reward/flat_orientation_penalty"),
    ("target_x_velocity_penalty_weight", "Episode_Reward/target_x_velocity_penalty"),
    ("symmetry_penalty_weight", "Episode_Reward/symmetry_penalty"),
    ("episode_symmetry_step_length_penalty_weight", "Episode_Reward/episode_symmetry_step_length_penalty"),
    ("episode_symmetry_joint_rom_penalty_weight", "Episode_Reward/episode_symmetry_joint_rom_penalty"),
    ("episode_symmetry_link_rom_penalty_weight", "Episode_Reward/episode_symmetry_link_rom_penalty"),
    (
        "episode_symmetry_single_stance_balance_penalty_weight",
        "Episode_Reward/episode_symmetry_single_stance_balance_penalty",
    ),
    (
        "episode_symmetry_hdf5_ankle_roll_y_track_penalty_weight",
        "Episode_Reward/episode_symmetry_hdf5_ankle_roll_y_track_penalty",
    ),
    ("step_length_penalty_weight", "Episode_Reward/step_length_penalty"),
    ("foot_crossing_penalty_weight", "Episode_Reward/foot_crossing_penalty"),
    ("imitation_joints_position_penalty_weight", "Episode_Reward/imitation_penalty"),
    ("imitation_velocity_penalty_weight", "Episode_Reward/imitation_velocity_penalty"),
    ("hdf5_link_local_imitation_penalty_weight", "Episode_Reward/hdf5_link_local_imitation_penalty"),
    ("hdf5_link_local_imitation_penalty_weight", "Episode_Reward/npz_link_local_imitation_penalty"),
    ("hdf5_link_local_imitation_velocity_penalty_weight", "Episode_Reward/hdf5_link_local_imitation_velocity_penalty"),
    ("hdf5_link_local_imitation_velocity_penalty_weight", "Episode_Reward/npz_link_local_imitation_velocity_penalty"),
    ("hdf5_link_local_imitation_err_derivative_penalty_weight", "Episode_Reward/hdf5_link_local_imitation_err_derivative_penalty"),
    ("hdf5_link_local_imitation_err_derivative_penalty_weight", "Episode_Reward/npz_link_local_imitation_err_derivative_penalty"),
    ("xy_drift_penalty_weight", "Episode_Reward/xy_drift_penalty"),
    ("stance_foot_slip_penalty_weight", "Episode_Reward/stance_foot_slip_penalty"),
    ("imitation_root_xy_penalty_weight", "Episode_Reward/imitation_root_xy_penalty"),
    ("root_lin_vel_jerk_penalty_weight", "Episode_Reward/root_lin_vel_jerk_penalty"),
    ("action_rate_penalty_scale", "Episode_Reward/policy_action_roughness_penalty"),
    ("joint_acceleration_penalty_scale", "Episode_Reward/joint_velocity_roughness_penalty"),
    ("hdf5_hip_knee_y_rot_imitation_penalty_weight", "Episode_Reward/hdf5_hip_knee_y_rot_imitation_penalty"),
    ("hdf5_ankle_roll_y_rot_imitation_penalty_weight", "Episode_Reward/hdf5_ankle_roll_y_rot_imitation_penalty"),
    ("hdf5_ankle_roll_link_pos_imitation_penalty_weight", "Episode_Reward/hdf5_ankle_roll_link_pos_imitation_penalty"),
    ("hdf5_hip_yaw_link_pos_imitation_penalty_weight", "Episode_Reward/hdf5_hip_yaw_link_pos_imitation_penalty"),
    ("hdf5_hip_yaw_link_pos_pair_penalty_weight", "Episode_Reward/hdf5_hip_yaw_link_pos_pair_penalty"),
)
# Penalties/costs with no config weight: always show if present in series.
REWARD_PENALTIES_ALWAYS = [
    "Episode_Reward/dof_at_limit_cost",
    "Episode_Reward/death_cost",
]

# Episode_Reward/* tail → short legend (matplotlib)
EPISODE_REWARD_SHORT_LABEL: dict[str, str] = {
    "velocity_reward": "vel (target vx)",
    "velocity_reward_pre_curriculum": "vel (pre-curr)",
    "x_drift_reward": "drift x (reward)",
    "y_drift_reward": "drift y (reward)",
    "alternating_foot_reward": "alt feet (L-R)",
    "symmetry_reward": "symmetry (L-R gait)",
    "symmetry_penalty": "symmetry pen (gait)",
    "torso_z_axis_world_align_reward": "torso +Z vs world +Z",
    "torso_z_axis_world_align_penalty": "torso +Z misalign pen",
    "palm_below_pelvis_z_reward": "palms z ≤ pelvis (rew)",
    "palm_below_pelvis_z_penalty": "palms z above cap (pen)",
    "com_velocity_tracking_reward": "CoM vel track",
    "linear_momentum_projection_reward": "forward lin momentum",
    "base_of_support_center_penalty": "BoS center pen",
    "dcm_capture_point_reward": "DCM capture-point",
    "dcm_capture_point_lateral_penalty": "DCM lateral pen",
    "early_termination_time_penalty": "early fall vs horizon",
    "episode_symmetry_reward": "ep sym (legacy blend)",
    "episode_symmetry_step_length_reward": "ep sym step L/R",
    "episode_symmetry_step_length_penalty": "ep sym step pen",
    "episode_symmetry_joint_rom_reward": "ep sym joint ROM",
    "episode_symmetry_joint_rom_penalty": "ep sym joint ROM pen",
    "episode_symmetry_link_rom_reward": "ep sym link ROM",
    "episode_symmetry_link_rom_penalty": "ep sym link ROM pen",
    "episode_symmetry_single_stance_balance_reward": "ep sym single-stance L/R",
    "episode_symmetry_single_stance_balance_penalty": "ep sym single-stance pen",
    "episode_symmetry_hdf5_ankle_roll_y_track_reward": "ep sym ankle y-rot track L/R",
    "episode_symmetry_hdf5_ankle_roll_y_track_penalty": "ep sym ankle y-rot track pen",
    "step_length_reward": "step length (≥ min)",
    "gait_single_stance_reward": "single stance",
    "at_least_one_foot_contact_reward": "≥1 foot contact",
    "ankle_roll_feet_separation_reward": "ankle L-R dist in band",
    "ankle_roll_feet_separation_penalty": "ankle L-R outside band",
    "ankle_roll_left_foot_height_reward": "L ankle z < tgt",
    "ankle_roll_right_foot_height_reward": "R ankle z < tgt",
    "hip_weight_shift_reward": "hip load |τ|",
    "foot_crossing_reward": "foot crossing",
    "imitation_reward": "imit joints pos",
    "imitation_velocity_reward": "imit joint vel",
    "hdf5_link_local_imitation_reward": "link pos (pelvis)",
    "npz_link_local_imitation_reward": "link pos (pelvis)",
    "hdf5_link_local_imitation_velocity_reward": "link vel (pelvis)",
    "npz_link_local_imitation_velocity_reward": "link vel (pelvis)",
    "base_height_reward": "base height z",
    "base_height_penalty": "base height pen (|z-ref|)",
    "imitation_legs_bonus": "imit legs bonus",
    "imitation_root_xy_reward": "root xy vs motion",
    "swing_foot_clearance_reward": "swing clearance",
    "swing_foot_clearance_penalty": "swing clearance pen",
    "hdf5_ankle_roll_y_rot_imitation_reward": "ankle roll (motion)",
    "hdf5_ankle_roll_y_rot_imitation_penalty": "ankle roll vs clip (MSE)",
    "hdf5_hip_knee_y_rot_imitation_reward": "hip/knee Y (motion)",
    "hand_position_reward": "hand pos",
    "progress_reward": "progress (+x)",
    "up_reward": "upright",
    "heading_reward": "heading",
    "heading_reward_gross": "heading (gross)",
    "z_orientation_reward": "yaw reward (exp)",
    "alive_reward": "alive",
    "policy_action_smoothness_reward": "policy daction smooth (exp)",
    "joint_velocity_smoothness_reward": "joint dvel smooth (exp)",
    "policy_action_roughness_penalty": "policy daction rough (Huber/L2)",
    "joint_velocity_roughness_penalty": "joint dvel rough (Huber/L2)",
    "smooth_reward": "smooth motion (legacy)",
    "joint_limit_cost": "joint limit (clip)",
    "joint_limit_reward": "joint limit (in-bounds)",
    "y_drift_penalty": "drift y",
    "x_drift_penalty": "drift x",
    "xy_drift_penalty": "drift xy",
    "z_orientation_penalty": "yaw penalty",
    "same_foot_double_non_contact_penalty": "same-foot tap",
    "both_feet_non_contact_penalty": "jump (both air)",
    "step_length_penalty": "step length pen",
    "foot_crossing_penalty": "foot crossing pen",
    "imitation_penalty": "imit joints pen",
    "imitation_velocity_penalty": "imit vel pen",
    "hdf5_link_local_imitation_velocity_penalty": "link vel pen",
    "npz_link_local_imitation_velocity_penalty": "link vel pen",
    "hdf5_link_local_imitation_penalty": "link pos pen",
    "npz_link_local_imitation_penalty": "link pos pen",
    "hdf5_link_local_imitation_err_derivative_penalty": "link err Δ pen",
    "npz_link_local_imitation_err_derivative_penalty": "link err Δ pen",
    "lin_vel_z_penalty": "lin vz",
    "ang_vel_xy_penalty": "ang vxy",
    "flat_orientation_penalty": "tilt pen",
    "target_x_velocity_penalty": "vx mismatch",
    "stance_foot_slip_penalty": "stance slip",
    "root_lin_vel_jerk_penalty": "root Δv jerk",
    "imitation_root_xy_penalty": "root xy pen",
    "hdf5_hip_knee_y_rot_imitation_penalty": "hip/knee Y (penalty)",
    "hdf5_ankle_roll_link_pos_imitation_reward": "ankle foot xyz (motion)",
    "hdf5_ankle_roll_link_pos_imitation_penalty": "ankle foot xyz pen",
    "hdf5_hip_yaw_link_pos_imitation_reward": "hip yaw xyz (motion)",
    "hdf5_hip_yaw_link_pos_imitation_penalty": "hip yaw xyz pen",
    "hdf5_hip_yaw_link_pos_pair_penalty": "hip yaw pair R-L",
    "actions_cost": "action cost",
    "electricity_cost": "energy",
    "jerk_cost": "jerk (legacy)",
    "death_cost": "death",
    "dof_at_limit_cost": "dof @ limit",
}

IMITATION_PLOT_EMPHASIS_LABELS: frozenset[str] = frozenset(
    {
        "imit joints pos",
        "imit joint vel",
        "imit legs bonus",
        "link pos (pelvis)",
        "link vel (pelvis)",
        "ankle roll (motion)",
        "ankle foot xyz (motion)",
        "hip yaw xyz (motion)",
        "hip yaw pair R-L",
        "hip/knee Y (motion)",
    }
)

# General penalty panel: thicker lines for high-signal motion-clip costs.
PENALTY_PLOT_EMPHASIS_LABELS: frozenset[str] = frozenset(
    {
        "ankle roll vs clip (MSE)",
        "hip/knee Y (penalty)",
        "ankle foot xyz pen",
    }
)

# Symmetry panel: emphasize newer episode-level terms when enabled.
SYMMETRY_PLOT_EMPHASIS_LABELS: frozenset[str] = frozenset(
    {
        "ep sym single-stance L/R",
        "ep sym single-stance pen",
        "ep sym ankle y-rot track L/R",
        "ep sym ankle y-rot track pen",
    }
)

# Dynamic-balance panels: emphasize the new CoM / DCM terms.
DYNAMIC_BALANCE_PLOT_EMPHASIS_LABELS: frozenset[str] = frozenset(
    {
        "CoM vel track",
        "forward lin momentum",
        "DCM capture-point",
        "BoS center pen",
        "DCM lateral pen",
    }
)

# Episode_Reward/* tails that belong to the **imitation** positive panel (V4 layout).
_V4_IMITATION_POSITIVE_TAILS: frozenset[str] = frozenset(
    {
        "imitation_reward",
        "imitation_velocity_reward",
        "imitation_legs_bonus",
        "hdf5_link_local_imitation_reward",
        "npz_link_local_imitation_reward",
        "hdf5_link_local_imitation_velocity_reward",
        "npz_link_local_imitation_velocity_reward",
        "hdf5_ankle_roll_y_rot_imitation_reward",
        "hdf5_ankle_roll_link_pos_imitation_reward",
        "hdf5_hip_yaw_link_pos_imitation_reward",
        "hdf5_hip_knee_y_rot_imitation_reward",
        "imitation_root_xy_reward",
    }
)


def _split_v4_positive_rewards(keys: list[str]) -> tuple[list[str], list[str]]:
    """Split positive Episode_Reward keys into locomotion-task vs imitation (V4 structure)."""
    loco: list[str] = []
    imi: list[str] = []
    for k in keys:
        if not k.startswith("Episode_Reward/"):
            loco.append(k)
            continue
        tail = k[len("Episode_Reward/") :]
        if tail in _V4_IMITATION_POSITIVE_TAILS:
            imi.append(k)
        else:
            loco.append(k)
    return _ordered_unique(loco), _ordered_unique(imi)


# Shown on the dedicated symmetry subplot only (removed from locomotion / general penalty panels).
_SYMMETRY_DEDICATED_POSITIVE: frozenset[str] = frozenset(
    {
        "Episode_Reward/symmetry_reward",
        "Episode_Reward/episode_symmetry_reward",
        "Episode_Reward/episode_symmetry_step_length_reward",
        "Episode_Reward/episode_symmetry_joint_rom_reward",
        "Episode_Reward/episode_symmetry_link_rom_reward",
        "Episode_Reward/episode_symmetry_single_stance_balance_reward",
        "Episode_Reward/episode_symmetry_hdf5_ankle_roll_y_track_reward",
    }
)
_SYMMETRY_DEDICATED_PENALTY: frozenset[str] = frozenset(
    {
        "Episode_Reward/symmetry_penalty",
        "Episode_Reward/episode_symmetry_step_length_penalty",
        "Episode_Reward/episode_symmetry_joint_rom_penalty",
        "Episode_Reward/episode_symmetry_link_rom_penalty",
        "Episode_Reward/episode_symmetry_single_stance_balance_penalty",
        "Episode_Reward/episode_symmetry_hdf5_ankle_roll_y_track_penalty",
    }
)
_SYMMETRY_PANEL_ORDER_POSITIVE: tuple[str, ...] = (
    "Episode_Reward/symmetry_reward",
    "Episode_Reward/episode_symmetry_reward",
    "Episode_Reward/episode_symmetry_step_length_reward",
    "Episode_Reward/episode_symmetry_joint_rom_reward",
    "Episode_Reward/episode_symmetry_link_rom_reward",
    "Episode_Reward/episode_symmetry_single_stance_balance_reward",
    "Episode_Reward/episode_symmetry_hdf5_ankle_roll_y_track_reward",
)
_SYMMETRY_PANEL_ORDER_PENALTY: tuple[str, ...] = (
    "Episode_Reward/symmetry_penalty",
    "Episode_Reward/episode_symmetry_step_length_penalty",
    "Episode_Reward/episode_symmetry_joint_rom_penalty",
    "Episode_Reward/episode_symmetry_link_rom_penalty",
    "Episode_Reward/episode_symmetry_single_stance_balance_penalty",
    "Episode_Reward/episode_symmetry_hdf5_ankle_roll_y_track_penalty",
)

# Shown on dedicated dynamic-balance panels only (removed from locomotion / general penalty panels).
_DYNAMIC_BALANCE_DEDICATED_POSITIVE: frozenset[str] = frozenset(
    {
        "Episode_Reward/com_velocity_tracking_reward",
        "Episode_Reward/linear_momentum_projection_reward",
        "Episode_Reward/dcm_capture_point_reward",
    }
)
_DYNAMIC_BALANCE_DEDICATED_PENALTY: frozenset[str] = frozenset(
    {
        "Episode_Reward/base_of_support_center_penalty",
        "Episode_Reward/dcm_capture_point_lateral_penalty",
    }
)
_DYNAMIC_BALANCE_PANEL_ORDER_POSITIVE: tuple[str, ...] = (
    "Episode_Reward/com_velocity_tracking_reward",
    "Episode_Reward/linear_momentum_projection_reward",
    "Episode_Reward/dcm_capture_point_reward",
)
_DYNAMIC_BALANCE_PANEL_ORDER_PENALTY: tuple[str, ...] = (
    "Episode_Reward/base_of_support_center_penalty",
    "Episode_Reward/dcm_capture_point_lateral_penalty",
)


def _extract_dynamic_balance_dedicated_keys(
    loco_positive: list[str],
    penalties: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Expose dynamic-balance metrics for dedicated panels without removing them from the main panels."""
    dyn_p = [k for k in loco_positive if k in _DYNAMIC_BALANCE_DEDICATED_POSITIVE]
    dyn_n = [k for k in penalties if k in _DYNAMIC_BALANCE_DEDICATED_PENALTY]
    loco2 = list(loco_positive)
    pen2 = list(penalties)
    return loco2, pen2, dyn_p, dyn_n


def _extract_symmetry_dedicated_keys(
    loco_positive: list[str],
    penalties: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Pull L/R symmetry metrics into a dedicated panel; return filtered loco list, penalties, sym+, sym-."""
    sym_p = [k for k in loco_positive if k in _SYMMETRY_DEDICATED_POSITIVE]
    sym_n = [k for k in penalties if k in _SYMMETRY_DEDICATED_PENALTY]
    loco2 = [k for k in loco_positive if k not in _SYMMETRY_DEDICATED_POSITIVE]
    pen2 = [k for k in penalties if k not in _SYMMETRY_DEDICATED_PENALTY]
    return loco2, pen2, sym_p, sym_n


def _order_symmetry_keys(keys: list[str], preferred: tuple[str, ...]) -> list[str]:
    """Stable order: *preferred* first, then remaining keys."""
    seen: set[str] = set()
    out: list[str] = []
    for k in preferred:
        if k in keys and k not in seen:
            out.append(k)
            seen.add(k)
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


# HDF5 vs legacy npz_* log keys for the same link-imitation terms (prefer HDF5 in plots when both listed).
_HDF5_NPZ_LINK_EPISODE_PAIRS: tuple[tuple[str, str], ...] = (
    ("Episode_Reward/hdf5_link_local_imitation_reward", "Episode_Reward/npz_link_local_imitation_reward"),
    ("Episode_Reward/hdf5_link_local_imitation_penalty", "Episode_Reward/npz_link_local_imitation_penalty"),
    (
        "Episode_Reward/hdf5_link_local_imitation_velocity_reward",
        "Episode_Reward/npz_link_local_imitation_velocity_reward",
    ),
    (
        "Episode_Reward/hdf5_link_local_imitation_velocity_penalty",
        "Episode_Reward/npz_link_local_imitation_velocity_penalty",
    ),
    (
        "Episode_Reward/hdf5_link_local_imitation_err_derivative_penalty",
        "Episode_Reward/npz_link_local_imitation_err_derivative_penalty",
    ),
)


def _dedupe_hdf5_npz_link_plot_keys(keys: list[str], series: dict[str, list]) -> list[str]:
    """Drop npz_* episode keys when the matching hdf5_* key is present (same metric, current env logs hdf5_*)."""
    drop: set[str] = set()
    for hdf5_k, npz_k in _HDF5_NPZ_LINK_EPISODE_PAIRS:
        if hdf5_k in keys and npz_k in keys and hdf5_k in series:
            drop.add(npz_k)
    return [k for k in keys if k not in drop]


def _ordered_unique(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _omit_zero_cfg_value_line(line: str) -> bool:
    """True → drop line from config sidebar (value after ': ' parses as numeric 0)."""
    if ":" not in line:
        return False
    _, sep, val = line.partition(": ")
    if sep != ": ":
        return False
    val = val.strip()
    if val in ("—", "None", ""):
        return False
    try:
        return float(val) == 0.0
    except ValueError:
        return False


def _filter_config_sidebar_lines(lines: list[str]) -> list[str]:
    """Remove ``key: 0.0`` (or ``0``) rows; keep section titles and non-numeric values."""
    return [ln for ln in lines if not _omit_zero_cfg_value_line(ln)]


def _last_finite_value(values: Sequence[float | None], *, penalty_negative_display: bool = False) -> float | None:
    """Return the last finite plotted y-value from a metric series."""
    for raw in reversed(values):
        if raw is None:
            continue
        try:
            y = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y):
            continue
        return -abs(y) if penalty_negative_display else y
    return None


def _format_last_value_suffix(values: Sequence[float | None], *, penalty_negative_display: bool = False) -> str:
    """Compact legend suffix showing the latest plotted y-value."""
    y = _last_finite_value(values, penalty_negative_display=penalty_negative_display)
    if y is None:
        return ""
    return f" [{y:.3g}]"


def _classify_episode_reward_key(ep_key: str) -> str | None:
    """Return 'positive' or 'penalty' for Episode_Reward/* tails, or None if not a standard reward metric."""
    if not ep_key.startswith("Episode_Reward/"):
        return None
    tail = ep_key[len("Episode_Reward/") :]
    if tail.endswith("_penalty") or tail.endswith("_cost"):
        return "penalty"
    if tail.endswith("_reward") or tail.endswith("_bonus"):
        return "positive"
    return None


def merge_episode_reward_keys_from_series(
    series: dict[str, list],
    positive: list[str],
    penalty: list[str],
) -> tuple[list[str], list[str]]:
    """Append any Episode_Reward/* present in *series* but missing from plot lists (future env keys / header gaps)."""
    known = set(positive) | set(penalty)
    for k in series:
        if not k.startswith("Episode_Reward/"):
            continue
        if k in known:
            continue
        cat = _classify_episode_reward_key(k)
        if cat == "positive":
            positive.append(k)
            known.add(k)
        elif cat == "penalty":
            penalty.append(k)
            known.add(k)
    positive = _dedupe_hdf5_npz_link_plot_keys(_ordered_unique(positive), series)
    penalty = _dedupe_hdf5_npz_link_plot_keys(_ordered_unique(penalty), series)
    return positive, penalty


CATEGORY_MAIN_LEFT = ["Mean reward"]
CATEGORY_LOSSES = [
    "Mean value_function loss",
    "Mean surrogate loss",
    "Mean entropy loss",
    "Mean action noise std",
]


def _format_progress_forward_signal_summary(reward_cfg: dict[str, float | str | None]) -> str:
    """One-line summary of world +x progress knobs for plot sidebar."""
    keys = (
        "progress_reward_multiplier",
        "progress_reward_world_vx_blend",
        "progress_reward_forward_displacement_weight",
        "progress_reward_forward_displacement_threshold_m",
    )
    parts: list[str] = []
    for k in keys:
        v = reward_cfg.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            parts.append(f"{k}={v}")
            continue
        if k.endswith("_threshold_m"):
            short = "th_m"
        elif k == "progress_reward_multiplier":
            short = "mult"
        elif k == "progress_reward_world_vx_blend":
            short = "vx_blend"
        else:
            short = "disp_w"
        parts.append(f"{short}={fv:g}")
    pg = reward_cfg.get("progress_goal_world_offset_xy_m")
    if pg is not None and str(pg).strip() not in ("", "None"):
        parts.append(f"goal_xy={pg}")
    return " · ".join(parts) if parts else "—"


def _joint_pos_imitation_header_weight(reward_cfg: dict[str, float | str | None], *, reward: bool) -> float | None:
    """Resolve joint-angle motion imitation weight from log header (new or deprecated keys)."""
    keys = (
        ("imitation_joints_position_reward_weight", "imitation_position_reward_weight", "imitation_reward_weight")
        if reward
        else ("imitation_joints_position_penalty_weight", "imitation_position_penalty_weight", "imitation_penalty_weight")
    )
    for k in keys:
        if k in reward_cfg and reward_cfg[k] is not None:
            try:
                return float(reward_cfg[k])
            except (TypeError, ValueError):
                continue
    return None


def _ensure_hdf5_link_weights_from_npz_aliases(reward_cfg: dict[str, float | str | None]) -> None:
    """Old logs used npz_link_local_* header keys; normalize to hdf5_* for REWARD_* tables."""
    pairs = (
        ("hdf5_link_local_imitation_reward_weight", "npz_link_local_imitation_reward_weight"),
        ("hdf5_link_local_imitation_penalty_weight", "npz_link_local_imitation_penalty_weight"),
        ("hdf5_link_local_imitation_velocity_reward_weight", "npz_link_local_imitation_velocity_reward_weight"),
        ("hdf5_link_local_imitation_velocity_penalty_weight", "npz_link_local_imitation_velocity_penalty_weight"),
        ("hdf5_link_local_imitation_err_derivative_penalty_weight", "npz_link_local_imitation_err_derivative_penalty_weight"),
    )
    for hdf5_k, npz_k in pairs:
        if reward_cfg.get(hdf5_k) is None and npz_k in reward_cfg:
            reward_cfg[hdf5_k] = reward_cfg[npz_k]


def _reward_lists_from_config(reward_cfg: dict[str, float | str | None] | None) -> tuple[list[str], list[str]]:
    """Build positive and penalty reward lists from parsed header; exclude weight=0."""
    positive: list[str] = []
    penalty: list[str] = []
    if reward_cfg is not None:
        _ensure_hdf5_link_weights_from_npz_aliases(reward_cfg)
        seen_ep: set[str] = set()
        for cfg_key, ep_key in REWARD_POSITIVE_WEIGHTS:
            if cfg_key == "imitation_joints_position_reward_weight":
                w = _joint_pos_imitation_header_weight(reward_cfg, reward=True)
            else:
                w = reward_cfg.get(cfg_key)
            if w is not None and float(w) != 0:
                if ep_key not in seen_ep:
                    positive.append(ep_key)
                    seen_ep.add(ep_key)
        # Net heading_reward is zeroed when z_orientation_penalty replaces it; gross is logged separately.
        try:
            zo = float(reward_cfg.get("z_orientation_penalty_weight") or 0.0)
            hw = float(reward_cfg.get("heading_weight") or 0.0)
            if zo != 0.0 and hw != 0.0:
                positive.append("Episode_Reward/heading_reward_gross")
        except (TypeError, ValueError):
            pass
        # Curriculum can scale net velocity_reward to ~0 in stage 1; unscaled kernel is logged separately.
        try:
            if bool(reward_cfg.get("enable_curriculum")) and float(reward_cfg.get("velocity_reward_weight") or 0.0) != 0.0:
                positive.append("Episode_Reward/velocity_reward_pre_curriculum")
        except (TypeError, ValueError):
            pass
        # Leg emphasis bonus logged only when imitation_legs_weight != 1
        ilw = reward_cfg.get("imitation_legs_weight")
        if ilw is not None:
            try:
                if float(ilw) != 1.0:
                    positive.append("Episode_Reward/imitation_legs_bonus")
            except (TypeError, ValueError):
                pass
        for cfg_key, ep_key in REWARD_PENALTY_WEIGHTS:
            if cfg_key == "imitation_joints_position_penalty_weight":
                w = _joint_pos_imitation_header_weight(reward_cfg, reward=False)
            else:
                w = reward_cfg.get(cfg_key)
            if w is not None and float(w) != 0:
                penalty.append(ep_key)
    penalty.extend(REWARD_PENALTIES_ALWAYS)
    positive = _ordered_unique(positive)
    penalty = _ordered_unique(penalty)
    return positive, penalty


def _series_has_episode_reward_key(series: dict[str, list], ep_key: str) -> bool:
    """True if *series* has usable values for this Episode_Reward key (or HDF5/npz pair alternate)."""
    vals = series.get(ep_key)
    if vals and any(v is not None for v in vals):
        return True
    for hdf5_k, npz_k in _HDF5_NPZ_LINK_EPISODE_PAIRS:
        if ep_key == npz_k:
            alt = series.get(hdf5_k)
            if alt and any(v is not None for v in alt):
                return True
        if ep_key == hdf5_k:
            alt = series.get(npz_k)
            if alt and any(v is not None for v in alt):
                return True
    return False


def _missing_active_reward_penalty_keys(
    reward_cfg: dict[str, float | str | None] | None,
    series: dict[str, list],
) -> tuple[list[str], list[str]]:
    """Expected active Episode_Reward keys (non-zero cfg) with no data in *series*."""
    if reward_cfg is None:
        return [], []
    exp_pos, exp_pen = _reward_lists_from_config(reward_cfg)
    miss_p = [k for k in exp_pos if not _series_has_episode_reward_key(series, k)]
    miss_n = [k for k in exp_pen if not _series_has_episode_reward_key(series, k)]
    return miss_p, miss_n


def _episode_reward_y_for_ylim(
    key: str,
    raw: float | None,
    *,
    penalty_negative_display: bool,
) -> float | None:
    """Same y transform as the reward/penalty subplots (for consistent ylim)."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if penalty_negative_display:
        if "death_cost" in key:
            return v
        return -abs(v)
    return v


def _last_episode_reward_y_per_key(
    series: dict[str, list],
    episode_keys: list[str],
    *,
    penalty_negative_display: bool,
) -> list[float]:
    """Last logged numeric value per key (plot coordinates), scanning from the end."""
    out: list[float] = []
    for k in episode_keys:
        if k not in series:
            continue
        vals = series[k]
        for j in range(len(vals) - 1, -1, -1):
            y = _episode_reward_y_for_ylim(k, vals[j], penalty_negative_display=penalty_negative_display)
            if y is not None:
                out.append(y)
                break
    return out


def _episode_component_ylim_from_tail(
    iterations: list[int],
    series: dict[str, list],
    episode_keys: list[str],
    *,
    focus_after_iteration: int,
    late_phase_frac: float = 0.22,
    quantile_low: float = 0.04,
    quantile_high: float = 0.88,
    margin_frac: float = 0.06,
    penalty_negative_display: bool = False,
    include_last_values: bool = True,
) -> tuple[float, float] | None:
    """Y-limits from a late training segment (robust quantiles) plus margin.

    Uses ``iteration >= max(focus_after_iteration, late_window_start)`` where
    ``late_window_start`` begins the last ``late_phase_frac`` of the logged span.
    If ``include_last_values`` is True, ``ymin``/``ymax`` are widened so each
    component's final logged point fits inside the axis (tighter view if False).
    """
    if focus_after_iteration < 0 or not iterations or not episode_keys:
        return None
    first = float(iterations[0])
    last = float(iterations[-1])
    span = max(last - first, 1.0)
    lf = min(max(float(late_phase_frac), 0.05), 0.95)
    late_start = first + (1.0 - lf) * span
    raw_cut = max(float(focus_after_iteration), late_start)
    cut_iter = min(raw_cut, last)

    def _collect(cut: float) -> list[float]:
        out: list[float] = []
        for i, it in enumerate(iterations):
            if float(it) < cut:
                continue
            for k in episode_keys:
                if k not in series:
                    continue
                vals = series[k]
                if i >= len(vals):
                    continue
                y = _episode_reward_y_for_ylim(k, vals[i], penalty_negative_display=penalty_negative_display)
                if y is not None:
                    out.append(y)
        return out

    collected = _collect(cut_iter)
    if not collected and cut_iter > first:
        collected = _collect(max(float(focus_after_iteration), first))
    if not collected:
        return None
    collected.sort()

    def _q(data: list[float], q: float) -> float:
        if len(data) == 1:
            return data[0]
        pos = q * (len(data) - 1)
        j = int(math.floor(pos))
        j2 = int(math.ceil(pos))
        if j == j2:
            return data[j]
        t = pos - j
        return data[j] * (1.0 - t) + data[j2] * t

    q_lo = min(max(quantile_low, 0.0), 1.0)
    q_hi = min(max(quantile_high, 0.0), 1.0)
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    ymin, ymax = _q(collected, q_lo), _q(collected, q_hi)
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    if include_last_values:
        last_ys = _last_episode_reward_y_per_key(
            series, episode_keys, penalty_negative_display=penalty_negative_display
        )
        if last_ys:
            ymin = min(ymin, min(last_ys))
            ymax = max(ymax, max(last_ys))

    if math.isclose(ymin, ymax, rel_tol=0.0, abs_tol=1e-12):
        pad = abs(ymin) * 0.15 if ymin != 0.0 else 0.1
        return ymin - pad, ymax + pad
    span_y = ymax - ymin
    pad = max(span_y * margin_frac, 1e-9)
    return ymin - pad, ymax + pad


def _symmetry_panel_ylim_from_tail(
    iterations: list[int],
    series: dict[str, list],
    sym_positive_keys: list[str],
    sym_penalty_keys: list[str],
    *,
    focus_after_iteration: int,
    late_phase_frac: float = 0.22,
    quantile_low: float = 0.04,
    quantile_high: float = 0.88,
    margin_frac: float = 0.06,
    include_last_values: bool = True,
) -> tuple[float, float] | None:
    """Late-window y-limits for the symmetry panel (positive rewards + penalties as negative)."""
    episode_keys = sym_positive_keys + sym_penalty_keys
    if focus_after_iteration < 0 or not iterations or not episode_keys:
        return None
    first = float(iterations[0])
    last = float(iterations[-1])
    span = max(last - first, 1.0)
    lf = min(max(float(late_phase_frac), 0.05), 0.95)
    late_start = first + (1.0 - lf) * span
    raw_cut = max(float(focus_after_iteration), late_start)
    cut_iter = min(raw_cut, last)

    def _neg_display(k: str) -> bool:
        return k in _SYMMETRY_DEDICATED_PENALTY

    def _collect(cut: float) -> list[float]:
        out: list[float] = []
        for i, it in enumerate(iterations):
            if float(it) < cut:
                continue
            for k in episode_keys:
                if k not in series:
                    continue
                vals = series[k]
                if i >= len(vals):
                    continue
                y = _episode_reward_y_for_ylim(
                    k, vals[i], penalty_negative_display=_neg_display(k)
                )
                if y is not None:
                    out.append(y)
        return out

    collected = _collect(cut_iter)
    if not collected and cut_iter > first:
        collected = _collect(max(float(focus_after_iteration), first))
    if not collected:
        return None
    collected.sort()

    def _q(data: list[float], q: float) -> float:
        if len(data) == 1:
            return data[0]
        pos = q * (len(data) - 1)
        j = int(math.floor(pos))
        j2 = int(math.ceil(pos))
        if j == j2:
            return data[j]
        t = pos - j
        return data[j] * (1.0 - t) + data[j2] * t

    q_lo = min(max(quantile_low, 0.0), 1.0)
    q_hi = min(max(quantile_high, 0.0), 1.0)
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    ymin, ymax = _q(collected, q_lo), _q(collected, q_hi)
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    if include_last_values:
        last_ys: list[float] = []
        for k in episode_keys:
            if k not in series:
                continue
            vals = series[k]
            for j in range(len(vals) - 1, -1, -1):
                y = _episode_reward_y_for_ylim(
                    k, vals[j], penalty_negative_display=_neg_display(k)
                )
                if y is not None:
                    last_ys.append(y)
                    break
        if last_ys:
            ymin = min(ymin, min(last_ys))
            ymax = max(ymax, max(last_ys))

    if math.isclose(ymin, ymax, rel_tol=0.0, abs_tol=1e-12):
        pad = abs(ymin) * 0.15 if ymin != 0.0 else 0.1
        return ymin - pad, ymax + pad
    span_y = ymax - ymin
    pad = max(span_y * margin_frac, 1e-9)
    return ymin - pad, ymax + pad


def generate_visual_analysis(  # noqa: C901
    log_dir: str,
    output_name: str = "training_progress_analysis.png",
    *,
    reward_penalty_ylim_focus_after: int | None = 300,
    reward_penalty_late_phase_frac: float = 0.22,
) -> str | None:
    """Generate training_progress_analysis.png in log_dir. Returns path or None on failure.

    V4 layout: **locomotion-task** positive rewards, **imitation / clip** positive rewards,
    **dynamic balance** rewards, **L/R symmetry** (gait + episode-level) mixed positive/negative,
    **dynamic balance** penalties, then general penalties.
    Late-window y scaling when focus iteration ≥ 0. Main row + sidebar unchanged.
    """
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

    reward_cfg = parse_header_reward_config(path)
    if reward_cfg is None:
        # Old log or no header: show all known reward keys, then add any Episode_Reward/* seen in series
        category_rewards_positive = [ep_key for _, ep_key in REWARD_POSITIVE_WEIGHTS]
        category_rewards_penalties = [ep_key for _, ep_key in REWARD_PENALTY_WEIGHTS] + REWARD_PENALTIES_ALWAYS
    else:
        category_rewards_positive, category_rewards_penalties = _reward_lists_from_config(reward_cfg)
    category_rewards_positive, category_rewards_penalties = merge_episode_reward_keys_from_series(
        series, category_rewards_positive, category_rewards_penalties
    )
    category_rewards_loco, category_rewards_imit = _split_v4_positive_rewards(category_rewards_positive)
    (
        category_rewards_loco,
        category_rewards_penalties,
        dyn_positive_keys,
        dyn_penalty_keys,
    ) = _extract_dynamic_balance_dedicated_keys(category_rewards_loco, category_rewards_penalties)
    (
        category_rewards_loco,
        category_rewards_penalties,
        sym_positive_keys,
        sym_penalty_keys,
    ) = _extract_symmetry_dedicated_keys(category_rewards_loco, category_rewards_penalties)
    dyn_positive_ordered = _order_symmetry_keys(dyn_positive_keys, _DYNAMIC_BALANCE_PANEL_ORDER_POSITIVE)
    dyn_penalty_ordered = _order_symmetry_keys(dyn_penalty_keys, _DYNAMIC_BALANCE_PANEL_ORDER_PENALTY)
    sym_positive_ordered = _order_symmetry_keys(sym_positive_keys, _SYMMETRY_PANEL_ORDER_POSITIVE)
    sym_penalty_ordered = _order_symmetry_keys(sym_penalty_keys, _SYMMETRY_PANEL_ORDER_PENALTY)

    fig = plt.figure(figsize=(18, 28))
    gs = GridSpec(
        8,
        3,
        figure=fig,
        # Dedicated spacer row kept only between locomotion and imitation.
        height_ratios=[0.85, 1.0, 0.24, 1.0, 0.92, 0.95, 0.88, 1.05],
        width_ratios=[1.2, 1.2, 0.65],
        hspace=0.32,
        wspace=0.3,
    )
    fig.suptitle(
        "G1 Locomotion V4 — training progress (locomotion, imitation, dynamic balance, symmetry)",
        fontsize=14,
        fontweight="bold",
    )

    _plot_style = {"markersize": 3.0, "linewidth": 1.5, "markeredgewidth": 0.8}
    # Unique (color, linestyle) per line: 24 colors × 6 styles = 144 combinations
    _cmap_colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    _line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    _markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]
    _palette = []
    for i in range(144):
        _palette.append((_cmap_colors[i % len(_cmap_colors)], _line_styles[i % len(_line_styles)]))

    def _series_style(idx: int, *, emphasize: bool = False) -> dict:
        c, ls = _palette[idx % len(_palette)]
        style = dict(**_plot_style)
        style["color"] = c
        style["linestyle"] = ls
        style["marker"] = _markers[idx % len(_markers)]
        style["markevery"] = max(1, len(iterations) // 18)
        style["markerfacecolor"] = "white"
        if emphasize:
            style["linewidth"] = 2.2
            style["markersize"] = 4.6
        return style

    def _ax_style(ax):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)

    ax_main = fig.add_subplot(gs[0, 0])
    _main_idx = 0
    for k in CATEGORY_MAIN_LEFT:
        if k in series and series[k]:
            ax_main.plot(
                iterations,
                series[k],
                label=k + _format_last_value_suffix(series[k]),
                **_series_style(_main_idx),
            )
            _main_idx += 1
    ax_main.set_ylabel("Mean reward", fontsize=9, color=_palette[0][0] if _main_idx else "C0")
    ax_main.tick_params(axis="y", labelcolor=_palette[0][0] if _main_idx else "C0")
    _ax_style(ax_main)
    ax_main.set_xlabel("")
    ax_main2 = ax_main.twinx()
    has_right = False
    if "Iteration time" in series and series["Iteration time"]:
        ax_main2.plot(
            iterations,
            series["Iteration time"],
            label="iter time (s)" + _format_last_value_suffix(series["Iteration time"]),
            **_series_style(_main_idx),
        )
        _main_idx += 1
        has_right = True
    if "Computation" in series and series["Computation"]:
        comp = [x * 1e-5 for x in series["Computation"]]
        ax_main2.plot(
            iterations,
            comp,
            label="throughput (×10⁵ steps/s)" + _format_last_value_suffix(comp),
            **_series_style(_main_idx),
        )
        has_right = True
    if has_right:
        ax_main2.set_ylabel("iter time (s) / throughput", fontsize=8, color="gray")
        ax_main2.tick_params(axis="y", labelcolor="gray")
        ax_main2.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 1.20),
            fontsize=7,
            framealpha=0.92,
            edgecolor="gray",
        )
        ax_main.set_title("Main metrics (reward + iter time / throughput)", fontsize=10, fontweight="bold")
    else:
        ax_main.set_title("Main metrics (reward only)", fontsize=10, fontweight="bold")
    ax_main.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.04),
        fontsize=8,
        framealpha=0.92,
        edgecolor="gray",
    )

    ax_loss = fig.add_subplot(gs[0, 1])
    for idx, k in enumerate(CATEGORY_LOSSES):
        if k in series and series[k]:
            ax_loss.plot(
                iterations,
                series[k],
                label=k.replace("Mean ", "") + _format_last_value_suffix(series[k]),
                **_series_style(idx),
            )
    ax_loss.set_title("Losses", fontsize=10, fontweight="bold")
    ax_loss.legend(loc="upper right", fontsize=8, framealpha=0.92, edgecolor="gray")
    _ax_style(ax_loss)
    ax_loss.set_xlabel("")

    def _plot_rewards_positive_subset(ax, key_list: list[str], *, palette_start: int, with_legend: bool) -> None:
        if not key_list:
            ax.text(
                0.5,
                0.5,
                "No components in this panel",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            _ax_style(ax)
            ax.set_xlabel("")
            ax.set_ylabel("Reward", fontsize=9)
            return
        for j, k in enumerate(key_list):
            if k in series and series[k]:
                tail = k.replace("Episode_Reward/", "")
                label = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
                emphasize = label in IMITATION_PLOT_EMPHASIS_LABELS or label in DYNAMIC_BALANCE_PLOT_EMPHASIS_LABELS
                ax.plot(
                    iterations,
                    series[k],
                    label=label + _format_last_value_suffix(series[k]),
                    **_series_style(palette_start + j, emphasize=emphasize),
                )
        if with_legend:
            ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.92, edgecolor="gray")
        _ax_style(ax)
        ax.set_ylabel("Reward", fontsize=9)
        ax.set_xlabel("")

    _fp_after = reward_penalty_ylim_focus_after

    # Locomotion-task rewards (positive) — row 1
    ax_loco = fig.add_subplot(gs[1, 0:2])
    _plot_rewards_positive_subset(ax_loco, category_rewards_loco, palette_start=0, with_legend=False)
    _fp_note_l = ""
    if _fp_after is not None and _fp_after >= 0 and category_rewards_loco:
        _lim_l = _episode_component_ylim_from_tail(
            iterations,
            series,
            category_rewards_loco,
            focus_after_iteration=_fp_after,
            late_phase_frac=reward_penalty_late_phase_frac,
            include_last_values=True,
        )
        if _lim_l is not None:
            ax_loco.set_ylim(_lim_l)
            _fp_note_l = " — y: late window + quantiles + last points"
    ax_loco.set_title(
        "Locomotion / task — rewards (positive)" + _fp_note_l,
        fontsize=10,
        fontweight="bold",
    )
    if category_rewards_loco:
        ax_loco.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fontsize=7,
            ncol=5,
            framealpha=0.92,
            edgecolor="gray",
        )

    # Imitation / motion-clip rewards (positive) — row 2
    ax_imit = fig.add_subplot(gs[3, 0:2])
    _plot_rewards_positive_subset(ax_imit, category_rewards_imit, palette_start=24, with_legend=True)
    _fp_note_i = ""
    if _fp_after is not None and _fp_after >= 0 and category_rewards_imit:
        _lim_i = _episode_component_ylim_from_tail(
            iterations,
            series,
            category_rewards_imit,
            focus_after_iteration=_fp_after,
            late_phase_frac=reward_penalty_late_phase_frac,
            include_last_values=True,
        )
        if _lim_i is not None:
            ax_imit.set_ylim(_lim_i)
            _fp_note_i = " — y: late window + quantiles + last points"
    ax_imit.set_title(
        "Imitation / motion clip — rewards (positive)" + _fp_note_i,
        fontsize=10,
        fontweight="bold",
    )

    # Dynamic balance rewards — row 3
    ax_dyn = fig.add_subplot(gs[4, 0:2])
    _plot_rewards_positive_subset(ax_dyn, dyn_positive_ordered, palette_start=40, with_legend=True)
    _fp_note_d = ""
    if _fp_after is not None and _fp_after >= 0 and dyn_positive_ordered:
        _lim_d = _episode_component_ylim_from_tail(
            iterations,
            series,
            dyn_positive_ordered,
            focus_after_iteration=_fp_after,
            late_phase_frac=reward_penalty_late_phase_frac,
            include_last_values=True,
        )
        if _lim_d is not None:
            ax_dyn.set_ylim(_lim_d)
            _fp_note_d = " — y: late window + quantiles + last points"
    ax_dyn.set_title(
        "Dynamic balance — rewards (positive)" + _fp_note_d,
        fontsize=10,
        fontweight="bold",
    )

    # Symmetry (gait + episode L/R) — row 4
    ax_sym = fig.add_subplot(gs[5, 0:2])
    _sym_plot_idx = 0
    _sym_palette_base = 48
    if not sym_positive_ordered and not sym_penalty_ordered:
        ax_sym.text(
            0.5,
            0.5,
            "No symmetry Episode_Reward/* keys in this run",
            ha="center",
            va="center",
            transform=ax_sym.transAxes,
            fontsize=10,
            color="gray",
        )
        _ax_style(ax_sym)
        ax_sym.set_xlabel("")
        ax_sym.set_ylabel("Reward / penalty", fontsize=9)
    else:
        for k in sym_positive_ordered:
            if k in series and series[k]:
                tail = k.replace("Episode_Reward/", "")
                label = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
                ax_sym.plot(
                    iterations,
                    series[k],
                    label=label + _format_last_value_suffix(series[k]),
                    **_series_style(
                        _sym_palette_base + _sym_plot_idx,
                        emphasize=label in SYMMETRY_PLOT_EMPHASIS_LABELS,
                    ),
                )
                _sym_plot_idx += 1
        for k in sym_penalty_ordered:
            if k in series and series[k]:
                vals = series[k]
                plot_vals = [-abs(v) for v in vals] if "death_cost" not in k else vals
                tail = k.replace("Episode_Reward/", "")
                label = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
                ax_sym.plot(
                    iterations,
                    plot_vals,
                    label=label + _format_last_value_suffix(plot_vals, penalty_negative_display=False),
                    **_series_style(
                        _sym_palette_base + _sym_plot_idx,
                        emphasize=label in SYMMETRY_PLOT_EMPHASIS_LABELS,
                    ),
                )
                _sym_plot_idx += 1
        ax_sym.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        _fp_note_s = ""
        if _fp_after is not None and _fp_after >= 0 and (sym_positive_ordered or sym_penalty_ordered):
            _lim_s = _symmetry_panel_ylim_from_tail(
                iterations,
                series,
                sym_positive_ordered,
                sym_penalty_ordered,
                focus_after_iteration=_fp_after,
                late_phase_frac=reward_penalty_late_phase_frac,
                include_last_values=True,
            )
            if _lim_s is not None:
                _s_lo, _s_hi = _lim_s
                _s_lo = min(_s_lo, 0.0)
                _s_hi = max(_s_hi, 0.0)
                ax_sym.set_ylim(_s_lo, _s_hi)
                _fp_note_s = " — y: late window + quantiles, incl. last points"
        ax_sym.set_title("Symmetry — rewards & penalties (penalties shown negative)" + _fp_note_s, fontsize=10, fontweight="bold")
        ax_sym.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.92, edgecolor="gray")
        _ax_style(ax_sym)
        ax_sym.set_ylabel("Value", fontsize=9)
        ax_sym.set_xlabel("")

    # Dynamic balance penalties — row 5
    ax_dyn_pen = fig.add_subplot(gs[6, 0:2])
    for idx, k in enumerate(dyn_penalty_ordered):
        if k in series and series[k]:
            vals = series[k]
            plot_vals = [-abs(v) for v in vals]
            tail = k.replace("Episode_Reward/", "")
            label = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
            ax_dyn_pen.plot(
                iterations,
                plot_vals,
                label=label + _format_last_value_suffix(plot_vals),
                **_series_style(64 + idx, emphasize=label in DYNAMIC_BALANCE_PLOT_EMPHASIS_LABELS),
            )
    ax_dyn_pen.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    _fp_note_dp = ""
    if _fp_after is not None and _fp_after >= 0 and dyn_penalty_ordered:
        _lim_dp = _episode_component_ylim_from_tail(
            iterations,
            series,
            dyn_penalty_ordered,
            focus_after_iteration=_fp_after,
            late_phase_frac=reward_penalty_late_phase_frac,
            penalty_negative_display=True,
            include_last_values=True,
        )
        if _lim_dp is not None:
            _dp_lo, _dp_hi = _lim_dp
            _dp_lo = min(_dp_lo, 0.0)
            _dp_hi = max(_dp_hi, 0.0)
            ax_dyn_pen.set_ylim(_dp_lo, _dp_hi)
            _fp_note_dp = " — y: late window + quantiles, incl. each series' last point"
    ax_dyn_pen.set_title(
        "Dynamic balance — penalties (shown negative)" + _fp_note_dp,
        fontsize=10,
        fontweight="bold",
    )
    if dyn_penalty_ordered:
        ax_dyn_pen.legend(loc="lower left", fontsize=7, ncol=3, framealpha=0.92, edgecolor="gray")
    else:
        ax_dyn_pen.text(
            0.5,
            0.5,
            "No dynamic-balance penalty keys in this run",
            ha="center",
            va="center",
            transform=ax_dyn_pen.transAxes,
            fontsize=10,
            color="gray",
        )
    _ax_style(ax_dyn_pen)
    ax_dyn_pen.set_ylabel("Penalty", fontsize=9)
    ax_dyn_pen.set_xlabel("")

    # Penalties — row 6
    ax_pen = fig.add_subplot(gs[7, 0:2])
    for idx, k in enumerate(category_rewards_penalties):
        if k in series and series[k]:
            vals = series[k]
            if "death_cost" in k:
                plot_vals = vals
            else:
                plot_vals = [-abs(v) for v in vals]
            tail = k.replace("Episode_Reward/", "")
            label = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
            ax_pen.plot(
                iterations,
                plot_vals,
                label=label + _format_last_value_suffix(plot_vals),
                **_series_style(idx, emphasize=label in PENALTY_PLOT_EMPHASIS_LABELS),
            )
    ax_pen.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    _fp_note_p = ""
    if _fp_after is not None and _fp_after >= 0:
        _lim_p = _episode_component_ylim_from_tail(
            iterations,
            series,
            category_rewards_penalties,
            focus_after_iteration=_fp_after,
            late_phase_frac=reward_penalty_late_phase_frac,
            penalty_negative_display=True,
            include_last_values=True,
        )
        if _lim_p is not None:
            _p_lo, _p_hi = _lim_p
            _p_lo = min(_p_lo, 0.0)
            _p_hi = max(_p_hi, 0.0)
            ax_pen.set_ylim(_p_lo, _p_hi)
            _fp_note_p = " — y: late window + quantiles, incl. each series' last point"
    ax_pen.set_title("Penalties / costs (shown negative)" + _fp_note_p, fontsize=10, fontweight="bold")
    ax_pen.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=7,
        ncol=5,
        framealpha=0.92,
        edgecolor="gray",
    )
    _ax_style(ax_pen)
    ax_pen.set_ylabel("Penalty", fontsize=9)
    ax_pen.set_xlabel("")

    ax_cfg = fig.add_subplot(gs[0:8, 2])
    smooth_jerk = parse_header_smooth_jerk(path)
    reward_cfg = parse_header_reward_config(path)
    lines = ["Config / Run Args (log header)", ""]
    if smooth_jerk:
        lines.extend([
            "Smooth / rough (G1 locomotion; run args / env header)",
            f"  action_rate_penalty_scale (L2 Δa): {smooth_jerk.get('action_rate_penalty_scale', '—')}",
            f"  policy_action_roughness_penalty_scale: {smooth_jerk.get('policy_action_roughness_penalty_scale', '—')}",
            f"  policy_action_roughness_huber_delta: {smooth_jerk.get('policy_action_roughness_huber_delta', '—')}",
            f"  smooth_motion_reward_weight / sigma: {smooth_jerk.get('smooth_motion_reward_weight', '—')} / {smooth_jerk.get('smooth_motion_sigma', '—')}",
            f"  joint_velocity_roughness_penalty_scale: {smooth_jerk.get('joint_velocity_roughness_penalty_scale', '—')}",
            f"  joint_velocity_roughness_huber_delta_rad_s: {smooth_jerk.get('joint_velocity_roughness_huber_delta_rad_s', '—')}",
            f"  joint_velocity_smoothness_reward_weight / sigma: {smooth_jerk.get('joint_velocity_smoothness_reward_weight', '—')} / {smooth_jerk.get('joint_velocity_smoothness_sigma', '—')}",
            f"  joint_acceleration_penalty_scale: {smooth_jerk.get('joint_acceleration_penalty_scale', '—')}",
            f"  smooth_motion_joint_accel_reward_weight / sigma: {smooth_jerk.get('smooth_motion_joint_accel_reward_weight', '—')} / {smooth_jerk.get('smooth_motion_joint_accel_sigma', '—')}",
            f"  root_lin_vel_jerk_penalty_weight: {smooth_jerk.get('root_lin_vel_jerk_penalty_weight', '—')}",
            "",
        ])
    if reward_cfg:
        _hdr_ijr = _joint_pos_imitation_header_weight(reward_cfg, reward=True)
        _hdr_ijp = _joint_pos_imitation_header_weight(reward_cfg, reward=False)
        lines.extend([
            "Run args",
            f"  experiment_name: {reward_cfg.get('experiment_name', '—')}",
            f"  num_envs: {reward_cfg.get('num_envs', '—')}",
            f"  max_iterations: {reward_cfg.get('max_iterations', '—')}",
            f"  seed: {reward_cfg.get('seed', '—')}",
            f"  device: {reward_cfg.get('device', '—')}",
            f"  frames_per_second: {reward_cfg.get('frames_per_second', '—')}",
            f"  gravity_scale: {reward_cfg.get('gravity_scale', '—')}",
            f"  starting_policy_path: {reward_cfg.get('starting_policy_path', '—')}",
            f"  divergence_mean_reward_min: {reward_cfg.get('divergence_mean_reward_min', '—')}",
            "",
            "PPO",
            f"  ppo_learning_rate: {reward_cfg.get('ppo_learning_rate', '—')}",
            f"  ppo_max_grad_norm: {reward_cfg.get('ppo_max_grad_norm', '—')}",
            f"  ppo_value_loss_coef: {reward_cfg.get('ppo_value_loss_coef', '—')}",
            f"  ppo_entropy_coef: {reward_cfg.get('ppo_entropy_coef', '—')}",
            f"  reward_clip_min / max: {reward_cfg.get('reward_clip_min', '—')} / {reward_cfg.get('reward_clip_max', '—')}",
            "",
            "Velocity",
            f"  target_velocity: {reward_cfg.get('target_velocity', '—')}",
            f"  velocity_reward_weight: {reward_cfg.get('velocity_reward_weight', '—')}",
            f"  velocity_reward_sharpness: {reward_cfg.get('velocity_reward_sharpness', '—')}",
            f"  target_x_velocity_penalty_weight: {reward_cfg.get('target_x_velocity_penalty_weight', '—')}",
            f"  velocity_command_range: {reward_cfg.get('velocity_command_range', '—')}",
            "",
            "Forward / progress (world +x → Episode_Reward/progress_reward)",
            f"  progress_reward_multiplier: {reward_cfg.get('progress_reward_multiplier', '—')}",
            f"  progress_reward_world_vx_blend: {reward_cfg.get('progress_reward_world_vx_blend', '—')}",
            f"  progress_reward_forward_displacement_weight: {reward_cfg.get('progress_reward_forward_displacement_weight', '—')}",
            f"  progress_reward_forward_displacement_threshold_m: {reward_cfg.get('progress_reward_forward_displacement_threshold_m', '—')}",
            f"  progress_goal_world_offset_xy_m: {reward_cfg.get('progress_goal_world_offset_xy_m', '—')}  # V1-style goal from origin (None = V1 default in env)",
            f"  agg: {_format_progress_forward_signal_summary(reward_cfg)}",
            "",
            "V4 — reward_profile & policy command (Isaac-G1-Locomotion-V4-Direct-v0)",
            f"  reward_profile: {reward_cfg.get('reward_profile', '—')}",
            f"  staged_imitation_start_optimizer_steps: {reward_cfg.get('staged_imitation_start_optimizer_steps', '—')}",
            f"  staged_imitation_ramp_optimizer_steps: {reward_cfg.get('staged_imitation_ramp_optimizer_steps', '—')}",
            f"  staged_ramp_start_scale: {reward_cfg.get('staged_ramp_start_scale', '—')}",
            f"  staged_dybl_start_optimizer_steps: {reward_cfg.get('staged_dybl_start_optimizer_steps', '—')}",
            f"  staged_dybl_ramp_optimizer_steps: {reward_cfg.get('staged_dybl_ramp_optimizer_steps', '—')}",
            f"  staged_symmetry_start_optimizer_steps: {reward_cfg.get('staged_symmetry_start_optimizer_steps', '—')}",
            f"  staged_symmetry_ramp_optimizer_steps: {reward_cfg.get('staged_symmetry_ramp_optimizer_steps', '—')}",
            f"  staged_symmetry_ramp_start_scale: {reward_cfg.get('staged_symmetry_ramp_start_scale', '—')}",
            f"  velocity_command_obs_enabled: {reward_cfg.get('velocity_command_obs_enabled', '—')}",
            f"  velocity_command_obs_scale: {reward_cfg.get('velocity_command_obs_scale', '—')}",
            f"  include_root_xy_error_obs: {reward_cfg.get('include_root_xy_error_obs', '—')}",
            f"  root_xy_error_obs_scale: {reward_cfg.get('root_xy_error_obs_scale', '—')}",
            f"  include_reward_profile_phase_obs: {reward_cfg.get('include_reward_profile_phase_obs', '—')}",
            f"  target_yaw_command_obs_enabled: {reward_cfg.get('target_yaw_command_obs_enabled', '—')}",
            f"  append_teleop_hand_target_obs: {reward_cfg.get('append_teleop_hand_target_obs', '—')}",
            f"  teleop_target_obs_scale: {reward_cfg.get('teleop_target_obs_scale', '—')}",
            "",
            "Decimation",
            f"  decimation: {reward_cfg.get('decimation', '—')}",
            "",
            "HDF5 joint ref scale (dof_pos vs clip)",
            f"  hdf5_joint_position_data_scale: {reward_cfg.get('hdf5_joint_position_data_scale', '—')}",
            f"  hdf5_joint_position_data_scale_ramp_enabled: {reward_cfg.get('hdf5_joint_position_data_scale_ramp_enabled', '—')}",
            f"  ramp_start → ramp_end: {reward_cfg.get('hdf5_joint_position_data_scale_ramp_start', '—')} → {reward_cfg.get('hdf5_joint_position_data_scale_ramp_end', '—')}",
            f"  ramp_steps / ramp_delay_steps: {reward_cfg.get('hdf5_joint_position_data_scale_ramp_steps', '—')} / {reward_cfg.get('hdf5_joint_position_data_scale_ramp_delay_steps', '—')}",
            f"  ramp_in_ppo_iterations: {reward_cfg.get('hdf5_joint_position_data_scale_ramp_in_ppo_iterations', '—')}",
            f"  exclude_ankles: {reward_cfg.get('hdf5_joint_position_data_scale_exclude_ankles', '—')}",
            f"  hdf5_imitation_max_frames: {reward_cfg.get('hdf5_imitation_max_frames', '—')}",
            f"  hdf5_imitation_reward_episode_time_scale_enabled: {reward_cfg.get('hdf5_imitation_reward_episode_time_scale_enabled', '—')}",
            "",
            "Drift / yaw",
            f"  x_drift_tol: {reward_cfg.get('x_drift_tolerance', '—')}",
            f"  x_drift_weight: {reward_cfg.get('x_drift_penalty_weight', '—')}",
            f"  x_drift_reward_weight: {reward_cfg.get('x_drift_reward_weight', '—')}",
            f"  x_drift_reward_sigma: {reward_cfg.get('x_drift_reward_sigma', '—')}",
            f"  y_drift_tol: {reward_cfg.get('y_drift_tolerance', '—')}",
            f"  y_drift_weight: {reward_cfg.get('y_drift_penalty_weight', '—')}",
            f"  y_drift_reward_weight: {reward_cfg.get('y_drift_reward_weight', '—')}",
            f"  y_drift_reward_sigma: {reward_cfg.get('y_drift_reward_sigma', '—')}",
            f"  z_orient_tol: {reward_cfg.get('z_orientation_tolerance', '—')}",
            f"  z_orient_pen_weight: {reward_cfg.get('z_orientation_penalty_weight', '—')}",
            f"  z_orient_reward_weight: {reward_cfg.get('z_orientation_reward_weight', '—')}",
            f"  z_orient_reward_sharpness: {reward_cfg.get('z_orientation_reward_sharpness', '—')}",
            "",
            "Torso +Z vs world +Z (V4 trunk upright)",
            f"  link_body_name: {reward_cfg.get('torso_z_align_link_body_name', '—')}",
            f"  align_reward_w: {reward_cfg.get('torso_z_axis_world_align_reward_weight', '—')}",
            f"  align_reward_sigma: {reward_cfg.get('torso_z_axis_world_align_reward_sigma', '—')}",
            f"  align_penalty_w: {reward_cfg.get('torso_z_axis_world_align_penalty_weight', '—')}",
            f"  align_penalty_tol (1−dot): {reward_cfg.get('torso_z_axis_world_align_penalty_tolerance', '—')}",
            f"  palm_below_z_max (pelvis m): {reward_cfg.get('palm_below_pelvis_z_max_pelvis_frame_m', '—')}",
            f"  palm_below_z_reward_w / σ: {reward_cfg.get('palm_below_pelvis_z_reward_weight', '—')} / {reward_cfg.get('palm_below_pelvis_z_reward_sigma_m', '—')}",
            f"  palm_below_z_pen_w / slack: {reward_cfg.get('palm_below_pelvis_z_penalty_weight', '—')} / {reward_cfg.get('palm_below_pelvis_z_penalty_tolerance_m', '—')}",
            f"  palm bodies R/L: {reward_cfg.get('palm_below_pelvis_right_body_name', '—')} / {reward_cfg.get('palm_below_pelvis_left_body_name', '—')}",
            "",
            "Early fall vs horizon (V4)",
            f"  early_termination_time_penalty_weight: {reward_cfg.get('early_termination_time_penalty_weight', '—')}  # fall only: w×(1−len/max_len)",
            "",
            "Dynamic balance (V4 controlled fall)",
            f"  com_velocity_tracking_reward_weight: {reward_cfg.get('com_velocity_tracking_reward_weight', '—')}",
            f"  com_velocity_tracking_sigma_mps: {reward_cfg.get('com_velocity_tracking_sigma_mps', '—')}",
            f"  linear_momentum_projection_reward_weight: {reward_cfg.get('linear_momentum_projection_reward_weight', '—')}",
            f"  linear_momentum_projection_scale: {reward_cfg.get('linear_momentum_projection_scale', '—')}",
            f"  base_of_support_center_penalty_weight: {reward_cfg.get('base_of_support_center_penalty_weight', '—')}",
            f"  base_of_support_forward_lead_min_m: {reward_cfg.get('base_of_support_forward_lead_min_m', '—')}",
            f"  dcm_capture_point_reward_weight: {reward_cfg.get('dcm_capture_point_reward_weight', '—')}",
            f"  dcm_capture_point_target_ahead_m: {reward_cfg.get('dcm_capture_point_target_ahead_m', '—')}",
            f"  dcm_capture_point_sigma_m: {reward_cfg.get('dcm_capture_point_sigma_m', '—')}",
            f"  dcm_capture_point_lateral_penalty_weight: {reward_cfg.get('dcm_capture_point_lateral_penalty_weight', '—')}",
            f"  dcm_capture_point_lateral_tolerance_m: {reward_cfg.get('dcm_capture_point_lateral_tolerance_m', '—')}",
            f"  dcm_com_height_min_m: {reward_cfg.get('dcm_com_height_min_m', '—')}",
            "",
            "Alternating foot / stance",
            f"  alt_reward_weight: {reward_cfg.get('alternating_foot_reward_weight', '—')}",
            f"  same_foot_tap_pen_weight: {reward_cfg.get('same_foot_tap_penalty_weight', '—')}",
            f"  same_foot_tap_min_air_steps: {reward_cfg.get('same_foot_tap_min_air_steps', '—')}",
            f"  gait_single_stance_reward_weight: {reward_cfg.get('gait_single_stance_reward_weight', '—')}",
            f"  hip_weight_shift_reward_weight: {reward_cfg.get('hip_weight_shift_reward_weight', '—')}",
            f"  hip_weight_shift_torque_scale: {reward_cfg.get('hip_weight_shift_torque_scale', '—')}",
            f"  alternating_foot_scale_with_forward_vel: {reward_cfg.get('alternating_foot_scale_with_forward_vel', '—')}",
            f"  alternating_foot_vel_scale_ref: {reward_cfg.get('alternating_foot_vel_scale_ref', '—')}",
            f"  jump_pen_weight: {reward_cfg.get('jump_penalty_weight', '—')}",
            f"  foot_contact_rew: {reward_cfg.get('at_least_one_foot_contact_reward_weight', '—')}",
            f"  ankle_roll_feet_target_separation_m: {reward_cfg.get('ankle_roll_feet_target_separation_m', '—')}",
            f"  ankle_roll_feet_separation_tolerance_m: {reward_cfg.get('ankle_roll_feet_separation_tolerance_m', '—')}",
            f"  ankle_roll_feet_separation_reward_weight: {reward_cfg.get('ankle_roll_feet_separation_reward_weight', '—')}",
            f"  ankle_roll_feet_separation_penalty_weight: {reward_cfg.get('ankle_roll_feet_separation_penalty_weight', '—')}",
            f"  ankle_roll_foot_height_reward_target_m: {reward_cfg.get('ankle_roll_foot_height_reward_target_m', '—')}  # minimal h ~0.105",
            f"  ankle_roll_foot_height_reward_weight: {reward_cfg.get('ankle_roll_foot_height_reward_weight', '—')}",
            "",
            "Stability (V3)",
            f"  lin_vel_z_penalty: {reward_cfg.get('lin_vel_z_penalty_weight', '—')}",
            f"  lin_vel_z_tolerance: {reward_cfg.get('lin_vel_z_tolerance', '—')}",
            f"  ang_vel_xy_penalty: {reward_cfg.get('ang_vel_xy_penalty_weight', '—')}",
            f"  flat_orientation_penalty: {reward_cfg.get('flat_orientation_penalty_weight', '—')}",
            f"  flat_orientation_tolerance: {reward_cfg.get('flat_orientation_tolerance', '—')}",
            "",
            "V3 optimizations",
            f"  base_height_reference: {reward_cfg.get('base_height_reference', '—')}",
            f"  base_height_reward_weight: {reward_cfg.get('base_height_reward_weight', '—')}",
            f"  base_height_sigma: {reward_cfg.get('base_height_sigma', '—')}",
            f"  base_height_penalty_weight: {reward_cfg.get('base_height_penalty_weight', '—')}",
            f"  base_height_penalty_tolerance_m: {reward_cfg.get('base_height_penalty_tolerance_m', '—')}",
            f"  enable_curriculum: {reward_cfg.get('enable_curriculum', '—')}",
            f"  curriculum_stage1_steps: {reward_cfg.get('curriculum_stage1_steps', '—')}",
            f"  curriculum_stage2_steps: {reward_cfg.get('curriculum_stage2_steps', '—')}",
            f"  num_steps_per_env: {reward_cfg.get('num_steps_per_env', '—')}",
            f"  enable_push_perturbation: {reward_cfg.get('enable_push_perturbation', '—')}",
            f"  max_push_vel_xy: {reward_cfg.get('max_push_vel_xy', '—')}",
            f"  motion_cycle_episode_margin_s: {reward_cfg.get('motion_cycle_episode_margin_s', '—')}",
            "",
            "Foot / root tracking (V3)",
            f"  imitation_root_xy_reward_weight: {reward_cfg.get('imitation_root_xy_reward_weight', '—')}",
            f"  imitation_root_xy_penalty_weight: {reward_cfg.get('imitation_root_xy_penalty_weight', '—')}",
            f"  imitation_root_xy_sigma_m: {reward_cfg.get('imitation_root_xy_sigma_m', '—')}",
            f"  swing_foot_clearance_reward_weight: {reward_cfg.get('swing_foot_clearance_reward_weight', '—')}",
            f"  swing_foot_clearance_penalty_weight: {reward_cfg.get('swing_foot_clearance_penalty_weight', '—')}",
            f"  swing_foot_clearance_sigma_m: {reward_cfg.get('swing_foot_clearance_sigma_m', '—')}",
            f"  swing_foot_clearance_margin_m: {reward_cfg.get('swing_foot_clearance_margin_m', '—')}",
            f"  stance_foot_slip_penalty_weight: {reward_cfg.get('stance_foot_slip_penalty_weight', '—')}",
            f"  hdf5_derive_foot_contact_from_link_z: {reward_cfg.get('hdf5_derive_foot_contact_from_link_z', '—')}",
            f"  hdf5_derived_foot_contact_beta: {reward_cfg.get('hdf5_derived_foot_contact_beta', '—')}",
            f"  hdf5_link_local_stance_tolerance_scale: {reward_cfg.get('hdf5_link_local_stance_tolerance_scale', '—')}",
            f"  hdf5_link_local_stance_ref_contact_threshold: {reward_cfg.get('hdf5_link_local_stance_ref_contact_threshold', '—')}",
            f"  hdf5_link_local_imitation_swing_penalty_mask: {reward_cfg.get('hdf5_link_local_imitation_swing_penalty_mask', '—')}",
            f"  hdf5_link_local_imitation_swing_ref_contact_threshold: {reward_cfg.get('hdf5_link_local_imitation_swing_ref_contact_threshold', '—')}",
            f"  imitation_joint_ref_contact_leg_scale: {reward_cfg.get('imitation_joint_ref_contact_leg_scale', '—')}",
            f"  hdf5_link_use_joint_motion_clock_when_aligned: {reward_cfg.get('hdf5_link_use_joint_motion_clock_when_aligned', '—')}",
            f"  hdf5_link_local_imitation_use_body_lin_vel: {reward_cfg.get('hdf5_link_local_imitation_use_body_lin_vel', '—')}",
            "",
            "Env",
            f"  termination_height: {reward_cfg.get('termination_height', '—')}",
            "",
            "Motion clip (joints_data)",
            f"  path: {reward_cfg.get('joints_data', '—')}",
            f"  joint_limit_penalty_weight: {reward_cfg.get('joint_limit_penalty_weight', '—')}",
            f"  joint_limit_reward_weight: {reward_cfg.get('joint_limit_reward_weight', '—')}",
            f"  joint_limit_reward_sigma: {reward_cfg.get('joint_limit_reward_sigma', '—')}",
            "",
            "Symmetry (L-R)",
            f"  sym_reward_weight: {reward_cfg.get('symmetry_reward_weight', '—')}",
            f"  sym_penalty_weight: {reward_cfg.get('symmetry_penalty_weight', '—')}",
            f"  symmetry_include_hip_torque: {reward_cfg.get('symmetry_include_hip_torque', '—')}",
            f"  symmetry_torque_terms_double_support_only: {reward_cfg.get('symmetry_torque_terms_double_support_only', '—')}",
            f"  symmetry_use_geometric_mean: {reward_cfg.get('symmetry_use_geometric_mean', '—')}",
            f"  symmetry_contact_sharpness: {reward_cfg.get('symmetry_contact_sharpness', '—')}",
            f"  symmetry_touchdown_sharpness: {reward_cfg.get('symmetry_touchdown_sharpness', '—')}",
            f"  symmetry_torque_sharpness: {reward_cfg.get('symmetry_torque_sharpness', '—')}",
            "",
            "Episode symmetry (L–R, end of episode)",
            f"  min_episode_steps: {reward_cfg.get('episode_symmetry_min_episode_steps', '—')}",
            f"  legacy ep_sym_reward_w: {reward_cfg.get('episode_symmetry_reward_weight', '—')}",
            f"  step blend / joint blend / link blend: {reward_cfg.get('episode_symmetry_step_length_blend', '—')} / {reward_cfg.get('episode_symmetry_joint_rom_blend', '—')} / {reward_cfg.get('episode_symmetry_link_rom_blend', '—')}",
            f"  step r/p w: {reward_cfg.get('episode_symmetry_step_length_reward_weight', '—')} / {reward_cfg.get('episode_symmetry_step_length_penalty_weight', '—')}",
            f"  joint ROM r/p w: {reward_cfg.get('episode_symmetry_joint_rom_reward_weight', '—')} / {reward_cfg.get('episode_symmetry_joint_rom_penalty_weight', '—')}",
            f"  link ROM r/p w: {reward_cfg.get('episode_symmetry_link_rom_reward_weight', '—')} / {reward_cfg.get('episode_symmetry_link_rom_penalty_weight', '—')}",
            f"  σ step_m / joint_rad / link_m: {reward_cfg.get('episode_symmetry_step_length_sigma_m', '—')} / {reward_cfg.get('episode_symmetry_joint_rom_sigma_rad', '—')} / {reward_cfg.get('episode_symmetry_link_rom_sigma_m', '—')}",
            f"  single-stance XOR balance r/p w: {reward_cfg.get('episode_symmetry_single_stance_balance_reward_weight', '—')} / {reward_cfg.get('episode_symmetry_single_stance_balance_penalty_weight', '—')}  σ_gss={reward_cfg.get('episode_symmetry_single_stance_balance_sigma', '—')}",
            f"  HDF5 ankle y-rot track L/R r/p w: {reward_cfg.get('episode_symmetry_hdf5_ankle_roll_y_track_reward_weight', '—')} / {reward_cfg.get('episode_symmetry_hdf5_ankle_roll_y_track_penalty_weight', '—')}  σ_ar={reward_cfg.get('episode_symmetry_hdf5_ankle_roll_y_track_sigma_rad', '—')}",
            "  # single-stance ep term: same L-only/R-only contact XOR as gait_single_stance_reward_weight (per-step); this block accumulates episode balance",
            "",
            "Step length",
            f"  step_length_min_m: {reward_cfg.get('step_length_min_m', '—')}",
            f"  step_length_reward: {reward_cfg.get('step_length_reward_weight', '—')}",
            f"  step_length_penalty: {reward_cfg.get('step_length_penalty_weight', '—')}",
            f"  step_length_use_smooth_reward: {reward_cfg.get('step_length_use_smooth_reward', '—')}",
            f"  step_length_smooth_sigma_m: {reward_cfg.get('step_length_smooth_sigma_m', '—')}",
            f"  step_length_penalty_squared: {reward_cfg.get('step_length_penalty_squared', '—')}",
            "",
            "Foot crossing",
            f"  foot_crossing_reward: {reward_cfg.get('foot_crossing_reward_weight', '—')}",
            f"  foot_crossing_penalty: {reward_cfg.get('foot_crossing_penalty_weight', '—')}",
            "",
            "Imit — joints (clip)",
            f"  joint_pos_rew: {_hdr_ijr if _hdr_ijr is not None else '—'}",
            f"  joint_pos_pen: {_hdr_ijp if _hdr_ijp is not None else '—'}",
            f"  joint_pos_reward_sharpness: {reward_cfg.get('imitation_joints_position_reward_sharpness', '—')}",
            f"  joint_vel_rew: {reward_cfg.get('imitation_velocity_reward_weight', '—')}",
            f"  joint_vel_pen: {reward_cfg.get('imitation_velocity_penalty_weight', '—')}",
            f"  motion_loops: {reward_cfg.get('hdf5_imitation_loop_count', reward_cfg.get('npz_imitation_loop_count', '—'))}",
            f"  motion_time_scale: {reward_cfg.get('hdf5_imitation_time_factor', reward_cfg.get('npz_imitation_time_factor', '—'))}",
            f"  motion_pos_tol_pct: {reward_cfg.get('hdf5_imitation_position_tolerance_pct', reward_cfg.get('npz_imitation_position_tolerance_pct', '—'))}",
            f"  motion_vel_tol_pct: {reward_cfg.get('hdf5_imitation_velocity_tolerance_pct', reward_cfg.get('npz_imitation_velocity_tolerance_pct', '—'))}",
            f"  leg_arm_w_pos: {reward_cfg.get('imitation_joint_position_use_leg_arm_weights', '—')}",
            f"  leg_arm_w_vel: {reward_cfg.get('imitation_joint_velocity_use_leg_arm_weights', '—')}",
            f"  imit_ramp_steps: {reward_cfg.get('imitation_strength_ramp_steps', '—')}",
            f"  imit_ramp_easing: {reward_cfg.get('imitation_strength_ramp_easing', '—')}",
            f"  adapter_phase: {reward_cfg.get('adapter_phase_enabled', '—')}",
            f"  adapter_duration: {reward_cfg.get('adapter_phase_duration_steps', '—')}",
            f"  adapter_start_scale: {reward_cfg.get('adapter_phase_start_scale', '—')}",
            f"  adapter_easing: {reward_cfg.get('adapter_phase_easing', '—')}",
            f"  legs_w: {reward_cfg.get('imitation_legs_weight', '—')}",
            f"  arms_w: {reward_cfg.get('imitation_arms_weight', '—')}",
            f"  joint_vel_sigma_rs: {reward_cfg.get('imitation_velocity_sigma_rad_s', '—')}",
            f"  periodic_joint_vel_fd: {reward_cfg.get('hdf5_imitation_periodic_joint_vel_fd', '—')}",
            f"  obs_phase: {reward_cfg.get('imitation_phase_in_obs', '—')}",
            f"  obs_link_err: {reward_cfg.get('imitation_pelvis_link_error_in_obs', '—')}",
            f"  obs_root_lv: {reward_cfg.get('imitation_root_lin_vel_in_obs', '—')}",
            "",
            "Imit — links 4× (pelvis)",
            f"  link_pos_rew: {reward_cfg.get('hdf5_link_local_imitation_reward_weight', reward_cfg.get('npz_link_local_imitation_reward_weight', '—'))}",
            f"  link_pos_pen: {reward_cfg.get('hdf5_link_local_imitation_penalty_weight', reward_cfg.get('npz_link_local_imitation_penalty_weight', '—'))}",
            f"  hdf5_link_local_imitation_periodic_fd: {reward_cfg.get('hdf5_link_local_imitation_periodic_fd', '—')}",
            f"  hdf5_link_local_imitation_position_reward_penalty_xy_only: {reward_cfg.get('hdf5_link_local_imitation_position_reward_penalty_xy_only', '—')}",
            f"  link_tol_m: {reward_cfg.get('hdf5_link_local_imitation_tolerance_m', reward_cfg.get('npz_link_local_imitation_tolerance_m', '—'))}",
            f"  link_vel_rew: {reward_cfg.get('hdf5_link_local_imitation_velocity_reward_weight', reward_cfg.get('npz_link_local_imitation_velocity_reward_weight', '—'))}",
            f"  link_vel_pen: {reward_cfg.get('hdf5_link_local_imitation_velocity_penalty_weight', reward_cfg.get('npz_link_local_imitation_velocity_penalty_weight', '—'))}",
            f"  link_vel_sigma: {reward_cfg.get('hdf5_link_local_imitation_velocity_sigma_m_per_s', reward_cfg.get('npz_link_local_imitation_velocity_sigma_m_per_s', '—'))}",
            f"  link_deriv_pen: {reward_cfg.get('hdf5_link_local_imitation_err_derivative_penalty_weight', reward_cfg.get('npz_link_local_imitation_err_derivative_penalty_weight', '—'))}",
            f"  link_slot_w: {reward_cfg.get('hdf5_link_local_imitation_slot_weights', reward_cfg.get('npz_link_local_imitation_slot_weights', '—'))}",
            f"  link_huber_m: {reward_cfg.get('hdf5_link_local_imitation_huber_delta_m', reward_cfg.get('npz_link_local_imitation_huber_delta_m', '—'))}",
            f"  link_dist_cap_m: {reward_cfg.get('hdf5_link_local_imitation_distance_cap_m', reward_cfg.get('npz_link_local_imitation_distance_cap_m', '—'))}",
            f"  link_contact_tol_scl: {reward_cfg.get('hdf5_link_local_contact_tolerance_scale', reward_cfg.get('npz_link_local_contact_tolerance_scale', '—'))}",
            f"  link_pos_scale_x: {reward_cfg.get('hdf5_link_local_imitation_pos_scale_x', reward_cfg.get('npz_link_local_imitation_pos_scale_x', '—'))}",
            "",
            "Imit — ankles & hip/knee (motion clip)",
            f"  ankle_roll_rew: {reward_cfg.get('hdf5_ankle_roll_y_rot_imitation_reward_weight', '—')}",
            f"  ankle_roll_pen (MSE vs clip): {reward_cfg.get('hdf5_ankle_roll_y_rot_imitation_penalty_weight', '—')}",
            f"  ankle_roll_sigma: {reward_cfg.get('hdf5_ankle_roll_y_rot_imitation_sigma_rad', '—')}",
            f"  ankle_roll_tol_rad: {reward_cfg.get('hdf5_ankle_roll_y_rot_imitation_tolerance_rad', '—')}",
            f"  hip_knee_y_rot_rew: {reward_cfg.get('hdf5_hip_knee_y_rot_imitation_reward_weight', '—')}",
            f"  hip_knee_y_rot_sigma: {reward_cfg.get('hdf5_hip_knee_y_rot_imitation_sigma_rad', '—')}",
            f"  hip_knee_y_rot_tol_rad: {reward_cfg.get('hdf5_hip_knee_y_rot_imitation_tolerance_rad', '—')}",
            f"  hip_knee_y_rot_pen: {reward_cfg.get('hdf5_hip_knee_y_rot_imitation_penalty_weight', '—')}",
            f"  ankle_foot_pos_rew: {reward_cfg.get('hdf5_ankle_roll_link_pos_imitation_reward_weight', '—')}",
            f"  ankle_foot_pos_sigma_m: {reward_cfg.get('hdf5_ankle_roll_link_pos_imitation_sigma_m', '—')}",
            f"  ankle_foot_pos_pen: {reward_cfg.get('hdf5_ankle_roll_link_pos_imitation_penalty_weight', '—')}",
            f"  ankle_foot_tol_m: {reward_cfg.get('hdf5_ankle_roll_link_pos_imitation_tolerance_m', '—')}",
            f"  hip_yaw_link_pos_rew: {reward_cfg.get('hdf5_hip_yaw_link_pos_imitation_reward_weight', '—')}",
            f"  hip_yaw_link_pos_sigma_m: {reward_cfg.get('hdf5_hip_yaw_link_pos_imitation_sigma_m', '—')}",
            f"  hip_yaw_link_pos_pen: {reward_cfg.get('hdf5_hip_yaw_link_pos_imitation_penalty_weight', '—')}",
            f"  hip_yaw_link_pos_tol_m: {reward_cfg.get('hdf5_hip_yaw_link_pos_imitation_tolerance_m', '—')}",
            f"  hip_yaw_link_pos_reward_tol_m: {reward_cfg.get('hdf5_hip_yaw_link_pos_imitation_reward_tolerance_m', '—')}",
            f"  hip_yaw_link_pos_pair_pen: {reward_cfg.get('hdf5_hip_yaw_link_pos_pair_penalty_weight', '—')}",
            f"  hip_yaw_link_pos_pair_tol_m: {reward_cfg.get('hdf5_hip_yaw_link_pos_pair_penalty_tolerance_m', '—')}",
        ])
        miss_p, miss_n = _missing_active_reward_penalty_keys(reward_cfg, series)
        if miss_p or miss_n:
            lines.append("")
            lines.append("Missing in log (non-zero cfg, no series)")
            for k in miss_p:
                tail = k.replace("Episode_Reward/", "")
                lab = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
                lines.append(f"  [reward] {lab}")
            for k in miss_n:
                tail = k.replace("Episode_Reward/", "")
                lab = EPISODE_REWARD_SHORT_LABEL.get(tail, tail.replace("_", " "))
                lines.append(f"  [penalty] {lab}")
            print(
                f"[INFO] training_progress_analysis: {len(miss_p)} active reward(s) and {len(miss_n)} active penalty/cost(s) "
                "have no Episode_Reward/* line in this log (listed in PNG sidebar).",
                file=sys.stderr,
            )
    lines = _filter_config_sidebar_lines(lines)
    ax_cfg.text(0.05, 0.98, "\n".join(lines), transform=ax_cfg.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, pad=0.8))
    ax_cfg.set_axis_off()

    out_path = os.path.join(log_dir, output_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print training_progress_log.txt for g1_locomotion_v4 and generate visual analysis "
            "(locomotion, imitation, symmetry, and general penalty panels)."
        )
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Run directory, e.g. logs/rsl_rl/g1_locomotion_v3/2026-03-18_12-00-00",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Print only the last N iteration blocks (default: all)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate the visual analysis PNG",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_progress_analysis.png",
        metavar="NAME",
        help="Output filename for the plot (default: training_progress_analysis.png)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate the visual analysis PNG; do not print the log.",
    )
    parser.add_argument(
        "--reward-penalty-ylim-focus-after",
        type=int,
        default=300,
        metavar="ITER",
        help=(
            "Rewards/penalties: y-limits use iterations >= max(ITER, start of last "
            "--reward-penalty-ylim-late-frac of the run). Default: 300. Use -1 for full autoscale."
        ),
    )
    parser.add_argument(
        "--reward-penalty-ylim-late-frac",
        type=float,
        default=0.22,
        metavar="FRAC",
        help=(
            "Fraction of the logged iteration *span* used as the late phase for y-axis scaling "
            "(default 0.22 = last 22%%). Combined with --reward-penalty-ylim-focus-after via max()."
        ),
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
        )
        if out_path:
            print(f"[INFO] Visual analysis saved to: {out_path}", file=sys.stderr)
        else:
            print("[WARNING] Could not generate visual analysis (missing data or matplotlib).", file=sys.stderr)


if __name__ == "__main__":
    main()

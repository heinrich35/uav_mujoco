# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with RSL-RL.",
    epilog="""
PPO Hyperparameter Tuning (ion vertical takeoff example):
  --max_iterations 20000
  --ppo_learning_rate 2e-5
  --ppo_entropy_coef 0.03          # Higher = more exploration
  --ppo_init_noise_std 0.6         # Higher initial exploration
  --ppo_desired_kl 0.005
  --ppo_num_steps_per_env 40       # More steps per env = more diverse rollouts
  --ppo_noise_std_type log         # Keep action std from collapsing (more exploration)
  --ppo_max_grad_norm 0.5          # Slightly higher = larger policy updates
  --ppo_lam 0.95                   # GAE lambda (lower = more exploratory advantage)

Full example with reward + exploration tuning:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
    --task Isaac-Ion-Vertical-Takeoff-Direct-v0 --headless \\
    --max_iterations 20000 \\
    --target_velocity_threshold 0.5 --positive_z_velocity_reward_weight 2.0 --z_velocity_sharpness 1.0 \\
    --low_altitude_penalty_threshold 0.02 \\
    --ppo_entropy_coef 0.04 --ppo_init_noise_std 1.5 \\
    --ppo_num_steps_per_env 40 --ppo_noise_std_type log
""",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--policy_path",
    type=str,
    default=None,
    help="Path to pre-trained policy model (.pt file) for transition learning.",
)
parser.add_argument(
    "--starting_policy_path",
    type=str,
    default=None,
    help="Path to checkpoint to use as initial policy before training (e.g. G1 standing for walking transfer). "
    "Used when not resuming. Default: none (train from scratch). G1 walking/teleop scripts pass this explicitly.",
)
parser.add_argument(
    "--joints_data",
    type=str,
    default=None,
    help="Path to NPZ file containing target joint positions or joint range of motion for limits (e.g. 'npz/stand_pose.npz').",
)
parser.add_argument(
    "--npz_only_for_limits",
    "--hdf5_only_for_limits",
    action="store_true",
    default=False,
    help="Motion file (.hdf5 / .npz) is used only for soft joint limits, not frame-by-frame imitation. Alias: --hdf5_only_for_limits.",
)
parser.add_argument(
    "--start_pose_path",
    type=str,
    default=None,
    help="Path to NPZ file containing start pose (e.g., 'npz/stand_pose.npz'). If not provided, uses initial robot pose.",
)
parser.add_argument(
    "--transition_duration",
    type=float,
    default=None,
    help="Duration (seconds) for smooth transition from start to target pose (default: 3.0).",
)
parser.add_argument(
    "--policy_control_duration",
    type=float,
    default=None,
    help="Duration (seconds) to use pre-trained policy before transitioning (for transition learning).",
)
parser.add_argument(
    "--policy_transition_duration",
    type=float,
    default=None,
    help="Duration (seconds) for smooth transition from policy to agent actions. Actions are blended during transition.",
)
parser.add_argument(
    "--joint_position_reward_weight",
    type=float,
    default=None,
    help="Weight for joint position reward (base weight, used as fallback if group weights not set).",
)
parser.add_argument(
    "--joint_position_reward_weight_legs",
    type=float,
    default=None,
    help="Weight for leg joint position reward (hip, knee, ankle joints).",
)
parser.add_argument(
    "--joint_position_reward_weight_arms",
    type=float,
    default=None,
    help="Weight for arm joint position reward (shoulder, elbow joints).",
)
parser.add_argument(
    "--joint_position_reward_weight_fingers",
    type=float,
    default=None,
    help="Weight for finger/hand joint position reward.",
)
parser.add_argument(
    "--joint_position_decay_legs",
    type=float,
    default=None,
    help="Decay constant for leg joint position reward (in rad^2). Larger = more forgiving. None = auto-compute from tolerance.",
)
parser.add_argument(
    "--joint_position_decay_arms",
    type=float,
    default=None,
    help="Decay constant for arm joint position reward (in rad^2). Larger = more forgiving. None = auto-compute from tolerance.",
)
parser.add_argument(
    "--joint_position_decay_fingers",
    type=float,
    default=None,
    help="Decay constant for finger joint position reward (in rad^2). Larger = more forgiving. None = auto-compute from tolerance.",
)
parser.add_argument(
    "--stability_weight",
    type=float,
    default=None,
    help="Weight for stability reward (staying upright).",
)
parser.add_argument(
    "--balance_weight",
    type=float,
    default=None,
    help="Weight for balance reward (penalizing excessive angular velocity).",
)
parser.add_argument(
    "--x_axis_drift_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for X-axis drift (forward/backward movement).",
)
parser.add_argument(
    "--y_axis_drift_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for Y-axis drift (lateral movement).",
)
parser.add_argument(
    "--z_axis_rotation_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for Z-axis rotation (yaw rotation).",
)
parser.add_argument(
    "--x_axis_alignment_reward_weight",
    type=float,
    default=None,
    help="Weight for X-axis alignment reward (encouraging robot to face X+ direction).",
)
parser.add_argument(
    "--feet_distance_reward_weight",
    type=float,
    default=None,
    help="Weight for feet distance reward (encouraging keeping feet at target distance).",
)
parser.add_argument(
    "--feet_on_ground_reward_weight",
    type=float,
    default=None,
    help="Weight for feet-on-ground reward (encouraging keeping both feet at starting z position).",
)
parser.add_argument(
    "--frames_per_second",
    "--motion_clip_fps",
    "--hdf5_fps",
    type=int,
    default=None,
    dest="frames_per_second",
    metavar="FPS",
    help="Motion clip sampling rate (rows/s in HDF5/NPZ). For G1 Locomotion V3/V4 sets hdf5_fps_override / npz_fps_override "
    "(else configs with frames_per_second only). Must match export unless you intentionally resample.",
)
parser.add_argument(
    "--enable_interpolation",
    type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'on']),
    default=None,
    help="Whether to enable interpolation between frames (True/False).",
)
parser.add_argument(
    "--move_forward_reward_weight",
    type=float,
    default=None,
    help="Weight for forward movement reward (X+ axis).",
)
parser.add_argument(
    "--loop_duration_seconds",
    type=float,
    default=None,
    help="Duration in seconds to extract from NPZ for looping (e.g., 1.0 = first 30 frames at 30fps). If None, uses all frames.",
)
parser.add_argument(
    "--loop_count",
    type=int,
    default=None,
    help="Number of times to repeat the loop_duration_seconds segment (e.g., 4 = repeat 4 times). If 1 or None, no looping.",
)
parser.add_argument(
    "--torso_pitch_forward_reward_weight",
    type=float,
    default=None,
    help="Reward weight for positive Y-axis rotation (forward pitch) to prevent backward lean.",
)
parser.add_argument(
    "--joint_limit_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for exceeding NPZ joint limits. Recommended: 5000-10000 for extreme violations (100+ degrees).",
)
parser.add_argument(
    "--joint_limit_reward_weight",
    type=float,
    default=None,
    help="Reward weight for staying inside motion-file soft joint limits (margin kernel; pairs with --joint_limit_penalty_weight). G1 locomotion.",
)
parser.add_argument(
    "--joint_limit_reward_sigma",
    type=float,
    default=None,
    help="σ (rad) for joint-limit margin reward: mean_j (1 - exp(-margin_j/σ)). G1 locomotion. Default from env cfg (e.g. 0.1).",
)
parser.add_argument(
    "--joint_limit_penalty_exponent",
    type=float,
    default=None,
    help="Exponent for joint limit penalty scaling (2.0=quadratic, 3.0=cubic). Higher = more aggressive for extreme violations. Default: 2.0",
)
parser.add_argument(
    "--dof_at_limit_cost_scale",
    type=float,
    default=None,
    help="Scale for URDF dof-at-limit penalty (joints with |dof_pos_scaled|>0.98). G1 locomotion: multiplies per-step count subtracted in compute_rewards. Default 1.0.",
)
parser.add_argument(
    "--hdf5_joint_limit_extend_deg",
    "--npz_joint_limit_extend_deg",
    dest="npz_joint_limit_extend_deg",
    type=float,
    default=None,
    help="Extend motion-clip (HDF5) joint limits in degrees: lower -= value, upper += value (e.g. 5 → [-34.72-5, 35.90+5] deg). G1 locomotion. Alias: --npz_joint_limit_extend_deg.",
)
parser.add_argument(
    "--hdf5_joint_limit_extend_deg_non_leg",
    "--npz_joint_limit_extend_deg_non_leg",
    dest="npz_joint_limit_extend_deg_non_leg",
    type=float,
    default=None,
    help="Extend motion-clip (HDF5) joint limits in degrees for non-leg joints only (torso + arms). Legs use --hdf5_joint_limit_extend_deg. G1 locomotion. Alias: --npz_joint_limit_extend_deg_non_leg.",
)
parser.add_argument(
    "--hdf5_limits_exclude_four_ankles",
    "--npz_limits_exclude_four_ankles",
    dest="npz_limits_exclude_four_ankles",
    action="store_true",
    help="Omit L/R ankle pitch and roll from motion-clip (HDF5) soft joint limits (no joint_limit penalty on those four DOFs). G1 locomotion. Alias: --npz_limits_exclude_four_ankles.",
)
# G1 Locomotion Walking V2 / V3: NPZ frame-by-frame joint-angle imitation
parser.add_argument(
    "--imit_joint_reward",
    "--imitation_joints_position_reward_weight",
    type=float,
    default=None,
    dest="imitation_joints_position_reward_weight",
    help="Joint-angle motion imitation reward (exp kernel). Alias: --imitation_joints_position_reward_weight. G1 Locomotion.",
)
parser.add_argument(
    "--imit_joint_penalty",
    "--imitation_joints_position_penalty_weight",
    type=float,
    default=None,
    dest="imitation_joints_position_penalty_weight",
    help="Joint-angle motion imitation penalty (MSE). Alias: --imitation_joints_position_penalty_weight. G1 Locomotion.",
)
parser.add_argument(
    "--imit_joint_reward_sharpness",
    "--imitation_joints_position_reward_sharpness",
    type=float,
    default=None,
    dest="imitation_joints_position_reward_sharpness",
    help="Exp kernel sharpness for joint position imitation: exp(-mse*sharp/sigma²). 1=default; >1=sharper. G1 Locomotion.",
)
# Deprecated: use --imitation_joints_position_reward_weight / --imitation_joints_position_penalty_weight
parser.add_argument("--imitation_position_reward_weight", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--imitation_position_penalty_weight", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--imitation_reward_weight", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--imitation_penalty_weight", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--motion_loops",
    "--hdf5_imitation_loop_count",
    "--npz_imitation_loop_count",
    type=int,
    default=None,
    dest="hdf5_imitation_loop_count",
    help="Repeat motion clip this many times end-to-end (1 = no repeat). Aliases: --hdf5_imitation_loop_count. G1 V2/V3.",
)
parser.add_argument(
    "--motion_max_frames",
    "--hdf5_imitation_max_frames",
    "--npz_imitation_max_frames",
    type=int,
    default=None,
    dest="hdf5_imitation_max_frames",
    help="Cap frames loaded from clip (0 = all). Aliases: --hdf5_imitation_max_frames. G1 Locomotion.",
)
parser.add_argument(
    "--motion_time_scale",
    "--hdf5_imitation_time_factor",
    "--npz_imitation_time_factor",
    type=float,
    default=None,
    dest="hdf5_imitation_time_factor",
    help="Slow (>1) or fast (<1) motion vs sim time. Aliases: --hdf5_imitation_time_factor. G1 V2/V3.",
)
parser.add_argument(
    "--motion_pos_tol_pct",
    "--hdf5_imitation_position_tolerance_pct",
    "--npz_imitation_position_tolerance_pct",
    type=float,
    default=None,
    dest="hdf5_imitation_position_tolerance_pct",
    help="Joint position imitation tolerance (%% width for exp reward). Aliases: --hdf5_imitation_position_tolerance_pct. G1 V2.",
)
parser.add_argument(
    "--motion_vel_tol_pct",
    "--hdf5_imitation_velocity_tolerance_pct",
    "--npz_imitation_velocity_tolerance_pct",
    type=float,
    default=None,
    dest="hdf5_imitation_velocity_tolerance_pct",
    help="Joint velocity imitation tolerance (%% → σ rad/s). Aliases: --hdf5_imitation_velocity_tolerance_pct. G1 V3.",
)
parser.add_argument(
    "--hdf5_imitation_reward_episode_time_scale_enabled",
    type=int,
    choices=[0, 1],
    default=None,
    help="1: multiply HDF5 link/ankle/hip imitation rewards and penalties by episode progress in [0,1]. 0: off. G1 V3+.",
)
parser.add_argument(
    "--hdf5_imitation_reward_episode_time_c",
    type=float,
    default=None,
    help="Constant c: HDF5 link/ankle/hip imitation scaled by c + (episode_length/max_length)^exp when episode-time scaling is on.",
)
parser.add_argument(
    "--hdf5_imitation_reward_episode_time_exponent",
    type=float,
    default=None,
    help="Exponent on (episode_length/max_episode_length) when HDF5 episode-time scaling is enabled (default 1.0 from cfg).",
)
# G1 Locomotion V3: imitation velocity (separate from position)
parser.add_argument(
    "--imit_vel_reward",
    "--imitation_velocity_reward_weight",
    type=float,
    default=None,
    dest="imitation_velocity_reward_weight",
    help="Joint velocity imitation reward (exp kernel). Alias: --imitation_velocity_reward_weight. G1 V3.",
)
parser.add_argument(
    "--imit_vel_penalty",
    "--imitation_velocity_penalty_weight",
    type=float,
    default=None,
    dest="imitation_velocity_penalty_weight",
    help="Joint velocity imitation penalty (L2). Alias: --imitation_velocity_penalty_weight. G1 V3.",
)
parser.add_argument(
    "--imitation_legs_only",
    action="store_true",
    help="Use only leg joints (hip, knee, ankle) for imitation position/velocity rewards and penalties. G1 Locomotion.",
)
parser.add_argument(
    "--imitation_exclude_joints",
    type=str,
    default=None,
    help="Comma-separated robot joint names to exclude from imitation rewards/penalties (exact names, e.g. left_ankle_roll_joint). G1 Locomotion.",
)
parser.add_argument(
    "--imitation_exclude_four_ankles",
    action="store_true",
    help="Exclude left/right ankle pitch and roll joints from imitation (four joints). G1 Locomotion.",
)
parser.add_argument(
    "--imitation_legs_weight",
    type=float,
    default=None,
    help="Weight for leg joints in imitation; >1 emphasizes legs. G1 Locomotion V3. Default 1.0.",
)
parser.add_argument(
    "--imitation_arms_weight",
    type=float,
    default=None,
    help="Weight for arm/torso joints in imitation. G1 Locomotion V3. Default 1.0.",
)
# G1 Locomotion V3: HDF5 motion foot/hand positions in pelvis frame (link_local_pos or *_ik keys)
parser.add_argument(
    "--link_imit_reward",
    "--hdf5_link_local_imitation_reward_weight",
    "--npz_link_local_imitation_reward_weight",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_reward_weight",
    help="Pelvis-frame link position reward (4 slots: feet+hands). Aliases: --hdf5_link_local_imitation_reward_weight. G1 V3.",
)
parser.add_argument(
    "--link_imit_penalty",
    "--hdf5_link_local_imitation_penalty_weight",
    "--npz_link_local_imitation_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_penalty_weight",
    help="Pelvis-frame link position penalty. Aliases: --hdf5_link_local_imitation_penalty_weight. G1 V3.",
)
parser.add_argument(
    "--link_imit_tol_m",
    "--hdf5_link_local_imitation_tolerance_m",
    "--npz_link_local_imitation_tolerance_m",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_tolerance_m",
    help="Link pos reward/penalty distance scale (m). Aliases: --hdf5_link_local_imitation_tolerance_m. G1 V3.",
)
parser.add_argument(
    "--link_imit_reward_penalty_xy_only",
    "--hdf5_link_local_imitation_position_reward_penalty_xy_only",
    "--npz_link_local_imitation_position_reward_penalty_xy_only",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    dest="hdf5_link_local_imitation_position_reward_penalty_xy_only",
    help="Use only pelvis-frame x,y for link position reward/penalty distance and err-derivative penalty (exclude z). G1 V3.",
)
parser.add_argument(
    "--link_vel_reward",
    "--hdf5_link_local_imitation_velocity_reward_weight",
    "--npz_link_local_imitation_velocity_reward_weight",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_velocity_reward_weight",
    help="Pelvis-frame link velocity reward. Aliases: --hdf5_link_local_imitation_velocity_reward_weight. G1 V3.",
)
parser.add_argument(
    "--link_vel_penalty",
    "--hdf5_link_local_imitation_velocity_penalty_weight",
    "--npz_link_local_imitation_velocity_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_velocity_penalty_weight",
    help="Pelvis-frame link velocity penalty. Aliases: --hdf5_link_local_imitation_velocity_penalty_weight. G1 V3.",
)
parser.add_argument(
    "--link_vel_sigma",
    "--hdf5_link_local_imitation_velocity_sigma_m_per_s",
    "--npz_link_local_imitation_velocity_sigma_m_per_s",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_velocity_sigma_m_per_s",
    help="σ (m/s) for link velocity exp reward. Aliases: --hdf5_link_local_imitation_velocity_sigma_m_per_s. G1 V3.",
)
parser.add_argument(
    "--link_imit_deriv_pen",
    "--hdf5_link_local_imitation_err_derivative_penalty_weight",
    "--npz_link_local_imitation_err_derivative_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_err_derivative_penalty_weight",
    help="Penalty on Δ(link error) vs motion. Aliases: --hdf5_link_local_imitation_err_derivative_penalty_weight. G1 V3.",
)
parser.add_argument(
    "--link_slot_weights",
    "--hdf5_link_local_imitation_slot_weights",
    "--npz_link_local_imitation_slot_weights",
    type=str,
    default=None,
    dest="hdf5_link_local_imitation_slot_weights",
    help="Four weights: R foot, L foot, R hand, L hand (comma-separated). Aliases: --hdf5_link_local_imitation_slot_weights. G1 V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_huber_delta_m",
    "--npz_link_local_imitation_huber_delta_m",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_huber_delta_m",
    help="If set, link position penalty uses Huber(||error||) with this delta (m) instead of hinge². G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_distance_cap_m",
    "--npz_link_local_imitation_distance_cap_m",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_distance_cap_m",
    help="Cap link distance (m) before position reward/penalty (robustness). G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_contact_tolerance_scale",
    "--npz_link_local_contact_tolerance_scale",
    type=float,
    default=None,
    dest="hdf5_link_local_contact_tolerance_scale",
    help="With motion right_foot_contact/left_foot_contact: tol *= 1+(scale-1)*contact. 1.0=no effect. G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_derive_foot_contact_from_link_z",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="If motion has no contact datasets, derive soft foot contact from ankle z (pelvis frame). G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_derived_foot_contact_beta",
    type=float,
    default=None,
    help="Softmax sharpness for derived foot contact from ankle z (larger = sharper single stance). G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_stance_tolerance_scale",
    type=float,
    default=None,
    help="Foot link slots: when ref contact >= stance threshold, multiply tol by this (<1 = stricter). 1.0=off. G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_stance_ref_contact_threshold",
    type=float,
    default=None,
    help="Ref foot contact above this counts as stance for stance_tolerance_scale. G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_swing_penalty_mask",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Mask link position penalty when ref and sim both say that foot is in swing. G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_swing_ref_contact_threshold",
    type=float,
    default=None,
    help="Ref contact below this = swing for swing_penalty_mask. G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_joint_ref_contact_leg_scale",
    type=float,
    default=None,
    help="Leg joint imitation weights *= 1 + scale * motion foot contact (L/R from joint name). 0=off. G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_root_xy_reward_weight",
    type=float,
    default=None,
    help="Reward horizontal root xy vs motion root_pos (world delta vs ref). G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_root_xy_penalty_weight",
    type=float,
    default=None,
    help="Penalty on horizontal root xy vs motion root_pos. G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_root_xy_sigma_m",
    type=float,
    default=None,
    help="Exp kernel scale (m) for imitation_root_xy_reward. G1 Locomotion V3.",
)
parser.add_argument(
    "--swing_foot_clearance_reward_weight",
    type=float,
    default=None,
    help="Reward swing ankle height above stance foot (single support). G1 Locomotion V3.",
)
parser.add_argument(
    "--swing_foot_clearance_penalty_weight",
    type=float,
    default=None,
    help="Penalty on swing ankle clearance shortfall below margin (single support). G1 Locomotion V3.",
)
parser.add_argument(
    "--swing_foot_clearance_sigma_m",
    type=float,
    default=None,
    help="Clearance shaping scale (m). G1 Locomotion V3.",
)
parser.add_argument(
    "--swing_foot_clearance_margin_m",
    type=float,
    default=None,
    help="Minimum clearance above stance foot (m) before reward. G1 Locomotion V3.",
)
parser.add_argument(
    "--stance_foot_slip_penalty_weight",
    type=float,
    default=None,
    help="Penalty on horizontal foot speed when ankle proxy says stance. G1 Locomotion V3.",
)
parser.add_argument(
    "--symmetry_torque_terms_double_support_only",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Accumulate ankle/hip torque symmetry only in double support. G1 walking / V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_pos_scale_x",
    "--npz_link_local_imitation_pos_scale_x",
    type=float,
    default=None,
    dest="hdf5_link_local_imitation_pos_scale_x",
    help="Scale pelvis-frame x for the four imitation links; writes a temp HDF5 and sets joints_data to it. 1.0=no copy. G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_joint_position_data_scale",
    type=float,
    default=None,
    help="Scale only the motion-file joint-position dataset dof_pos; writes a temp HDF5 and sets joints_data to it. 1.0=no copy. G1 locomotion imitation.",
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_ramp_enabled",
    type=int,
    choices=[0, 1],
    default=None,
    help=(
        "1: ramp joint-angle imitation scale at runtime (common_step_counter); train.py skips temp dof_pos HDF5. "
        "0: off. G1 V3/V4."
    ),
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_ramp_start",
    type=float,
    default=None,
    help="Joint imitation dof scale at ramp start (after optional delay). G1 V3/V4.",
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_ramp_end",
    type=float,
    default=None,
    help="Joint imitation dof scale after ramp_steps MDP steps. G1 V3/V4.",
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_ramp_steps",
    type=int,
    default=None,
    help=(
        "Linear ramp length in common_step_counter ticks after delay (or × --num_steps_per_env if "
        "--hdf5_joint_position_data_scale_ramp_in_ppo_iterations=1). G1 V3/V4."
    ),
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_ramp_delay_steps",
    type=int,
    default=None,
    help=(
        "Hold ramp_start while common_step_counter <= this value; then linear ramp over ramp_steps to ramp_end "
        "(reaches ramp_end at counter delay+ramp_steps). Shell: HDF5_START_SCALER_STEPS sets this (overrides "
        "HDF5_JOINT_POSITION_DATA_SCALE_RAMP_DELAY_STEPS in G1_LOCOMOTION_* scripts). G1 V3/V4."
    ),
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_ramp_in_ppo_iterations",
    type=int,
    choices=[0, 1],
    default=None,
    help=(
        "1: interpret ramp_delay_steps and ramp_steps as PPO rollout-cycle counts (× --num_steps_per_env for "
        "common_step_counter). 0: values are raw env step counts. G1 V3/V4."
    ),
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_mode",
    type=str,
    default=None,
    help="all | symmetric_legs — which imitation joints receive the goaled scale factor. G1 V3/V4.",
)
parser.add_argument(
    "--hdf5_joint_position_data_scale_exclude_ankles",
    type=int,
    default=None,
    help="1: ankle DOFs (name contains ankle) keep joint-angle ref scale 1.0 — no dof_pos bake scale and no runtime ramp delta on those columns. G1 V3/V4.",
)
parser.add_argument(
    "--ankle_roll_rew",
    "--hdf5_ankle_roll_y_rot_imitation_reward_weight",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_y_rot_imitation_reward_weight",
    help=(
        "Reward: pelvis-frame ankle link roll (euler_xyz X) vs clip *y_rot* channels. "
        "Alias: --hdf5_ankle_roll_y_rot_imitation_reward_weight. G1 V3."
    ),
)
parser.add_argument(
    "--ankle_roll_sigma",
    "--hdf5_ankle_roll_y_rot_imitation_sigma_rad",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_y_rot_imitation_sigma_rad",
    help="σ (rad) for ankle roll exp reward. Alias: --hdf5_ankle_roll_y_rot_imitation_sigma_rad. G1 V3.",
)
parser.add_argument(
    "--ankle_roll_tol_rad",
    "--hdf5_ankle_roll_y_rot_imitation_tolerance_rad",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_y_rot_imitation_tolerance_rad",
    help="Angular deadband (rad) before exp error. Alias: --hdf5_ankle_roll_y_rot_imitation_tolerance_rad. G1 V3.",
)
parser.add_argument(
    "--ankle_roll_pen",
    "--hdf5_ankle_roll_y_rot_imitation_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_y_rot_imitation_penalty_weight",
    help="Penalty: mean squared wrapped angle error vs clip (pairs with --ankle_roll_rew). G1 V3.",
)
parser.add_argument(
    "--hip_knee_y_rot_rew",
    "--hdf5_hip_knee_y_rot_imitation_reward_weight",
    type=float,
    default=None,
    dest="hdf5_hip_knee_y_rot_imitation_reward_weight",
    help=(
        "Reward: pelvis-frame hip/knee link pitch (Euler Y vs clip) for four link_*_y_rot channels. "
        "Alias: --hdf5_hip_knee_y_rot_imitation_reward_weight. G1 V3."
    ),
)
parser.add_argument(
    "--hip_knee_y_rot_sigma",
    "--hdf5_hip_knee_y_rot_imitation_sigma_rad",
    type=float,
    default=None,
    dest="hdf5_hip_knee_y_rot_imitation_sigma_rad",
    help="σ (rad) for hip/knee link y-rot exp reward. Alias: --hdf5_hip_knee_y_rot_imitation_sigma_rad. G1 V3.",
)
parser.add_argument(
    "--hip_knee_y_rot_tol_rad",
    "--hdf5_hip_knee_y_rot_imitation_tolerance_rad",
    type=float,
    default=None,
    dest="hdf5_hip_knee_y_rot_imitation_tolerance_rad",
    help="Angular deadband (rad) before exp error. Alias: --hdf5_hip_knee_y_rot_imitation_tolerance_rad. G1 V3.",
)
parser.add_argument(
    "--hip_knee_y_rot_pen",
    "--hdf5_hip_knee_y_rot_imitation_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_hip_knee_y_rot_imitation_penalty_weight",
    help=(
        "Penalty: mean squared wrapped pitch error (rad²) vs clip for four links. "
        "Alias: --hdf5_hip_knee_y_rot_imitation_penalty_weight. G1 V3."
    ),
)
parser.add_argument(
    "--ankle_foot_pos_pen",
    "--hdf5_ankle_roll_link_pos_imitation_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_link_pos_imitation_penalty_weight",
    help="Penalty: pelvis-frame ankle link xyz vs clip. Alias: --hdf5_ankle_roll_link_pos_imitation_penalty_weight. G1 V3.",
)
parser.add_argument(
    "--ankle_foot_tol_m",
    "--hdf5_ankle_roll_link_pos_imitation_tolerance_m",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_link_pos_imitation_tolerance_m",
    help="3D deadband (m) for ankle foot pos penalty/reward; 0 = pure MSE kernel. Alias: --hdf5_ankle_roll_link_pos_imitation_tolerance_m. G1 V3.",
)
parser.add_argument(
    "--ankle_foot_pos_rew",
    "--hdf5_ankle_roll_link_pos_imitation_reward_weight",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_link_pos_imitation_reward_weight",
    help="Reward: exp on pelvis-frame ankle xyz vs clip (pairs with ankle_foot_pos_pen). G1 V3.",
)
parser.add_argument(
    "--ankle_foot_pos_sigma_m",
    "--hdf5_ankle_roll_link_pos_imitation_sigma_m",
    type=float,
    default=None,
    dest="hdf5_ankle_roll_link_pos_imitation_sigma_m",
    help="σ (m) for ankle foot position exp reward. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_pen",
    "--hdf5_hip_yaw_link_pos_imitation_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_imitation_penalty_weight",
    help="Penalty: pelvis-frame hip yaw link xyz vs clip. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_tol_m",
    "--hdf5_hip_yaw_link_pos_imitation_tolerance_m",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_imitation_tolerance_m",
    help="3D deadband (m) for hip yaw link pos penalty/reward. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_rew",
    "--hdf5_hip_yaw_link_pos_imitation_reward_weight",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_imitation_reward_weight",
    help="Reward: exp on pelvis-frame hip yaw xyz vs clip. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_sigma_m",
    "--hdf5_hip_yaw_link_pos_imitation_sigma_m",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_imitation_sigma_m",
    help="σ (m) for hip yaw link position exp reward. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_reward_tol_m",
    "--hdf5_hip_yaw_link_pos_imitation_reward_tolerance_m",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_imitation_reward_tolerance_m",
    help="Per-link deadband (m) for hip yaw pos exp reward only; <0 uses imitation_tolerance_m. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_pair_pen",
    "--hdf5_hip_yaw_link_pos_pair_penalty_weight",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_pair_penalty_weight",
    help="Penalty when (right−left) pelvis vector ≠ reference. G1 V3.",
)
parser.add_argument(
    "--hip_yaw_link_pos_pair_tol_m",
    "--hdf5_hip_yaw_link_pos_pair_penalty_tolerance_m",
    type=float,
    default=None,
    dest="hdf5_hip_yaw_link_pos_pair_penalty_tolerance_m",
    help="Deadband (m) on ‖pair_err‖ before hip yaw pair penalty. G1 V3.",
)
parser.add_argument(
    "--include_root_xy_error_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Append root (x,y) minus env origin (world drift) to policy obs. G1 Locomotion V1 Walking / V2 / V3.",
)
parser.add_argument(
    "--root_xy_error_obs_scale",
    type=float,
    default=None,
    help="Scale for root XY error obs (meters * scale). E.g. 1/y_drift_tolerance for O(1) values. G1 locomotion walking.",
)
parser.add_argument(
    "--imitation_phase_in_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Add sin/cos motion phase to policy observation. G1 Locomotion V3. Default False.",
)
parser.add_argument(
    "--imitation_pelvis_link_error_in_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Append normalized pelvis-frame link errors (12) to observation. G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_root_lin_vel_in_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Append motion root_lin_vel (3) from normalized HDF5. G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_root_lin_vel_obs_scale",
    type=float,
    default=None,
    help="Scale factor for root_lin_vel observation channel. G1 Locomotion V3.",
)
# G1 Locomotion V4: reward profile split + command observations
parser.add_argument(
    "--reward_profile",
    type=str,
    default=None,
    help="G1 V4 / Ion V2 / cruise V2 / G1 standing IK V2: profile (G1 V4: locomotion|imitation|full|staged; standing IK V2: standing|ik|full|staged; Ion/cruise: see task docs).",
)
parser.add_argument(
    "--staged_imitation_start_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps before imitation scale ramps (reward_profile=staged).",
)
parser.add_argument(
    "--staged_imitation_ramp_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps over which imitation scale goes 0→1 after start.",
)
parser.add_argument(
    "--staged_ramp_start_scale",
    type=float,
    default=None,
    help="G1 V4: starting scale at ramp start for staged imitation/dynamic-balance/contact-gait/symmetry groups (e.g. 0.5 gives 0.5→1 ramp).",
)
parser.add_argument(
    "--staged_dybl_start_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps before dynamic-balance reward/penalty weights ramp from 0.",
)
parser.add_argument(
    "--staged_dybl_ramp_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps over which dynamic-balance reward/penalty weights ramp 0→1.",
)
parser.add_argument(
    "--staged_symmetry_start_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps before symmetry reward/penalty weights ramp from 0.",
)
parser.add_argument(
    "--staged_symmetry_ramp_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps over which symmetry reward/penalty weights ramp 0→1.",
)
parser.add_argument(
    "--staged_symmetry_ramp_start_scale",
    type=float,
    default=None,
    help="G1 V4: symmetry weight ramp start in [0,1] (linear ramp to 1). G1_LOCOMOTION_V4_WALK_IM_60fps_TRAIN.sh defaults to --staged_ramp_start_scale; omit CLI arg for cfg None → runtime uses staged_ramp_start_scale.",
)
parser.add_argument(
    "--staged_contact_gait_start_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps before contact/gait reward weights ramp (jump/foot-contact/gait/swing/stance slip).",
)
parser.add_argument(
    "--staged_contact_gait_ramp_optimizer_steps",
    type=int,
    default=None,
    help="G1 V4: optimizer steps over which contact/gait weights ramp staged_ramp_start_scale→1; 0 disables.",
)
parser.add_argument(
    "--staged_ik_start_env_steps",
    type=int,
    default=None,
    help="G1 Standing IK V2: env steps before IK reward group ramps (reward_profile=staged).",
)
parser.add_argument(
    "--staged_ik_ramp_env_steps",
    type=int,
    default=None,
    help="G1 Standing IK V2: env steps over which IK scale goes 0→1 after start.",
)
parser.add_argument(
    "--velocity_command_obs_enabled",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="G1 V4: append normalized per-episode vx command to policy observation.",
)
parser.add_argument(
    "--velocity_command_obs_scale",
    type=float,
    default=None,
    help="G1 V4: divide command vx by this for observation (e.g. 1.5).",
)
parser.add_argument(
    "--include_reward_profile_phase_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="G1 V4: append scalar imitation-phase progress [0,1] (staged profile).",
)
parser.add_argument(
    "--target_yaw_command_obs_enabled",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="G1 V4: append sin/cos(target_z_orientation) to policy observation.",
)
parser.add_argument(
    "--append_teleop_hand_target_obs",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="G1 V4: append constant hand_target_pos_b×scale (3 dims, last) for teleop / "
    "G1StandingForTeleoperation-style obs alignment. Default: env cfg.",
)
parser.add_argument(
    "--gravity_scale",
    type=float,
    default=None,
    metavar="FACTOR",
    help="G1 V4 / G1 Standing IK V2: world gravity z is -9.80665×FACTOR (m/s²). Default: env cfg (1.0).",
)
parser.add_argument(
    "--progress_goal_world_offset_xy_m",
    type=str,
    default=None,
    help="G1 V4: progress/heading goal offset from each env origin in world xy (m), 'forward,lateral' e.g. "
    "'1000,0' (V1 default). Use 'none' to keep env cfg None (inherit V1 targets without override).",
)
# G1 Locomotion V4: episode-level L-R symmetry (accumulated ROM + step lengths)
parser.add_argument(
    "--episode_symmetry_min_episode_steps",
    type=int,
    default=None,
    help="G1 V4: minimum episode steps before applying episode_symmetry reward/penalty terms.",
)
parser.add_argument(
    "--episode_symmetry_reward_weight",
    type=float,
    default=None,
    help="G1 V4: legacy aggregate episode symmetry reward weight (blended step/joint/link).",
)
parser.add_argument(
    "--episode_symmetry_step_length_blend",
    type=float,
    default=None,
    help="G1 V4: blend weight for step-length term inside legacy aggregate symmetry reward.",
)
parser.add_argument(
    "--episode_symmetry_joint_rom_blend",
    type=float,
    default=None,
    help="G1 V4: blend weight for joint-ROM term inside legacy aggregate symmetry reward.",
)
parser.add_argument(
    "--episode_symmetry_link_rom_blend",
    type=float,
    default=None,
    help="G1 V4: blend weight for link-ROM term inside legacy aggregate symmetry reward.",
)
parser.add_argument(
    "--episode_symmetry_step_length_reward_weight",
    type=float,
    default=None,
    help="G1 V4: per-category episode symmetry reward for mean L/R step length match.",
)
parser.add_argument(
    "--episode_symmetry_step_length_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: per-category penalty for L/R step length asymmetry (1 - symmetry score).",
)
parser.add_argument(
    "--episode_symmetry_joint_rom_reward_weight",
    type=float,
    default=None,
    help="G1 V4: per-category reward for imitation-joint ROM L/R symmetry.",
)
parser.add_argument(
    "--episode_symmetry_joint_rom_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: per-category penalty for imitation-joint ROM asymmetry.",
)
parser.add_argument(
    "--episode_symmetry_link_rom_reward_weight",
    type=float,
    default=None,
    help="G1 V4: per-category reward for HDF5 link (pelvis-frame) ROM L/R symmetry.",
)
parser.add_argument(
    "--episode_symmetry_link_rom_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: per-category penalty for link ROM asymmetry.",
)
parser.add_argument(
    "--episode_symmetry_step_length_sigma_m",
    type=float,
    default=None,
    help="G1 V4: exp-kernel sigma (m) for step-length symmetry score.",
)
parser.add_argument(
    "--episode_symmetry_joint_rom_sigma_rad",
    type=float,
    default=None,
    help="G1 V4: exp-kernel sigma (rad) for joint-ROM symmetry score.",
)
parser.add_argument(
    "--episode_symmetry_link_rom_sigma_m",
    type=float,
    default=None,
    help="G1 V4: exp-kernel sigma (m) for link-ROM symmetry score.",
)
parser.add_argument(
    "--episode_symmetry_single_stance_balance_reward_weight",
    type=float,
    default=None,
    help="G1 V4: reward L/R single-support time balance (same XOR as gait_single_stance_reward).",
)
parser.add_argument(
    "--episode_symmetry_single_stance_balance_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: penalty for L/R single-support time imbalance.",
)
parser.add_argument(
    "--episode_symmetry_single_stance_balance_sigma",
    type=float,
    default=None,
    help="G1 V4: σ on normalized |n_L−n_R|/(n_L+n_R) for single-stance balance score.",
)
parser.add_argument(
    "--episode_symmetry_hdf5_ankle_roll_y_track_reward_weight",
    type=float,
    default=None,
    help="G1 V4: reward symmetry of mean |ankle roll tracking error| L vs R (HDF5 clip).",
)
parser.add_argument(
    "--episode_symmetry_hdf5_ankle_roll_y_track_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: penalty when L/R mean tracking error differs (ankle roll y-rot vs clip).",
)
parser.add_argument(
    "--episode_symmetry_hdf5_ankle_roll_y_track_sigma_rad",
    type=float,
    default=None,
    help="G1 V4: σ (rad) for ankle-roll tracking symmetry score.",
)
parser.add_argument(
    "--torso_z_align_link_body_name",
    type=str,
    default=None,
    help="G1 V4: rigid body name for torso/trunk z-axis upright reward (default from env cfg, e.g. torso_link).",
)
parser.add_argument(
    "--torso_z_axis_world_align_reward_weight",
    type=float,
    default=None,
    help="G1 V4: reward weight × exp(−err/σ) with err = 1 − (torso +Z)·world +Z.",
)
parser.add_argument(
    "--torso_z_axis_world_align_reward_sigma",
    type=float,
    default=None,
    help="G1 V4: σ for torso +Z vs world +Z alignment reward (same units as err).",
)
parser.add_argument(
    "--torso_z_axis_world_align_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: penalty weight × max(0, err − tolerance) for torso +Z vs world +Z misalignment.",
)
parser.add_argument(
    "--torso_z_axis_world_align_penalty_tolerance",
    type=float,
    default=None,
    help="G1 V4: no penalty when (1 − dot) ≤ this (0 = penalize any tilt).",
)
parser.add_argument(
    "--palm_below_pelvis_right_body_name",
    type=str,
    default=None,
    help="G1 V4: right palm body for palm-below-pelvis-z reward/penalty (default right_palm_link).",
)
parser.add_argument(
    "--palm_below_pelvis_left_body_name",
    type=str,
    default=None,
    help="G1 V4: left palm body (default left_palm_link).",
)
parser.add_argument(
    "--palm_below_pelvis_z_max_pelvis_frame_m",
    type=float,
    default=None,
    help="G1 V4: palm origin z in pelvis/root frame must stay ≤ this (m); +z is root up.",
)
parser.add_argument(
    "--palm_below_pelvis_z_reward_weight",
    type=float,
    default=None,
    help="G1 V4: reward × mean exp(−relu(z−z_max)²/σ²) over resolved palms.",
)
parser.add_argument(
    "--palm_below_pelvis_z_reward_sigma_m",
    type=float,
    default=None,
    help="G1 V4: σ (m) for palm-below-pelvis-z reward kernel.",
)
parser.add_argument(
    "--palm_below_pelvis_z_penalty_weight",
    type=float,
    default=None,
    help="G1 V4: penalty × mean relu(z − z_max − tol)² over palms.",
)
parser.add_argument(
    "--palm_below_pelvis_z_penalty_tolerance_m",
    type=float,
    default=None,
    help="G1 V4: slack above z_max before squared z penalty applies.",
)
parser.add_argument(
    "--com_velocity_tracking_reward_weight",
    type=float,
    default=None,
    help="G1 Locomotion V4: reward weight for tracking desired root CoM xy velocity along the travel direction.",
)
parser.add_argument(
    "--com_velocity_tracking_sigma_mps",
    type=float,
    default=None,
    help="G1 Locomotion V4: sigma (m/s) for CoM velocity tracking reward kernel.",
)
parser.add_argument(
    "--linear_momentum_projection_reward_weight",
    type=float,
    default=None,
    help="G1 Locomotion V4: reward forward linear momentum projected onto the command/progress direction.",
)
parser.add_argument(
    "--linear_momentum_projection_scale",
    type=float,
    default=None,
    help="G1 Locomotion V4: scale for tanh forward linear momentum reward.",
)
parser.add_argument(
    "--base_of_support_center_penalty_weight",
    type=float,
    default=None,
    help="G1 Locomotion V4: penalty when CoM stays too centered over the support center instead of leaning forward.",
)
parser.add_argument(
    "--base_of_support_forward_lead_min_m",
    type=float,
    default=None,
    help="G1 Locomotion V4: minimum forward CoM lead (m) relative to support center before BoS penalty is zero.",
)
parser.add_argument(
    "--dcm_capture_point_reward_weight",
    type=float,
    default=None,
    help="G1 Locomotion V4: reward weight for capture-point / DCM target-ahead shaping.",
)
parser.add_argument(
    "--dcm_capture_point_target_ahead_m",
    type=float,
    default=None,
    help="G1 Locomotion V4: desired forward DCM/capture-point distance ahead of CoM (m).",
)
parser.add_argument(
    "--dcm_capture_point_sigma_m",
    type=float,
    default=None,
    help="G1 Locomotion V4: sigma (m) for DCM/capture-point forward reward kernel.",
)
parser.add_argument(
    "--dcm_capture_point_lateral_penalty_weight",
    type=float,
    default=None,
    help="G1 Locomotion V4: penalty weight for lateral DCM/capture-point deviation outside tolerance.",
)
parser.add_argument(
    "--dcm_capture_point_lateral_tolerance_m",
    type=float,
    default=None,
    help="G1 Locomotion V4: lateral tolerance (m) before DCM/capture-point penalty applies.",
)
parser.add_argument(
    "--dcm_com_height_min_m",
    type=float,
    default=None,
    help="G1 Locomotion V4: minimum CoM height (m) used in omega=sqrt(g/h) for DCM computation.",
)
parser.add_argument(
    "--imitation_joint_position_use_leg_arm_weights",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Use imitation_legs_weight/imitation_arms_weight in joint position imitation mean. V3.",
)
parser.add_argument(
    "--imitation_joint_velocity_use_leg_arm_weights",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Use same leg/arm per-joint weights in joint velocity imitation (DeepMimic-style). G1 Locomotion V3.",
)
parser.add_argument(
    "--imit_ramp_steps",
    "--imitation_strength_ramp_steps",
    type=int,
    default=None,
    dest="imitation_strength_ramp_steps",
    help="Ramp all V3 imitation strengths 0→1 over this many optimizer steps. Alias: --imitation_strength_ramp_steps. V3.",
)
parser.add_argument(
    "--imitation_strength_ramp_easing",
    type=str,
    default=None,
    choices=["linear", "smoothstep", "cosine"],
    help="Easing for imitation strength ramp: linear, smoothstep (S-curve), or cosine. G1 Locomotion V3.",
)
parser.add_argument(
    "--adapter_phase",
    action="store_true",
    help="G1 Locomotion V3: enable adapter phase — ramp total per-step reward scale from --adapter_phase_start_scale to 1.0 "
    "over --adapter_phase_duration (common_step_counter steps; same clock as --imit_ramp_steps). Use when resuming / "
    "transferring with changed reward shaping.",
)
parser.add_argument(
    "--adapter_phase_duration",
    type=int,
    default=None,
    dest="adapter_phase_duration",
    metavar="N",
    help="Adapter phase length in env common_step_counter steps (not PPO iterations). 0 disables ramp. G1 Locomotion V3.",
)
parser.add_argument(
    "--adapter_phase_start_scale",
    type=float,
    default=None,
    dest="adapter_phase_start_scale",
    metavar="S",
    help="Reward multiplier at step 0 during adapter phase; ramps to 1.0. Typical 0.15–0.5. G1 Locomotion V3.",
)
parser.add_argument(
    "--adapter_phase_easing",
    type=str,
    default=None,
    choices=["linear", "smoothstep", "cosine"],
    help="Easing for adapter ramp (linear, smoothstep, cosine). G1 Locomotion V3.",
)
parser.add_argument(
    "--imitation_velocity_sigma_rad_s",
    type=float,
    default=None,
    help="If >0, joint vel imitation exp-kernel σ in rad/s (overrides hdf5_imitation_velocity_tolerance_pct). V3.",
)
parser.add_argument(
    "--motion_cycle_episode_margin_s",
    type=float,
    default=None,
    help="Extra seconds beyond one motion loop when auto-raising episode_length_s. G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_use_joint_motion_clock_when_aligned",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="If True, link interp uses joint motion clock when L and fps match. V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_use_body_lin_vel",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Use physics body COM lin vel (root frame) for link vel matching vs FD on pelvis positions. V3.",
)
parser.add_argument(
    "--hdf5_imitation_periodic_joint_vel_fd",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Periodic central FD when inferring joint ref velocities from dof_pos (looping clip). G1 Locomotion V3.",
)
parser.add_argument(
    "--hdf5_link_local_imitation_periodic_fd",
    "--npz_link_local_imitation_periodic_fd",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    dest="hdf5_link_local_imitation_periodic_fd",
    metavar="True|False",
    help="Periodic finite differences for link ref velocity when link_local_vel missing. V3.",
)
parser.add_argument(
    "--debug_training_fixed_base",
    action="store_true",
    help="G1 Locomotion V3: fix robot root (pelvis) at env-local (0,0,z); joints still move. Default z=0.75.",
)
parser.add_argument(
    "--debug_training_fixed_base_spawn_z",
    type=float,
    default=None,
    help="Root spawn z (m, env-local) with --debug_training_fixed_base. Default 0.75 from env cfg.",
)
# G1 locomotion smooth/jerk motion (add only if not already present, e.g. avoid duplicate from another module)
if "--action_rate_penalty_scale" not in (getattr(parser, "_option_string_actions", None) or {}):
    parser.add_argument(
        "--action_rate_penalty_scale",
        "--policy_action_roughness_penalty_scale",
        type=float,
        default=None,
        help="Pairs with policy_action_smoothness_*: penalty scale × roughness metric on Δaction (Huber sum if "
        "--policy_action_roughness_huber_delta > 0, else Σ‖Δa‖²). Logged policy_action_roughness_penalty. G1 locomotion.",
    )
    parser.add_argument(
        "--joint_acceleration_penalty_scale",
        "--joint_velocity_roughness_penalty_scale",
        type=float,
        default=None,
        help="Pairs with joint_velocity_smoothness_*: penalty scale × roughness metric on Δdof_vel (Huber sum if "
        "--joint_velocity_roughness_huber_delta_rad_s > 0, else Σ‖Δ‖²). Logged joint_velocity_roughness_penalty. G1 locomotion.",
    )
    parser.add_argument(
        "--smooth_motion_reward_weight",
        "--policy_action_smoothness_reward_weight",
        type=float,
        default=None,
        help="Smooth-motion reward weight (G1: exp(-roughness_metric/σ) on Δaction; ion VTOL: same cfg field). Default 0.0 = disabled.",
    )
    parser.add_argument(
        "--smooth_motion_sigma",
        "--policy_action_smoothness_sigma",
        type=float,
        default=None,
        help="σ for policy Δaction smoothness exp term; smaller = tighter. G1 locomotion. Default 0.1.",
    )
    parser.add_argument(
        "--policy_action_roughness_huber_delta",
        type=float,
        default=None,
        help="Per-action Huber threshold on |Δa| for policy_action rough/smooth pair (normalized cmd units). 0 = pure Σ(Δa)². "
        "Larger δ → more permissive on sharp command steps.",
    )
    parser.add_argument(
        "--smooth_motion_joint_accel_reward_weight",
        "--joint_velocity_smoothness_reward_weight",
        type=float,
        default=None,
        help="Smooth reward exp(-roughness_metric/σ) on joint Δq̇ (same metric as joint roughness penalty; G1 locomotion / standing IK).",
    )
    parser.add_argument(
        "--smooth_motion_joint_accel_sigma",
        "--joint_velocity_smoothness_sigma",
        type=float,
        default=None,
        help="σ for joint-velocity smoothness reward term. G1 locomotion / standing IK.",
    )
    parser.add_argument(
        "--joint_velocity_roughness_huber_delta_rad_s",
        type=float,
        default=None,
        help="Per-DOF Huber threshold on Δq̇ (rad/s per step) for joint_velocity rough/smooth pair. 0 = pure Σ(Δq̇)². "
        "Default from env cfg (~0.2); larger δ → more permissive on sharp velocity spikes.",
    )
parser.add_argument(
    "--npz_joint_position_tolerance_deg",
    type=float,
    default=None,
    help="Tolerance in degrees for NPZ joint position success metric. Default: 10.0 degrees.",
)
# Granular joint pair reward weights (left/right pairs and torso)
parser.add_argument(
    "--joint_position_reward_weight_left_hip",
    type=float,
    default=None,
    help="Reward weight for left hip joints (yaw, roll, pitch). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_right_hip",
    type=float,
    default=None,
    help="Reward weight for right hip joints (yaw, roll, pitch). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_left_knee",
    type=float,
    default=None,
    help="Reward weight for left knee joint. 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_right_knee",
    type=float,
    default=None,
    help="Reward weight for right knee joint. 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_left_ankle",
    type=float,
    default=None,
    help="Reward weight for left ankle joints (pitch, roll). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_right_ankle",
    type=float,
    default=None,
    help="Reward weight for right ankle joints (pitch, roll). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_torso",
    type=float,
    default=None,
    help="Reward weight for torso joint. 0.0 = use fallback weight.",
)
parser.add_argument(
    "--joint_position_reward_weight_left_shoulder",
    type=float,
    default=None,
    help="Reward weight for left shoulder joints (pitch, roll, yaw). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_right_shoulder",
    type=float,
    default=None,
    help="Reward weight for right shoulder joints (pitch, roll, yaw). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_left_elbow",
    type=float,
    default=None,
    help="Reward weight for left elbow joints (pitch, roll). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_right_elbow",
    type=float,
    default=None,
    help="Reward weight for right elbow joints (pitch, roll). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_left_hand",
    type=float,
    default=None,
    help="Reward weight for left hand joints (five, six, three). 0.0 = use group weight or fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_right_hand",
    type=float,
    default=None,
    help="Reward weight for right hand joints (five, six, three). 0.0 = use group weight or fallback.",
)
# Joint type pair reward weights (combines left+right of same joint type, highest priority)
parser.add_argument(
    "--joint_position_reward_weight_hip_pitch",
    type=float,
    default=None,
    help="Reward weight for both hip_pitch joints (left+right together). 0.0 = use pair/group/fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_hip_roll",
    type=float,
    default=None,
    help="Reward weight for both hip_roll joints (left+right together). 0.0 = use pair/group/fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_hip_yaw",
    type=float,
    default=None,
    help="Reward weight for both hip_yaw joints (left+right together). 0.0 = use pair/group/fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_knee",
    type=float,
    default=None,
    help="Reward weight for both knee joints (left+right together). 0.0 = use pair/group/fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_ankle_pitch",
    type=float,
    default=None,
    help="Reward weight for both ankle_pitch joints (left+right together). 0.0 = use pair/group/fallback.",
)
parser.add_argument(
    "--joint_position_reward_weight_ankle_roll",
    type=float,
    default=None,
    help="Reward weight for both ankle_roll joints (left+right together). 0.0 = use pair/group/fallback.",
)
# Ion VTOL vertical takeoff task arguments
parser.add_argument(
    "--target_takeoff_speed",
    type=float,
    default=None,
    help="Target upward speed (m/s). Upward reward peaks when vz equals this (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--target_velocity_tolerance",
    type=float,
    default=None,
    help="Only reward when |vz - target| <= this (m/s). No reward outside band (ion_vertical_takeoff). Default: 0.5.",
)
parser.add_argument(
    "--positive_z_velocity_reward_weight",
    type=float,
    default=None,
    help="Weight for upward velocity proximity reward: weight * exp(-sharpness*|vz-target|) when vz>0 (ion_vertical_takeoff). Default: 2.0.",
)
parser.add_argument(
    "--z_velocity_sharpness",
    type=float,
    default=None,
    help="Proximity falloff: reward = exp(-sharpness*|vz-target|). Higher = reward only when vz very close to target (ion_vertical_takeoff). Default: 2.0.",
)
parser.add_argument(
    "--upward_velocity_shaping_weight",
    type=float,
    default=None,
    help="Shaping: reward = weight*vz*dt for 0<vz<vz_low to encourage climbing toward band; 0=disabled (ion_vertical_takeoff). Default: 0.",
)
parser.add_argument(
    "--thrust_output_bias",
    type=float,
    default=None,
    help="Thrust mapping: policy output maps to thrust fraction [bias, 1]. Higher = default more thrust (ion_vertical_takeoff). Default: 0.4.",
)
parser.add_argument(
    "--idle_velocity_penalty_weight",
    type=float,
    default=None,
    help="Per-step penalty when |vz| < idle_velocity_threshold (ion_vertical_takeoff). Default: 2.0.",
)
parser.add_argument(
    "--idle_velocity_threshold",
    type=float,
    default=None,
    help="|vz| below this (m/s) counts as idle for idle penalty (ion_vertical_takeoff). Default: 0.05.",
)
parser.add_argument(
    "--x_drift_penalty",
    type=float,
    default=None,
    help="Penalty scale for X-axis drift (ion_vertical_takeoff).",
)
parser.add_argument(
    "--y_drift_penalty",
    type=float,
    default=None,
    help="Penalty scale for Y-axis drift (ion_vertical_takeoff).",
)
parser.add_argument(
    "--jerky_penalty",
    type=float,
    default=None,
    help="Penalty scale for jerky/flickering motion (ion_vertical_takeoff).",
)
parser.add_argument(
    "--orientation_reward_weight",
    type=float,
    default=None,
    help="Reward weight for maintaining upright orientation (ion_vertical_takeoff).",
)
parser.add_argument(
    "--orientation_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight vs spawn orientation (Isaac-Ion-Vertical-Takeoff-Direct-v0). V1 uses xy/z split instead.",
)
parser.add_argument(
    "--orientation_penalty_sharpness",
    type=float,
    default=None,
    help="Orientation penalty deviation^sharpness (ion_vertical_takeoff Direct). Default from env cfg.",
)
parser.add_argument(
    "--orientation_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max orientation penalty per step (ion_vertical_takeoff Direct). Default from env cfg.",
)
parser.add_argument(
    "--xy_orientation_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for tilt (roll/pitch) deviation from upright (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--xy_orientation_penalty_sharpness",
    type=float,
    default=None,
    help="Sharpness of XY orientation penalty: deviation^sharpness (ion_vertical_takeoff). Default: 2.0.",
)
parser.add_argument(
    "--xy_orientation_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max XY orientation penalty magnitude per step (ion_vertical_takeoff). Default: 0.06.",
)
parser.add_argument(
    "--z_orientation_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for yaw deviation from spawn (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--z_orientation_reward_weight",
    type=float,
    default=None,
    help="G1 walking: reward weight * exp(-|yaw_err| / sharpness) toward target_z_orientation. Default: 0.",
)
parser.add_argument(
    "--z_orientation_reward_sharpness",
    type=float,
    default=None,
    help="G1 walking: rad scale for z_orientation_reward exp kernel. Default: 0.2.",
)
parser.add_argument(
    "--alternating_foot_reward_weight",
    type=float,
    default=None,
    help="Reward when foot contact event alternates (L after R, R after L). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--same_foot_tap_penalty_weight",
    type=float,
    default=None,
    help="Penalty when same foot does brief tap (F→T→F) while other stays in contact. G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--jump_penalty_weight",
    type=float,
    default=None,
    help="Penalty when both feet off. G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--at_least_one_foot_contact_reward_weight",
    "--foot_contact_rew",
    type=float,
    default=None,
    dest="at_least_one_foot_contact_reward_weight",
    help="Reward per step when ≥1 foot has contact (pairs with jump_penalty_weight). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--ankle_roll_feet_target_separation_m",
    type=float,
    default=None,
    help="Target 3D distance (m) between left_ankle_roll_link and right_ankle_roll_link. "
    "Reward in (target±tol) band; penalty outside. G1 walking/V4. Default: 0.25 (env cfg).",
)
parser.add_argument(
    "--ankle_roll_feet_separation_tolerance_m",
    type=float,
    default=None,
    help="Half-width (m) of acceptable foot separation band around target (reward inside; penalty outside). "
    "G1 walking/V4. Default: 0.07 (env cfg).",
)
parser.add_argument(
    "--ankle_roll_feet_separation_reward_weight",
    type=float,
    default=None,
    help="Weight × triangular kernel in [target-tol, target+tol] (peak 1 at target). G1 walking/V4. Default: 0.0.",
)
parser.add_argument(
    "--ankle_roll_feet_separation_penalty_weight",
    type=float,
    default=None,
    help="Weight × (excess too-wide + excess too-narrow) outside target±tol (m). G1 walking/V4. Default: 0.0.",
)
parser.add_argument(
    "--ankle_roll_foot_height_reward_target_m",
    type=float,
    default=None,
    help="World z (m); per-foot reward weight × (target - z)+ for each ankle_roll_link when z < target. "
    "Default 0.11 (env cfg); minimal foot height often ~0.105.",
)
parser.add_argument(
    "--ankle_roll_foot_height_reward_weight",
    type=float,
    default=None,
    help="Per-foot linear margin reward on world z of L/R ankle_roll_link (see --ankle_roll_foot_height_reward_target_m). Default: 0.0.",
)
parser.add_argument(
    "--symmetry_reward_weight",
    type=float,
    default=None,
    help="Reward for L-R symmetry (contact ratio, touchdown ratio, torque magnitude). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--symmetry_penalty_weight",
    type=float,
    default=None,
    help="Penalty for L-R asymmetry (same logic as symmetry reward). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--symmetry_contact_sharpness",
    type=float,
    default=None,
    help="Exp scale for contact-ratio symmetry (0.5=perfect). G1 walking. Default: 0.2.",
)
parser.add_argument(
    "--symmetry_touchdown_sharpness",
    type=float,
    default=None,
    help="Exp scale for touchdown-ratio symmetry. G1 walking. Default: 0.2.",
)
parser.add_argument(
    "--symmetry_torque_sharpness",
    type=float,
    default=None,
    help="Nm; exp scale for ankle+hip torque symmetry (pitch L≈R, roll/yaw L≈-R). G1 walking. Default: 5.0.",
)
parser.add_argument(
    "--symmetry_include_hip_torque",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="If True, symmetry term_torque includes L/R hip applied torques (roll,pitch,yaw) with ankles. G1 walking.",
)
parser.add_argument(
    "--step_length_min_m",
    type=float,
    default=None,
    help="Min forward step length (m) to reward; penalty when step x < this. G1 walking. Default: 0.15.",
)
parser.add_argument(
    "--step_length_reward_weight",
    type=float,
    default=None,
    help="Reward when foot x displacement (liftoff→touchdown) >= step_length_min_m. G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--step_length_penalty_weight",
    type=float,
    default=None,
    help="Penalty when foot x displacement < step_length_min_m (linear in shortfall). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--foot_crossing_reward_weight",
    type=float,
    default=None,
    help="Reward when stepping foot crosses from behind to ahead of other (rel x neg→pos). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--foot_crossing_penalty_weight",
    type=float,
    default=None,
    help="Penalty when valid step but foot did not cross (rel x neg→pos). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--same_foot_tap_min_air_steps",
    type=int,
    default=None,
    help="Same-foot tap: require N decimated swing steps before F→T counts (0=legacy). G1 walking. Default: 0.",
)
parser.add_argument(
    "--symmetry_use_geometric_mean",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Symmetry: use (t_contact·t_td·t_torque)^(1/3) for reward and 1-geom for penalty. Default: False.",
)
parser.add_argument(
    "--step_length_use_smooth_reward",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Step length: exp(-shortfall/sigma) reward vs binary at min. G1 walking. Default: False.",
)
parser.add_argument(
    "--step_length_smooth_sigma_m",
    type=float,
    default=None,
    help="Step-length smooth reward exp scale (m). G1 walking. Default: 0.08.",
)
parser.add_argument(
    "--step_length_penalty_squared",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Step-length penalty uses shortfall² instead of linear. G1 walking. Default: False.",
)
parser.add_argument(
    "--gait_single_stance_reward_weight",
    type=float,
    default=None,
    help="Bonus when exactly one foot in contact (XOR). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--hip_weight_shift_reward_weight",
    type=float,
    default=None,
    help="Single-stance: reward when sum(|hip τ|) is larger on the stance leg (tanh vs scale). G1 walking. Default: 0.0.",
)
parser.add_argument(
    "--hip_weight_shift_torque_scale",
    type=float,
    default=None,
    help="Nm scale for hip weight-shift tanh (larger = softer). G1 walking. Default: 40.0.",
)
parser.add_argument(
    "--alternating_foot_scale_with_forward_vel",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Scale alternating-foot reward by clip(v_x/ref,0,1.5). G1 walking. Default: False.",
)
parser.add_argument(
    "--alternating_foot_vel_scale_ref",
    type=float,
    default=None,
    help="Reference forward speed (m/s) for alternating_foot_scale_with_forward_vel. Default: 0.5.",
)
parser.add_argument(
    "--lin_vel_z_penalty_weight",
    type=float,
    default=None,
    help="L2 penalty weight on base vertical velocity (Isaac-Lab lin_vel_z_l2). G1 Walking V2. Typical -0.2 to -2.0.",
)
parser.add_argument(
    "--lin_vel_z_tolerance",
    type=float,
    default=None,
    help="No lin_vel_z penalty when |v_z| <= this (m/s). G1 Walking V2. E.g. 0.05.",
)
parser.add_argument(
    "--ang_vel_xy_penalty_weight",
    type=float,
    default=None,
    help="L2 penalty weight on roll/pitch angular velocity (Isaac-Lab ang_vel_xy_l2). G1 Walking V2. Typical -0.05.",
)
parser.add_argument(
    "--flat_orientation_penalty_weight",
    type=float,
    default=None,
    help="L2 penalty weight on projected_gravity_b xy / tilt (Isaac-Lab flat_orientation_l2). G1 Walking V2. Typical 0 or -1.0.",
)
parser.add_argument(
    "--flat_orientation_tolerance",
    type=float,
    default=None,
    help="No flat_orientation penalty when grav_xy L2² <= tolerance². G1 Walking V2. E.g. 0.1 allows small tilt.",
)
parser.add_argument(
    "--z_orientation_penalty_sharpness",
    type=float,
    default=None,
    help="Sharpness of Z orientation penalty: deviation^sharpness (ion_vertical_takeoff). Default: 2.0.",
)
parser.add_argument(
    "--z_orientation_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max Z orientation penalty magnitude per step (ion_vertical_takeoff). Default: 0.06.",
)
parser.add_argument(
    "--electricity_cost_penalty",
    type=float,
    default=None,
    help="Penalty scale for electricity/power cost (ion_vertical_takeoff).",
)
parser.add_argument(
    "--idle_velocity_reward_weight",
    type=float,
    default=None,
    help="Reward when |vz| >= idle threshold (pairs with idle_velocity_penalty; ion_vertical_takeoff Direct). Default: 0.",
)
parser.add_argument(
    "--x_drift_free_reward_weight",
    type=float,
    default=None,
    help="Reward when |vx| within drift_tolerance (pairs with x_drift_penalty; ion_vertical_takeoff Direct). Default: 0.",
)
parser.add_argument(
    "--y_drift_free_reward_weight",
    type=float,
    default=None,
    help="Reward when |vy| within drift_tolerance (pairs with y_drift_penalty; ion_vertical_takeoff Direct). Default: 0.",
)
# --smooth_motion_reward_weight is registered with G1 locomotion / action-rate args (above); ion VTOL uses the same cfg field.
parser.add_argument(
    "--smooth_motion_reward_sharpness",
    type=float,
    default=None,
    help="Sharpness in exp(-sharpness * jerk_energy) for smooth_motion_reward (ion_vertical_takeoff Direct). Default: 0.01.",
)
parser.add_argument(
    "--electricity_efficiency_reward_weight",
    type=float,
    default=None,
    help="Exp reward on low thrust^2 (pairs with electricity_cost; ion_vertical_takeoff Direct). Default: 0.",
)
parser.add_argument(
    "--electricity_efficiency_sigma",
    type=float,
    default=None,
    help="Scale for exp(-sigma * sum(thrust^2)) in electricity_efficiency_reward (ion_vertical_takeoff Direct). Default: 1e-3.",
)
parser.add_argument(
    "--upward_velocity_shaping_miss_penalty_weight",
    type=float,
    default=None,
    help="Penalty for vz < 0 (pairs with upward_velocity_shaping; ion_vertical_takeoff Direct). Default: 0.",
)
parser.add_argument(
    "--low_altitude_penalty_threshold",
    type=float,
    default=None,
    help="Altitude threshold (m); penalize when z < this, reward when z > this (ion_vertical_takeoff). Default: 0.1.",
)
parser.add_argument(
    "--low_altitude_penalty_weight",
    type=float,
    default=None,
    help="Per-step penalty weight when z < low_altitude_penalty_threshold (ion_vertical_takeoff).",
)
parser.add_argument(
    "--low_altitude_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max low-altitude penalty magnitude per step; keeps returns bounded to avoid surrogate NaN (ion_vertical_takeoff). Default: 0.05.",
)
parser.add_argument(
    "--low_altitude_penalty_ramp",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    help="Use smooth ramp for low-altitude penalty (True) or binary (False). Default: True.",
)
parser.add_argument(
    "--airborne_reward_weight",
    type=float,
    default=None,
    help="Per-step reward when z > low_altitude_penalty_threshold to encourage takeoff (ion_vertical_takeoff). Default: 0.5.",
)
parser.add_argument(
    "--takeoff_height_shaping_weight",
    type=float,
    default=None,
    help="Weight * height AGL (capped) * dt to encourage leaving the ground (ion_vertical_takeoff Direct). Default: 0.",
)
parser.add_argument(
    "--takeoff_height_shaping_cap_m",
    type=float,
    default=None,
    help="Cap (m) on AGL in takeoff_height_shaping (ion_vertical_takeoff Direct). Default: 2.0.",
)
parser.add_argument(
    "--drift_free_requires_airborne",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    help="If true, x/y drift-free rewards only when z >= low_altitude_penalty_threshold (ion_vertical_takeoff Direct). Default: true.",
)
parser.add_argument(
    "--reward_scale",
    type=float,
    default=None,
    help="Multiply reward by this before return; <1 reduces value targets (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--spawn_height",
    type=float,
    default=None,
    help="Height (m) above terrain origin to spawn the robot (ion_vertical_takeoff v1 / Direct / V2). Default from env cfg.",
)
parser.add_argument(
    "--clamp_rewards",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    help="Ion vertical takeoff Direct: apply per-step penalty caps, takeoff height cap, and global reward_clip_range. Default: false.",
)
parser.add_argument(
    "--clamp_xy_drift_penalties",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    help="Ion vertical takeoff Direct: clamp x_drift and y_drift to [-drift_penalty_cap_per_step, 0] per step. Independent of --clamp_rewards. Default: false.",
)
parser.add_argument(
    "--clamp_off_band_z_velocity_penalty",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    help="Ion vertical takeoff Direct: clamp off-band vz penalty to [-off_band_z_velocity_penalty_cap_per_step, 0] per step. Independent of --clamp_rewards. Default: false.",
)
parser.add_argument(
    "--reward_clip_min",
    type=float,
    default=None,
    help="Lower bound for reward clipping per step (ion_vertical_takeoff). Default: -1.0.",
)
parser.add_argument(
    "--reward_clip_max",
    type=float,
    default=None,
    help="Upper bound for reward clipping per step (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help="Physics steps per policy step. Policy step duration = sim.dt * decimation (e.g. sim.dt=1/120, decimation=2 → 60 Hz).",
)
parser.add_argument(
    "--sim_dt",
    type=float,
    default=None,
    help="Override SimulationCfg.dt (seconds). G1 locomotion default is often 1/120. Combined with --decimation sets MDP step rate.",
)
parser.add_argument(
    "--staged_stability_start_env_steps",
    type=int,
    default=None,
    help="Ion V2 / horizontal cruise V2: common_step_counter steps before stability reward weights ramp (reward_profile=staged).",
)
parser.add_argument(
    "--staged_stability_ramp_env_steps",
    type=int,
    default=None,
    help="Ion V2 / horizontal cruise V2: env steps over which stability scale goes 0→1 after the start delay.",
)
parser.add_argument(
    "--target_vz_command_obs_enabled",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Ion vertical takeoff V2 / horizontal cruise V2: append normalized target_takeoff_speed / scale to policy observation.",
)
parser.add_argument(
    "--vz_command_obs_scale",
    type=float,
    default=None,
    help="Ion vertical takeoff V2 / horizontal cruise V2: scale divisor for target vz (target_takeoff_speed) obs channel.",
)
parser.add_argument(
    "--target_vx_command_obs_enabled",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Ion horizontal cruise V2: append normalized target_forward_speed / vx_command_obs_scale to policy observation.",
)
parser.add_argument(
    "--vx_command_obs_scale",
    type=float,
    default=None,
    help="Ion horizontal cruise V2: divide target_forward_speed by this for the forward command observation channel.",
)
# Ion horizontal cruise: z position and positive x velocity (forward) rewards
parser.add_argument(
    "--z_position_reward_weight",
    type=float,
    default=None,
    help="Weight for altitude hold reward (AGL vs target; ion vertical takeoff Direct/V2, ion_horizontal_cruise). Default from env cfg.",
)
parser.add_argument(
    "--z_position_target",
    type=float,
    default=None,
    help="Target altitude AGL (m) for z hold reward; match spawn_height to stay at spawn (ion vertical takeoff Direct/V2, ion_horizontal_cruise). Default from env cfg.",
)
parser.add_argument(
    "--z_position_sharpness",
    type=float,
    default=None,
    help="Sharpness for z position reward: exp(-sharpness * |AGL - target|) (ion vertical takeoff Direct/V2, ion_horizontal_cruise). Default from env cfg.",
)
parser.add_argument(
    "--xy_position_reward_weight",
    type=float,
    default=None,
    help="Exp reward for root XY near env origin (ion vertical takeoff Direct/V2). Default: 0.",
)
parser.add_argument(
    "--xy_position_reward_sharpness",
    type=float,
    default=None,
    help="Sharpness for xy position exp reward vs planar distance (m). Default from env cfg.",
)
parser.add_argument(
    "--xy_position_penalty_weight",
    type=float,
    default=None,
    help="Penalty when planar distance from env origin exceeds tolerance (ion vertical takeoff Direct/V2). Default: 0.",
)
parser.add_argument(
    "--xy_position_penalty_tolerance",
    type=float,
    default=None,
    help="Planar distance (m) from env origin with no XY position penalty. Default from env cfg.",
)
parser.add_argument(
    "--xy_position_penalty_sharpness",
    type=float,
    default=None,
    help="Exponent on (distance - tolerance) for XY position penalty. Default from env cfg.",
)
parser.add_argument(
    "--xy_position_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max XY position penalty magnitude per step when clamp_rewards. Default from env cfg.",
)
parser.add_argument(
    "--target_forward_speed",
    type=float,
    default=None,
    help="Target forward speed vx (m/s). Forward reward peaks when vx equals this (ion_horizontal_cruise). Default: 5.0.",
)
parser.add_argument(
    "--target_x_velocity_tolerance",
    type=float,
    default=None,
    help="Only reward when |vx - target| <= this (m/s) (ion_horizontal_cruise). Default: 0.5.",
)
parser.add_argument(
    "--positive_x_velocity_reward_weight",
    type=float,
    default=None,
    help="Weight for forward velocity proximity reward when vx > 0 in band (ion_horizontal_cruise). Default: 2.0.",
)
parser.add_argument(
    "--x_velocity_sharpness",
    type=float,
    default=None,
    help="Proximity falloff for x velocity reward: exp(-sharpness * |vx - target|) (ion_horizontal_cruise). Default: 2.0.",
)
parser.add_argument(
    "--forward_velocity_shaping_weight",
    type=float,
    default=None,
    help="Shaping: reward = weight*vx*dt for 0 < vx < vx_low to encourage accelerating toward band; 0 = disabled (ion_horizontal_cruise). Default: 0.",
)
parser.add_argument(
    "--off_band_x_velocity_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for vx outside rewarded band [target±tolerance] (ion_horizontal_cruise). 0 = disabled. Default: 0.",
)
parser.add_argument(
    "--off_band_x_velocity_penalty_sharpness",
    type=float,
    default=None,
    help="Exponent for off-band x velocity penalty (ion_horizontal_cruise). Default: 1.0.",
)
parser.add_argument(
    "--reward_log_interval",
    type=int,
    default=200,
    help="Log reward debug every N iterations. Higher = less I/O, faster. Default: 200.",
)
parser.add_argument(
    "--progress_log_interval",
    type=int,
    default=100,
    help="Append full iteration block to log_dir/training_progress_log.txt every N iterations. Default: 100.",
)
# G1 locomotion V1 (and other direct locomotion envs with same cfg attributes)
parser.add_argument(
    "--up_weight",
    type=float,
    default=None,
    help="Weight for upright reward (G1 locomotion, etc.). Default env-dependent.",
)
parser.add_argument(
    "--heading_weight",
    type=float,
    default=None,
    help="Weight for heading reward (G1 locomotion, etc.). Default env-dependent.",
)
parser.add_argument(
    "--actions_cost_scale",
    type=float,
    default=None,
    help="Scale for action cost penalty (G1 locomotion, etc.).",
)
parser.add_argument(
    "--energy_cost_scale",
    type=float,
    default=None,
    help="Scale for energy/electricity cost penalty (G1 locomotion, etc.).",
)
parser.add_argument(
    "--alive_reward_scale",
    type=float,
    default=None,
    help="Scale for alive reward per step (G1 locomotion, etc.).",
)
parser.add_argument(
    "--progress_reward_multiplier",
    type=float,
    default=None,
    help="Multiplier for progress reward (0=standing still, 1=walk forward). G1 locomotion V1.",
)
parser.add_argument(
    "--progress_reward_world_vx_blend",
    type=float,
    default=None,
    help="Adds mult*blend*max(vx,0)*step_dt to progress (world +x), on top of potential-based progress. G1 locomotion.",
)
parser.add_argument(
    "--progress_reward_forward_displacement_weight",
    type=float,
    default=None,
    help="Adds mult*weight*max(0, Δroot_x-threshold) per env step to progress (world +x). G1 locomotion.",
)
parser.add_argument(
    "--progress_reward_forward_displacement_threshold_m",
    type=float,
    default=None,
    help="Meters of forward root displacement per step below which displacement progress is zero. G1 locomotion.",
)
parser.add_argument(
    "--progress_forward_small",
    action="store_true",
    help="Preset small forward signal: sets mult/blend/displacement defaults if those CLI args were omitted.",
)
parser.add_argument(
    "--target_velocity",
    type=float,
    default=None,
    help="Target forward (+x) velocity in m/s (G1 locomotion V1 walking).",
)
parser.add_argument(
    "--velocity_reward_weight",
    type=float,
    default=None,
    help="Reward weight for maintaining target +x velocity (G1 locomotion V1 walking / Walking V2).",
)
parser.add_argument(
    "--velocity_reward_sharpness",
    type=float,
    default=None,
    help="Sharpness (m/s) for velocity reward exp(-|v_x - target| / sharpness). G1 walking / Walking V2. Default: 0.5.",
)
parser.add_argument(
    "--target_x_velocity_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for squared mismatch from target +x velocity (v_x - target)^2. G1 Locomotion V3. Default: 0.0.",
)
parser.add_argument(
    "--termination_height",
    type=float,
    default=None,
    help="Torso height (m) below which episode ends (fall). G1 locomotion. Default 0.6. Use 0.5 to reduce early termination.",
)
parser.add_argument(
    "--early_termination_time_penalty_weight",
    type=float,
    default=None,
    help="G1 Locomotion V4: on fall (not time-out), subtract weight×(1−episode_len/max_episode_len). 0=off.",
)
# G1 Locomotion V3 (base height, curriculum, push perturbation)
parser.add_argument(
    "--base_height_reference",
    type=float,
    default=None,
    help="Reference base height (m) for base_height reward. G1 Locomotion V3. Default: 0.95.",
)
parser.add_argument(
    "--base_height_reward_weight",
    type=float,
    default=None,
    help="Weight for base height reward (exp kernel). G1 Locomotion V3. 0 = off. Default: 0.0.",
)
parser.add_argument(
    "--base_height_sigma",
    type=float,
    default=None,
    help="Sigma for base height reward exp(-(z-ref)^2/sigma^2). G1 Locomotion V3. Default: 0.08.",
)
parser.add_argument(
    "--base_height_penalty_weight",
    type=float,
    default=None,
    help="G1 Locomotion V3/V4: penalty weight × max(0, |root_z − base_height_reference| − tolerance_m). 0 = off.",
)
parser.add_argument(
    "--base_height_penalty_tolerance_m",
    type=float,
    default=None,
    help="G1 Locomotion V3/V4: no base-height penalty when |z − ref| ≤ this (m). Default: 0.0.",
)
parser.add_argument(
    "--curriculum_stage1_steps",
    type=int,
    default=None,
    help="Curriculum stage1 end (env steps); velocity reward = 0 before this. G1 Locomotion V3. Default: 5000.",
)
parser.add_argument(
    "--curriculum_stage2_steps",
    type=int,
    default=None,
    help="Curriculum stage2 end (env steps); velocity reward ramps 0->1 between stage1 and stage2. G1 Locomotion V3. Default: 15000.",
)
parser.add_argument(
    "--num_steps_per_env",
    type=int,
    default=None,
    help="Steps per env per iteration (for curriculum step counting). G1 Locomotion V3. Should match PPO num_steps_per_env. Default: 24.",
)
parser.add_argument(
    "--enable_curriculum",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Enable curriculum (velocity reward 0 -> ramp -> 1). G1 Locomotion V3. Default: False.",
)
parser.add_argument(
    "--enable_push_perturbation",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Apply random xy velocity impulse at reset. G1 Locomotion V3. Default: False.",
)
parser.add_argument(
    "--max_push_vel_xy",
    type=float,
    default=None,
    help="Max absolute xy velocity impulse (m/s) at reset when enable_push_perturbation. G1 Locomotion V3. Default: 0.3.",
)
parser.add_argument(
    "--episode_start_force_impulse_enabled",
    type=lambda x: (str(x).lower() in ("true", "1", "yes", "on")),
    default=None,
    metavar="True|False",
    help="Apply a short random external-force impulse on the root body at the beginning of each episode. G1 Locomotion V4. Default: False.",
)
parser.add_argument(
    "--episode_start_force_impulse_duration_steps",
    type=int,
    default=None,
    help="Duration of the episode-start external-force impulse in policy steps. G1 Locomotion V4. Default: 0.",
)
parser.add_argument(
    "--episode_start_force_impulse_max_force_xy_n",
    type=float,
    default=None,
    help="Max absolute world-frame xy force (N) for the episode-start external-force impulse. G1 Locomotion V4. Default: 0.0.",
)
parser.add_argument(
    "--episode_start_force_impulse_max_force_z_n",
    type=float,
    default=None,
    help="Max absolute world-frame z force (N) for the episode-start external-force impulse. G1 Locomotion V4. Default: 0.0.",
)
parser.add_argument(
    "--hand_target_pos_b",
    type=str,
    default=None,
    help="Hand target in pelvis frame: 'x,y,z' meters (e.g. '0.12,-0.2,-0.08'). "
    "Used for hand-position reward on standing tasks; with G1 V4 --append_teleop_hand_target_obs True, also "
    "the constant last 3 observation channels (teleop alignment).",
)
parser.add_argument(
    "--hand_reward_weight",
    type=float,
    default=None,
    help="Weight for right-hand position reward (hand at hand_target_pos_b in pelvis frame). 0 = disabled. G1 locomotion V1 standing.",
)
parser.add_argument(
    "--hand_reward_sigma",
    type=float,
    default=None,
    help="Sigma for hand position reward exp(-squared_error/sigma). Smaller = sharper peak. G1 locomotion V1 standing. Default: 0.01.",
)
parser.add_argument(
    "--xy_drift_tolerance",
    type=float,
    default=None,
    help="XY drift tolerance (m); no penalty when |x - origin_x| and |y - origin_y| <= this (G1 standing hand position V1). Default: 0.1.",
)
parser.add_argument(
    "--xy_drift_penalty_weight",
    type=float,
    default=None,
    help="Weight for XY drift penalty when beyond xy_drift_tolerance (G1 standing hand position V1). Default: 0.0.",
)
parser.add_argument(
    "--x_drift_tolerance",
    type=float,
    default=None,
    help="X drift tolerance (m); no penalty when |x - origin_x| <= this (G1 standing hand position V2). Default: 0.1.",
)
parser.add_argument(
    "--x_drift_penalty_weight",
    type=float,
    default=None,
    help="Weight for X-axis drift penalty when |x - origin_x| > x_drift_tolerance (G1 standing hand position V2). Default: 0.0.",
)
parser.add_argument(
    "--y_drift_tolerance",
    type=float,
    default=None,
    help="Y drift tolerance (m); no penalty when |y - origin_y| <= this (G1 standing hand position V2). Default: 0.1.",
)
parser.add_argument(
    "--y_drift_penalty_weight",
    type=float,
    default=None,
    help="Weight for Y-axis drift penalty (G1 standing hand position V2) or y drift (G1 locomotion V1 walking). Default: 0.0.",
)
parser.add_argument(
    "--x_drift_reward_weight",
    type=float,
    default=None,
    help="Weight for exp reward keeping root x near env origin (G1 locomotion / standing IK). 0 = off. σ: --x_drift_reward_sigma.",
)
parser.add_argument(
    "--y_drift_reward_weight",
    type=float,
    default=None,
    help="Weight for exp reward keeping root y near env origin (g1_standing_ik). 0 = off.",
)
parser.add_argument(
    "--x_drift_reward_sigma",
    type=float,
    default=None,
    help="σ (m) for x drift exp reward; 0 = use x_drift_tolerance (G1 locomotion / standing IK).",
)
parser.add_argument(
    "--y_drift_reward_sigma",
    type=float,
    default=None,
    help="σ (m) for y drift exp reward; 0 = use y_drift_tolerance (g1_standing_ik).",
)
parser.add_argument(
    "--ik_target_motion_speed",
    type=float,
    default=None,
    help="IK target motion speed: higher = faster along waypoint path (G1 standing hand position V2). Default: 1.0.",
)
parser.add_argument(
    "--ik_target_motion_period_base",
    type=float,
    default=None,
    help="Base period (s) for one full waypoint cycle when speed=1.0 (G1 standing hand position V2). Default: 8.0.",
)
parser.add_argument(
    "--ik_joint_reward_weight",
    type=float,
    default=None,
    help="Weight for IK joint-following reward (G1 standing hand position V2). Default: 0.5.",
)
parser.add_argument(
    "--ik_joint_reward_sigma",
    type=float,
    default=None,
    help="Sigma for IK joint reward exp(-sq_err/sigma) (G1 standing hand position V2). Default: 0.01.",
)
parser.add_argument(
    "--ik_joint_penalty_weight",
    type=float,
    default=None,
    help="Weight for penalty when joints deviate from IK solution (G1 standing hand position V2). Default: 0.2.",
)
parser.add_argument(
    "--ik_joint_penalty_threshold",
    type=float,
    default=None,
    help="No IK joint penalty when joint error (rad) below this (G1 standing hand position V2). Default: 0.05.",
)
parser.add_argument(
    "--ik_joint_tolerance_percent",
    type=float,
    default=None,
    help="Joint position tolerance as percent of joint range for g1_standing_ik (within-tolerance reward / outside penalty). Default: 5.0.",
)
parser.add_argument(
    "--ee_position_reward_weight",
    type=float,
    default=None,
    help="Weight for end-effector-at-target reward (g1_standing_ik). Default: 1.0.",
)
parser.add_argument(
    "--ee_position_penalty_weight",
    type=float,
    default=None,
    help="Weight for end-effector-not-at-target penalty (g1_standing_ik). Default: 0.5.",
)
parser.add_argument(
    "--ee_position_reward_sigma",
    type=float,
    default=None,
    help="Sigma for EE position reward exp(-sq_err/sigma) (g1_standing_ik). Default from env cfg.",
)
parser.add_argument(
    "--ik_ee_smoothness_penalty_weight",
    type=float,
    default=None,
    help="Penalty on pelvis-frame hand motion jerk ||Δee_pos_b/step_dt||² (g1_standing_ik). Default from env cfg.",
)
parser.add_argument(
    "--ik_target_obs_scale",
    type=float,
    default=None,
    help="Scale for IK target position in policy observation (g1_standing_ik). Default from env cfg.",
)
parser.add_argument(
    "--ik_target_box_min",
    type=str,
    default=None,
    help="Random IK target box minimum in pelvis frame 'x,y,z' meters (g1_standing_ik). Default from env cfg.",
)
parser.add_argument(
    "--ik_target_box_max",
    type=str,
    default=None,
    help="Random IK target box maximum in pelvis frame 'x,y,z' meters (g1_standing_ik). Default from env cfg.",
)
parser.add_argument(
    "--g1_standing_fixed_base",
    action="store_true",
    help="Spawn G1 with fix_root_link=True (fixed, non-moveable base; Isaac-G1-Standing-IK-Direct-v0).",
)
parser.add_argument(
    "--teleop_target_obs_scale",
    type=float,
    default=None,
    help="Scale for teleop target in observation (G1 teleoperation). Default: 1.0.",
)
parser.add_argument(
    "--target_z_orientation",
    type=float,
    default=None,
    help="Target yaw in rad (G1 locomotion V1 walking). Default: 0.0.",
)
parser.add_argument(
    "--z_orientation_tolerance",
    type=float,
    default=None,
    help="No z-orientation penalty when |yaw - target_z_orientation| <= this in rad (G1 locomotion V1 walking). Default: 0.2.",
)
parser.add_argument(
    "--drift_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max drift penalty magnitude per step per axis (ion_vertical_takeoff). Default: 0.1.",
)
parser.add_argument(
    "--drift_tolerance",
    type=float,
    default=None,
    help="No X/Y drift penalty when |vx|,|vy| <= this (m/s). Default: 0.05.",
)
parser.add_argument(
    "--x_drift_penalty_sharpness",
    type=float,
    default=None,
    help="X drift penalty exponent: weight * (effective_|vx|)^sharpness (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--y_drift_penalty_sharpness",
    type=float,
    default=None,
    help="Y drift penalty exponent: weight * (effective_|vy|)^sharpness (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--off_band_z_velocity_penalty_weight",
    type=float,
    default=None,
    help="Penalty weight for z velocity outside rewarded band [target±tolerance] (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--off_band_z_velocity_penalty_sharpness",
    type=float,
    default=None,
    help="Exponent for off-band z velocity penalty: distance below/above band^sharpness (ion_vertical_takeoff). Default: 1.0.",
)
parser.add_argument(
    "--off_band_z_velocity_penalty_cap_per_step",
    type=float,
    default=None,
    help="Max off-band z velocity penalty magnitude per step; avoids total reward clip saturation (ion_vertical_takeoff). Default: 0.5.",
)
# PPO hyperparameter overrides (apply to any task using PPO)
parser.add_argument(
    "--ppo_learning_rate",
    type=float,
    default=None,
    help="PPO learning rate (e.g., 2e-5).",
)
parser.add_argument(
    "--ppo_entropy_coef",
    type=float,
    default=None,
    help="PPO entropy coefficient (higher = more exploration).",
)
parser.add_argument(
    "--ppo_entropy_coef_final",
    type=float,
    default=None,
    help=(
        "Isaac-Rover-DiskWorld-Localization only: if set, linearly anneal PPO entropy_coef from "
        "--ppo_entropy_coef (or agent cfg) to this value over max_iterations (each PPO update)."
    ),
)
parser.add_argument(
    "--ppo_value_loss_coef",
    type=float,
    default=None,
    help="PPO value loss coefficient.",
)
parser.add_argument(
    "--ppo_init_noise_std",
    type=float,
    default=None,
    help="Initial action noise std (policy exploration).",
)
parser.add_argument(
    "--ppo_clip_param",
    type=float,
    default=None,
    help="PPO clip parameter for policy updates.",
)
parser.add_argument(
    "--ppo_desired_kl",
    type=float,
    default=None,
    help="PPO desired KL divergence (adaptive LR target).",
)
parser.add_argument(
    "--divergence_mean_reward_min",
    type=float,
    default=None,
    help="DivergenceAwareOnPolicyRunner: stop if rolling mean episode return is below this (omit to use default -1e6).",
)
parser.add_argument(
    "--divergence_mean_reward_max",
    type=float,
    default=None,
    help="DivergenceAwareOnPolicyRunner: stop if rolling mean episode return exceeds this (omit to use default 1e6). Long multi-reward locomotion can exceed 1e6.",
)
parser.add_argument(
    "--ppo_gamma",
    type=float,
    default=None,
    help="PPO discount factor gamma.",
)
parser.add_argument(
    "--ppo_num_learning_epochs",
    type=int,
    default=None,
    help="PPO number of learning epochs per update.",
)
parser.add_argument(
    "--ppo_num_mini_batches",
    type=int,
    default=None,
    help="PPO number of mini-batches per update.",
)
# Exploration-focused overrides (tune toward more explorative behavior)
parser.add_argument(
    "--ppo_num_steps_per_env",
    type=int,
    default=None,
    help="Steps per env per update. Higher = more diverse rollouts, more exploration (e.g. 32–48).",
)
parser.add_argument(
    "--ppo_noise_std_type",
    type=str,
    default=None,
    choices=["scalar", "log"],
    help="Policy std type: 'log' keeps action std from collapsing too fast (more exploration).",
)
parser.add_argument(
    "--ppo_max_grad_norm",
    type=float,
    default=None,
    help="Max gradient norm. Slightly higher can allow larger policy updates (e.g. 0.5–1.0).",
)
parser.add_argument(
    "--ppo_lam",
    type=float,
    default=None,
    help="GAE lambda (0–1). Higher = lower variance, lower = more exploratory advantage estimate.",
)
# ---------------------------------------------------------------------------
# Rover disk-world task reward weight overrides
# (Isaac-Rover-DiskWorld-Forward-v0)
# ---------------------------------------------------------------------------
parser.add_argument(
    "--rover_forward_progress_weight",
    type=float,
    default=None,
    help="[Rover] Weight for forward_angular_progress reward (arc-length per step). Default 10.0.",
)
parser.add_argument(
    "--rover_speed_tracking_weight",
    type=float,
    default=None,
    help="[Rover] Weight for forward_speed_tracking reward (penalises speed < target). Default 0.5.",
)
parser.add_argument(
    "--rover_speed_target",
    type=float,
    default=None,
    help="[Rover] Target tangential speed (m/s) for speed_tracking reward. Default 1.0.",
)
parser.add_argument(
    "--rover_heading_weight",
    type=float,
    default=None,
    help="[Rover] Weight for heading_alignment reward (cos of heading error vs CCW tangent). Default 0.5.",
)
parser.add_argument(
    "--rover_action_rate_weight",
    type=float,
    default=None,
    help="[Rover] Weight for action_rate_penalty (smoothness regulariser). Default 0.05.",
)
parser.add_argument(
    "--rover_motion_jerk_weight",
    type=float,
    default=None,
    help="[Rover] Weight for motion_jerk_penalty (root lin/ang jerk; smoother body motion). Default 0.02.",
)
parser.add_argument(
    "--rover_heading_oscillation_weight",
    type=float,
    default=None,
    help="[Rover] Weight for heading_oscillation_penalty (reduces heading zig-zag). Default from task cfg if omitted.",
)
parser.add_argument(
    "--rover_motion_jerk_bonus_weight",
    type=float,
    default=None,
    help="[Rover] Weight for motion_jerk_smoothness_bonus (exp(-scale*jerk²); pairs with motion_jerk_penalty).",
)
parser.add_argument(
    "--rover_heading_stability_bonus_weight",
    type=float,
    default=None,
    help="[Rover] Weight for heading_stability_bonus (exp(-scale*Δheading²); pairs with heading_oscillation_penalty).",
)
parser.add_argument(
    "--rover_motion_jerk_bonus_exp_scale",
    type=float,
    default=None,
    help="[Rover] exp_scale param for motion_jerk_smoothness_bonus (larger → bonus drops faster with jerk).",
)
parser.add_argument(
    "--rover_heading_stability_bonus_exp_scale",
    type=float,
    default=None,
    help="[Rover] exp_scale param for heading_stability_bonus (larger → bonus drops faster with heading oscillation).",
)
parser.add_argument(
    "--rover_alive_weight",
    type=float,
    default=None,
    help="[Rover] Weight for alive_bonus (per-step survival reward). Default 0.1.",
)
parser.add_argument(
    "--rover_tilt_weight",
    type=float,
    default=None,
    help="[Rover] Weight for tilt_penalty (body-tilt from disk normal). Default 0.2.",
)
parser.add_argument(
    "--rover_radial_z_alignment_reward_weight",
    type=float,
    default=None,
    help=(
        "[Rover] Weight for radial_z_alignment_reward "
        "(body +Z · outward_radial; 1=aligned, -1=inward). Default 0.3."
    ),
)
parser.add_argument(
    "--rover_radial_z_alignment_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover] Weight for radial_z_alignment_penalty "
        "(-(1-dot)^2 quadratic misalignment cost). Default 0.2."
    ),
)
parser.add_argument(
    "--rover_goal_bonus_weight",
    type=float,
    default=None,
    help="[Rover] Weight for goal_bonus (one-shot reward on reaching 1 rad progress). Default 50.0.",
)
parser.add_argument(
    "--rover_goal_rad",
    type=float,
    default=None,
    help="[Rover] Cumulative CCW angular progress (rad) required for goal termination. Default 1.0.",
)
parser.add_argument(
    "--rover_fall_radius",
    type=float,
    default=None,
    help="[Rover] Radial distance threshold (m) below which the rover is considered fallen. Default 80.0.",
)
parser.add_argument(
    "--rover_episode_length_s",
    type=float,
    default=None,
    help="[Rover] Episode timeout duration in seconds. Default 30.0.",
)
parser.add_argument(
    "--rover_localization_visual_obstacle_max_envs",
    type=int,
    default=None,
    help=(
        "[Rover Localization] Spawn USD landmark meshes only for env indices [0, K) with "
        "K = min(num_envs, this) when this > 0. When <= 0, K = min(num_envs, "
        "visual_obstacle_usd_all_envs_cap) from env cfg (default 64), not num_envs, "
        "to avoid GPU saturation at large num_envs. Default from env cfg."
    ),
)
parser.add_argument(
    "--rover_localization_visual_usd_all_envs_cap",
    type=int,
    default=None,
    help=(
        "[Rover Localization] When visual_obstacle_max_spawned_envs <= 0, cap USD spawns "
        "to this many envs (default from env cfg). Ignored when visual_obstacle_max_spawned_envs > 0."
    ),
)
parser.add_argument(
    "--rover_localization_goal_collision",
    type=int,
    default=None,
    help="[Rover Localization] Goal collision toggle: 1=on, 0=off (default from env cfg).",
)
parser.add_argument(
    "--rover_localization_goal_asset_path",
    type=str,
    default=None,
    help="[Rover Localization] Optional goal mesh path override for visual obstacle spawning.",
)
parser.add_argument(
    "--rover_localization_box_collision",
    type=int,
    default=None,
    help="[Rover Localization] Blue-box collision toggle: 1=on, 0=off (default from env cfg).",
)
parser.add_argument(
    "--rover_localization_goal_count",
    type=int,
    default=None,
    help="[Rover Localization] Number of goal objects to spawn/use in landmark pool.",
)
parser.add_argument(
    "--rover_localization_box_count",
    type=int,
    default=None,
    help="[Rover Localization] Number of blue-box objects to spawn/use in landmark pool.",
)
parser.add_argument(
    "--rover_localization_train_debug",
    action="store_true",
    default=False,
    help=(
        "[Rover Localization] Print effective reward/PPO/env snapshots and append "
        "localization_train_debug_snap.txt under the run log_dir. Also enables "
        "localization_debug_metrics.csv (same as ROVER_LOCALIZATION_TRAIN_DEBUG=1)."
    ),
)
# ---------------------------------------------------------------------------
# Rover ObstacleAvoidance task (Isaac-Rover-DiskWorld-ObstacleAvoidance-v0)
# ---------------------------------------------------------------------------
parser.add_argument(
    "--rover_oa_forward_progress_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for forward_angular_progress reward (clockwise arc). Default 30.0.",
)
parser.add_argument(
    "--rover_oa_forward_speed_tracking_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight for forward_speed_tracking penalty "
        "(penalizes falling below target clockwise speed). Default 4.0."
    ),
)
parser.add_argument(
    "--rover_oa_forward_speed_target_ms",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Target clockwise tangential speed (m/s) for speed-tracking penalty. Default 1.5.",
)
parser.add_argument(
    "--rover_oa_blue_box_penalty_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for blue_box_avoidance_penalty (saturating Gaussian). Default 2.0.",
)
parser.add_argument(
    "--rover_oa_blue_box_danger_radius_m",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Sigma (m) of blue-box avoidance Gaussian falloff. Default 5.0.",
)
parser.add_argument(
    "--rover_oa_goal_pass_through_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for one-shot bonus per goal passed. Default 20.0.",
)
parser.add_argument(
    "--rover_oa_goal_pass_threshold_m",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] 3-D distance (m) at which the rover is considered to have "
        "passed the next-ahead goal. Default 2.0."
    ),
)
parser.add_argument(
    "--rover_oa_heading_alignment_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for disk_tangent_heading_alignment_reward. Default 1.0.",
)
parser.add_argument(
    "--rover_oa_heading_alignment_tolerance_rad",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] ± window (rad) for heading alignment with disk clockwise tangent. "
        "Default 1.0 rad."
    ),
)
parser.add_argument(
    "--rover_oa_radial_z_alignment_reward_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight for radial_z_alignment_reward "
        "(rover +Z aligned with vector disk_origin -> rover_origin). Default 3.0."
    ),
)
parser.add_argument(
    "--rover_oa_radial_z_alignment_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight for radial_z_alignment_penalty "
        "(misalignment from vector disk_origin -> rover_origin). Default 1.0."
    ),
)
parser.add_argument(
    "--rover_oa_upside_down_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Magnitude of termination penalty applied when "
        "upside_down termination triggers. Default 100.0."
    ),
)
parser.add_argument(
    "--rover_oa_fell_off_disk_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Magnitude of termination penalty applied when "
        "fell_off_disk termination triggers. Default 200.0."
    ),
)
parser.add_argument(
    "--rover_oa_upside_down_min_radial_dot",
    "--rover_oa_upside_down_min_dot_z",
    dest="rover_oa_upside_down_min_radial_dot",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Upside-down termination: body +Z in world frame, "
        "cos(tilt from world +Z) = dot(body_z_w, e_z). Episode ends when this falls "
        "below the threshold. Default 0.0 (past horizontal / inverted). Positive values "
        "are stricter (e.g. 0.15 ends if tilt exceeds ~81°). The legacy alias "
        "--rover_oa_upside_down_min_dot_z is also accepted."
    ),
)
parser.add_argument(
    "--rover_oa_action_rate_penalty_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for action_rate_penalty (smoothness). Default 0.04.",
)
parser.add_argument(
    "--rover_oa_motion_jerk_penalty_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for motion_jerk_penalty. Default 0.01.",
)
parser.add_argument(
    "--rover_oa_heading_oscillation_penalty_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for heading_oscillation_penalty. Default 0.05.",
)
parser.add_argument(
    "--rover_oa_alive_bonus_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for alive_bonus. Default 0.05.",
)
parser.add_argument(
    "--rover_oa_teacher_imitation_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Weight for teacher_action_imitation reward. Default 2.0.",
)
parser.add_argument(
    "--rover_oa_teacher_imitation_decay_env_steps",
    type=int,
    default=None,
    help="[Rover ObstacleAvoidance] Decay horizon (env steps) for teacher imitation shaping. Default 20000.",
)
parser.add_argument(
    "--rover_oa_teacher_imitation_match_sigma",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Sigma for student-vs-teacher action matching. Default 0.25.",
)
parser.add_argument(
    "--rover_oa_teacher_policy_enable",
    type=int,
    default=None,
    help="[Rover ObstacleAvoidance] Enable teacher controller bootstrap (1=yes, 0=no).",
)
parser.add_argument(
    "--rover_oa_teacher_policy_path",
    type=str,
    default=None,
    help="[Rover ObstacleAvoidance] Checkpoint path for teacher forward policy (final_model.pt).",
)
parser.add_argument(
    "--rover_oa_teacher_action_blend_weight",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Initial teacher action blend alpha: 0=student only, 1=teacher only. Default 0.",
)
parser.add_argument(
    "--rover_oa_teacher_action_blend_decay_env_steps",
    type=int,
    default=None,
    help="[Rover ObstacleAvoidance] Env-step horizon for teacher action blend to decay to zero. Default 20000.",
)
parser.add_argument(
    "--rover_oa_fall_min_radius_m",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Radial threshold (m) below which the episode terminates. Default 99.5.",
)
parser.add_argument(
    "--rover_oa_radial_z_penalty_max_abs_m",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Clip |z−z_nominal| to this before squaring for radial_z_alignment_penalty. "
        "Default 2.5."
    ),
)
parser.add_argument(
    "--rover_oa_near_fall_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight of dense edge-of-disk safety penalty "
        "(near_fall_radial_penalty). Provides a continuous gradient toward staying "
        "near the spawn radius rather than only the sparse fall-termination penalty. "
        "Default 8.0."
    ),
)
parser.add_argument(
    "--rover_oa_near_fall_danger_band_m",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Width (m) of the danger strip just outside fall "
        "radius for the dense edge-safety penalty. Default 1.5."
    ),
)
parser.add_argument(
    "--rover_oa_visual_box_ahead_penalty_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight of dense visual-channel blue-box-ahead "
        "penalty. Reads `_visual_object_tracks` so the camera output is on the "
        "policy gradient path. Default 4.0."
    ),
)
parser.add_argument(
    "--rover_oa_visual_box_ahead_bearing_gate_rad",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Sigma of bearing falloff for visual box-ahead penalty. Default 0.4.",
)
parser.add_argument(
    "--rover_oa_visual_box_ahead_range_gate_m",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Sigma of range falloff for visual box-ahead penalty. Default 6.0.",
)
parser.add_argument(
    "--rover_oa_visual_goal_ahead_reward_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight of dense visual-channel red-goal-ahead "
        "alignment reward. Default 2.0."
    ),
)
parser.add_argument(
    "--rover_oa_visual_goal_ahead_bearing_gate_rad",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Sigma of bearing falloff for visual goal-ahead reward. Default 0.4.",
)
parser.add_argument(
    "--rover_oa_visual_goal_ahead_range_gate_m",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Sigma of range falloff for visual goal-ahead reward. Default 12.0.",
)
parser.add_argument(
    "--rover_oa_goal_approach_shaping_weight",
    type=float,
    default=None,
    help=(
        "[Rover ObstacleAvoidance] Weight of dense next-goal-distance shaping reward "
        "(complements the sparse pass-through bonus). Default 1.5."
    ),
)
parser.add_argument(
    "--rover_oa_goal_approach_sigma_m",
    type=float,
    default=None,
    help="[Rover ObstacleAvoidance] Gaussian sigma (m) for goal-approach shaping. Default 5.0.",
)
parser.add_argument(
    "--rover_oa_objective_ramp_enable",
    type=int,
    default=0,
    choices=(0, 1),
    help=(
        "[Rover ObstacleAvoidance] If 1, linearly ramp blue/goal/visual/approach reward weights from "
        "--rover_oa_objective_ramp_start_scale to 1.0 over --rover_oa_objective_ramp_iters learning iterations "
        "inside one Isaac session (no chunk restarts)."
    ),
)
parser.add_argument(
    "--rover_oa_objective_ramp_iters",
    type=int,
    default=1200,
    help="[Rover ObstacleAvoidance] Learning-iteration horizon for the in-process objective ramp. Default 1200.",
)
parser.add_argument(
    "--rover_oa_objective_ramp_start_scale",
    type=float,
    default=0.0,
    help="[Rover ObstacleAvoidance] Starting multiplicative scale for objective terms (0–1). Default 0.0.",
)
parser.add_argument(
    "--rover_oa_objective_ramp_forward_boost",
    type=float,
    default=0.12,
    help=(
        "[Rover ObstacleAvoidance] While objectives are weak, scale forward_angular_progress weight by "
        "(1 + boost * (1 - objective_scale)). Default 0.12. Ignored when --rover_oa_objective_ramp_blend_mode=full_lerp."
    ),
)
parser.add_argument(
    "--rover_oa_objective_ramp_blend_mode",
    type=str,
    default="scale",
    choices=("scale", "full_lerp"),
    help=(
        "[Rover ObstacleAvoidance] When ramp_enable=1: 'scale' = legacy objective-only multiplier + forward_boost; "
        "'full_lerp' = linear blend of a fixed anchor preset → CLI-built weights over ramp_iters (G1 staged style)."
    ),
)
parser.add_argument(
    "--rover_oa_objective_ramp_anchor_preset",
    type=str,
    default="moderate",
    choices=("moderate", "recovery"),
    help=(
        "[Rover ObstacleAvoidance] Anchor weights at ramp start for blend_mode=full_lerp: "
        "'moderate' matches typical pre-ramp TRAIN.sh defaults; 'recovery' matches curriculum v2 phase-1."
    ),
)
parser.add_argument(
    "--rover_oa_init_random_episode_length",
    type=int,
    default=None,
    choices=(0, 1),
    help=(
        "[Rover ObstacleAvoidance] Force RSL-RL init_at_random_ep_len: 0 = full max episode length at reset, "
        "1 = random truncated starts. When unset and in-process objective ramp is enabled, defaults to 0 "
        "(reduces early-reset / timeout statistics collapse during ramp)."
    ),
)
# ---------------------------------------------------------------------------
# Rover FlatWorld-GoalNav task (Isaac-Rover-FlatWorld-GoalNav-v0)
# ---------------------------------------------------------------------------
parser.add_argument(
    "--rover_flat_distance_to_target_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for distance_to_target_reward (negative distance / scale). Default 1.0.",
)
parser.add_argument(
    "--rover_flat_distance_to_target_scale_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] scale_m parameter of distance_to_target_reward (m). Default 50.0.",
)
parser.add_argument(
    "--rover_flat_distance_progress_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal/OA-Flat] Weight for distance_progress_reward (+ when distance decreases). Default ~7–8.",
)
parser.add_argument(
    "--rover_flat_distance_increase_penalty_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for distance_increase_penalty (- when distance grows). Default 2.0.",
)
parser.add_argument(
    "--rover_flat_heading_to_target_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal/OA-Flat] Weight for heading_to_target_reward (cos heading; can reward spin without closure). Default 0.2 (flat goal) / 0.18 (OA-Flat).",
)
parser.add_argument(
    "--rover_flat_heading_goal_closure_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal/OA-Flat] Weight for heading_goal_closure_speed_reward (cos+ × positive v·goal/cap). Default 0.9 (flat goal) / 1.25 (OA-Flat).",
)
parser.add_argument(
    "--rover_flat_yaw_no_closure_penalty_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal/OA-Flat] Magnitude weight for yaw_without_goal_closure_penalty (penalize |yaw| when far and not closing). Default 0.14 on OA-Flat; 0 disables.",
)
parser.add_argument(
    "--rover_flat_forward_velocity_to_target_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for forward_velocity_to_target_reward. Default ~1.0 (flat goal) / 1.15 (OA-Flat).",
)
parser.add_argument(
    "--rover_flat_forward_velocity_to_target_cap_ms",
    type=float,
    default=None,
    help="[Rover FlatGoal] cap_ms for forward_velocity_to_target_reward (m/s). Default 3.0.",
)
parser.add_argument(
    "--rover_flat_close_to_target_bonus_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for close_to_target_bonus. Default 1.0.",
)
parser.add_argument(
    "--rover_flat_close_to_target_threshold_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] threshold_m for close_to_target_bonus (m). Default 5.0.",
)
parser.add_argument(
    "--rover_flat_goal_reached_bonus_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for goal_reached_bonus (one-shot success). Default 50.0.",
)
parser.add_argument(
    "--rover_flat_goal_reached_threshold_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Distance (m) for goal_reached_bonus one-shot success. Default 1.5.",
)
parser.add_argument(
    "--rover_flat_goal_relocate_threshold_m",
    type=float,
    default=None,
    help="[Rover FlatGoal/OA-Flat] Planar distance (m) under which the interval event moves the goal to a new random XY (default 5.0; should be >= close_to_target bonus radius to avoid stalling just outside relocate).",
)
parser.add_argument(
    "--rover_flat_action_rate_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for action_rate_penalty. Default 0.05.",
)
parser.add_argument(
    "--rover_flat_tilt_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for tilt_penalty. Default 0.2.",
)
parser.add_argument(
    "--rover_flat_boundary_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for boundary_penalty. Default 0.5.",
)
parser.add_argument(
    "--rover_flat_boundary_margin_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] margin_m for boundary_penalty (m). Default 3.0.",
)
parser.add_argument(
    "--rover_flat_four_wheels_ground_contact_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for four_wheels_ground_contact_reward. Default 0.2.",
)
parser.add_argument(
    "--rover_flat_front_wheels_ground_contact_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for front_wheels_ground_contact_reward. Default 0.5.",
)
parser.add_argument(
    "--rover_flat_rear_wheels_ground_contact_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for rear_wheels_ground_contact_reward. Default 0.5.",
)
parser.add_argument(
    "--rover_flat_not_four_wheels_ground_contact_penalty_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for not_four_wheels_ground_contact_penalty. Default 0.5.",
)
parser.add_argument(
    "--rover_flat_wheel_ground_contact_threshold_n",
    type=float,
    default=None,
    help="[Rover FlatGoal] Contact-force threshold in newtons for wheel-ground contact. Default 1.0.",
)
parser.add_argument(
    "--rover_flat_alive_weight",
    type=float,
    default=None,
    help="[Rover FlatGoal] Weight for alive_bonus. Default 0.1.",
)
parser.add_argument(
    "--rover_flat_wheel_effort_limit_sim",
    type=float,
    default=None,
    help="[Rover FlatGoal] Wheel actuator effort limit in N*m. Default 2500.0.",
)
parser.add_argument(
    "--rover_flat_wheel_target_slew_rad_s2",
    type=float,
    default=None,
    help="[Rover FlatGoal] Max wheel velocity target slew rate in rad/s^2. Default 350.0.",
)
parser.add_argument(
    "--rover_flat_plane_half_size_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Half-size of the ground plane (m). Default 25.0 (50 m plane).",
)
parser.add_argument(
    "--rover_flat_spawn_half_size_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Half-extent (m) used when sampling the rover spawn XY. Default 15.0.",
)
parser.add_argument(
    "--rover_flat_target_half_size_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Half-extent (m) used when sampling the target XY. Default 23.0.",
)
parser.add_argument(
    "--rover_flat_target_exclusion_half_size_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Target is sampled outside this central square half-extent (m). Default 15.5.",
)
parser.add_argument(
    "--rover_flat_target_min_spawn_dist_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Min spawn-to-target distance (m) — target is resampled until satisfied. Default 5.0.",
)
parser.add_argument(
    "--rover_flat_spawn_z_m",
    type=float,
    default=None,
    help="[Rover FlatGoal] Rover root spawn height above the ground plane (world Z, m). Default 0.2.",
)
parser.add_argument(
    "--rover_flat_episode_length_s",
    type=float,
    default=None,
    help="[Rover FlatGoal] Episode timeout (s). Default 30.0.",
)
parser.add_argument(
    "--rover_flat_spawn_target_marker",
    type=int,
    default=None,
    help="[Rover FlatGoal] Spawn yellow target marker cuboid: 1=on, 0=off. Default 1.",
)
parser.add_argument(
    "--rover_flat_upside_down_min_dot",
    type=float,
    default=None,
    help=(
        "[Rover FlatGoal] Terminate episode when body+Z · world+Z drops below this value. "
        "Geometric: cos(max_tilt_angle). Examples: 0.7 (~45°, strict), 0.5 (~60°), "
        "0.2 (~78°, only catastrophic flips, default), -1.1 (effectively disabled)."
    ),
)
parser.add_argument(
    "--rover_flat_wheelie_min_pitch_sin",
    type=float,
    default=None,
    help=(
        "[Rover FlatGoal] Wheelie termination pitch gate: terminate when both front wheels are airborne "
        "and body+X · world+Z exceeds this value. Default 0.20 (~11.5 deg nose-up)."
    ),
)
parser.add_argument(
    "--rover_flat_goal_stage",
    type=int,
    default=None,
    help="[Rover FlatGoal] Optional curriculum stage (0=crawl, 1=drive, 2=full) for training_progress_log header.",
)
parser.add_argument(
    "--rover_flat_goal_stage_note",
    type=str,
    default=None,
    help="[Rover FlatGoal] Optional short note recorded next to flat_goal_train_stage in the progress log.",
)
parser.add_argument(
    "--rover_flat_body_reverse_vel_penalty_weight",
    type=float,
    default=None,
    help="[Rover OA-Flat] Weight for penalizing reverse motion (body +X velocity negative). Default from env cfg.",
)
parser.add_argument(
    "--rover_flat_body_reverse_vel_scale_ms",
    type=float,
    default=None,
    help="[Rover OA-Flat] Normalizes reverse-speed penalty; typically match max forward speed (m/s).",
)
parser.add_argument(
    "--rover_flat_reset_max_target_samples",
    type=int,
    default=None,
    help="[Rover OA-Flat] Max rejection samples when placing goal outside spawn exclusion (reset + relocate).",
)
parser.add_argument(
    "--rover_oa_flat_box_count",
    type=int,
    default=None,
    help="[Rover OA-Flat] Number of blue obstacle cubes.",
)
parser.add_argument(
    "--rover_oa_flat_box_extent_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Blue box cube extent (m).",
)
parser.add_argument(
    "--rover_oa_flat_box_arena_half_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Half-extent (m) of XY grid for blue boxes; decouples from plane when set.",
)
parser.add_argument(
    "--rover_oa_flat_box_grid_step_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Nominal grid spacing for blue box placement (m).",
)
parser.add_argument(
    "--rover_oa_flat_box_min_spawn_clear_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Min clearance from rover spawn to any blue box center (m).",
)
parser.add_argument(
    "--rover_oa_flat_box_min_pair_clear_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Min pairwise XY clearance between blue box centers (m).",
)
parser.add_argument(
    "--rover_oa_flat_box_half_height_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Half-height of blue boxes for Z placement (m).",
)
parser.add_argument(
    "--rover_oa_flat_blue_avoidance_weight",
    type=float,
    default=None,
    help="[Rover OA-Flat] Weight for soft Gaussian avoidance penalty vs nearest blue box.",
)
parser.add_argument(
    "--rover_oa_flat_blue_avoidance_danger_radius_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Danger radius (m) shaping avoidance penalty vs distance.",
)
parser.add_argument(
    "--rover_oa_flat_blue_collision_weight",
    type=float,
    default=None,
    help="[Rover OA-Flat] Weight for near-contact collision shaping vs blue boxes.",
)
parser.add_argument(
    "--rover_oa_flat_blue_collision_radius_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Planar radius (m) for collision penalty vs blue box centers.",
)
parser.add_argument(
    "--rover_oa_flat_target_min_blue_box_clearance_m",
    type=float,
    default=None,
    help="[Rover OA-Flat] Goal sampling: min planar distance from goal to each blue box center (m).",
)
parser.add_argument(
    "--rover_oa_flat_blue_avoidance_near_ramp_weight",
    type=float,
    default=None,
    help="[Rover OA-Flat] Extra (1-d/danger)^2 avoidance magnitude inside danger_radius (oa_flat_blue_box_avoidance_penalty).",
)
parser.add_argument(
    "--rover_oa_flat_blue_avoidance_gaussian_sigma_scale",
    type=float,
    default=None,
    help="[Rover OA-Flat] Gaussian sigma = danger_radius * this scale in soft blue avoidance penalty.",
)
parser.add_argument(
    "--rover_oa_flat_obstacle_avoidance_reward_scale",
    type=float,
    default=None,
    help="[Rover OA-Flat] Global scalar for all blue-box obstacle-avoidance rewards (0=off, 1=full).",
)
parser.add_argument(
    "--rover_oa_flat_per_env_seed_streams",
    type=int,
    default=None,
    help="[Rover OA-Flat] 1=per-env RNG streams for goal/box sampling; 0=global torch.rand.",
)
parser.add_argument(
    "--rover_oa_flat_camera_width",
    type=int,
    default=None,
    help="[Rover OA-Flat] RGB camera width (px). Use ≥~300 for stable Replicator/RTX on some builds.",
)
parser.add_argument(
    "--rover_oa_flat_camera_height",
    type=int,
    default=None,
    help="[Rover OA-Flat] RGB camera height (px).",
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_goal_max_detections",
    type=int,
    default=None,
    help="[Rover OA-Flat] Policy obs: number of yellow-goal blob slots (×4 features each).",
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_goal_enrichment",
    type=int,
    choices=(0, 1),
    default=None,
    help=(
        "[Rover OA-Flat] 1=include yellow enrichment (+parity stats/hist + yellow-vs-blue slot0); "
        "0=policy observes yellow tracks only (legacy width)."
    ),
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_excess_threshold",
    type=float,
    default=None,
    help="[Rover OA-Flat] Pooled-map mass threshold for yellow goal blob confidence.",
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_conf_sigmoid_slope",
    type=float,
    default=None,
    help="[Rover OA-Flat] Sigmoid steepness for yellow blob confidence.",
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_track_conf_gate",
    type=float,
    default=None,
    help="[Rover OA-Flat] Below this confidence, zero sin/cos/tanh(range) yellow track dims.",
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_score_boost",
    type=float,
    default=None,
    help="[Rover OA-Flat] Multiplier on yellow raw score map before pooling.",
)
parser.add_argument(
    "--rover_oa_flat_visual_yellow_score_relu",
    type=int,
    choices=(0, 1),
    default=None,
    help="[Rover OA-Flat] 1=ReLU(min(R,G)-B) for yellow cue (suppress negative clutter); 0=raw difference.",
)
parser.add_argument(
    "--rover_oa_flat_policy_include_yellow_goal_visual",
    type=int,
    choices=(0, 1),
    default=None,
    help=(
        "[Rover OA-Flat] 1=include yellow goal RGB+D blob ObsTerms on policy; 0=omit (recommended baseline). "
        "Changing width invalidates checkpoints trained with the opposite layout."
    ),
)
# Rover Localization task arguments
parser.add_argument(
    "--rover_localization_expert_policy_path",
    type=str,
    default=None,
    help="Path to the frozen OA-Flat expert policy checkpoint.",
)
parser.add_argument(
    "--rover_localization_position_error_scale_m",
    type=float,
    default=None,
    help="Position error scale (m) for the localization reward.",
)
parser.add_argument(
    "--rover_localization_position_error_exp_scale",
    type=float,
    default=None,
    help="Exponential penalty scale (1/m) for localization reward.",
)
parser.add_argument(
    "--rover_localization_position_error_linear_weight",
    type=float,
    default=None,
    help="Weight of linear position-error penalty.",
)
parser.add_argument(
    "--rover_localization_position_error_exp_weight",
    type=float,
    default=None,
    help="Weight of exponential position-error penalty.",
)
parser.add_argument(
    "--rover_localization_position_bonus_threshold_m",
    type=float,
    default=None,
    help="Distance threshold (m) for sparse accuracy bonus.",
)
parser.add_argument(
    "--rover_localization_position_bonus_weight",
    type=float,
    default=None,
    help="Weight for sparse position-estimation bonus reward.",
)
parser.add_argument(
    "--rover_localization_alive_weight",
    type=float,
    default=None,
    help="Weight for alive bonus.",
)
parser.add_argument(
    "--rover_localization_distance_scale_m",
    type=float,
    default=None,
    help="Scale (m) for goal distance observations.",
)
parser.add_argument(
    "--rover_localization_episode_length_s",
    type=float,
    default=None,
    help="Episode timeout (s).",
)
parser.add_argument(
    "--rover_localization_debug_mode",
    type=int,
    default=None,
    choices=(0, 1),
    help="Enable debug visualization (red sphere at estimated position). 1=on, 0=off.",
)
parser.add_argument(
    "--rover_localization_goal_reached_bonus_weight",
    type=float,
    default=None,
    help="Goal-reached bonus weight. Must be >0 for the reward manager to call the term (it skips weight==0). The bonus itself is negligible at 1e-8, but the side-effect triggers goal relocation."
)
parser.add_argument(
    "--rover_localization_goal_reached_threshold_m",
    type=float,
    default=None,
    help="Distance threshold (m) under which the expert reaching the goal triggers relocation.",
)
parser.add_argument(
    "--rover_localization_include_yellow_goal_visual",
    type=int,
    default=None,
    choices=(0, 1),
    help="Include yellow goal visual observation terms for the expert policy (1=on, 0=off). Default True to prevent circling.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Training runs headless by default. Respect explicit --enable_cameras instead
# of forcing cameras off when --video is not requested.
if args_cli.video:
    args_cli.enable_cameras = True
    print("[INFO] Video recording enabled - cameras will be active")
elif args_cli.enable_cameras:
    print("[INFO] Cameras explicitly enabled via CLI flag")
elif getattr(args_cli, "task", None) and ("OAFlat" in args_cli.task or "Localization" in args_cli.task):
    # Isaac-Rover-OAFlat-* and Isaac-Rover-Localization-* scenes spawn CameraCfg (vision obs).
    args_cli.enable_cameras = True
    print("[INFO] OA-Flat / Localization task uses onboard RGB camera; enabling cameras.")
else:
    # AppLauncher auto-detects headless mode when render_mode=None.
    args_cli.enable_cameras = False
    print("[INFO] Running in headless mode (no video, no GUI)")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app (headless by default when video=False)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import re
import tempfile
import time
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl.divergence_aware_runner import DivergenceAwareOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, LoggingWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)


def _get_g1_joints_npz_link_helpers():
    """NPZ/HDF5 motion helpers for G1 imitation; return three ``None`` if ``g1_locomotion_v1`` was removed."""
    try:
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_joints_npz_link_util import (
            resolve_g1_joints_data_npz_path,
            write_npz_dof_pos_scaled_copy,
            write_npz_link_local_pos_x_scaled_copy,
        )

        return resolve_g1_joints_data_npz_path, write_npz_dof_pos_scaled_copy, write_npz_link_local_pos_x_scaled_copy
    except ModuleNotFoundError:
        return None, None, None


def _g1_v1_apply_debug_fixed_base_if_available(env_cfg) -> None:
    try:
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v3_env_cfg import (
            apply_g1_locomotion_v3_debug_training_fixed_base,
        )
    except ModuleNotFoundError:
        return
    apply_g1_locomotion_v3_debug_training_fixed_base(env_cfg)


def _infer_rsl_rl_actor_obs_dim_from_checkpoint(checkpoint_path: str) -> int | None:
    """Return actor MLP input width from an RSL-RL ``.pt`` (``actor.0.weight`` shape[1]), or None."""
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    sd = blob.get("model_state_dict", blob)
    if not isinstance(sd, dict):
        return None
    w = sd.get("actor.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) < 2:
        return None
    return int(w.shape[1])


def _align_g1_v4_env_cfg_obs_for_resume_checkpoint(env_cfg, checkpoint_obs_dim: int) -> bool:
    """If V4 cfg obs size ≠ checkpoint, try common toggle presets (103, 104, …). Returns True if aligned."""
    from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v4_env_cfg import (
        compute_g1_locomotion_v4_policy_obs_dim,
    )

    keys = (
        "include_root_xy_error_obs",
        "velocity_command_obs_enabled",
        "append_teleop_hand_target_obs",
        "include_reward_profile_phase_obs",
        "target_yaw_command_obs_enabled",
    )
    saved = {k: bool(getattr(env_cfg, k)) if hasattr(env_cfg, k) else False for k in keys}

    def _dim() -> int:
        return int(compute_g1_locomotion_v4_policy_obs_dim(env_cfg))

    if _dim() == checkpoint_obs_dim:
        return False

    presets: list[dict[str, bool]] = [
        # (root_xy, vel_cmd, teleop, phase, yaw) — base 99+4 torque = 103 when imitation extras off
        {k: v for k, v in zip(keys, (False, False, False, False, False), strict=True)},
        {k: v for k, v in zip(keys, (False, True, False, False, False), strict=True)},
        {k: v for k, v in zip(keys, (True, False, False, False, False), strict=True)},
        {k: v for k, v in zip(keys, (False, False, True, False, False), strict=True)},
        {k: v for k, v in zip(keys, (False, True, True, False, False), strict=True)},
        {k: v for k, v in zip(keys, (True, False, True, False, False), strict=True)},
        {k: v for k, v in zip(keys, (True, True, False, False, False), strict=True)},
        {k: v for k, v in zip(keys, (True, True, True, False, False), strict=True)},
        {k: v for k, v in zip(keys, (True, True, True, True, False), strict=True)},
        {k: v for k, v in zip(keys, (True, True, True, False, True), strict=True)},
        {k: v for k, v in zip(keys, (True, True, True, True, True), strict=True)},
    ]

    for preset in presets:
        for k, v in preset.items():
            if hasattr(env_cfg, k):
                setattr(env_cfg, k, v)
        if _dim() == checkpoint_obs_dim:
            print(
                f"[INFO] G1 V4 resume: observation toggles matched to checkpoint "
                f"(observation_space={checkpoint_obs_dim}): {preset}"
            )
            return True

    for k, v in saved.items():
        if hasattr(env_cfg, k):
            setattr(env_cfg, k, v)
    print(
        f"[WARNING] G1 V4 resume: could not match env obs toggles to checkpoint "
        f"(checkpoint obs_dim={checkpoint_obs_dim}, cfg gives {_dim()}). "
        f"Pass --include_root_xy_error_obs / --velocity_command_obs_enabled / --append_teleop_hand_target_obs explicitly."
    )
    return False


# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# Patch torch.normal to clamp scale (avoids "normal expects std >= 0" in PPO when policy std collapses)
_torch_normal_orig = torch.normal
_MIN_STD = 1e-6


def _clamp_std(s):
    if isinstance(s, torch.Tensor):
        # Replace invalid (<=0, NaN, inf) with _MIN_STD
        valid = (s > _MIN_STD) & torch.isfinite(s)
        return torch.where(valid, s, torch.full_like(s, _MIN_STD))
    v = float(s)
    return v if v > _MIN_STD and v == v else _MIN_STD  # v==v is False for NaN


def _torch_normal_patched(*args, **kwargs):
    if len(args) >= 2:
        args = (args[0], _clamp_std(args[1]), *args[2:])
    if "std" in kwargs:
        kwargs = dict(kwargs)
        kwargs["std"] = _clamp_std(kwargs["std"])
    return _torch_normal_orig(*args, **kwargs)


def _truncate_training_progress_log_to_checkpoint_iter(progress_log_path: str, checkpoint_spec: str) -> None:
    """When resuming from model_N.pt, drop copied training_progress_log blocks with iteration > N.

    Full-run logs copied on resume would otherwise include iterations past the checkpoint; final_model.pt
    and other checkpoint names leave the file unchanged.
    """
    ck = os.path.basename(checkpoint_spec)
    m = re.match(r"^model_(\d+)\.pt$", ck, re.IGNORECASE)
    if not m:
        return
    max_iter = int(m.group(1))
    try:
        with open(progress_log_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return
    pat = re.compile(r"Learning iteration\s+(\d+)/")
    matches = list(pat.finditer(text))
    if not matches:
        return
    cut = None
    for mo in matches:
        if int(mo.group(1)) > max_iter:
            cut = mo.start()
            break
    if cut is None:
        return
    with open(progress_log_path, "w", encoding="utf-8") as f:
        f.write(text[:cut])
    print(
        f"[INFO] Truncated copied training_progress_log.txt to iterations <= {max_iter} "
        f"(resume checkpoint {ck})."
    )


def _replace_training_progress_log_run_arguments_section(progress_log_path: str, args_cli) -> None:
    """On resume, refresh the 'Run arguments (this run)' block so it matches the current CLI invocation."""
    try:
        with open(progress_log_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return
    new_lines = [
        "################################################################################",
        " Run arguments (this run)",
        "################################################################################",
    ]
    for k in sorted(vars(args_cli).keys()):
        if k.startswith("_"):
            continue
        v = getattr(args_cli, k)
        if v is None:
            continue
        new_lines.append(f"  {k}: {v}")
    new_block = "\n".join(new_lines)
    pat = re.compile(
        r"(?ms)^################################################################################\n Run arguments \(this run\)\n################################################################################\n.*?(?=^################################################################################\n Env config)",
    )
    new_content, n_subs = pat.subn(new_block + "\n", content, count=1)
    if n_subs:
        with open(progress_log_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return
    try:
        lines = content.split("\n")
        start_it = None
        for i, line in enumerate(lines):
            if "Learning iteration" in re.sub(r"\x1b\[[0-9;]*m", "", line):
                start_it = i
                break
        if start_it is None:
            return
        body = "\n".join(lines[start_it:])
        with open(progress_log_path, "w", encoding="utf-8") as f:
            f.write(new_block + "\n\n" + body)
    except OSError:
        return


def _rover_oa_flat_training_log_snapshot_lines(env_cfg) -> list[str]:
    """Lines for the OA-Flat env snapshot + Effective RewTerm block (used in training_progress_log header)."""
    header_lines: list[str] = []
    header_lines.append("")
    header_lines.append("################################################################################")
    header_lines.append(" Rover OA-Flat – config snapshot (depth scan + privileged obstacles + aux)")
    header_lines.append("################################################################################")
    for _a in (
        "oa_flat_camera_sensor_name",
        "oa_flat_camera_width",
        "oa_flat_camera_height",
        "oa_flat_depth_scan_num_bins",
        "oa_flat_privileged_k_nearest",
        "oa_flat_box_count",
        "oa_flat_box_extent_m",
        "oa_flat_box_arena_half_m",
        "oa_flat_box_arena_use_plane",
        "oa_flat_box_grid_step_m",
        "oa_flat_box_min_spawn_clear_m",
        "oa_flat_box_min_pair_clear_m",
        "oa_flat_box_half_height_m",
        "oa_flat_target_min_blue_box_clearance_m",
        "oa_flat_goal_marker_z_above_spawn_m",
        "oa_flat_visual_yellow_goal_max_detections",
        "oa_flat_visual_yellow_goal_enrichment_enabled",
        "oa_flat_policy_include_yellow_goal_visual",
        "oa_flat_visual_yellow_goal_enrichment_extra_dim",
        "oa_flat_visual_yellow_excess_threshold",
        "oa_flat_visual_yellow_conf_sigmoid_slope",
        "oa_flat_visual_yellow_track_conf_gate",
        "oa_flat_visual_yellow_score_relu",
        "oa_flat_visual_yellow_score_boost",
        "oa_flat_blue_avoidance_weight",
        "oa_flat_blue_avoidance_danger_radius_m",
        "oa_flat_blue_avoidance_near_ramp_weight",
        "oa_flat_blue_avoidance_gaussian_sigma_scale",
        "oa_flat_blue_collision_weight",
        "oa_flat_blue_collision_radius_m",
        "oa_flat_obstacle_avoidance_reward_scale",
        "oa_flat_per_env_seed_streams",
        "flat_body_reverse_vel_penalty_weight",
        "flat_body_reverse_vel_scale_ms",
        "flat_body_reverse_far_boost_start_m",
        "flat_body_reverse_far_boost_max_mul",
        "flat_body_reverse_far_boost_ramp_m",
        "flat_yaw_closure_speed_threshold_m_s",
        "flat_reset_max_target_samples",
        "flat_distance_to_target_weight",
        "flat_distance_to_target_scale_m",
        "flat_distance_progress_weight",
        "flat_distance_increase_penalty_weight",
        "flat_goal_reached_threshold_m",
        "flat_goal_relocate_threshold_m",
        "flat_close_to_target_bonus_weight",
        "flat_close_to_target_threshold_m",
        "flat_goal_reached_bonus_weight",
        "flat_heading_to_target_weight",
        "flat_heading_goal_closure_weight",
        "flat_yaw_no_closure_penalty_weight",
        "flat_forward_velocity_to_target_weight",
        "flat_forward_velocity_to_target_cap_ms",
        "plane_half_size_m",
        "spawn_half_size_m",
        "target_half_size_m",
        "spawn_z_m",
        "episode_length_s",
        "flat_goal_train_stage",
        "flat_goal_train_stage_note",
    ):
        if hasattr(env_cfg, _a):
            header_lines.append(f"  {_a}: {getattr(env_cfg, _a)}")
    _rew = getattr(env_cfg, "rewards", None)
    if _rew is not None:
        header_lines.append("")
        header_lines.append("  # Effective RewTerm weights (post overrides):")
        for _rt_name in (
            "distance_to_target",
            "distance_progress",
            "distance_increase_penalty",
            "heading_to_target",
            "heading_goal_closure_speed",
            "yaw_without_goal_closure",
            "forward_velocity_to_target",
            "close_to_target_bonus",
            "goal_reached_bonus",
            "action_rate_penalty",
            "tilt_penalty",
            "boundary_penalty",
            "four_wheels_contact",
            "front_wheels_contact",
            "rear_wheels_contact",
            "not_four_wheels_contact_penalty",
            "alive_bonus",
            "blue_box_avoidance_penalty",
            "blue_box_collision_penalty",
            "body_reverse_vel_penalty",
        ):
            _rt = getattr(_rew, _rt_name, None)
            if _rt is not None:
                _params = getattr(_rt, "params", {}) or {}
                header_lines.append(
                    f"  reward.{_rt_name}.weight: {getattr(_rt, 'weight', None)}"
                    + ("" if not _params else f"  params={_params}")
                )
    return header_lines


def _replace_training_progress_log_rover_oa_flat_snapshot(progress_log_path: str, env_cfg) -> None:
    """After resume copies training_progress_log from an older run, rewrite the OA-Flat snapshot from live env_cfg.

    Only the CLI \"Run arguments\" block was refreshed before; flat_* / RewTerm lines stayed stale and could
    contradict the current invocation (e.g. distance_progress_weight mismatch).
    """
    try:
        snap_lines = _rover_oa_flat_training_log_snapshot_lines(env_cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[WARNING] Could not build rover OA-Flat snapshot for training_progress_log.txt: %s",
            exc,
        )
        return
    block = "\n".join(snap_lines + ["################################################################################", "################################################################################"]) + "\n"
    try:
        with open(progress_log_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return
    marker = (
        "################################################################################\n"
        " Rover OA-Flat – config snapshot (depth scan + privileged obstacles + aux)\n"
        "################################################################################\n"
    )
    idx = content.find(marker)
    if idx < 0:
        return
    m = re.search(
        r"\n################################################################################\n################################################################################\n",
        content[idx + len(marker) :],
    )
    if not m:
        logger.warning(
            "[WARNING] Could not find end of rover OA-Flat block in %s; snapshot not replaced.",
            progress_log_path,
        )
        return
    end_abs = idx + len(marker) + m.end()
    new_content = content[:idx] + block + content[end_abs:]
    try:
        with open(progress_log_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as exc:
        logger.warning("[WARNING] Could not rewrite rover OA-Flat snapshot in training_progress_log.txt: %s", exc)


def _rover_localization_train_debug_active(args_cli) -> bool:
    """True when CLI flag or ROVER_LOCALIZATION_TRAIN_DEBUG env requests localization train debug."""
    if getattr(args_cli, "rover_localization_train_debug", False):
        return True
    return os.environ.get("ROVER_LOCALIZATION_TRAIN_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _rover_localization_task(args_cli) -> str:
    return (getattr(args_cli, "task", None) or "").split(":")[-1]


def _write_rover_localization_train_debug_snap(
    *,
    log_dir: str | None,
    stage: str,
    env_cfg,
    agent_cfg,
    args_cli,
    runner=None,
    entropy_anneal_installed: bool | None = None,
    env=None,
) -> None:
    """Console + append-only file snapshot for rover localization training (debug builds)."""
    if not _rover_localization_train_debug_active(args_cli):
        return
    if "Isaac-Rover-DiskWorld-Localization" not in _rover_localization_task(args_cli):
        return
    lines: list[str] = [
        f"[ROVER_LOC_TRAIN_DEBUG] === {stage} ===",
        "  Expected fixes (verify below):",
        "  - LocalizationEstimateAction: processed_actions = nan_to_num(raw) only (no ±1.5 clamp).",
        "  - Rewards: position scale_m≈100, close threshold_m≈3, strong wheel-odom + raw_wheel_odom_seed terms.",
        "  - Termination: fell_off_disk min_radius≈85 m.",
        "  - PPO: lower init_noise_std, entropy_coef anneal when --ppo_entropy_coef_final is set.",
        "",
        "  env_cfg.scene.num_envs:",
        f"    {getattr(getattr(env_cfg, 'scene', None), 'num_envs', getattr(env_cfg, 'num_envs', '?'))}",
    ]
    for k in (
        "loc_position_error_weight",
        "loc_position_error_scale_m",
        "loc_position_close_weight",
        "loc_position_close_threshold_m",
        "loc_orientation_error_weight",
        "loc_yaw_unit_norm_penalty_weight",
        "loc_estimate_smoothness_weight",
        "loc_estimate_near_wheel_odom_weight",
        "loc_estimate_near_wheel_odom_scale_m",
        "loc_estimate_raw_wheel_odom_seed_weight",
        "loc_estimate_raw_wheel_odom_seed_decay_env_steps",
        "loc_estimate_alive_bonus_weight",
        "loc_fall_min_radius_m",
        "estimate_pos_norm_m",
        "drive_policy_path",
        "box_count",
        "goal_count",
    ):
        if hasattr(env_cfg, k):
            lines.append(f"  {k}: {getattr(env_cfg, k)}")
    _rew = getattr(env_cfg, "rewards", None)
    if _rew is not None:
        lines.append("  Effective RewTerm (post __post_init__ / CLI):")
        for _n in (
            "position_error_reward",
            "position_close_bonus",
            "orientation_error_reward",
            "yaw_unit_norm_penalty",
            "estimate_smoothness",
            "estimate_near_wheel_odom",
            "estimate_raw_wheel_odom_seed",
            "estimate_alive_bonus",
        ):
            _rt = getattr(_rew, _n, None)
            if _rt is None:
                continue
            _p = getattr(_rt, "params", None) or {}
            lines.append(f"    reward.{_n}: weight={getattr(_rt, 'weight', None)} params={dict(_p)}")
    _term = getattr(env_cfg, "terminations", None)
    if _term is not None and getattr(_term, "fell_off_disk", None) is not None:
        _fd = _term.fell_off_disk
        lines.append(f"  terminations.fell_off_disk.params: {getattr(_fd, 'params', {})}")
    lines.extend(
        [
            "  agent_cfg:",
            f"    num_steps_per_env: {getattr(agent_cfg, 'num_steps_per_env', None)}",
            f"    max_iterations: {getattr(agent_cfg, 'max_iterations', None)}",
            f"    clip_actions: {getattr(agent_cfg, 'clip_actions', None)}",
            f"    policy.init_noise_std: {getattr(getattr(agent_cfg, 'policy', None), 'init_noise_std', None)}",
            f"    algorithm.entropy_coef: {getattr(getattr(agent_cfg, 'algorithm', None), 'entropy_coef', None)}",
            f"    algorithm.learning_rate: {getattr(getattr(agent_cfg, 'algorithm', None), 'learning_rate', None)}",
        ]
    )
    if getattr(args_cli, "ppo_entropy_coef_final", None) is not None:
        lines.append(f"    CLI ppo_entropy_coef_final (anneal target): {args_cli.ppo_entropy_coef_final}")
    if entropy_anneal_installed is not None:
        lines.append(f"    entropy_coef anneal hook installed on runner.alg.update: {entropy_anneal_installed}")
    if runner is not None and hasattr(runner, "alg"):
        lines.append(f"    runtime algorithm.entropy_coef: {getattr(runner.alg, 'entropy_coef', None)}")
    if env is not None:
        try:
            eu = env
            while hasattr(eu, "env"):
                eu = eu.env
            lines.append(f"  env: {type(eu).__name__}")
            if hasattr(eu, "action_space") and eu.action_space is not None:
                lines.append(f"    action_space.shape: {eu.action_space.shape}")
            if hasattr(eu, "observation_space") and eu.observation_space is not None:
                lines.append(f"    observation_space: {eu.observation_space}")
        except Exception as exc:
            lines.append(f"  env (unwrapped) introspection failed: {exc}")
    text = "\n".join(lines)
    print(text)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            snap_path = os.path.join(log_dir, "localization_train_debug_snap.txt")
            with open(snap_path, "a", encoding="utf-8") as f:
                f.write(text + "\n\n")
        except OSError as exc:
            print(f"[WARN] ROVER_LOC_TRAIN_DEBUG: could not write snap file: {exc}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    torch.normal = _torch_normal_patched
    if getattr(args_cli, "rover_localization_train_debug", False):
        os.environ["ROVER_LOCALIZATION_TRAIN_DEBUG"] = "1"
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # PPO hyperparameter overrides
    ppo_overrides = [
        ("ppo_learning_rate", "algorithm", "learning_rate"),
        ("ppo_entropy_coef", "algorithm", "entropy_coef"),
        ("ppo_value_loss_coef", "algorithm", "value_loss_coef"),
        ("ppo_clip_param", "algorithm", "clip_param"),
        ("ppo_desired_kl", "algorithm", "desired_kl"),
        ("ppo_gamma", "algorithm", "gamma"),
        ("ppo_lam", "algorithm", "lam"),
        ("ppo_num_learning_epochs", "algorithm", "num_learning_epochs"),
        ("ppo_num_mini_batches", "algorithm", "num_mini_batches"),
        ("ppo_max_grad_norm", "algorithm", "max_grad_norm"),
        ("ppo_init_noise_std", "policy", "init_noise_std"),
        ("ppo_noise_std_type", "policy", "noise_std_type"),
    ]
    for arg_name, cfg_section, attr_name in ppo_overrides:
        val = getattr(args_cli, arg_name, None)
        if val is not None and hasattr(agent_cfg, cfg_section):
            section = getattr(agent_cfg, cfg_section)
            if hasattr(section, attr_name):
                setattr(section, attr_name, val)
                print(f"[INFO] PPO: {cfg_section}.{attr_name} = {val}")

    # Runner-level exploration: steps per env per update (more = more diverse rollouts)
    if getattr(args_cli, "ppo_num_steps_per_env", None) is not None:
        if hasattr(agent_cfg, "num_steps_per_env"):
            agent_cfg.num_steps_per_env = args_cli.ppo_num_steps_per_env
            print(f"[INFO] PPO (runner): num_steps_per_env = {args_cli.ppo_num_steps_per_env}")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Override policy_path and joints_data from command line arguments (for transition learning)
    if args_cli.policy_path is not None:
        if hasattr(env_cfg, 'policy_path'):
            env_cfg.policy_path = args_cli.policy_path
            print(f"[INFO] Policy path set to: {env_cfg.policy_path}")
        else:
            logger.warning(f"Config does not have 'policy_path' attribute. Ignoring --policy_path argument.")

    if args_cli.joints_data is not None:
        if hasattr(env_cfg, 'joints_data'):
            env_cfg.joints_data = args_cli.joints_data
            print(f"[INFO] Joints data file set to: {env_cfg.joints_data}")
        else:
            logger.warning(f"Config does not have 'joints_data' attribute. Ignoring --joints_data argument.")
    if getattr(args_cli, 'npz_only_for_limits', False):
        if hasattr(env_cfg, 'npz_only_for_limits'):
            env_cfg.npz_only_for_limits = True
            print("[INFO] hdf5_only_for_limits / npz_only_for_limits: motion file used only for joint limits")
        else:
            logger.warning(
                "Config does not have 'npz_only_for_limits' attribute. Ignoring --hdf5_only_for_limits / --npz_only_for_limits."
            )

    if args_cli.start_pose_path is not None:
        if hasattr(env_cfg, 'start_pose_path'):
            env_cfg.start_pose_path = args_cli.start_pose_path
            print(f"[INFO] Start pose path set to: {env_cfg.start_pose_path}")
        else:
            logger.warning(f"Config does not have 'start_pose_path' attribute. Ignoring --start_pose_path argument.")

    if args_cli.transition_duration is not None:
        if hasattr(env_cfg, 'transition_duration'):
            env_cfg.transition_duration = args_cli.transition_duration
            print(f"[INFO] Transition duration set to: {env_cfg.transition_duration} seconds")
        else:
            logger.warning(f"Config does not have 'transition_duration' attribute. Ignoring --transition_duration argument.")

    # Override transition learning parameters from command line arguments
    if args_cli.policy_control_duration is not None:
        if hasattr(env_cfg, 'policy_control_duration'):
            env_cfg.policy_control_duration = args_cli.policy_control_duration
            print(f"[INFO] Policy control duration set to: {env_cfg.policy_control_duration}")
    
    if args_cli.policy_transition_duration is not None:
        if hasattr(env_cfg, 'policy_transition_duration'):
            env_cfg.policy_transition_duration = args_cli.policy_transition_duration
            print(f"[INFO] Policy transition duration set to: {env_cfg.policy_transition_duration}")
    
    if args_cli.joint_position_reward_weight is not None:
        if hasattr(env_cfg, 'joint_position_reward_weight'):
            env_cfg.joint_position_reward_weight = args_cli.joint_position_reward_weight
            print(f"[INFO] Joint position reward weight set to: {env_cfg.joint_position_reward_weight}")
    
    if args_cli.joint_position_reward_weight_legs is not None:
        if hasattr(env_cfg, 'joint_position_reward_weight_legs'):
            env_cfg.joint_position_reward_weight_legs = args_cli.joint_position_reward_weight_legs
            print(f"[INFO] Leg joint position reward weight set to: {env_cfg.joint_position_reward_weight_legs}")
    
    if args_cli.joint_position_reward_weight_arms is not None:
        if hasattr(env_cfg, 'joint_position_reward_weight_arms'):
            env_cfg.joint_position_reward_weight_arms = args_cli.joint_position_reward_weight_arms
            print(f"[INFO] Arm joint position reward weight set to: {env_cfg.joint_position_reward_weight_arms}")
    
    if args_cli.joint_position_reward_weight_fingers is not None:
        if hasattr(env_cfg, 'joint_position_reward_weight_fingers'):
            env_cfg.joint_position_reward_weight_fingers = args_cli.joint_position_reward_weight_fingers
            print(f"[INFO] Finger joint position reward weight set to: {env_cfg.joint_position_reward_weight_fingers}")
    
    if args_cli.joint_position_decay_legs is not None:
        if hasattr(env_cfg, 'joint_position_decay_legs'):
            env_cfg.joint_position_decay_legs = args_cli.joint_position_decay_legs
            print(f"[INFO] Leg joint position decay constant set to: {env_cfg.joint_position_decay_legs} rad²")
        else:
            logger.warning(f"Config does not have 'joint_position_decay_legs' attribute. Ignoring --joint_position_decay_legs argument.")
    
    if args_cli.joint_position_decay_arms is not None:
        if hasattr(env_cfg, 'joint_position_decay_arms'):
            env_cfg.joint_position_decay_arms = args_cli.joint_position_decay_arms
            print(f"[INFO] Arm joint position decay constant set to: {env_cfg.joint_position_decay_arms} rad²")
        else:
            logger.warning(f"Config does not have 'joint_position_decay_arms' attribute. Ignoring --joint_position_decay_arms argument.")
    
    if args_cli.joint_position_decay_fingers is not None:
        if hasattr(env_cfg, 'joint_position_decay_fingers'):
            env_cfg.joint_position_decay_fingers = args_cli.joint_position_decay_fingers
            print(f"[INFO] Finger joint position decay constant set to: {env_cfg.joint_position_decay_fingers} rad²")
        else:
            logger.warning(f"Config does not have 'joint_position_decay_fingers' attribute. Ignoring --joint_position_decay_fingers argument.")
    
    # -----------------------------------------------------------------------
    # Rover disk-world task overrides  (Isaac-Rover-DiskWorld-Forward-v0)
    # -----------------------------------------------------------------------
    _task_name = getattr(args_cli, "task", "") or ""
    _exp_name = getattr(agent_cfg, "experiment_name", "") or ""
    _rover_dw_obstacle = "ObstacleAvoidance" in _task_name or "rover_disk_world_obstacle_avoidance" in _exp_name
    _rover_dw_shared = "Rover-DiskWorld" in _task_name or "rover_disk_world" in _exp_name
    if not _rover_dw_obstacle and (
        "Rover-DiskWorld" in _task_name
        or "rover_disk_world_physparam" in _exp_name
        or "rover_disk_world_idle_to_forward" in _exp_name
        or ("rover_disk_world" in _exp_name and "obstacle_avoidance" not in _exp_name)
    ):
        _rew = getattr(env_cfg, "rewards", None)
        _term = getattr(env_cfg, "terminations", None)

        def _set_rew_weight(term_name, value, label):
            term = getattr(_rew, term_name, None) if _rew else None
            if term is not None:
                term.weight = value
                print(f"[INFO] Rover: rewards.{term_name}.weight = {value}")
            else:
                logger.warning(f"[Rover] rewards.{term_name} not found – ignoring --{label}")

        def _set_rew_param(term_name, param, value, label):
            term = getattr(_rew, term_name, None) if _rew else None
            if term is not None:
                if term.params is None:
                    term.params = {}
                term.params[param] = value
                print(f"[INFO] Rover: rewards.{term_name}.params[{param!r}] = {value}")
            else:
                logger.warning(f"[Rover] rewards.{term_name} not found – ignoring --{label}")

        def _set_term_param(term_name, param, value, label):
            term = getattr(_term, term_name, None) if _term else None
            if term is not None:
                if term.params is None:
                    term.params = {}
                term.params[param] = value
                print(f"[INFO] Rover: terminations.{term_name}.params[{param!r}] = {value}")
            else:
                logger.warning(f"[Rover] terminations.{term_name} not found – ignoring --{label}")

        if args_cli.rover_forward_progress_weight is not None:
            _set_rew_weight("forward_angular_progress", args_cli.rover_forward_progress_weight, "rover_forward_progress_weight")
        if args_cli.rover_speed_tracking_weight is not None:
            _set_rew_weight("forward_speed_tracking", args_cli.rover_speed_tracking_weight, "rover_speed_tracking_weight")
        if args_cli.rover_speed_target is not None:
            _set_rew_param("forward_speed_tracking", "target_tangential_speed", args_cli.rover_speed_target, "rover_speed_target")
        if args_cli.rover_heading_weight is not None:
            _set_rew_weight("heading_alignment", args_cli.rover_heading_weight, "rover_heading_weight")
        if args_cli.rover_action_rate_weight is not None:
            _set_rew_weight("action_rate_penalty", args_cli.rover_action_rate_weight, "rover_action_rate_weight")
        if args_cli.rover_motion_jerk_weight is not None:
            _set_rew_weight("motion_jerk_penalty", args_cli.rover_motion_jerk_weight, "rover_motion_jerk_weight")
        if args_cli.rover_heading_oscillation_weight is not None:
            _set_rew_weight(
                "heading_oscillation_penalty",
                args_cli.rover_heading_oscillation_weight,
                "rover_heading_oscillation_weight",
            )
        if args_cli.rover_motion_jerk_bonus_weight is not None:
            _set_rew_weight(
                "motion_jerk_smoothness_bonus",
                args_cli.rover_motion_jerk_bonus_weight,
                "rover_motion_jerk_bonus_weight",
            )
        if args_cli.rover_heading_stability_bonus_weight is not None:
            _set_rew_weight(
                "heading_stability_bonus",
                args_cli.rover_heading_stability_bonus_weight,
                "rover_heading_stability_bonus_weight",
            )
        if args_cli.rover_motion_jerk_bonus_exp_scale is not None:
            _set_rew_param(
                "motion_jerk_smoothness_bonus",
                "exp_scale",
                args_cli.rover_motion_jerk_bonus_exp_scale,
                "rover_motion_jerk_bonus_exp_scale",
            )
        if args_cli.rover_heading_stability_bonus_exp_scale is not None:
            _set_rew_param(
                "heading_stability_bonus",
                "exp_scale",
                args_cli.rover_heading_stability_bonus_exp_scale,
                "rover_heading_stability_bonus_exp_scale",
            )
        if args_cli.rover_alive_weight is not None:
            _set_rew_weight("alive_bonus", args_cli.rover_alive_weight, "rover_alive_weight")
        if args_cli.rover_tilt_weight is not None:
            _set_rew_weight("tilt_penalty", args_cli.rover_tilt_weight, "rover_tilt_weight")
        if args_cli.rover_radial_z_alignment_reward_weight is not None:
            _set_rew_weight(
                "radial_z_alignment_reward",
                args_cli.rover_radial_z_alignment_reward_weight,
                "rover_radial_z_alignment_reward_weight",
            )
        if args_cli.rover_radial_z_alignment_penalty_weight is not None:
            _set_rew_weight(
                "radial_z_alignment_penalty",
                args_cli.rover_radial_z_alignment_penalty_weight,
                "rover_radial_z_alignment_penalty_weight",
            )
        if args_cli.rover_goal_bonus_weight is not None:
            _set_rew_weight("goal_bonus", args_cli.rover_goal_bonus_weight, "rover_goal_bonus_weight")
        if args_cli.rover_goal_rad is not None:
            _set_term_param("forward_progress_goal", "goal_rad", args_cli.rover_goal_rad, "rover_goal_rad")
        if args_cli.rover_fall_radius is not None:
            _set_term_param("fell_off_disk", "min_radius", args_cli.rover_fall_radius, "rover_fall_radius")

    # Episode length + localization-style scene flags for all disk-world rover tasks (incl. obstacle avoidance).
    if _rover_dw_shared:
        if args_cli.rover_episode_length_s is not None:
            env_cfg.episode_length_s = args_cli.rover_episode_length_s
            print(f"[INFO] Rover: episode_length_s = {args_cli.rover_episode_length_s}")
        if getattr(args_cli, "rover_localization_visual_obstacle_max_envs", None) is not None:
            if hasattr(env_cfg, "visual_obstacle_max_spawned_envs"):
                env_cfg.visual_obstacle_max_spawned_envs = int(
                    args_cli.rover_localization_visual_obstacle_max_envs
                )
                print(
                    "[INFO] Rover localization: visual_obstacle_max_spawned_envs = "
                    f"{env_cfg.visual_obstacle_max_spawned_envs}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no visual_obstacle_max_spawned_envs – ignoring "
                    "--rover_localization_visual_obstacle_max_envs"
                )
        if getattr(args_cli, "rover_localization_visual_usd_all_envs_cap", None) is not None:
            if hasattr(env_cfg, "visual_obstacle_usd_all_envs_cap"):
                env_cfg.visual_obstacle_usd_all_envs_cap = int(
                    args_cli.rover_localization_visual_usd_all_envs_cap
                )
                print(
                    "[INFO] Rover localization: visual_obstacle_usd_all_envs_cap = "
                    f"{env_cfg.visual_obstacle_usd_all_envs_cap}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no visual_obstacle_usd_all_envs_cap – ignoring "
                    "--rover_localization_visual_usd_all_envs_cap"
                )
        if getattr(args_cli, "rover_localization_goal_collision", None) is not None:
            if hasattr(env_cfg, "goal_collision_enabled"):
                env_cfg.goal_collision_enabled = bool(int(args_cli.rover_localization_goal_collision))
                print(
                    "[INFO] Rover localization: goal_collision_enabled = "
                    f"{env_cfg.goal_collision_enabled}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no goal_collision_enabled – ignoring "
                    "--rover_localization_goal_collision"
                )
        if getattr(args_cli, "rover_localization_goal_asset_path", None):
            if hasattr(env_cfg, "goal_asset_path"):
                env_cfg.goal_asset_path = str(args_cli.rover_localization_goal_asset_path)
                print(
                    "[INFO] Rover localization: goal_asset_path = "
                    f"{env_cfg.goal_asset_path}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no goal_asset_path – ignoring "
                    "--rover_localization_goal_asset_path"
                )
        if getattr(args_cli, "rover_localization_box_collision", None) is not None:
            if hasattr(env_cfg, "box_collision_enabled"):
                env_cfg.box_collision_enabled = bool(int(args_cli.rover_localization_box_collision))
                print(
                    "[INFO] Rover localization: box_collision_enabled = "
                    f"{env_cfg.box_collision_enabled}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no box_collision_enabled – ignoring "
                    "--rover_localization_box_collision"
                )
        if getattr(args_cli, "rover_localization_goal_count", None) is not None:
            if hasattr(env_cfg, "goal_count"):
                env_cfg.goal_count = int(args_cli.rover_localization_goal_count)
                print(f"[INFO] Rover localization: goal_count = {env_cfg.goal_count}")
            else:
                logger.warning(
                    "[Rover] env_cfg has no goal_count – ignoring "
                    "--rover_localization_goal_count"
                )
        if getattr(args_cli, "rover_localization_box_count", None) is not None:
            if hasattr(env_cfg, "box_count"):
                env_cfg.box_count = int(args_cli.rover_localization_box_count)
                print(f"[INFO] Rover localization: box_count = {env_cfg.box_count}")
            else:
                logger.warning(
                    "[Rover] env_cfg has no box_count – ignoring "
                    "--rover_localization_box_count"
                )
        # Keep reset-event obstacle layout in sync with env_cfg.* (CLI may have updated counts).
        _ev = getattr(env_cfg, "events", None)
        if _ev is not None and hasattr(_ev, "randomize_obstacles"):
            _rt = _ev.randomize_obstacles
            if hasattr(env_cfg, "goal_count"):
                _rt.params["goal_count"] = int(env_cfg.goal_count)
            if hasattr(env_cfg, "box_count"):
                _rt.params["box_count"] = int(env_cfg.box_count)
            if hasattr(env_cfg, "goal_radial_distance"):
                _rt.params["goal_radial_distance"] = float(env_cfg.goal_radial_distance)
            if hasattr(env_cfg, "oa_box_radial_distance_m"):
                _rt.params["box_radial_distance_m"] = float(env_cfg.oa_box_radial_distance_m)
            if hasattr(env_cfg, "oa_cylinder_y_half_width_m"):
                _rt.params["cylinder_y_half_width_m"] = float(env_cfg.oa_cylinder_y_half_width_m)
            if hasattr(env_cfg, "oa_rover_spawn_exclusion_radius_m"):
                _rt.params["rover_spawn_exclusion_radius_m"] = float(env_cfg.oa_rover_spawn_exclusion_radius_m)

    # -----------------------------------------------------------------------
    # Rover ObstacleAvoidance task overrides  (Isaac-Rover-DiskWorld-ObstacleAvoidance-v0)
    # -----------------------------------------------------------------------
    if (
        "Rover-DiskWorld-ObstacleAvoidance" in _task_name
        or "rover_disk_world_obstacle_avoidance"
        in (getattr(agent_cfg, "experiment_name", "") or "")
    ):
        _oa_scalar_pairs = (
            ("rover_oa_forward_progress_weight", "oa_forward_progress_weight"),
            (
                "rover_oa_forward_speed_tracking_penalty_weight",
                "oa_forward_speed_tracking_penalty_weight",
            ),
            ("rover_oa_forward_speed_target_ms", "oa_forward_speed_target_ms"),
            ("rover_oa_blue_box_penalty_weight", "oa_blue_box_penalty_weight"),
            ("rover_oa_blue_box_danger_radius_m", "oa_blue_box_danger_radius_m"),
            ("rover_oa_goal_pass_through_weight", "oa_goal_pass_through_weight"),
            ("rover_oa_heading_alignment_weight", "oa_heading_alignment_weight"),
            ("rover_oa_heading_alignment_tolerance_rad", "oa_heading_alignment_tolerance_rad"),
            (
                "rover_oa_radial_z_alignment_reward_weight",
                "oa_radial_z_alignment_reward_weight",
            ),
            (
                "rover_oa_radial_z_alignment_penalty_weight",
                "oa_radial_z_alignment_penalty_weight",
            ),
            ("rover_oa_upside_down_penalty_weight", "oa_upside_down_penalty_weight"),
            ("rover_oa_fell_off_disk_penalty_weight", "oa_fell_off_disk_penalty_weight"),
            ("rover_oa_near_fall_penalty_weight", "oa_near_fall_penalty_weight"),
            ("rover_oa_near_fall_danger_band_m", "oa_near_fall_danger_band_m"),
            (
                "rover_oa_visual_box_ahead_penalty_weight",
                "oa_visual_box_ahead_penalty_weight",
            ),
            (
                "rover_oa_visual_box_ahead_bearing_gate_rad",
                "oa_visual_box_ahead_bearing_gate_rad",
            ),
            (
                "rover_oa_visual_box_ahead_range_gate_m",
                "oa_visual_box_ahead_range_gate_m",
            ),
            (
                "rover_oa_visual_goal_ahead_reward_weight",
                "oa_visual_goal_ahead_reward_weight",
            ),
            (
                "rover_oa_visual_goal_ahead_bearing_gate_rad",
                "oa_visual_goal_ahead_bearing_gate_rad",
            ),
            (
                "rover_oa_visual_goal_ahead_range_gate_m",
                "oa_visual_goal_ahead_range_gate_m",
            ),
            ("rover_oa_goal_approach_shaping_weight", "oa_goal_approach_shaping_weight"),
            ("rover_oa_goal_approach_sigma_m", "oa_goal_approach_sigma_m"),
            ("rover_oa_upside_down_min_radial_dot", "oa_upside_down_min_radial_dot"),
            ("rover_oa_action_rate_penalty_weight", "oa_action_rate_penalty_weight"),
            ("rover_oa_motion_jerk_penalty_weight", "oa_motion_jerk_penalty_weight"),
            (
                "rover_oa_heading_oscillation_penalty_weight",
                "oa_heading_oscillation_penalty_weight",
            ),
            ("rover_oa_alive_bonus_weight", "oa_alive_bonus_weight"),
            ("rover_oa_teacher_imitation_weight", "oa_teacher_imitation_weight"),
            (
                "rover_oa_teacher_imitation_match_sigma",
                "oa_teacher_imitation_match_sigma",
            ),
            ("rover_oa_teacher_action_blend_weight", "oa_teacher_action_blend_weight"),
            ("rover_oa_fall_min_radius_m", "oa_fall_min_radius_m"),
            ("rover_oa_radial_z_penalty_max_abs_m", "oa_radial_z_penalty_max_abs_m"),
        )
        for cli_attr, cfg_attr in _oa_scalar_pairs:
            v = getattr(args_cli, cli_attr, None)
            if v is None:
                continue
            if hasattr(env_cfg, cfg_attr):
                setattr(env_cfg, cfg_attr, float(v))
                print(f"[INFO] Rover obstacle-avoidance: {cfg_attr} = {float(v)}")
            else:
                logger.warning(
                    f"[Rover] env_cfg has no {cfg_attr} – ignoring --{cli_attr}"
                )
        # Pass-through threshold also lives on env_cfg directly (used by env runtime).
        if getattr(args_cli, "rover_oa_goal_pass_threshold_m", None) is not None:
            if hasattr(env_cfg, "goal_pass_threshold_m"):
                env_cfg.goal_pass_threshold_m = float(args_cli.rover_oa_goal_pass_threshold_m)
                print(
                    "[INFO] Rover obstacle-avoidance: goal_pass_threshold_m = "
                    f"{env_cfg.goal_pass_threshold_m}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no goal_pass_threshold_m – ignoring "
                    "--rover_oa_goal_pass_threshold_m"
                )
        _oa_int_pairs = (
            (
                "rover_oa_teacher_imitation_decay_env_steps",
                "oa_teacher_imitation_decay_env_steps",
            ),
            (
                "rover_oa_teacher_action_blend_decay_env_steps",
                "oa_teacher_action_blend_decay_env_steps",
            ),
        )
        for cli_attr, cfg_attr in _oa_int_pairs:
            v = getattr(args_cli, cli_attr, None)
            if v is None:
                continue
            if hasattr(env_cfg, cfg_attr):
                setattr(env_cfg, cfg_attr, int(max(1, v)))
                print(f"[INFO] Rover obstacle-avoidance: {cfg_attr} = {getattr(env_cfg, cfg_attr)}")
            else:
                logger.warning(
                    f"[Rover] env_cfg has no {cfg_attr} – ignoring --{cli_attr}"
                )
        if getattr(args_cli, "rover_oa_teacher_policy_enable", None) is not None:
            if hasattr(env_cfg, "oa_teacher_policy_enable"):
                env_cfg.oa_teacher_policy_enable = bool(int(args_cli.rover_oa_teacher_policy_enable))
                print(
                    "[INFO] Rover obstacle-avoidance: oa_teacher_policy_enable = "
                    f"{env_cfg.oa_teacher_policy_enable}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no oa_teacher_policy_enable – ignoring "
                    "--rover_oa_teacher_policy_enable"
                )
        if getattr(args_cli, "rover_oa_teacher_policy_path", None):
            if hasattr(env_cfg, "oa_teacher_policy_path"):
                env_cfg.oa_teacher_policy_path = str(args_cli.rover_oa_teacher_policy_path)
                env_cfg.drive_policy_path = str(args_cli.rover_oa_teacher_policy_path)
                print(
                    "[INFO] Rover obstacle-avoidance: oa_teacher_policy_path = "
                    f"{env_cfg.oa_teacher_policy_path}"
                )
            else:
                logger.warning(
                    "[Rover] env_cfg has no oa_teacher_policy_path – ignoring "
                    "--rover_oa_teacher_policy_path"
                )
        # -----------------------------------------------------------------------
        # Rover Localization task overrides  (Isaac-Rover-Localization-v0)
        # -----------------------------------------------------------------------
        if "Isaac-Rover-Localization" in (_task_name or ""):
            _loc_expert = getattr(args_cli, "rover_localization_expert_policy_path", None)
            if _loc_expert is not None and hasattr(env_cfg, "expert_policy_path"):
                env_cfg.expert_policy_path = str(_loc_expert)
                print(f"[INFO] Rover Localization: expert_policy_path = {env_cfg.expert_policy_path}")
            _loc_pairs = (
                ("rover_localization_position_error_scale_m", "localization_position_error_scale_m"),
                ("rover_localization_position_error_exp_scale", "localization_position_error_exp_scale"),
                ("rover_localization_position_error_linear_weight", "localization_position_error_linear_weight"),
                ("rover_localization_position_error_exp_weight", "localization_position_error_exp_weight"),
                ("rover_localization_position_bonus_threshold_m", "localization_position_bonus_threshold_m"),
                ("rover_localization_position_bonus_weight", "localization_position_bonus_weight"),
                ("rover_localization_alive_weight", "localization_alive_weight"),
                ("rover_localization_distance_scale_m", "localization_distance_scale_m"),
                ("rover_localization_episode_length_s", "episode_length_s"),
                ("rover_localization_goal_reached_bonus_weight", "flat_goal_reached_bonus_weight"),
                ("rover_localization_goal_reached_threshold_m", "flat_goal_reached_threshold_m"),
                ("rover_flat_goal_relocate_threshold_m", "flat_goal_relocate_threshold_m"),
                ("rover_localization_include_yellow_goal_visual", "oa_flat_policy_include_yellow_goal_visual"),
            )
            for cli_attr, cfg_attr in _loc_pairs:
                v = getattr(args_cli, cli_attr, None)
                if v is None:
                    continue
                if hasattr(env_cfg, cfg_attr):
                    setattr(env_cfg, cfg_attr, float(v))
                    print(f"[INFO] Rover Localization: {cfg_attr} = {getattr(env_cfg, cfg_attr)}")
                else:
                    print(f"[WARNING] Rover Localization: env_cfg has no {cfg_attr} – ignoring --{cli_attr}")
            # Geometry / environment overrides shared with OA-Flat
            _loc_geo_pairs = (
                ("rover_flat_plane_half_size_m", "plane_half_size_m"),
                ("rover_flat_spawn_half_size_m", "spawn_half_size_m"),
                ("rover_flat_target_half_size_m", "target_half_size_m"),
                ("rover_flat_target_exclusion_half_size_m", "target_exclusion_half_size_m"),
                ("rover_flat_target_min_spawn_dist_m", "target_min_spawn_dist_m"),
                ("rover_flat_spawn_z_m", "spawn_z_m"),
                ("rover_oa_flat_box_count", "oa_flat_box_count"),
                ("rover_oa_flat_box_min_pair_clear_m", "oa_flat_box_min_pair_clear_m"),
            )
            for cli_attr, cfg_attr in _loc_geo_pairs:
                v = getattr(args_cli, cli_attr, None)
                if v is None:
                    continue
                if hasattr(env_cfg, cfg_attr):
                    if isinstance(v, int) or cfg_attr in ("oa_flat_box_count",):
                        setattr(env_cfg, cfg_attr, int(v))
                    else:
                        setattr(env_cfg, cfg_attr, float(v))
                    print(f"[INFO] Rover Localization: {cfg_attr} = {getattr(env_cfg, cfg_attr)}")
                else:
                    print(f"[WARNING] Rover Localization: env_cfg has no {cfg_attr} – ignoring --{cli_attr}")
            _loc_debug = getattr(args_cli, "rover_localization_debug_mode", None)
            if _loc_debug is not None and hasattr(env_cfg, "debug_mode"):
                env_cfg.debug_mode = bool(int(_loc_debug))
                print(f"[INFO] Rover Localization: debug_mode = {env_cfg.debug_mode}")
            # Re-mirror localization weights → RewTerm instances after CLI overrides.
            try:
                env_cfg.rewards.position_estimation_reward.weight = 1.0
                env_cfg.rewards.position_estimation_reward.params["position_error_scale_m"] = float(
                    env_cfg.localization_position_error_scale_m
                )
                env_cfg.rewards.position_estimation_reward.params["position_error_exp_scale"] = float(
                    env_cfg.localization_position_error_exp_scale
                )
                env_cfg.rewards.position_estimation_reward.params["position_error_linear_weight"] = float(
                    env_cfg.localization_position_error_linear_weight
                )
                env_cfg.rewards.position_estimation_reward.params["position_error_exp_weight"] = float(
                    env_cfg.localization_position_error_exp_weight
                )
                env_cfg.rewards.position_estimation_bonus.weight = float(
                    env_cfg.localization_position_bonus_weight
                )
                env_cfg.rewards.position_estimation_bonus.params["threshold_m"] = float(
                    env_cfg.localization_position_bonus_threshold_m
                )
                env_cfg.rewards.alive_bonus.weight = float(env_cfg.localization_alive_weight)
                env_cfg.rewards.goal_reached_bonus.weight = float(env_cfg.flat_goal_reached_bonus_weight)
                env_cfg.rewards.goal_reached_bonus.params["threshold_m"] = float(
                    env_cfg.flat_goal_reached_threshold_m
                )
                env_cfg.rewards.goal_reached_bonus.params["set_relocate_pending"] = True
                env_cfg.observations.policy.goal_vec_xy.params["scale_m"] = float(
                    env_cfg.localization_distance_scale_m
                )
                env_cfg.observations.policy.dist_goal.params["scale_m"] = float(
                    env_cfg.localization_distance_scale_m
                )
            except (AttributeError, KeyError) as _exc:
                print(f"[WARNING] Rover Localization: failed to mirror CLI weights to RewTerms: {_exc}")

        # Re-mirror flat weights → RewTerm instances + termination params after CLI.
        # __post_init__ ran before the overrides above, so refresh now.
        try:
            env_cfg.rewards.forward_angular_progress.weight = float(
                env_cfg.oa_forward_progress_weight
            )
            env_cfg.rewards.forward_speed_tracking_penalty.weight = float(
                env_cfg.oa_forward_speed_tracking_penalty_weight
            )
            env_cfg.rewards.forward_speed_tracking_penalty.params["target_tangential_speed"] = float(
                env_cfg.oa_forward_speed_target_ms
            )
            env_cfg.rewards.blue_box_avoidance_penalty.weight = float(
                env_cfg.oa_blue_box_penalty_weight
            )
            env_cfg.rewards.blue_box_avoidance_penalty.params["danger_radius_m"] = float(
                env_cfg.oa_blue_box_danger_radius_m
            )
            env_cfg.rewards.next_goal_pass_through_reward.weight = float(
                env_cfg.oa_goal_pass_through_weight
            )
            env_cfg.rewards.disk_tangent_heading_alignment.weight = float(
                env_cfg.oa_heading_alignment_weight
            )
            env_cfg.rewards.disk_tangent_heading_alignment.params["tolerance_rad"] = float(
                env_cfg.oa_heading_alignment_tolerance_rad
            )
            env_cfg.rewards.radial_z_alignment_reward.weight = float(
                env_cfg.oa_radial_z_alignment_reward_weight
            )
            env_cfg.rewards.radial_z_alignment_penalty.weight = float(
                env_cfg.oa_radial_z_alignment_penalty_weight
            )
            if hasattr(env_cfg.rewards, "radial_z_alignment_penalty"):
                env_cfg.rewards.radial_z_alignment_penalty.params["max_abs_z_error_before_clip_m"] = float(
                    env_cfg.oa_radial_z_penalty_max_abs_m
                )
            env_cfg.rewards.upside_down_termination_penalty.weight = -abs(
                float(env_cfg.oa_upside_down_penalty_weight)
            )
            env_cfg.rewards.fell_off_disk_termination_penalty.weight = -abs(
                float(env_cfg.oa_fell_off_disk_penalty_weight)
            )
            # Dense edge-of-disk safety (mirror CLI weights to RewTerm).
            if hasattr(env_cfg.rewards, "near_fall_radial_penalty"):
                env_cfg.rewards.near_fall_radial_penalty.weight = abs(
                    float(env_cfg.oa_near_fall_penalty_weight)
                )
                env_cfg.rewards.near_fall_radial_penalty.params["fall_radius_m"] = float(
                    env_cfg.oa_fall_min_radius_m
                )
                env_cfg.rewards.near_fall_radial_penalty.params["danger_band_m"] = float(
                    env_cfg.oa_near_fall_danger_band_m
                )
            # Dense visual obstacle / goal shaping.
            if hasattr(env_cfg.rewards, "visual_blue_box_ahead_penalty"):
                env_cfg.rewards.visual_blue_box_ahead_penalty.weight = abs(
                    float(env_cfg.oa_visual_box_ahead_penalty_weight)
                )
                env_cfg.rewards.visual_blue_box_ahead_penalty.params["bearing_gate_rad"] = float(
                    env_cfg.oa_visual_box_ahead_bearing_gate_rad
                )
                env_cfg.rewards.visual_blue_box_ahead_penalty.params["range_gate_m"] = float(
                    env_cfg.oa_visual_box_ahead_range_gate_m
                )
            if hasattr(env_cfg.rewards, "visual_goal_ahead_alignment_reward"):
                env_cfg.rewards.visual_goal_ahead_alignment_reward.weight = float(
                    env_cfg.oa_visual_goal_ahead_reward_weight
                )
                env_cfg.rewards.visual_goal_ahead_alignment_reward.params["bearing_gate_rad"] = float(
                    env_cfg.oa_visual_goal_ahead_bearing_gate_rad
                )
                env_cfg.rewards.visual_goal_ahead_alignment_reward.params["range_gate_m"] = float(
                    env_cfg.oa_visual_goal_ahead_range_gate_m
                )
            if hasattr(env_cfg.rewards, "next_goal_approach_shaping"):
                env_cfg.rewards.next_goal_approach_shaping.weight = float(
                    env_cfg.oa_goal_approach_shaping_weight
                )
                env_cfg.rewards.next_goal_approach_shaping.params["sigma_m"] = float(
                    env_cfg.oa_goal_approach_sigma_m
                )
            env_cfg.rewards.action_rate_penalty.weight = float(
                env_cfg.oa_action_rate_penalty_weight
            )
            env_cfg.rewards.motion_jerk_penalty.weight = float(
                env_cfg.oa_motion_jerk_penalty_weight
            )
            env_cfg.rewards.heading_oscillation_penalty.weight = float(
                env_cfg.oa_heading_oscillation_penalty_weight
            )
            env_cfg.rewards.alive_bonus.weight = float(env_cfg.oa_alive_bonus_weight)
            env_cfg.rewards.teacher_action_imitation.weight = float(
                env_cfg.oa_teacher_imitation_weight
            )
            env_cfg.rewards.teacher_action_imitation.params["decay_env_steps"] = int(
                max(1, env_cfg.oa_teacher_imitation_decay_env_steps)
            )
            env_cfg.rewards.teacher_action_imitation.params["match_sigma"] = float(
                env_cfg.oa_teacher_imitation_match_sigma
            )
            env_cfg.terminations.fell_off_disk.params["min_radius"] = float(
                env_cfg.oa_fall_min_radius_m
            )
            env_cfg.terminations.upside_down.params["min_radial_dot"] = float(
                env_cfg.oa_upside_down_min_radial_dot
            )
        except (AttributeError, KeyError) as _exc:
            logger.warning(
                f"[Rover obstacle-avoidance] Failed to mirror CLI weights to RewTerms: {_exc}"
            )
        # EnvCfg.__post_init__ ran before localization CLI set goal_asset_path; rebuild USD spawns now.
        try:
            from isaaclab.assets import RigidObjectCollectionCfg

            from isaaclab_tasks.manager_based.rover.rover_disk_assets import (
                build_oa_obstacle_rigid_objects_dict,
                default_goal_mesh_path,
            )

            _oa_gp = str(getattr(env_cfg, "goal_asset_path", "") or "").strip()
            if not _oa_gp:
                _oa_gp = default_goal_mesh_path()
                env_cfg.goal_asset_path = _oa_gp
            env_cfg.scene.oa_obstacles = RigidObjectCollectionCfg(
                rigid_objects=build_oa_obstacle_rigid_objects_dict(
                    goal_mesh_path=_oa_gp,
                    box_collision_enabled=bool(getattr(env_cfg, "box_collision_enabled", False)),
                    goal_collision_enabled=bool(getattr(env_cfg, "goal_collision_enabled", False)),
                )
            )
            print(
                "[INFO] Rover obstacle-avoidance: scene.oa_obstacles (boxes + goals) built from "
                f"goal_asset_path={_oa_gp!r}."
            )
        except Exception as _exc:
            logger.warning(f"[Rover obstacle-avoidance] Could not build scene.oa_obstacles: {_exc}")

    # -----------------------------------------------------------------------
    # Rover FlatWorld-GoalNav task overrides  (Isaac-Rover-FlatWorld-GoalNav-v0)
    # -----------------------------------------------------------------------
    if (
        "Rover-FlatWorld" in _task_name
        or "Isaac-Rover-OAFlat" in (_task_name or "")
        or "rover_flat_world" in (getattr(agent_cfg, "experiment_name", "") or "")
        or getattr(agent_cfg, "experiment_name", "") in ("rover_oa_flat_env", "rover_oa_flat_physparam_env")
    ):
        _fg_scalar_pairs = (
            ("rover_flat_distance_to_target_weight", "flat_distance_to_target_weight"),
            ("rover_flat_distance_to_target_scale_m", "flat_distance_to_target_scale_m"),
            ("rover_flat_distance_progress_weight", "flat_distance_progress_weight"),
            ("rover_flat_distance_increase_penalty_weight", "flat_distance_increase_penalty_weight"),
            ("rover_flat_heading_to_target_weight", "flat_heading_to_target_weight"),
            ("rover_flat_heading_goal_closure_weight", "flat_heading_goal_closure_weight"),
            ("rover_flat_yaw_no_closure_penalty_weight", "flat_yaw_no_closure_penalty_weight"),
            ("rover_flat_forward_velocity_to_target_weight", "flat_forward_velocity_to_target_weight"),
            ("rover_flat_forward_velocity_to_target_cap_ms", "flat_forward_velocity_to_target_cap_ms"),
            ("rover_flat_close_to_target_bonus_weight", "flat_close_to_target_bonus_weight"),
            ("rover_flat_close_to_target_threshold_m", "flat_close_to_target_threshold_m"),
            ("rover_flat_goal_reached_bonus_weight", "flat_goal_reached_bonus_weight"),
            ("rover_flat_goal_reached_threshold_m", "flat_goal_reached_threshold_m"),
            ("rover_flat_goal_relocate_threshold_m", "flat_goal_relocate_threshold_m"),
            ("rover_flat_action_rate_weight", "flat_action_rate_weight"),
            ("rover_flat_tilt_weight", "flat_tilt_weight"),
            ("rover_flat_boundary_weight", "flat_boundary_weight"),
            ("rover_flat_boundary_margin_m", "flat_boundary_margin_m"),
            ("rover_flat_four_wheels_ground_contact_weight", "flat_four_wheels_ground_contact_weight"),
            ("rover_flat_front_wheels_ground_contact_weight", "flat_front_wheels_ground_contact_weight"),
            ("rover_flat_rear_wheels_ground_contact_weight", "flat_rear_wheels_ground_contact_weight"),
            (
                "rover_flat_not_four_wheels_ground_contact_penalty_weight",
                "flat_not_four_wheels_ground_contact_penalty_weight",
            ),
            ("rover_flat_wheel_ground_contact_threshold_n", "flat_wheel_ground_contact_threshold_n"),
            ("rover_flat_alive_weight", "flat_alive_weight"),
            ("rover_flat_wheel_effort_limit_sim", "flat_wheel_effort_limit_sim"),
            ("rover_flat_wheel_target_slew_rad_s2", "flat_wheel_target_slew_rad_s2"),
            ("rover_flat_plane_half_size_m", "plane_half_size_m"),
            ("rover_flat_spawn_half_size_m", "spawn_half_size_m"),
            ("rover_flat_target_half_size_m", "target_half_size_m"),
            ("rover_flat_target_exclusion_half_size_m", "target_exclusion_half_size_m"),
            ("rover_flat_target_min_spawn_dist_m", "target_min_spawn_dist_m"),
            ("rover_flat_spawn_z_m", "spawn_z_m"),
            ("rover_flat_upside_down_min_dot", "flat_upside_down_min_dot"),
            ("rover_flat_wheelie_min_pitch_sin", "flat_wheelie_min_pitch_sin"),
        )
        _oa_scalar_pairs = ()
        if "Isaac-Rover-OAFlat" in (_task_name or "") or getattr(agent_cfg, "experiment_name", "") in (
            "rover_oa_flat_env",
            "rover_oa_flat_physparam_env",
        ):
            _oa_scalar_pairs = (
                ("rover_flat_body_reverse_vel_penalty_weight", "flat_body_reverse_vel_penalty_weight"),
                ("rover_flat_body_reverse_vel_scale_ms", "flat_body_reverse_vel_scale_ms"),
                ("rover_oa_flat_box_extent_m", "oa_flat_box_extent_m"),
                ("rover_oa_flat_box_grid_step_m", "oa_flat_box_grid_step_m"),
                ("rover_oa_flat_box_min_spawn_clear_m", "oa_flat_box_min_spawn_clear_m"),
                ("rover_oa_flat_box_min_pair_clear_m", "oa_flat_box_min_pair_clear_m"),
                ("rover_oa_flat_box_half_height_m", "oa_flat_box_half_height_m"),
                ("rover_oa_flat_blue_avoidance_weight", "oa_flat_blue_avoidance_weight"),
                ("rover_oa_flat_blue_avoidance_danger_radius_m", "oa_flat_blue_avoidance_danger_radius_m"),
                ("rover_oa_flat_blue_avoidance_near_ramp_weight", "oa_flat_blue_avoidance_near_ramp_weight"),
                ("rover_oa_flat_blue_avoidance_gaussian_sigma_scale", "oa_flat_blue_avoidance_gaussian_sigma_scale"),
                ("rover_oa_flat_blue_collision_weight", "oa_flat_blue_collision_weight"),
                ("rover_oa_flat_blue_collision_radius_m", "oa_flat_blue_collision_radius_m"),
                ("rover_oa_flat_target_min_blue_box_clearance_m", "oa_flat_target_min_blue_box_clearance_m"),
                ("rover_oa_flat_obstacle_avoidance_reward_scale", "oa_flat_obstacle_avoidance_reward_scale"),
                ("rover_oa_flat_visual_yellow_excess_threshold", "oa_flat_visual_yellow_excess_threshold"),
                ("rover_oa_flat_visual_yellow_conf_sigmoid_slope", "oa_flat_visual_yellow_conf_sigmoid_slope"),
                ("rover_oa_flat_visual_yellow_track_conf_gate", "oa_flat_visual_yellow_track_conf_gate"),
                ("rover_oa_flat_visual_yellow_score_boost", "oa_flat_visual_yellow_score_boost"),
            )
        for _cli_name, _cfg_name in _fg_scalar_pairs + _oa_scalar_pairs:
            _val = getattr(args_cli, _cli_name, None)
            if _val is None:
                continue
            if hasattr(env_cfg, _cfg_name):
                setattr(env_cfg, _cfg_name, float(_val))
                print(f"[INFO] Rover FlatGoal: env_cfg.{_cfg_name} = {float(_val)}")
            else:
                logger.warning(
                    f"[Rover FlatGoal] env_cfg has no '{_cfg_name}' – ignoring --{_cli_name}"
                )
        if "Isaac-Rover-OAFlat" in (_task_name or "") or getattr(agent_cfg, "experiment_name", "") in (
            "rover_oa_flat_env",
            "rover_oa_flat_physparam_env",
        ):
            for _cli_name, _cfg_name in (
                ("rover_oa_flat_box_count", "oa_flat_box_count"),
                ("rover_flat_reset_max_target_samples", "flat_reset_max_target_samples"),
                ("rover_oa_flat_camera_width", "oa_flat_camera_width"),
                ("rover_oa_flat_camera_height", "oa_flat_camera_height"),
                ("rover_oa_flat_visual_yellow_goal_max_detections", "oa_flat_visual_yellow_goal_max_detections"),
            ):
                _ival = getattr(args_cli, _cli_name, None)
                if _ival is None:
                    continue
                if hasattr(env_cfg, _cfg_name):
                    setattr(env_cfg, _cfg_name, int(_ival))
                    print(f"[INFO] Rover OA-Flat: env_cfg.{_cfg_name} = {int(_ival)}")
                else:
                    logger.warning(
                        f"[Rover OA-Flat] env_cfg has no '{_cfg_name}' – ignoring --{_cli_name}"
                    )
            if getattr(args_cli, "rover_oa_flat_box_arena_half_m", None) is not None:
                env_cfg.oa_flat_box_arena_half_m = float(args_cli.rover_oa_flat_box_arena_half_m)
                if hasattr(env_cfg, "oa_flat_box_arena_use_plane"):
                    env_cfg.oa_flat_box_arena_use_plane = False
                print(
                    "[INFO] Rover OA-Flat: oa_flat_box_arena_half_m = "
                    f"{env_cfg.oa_flat_box_arena_half_m} (not synced to plane_half_size_m)"
                )
            if getattr(args_cli, "rover_oa_flat_visual_yellow_goal_enrichment", None) is not None:
                if hasattr(env_cfg, "oa_flat_visual_yellow_goal_enrichment_enabled"):
                    env_cfg.oa_flat_visual_yellow_goal_enrichment_enabled = bool(
                        int(args_cli.rover_oa_flat_visual_yellow_goal_enrichment)
                    )
                    print(
                        "[INFO] Rover OA-Flat: oa_flat_visual_yellow_goal_enrichment_enabled = "
                        f"{env_cfg.oa_flat_visual_yellow_goal_enrichment_enabled}"
                    )
            if getattr(args_cli, "rover_oa_flat_policy_include_yellow_goal_visual", None) is not None:
                if hasattr(env_cfg, "oa_flat_policy_include_yellow_goal_visual"):
                    env_cfg.oa_flat_policy_include_yellow_goal_visual = bool(
                        int(args_cli.rover_oa_flat_policy_include_yellow_goal_visual)
                    )
                    print(
                        "[INFO] Rover OA-Flat: oa_flat_policy_include_yellow_goal_visual = "
                        f"{env_cfg.oa_flat_policy_include_yellow_goal_visual}"
                    )
            if getattr(args_cli, "rover_oa_flat_visual_yellow_score_relu", None) is not None:
                if hasattr(env_cfg, "oa_flat_visual_yellow_score_relu"):
                    env_cfg.oa_flat_visual_yellow_score_relu = bool(
                        int(args_cli.rover_oa_flat_visual_yellow_score_relu)
                    )
                    print(
                        "[INFO] Rover OA-Flat: oa_flat_visual_yellow_score_relu = "
                        f"{env_cfg.oa_flat_visual_yellow_score_relu}"
                    )
            if bool(getattr(env_cfg, "oa_flat_policy_include_yellow_goal_visual", False)):
                from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import sync_oa_flat_policy_yellow_visual_obs_terms

                sync_oa_flat_policy_yellow_visual_obs_terms(env_cfg)
            if getattr(args_cli, "rover_oa_flat_per_env_seed_streams", None) is not None:
                env_cfg.oa_flat_per_env_seed_streams = bool(int(args_cli.rover_oa_flat_per_env_seed_streams))
                print(f"[INFO] Rover OA-Flat: oa_flat_per_env_seed_streams = {env_cfg.oa_flat_per_env_seed_streams}")
            _oa_box_cli_any = any(
                getattr(args_cli, k, None) is not None
                for k in (
                    "rover_oa_flat_box_count",
                    "rover_oa_flat_box_extent_m",
                    "rover_oa_flat_box_arena_half_m",
                    "rover_oa_flat_box_grid_step_m",
                    "rover_oa_flat_box_min_spawn_clear_m",
                    "rover_oa_flat_box_min_pair_clear_m",
                    "rover_oa_flat_box_half_height_m",
                    "rover_oa_flat_target_min_blue_box_clearance_m",
                )
            )
            if _oa_box_cli_any:
                from isaaclab_tasks.manager_based.rover.rover_oa_flat_env_cfg import (
                    refresh_oa_flat_blue_box_assets_from_env_fields,
                )

                refresh_oa_flat_blue_box_assets_from_env_fields(env_cfg)
                print("[INFO] Rover OA-Flat: rebuilt scene.oa_blue_boxes + event box params after CLI overrides.")
        if getattr(args_cli, "rover_flat_episode_length_s", None) is not None:
            env_cfg.episode_length_s = float(args_cli.rover_flat_episode_length_s)
            print(f"[INFO] Rover FlatGoal: episode_length_s = {env_cfg.episode_length_s}")
        if getattr(args_cli, "rover_flat_spawn_target_marker", None) is not None:
            if hasattr(env_cfg, "spawn_target_marker"):
                env_cfg.spawn_target_marker = bool(int(args_cli.rover_flat_spawn_target_marker))
                print(f"[INFO] Rover FlatGoal: spawn_target_marker = {env_cfg.spawn_target_marker}")
        _fg_stage = getattr(args_cli, "rover_flat_goal_stage", None)
        if _fg_stage is not None and hasattr(env_cfg, "flat_goal_train_stage"):
            if int(_fg_stage) in (0, 1, 2):
                env_cfg.flat_goal_train_stage = int(_fg_stage)
                print(f"[INFO] Rover FlatGoal: flat_goal_train_stage = {env_cfg.flat_goal_train_stage}")
            else:
                logger.warning("[Rover FlatGoal] rover_flat_goal_stage must be 0, 1, or 2; ignoring %s", _fg_stage)
        _fg_note = getattr(args_cli, "rover_flat_goal_stage_note", None)
        if _fg_note is not None and hasattr(env_cfg, "flat_goal_train_stage_note"):
            env_cfg.flat_goal_train_stage_note = str(_fg_note)[:240]
            print(f"[INFO] Rover FlatGoal: flat_goal_train_stage_note set ({len(env_cfg.flat_goal_train_stage_note)} chars)")
        if "Isaac-Rover-OAFlat" in (_task_name or "") or getattr(agent_cfg, "experiment_name", "") in (
            "rover_oa_flat_env",
            "rover_oa_flat_physparam_env",
        ):
            _rvp = float(getattr(env_cfg, "flat_body_reverse_vel_penalty_weight", 0.15))
            if _rvp > 0.65:
                logger.warning(
                    "[Rover OA-Flat] flat_body_reverse_vel_penalty_weight=%s is very high; "
                    "clamping to 0.65 (extreme values can block useful skid-steer rotation).",
                    _rvp,
                )
                env_cfg.flat_body_reverse_vel_penalty_weight = 0.65
        # Re-run the cfg's __post_init__ so the scalar fields above get mirrored
        # into the RewTerm / DoneTerm instances (same pattern as the env_cfg defaults).
        try:
            env_cfg.__post_init__()
        except Exception as _e_fg_post:
            logger.warning(
                f"[Rover FlatGoal] Could not re-run env_cfg.__post_init__ after CLI overrides: {_e_fg_post}"
            )

    # -----------------------------------------------------------------------
    _write_rover_localization_train_debug_snap(
        log_dir=log_dir,
        stage="after_env_cfg_overrides",
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        args_cli=args_cli,
    )

    if args_cli.action_rate_penalty_scale is not None:
        if hasattr(env_cfg, 'action_rate_penalty_scale'):
            env_cfg.action_rate_penalty_scale = args_cli.action_rate_penalty_scale
            print(f"[INFO] Action rate penalty scale set to: {env_cfg.action_rate_penalty_scale}")
    
    if args_cli.joint_acceleration_penalty_scale is not None:
        if hasattr(env_cfg, 'joint_acceleration_penalty_scale'):
            env_cfg.joint_acceleration_penalty_scale = args_cli.joint_acceleration_penalty_scale
            print(f"[INFO] Joint acceleration penalty scale set to: {env_cfg.joint_acceleration_penalty_scale}")
    
    if args_cli.stability_weight is not None:
        if hasattr(env_cfg, 'stability_weight'):
            env_cfg.stability_weight = args_cli.stability_weight
            print(f"[INFO] Stability weight set to: {env_cfg.stability_weight}")
        else:
            logger.warning(f"Config does not have 'stability_weight' attribute. Ignoring --stability_weight argument.")
    
    if args_cli.balance_weight is not None:
        if hasattr(env_cfg, 'balance_weight'):
            env_cfg.balance_weight = args_cli.balance_weight
            print(f"[INFO] Balance weight set to: {env_cfg.balance_weight}")
        else:
            logger.warning(f"Config does not have 'balance_weight' attribute. Ignoring --balance_weight argument.")
    
    if bool(getattr(args_cli, "progress_forward_small", False)):
        _pf_small_defaults = (
            ("progress_reward_multiplier", 0.28),
            ("progress_reward_world_vx_blend", 0.55),
            ("progress_reward_forward_displacement_weight", 0.38),
            ("progress_reward_forward_displacement_threshold_m", 0.001),
        )
        for name, dval in _pf_small_defaults:
            if getattr(args_cli, name, None) is None:
                setattr(args_cli, name, dval)
        print(
            "[INFO] progress_forward_small: applied defaults for "
            "progress_reward_multiplier / world_vx_blend / forward_displacement_* (only where CLI omitted)."
        )

    # G1 locomotion V1 / Walking V2 (and other direct locomotion envs with same attributes)
    for attr_name in (
        "up_weight", "heading_weight", "actions_cost_scale", "energy_cost_scale",
        "alive_reward_scale", "progress_reward_multiplier", "progress_reward_world_vx_blend",
        "progress_reward_forward_displacement_weight", "progress_reward_forward_displacement_threshold_m",
        "target_velocity", "velocity_reward_weight", "velocity_reward_sharpness", "target_x_velocity_penalty_weight", "termination_height",
        "y_drift_tolerance", "y_drift_penalty_weight",
        "y_drift_reward_weight", "y_drift_reward_sigma",
        "x_drift_reward_weight", "x_drift_reward_sigma",
        "target_z_orientation", "z_orientation_tolerance", "z_orientation_penalty_weight",
        "z_orientation_reward_weight", "z_orientation_reward_sharpness",
        "alternating_foot_reward_weight", "same_foot_tap_penalty_weight",
        "jump_penalty_weight", "at_least_one_foot_contact_reward_weight",
        "ankle_roll_feet_target_separation_m",
        "ankle_roll_feet_separation_tolerance_m",
        "ankle_roll_feet_separation_reward_weight",
        "ankle_roll_feet_separation_penalty_weight",
        "ankle_roll_foot_height_reward_target_m",
        "ankle_roll_foot_height_reward_weight",
        "symmetry_reward_weight", "symmetry_penalty_weight", "symmetry_contact_sharpness",
        "symmetry_touchdown_sharpness", "symmetry_torque_sharpness", "symmetry_include_hip_torque",
        "symmetry_torque_terms_double_support_only",
        "symmetry_use_geometric_mean",
        "step_length_min_m", "step_length_reward_weight", "step_length_penalty_weight",
        "step_length_use_smooth_reward", "step_length_smooth_sigma_m", "step_length_penalty_squared",
        "foot_crossing_reward_weight", "foot_crossing_penalty_weight",
        "same_foot_tap_min_air_steps", "gait_single_stance_reward_weight",
        "hip_weight_shift_reward_weight", "hip_weight_shift_torque_scale",
        "alternating_foot_scale_with_forward_vel", "alternating_foot_vel_scale_ref",
        "lin_vel_z_penalty_weight", "lin_vel_z_tolerance", "ang_vel_xy_penalty_weight", "flat_orientation_penalty_weight", "flat_orientation_tolerance",
        "base_height_reference",
        "base_height_reward_weight",
        "base_height_sigma",
        "base_height_penalty_weight",
        "base_height_penalty_tolerance_m",
        "curriculum_stage1_steps", "curriculum_stage2_steps", "num_steps_per_env",
        "enable_curriculum", "enable_push_perturbation", "max_push_vel_xy",
        "episode_start_force_impulse_enabled", "episode_start_force_impulse_duration_steps",
        "episode_start_force_impulse_max_force_xy_n", "episode_start_force_impulse_max_force_z_n",
        "imitation_legs_weight", "imitation_arms_weight",
        "include_root_xy_error_obs", "root_xy_error_obs_scale",
        "imitation_phase_in_obs",
        "imitation_pelvis_link_error_in_obs", "imitation_root_lin_vel_in_obs",
        "imitation_root_lin_vel_obs_scale", "imitation_joint_position_use_leg_arm_weights",
        "imitation_joint_velocity_use_leg_arm_weights",
        "imitation_strength_ramp_steps", "imitation_strength_ramp_easing",
        "imitation_velocity_sigma_rad_s", "motion_cycle_episode_margin_s",
        "hdf5_link_use_joint_motion_clock_when_aligned", "hdf5_link_local_imitation_use_body_lin_vel",
        "hdf5_derive_foot_contact_from_link_z",
        "hdf5_imitation_periodic_joint_vel_fd",
        "hdf5_joint_position_data_scale_ramp_enabled",
        "hdf5_joint_position_data_scale_ramp_start",
        "hdf5_joint_position_data_scale_ramp_end",
        "hdf5_joint_position_data_scale_ramp_steps",
        "hdf5_joint_position_data_scale_ramp_delay_steps",
        "hdf5_joint_position_data_scale_ramp_in_ppo_iterations",
        "hdf5_joint_position_data_scale_mode",
        "hdf5_joint_position_data_scale_exclude_ankles",
        "hdf5_imitation_reward_episode_time_c",
        "hdf5_imitation_reward_episode_time_exponent",
        "hdf5_link_local_imitation_periodic_fd",
        "hdf5_link_local_stance_tolerance_scale", "hdf5_link_local_stance_ref_contact_threshold",
        "hdf5_link_local_imitation_position_reward_penalty_xy_only",
        "hdf5_link_local_imitation_swing_penalty_mask", "hdf5_link_local_imitation_swing_ref_contact_threshold",
        "imitation_joint_ref_contact_leg_scale",
        "imitation_root_xy_reward_weight", "imitation_root_xy_penalty_weight", "imitation_root_xy_sigma_m",
        "swing_foot_clearance_reward_weight", "swing_foot_clearance_penalty_weight",
        "swing_foot_clearance_sigma_m", "swing_foot_clearance_margin_m",
        "stance_foot_slip_penalty_weight",
        # G1 Locomotion V4
        "reward_profile",
        "staged_imitation_start_optimizer_steps",
        "staged_imitation_ramp_optimizer_steps",
        "staged_ramp_start_scale",
        "staged_dybl_start_optimizer_steps",
        "staged_dybl_ramp_optimizer_steps",
        "staged_contact_gait_start_optimizer_steps",
        "staged_contact_gait_ramp_optimizer_steps",
        "staged_symmetry_start_optimizer_steps",
        "staged_symmetry_ramp_optimizer_steps",
        "staged_symmetry_ramp_start_scale",
        "velocity_command_obs_enabled",
        "velocity_command_obs_scale",
        "include_reward_profile_phase_obs",
        "target_yaw_command_obs_enabled",
        "append_teleop_hand_target_obs",
        "gravity_scale",
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
        "torso_z_align_link_body_name",
        "torso_z_axis_world_align_reward_weight",
        "torso_z_axis_world_align_reward_sigma",
        "torso_z_axis_world_align_penalty_weight",
        "torso_z_axis_world_align_penalty_tolerance",
        "palm_below_pelvis_right_body_name",
        "palm_below_pelvis_left_body_name",
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
        # Ion vertical takeoff V2 (ignored if env cfg has no attribute)
        "staged_stability_start_env_steps",
        "staged_stability_ramp_env_steps",
        "target_vz_command_obs_enabled",
        "vz_command_obs_scale",
        "target_vx_command_obs_enabled",
        "vx_command_obs_scale",
        "staged_ik_start_env_steps",
        "staged_ik_ramp_env_steps",
    ):
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None and hasattr(env_cfg, attr_name):
            setattr(env_cfg, attr_name, arg_value)
            print(f"[INFO] env cfg: {attr_name} set to: {arg_value}")

    if getattr(args_cli, "hdf5_imitation_reward_episode_time_scale_enabled", None) is not None and hasattr(
        env_cfg, "hdf5_imitation_reward_episode_time_scale_enabled"
    ):
        env_cfg.hdf5_imitation_reward_episode_time_scale_enabled = bool(
            int(args_cli.hdf5_imitation_reward_episode_time_scale_enabled)
        )
        print(
            "[INFO] env cfg: hdf5_imitation_reward_episode_time_scale_enabled set to: "
            f"{env_cfg.hdf5_imitation_reward_episode_time_scale_enabled}"
        )

    if getattr(args_cli, "hdf5_joint_position_data_scale_ramp_enabled", None) is not None and hasattr(
        env_cfg, "hdf5_joint_position_data_scale_ramp_enabled"
    ):
        env_cfg.hdf5_joint_position_data_scale_ramp_enabled = bool(
            int(args_cli.hdf5_joint_position_data_scale_ramp_enabled)
        )
        print(
            "[INFO] env cfg: hdf5_joint_position_data_scale_ramp_enabled set to: "
            f"{env_cfg.hdf5_joint_position_data_scale_ramp_enabled}"
        )
    if getattr(args_cli, "hdf5_joint_position_data_scale_ramp_in_ppo_iterations", None) is not None and hasattr(
        env_cfg, "hdf5_joint_position_data_scale_ramp_in_ppo_iterations"
    ):
        env_cfg.hdf5_joint_position_data_scale_ramp_in_ppo_iterations = bool(
            int(args_cli.hdf5_joint_position_data_scale_ramp_in_ppo_iterations)
        )
        print(
            "[INFO] env cfg: hdf5_joint_position_data_scale_ramp_in_ppo_iterations set to: "
            f"{env_cfg.hdf5_joint_position_data_scale_ramp_in_ppo_iterations}"
        )
    for _ramp_attr in ("hdf5_joint_position_data_scale_ramp_start", "hdf5_joint_position_data_scale_ramp_end"):
        _rv = getattr(args_cli, _ramp_attr, None)
        if _rv is not None and hasattr(env_cfg, _ramp_attr):
            setattr(env_cfg, _ramp_attr, float(_rv))
            print(f"[INFO] env cfg: {_ramp_attr} set to: {getattr(env_cfg, _ramp_attr)}")
    for _ramp_attr in ("hdf5_joint_position_data_scale_ramp_steps", "hdf5_joint_position_data_scale_ramp_delay_steps"):
        _rv = getattr(args_cli, _ramp_attr, None)
        if _rv is not None and hasattr(env_cfg, _ramp_attr):
            setattr(env_cfg, _ramp_attr, int(_rv))
            print(f"[INFO] env cfg: {_ramp_attr} set to: {getattr(env_cfg, _ramp_attr)}")
    if getattr(args_cli, "hdf5_joint_position_data_scale_mode", None) is not None and hasattr(
        env_cfg, "hdf5_joint_position_data_scale_mode"
    ):
        env_cfg.hdf5_joint_position_data_scale_mode = str(args_cli.hdf5_joint_position_data_scale_mode).strip()
        print(f"[INFO] env cfg: hdf5_joint_position_data_scale_mode = {env_cfg.hdf5_joint_position_data_scale_mode}")
    if getattr(args_cli, "hdf5_joint_position_data_scale_exclude_ankles", None) is not None and hasattr(
        env_cfg, "hdf5_joint_position_data_scale_exclude_ankles"
    ):
        env_cfg.hdf5_joint_position_data_scale_exclude_ankles = bool(
            int(args_cli.hdf5_joint_position_data_scale_exclude_ankles)
        )
        print(
            "[INFO] env cfg: hdf5_joint_position_data_scale_exclude_ankles = "
            f"{env_cfg.hdf5_joint_position_data_scale_exclude_ankles}"
        )

    # G1 Locomotion V3: adapter phase (reward ramp after resume / reward reshape)
    if hasattr(env_cfg, "adapter_phase_enabled"):
        if getattr(args_cli, "adapter_phase", False):
            env_cfg.adapter_phase_enabled = True
            print("[INFO] G1 locomotion V3: adapter_phase_enabled = True")
        if getattr(args_cli, "adapter_phase_duration", None) is not None:
            env_cfg.adapter_phase_duration_steps = int(args_cli.adapter_phase_duration)
            print(f"[INFO] G1 locomotion V3: adapter_phase_duration_steps = {env_cfg.adapter_phase_duration_steps}")
        if getattr(args_cli, "adapter_phase_start_scale", None) is not None:
            env_cfg.adapter_phase_start_scale = float(args_cli.adapter_phase_start_scale)
            print(f"[INFO] G1 locomotion V3: adapter_phase_start_scale = {env_cfg.adapter_phase_start_scale}")
        if getattr(args_cli, "adapter_phase_easing", None) is not None:
            env_cfg.adapter_phase_easing = str(args_cli.adapter_phase_easing)
            print(f"[INFO] G1 locomotion V3: adapter_phase_easing = {env_cfg.adapter_phase_easing}")

    _g1_walk_obs_sync_tasks = (
        "Isaac-G1-Locomotion-V1-Walking-Direct-v0",
        "Isaac-G1-Locomotion-Walking-V2-Direct-v0",
        "Isaac-G1-Locomotion-V3-Direct-v0",
    )
    if getattr(args_cli, "task", None) in _g1_walk_obs_sync_tasks and hasattr(env_cfg, "observation_space"):
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v1_env_cfg import compute_g1_locomotion_policy_obs_dim

        env_cfg.observation_space = compute_g1_locomotion_policy_obs_dim(env_cfg)
        print(f"[INFO] G1 locomotion walking: observation_space synced to {env_cfg.observation_space} (policy obs toggles)")
    if getattr(args_cli, "task", None) == "Isaac-G1-Locomotion-V4-Direct-v0" and hasattr(env_cfg, "observation_space"):
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v4_env_cfg import compute_g1_locomotion_v4_policy_obs_dim

        env_cfg.observation_space = compute_g1_locomotion_v4_policy_obs_dim(env_cfg)
        print(
            f"[INFO] G1 locomotion V4: observation_space={env_cfg.observation_space} from cfg/CLI "
            f"(with --resume, checkpoint obs width may adjust toggles before gym.make)"
        )

    if (
        getattr(args_cli, "task", None) == "Isaac-G1-Locomotion-V4-Direct-v0"
        and getattr(args_cli, "progress_goal_world_offset_xy_m", None) is not None
        and hasattr(env_cfg, "progress_goal_world_offset_xy_m")
    ):
        s = str(args_cli.progress_goal_world_offset_xy_m).strip()
        slow = s.lower()
        if slow in ("none", "null", ""):
            env_cfg.progress_goal_world_offset_xy_m = None
            print("[INFO] G1 V4: progress_goal_world_offset_xy_m = None (V1 default targets, no V4 override).")
        else:
            try:
                parts = [float(x.strip()) for x in s.split(",")]
                if len(parts) != 2:
                    raise ValueError("need exactly 2 numbers: forward_m,lateral_m")
                env_cfg.progress_goal_world_offset_xy_m = (parts[0], parts[1])
                print(f"[INFO] G1 V4: progress_goal_world_offset_xy_m = {env_cfg.progress_goal_world_offset_xy_m}")
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid --progress_goal_world_offset_xy_m {s!r}: {e}. Use 'x,y' in meters (e.g. '1000,0') or 'none'."
                )

    if getattr(args_cli, "task", None) == "Isaac-Ion-Vertical-Takeoff-V2-Direct-v0" and hasattr(env_cfg, "observation_space"):
        from isaaclab_tasks.direct.ion_vertical_takeoff.ion_vertical_takeoff_v2_env_cfg import (
            compute_ion_vertical_takeoff_v2_policy_obs_dim,
        )

        env_cfg.observation_space = compute_ion_vertical_takeoff_v2_policy_obs_dim(env_cfg)
        print(f"[INFO] Ion vertical takeoff V2: observation_space = {env_cfg.observation_space} (command/phase obs)")

    if getattr(args_cli, "task", None) == "Isaac-Ion-Horizontal-Cruise-V2-Direct-v0" and hasattr(env_cfg, "observation_space"):
        from isaaclab_tasks.direct.ion_horizontal_cruise_v2.ion_horizontal_cruise_v2_env_cfg import (
            compute_ion_horizontal_cruise_v2_policy_obs_dim,
        )

        env_cfg.observation_space = compute_ion_horizontal_cruise_v2_policy_obs_dim(env_cfg)
        print(f"[INFO] Ion horizontal cruise V2: observation_space = {env_cfg.observation_space} (command/phase obs)")

    if getattr(args_cli, "task", None) == "Isaac-G1-Standing-IK-V2-Direct-v0" and hasattr(env_cfg, "observation_space"):
        from isaaclab_tasks.direct.g1_locomotion_v1.g1_standing_ik_v2_env_cfg import compute_g1_standing_ik_v2_policy_obs_dim

        env_cfg.observation_space = compute_g1_standing_ik_v2_policy_obs_dim(env_cfg)
        print(f"[INFO] G1 standing IK V2: observation_space = {env_cfg.observation_space} (optional phase obs)")

    # G1 standing hand position V1: hand target in pelvis frame (x,y,z)
    if getattr(args_cli, "hand_target_pos_b", None) is not None and hasattr(env_cfg, "hand_target_pos_b"):
        s = args_cli.hand_target_pos_b.strip()
        try:
            parts = [float(x.strip()) for x in s.split(",")]
            if len(parts) != 3:
                raise ValueError("hand_target_pos_b must have exactly 3 values: x,y,z")
            env_cfg.hand_target_pos_b = (float(parts[0]), float(parts[1]), float(parts[2]))
            print(f"[INFO] G1 locomotion / standing hand position: hand_target_pos_b set to: {env_cfg.hand_target_pos_b}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid --hand_target_pos_b '{s}': {e}. Use 'x,y,z' in meters (e.g. '0.12,-0.2,-0.08').")

    if getattr(args_cli, "hand_reward_weight", None) is not None and hasattr(env_cfg, "hand_reward_weight"):
        env_cfg.hand_reward_weight = args_cli.hand_reward_weight
        print(f"[INFO] G1 locomotion standing: hand_reward_weight set to: {env_cfg.hand_reward_weight}")
    if getattr(args_cli, "hand_reward_sigma", None) is not None and hasattr(env_cfg, "hand_reward_sigma"):
        env_cfg.hand_reward_sigma = args_cli.hand_reward_sigma
        print(f"[INFO] G1 locomotion standing: hand_reward_sigma set to: {env_cfg.hand_reward_sigma}")

    if getattr(args_cli, "xy_drift_tolerance", None) is not None and hasattr(env_cfg, "xy_drift_tolerance"):
        env_cfg.xy_drift_tolerance = args_cli.xy_drift_tolerance
        print(f"[INFO] G1 standing hand position: xy_drift_tolerance set to: {env_cfg.xy_drift_tolerance}")
    if getattr(args_cli, "xy_drift_penalty_weight", None) is not None and hasattr(env_cfg, "xy_drift_penalty_weight"):
        env_cfg.xy_drift_penalty_weight = args_cli.xy_drift_penalty_weight
        print(f"[INFO] G1 standing hand position: xy_drift_penalty_weight set to: {env_cfg.xy_drift_penalty_weight}")

    # G1 standing hand position V2: separate X and Y drift penalties
    if getattr(args_cli, "x_drift_tolerance", None) is not None and hasattr(env_cfg, "x_drift_tolerance"):
        env_cfg.x_drift_tolerance = args_cli.x_drift_tolerance
        print(f"[INFO] G1 standing hand position V2: x_drift_tolerance set to: {env_cfg.x_drift_tolerance}")
    if getattr(args_cli, "x_drift_penalty_weight", None) is not None and hasattr(env_cfg, "x_drift_penalty_weight"):
        env_cfg.x_drift_penalty_weight = args_cli.x_drift_penalty_weight
        print(f"[INFO] G1 standing hand position V2: x_drift_penalty_weight set to: {env_cfg.x_drift_penalty_weight}")
    if getattr(args_cli, "y_drift_tolerance", None) is not None and hasattr(env_cfg, "y_drift_tolerance"):
        env_cfg.y_drift_tolerance = args_cli.y_drift_tolerance
        print(f"[INFO] G1 standing hand position V2: y_drift_tolerance set to: {env_cfg.y_drift_tolerance}")
    if getattr(args_cli, "y_drift_penalty_weight", None) is not None and hasattr(env_cfg, "y_drift_penalty_weight"):
        env_cfg.y_drift_penalty_weight = args_cli.y_drift_penalty_weight
        print(f"[INFO] G1 standing hand position V2: y_drift_penalty_weight set to: {env_cfg.y_drift_penalty_weight}")
    for attr in (
        "x_drift_reward_weight",
        "y_drift_reward_weight",
        "x_drift_reward_sigma",
        "y_drift_reward_sigma",
    ):
        if getattr(args_cli, attr, None) is not None and hasattr(env_cfg, attr):
            setattr(env_cfg, attr, getattr(args_cli, attr))
            print(f"[INFO] G1 drift reward: {attr} set to: {getattr(env_cfg, attr)}")

    # G1 standing hand position V2 / G1 teleoperation / g1_standing_ik: IK target motion and IK reward/penalty
    for attr in (
        "ik_target_motion_speed", "ik_target_motion_period_base",
        "ik_joint_reward_weight", "ik_joint_reward_sigma", "ik_joint_penalty_weight", "ik_joint_penalty_threshold",
        "ik_joint_tolerance_percent", "ee_position_reward_weight", "ee_position_penalty_weight", "ee_position_reward_sigma",
        "ik_ee_smoothness_penalty_weight",
        "ik_target_obs_scale",
        "teleop_target_obs_scale",
    ):
        if getattr(args_cli, attr, None) is not None and hasattr(env_cfg, attr):
            setattr(env_cfg, attr, getattr(args_cli, attr))
            print(f"[INFO] G1 standing hand position V2: {attr} set to: {getattr(env_cfg, attr)}")

    for box_cli, box_cfg in (
        ("ik_target_box_min", "ik_target_box_min"),
        ("ik_target_box_max", "ik_target_box_max"),
    ):
        if getattr(args_cli, box_cli, None) is not None and hasattr(env_cfg, box_cfg):
            s = str(getattr(args_cli, box_cli)).strip()
            try:
                parts = [float(x.strip()) for x in s.split(",")]
                if len(parts) != 3:
                    raise ValueError("need exactly 3 comma-separated values x,y,z")
                setattr(env_cfg, box_cfg, (float(parts[0]), float(parts[1]), float(parts[2])))
                print(f"[INFO] G1 standing IK: {box_cfg} set to: {getattr(env_cfg, box_cfg)}")
            except (ValueError, TypeError) as e:
                logger.warning("Invalid --%s %r: %s. Use 'x,y,z' in meters.", box_cli, s, e)

    if getattr(args_cli, "g1_standing_fixed_base", False):
        if hasattr(env_cfg, "spawn_fixed_base"):
            env_cfg.spawn_fixed_base = True
        if hasattr(env_cfg, "robot") and hasattr(env_cfg.robot.spawn, "articulation_props"):
            env_cfg.robot.spawn.articulation_props.fix_root_link = True
            print("[INFO] G1 standing IK: fixed base spawn (fix_root_link=True)")

    if args_cli.x_axis_drift_penalty_weight is not None:
        if hasattr(env_cfg, 'x_axis_drift_penalty_weight'):
            env_cfg.x_axis_drift_penalty_weight = args_cli.x_axis_drift_penalty_weight
            print(f"[INFO] X-axis drift penalty weight set to: {env_cfg.x_axis_drift_penalty_weight}")
        else:
            logger.warning(f"Config does not have 'x_axis_drift_penalty_weight' attribute. Ignoring --x_axis_drift_penalty_weight argument.")
    
    if args_cli.y_axis_drift_penalty_weight is not None:
        if hasattr(env_cfg, 'y_axis_drift_penalty_weight'):
            env_cfg.y_axis_drift_penalty_weight = args_cli.y_axis_drift_penalty_weight
            print(f"[INFO] Y-axis drift penalty weight set to: {env_cfg.y_axis_drift_penalty_weight}")
        else:
            logger.warning(f"Config does not have 'y_axis_drift_penalty_weight' attribute. Ignoring --y_axis_drift_penalty_weight argument.")
    
    if args_cli.z_axis_rotation_penalty_weight is not None:
        if hasattr(env_cfg, 'z_axis_rotation_penalty_weight'):
            env_cfg.z_axis_rotation_penalty_weight = args_cli.z_axis_rotation_penalty_weight
            print(f"[INFO] Z-axis rotation penalty weight set to: {env_cfg.z_axis_rotation_penalty_weight}")
        else:
            logger.warning(f"Config does not have 'z_axis_rotation_penalty_weight' attribute. Ignoring --z_axis_rotation_penalty_weight argument.")
    
    if args_cli.x_axis_alignment_reward_weight is not None:
        if hasattr(env_cfg, 'x_axis_alignment_reward_weight'):
            env_cfg.x_axis_alignment_reward_weight = args_cli.x_axis_alignment_reward_weight
            print(f"[INFO] X-axis alignment reward weight set to: {env_cfg.x_axis_alignment_reward_weight}")
        else:
            logger.warning(f"Config does not have 'x_axis_alignment_reward_weight' attribute. Ignoring --x_axis_alignment_reward_weight argument.")
    
    if args_cli.feet_distance_reward_weight is not None:
        if hasattr(env_cfg, 'feet_distance_reward_weight'):
            env_cfg.feet_distance_reward_weight = args_cli.feet_distance_reward_weight
            print(f"[INFO] Feet distance reward weight set to: {env_cfg.feet_distance_reward_weight}")
        else:
            logger.warning(f"Config does not have 'feet_distance_reward_weight' attribute. Ignoring --feet_distance_reward_weight argument.")
    
    if args_cli.feet_on_ground_reward_weight is not None:
        if hasattr(env_cfg, 'feet_on_ground_reward_weight'):
            env_cfg.feet_on_ground_reward_weight = args_cli.feet_on_ground_reward_weight
            print(f"[INFO] Feet on ground reward weight set to: {env_cfg.feet_on_ground_reward_weight}")
        else:
            logger.warning(f"Config does not have 'feet_on_ground_reward_weight' attribute. Ignoring --feet_on_ground_reward_weight argument.")
    
    if args_cli.torso_pitch_forward_reward_weight is not None:
        if hasattr(env_cfg, 'torso_pitch_forward_reward_weight'):
            env_cfg.torso_pitch_forward_reward_weight = args_cli.torso_pitch_forward_reward_weight
            print(f"[INFO] Torso pitch forward reward weight set to: {env_cfg.torso_pitch_forward_reward_weight}")
        else:
            logger.warning(f"Config does not have 'torso_pitch_forward_reward_weight' attribute. Ignoring --torso_pitch_forward_reward_weight argument.")

    if args_cli.joint_limit_penalty_weight is not None:
        if hasattr(env_cfg, 'joint_limit_penalty_weight'):
            env_cfg.joint_limit_penalty_weight = args_cli.joint_limit_penalty_weight
            print(f"[INFO] Joint limit penalty weight set to: {env_cfg.joint_limit_penalty_weight}")
        else:
            logger.warning(f"Config does not have 'joint_limit_penalty_weight' attribute. Ignoring --joint_limit_penalty_weight argument.")
    if getattr(args_cli, "joint_limit_reward_weight", None) is not None:
        if hasattr(env_cfg, "joint_limit_reward_weight"):
            env_cfg.joint_limit_reward_weight = args_cli.joint_limit_reward_weight
            print(f"[INFO] Joint limit reward weight set to: {env_cfg.joint_limit_reward_weight}")
        else:
            logger.warning(
                "Config does not have 'joint_limit_reward_weight' attribute. Ignoring --joint_limit_reward_weight argument."
            )
    if getattr(args_cli, "joint_limit_reward_sigma", None) is not None:
        if hasattr(env_cfg, "joint_limit_reward_sigma"):
            env_cfg.joint_limit_reward_sigma = args_cli.joint_limit_reward_sigma
            print(f"[INFO] Joint limit reward sigma set to: {env_cfg.joint_limit_reward_sigma}")
        else:
            logger.warning(
                "Config does not have 'joint_limit_reward_sigma' attribute. Ignoring --joint_limit_reward_sigma argument."
            )
    
    if args_cli.joint_limit_penalty_exponent is not None:
        if hasattr(env_cfg, 'joint_limit_penalty_exponent'):
            env_cfg.joint_limit_penalty_exponent = args_cli.joint_limit_penalty_exponent
            print(f"[INFO] Joint limit penalty exponent set to: {env_cfg.joint_limit_penalty_exponent}")
        else:
            logger.warning(f"Config does not have 'joint_limit_penalty_exponent' attribute. Ignoring --joint_limit_penalty_exponent argument.")

    if getattr(args_cli, "dof_at_limit_cost_scale", None) is not None:
        if hasattr(env_cfg, "dof_at_limit_cost_scale"):
            env_cfg.dof_at_limit_cost_scale = args_cli.dof_at_limit_cost_scale
            print(f"[INFO] dof_at_limit_cost_scale set to: {env_cfg.dof_at_limit_cost_scale}")
        else:
            logger.warning(
                "Config does not have 'dof_at_limit_cost_scale' attribute. Ignoring --dof_at_limit_cost_scale argument."
            )

    if args_cli.npz_joint_limit_extend_deg is not None:
        if hasattr(env_cfg, "npz_joint_limit_extend_deg"):
            env_cfg.npz_joint_limit_extend_deg = args_cli.npz_joint_limit_extend_deg
            print(f"[INFO] HDF5/motion-clip joint limit extend (deg) set to: {env_cfg.npz_joint_limit_extend_deg}")
        else:
            logger.warning(
                "Config does not have 'npz_joint_limit_extend_deg' attribute. Ignoring --hdf5_joint_limit_extend_deg / --npz_joint_limit_extend_deg argument."
            )

    if getattr(args_cli, "npz_joint_limit_extend_deg_non_leg", None) is not None:
        if hasattr(env_cfg, "npz_joint_limit_extend_deg_non_leg"):
            env_cfg.npz_joint_limit_extend_deg_non_leg = args_cli.npz_joint_limit_extend_deg_non_leg
            print(f"[INFO] HDF5/motion-clip joint limit extend non-leg (deg) set to: {env_cfg.npz_joint_limit_extend_deg_non_leg}")
        else:
            logger.warning(
                "Config does not have 'npz_joint_limit_extend_deg_non_leg' attribute. Ignoring --hdf5_joint_limit_extend_deg_non_leg / --npz_joint_limit_extend_deg_non_leg argument."
            )

    if getattr(args_cli, "npz_limits_exclude_four_ankles", False) and hasattr(env_cfg, "npz_limits_exclude_four_ankles"):
        env_cfg.npz_limits_exclude_four_ankles = True
        print("[INFO] Motion-clip joint limits: hdf5_limits_exclude_four_ankles=True (ankles not in soft limit penalty)")

    # G1 Locomotion Walking V2 / V3: motion clip imitation (position + velocity)
    for attr_name in (
        "imitation_joints_position_reward_weight",
        "imitation_joints_position_penalty_weight",
        "imitation_joints_position_reward_sharpness",
        "imitation_velocity_reward_weight",
        "imitation_velocity_penalty_weight",
    ):
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None and hasattr(env_cfg, attr_name):
            setattr(env_cfg, attr_name, arg_value)
            print(f"[INFO] G1 Locomotion imitation: {attr_name} set to: {arg_value}")

    _G1_MOTION_IMITATION_HDF5_NPZ = (
        ("hdf5_imitation_loop_count", "npz_imitation_loop_count"),
        ("hdf5_imitation_max_frames", "npz_imitation_max_frames"),
        ("hdf5_imitation_time_factor", "npz_imitation_time_factor"),
        ("hdf5_imitation_position_tolerance_pct", "npz_imitation_position_tolerance_pct"),
        ("hdf5_imitation_velocity_tolerance_pct", "npz_imitation_velocity_tolerance_pct"),
    )
    for hdf5_attr, npz_attr in _G1_MOTION_IMITATION_HDF5_NPZ:
        arg_value = getattr(args_cli, hdf5_attr, None)
        if arg_value is None:
            continue
        if hasattr(env_cfg, hdf5_attr):
            setattr(env_cfg, hdf5_attr, arg_value)
            print(f"[INFO] G1 Locomotion imitation: {hdf5_attr} set to: {arg_value}")
        if hasattr(env_cfg, npz_attr):
            setattr(env_cfg, npz_attr, arg_value)
            if not hasattr(env_cfg, hdf5_attr):
                print(f"[INFO] G1 Locomotion imitation: {npz_attr} set to: {arg_value}")

    for attr_name in (
        "hdf5_link_local_imitation_reward_weight",
        "hdf5_link_local_imitation_penalty_weight",
        "hdf5_link_local_imitation_tolerance_m",
        "hdf5_link_local_imitation_position_reward_penalty_xy_only",
        "hdf5_link_local_imitation_velocity_reward_weight",
        "hdf5_link_local_imitation_velocity_penalty_weight",
        "hdf5_link_local_imitation_velocity_sigma_m_per_s",
        "hdf5_link_local_imitation_err_derivative_penalty_weight",
        "hdf5_link_local_imitation_huber_delta_m",
        "hdf5_link_local_imitation_distance_cap_m",
        "hdf5_link_local_contact_tolerance_scale",
        "hdf5_derived_foot_contact_beta",
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
    ):
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None and hasattr(env_cfg, attr_name):
            setattr(env_cfg, attr_name, arg_value)
            print(f"[INFO] G1 Locomotion imitation: {attr_name} set to: {arg_value}")
    if (
        getattr(args_cli, "hdf5_link_local_imitation_slot_weights", None) is not None
        and hasattr(env_cfg, "hdf5_link_local_imitation_slot_weights")
    ):
        try:
            parts = [float(x.strip()) for x in args_cli.hdf5_link_local_imitation_slot_weights.split(",")]
            if len(parts) == 4:
                env_cfg.hdf5_link_local_imitation_slot_weights = (parts[0], parts[1], parts[2], parts[3])
                print(
                    f"[INFO] G1 Locomotion imitation: hdf5_link_local_imitation_slot_weights = {env_cfg.hdf5_link_local_imitation_slot_weights}"
                )
            else:
                logger.warning("hdf5_link_local_imitation_slot_weights must have exactly 4 comma-separated floats.")
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid --hdf5_link_local_imitation_slot_weights: {e}")

    _link_sx = getattr(env_cfg, "hdf5_link_local_imitation_pos_scale_x", 1.0)
    try:
        _link_sx_f = float(_link_sx)
    except (TypeError, ValueError):
        _link_sx_f = 1.0
    resolve_g1_joints_data_npz_path, write_npz_dof_pos_scaled_copy, write_npz_link_local_pos_x_scaled_copy = (
        _get_g1_joints_npz_link_helpers()
    )
    if abs(_link_sx_f - 1.0) > 1e-9 and hasattr(env_cfg, "joints_data"):
        if resolve_g1_joints_data_npz_path is None or write_npz_link_local_pos_x_scaled_copy is None:
            logger.warning(
                "hdf5_link_local_imitation_pos_scale_x=%s ignored: g1_locomotion_v1 motion utilities are not available "
                "(package removed from isaaclab_tasks).",
                _link_sx_f,
            )
        else:
            _jdf = getattr(env_cfg, "joints_data", None)
            if _jdf and str(_jdf).strip():
                _src = resolve_g1_joints_data_npz_path(str(_jdf).strip())
                if _src is None:
                    logger.warning(
                        "hdf5_link_local_imitation_pos_scale_x=%s but joints_data motion file not found (%r); not writing scaled copy.",
                        _link_sx_f,
                        _jdf,
                    )
                else:
                    try:
                        _fd, _tmp = tempfile.mkstemp(prefix="g1_link_pos_x_scaled_", suffix=".hdf5")
                        os.close(_fd)
                        write_npz_link_local_pos_x_scaled_copy(_src, _tmp, _link_sx_f)
                        env_cfg.joints_data = _tmp
                        print(
                            f"[INFO] Motion link pelvis-frame x scaled by {_link_sx_f}; temp joints_data: {_tmp} (source: {_src})"
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to write scaled motion file (hdf5_link_local_imitation_pos_scale_x): %s", e
                        )

    _joint_pos_ramp_en = bool(getattr(env_cfg, "hdf5_joint_position_data_scale_ramp_enabled", False))

    _joint_pos_scale = getattr(args_cli, "hdf5_joint_position_data_scale", None)
    try:
        _joint_pos_scale_f = 1.0 if _joint_pos_scale is None else float(_joint_pos_scale)
    except (TypeError, ValueError):
        _joint_pos_scale_f = 1.0
    if _joint_pos_ramp_en and abs(_joint_pos_scale_f - 1.0) > 1e-9:
        logger.warning(
            "hdf5_joint_position_data_scale=%s is ignored while joint-position ramp is enabled "
            "(no temp dof_pos copy; use ramp start/end or disable ramp).",
            _joint_pos_scale_f,
        )
    if (
        (not _joint_pos_ramp_en)
        and abs(_joint_pos_scale_f - 1.0) > 1e-9
        and hasattr(env_cfg, "joints_data")
    ):
        if resolve_g1_joints_data_npz_path is None or write_npz_dof_pos_scaled_copy is None:
            logger.warning(
                "hdf5_joint_position_data_scale=%s ignored: g1_locomotion_v1 motion utilities are not available "
                "(package removed from isaaclab_tasks).",
                _joint_pos_scale_f,
            )
        else:
            _jdf = getattr(env_cfg, "joints_data", None)
            if _jdf and str(_jdf).strip():
                _src = resolve_g1_joints_data_npz_path(str(_jdf).strip())
                if _src is None:
                    logger.warning(
                        "hdf5_joint_position_data_scale=%s but joints_data motion file not found (%r); not writing scaled copy.",
                        _joint_pos_scale_f,
                        _jdf,
                    )
                else:
                    try:
                        _fd, _tmp = tempfile.mkstemp(prefix="g1_dof_pos_scaled_", suffix=".hdf5")
                        os.close(_fd)
                        write_npz_dof_pos_scaled_copy(
                            _src,
                            _tmp,
                            _joint_pos_scale_f,
                            exclude_ankles=bool(getattr(env_cfg, "hdf5_joint_position_data_scale_exclude_ankles", False)),
                        )
                        env_cfg.joints_data = _tmp
                        print(
                            f"[INFO] Motion dof_pos scaled by {_joint_pos_scale_f}; temp joints_data: {_tmp} (source: {_src})"
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to write scaled motion file (hdf5_joint_position_data_scale): %s", e
                        )

    if getattr(args_cli, "debug_training_fixed_base", False) and hasattr(env_cfg, "debug_training_fixed_base"):
        env_cfg.debug_training_fixed_base = True
        print("[INFO] G1 Locomotion V3: debug_training_fixed_base=True (fixed root, joints actuated)")
    if getattr(args_cli, "debug_training_fixed_base_spawn_z", None) is not None and hasattr(
        env_cfg, "debug_training_fixed_base_spawn_z"
    ):
        env_cfg.debug_training_fixed_base_spawn_z = float(args_cli.debug_training_fixed_base_spawn_z)
        print(f"[INFO] G1 Locomotion V3: debug_training_fixed_base_spawn_z={env_cfg.debug_training_fixed_base_spawn_z}")

    if getattr(args_cli, "imitation_legs_only", False) and hasattr(env_cfg, "imitation_legs_only"):
        env_cfg.imitation_legs_only = True
        print("[INFO] G1 Locomotion imitation: imitation_legs_only set to True (leg joints only)")
    if hasattr(env_cfg, "imitation_exclude_joint_names"):
        if getattr(args_cli, "imitation_exclude_joints", None):
            ex_names = tuple(s.strip() for s in args_cli.imitation_exclude_joints.split(",") if s.strip())
            if ex_names:
                env_cfg.imitation_exclude_joint_names = ex_names
                print(f"[INFO] G1 Locomotion imitation: imitation_exclude_joint_names = {ex_names}")
        elif getattr(args_cli, "imitation_exclude_four_ankles", False):
            env_cfg.imitation_exclude_joint_names = (
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            )
            print(
                "[INFO] G1 Locomotion imitation: imitation_exclude_four_ankles → "
                "left_ankle_pitch_joint, left_ankle_roll_joint, right_ankle_pitch_joint, right_ankle_roll_joint"
            )
    # Deprecated CLI → imitation_joints_position_* (only if new flags not passed)
    if hasattr(env_cfg, "imitation_joints_position_reward_weight"):
        if getattr(args_cli, "imitation_joints_position_reward_weight", None) is None:
            jr = getattr(args_cli, "imitation_position_reward_weight", None)
            if jr is None:
                jr = getattr(args_cli, "imitation_reward_weight", None)
            if jr is not None:
                env_cfg.imitation_joints_position_reward_weight = jr
                src = (
                    "--imitation_position_reward_weight"
                    if getattr(args_cli, "imitation_position_reward_weight", None) is not None
                    else "--imitation_reward_weight"
                )
                print(f"[INFO] G1 Locomotion imitation: imitation_joints_position_reward_weight={jr} (deprecated {src})")
        if getattr(args_cli, "imitation_joints_position_penalty_weight", None) is None:
            jp = getattr(args_cli, "imitation_position_penalty_weight", None)
            if jp is None:
                jp = getattr(args_cli, "imitation_penalty_weight", None)
            if jp is not None:
                env_cfg.imitation_joints_position_penalty_weight = jp
                srcp = (
                    "--imitation_position_penalty_weight"
                    if getattr(args_cli, "imitation_position_penalty_weight", None) is not None
                    else "--imitation_penalty_weight"
                )
                print(f"[INFO] G1 Locomotion imitation: imitation_joints_position_penalty_weight={jp} (deprecated {srcp})")

    for attr_name in (
        "action_rate_penalty_scale",
        "joint_acceleration_penalty_scale",
        "smooth_motion_reward_weight",
        "smooth_motion_sigma",
        "policy_action_roughness_huber_delta",
        "smooth_motion_joint_accel_reward_weight",
        "smooth_motion_joint_accel_sigma",
        "joint_velocity_roughness_huber_delta_rad_s",
    ):
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None and hasattr(env_cfg, attr_name):
            setattr(env_cfg, attr_name, arg_value)
            print(f"[INFO] G1 locomotion: {attr_name} set to: {arg_value}")

    if args_cli.npz_joint_position_tolerance_deg is not None:
        if hasattr(env_cfg, 'npz_joint_position_tolerance_deg'):
            env_cfg.npz_joint_position_tolerance_deg = args_cli.npz_joint_position_tolerance_deg
            print(f"[INFO] NPZ joint position tolerance set to: {env_cfg.npz_joint_position_tolerance_deg} degrees")
        else:
            logger.warning(f"Config does not have 'npz_joint_position_tolerance_deg' attribute. Ignoring --npz_joint_position_tolerance_deg argument.")
    
    # Joint type pair reward weights (combines left+right of same joint type, highest priority)
    joint_type_pair_weight_args = [
        ('joint_position_reward_weight_hip_pitch', 'Hip pitch (both)'),
        ('joint_position_reward_weight_hip_roll', 'Hip roll (both)'),
        ('joint_position_reward_weight_hip_yaw', 'Hip yaw (both)'),
        ('joint_position_reward_weight_knee', 'Knee (both)'),
        ('joint_position_reward_weight_ankle_pitch', 'Ankle pitch (both)'),
        ('joint_position_reward_weight_ankle_roll', 'Ankle roll (both)'),
    ]
    
    for attr_name, display_name in joint_type_pair_weight_args:
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None:
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, arg_value)
                print(f"[INFO] {display_name} joint position reward weight set to: {arg_value}")
            else:
                logger.warning(f"Config does not have '{attr_name}' attribute. Ignoring --{attr_name} argument.")
    
    # Granular joint pair reward weights
    granular_weight_args = [
        ('joint_position_reward_weight_left_hip', 'Left hip'),
        ('joint_position_reward_weight_right_hip', 'Right hip'),
        ('joint_position_reward_weight_left_knee', 'Left knee'),
        ('joint_position_reward_weight_right_knee', 'Right knee'),
        ('joint_position_reward_weight_left_ankle', 'Left ankle'),
        ('joint_position_reward_weight_right_ankle', 'Right ankle'),
        ('joint_position_reward_weight_torso', 'Torso'),
        ('joint_position_reward_weight_left_shoulder', 'Left shoulder'),
        ('joint_position_reward_weight_right_shoulder', 'Right shoulder'),
        ('joint_position_reward_weight_left_elbow', 'Left elbow'),
        ('joint_position_reward_weight_right_elbow', 'Right elbow'),
        ('joint_position_reward_weight_left_hand', 'Left hand'),
        ('joint_position_reward_weight_right_hand', 'Right hand'),
    ]
    
    for attr_name, display_name in granular_weight_args:
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None:
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, arg_value)
                print(f"[INFO] {display_name} joint position reward weight set to: {arg_value}")
            else:
                logger.warning(f"Config does not have '{attr_name}' attribute. Ignoring --{attr_name} argument.")

    # Override action following parameters from command line arguments
    if args_cli.frames_per_second is not None:
        if hasattr(env_cfg, "hdf5_fps_override"):
            env_cfg.hdf5_fps_override = int(args_cli.frames_per_second)
            if hasattr(env_cfg, "npz_fps_override"):
                env_cfg.npz_fps_override = int(args_cli.frames_per_second)
            print(
                f"[INFO] G1 locomotion: motion clip FPS (hdf5_fps_override / npz_fps_override) set to: "
                f"{env_cfg.hdf5_fps_override}"
            )
        elif hasattr(env_cfg, "frames_per_second"):
            env_cfg.frames_per_second = args_cli.frames_per_second
            print(f"[INFO] Frames per second set to: {env_cfg.frames_per_second}")
        else:
            logger.warning(
                "Config has neither hdf5_fps_override nor frames_per_second. Ignoring --frames_per_second / --motion_clip_fps."
            )

    if getattr(args_cli, "sim_dt", None) is not None and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "dt"):
        env_cfg.sim.dt = float(args_cli.sim_dt)
        print(f"[INFO] sim.dt set to: {env_cfg.sim.dt}")

    if args_cli.enable_interpolation is not None:
        if hasattr(env_cfg, 'enable_interpolation'):
            env_cfg.enable_interpolation = args_cli.enable_interpolation
            print(f"[INFO] Enable interpolation set to: {env_cfg.enable_interpolation}")
        else:
            logger.warning(f"Config does not have 'enable_interpolation' attribute. Ignoring --enable_interpolation argument.")

    if args_cli.move_forward_reward_weight is not None:
        if hasattr(env_cfg, 'move_forward_reward_weight'):
            env_cfg.move_forward_reward_weight = args_cli.move_forward_reward_weight
            print(f"[INFO] Move forward reward weight set to: {env_cfg.move_forward_reward_weight}")
        else:
            logger.warning(f"Config does not have 'move_forward_reward_weight' attribute. Ignoring --move_forward_reward_weight argument.")

    if args_cli.loop_duration_seconds is not None:
        if hasattr(env_cfg, 'loop_duration_seconds'):
            env_cfg.loop_duration_seconds = args_cli.loop_duration_seconds
            print(f"[INFO] Loop duration seconds set to: {env_cfg.loop_duration_seconds}")
        else:
            logger.warning(f"Config does not have 'loop_duration_seconds' attribute. Ignoring --loop_duration_seconds argument.")

    if args_cli.loop_count is not None:
        if hasattr(env_cfg, 'loop_count'):
            env_cfg.loop_count = args_cli.loop_count
            print(f"[INFO] Loop count set to: {env_cfg.loop_count}")
        else:
            logger.warning(f"Config does not have 'loop_count' attribute. Ignoring --loop_count argument.")

    # Ion VTOL vertical takeoff task overrides
    ion_takeoff_args = [
        ('target_takeoff_speed', 'Target takeoff speed (m/s)'),
        ('target_velocity_tolerance', 'Target velocity tolerance (|vz-target| <= m/s)'),
        ('positive_z_velocity_reward_weight', 'Positive Z velocity reward weight'),
        ('z_velocity_sharpness', 'Z velocity sharpness (exponent for vz>0)'),
        ('upward_velocity_shaping_weight', 'Upward velocity shaping weight (0<vz<vz_low)'),
        ('thrust_output_bias', 'Thrust output bias (min thrust fraction)'),
        ('idle_velocity_penalty_weight', 'Idle velocity penalty weight'),
        ('idle_velocity_reward_weight', 'Idle velocity reward weight (active vz)'),
        ('idle_velocity_threshold', 'Idle velocity threshold (m/s)'),
        ('drift_penalty_cap_per_step', 'Drift penalty cap per step'),
        ('drift_tolerance', 'Drift tolerance (no penalty when |vx|,|vy| <= m/s)'),
        ('x_drift_penalty_sharpness', 'X drift penalty sharpness (exponent)'),
        ('y_drift_penalty_sharpness', 'Y drift penalty sharpness (exponent)'),
        ('off_band_z_velocity_penalty_weight', 'Off-band Z velocity penalty weight'),
        ('off_band_z_velocity_penalty_sharpness', 'Off-band Z velocity penalty sharpness'),
        ('off_band_z_velocity_penalty_cap_per_step', 'Off-band Z velocity penalty cap per step'),
        ('x_drift_penalty', 'X drift penalty'),
        ('y_drift_penalty', 'Y drift penalty'),
        ('x_drift_free_reward_weight', 'X drift-free reward weight'),
        ('y_drift_free_reward_weight', 'Y drift-free reward weight'),
        ('jerky_penalty', 'Jerky motion penalty'),
        ('smooth_motion_reward_weight', 'Smooth motion reward weight'),
        ('smooth_motion_reward_sharpness', 'Smooth motion reward sharpness'),
        ('electricity_cost_penalty', 'Electricity cost penalty'),
        ('electricity_efficiency_reward_weight', 'Electricity efficiency reward weight'),
        ('electricity_efficiency_sigma', 'Electricity efficiency sigma'),
        ('upward_velocity_shaping_miss_penalty_weight', 'Upward shaping miss penalty weight'),
        ('orientation_reward_weight', 'Orientation reward weight'),
        ('orientation_penalty_weight', 'Orientation penalty weight (Direct)'),
        ('orientation_penalty_sharpness', 'Orientation penalty sharpness (Direct)'),
        ('orientation_penalty_cap_per_step', 'Orientation penalty cap per step (Direct)'),
        ('xy_orientation_penalty_weight', 'XY orientation (tilt) penalty weight'),
        ('xy_orientation_penalty_sharpness', 'XY orientation penalty sharpness'),
        ('xy_orientation_penalty_cap_per_step', 'XY orientation penalty cap per step'),
        ('z_orientation_penalty_weight', 'Z orientation (yaw) penalty weight'),
        ('z_orientation_penalty_sharpness', 'Z orientation penalty sharpness'),
        ('z_orientation_penalty_cap_per_step', 'Z orientation penalty cap per step'),
        ('low_altitude_penalty_threshold', 'Low altitude penalty threshold (z < m)'),
        ('low_altitude_penalty_weight', 'Low altitude penalty weight'),
        ('low_altitude_penalty_cap_per_step', 'Low altitude penalty cap per step'),
        ('low_altitude_penalty_ramp', 'Low altitude penalty ramp (smooth vs binary)'),
        ('airborne_reward_weight', 'Airborne reward weight (z > threshold)'),
        ('takeoff_height_shaping_weight', 'Takeoff height shaping weight (AGL)'),
        ('takeoff_height_shaping_cap_m', 'Takeoff height shaping cap (m)'),
        ('drift_free_requires_airborne', 'Drift-free rewards require z >= low-altitude threshold'),
        ('clamp_rewards', 'Clamp rewards (per-step caps + global clip)'),
        ('clamp_xy_drift_penalties', 'Clamp XY drift penalties (per-step cap each axis)'),
        ('clamp_off_band_z_velocity_penalty', 'Clamp off-band vz velocity penalty (per-step cap)'),
        ('reward_scale', 'Reward scale (multiplier before return)'),
        ('spawn_height', 'Spawn height above terrain (m)'),
        ('decimation', 'Decimation (physics steps per policy step)'),
        ('z_position_reward_weight', 'Z position reward weight (keep altitude AGL)'),
        ('z_position_target', 'Z position target AGL (m)'),
        ('z_position_sharpness', 'Z position reward sharpness'),
        ('xy_position_reward_weight', 'XY position exp reward weight (hold origin)'),
        ('xy_position_reward_sharpness', 'XY position reward sharpness'),
        ('xy_position_penalty_weight', 'XY position penalty weight (drift from origin)'),
        ('xy_position_penalty_tolerance', 'XY position penalty tolerance (m)'),
        ('xy_position_penalty_sharpness', 'XY position penalty sharpness'),
        ('xy_position_penalty_cap_per_step', 'XY position penalty cap per step'),
        ('target_forward_speed', 'Target forward speed vx (m/s)'),
        ('target_x_velocity_tolerance', 'Target x velocity tolerance (m/s)'),
        ('positive_x_velocity_reward_weight', 'Positive X velocity reward weight'),
        ('x_velocity_sharpness', 'X velocity sharpness'),
        ('forward_velocity_shaping_weight', 'Forward velocity shaping weight'),
        ('off_band_x_velocity_penalty_weight', 'Off-band X velocity penalty weight'),
        ('off_band_x_velocity_penalty_sharpness', 'Off-band X velocity penalty sharpness'),
        # Ion vertical takeoff V2 (--reward_profile / --include_reward_profile_phase_obs: shared G1 V4 args)
        ('staged_stability_start_env_steps', 'Ion V2: env steps before stability group ramps'),
        ('staged_stability_ramp_env_steps', 'Ion V2: stability ramp length (env steps)'),
        ('target_vz_command_obs_enabled', 'Ion V2 / cruise V2: normalized target vz obs channel'),
        ('vz_command_obs_scale', 'Ion V2 / cruise V2: target vz obs scale divisor'),
        ('target_vx_command_obs_enabled', 'Cruise V2: normalized target vx obs channel'),
        ('vx_command_obs_scale', 'Cruise V2: target vx obs scale divisor'),
    ]
    for attr_name, display_name in ion_takeoff_args:
        arg_value = getattr(args_cli, attr_name, None)
        if arg_value is not None:
            if hasattr(env_cfg, attr_name):
                setattr(env_cfg, attr_name, arg_value)
                print(f"[INFO] Ion env (takeoff/cruise): {display_name} set to: {arg_value}")
            else:
                logger.warning(f"Config does not have '{attr_name}' attribute. Ignoring --{attr_name} argument.")

    # Reward clip range (tuple); set when either bound is provided
    if getattr(args_cli, "reward_clip_min", None) is not None or getattr(args_cli, "reward_clip_max", None) is not None:
        if hasattr(env_cfg, "reward_clip_range"):
            try:
                lo, hi = env_cfg.reward_clip_range[0], env_cfg.reward_clip_range[1]
            except (TypeError, IndexError):
                lo, hi = -1.0, 1.0
            lo = args_cli.reward_clip_min if args_cli.reward_clip_min is not None else lo
            hi = args_cli.reward_clip_max if args_cli.reward_clip_max is not None else hi
            env_cfg.reward_clip_range = (lo, hi)
            print(f"[INFO] reward_clip_range set to ({lo}, {hi})")
        else:
            logger.warning("Config does not have 'reward_clip_range' attribute. Ignoring --reward_clip_min/--reward_clip_max.")

    # When decimation is set, sync sim.render_interval so rendering is consistent (direct envs)
    decimation_val = getattr(args_cli, "decimation", None)
    if decimation_val is not None and hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = decimation_val
        print(f"[INFO] Ion vertical takeoff: sim.render_interval set to {decimation_val} (match decimation)")

    _g1_v1_apply_debug_fixed_base_if_available(env_cfg)

    # G1 V4 + resume: align observation toggles to checkpoint *before* gym.make (fixes 103 vs 109 actor width).
    if getattr(args_cli, "task", None) == "Isaac-G1-Locomotion-V4-Direct-v0" and getattr(agent_cfg, "resume", False):
        try:
            _resume_ckpt_probe = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
            if os.path.isfile(_resume_ckpt_probe):
                _ck_dim = _infer_rsl_rl_actor_obs_dim_from_checkpoint(_resume_ckpt_probe)
                if _ck_dim is not None:
                    from isaaclab_tasks.direct.g1_locomotion_v1.g1_locomotion_v4_env_cfg import (
                        compute_g1_locomotion_v4_policy_obs_dim,
                    )

                    if _align_g1_v4_env_cfg_obs_for_resume_checkpoint(env_cfg, _ck_dim):
                        env_cfg.observation_space = compute_g1_locomotion_v4_policy_obs_dim(env_cfg)
                        print(
                            f"[INFO] G1 locomotion V4: observation_space={env_cfg.observation_space} "
                            "(final before env; aligned to resume checkpoint)"
                        )
        except Exception as e:
            logger.warning("G1 V4 resume obs alignment skipped: %s", e)

    # create isaac environment
    # render_mode=None means headless (no GUI, no rendering) - this is the default for training
    # render_mode="rgb_array" is only used when recording videos
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _write_rover_localization_train_debug_snap(
        log_dir=log_dir,
        stage="after_gym_make",
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        args_cli=args_cli,
        env=env,
    )

    # Write header to training_progress_log.txt (run arguments + joint limits from NPZ if any); only when file does not exist
    progress_log_path = os.path.join(log_dir, "training_progress_log.txt")
    resume_path = None
    resume_progress_copied = False
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        _oa_flat_resume_dirs = None
        if getattr(agent_cfg, "experiment_name", "") in (
            "rover_oa_flat_env",
            "rover_oa_flat_physparam_env",
        ) or "Isaac-Rover-OAFlat" in (getattr(args_cli, "task", "") or ""):
            _oa_flat_resume_dirs = ["checkpoints"]
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, other_dirs=_oa_flat_resume_dirs
        )
        if agent_cfg.resume:
            resume_run_dir = os.path.dirname(resume_path)
            # Checkpoints may live in policy_checkpoints/; logs stay at run root
            if os.path.basename(resume_run_dir) == "policy_checkpoints":
                resume_run_dir = os.path.dirname(resume_run_dir)
            if os.path.basename(resume_run_dir) == "checkpoints":
                resume_run_dir = os.path.dirname(resume_run_dir)
            src_tp = os.path.join(resume_run_dir, "training_progress_log.txt")
            src_rd = os.path.join(resume_run_dir, "reward_debug_log.txt")
            import shutil

            if os.path.isfile(src_tp):
                os.makedirs(log_dir, exist_ok=True)
                shutil.copy2(src_tp, progress_log_path)
                resume_progress_copied = True
                print(f"[INFO] Copied training_progress_log.txt from checkpoint run for resume: {progress_log_path}")
                _truncate_training_progress_log_to_checkpoint_iter(progress_log_path, agent_cfg.load_checkpoint)
                _replace_training_progress_log_run_arguments_section(progress_log_path, args_cli)
                if "Isaac-Rover-OAFlat" in (getattr(args_cli, "task", "") or ""):
                    _replace_training_progress_log_rover_oa_flat_snapshot(progress_log_path, env_cfg)
                    print(
                        "[INFO] Refreshed Rover OA-Flat env snapshot + Effective RewTerm lines in "
                        f"training_progress_log.txt from live env_cfg (resume)."
                    )
            elif os.path.isfile(src_rd):
                os.makedirs(log_dir, exist_ok=True)
                shutil.copy2(src_rd, progress_log_path)
                resume_progress_copied = True
                print(f"[INFO] Copied legacy reward_debug_log.txt to training_progress_log.txt for resume: {progress_log_path}")
                _truncate_training_progress_log_to_checkpoint_iter(progress_log_path, agent_cfg.load_checkpoint)
                _replace_training_progress_log_run_arguments_section(progress_log_path, args_cli)
                if "Isaac-Rover-OAFlat" in (getattr(args_cli, "task", "") or ""):
                    _replace_training_progress_log_rover_oa_flat_snapshot(progress_log_path, env_cfg)
                    print(
                        "[INFO] Refreshed Rover OA-Flat env snapshot + Effective RewTerm lines in "
                        f"training_progress_log.txt from live env_cfg (resume)."
                    )
            else:
                logger.warning(
                    f"[WARNING] No training_progress_log.txt or reward_debug_log.txt in checkpoint run dir: {resume_run_dir}. "
                    "Resumed run will start a new training progress log."
                )

    if not resume_progress_copied and not os.path.isfile(progress_log_path):
        try:
            os.makedirs(log_dir, exist_ok=True)
            header_lines = [
                "################################################################################",
                " Run arguments (this run)",
                "################################################################################",
            ]
            for k in sorted(vars(args_cli).keys()):
                if k.startswith("_"):
                    continue
                v = getattr(args_cli, k)
                if v is None:
                    continue
                header_lines.append(f"  {k}: {v}")
            header_lines.append("")
            header_lines.append("################################################################################")
            header_lines.append(" Env config (reward weights, velocity, smooth/jerk motion scales)")
            header_lines.append("################################################################################")
            for attr in (
                "joints_data", "joint_limit_penalty_weight", "joint_limit_reward_weight", "joint_limit_reward_sigma",
                "dof_at_limit_cost_scale", "npz_joint_limit_extend_deg", "npz_joint_limit_extend_deg_non_leg",
                "npz_limits_exclude_four_ankles", "npz_only_for_limits",
                "imitation_joints_position_reward_weight", "imitation_joints_position_penalty_weight",
                "imitation_joints_position_reward_sharpness",
                "imitation_velocity_reward_weight", "imitation_velocity_penalty_weight",
                "hdf5_imitation_loop_count", "hdf5_imitation_max_frames", "hdf5_imitation_time_factor",
                "hdf5_imitation_position_tolerance_pct", "hdf5_imitation_velocity_tolerance_pct",
                "hdf5_imitation_reward_episode_time_scale_enabled",
                "hdf5_imitation_reward_episode_time_c",
                "hdf5_imitation_reward_episode_time_exponent",
                "npz_imitation_loop_count", "npz_imitation_max_frames", "npz_imitation_time_factor",
                "npz_imitation_position_tolerance_pct", "npz_imitation_velocity_tolerance_pct",
                "progress_reward_multiplier",
                "progress_reward_world_vx_blend",
                "progress_reward_forward_displacement_weight",
                "progress_reward_forward_displacement_threshold_m",
                "progress_goal_world_offset_xy_m",
                "up_weight",
                "heading_weight",
                "alive_reward_scale",
                "actions_cost_scale", "energy_cost_scale", "decimation",
                "target_velocity", "velocity_reward_weight", "velocity_reward_sharpness", "target_x_velocity_penalty_weight", "termination_height",
                "drift_tolerance",  # ion velocity drift
                "y_drift_tolerance", "y_drift_penalty_weight",  # G1 walking lateral y drift
                "y_drift_reward_weight", "y_drift_reward_sigma",
                "target_z_orientation", "z_orientation_tolerance", "z_orientation_penalty_weight",
                "z_orientation_reward_weight", "z_orientation_reward_sharpness",
                "alternating_foot_reward_weight", "same_foot_tap_penalty_weight",
                "jump_penalty_weight", "at_least_one_foot_contact_reward_weight",
                "ankle_roll_feet_target_separation_m",
                "ankle_roll_feet_separation_tolerance_m",
                "ankle_roll_feet_separation_reward_weight",
                "ankle_roll_feet_separation_penalty_weight",
                "ankle_roll_foot_height_reward_target_m",
                "ankle_roll_foot_height_reward_weight",
                "symmetry_reward_weight", "symmetry_penalty_weight", "symmetry_contact_sharpness",
                "symmetry_touchdown_sharpness", "symmetry_torque_sharpness", "symmetry_include_hip_torque",
                "symmetry_torque_terms_double_support_only",
                "symmetry_use_geometric_mean",
                "step_length_min_m", "step_length_reward_weight", "step_length_penalty_weight",
                "step_length_use_smooth_reward", "step_length_smooth_sigma_m", "step_length_penalty_squared",
                "foot_crossing_reward_weight", "foot_crossing_penalty_weight",
                "same_foot_tap_min_air_steps", "gait_single_stance_reward_weight",
                "hip_weight_shift_reward_weight", "hip_weight_shift_torque_scale",
                "alternating_foot_scale_with_forward_vel", "alternating_foot_vel_scale_ref",
                "lin_vel_z_penalty_weight", "lin_vel_z_tolerance", "ang_vel_xy_penalty_weight", "flat_orientation_penalty_weight", "flat_orientation_tolerance",
                "action_rate_penalty_scale", "joint_acceleration_penalty_scale",
                "smooth_motion_reward_weight", "smooth_motion_sigma",
                "policy_action_roughness_huber_delta",
                "smooth_motion_joint_accel_reward_weight", "smooth_motion_joint_accel_sigma",
                "joint_velocity_roughness_huber_delta_rad_s",
                "smoothness_skip_first_episode_step",
                "hand_target_pos_b", "right_hand_body_name", "hand_reward_weight", "hand_reward_sigma",
                "xy_drift_tolerance", "xy_drift_penalty_weight",
                "x_drift_tolerance", "x_drift_penalty_weight", "y_drift_tolerance", "y_drift_penalty_weight",  # G1 standing hand position V2
                "x_drift_reward_weight", "x_drift_reward_sigma",  # exp reward vs origin (pairs with x_drift_penalty)
                "ik_target_motion_speed", "ik_target_motion_period_base",
                "ik_joint_reward_weight", "ik_joint_reward_sigma", "ik_joint_penalty_weight", "ik_joint_penalty_threshold",
                "ik_joint_tolerance_percent", "ee_position_reward_weight", "ee_position_penalty_weight", "ee_position_reward_sigma",
                "ik_ee_smoothness_penalty_weight",
                "spawn_fixed_base", "ik_target_box_min", "ik_target_box_max", "ik_target_obs_scale",
                "staged_ik_start_env_steps", "staged_ik_ramp_env_steps",
                "teleop_target_obs_scale",
                "reward_clip_range",
                "base_height_reference",
                "base_height_reward_weight",
                "base_height_sigma",
                "base_height_penalty_weight",
                "base_height_penalty_tolerance_m",
                "velocity_command_range", "enable_curriculum", "curriculum_stage1_steps",
                "curriculum_stage2_steps", "num_steps_per_env",
                "enable_push_perturbation", "max_push_vel_xy",
                "episode_start_force_impulse_enabled", "episode_start_force_impulse_duration_steps",
                "episode_start_force_impulse_max_force_xy_n", "episode_start_force_impulse_max_force_z_n",
                "imitation_joints_position_reward_weight", "imitation_joints_position_penalty_weight",
                "imitation_joints_position_reward_sharpness",
                "imitation_velocity_reward_weight", "imitation_velocity_penalty_weight",
                "imitation_legs_only", "imitation_exclude_joint_names",
                "imitation_legs_weight", "imitation_arms_weight",
                "include_root_xy_error_obs", "root_xy_error_obs_scale",
                "reward_profile",
                "staged_imitation_start_optimizer_steps",
                "staged_imitation_ramp_optimizer_steps",
                "staged_ramp_start_scale",
                "staged_symmetry_start_optimizer_steps",
                "staged_symmetry_ramp_optimizer_steps",
                "staged_symmetry_ramp_start_scale",
                "staged_contact_gait_start_optimizer_steps",
                "staged_contact_gait_ramp_optimizer_steps",
                "velocity_command_obs_enabled",
                "velocity_command_obs_scale",
                "include_reward_profile_phase_obs",
                "target_yaw_command_obs_enabled",
                "append_teleop_hand_target_obs",
                "gravity_scale",
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
                "torso_z_align_link_body_name",
                "torso_z_axis_world_align_reward_weight",
                "torso_z_axis_world_align_reward_sigma",
                "torso_z_axis_world_align_penalty_weight",
                "torso_z_axis_world_align_penalty_tolerance",
                "palm_below_pelvis_right_body_name",
                "palm_below_pelvis_left_body_name",
                "palm_below_pelvis_z_max_pelvis_frame_m",
                "palm_below_pelvis_z_reward_weight",
                "palm_below_pelvis_z_reward_sigma_m",
                "palm_below_pelvis_z_penalty_weight",
                "palm_below_pelvis_z_penalty_tolerance_m",
                "early_termination_time_penalty_weight",
                "teleop_target_obs_scale",
                "staged_stability_start_env_steps",
                "staged_stability_ramp_env_steps",
                "target_vz_command_obs_enabled",
                "vz_command_obs_scale",
                "target_vx_command_obs_enabled",
                "vx_command_obs_scale",
                "imitation_phase_in_obs",
                "imitation_pelvis_link_error_in_obs", "imitation_root_lin_vel_in_obs",
                "imitation_root_lin_vel_obs_scale", "imitation_joint_position_use_leg_arm_weights",
                "imitation_joint_velocity_use_leg_arm_weights", "imitation_strength_ramp_steps",
                "imitation_strength_ramp_easing",
                "adapter_phase_enabled", "adapter_phase_duration_steps", "adapter_phase_start_scale", "adapter_phase_easing",
                "imitation_velocity_sigma_rad_s",
                "motion_cycle_episode_margin_s",
                "hdf5_link_use_joint_motion_clock_when_aligned", "hdf5_link_local_imitation_use_body_lin_vel",
                "hdf5_imitation_periodic_joint_vel_fd",
                "hdf5_joint_position_data_scale_ramp_enabled",
                "hdf5_joint_position_data_scale_ramp_start",
                "hdf5_joint_position_data_scale_ramp_end",
                "hdf5_joint_position_data_scale_ramp_steps",
                "hdf5_joint_position_data_scale_ramp_delay_steps",
                "hdf5_joint_position_data_scale_ramp_in_ppo_iterations",
                "hdf5_joint_position_data_scale_mode",
                "hdf5_joint_position_data_scale_exclude_ankles",
                "hdf5_link_local_imitation_periodic_fd",
                "hdf5_link_local_imitation_reward_weight", "hdf5_link_local_imitation_penalty_weight",
                "hdf5_link_local_imitation_tolerance_m",
                "hdf5_link_local_imitation_position_reward_penalty_xy_only",
                "hdf5_link_local_imitation_velocity_reward_weight", "hdf5_link_local_imitation_velocity_penalty_weight",
                "hdf5_link_local_imitation_velocity_sigma_m_per_s",
                "hdf5_link_local_imitation_err_derivative_penalty_weight",
                "hdf5_link_local_imitation_slot_weights",
                "hdf5_link_local_imitation_huber_delta_m", "hdf5_link_local_imitation_distance_cap_m",
                "hdf5_link_local_contact_tolerance_scale",
                "hdf5_derive_foot_contact_from_link_z", "hdf5_derived_foot_contact_beta",
                "hdf5_link_local_stance_tolerance_scale", "hdf5_link_local_stance_ref_contact_threshold",
                "hdf5_link_local_imitation_swing_penalty_mask", "hdf5_link_local_imitation_swing_ref_contact_threshold",
                "imitation_joint_ref_contact_leg_scale",
                "imitation_root_xy_reward_weight", "imitation_root_xy_penalty_weight", "imitation_root_xy_sigma_m",
                "swing_foot_clearance_reward_weight", "swing_foot_clearance_penalty_weight",
                "swing_foot_clearance_sigma_m", "swing_foot_clearance_margin_m",
                "stance_foot_slip_penalty_weight",
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
                "z_position_reward_weight", "z_position_target", "z_position_sharpness",
                "target_forward_speed", "target_x_velocity_tolerance", "positive_x_velocity_reward_weight",
                "x_velocity_sharpness", "forward_velocity_shaping_weight",
                "off_band_x_velocity_penalty_weight", "off_band_x_velocity_penalty_sharpness",
            ):
                if hasattr(env_cfg, attr):
                    val = getattr(env_cfg, attr)
                    header_lines.append(f"  {attr}: {val}")
            # Explicit smooth/jerk args for G1 locomotion (copy-paste for shell)
            if hasattr(env_cfg, "action_rate_penalty_scale"):
                header_lines.append("")
                header_lines.append("# Policy ‖Δa‖² + joint ‖Δq̇‖² smooth/rough pairs (G1 locomotion) — copy-paste:")
                header_lines.append(f"#   --action_rate_penalty_scale {getattr(env_cfg, 'action_rate_penalty_scale')} \\")
                header_lines.append(f"#   --smooth_motion_reward_weight {getattr(env_cfg, 'smooth_motion_reward_weight')} \\")
                header_lines.append(f"#   --smooth_motion_sigma {getattr(env_cfg, 'smooth_motion_sigma')} \\")
                if hasattr(env_cfg, "policy_action_roughness_huber_delta"):
                    header_lines.append(
                        f"#   --policy_action_roughness_huber_delta {getattr(env_cfg, 'policy_action_roughness_huber_delta')} \\"
                    )
                header_lines.append(f"#   --joint_acceleration_penalty_scale {getattr(env_cfg, 'joint_acceleration_penalty_scale')} \\")
                if hasattr(env_cfg, "smooth_motion_joint_accel_reward_weight"):
                    header_lines.append(
                        f"#   --smooth_motion_joint_accel_reward_weight {getattr(env_cfg, 'smooth_motion_joint_accel_reward_weight')} \\"
                    )
                if hasattr(env_cfg, "smooth_motion_joint_accel_sigma"):
                    header_lines.append(
                        f"#   --smooth_motion_joint_accel_sigma {getattr(env_cfg, 'smooth_motion_joint_accel_sigma')} \\"
                    )
                if hasattr(env_cfg, "joint_velocity_roughness_huber_delta_rad_s"):
                    header_lines.append(
                        f"#   --joint_velocity_roughness_huber_delta_rad_s {getattr(env_cfg, 'joint_velocity_roughness_huber_delta_rad_s')}"
                    )
            _task_hdr = getattr(args_cli, "task", None) or ""
            if "Isaac-Rover-DiskWorld-Localization" in _task_hdr:
                try:
                    header_lines.append("")
                    header_lines.append("################################################################################")
                    header_lines.append(" Rover Localization task – reward weights, sensor / marker / scene params")
                    header_lines.append("################################################################################")
                    _loc_attrs = (
                        # Frozen drive policy
                        "drive_policy_path",
                        "drive_policy_activation",
                        "drive_max_forward_vel",
                        "drive_max_turn_diff_vel",
                        # Visual obstacle scene
                        "box_count",
                        "goal_count",
                        "box_size",
                        "goal_scale",
                        "obstacle_radial_distance",
                        "goal_radial_distance",
                        "box_y_range",
                        "min_object_spacing",
                        "spawn_exclusion_radius",
                        "spawn_xyz",
                        "landmark_seed",
                        "num_visible_landmarks",
                        "landmark_max_range_m",
                        "camera_fov_half_angle_rad",
                        "spawn_visual_obstacles",
                        "box_collision_enabled",
                        "goal_collision_enabled",
                        "goal_asset_path",
                        "visual_obstacle_max_spawned_envs",
                        "visual_obstacle_usd_all_envs_cap",
                        "env_spacing",
                        # Estimate-marker (red cuboid)
                        "spawn_estimate_marker",
                        "estimate_marker_size",
                        "estimate_marker_color",
                        "estimate_marker_y_offset_m",
                        "estimate_pos_norm_m",
                        # Flat reward weights / scales
                        "loc_position_error_weight",
                        "loc_position_error_scale_m",
                        "loc_position_close_weight",
                        "loc_position_close_threshold_m",
                        "loc_orientation_error_weight",
                        "loc_yaw_unit_norm_penalty_weight",
                        "loc_estimate_smoothness_weight",
                        "loc_estimate_near_wheel_odom_weight",
                        "loc_estimate_near_wheel_odom_scale_m",
                        "loc_estimate_raw_wheel_odom_seed_weight",
                        "loc_estimate_raw_wheel_odom_seed_decay_env_steps",
                        "oa_teacher_action_blend_weight",
                        "oa_teacher_action_blend_decay_env_steps",
                        "loc_estimate_alive_bonus_weight",
                        "loc_fall_min_radius_m",
                        "landmark_camera_enable",
                        "landmark_camera_width",
                        "landmark_camera_height",
                        "landmark_camera_hfov_deg",
                        "landmark_camera_max_range_m",
                        "landmark_camera_pixel_stride",
                        "landmark_camera_debug_mode",
                        "landmark_camera_debug_show_window",
                        "landmark_camera_debug_log_every",
                        "landmark_camera_debug_max_print",
                        "landmark_camera_debug_window_every",
                    )
                    for _a in _loc_attrs:
                        if hasattr(env_cfg, _a):
                            header_lines.append(f"  {_a}: {getattr(env_cfg, _a)}")
                    _loc_vis_cli = getattr(
                        args_cli, "rover_localization_visual_obstacle_max_envs", None
                    )
                    if _loc_vis_cli is not None:
                        header_lines.append(
                            f"  rover_localization_visual_obstacle_max_envs: {_loc_vis_cli}"
                        )
                    _loc_usd_all = getattr(
                        args_cli, "rover_localization_visual_usd_all_envs_cap", None
                    )
                    if _loc_usd_all is not None:
                        header_lines.append(
                            f"  rover_localization_visual_usd_all_envs_cap: {_loc_usd_all}"
                        )
                    _loc_goal_col = getattr(args_cli, "rover_localization_goal_collision", None)
                    if _loc_goal_col is not None:
                        header_lines.append(f"  rover_localization_goal_collision: {_loc_goal_col}")
                    _loc_box_col = getattr(args_cli, "rover_localization_box_collision", None)
                    if _loc_box_col is not None:
                        header_lines.append(f"  rover_localization_box_collision: {_loc_box_col}")
                    _loc_goal_count = getattr(args_cli, "rover_localization_goal_count", None)
                    if _loc_goal_count is not None:
                        header_lines.append(f"  rover_localization_goal_count: {_loc_goal_count}")
                    _loc_box_count = getattr(args_cli, "rover_localization_box_count", None)
                    if _loc_box_count is not None:
                        header_lines.append(f"  rover_localization_box_count: {_loc_box_count}")
                    # Pull per-term weights directly from the RewardsCfg so they
                    # reflect whatever overrides (if any) were applied later.
                    try:
                        _rew = getattr(env_cfg, "rewards", None)
                        if _rew is not None:
                            header_lines.append("")
                            header_lines.append("  # Effective RewTerm weights (post overrides):")
                            for _rt_name in (
                                "position_error_reward",
                                "position_close_bonus",
                                "orientation_error_reward",
                                "yaw_unit_norm_penalty",
                                "estimate_smoothness",
                                "estimate_near_wheel_odom",
                                "estimate_raw_wheel_odom_seed",
                                "estimate_alive_bonus",
                            ):
                                _rt = getattr(_rew, _rt_name, None)
                                if _rt is not None:
                                    _params = getattr(_rt, "params", {}) or {}
                                    header_lines.append(
                                        f"  reward.{_rt_name}.weight: {getattr(_rt, 'weight', None)}"
                                        + ("" if not _params else f"  params={_params}")
                                    )
                    except Exception as _e_rw:
                        logger.warning(
                            "[WARNING] Could not enumerate rover-localization RewTerm weights: %s", _e_rw
                        )
                except Exception as _e_loc_hdr:
                    logger.warning(
                        "[WARNING] Could not write rover-localization header block: %s", _e_loc_hdr
                    )
            if "Isaac-Rover-FlatWorld-GoalNav" in _task_hdr:
                try:
                    header_lines.append("")
                    header_lines.append("################################################################################")
                    header_lines.append(" Rover FlatWorld-GoalNav task – reward weights, geometry, episode")
                    header_lines.append("################################################################################")
                    _fg_attrs = (
                        # Scene geometry
                        "plane_half_size_m",
                        "spawn_half_size_m",
                        "target_half_size_m",
                        "target_exclusion_half_size_m",
                        "target_min_spawn_dist_m",
                        "spawn_z_m",
                        "target_marker_z_m",
                        "target_marker_size_m",
                        "spawn_target_marker",
                        "continuous_target_reset_enabled",
                        "continuous_target_reset_threshold_m",
                        # Flat reward weights / params
                        "flat_distance_to_target_weight",
                        "flat_distance_to_target_scale_m",
                        "flat_distance_progress_weight",
                        "flat_distance_increase_penalty_weight",
                        "flat_heading_to_target_weight",
                        "flat_heading_goal_closure_weight",
                        "flat_yaw_no_closure_penalty_weight",
                        "flat_forward_velocity_to_target_weight",
                        "flat_forward_velocity_to_target_cap_ms",
                        "flat_close_to_target_bonus_weight",
                        "flat_close_to_target_threshold_m",
                        "flat_goal_reached_bonus_weight",
                        "flat_goal_reached_threshold_m",
                        "flat_goal_relocate_threshold_m",
                        "flat_action_rate_weight",
                        "flat_tilt_weight",
                        "flat_boundary_weight",
                        "flat_boundary_margin_m",
                        "flat_four_wheels_ground_contact_weight",
                        "flat_front_wheels_ground_contact_weight",
                        "flat_rear_wheels_ground_contact_weight",
                        "flat_not_four_wheels_ground_contact_penalty_weight",
                        "flat_wheel_ground_contact_threshold_n",
                        "flat_alive_weight",
                        "flat_wheel_effort_limit_sim",
                        "flat_wheel_target_slew_rad_s2",
                        "flat_upside_down_min_dot",
                        "flat_wheelie_min_pitch_sin",
                        "flat_goal_train_stage",
                        "flat_goal_train_stage_note",
                        "episode_length_s",
                    )
                    for _a in _fg_attrs:
                        if hasattr(env_cfg, _a):
                            header_lines.append(f"  {_a}: {getattr(env_cfg, _a)}")
                    try:
                        _rew = getattr(env_cfg, "rewards", None)
                        if _rew is not None:
                            header_lines.append("")
                            header_lines.append("  # Effective RewTerm weights (post overrides):")
                            for _rt_name in (
                                "distance_to_target",
                                "distance_progress",
                                "distance_increase_penalty",
                                "heading_to_target",
                                "heading_goal_closure_speed",
                                "yaw_without_goal_closure",
                                "forward_velocity_to_target",
                                "close_to_target_bonus",
                                "goal_reached_bonus",
                                "action_rate_penalty",
                                "tilt_penalty",
                                "boundary_penalty",
                                "four_wheels_contact",
                                "front_wheels_contact",
                                "rear_wheels_contact",
                                "not_four_wheels_contact_penalty",
                                "alive_bonus",
                            ):
                                _rt = getattr(_rew, _rt_name, None)
                                if _rt is not None:
                                    _params = getattr(_rt, "params", {}) or {}
                                    header_lines.append(
                                        f"  reward.{_rt_name}.weight: {getattr(_rt, 'weight', None)}"
                                        + ("" if not _params else f"  params={_params}")
                                    )
                        _term = getattr(env_cfg, "terminations", None)
                        if _term is not None:
                            header_lines.append("")
                            header_lines.append("  # Effective DoneTerm params:")
                            for _tn in ("time_out", "goal_reached", "out_of_bounds", "fell_below_plane", "upside_down"):
                                _tt = getattr(_term, _tn, None)
                                if _tt is not None:
                                    _p = getattr(_tt, "params", {}) or {}
                                    header_lines.append(
                                        f"  termination.{_tn}: time_out={getattr(_tt, 'time_out', False)}"
                                        + ("" if not _p else f"  params={_p}")
                                    )
                    except Exception as _e_fg_rw:
                        logger.warning(
                            "[WARNING] Could not enumerate rover-flat-goal RewTerm weights: %s", _e_fg_rw
                        )
                except Exception as _e_fg_hdr:
                    logger.warning(
                        "[WARNING] Could not write rover-flat-goal header block: %s", _e_fg_hdr
                    )
            if "Isaac-Rover-OAFlat" in _task_hdr:
                try:
                    header_lines.extend(_rover_oa_flat_training_log_snapshot_lines(env_cfg))
                except Exception as _e_oa_hdr:
                    logger.warning("[WARNING] Could not write rover OA-flat header block: %s", _e_oa_hdr)
            if "Isaac-Rover-DiskWorld-Physparam" in _task_hdr:
                try:
                    from rover_physparam_log_docs import append_physparam_latch_doc_to_header  # noqa: PLC0415

                    _ru_phy_header = None
                    try:
                        _train_py_dir = os.path.dirname(os.path.abspath(__file__))
                        _scripts_rover_hdr = os.path.normpath(
                            os.path.join(_train_py_dir, "..", "..", "..", "scripts", "rover")
                        )
                        if _scripts_rover_hdr not in sys.path:
                            sys.path.insert(0, _scripts_rover_hdr)
                        import rover_utils as _ru_phy_header  # noqa: PLC0415
                    except Exception:
                        pass
                    append_physparam_latch_doc_to_header(header_lines, _ru_phy_header)
                except Exception as _e_phy_doc:
                    logger.warning(
                        "[WARNING] Could not append physparam latch doc to progress log header: %s",
                        _e_phy_doc,
                    )
            unwrapped = env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            if hasattr(unwrapped, "get_joint_limits_for_log_header"):
                limits = unwrapped.get_joint_limits_for_log_header()
                if limits:
                    header_lines.append("")
                    header_lines.append("################################################################################")
                    header_lines.append(" Joint limits from motion file / HDF5 or legacy NPZ (rad and deg)")
                    header_lines.append("################################################################################")
                    header_lines.extend(limits)
            header_lines.append("################################################################################")
            header_lines.append("")
            with open(progress_log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(header_lines))
        except Exception as e_header:
            logger.warning(f"[WARNING] Could not write training_progress_log.txt header: {e_header}")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        from isaaclab_rl.rsl_rl.rsl_rl_construct_patch import apply_rsl_rl_aux_obstacle_construct_patch

        apply_rsl_rl_aux_obstacle_construct_patch()
    except Exception:
        pass

    # create runner from rsl-rl (use DivergenceAwareOnPolicyRunner for early NaN/Inf detection)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = DivergenceAwareOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        if getattr(agent_cfg, "experiment_name", "") in (
            "rover_oa_flat_env",
            "rover_oa_flat_physparam_env",
        ) or "Isaac-Rover-OAFlat" in (getattr(args_cli, "task", "") or ""):
            _oa_ck = os.path.join(log_dir, "checkpoints")
            os.makedirs(_oa_ck, exist_ok=True)
            runner.log_dir = _oa_ck
            print(f"[INFO] Rover OA-Flat: periodic checkpoints -> {runner.log_dir}")
        _dr_min = getattr(args_cli, "divergence_mean_reward_min", None)
        _dr_max = getattr(args_cli, "divergence_mean_reward_max", None)
        if _dr_min is not None or _dr_max is not None:
            if _dr_min is not None:
                runner.REWARD_MIN = _dr_min
            if _dr_max is not None:
                runner.REWARD_MAX = _dr_max
            print(f"[INFO] Divergence checker mean episode return bounds: [{runner.REWARD_MIN:g}, {runner.REWARD_MAX:g}]")
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        _oa_resume_partial_ok = getattr(agent_cfg, "experiment_name", "") in (
            "rover_oa_flat_env",
            "rover_oa_flat_physparam_env",
        ) or "Isaac-Rover-OAFlat" in (getattr(args_cli, "task", "") or "")
        if agent_cfg.algorithm.class_name == "Distillation":
            runner.load(resume_path)
        else:
            try:
                runner.load(resume_path)
            except RuntimeError as e:
                err = str(e)
                _looks_like_obs_arch_mismatch = (
                    "missing key(s) in state_dict" in err.lower()
                    or "unexpected key(s)" in err.lower()
                    or "size mismatch" in err.lower()
                )
                if not (_oa_resume_partial_ok and _looks_like_obs_arch_mismatch):
                    raise
                logger.warning(
                    "OA-Flat resume: checkpoint policy does not match current network (OBS/AAC/aux or normalizer)."
                    " Loading only shape-compatible weights; auxiliary head / mismatched layers init randomly;"
                    " optimizer state reset.",
                )
                checkpoint = torch.load(resume_path, map_location=runner.device, weights_only=False)
                model_state_dict = checkpoint.get("model_state_dict", checkpoint)
                current_state_dict = runner.alg.policy.state_dict()
                filtered_state_dict = {}
                for key, value in model_state_dict.items():
                    if "normalizer" in key.lower():
                        continue
                    if key in current_state_dict:
                        if current_state_dict[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            print(
                                "[WARNING]: Skipping "
                                f"{key} (shape mismatch: {tuple(value.shape)} vs "
                                f"{tuple(current_state_dict[key].shape)})"
                            )
                    else:
                        print(f"[WARNING]: Skipping {key} (not in current model)")
                _res = runner.alg.policy.load_state_dict(filtered_state_dict, strict=False)
                if hasattr(_res, "missing_keys") and _res.missing_keys:
                    print(f"[INFO]: Random-init / missing-after-filter: {len(_res.missing_keys)} tensors")
                if hasattr(_res, "unexpected_keys") and _res.unexpected_keys:
                    print(f"[INFO]: Unexpected keys ignored: {len(_res.unexpected_keys)}")
                print(f"[INFO]: Loaded {len(filtered_state_dict)} tensors from checkpoint (partial transfer).")
                runner.current_learning_iteration = int(checkpoint.get("iter", 0))
                runner.alg.optimizer = torch.optim.Adam(
                    runner.alg.policy.parameters(), lr=float(runner.alg.learning_rate)
                )
                print(
                    f"[INFO]: Resuming at learning iteration {runner.current_learning_iteration} with fresh Adam on policy."
                )
    else:
        # Load starting policy checkpoint if provided (e.g. standing model for walking transfer)
        starting_policy_path = getattr(args_cli, "starting_policy_path", None)
        if starting_policy_path:
            starting_policy_path = os.path.abspath(starting_policy_path)
        if starting_policy_path and os.path.isfile(starting_policy_path):
            print(f"[INFO]: Loading starting policy from: {starting_policy_path}")
            try:
                runner.load(starting_policy_path)
                print("[INFO]: Successfully loaded starting policy. Training will optimize from this checkpoint.")
            except RuntimeError as e:
                error_str = str(e)
                is_normalizer_error = "Missing key(s)" in error_str and "normalizer" in error_str
                is_shape_mismatch = "size mismatch" in error_str.lower()
                if is_normalizer_error or is_shape_mismatch:
                    if is_shape_mismatch:
                        print("[WARNING]: Checkpoint has different network architecture (e.g. observation size).")
                    else:
                        print("[WARNING]: Checkpoint has different observation normalization settings.")
                    print("[INFO]: Attempting to load only compatible weights (actor/critic)...")
                    try:
                        checkpoint = torch.load(starting_policy_path, map_location=runner.device)
                        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
                        current_state_dict = runner.alg.policy.state_dict()
                        filtered_state_dict = {}
                        for key, value in model_state_dict.items():
                            if "normalizer" in key.lower():
                                continue
                            if key in current_state_dict:
                                if current_state_dict[key].shape == value.shape:
                                    filtered_state_dict[key] = value
                                else:
                                    print(f"[WARNING]: Skipping {key} (shape mismatch: {value.shape} vs {current_state_dict[key].shape})")
                            else:
                                print(f"[WARNING]: Skipping {key} (not in current model)")
                        result = runner.alg.policy.load_state_dict(filtered_state_dict, strict=False)
                        if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
                            if result.missing_keys:
                                print(f"[INFO]: Missing keys (random init): {len(result.missing_keys)}")
                            if result.unexpected_keys:
                                print(f"[INFO]: Unexpected keys (ignored): {len(result.unexpected_keys)}")
                        print(f"[INFO]: Loaded {len(filtered_state_dict)} compatible weights from starting policy.")
                    except Exception as e2:
                        print(f"[ERROR]: Failed to load with filtering: {e2}. Starting from scratch.")
                else:
                    raise
        elif starting_policy_path:
            print(f"[WARNING]: Starting policy not found at {starting_policy_path}. Training from scratch.")

    # Rover obstacle-avoidance: in-process objective weight ramp (no per-chunk Isaac restarts).
    _rover_oa_ramp_installed = False
    try:
        import importlib.util

        _rover_ramp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rover_oa_objective_ramp.py")
        if os.path.isfile(_rover_ramp_path):
            _spec_rr = importlib.util.spec_from_file_location("_rover_oa_objective_ramp", _rover_ramp_path)
            if _spec_rr is not None and _spec_rr.loader is not None:
                _mod_rr = importlib.util.module_from_spec(_spec_rr)
                _spec_rr.loader.exec_module(_mod_rr)
                _rover_oa_ramp_installed = bool(
                    _mod_rr.install_rover_obstacle_avoidance_objective_ramp(runner, env, args_cli)
                )
    except Exception as _e_rr:
        logger.warning("[Rover OA] objective ramp install failed: %s", _e_rr)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # wrap runner to control logging frequency (default: every 200 iters for less I/O)
    reward_log_interval = getattr(args_cli, "reward_log_interval", 200)
    progress_log_interval = getattr(args_cli, "progress_log_interval", 100)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _workspace_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
    _print_progress_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_training_progress_log.py")
    _task = getattr(args_cli, "task", None) or ""
    _exp = getattr(agent_cfg, "experiment_name", None) or ""
    _exp_cli = getattr(args_cli, "experiment_name", None) or ""
    if _exp == "g1_standing_hand_position_v2" or "G1-Standing-Hand-Position-V2" in _task:
        _standing_v2_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_standing_hand_position_v2_progress_log.py")
        if os.path.isfile(_standing_v2_script):
            _print_progress_script = _standing_v2_script
    elif _exp == "g1_teleoperation_fixed_base" or "G1-Teleoperation-Fixed-Base" in _task:
        _fixed_base_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_teleoperation_fixed_base_progress_log.py")
        if os.path.isfile(_fixed_base_script):
            _print_progress_script = _fixed_base_script
    elif _exp == "g1_teleoperation" or "G1-Teleoperation" in _task:
        _teleop_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_teleoperation_progress_log.py")
        if os.path.isfile(_teleop_script):
            _print_progress_script = _teleop_script
    elif _exp == "g1_locomotion_v1_standing" or _exp == "g1_standing_hand_position_v1" or ("G1-Locomotion-V1-Direct" in _task and "Walking" not in _task) or "G1-Standing-Hand-Position-V1" in _task:
        _standing_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v1_standing_progress_log.py")
        if os.path.isfile(_standing_script):
            _print_progress_script = _standing_script
    elif _exp == "g1_locomotion_v1_walking" or "G1-Locomotion-V1-Walking" in _task:
        _walking_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v1_walking_progress_log.py")
        if os.path.isfile(_walking_script):
            _print_progress_script = _walking_script
    elif _exp == "g1_locomotion_walking_v2" or "G1-Locomotion-Walking-V2" in _task:
        _walking_v2_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_walking_v2_progress_log.py")
        if os.path.isfile(_walking_v2_script):
            _print_progress_script = _walking_v2_script
    elif _exp == "g1_locomotion_v3" or "G1-Locomotion-V3" in _task:
        _v3_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v3_progress_log.py")
        if os.path.isfile(_v3_script):
            _print_progress_script = _v3_script
    elif "g1_locomotion_v4" in (_exp or "") or "G1-Locomotion-V4" in _task:
        _v4_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v4_progress_log.py")
        if os.path.isfile(_v4_script):
            _print_progress_script = _v4_script
    elif _exp == "ion_horizontal_cruise_v1":
        _cruise_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_horizontal_cruise_v1_progress_log.py")
        if os.path.isfile(_cruise_script):
            _print_progress_script = _cruise_script
    elif _exp == "ion_horizontal_cruise_v2" or "Isaac-Ion-Horizontal-Cruise-V2-Direct-v0" in _task:
        _cruise_v2_script = os.path.join(
            _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_horizontal_cruise_v2_progress_log.py"
        )
        if os.path.isfile(_cruise_v2_script):
            _print_progress_script = _cruise_v2_script
    elif _exp == "ion_vertical_takeoff_direct" or "Isaac-Ion-Vertical-Takeoff-Direct-v0" in _task:
        _ion_vt_script = os.path.join(
            _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_vertical_takeoff_direct_progress_log.py"
        )
        if os.path.isfile(_ion_vt_script):
            _print_progress_script = _ion_vt_script
    elif _exp == "ion_vertical_takeoff_v2" or "Isaac-Ion-Vertical-Takeoff-V2-Direct-v0" in _task:
        _ion_v2_script = os.path.join(
            _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_vertical_takeoff_v2_progress_log.py"
        )
        if os.path.isfile(_ion_v2_script):
            _print_progress_script = _ion_v2_script
    elif _exp == "g1_standing_ik_v2" or "g1_standing_ik_v2_phase" in (_exp or "") or "G1-Standing-IK-V2" in _task:
        # Dedicated analyzer: scripts/reinforcement_learning/rsl_rl/print_g1_standing_ik_v2_progress_log.py
        _standing_ik_v2_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_standing_ik_v2_progress_log.py")
        if os.path.isfile(_standing_ik_v2_script):
            _print_progress_script = _standing_ik_v2_script
    elif _exp == "g1_standing_ik" or (
        "G1-Standing-IK" in _task and "G1-Standing-IK-V2" not in _task
    ):
        _standing_ik_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_standing_ik_progress_log.py")
        if os.path.isfile(_standing_ik_script):
            _print_progress_script = _standing_ik_script
    elif (
        "rover_disk_world_idle_to_forward" in (_exp or "")
        or "rover_disk_world_idle_to_forward" in (_exp_cli or "")
        or "Isaac-Rover-DiskWorld-IdleToForward" in _task
    ):
        _idle_fwd_script = os.path.join(
            _workspace_root,
            "scripts",
            "reinforcement_learning",
            "rsl_rl",
            "print_rover_idle_to_forward_progress_log.py",
        )
        if os.path.isfile(_idle_fwd_script):
            _print_progress_script = _idle_fwd_script
    elif (
        "rover_disk_world_obstacle_avoidance" in (_exp or "")
        or "rover_disk_world_obstacle_avoidance" in (_exp_cli or "")
        or "Rover-DiskWorld-ObstacleAvoidance" in _task
        or "Isaac-Rover-DiskWorld-ObstacleAvoidance" in _task
    ):
        _rover_oa_script = os.path.join(
            _workspace_root,
            "scripts",
            "reinforcement_learning",
            "rsl_rl",
            "print_rover_disk_world_obstacle_avoidance_progress_log.py",
        )
        if os.path.isfile(_rover_oa_script):
            _print_progress_script = _rover_oa_script
    elif getattr(agent_cfg, "experiment_name", "") in (
        "rover_oa_flat_env",
        "rover_oa_flat_physparam_env",
    ) or "Isaac-Rover-OAFlat" in _task:
        _rover_oa_flat_script = os.path.join(
            _workspace_root,
            "scripts",
            "reinforcement_learning",
            "rsl_rl",
            "print_rover_oa_flat_progress_log.py",
        )
        if os.path.isfile(_rover_oa_flat_script):
            _print_progress_script = _rover_oa_flat_script
    elif "rover_localization_env" in (_exp or "") or "Isaac-Rover-Localization" in _task:
        _rover_loc_script = os.path.join(
            _workspace_root,
            "scripts",
            "reinforcement_learning",
            "rsl_rl",
            "print_rover_localization_progress_log.py",
        )
        if os.path.isfile(_rover_loc_script):
            _print_progress_script = _rover_loc_script
    elif (
        "rover_disk_world" in (_exp or "")
        or "rover_disk_world" in (_exp_cli or "")
        or "Rover-DiskWorld" in _task
        or "Isaac-Rover-DiskWorld" in _task
        or "rover_disk_world_physparam" in (_exp or "")
        or "rover_disk_world_physparam" in (_exp_cli or "")
    ):
        _rover_script = os.path.join(
            _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_rover_disk_world_progress_log.py"
        )
        if os.path.isfile(_rover_script):
            _print_progress_script = _rover_script
    if not os.path.isfile(_print_progress_script):
        _print_progress_script = None

    # Rover localization: linear entropy-coefficient anneal (rsl_rl PPO has no entropy schedule).
    _loc_task_tail = (_task.split(":")[-1] if ":" in _task else _task)
    _loc_entropy_anneal_installed = False
    if "Isaac-Rover-DiskWorld-Localization" in _loc_task_tail:
        _ent_final = getattr(args_cli, "ppo_entropy_coef_final", None)
        if _ent_final is not None:
            _ent_start = getattr(args_cli, "ppo_entropy_coef", None)
            if _ent_start is None:
                _ent_start = float(agent_cfg.algorithm.entropy_coef)
            else:
                _ent_start = float(_ent_start)
            _ent_final = float(_ent_final)
            _ent_max_it = max(int(agent_cfg.max_iterations), 1)
            _ent_upd_idx = [int(getattr(runner, "current_learning_iteration", 0))]
            _orig_alg_update = runner.alg.update

            def _localization_alg_update_with_entropy_anneal():
                den = max(_ent_max_it - 1, 1)
                prog = min(1.0, max(0.0, float(_ent_upd_idx[0]) / float(den)))
                runner.alg.entropy_coef = _ent_start + (_ent_final - _ent_start) * prog
                out = _orig_alg_update()
                _ent_upd_idx[0] += 1
                return out

            runner.alg.update = _localization_alg_update_with_entropy_anneal
            _loc_entropy_anneal_installed = True
            print(
                "[INFO] Rover localization: entropy_coef linear anneal "
                f"{_ent_start:g} → {_ent_final:g} over {_ent_max_it} iterations (per PPO update)."
            )

    _write_rover_localization_train_debug_snap(
        log_dir=log_dir,
        stage="after_runner_ready_before_logging_wrapper",
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        args_cli=args_cli,
        runner=runner,
        entropy_anneal_installed=_loc_entropy_anneal_installed,
        env=env,
    )

    runner = LoggingWrapper(
        runner,
        log_interval=reward_log_interval,
        progress_log_dir=log_dir,
        progress_log_interval=progress_log_interval,
        progress_plot_script=_print_progress_script,
    )

    # run training — RSL-RL default True for most tasks; rover OA defaults False (random truncation was ~37 steps).
    _init_random_ep_len = True
    if _rover_dw_obstacle:
        _init_random_ep_len = False
    if int(getattr(args_cli, "rover_oa_objective_ramp_enable", 0) or 0) == 1:
        if getattr(args_cli, "rover_oa_init_random_episode_length", None) is None:
            _init_random_ep_len = False
        else:
            _init_random_ep_len = bool(int(args_cli.rover_oa_init_random_episode_length))
    elif getattr(args_cli, "rover_oa_init_random_episode_length", None) is not None:
        _init_random_ep_len = bool(int(args_cli.rover_oa_init_random_episode_length))
    if _rover_dw_obstacle:
        print(
            f"[INFO] Rover obstacle-avoidance: init_at_random_ep_len={_init_random_ep_len} "
            "(False = full max episode length at reset; True = RSL-RL random truncation — usually worse for OA)."
        )
        if not _init_random_ep_len:
            try:
                _mel = int(getattr(runner.env, "max_episode_length", 0) or 0)
                _nsp = int(getattr(runner, "num_steps_per_env", 64) or 64)
                if _mel > 0 and _nsp > 0:
                    _it_first_done = (_mel + _nsp - 1) // _nsp
                    print(
                        "[INFO] Rover obstacle-avoidance: IsaacLab only writes Episode_Reward/* to extras on env "
                        "reset. With a long horizon, console Episode_Reward/* and rsl_rl Train/mean_reward stay at "
                        f"0 until the first timeouts/terminations (often around iteration ≳ {_it_first_done} at "
                        f"{_nsp} steps/env/iter). PPO still optimizes per-step reward."
                    )
            except Exception:
                pass
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=_init_random_ep_len)

    # Save final model as final_model.pt
    # RSL-RL saves checkpoints to log_dir/policy_checkpoints/model_<iteration>.pt
    # Copy the last checkpoint to log_dir/final_model.pt
    try:
        import glob
        import shutil

        # Check policy_checkpoints subfolder first, then root (backward compat)
        _ckpt_subdir = "policy_checkpoints"
        checkpoint_pattern = os.path.join(log_dir, _ckpt_subdir, "model_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        if not checkpoint_files:
            checkpoint_pattern = os.path.join(log_dir, "checkpoints", "model_*.pt")
            checkpoint_files = glob.glob(checkpoint_pattern)
        if not checkpoint_files:
            checkpoint_pattern = os.path.join(log_dir, "model_*.pt")
            checkpoint_files = glob.glob(checkpoint_pattern)

        if checkpoint_files:
            # Sort by iteration number (extract number from filename)
            def get_iteration(filename):
                import re
                match = re.search(r'model_(\d+)\.pt', os.path.basename(filename))
                return int(match.group(1)) if match else 0

            checkpoint_files.sort(key=get_iteration, reverse=True)
            last_checkpoint = checkpoint_files[0]

            # Copy to final_model.pt
            final_model_path = os.path.join(log_dir, "final_model.pt")
            shutil.copy2(last_checkpoint, final_model_path)
            print(f"[INFO] Final model saved to: {final_model_path} (copied from {os.path.basename(last_checkpoint)})")
        else:
            logger.warning(f"[WARNING] No checkpoint files found in {log_dir}. Cannot create final_model.pt")
    except Exception as e:
        logger.warning(f"[WARNING] Failed to save final model: {e}")
        import traceback
        logger.debug(f"[WARNING] Traceback: {traceback.format_exc()}")

    # Generate training_progress_analysis.png from training_progress_log.txt (written every progress_log_interval)
    training_progress_log = os.path.join(log_dir, "training_progress_log.txt")
    if os.path.isfile(training_progress_log):
        try:
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _workspace_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
            _print_progress_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_training_progress_log.py")
            _task = getattr(args_cli, "task", None) or ""
            _exp = getattr(agent_cfg, "experiment_name", None) or ""
            _exp_cli = getattr(args_cli, "experiment_name", None) or ""
            if _exp == "g1_standing_hand_position_v2" or "G1-Standing-Hand-Position-V2" in _task:
                _standing_v2_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_standing_hand_position_v2_progress_log.py")
                if os.path.isfile(_standing_v2_script):
                    _print_progress_script = _standing_v2_script
            elif _exp == "g1_teleoperation_fixed_base" or "G1-Teleoperation-Fixed-Base" in _task:
                _fixed_base_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_teleoperation_fixed_base_progress_log.py")
                if os.path.isfile(_fixed_base_script):
                    _print_progress_script = _fixed_base_script
            elif _exp == "g1_teleoperation" or "G1-Teleoperation" in _task:
                _teleop_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_teleoperation_progress_log.py")
                if os.path.isfile(_teleop_script):
                    _print_progress_script = _teleop_script
            elif _exp == "g1_locomotion_v1_standing" or _exp == "g1_standing_hand_position_v1" or ("G1-Locomotion-V1-Direct" in _task and "Walking" not in _task) or "G1-Standing-Hand-Position-V1" in _task:
                _standing_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v1_standing_progress_log.py")
                if os.path.isfile(_standing_script):
                    _print_progress_script = _standing_script
            elif _exp == "g1_locomotion_v1_walking" or "G1-Locomotion-V1-Walking" in _task:
                _walking_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v1_walking_progress_log.py")
                if os.path.isfile(_walking_script):
                    _print_progress_script = _walking_script
            elif _exp == "g1_locomotion_walking_v2" or "G1-Locomotion-Walking-V2" in _task:
                _walking_v2_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_walking_v2_progress_log.py")
                if os.path.isfile(_walking_v2_script):
                    _print_progress_script = _walking_v2_script
            elif _exp == "g1_locomotion_v3" or "G1-Locomotion-V3" in _task:
                _v3_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v3_progress_log.py")
                if os.path.isfile(_v3_script):
                    _print_progress_script = _v3_script
            elif "g1_locomotion_v4" in (_exp or "") or "G1-Locomotion-V4" in _task:
                _v4_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_locomotion_v4_progress_log.py")
                if os.path.isfile(_v4_script):
                    _print_progress_script = _v4_script
            elif _exp == "ion_horizontal_cruise_v1":
                _cruise_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_horizontal_cruise_v1_progress_log.py")
                if os.path.isfile(_cruise_script):
                    _print_progress_script = _cruise_script
            elif _exp == "ion_horizontal_cruise_v2" or "Isaac-Ion-Horizontal-Cruise-V2-Direct-v0" in _task:
                _cruise_v2_script = os.path.join(
                    _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_horizontal_cruise_v2_progress_log.py"
                )
                if os.path.isfile(_cruise_v2_script):
                    _print_progress_script = _cruise_v2_script
            elif _exp == "ion_vertical_takeoff_direct" or "Isaac-Ion-Vertical-Takeoff-Direct-v0" in _task:
                _ion_vt_script = os.path.join(
                    _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_vertical_takeoff_direct_progress_log.py"
                )
                if os.path.isfile(_ion_vt_script):
                    _print_progress_script = _ion_vt_script
            elif _exp == "ion_vertical_takeoff_v2" or "Isaac-Ion-Vertical-Takeoff-V2-Direct-v0" in _task:
                _ion_v2_script = os.path.join(
                    _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_ion_vertical_takeoff_v2_progress_log.py"
                )
                if os.path.isfile(_ion_v2_script):
                    _print_progress_script = _ion_v2_script
            elif _exp == "g1_standing_ik_v2" or "g1_standing_ik_v2_phase" in (_exp or "") or "G1-Standing-IK-V2" in _task:
                _standing_ik_v2_script = os.path.join(
                    _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_standing_ik_v2_progress_log.py"
                )
                if os.path.isfile(_standing_ik_v2_script):
                    _print_progress_script = _standing_ik_v2_script
            elif _exp == "g1_standing_ik" or (
                "G1-Standing-IK" in _task and "G1-Standing-IK-V2" not in _task
            ):
                _standing_ik_script = os.path.join(_workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_g1_standing_ik_progress_log.py")
                if os.path.isfile(_standing_ik_script):
                    _print_progress_script = _standing_ik_script
            elif (
                "rover_disk_world_idle_to_forward" in (_exp or "")
                or "rover_disk_world_idle_to_forward" in (_exp_cli or "")
                or "Isaac-Rover-DiskWorld-IdleToForward" in _task
            ):
                _idle_fwd_script = os.path.join(
                    _workspace_root,
                    "scripts",
                    "reinforcement_learning",
                    "rsl_rl",
                    "print_rover_idle_to_forward_progress_log.py",
                )
                if os.path.isfile(_idle_fwd_script):
                    _print_progress_script = _idle_fwd_script
            elif (
                "rover_disk_world_obstacle_avoidance" in (_exp or "")
                or "rover_disk_world_obstacle_avoidance" in (_exp_cli or "")
                or "Rover-DiskWorld-ObstacleAvoidance" in _task
                or "Isaac-Rover-DiskWorld-ObstacleAvoidance" in _task
            ):
                _rover_oa_script = os.path.join(
                    _workspace_root,
                    "scripts",
                    "reinforcement_learning",
                    "rsl_rl",
                    "print_rover_disk_world_obstacle_avoidance_progress_log.py",
                )
                if os.path.isfile(_rover_oa_script):
                    _print_progress_script = _rover_oa_script
            elif _exp in ("rover_oa_flat_env",) or "Isaac-Rover-OAFlat" in _task:
                if "Physparam" not in (_task or "") and _exp != "rover_oa_flat_physparam_env":
                    _rover_oa_flat_script = os.path.join(
                        _workspace_root, "scripts", "reinforcement_learning", "rsl_rl",
                        "print_rover_oa_flat_progress_log.py",
                    )
                    if os.path.isfile(_rover_oa_flat_script):
                        _print_progress_script = _rover_oa_flat_script
            if _exp in ("rover_oa_flat_physparam_env",) or "Physparam" in (_task or ""):
                if "Isaac-Rover-OAFlat" in (_task or ""):
                    _rover_physparam_script = os.path.join(
                        _workspace_root, "scripts", "reinforcement_learning", "rsl_rl",
                        "print_rover_oa_flat_physparam_progress_log.py",
                    )
                    if os.path.isfile(_rover_physparam_script):
                        _print_progress_script = _rover_physparam_script
            elif "rover_localization_env" in (_exp or "") or "Isaac-Rover-Localization" in _task:
                _rover_loc_script = os.path.join(
                    _workspace_root,
                    "scripts",
                    "reinforcement_learning",
                    "rsl_rl",
                    "print_rover_localization_progress_log.py",
                )
                if os.path.isfile(_rover_loc_script):
                    _print_progress_script = _rover_loc_script
            elif (
                "rover_disk_world" in (_exp or "")
                or "rover_disk_world" in (_exp_cli or "")
                or "Rover-DiskWorld" in _task
                or "Isaac-Rover-DiskWorld" in _task
                or "rover_disk_world_physparam" in (_exp or "")
                or "rover_disk_world_physparam" in (_exp_cli or "")
            ):
                _rover_script = os.path.join(
                    _workspace_root, "scripts", "reinforcement_learning", "rsl_rl", "print_rover_disk_world_progress_log.py"
                )
                if os.path.isfile(_rover_script):
                    _print_progress_script = _rover_script
            if os.path.isfile(_print_progress_script):
                import subprocess
                result = subprocess.run(
                    [sys.executable, _print_progress_script, log_dir, "--plot-only"],
                    cwd=_workspace_root,
                    timeout=60,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and os.path.isfile(os.path.join(log_dir, "training_progress_analysis.png")):
                    print(f"[INFO] Visual analysis saved to {os.path.join(log_dir, 'training_progress_analysis.png')}")
                elif result.returncode != 0 and result.stderr:
                    logger.warning(f"[WARNING] Could not generate training_progress_analysis.png: {result.stderr.strip()}")
        except Exception as e_plot:
            logger.warning(f"[WARNING] Could not generate training_progress_analysis.png: {e_plot}")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

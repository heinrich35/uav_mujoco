# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# In-process inverse kinematics for G1 right arm using ikpy (pure Python).
# No server: load chain once from URDF, compute IK in the same process.
"""G1 right-arm chain IK via ikpy. Use with --ik_chain_urdf in play_g1_teleop_ik_target_vis.py."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

# Optional: avoid hard dependency on scipy if we have a tiny quat->matrix
try:
    from scipy.spatial.transform import Rotation as R
    _has_scipy = True
except ImportError:
    _has_scipy = False

_chain_cache: Any = None
_chain_urdf_path: Optional[str] = None

# Joint order must match RIGHT_ARM_JOINT_EXPR in the teleop script
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]


def _quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3 rotation matrix."""
    if _has_scipy:
        # scipy: from_quat expects (x,y,z,w)
        r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return r.as_matrix()
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_chain(urdf_path: str, use_cache: bool = True) -> Any:
    """Load ikpy chain from URDF. Returns None if ikpy missing or load fails.
    use_cache: reuse same chain if urdf_path unchanged."""
    global _chain_cache, _chain_urdf_path
    if use_cache and _chain_cache is not None and _chain_urdf_path == urdf_path:
        return _chain_cache
    try:
        import ikpy.chain
    except ImportError:
        return None
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        return None
    try:
        # G1_right_arm_only.urdf: base_link -> torso -> 5 revolute -> right_palm
        # ikpy chain = OriginLink (0) + 7 URDF joints (1=base_to_torso fixed, 2-6=5 revolute, 7=right_palm fixed) = 8 links
        # Pass mask at build time to avoid "fixed but set as active" warnings
        active_links_mask = [False, False, True, True, True, True, True, False]
        chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["base_link"],
            base_element_type="link",
            active_links_mask=active_links_mask,
            name="g1_right_arm",
            symbolic=False,
        )
        _chain_cache = chain
        _chain_urdf_path = urdf_path
        return chain
    except Exception:
        return None


def compute_ik(
    chain: Any,
    target_position: np.ndarray,
    target_quat_wxyz: np.ndarray,
    seed_joints: np.ndarray,
) -> Optional[np.ndarray]:
    """Solve IK for target pose in chain base frame.
    target_position: (3,) in meters; target_quat_wxyz: (4,) w,x,y,z.
    seed_joints: (5,) current joint positions for the 5 revolute joints.
    Returns (5,) joint positions or None on failure."""
    target_position = np.asarray(target_position, dtype=np.float64).ravel()
    target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64).ravel()
    seed_joints = np.asarray(seed_joints, dtype=np.float64).ravel()
    if target_position.size != 3 or target_quat_wxyz.size != 4 or seed_joints.size != 5:
        return None
    # Build 4x4 target matrix
    rot = _quat_to_rotation_matrix(target_quat_wxyz)
    target_frame = np.eye(4, dtype=np.float64)
    target_frame[:3, :3] = rot
    target_frame[:3, 3] = target_position
    # Full chain positions: [origin, base_to_torso, sh_pitch, sh_roll, sh_yaw, el_pitch, el_roll, palm]
    full_seed = np.zeros(len(chain.links), dtype=np.float64)
    full_seed[2:7] = seed_joints
    # Solve
    result = chain.inverse_kinematics_frame(
        target_frame,
        initial_position=full_seed,
        max_iter=80,
        orientation_mode=None,  # position only (more robust)
    )
    if result is None:
        return None
    # Try with orientation if position-only fails or we want orientation
    try:
        active = np.array(chain.active_from_full(result), dtype=np.float64)
    except Exception:
        return None
    if active.size != 5:
        return None
    # Sanity: no NaN
    if not np.all(np.isfinite(active)):
        return None
    return active


def compute_ik_position_only(
    chain: Any,
    target_position: np.ndarray,
    seed_joints: np.ndarray,
) -> Optional[np.ndarray]:
    """IK targeting only position (identity orientation). Convenience wrapper."""
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return compute_ik(chain, target_position, identity_quat, seed_joints)

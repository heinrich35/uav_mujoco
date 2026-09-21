# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP functions for the pond environment."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv


def reset_pond_scene(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset all pond birds and distractors with random positions.

    Gosling is placed at (0, 0, 0) with identity rotation.
    Ducks (3×), swans (3×), and distractors (3× each: rock, turtle, lilypad, log)
    are randomly placed with >2m clearance between all instances.

    Birds get random yaw (Z-axis) rotations. Distractors get random yaw.
    Distractors are kinematic (static) rigid objects.

    Spawn area: circle of radius 8m, z=0 (water surface).
    """
    # Pond parameters
    pond_radius = 8.0
    min_clearance = 2.0
    max_attempts = 200

    num_ducks = 3
    num_swans = 3
    num_distractors = 12  # 3 rock + 3 turtle + 3 lilypad + 3 log
    total_birds = 1 + num_ducks + num_swans
    total_objects = total_birds + num_distractors

    device = env.device

    # Fixed gosling position
    positions = [(0.0, 0.0)]
    occupied = [(0.0, 0.0)]  # (x, y) list for clearance checks

    def try_place():
        """Try to find a random position with clearance.  Returns (x, y) or None."""
        for _ in range(max_attempts):
            angle = torch.rand(1, device=device).item() * 2.0 * 3.14159265
            radius = torch.rand(1, device=device).item() * pond_radius
            x = radius * torch.cos(torch.tensor(angle, device=device)).item()
            y = radius * torch.sin(torch.tensor(angle, device=device)).item()
            if all((x - ox) ** 2 + (y - oy) ** 2 >= min_clearance ** 2 for ox, oy in occupied):
                return (x, y)
        return None

    def fallback_place(idx: int):
        """Fallback: place on a ring at 70% pond radius, evenly spaced."""
        angle = idx * 2.0 * 3.14159265 / max(total_objects, 1)
        x = pond_radius * 0.7 * torch.cos(torch.tensor(angle)).item()
        y = pond_radius * 0.7 * torch.sin(torch.tensor(angle)).item()
        return (x, y)

    # Random positions for ducks
    for i in range(num_ducks):
        pos = try_place()
        if pos is None:
            pos = fallback_place(len(positions))
        positions.append(pos)
        occupied.append(pos)

    # Random positions for swans
    for i in range(num_swans):
        pos = try_place()
        if pos is None:
            pos = fallback_place(len(positions))
        positions.append(pos)
        occupied.append(pos)

    # Apply bird positions: index 0 = gosling, 1-3 = ducks, 4-6 = swans
    robot_names = ["gosling"] + [f"duck_{i}" for i in range(num_ducks)] + [f"swan_{i}" for i in range(num_swans)]

    for i, name in enumerate(robot_names):
        if name not in env.scene.articulations:
            continue
        robot = env.scene[name]
        x, y = positions[i]

        # Random yaw angle (Z rotation) for ducks and swans
        if name.startswith("duck") or name.startswith("swan"):
            yaw = torch.rand(1, device=device).item() * 2.0 * 3.14159265
            half = yaw / 2.0
            qw = torch.cos(torch.tensor(half, device=device)).item()
            qz = torch.sin(torch.tensor(half, device=device)).item()
            root_quat = torch.tensor([qw, 0.0, 0.0, qz], device=device)
        else:
            root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        # Set root state (pos, quat, lin_vel, ang_vel) = 13 elements
        root_state = torch.zeros(1, 13, device=device)
        root_state[:, :3] = torch.tensor([x, y, 0.0], device=device)
        root_state[:, 3:7] = root_quat
        robot.write_root_state_to_sim(root_state)

    # ------------------------------------------------------------------
    # Place distractors (kinematic rigid objects)
    # ------------------------------------------------------------------
    distractor_types = ["rock", "turtle", "lilypad", "log"]
    distractor_names = []
    for dtype in distractor_types:
        for i in range(3):
            distractor_names.append(f"{dtype}_{i}")

    for name in distractor_names:
        pos = try_place()
        if pos is None:
            pos = fallback_place(len(positions))
        positions.append(pos)
        occupied.append(pos)

    for i, name in enumerate(distractor_names):
        if name not in env.scene.rigid_objects:
            continue
        obj = env.scene[name]
        x, y = positions[len(robot_names) + i]

        # Random yaw for distractors too
        yaw = torch.rand(1, device=device).item() * 2.0 * 3.14159265
        half = yaw / 2.0
        qw = torch.cos(torch.tensor(half, device=device)).item()
        qz = torch.sin(torch.tensor(half, device=device)).item()
        root_quat = torch.tensor([qw, 0.0, 0.0, qz], device=device)

        # write_root_pose_to_sim takes position + quaternion
        root_pose = torch.zeros(1, 7, device=device)
        root_pose[:, :3] = torch.tensor([x, y, 0.0], device=device)
        root_pose[:, 3:7] = root_quat
        obj.write_root_pose_to_sim(root_pose)

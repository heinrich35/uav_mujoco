# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pond RL environment — fluid dynamics simulation with teleoperable floating robots."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class PondRLEnv(ManagerBasedRLEnv):
    """RL environment for the pond with fluid dynamics simulation.

    Simulates birds floating on a water surface with buoyancy, hydrodynamic
    drag, and paddle actuation. The water surface is at world z = 0.

    * **Buoyancy**: upward force proportional to submersion depth
    * **Linear drag**: resistive force opposing velocity (anisotropic)
    * **Angular drag**: rotational damping
    * **Paddle actuation**: forward force + yaw torque per robot

    Only the gosling is teleoperable via keyboard.
    Duck and swan instances float passively but accept paddle commands
    via :meth:`set_robot_paddle_command` for future programmatic actuation.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._fluid_cfg = cfg.fluid

        # All articulation names in the scene
        self._robot_names = list(self.scene.articulations.keys())
        print(f"[INFO] Pond robots ({len(self._robot_names)}): {self._robot_names}")

        # Paddle commands per robot: {name: tensor(num_envs, 2)}
        self._paddle_cmds: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, 2, device=self.device)
            for name in self._robot_names
        }

        # Body IDs cached after first step
        self._body_ids: dict[str, torch.Tensor] = {}

        # Per-robot displaced volume (m³) and drag scale
        self._displaced_vol: dict[str, float] = {}
        self._drag_scale: dict[str, float] = {}
        for name in self._robot_names:
            if "gosling" in name:
                self._displaced_vol[name] = 0.008   # 8 L
                self._drag_scale[name] = 1.0
            elif "duck" in name:
                self._displaced_vol[name] = 0.014   # 14 L
                self._drag_scale[name] = 1.5
            elif "swan" in name:
                self._displaced_vol[name] = 0.025   # 25 L
                self._drag_scale[name] = 2.5
            else:
                self._displaced_vol[name] = 0.008
                self._drag_scale[name] = 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_paddle_command(self, forward_force_n: float, yaw_torque_nm: float):
        """Set paddle command for gosling (keyboard teleoperation)."""
        self.set_robot_paddle_command("gosling", forward_force_n, yaw_torque_nm)

    def set_robot_paddle_command(self, robot_name: str, forward_force_n: float, yaw_torque_nm: float):
        """Set paddle actuation command for any robot.

        Args:
            robot_name: Name of the robot in the scene (e.g. 'gosling', 'duck_0').
            forward_force_n: Forward force in body +X (N).
            yaw_torque_nm: Yaw torque around body +Z (N·m).
        """
        if robot_name not in self._paddle_cmds:
            return
        self._paddle_cmds[robot_name][:] = torch.tensor(
            [forward_force_n, yaw_torque_nm],
            dtype=torch.float32,
            device=self.device,
        )

    def get_robot_position(self) -> torch.Tensor:
        """Get gosling world position. Returns (num_envs, 3)."""
        return self.scene["gosling"].data.root_pos_w

    def get_robot_velocity(self) -> torch.Tensor:
        """Get gosling world linear velocity. Returns (num_envs, 3)."""
        return self.scene["gosling"].data.root_lin_vel_w

    def get_submersion_depth(self) -> torch.Tensor:
        """Get gosling submersion depth (positive = below water). Returns (num_envs,)."""
        pos_z = self.scene["gosling"].data.root_pos_w[:, 2]
        return torch.clamp(-pos_z, min=0.0)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def step(self, action):
        """Apply fluid forces to all robots, then step physics."""
        if not self._body_ids:
            for name in self._robot_names:
                self._body_ids[name] = self.scene[name]._ALL_INDICES.to(self.device)

        for name in self._robot_names:
            self._apply_fluid_forces(name)

        return super().step(action)

    # ------------------------------------------------------------------
    # Fluid dynamics
    # ------------------------------------------------------------------

    def _apply_fluid_forces(self, robot_name: str):
        """Compute and apply buoyancy, drag, and paddle forces to one robot."""
        robot = self.scene[robot_name]
        num_envs = self.num_envs
        device = self.device
        cfg = self._fluid_cfg
        paddle = self._paddle_cmds[robot_name]  # (N, 2)
        displaced_vol = self._displaced_vol[robot_name]
        dscale = self._drag_scale.get(robot_name, 1.0)

        # State
        pos_w = robot.data.root_pos_w
        vel_w = robot.data.root_lin_vel_w
        ang_vel_w = robot.data.root_ang_vel_w
        quat_w = robot.data.root_quat_w

        body_x, body_y, body_z = _quat_to_body_axes(quat_w)

        # Submersion
        depth = torch.clamp(-pos_w[:, 2], min=0.0)
        frac = torch.clamp(depth / cfg.buoyancy_ramp_depth, 0.0, 1.0)
        frac_unsq = frac.unsqueeze(-1)

        # --- Buoyancy (world +Z) ---
        full_buoyancy = displaced_vol * cfg.water_density * cfg.gravity_mag
        force_buoyancy = torch.zeros(num_envs, 3, device=device)
        force_buoyancy[:, 2] = frac * full_buoyancy

        # --- Linear drag (body-frame) ---
        vel_body = _world_to_body(vel_w, body_x, body_y, body_z)
        drag_air = torch.tensor(cfg.drag_linear_air, device=device)
        drag_water = torch.tensor(cfg.drag_linear_water, device=device)
        drag_coeffs = (1.0 - frac_unsq) * drag_air + frac_unsq * drag_water * dscale
        force_drag_body = -drag_coeffs * vel_body
        force_drag_w = _body_to_world(force_drag_body, body_x, body_y, body_z)

        # --- Angular drag (body-frame) ---
        ang_vel_body = _world_to_body(ang_vel_w, body_x, body_y, body_z)
        ang_drag_air = torch.tensor(cfg.drag_angular_air, device=device)
        ang_drag_water = torch.tensor(cfg.drag_angular_water, device=device)
        ang_drag_coeffs = (1.0 - frac_unsq) * ang_drag_air + frac_unsq * ang_drag_water * dscale
        torque_drag_body = -ang_drag_coeffs * ang_vel_body
        torque_drag_w = _body_to_world(torque_drag_body, body_x, body_y, body_z)

        # --- Paddle actuation (body-frame) ---
        force_paddle_body = torch.zeros(num_envs, 3, device=device)
        force_paddle_body[:, 0] = paddle[:, 0]
        torque_paddle_body = torch.zeros(num_envs, 3, device=device)
        torque_paddle_body[:, 2] = paddle[:, 1]
        force_paddle_w = _body_to_world(force_paddle_body, body_x, body_y, body_z)
        torque_paddle_w = _body_to_world(torque_paddle_body, body_x, body_y, body_z)

        # --- Total ---
        total_force_w = force_buoyancy + force_drag_w + force_paddle_w
        total_torque_w = torque_drag_w + torque_paddle_w

        robot.set_external_force_and_torque(
            forces=total_force_w.unsqueeze(1),
            torques=total_torque_w.unsqueeze(1),
            body_ids=self._body_ids[robot_name],
            is_global=True,
        )


# ------------------------------------------------------------------
# Frame conversion utilities (module-level)
# ------------------------------------------------------------------

def _quat_to_body_axes(q_wxyz: torch.Tensor):
    """Convert quaternion (w,x,y,z) to body-frame axis vectors. Each (N,3)."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    body_x = torch.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], dim=-1)
    body_y = torch.stack([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], dim=-1)
    body_z = torch.stack([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)], dim=-1)
    return body_x, body_y, body_z


def _world_to_body(v_w, bx, by, bz):
    """Convert world-frame vectors to body frame (N,3)."""
    return torch.stack([(v_w*bx).sum(-1), (v_w*by).sum(-1), (v_w*bz).sum(-1)], dim=-1)


def _body_to_world(v_b, bx, by, bz):
    """Convert body-frame vectors to world frame (N,3)."""
    return bx*v_b[:, 0:1] + by*v_b[:, 1:2] + bz*v_b[:, 2:3]

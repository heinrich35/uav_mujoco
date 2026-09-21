"""Motor bank: 3 thrust points on a line along the y axis at the BACK (-x).

Layout (body frame, from design.yaml `motors`):

    middle motor at y=0, outers at y=±motor_gap, all at x = motor_x(geom)
    (aft, i.e. the -x side of the vehicle), z=0.

Each motor pushes along +x_body with thrust T_i = throttle_i * max_thrust,
throttle in [0, 1], through a first-order spool lag dT/dt = (T_cmd - T)/tau.

Resultant (planar):
    Fx = sum_i T_i                        (forward force)
    Mz = sum_i (r_i x F_i)_z = -sum_i y_i * T_i

(x drops out of Mz because every force is parallel to x — the motor x still
matters for where the line sits on the airframe and for any future pitch DOF.)
Differential throttle across the y line is the yaw actuator.
"""
import numpy as np


class MotorBank:
    def __init__(self, cfg, x_pos, dt):
        m = cfg["motors"]
        assert m["count"] == 3, "v0 bank is exactly the 3-motor y line"
        self.gap = float(m["motor_gap"])
        self.y = np.array([-self.gap, 0.0, self.gap])
        self.x = float(x_pos)
        self.t_max = float(m["max_thrust"])
        self.tau = max(float(m["tau"]), 1e-3)
        self.dt = float(dt)
        self.thrust = np.zeros(3)          # current N per motor

    def reset(self, thrust=None):
        if thrust is None:
            self.thrust = np.full(3, float(self.t_max) / 3.0)
        else:  # a scalar trim thrust is legal — broadcast it across the bank
            self.thrust = np.broadcast_to(np.asarray(thrust, float),
                                          (3,)).copy()

    def command(self, throttle):
        """Advance the spool lag one control step toward the commanded throttle."""
        tgt = np.clip(np.asarray(throttle, float), 0.0, 1.0) * self.t_max
        self.thrust += (tgt - self.thrust) * min(1.0, self.dt / self.tau)
        return self.thrust

    def fx_mz(self):
        """(total forward force [N], yaw moment about CG [N m])."""
        fx = float(np.sum(self.thrust))
        mz = -float(self.y @ self.thrust)
        return fx, mz

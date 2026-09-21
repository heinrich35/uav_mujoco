"""VehicleEnv — planar flight control env for the 3-motor aerial vehicle.

Objective: fly FORWARD (+x) at CONSTANT speed (aero.cruise_speed), wings level
with the x axis (y->0, yaw->0), cheaply.

MDP (per design — the geometry is FIXED within an episode and passed in as an
analyzed VehicleGeom; the design descriptor rides in the obs so one policy can
amortize across designs — pipeline pattern A):

    action  (3,) throttle per motor, Box(0, 1)
    obs     [u/vt, v/vt, psi, r*b/(2vt), y, throttle(3)/1, design_vector(9)]
    reward  +w_alive - w_speed|u-vt| - w_lateral|y| - w_yaw|psi|
            - w_effort*sum(a_i^2)   (crash: terminal -crash_penalty)

Physics (planar, body frame, first principles only where cheap):
    udot = (Fx - D*cos(beta_side)) / m + v*r
    vdot = (-D*sin(beta_side) - Y_beta) / m - u*r
    rdot = (Mz_thrust + Mz_weathervane) / Izz
with D = q*S*CD at the cruise polar (frozen alpha — pitch is out of scope v0),
beta_side = atan2(v, u), Y_beta = q*S*cy_beta*beta_side,
Mz_weathervane = -q*S*b*cl_beta*beta_side (restores nose into the wind).

State is (x, y, psi, u, v, r) in a world with +x downwind-free: thrust fights
drag only. Crash = |y| > y_limit or |psi| > yaw_limit.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import build, load_design, overrides_to_cfg  # noqa: E402
from motors import MotorBank  # noqa: E402


class VehicleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, geom=None, cfg=None, overrides=None, seed=None):
        super().__init__()
        cfg = cfg if cfg is not None else load_design()
        cfg = overrides_to_cfg(cfg, overrides)
        self.cfg = cfg
        self.geom = geom if geom is not None else build(cfg)
        self.sim = cfg["sim"]
        self.rw = cfg["reward"]
        self.aero_cfg = cfg["aero"]
        self.vt = float(self.aero_cfg["cruise_speed"])
        self.dt = float(self.sim["dt"])
        self.n_steps = int(round(float(self.sim["episode_s"]) / self.dt))

        self.motors = MotorBank(cfg, x_pos=0.0, dt=self.dt)
        from geometry import motor_x
        self.motors.x = float(motor_x(self.geom))

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(5 + 3 + len(self.geom.design_vector),),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self._state = None
        self._throttle = np.zeros(3)
        self._steps = 0

    # -- model -----------------------------------------------------------
    def _derivs(self, state, fx_thrust, mz_thrust):
        _x, _y, psi, u, v, r = state
        a = self.aero_cfg
        V = max(np.hypot(u, v), 1e-6)
        q = 0.5 * a["rho"] * V * V
        beta = np.arctan2(v, max(u, 1e-3))
        CD = self.geom.CD0 + self.geom.k_ind * self.geom.cl_c ** 2
        D = q * self.geom.S_ref * CD
        Y = -q * self.geom.S_ref * a["cy_beta"] * beta
        # weathervane yaws the nose TOWARD the relative wind (beta > 0 =
        # wind from the right -> yaw right = +Mz); cl_r damps the yaw rate.
        # (The pre-09-18 build had the weathervane sign flipped — it fought
        # the policy instead of stabilizing it.)
        Mwv = q * self.geom.S_ref * self.geom.span * a["cl_beta"] * beta
        Mrd = -(q * self.geom.S_ref * self.geom.span
                * a["cl_r"] * self.geom.span / (2.0 * V)) * r

        m, izz = self.geom.mass, self.geom.izz
        udot = (fx_thrust - D * (u / V)) / m + v * r
        vdot = (Y - D * (v / V)) / m - u * r
        rdot = (mz_thrust + Mwv + Mrd) / izz
        return np.array([u * np.cos(psi) - v * np.sin(psi),
                         u * np.sin(psi) + v * np.cos(psi),
                         r, udot, vdot, rdot])

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        u0 = 0.7 * self.vt + self.rng.uniform(-0.3, 0.3)
        trim = (0.5 * self.aero_cfg["rho"] * u0 ** 2 * self.geom.S_ref *
                (self.geom.CD0 + self.geom.k_ind * self.geom.cl_c ** 2) / 3.0)
        self._state = np.array([0.0,
                                self.rng.uniform(-0.3, 0.3),
                                self.rng.uniform(-0.15, 0.15),
                                max(u0, 0.5), 0.0, 0.0])
        self._throttle = np.zeros(3)
        self._steps = 0
        self.motors.reset(thrust=np.clip(trim / self.motors.t_max, 0, 1) *
                          self.motors.t_max)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float).ravel(), 0.0, 1.0)
        self._throttle = a
        self.motors.command(a)
        fx, mz = self.motors.fx_mz()

        # substep the integrator for a stable lag/drag coupling
        n_sub = 2
        h = self.dt / n_sub
        for _ in range(n_sub):
            self._state = self._state + self._derivs(self._state, fx, mz) * h
        self._steps += 1

        x, y, psi, u, v, r = self._state
        crashed = (abs(y) > self.sim["y_limit"] or
                   abs(psi) > np.deg2rad(self.sim["yaw_limit_deg"]))
        rw = self.rw
        reward = (rw.get("w_alive", 0.0)
                  - rw["w_speed"] * abs(u - self.vt)
                  - rw.get("w_side_vel", 0.0) * abs(v) / self.vt
                  - rw["w_lateral"] * abs(y)
                  - rw["w_yaw"] * abs(psi)
                  - rw["w_effort"] * float(np.sum(a ** 2)))
        terminated = bool(crashed)
        if crashed:
            reward -= rw["crash_penalty"]
        truncated = bool(self._steps >= self.n_steps)
        info = dict(speed_err=abs(u - self.vt), y=float(y), psi=float(psi),
                    x=float(x), drag_est=0.0, thrust=self.motors.thrust.copy())
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        _x, y, psi, u, v, r = self._state
        return np.concatenate([
            [u / self.vt, v / self.vt, psi,
             r * self.geom.span / (2 * self.vt), y / self.sim["y_limit"]],
            self._throttle,
            self.geom.design_vector,
        ]).astype(np.float32)


def make_env_fn(cfg=None, overrides=None, seed=None):
    def _init():
        return VehicleEnv(cfg=cfg, overrides=overrides, seed=seed)
    return _init

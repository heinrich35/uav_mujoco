"""AirfoilDesignEnv — Gymnasium env for the 2D shape-design loop.

MDP
   action   2*n_coef in [-1, 1]; per-step delta on lower/upper CST coefficients
            (scaled by action_scale). n_coef=5 per surface -> 10-D action.
   obs      [cst_lower/norm, cst_upper/norm, alpha/10, L/D/100, t/0.15]
   reward   (L/D_new - L/D_prev)/reward_scale  -  penalties
            thickness floor (airfoil must stay >= min_thickness),
            non-convergence (solver refused the shape).
   episode  episode_steps refinement steps, starting from NACA0012 + noise.

Every evaluation goes through PolarCache — repeated shapes are free.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.airfoil import cst_airfoil, naca0012_cst, thickness_camber  # noqa: E402
from common.caching import PolarCache, polar_key                        # noqa: E402
from airfoil2d.xfoil_eval import eval_polar                             # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "results" / "polar_cache.db"


class AirfoilDesignEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_coef=5, episode_steps=8, re=5e5, alpha=5.0,
                 action_scale=0.05, min_thickness=0.06, backend="module",
                 seed=None):
        super().__init__()
        self.n_coef = n_coef
        self.episode_steps = episode_steps
        self.re = re
        self.alpha = alpha
        self.action_scale = action_scale
        self.min_thickness = min_thickness
        self.backend = backend
        self.rng = np.random.default_rng(seed)
        self.cache = PolarCache(CACHE)

        c_up, c_lo = naca0012_cst(n_coef)
        self._base = (c_lo, c_up)
        norm = 0.1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(2 * n_coef + 3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0,
                                           shape=(2 * n_coef,), dtype=np.float32)
        self._norm = norm
        self._coefs = None
        self._prev_ld = None
        self._steps = 0

    # -- helpers ---------------------------------------------------------
    def _coords(self, c_lo, c_up):
        return cst_airfoil(c_up, c_lo, n=160)

    def _evaluate(self, c_lo, c_up):
        x, y = self._coords(c_lo, c_up)
        key = polar_key(x, y, self.re, self.alpha, self.backend)
        hit = self.cache.get(key)
        if hit is not None:
            cl, cd, cm, _ = hit
            return dict(cl=cl, cd=cd, cm=cm, converged=True), x, y
        res = eval_polar(x, y, self.re, self.alpha, backend=self.backend)
        if res["converged"]:
            self.cache.put(key, res["cl"], res["cd"], res["cm"], self.backend)
        return res, x, y

    def _obs(self, res):
        c_lo, c_up = self._coefs
        t, _ = thickness_camber(*self._coords(c_lo, c_up))
        ld = res["cl"] / res["cd"] if res["converged"] else 0.0
        return np.concatenate([
            c_lo / self._norm, c_up / self._norm,
            [self.alpha / 10.0, ld / 100.0, t / 0.15],
        ]).astype(np.float32)

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c_lo, c_up = self._base
        noise = self.rng.uniform(-0.02, 0.02, size=(2, self.n_coef))
        self._coefs = (c_lo + noise[0], c_up + noise[1])
        res, _x, _y = self._evaluate(*self._coefs)
        if not res["converged"]:  # nudge until the start shape solves
            self._coefs = self._base
            res, _x, _y = self._evaluate(*self._coefs)
        self._prev_ld = res["cl"] / res["cd"] if res["converged"] else 1.0
        self._steps = 0
        return self._obs(res), {}

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1.0, 1.0).reshape(2, self.n_coef)
        c_lo = self._coefs[0] + self.action_scale * action[0]
        c_up = self._coefs[1] + self.action_scale * action[1]
        res, x, y = self._evaluate(c_lo, c_up)

        reward, converged = 0.0, res["converged"]
        if converged:
            self._coefs = (c_lo, c_up)
            ld = res["cl"] / res["cd"]
            reward += (ld - self._prev_ld) / 20.0
            self._prev_ld = ld
        else:
            reward -= 0.5  # un-solvable shape: keep the old coefficients

        t, _ = thickness_camber(x, y)
        if t < self.min_thickness:
            reward -= 2.0 * (self.min_thickness - t) / self.min_thickness

        self._steps += 1
        done = self._steps >= self.episode_steps
        return self._obs(res if converged else
                         dict(cl=np.nan, cd=1.0, cm=np.nan, converged=False)
                         ), reward, done, False, {}

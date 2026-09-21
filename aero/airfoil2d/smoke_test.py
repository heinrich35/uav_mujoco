"""Smoke tests for the aero stack — run inside aero/env.sh (CPU-only).

    python /home/heinz/isaaclab_uav/aero/airfoil2d/smoke_test.py [--skip-slow]

Each check prints "PASS <name> ..." or "FAIL <name> ..."; exit 1 on any FAIL.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, fn, *a, **k):
    t0 = time.perf_counter()
    try:
        msg = fn(*a, **k) or ""
        print(f"PASS {name} ({(time.perf_counter() - t0) * 1000:.0f} ms) {msg}")
    except Exception as e:
        FAILURES.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]}")


def xfoil_eval():
    from airfoil2d.xfoil_eval import eval_polar
    from common.airfoil import naca4_coords

    x, y = naca4_coords()
    r = eval_polar(x, y, 5e5, 5.0)
    assert r["converged"] and 0.4 < r["cl"] < 0.7, r
    return f"cl={r['cl']:.4f} cd={r['cd']:.5f} L/D={r['cl'] / r['cd']:.1f}"


def cst_roundtrip():
    from common.airfoil import cst_airfoil, fit_cst, naca4_coords
    x, y = naca4_coords()
    n = (len(x) + 1) // 2
    cu = fit_cst(x[:n], y[:n], 8)
    cl_ = fit_cst(x[n - 1:][::-1], -y[n - 1:][::-1], 8)
    xr, yr = cst_airfoil(cu, cl_, n=160)
    err = np.abs(yr - y).max()
    assert err < 5e-3, f"CST roundtrip max |dy| = {err:.2e}"
    return f"CST-8 roundtrip max |dy| = {err:.1e}"


def env_rollout():
    from airfoil2d.env_xfoil import AirfoilDesignEnv

    env = AirfoilDesignEnv(seed=0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    total = 0.0
    for _ in range(4):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r), f"non-finite reward {r}"
        total += float(r)
        assert not (term or trunc)
    return f"4 random steps, return={total:.3f}, cache entries={len(env.cache)}"


def sb3_short(n=300):
    os_guard()  # noqa: F821 — defined below
    import torch
    from stable_baselines3 import SAC

    import airfoil2d.env_xfoil as m
    env = m.AirfoilDesignEnv(seed=1)
    model = SAC("MlpPolicy", env, seed=1, device="cpu", verbose=0,
                buffer_size=5000, train_freq=1)
    model.learn(total_timesteps=n)
    obs, _ = env.reset()
    a, _ = model.predict(obs, deterministic=True)
    assert np.all(np.isfinite(a))
    return f"SAC {n} steps on CPU, finite actions"


def os_guard():
    import os
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", (
        "CUDA_VISIBLE_DEVICES not empty — source aero/env.sh first")
    import torch
    assert not torch.cuda.is_available()
    assert torch.cuda.device_count() == 0


def pygem_ffd():
    from pygem import FFD

    # PyGeM 2.x: displacements live in array_mu_{x,y,z} as FRACTIONS of
    # box_length; control_points() is a read-only convenience. Points outside
    # the box don't deform, and the Bernstein weights vanish on box faces —
    # so probe strictly interior points.
    ffd = FFD(n_control_points=[2, 2, 2])
    ffd.array_mu_x[1, 1, 1] = 0.3
    ffd.array_mu_y[1, 1, 1] = 0.2
    ffd.array_mu_z[1, 1, 1] = -0.1
    pts = np.random.default_rng(0).uniform(0.05, 0.95, size=(200, 3))
    new = np.asarray(ffd(pts))
    assert new.shape == pts.shape
    assert np.abs(new - pts).max() > 1e-3, "FFD produced an identity deformation"
    return f"lattice 2^3, 200 interior pts, max|d| = {np.abs(new - pts).max():.3f}"


def gmsh_cmsh():
    from common.airfoil import naca4_coords
    from tools.mk_cmesh import build_cmsh, write_su2

    out = ROOT / "results" / "cmsh"
    out.mkdir(parents=True, exist_ok=True)
    x, y = naca4_coords()
    build_cmsh(x, y, out / "cmesh.msh")
    write_su2(out / "cmesh.msh", out / "cmesh.su2")
    txt = (out / "cmesh.su2").read_text()
    assert "NDIME= 2" in txt and "MARKER_TAG= airfoil" in txt
    n_el = int(next(ln.split()[-1] for ln in txt.splitlines() if ln.startswith("NELEM")))
    return f"{n_el} triangles, markers airfoil+farfield"


def su2_euler(np_mpi=0):
    import subprocess

    mesh = ROOT / "results" / "cmsh" / "cmesh.su2"
    if not mesh.exists():
        gmsh_cmsh()
    run = ROOT / "results" / "su2_smoke"
    run.mkdir(parents=True, exist_ok=True)
    cfg = run / "euler.cfg"
    cfg.write_text(
        f"MESH_FILENAME= {mesh}\n"
        "SOLVER= EULER\n"
        "MACH_NUMBER= 0.5\n"
        "AOA= 5.0\n"
        "MARKER_FAR= ( farfield )\n"
        "MARKER_SYM= ( airfoil )\n"          # Euler wall = slip/symmetry BC in v8
        "MARKER_MONITORING= ( airfoil )\n"   # forces/coefficients evaluated here
        "CONV_NUM_METHOD_FLOW= ROE\n"
        "MUSCL_FLOW= NO\n"
        "SLOPE_LIMITER_FLOW= NONE\n"
        "TIME_DISCRE_FLOW= EULER_IMPLICIT\n"
        "CFL_NUMBER= 5.0\n"
        "ITER= 30\n"
        "CONV_FIELD= RMS_DENSITY\n"
        "CONV_RESIDUAL_MINVAL= -6\n"
        "HISTORY_OUTPUT= (ITER, RMS_CL, RMS_CD)\n"
        "SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, RMS_LIFT, RMS_DRAG)\n"
    )
    su2_bin = str(Path.home() / "SU2-v8.5.0" / "bin" / "SU2_CFD")
    cmd = ([ "mpirun", "--bind-to", "none", "-np", str(np_mpi), su2_bin, str(cfg)]
           if np_mpi else [su2_bin, str(cfg)])
    env = {**os.environ, "OMPI_MCA_osc": "pt2pt"}
    r = subprocess.run(cmd, cwd=run, capture_output=True, text=True, timeout=600, env=env)
    log = (run / f"euler_np{np_mpi}.log")
    log.write_text(r.stdout + r.stderr)
    iters = [ln for ln in r.stdout.splitlines() if "RMS_DENSITY" in ln or "|  " in ln]
    assert r.returncode == 0, f"SU2_CFD rc={r.returncode}, see {log}"
    assert len(iters) >= 10, f"only {len(iters)} iteration lines, see {log}"
    return f"{len(iters)} iterations logged -> {log.name}"


def neuralfoil_xcheck():
    from airfoil2d.xfoil_eval import eval_polar
    from common.airfoil import naca4_coords

    x, y = naca4_coords()
    r = eval_polar(x, y, 5e5, 5.0, backend="neuralfoil")
    assert r["converged"], "neuralfoil eval failed (installed --no-deps?)"
    return f"cl={r['cl']:.4f} (XFoil reference 0.627)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args()

    check("torch-cpu-guard", os_guard)
    check("cst-roundtrip", cst_roundtrip)
    check("xfoil-eval", xfoil_eval)
    check("env-rollout", env_rollout)
    check("pygem-ffd", pygem_ffd)
    check("gmsh-cmesh", gmsh_cmsh)
    if not args.skip_slow:
        check("su2-euler-serial", su2_euler, np_mpi=0)
        check("su2-euler-mpi4", su2_euler, np_mpi=4)
        check("neuralfoil-xcheck", neuralfoil_xcheck)
        check("sb3-sac-300", sb3_short, 300)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()

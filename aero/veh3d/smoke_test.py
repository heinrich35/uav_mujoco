"""Smoke tests for veh3d — run inside aero/env.sh (CPU-only).

    python /home/heinz/isaaclab_uav/aero/veh3d/smoke_test.py

Each check prints "PASS <name> ..." or "FAIL <name> ..."; exit 1 on any FAIL.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FAILURES = []


def check(name, fn, *a, **k):
    t0 = time.perf_counter()
    try:
        msg = fn(*a, **k) or ""
        print(f"PASS {name} ({(time.perf_counter() - t0) * 1000:.0f} ms) {msg}")
    except Exception as e:
        FAILURES.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:200]}")


def os_guard():
    import os
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", (
        "CUDA_VISIBLE_DEVICES not empty — source aero/env.sh first")
    import torch
    assert not torch.cuda.is_available()
    assert torch.cuda.device_count() == 0


def geometry_build():
    from geometry import build, feasibility, load_design

    geom = build(load_design())
    m = geom.mesh
    assert m.is_watertight, "default loft is not watertight"
    assert geom.mass > 0.05 and np.isfinite(geom.izz) and geom.izz > 1e-4
    assert geom.S_ref > 0.05 and geom.A_front > 1e-4
    assert geom.polar_ok and geom.cl_c > 0.1, f"polar cl={geom.cl_c}"
    ok, reasons = feasibility(geom)
    assert ok, f"default design must be feasible, got: {reasons}"
    a = geom.aero
    return (f"watertight, {len(m.faces)} faces | mass {geom.mass:.3f} kg "
            f"Izz {geom.izz:.4f} | S {geom.S_ref:.3f} A_f {geom.A_front:.4f} "
            f"CD0 {geom.CD0:.3f} | L {a['lift']:.2f} W {a['weight']:.2f} "
            f"D {a['drag']:.2f} N")


def geometry_stl_roundtrip():
    import trimesh

    from geometry import build, export_stl, load_design

    geom = build(load_design())
    p = ROOT / "results" / "veh3d_smoke" / "roundtrip.stl"
    export_stl(geom, p)
    m2 = trimesh.load(str(p))
    err = abs(m2.volume - geom.mesh.volume) / geom.mesh.volume
    assert err < 1e-4, f"STL roundtrip volume drift {err:.2e}"
    return f"volume drift {err:.1e} -> {p.name}"


def motors_torque():
    from motors import MotorBank

    bank = MotorBank.__new__(MotorBank)   # bypass cfg; hand-set the math case
    bank.y = np.array([-0.4, 0.0, 0.4])
    bank.t_max, bank.tau, bank.dt = 1.0, 0.08, 0.02
    bank.thrust = np.array([1.0, 0.5, 0.0])
    fx, mz = bank.fx_mz()
    assert abs(fx - 1.5) < 1e-12
    assert abs(mz - 0.4) < 1e-12, f"Mz {mz} != -sum(y*T) = 0.4"
    # spool lag: one step toward full command at dt/tau = 0.25
    bank2 = MotorBank.__new__(MotorBank)
    bank2.y, bank2.t_max, bank2.tau, bank2.dt = bank.y, 1.0, 0.08, 0.02
    bank2.thrust = np.zeros(3)
    bank2.command(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(bank2.thrust, 0.25), bank2.thrust
    return "fx=1.5 N, Mz=+0.4 N·m (=-ΣyᵢTᵢ); spool step 0->0.25 @ dt/tau=0.25"


def env_rollout():
    from env_veh import VehicleEnv
    from geometry import load_design

    env = VehicleEnv(cfg=load_design(), seed=0)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    # full throttle: must accelerate toward (and past) cruise trim
    u0 = env._state[3]
    ret = 0.0
    for _ in range(300):
        obs, r, term, trunc, info = env.step([1.0, 1.0, 1.0])
        ret += r
        assert np.isfinite(r) and np.all(np.isfinite(obs))
        if term or trunc:
            break
    u1 = env._state[3]
    assert u1 > u0 + 1.0, f"full throttle did not accelerate: {u0:.2f}->{u1:.2f}"
    # zero throttle: drag must decay the speed
    env.reset()
    for _ in range(300):
        obs, r, term, trunc, info = env.step([0.0, 0.0, 0.0])
        if term or trunc:
            break
    u2 = env._state[3]
    assert u2 < 2.0, f"zero throttle did not decay: {u2:.2f}"
    return (f"throttle-up {u0:.2f}->{u1:.2f} m/s, coast-down ->{u2:.2f} m/s, "
            f"return={ret:.1f}")


def sb3_ppo_short(n=2000):
    from stable_baselines3 import PPO

    from env_veh import VehicleEnv
    from geometry import load_design

    env = VehicleEnv(cfg=load_design(), seed=1)
    model = PPO("MlpPolicy", env, seed=1, device="cpu", verbose=0,
                n_steps=200, batch_size=200)
    model.learn(total_timesteps=n)
    obs, _ = env.reset()
    a, _ = model.predict(obs, deterministic=True)
    assert np.all(np.isfinite(a)) and np.all((a >= 0) & (a <= 1))
    return f"PPO {n} steps on CPU, finite throttles {np.round(a, 2)}"


def feasibility_gate():
    from geometry import build, feasibility, load_design, overrides_to_cfg

    # absurd: tiny span but default motor gap -> the motor gate must fire
    cfg = overrides_to_cfg(load_design(), {"geometry.span": 0.4})
    geom = build(cfg)
    ok, reasons = feasibility(geom)
    assert not ok and any("motor gate" in r for r in reasons), (ok, reasons)
    return f"span=0.4 rejected: {reasons[0]}"


def design_loop_quick(n_designs=2, steps=2000):
    import shutil

    from geometry import load_design
    from train_design import read_rows, run_design_loop

    out = ROOT / "results" / "veh3d_smoke" / "loop"
    if out.exists():
        shutil.rmtree(out)          # the loop is resumable; smoke wants clean
    cfg = load_design()
    out = ROOT / "results" / "veh3d_smoke" / "loop"
    best, csv_path = run_design_loop(
        cfg, out, n_designs=n_designs, steps_per_design=steps, n_envs=2,
        seed=7, eval_episodes=2, tag="smoke")
    rows = read_rows(csv_path)
    assert len(rows) == n_designs, f"expected {n_designs} rows, got {len(rows)}"
    n_feas = sum(int(r["feasible"]) for r in rows)
    if n_feas:
        assert (out / "best_design.json").exists()
        assert (out / "best.stl").exists()
    return (f"{len(rows)} ledger rows, {n_feas} feasible trained"
            + (f", best idx {best['idx']} score {best['score']}" if best else ""))


def main():
    check("torch-cpu-guard", os_guard)
    check("geometry-build", geometry_build)
    check("geometry-stl-roundtrip", geometry_stl_roundtrip)
    check("motors-torque", motors_torque)
    check("feasibility-gate", feasibility_gate)
    check("env-rollout", env_rollout)
    check("sb3-ppo-short", sb3_ppo_short)
    check("design-loop-quick", design_loop_quick)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()

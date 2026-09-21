"""SLOW loop: search the DESIGN space — each candidate earns a brief control run.

One slow-loop iteration (the "design generation"):

    sample theta ~ Uniform(ranges from design.yaml)
      -> build geometry (mesh, mass, areas, NeuralFoil polar)
      -> feasibility gate (lift / thrust / motor-in-span / watertight)
           infeasible: ledger row with the reason, NO training
      -> brief PPO control training (fast loop, pattern A: design in obs)
      -> score = deterministic tracking eval
      -> append row to results/veh3d/<tag>/design_loop.csv

Resumable: rows are keyed by idx; existing rows are skipped on re-run.
Ends with best_design.json + best STL + the winner's render.

Usage (source aero/env.sh first):
    python .../veh3d/train_design.py                     # 6 designs, 60k steps each
    python .../veh3d/train_design.py --n-designs 3 --steps-per-design 30000
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

THETA_KEYS = [
    "geometry.span", "geometry.taper", "geometry.sweep_deg",
    "geometry.thickness_root", "geometry.camber", "motors.motor_gap",
    "aero.cruise_alpha_deg",
]

CSV_FIELDS = (["idx", "feasible", "reason", "score", "speed_err_rms",
               "lateral_rms", "yaw_rms_deg", "progress_mean", "train_s"]
              + [k.split(".")[-1] for k in THETA_KEYS]   # short names in the csv
              + ["mass", "S_ref", "A_front", "AR", "CD0", "CL_c", "lift_N",
                 "drag_N"])


def sample_theta(rng, ranges):
    return {k: float(rng.uniform(*ranges[k])) for k in THETA_KEYS}


def theta_row(theta):
    return {k.split(".")[-1]: round(v, 4) for k, v in theta.items()}


def geom_row(geom):
    a = geom.aero
    return dict(mass=round(geom.mass, 4), S_ref=round(geom.S_ref, 4),
                A_front=round(geom.A_front, 5), AR=round(geom.AR, 2),
                CD0=round(geom.CD0, 4), CL_c=round(float(geom.cl_c), 3),
                lift_N=round(a["lift"], 2), drag_N=round(a["drag"], 2))


def run_design_loop(cfg, out_dir, n_designs, steps_per_design, n_envs, seed,
                    eval_episodes, verbose=0, tag="loop"):
    """Returns (best_row, csv_path). Resumable by idx."""
    from geometry import build, export_stl, overrides_to_cfg, render_png
    from train_control import train_once

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "design_loop.csv"
    rows = read_rows(csv_path)
    rng = np.random.default_rng(seed)
    ranges = cfg["slow_loop"]["ranges"]

    best, best_row = None, None
    for idx in range(n_designs):
        if any(str(r["idx"]) == str(idx) for r in rows):  # csv reads back as str
            print(f"[{tag}] design {idx}: already in ledger, skip")
            continue
        theta = sample_theta(rng, ranges)
        t0 = time.perf_counter()
        cfg_o = overrides_to_cfg(cfg, theta)
        geom = build(cfg_o)
        feasible, reasons = check_feasible(geom)

        row = dict(idx=idx, feasible=int(feasible),
                   reason="; ".join(reasons) if reasons else "",
                   score="", speed_err_rms="", lateral_rms="", yaw_rms_deg="",
                   progress_mean="", train_s="", **theta_row(theta),
                   **geom_row(geom))
        if not feasible:
            print(f"[{tag}] design {idx}: INFEASIBLE — {reasons[0]}")
        else:
            d_out = out_dir / f"design_{idx:02d}"
            try:
                _model, metrics, _d, _g = train_once(
                    cfg, theta, steps_per_design, n_envs, seed + 100 + idx,
                    d_out, verbose=verbose, eval_episodes=eval_episodes)
            except Exception as e:   # a design may train badly; keep the loop alive
                row["reason"] = f"train failed: {type(e).__name__}: {str(e)[:80]}"
                print(f"[{tag}] design {idx}: {row['reason']}")
            else:
                row.update(score=round(metrics["return_mean"], 2),
                           speed_err_rms=round(metrics["speed_err_rms"], 4),
                           lateral_rms=round(metrics["lateral_rms"], 4),
                           yaw_rms_deg=round(metrics["yaw_rms_deg"], 2),
                           progress_mean=round(metrics["progress_mean"], 1),
                           train_s=metrics["train_seconds"])
                if best is None or metrics["return_mean"] > best:
                    best = metrics["return_mean"]
                    best_row = row
                    export_stl(geom, out_dir / "best.stl")
                    render_png(geom, out_dir / "best.png")
                    (out_dir / "best_design.json").write_text(json.dumps(
                        dict(idx=idx, theta=theta, metrics=metrics,
                             geometry=geom_row(geom),
                             cfg_geometry=cfg_o["geometry"],
                             cfg_motors=cfg_o["motors"]), indent=2, default=str))
                print(f"[{tag}] design {idx}: score {metrics['return_mean']:.1f} "
                      f"speed_rms {metrics['speed_err_rms']:.3f} "
                      f"lat {metrics['lateral_rms']:.3f} "
                      f"({time.perf_counter() - t0:.0f}s)")
        rows.append(row)
        write_rows(csv_path, rows)

    print(f"[{tag}] ledger -> {csv_path}"
          + (f" | best idx {best_row['idx']} score {best_row['score']}"
             if best_row else " | NO feasible design trained"))
    return best_row, csv_path


# -- ledger ------------------------------------------------------------------
def read_rows(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def write_rows(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


# thin indirection so the smoke can monkeypatch feasibility cheaply
def check_feasible(geom):
    from geometry import feasibility
    return feasibility(geom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", default=str(Path(__file__).parent / "design.yaml"))
    p.add_argument("--n-designs", type=int, default=None)
    p.add_argument("--steps-per-design", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tag", default=time.strftime("%H%M%S"))
    args = p.parse_args()

    import torch
    assert not torch.cuda.is_available(), (
        "GPU visible — source aero/env.sh (CUDA_VISIBLE_DEVICES='') first")

    from geometry import load_design

    cfg = load_design(args.design)
    sl = cfg["slow_loop"]
    run_design_loop(
        cfg, ROOT / "results" / "veh3d" / f"loop_{args.tag}",
        n_designs=args.n_designs or sl["n_designs"],
        steps_per_design=args.steps_per_design or sl["steps_per_design"],
        n_envs=args.n_envs or sl["n_envs"],
        seed=args.seed if args.seed is not None else sl["seed"],
        eval_episodes=sl["eval_episodes"], tag=args.tag)


if __name__ == "__main__":
    main()

# aero/ — aerodynamic shape-design + RL loop (CPU-only)

Pipeline stage by stage: **shape parameters → mesh → CFD → reward → policy**.
Everything here is CPU-only by construction: `env.sh` exports
`CUDA_VISIBLE_DEVICES=""` before torch is ever imported (the GPU belongs to the
IsaacLab training run — never load it).

## Activate

```bash
cd /home/heinz/isaaclab_uav/aero && . env.sh     # venv + SU2 + CUDA guard
. of14-env.sh                                    # only when using OpenFOAM (clobbers LD_LIBRARY_PATH)
```

## Layout

| Path | Role |
|---|---|
| `common/airfoil.py` | NACA4 coords, cosine spacing, CST parameterization (design vector ↔ surface), CST least-squares fit |
| `common/caching.py` | sqlite polar cache keyed on (coord-bytes, Re, α, backend) — RL never re-pays for a CFD eval |
| `airfoil2d/xfoil_eval.py` | unified `eval_polar()`: backends `module` (xfoil-python), `binary` (apt XFoil), `neuralfoil` |
| `airfoil2d/env_xfoil.py` | `AirfoilDesignEnv` — Gymnasium env: action = CST deltas, obs = coeffs+α+L/D+thickness, reward = ΔL/D − penalties |
| `airfoil2d/train_sb3.py` | SAC/TD3 + SubprocVecEnv fan-out, CPU, saves to `results/sb3_smoke/` |
| `airfoil2d/smoke_test.py` | one-shot verification of every installed piece (see below) |
| `tools/mk_cmesh.py` | gmsh C-mesh → meshio → hand-written `.su2` (meshio's su2 writer drops marker names) |
| `tests/test_smoke.py` | pytest wrappers that skip when a dep is absent |
| `duct3d/` | placeholder — PyGeM FFD + gmsh/snappy + SU2 runner next pass |
| `results/` | polar cache db, meshes, SU2 logs, SB3 checkpoints |
| `guard/` | before-snapshots of `env_isaaclab` proving the live env was untouched |

## Verify

```bash
python airfoil2d/smoke_test.py              # full: xfoil, env, PyGeM, gmsh→su2, SU2 serial+MPI4, neuralfoil, SAC
python airfoil2d/smoke_test.py --skip-slow  # fast subset
pytest tests/ -q
```

Reference numbers the stack must reproduce (NACA0012, Re 5e5, AoA 5°):
`cl ≈ 0.627`, `cd ≈ 0.0104`, `L/D ≈ 60`. XFoil-module backend gives
cl=0.6271/cd=0.01040 in well under a second — if yours differ wildly, suspect
the coordinate generator (must be `xc = 0.5(1−cos πt)`) or a repanel call on
already-fine cosine coords (de-converges).

## Install scripts (re-runnable)

- `setup_env_aero.sh` — builds `~/env_aero` (torch-cpu, gymnasium, SB3, gmsh,
  meshio, pyvista, xfoil-python **from git with the gfortran-flag patch**, PyGeM
  from mathLab git — PyPI `pygem` is a glacier model, never install it —,
  neuralfoil `--no-deps`).
- `build_su2.sh` — SU2 v8.5.0 from source with MPI (release binaries segfault
  on Ubuntu 24.04); installs to `~/SU2-v8.5.0`.
- `apt_layer.sh` — **sudo, run by hand**: `xfoil gmsh zlib1g-dev`, OpenFOAM 14
  from dl.openfoam.org, paraview+xvfb. OpenFOAM smoke is pending this step.

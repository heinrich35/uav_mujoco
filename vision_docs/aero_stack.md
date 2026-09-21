# Aero/fluid shape-design + RL stack — install record (2026-09-14/15)

Target project: `/home/heinz/isaaclab_uav` (aero/ skeleton + scripts in `aero/`).
Hard rule for this install: **CPU only** — the GPU is occupied by the IsaacLab
training env. `CUDA_VISIBLE_DEVICES=""` is exported in `aero/env.sh` *before*
the first `import torch` (empty string, not `-1`), and `train_sb3.py` /
`smoke_test.py` assert `torch.cuda.device_count() == 0`.

## Where everything landed

| Piece | Where | Source / version | Notes |
|---|---|---|---|
| Python env | `/home/heinz/env_aero` | uv venv, python3.12 | NEVER touch `/home/heinz/env_isaaclab` (live training env; guarded by before/after freeze+tree diffs in `aero/guard/`) |
| torch | env_aero | 2.14.0+cpu (pytorch.org/whl/cpu) | |
| RL | env_aero | gymnasium 1.3.0, stable-baselines3 2.9.0 | SAC/TD3 + SubprocVecEnv (4 workers × `OMP_NUM_THREADS=1`) |
| XFoil binding | env_aero | xfoil-python 1.1.1 **from git, locally patched** | PyPI sdist is broken (12 kB, no CMakeLists.txt) |
| XFoil binary | `/usr/bin/xfoil` | apt | backup backend (`backend="binary"`); apt step run by user |
| NeuralFoil | env_aero | `--no-deps` | its aerosandbox dep is heavy+circular; core is numpy-only |
| PyGeM | env_aero | git+mathLab/PyGeM | PyPI `pygem` = **glacier** model — never install that name |
| mesh/viz | env_aero | gmsh 4.15.2, meshio 5.3.5, pyvista 0.49, vtk 9.7 | pip gmsh > apt 4.12 |
| SU2 | `/home/heinz/SU2-v8.5.0/bin` | v8.5.0 "Harrier", source build, MPI enabled | release tarball binaries segfault on Ubuntu 24.04 (su2code #2682) and are serial-only |
| SU2 src | `/home/heinz/src/SU2-8.5.0` | git v8.5.0 | built via meson: `with-mpi=enabled with-omp=false enable-{autodiff,directdiff,pywrapper}=false` |
| OpenFOAM | `/opt/openfoam14` (pending) | apt, dl.openfoam.org noble | **user runs `aero/apt_layer.sh` by hand** (sudo password); openfoam13 fallback |
| ParaView | apt (pending) | same sudo script | separate transaction (~1 GB Qt) |
| trimesh | env_aero | 5.1.0 (+ pyyaml 6.0.3) | veh3d loft/mass/STL; 5.x `mass_properties` is a **dataclass** (`.mass/.inertia/.center_mass`), not a dict |
| mujoco | env_aero | 3.13.0 | veh3d/flight_sim.py interactive viewer (GLFW, desktop GL — no CUDA); EGL offscreen for headless previews |

Activation: `cd /home/heinz/isaaclab_uav/aero && . env.sh`
(SU2_HOME/SU2_RUN/PATH/PYTHONPATH + CUDA guard + thread caps).
OpenFOAM: `. of14-env.sh` separately — its bashrc is non-idempotent and
clobbers `LD_LIBRARY_PATH`/`PS1`, so it must not live in `env.sh`.

## Pitfalls hit and fixed during install (do not re-learn these)

1. **xfoil-python gfortran flags**: upstream `CMakeLists.txt` sets
   `-fbounds-check -ffpe-trap=invalid,zero`. XFoil's 1980s core writes the
   Kutta row at `AIJ(N+1, …)` (`m_xpanel.f90:1150`) and takes benign FP
   detours → every single eval aborted. Fix: delete both flags from
   `CMakeLists.txt` before building (`setup_env_aero.sh` does the sed).
2. **Coordinate generator**: NACA thickness distribution needs
   `xc = 0.5*(1−cos(πt))` — missing 0.5 factor silently spans [0,2] and yields
   garbage polars. Probe coordinates through the library getter, not print.
3. **No repanel on fine cosine coords**: 319-pt cosine spacing + `repanel(180)`
   de-converges; use the coords as generated.
4. **xfoil-python usage**: assign `xf.airfoil = Airfoil(x, y)`, set
   `xf.print = False`; a false convergence flag returns `(nan,nan,nan,nan)`.
5. **Default gcc is 11.5** (update-alternatives), not 13.2 — SU2 must be built
   with `CC=gcc-13 CXX=g++-13 OMPI_CC=gcc-13 OMPI_CXX=g++-13`. Never run
   update-alternatives (live env builds torch extensions against it).
6. **meshio `.su2` writer drops marker names** → `tools/mk_cmesh.py` writes the
   minimal `.su2` text itself (NDIME/NPOIN/NELEM/NMARK/MARKER_TAG).
7. **uv venvs ship no pip** → `uv venv --seed`; and always
   `uv pip install --python /home/heinz/env_aero/bin/python` because PATH
   starts with `env_isaaclab/bin`.
8. **numpy downgrade risk** = vtk (via pyvista/PyGeM) → constraints.txt pins
   `numpy>=2.3,<3` on every install.
9. **OpenFOAM 14 has no `simpleFoam`** — solvers run as
   `foamRun -solver incompressibleFluid`.
10. **CST fit**: never divide by the class function (y/C is 0/0 at both ends
    and blows up like 1/(1−ψ) near an open TE — lstsq then chases the TE spike
    and returns fitted thickness 0.008 instead of 0.053). Fit y DIRECTLY
    against columns C·B_k; fitted surface closes at the TE by construction.
11. **gmsh write drops ungrouped elements**: as soon as ANY physical group
    exists, `gmsh.write` writes only grouped entities — group the surface
    (dim 2) too, or the .msh silently contains just boundary lines.
12. **`.su2` element codes are VTK ids**, not gmsh ids: line=3, triangle=5.
    gmsh's 2/3 make SU2 abort with "Element type not supported".
13. **SU2 v8.5 cfg**: `MARKER_MONITORING` is NOT a boundary condition — every
    mesh marker needs one; the Euler wall is `MARKER_SYM= ( airfoil )`.
    `OUTPUT=` is gone (use `HISTORY_OUTPUT=`), `MESH_FILENAME` defaults to
    `mesh.su2`, and a minimal case also needs `CONV_NUM_METHOD_FLOW`/
    `TIME_DISCRE_FLOW`/`CFL_NUMBER` or preprocessing aborts.
14. **PyGeM 2.x**: deformations live in `array_mu_x/y/z` as FRACTIONS of
    `box_length`; `control_points()` is a read-only convenience returning a
    copy. Only points INSIDE the FFD box deform, and Bernstein weights vanish
    on box faces — probe with strictly interior points, not an integer grid.
15. **NeuralFoil needs aerosandbox** at import time (`--no-deps` install is
    not enough). `get_aero_from_coordinates` takes a stacked (N,2) array +
    `alpha=`/`Re=` kwargs, and returns 0-d arrays — use `.item()`, never
    `float()` directly (numpy 2 raises "only 0-dimensional arrays...").
16. **veh3d loft, watertightness**: a section ring must be a strictly simple
    polygon — duplicate LE/TE seam points become zero-length edges whose
    merged vertices turn strip triangles degenerate and punch holes. Share
    vertex INDICES between strips/caps (never re-append ring per strip), and
    call `trimesh.repair.fix_normals()`: strip/cap windings disagree locally,
    which silently yields a NEGATIVE inertia tensor. A mirror-symmetric loft
    still gets com_y ≈ 1e-4 m — diagonal-split of non-planar quads is
    inherently mirror-asymmetric; irrelevant, don't chase it.
17. **veh3d reward shaping (both bit within one hour)**: a crash penalty
    smaller than one episode's accumulated cost trains "crash ASAP" (eval
    return −79 looked GREAT — it was 20 steps of penalties + terminal); fix =
    per-step alive bonus + crash penalty ~10× episode cost. And penalize
    lateral VELOCITY (the drift driver), not just position — position-only
    plateaued at 1.7 m RMS with no gradient out. Also: a weathervane
    (beta-proportional) yaw moment with NO Cl_r rate damping is an undamped
    oscillator — the plant, not the policy, was untrainable.
18. **CSV ledgers round-trip as strings** — `int(row["idx"]) == idx` never
    true when compared raw; resume/skip logic must str()-both sides.
19. **trimesh 5.x mass_properties is a dataclass** (see table row); density
    DOES scale inertia (verify against the analytic box: Izz=m(a²+b²)/12).
    WORSE: the cache survives `fix_normals`/`invert()` — it returned a
    centroid OUTSIDE the mesh bounding box while volume stayed correct.
    veh3d now uses its own signed-tetrahedron integral (`geometry.mass_properties`),
    cross-checked against MuJoCo's independent mesh integral (exact agree).
20. **MuJoCo `<inertial fullinertia>`** reorders to principal axes
    (`body_inertia` ≠ your (Ixx,Iyy,Izz), plus `body_iquat`) — external
    moment bookkeeping gets treacherous. Compose body inertia from geoms
    instead (mesh geom with `mass=` + payload sphere), and NAME the freejoint
    (`model.joint("vehicle")` looks up a JOINT, unnamed freejoints match
    nothing).
21. **Moment sign conventions (wrote them down or flip them hourly)**: body
    x fwd / y right / z up ⇒ M_x>0 rolls LEFT, M_y>0 pitches DOWN, M_z>0
    yaws RIGHT. Restoring stiffness: α>α_trim needs +M_y (nose down); the
    minus-sign "fix" inverts it into a divergent dive. Weathervane: β>0
    (wind from right) → nose toward the wind = +M_z (the planar env had this
    inverted for its first day — fixing it HALVED lateral RMS, 5/5 episodes).
22. **Explicit rate-damping through xfrc is only conditionally stable**:
    coeff·dt/I must stay < 1 — with Iyy = 0.0012 kg·m² the pitch axis blew up
    in 7 steps at cm_q=10, dt=4 ms (2-step-period oscillation doubling every
    step). Set cm_q=1.2 + dt=2 ms. If rate damping grows legs, check this
    first.

## Reference verification (XFoil module backend, CPU)

NACA0012, Re 5e5, AoA 5°: `cl=0.6271`, `cd=0.01040`, `L/D=60.3` — matches the
standard published polar (0.627 / 0.0104 / ~60) in well under a second per
eval. Same coords through NeuralFoil agree to ~1e-2 in cl.

## Status / TODO

- [x] env_aero, SU2 build, skeleton, **all 9 smoke checks green** (2026-09-15:
      torch-guard, CST roundtrip, XFoil eval cl=0.6271/cd=0.01040, RL env
      rollout + sqlite cache, PyGeM FFD, gmsh→.su2 95 627 tris, SU2 Euler
      serial + MPI-4, NeuralFoil cl=0.6217 (−0.9% vs XFoil), SAC-300 CPU)
- [x] guard: `env_isaaclab` freeze + tree diffs vs before-snapshots = empty
- [ ] user runs `sudo bash aero/apt_layer.sh` (xfoil/gmsh apt dupes, OpenFOAM 14, paraview)
- [ ] OpenFOAM smoke: `foamRun -solver incompressibleFluid` cavity + `decomposePar`/`mpirun -np 4` + `snappyHexMesh -help`
- [x] **veh3d/ (2026-09-18): parametric aerial vehicle, both loops wired** —
      design.yaml (params + search ranges) → section loft → trimesh (2900
      faces, watertight, fix_normals'd) → mass/areas/NeuralFoil polar →
      feasibility gates (lift/thrust/motor/watertight; all three observed
      firing on random samples) → Gymnasium planar flight env (3 y-axis
      motors at the back, +x thrust, differential = yaw, spool lag) → SB3 PPO
      600k: speed err 0.50 m/s RMS, yaw 1.2°, 62 m/8 s; slow loop resumable
      CSV ledger + best_design.json/stl/png; 8/8 smoke checks green. Winner
      polish handoff demonstrated (loop_1002 → winner_1002_polished, 4/5
      full episodes). README in veh3d/.
- [x] **veh3d/flight_sim.py (2026-09-18): interactive 6-DOF flight sim** —
      same parametric model → MJCF (ground plane, skybox, shadowed lights,
      tracking camera, motor sites) via MuJoCo 3.13, desktop-GL rendering
      (no CUDA). Gravity + lift/drag/sideforce + soft stall + weathervane/
      rate damping + steady wind + OU turbulence; keys W/S/A/D/Space/R.
      Selftest: steady 7.9 m/s at design trim α=3.0°, 95 m/12 s calm;
      crosswind+turbulence drift verified. Weathervane sign fix backported
      to the planar env → lateral RMS 0.34→0.13 m, 5/5 full episodes.
- [ ] duct3d/: PyGeM FFD over the shahed-136/duct STL, gmsh 3D or snappy mesh, SU2/OpenFOAM runner
- [ ] veh3d next rungs: SU2 confirmation runs on gate-passing winners; CST
      sections (common/airfoil.py); CMA-ES outer optimizer replacing uniform
      sampling; Isaac Lab 3D port once the GPU frees up
- [ ] later: feed `results/` polars into `scripts/mujoco/uav_sim_1.py` k_drag/k_lift

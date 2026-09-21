# Reinforcement learning for physical design — the veh3d guide

How this project trains a machine to design an aircraft: a FAST loop where RL
learns motor/surface CONTROL for a fixed airframe, and a SLOW loop where the
AIRFRAME ITSELF (shape + surface parameters) is optimized against how well the
fast loop can fly it. Everything below reflects what is actually built and
verified in `aero/veh3d/` (see also `flight_sim_play.md` for the simulator).

Launch rule for every command in this file:

```bash
cd /home/heinz/isaaclab_uav/aero && source env.sh    # CPU-only; GPU belongs to IsaacLab
```

Wrong python = `FATAL: section polar failed` (env_isaaclab has mujoco but no
neuralfoil). The first line of every run prints the interpreter — it must say
`.../env_aero/bin/python`.

---

## 1. The design pipeline

```
design.yaml                       THE design source of truth (params + ranges)
    |
    v  geometry.py
airfoil SECTIONS (2D curves: NACA thickness + camber)
    lofted along spanwise 3D CURVES (taper, sweep, twist, dihedral)
    -> watertight triangle mesh (trimesh)
    -> mass, inertia, com          (own signed-tetrahedron integral;
    -> projected areas               mesh-cache bugs are why we don't use
    -> section polar                 trimesh's mass_properties here)
    |                              (NeuralFoil CL/CD at the design's trim)
    v  feasibility gates           hard, pre-flight:
      lift    >= 1.2 x weight at cruise      (wing too small -> reject)
      thrust  >= 1.15 x cruise drag          (too draggy    -> reject)
      motors inside half-span, mesh watertight
    |
    +--------------------> FAST loop (control)   env_veh.py + train_control.py
    |                       Gymnasium planar flight env, SB3 PPO.
    |                       Trains motor/surface CONTROL for THIS airframe.
    |                       Design descriptor rides in the observation, so one
    |                       policy amortizes across designs.
    v
SLOW loop (shape)   train_design.py
    sample theta ~ Uniform(slow_loop.ranges)
      -> build geometry -> gates -> BRIEF PPO -> deterministic flight eval
      -> one row in the CSV ledger (resumable, by idx)
    best design -> best_design.json + best.stl -> full-budget polish run
```

Two timescales, deliberately separated: control learns per physics step
(milliseconds); shape moves once per design generation (minutes). Naive
single-loop co-optimization collapses (the controller exploits whatever body
is momentarily cheap); the ledger + gates keep credit assignment clean.

The coupling pattern used here is **A — design-in-observation**: the fast-loop
policy SEES the design descriptor (span, chord, camber, mass, CD0, trim
alpha...), so it becomes a robust controller across designs, and the slow loop
ranks designs by how well that controller flies them. Upgrades, in order of
cost: (B) gradient through a differentiable aero surrogate into the design
vector; (C) a true hierarchical "designer" policy that emits theta as actions.

---

## 2. What the slow loop optimizes

Sampled parameters (from `slow_loop.ranges` in `veh3d/design.yaml`) — each is
a real aerodynamic/structural tradeoff:

| Parameter | Range | What it trades |
|---|---|---|
| `geometry.span` | 0.8 – 1.8 m | bigger span: higher AR -> less induced drag, more yaw authority — but more mass and wetted area |
| `geometry.taper` | 0.40 – 1.00 | tip chord: outboard loading vs structural depth at the tip |
| `geometry.sweep_deg` | 0 – 20° | pitch stability & looks vs tip stall behavior |
| `geometry.thickness_root` | 0.08 – 0.16 t/c | structural depth vs frontal drag (the biggest drag lever) |
| `geometry.camber` | 0.00 – 0.05 | lift at low alpha vs profile drag & pitching moment |
| `motors.motor_gap` | 0.15 – 0.55 m | differential yaw authority (bigger arm) vs fitting inside the span |
| `aero.cruise_alpha_deg` | 1.0 – 5.0° | trim point — a design OUTCOME: low alpha starves lift (lift gate), high alpha inflates induced drag (thrust gate) |

The **feasibility gates** sandwich the space so most random samples are
rejected for a named reason (the ledger records it): wing too small (lift
gate), wing too draggy (thrust gate), motor gap beyond half-span (motor gate),
broken mesh (watertight). This is deliberate — the optimizer explores a space
where physics says no as well as yes.

Fixed-but-editable knobs NOT yet sampled (candidates for future ranges):
`material.payload_mass`, `flight.motor_thrust_total`, flap/brake geometry
fractions, foam density. Add a dotted path to `slow_loop.ranges` and it is
automatically sampled, gated, and ledgered — nothing else to wire.

---

## 3. How to train

### Fast loop — control for one design (~67 s at 600k steps, CPU)

```bash
python veh3d/train_control.py                          # default design, 600k PPO
python veh3d/train_control.py --set geometry.span=1.0 --steps 100000
python veh3d/train_control.py --no-train --model results/veh3d/<run>/model.zip
```

Budget note: 300k plateaued below a hand-written PD controller; 600k clears
it. Judge by the printed metrics: `speed_err_rms` (PD floor ≈ 0.0 with the
planar env's trim), `lateral_rms`, `yaw_rms_deg`, `episodes_survived`.
Artifacts: `model.zip`, `metrics.json`, `design.stl`, `design.png`.

### Slow loop — the design search (~12 s per design at 100k steps)

```bash
python veh3d/train_design.py --n-designs 6 --tag myarm
```

Each sampled design: build -> gate -> brief PPO -> scored eval -> one CSV row
in `results/veh3d/loop_<tag>/design_loop.csv`. Re-running skips rows already
in the ledger (resumable). Read the ledger like a fix-log: `feasible=0` rows
carry the rejecting gate in `reason`; `score` is the deterministic tracking
return — higher is better. The winner gets `best_design.json`/`best.stl`,
then polish it at full budget:

```bash
python veh3d/train_control.py --set geometry.span=<winner> ... --out results/veh3d/winner_polished
```

Rules of the road: source `env.sh` first; keep `--n-envs` under half the
cores while the GPU training job runs; every gate rejection is a result, not
a bug — record it.

### Verify

```bash
python veh3d/smoke_test.py            # 8 checks: geometry, motors, gates, env, PPO, ledger
python veh3d/flight_sim.py --selftest # headless flight rollout + preview png
```

---

## 4. The tools we have (and what each is for)

| Tool | Where | Role in the pipeline |
|---|---|---|
| torch 2.14+cpu, SB3 2.9, gymnasium 1.3 | env_aero | RL: PPO for control, the fast loop's learner |
| NeuralFoil 0.3.3 (+ aerosandbox) | env_aero | microsecond section polars — the slow loop's aero oracle (≈1 % vs XFoil) |
| xfoil-python (patched) + apt xfoil | env_aero | higher-trust polars for confirmation |
| SU2 v8.5.0 (MPI, source build) | ~/SU2-v8.5.0 | 3-D CFD confirmation of gate-passing winners (next rung) |
| OpenFOAM 14 | pending `sudo aero/apt_layer.sh` | second CFD opinion / duct flows |
| trimesh 5.1 | env_aero | loft meshing, STL export, watertight checks (NOT mass properties — see pitfalls) |
| gmsh 4.15 / meshio | env_aero | CFD meshing for SU2/OpenFOAM |
| PyGeM | env_aero | FFD freeform morphing (the duct3d route) |
| mujoco 3.13 | env_aero | the interactive 6-DOF flight simulator + headless physics |
| PyYAML / matplotlib | env_aero | config + renders |

Everything is CPU-only by hard rule; the GPU runs IsaacLab training.

---

## 5. How to use the tools

**design.yaml — the design itself.** Edit parameters with any editor, or
override per run with dotted paths: `--set geometry.span=1.5 --set
geometry.camber=0.04`. Everything downstream (mesh, mass, aero, env, sim)
regenerates; never hand-edit an exported STL.

**geometry.py — preview & export.**

```bash
python veh3d/geometry.py --png /tmp/preview.png --set geometry.sweep_deg=15
python veh3d/geometry.py --stl veh3d/meshes/mydesign.stl
```

Prints mass/inertia/areas/polar and the feasibility verdict — a 1-second
design sanity check before any training.

**NeuralFoil/XFoil — section polars.** Always through the shared wrapper:
`from airfoil2d.xfoil_eval import eval_polar; eval_polar(x, y, Re, alpha,
backend="neuralfoil")` (backends: `neuralfoil`, `module`, `binary`). It
normalizes NaNs to `converged=False`. Beware numpy-2 zero-d arrays — use
`.item()`.

**trimesh — meshing only.** Watertight checks, exports, bounds. Do NOT use
`trimesh.mass_properties` after repairs: its cache has been observed
returning a centroid outside the mesh bounding box. `geometry.mass_properties`
(signed-tetrahedron integral) is the project's source of truth, cross-checked
against MuJoCo's independent mesh integral.

**SB3 PPO — the fast loop.** Use `train_control.py`'s settings as the baseline
(n_steps 200, gamma 0.99, lr default). Reward lessons already paid for: a
crash penalty must exceed one episode's avoided cost (else crashing early
looks optimal), and penalize lateral VELOCITY, not just position.

**mujoco — the flight sim.** `flight_sim.py` (interactive, desktop GL) and
`--selftest` (headless EGL). The sim consumes the SAME `design.yaml + --set`
model: lofted STL as collision-visual hull, mesh-derived mass, box-collision
collection, flaps/airbrake, wind and turbulence. See
`flight_sim_play.md` for keys.

**SU2 — winner confirmation (next rung).** The airfoil2d runner already
demonstrates serial + `mpirun -np 4` Euler runs; extend to 3-D volumes of
gate-passing winners before trusting a design for hardware.

---

## 6. Pitfalls already paid for (do not re-learn)

1. Wrong python → NaN polar → dead-looking sim. Source `env.sh`; check line 1.
2. Crash penalty < episode cost ⇒ "crash ASAP" policies with great-looking
   returns. Alive-bonus + penalty ≫ episode cost.
3. Penalize lateral velocity, not just position (drift plateau).
4. Weathervane/stiffness sign conventions: write them down before tuning
   (M_y>0 = nose down; restoring needs +cm_alpha·(α−α_trim)).
5. Explicit rate damping through xfrc is stable only while coeff·dt/I < 1
   (pitch, Iyy≈0.0012, is the stiff axis).
6. trimesh mass_properties cache lies after repair — use the project's own
   integral.
7. Adding ANY joint shifts freejoint qpos/dof addresses — always index through
   `model.jnt_qposadr[model.joint("root").id]`, never hardcoded slices.
8. YAML block edits: validate after programmatic edits (duplicate keys
   silently override — last one wins).

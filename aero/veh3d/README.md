# veh3d — parametric aerial vehicle: design loop + motor-control loop

A minimal end-to-end version of the two-timescale pipeline:

```
SLOW (design)  sample θ ∈ ranges(design.yaml) → mesh → mass/areas → NeuralFoil polar
                     → feasibility gates → brief PPO → score → ledger CSV
                     └── winner → full-budget control run (handoff)
FAST (control) Gymnasium planar flight env: 3 motors + drag from the mesh
                     geometry; objective: fly +x at constant speed, level
```

Both loops are CPU-only (`source ../env.sh`; the GPU belongs to the IsaacLab
training job).

## The config file is the design

`design.yaml` is the single source of truth. Mesh, mass properties, aero,
env, and search ranges all REGENERATE from it. There is no hand-modeled mesh:

| Block | Meaning |
|---|---|
| `geometry` | 2D section (NACA-type thickness + parabolic camber) lofted across the span: `span, root_chord, taper, sweep_deg, thickness_root/tip, twist_deg, camber, dihedral_deg` + loft resolutions |
| `material` | foam density + lumped payload mass |
| `motors` | **3 thrust points on the y axis at the back (−x)**: `motor_gap` (outer motors at y=±gap, middle at 0), `x_frac` (aft of local LE), `max_thrust`, spool lag `tau` |
| `aero` | cruise speed / trim alpha, drag build-up coefficients, `cy_beta`/`cl_beta`/`cl_r` stability derivatives |
| `sim`, `reward` | episode, limits, reward weights |
| `slow_loop.ranges` | the closed min/max the design search may sample (dotted paths into this file) |

Preview any parameter change instantly:

```bash
cd /home/heinz/isaaclab_uav/aero && source env.sh
python veh3d/geometry.py --png /tmp/preview.png --set geometry.span=1.6 \
    --set geometry.camber=0.04 --set geometry.sweep_deg=15
```

## Which software edits what

| Task | Tool | Why |
|---|---|---|
| Change the design | **any editor on `design.yaml`** (or `--set k=v` overrides) | parameters ARE the design; everything downstream rebuilds |
| Instant visual check | `python veh3d/geometry.py --png out.png` (matplotlib, headless) | plan view + 2D sections + lofted mesh in one image |
| Inspect the mesh artifact | **gmsh GUI**: `gmsh veh3d/meshes/veh_default.stl`; or **Blender** (import STL) | gmsh is installed; Blender is the UR5 collision pipeline's tool |
| Human GUI parametric design studies | **OpenVSP** (NASA, aviation-native) | same vocabulary (span/sweep/taper/twist/dihedral), built-in drag estimates; re-type values you settle on back into the YAML |
| Freeform morphs later | PyGeM FFD on the STL | the planned `duct3d/` route |
| NEVER | hand-editing the STL | it is a build artifact; edits break the parameter link and get overwritten |

The example mesh is GENERATED, not downloaded: it must be parameter-linked
and regenerable (a downloaded mesh has no parametric features to modify).

```bash
python veh3d/geometry.py --stl veh3d/meshes/veh_default.stl --png results/veh3d/veh_default.png
```

## Motors

`motors.MotorBank`: three points at `(x_mot, −gap/0/+gap, 0)`, `x_mot` on the
back (−x) side, each pushing **+x (forward)** through a first-order spool lag:

```
Fx = Σ T_i                    (forward force)
Mz = Σ (r_i × F_i)_z = −Σ y_i T_i      (differential throttle = yaw actuator)
```

(x drops out of Mz because all forces are ∥x — `motor_x()` still matters for
airframe placement and any future pitch DOF.) Hand-checked in the smoke:
`thrust=(1, .5, 0), y=(−.4, 0, .4) → Mz=+0.4 N·m`.

## Fast loop — control training

```bash
python veh3d/train_control.py                       # 600k PPO steps, default design
python veh3d/train_control.py --set geometry.span=1.0 --steps 60000
python veh3d/train_control.py --no-train --model results/veh3d/<run>/model.zip
```

Obs = `[u/vt, v/vt, ψ, r·b/2vt, y/ylim, throttles, design_vector(11)]` — the
design descriptor rides in the obs, so one policy amortizes across designs
(pipeline pattern A). Reward = `+w_alive − w_speed|u−vt| − w_side_vel|v|/vt −
w_lateral|y| − w_yaw|ψ| − w_effort Σa²`, crash `−100`.

Physics: planar (x, y, ψ), drag from the mesh-derived `CD0` + induced term
(`AR` from the actual loft), side-force and weathervane/yaw-damping
derivatives. Known reward lessons (both hit and fixed):
* crash penalty must dominate an episode's avoided cost, else crashing early
  is optimal (the 10.0 penalty trained a crash-asap policy at "return −79");
* penalize lateral VELOCITY, not just position — position-only left a
  drifting plateau.

## Slow loop — design search

```bash
python veh3d/train_design.py                        # 6 designs × 100k from ranges
python veh3d/train_design.py --n-designs 12 --tag myarm
```

Per design: sample θ → build geometry → **feasibility gates** (watertight /
lift ≥ 1.2·W at cruise / motors inside span / thrust ≥ 1.15·D) → brief PPO →
deterministic tracking eval → row in `results/veh3d/loop_<tag>/design_loop.csv`
(resumable by `idx`). Gates are the interesting output: sampled designs die on
the lift gate (wing too small), thrust gate (too draggy), or motor gate (gap
beyond half-span) — the search space is sandwiched, not everything flies.
Winner → `best_design.json` + `best.stl` + `best.png`, then polish with a
full-budget `train_control.py --set <winner params>` run.

## Interactive flight simulator (MuJoCo, 6-DOF)

`flight_sim.py` sends the SAME parametric model — `design.yaml` + `--set`
overrides → `VehicleGeom` → lofted STL, mesh-derived mass/inertia, NeuralFoil
polar, motor-line geometry — into a MuJoCo world: ground plane with checker
texture, skybox gradient, shadowed key + fill lights, tracking camera. Full
6-DOF free flight: gravity, lift/drag/side-force from relative airflow, soft
stall, weathervane + rate damping, steady wind + Ornstein-Uhlenbeck turbulence.

**Tools needed**: just the `mujoco` wheel (3.13, installed in env_aero) and a
desktop session (X11) for the viewer window. Rendering goes through desktop
OpenGL/GLFW — no CUDA, no GPU compute; the IsaacLab training job is untouched.
Headless users: `--selftest` runs the physics with EGL offscreen rendering.

```bash
cd /home/heinz/isaaclab_uav/aero && source env.sh
python veh3d/flight_sim.py                                          # calm air
python veh3d/flight_sim.py --set geometry.span=1.5 --wind 3 --wind-dir-deg 90 --turb 0.6
python veh3d/flight_sim.py --selftest              # headless 12 s + preview png
```

**Control scheme (remade 2026-09-18)**: ONE motor at the back center + TWO
trailing-edge flaps, each a real articulated body (`wing link > revolute
joint > flap link`) driven by position actuators. Flap aero comes from each
flap's OWN attack angle against the local airflow (`v_rel + ω×r`, projected
into the deflected surface frame) — not a canned roll moment.

**Keys (numpad, NUM LOCK on; arrows + row digits as fallback)**: `8/2` motor
throttle ±, `6/4` coordinated flap pair (turn), `7/1` elevator (symmetric
flaps — the pitch axis), `9` airbrake toggle (spine plates, ~3 m/s² at
cruise), `5` center flaps, `0` cut + center, `R` flight-spawn reset, `Z`
zero-park reset, `ESC` quit. HUD prints alt/speed/α/throttle/turn/elev/flap
angles/brake angle at 4 Hz. **Turn by pulsing** `6` + `7` then `5`. Full
play guide: `/home/heinz/isaaclab_uav/aero_docs/flight_sim_play.md`.

Verified: 12 s headless rollout → steady 7.9 m/s at exactly the design trim
α = 3.0°, 95 m flown, no drift; 3 m/s crosswind + turbulence → realistic
drift/buffet, still airborne. Landings/crashes are real contacts with the
ground plane.

## Verification

```bash
python veh3d/smoke_test.py    # 8 checks: cpu-guard, geometry+watertight,
                              # STL roundtrip, motor torque math, feasibility
                              # gate, env physics rollout, PPO short,
                              # design-loop mini
```

All 8 green as of 2026-09-18. Reference results (CPU): default design
mass 0.383 kg, S 0.336 m², CD0 0.081, D 1.54 N at 8 m/s, thrust margin 1.36;
600k PPO → speed err 0.50 m/s RMS, lateral 0.34 m, yaw 1.2°, 62 m flown/8 s
(winner-polish run: 4/5 full episodes).

## v0 simplifications (deliberate)

Root-section polar only (no spanwise lift distribution); frozen-alpha drag in
the env; Cauchy frontal area (exact only for convex bodies); planar dynamics
(no pitch/roll — a "planar airship" testbed); no CFD in either loop yet.
Next rungs: SU2 confirmation runs on gate-passing winners (the airfoil2d
runner already exists), CST sections via `common/airfoil.py`, CMA-ES outer
optimizer, Isaac Lab 3D port when the GPU frees up.

# Flight simulator — play guide

Interactive 6-DOF flight sim for the parametric veh3d vehicle
(`aero/veh3d/flight_sim.py`, MuJoCo 3.13, CPU-only render path — desktop
OpenGL, no CUDA; the IsaacLab training job is untouched).

## Launch

```bash
cd /home/heinz/isaaclab_uav/aero && source env.sh
python veh3d/flight_sim.py                                    # calm air
python veh3d/flight_sim.py --set geometry.span=1.5 --wind 3 --wind-dir-deg 90 --turb 0.6
python veh3d/flight_sim.py --selftest        # headless: 12 s physics + preview png
```

| Flag | Meaning | Default |
|---|---|---|
| `--set k=v` | design override, same syntax as `train_control.py` / `train_design.py` (repeatable) — the sim ALWAYS flies the parameter-generated model | — |
| `--wind` | steady wind speed [m/s] | 0 |
| `--wind-dir-deg` | direction the wind blows TOWARD (world +x = 0) | 0 |
| `--turb` | turbulence intensity (OU sigma, m/s) | 0 |
| `--alt` / `--speed` | reset-spawn altitude [m] / speed [m/s] (defaults = the above-rail, motor-off spec) | 1.3 / 0 |
| `--selftest` | no window: scripted rollout + `results/veh3d/sim_preview.png` | off |

## Keys — control scheme (numpad, remade 2026-09-18/20)

ONE motor at the back center (3.2 N, thrust +x), TWO trailing-edge flaps
(real articulated bodies, center-pivoted on their joints — they MIX into
turn + elevator like real elevons), and a split AIRBRAKE (two plates on the
spine that fold up into the flow). The HUD shows
`thr ... turn ... elev ... flaps r/l brake deg`.

**NUM LOCK ON** — or just use the **arrow keys** (↑↓ throttle, ←→ turn).
Main-row digits work as fallbacks too.

| Key | Action | What it does physically |
|---|---|---|
| `KP 8` / `↑` | motor throttle +0.12 | more thrust → speed up, then CLIMB (throttle is the climb lever; motor 3.2 N: one press ≈ +0.38 N ≈ 1 m/s², full ≈ 5 m/s²) |
| `KP 2` / `↓` | motor throttle −0.12 | glide down (clamps at 0) |
| `KP 6` / `→` | turn right +0.1 | right flap TE-up + left flap TE-down (±18°) → bank → coordinated right turn |
| `KP 4` / `←` | turn left −0.1 | mirrored pair → left turn |
| `KP 7` | ELEVATOR nose-up +0.15 | BOTH flaps TE-up together (±15°) → tail download → nose rises; use DURING turns to hold altitude |
| `KP 1` | ELEVATOR nose-down −0.15 | both flaps TE-down → nose drops, speed builds |
| `KP 9` | AIRBRAKE toggle | spine plates fold up to 70°: ~1–1.8 N of extra drag → decelerates ~3 m/s² at cruise; press again to retract |
| `KP 5` | center flaps | turn + elevator to neutral, throttle unchanged |
| `KP 0` / `Insert` | cut + center everything | dead-stick glide |
| `KP .` | SHOOT the cart | 8 m/s impulse along the borrowed launch rail (watch `cart s/v` in the HUD); the cart's spring pulls it back |
| `V` | toggle geom name labels | draws every geom name in the viewport (`cart_col_*`, hull, flaps, …) |
| `C` | (no collision boxes in this model) | reports that if pressed — reserved for box-collision models |
| HOLD a direction key | CONTINUOUS ramp | values move smoothly the whole time the key is down: throttle ±0.6/s, turn ±1.2/s, pitch ±1.0/s; a tap nudges and holds (no drift). Release stops exactly where you are |

**Standard circuit recipe**: `8`×2 to climb away → `6`+`7` together to bank
into a turn while holding the nose up → `5` to center and carve → level with
`8`/`2` → line up and `9` to brake for landing → flare with `7`, touch, `0`.

**Flap aerodynamics** are computed from each flap's OWN attack angle against
the local airflow: the flow at the surface is `v_rel + ω × r` (vehicle motion
plus rotation-induced flow at the hinge), projected into the deflected flap's
frame, `α_flap = atan2(−w, u)` there, lift = `q·A·cl_α·α_flap` (soft cap
±1.1) along the surface normal plus profile drag. So deflection, vehicle α,
and body rates all enter — the flaps also act as dampers and feel gusts.

**How to turn (important):** tap `KP 6` a few times, add `KP 7` (elevator)
to hold the nose, then `KP 5` to center and let it carve. There is NO roll
spring (only rate damping) — a held turn deflection rolls into a spiral;
pulse and center, like a real stick. Banked turns still bleed some altitude
even with elevator — add `KP 8` for a sustained level turn.

**Airbrake**: `KP 9` toggles the two spine plates (watch `brake` deg in the
HUD swing to 70). Expect ~2–3 m/s² deceleration at cruise plus a slight
sink — combine with `KP 7` to hold the nose during the slowdown.

## Keys — episode reset

| Key | Reset to | State |
|---|---|---|
| `R` | **above the rail** | position (−0.35, 0, 1.3 m) — or `--alt` — AT REST, **MOTOR OFF**, flaps neutral; cart re-zeroed to the rail start; gust state KEPT |
| `Z` | **zero / parked** | position (0, 0, on-the-ground), attitude level, ALL velocities 0 (flap hinges included), motor off, everything neutral, gust zeroed |

⚠️ Both resets leave the MOTOR OFF (spec): from rest or from a 1.3 m drop the
drone cannot fly on thrust alone — it will free-fall to the ground. To fly,
spawn (`--speed 8 --alt 6`) or edit the spawn in `main()`; the airborne
selftest does exactly that.

**Borrowed rail** (from `scripts/mujoco/uav_rail_scene.xml`, spawned at the
origin): `base_link` truss (visual-only mesh, exactly as upstream — their
note: colliding it blasts the airframe) → `rail_link` tilted 11.5° →
`launch_cart` (5 kg) on a prismatic joint with spring return. `KP .` kicks
the cart up-rail at 8 m/s; spring+damping walk it back. The drone spawns
above it at (−0.35, 0, 1.3).

**Collision (blend-authored)**: the drone's collision boxes come from
the BLEND EXPORT by default (`collision source:` line at startup shows the
file). To edit them: change the `veh.Cube.*` objects in
`assets/shahed-136/rail_cart_shahed_collision.blend`, run the
`export_collision_blend_mujoco_typed.py` command from the collision
section's history (or `--validate` to check), and rerun the sim — new sizes
and positions land automatically, mapped proportionally onto our airframe.
The hull mesh is visual-only when boxes exist (they carry the collision);
with no export present the sim falls back to `uav_rail_scene_prim.xml`.
Knobs that still apply on top: the `col_x_shift` dial and per-box
`col_box_overrides` (`col_box_scale` only affects the legacy "envelope"
mode). The launch cart collides through its `cart_col_*` boxes; the rail
truss stays visual-only. Landings and crashes are real contacts; there is
no damage model.


## Reading the HUD (terminal, 4 Hz)

```
t  120.3s | alt   6.1 m | V   7.9 m/s | a  +3.0deg | thr 0.55 turn +0.0 elev +0.0 flaps  +0.1/ -0.1 brake   0deg | airborne
```

- `alt` — height of the body origin above the ground plane
- `V` — airspeed (relative to the wind, not ground speed)
- `a` — angle of attack α (trimmed ≈ the design's `cruise_alpha_deg`)
- `thr` — motor throttle; `turn` / `elev` — your turn and pitch commands
- `flaps` — ACTUAL joint angles right/left in degrees (they lag the command
  under aero load — the position servo is springy, like a real one)
- `brake` — airbrake plate angle (0 = retracted, 70 = fully open)
- `cart s/v` — borrowed launch-rail cart: stroke position and velocity
- `GROUNDED` — hull is in contact with the ground plane

## Mouse / viewer

Left-drag orbits (around the vehicle — the camera tracks it), scroll zooms,
right-drag pans. `Tab` toggles the MuJoCo info overlay, `F1`/`F2` hide the
side panels. The vehicle body origin sits at its mesh origin (LE of the root
section); the red dots are the motor sites.

## Physics in one paragraph

Gravity + aerodynamics from the RELATIVE airflow: linearized lift from the
design's NeuralFoil polar (CL = cl_c·α/α_trim, soft stall ±1.3), drag =
mesh-derived CD0 + induced (real AR/S from the loft), side force, weathervane
yaw + pitch/roll/yaw rate damping. Every hinged surface — flaps AND brake
plates — contributes its own force from its LOCAL attack angle (`v_rel +
ω×r` projected into the deflected surface frame). Pitch trims near the
design cruise α; `flight:`/`flaps:`/`brakes:` blocks in design.yaml are
sim-world constants the design loop does NOT yet optimize; the RL training
env (`env_veh.py`) still uses the older 3-motor bank — the two control
schemes are intentionally separate until the RL side is remade. Landings
and crashes are real contacts; there is no damage model.

## Troubleshooting

- **`cruise lift margin nanx` + keys feel dead**: the sim refused to start
  (FATAL: section polar failed) — you are in the WRONG PYTHON. `env_isaaclab`
  has mujoco/trimesh but NO neuralfoil, so the aerodynamic polar comes out
  NaN and the vehicle thrashes in place (this exact state used to look like
  "keys not responsive"). Fix:
  `cd /home/heinz/isaaclab_uav/aero && source env.sh`, rerun. The sim now
  hard-exits with the interpreter path printed on the first line — check it
  says `.../env_aero/bin/python`.
- `X Error ... GLXBadDrawable` at the very END after quitting normally is a
  known cosmetic driver quirk — harmless. If it appears mid-run, it means a
  python exception killed the loop; check the traceback above it.
- No window on a headless/SSH box: use `--selftest` (EGL offscreen).
- The sim paces itself to real time; if the machine is loaded it runs slow
  but never fast.

## File map

- `aero_docs/rl_physical_design.md` — the full RL-for-physical-design guide:
  pipeline, slow-loop parameters, training commands, tools inventory
- `scripts/mujoco/export_collision_blend_mujoco_typed.py` — blend → MuJoCo
  typed collision export: replaces ONLY box/cylinder collisions with the
  blend's primitives (spheres/capsules/meshes stay in place), can rename
  blend groups to our naming (`--rename-in-blend shahed=veh`), and
  `--validate` checks the result against the blend + reference scene
- `aero/veh3d/flight_sim.py` — the sim (MJCF builder, 6-DOF aero, teleop)
- `aero/veh3d/design.yaml` — the design + `flight:` sim constants
- `aero/veh3d/README.md` — full pipeline (design loop, control training)
- `aero/results/veh3d/sim_cache/` — parameter-hashed STLs the sim spawns

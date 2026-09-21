"""Interactive 6-DOF flight simulator for the parametric veh3d vehicle.

The SAME model the design loop trains on (design.yaml + `--set` overrides ->
VehicleGeom -> lofted STL, mass, inertia, NeuralFoil polar, motor line) is
packaged into a MuJoCo scene: ground plane, skybox, shadowed key light,
tracking camera. Full 6-DOF free flight with gravity, lift/drag/sideforce,
stall soft-cap, weathervane + rate damping, steady wind + OU turbulence.

    cd /home/heinz/isaaclab_uav/aero && source env.sh
    python veh3d/flight_sim.py                     # default design, calm air
    python veh3d/flight_sim.py --set geometry.span=1.6 --wind 3 --wind-dir-deg 40 --turb 0.6
    python veh3d/flight_sim.py --selftest          # headless physics + EGL preview png

Keys (teleop)  — numpad with NUM LOCK ON, or the ARROW KEYS (Num Lock off);
-------------  main-row digits also work
  KP 8 / 2   motor throttle up / down (single motor, back center)
  KP 4 / 6   coordinated FLAP pair: turn LEFT / RIGHT
             (6 -> right flap TE-up + left flap TE-down -> bank + turn right)
  KP 7 / 1   ELEVATOR (both flaps symmetric): 7 = nose up, 1 = nose down
  KP 9       AIRBRAKE toggle: spine plates fold up into the flow (drag)
  KP 5       center flaps (turn + elevator to neutral)
  KP 0       cut throttle AND center everything (Insert = same)
  KP .       SHOOT the cart along the rail (15 m/s impulse;
             --shoot-speed to change)
  V          toggle GEOM NAME LABELS in the viewport (see the collision boxes'
             names: veh_col_*, cart_col_*, ...); double-click selects a body
  R          reset: drone at (-0.35, 0, 1.3) above the rail, at rest, MOTOR OFF
  Z          ZERO: parked at origin, pos/vel/throttle = 0 (no takeoff from rest)
  ESC        quit
  mouse      orbit/zoom (camera keeps tracking the vehicle)
  Tab/F1/F2  built-in MuJoCo viewer panels

Flight-model notes: the wing is pitch-trimmed near its design cruise alpha
(cm_alpha restoring) — throttle acts as the climb lever, like a motor
glider. Each flap is a REAL hinged body (wing link > revolute joint > flap
link) whose aero force comes from its OWN attack angle against the local
airflow (v_rel + w x r at the surface) — so flap deflection, vehicle alpha
and pitch rate all enter, and rate changes give natural damping. cm_*/cl_p
are sim-world constants (design.yaml `flight:`); the RL training env
(env_veh.py) still uses the older 3-motor bank and has no flaps.

Render path: MuJoCo's native viewer draws through desktop OpenGL (GLFW) —
no CUDA, no GPU compute; the IsaacLab training job is untouched.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import glfw          # module import is display-safe; only windows need X
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import build, load_design, motor_x, overrides_to_cfg  # noqa: E402

# Borrowed rail object (scripts/mujoco/uav_rail_scene.xml): base_link ->
# rail_link (11.5 deg tilt) -> launch_cart on a prismatic joint. Spawned at
# the origin; the rail truss mesh is VISUAL-ONLY exactly as upstream (their
# note: the mesh encloses the start pose, colliding it blasts the airframe).
RAIL_ASSET_DIR = Path("/home/heinz/isaaclab_uav/assets/shahed-136/source")
PRIM_SCENE = Path("/home/heinz/isaaclab_uav/scripts/mujoco/"
                  "uav_rail_scene_prim.xml")   # cart_col_* box source
TYPED_SCENE = Path("/home/heinz/isaaclab_uav/scripts/mujoco/"
                   "uav_rail_scene_typed.xml")  # blend export (DEFAULT source)
CART_SHOOT_SPEED = 35.0       # m/s along the rail, one KP_DECIMAL press (user: stronger)


def _prim_boxes(prefix, path=None):
    """Box collision collection borrowed from a prim scene (default
    uav_rail_scene_prim.xml).

    Returns [(size(3,), pos(3,), quat(4,))] for the <prefix>_col_* geoms."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path or str(PRIM_SCENE)).getroot()
    out = []
    for g in root.iter("geom"):
        name = g.get("name") or ""
        if g.get("type") == "box" and name.startswith(prefix + "_col"):
            quat = (np.array([float(x) for x in g.get("quat").split()])
                    if g.get("quat") else np.array([1.0, 0, 0, 0]))
            out.append((np.array([float(x) for x in g.get("size").split()]),
                        np.array([float(x) for x in g.get("pos").split()]),
                        quat))
    return out


def _box_geom_xml(name, size, pos, quat, extra):
    q = f' quat="{quat[0]:.5f} {quat[1]:.5f} {quat[2]:.5f} {quat[3]:.5f}"'
    return (f'      <geom name="{name}" type="box"'
            f' size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"'
            f' pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"{q} {extra}/>')

# -- MJCF --------------------------------------------------------------------
def default_collision_source():
    """Collision-box source: the BLEND EXPORT when present (edit the boxes in
    Blender, run export_collision_blend_mujoco_typed.py, rerun the sim),
    else the original prim scene."""
    if TYPED_SCENE.exists():
        return str(TYPED_SCENE)
    print(f"[COLLISION] note: no blend export found ({TYPED_SCENE.name}) -- "
          f"falling back to {PRIM_SCENE.name}")
    return str(PRIM_SCENE)


def _map_boxes_onto(boxes, target_min, target_max):
    """Affinely map a box collection's envelope onto a target bounding box
    (per-axis scale + translate; quats kept — standard proxy approximation
    for rotated boxes under non-uniform scaling)."""
    lo = np.array([b[1] - b[0] for b in boxes]).min(axis=0)
    hi = np.array([b[1] + b[0] for b in boxes]).max(axis=0)
    scale = (np.asarray(target_max) - np.asarray(target_min)) / (hi - lo)
    off = np.asarray(target_min) - lo * scale
    return [(b[0] * scale, b[1] * scale + off, b[2]) for b in boxes]


def flap_layout(geom, cfg):
    """Trailing-edge flap geometry derived from the wing, both sides.

    Returns dict with, per side: hinge/TE reference position (vehicle frame),
    chord, span, mid-span arm y, area. The flap BODY is centered on its
    joint (shape center = hinge = com); the box extends aft of the wing TE.
    """
    g = cfg["geometry"]
    fl = cfg["flaps"]
    b2 = geom.span / 2.0
    y_in, y_out = fl["span_in_frac"] * b2, fl["span_out_frac"] * b2
    y_mid = 0.5 * (y_in + y_out)
    eta = y_mid / b2
    chord = g["root_chord"] * (1.0 - (1.0 - g["taper"]) * eta)
    c_f = fl["chord_frac"] * chord
    span_f = y_out - y_in
    # wing TE at the flap mid-span station: LE x at |y| minus local chord
    x_te = -np.tan(np.deg2rad(g["sweep_deg"])) * y_mid - chord
    return dict(y_mid=y_mid, chord=c_f, span=span_f, x_hinge=x_te,
                area=c_f * span_f, max_rad=np.deg2rad(fl["max_deflect_deg"]))


def brake_layout(geom, cfg):
    """Split airbrake geometry: two plates flanking the spine at the CG x
    (drag without a pitch moment), hinged about y, folding UP into the flow."""
    b = cfg["brakes"]
    return dict(chord=b["chord"], span=b["span"], y_off=b["y_offset"],
                z=b["z_mount"], x=float(geom.cg[0]),
                area=b["chord"] * b["span"],
                max_rad=np.deg2rad(b["max_deg"]))


def build_mjcf(geom, cfg, spawn):
    """Scene + vehicle MJCF from the analyzed geometry. Returns (xml, assets).

    Control scheme (remade 2026-09-18): ONE motor at the back center + TWO
    trailing-edge flaps, each a real body: wing link > hinge (revolute, y
    axis) > flap link, driven by a position actuator. Body inertia is
    composed by MuJoCo from geoms (mesh + payload sphere + flap boxes).
    """
    import hashlib
    import json

    com = geom.cg
    foam_mass = geom.mass - cfg["material"]["payload_mass"]
    # cache the lofted STL keyed by the parameters that built it
    key = hashlib.md5(json.dumps(geom.cfg["geometry"], sort_keys=True,
                                 default=str).encode()).hexdigest()[:10]
    stl = base_dir().parent / "results" / "veh3d" / "sim_cache" / f"veh_{key}.stl"
    stl.parent.mkdir(parents=True, exist_ok=True)
    if not stl.exists():
        from geometry import export_stl
        export_stl(geom, stl)
    stl_bytes = stl.read_bytes()
    rail_bytes = (RAIL_ASSET_DIR / "rail_collision.stl").read_bytes()
    cart_bytes = (RAIL_ASSET_DIR / "launch_cart.stl").read_bytes()

    x_mot = motor_x(geom)
    fg = flap_layout(geom, cfg)
    bg = brake_layout(geom, cfg)
    fl = cfg["flaps"]
    br = cfg["brakes"]
    # ONE collision source for every box group (drone + cart): the blend
    # export by default, --collision-xml to override
    col_src = (cfg["flight"].get("collision_xml")
               or default_collision_source())
    flaps_xml = ""
    for side, sgn in (("right", +1), ("left", -1)):
        # body origin AT the flap shape center (user spec): the box is
        # centered on the joint, so the frame/COM marker sits mid-chord,
        # mid-span. The hinge line passes through that center.
        x_c = fg["x_hinge"] - fg["chord"] / 2.0
        flaps_xml += f"""
    <body name="flap_{side}" pos="{x_c:.5f} {sgn * fg['y_mid']:.5f} 0">
      <joint name="j_flap_{side}" type="hinge" axis="0 1 0"
             range="-{fg['max_rad']:.4f} {fg['max_rad']:.4f}"
             damping="{fl['damping']}"/>
      <geom name="flap_{side}_geom" type="box"
            size="{fg['chord'] / 2:.5f} {fg['span'] / 2:.5f} 0.002"
            pos="0 0 0"
            rgba="0.15 0.15 0.9 1" mass="{fl['mass']}"
            friction="0.8 0.02 0.002" condim="3"/>
    </body>"""
    # cart collision: same source as the drone boxes (blend export by
    # default) — edit cart.Cube.* in the blend, re-export, cart updates too
    cart_col_xml = "\n".join(
        _box_geom_xml(f"cart_col_{i:03d}", sz, ps, q,
                      'density="0" rgba="0.2 0.9 0.4 0.25" group="0"')
        for i, (sz, ps, q) in enumerate(_prim_boxes("cart", path=col_src)))
    # drone collision boxes: the BLEND EXPORT mapped onto OUR airframe
    # envelope, non-wing boxes inflated by col_box_scale, whole cloud shifted
    # by col_x_shift, per-box overrides on top. When boxes exist the hull
    # mesh is visual-only (they carry the collision); with none, the hull
    # collides. veh_col_012 (the 1.35 m fuselage slab) stays deleted.
    # col_map_mode: "verbatim" (default) uses the blend boxes EXACTLY as
    # authored — what you model is what collides. "envelope" is the legacy
    # auto-fit onto our airframe bbox (col_box_scale then applies; it does
    # nothing in verbatim mode).
    bmin, bmax = geom.mesh.bounds
    col_mode = cfg["flight"].get("col_map_mode", "verbatim")
    col_scale = float(cfg["flight"].get("col_box_scale", 1.0))
    x_shift = float(cfg["flight"].get("col_x_shift", 0.0))
    col_over = cfg["flight"].get("col_box_overrides") or {}
    col_prefix = "veh" if "typed" in os.path.basename(col_src) else "shahed"
    scaled = []
    for i, (sz, ps, q) in enumerate(_map_boxes_onto(
            _prim_boxes(col_prefix, path=col_src), bmin, bmax)
            if col_mode == "envelope" else _prim_boxes(col_prefix, path=col_src)):
        if i == 11:                     # veh_col_012: deleted (user spec)
            continue
        wing = (i == 10) and col_mode == "envelope"   # tint only in envelope mode
        if col_mode == "envelope":
            k = 1.0 if wing else col_scale
            sz, ps = sz * k, ps * k
        name = f"veh_col_{len(scaled):03d}"
        ov = col_over.get(name) or {}
        if "size" in ov:
            sz = np.array(ov["size"], float)
        if "pos" in ov:
            ps = np.array(ov["pos"], float)
        ps = ps + np.array([x_shift, 0.0, 0.0])
        scaled.append((name, sz, ps, q, wing))
    veh_col_xml = "\n".join(
        _box_geom_xml(name, sz, ps, q,
                      'mass="0" rgba="0.9 0.2 0.2 0.10" group="0"'
                      if wing else
                      'mass="0" rgba="0.2 0.9 0.4 0.18" group="0"')
        for name, sz, ps, q, wing in scaled)
    hull_col = "0" if scaled else "1"
    hull_group = "1" if scaled else "0"
    brake_xml = ""
    for side, sgn in (("left", -1), ("right", +1)):
        brake_xml += f"""
    <body name="brake_{side}" pos="{bg['x']:.5f} {sgn * bg['y_off']:.5f} {bg['z']:.4f}">
      <joint name="j_brake_{side}" type="hinge" axis="0 1 0"
             range="0 {bg['max_rad']:.4f}" damping="{br['damping']}"/>
      <geom name="brake_{side}_geom" type="box"
            size="{bg['chord'] / 2:.5f} {bg['span'] / 2:.5f} 0.002"
            pos="0 0 0"
            rgba="0.9 0.6 0.1 1" mass="{br['mass']}"
            contype="0" conaffinity="0"/>
    </body>"""

    xml = f"""
<mujoco model="veh3d_flight">
  <option timestep="{spawn['dt']:.5f}" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>
    <global offwidth="1600" offheight="900"/>
    <map shadowclip="0.5"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.55 0.85"
             rgb2="0.85 0.92 1.0" width="512" height="512"/>
    <texture type="2d" name="groundtex" builtin="checker"
             rgb1="0.22 0.38 0.25" rgb2="0.34 0.50 0.36"
             width="512" height="512"/>
    <material name="groundmat" texture="groundtex" texrepeat="25 25"
              reflectance="0.08"/>
    <mesh name="veh_mesh" file="veh.stl"/>
    <mesh name="rail_mesh" file="rail_collision.stl"/>
    <mesh name="cart_mesh" file="launch_cart.stl"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="0 -8 20" dir="0 0.35 -1"
           diffuse="0.9 0.9 0.85" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" directional="true" pos="8 6 10" dir="-0.5 -0.4 -1"
           diffuse="0.35 0.38 0.45" castshadow="false"/>
    <geom name="ground" type="plane" size="200 200 0.1" material="groundmat"
          friction="0.9 0.02 0.002"/>
    <body name="base_link" pos="0 0 0" quat="1 0 0 0">
      <geom name="base_geom" type="mesh" mesh="rail_mesh"
            rgba="0.25 0.30 0.35 0.45"
            density="0" contype="0" conaffinity="0"/>
      <body name="rail_link" pos="-0.65 0 0.71" euler="0 -11.5 0">
        <body name="launch_cart" pos="0 0 0">
          <joint name="rail_joint" type="slide" axis="1 0 0"
                 limited="true" range="0 2"
                 damping="5" stiffness="10" armature="0.01"/>
          <inertial pos="0 0 0" mass="5" diaginertia="0.05 0.05 0.02"/>
          <geom name="cart_geom" type="mesh" mesh="cart_mesh"
                rgba="1.0 0.4 0.1 1" density="0" contype="0" conaffinity="0"
                group="1"/>
{cart_col_xml}
        </body>
      </body>
    </body>
    <body name="vehicle" pos="{spawn['pos'][0]} {spawn['pos'][1]} {spawn['pos'][2]}">
      <freejoint name="root"/>
      <geom name="hull" type="mesh" mesh="veh_mesh" rgba="0.85 0.15 0.15 1"
            mass="{foam_mass:.5f}" friction="0.8 0.02 0.002" condim="3"
            contype="{hull_col}" conaffinity="{hull_col}" group="{hull_group}"/>
{veh_col_xml}
      <geom name="payload" type="sphere" pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}"
            size="0.02" mass="{cfg['material']['payload_mass']:.5f}"
            rgba="0 0 0 0" contype="0" conaffinity="0"/>
      <site name="motor" pos="{x_mot:.5f} 0 0" size="0.015" rgba="1 0.2 0.2 0.9"/>
{flaps_xml}
{brake_xml}
      <camera name="chase" mode="trackcom" pos="0 -3 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act_flap_right" joint="j_flap_right" kp="{fl['kp']}"
              ctrlrange="-{fg['max_rad']:.4f} {fg['max_rad']:.4f}"/>
    <position name="act_flap_left" joint="j_flap_left" kp="{fl['kp']}"
              ctrlrange="-{fg['max_rad']:.4f} {fg['max_rad']:.4f}"/>
    <position name="act_brake_left" joint="j_brake_left" kp="{br['kp']}"
              ctrlrange="0 {bg['max_rad']:.4f}"/>
    <position name="act_brake_right" joint="j_brake_right" kp="{br['kp']}"
              ctrlrange="0 {bg['max_rad']:.4f}"/>
  </actuator>
</mujoco>
"""
    return xml, {"veh.stl": stl_bytes, "rail_collision.stl": rail_bytes,
                 "launch_cart.stl": cart_bytes}


def base_dir():
    return Path(__file__).resolve().parent


# -- wind --------------------------------------------------------------------
class Wind:
    """Steady wind + Ornstein-Uhlenbeck turbulence (world frame)."""

    def __init__(self, speed, dir_deg, turb, rng):
        self.steady = speed * np.array([np.cos(np.deg2rad(dir_deg)),
                                        np.sin(np.deg2rad(dir_deg)), 0.0])
        self.sigma, self.tau = turb, 1.5
        self.gust = np.zeros(3)
        self.rng = rng

    def step(self, dt):
        if self.sigma <= 0:
            return self.steady
        a = dt / self.tau
        self.gust += -a * self.gust + self.sigma * np.sqrt(2 * a) * \
            self.rng.standard_normal(3)
        return self.steady + self.gust


# -- teleop state ------------------------------------------------------------
class CenterMotor:
    """Single motor at the back center: +x thrust through a spool lag.

    Sim-world sizing from flight.motor_thrust_total (falls back to the
    design bank equivalent count * max_thrust if absent)."""

    def __init__(self, cfg, dt):
        m = cfg["motors"]
        self.t_max = float(cfg["flight"].get(
            "motor_thrust_total", m["count"] * m["max_thrust"]))
        self.tau = max(float(m["tau"]), 1e-3)
        self.dt = float(dt)
        self.thrust = 0.0

    def reset(self, thrust=0.0):
        self.thrust = float(thrust)

    def command(self, throttle):
        tgt = float(np.clip(throttle, 0.0, 1.0)) * self.t_max
        self.thrust += (tgt - self.thrust) * min(1.0, self.dt / self.tau)


class Teleop:
    """Numpad key state (Num Lock on; main-row digits + arrows as fallback):
    8/2 motor throttle, 4/6 turn (flap pair), 7/1 elevator (symmetric flaps),
    9 airbrake toggle, 5 center flaps, 0 cut + center."""

    _UP = frozenset({glfw.KEY_KP_8, glfw.KEY_8, glfw.KEY_UP})
    _DOWN = frozenset({glfw.KEY_KP_2, glfw.KEY_2, glfw.KEY_DOWN})
    _LEFT = frozenset({glfw.KEY_KP_4, glfw.KEY_4, glfw.KEY_LEFT})
    _RIGHT = frozenset({glfw.KEY_KP_6, glfw.KEY_6, glfw.KEY_RIGHT})
    _PITCH_UP = frozenset({glfw.KEY_KP_7, glfw.KEY_7})
    _PITCH_DN = frozenset({glfw.KEY_KP_1, glfw.KEY_1})
    _BRAKE = frozenset({glfw.KEY_KP_9, glfw.KEY_9})
    _CENTER = frozenset({glfw.KEY_KP_5, glfw.KEY_5})
    _CUT = frozenset({glfw.KEY_KP_0, glfw.KEY_0, glfw.KEY_INSERT})

    RAMP_HOLD = 0.35      # s of ramping per key event (repeat refreshes it)
    THROTTLE_RATE = 0.60  # throttle fraction per second (full sweep ~1.7 s)
    TURN_RATE = 1.20      # turn command per second (full sweep ~0.9 s)
    PITCH_RATE = 1.00     # elevator command per second (full sweep ~1.0 s)

    def __init__(self, throttle0):
        self.common = float(np.clip(throttle0, 0.0, 1.0))
        self.turn = 0.0                 # -1 full left .. +1 full right
        self.pitch = 0.0                # -1 nose down .. +1 nose up (elevator)
        self.brake = False              # airbrake toggle
        self._ramp = {"common": 0.0, "turn": 0.0, "pitch": 0.0}
        self._ramp_until = 0.0

    def on_key(self, key, now):
        if key in self._UP:
            self._ramp = {"common": +1.0, "turn": 0.0, "pitch": 0.0}
        elif key in self._DOWN:
            self._ramp = {"common": -1.0, "turn": 0.0, "pitch": 0.0}
        elif key in self._LEFT:
            self._ramp = {"common": 0.0, "turn": -1.0, "pitch": 0.0}
        elif key in self._RIGHT:
            self._ramp = {"common": 0.0, "turn": +1.0, "pitch": 0.0}
        elif key in self._PITCH_UP:
            self._ramp = {"common": 0.0, "turn": 0.0, "pitch": +1.0}
        elif key in self._PITCH_DN:
            self._ramp = {"common": 0.0, "turn": 0.0, "pitch": -1.0}
        elif key in self._BRAKE:
            self.brake = not self.brake
        elif key in self._CENTER:
            self.turn, self.pitch = 0.0, 0.0
            self._ramp["turn"] = self._ramp["pitch"] = 0.0
        elif key in self._CUT:
            self.common, self.turn, self.pitch = 0.0, 0.0, 0.0
            self._ramp = {"common": 0.0, "turn": 0.0, "pitch": 0.0}
        else:
            return False
        self._ramp_until = now + self.RAMP_HOLD
        return True

    def update(self, dt, now):
        """Advance the active ramp. Call once per frame with the wall dt."""
        if now > self._ramp_until:
            self._ramp = {"common": 0.0, "turn": 0.0, "pitch": 0.0}
            return
        self.common = float(np.clip(
            self.common + self._ramp["common"] * self.THROTTLE_RATE * dt,
            0.0, 1.0))
        self.turn = float(np.clip(
            self.turn + self._ramp["turn"] * self.TURN_RATE * dt, -1.0, 1.0))
        self.pitch = float(np.clip(
            self.pitch + self._ramp["pitch"] * self.PITCH_RATE * dt, -1.0, 1.0))


# -- surface aerodynamics (flaps AND brakes) ---------------------------------
_RHO = 1.225


def plate_aero(v_rel_body, w_body, theta, geo, surf, side_sign):
    """Force [N, body frame] on ONE hinged surface from its LOCAL AoA.

    geo  = dict(chord, y_mid)           (placement / moment arm)
    surf = dict(cl_alpha, cl_max, cd0)  (aero coefficients)
    theta>0 = trailing edge UP. Local airflow = vehicle flow + rotational
    flow (w x r_cp) — the surface feels its own attack angle against the
    air, which gives every surface natural damping authority.
    """
    chord_dir = np.array([-np.cos(theta), 0.0, np.sin(theta)])  # hinge->TE
    x_f = -chord_dir                                            # surface fwd
    z_f = np.array([np.sin(theta), 0.0, np.cos(theta)])         # normal
    r_cp = chord_dir * (geo["chord"] / 2.0)
    r_cp[1] = side_sign * geo["y_mid"]

    v_f = np.asarray(v_rel_body) + np.cross(w_body, r_cp)
    V = max(np.linalg.norm(v_f), 1e-4)
    u_f, w_f = float(v_f @ x_f), float(v_f @ z_f)
    alpha_f = np.arctan2(-w_f, u_f)
    q = 0.5 * _RHO * V * V

    CL = surf["cl_max"] * np.tanh(surf["cl_alpha"] * alpha_f / surf["cl_max"])
    CD = surf["cd0"] + 0.20 * CL ** 2
    F = q * geo["area"] * (CL * z_f - CD * v_f / V)
    return F, dict(alpha_f=alpha_f, CL=CL, V=V)


def flap_aero(v_rel_body, w_body, theta, fg, cfg, side_sign):
    """Flap wrapper of plate_aero (flaps coefficients + geometry)."""
    return plate_aero(v_rel_body, w_body, theta,
                      dict(chord=fg["chord"], y_mid=fg["y_mid"],
                           area=fg["area"]),
                      cfg["flaps"], side_sign)


# -- 6-DOF aero + motors -----------------------------------------------------
def aero_forces(geom, cfg, v_world, w_body, R, wind_vec):
    """Body-frame aero force [N] and torque [N m] from relative airflow."""
    a, fl = cfg["aero"], cfg["flight"]
    v_rel = R.T @ (np.asarray(v_world) - np.asarray(wind_vec))   # body frame
    V = max(np.linalg.norm(v_rel), 1e-4)
    vhat = v_rel / V
    alpha = np.arctan2(-v_rel[2], v_rel[0])
    beta = np.arcsin(np.clip(v_rel[1] / V, -1.0, 1.0))
    q = 0.5 * cfg["aero"]["rho"] * V * V

    cl_alpha = geom.cl_c / np.deg2rad(a["cruise_alpha_deg"])
    CL = cl_alpha * alpha                                   # linearized polar
    CL = 1.3 * np.tanh(CL / 1.3)                            # soft stall cap
    CD = geom.CD0 + geom.k_ind * CL ** 2

    F = -q * geom.S_ref * CD * vhat                         # drag
    up = np.array([0.0, 0.0, 1.0]) - np.dot([0.0, 0.0, 1.0], vhat) * vhat
    up /= max(np.linalg.norm(up), 1e-6)
    F = F + q * geom.S_ref * CL * up                        # lift
    F = F + np.array([0.0, -q * geom.S_ref * a["cy_beta"] * beta, 0.0])

    # moments, body convention: M_x>0 rolls left, M_y>0 pitches DOWN,
    # M_z>0 yaws RIGHT. Stored derivatives are positive magnitudes.
    #   pitch stiffness:  alpha > trim  ->  want nose DOWN  ->  +M_y
    #   pitch damping:    oppose omega_y
    #   weathervane:      beta > 0 (wind from right) -> yaw right -> +M_z
    #   yaw/roll damping: oppose the rate
    S, b, c = geom.S_ref, geom.span, geom.chord_root
    M = np.array([
        -q * S * b * fl["cl_p"] * b / (2 * V) * w_body[0],
        q * S * c * fl["cm_alpha"] * (alpha - np.deg2rad(a["cruise_alpha_deg"]))
        - q * S * c * fl["cm_q"] * c / (2 * V) * w_body[1],
        q * S * b * (a["cl_beta"] * beta - a["cl_r"] * b / (2 * V) * w_body[2]),
    ])
    return F, M, dict(V=V, alpha=alpha, beta=beta, CL=CL, CD=CD)


def quat_to_mat(quat):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    return R.reshape(3, 3)


# -- shared sim loop pieces --------------------------------------------------
def make_model(geom, cfg, spawn):
    import mujoco
    xml, assets = build_mjcf(geom, cfg, spawn)
    return mujoco.MjModel.from_xml_string(xml, assets)


def reset_state(data, model, spawn, motor, teleop, throttle0, zero=False,
                geom=None, wind=None):
    """R-key reset (spawn: flight state, trim throttle) or Z-key zero reset.

    zero=True parks the vehicle AT REST ON THE GROUND at the origin:
    position (0, 0, hull-on-ground), attitude level, all velocities zero
    (flap hinges included), motor off, flaps neutral, gust state zeroed.
    With contacts holding it, the acceleration is ~0 — note the motor pushes
    +x only, so a parked vehicle cannot take off; use R for a flying start.
    """
    jnt = model.jnt_qposadr[model.joint("root").id]
    dof = model.jnt_dofadr[model.joint("root").id]
    data.qpos[:] = 0.0
    data.qpos[jnt + 3:jnt + 7] = spawn["quat"]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    # borrowed rail: cart back to the start of its stroke
    rj = model.jnt_qposadr[model.joint("rail_joint").id]
    rdof = model.jnt_dofadr[model.joint("rail_joint").id]
    data.qpos[rj] = 0.0
    data.qvel[rdof] = 0.0
    if zero:
        z_park = -float(geom.mesh.bounds[0][2]) + 0.002 if geom is not None \
            else 0.05
        data.qpos[jnt:jnt + 3] = (0.0, 0.0, z_park)
        motor.reset(0.0)
        teleop.common, teleop.turn, teleop.pitch, teleop.brake = 0.0, 0.0, 0.0, False
        if wind is not None:
            wind.gust = np.zeros(3)
    else:
        data.qpos[jnt:jnt + 3] = spawn["pos"]
        data.qvel[dof:dof + 3] = spawn["vel"]
        thr0 = float(spawn.get("throttle", 0.0))   # standard reset = motor OFF
        motor.reset(thr0 * motor.t_max)
        teleop.common, teleop.turn = thr0, 0.0
        teleop.pitch, teleop.brake = 0.0, False
    data.xfrc_applied[:] = 0.0
    import mujoco
    mujoco.mj_forward(model, data)


def physics_step(model, data, geom, cfg, motor, teleop, wind, fg=None, bg=None):
    import mujoco
    motor.command(teleop.common)
    jnt = model.jnt_qposadr[model.joint("root").id]
    dof = model.jnt_dofadr[model.joint("root").id]
    # NEVER hardcode the freejoint slice — the borrowed rail's slide joint
    # sits BEFORE it in the tree, shifting every address by one
    R = quat_to_mat(data.qpos[jnt + 3:jnt + 7])
    v_world = data.qvel[dof:dof + 3]
    w_body = data.qvel[dof + 3:dof + 6]
    wind_vec = wind.step(cfg["sim"]["dt"])
    F_a, M_a, info = aero_forces(geom, cfg, v_world, w_body, R, wind_vec)

    bid = model.body("vehicle").id
    F_m = R @ np.array([motor.thrust, 0.0, 0.0])   # +x thrust, back center
    F_tot = R @ F_a + F_m
    F_tot[2] -= geom.mass * 9.81
    data.xfrc_applied[bid, :3] = F_tot
    data.xfrc_applied[bid, 3:6] = R @ M_a

    v_rel = R.T @ (v_world - wind_vec)
    if fg is not None:
        # elevon mixing: turn = differential, elevator = symmetric (7/1
        # keys). Key authorities sit inside the actuator range; combined
        # commands clamp per actuator. A sustained full roll still spirals
        # (rate-only roll damping, no roll spring) — pulse and center.
        fl = cfg["flaps"]
        d_turn = teleop.turn * np.deg2rad(fl["turn_deflect_deg"])
        d_elev = teleop.pitch * np.deg2rad(fl["elev_deflect_deg"])
        data.ctrl[0] = np.clip(d_turn + d_elev, -fg["max_rad"], fg["max_rad"])
        data.ctrl[1] = np.clip(-d_turn + d_elev, -fg["max_rad"], fg["max_rad"])
        for side, sgn in (("right", +1.0), ("left", -1.0)):
            theta = data.qpos[model.jnt_qposadr[
                model.joint(f"j_flap_{side}").id]]
            F_f, _ = flap_aero(v_rel, w_body, theta, fg, cfg, sgn)
            fbid = model.body(f"flap_{side}").id
            data.xfrc_applied[fbid, :3] = R @ F_f
            data.xfrc_applied[fbid, 3:6] = 0.0  # force at the flap com; the
            #   hinge transmits the moment — no double bookkeeping
    if bg is not None:
        # split airbrake: 9 toggles both plates between closed (flat) and
        # open (folded up into the flow) — pure drag, mounted at the CG x
        b_cmd = bg["max_rad"] if teleop.brake else 0.0
        data.ctrl[2] = data.ctrl[3] = b_cmd
        for side, sgn in (("left", -1.0), ("right", +1.0)):
            theta = data.qpos[model.jnt_qposadr[
                model.joint(f"j_brake_{side}").id]]
            F_b, _ = plate_aero(v_rel, w_body, theta,
                                dict(chord=bg["chord"], y_mid=bg["y_off"],
                                     area=bg["area"]),
                                cfg["brakes"], sgn)
            bbid = model.body(f"brake_{side}").id
            data.xfrc_applied[bbid, :3] = R @ F_b
            data.xfrc_applied[bbid, 3:6] = 0.0

    mujoco.mj_step(model, data)
    grounded = data.ncon > 0
    return info, grounded


# -- interactive run ---------------------------------------------------------
def run_interactive(model, data, geom, cfg, spawn, wind, shoot_speed=CART_SHOOT_SPEED):
    import glfw
    import mujoco
    import mujoco.viewer

    fg = flap_layout(geom, cfg)
    bg = brake_layout(geom, cfg)
    motor = CenterMotor(cfg, dt=cfg["sim"]["dt"])
    throttle0 = level_trim_throttle(geom, cfg, motor.t_max)
    teleop = Teleop(throttle0)
    reset_state(data, model, spawn, motor, teleop, throttle0)

    key_queue = []

    def key_callback(key):
        key_queue.append(key)

    # collision-box shuffle: C cycles the veh_col_* boxes, highlighting one
    # at a time (render color) and showing its name in an overlay figure
    col_ids = [g for g in range(model.ngeom)
               if (model.geom(g).name or "").startswith("veh_col")]
    # NOTE: the drone's box-collision cloud was REMOVED (user spec 09-20) --
    # the hull mesh carries all vehicle collision again, so col_ids is empty
    # and C reports "no boxes". cart_col_* boxes still exist on the cart.
    col_orig = {g: np.array(model.geom_rgba[g]) for g in col_ids}
    sel = {"i": -1}

    def select_collision(i):
        if 0 <= sel["i"] < len(col_ids):                 # restore previous
            model.geom_rgba[col_ids[sel["i"]]] = col_orig[col_ids[sel["i"]]]
        sel["i"] = i
        if i < 0:
            handle.set_figures([])
            print(f"[t={sim_t:6.1f}s] collision highlight cleared")
            return
        g = col_ids[i]
        model.geom_rgba[g] = [1.0, 0.15, 0.85, 1.0]      # bright magenta
        name = model.geom(g).name
        size = np.round(model.geom_size[g], 4).tolist()
        # the NAME itself gets a different color + font size: magenta text in
        # a TALL overlay rect (mjr_figure scales the font with rect height;
        # this binding exposes no textheight field)
        fig = mujoco.MjvFigure()
        fig.figurergba = [0.05, 0.02, 0.08, 0.78]
        fig.textrgb = [1.0, 0.2, 0.85]                   # magenta name text
        fig.flg_legend = 1
        fig.linename[0] = f"{name}  size(half)={size}"
        fig.linergb[0] = [1.0, 0.2, 0.85]
        fig.title = f"COLLISION {i + 1}/{len(col_ids)}  {name}"
        vp = handle.viewport
        handle.set_figures([(mujoco.MjrRect(10, max(vp.height - 110, 10),
                                            max(vp.width - 20, 100), 95), fig)])
        print(f"[t={sim_t:6.1f}s] COLLISION {i + 1}/{len(col_ids)}: {name} "
              f"size(half)={size}")

    print(__doc__.split("Keys (teleop)")[1].split("Flight-model")[0])
    print(f"spawn: alt {spawn['pos'][2]:.1f} m, speed {spawn['speed']:.1f} m/s, "
          f"motor {motor.t_max:.1f} N (trim throttle {throttle0:.2f}) | "
          f"wind {cfg_get_wind(wind)}")

    handle = None
    try:
        handle = mujoco.viewer.launch_passive(model, data,
                                              key_callback=key_callback)
        # tracking camera: follow the vehicle; the user still orbits/zooms
        handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        handle.cam.trackbodyid = model.body("vehicle").id

        sim_t, wall0, last_hud = 0.0, time.time(), 0.0
        last_frame = time.perf_counter()
        n_sub = max(1, int(round(0.01 / spawn["dt"])))   # 100 Hz physics chunks
        while handle.is_running():
            now_wall = time.perf_counter()
            for k in key_queue:
                if k == glfw.KEY_ESCAPE:
                    return
                if k in (glfw.KEY_KP_DECIMAL, glfw.KEY_PERIOD):
                    rdof = model.jnt_dofadr[model.joint("rail_joint").id]
                    data.qvel[rdof] = shoot_speed
                    print(f"[t={sim_t:6.1f}s] CART SHOT -> "
                          f"{shoot_speed:.0f} m/s along the rail")
                    continue
                if k == glfw.KEY_C:
                    if not col_ids:
                        print("[t] this model has no collision boxes to cycle")
                        continue
                    select_collision((sel["i"] + 1) % len(col_ids))
                    continue
                if k == glfw.KEY_V:
                    geom_l = mujoco.mjtLabel.mjLABEL_GEOM
                    none_l = mujoco.mjtLabel.mjLABEL_NONE
                    handle.opt.label = none_l if handle.opt.label == geom_l \
                        else geom_l
                    print(f"[t={sim_t:6.1f}s] geom name labels "
                          f"{'ON — read veh_col_*/cart_col_* names in the viewport'
                          if handle.opt.label == geom_l else 'OFF'}")
                    continue
                if k == glfw.KEY_R:
                    reset_state(data, model, spawn, motor, teleop, throttle0)
                    print(f"[t={sim_t:6.1f}s] RESET -> above the rail "
                          f"({spawn['pos'][0]:.2f}, 0, {spawn['pos'][2]:.1f} m), "
                          f"at rest, MOTOR OFF — press 8 to throttle up")
                elif k == glfw.KEY_Z:
                    reset_state(data, model, spawn, motor, teleop, throttle0,
                                zero=True, geom=geom, wind=wind)
                    print(f"[t={sim_t:6.1f}s] ZERO -> parked at origin, "
                          f"pos/vel/throttle 0 (motor pushes +x only: "
                          f"no takeoff from rest — R for a flying start)")
                else:
                    teleop.on_key(int(k), now_wall)
            key_queue.clear()
            teleop.update(min(now_wall - last_frame, 0.05), now_wall)
            last_frame = now_wall

            for _ in range(n_sub):
                info, grounded = physics_step(model, data, geom, cfg, motor,
                                              teleop, wind, fg, bg)
                sim_t += cfg["sim"]["dt"]

            handle.sync()

            if sim_t - last_hud > 0.25:
                last_hud = sim_t
                jnt = model.jnt_qposadr[model.joint("root").id]
                pos = data.qpos[jnt:jnt + 3]
                d_r = np.degrees(data.qpos[model.jnt_qposadr[
                    model.joint("j_flap_right").id]])
                d_l = np.degrees(data.qpos[model.jnt_qposadr[
                    model.joint("j_flap_left").id]])
                b_deg = np.degrees(data.qpos[model.jnt_qposadr[
                    model.joint("j_brake_left").id]])
                rj = model.jnt_qposadr[model.joint("rail_joint").id]
                rdof = model.jnt_dofadr[model.joint("rail_joint").id]
                print(f"t {sim_t:6.1f}s | alt {pos[2]:5.1f} m | V {info['V']:5.1f} m/s "
                      f"| a {np.degrees(info['alpha']):+5.1f}deg "
                      f"| thr {teleop.common:.2f} turn {teleop.turn:+.1f} "
                      f"elev {teleop.pitch:+.1f} "
                      f"flaps {d_r:+5.1f}/{d_l:+5.1f} brake {b_deg:4.0f}deg "
                      f"| cart s={data.qpos[rj]:4.2f} v={data.qvel[rdof]:+5.1f} "
                      f"| {'GROUNDED' if grounded else 'airborne'}")

            sleep = sim_t - (time.time() - wall0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        if handle is not None:
            handle.close()
        for g, rgba in col_orig.items():
            model.geom_rgba[g] = rgba
        print("sim ended")


def cfg_get_wind(wind):
    s = np.linalg.norm(wind.steady)
    return f"{s:.1f} m/s + turbulence sigma {wind.sigma:.1f}" if s else \
        f"calm + turbulence sigma {wind.sigma:.1f}"


def level_trim_throttle(geom, cfg, t_max):
    """Throttle holding LEVEL flight at cruise speed (lift = weight)."""
    a = cfg["aero"]
    q = 0.5 * a["rho"] * a["cruise_speed"] ** 2
    CL = geom.mass * 9.81 / (q * geom.S_ref)
    CD = geom.CD0 + geom.k_ind * CL ** 2
    return float(np.clip(q * geom.S_ref * CD / t_max, 0.05, 1.0))


# -- headless selftest -------------------------------------------------------
def run_selftest(model, data, geom, cfg, spawn, wind, seconds=14.0):
    fg = flap_layout(geom, cfg)
    bg = brake_layout(geom, cfg)
    motor = CenterMotor(cfg, dt=cfg["sim"]["dt"])
    # climb margin over the level trim: at exactly level thrust the launch
    # phugoid bleeds energy and the vehicle greases onto the ground
    throttle0 = min(1.0, level_trim_throttle(geom, cfg, motor.t_max) * 1.25)
    teleop = Teleop(throttle0)
    spawn["pos"] = [0.0, 0.0, 8.0]          # the flight scenario spawns in
    spawn["vel"] = [8.0, 0.0, 0.0]          # the air with the motor on, LEVEL
    spawn["quat"] = [1.0, 0.0, 0.0, 0.0]    # attitude — NOT the above-rail
    spawn["throttle"] = throttle0           # motor-off reset
    reset_state(data, model, spawn, motor, teleop, throttle0)

    # borrowed-rail check: shoot the cart, it must travel up its stroke
    import mujoco
    rdof = model.jnt_dofadr[model.joint("rail_joint").id]
    rj = model.jnt_qposadr[model.joint("rail_joint").id]
    data.qvel[rdof] = CART_SHOOT_SPEED
    for _ in range(50):
        mujoco.mj_step(model, data)
    s_travel = float(data.qpos[rj])
    assert s_travel > 0.3, f"cart did not move after the shot: {s_travel:.2f} m"
    print(f"rail: cart shot -> travelled {s_travel:.2f} m along its stroke")
    reset_state(data, model, spawn, motor, teleop, throttle0)
    print(f"selftest: {seconds:.0f}s at throttle {throttle0:.3f} "
          f"(level trim {level_trim_throttle(geom, cfg, motor.t_max):.3f} "
          f"x1.25, motor {motor.t_max:.1f} N), wind {cfg_get_wind(wind)}; "
          f"flap pulse at 40% then center")

    jnt = model.jnt_qposadr[model.joint("root").id]
    log, t, last_print = [], 0.0, -1.0
    n_sub = max(1, int(round(0.01 / spawn["dt"])))
    # scenario: straight cruise -> elevator-assisted banked turn (pitch hold
    # against the turn's dive) -> airbrake slowdown segment -> glide out
    turn_pulse = (0.4 * seconds, 0.4 * seconds + 1.2)
    brake_on = 0.7 * seconds
    brake_off = min(seconds - 1.0, brake_on + 2.5)
    v_before_brake, v_min_brake = None, None
    while t < seconds:
        in_turn = turn_pulse[0] <= t < turn_pulse[1]
        teleop.turn = 0.6 if in_turn else 0.0
        teleop.pitch = 0.25 if in_turn else 0.0     # hold the nose in the turn
        teleop.common = min(1.0, throttle0 + (0.24 if in_turn else 0.0))
        if in_turn or t >= turn_pulse[1]:
            teleop.brake = brake_on <= t < brake_off
        for _ in range(n_sub):
            info, grounded = physics_step(model, data, geom, cfg, motor,
                                          teleop, wind, fg, bg)
            t += cfg["sim"]["dt"]
        pos = data.qpos[jnt:jnt + 3]
        airborne_now = pos[2] > 1.0
        if abs(t - (brake_on - 0.1)) < 0.05 and airborne_now:
            v_before_brake = info["V"]
        if brake_on <= t < brake_off and airborne_now:
            v_min_brake = info["V"] if v_min_brake is None \
                else min(v_min_brake, info["V"])
        d_r = np.degrees(data.qpos[model.jnt_qposadr[
            model.joint("j_flap_right").id]])
        log.append((t, *pos, info["V"], np.degrees(info["alpha"])))
        if not np.all(np.isfinite(data.qpos)):
            raise AssertionError(f"state blew up at t={t:.1f}s")
        if t - last_print >= 0.7:
            last_print = t
            print(f"  t {t:5.1f}s pos ({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:5.1f}) m "
                  f"V {info['V']:5.1f} a {np.degrees(info['alpha']):+5.1f}deg "
                  f"flaps {d_r:+5.1f}deg {'GROUND' if grounded else ''}")

    airborne = [(r[0], r[3]) for r in log if r[3] > 1.0]
    t_air = airborne[-1][0] if airborne else 0.0
    x_end = log[-1][1]
    # no pitch actuator: banked turns descend to a landing — require a real
    # flight (most of the episode) plus the demonstrated carve, not endless
    # flight
    assert t_air >= 0.75 * seconds, f"airborne only {t_air:.1f}s"
    assert x_end >= 20.0, f"no forward progress: x={x_end:.1f} m"
    y_turn = float(pos[1])
    assert y_turn > 1.5, (f"flap turn went the wrong way or not at all: "
                          f"y={y_turn:.2f} m after the flap pulse")
    assert v_before_brake is not None and v_min_brake is not None, \
        "brake segment never ran"
    drop = v_before_brake - v_min_brake
    assert drop >= 0.5, (f"airbrake did not slow the vehicle: "
                         f"{v_before_brake:.2f} -> {v_min_brake:.2f} m/s")
    assert np.isfinite(data.qpos).all() and np.abs(data.qvel).max() < 50
    render_preview(model, data)
    print(f"SELFTEST PASS: airborne {t_air:.1f}s, flew {x_end:.0f} m, "
          f"turn +y {y_turn:.1f} m, airbrake slowed "
          f"{v_before_brake:.1f} -> {v_min_brake:.1f} m/s")


def render_preview(model, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    r = mujoco.Renderer(model, height=540, width=960)
    views = [(az, el) for az, el in [(130, -18), (90, -8)]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
    for ax, (az, el) in zip(axes, views):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation = az, el
        bid = model.body("vehicle").id
        cam.lookat[:] = data.xpos[bid]
        cam.distance = 3.0
        r.update_scene(data, camera=cam)
        ax.imshow(r.render())
        ax.set_title(f"az {az} el {el}"); ax.axis("off")
    out = base_dir().parent / "results" / "veh3d" / "sim_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); r.close()
    print(f"preview -> {out}")


# -- main --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="veh3d interactive flight sim (MuJoCo, CPU render path)")
    p.add_argument("--design", default=str(base_dir() / "design.yaml"))
    p.add_argument("--set", action="append", default=[],
                   help="dotted.path=value design override, repeatable")
    p.add_argument("--wind", type=float, default=0.0, help="steady wind [m/s]")
    p.add_argument("--wind-dir-deg", type=float, default=0.0,
                   help="direction the wind blows TOWARD, world +x = 0")
    p.add_argument("--turb", type=float, default=0.0,
                   help="turbulence intensity (OU sigma, m/s)")
    p.add_argument("--alt", type=float, default=1.3,
                   help="reset spawn altitude [m] (default: above the rail)")
    p.add_argument("--shoot-speed", type=float, default=CART_SHOOT_SPEED,
                   help="cart launch speed [m/s] for the KP . key")
    p.add_argument("--pitch-deg", type=float, default=12.0,
                   help="reset spawn pitch [deg] about y: positive = NOSE UP "
                        "(12 = aligned with the 11.5 deg launch rail)")
    p.add_argument("--speed", type=float, default=None,
                   help="spawn speed [m/s] (default: cruise)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=12.0,
                   help="selftest rollout duration")
    p.add_argument("--collision-xml", default=None, metavar="PATH",
                   help="collision-box source scene (default: the blend "
                        "export uav_rail_scene_typed.xml)")
    p.add_argument("--tree", action="store_true",
                   help="print the model tree (bodies -> geoms with names, "
                        "types, sizes, collision flags) and exit")
    p.add_argument("--selftest", action="store_true",
                   help="headless: 12 s physics rollout + EGL preview png")
    args = p.parse_args()

    import os
    if args.selftest:
        os.environ.setdefault("MUJOCO_GL", "egl")   # headless GL for the preview

    import yaml
    cfg = overrides_to_cfg(load_design(args.design), parse_sets(args.set))
    cfg["flight"] = cfg.get("flight") or {"cm_alpha": 0.40, "cm_q": 1.2,
                                          "cl_p": 0.50}
    if args.collision_xml:
        cfg["flight"]["collision_xml"] = args.collision_xml
    cfg["flight"].setdefault("collision_xml", default_collision_source())
    print(f"collision source: {cfg['flight']['collision_xml']}")
    from geometry import feasibility
    geom = build(cfg)
    ok, reasons = feasibility(geom)
    print(f"python {sys.executable}")
    print(f"model: mass {geom.mass:.3f} kg S {geom.S_ref:.3f} m2 AR {geom.AR:.2f} "
          f"CD0 {geom.CD0:.3f} | cruise lift margin "
          f"{geom.aero['lift'] / geom.aero['weight']:.2f}x"
          + ("" if ok else f" | WARNING: {reasons[0]}"))
    if not geom.polar_ok:
        # never fly NaN: a failed NeuralFoil polar poisoned cl_c -> every aero
        # force is NaN -> MuJoCo diverges and keys LOOK dead while the vehicle
        # thrashes in auto-reset. This is almost always the WRONG PYTHON.
        sys.exit(
            "FATAL: section polar failed — NeuralFoil unavailable or "
            "diverged.\n"
            f"  interpreter: {sys.executable}\n"
            "  Fix: source the aero env first:\n"
            "    cd /home/heinz/isaaclab_uav/aero && source env.sh\n"
            "  then rerun. (env_isaaclab has mujoco but NO neuralfoil.)")

    speed0 = args.speed if args.speed is not None else 0.0
    rad = math.radians(args.pitch_deg)          # user input, deg
    spawn = dict(pos=[-0.47, 0.0, args.alt],
                 quat=[math.cos(rad / 2), 0.0, -math.sin(rad / 2), 0.0],
                 vel=[speed0, 0.0, 0.0],
                 dt=min(cfg["sim"]["dt"], 0.002), speed=speed0,
                 throttle=0.0)   # reset spawns ABOVE THE RAIL, motor off,
                                 # pitched nose-up by --pitch-deg
    cfg["sim"]["dt"] = spawn["dt"]                  # one dt everywhere
    wind = Wind(args.wind, args.wind_dir_deg, args.turb,
                np.random.default_rng(args.seed))
    model = make_model(geom, cfg, spawn)
    import mujoco
    data = mujoco.MjData(model)

    if args.tree:
        print_model_tree(model)
        return

    if args.selftest:
        run_selftest(model, data, geom, cfg, spawn, wind, seconds=args.seconds)
    else:
        run_interactive(model, data, geom, cfg, spawn, wind,
                        shoot_speed=args.shoot_speed)


def print_model_tree(model):
    """Terminal model tree: bodies -> geoms (name, type, size, collision)."""
    import mujoco
    types = {int(v): k.replace("mjGEOM_", "").lower()
             for k, v in vars(mujoco.mjtGeom).items() if k.startswith("mjGEOM_")}
    for b in range(model.nbody):
        depth, p = 0, model.body_parentid[b]
        while p != b and p != 0:
            depth += 1
            p = model.body_parentid[p]
        print("  " * depth + f"body {model.body(b).name}")
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b:
                continue
            sz = np.round(model.geom_size[g], 4).tolist()
            col = "collide" if (model.geom_contype[g] or
                                model.geom_conaffinity[g]) else "visual"
            print("  " * (depth + 1)
                  + f"geom {model.geom(g).name or '(unnamed)'}: "
                  f"{types.get(int(model.geom_type[g]), '?')} size={sz} [{col}]")


def parse_sets(items):
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


if __name__ == "__main__":
    main()

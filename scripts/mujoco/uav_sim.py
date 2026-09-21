#!/usr/bin/env python3
"""
MuJoCo UAV Aerodynamics Simulation with Keyboard Controls.

A fixed-wing UAV (Shahed-136 model) flies in a MuJoCo physics environment
with ground plane, aerodynamic forces (drag, lift, rotational damping), and
a static rail object.

Controls
--------
  0 / 5  Apply force @ COM (+Z) / nose (+X)        (numpad OK)
  4 / 6  Left / right flap turn                     (numpad OK)
  7 / 9  Flap magnitude ↓ / ↑                       (numpad OK)
  + / -  Force multiplier ↑ / ↓                     (numpad OK)
  L / Space  Rail launch: scripted catapult stroke, releases
             the UAV at ~11 m/s at the end of the rail
  R      Reset UAV, gentle auto push after 0.75 s (drone stays hooked)
  V      Hide / show visual mesh hulls (only collision prims)
  T      Log shahed + cart positions (world), once per press
  ESC    Exit
  ------  built-in MuJoCo viewer shortcuts ------
  Tab    Toggle info overlay
  F1     Toggle left  UI panel   (for more view space)
  F2     Toggle right UI panel   (for more view space)

Aerodynamic Model
-----------------
  - Quadratic body-axis drag with per-axis coefficients
  - Lift proportional to forward velocity squared in body +Z
  - Rotational damping (angular velocity damping in body frame)
  - Flap forces: asymmetric vertical forces at wing positions
    (-1, ±1, 0) body-frame, proportional to dynamic pressure,
    flap deflection, and flap magnitude setting
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import glfw as _glfw
import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_XML_PATH = str(_SCRIPT_DIR / "uav_rail_scene.xml")

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """All tunable simulation parameters gathered from CLI arguments."""

    # -- model ----------------------------------------------------------------
    xml_path: str = _DEFAULT_XML_PATH

    # -- initial pose ---------------------------------------------------------
    init_pos: Tuple[float, float, float] = (-0.22, 0.0, 1.43)
    init_pitch_deg: float = 12.8

    # -- physics --------------------------------------------------------------
    timestep: float = 0.005
    gravity: float = 9.81  # m/s² in -Z

    # -- key forces -----------------------------------------------------------
    force_up_z: float = 8000.0   # N, key 0 — +Z at COM
    force_nose_x: float = 2000.0 # N, key 5 — +X at nose
    nose_offset: Tuple[float, float, float] = (-0.8, 0.0, 0.0)

    # -- aerodynamics (body-frame) --------------------------------------------
    k_drag: Tuple[float, float, float] = (5.0, 30.0, 60.0)
    k_lift: float = 40.0
    k_rot_damp: Tuple[float, float, float] = (300.0, 300.0, 150.0)

    # -- flaps ----------------------------------------------------------------
    flap_k: float = 40.0                        # flap force coefficient
    flap_init_magnitude: float = 1.0            # initial flap magnitude (keys 7/9)
    flap_pos_right: Tuple[float, float, float] = (-1.0, 1.0, 0.0)   # body frame
    flap_pos_left: Tuple[float, float, float] = (-1.0, -1.0, 0.0)   # body frame

    # -- force scaling --------------------------------------------------------
    force_init_multiplier: float = 1.0          # initial key-force multiplier (+/-)

    # -- rail launch ----------------------------------------------------------
    # Force sizing: lift = k_lift * v_x^2 must exceed the 1962 N weight, so
    # v_release >= sqrt(1962 / k_lift) = 7.0 m/s.  The cradle weld makes the
    # cart + UAV accelerate as one 205 kg mass over the 2 m stroke, and
    # gravity eats ~400 N along the 11.5 deg slope:
    #   F = m * (v^2 / (2 * stroke) + g * sin(11.5 deg))
    #     5500 N -> ~10.5 m/s,  6500 N -> ~11.5 m/s at release.
    # NOTE: the motor ctrl is clamped to the XML ctrlrange; forces beyond it
    # are silently truncated (the old 90 MN command only ever delivered 500 N).
    rail_force_manual: float = 6500.0           # N, L / Space impulse
    rail_force_reset: float = 5500.0            # N, auto push motor assist
    rail_impulse_duration_manual: float = 1.0   # s, L / Space impulse
    rail_impulse_duration_reset: float = 0.5    # s, auto impulse (reset / start)
    rail_release_q: float = 1.9                 # m, cart travel that ends the launch
    rail_cradle_gap: float = 1.0                # m, max UAV-cart gap to re-cradle
    # Kinematic launch: the stroke velocity is scripted (v = a*t) instead of
    # force-driven, so nothing in the contact solver can absorb the launch.
    # rail_force_* only back the motor visually / hold the brake; the exit
    # speed is what matters.
    rail_exit_speed: float = 11.0               # m/s at release (lift ~2.4x weight)
    rail_auto_exit_speed: float = 1.0           # m/s, gentle auto push (reset/start)
    rail_auto_accel: float = 3.0                # m/s², gentle auto push accel

    # -- viewer ---------------------------------------------------------------
    hide_ui: bool = True                        # start with UI panels hidden
    hide_visual: bool = True                    # hide mesh hulls; show collision prims

    @property
    def init_pitch_rad(self) -> float:
        return math.radians(self.init_pitch_deg)

    @property
    def init_quat(self) -> np.ndarray:
        """[qw, qx, qy, qz] for roll=0, pitch, yaw=0."""
        cp2 = math.cos(self.init_pitch_rad / 2.0)
        sp2 = math.sin(self.init_pitch_rad / 2.0)
        return np.array([cp2, 0.0, sp2, 0.0])

    @property
    def init_qpos(self) -> np.ndarray:
        """Full qpos vector: [x, y, z, qw, qx, qy, qz]."""
        q = self.init_quat
        return np.array([
            self.init_pos[0], self.init_pos[1], self.init_pos[2],
            q[0], q[1], q[2], q[3],
        ])


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="uav_sim.py",
        description="MuJoCo fixed-wing UAV simulation with keyboard controls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Default settings
  python uav_sim.py

  # Custom initial pose (z=5 m, pitch=5°)
  python uav_sim.py --pos-z 5.0 --pitch 5.0

  # Stronger lift boost, weaker nose force
  python uav_sim.py --force-up 12000 --force-nose 1000

  # Different aerodynamic profile (high drag)
  python uav_sim.py --drag 10 50 100 --lift 20

  # Double the physics rate
  python uav_sim.py --timestep 0.0025

Controls in-sim
---------------
  0/5     apply force @ COM / nose      (numpad OK)
  4/6     left / right flap turn        (numpad OK)
  7/9     flap magnitude ↓/↑            (numpad OK)
  +/-     force multiplier ↑/↓          (numpad OK)
  L/Space rail launch (scripted catapult, ~11 m/s exit)
  R       reset pose, gentle auto push (drone stays hooked on cart)
  V       hide/show visual meshes       (only collision prims)
  T       log shahed + cart positions   (world, once per press)
  F1      toggle left  UI panel         (built-in viewer)
  F2      toggle right UI panel         (built-in viewer)
  Tab     toggle info overlay           (built-in viewer)
  ESC     exit
""",
    )

    # ---- model ------------------------------------------------------------
    p.add_argument(
        "--model", metavar="PATH", default=_DEFAULT_XML_PATH,
        help="Path to MJCF model XML (default: %(default)s)",
    )

    # ---- initial pose -----------------------------------------------------
    g_pose = p.add_argument_group("Initial Pose")
    g_pose.add_argument("--pos-x", type=float, default=None, metavar="X",
                        help="Initial X position [m]")
    g_pose.add_argument("--pos-y", type=float, default=None, metavar="Y",
                        help="Initial Y position [m]")
    g_pose.add_argument("--pos-z", type=float, default=None, metavar="Z",
                        help="Initial Z position [m]")
    g_pose.add_argument("--pitch", type=float, default=None, metavar="DEG",
                        help="Initial pitch angle [deg]")

    # ---- physics ----------------------------------------------------------
    g_phys = p.add_argument_group("Physics")
    g_phys.add_argument("--timestep", type=float, default=0.005, metavar="DT",
                        help="Simulation timestep [s] (default: 0.005)")
    g_phys.add_argument("--gravity", type=float, default=9.81, metavar="G",
                        help="Gravity magnitude [m/s²] in -Z (default: 9.81)")

    # ---- key forces -------------------------------------------------------
    g_forces = p.add_argument_group("Key Forces")
    g_forces.add_argument("--force-up", type=float, default=8000.0, metavar="N",
                          help="Key-0 +Z force at COM [N] (default: 8000)")
    g_forces.add_argument("--force-nose", type=float, default=2000.0, metavar="N",
                          help="Key-5 +X force at nose [N] (default: 2000)")

    # ---- aerodynamics -----------------------------------------------------
    g_aero = p.add_argument_group("Aerodynamics (body-frame)")
    g_aero.add_argument(
        "--drag", type=float, nargs=3, default=[5.0, 30.0, 60.0],
        metavar=("KX", "KY", "KZ"),
        help="Quadratic drag coefficients k_drag_%%(metavar)s "
             "(default: 5 30 60).  F_i = -k_i * v_i * |v_i|",
    )
    g_aero.add_argument(
        "--lift", type=float, default=40.0, metavar="KL",
        help="Lift coefficient (default: 40).  F_z_body += KL * v_x * |v_x|",
    )
    g_aero.add_argument(
        "--rot-damp", type=float, nargs=3, default=[300.0, 300.0, 150.0],
        metavar=("KRX", "KRY", "KRZ"),
        help="Rotational damping coefficients (default: 300 300 150).  "
             "tau_i = -k_i * omega_i",
    )

    # ---- flaps -------------------------------------------------------------
    g_flaps = p.add_argument_group("Flaps")
    g_flaps.add_argument(
        "--flap-k", type=float, default=40.0, metavar="KF",
        help="Flap force coefficient (default: 40).  "
             "F_flap = KF * mag * v_x * |v_x|",
    )
    g_flaps.add_argument(
        "--flap-mag", type=float, default=1.0, metavar="M",
        help="Initial flap magnitude (default: 1.0).  Adjust in-sim with 7/9.",
    )

    # ---- force scaling -----------------------------------------------------
    g_fscale = p.add_argument_group("Force Scaling")
    g_fscale.add_argument(
        "--force-mult", type=float, default=1.0, metavar="FM",
        help="Initial key-force multiplier (default: 1.0).  Adjust in-sim with +/-.",
    )

    # ---- rail launch -------------------------------------------------------
    g_rail = p.add_argument_group("Rail Launch")
    g_rail.add_argument(
        "--rail-force", type=float, default=6500.0, metavar="N",
        help="Manual rail impulse force [N] along joint + direction "
             "(default: 6500).  Triggered by keys L / Space.",
    )
    g_rail.add_argument(
        "--rail-force-reset", type=float, default=5500.0, metavar="N",
        help="Auto rail impulse force [N] fired after reset and at sim start "
             "(default: 5500).",
    )
    g_rail.add_argument(
        "--rail-duration", type=float, default=1.0, metavar="S",
        help="Manual rail impulse duration [s] (default: 1.0).  "
             "Triggered by keys L / Space.",
    )
    g_rail.add_argument(
        "--rail-duration-reset", type=float, default=0.5, metavar="S",
        help="Auto rail impulse duration [s] fired after reset and at sim "
             "start (default: 0.5).",
    )
    g_rail.add_argument(
        "--rail-exit-speed", type=float, default=11.0, metavar="V",
        help="Scripted exit speed at rail release [m/s] (default: 11). "
             "Takeoff needs >= 7 m/s with the default lift coefficient.",
    )
    g_rail.add_argument(
        "--rail-auto-exit-speed", type=float, default=1.0, metavar="V",
        help="Gentle auto push speed after reset / at sim start [m/s] "
             "(default: 1.0).  A nudge, not a launch.",
    )

    # ---- viewer ------------------------------------------------------------
    g_view = p.add_argument_group("Viewer")
    g_view.add_argument(
        "--show-ui", action="store_true", default=False,
        help="Show MuJoCo UI panels at start.  "
             "(They are hidden by default; press Tab/F1/F2 to toggle.)",
    )
    g_view.add_argument(
        "--show-visual", action="store_true", default=False,
        help="Show the visual mesh hulls at start.  "
             "(They are hidden by default so only the collision primitives "
             "render; press V in-sim to toggle.)",
    )

    return p


def _args_to_config(args: argparse.Namespace) -> SimConfig:
    """Convert parsed CLI args to a SimConfig instance.

    Uses SimConfig defaults as fallback; CLI arguments only override
    when explicitly provided (i.e. not None).  This way editing
    ``init_pos`` / ``init_pitch_deg`` in the SimConfig dataclass is
    sufficient — no need to keep argparse defaults in sync.
    """
    defaults = SimConfig()
    return SimConfig(
        xml_path=args.model,
        init_pos=(
            args.pos_x if args.pos_x is not None else defaults.init_pos[0],
            args.pos_y if args.pos_y is not None else defaults.init_pos[1],
            args.pos_z if args.pos_z is not None else defaults.init_pos[2],
        ),
        init_pitch_deg=(args.pitch if args.pitch is not None
                        else defaults.init_pitch_deg),
        timestep=args.timestep,
        gravity=args.gravity,
        force_up_z=args.force_up,
        force_nose_x=args.force_nose,
        k_drag=tuple(args.drag),
        k_lift=args.lift,
        k_rot_damp=tuple(args.rot_damp),
        flap_k=args.flap_k,
        flap_init_magnitude=args.flap_mag,
        force_init_multiplier=args.force_mult,
        rail_force_manual=args.rail_force,
        rail_force_reset=args.rail_force_reset,
        rail_impulse_duration_manual=args.rail_duration,
        rail_impulse_duration_reset=args.rail_duration_reset,
        rail_exit_speed=args.rail_exit_speed,
        rail_auto_exit_speed=args.rail_auto_exit_speed,
        hide_ui=not args.show_ui,
        hide_visual=not args.show_visual,
    )


# ---------------------------------------------------------------------------
# Key Queue  (thread-safe bridge between viewer thread and physics thread)
# ---------------------------------------------------------------------------
_key_queue: queue.Queue[int] = queue.Queue()

# ---------------------------------------------------------------------------
# Saved Initial State  (for reset)
# ---------------------------------------------------------------------------
_initial_qpos: np.ndarray | None = None
_initial_qvel: np.ndarray | None = None

# ---------------------------------------------------------------------------
# Runtime config (set in main())
# ---------------------------------------------------------------------------
_cfg: Optional[SimConfig] = None

# ---------------------------------------------------------------------------
# Runtime mutable state  (adjusted by key presses during simulation)
# ---------------------------------------------------------------------------
_flap_mode: str = "neutral"       # "neutral" | "left" | "right"
_flap_magnitude: float = 1.0      # flap effect strength (keys 7 / 9)
_force_multiplier: float = 1.0    # key-force scaling (keys + / -)

_FLAP_MAG_STEP: float = 0.25      # increment per key-7/9 press
_FORCE_MULT_STEP: float = 0.5     # increment per key-+/- press
_FLAP_MAG_MIN: float = 0.0
_FLAP_MAG_MAX: float = 10.0
_FORCE_MULT_MIN: float = 0.0
_FORCE_MULT_MAX: float = 20.0

# Rail impulse state
_rail_impulse_remaining: int = 0  # steps remaining in current impulse
_rail_impulse_force: float = 0.0  # N applied by the current impulse (manual or auto)
_rail_hold_position: float = 0.0   # q-position to servo to when idle
_RAIL_HOLD_KP: float = 5000.0      # position-servo gain [N/m] for cart brake

# Resolved once in main(): rail motor actuator id and cradle weld equality id
_rail_act_id: int = -1
_rail_eq_id: int = -1
_rail_qposadr: int = 0            # rail joint qpos address
_rail_dofadr: int = 0             # rail joint dof address
_rail_axis_w: np.ndarray = np.array([1.0, 0.0, 0.0])  # rail axis, world frame
_rail_launch_t: float = 0.0        # seconds since the current stroke started
_rail_launch_speed: float = 0.0    # target speed of the current stroke [m/s]
_rail_launch_accel: float = 0.0    # accel profile of the current stroke [m/s²]
_rail_carry_uav: bool = True       # carry the UAV by gap check if weld is dead

# Post-reset auto rail impulse: after key R, wait this long (sim time), then
# fire the same rail impulse as L / Space.
_RESET_IMPULSE_DELAY: float = 0.75  # seconds to hold after reset
_reset_hold_remaining: int = 0     # steps left before the gentle auto push fires

# Visual-mesh visibility (key V)
_viewer_handle = None                 # mujoco.viewer.Handle, set in main()
_visual_hidden: bool = True           # current visibility state
_VISUAL_GROUP: int = 1                # group the mesh hulls are moved into


def _apply_visual_visibility(
    model: mujoco.MjModel,
    handle,
    hide: bool,
) -> bool:
    """Render only the collision primitives: move every mesh geom (the visual
    hulls) into ``_VISUAL_GROUP`` and toggle that group's viewer flag.

    Returns the visibility state actually applied.  In scenes where the meshes
    ARE the collision hulls (e.g. uav_rail_scene.xml — every object geom is a
    mesh, only the ground is a plane) hiding them would blank the viewer, so a
    hide request there is ignored and the meshes stay visible.  The feature
    only truly hides in scenes that have separate collision prims
    (uav_rail_scene_prim.xml: boxes/cylinders stay in group 0).
    Model + viewer option are guarded by the viewer lock (render thread)."""
    if hide:
        non_mesh = (int(mujoco.mjtGeom.mjGEOM_MESH),
                    int(mujoco.mjtGeom.mjGEOM_PLANE))
        prim_ids = [g for g in range(model.ngeom)
                    if int(model.geom_type[g]) not in non_mesh]
        if not prim_ids:
            _log("V      |  scene has no separate collision prims — "
                 "meshes must stay visible")
            return False
    mesh_ids = [g for g in range(model.ngeom)
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    for g in mesh_ids:
        model.geom_group[g] = _VISUAL_GROUP
    if handle is not None:
        with handle.lock():
            handle.opt.geomgroup[_VISUAL_GROUP] = 0 if hide else 1
    return hide

# ---------------------------------------------------------------------------
# Key-group constants  (top-row + numpad for each action key)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Key-group constants  (top-row + numpad for each action key)
# ---------------------------------------------------------------------------
_KEY_0 = frozenset({_glfw.KEY_0, _glfw.KEY_KP_0})
_KEY_4 = frozenset({_glfw.KEY_4, _glfw.KEY_KP_4})
_KEY_5 = frozenset({_glfw.KEY_5, _glfw.KEY_KP_5})
_KEY_6 = frozenset({_glfw.KEY_6, _glfw.KEY_KP_6})
_KEY_7 = frozenset({_glfw.KEY_7, _glfw.KEY_KP_7})
_KEY_9 = frozenset({_glfw.KEY_9, _glfw.KEY_KP_9})
_KEY_R = frozenset({_glfw.KEY_R})
_KEY_V = frozenset({_glfw.KEY_V})
_KEY_T = frozenset({_glfw.KEY_T})
_KEY_L = frozenset({_glfw.KEY_L})
_KEY_SPACE = frozenset({_glfw.KEY_SPACE})
_KEY_PLUS  = frozenset({_glfw.KEY_EQUAL, _glfw.KEY_KP_ADD})
_KEY_MINUS = frozenset({_glfw.KEY_MINUS, _glfw.KEY_KP_SUBTRACT})

# Keys that the MuJoCo viewer handles natively (skip in our handler):
#   Tab  — toggle info overlay          F1  — toggle left  UI panel
#   F2   — toggle right UI panel
#   Space is NOT here so our handler can use it for rail impulse.
_VIEWER_KEYS = frozenset({
    _glfw.KEY_TAB,
    _glfw.KEY_F1,
    _glfw.KEY_F2,
    _glfw.KEY_ESCAPE,
})

# ---------------------------------------------------------------------------
# Aerodynamic Passive-Force Callback
# ---------------------------------------------------------------------------


def _aero_passive_callback(
    model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """Compute aerodynamic forces (drag, lift, rot-damp, flaps) + add to qfrc_passive."""
    cfg = _cfg
    assert cfg is not None

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
    if body_id < 0:
        return

    # Body-frame velocities (use joint DOF address — not hardcoded)
    jnt_adr = model.body_dofadr[body_id]
    v_world = data.qvel[jnt_adr : jnt_adr + 3].copy()
    w_body  = data.qvel[jnt_adr + 3 : jnt_adr + 6].copy()

    body_quat  = data.qpos[jnt_adr + 3 : jnt_adr + 7].copy()
    body_quatc = np.array([body_quat[0], -body_quat[1],
                           -body_quat[2], -body_quat[3]])
    v_body = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(v_body, v_world, body_quatc)

    # ---- drag ---------------------------------------------------------------
    force_body = np.zeros(3, dtype=np.float64)
    for i in range(3):
        force_body[i] = -cfg.k_drag[i] * v_body[i] * abs(v_body[i])

    # ---- lift ---------------------------------------------------------------
    force_body[2] += cfg.k_lift * v_body[0] * abs(v_body[0])

    # ---- rotational damping -------------------------------------------------
    torque_body = np.zeros(3, dtype=np.float64)
    for i in range(3):
        torque_body[i] = -cfg.k_rot_damp[i] * w_body[i]

    # ---- flap forces --------------------------------------------------------
    # Flaps generate vertical (body ±Z) forces at offset positions.
    # Dynamic pressure ~ v_x * |v_x|.
    # Left turn  (key 4): right flap ↓, left flap ↑  →  roll left
    # Right turn (key 6): right flap ↑, left flap ↓  →  roll right
    if _flap_mode != "neutral":
        q_dyn = v_body[0] * abs(v_body[0])  # forward dynamic pressure
        if q_dyn > 0.01:  # only apply at meaningful forward speed
            if _flap_mode == "left":
                sign_right = -1.0  # right flap down
                sign_left  = +1.0  # left  flap up
            else:  # "right"
                sign_right = +1.0
                sign_left  = -1.0

            f_mag = cfg.flap_k * _flap_magnitude * q_dyn

            for sign, pos in [(sign_right, cfg.flap_pos_right),
                              (sign_left,  cfg.flap_pos_left)]:
                f_body = np.array([0.0, 0.0, sign * f_mag], dtype=np.float64)
                tau_flap_body = np.cross(pos, f_body)

                force_body += f_body
                torque_body += tau_flap_body

    # ---- rotate body-frame force / torque → world frame ---------------------
    force_world = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(force_world, force_body, body_quat)
    torque_world = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(torque_world, torque_body, body_quat)

    # ---- add to generalized passive forces ----------------------------------
    data.qfrc_passive[jnt_adr : jnt_adr + 3] += force_world
    data.qfrc_passive[jnt_adr + 3 : jnt_adr + 6] += torque_world


# ---------------------------------------------------------------------------
# Key Callback  (runs in viewer / GLFW thread)
# ---------------------------------------------------------------------------


def _key_callback(key: int) -> None:
    """Enqueue key-code for processing by the physics thread."""
    _key_queue.put(key)


# ---------------------------------------------------------------------------
# Key-Action Handler  (runs in physics thread)
# ---------------------------------------------------------------------------


def _apply_key_action(model: mujoco.MjModel, data: mujoco.MjData, key: int) -> None:
    global _flap_mode, _flap_magnitude, _force_multiplier
    global _rail_impulse_remaining, _rail_hold_position
    global _visual_hidden, _reset_hold_remaining
    global _rail_impulse_force, _rail_launch_t
    global _rail_launch_speed, _rail_launch_accel
    cfg = _cfg
    assert cfg is not None

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
    if body_id < 0:
        return

    # Don't process keys the viewer handles natively (Tab, F1, F2, Space …).
    # These are consumed by the viewer's own GLFW callback.
    if key in _VIEWER_KEYS:
        return

    # ---- force application --------------------------------------------------

    if key in _KEY_0:
        fz = cfg.force_up_z * _force_multiplier
        data.xfrc_applied[body_id, 2] += fz
        _log(f"KEY 0  |  +{fz:.0f} N  +Z @ COM  (×{_force_multiplier:.1f})")

    elif key in _KEY_5:
        fx = cfg.force_nose_x * _force_multiplier
        _apply_force_at_body_point(
            model, data, body_id,
            force_world=np.array([fx, 0.0, 0.0]),
            point_body=np.array(cfg.nose_offset),
        )
        _log(f"KEY 5  |  +{fx:.0f} N  +X @ nose {cfg.nose_offset}  (×{_force_multiplier:.1f})")

    # ---- flap controls ------------------------------------------------------

    elif key in _KEY_4:
        if _flap_mode == "left":
            _flap_mode = "neutral"
            _log("KEY 4  |  Flaps OFF  (neutral)")
        else:
            _flap_mode = "left"
            _log(f"KEY 4  |  Flaps LEFT turn  (mag={_flap_magnitude:.2f})")

    elif key in _KEY_6:
        if _flap_mode == "right":
            _flap_mode = "neutral"
            _log("KEY 6  |  Flaps OFF  (neutral)")
        else:
            _flap_mode = "right"
            _log(f"KEY 6  |  Flaps RIGHT turn  (mag={_flap_magnitude:.2f})")

    # ---- flap magnitude -----------------------------------------------------

    elif key in _KEY_9:
        _flap_magnitude = min(_flap_magnitude + _FLAP_MAG_STEP, _FLAP_MAG_MAX)
        _log(f"KEY 9  |  Flap magnitude ↑  →  {_flap_magnitude:.2f} "
             f"(range [{_FLAP_MAG_MIN:.1f}–{_FLAP_MAG_MAX:.1f}])")

    elif key in _KEY_7:
        _flap_magnitude = max(_flap_magnitude - _FLAP_MAG_STEP, _FLAP_MAG_MIN)
        _log(f"KEY 7  |  Flap magnitude ↓  →  {_flap_magnitude:.2f} "
             f"(range [{_FLAP_MAG_MIN:.1f}–{_FLAP_MAG_MAX:.1f}])")

    # ---- force multiplier ---------------------------------------------------

    elif key in _KEY_PLUS:
        _force_multiplier = min(_force_multiplier + _FORCE_MULT_STEP, _FORCE_MULT_MAX)
        _log(f"KEY +  |  Force multiplier ↑  →  ×{_force_multiplier:.1f} "
             f"(range [×{_FORCE_MULT_MIN:.1f}–×{_FORCE_MULT_MAX:.1f}])")

    elif key in _KEY_MINUS:
        _force_multiplier = max(_force_multiplier - _FORCE_MULT_STEP, _FORCE_MULT_MIN)
        _log(f"KEY -  |  Force multiplier ↓  →  ×{_force_multiplier:.1f} "
             f"(range [×{_FORCE_MULT_MIN:.1f}–×{_FORCE_MULT_MAX:.1f}])")

    # ---- visual mesh visibility ---------------------------------------------

    elif key in _KEY_V:
        _visual_hidden = _apply_visual_visibility(
            model, _viewer_handle, not _visual_hidden)
        _log(f"KEY V  |  Visual meshes {'hidden' if _visual_hidden else 'shown'}"
             + ("  (only collision primitives render)" if _visual_hidden else ""))

    # ---- log UAV position ---------------------------------------------------

    elif key in _KEY_T:
        pos = data.xpos[body_id]
        _log(f"KEY T  |  Shahed position (world): "
             f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")
        cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "launch_cart")
        cart_pos = data.xpos[cart_id] if cart_id >= 0 else None
        if cart_pos is not None:
            _log(f"         Cart position   (world): "
                 f"({cart_pos[0]:+.3f}, {cart_pos[1]:+.3f}, {cart_pos[2]:+.3f}) m")

    # ---- reset --------------------------------------------------------------

    elif key in _KEY_R:
        data.qpos[:] = _initial_qpos.copy()
        data.qvel[:] = _initial_qvel.copy()
        data.xfrc_applied[body_id, :] = 0.0
        data.ctrl[:] = 0.0
        _flap_mode = "neutral"
        _rail_impulse_remaining = 0
        _rail_hold_position = 0.0
        _rail_launch_t = 0.0
        # mj_forward FIRST: _set_cradle reads xpos/xquat, which must reflect
        # the reset pose — otherwise the weld targets the old (mid-flight)
        # relative pose and snaps the airframe on the next step.
        mujoco.mj_forward(model, data)
        if _cradle_gap(data, model) <= cfg.rail_cradle_gap:
            _set_cradle(model, data, True)     # UAV is back on the cart
            _log_weld_fit(model, data, "after reset")
        _reset_hold_remaining = _reset_hold_steps(cfg)
        _log(f"KEY R  |  Reset → pos={tuple(data.xpos[body_id].round(3))}  "
             f"(flaps neutral, rail reset — gentle auto push in "
             f"{_RESET_IMPULSE_DELAY:.2f} s; drone stays hooked on the cart, "
             f"press L/Space to launch)")

    # ---- rail impulse -------------------------------------------------------

    elif key in _KEY_L or key in _KEY_SPACE:
        if _rail_impulse_remaining > 0:
            _log("KEY L/Space  |  launch already in progress — ignored "
                 "(press R to reset first)")
            return
        _reset_hold_remaining = 0          # manual impulse cancels the pending one
        _rail_impulse_force = cfg.rail_force_manual
        _rail_impulse_remaining = _rail_impulse_steps(
            cfg, cfg.rail_impulse_duration_manual)
        _rail_launch_t = 0.0
        # Full launch: size the accel so the exit speed is reached exactly at
        # the release position from wherever the cart currently sits.
        _rail_launch_speed = cfg.rail_exit_speed
        stroke = max(0.1, cfg.rail_release_q - data.qpos[_rail_qposadr])
        _rail_launch_accel = cfg.rail_exit_speed ** 2 / (2.0 * stroke)
        # Cradle the UAV to the cart so the stroke carries it.  Only when the
        # UAV is actually seated: engaging the weld mid-flight would snap it
        # back onto the cart.
        gap = _cradle_gap(data, model)
        if _rail_eq_id < 0:
            cradle_msg = "WARNING: no uav_cradle weld in XML — carrying by proximity"
        elif gap <= cfg.rail_cradle_gap:
            _set_cradle(model, data, True)
            cradle_msg = "cradle engaged"
        else:
            cradle_msg = f"cradle NOT engaged (UAV {gap:.1f} m from cart)"
        _log(f"KEY L/Space  |  LAUNCH  →  {cfg.rail_exit_speed:.1f} m/s exit  "
             f"({cradle_msg}, release at cart q={cfg.rail_release_q:.1f} m)")


# ---------------------------------------------------------------------------
# Force-application helper
# ---------------------------------------------------------------------------


def _apply_force_at_body_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    force_world: np.ndarray,
    point_body: np.ndarray,
) -> None:
    """Apply a world-frame force at a body-frame point on a body."""
    # COM position in world frame
    com_world = data.subtree_com[body_id].copy()

    # Body orientation quaternion
    body_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(body_quat, data.xmat[body_id].reshape(9))

    # Transform point from body to world frame
    point_world = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(point_world, point_body, body_quat)
    point_world += data.xpos[body_id]

    # Torque about COM
    r = point_world - com_world
    torque_world = np.cross(r, force_world)

    for i in range(3):
        data.xfrc_applied[body_id, i] += force_world[i]
        data.xfrc_applied[body_id, i + 3] += torque_world[i]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_start_time = time.time()


def _rail_impulse_steps(cfg: SimConfig, duration: float) -> int:
    """Number of physics steps for one rail impulse of `duration` seconds."""
    return max(1, int(duration / cfg.timestep))


def _set_cradle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    engage: bool,
) -> None:
    """Engage / release the cart<->UAV cradle weld (equality constraint).

    When engaging, the weld's target relative pose is rewritten to the UAV's
    *current* pose relative to the cart, so the constraint is exactly
    satisfied at the moment of engagement — otherwise the solver yanks the
    200 kg airframe towards the (arbitrary) pose stored in the XML.
    """
    if _rail_eq_id < 0:
        return
    if not engage:
        data.eq_active[_rail_eq_id] = 0
        return

    uav_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
    cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "launch_cart")
    if uav_id < 0 or cart_id < 0:
        return

    # eq_data layout (mjNEQDATA = 11): [0:3] anchor, [3:6] relpose position,
    # [6:10] relpose quaternion, [10] torquescale.  relpose = pose of body2
    # (uav) in body1's (cart) frame: position rotated into the cart frame,
    # orientation = q_cart^-1 * q_uav.  Anchor and torquescale stay as
    # compiled — only the relpose is rewritten.
    R_cart = data.xmat[cart_id].reshape(3, 3)
    pos_rel = R_cart.T @ (data.xpos[uav_id] - data.xpos[cart_id])
    q_cart = data.xquat[cart_id]
    q_uav = data.xquat[uav_id]
    q_cart_inv = np.zeros(4, dtype=np.float64)
    mujoco.mju_negQuat(q_cart_inv, q_cart)
    q_rel = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_rel, q_cart_inv, q_uav)

    model.eq_data[_rail_eq_id][3:6] = pos_rel
    model.eq_data[_rail_eq_id][6:10] = q_rel
    data.eq_active[_rail_eq_id] = 1


def _cradle_gap(data: mujoco.MjData, model: mujoco.MjModel) -> float:
    """Distance [m] between the UAV and the cart (are they seated together?)."""
    uav_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
    cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "launch_cart")
    if uav_id < 0 or cart_id < 0:
        return float("inf")
    return float(np.linalg.norm(data.xpos[uav_id] - data.xpos[cart_id]))


def _log_weld_fit(model: mujoco.MjModel, data: mujoco.MjData, when: str) -> None:
    """Log how well the cradle weld target matches the current pose.

    0 = constraint exactly satisfied.  A large residual means the weld is
    fighting the pose — it presses the airframe into the cart hull and the
    resulting contact force can swallow the whole launch force (observed as
    the cart creeping up the rail with full ctrl applied).
    """
    if _rail_eq_id < 0:
        return
    mujoco.mj_forward(model, data)
    rows = np.where((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    & (data.efc_id == _rail_eq_id))[0]
    if len(rows):
        resid = float(np.linalg.norm(data.efc_pos[rows]))
        _log(f"RAIL   |  Cradle target fit ({when}): residual {resid*100:.2f} cm"
             + ("" if resid < 0.005 else "  << MISMATCH — cradle is fighting!"))


def _reset_hold_steps(cfg: SimConfig) -> int:
    """Physics steps to hold after key R before the auto rail impulse fires."""
    return max(1, int(_RESET_IMPULSE_DELAY / cfg.timestep))


def _step_reset_hold(cfg: SimConfig) -> None:
    """Count down the post-reset hold, then fire the gentle auto push.
    Called once per physics step from the main loop.

    The auto push is deliberately NOT a launch: the drone stays hooked on the
    cart bar (cradle weld stays engaged, no release at the rail end) and the
    whole cart+UAV assembly is just nudged a few decimetres up the rail.
    Only a manual L / Space press performs the full catapult stroke that
    releases the UAV into the air.
    """
    global _reset_hold_remaining, _rail_impulse_remaining
    global _rail_impulse_force, _rail_launch_t
    global _rail_launch_speed, _rail_launch_accel
    if _reset_hold_remaining > 0:
        _reset_hold_remaining -= 1
        if _reset_hold_remaining == 0:
            _rail_impulse_force = cfg.rail_force_reset
            _rail_impulse_remaining = _rail_impulse_steps(
                cfg, cfg.rail_impulse_duration_reset)
            _rail_launch_t = 0.0
            _rail_launch_speed = cfg.rail_auto_exit_speed   # gentle nudge
            _rail_launch_accel = cfg.rail_auto_accel
            _log(f"AUTO   |  Gentle push  →  {cfg.rail_auto_exit_speed:.1f} m/s  "
                 f"(drone stays hooked on the cart — press L/Space to launch)")


def _rail_release(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """End the stroke: open the cradle so the UAV flies on with its momentum."""
    global _rail_impulse_remaining
    cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "launch_cart")
    cart_xpos = data.xpos[cart_id]
    _set_cradle(model, data, False)
    _log(f"RAIL   |  RELEASE  cart q={data.qpos[_rail_qposadr]:.2f} m  "
         f"cart v={data.qvel[_rail_dofadr]:+.2f} m/s  "
         f"xpos=({cart_xpos[0]:.2f}, {cart_xpos[1]:.2f}, {cart_xpos[2]:.2f})")
    _rail_impulse_remaining = 0


def _rail_control(model: mujoco.MjModel, data: mujoco.MjData, cfg: SimConfig) -> None:
    """Pre-step launcher control: motor assist during the stroke (the motion
    itself is scripted by _rail_drive), position-servo brake when idle."""
    global _rail_hold_position
    cart_q = data.qpos[_rail_qposadr]
    cart_dof = _rail_dofadr

    if _rail_impulse_remaining > 0:
        # Motor strains along; the kinematic driver sets the actual motion.
        data.ctrl[_rail_act_id] = _rail_impulse_force
        _rail_hold_position = cart_q
    else:
        # Idle — position-servo brake: hold cart at last position
        error = _rail_hold_position - cart_q
        hold_force = _RAIL_HOLD_KP * error
        # Also damp to prevent oscillation
        hold_force -= 200.0 * data.qvel[cart_dof]
        data.ctrl[_rail_act_id] = hold_force


def _rail_drive(model: mujoco.MjModel, data: mujoco.MjData, cfg: SimConfig) -> None:
    """Post-step kinematic launch driver — called right after mj_step.

    The cart's slide velocity is scripted (v = a*t, capped at the exit speed,
    a sized to reach the exit speed exactly at the release position).  While
    the UAV is cradled (weld engaged, or seated close to the cart) its world
    velocity is scripted identically, so the pair rides the rail as one rigid
    unit regardless of what the contact solver is doing.  At the release
    position the cradle opens and normal dynamics take over.
    """
    global _rail_launch_t, _rail_impulse_remaining
    if _rail_impulse_remaining <= 0 or _rail_act_id < 0:
        return

    _rail_launch_t += cfg.timestep
    v = min(_rail_launch_accel * _rail_launch_t, _rail_launch_speed)
    data.qvel[_rail_dofadr] = v

    # Carry the UAV: weld engaged, or merely seated near the cart (covers a
    # missing/broken weld constraint).
    cradled = (_rail_eq_id >= 0 and bool(data.eq_active[_rail_eq_id]))
    if not cradled and _rail_carry_uav:
        cradled = _cradle_gap(data, model) <= cfg.rail_cradle_gap
    if cradled:
        uav_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
        uav_dof = model.body_dofadr[uav_id]
        data.qvel[uav_dof:uav_dof + 3] = v * _rail_axis_w

    if _rail_impulse_remaining % 20 == 0:
        _log(f"  RAIL stroke [{_rail_impulse_remaining:3d} steps left]  "
             f"cart q={data.qpos[_rail_qposadr]:.3f}  v={v:5.2f} m/s")

    _rail_impulse_remaining -= 1
    _rail_hold_position = data.qpos[_rail_qposadr]

    if (data.qpos[_rail_qposadr] >= cfg.rail_release_q
            or _rail_impulse_remaining <= 0):
        if data.qpos[_rail_qposadr] >= cfg.rail_release_q:
            _rail_release(model, data)
        else:
            # Stroke ended before the release position: that is the normal end
            # of a gentle auto push (drone stays hooked on the cart bar).  For
            # a manual launch it would mean the duration was too short to
            # reach the exit speed — increase --rail-duration.
            _rail_impulse_remaining = 0


def _log(msg: str) -> None:
    t = time.time() - _start_time
    print(f"[t={t:7.2f}s]  {msg}", flush=True)


def _print_config(cfg: SimConfig) -> None:
    """Print a readable configuration summary."""
    w = 200 * cfg.gravity  # weight in N
    print()
    print("=" * 62)
    print("  SIMULATION CONFIGURATION")
    print("=" * 62)
    print(f"  Model       : {cfg.xml_path}")
    print(f"  Timestep    : {cfg.timestep} s  |  Gravity : {cfg.gravity} m/s²")
    print(f"  Init pos    : ({cfg.init_pos[0]:.1f}, {cfg.init_pos[1]:.1f}, "
          f"{cfg.init_pos[2]:.1f}) m")
    print(f"  Init pitch  : {cfg.init_pitch_deg:.1f}°")
    print(f"  UAV weight  : {w:.0f} N")
    print(f"  --- Forces ---")
    print(f"  Key 0 (+Z @ COM)        : {cfg.force_up_z:.0f} N")
    print(f"  Key 5 (+X @ nose)       : {cfg.force_nose_x:.0f} N")
    print(f"  --- Aerodynamics (body-frame) ---")
    print(f"  Drag  (kx, ky, kz)      : {cfg.k_drag[0]:.0f}  {cfg.k_drag[1]:.0f}  "
          f"{cfg.k_drag[2]:.0f}")
    print(f"  Lift  (k_lift)           : {cfg.k_lift:.0f}")
    print(f"  Rot damp (krx, kry, krz) : {cfg.k_rot_damp[0]:.0f}  "
          f"{cfg.k_rot_damp[1]:.0f}  {cfg.k_rot_damp[2]:.0f}")
    print(f"  --- Flaps ---")
    print(f"  Flap coefficient (k_flap) : {cfg.flap_k:.0f}")
    print(f"  Flap init magnitude       : {cfg.flap_init_magnitude:.2f}")
    print(f"  Flap right pos (body)     : {cfg.flap_pos_right}")
    print(f"  Flap left  pos (body)     : {cfg.flap_pos_left}")
    print(f"  --- Force Scaling ---")
    print(f"  Force multiplier (init)   : ×{cfg.force_init_multiplier:.1f}")
    print(f"  --- Rail Launch ---")
    print(f"  Rail impulse force manual : {cfg.rail_force_manual:.0f} N  (L / Space)")
    print(f"  Rail exit speed           : {cfg.rail_exit_speed:.1f} m/s (L/Space)  /  "
          f"{cfg.rail_auto_exit_speed:.1f} m/s gentle push (auto)")
    print(f"  Rail impulse duration man.: {cfg.rail_impulse_duration_manual:.2f} s  "
          f"({_rail_impulse_steps(cfg, cfg.rail_impulse_duration_manual)} steps)")
    print(f"  Rail impulse duration auto: {cfg.rail_impulse_duration_reset:.2f} s  "
          f"({_rail_impulse_steps(cfg, cfg.rail_impulse_duration_reset)} steps)")
    print(f"  --- Viewer ---")
    print(f"  Visual mesh hulls         : "
          f"{'hidden (V toggles)' if cfg.hide_visual else 'shown (V toggles)'}")
    print("=" * 62)
    print()
    print("  CONTROLS  (top-row or numpad)")
    print("    0 / 5  —  apply force @ COM / nose")
    print("    4 / 6  —  left / right flap turn")
    print("    7 / 9  —  flap magnitude  ↓/↑")
    print("    + / -  —  force multiplier ↑/↓")
    print("    L / Space —  rail launch impulse")
    print("    R      —  reset pose, auto rail impulse after 0.75 s")
    print("    V      —  hide/show visual meshes (only collision prims)")
    print("    T      —  log shahed + cart positions (world)")
    print("    ESC    —  exit")
    print("  -----------  built-in MuJoCo viewer shortcuts  -----------")
    print("    Tab    —  toggle info overlay")
    print("    F1     —  toggle left  UI panel  (more view)")
    print("    F2     —  toggle right UI panel  (more view)")
    print("    RMB    —  rotate camera  |  MMB — zoom  |  LMB — pan")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    global _cfg, _initial_qpos, _initial_qvel
    global _viewer_handle, _visual_hidden
    global _reset_hold_remaining
    global _rail_act_id, _rail_eq_id
    global _rail_qposadr, _rail_dofadr, _rail_axis_w, _rail_carry_uav

    # -- parse CLI ------------------------------------------------------------
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _args_to_config(args)
    _cfg = cfg

    # -- initialise runtime mutable state --------------------------------------
    global _flap_mode, _flap_magnitude, _force_multiplier
    global _rail_impulse_remaining, _rail_hold_position
    _flap_mode = "neutral"
    _flap_magnitude = cfg.flap_init_magnitude
    _force_multiplier = cfg.force_init_multiplier
    _rail_impulse_remaining = 0
    _rail_hold_position = 0.0
    _reset_hold_remaining = 0

    # -- load model & data ----------------------------------------------------
    _log(f"Loading model: {cfg.xml_path}")
    model = mujoco.MjModel.from_xml_path(cfg.xml_path)
    data = mujoco.MjData(model)

    # -- override physics parameters from CLI ---------------------------------
    model.opt.timestep = cfg.timestep
    model.opt.gravity[2] = -cfg.gravity

    # -- install aerodynamic callback -----------------------------------------
    mujoco.set_mjcb_passive(_aero_passive_callback)

    # -- set initial state ----------------------------------------------------
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "uav")
    # Set only the UAV free-joint portion of qpos / qvel
    uav_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "uav_joint")
    ]
    uav_dofadr = model.body_dofadr[body_id]
    data.qpos[uav_qadr : uav_qadr + 7] = cfg.init_qpos
    data.qvel[uav_dofadr : uav_dofadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    # The cradle weld's compiled relpose matches the XML pose; re-target it to
    # the actual init pose (which --pos-x/y/z / --pitch may have moved) so the
    # constraint starts satisfied instead of yanking the airframe on step 1.
    _rail_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     "rail_motor")
    _rail_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                    "uav_cradle")
    rail_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    "rail_joint")
    _rail_qposadr = model.jnt_qposadr[rail_jnt_id]
    _rail_dofadr = model.jnt_dofadr[rail_jnt_id]
    rail_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     "rail_link")
    _quat_tmp = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(_quat_tmp, data.xmat[rail_link_id].reshape(9))
    _rail_axis_w = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(_rail_axis_w, model.jnt_axis[rail_jnt_id], _quat_tmp)
    _rail_carry_uav = True
    if _rail_eq_id < 0:
        _log("WARNING: no 'uav_cradle' weld in the model — the UAV will be "
             "carried by proximity only; consider updating the XML")
    if _rail_act_id >= 0 and cfg.rail_force_manual > model.actuator_ctrlrange[_rail_act_id][1]:
        _log(f"WARNING: rail force {cfg.rail_force_manual:.0f} N exceeds the "
             f"actuator ctrlrange "
             f"{model.actuator_ctrlrange[_rail_act_id]} — force will be clamped "
             f"(launch motion is scripted, so the stroke is unaffected)")
    _log(f"Launch profile: L/Space → scripted stroke, {cfg.rail_exit_speed:.1f} m/s "
         f"at release (cart q = {cfg.rail_release_q:.1f} m, accel "
         f"{cfg.rail_exit_speed ** 2 / (2 * cfg.rail_release_q):.1f} m/s²);  "
         f"reset/start → gentle {cfg.rail_auto_exit_speed:.1f} m/s push "
         f"(drone stays hooked on the cart)")
    # Engage the cradle only when the UAV actually starts seated on the cart
    # (presets like "high" place it mid-air — welding it there would snap it
    # onto the cart on the first step).
    if _cradle_gap(data, model) <= cfg.rail_cradle_gap:
        _set_cradle(model, data, True)
        _log_weld_fit(model, data, "sim start")
    _initial_qpos = data.qpos.copy()
    _initial_qvel = data.qvel.copy()

    # -- rail actuator & cradle weld -------------------------------------------
    # (ids already resolved above, before the initial cradle engagement)
    _log(f"Rail actuator id: {_rail_act_id}  |  cradle weld eq id: {_rail_eq_id}"
         + ("" if _rail_eq_id >= 0
            else "  (MISSING — add <weld name=\"uav_cradle\"/> to the XML!)"))
    # Exit speed at the end of the 2 m stroke (205 kg, slope eats ~400 N):
    # v = sqrt(2 * (F/m - g*sin(11.5 deg)) * stroke), release at rail_release_q
    for label, f in (("manual", cfg.rail_force_manual),
                     ("auto  ", cfg.rail_force_reset)):
        a = f / 205.0 - cfg.gravity * math.sin(math.radians(11.5))
        v_exit = math.sqrt(max(0.0, 2.0 * a * cfg.rail_release_q))
        _log(f"Rail impulse {label}: {f:.0f} N × "
             f"{cfg.rail_impulse_duration_manual if label == 'manual' else cfg.rail_impulse_duration_reset:.2f} s"
             f"  →  ~{v_exit:.1f} m/s at release "
             f"(lift ~{cfg.k_lift * v_exit ** 2:.0f} N vs "
             f"{model.body_subtreemass[body_id] * cfg.gravity:.0f} N weight)")
    # -- auto rail impulse at sim start (same as after reset) ------------------
    _reset_hold_remaining = _reset_hold_steps(cfg)
    _log(f"Sim start |  Gentle auto push in {_RESET_IMPULSE_DELAY:.2f} s  "
         f"({cfg.rail_auto_exit_speed:.1f} m/s — drone stays hooked on the cart)")

    # -- print info -----------------------------------------------------------
    _print_config(cfg)

    _log(f"UAV mass : {model.body_subtreemass[body_id]:.1f} kg")
    _log(f"Init pos : {tuple(_initial_qpos[uav_qadr:uav_qadr+3])}")
    _log(f"Init quat: {tuple(_initial_qpos[uav_qadr+3:uav_qadr+7].round(4))}  "
         f"(roll=0°, pitch={cfg.init_pitch_deg}°, yaw=0°)")

    # -- launch viewer --------------------------------------------------------
    show_ui = not cfg.hide_ui
    handle = mujoco.viewer.launch_passive(
        model, data,
        key_callback=_key_callback,
        show_left_ui=show_ui,
        show_right_ui=show_ui,
    )
    # Move camera +1 m along Y for better side view of the rail + UAV
    handle.cam.lookat[0] -= 0.0
    handle.cam.lookat[1] += 0.0
    handle.cam.lookat[2] += 2.0

    # -- viewer visibility: hide visual hulls, show collision prims ------------
    _viewer_handle = handle
    _visual_hidden = cfg.hide_visual
    _visual_hidden = _apply_visual_visibility(model, handle, _visual_hidden)
    _log(f"Visual meshes {'hidden — only collision prims render' if _visual_hidden else 'shown'}"
         "  (press V to toggle)")
    
    _log(f"Viewer launched (UI panels {'hidden' if cfg.hide_ui else 'visible'}).  "
         "Press keys to interact.")

    # -- physics loop ---------------------------------------------------------
    _log(f"Target real-time rate: {1.0 / cfg.timestep:.0f} Hz  "
         f"(dt={cfg.timestep:.4f} s)")

    # Real-time pacing: track sim time vs wall time
    _sim_time = 0.0
    _wall_start = time.time()
    _last_report = 0.0

    try:
        while handle.is_running():
            # --- process keys ------------------------------------------------
            while not _key_queue.empty():
                try:
                    key = _key_queue.get_nowait()
                    if key == _glfw.KEY_ESCAPE:
                        _log("ESC pressed — exiting.")
                        handle.close()
                        break
                    _apply_key_action(model, data, key)
                except queue.Empty:
                    break

            # ---- post-reset auto rail impulse --------------------------------
            _step_reset_hold(cfg)

            # ---- rail control (applied before mj_step) -----------------------
            _rail_control(model, data, cfg)

            mujoco.mj_step(model, data)
            _rail_drive(model, data, cfg)
            data.xfrc_applied[body_id, :] = 0.0
            handle.sync()

            # --- real-time pacing -------------------------------------------
            _sim_time += cfg.timestep
            _wall_elapsed = time.time() - _wall_start
            _sleep = _sim_time - _wall_elapsed
            if _sleep > 0:
                time.sleep(_sleep)

    except KeyboardInterrupt:
        _log("Interrupted (Ctrl+C).")
        print()
    finally:
        handle.close()
        _log("Simulation ended.")


if __name__ == "__main__":
    main()

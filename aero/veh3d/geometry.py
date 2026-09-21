"""Parametric aerial-vehicle geometry: 2D sections lofted along the span.

Pipeline (all parameter-driven, design.yaml is the source of truth):

    yaml params -> airfoil section ring per span station -> lofted triangle
    mesh (trimesh) -> mass properties + projected areas -> NeuralFoil section
    polar at cruise -> VehicleGeom (everything the control env needs).

Frame: x = forward, y = right span, z = up. Sections live in the local x-z
plane at span station y; section x runs 0 (LE) to 1 (TE) * local chord, so a
section is a 2D curve and the vehicle is the 3D curve (loft) swept through the
span stations. The root section is on the mirror plane y=0 — the mesh is built
symmetric by construction.

Simplifications, on purpose (v0):
  * sections are closed NACA-style thickness + parabolic camber (CST hooks
    come later via common/airfoil.py);
  * frontal area uses Cauchy's projection formula — exact for convex bodies,
    a small over-count on this near-convex loft;
  * one section polar (root) stands in for the full spanwise lift distribution.

CLI:
    python geometry.py --stl meshes/veh_default.stl --png results/veh3d/veh_default.png
    python geometry.py --set geometry.span=1.6 --set geometry.camber=0.04 ...
"""
import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.airfoil import cosine_spacing  # noqa: E402

DESIGN_YAML = Path(__file__).resolve().parent / "design.yaml"


# -- config ------------------------------------------------------------------
def load_design(path=DESIGN_YAML):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _split_path(dotted):
    parent, _, leaf = dotted.rpartition(".")
    return parent, leaf


def set_param(cfg, dotted, value):
    """Merge one dotted-path override into a (deep-copied) config dict."""
    out = copy.deepcopy(cfg)
    node = out
    for part in dotted.split(".")[:-1]:
        node = node[part]
    node[dotted.split(".")[-1]] = value
    return out


def overrides_to_cfg(cfg, overrides):
    """dict of dotted-path -> value (or None) -> merged config."""
    merged = cfg
    for k, v in (overrides or {}).items():
        if v is not None:
            merged = set_param(merged, k, v)
    return merged


# -- sections ----------------------------------------------------------------
def section_ring(t, camber, n_half):
    """Closed 2D section, unit chord: (2*n_half - 2, 2) array of (x, z).

    A strictly simple polygon — no duplicated seam points: the LE and TE may
    appear only ONCE per ring, or the loft grows zero-length edges whose
    merged vertices turn strip triangles degenerate and punch holes.
    Order: LE -> upper -> TE -> lower(interior) -> wrap to LE.
    TE gap of the classic NACA polynomial is closed by the -0.1036 coefficient
    (yt(TE)=0 exactly).
    """
    xc = cosine_spacing(n_half)
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                  + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)   # closed-TE coefficient
    zc = 4.0 * camber * xc * (1.0 - xc)                    # parabolic camber
    upper = np.column_stack([xc, zc + yt])                 # LE -> TE, n points
    lower = np.column_stack([xc[1:-1][::-1], (zc - yt)[1:-1][::-1]])  # n-2
    return np.concatenate([upper, lower], axis=0)


def _rot_xz(pts, theta, pivot_x):
    """Rotate (x, z) points by theta about the vertical line x=pivot_x."""
    c, s = np.cos(theta), np.sin(theta)
    x, z = pts[:, 0] - pivot_x, pts[:, 1]
    return np.column_stack([pivot_x + c * x + s * z, -s * x + c * z])


# -- loft --------------------------------------------------------------------
# -- mass properties (own divergence integral — do NOT use trimesh's
#    mass_properties here: its cache survives fix_normals/invert and has been
#    observed returning a centroid OUTSIDE the mesh bounding box)
def mass_properties(verts, faces, density):
    """(volume, com, inertia-about-com) via signed tetrahedron integrals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cr = np.cross(v1, v2)
    vol6 = np.einsum("ij,ij->i", v0, cr)             # 6V per tet (origin, f)
    volume = float(vol6.sum()) / 6.0
    if volume < 0:                                   # caller fixes winding first
        raise ValueError("mesh volume negative — winding not repaired")

    # com and second moments of a tet (origin, v0, v1, v2):
    #   int(x) dV = V/24 * (sum of edge midpoints stuff) — standard forms:
    s = [v0, v1, v2]
    sum_v = v0 + v1 + v2
    com_tet = (vol6[:, None] * sum_v / 4.0)          # int(x)dV = V*mean*... =
    com = com_tet.sum(axis=0) / (6.0 * volume)       #   (6V)(v0+v1+v2)/24

    # inertia about origin: int (|x|^2 I - x x^T) dV per tet
    # int x_i x_j dV = V/60 * (sum over the 15 pairings) — use the compact form
    a, b, c = v0, v1, v2
    int_xx = (vol6 / 60.0) * (
        (a[:, 0] ** 2 + b[:, 0] ** 2 + c[:, 0] ** 2
         + a[:, 0] * b[:, 0] + a[:, 0] * c[:, 0] + b[:, 0] * c[:, 0]))
    int_yy = (vol6 / 60.0) * (
        (a[:, 1] ** 2 + b[:, 1] ** 2 + c[:, 1] ** 2
         + a[:, 1] * b[:, 1] + a[:, 1] * c[:, 1] + b[:, 1] * c[:, 1]))
    int_zz = (vol6 / 60.0) * (
        (a[:, 2] ** 2 + b[:, 2] ** 2 + c[:, 2] ** 2
         + a[:, 2] * b[:, 2] + a[:, 2] * c[:, 2] + b[:, 2] * c[:, 2]))
    int_xy = (vol6 / 120.0) * (
        (2 * (a[:, 0] * a[:, 1] + b[:, 0] * b[:, 1] + c[:, 0] * c[:, 1])
         + a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]
         + a[:, 0] * c[:, 1] + a[:, 1] * c[:, 0]
         + b[:, 0] * c[:, 1] + b[:, 1] * c[:, 0]))
    int_xz = (vol6 / 120.0) * (
        (2 * (a[:, 0] * a[:, 2] + b[:, 0] * b[:, 2] + c[:, 0] * c[:, 2])
         + a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0]
         + a[:, 0] * c[:, 2] + a[:, 2] * c[:, 0]
         + b[:, 0] * c[:, 2] + b[:, 2] * c[:, 0]))
    int_yz = (vol6 / 120.0) * (
        (2 * (a[:, 1] * a[:, 2] + b[:, 1] * b[:, 2] + c[:, 1] * c[:, 2])
         + a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1]
         + a[:, 1] * c[:, 2] + a[:, 2] * c[:, 1]
         + b[:, 1] * c[:, 2] + b[:, 2] * c[:, 1]))

    I_o = np.array([
        [int_yy.sum() + int_zz.sum(), -int_xy.sum(), -int_xz.sum()],
        [-int_xy.sum(), int_xx.sum() + int_zz.sum(), -int_yz.sum()],
        [-int_xz.sum(), -int_yz.sum(), int_xx.sum() + int_yy.sum()],
    ]) * density
    m = density * volume
    I_com = I_o - m * (com @ com) * np.eye(3) + m * np.outer(com, com)
    return volume, com, I_com


def loft_mesh(g):
    """Loft section rings across span stations -> watertight trimesh."""
    import trimesh

    n_st, n_pts = g["n_sections"], 2 * g["n_half"] - 2
    ys = np.linspace(-g["span"] / 2.0, g["span"] / 2.0, n_st)
    eta = np.abs(ys) / (g["span"] / 2.0)

    rings = []
    for y, e in zip(ys, eta):
        chord = g["root_chord"] * (1.0 - (1.0 - g["taper"]) * e)
        ring = section_ring(
            g["thickness_root"] + (g["thickness_tip"] - g["thickness_root"]) * e,
            g["camber"], g["n_half"])
        ring = _rot_xz(ring, np.deg2rad(g["twist_deg"]) * e, pivot_x=0.25)
        x_le = -np.tan(np.deg2rad(g["sweep_deg"])) * abs(y)
        z_off = np.tan(np.deg2rad(g["dihedral_deg"])) * abs(y)
        r3 = np.empty((n_pts, 3))
        # ring x runs LE(0)->TE(1); x is FORWARD, so the chord extends AFT (-x)
        r3[:, 0] = (x_le - ring[:, 0]) * chord
        r3[:, 1] = y
        r3[:, 2] = ring[:, 1] * chord + z_off
        rings.append(r3)

    verts, faces = np.vstack(rings), []
    c0, c1 = n_st * n_pts, n_st * n_pts + 1        # cap centroids
    verts = np.vstack([verts, rings[0].mean(axis=0), rings[-1].mean(axis=0)])
    for s in range(n_st - 1):
        for j in range(n_pts):                     # shared-index quad strips
            k = (j + 1) % n_pts
            a, b = s * n_pts + j, s * n_pts + k
            c, d = (s + 1) * n_pts + k, (s + 1) * n_pts + j
            faces += [[a, b, c], [a, c, d]]
    for j in range(n_pts):                         # end caps, opposite winding
        k = (j + 1) % n_pts
        faces.append([j, k, c0])
        faces.append([(n_st - 1) * n_pts + k, (n_st - 1) * n_pts + j, c1])

    mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces),
                           process=True)
    # strip/cap windings can disagree locally -> inconsistent normals -> a
    # garbage (possibly negative) inertia tensor; repair + assert orientation
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


# -- analysis ----------------------------------------------------------------
@dataclass
class VehicleGeom:
    cfg: dict
    mesh: object
    mass: float                 # kg, foam + payload
    izz: float                  # kg m^2 about CG (yaw inertia)
    cg: np.ndarray
    span: float
    chord_root: float
    S_ref: float                # planform area (top-view projection)
    S_wet: float                # wetted area proxy = full surface area
    A_front: float              # y-z projected area
    AR: float
    re_root: float
    cl_c: float                 # section CL at cruise alpha (NeuralFoil)
    cd_c: float                 # section CD at cruise alpha
    polar_ok: bool
    CD0: float                  # skin + frontal, referred to S_ref
    k_ind: float                # 1/(pi e AR)
    design_vector: np.ndarray = field(default=None)

    # derived, cruise conditions
    @property
    def aero(self):
        a = self.cfg["aero"]
        q = 0.5 * a["rho"] * a["cruise_speed"] ** 2
        CL = self.cl_c
        CD = self.CD0 + self.k_ind * CL ** 2
        L = q * self.S_ref * CL
        D = q * self.S_ref * CD
        m_tot = self.mass
        return dict(q=q, CL=CL, CD=CD, lift=L, drag=D, weight=m_tot * 9.81)


def feasibility(geom):
    """Hard gates every design must pass before it earns a control run."""
    a, cfg = geom.aero, geom.cfg
    reasons = []
    if not geom.mesh.is_watertight:
        reasons.append("mesh not watertight")
    if geom.aero["CL"] <= 0.0 or not geom.polar_ok:
        reasons.append("section polar failed (NeuralFoil)")
    else:
        if a["lift"] < 1.2 * a["weight"]:
            reasons.append(f"lift gate: L={a['lift']:.2f}N < "
                           f"1.2*W={1.2 * a['weight']:.2f}N @ cruise")
    gap = cfg["motors"]["motor_gap"]
    if gap > geom.span / 2.0 - 0.05:
        reasons.append(f"motor gate: gap {gap:.2f} exceeds half-span margin "
                       f"{geom.span / 2.0 - 0.05:.2f}")
    n = cfg["motors"]["count"]
    if n * cfg["motors"]["max_thrust"] < 1.15 * a["drag"]:
        reasons.append(f"thrust gate: {n}*{cfg['motors']['max_thrust']:.2f}N < "
                       f"1.15*D={1.15 * a['drag']:.2f}N @ cruise")
    return (len(reasons) == 0), reasons


def _design_vector(geom):
    """Fixed-order normalized design descriptor -> obs (amortized control).

    Same vector whatever the design, so ONE policy can be conditioned on the
    design (pipeline pattern A). Normalizations are arbitrary but frozen.
    """
    g, cfg = geom, geom.cfg
    return np.array([
        g.span / 2.0,
        g.chord_root / 0.5,
        g.cfg["geometry"]["taper"],
        g.cfg["geometry"]["sweep_deg"] / 30.0,
        g.cfg["geometry"]["thickness_root"] * 8.0 - 0.5,
        g.cfg["geometry"]["camber"] * 20.0,
        cfg["motors"]["motor_gap"] / 0.6,
        g.mass / 1.0,
        g.CD0 * 8.0,
        g.cl_c / 1.2,                          # section lift at the trim point
        cfg["aero"]["cruise_alpha_deg"] / 5.0,
    ], dtype=np.float32)


def build(cfg, overrides=None, polar_backend="neuralfoil"):
    """cfg (+ dotted-path overrides) -> analyzed VehicleGeom."""
    from airfoil2d.xfoil_eval import eval_polar

    g = cfg["geometry"]
    mesh = loft_mesh(g)
    density = cfg["material"]["foam_density"]
    volume, com, I_foam = mass_properties(mesh.vertices, mesh.faces, density)

    mass = density * volume + cfg["material"]["payload_mass"]
    izz = float(I_foam[2, 2])

    # Cauchy projection: exact for convex bodies, tiny over-count here
    n = mesh.face_normals
    A_front = 0.5 * float(np.abs(n[:, 0]) @ mesh.area_faces)
    S_ref = 0.5 * float(np.abs(n[:, 2]) @ mesh.area_faces)   # top view = planform
    S_wet = float(mesh.area)

    # root section polar at cruise — reorder ring to the xfoil convention
    # (TE -> upper -> LE -> lower -> TE, open TE pair) the backends expect
    a = cfg["aero"]
    ring = section_ring(g["thickness_root"], g["camber"], g["n_half"])
    n_half = g["n_half"]
    sec = np.concatenate([ring[:n_half][::-1], ring[n_half:][::-1]], axis=0)
    x, y = sec[:, 0], sec[:, 1]
    re = a["cruise_speed"] * g["root_chord"] / a["re_nu"]
    pol = eval_polar(x, y, re, a["cruise_alpha_deg"], backend=polar_backend)

    geom = VehicleGeom(
        cfg=cfg, mesh=mesh, mass=mass, izz=izz,
        cg=com, span=g["span"], chord_root=g["root_chord"],
        S_ref=S_ref, S_wet=S_wet, A_front=A_front,
        AR=g["span"] ** 2 / max(S_ref, 1e-9),
        re_root=re, cl_c=pol["cl"], cd_c=pol["cd"], polar_ok=pol["converged"],
        CD0=a["cd_skin"] * S_wet / S_ref + a["cd_frontal"] * A_front / S_ref,
        k_ind=1.0 / (np.pi * a["oswald_e"] * (g["span"] ** 2 / S_ref)),
    )
    geom.design_vector = _design_vector(geom)
    return geom


def export_stl(geom, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geom.mesh.export(str(path))
    return path


def motor_x(geom):
    """World x of the motor line: LE at that span station, then x_frac*chord
    aft (x_frac < 0 = the back, -x side)."""
    g, m = geom.cfg["geometry"], geom.cfg["motors"]
    gap = m["motor_gap"]
    eta = gap / (geom.span / 2.0)
    chord = g["root_chord"] * (1.0 - (1.0 - g["taper"]) * eta)
    return -np.tan(np.deg2rad(g["sweep_deg"])) * gap + m["x_frac"] * chord


def render_png(geom, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, cfg = geom.cfg["geometry"], geom.cfg
    fig = plt.figure(figsize=(11, 4), dpi=140)
    ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), \
        fig.add_subplot(1, 3, 3, projection="3d")

    # plan view: LE/TE traces + motor line
    ys = np.linspace(-geom.span / 2, geom.span / 2, 200)
    eta = np.abs(ys) / (geom.span / 2)
    chord = g["root_chord"] * (1 - (1 - g["taper"]) * eta)
    x_le = -np.tan(np.deg2rad(g["sweep_deg"])) * np.abs(ys)
    ax1.fill_betweenx(ys, x_le, x_le - chord, alpha=0.35)
    ax1.plot(x_le, ys, "b", x_le - chord, ys, "b")
    mg = cfg["motors"]["motor_gap"]
    ax1.plot([motor_x(geom), motor_x(geom)], [-mg, mg], "r^-", ms=8, lw=2)
    ax1.set_title("plan view (+x right)"); ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.axis("equal"); ax1.grid(alpha=0.3)

    # root / tip sections (x mirrored: nose right, matching the plan view)
    for i, (tag, t, c) in enumerate([("root", g["thickness_root"], g["root_chord"]),
                                     ("tip", g["thickness_tip"],
                                      g["root_chord"] * g["taper"])]):
        ring = section_ring(t, g["camber"], g["n_half"]) * c
        ax2.plot(-ring[:, 0] + 0.05 * i, ring[:, 1],
                 label=f"{tag} t/c={t:.2f}")
    ax2.set_title("2D sections"); ax2.legend(); ax2.axis("equal"); ax2.grid(alpha=0.3)

    # 3D loft
    v, f = geom.mesh.vertices, geom.mesh.faces
    ax3.plot_trisurf(v[:, 0], v[:, 1], triangles=f, Z=v[:, 2],
                     cmap="viridis", edgecolor="none", alpha=0.9)
    ax3.set_title("lofted mesh"); ax3.set_box_aspect((2.5, 2.5, 0.6))
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")

    fig.suptitle(f"{cfg['meta']['name']}  mass={geom.mass:.3f} kg  "
                 f"S={geom.S_ref:.3f} m²  AR={geom.AR:.2f}  "
                 f"A_front={geom.A_front:.4f} m²  CD0={geom.CD0:.3f}  "
                 f"CL_c={geom.cl_c:.2f}")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", default=str(DESIGN_YAML))
    p.add_argument("--set", action="append", default=[],
                   help="dotted.path=value override, repeatable")
    p.add_argument("--stl", default=None)
    p.add_argument("--png", default=None)
    args = p.parse_args()

    cfg = load_design(args.design)
    ov = {}
    for s in args.set:
        k, v = s.split("=", 1)
        ov[k] = yaml_scalar(v)
    geom = build(overrides_to_cfg(cfg, ov))
    ok, reasons = feasibility(geom)
    a = geom.aero
    print(f"mass {geom.mass:.3f} kg | Izz {geom.izz:.4f} | S_ref {geom.S_ref:.3f} | "
          f"AR {geom.AR:.2f} | A_front {geom.A_front:.4f} | CD0 {geom.CD0:.3f}")
    print(f"cruise: CL {a['CL']:.3f} CD {a['CD']:.3f} L {a['lift']:.2f} N "
          f"W {a['weight']:.2f} N D {a['drag']:.2f} N | Re_root {geom.re_root:.2e}")
    print(f"feasible: {ok}" + ("" if ok else f"  reasons: {reasons}"))
    if args.stl:
        print("stl ->", export_stl(geom, args.stl))
    if args.png:
        print("png ->", render_png(geom, args.png))


def yaml_scalar(s):
    import yaml
    return yaml.safe_load(s)


if __name__ == "__main__":
    main()

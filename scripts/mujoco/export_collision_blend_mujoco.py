"""Export Blender collision primitives into a MuJoCo MJCF scene (MJCF rewrite).

Replaces the mesh collision geoms of a MuJoCo scene with multiple box /
cylinder primitives authored in a ``.blend`` file, so ``uav_sim.py`` sims on
lightweight primitive contacts instead of the heavy decimated STL hulls.

Blend naming convention (each collision shape is one mesh object)::

    <key>.<Shape>[.<idx>]

where
  - ``<key>``     is the collision group (``shahed``, ``rail``, ``cart`` ...);
  - ``<Shape>``   is ``Cube`` or ``Cylinder``;
  - ``<idx>``     is Blender's optional duplicate suffix (``.001``, ``.002`` ...),
                  gaps from deleted shapes are fine.

Mesh objects WITHOUT a shape token (visual references such as
``shahed_collision_2``, ``rail_collision``, ``launch_cart``) are ignored.

Example (``assets/shahed-136/rail_cart_shahed_collision.blend``)::

    shahed.Cube.001 ... shahed.Cube.010     rail.Cube.011  rail.Cube.012
    cart.Cube.001  cart.Cube.002  cart.Cube.011

Group -> body mapping
  Each group's primitives are inserted into one MuJoCo ``<body>``; the blend
  ``Item > Position`` / ``Item > Rotation`` values are copied VERBATIM as that
  body-local ``pos`` / ``quat`` of the new geoms (Blender and MuJoCo are both
  Z-up right-handed and metric, so no conversion happens).  So author each
  group in the LOCAL frame of the body it belongs to::

      shahed -> uav          (drone local frame)
      rail   -> base_link    (base frame == world frame here)
      cart   -> launch_cart  (cart local frame)

  The defaults live in ``DEFAULT_KEY_TO_BODY`` below; override or extend any
  of them with repeatable ``--map key=body`` arguments.

What happens in the scene XML
  For every mapped body:
    - previous ``<key>_col_*`` primitives are removed (re-runs are idempotent);
    - every remaining collidable geom of the body (the old STL collision) is
      demoted to VISUAL ONLY (``contype="0" conaffinity="0"``, group 1), or
      deleted entirely with ``--delete-visual-meshes``;
    - one ``<geom type="box">`` / ``<geom type="cylinder">`` per blend shape is
      inserted, named ``<key>_col_<NNN>`` (NNN = the blend trailing index, so
      names stay stable across runs).  Boxes take half-extents from
      ``obj.dimensions``; cylinders take radius and HALF-length from the local
      bounding box (Z-axis cylinder, rotate the OBJECT not the mesh).  Rotation
      is emitted as a convention-free ``quat``.  ``density``/``contype``/
      ``conaffinity`` are inherited from the geom they replace, so body masses
      and the rail's contacts-via-pairs behaviour are unchanged.
  Every explicit ``<contact><pair>`` referencing a replaced geom is re-pointed
  to the new primitives (cross product), keeping ``solref``/``solimp`` and
  getting a deterministic ``<name>_p<N>`` name -- MuJoCo evaluates explicit
  pairs regardless of contype/conaffinity, so this is what keeps the rail
  colliding with the UAV after the old mesh pair goes away.
  With ``--delete-visual-meshes``, ``<mesh>`` assets left unreferenced are
  removed as well.

Run headless::

    blender --background assets/shahed-136/rail_cart_shahed_collision.blend \\
            --python scripts/mujoco/export_collision_blend_mujoco.py -- \\
            [--xml PATH] [--out PATH | --in-place] [--map key=body ...]
            [--delete-visual-meshes] [--allow-empty] [--dry-run]

Defaults: ``--xml`` is the scene next to this script
(``uav_rail_scene.xml``, the scene ``uav_sim.py`` loads); ``--out`` is
``<xml>_prim.xml`` next to it.  Then run the sim on it::

    python scripts/mujoco/uav_sim.py --model scripts/mujoco/uav_rail_scene_prim.xml
"""

import os
import re
import sys
import math
import argparse
import xml.etree.ElementTree as ET

import bpy


# Shape tokens we know how to export.  The token must appear as a name segment
# (between dots), so a reference mesh whose name contains "Cube" verbatim is
# not misread.  Extend this tuple to support Sphere / Capsule later.
SHAPE_TOKENS = ("Cube", "Cylinder")

# Generated geom names are <key>_col_<NNN>; the stable suffix is what makes
# in-place re-runs (newer shape set replaces the previous one) work.
PRIM_TAG = "_col_"
PAIR_IDX_RE = re.compile(r"_p\d+$")     # strip to recover the original pair name

# blend group -> MuJoCo <body> that receives the primitives (see docstring).
DEFAULT_KEY_TO_BODY = {
    "shahed": "uav",
    "rail": "base_link",
    "cart": "launch_cart",
}

# Generated primitives: translucent green, group 0 -- one of the groups
# MuJoCo's viewer shows BY DEFAULT (mjv_defaultOption enables only groups
# 0-2; anything in 3+ is invisible until a flag is flipped).
PRIM_RGBA = "0.2 0.9 0.4 0.35"
PRIM_GROUP = "0"
# The demoted visual hulls go to group 1 (the conventional visual group):
# still visible in a plain viewer, hideable with the standard group toggle
# (digit 1, and hidden automatically by uav_sim.py's collision-only view).
VISUAL_GROUP = "1"


# ---------------------------------------------------------------- Blender --

def _parse_shape_obj(name):
    """Split ``<key>.<Shape>[.<idx>]`` into ``(key, shape_token)``.

    Returns ``(None, None)`` for names that do not follow the convention
    (visual reference meshes etc.).
    """
    segs = name.split(".")
    for i, seg in enumerate(segs):
        if seg in SHAPE_TOKENS:
            key = ".".join(segs[:i])
            return (key or None), seg
    return None, None


def _trailing_int(name):
    """Numeric sort / naming key from the trailing index (``rail.Cube.012`` -> 12)."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _euler_xyz(obj):
    """Item > Rotation X / Y / Z in radians (summary printing only)."""
    if obj.rotation_mode.startswith("XYZ"):
        e = obj.rotation_euler
        return float(e[0]), float(e[1]), float(e[2])
    e = obj.matrix_basis.to_euler("XYZ")
    return float(e.x), float(e.y), float(e.z)


def _rot_quat_wxyz(obj):
    """Object rotation as MuJoCo ``quat`` (w x y z), convention-free."""
    q = obj.matrix_basis.to_quaternion()
    return (float(q.w), float(q.x), float(q.y), float(q.z))


def collect_shapes():
    """Read every convention-named mesh object into
    ``{key: [shape dict, ...]}`` sorted by trailing index."""
    groups = {}
    skipped = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        key, shape = _parse_shape_obj(obj.name)
        if not key or not shape:
            skipped.append(obj.name)
            continue
        dx, dy, dz = (float(v) for v in obj.dimensions)
        if min(dx, dy, dz) <= 1e-9:
            print(f"[BLEND] WARNING {obj.name}: zero extent "
                  f"({dx:.4g},{dy:.4g},{dz:.4g}) -- skipped")
            continue
        pos = tuple(float(v) for v in obj.location)
        quat = _rot_quat_wxyz(obj)
        euler = _euler_xyz(obj)
        if shape == "Cube":
            geom = {"name": obj.name, "type": "box",
                    "size": (dx / 2.0, dy / 2.0, dz / 2.0),   # MuJoCo half-extents
                    "pos": pos, "quat": quat, "euler": euler}
        else:  # Cylinder: Z-axis mesh; MuJoCo size is (radius, HALF length)
            geom = {"name": obj.name, "type": "cylinder",
                    "size": ((dx + dy) / 4.0, dz / 2.0),
                    "pos": pos, "quat": quat, "euler": euler}
        groups.setdefault(key, []).append(geom)

    for geoms in groups.values():
        geoms.sort(key=lambda g: (_trailing_int(g["name"]), g["name"]))
    if skipped:
        preview = ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else "")
        print(f"[BLEND] ignored {len(skipped)} non-convention object(s): {preview}")
    return groups


# ------------------------------------------------------------------- MJCF --

def _fmt(values):
    return " ".join(f"{v:.6g}" for v in values)


def _is_identity(quat):
    return all(abs(a - b) < 1e-9 for a, b in zip(quat, (1.0, 0.0, 0.0, 0.0)))


def load_tree(path):
    """Parse the MJCF keeping comments (they are part of this scene's docs)."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _geom_children(el):
    return [c for c in el if c.tag == "geom"]


def _drop_with_banner(parent, el, key):
    """Remove ``el`` and, if it is our own prim banner comment, that too."""
    children = list(parent)
    i = children.index(el)
    if i > 0 and children[i - 1].tag is ET.Comment and key in (children[i - 1].text or ""):
        parent.remove(children[i - 1])
    parent.remove(el)


def _prim_names(key, geoms, taken):
    """Stable geom names ``<key>_col_<NNN>`` from the blend trailing index."""
    names = []
    for g in geoms:
        base = f"{key}{PRIM_TAG}{_trailing_int(g['name']):03d}"
        name, n = base, 0
        while name in taken:               # same index in two shape types
            n += 1
            name = f"{base}{chr(96 + n)}"
        taken.add(name)
        names.append(name)
    return names


def patch_scene(tree, key_to_body, shapes, delete_visual, dry):
    """Rewrite the tree in place; returns printable change records."""
    root = tree.getroot()
    body_els = {b.get("name"): b for b in root.iter("body")}

    # geom name -> (owning body name or None for worldbody, element)
    geom_owner = {}
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for g in worldbody.findall("geom"):
            geom_owner[g.get("name")] = (None, g)
    for bname, bel in body_els.items():
        for g in _geom_children(bel):
            geom_owner[g.get("name")] = (bname, g)

    contact = root.find("contact")
    pairs = list(contact.findall("pair")) if contact is not None else []
    pair_refs = {p.get(a) for p in pairs for a in ("geom1", "geom2")}

    # ---- resolve the group -> body mapping before touching anything --------
    body_to_key, planned, errors = {}, [], []
    for key in sorted(shapes):
        bname = key_to_body.get(key)
        if bname is None:
            errors.append(f"blend group '{key}' ({len(shapes[key])} shape(s)) has no "
                          f"--map target; bodies: {', '.join(sorted(body_els))}")
        elif bname not in body_els:
            errors.append(f"--map {key}={bname}: no body '{bname}' in the scene")
        elif bname in body_to_key:
            errors.append(f"body '{bname}' is the target of both '{body_to_key[bname]}' and '{key}'")
        else:
            body_to_key[bname] = key
            planned.append((key, bname, body_els[bname]))
    for key, bname in sorted(key_to_body.items()):
        if key not in shapes:
            msg = f"mapped group '{key}' has no shapes in the blend"
            if not dry:  # a silent no-op would leave the OLD collision in place
                errors.append(msg)
            else:
                print(f"[MJCF] note: {msg}")
    if errors:
        for e in errors:
            print(f"[MJCF] ERROR {e}")
        raise SystemExit(2)

    # ---- per body: drop old prims, demote/delete old collision, add prims --
    key_prims, stats = {}, []
    for key, bname, bel in planned:
        geoms = _geom_children(bel)
        old_prims = {g for g in geoms
                     if (g.get("name") or "").startswith(key + PRIM_TAG)}
        fresh = [g for g in geoms if g not in old_prims]

        # Flags donor: a pair-referenced geom first (the rail is 0/0 and only
        # collides through explicit pairs), then any collidable geom, then a
        # previous prim, so re-runs keep the same contact behaviour.
        donor = next((g for g in fresh if g.get("name") in pair_refs), None)
        if donor is None:
            donor = next((g for g in fresh
                          if g.get("contype", "1") != "0"
                          or g.get("conaffinity", "1") != "0"), None)
        if donor is None:
            donor = next(iter(old_prims), None)
        contype = donor.get("contype") if donor is not None else None
        conaffinity = donor.get("conaffinity") if donor is not None else None
        density = (donor.get("density") or "0") if donor is not None else "0"

        for g in old_prims:                                   # previous run
            _drop_with_banner(bel, g, key)
        replaced = []
        for g in fresh:                                       # the old STL hit
            if g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0":
                continue                                      # already visual-only
            replaced.append(g)
            if delete_visual:
                bel.remove(g)
            else:
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("group", VISUAL_GROUP)

        taken = {g.get("name") for g in _geom_children(bel)}
        names = _prim_names(key, shapes[key], taken)
        key_prims[key] = names

        children = list(bel)
        last = max((children.index(g) for g in _geom_children(bel)),
                   default=len(children) - 1)
        at = last + 1
        for name, s in zip(names, shapes[key]):
            bel.insert(at, ET.Comment(f" {s['name']} ({s['type']}) "))
            at += 1
            g = ET.Element("geom", {"name": name, "type": s["type"],
                                    "size": _fmt(s["size"]), "pos": _fmt(s["pos"])})
            if not _is_identity(s["quat"]):
                g.set("quat", _fmt(s["quat"]))
            if contype is not None:
                g.set("contype", contype)
                g.set("conaffinity", conaffinity)
            g.set("density", density)
            g.set("rgba", PRIM_RGBA)
            g.set("group", PRIM_GROUP)
            bel.insert(at, g)
            at += 1

        stats.append((key, bname, donor.get("name") if donor is not None else "-",
                      contype, conaffinity, zip(names, shapes[key])))

    # ---- re-point explicit contact pairs at the new primitives -------------
    # Expanded pairs from a previous run (<base>_p<N>) are collapsed back to
    # their logical definition first, so re-runs regenerate the same set
    # instead of re-expanding every _p<N> into N more.
    new_pairs, dropped, pair_info, defs = [], [], [], {}
    for p in pairs:
        base = PAIR_IDX_RE.sub("", p.get("name"))
        sides = tuple(
            ("body", geom_owner.get(r, (None, None))[0])
            if geom_owner.get(r, (None, None))[0] in body_to_key else ("geom", r)
            for r in (p.get("geom1"), p.get("geom2")))
        attrs = tuple(sorted((k, v) for k, v in p.attrib.items() if k != "name"))
        if base in defs:
            if defs[base][1:] != (sides, attrs):
                print(f"[MJCF] WARNING pair group '{base}' is inconsistent; "
                      f"keeping the first definition")
            dropped.append(p)                                 # duplicate of its group
            continue
        defs[base] = (p, sides, attrs)
        n_prims = sum(1 for kind, _ in sides if kind == "body")
        if n_prims == 0:
            continue                                          # untouched original
        pair_info.append(f"[MJCF]   pair '{base}' ({p.get('geom1')} <-> "
                         f"{p.get('geom2')}) -> primitive pair(s)")

    for base, (p, sides, attrs) in defs.items():
        choices = [key_prims[body_to_key[v]] if kind == "body" else [v]
                   for kind, v in sides]
        if all(len(c) == 1 for c in choices):
            continue                                          # nothing replaced here
        dropped.append(p)
        for n, (a, b) in enumerate((a, b) for a in choices[0] for b in choices[1]):
            np_ = ET.Element("pair", dict(attrs))
            np_.set("name", f"{base}_p{n}")
            np_.set("geom1", a)
            np_.set("geom2", b)
            new_pairs.append(np_)
    for p in dropped:
        contact.remove(p)
    for p in new_pairs:
        contact.append(p)

    # ---- dangling-pair guard ----------------------------------------------
    known = set(geom_owner) | set(n for ns in key_prims.values() for n in ns)
    for p in contact.findall("pair"):
        for a in ("geom1", "geom2"):
            if p.get(a) not in known:
                raise SystemExit(f"[MJCF] ERROR pair '{p.get('name')}' references "
                                 f"unknown geom '{p.get(a)}'")

    # ---- unreferenced mesh assets (only when visuals are deleted) ---------
    if delete_visual:
        asset = root.find("asset")
        used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
        for m in list(asset.findall("mesh")) if asset is not None else []:
            if m.get("name") not in used:
                asset.remove(m)

    return stats, pair_info, new_pairs


# -------------------------------------------------------------------- CLI --

def main():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xml", default=os.path.join(here, "uav_rail_scene.xml"),
                   help="MJCF scene to patch (default: uav_rail_scene.xml)")
    p.add_argument("--out", default=None,
                   help="output MJCF (default: <xml>_prim.xml)")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite --xml (a .pre_col.bak backup is kept)")
    p.add_argument("--map", action="append", default=[], metavar="KEY=BODY",
                   help="blend group -> MuJoCo body; repeats and overrides "
                        f"the defaults {DEFAULT_KEY_TO_BODY}")
    p.add_argument("--delete-visual-meshes", action="store_true",
                   help="remove the replaced mesh geoms (and unused <mesh> "
                        "assets) instead of keeping them visual-only")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned edits, write nothing")
    args = p.parse_args(rest)

    key_to_body = dict(DEFAULT_KEY_TO_BODY)
    for m in args.map:
        if "=" not in m:
            raise SystemExit(f"[MJCF] ERROR --map expects KEY=BODY, got '{m}'")
        k, b = m.split("=", 1)
        key_to_body[k.strip()] = b.strip()

    if args.in_place and args.out:
        raise SystemExit("[MJCF] ERROR --out and --in-place are mutually exclusive")
    if not os.path.isfile(args.xml):
        raise SystemExit(f"[MJCF] ERROR no such scene: {args.xml}")
    out_path = args.xml if args.in_place else \
        args.out or re.sub(r"\.xml$", "", args.xml) + "_prim.xml"

    shapes = collect_shapes()
    if not shapes:
        raise SystemExit("[BLEND] ERROR no <key>.<Shape> objects in "
                         f"{bpy.data.filepath or '(unsaved blend)'}")
    total = sum(len(v) for v in shapes.values())
    print(f"[BLEND] {bpy.data.filepath}")
    for key in sorted(shapes):
        print(f"[BLEND]   {key}: {len(shapes[key])} shape(s) "
              f"({sum(g['type'] == 'box' for g in shapes[key])} box / "
              f"{sum(g['type'] == 'cylinder' for g in shapes[key])} cylinder)")
    print(f"[BLEND] {total} shape(s) in {len(shapes)} group(s)")

    tree = load_tree(args.xml)
    stats, pair_info, added = patch_scene(tree, key_to_body, shapes,
                                          args.delete_visual_meshes, args.dry_run)

    print(f"[MJCF] {args.xml}")
    for key, bname, donor, ct, ca, prims in stats:
        print(f"[MJCF]   {key} -> body '{bname}'  "
              f"(flags from '{donor}': contype={ct} conaffinity={ca})")
        for name, s in prims:
            e = s["euler"]
            print(f"[MJCF]     {name}: {s['type']} size=({_fmt(s['size'])}) "
                  f"pos=({_fmt(s['pos'])}) "
                  f"rotXYZ=({math.degrees(e[0]):.2f},"
                  f"{math.degrees(e[1]):.2f},{math.degrees(e[2]):.2f})deg")
    for line in pair_info:
        print(line)
    if added:
        print(f"[MJCF]   e.g. {added[0].get('name')}: "
              f"{added[0].get('geom1')} <-> {added[0].get('geom2')}")

    if args.dry_run:
        print("[MJCF] dry run -- nothing written")
        return
    if args.in_place and not os.path.exists(args.xml + ".pre_col.bak"):
        with open(args.xml, "rb") as fsrc, open(args.xml + ".pre_col.bak", "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"[MJCF] backup: {args.xml}.pre_col.bak")
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode")
    print(f"[MJCF] wrote {out_path}")
    if not args.in_place:
        print(f"[MJCF] run:  python {os.path.join(here, 'uav_sim.py')} --model {out_path}")


if __name__ == "__main__":
    main()

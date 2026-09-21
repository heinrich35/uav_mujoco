"""Typed collision export: Blender boxes/cylinders -> MuJoCo, replacing ONLY
box & cylinder collisions (based on export_collision_blend_mujoco.py).

Delta vs the base script, per user spec:
  1. TYPE-SCOPED replacement -- on each mapped body, only collidable geoms of
     type box / cylinder are replaced by the blend primitives. Every OTHER
     collision type (sphere, capsule, mesh, hfield, ...) is LEFT IN PLACE,
     untouched and still collidable.
  2. --rename-in-blend OLD=NEW -- renames blend object groups (e.g.
     ``shahed`` -> ``veh``) so the generated geoms match OUR sim's naming
     (``veh_col_*``), then saves the blend (a .bak backup is kept).
  3. --validate -- re-opens the written scene and checks every ``<key>_col_*``
     prim against the blend shapes (count / size / pos / quat, tol 1e-3) and,
     if the reference scene ``uav_rail_scene_prim.xml`` sits next to it,
     against that too (group aliases from the rename map).

Shared with the base script: naming convention ``<key>.<Shape>[.<idx>]`` per
collision shape (Cube / Cylinder), group -> body mapping (defaults below,
``--map key=body`` to override), verbatim blend pos/quat, half-extent sizes,
stable ``<key>_col_<NNN>`` geom names, idempotent re-runs, banner comments,
.pre_col.bak backups, --dry-run.

Run headless::

    blender --background assets/shahed-136/rail_cart_shahed_collision.blend \\
            --python scripts/mujoco/export_collision_blend_mujoco_typed.py -- \\
            [--xml PATH] [--out PATH | --in-place] [--map key=body ...]
            [--rename-in-blend OLD=NEW] [--validate] [--delete-visual-meshes]
            [--dry-run]
"""

import os
import re
import sys
import math
import shutil
import argparse
import xml.etree.ElementTree as ET

import bpy


# Shape tokens we know how to export.  The token must appear as a name segment
# (between dots).  Extend to Sphere / Capsule when the blend grows them --
# they would then also JOIN the replaceable set.
SHAPE_TOKENS = ("Cube", "Cylinder")
REPLACEABLE_TYPES = {"box", "cylinder"}

PRIM_TAG = "_col_"
PAIR_IDX_RE = re.compile(r"_p\d+$")

# blend group -> MuJoCo <body>.  'veh' is the renamed spelling of 'shahed'
# (--rename-in-blend shahed=veh) so generated geoms read veh_col_* like our
# veh3d sim.  The original spelling stays mapped for un-renamed blends.
DEFAULT_KEY_TO_BODY = {
    "veh": "vehicle",
    "shahed": "uav",
    "rail": "base_link",
    "cart": "launch_cart",
}

PRIM_RGBA = "0.2 0.9 0.4 0.35"
PRIM_GROUP = "0"
VISUAL_GROUP = "1"


# ---------------------------------------------------------------- Blender --

def _parse_shape_obj(name):
    """Split ``<key>.<Shape>[.<idx>]`` into ``(key, shape_token)``."""
    segs = name.split(".")
    for i, seg in enumerate(segs):
        if seg in SHAPE_TOKENS:
            key = ".".join(segs[:i])
            return (key or None), seg
    return None, None


def _trailing_int(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _rot_quat_wxyz(obj):
    q = obj.matrix_basis.to_quaternion()
    return (float(q.w), float(q.x), float(q.y), float(q.z))


def rename_in_blend(old, new):
    """Rename object group OLD -> NEW (shahed -> veh) and save the blend.

    Only the group PREFIX changes (shahed.Cube.001 -> veh.Cube.001); the
    shape token and trailing index are preserved, so --validate aliases keep
    working.  The blend on disk is backed up to <file>.pre_rename.bak first.
    """
    blend = bpy.data.filepath
    if blend:
        bak = blend + ".pre_rename.bak"
        if not os.path.exists(bak):
            shutil.copyfile(blend, bak)
            print(f"[BLEND] backup: {bak}")
    renamed = []
    for obj in bpy.data.objects:
        segs = obj.name.split(".")
        if segs and segs[0] == old:
            obj.name = ".".join([new] + segs[1:])
            renamed.append(obj.name)
    if not renamed:
        print(f"[BLEND] no objects starting '{old}.' -- nothing to rename")
        return renamed
    if blend:
        bpy.ops.wm.save_mainfile()
        print(f"[BLEND] saved {blend} with {len(renamed)} renamed object(s)")
    preview = ", ".join(renamed[:5]) + ("..." if len(renamed) > 5 else "")
    print(f"[BLEND] renamed: {preview}")
    return renamed


def collect_shapes(align_world=False):
    """Every convention-named mesh object -> {key: [shape dict, ...]}.

    Default (faithful OBB): size = Item > Dimensions / 2 in the object's OWN
    axes, rotation carried as quat — a rotated box exports exactly as it
    sits. Item > Dimensions is in LOCAL axes: if the object was rotated and
    then edited, the local y/z can read "switched" vs the world view; either
    Apply Rotation in Blender, or pass --align-world to export every box as
    its WORLD-axis-aligned bounding box (identity quat — rotated boxes become
    their AABB, which fattens thin rotated boxes like the rail posts).
    """
    from mathutils import Vector
    groups = {}
    skipped = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        key, shape = _parse_shape_obj(obj.name)
        if not key or not shape:
            skipped.append(obj.name)
            continue
        if shape == "Cube" and align_world:
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            lo = [min(c[i] for c in corners) for i in range(3)]
            hi = [max(c[i] for c in corners) for i in range(3)]
            ext = [hi[i] - lo[i] for i in range(3)]
            if min(ext) <= 1e-9:
                print(f"[BLEND] WARNING {obj.name}: zero extent -- skipped")
                continue
            geom = {"name": obj.name, "type": "box",
                    "size": tuple(e / 2.0 for e in ext),
                    "pos": tuple((hi[i] + lo[i]) / 2.0 for i in range(3)),
                    "quat": (1.0, 0.0, 0.0, 0.0)}
        else:
            dx, dy, dz = (float(v) for v in obj.dimensions)
            if min(dx, dy, dz) <= 1e-9:
                print(f"[BLEND] WARNING {obj.name}: zero extent -- skipped")
                continue
            geom = {"name": obj.name, "quat": _rot_quat_wxyz(obj),
                    "pos": tuple(float(v) for v in obj.location)}
            if shape == "Cube":
                geom.update(type="box", size=(dx / 2.0, dy / 2.0, dz / 2.0))
            else:  # Cylinder: Z-axis mesh; size is (radius, HALF length)
                geom.update(type="cylinder",
                            size=((dx + dy) / 4.0, dz / 2.0))
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
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _geom_children(el):
    return [c for c in el if c.tag == "geom"]


def _drop_with_banner(parent, el, key):
    children = list(parent)
    i = children.index(el)
    if i > 0 and children[i - 1].tag is ET.Comment and key in (children[i - 1].text or ""):
        parent.remove(children[i - 1])
    parent.remove(el)


def _prim_names(key, geoms, taken):
    names = []
    for g in geoms:
        base = f"{key}{PRIM_TAG}{_trailing_int(g['name']):03d}"
        name, n = base, 0
        while name in taken:
            n += 1
            name = f"{base}{chr(96 + n)}"
        taken.add(name)
        names.append(name)
    return names


def patch_scene(tree, key_to_body, shapes, delete_visual, dry):
    """Typed rewrite: replace ONLY collidable box/cylinder geoms per mapped
    body; every other collidable type stays exactly as it is."""
    root = tree.getroot()
    body_els = {b.get("name"): b for b in root.iter("body")}

    geom_owner = {}
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for g in worldbody.findall("geom"):
            geom_owner[g.get("name")] = (None, g)
    for bname, bel in body_els.items():
        for g in _geom_children(bel):
            geom_owner[g.get("name")] = (bname, g)

    contact = root.find("contact")
    if contact is None:                 # scenes without explicit pairs: create
        contact = ET.Element("contact")  # the section so re-pointing works
        root.append(contact)
    pairs = list(contact.findall("pair"))
    pair_refs = {p.get(a) for p in pairs for a in ("geom1", "geom2")}

    body_to_key, planned, errors = {}, [], []
    for key in sorted(shapes):
        bname = key_to_body.get(key)
        if bname is None:
            continue                      # unmapped group: ignore silently
        if bname not in body_els:
            errors.append(f"--map {key}={bname}: no body '{bname}' in the scene")
        elif bname in body_to_key and body_to_key[bname] != key:
            errors.append(f"body '{bname}' is the target of both "
                          f"'{body_to_key[bname]}' and '{key}'")
        else:
            body_to_key[bname] = key
            planned.append((key, bname, body_els[bname]))
    for key, bname in sorted(key_to_body.items()):
        if key in ("veh", "shahed") and key not in shapes:
            continue                      # renamed-away spelling: fine
        if key not in shapes and not dry:
            errors.append(f"mapped group '{key}' has no shapes in the blend")
    if errors:
        for e in errors:
            print(f"[MJCF] ERROR {e}")
        raise SystemExit(2)

    # ---- per body: replace only collidable box/cylinder geoms --------------
    key_prims, replaced_names, stats = {}, set(), []
    left_report = []
    for key, bname, bel in planned:
        geoms = _geom_children(bel)
        old_prims = {g for g in geoms
                     if (g.get("name") or "").startswith(key + PRIM_TAG)}
        fresh = [g for g in geoms if g not in old_prims]

        # TYPE-SCOPED: the replaceable set is collidable box/cylinder only.
        replaceable, left_in_place = [], []
        for g in fresh:
            if g in old_prims or g.get("type") not in REPLACEABLE_TYPES:
                continue
            if g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0":
                continue                  # already visual-only
            replaceable.append(g)
        for g in fresh:
            if g in replaceable:
                continue
            collidable = (g.get("contype", "1") != "0"
                          or g.get("conaffinity", "1") != "0")
            if collidable and g.get("type") not in REPLACEABLE_TYPES:
                left_in_place.append(g)   # sphere / capsule / mesh / ... stays

        # Flags donor: a pair-referenced replaceable first, then any
        # replaceable, then a previous prim of ours.
        donor = next((g for g in replaceable if g.get("name") in pair_refs), None)
        if donor is None:
            donor = next(iter(replaceable), None)
        if donor is None:
            donor = next(iter(old_prims), None)
        contype = donor.get("contype") if donor is not None else None
        conaffinity = donor.get("conaffinity") if donor is not None else None
        density = (donor.get("density") or "0") if donor is not None else "0"

        for g in old_prims:
            _drop_with_banner(bel, g, key)
        replaced = []
        for g in replaceable:             # ONLY these lose their collision
            replaced.append(g)
            if delete_visual:
                bel.remove(g)
            else:
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("group", VISUAL_GROUP)
        for g in left_in_place:           # LEFT IN PLACE: untouched
            left_report.append((bname, g.get("name"), g.get("type")))

        replaced_names |= {g.get("name") for g in replaced}

        taken = {g.get("name") for g in _geom_children(bel)}
        names = _prim_names(key, shapes[key], taken)
        key_prims[key] = names

        children = list(bel)
        last = max((children.index(g) for g in _geom_children(bel)),
                   default=len(children) - 1)
        at = last + 1
        for name, sh in zip(names, shapes[key]):
            bel.insert(at, ET.Comment(f" {sh['name']} ({sh['type']}) "))
            at += 1
            g = ET.Element("geom", {"name": name, "type": sh["type"],
                                    "size": _fmt(sh["size"]),
                                    "pos": _fmt(sh["pos"])})
            if not _is_identity(sh["quat"]):
                g.set("quat", _fmt(sh["quat"]))
            if contype is not None:
                g.set("contype", contype)
                g.set("conaffinity", conaffinity)
            g.set("density", density)
            g.set("rgba", PRIM_RGBA)
            g.set("group", PRIM_GROUP)
            bel.insert(at, g)
            at += 1

        stats.append((key, bname, donor.get("name") if donor is not None else "-",
                      len(replaced), len(left_in_place),
                      zip(names, shapes[key])))

    # ---- pairs: re-point only sides that referenced a REPLACED geom -------
    # (left-in-place collisions keep their exact original pair references)
    new_pairs, dropped, defs = [], [], {}
    for p in pairs:
        base = PAIR_IDX_RE.sub("", p.get("name"))
        sides = []
        for r in (p.get("geom1"), p.get("geom2")):
            owner = geom_owner.get(r, (None, None))[0]
            if r in replaced_names:
                sides.append(("replaced", body_to_key[owner]))
            else:
                sides.append(("geom", r))
        attrs = tuple(sorted((k, v) for k, v in p.attrib.items() if k != "name"))
        if base in defs:
            dropped.append(p)
            continue
        defs[base] = (p, sides, attrs)
    for base, (p, sides, attrs) in defs.items():
        choices = [key_prims[k] if kind == "replaced" else [v]
                   for kind, v in sides]
        if all(len(c) == 1 for c in choices):
            continue
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

    known = set(geom_owner) | set(n for ns in key_prims.values() for n in ns)
    for p in contact.findall("pair"):
        for a in ("geom1", "geom2"):
            if p.get(a) not in known:
                raise SystemExit(f"[MJCF] ERROR pair '{p.get('name')}' references "
                                 f"unknown geom '{p.get(a)}'")

    if delete_visual:
        asset = root.find("asset")
        used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
        for m in list(asset.findall("mesh")) if asset is not None else []:
            if m.get("name") not in used:
                asset.remove(m)

    return stats, left_report, new_pairs


# ---------------------------------------------------------------- validate --

def _trailing_int(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def validate_scene(path, key_to_body, shapes, alias, tol=1e-3,
                   align_world=False):
    """Check <key>_col_* prims against the blend shapes (and, when a
    reference scene sits next to this script, against that too)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    by_name = {g.get("name"): g for g in root.iter("geom") if g.get("name")}

    failures = 0
    print(f"[VALIDATE] {path} vs {os.path.basename(os.path.dirname(''))}"
          f"blend shapes (tol {tol})")
    for key in sorted(k for k in shapes if k in key_to_body):
        prims = sorted((n for n in by_name
                        if n.startswith(key + PRIM_TAG)), key=_trailing_int)
        want = {(_trailing_int(s["name"])): s for s in shapes[key]}
        got = {_trailing_int(n): by_name[n] for n in prims}
        if set(got) != set(want):
            print(f"[VALIDATE] FAIL {key}: prim indices {sorted(got)} != "
                  f"blend {sorted(want)}")
            failures += 1
            continue
        worst = 0.0
        for idx in sorted(want):
            g, s = got[idx], want[idx]
            size = [float(x) for x in g.get("size").split()]
            if g.get("type") != s["type"]:
                print(f"[VALIDATE] FAIL {key}[{idx}] type {g.get('type')} "
                      f"!= {s['type']}")
                failures += 1
                continue
            worst = max(worst, max(abs(a - b) for a, b in zip(size, s["size"])))
            worst = max(worst, max(abs(a - b) for a, b in
                                   zip([float(x) for x in g.get("pos").split()],
                                       s["pos"])))
            q1 = [float(x) for x in (g.get("quat") or "1 0 0 0").split()]
            worst = max(worst, max(abs(a - b) for a, b in zip(q1, s["quat"])))
        status = "PASS" if worst <= tol else "FAIL"
        failures += status == "FAIL"
        print(f"[VALIDATE] {status} {key}: {len(prims)} prim(s), "
              f"max deviation {worst:.2e}")

    # reference comparison (group aliases from the rename: veh <-> shahed)
    # reference comparison: per-group world ENVELOPE (boxes may legitimately
    # differ box-by-box — the old reference carries local-dims + quats; the
    # new export is world-axis-aligned)
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "uav_rail_scene_prim.xml")
    if os.path.isfile(ref) and alias:
        old, new = alias
        rroot = ET.parse(ref).getroot()
        rname = {g.get("name"): g for g in rroot.iter("geom") if g.get("name")}
        print(f"[VALIDATE] vs reference {os.path.basename(ref)} "
              f"(alias {old}->{new}, "
              f"{'world envelope' if align_world else 'per-box OBB'})")
        for key in sorted(k for k in shapes if k in key_to_body):
            ref_key = old if key == new else key
            rnames = sorted((n for n in rname
                             if n.startswith(ref_key + PRIM_TAG)),
                            key=_trailing_int)
            pnames = sorted((n for n in by_name
                             if n.startswith(key + PRIM_TAG)),
                            key=_trailing_int)
            if not rnames or not pnames:
                continue
            if align_world:
                # world-AABB mode: compare the per-group envelope (rotated
                # boxes legitimately differ box-by-box vs the old OBB ref)
                def env(gs):
                    lo = [min(float(g.get("pos").split()[a]) -
                              float(g.get("size").split()[a]) for g in gs)
                          for a in range(3)]
                    hi = [max(float(g.get("pos").split()[a]) +
                              float(g.get("size").split()[a]) for g in gs)
                          for a in range(3)]
                    return lo, hi
                rlo, rhi = env([rname[n] for n in rnames])
                plo, phi = env([by_name[n] for n in pnames])
                devs = [abs(a - b) for a, b in zip(rlo, plo)] + \
                       [abs(a - b) for a, b in zip(rhi, phi)]
                worst = max(devs)
                status = "DRIFT" if worst > 0.05 else "PASS"
                print(f"[VALIDATE] {status} {key} vs {ref_key}: envelope max "
                      f"dev {worst:.2e} m"
                      + ("" if status == "PASS" else "  (intentional edit?)"))
                continue
            # faithful OBB mode: strict per-box comparison (same export type
            # as the reference), max deviation over size + pos
            same = (len(pnames) == len(rnames) and all(
                _trailing_int(a) == _trailing_int(b)
                for a, b in zip(pnames, rnames)))
            worst = 0.0
            if same:
                for rn, nn in zip(rnames, pnames):
                    rg, ng = rname[rn], by_name[nn]
                    for attr in ("size", "pos"):
                        worst = max(worst, max(
                            abs(a - b) for a, b in zip(
                                [float(x) for x in rg.get(attr).split()],
                                [float(x) for x in ng.get(attr).split()])))
            # informational: the reference is the OLD baseline; intentional
            # blend edits show up here as drift. The hard gate is the
            # blend-vs-output comparison above.
            status = "DRIFT" if not (same and worst <= tol) else "PASS"
            print(f"[VALIDATE] {status} {key} vs {ref_key}: per-box OBB, "
                  f"max dev {worst:.2e}"
                  + ("" if status == "PASS" else "  (intentional edit?)"))
    return failures


# -------------------------------------------------------------------- CLI --

def main():
    argv = sys.argv
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--xml", default=os.path.join(here, "uav_rail_scene.xml"),
                   help="MJCF scene to patch (default: uav_rail_scene.xml)")
    p.add_argument("--out", default=None,
                   help="output MJCF (default: <xml>_typed.xml)")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite --xml (a .pre_col.bak backup is kept)")
    p.add_argument("--map", action="append", default=[], metavar="KEY=BODY",
                   help="blend group -> MuJoCo body; overrides the defaults "
                        f"{DEFAULT_KEY_TO_BODY}")
    p.add_argument("--rename-in-blend", action="append", default=[],
                   metavar="OLD=NEW",
                   help="rename a blend object group (e.g. shahed=veh) and "
                        "save the blend; repeatable")
    p.add_argument("--validate", action="store_true",
                   help="after writing, re-check every prim against the "
                        "blend shapes and the reference prim scene")
    p.add_argument("--delete-visual-meshes", action="store_true",
                   help="remove replaced mesh geoms instead of keeping them "
                        "visual-only")
    p.add_argument("--align-world", action="store_true",
                   help="export every box as its WORLD-axis-aligned bounding "
                        "box (identity quat) instead of the faithful "
                        "local-dims + rotation")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned edits, write nothing")
    args = p.parse_args(rest)

    key_to_body = dict(DEFAULT_KEY_TO_BODY)
    for m in args.map:
        if "=" not in m:
            raise SystemExit(f"[MJCF] ERROR --map expects KEY=BODY, got '{m}'")
        k, b = m.split("=", 1)
        key_to_body[k.strip()] = b.strip()

    alias = None
    for m in args.rename_in_blend:
        if "=" not in m:
            raise SystemExit("[MJCF] ERROR --rename-in-blend expects OLD=NEW")
        old, new = m.split("=", 1)
        rename_in_blend(old.strip(), new.strip())
        alias = (old.strip(), new.strip())
        if old in key_to_body:                     # keep the body target
            key_to_body[new] = key_to_body.pop(old)

    if args.in_place and args.out:
        raise SystemExit("[MJCF] ERROR --out and --in-place are mutually exclusive")
    if not os.path.isfile(args.xml):
        raise SystemExit(f"[MJCF] ERROR no such scene: {args.xml}")
    out_path = args.xml if args.in_place else \
        args.out or re.sub(r"\.xml$", "", args.xml) + "_typed.xml"

    shapes = collect_shapes(align_world=args.align_world)
    if not shapes:
        raise SystemExit("[BLEND] ERROR no <key>.<Shape> objects in "
                         f"{bpy.data.filepath or '(unsaved blend)'}")
    total = sum(len(v) for v in shapes.values())
    print(f"[BLEND] {bpy.data.filepath}")
    for key in sorted(shapes):
        print(f"[BLEND]   {key}: {len(shapes[key])} shape(s) "
              f"({sum(g['type'] == 'box' for g in shapes[key])} box / "
              f"{sum(g['type'] == 'cylinder' for g in shapes[key])} cylinder)")

    tree = load_tree(args.xml)
    stats, left_report, added = patch_scene(tree, key_to_body, shapes,
                                            args.delete_visual_meshes,
                                            args.dry_run)

    print(f"[MJCF] {args.xml}")
    for key, bname, donor, n_repl, n_left, prims in stats:
        print(f"[MJCF]   {key} -> body '{bname}'  (flags from '{donor}'): "
              f"{n_repl} box/cyl collision(s) replaced, "
              f"{n_left} other collision(s) LEFT IN PLACE")
        for name, sh in prims:
            print(f"[MJCF]     {name}: {sh['type']} size=({_fmt(sh['size'])}) "
                  f"pos=({_fmt(sh['pos'])})")
    for bname, gname, gtype in left_report:
        print(f"[MJCF]   LEFT IN PLACE: {bname}/{gname} ({gtype})")
    if added:
        print(f"[MJCF]   e.g. pair {added[0].get('name')}: "
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

    if args.validate:
        fails = validate_scene(out_path, key_to_body, shapes, alias,
                               align_world=args.align_world)
        if fails:
            raise SystemExit(f"[VALIDATE] {fails} failure(s)")
        print("[VALIDATE] ALL PRIM CHECKS PASSED")


if __name__ == "__main__":
    main()

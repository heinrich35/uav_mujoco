#!/usr/bin/env python3
"""
Stage 0 — Synthetic training data generation for duck/swan detection pipeline.

Three capture modes:
  Mode A — Isolated single object (feeds Stage 2 classifier)
  Mode B — Full pond scene      (feeds Stage 1 detector)
  Mode C — Distractor-only negs  (anti-hallucination)

Usage:
  ./isaaclab.sh -p scripts/pond/generate_training_data.py
"""

import argparse, json, math, os, random, shutil, struct as _struct, time, zlib
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--mode_a_per_object", type=int, default=500,
                    help="Frames per object in Mode A (isolated single-object)")
parser.add_argument("--mode_a_objects", type=str, default=None,
                    help="Comma-separated object names for Mode A (default: all 6). "
                         "Example: --mode_a_objects swan,rock  to only generate swan+rock frames")
parser.add_argument("--mode_b_count", type=int, default=1000,
                    help="Total frames for Mode B (full pond scene)")
parser.add_argument("--mode_c_count", type=int, default=1000,
                    help="Total frames for Mode C (distractor-only negatives)")
parser.add_argument("--resume", action="store_true",
                    help="Skip modes whose output dirs already have annotation JSON")
parser.add_argument("--force", action="store_true",
                    help="Delete existing output before starting (default: True; "
                         "use with --resume to keep existing)")
parser.add_argument("--slice", type=str, default=None,
                    help="Worker slice for parallel runs: 'I/N' (e.g. '0/2' runs first half)")
parser.add_argument("--no_reset", action="store_true", default=True,
                    help="Avoid sim.reset() between modes — prevents GPU reinit stalls "
                         "(default: True, use --reset to re-enable resets)")
parser.add_argument("--reset", action="store_true", dest="do_reset",
                    help="Call sim.reset() between modes (legacy behaviour)")
parser.add_argument("--progress_interval", type=int, default=25,
                    help="Print progress every N frames in Mode B/C (default: 25)")
parser.add_argument("--mode_a_only", action="store_true",
                    help="Generate Mode A only, skip B/C")
parser.add_argument("--mode_b_only", action="store_true",
                    help="Generate Mode B only, skip A/C")
parser.add_argument("--mode_c_only", action="store_true",
                    help="Generate Mode C only, skip A/B")
parser.add_argument("--mode_bc_only", action="store_true",
                    help="Generate Mode B/C only, skip A (requires pre-existing Mode A data)")
parser.add_argument("--coco_only", action="store_true",
                    help="Skip generation — only rebuild COCO annotations from existing "
                         "PNGs/bboxes (use after parallel workers finish)")
parser.add_argument("--no_coco", action="store_true",
                    help="Skip per-mode COCO building (for parallel workers; final "
                         "worker or --coco_only run rebuilds all COCO afterwards)")
parser.add_argument("--fast", action="store_true",
                    help="Disable physics on spawned objects — renders only, "
                         "much faster (~5s/frame instead of ~30s)")
args = parser.parse_args()
app = AppLauncher(args)
sim_app = app.app

import omni.replicator.core as rep
from isaacsim.core.api.simulation_context import SimulationContext
from omni.usd import get_context
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
REPO = Path("/home/heinz/isaaclab_uav")
USD_BASE = REPO / "assets/pond/glb/usd"
OUTPUT_ROOT = REPO / "outputs" / "training_data"

# --- Frame counts (from CLI; defaults: 500/1000/1000) ---
MODE_A_PER_OBJECT = args.mode_a_per_object
MODE_B = args.mode_b_count
MODE_C = args.mode_c_count

# --- Slice / parallel worker ---
_SLICE_IDX = 0
_SLICE_TOTAL = 1
if args.slice:
    parts = args.slice.split("/")
    if len(parts) != 2:
        raise SystemExit("--slice must be 'I/N' (e.g. '0/2')")
    _SLICE_IDX = int(parts[0])
    _SLICE_TOTAL = int(parts[1])
    if _SLICE_IDX < 0 or _SLICE_IDX >= _SLICE_TOTAL:
        raise SystemExit(f"--slice worker ID {_SLICE_IDX} out of range [0, {_SLICE_TOTAL - 1}]")

# --- Timing helper ---
_TIMERS: dict[str, float] = {}
def _tick(name: str):
    _TIMERS[name] = time.time()
def _tock(name: str, label: str = "") -> float:
    """Return elapsed seconds since _tick(name); optionally print with label."""
    elapsed = time.time() - _TIMERS.get(name, time.time())
    if label:
        print(f"  [TIMER] {label}: {elapsed:.1f}s")
    return elapsed

# --- Object definitions ---
OBJECTS = {
    "duck":    {"usd": str(USD_BASE / "duck" / "duck_articulation.usd"),   "label": "duck"},
    "swan":    {"usd": str(USD_BASE / "swan" / "swan_articulation.usd"),   "label": "swan"},
    "log":     {"usd": str(USD_BASE / "log" / "log.usd"),                  "label": "log"},
    "lilypad": {"usd": str(USD_BASE / "lilypad" / "lilypad.usd"),          "label": "lilypad"},
    "turtle":  {"usd": str(USD_BASE / "turtle" / "turtle.usd"),            "label": "turtle"},
    "rock":    {"usd": str(USD_BASE / "rock" / "rock.usd"),                "label": "rock"},
}
TARGETS = ["duck", "swan"]
DISTRACTORS = ["log", "lilypad", "turtle", "rock"]
# Mode A: capture each object in isolation, one session per object
MODE_A_OBJECTS = ["duck", "swan", "turtle", "lilypad", "log", "rock"]

# --- COCO class mapping ---
# Six fine-grained classes (no "other" collapse)
FINE_TO_COCO = {
    "duck": 0, "swan": 1, "turtle": 2,
    "lilypad": 3, "log": 4, "rock": 5,
}
COCO_CATEGORIES = [
    {"id": 0, "name": "duck"},
    {"id": 1, "name": "swan"},
    {"id": 2, "name": "turtle"},
    {"id": 3, "name": "lilypad"},
    {"id": 4, "name": "log"},
    {"id": 5, "name": "rock"},
]

# --- Placement ---
# Y range is relative to camera at (0, -6, 0.35):  y=-2 → 4m from camera, y=9 → 15m
# X range is clamped per-object based on Y distance to stay within camera FOV.
PLACEMENT_ZONE = {"x": (-6.0, 6.0), "y": (-4.0, 9.0)}
WATER_Z = (-0.03, -0.01)

# --- Object randomization ---
YAW_RANGE = (0.0, 360.0)
PITCH_RANGE = (-10.0, 10.0)
SCALE_RANGE = (0.88, 1.12)
# Per-object scale overrides (prevents logs/rocks from dominating the scene)
OBJECT_SCALE = {
    "log":  (0.70, 0.95),
    "rock": (0.70, 0.95),
}

# --- Camera (static gosling viewpoint for Modes B/C) ---
CAMERA_POS = (0.0, -6.0, 0.35)
CAMERA_LOOK_AT = (0.0, 6.0, -0.02)
# --- Camera for Mode A (gosling at origin, looking forward +X) ---
CAMERA_A_POS = (0.0, 0.0, 0.35)
CAMERA_A_LOOK_AT = (5.0, 0.0, -0.02)
FOCAL_LENGTH = 24.0
RESOLUTION = (640, 480)
CLIPPING_RANGE = (0.1, 200.0)

# --- Object placement for Mode A ---
OBJ_A_DISTANCE = (1.5, 8.0)   # distance range in front of gosling (m) — wide range
OBJ_A_XY_JITTER = 2.0         # ± random offset on Y (clamped to FOV)
OBJ_A_PITCH = (-20.0, 20.0)   # pitch variation for Mode A — more tilt diversity
# Trapezoidal FOV: visible half-extent at distance d = d * tan(FOV/2)
# Default USD aperture: 20.955mm horiz, 15.291mm vert, focal=24mm
_CAM_A_HFOV = 2 * math.atan(20.955 / (2 * FOCAL_LENGTH))
_CAM_A_VFOV = 2 * math.atan(15.291 / (2 * FOCAL_LENGTH))
# --- Lighting (matches pond_env_cfg.py exactly) ---
DOME_INTENSITY = 500
DOME_COLOR = (0.85, 0.90, 1.0)
SUN_INTENSITY = 2000
SUN_COLOR = (1.0, 0.95, 0.85)
# Mode A: isolated object shots — moderate boost over base (no water bounce fill)
DOME_INTENSITY_A = (250, 500)
# Mode B/C: full pond scene with subtle variation around base 100
DOME_INTENSITY_BC = (70, 200)
# Subtle colour temperature variation (± these deltas per channel)
DOME_COLOR_DELTA = 0.06

# --- Water ---
WATER_SCALE = 30

# ═══════════════════════════════════════════════════════════════════════════
# Infrastructure (shared across all modes)
# ═══════════════════════════════════════════════════════════════════════════
sim = SimulationContext(
    physics_dt=1.0 / 30.0, rendering_dt=1.0 / 30.0,
    sim_params={"use_gpu": True, "use_gpu_pipeline": True, "use_fabric": True},
    backend="torch", device="cuda:0",
)

# Camera for Modes B/C (gosling behind pond, looking across)
camera = rep.create.camera(
    position=CAMERA_POS,
    look_at=CAMERA_LOOK_AT,
    focal_length=FOCAL_LENGTH,
    clipping_range=CLIPPING_RANGE,
)
render_product = rep.create.render_product(camera, resolution=RESOLUTION)
rp_path = render_product if isinstance(render_product, str) else render_product.path

# Camera for Mode A (gosling at origin, looking forward +X)
camera_a = rep.create.camera(
    position=CAMERA_A_POS,
    look_at=CAMERA_A_LOOK_AT,
    focal_length=FOCAL_LENGTH,
    clipping_range=CLIPPING_RANGE,
)
render_product_a = rep.create.render_product(camera_a, resolution=RESOLUTION)
rp_path_a = render_product_a if isinstance(render_product_a, str) else render_product_a.path

# Annotators for Mode B/C
annotators = {}
for name in ["rgb", "bounding_box_2d_tight"]:
    dev = "cpu" if name == "bounding_box_2d_tight" else "cuda"
    ann = rep.AnnotatorRegistry.get_annotator(name, device=dev)
    ann.attach(rp_path)
    annotators[name] = ann

# Annotators for Mode A
annotators_a = {}
for name in ["rgb", "bounding_box_2d_tight"]:
    dev = "cpu" if name == "bounding_box_2d_tight" else "cuda"
    ann = rep.AnnotatorRegistry.get_annotator(name, device=dev)
    ann.attach(rp_path_a)
    annotators_a[name] = ann

# Scene
from isaacsim.core.utils.prims import define_prim
_water_path = "/World/waterSurface"
define_prim(_water_path, "Cube")  # placeholder — material applied below after stage init
dome = rep.create.light(light_type="dome", intensity=DOME_INTENSITY, color=DOME_COLOR)
# Sun light
rep.create.light(light_type="distant", intensity=SUN_INTENSITY, color=SUN_COLOR,
                 position=(20.0, 10.0, 30.0))

# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════
stage = get_context().get_stage()

# Apply water material matching pond env (moved here because stage needed)
_water = stage.GetPrimAtPath(_water_path)
if _water and _water.IsValid():
    from pxr import UsdGeom, UsdShade
    from isaacsim.core.utils.semantics import add_labels
    UsdGeom.XformCommonAPI(_water).SetScale((WATER_SCALE, WATER_SCALE, 0.02))
    add_labels(_water, labels=["water"], instance_name="class")
    _mat_path = f"{_water_path}/material"
    _mat = UsdShade.Material.Define(stage, _mat_path)
    _sh = UsdShade.Shader.Define(stage, f"{_mat_path}/Shader")
    _sh.CreateIdAttr("UsdPreviewSurface")
    _sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.5, 0.8))
    _sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    _sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
    _sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.6)
    _mat.CreateSurfaceOutput().ConnectToSource(_sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(_water).Bind(_mat)

def _silent_cleanup():
    """Remove dynamic objects via stage.RemovePrim().
    Safe for Mode A (1-2 prims per frame). Mode B/C skip this to avoid
    PhysX fabric errors when removing many prims at once.
    """
    replicator = stage.GetPrimAtPath("/Replicator")
    if not replicator or not replicator.IsValid():
        return
    to_remove = []
    for child in replicator.GetAllChildren():
        name = child.GetName()
        if name.startswith("Ref_Xform") or name.startswith("ModeA_"):
            to_remove.append(child.GetPath())
    for p in to_remove:
        try:
            stage.RemovePrim(p)
        except Exception:
            pass

def _find_dome_prim():
    """Find the DomeLight prim in the stage."""
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.GetTypeName() == "DomeLight":
            return prim
    return None

def _reset_dome_lighting(dome_prim):
    """Reset dome light to base pond-environment values."""
    if dome_prim is None:
        return
    try:
        dome_prim.GetAttribute("inputs:intensity").Set(DOME_INTENSITY)
        dome_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*DOME_COLOR))
    except Exception:
        pass


def _randomize_lighting(dome_prim, intensity_range=None, color_delta=None):
    """Randomize dome light intensity and colour temperature.

    When *intensity_range* is None a subtle default range around the base
    pond intensity is used so the call is never a silent no-op.
    """
    if dome_prim is None:
        return
    # Intensity
    lo, hi = intensity_range if intensity_range is not None else (70, 200)
    intensity = random.uniform(lo, hi)
    attr_i = dome_prim.GetAttribute("inputs:intensity")
    if attr_i:
        attr_i.Set(intensity)
    # Colour temperature (subtle per-channel jitter)
    delta = color_delta if color_delta is not None else DOME_COLOR_DELTA
    r = max(0.0, min(1.0, DOME_COLOR[0] + random.uniform(-delta, delta)))
    g = max(0.0, min(1.0, DOME_COLOR[1] + random.uniform(-delta, delta)))
    b = max(0.0, min(1.0, DOME_COLOR[2] + random.uniform(-delta, delta)))
    attr_c = dome_prim.GetAttribute("inputs:color")
    if attr_c:
        attr_c.Set(Gf.Vec3f(r, g, b))

def _spawn_object(obj_name, position, yaw_deg, pitch_deg=0.0, scale_factor=1.0,
                  name=None, parent=None):
    """Spawn one object with semantic label via rep.create.from_usd.

    Returns the expected prim path (parent/name) so callers can track it
    for later repositioning without scanning the stage.
    """
    cfg = OBJECTS[obj_name]
    if obj_name in OBJECT_SCALE:
        sx = sy = sz = random.uniform(*OBJECT_SCALE[obj_name])
    else:
        sx = sy = sz = scale_factor
    kwargs = dict(
        usd=cfg["usd"],
        semantics=[("class", cfg["label"])],
        position=position,
        rotation=(pitch_deg, 0.0, yaw_deg),
        scale=(sx, sy, sz),
    )
    if name is not None:
        kwargs["name"] = name
    if parent is not None:
        kwargs["parent"] = parent
    rep.create.from_usd(**kwargs)
    time.sleep(0.01)
    if parent is not None and name is not None:
        return f"{parent}/{name}"
    return None

def _get_anno(name, anno_dict=None):
    """Collect annotator data as numpy array."""
    if anno_dict is None:
        anno_dict = annotators
    try:
        d = anno_dict[name].get_data()
        if d is not None and hasattr(d, "numpy"):
            return np.ascontiguousarray(d.numpy())
        return np.asarray(d) if d is not None else None
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════
        return None

def _write_png(data_uint8, path):
    """Write [H,W,3] or [H,W,4] uint8 array as PNG (pure Python, no PIL)."""
    h, w = data_uint8.shape[:2]
    depth = data_uint8.shape[2] if data_uint8.ndim == 3 else 1
    color_type = {1: 0, 3: 2, 4: 6}[depth]
    def _chunk(ctype, cdata):
        body = ctype + cdata
        return _struct.pack('>I', len(cdata)) + body + _struct.pack('>I', zlib.crc32(body) & 0xffffffff)
    raw = np.ascontiguousarray(data_uint8).tobytes()
    filtered = b''.join(b'\x00' + raw[i * w * depth:(i + 1) * w * depth] for i in range(h))
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(_chunk(b'IHDR', _struct.pack('>IIBBBBB', w, h, 8, color_type, 0, 0, 0)))
        f.write(_chunk(b'IDAT', zlib.compress(filtered)))
        f.write(_chunk(b'IEND', b''))

def _save_frame(frame_idx, output_dir, anno_dict=None, bucket_size=1000):
    """Collect annotator data and save PNG + NPY.

    Files are written to bucket_NNN/ subdirectories (up to *bucket_size*
    frames each) to prevent directory entry saturation on ext4, which
    causes progressive slowdown above ~500 files per directory.
    """
    bucket = frame_idx // bucket_size
    bucket_dir = output_dir / f"bucket_{bucket:03d}"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    rgb = _get_anno("rgb", anno_dict)
    bbox = _get_anno("bounding_box_2d_tight", anno_dict)
    if rgb is not None and rgb.size > 0:
        np.save(str(bucket_dir / f"rgb_{frame_idx:04d}.npy"), rgb)
        try:
            arr = rgb
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                _write_png(arr, str(bucket_dir / f"rgb_{frame_idx:04d}.png"))
        except Exception as e:
            print(f"    [WARN] PNG frame {frame_idx}: {e}")
    if bbox is not None and bbox.size > 0:
        np.save(str(bucket_dir / f"bbox_{frame_idx:04d}.npy"), bbox)
    return bbox

# ═══════════════════════════════════════════════════════════════════════════
# COCO builder
# ═══════════════════════════════════════════════════════════════════════════
def _build_coco(output_dir, mode_name):
    """Scan bbox NPYs, build COCO JSON with 6-class fine-grained labels."""
    coco = {
        "info": {"description": f"Pond dataset — {mode_name}", "mode": mode_name},
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES,
        "semantic_id_map": {},  # global: semId → {label, class}
    }
    ann_id = 0
    # Scan recursively — files are in bucket_NNN/ subdirectories
    bbox_files = sorted(output_dir.glob("**/bbox_*.npy"))
    for bbox_file in bbox_files:
        frame_num = int(bbox_file.stem.split("_")[-1])
        coco["images"].append({
            "id": frame_num,
            "file_name": f"rgb_{frame_num:04d}.png",
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
        })
        try:
            item = np.load(str(bbox_file), allow_pickle=True)
            item = item.item() if isinstance(item, np.ndarray) and item.shape == () else item
            if not isinstance(item, dict):
                continue
            # Format: {'data': structured_array(dtype=[semanticId,x_min,y_min,x_max,y_max,occlusionRatio]),
            #          'info': {'idToLabels': {semId: {'class': label}}}}
            data_arr = item.get("data", None)
            id_to_labels = item.get("info", {}).get("idToLabels", {})
            # Collect all unique semId→label mappings for semantic_id_map
            for sem_id_str, lbl_info in id_to_labels.items():
                sid = int(sem_id_str)
                fine = lbl_info.get("class", "other")
                # Keep fine-grained label; class mirrors label for all 6 types
                coco_class = fine if fine in FINE_TO_COCO else "other"
                coco["semantic_id_map"][str(sid)] = {"label": fine, "class": coco_class}
            if data_arr is None or len(data_arr) == 0:
                continue
            for row in data_arr:
                sem_id = int(row[0])
                xmin, ymin, xmax, ymax = int(row[1]), int(row[2]), int(row[3]), int(row[4])
                label_info = id_to_labels.get(str(sem_id), {})
                fine_label = label_info.get("class", "other")
                cat_id = FINE_TO_COCO.get(fine_label)
                if cat_id is None:          # skip unknown labels (water, …)
                    continue
                w, h = xmax - xmin, ymax - ymin
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": frame_num,
                    "category_id": cat_id,
                    "bbox": [float(xmin), float(ymin), float(w), float(h)],
                    "bbox_mode": "XYWH_ABS",
                    "area": float(w * h),
                    "fine_label": fine_label,
                })
                ann_id += 1
        except Exception as e:
            print(f"    [WARN] COCO parse {bbox_file}: {e}")
    with open(output_dir / f"annotations_{mode_name}.json", "w") as f:
        json.dump(coco, f, indent=2)
    return coco

def _merge_coco(per_mode_data_list, output_path):
    """Combine per-mode COCO dicts with globally unique IDs."""
    merged = {
        "info": {"description": "Pond dataset — all modes merged"},
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES,
        "semantic_id_map": {},
    }
    img_offset, ann_offset = 0, 0
    for mode_coco in per_mode_data_list:
        merged["semantic_id_map"].update(mode_coco.get("semantic_id_map", {}))
        for img in mode_coco["images"]:
            img["id"] += img_offset
            merged["images"].append(img)
        for ann in mode_coco["annotations"]:
            ann["id"] += ann_offset
            ann["image_id"] += img_offset
            merged["annotations"].append(ann)
        img_offset += len(mode_coco["images"])
        ann_offset += len(mode_coco["annotations"])
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    return merged

# ═══════════════════════════════════════════════════════════════════════════
# Mode A — Isolated single-object sessions (6 objects × 30 frames each)
# ═══════════════════════════════════════════════════════════════════════════
def run_mode_a(output_dir, dome_prim, object_list=None, start_frame=0,
               per_obj_remaining=None):
    """Generate Mode A frames.

    If *per_obj_remaining* is a dict mapping obj_name→frames_to_generate,
    uses those per-object counts instead of the global MODE_A_PER_OBJECT.
    This enables per-object resume (each object may have different existing
    counts).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if object_list is None:
        object_list = MODE_A_OBJECTS
    total_objects = len(object_list)

    if per_obj_remaining:
        # Per-object resume: each object has its own target
        total_frames = sum(per_obj_remaining.values())
    else:
        total_frames = total_objects * MODE_A_PER_OBJECT

    worker_tag = f"  [worker {_SLICE_IDX}/{_SLICE_TOTAL}]" if args.slice else ""
    resumed = " (resume)" if start_frame > 0 else ""
    print(f"\n[Mode A] Isolated single-object sessions{resumed} — "
          f"{total_objects} objects, {total_frames} new frames "
          f"from global frame {start_frame}"
          f"{worker_tag}")

    worker_tag = f"  [worker {_SLICE_IDX}/{_SLICE_TOTAL}]" if args.slice else ""
    resumed = " (resume)" if start_frame > 0 else ""
    print(f"\n[Mode A] Isolated single-object sessions{resumed} — "
          f"{total_objects} objects, {total_frames} new frames "
          f"from global frame {start_frame}"
          f"{worker_tag}")

    global_frame = start_frame

    for obj_idx, obj_name in enumerate(object_list):
        obj_count = per_obj_remaining.get(obj_name, MODE_A_PER_OBJECT) if per_obj_remaining else MODE_A_PER_OBJECT
        print(f"\n  --- Session {obj_idx+1}/{total_objects}: {obj_name} "
              f"({obj_count} frames) ---")

        # Warmup: 7 spawn+render cycles to load textures
        for _ in range(7):
            _spawn_object(obj_name,
                          (random.uniform(*OBJ_A_DISTANCE), 0.0, random.uniform(*WATER_Z)),
                          random.uniform(*YAW_RANGE), random.uniform(*OBJ_A_PITCH),
                          1.0, name=f"ModeA_warmup_{obj_name}")
            sim.step()
            _silent_cleanup()

        for frame_local in range(obj_count):
            obj_dist = random.uniform(*OBJ_A_DISTANCE)
            vis_half_y = obj_dist * math.tan(_CAM_A_HFOV / 2) * 0.80
            max_jitter_y = min(OBJ_A_XY_JITTER, max(vis_half_y, 0.3))
            obj_x = obj_dist + random.uniform(-0.3, 0.3)
            obj_y = 0.0 + random.uniform(-max_jitter_y, max_jitter_y)
            obj_z = random.uniform(*WATER_Z)
            obj_yaw = random.uniform(*YAW_RANGE)
            obj_pitch = random.uniform(*OBJ_A_PITCH)
            obj_scale = random.uniform(*SCALE_RANGE)

            _spawn_object(obj_name, (obj_x, obj_y, obj_z), obj_yaw, obj_pitch, obj_scale,
                          name=f"ModeA_{obj_name}_{frame_local}")
            _randomize_lighting(dome_prim, intensity_range=DOME_INTENSITY_A)
            sim.step()
            _save_frame(global_frame, output_dir, anno_dict=annotators_a)
            _silent_cleanup()

            print(f"    [{obj_name:<7s} {frame_local+1:>4d}/{obj_count}] "
                  f"d={obj_dist:4.1f}m pos=({obj_x:+5.2f},{obj_y:+5.2f}) "
                  f"yaw={obj_yaw:6.1f}° pitch={obj_pitch:+5.1f}° scale={obj_scale:.2f}")
            global_frame += 1

        _silent_cleanup()

    return _build_coco_for_mode(output_dir, "mode_a")

# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# Mode B — smart scene composition with variable object density
# ═══════════════════════════════════════════════════════════════════════════
# Scene templates define different object distributions so the detector
# sees realistic variation: dense duck clusters, empty-ish frames, etc.
# Each template specifies (min, max) instances per type and a clearance
# override.  Duck + swan are oversampled to improve recall.
_B_SCENE_TEMPLATES = [
    # (label, duck, swan, turtle, lilypad, log, rock, clearance)
    ("duck-heavy",     (5, 10), (1, 4), (0, 3), (0, 4), (0, 3), (0, 3), (0.5, 2.0)),
    ("swan-heavy",     (1, 4), (5, 10), (0, 3), (0, 4), (0, 3), (0, 3), (0.5, 2.0)),
    ("balanced",       (2, 6), (2, 6), (1, 5), (1, 5), (1, 5), (1, 5), (1.0, 4.0)),
    ("crowded",        (3, 8), (3, 8), (2, 6), (2, 6), (2, 6), (2, 6), (0.0, 1.5)),
    ("sparse-targets", (1, 3), (1, 3), (0, 2), (0, 3), (0, 2), (0, 2), (2.0, 5.0)),
    ("distractor-heavy", (0, 2), (0, 2), (2, 6), (3, 7), (2, 6), (2, 6), (1.0, 3.0)),
    ("close-range",    (1, 4), (1, 4), (0, 2), (0, 2), (0, 2), (0, 2), (0.5, 2.0)),
]
# Extended Y range: objects 1.5m to 18m from camera (was -4..9, now -5..12)
_B_Y_RANGE = (-5.0, 12.0)

def _xform_set(prim_path, position, yaw, pitch):
    """Set USD transform on a prim — works when physics is not simulating."""
    _stage = get_context().get_stage()
    p = _stage.GetPrimAtPath(prim_path)
    if p and p.IsValid():
        xf = UsdGeom.XformCommonAPI(p)
        xf.SetTranslate(position)
        xf.SetRotate((pitch, yaw, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

def _disable_physics(parent_path):
    """Try to disable physics on all prims under *parent_path*.

    Sets kinematic flag on rigid bodies — PhysX skips them, rendering still works.
    Articulations (duck/swan) may not respond, but they're a minority.
    """
    _stage = get_context().get_stage()
    parent = _stage.GetPrimAtPath(parent_path)
    if not parent or not parent.IsValid():
        return
    for prim in Usd.PrimRange(parent):
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.CreateAttribute("physxRigidBody:kinematicEnabled",
                                     Sdf.ValueTypeNames.Bool, False).Set(True)
        except Exception:
            pass

def _populate_frame_smart(parent_path, object_types, template):
    """Spawn objects according to a scene template with variable clearance."""
    _label, *counts, (cl_lo, cl_hi) = template
    clearance = random.uniform(cl_lo, cl_hi)
    clearance_sq = clearance ** 2

    UsdGeom.Xform.Define(stage, parent_path)
    placed_xy = []
    for obj_name, (lo, hi) in zip(object_types, counts):
        n = random.randint(lo, hi)
        for _ in range(n):
            for _ in range(80):
                x = random.uniform(*PLACEMENT_ZONE["x"])
                y = random.uniform(*_B_Y_RANGE)
                if not placed_xy or all(
                    (x - px) ** 2 + (y - py) ** 2 >= clearance_sq
                    for px, py in placed_xy
                ):
                    break
            placed_xy.append((x, y))
            z = random.uniform(*WATER_Z)
            yaw = random.uniform(*YAW_RANGE)
            pitch = random.uniform(*PITCH_RANGE)
            scale = random.uniform(*SCALE_RANGE)
            _spawn_object(obj_name, (x, y, z), yaw, pitch, scale, parent=parent_path)
    if args.fast:
        _disable_physics(parent_path)

def run_mode_b(output_dir, dome_prim, num_frames=None, start_frame=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_frames is None:
        num_frames = MODE_B
    types = list(OBJECTS.keys())

    if args.fast:
        # ── Fast path: spawn max objects ONCE, reposition via USD each frame ─
        return _run_mode_b_fast(output_dir, dome_prim, num_frames, start_frame, types)

    worker_tag = f"  [worker {_SLICE_IDX}/{_SLICE_TOTAL}]" if args.slice else ""
    print(f"\n[Mode B] Full pond scene — {num_frames} frames "
          f"(7 scene templates, variable objects/frame, Y={_B_Y_RANGE}){worker_tag}")
    prev_parent = None
    for local_idx in range(num_frames + 2):
        if prev_parent is not None:
            try: stage.RemovePrim(prev_parent)
            except: pass
        parent_path = f"/Replicator/Frame_B_{local_idx}"
        _populate_frame_smart(parent_path, types, random.choice(_B_SCENE_TEMPLATES))
        _randomize_lighting(dome_prim, intensity_range=DOME_INTENSITY_BC)
        sim.step()
        if local_idx >= 2:
            _save_frame(start_frame + local_idx - 2, output_dir)
        prev_parent = parent_path
        if local_idx >= 2 and (local_idx - 2) % args.progress_interval == 0:
            print(f"  [B {start_frame+local_idx-2:>4d}/{start_frame+num_frames}]")
    if prev_parent is not None:
        try: stage.RemovePrim(prev_parent)
        except: pass
    return _build_coco_for_mode(output_dir, "mode_b")

def _run_mode_b_fast(output_dir, dome_prim, num_frames, start_frame, types):
    """Spawn-per-frame, but with pre-cached USD + render-only app.update().
    USD cache priming makes rep.create.from_usd near-instant after first use."""
    import omni.kit.app
    _app = omni.kit.app.get_app()

    # Prime the USD cache — open each file once so subsequent loads are instant
    print(f"  [Mode B FAST] Priming USD cache ({len(types)} files)...")
    for obj_name in types:
        Usd.Stage.Open(OBJECTS[obj_name]["usd"])
    print(f"  [Mode B FAST] Cache ready, render-only mode, ~2-5s/frame expected")

    prev_parent = None
    for local_idx in range(num_frames + 2):
        if prev_parent is not None:
            try: stage.RemovePrim(prev_parent)
            except: pass
        parent_path = f"/Replicator/Frame_B_{local_idx}"
        _populate_frame_smart(parent_path, types, random.choice(_B_SCENE_TEMPLATES))
        _randomize_lighting(dome_prim, intensity_range=DOME_INTENSITY_BC)
        if local_idx >= 2:
            _app.update()  # render only, no physics
        else:
            sim.step()     # warmup: physics init
        if local_idx >= 2:
            _save_frame(start_frame + local_idx - 2, output_dir)
        prev_parent = parent_path
        if local_idx >= 2 and (local_idx - 2) % args.progress_interval == 0:
            print(f"  [B {start_frame+local_idx-2:>4d}/{start_frame+num_frames}]")
    if prev_parent is not None:
        try: stage.RemovePrim(prev_parent)
        except: pass
    return _build_coco_for_mode(output_dir, "mode_b")

_C_DISTRACTOR_TEMPLATES = [
    # (label, turtle, lilypad, log, rock, clearance)
    ("dense",     (3, 8), (3, 8), (2, 6), (2, 6), (0.0, 2.0)),
    ("sparse",    (0, 3), (0, 3), (0, 2), (0, 2), (2.0, 5.0)),
    ("medium",    (1, 5), (1, 5), (1, 4), (1, 4), (1.0, 3.0)),
    ("close",     (1, 3), (1, 3), (0, 2), (0, 2), (0.5, 2.0)),
    ("empty-ish", (0, 1), (0, 1), (0, 1), (0, 1), (3.0, 6.0)),
]

def _run_mode_c_fast(output_dir, dome_prim, num_frames, start_frame):
    """Spawn-per-frame with pre-cached USD + render-only."""
    import omni.kit.app
    _app = omni.kit.app.get_app()

    distractor_types = list(DISTRACTORS)
    print(f"  [Mode C FAST] Priming USD cache ({len(distractor_types)} files)...")
    for obj_name in distractor_types:
        Usd.Stage.Open(OBJECTS[obj_name]["usd"])
    print(f"  [Mode C FAST] Cache ready, render-only mode")

    prev_parent = None
    for local_idx in range(num_frames + 2):
        if prev_parent is not None:
            try: stage.RemovePrim(prev_parent)
            except: pass
        parent_path = f"/Replicator/Frame_C_{local_idx}"
        _label, *counts, cl_range = random.choice(_C_DISTRACTOR_TEMPLATES)
        clearance = random.uniform(*cl_range)
        clearance_sq = clearance ** 2
        UsdGeom.Xform.Define(stage, parent_path)
        placed_xy = []
        for obj_name, (lo, hi) in zip(distractor_types, counts):
            n = random.randint(lo, hi)
            for _ in range(n):
                for _ in range(80):
                    x = random.uniform(*PLACEMENT_ZONE["x"])
                    y = random.uniform(*_B_Y_RANGE)
                    if all((x-px)**2+(y-py)**2 >= clearance_sq for px,py in placed_xy):
                        break
                placed_xy.append((x, y))
                z = random.uniform(*WATER_Z)
                yaw = random.uniform(*YAW_RANGE)
                _spawn_object(obj_name, (x, y, z), yaw, parent=parent_path)
        _randomize_lighting(dome_prim, intensity_range=DOME_INTENSITY_BC)
        if local_idx >= 2:
            _app.update()
        else:
            sim.step()
        if local_idx >= 2:
            _save_frame(start_frame + local_idx - 2, output_dir)
        prev_parent = parent_path
        if local_idx >= 2 and (local_idx - 2) % args.progress_interval == 0:
            print(f"  [C {start_frame+local_idx-2:>4d}/{start_frame+num_frames}]")
    if prev_parent is not None:
        try: stage.RemovePrim(prev_parent)
        except: pass
    return _build_coco_for_mode(output_dir, "mode_c")

def run_mode_c(output_dir, dome_prim, num_frames=None, start_frame=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_frames is None:
        num_frames = MODE_C

    if args.fast:
        return _run_mode_c_fast(output_dir, dome_prim, num_frames, start_frame)
    print(f"\n[Mode C] Distractor-only negatives — {num_frames} frames "
          f"(5 templates, variable objects/frame, no duck/swan){worker_tag}")
    distractor_types = list(DISTRACTORS)
    prev_parent = None
    for local_idx in range(num_frames + 2):
        if prev_parent is not None:
            try:
                stage.RemovePrim(prev_parent)
            except Exception:
                pass
        parent_path = f"/Replicator/Frame_C_{local_idx}"
        _label, *counts, cl_range = random.choice(_C_DISTRACTOR_TEMPLATES)
        clearance = random.uniform(*cl_range)
        clearance_sq = clearance ** 2
        UsdGeom.Xform.Define(stage, parent_path)
        placed_xy = []
        for obj_name, (lo, hi) in zip(distractor_types, counts):
            n = random.randint(lo, hi)
            for _ in range(n):
                for _ in range(80):
                    x = random.uniform(*PLACEMENT_ZONE["x"])
                    y = random.uniform(*_B_Y_RANGE)
                    if not placed_xy or all((x-px)**2+(y-py)**2 >= clearance_sq for px,py in placed_xy):
                        break
                placed_xy.append((x, y))
                z = random.uniform(*WATER_Z)
                yaw = random.uniform(*YAW_RANGE)
                _spawn_object(obj_name, (x, y, z), yaw, parent=parent_path)
        if args.fast:
            _disable_physics(parent_path)
        _randomize_lighting(dome_prim, intensity_range=DOME_INTENSITY_BC)
        # Warmup (first 2 frames) always uses physics to init bodies.
        # Data frames with --fast skip physics entirely — render only.
        if args.fast and local_idx >= 2:
            import omni.kit.app
            omni.kit.app.get_app().update()
        else:
            sim.step()
        if local_idx >= 2:
            _save_frame(start_frame + local_idx - 2, output_dir)
        prev_parent = parent_path
        if local_idx >= 2 and (local_idx - 2) % args.progress_interval == 0:
            print(f"  [C {start_frame+local_idx-2:>4d}/{start_frame+num_frames}]")
    if prev_parent is not None:
        try:
            stage.RemovePrim(prev_parent)
        except Exception:
            pass
    return _build_coco_for_mode(output_dir, "mode_c")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
_tick("total")

# --- Determine which modes to run ---
if args.mode_a_only:
    RUN_A, RUN_B, RUN_C = True, False, False
elif args.mode_b_only:
    RUN_A, RUN_B, RUN_C = False, True, False
elif args.mode_c_only:
    RUN_A, RUN_B, RUN_C = False, False, True
elif args.mode_bc_only:
    RUN_A, RUN_B, RUN_C = False, True, True
else:
    RUN_A, RUN_B, RUN_C = True, True, True

# --- Mode A object filter (--mode_a_objects) ---
if args.mode_a_objects:
    _filtered = [o.strip() for o in args.mode_a_objects.split(",")]
    _valid = set(MODE_A_OBJECTS)
    _a_objects_all = [o for o in _filtered if o in _valid]
    if not _a_objects_all:
        raise SystemExit(f"No valid objects in --mode_a_objects. Valid: {MODE_A_OBJECTS}")
    unknown = [o for o in _filtered if o not in _valid]
    if unknown:
        print(f"[WARN] Unknown objects ignored: {unknown}")
else:
    _a_objects_all = list(MODE_A_OBJECTS)

# --- Compute slice ranges for parallel workers ---
if args.slice and _SLICE_TOTAL > 1:
    # Mode A: split objects among workers
    chunk_a = math.ceil(len(_a_objects_all) / _SLICE_TOTAL)
    _a_objects = _a_objects_all[_SLICE_IDX * chunk_a : (_SLICE_IDX + 1) * chunk_a]

    # Mode B/C: split frame counts
    _b_per_worker = math.ceil(MODE_B / _SLICE_TOTAL)
    _c_per_worker = math.ceil(MODE_C / _SLICE_TOTAL)
    _b_start = _SLICE_IDX * _b_per_worker
    _c_start = _SLICE_IDX * _c_per_worker
else:
    _a_objects = list(_a_objects_all)
    _b_per_worker = MODE_B
    _c_per_worker = MODE_C
    _b_start = 0
    _c_start = 0

# --- Resume: detect existing frames and set start offsets ---
def _count_existing_pngs(mode_dir_name: str) -> int:
    """Return number of existing rgb_*.png files in a mode directory."""
    d = OUTPUT_ROOT / mode_dir_name
    if not d.is_dir():
        return 0
    return len(list(d.glob("**/rgb_*.png")))

def _count_existing_per_object(mode_dir_name: str) -> dict[str, int]:
    """Count existing frames per object by scanning bbox NPY files.

    Returns {obj_name: frame_count}.  Used for Mode A resume so we know
    exactly how many frames each object already has.
    """
    from collections import Counter

    d = OUTPUT_ROOT / mode_dir_name
    if not d.is_dir():
        return {}

    counts: Counter = Counter()
    for bbox_path in sorted(d.glob("**/bbox_*.npy")):
        try:
            item = np.load(str(bbox_path), allow_pickle=True)
            item = item.item() if isinstance(item, np.ndarray) and item.shape == () else item
            if isinstance(item, dict):
                id_to_labels = item.get("info", {}).get("idToLabels", {})
                data_arr = item.get("data", [])
                for row in data_arr:
                    sem_id = int(row[0])
                    label = id_to_labels.get(str(sem_id), {}).get("class", "")
                    if label and label != "water":
                        counts[label] += 1
        except Exception:
            pass
    return dict(counts)

_a_start_frame = 0
_a_per_obj_remaining = None  # default (no resume, use MODE_A_PER_OBJECT for all)

if args.resume:
    # Mode A: count existing frames PER OBJECT, append up to target.
    _a_existing_per_obj = _count_existing_per_object("mode_a_isolated")
    _a_existing_total = sum(_a_existing_per_obj.values())
    _a_target_total = len(_a_objects) * MODE_A_PER_OBJECT

    if _a_existing_total > 0:
        # Show per-object status
        parts = []
        all_done = True
        for obj_name in _a_objects:
            n = _a_existing_per_obj.get(obj_name, 0)
            needed = MODE_A_PER_OBJECT - n
            if needed > 0:
                all_done = False
                parts.append(f"{obj_name}: {n}/{MODE_A_PER_OBJECT} (need {needed})")
        if parts:
            print(f"[resume] Mode A per-object: " + " | ".join(parts))
        if all_done:
            print(f"[resume] Mode A: all objects at {MODE_A_PER_OBJECT}/obj — skipping")
            RUN_A = False
        else:
            _a_missing = _a_target_total - _a_existing_total
            # Compute global start frame (append after existing)
            _a_start_frame = _a_existing_total
            print(f"[resume] Mode A: {_a_existing_total} total existing, "
                  f"need {_a_missing} more → targeting {MODE_A_PER_OBJECT}/obj")

            # Build per-object list with remaining counts for run_mode_a
            _a_per_obj_remaining = {}
            for obj_name in _a_objects:
                have = _a_existing_per_obj.get(obj_name, 0)
                _a_per_obj_remaining[obj_name] = max(0, MODE_A_PER_OBJECT - have)
            # Remove fully-complete objects from the run list
            _a_objects = [o for o in _a_objects if _a_per_obj_remaining.get(o, 0) > 0]
    else:
        _a_start_frame = 0
        _a_per_obj_remaining = None

    # Mode B/C: --mode_b_count / --mode_c_count = DESIRED TOTAL; append if short
    _b_existing = _count_existing_pngs("mode_b_mixed")
    if not args.slice:
        _b_start = _b_existing
    if not RUN_B:
        pass
    elif _b_existing >= MODE_B:
        print(f"[resume] Mode B: {_b_existing} frames >= target {MODE_B} — skipping")
        RUN_B = False
    elif _b_existing > 0:
        _b_new = MODE_B - _b_existing
        _b_per_worker = _b_new
        print(f"[resume] Mode B: {_b_existing} existing, need {_b_new} more "
              f"→ targeting {MODE_B} total, starting at idx {_b_start}")

    _c_existing = _count_existing_pngs("mode_c_negatives")
    if not args.slice:
        _c_start = _c_existing
    if not RUN_C:
        pass
    elif _c_existing >= MODE_C:
        print(f"[resume] Mode C: {_c_existing} frames >= target {MODE_C} — skipping")
        RUN_C = False
    elif _c_existing > 0:
        _c_new = MODE_C - _c_existing
        _c_per_worker = _c_new
        print(f"[resume] Mode C: {_c_existing} existing, need {_c_new} more "
              f"→ targeting {MODE_C} total, starting at idx {_c_start}")

    if not RUN_A and not RUN_B and not RUN_C:
        print("[resume] All modes already at target — nothing to do")
        if args.no_coco:
            print("[resume] Run without --no_coco or with --coco_only to rebuild annotations")
        print("=" * 70)
        sim_app.close()
        exit(0)

def _build_coco_for_mode(mode_dir, mode_name):
    """Build COCO JSON for one mode and write to disk (unless --no_coco)."""
    if args.no_coco:
        return {}  # parallel worker: skip COCO, will be rebuilt later
    coco = _build_coco(mode_dir, mode_name)
    return coco

# --- COCO-only mode: rebuild annotations from existing PNGs/bboxes ---
if args.coco_only:
    print("=" * 70)
    print("  COCO-ONLY — Rebuilding annotations from existing data")
    print(f"  Output: {OUTPUT_ROOT}")
    print("=" * 70)
    mode_coco_data = []
    for mode_name, mode_dir_name in [
        ("mode_a", "mode_a_isolated"),
        ("mode_b", "mode_b_mixed"),
        ("mode_c", "mode_c_negatives"),
    ]:
        d = OUTPUT_ROOT / mode_dir_name
        if d.is_dir() and list(d.glob("rgb_*.png")):
            coco = _build_coco(d, mode_name)
            mode_coco_data.append(coco)
            print(f"  {mode_dir_name}: {len(coco['images'])} imgs, "
                  f"{len(coco['annotations'])} anns")
        else:
            print(f"  {mode_dir_name}: no data — skipping")
    if mode_coco_data:
        merged = _merge_coco(mode_coco_data, OUTPUT_ROOT / "annotations_all.json")
        print(f"\n  Merged → {OUTPUT_ROOT / 'annotations_all.json'}")
        print(f"  Total: {len(merged['images'])} images, "
              f"{len(merged['annotations'])} annotations")
    print("=" * 70)
    sim_app.close()
    exit(0)

# --- Clear output (unless resuming or coco_only) ---
if not args.resume:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

dome_prim = _find_dome_prim()

total_frames = ((len(_a_objects) * MODE_A_PER_OBJECT - _a_start_frame) if RUN_A else 0) \
             + (_b_per_worker if RUN_B else 0) \
             + (_c_per_worker if RUN_C else 0)
print("=" * 70)
print("  STAGE 0 — Training Data Generation")
print(f"  Output: {OUTPUT_ROOT}")
print(f"  Config: Mode A {len(_a_objects)}obj×{MODE_A_PER_OBJECT} | "
      f"B {_b_per_worker} | C {_c_per_worker} = {total_frames} total frames")
if args.slice:
    print(f"  Slice:  worker {_SLICE_IDX}/{_SLICE_TOTAL}")
if args.resume:
    print(f"  Resume: skipping completed modes")
if args.no_coco:
    print(f"  COCO:   disabled (--no_coco) — run with --coco_only after all workers finish")
print(f"  Reset:  {'disabled (fast)' if args.no_reset and not args.do_reset else 'enabled'}")
print(f"  Camera: focal={FOCAL_LENGTH}mm, res={RESOLUTION}")
print(f"  Dome:   {str(dome_prim.GetPath()) if dome_prim else 'NOT FOUND'}")
print("=" * 70)


# --- Helpers for inter-mode transition ---
def _transition_to_next_mode():
    """Clean up objects between modes. Avoids sim.reset() by default
    to prevent GPU pipeline teardown that causes CUDA memcpy stalls."""
    global dome_prim
    _silent_cleanup()
    if args.do_reset or (not args.no_reset):
        # Legacy: full simulation reset (may cause CUDA reinit overhead)
        sim.reset()
        dome_prim = _find_dome_prim()
    else:
        # Fast path: manual cleanup only, preserve GPU pipeline
        # Remove any leftover persistent parents from previous mode
        for prefix in ["Persistent_B", "Persistent_C", "Frame_B_", "Frame_C_"]:
            for child in list(stage.GetPrimAtPath("/Replicator").GetAllChildren()):
                try:
                    if child.GetName().startswith(prefix):
                        stage.RemovePrim(child.GetPath())
                except Exception:
                    pass
        # Re-find dome prim (may have been recreated) and restore lighting
        dome_prim = _find_dome_prim()
    _reset_dome_lighting(dome_prim)
    # Several warmup steps to let GPU pipeline reinitialise after
    # camera/render-product teardown from the previous mode.
    # The first 1-2 steps may log transient CUDA memcpy warnings;
    # these are harmless and clear up before real data frames.
    for _ in range(5):
        sim.step()

# --- Initial sim reset to initialise physics/fabric ---
sim.reset()
dome_prim = _find_dome_prim()
_reset_dome_lighting(dome_prim)

mode_coco_data = []

# --- Mode A: isolated single-object ---
if RUN_A:
    mode_a_coco = run_mode_a(OUTPUT_ROOT / "mode_a_isolated", dome_prim,
                              object_list=_a_objects, start_frame=_a_start_frame,
                              per_obj_remaining=_a_per_obj_remaining)
    mode_coco_data.append(mode_a_coco)
    _transition_to_next_mode()
else:
    # Load existing Mode A COCO if resuming
    _a_json = OUTPUT_ROOT / "mode_a_isolated" / "annotations_mode_a.json"
    if _a_json.exists():
        with open(_a_json) as f:
            mode_coco_data.append(json.load(f))

# --- Mode B: full pond scene ---
if RUN_B:
    mode_b_coco = run_mode_b(OUTPUT_ROOT / "mode_b_mixed", dome_prim,
                              num_frames=_b_per_worker, start_frame=_b_start)
    mode_coco_data.append(mode_b_coco)
    _transition_to_next_mode()
else:
    _b_json = OUTPUT_ROOT / "mode_b_mixed" / "annotations_mode_b.json"
    if _b_json.exists():
        with open(_b_json) as f:
            mode_coco_data.append(json.load(f))

# --- Mode C: distractor-only negatives ---
if RUN_C:
    mode_c_coco = run_mode_c(OUTPUT_ROOT / "mode_c_negatives", dome_prim,
                              num_frames=_c_per_worker, start_frame=_c_start)
    mode_coco_data.append(mode_c_coco)
else:
    _c_json = OUTPUT_ROOT / "mode_c_negatives" / "annotations_mode_c.json"
    if _c_json.exists():
        with open(_c_json) as f:
            mode_coco_data.append(json.load(f))

# --- Merge COCO ---
# Filter out empty dicts (from skipped modes or --no_coco runs)
valid_coco = [c for c in mode_coco_data if c and c.get("images")]
if valid_coco:
    merged = _merge_coco(valid_coco, OUTPUT_ROOT / "annotations_all.json")
else:
    merged = {"images": [], "annotations": [], "categories": COCO_CATEGORIES,
              "semantic_id_map": {}}
    if args.no_coco:
        print("\n  ⚠️  --no_coco: per-mode annotations not built by workers")
        print("  After all workers finish, run:")
        print(f"    ./isaaclab.sh -p scripts/pond/generate_training_data.py --coco_only --resume")

sim.stop()
_tock("total", "Total generation time")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
total_png = len(list(OUTPUT_ROOT.rglob("rgb_*.png")))
total_bbox = len(list(OUTPUT_ROOT.rglob("bbox_*.npy")))
print("\n" + "=" * 70)
print("  GENERATION COMPLETE")
print(f"  Frames: {total_png} PNG, {total_bbox} bbox")
if merged["images"]:
    print(f"  COCO:   {len(merged['annotations'])} boxes / {len(merged['images'])} images")
print("  By mode:")
for mode_dir in sorted(OUTPUT_ROOT.iterdir()):
    if mode_dir.is_dir():
        png = len(list(mode_dir.glob("rgb_*.png")))
        bbox = len(list(mode_dir.glob("bbox_*.npy")))
        print(f"    {mode_dir.name}: {png} frames, {bbox} bbox")
print("=" * 70)

sim_app.close()

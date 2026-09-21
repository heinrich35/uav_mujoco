"""
validate_dataset.py — Validates training data from generate_training_data.py

Four checks in order:
  1 — annotations JSON semantic_id_map: every semId has valid label + class
  2 — Per-frame bbox files: unknown IDs, degenerate boxes, OOB, high occlusion
  3 — Visual overlay: bboxes on RGB with waterline, thick=target thin=other
  4 — Class distribution: other >= duck+swan, duck/swan within 2:1

Usage:
    python validate_dataset.py --data_dir /home/heinz/isaaclab_uav/outputs/training_data
    python validate_dataset.py --data_dir ... --mode mode_a_isolated --out ./report
"""

import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Known valid labels / classes ─────────────────────────────────────────
VALID_LABELS   = {'duck', 'swan', 'rock', 'log', 'lilypad', 'turtle', 'water'}
VALID_CLASSES  = {'duck', 'swan', 'turtle', 'lilypad', 'log', 'rock'}
TARGET_LABELS  = {'duck', 'swan'}
DISTRACTOR_LABELS = {'turtle', 'lilypad', 'log', 'rock'}
OCCLUSION_THRESHOLD = 0.60

# ── Visual overlay colours ───────────────────────────────────────────────
LABEL_COLORS = {
    'duck':    (255, 200, 0),
    'swan':    (255, 255, 255),
    'rock':    (160, 120, 80),
    'log':     (120, 90, 60),
    'lilypad': (80, 180, 80),
    'turtle':  (100, 140, 100),
    'water':   (0, 100, 200),
    'UNKNOWN': (255, 0, 0),
}


def load_bbox(path: Path) -> dict | None:
    """Load a bbox NPY. Returns {'data': [dict], 'labels': {semId: label}}."""
    item = np.load(path, allow_pickle=True)
    if hasattr(item, 'item') and item.ndim == 0:
        item = item.item()
    if not isinstance(item, dict):
        return None
    data_arr = item.get("data", [])
    id_to_labels = item.get("info", {}).get("idToLabels", {})
    out = []
    labels = {}
    for row in data_arr:
        sem_id = int(row[0])
        label = id_to_labels.get(str(sem_id), {}).get("class", f"UNKNOWN_{sem_id}")
        out.append({
            "semanticId": sem_id,
            "x_min": int(row[1]), "y_min": int(row[2]),
            "x_max": int(row[3]), "y_max": int(row[4]),
            "occlusion": float(row[5]),
            "label": label,
        })
        labels[sem_id] = label
    return {"data": out, "labels": labels}


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1 — semantic_id_map in annotations JSON
# ═══════════════════════════════════════════════════════════════════════════
def check_semantic_map(ann: dict) -> list[str]:
    """Every semanticId must have a valid label and class."""
    issues = []
    smap = ann.get("semantic_id_map", {})
    if not smap:
        issues.append("CRITICAL: 'semantic_id_map' missing from annotations JSON — "
                      "cannot validate labels")
        return issues

    for sid, info in sorted(smap.items()):
        label = info.get("label", "")
        cls   = info.get("class", "")
        if label not in VALID_LABELS:
            issues.append(f"  semId {sid}: label '{label}' not in {sorted(VALID_LABELS)}")
        if cls not in VALID_CLASSES and cls != "other":
            issues.append(f"  semId {sid}: class '{cls}' not in {sorted(VALID_CLASSES)}")
        if label in TARGET_LABELS and cls != label:
            issues.append(f"  semId {sid}: target label '{label}' has wrong class '{cls}'")
        if label in DISTRACTOR_LABELS and cls != label:
            issues.append(f"  semId {sid}: distractor '{label}' has class '{cls}' "
                          f"(should be '{label}')")
    return issues


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2 — Per-frame bbox validation
# ═══════════════════════════════════════════════════════════════════════════
def check_frames(mode_dir: Path, ann: dict, img_w: int, img_h: int) -> dict:
    """Load every bbox_*.npy and check for ID/label/degenerate/OOB/occlusion issues."""
    smap = ann.get("semantic_id_map", {})

    stats = {
        "total_frames": 0,
        "frames_with_targets": 0,
        "frames_no_objects": 0,
        "label_counts": Counter(),
        "occlusion_flags": [],
        "bbox_degenerate": [],
        "bbox_oob": [],
        "unknown_ids": [],
    }

    bbox_files = sorted(mode_dir.glob("**/bbox_*.npy"))
    if not bbox_files:
        print("  WARNING: No bbox_*.npy files found")
        return stats

    for bf in bbox_files:
        tag = bf.stem.replace("bbox_", "")
        stats["total_frames"] += 1

        raw = load_bbox(bf)
        if raw is None or not raw["data"]:
            stats["frames_no_objects"] += 1
            continue

        has_target = False
        for box in raw["data"]:
            sid   = box["semanticId"]
            label = box["label"]
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            occ   = box["occlusion"]

            # Unknown semantic ID?
            if str(sid) not in smap:
                stats["unknown_ids"].append((tag, sid, label))
                continue

            stats["label_counts"][label] += 1
            if label in TARGET_LABELS:
                has_target = True

            # Occlusion check for targets
            if label in TARGET_LABELS and occ > OCCLUSION_THRESHOLD:
                stats["occlusion_flags"].append((tag, label, round(occ, 3)))

            # Degenerate bbox
            if x2 <= x1 or y2 <= y1:
                stats["bbox_degenerate"].append((tag, label, (x1, y1, x2, y2)))
            # Out-of-bounds
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                stats["bbox_oob"].append((tag, label, (x1, y1, x2, y2)))

        if has_target:
            stats["frames_with_targets"] += 1

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3 — Visual overlay with bboxes + waterline
# ═══════════════════════════════════════════════════════════════════════════
def draw_visual_check(mode_dir: Path, ann: dict, out_dir: Path, n: int = 9):
    """Draw bounding boxes on RGB images. Thick border = duck/swan, thin = other.
    Yellow horizontal line at image midpoint marks the waterline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    smap = ann.get("semantic_id_map", {})

    bbox_files = sorted(mode_dir.glob("**/bbox_*.npy"))[:n]
    for bf in bbox_files:
        tag      = bf.stem.replace("bbox_", "")
        rgb_path = bf.parent / f"rgb_{tag}.png"
        if not rgb_path.exists():
            continue

        img  = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size

        # Waterline at image midpoint (yellow)
        draw.line([(0, H // 2), (W, H // 2)], fill=(255, 255, 0), width=1)

        raw = load_bbox(bf)
        if raw is None:
            img.save(out_dir / f"check_{tag}.png")
            continue

        for box in raw["data"]:
            x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]
            label = box["label"]
            occ   = box["occlusion"]
            color = LABEL_COLORS.get(label, LABEL_COLORS["UNKNOWN"])
            lw    = 3 if label in TARGET_LABELS else 1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            draw.text((x1 + 2, y1 + 2), f"{label} occ={occ:.2f}", fill=color)

        img.save(out_dir / f"check_{tag}.png")

    print(f"  Visual overlays → {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4 — Class distribution
# ═══════════════════════════════════════════════════════════════════════════
def check_distribution(label_counts: Counter, mode_name: str = "") -> list[str]:
    """Ensure duck+swan present (except Mode C), and distractor classes not empty.
    Mode C (negatives) is expected to have no duck/swan."""
    issues = []
    total = sum(label_counts.values())
    if total == 0:
        return ["CRITICAL: No labeled objects found"]

    duck  = label_counts.get("duck", 0)
    swan  = label_counts.get("swan", 0)
    distractors = sum(v for k, v in label_counts.items()
                      if k in DISTRACTOR_LABELS)
    is_neg = "negatives" in mode_name

    if is_neg:
        if duck > 0:
            issues.append(f"❌ Mode C has {duck} 'duck' labels — should be zero")
        if swan > 0:
            issues.append(f"❌ Mode C has {swan} 'swan' labels — should be zero")
    else:
        if duck == 0:
            issues.append("CRITICAL: No 'duck' labels found")
        if swan == 0:
            issues.append("CRITICAL: No 'swan' labels found")
        if duck > 0 and swan > 0:
            ratio = max(duck, swan) / min(duck, swan)
            if ratio > 2.0:
                issues.append(f"Class imbalance: duck={duck} / swan={swan} "
                              f"(ratio {ratio:.1f}:1) — oversample minority class")
        if distractors < (duck + swan):
            issues.append(f"Distractor count ({distractors}) < duck+swan ({duck + swan}) — "
                          f"add more distractor frames (Mode C)")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="Validate pond training data")
    p.add_argument("--data_dir", default="/home/heinz/isaaclab_uav/outputs/training_data")
    p.add_argument("--mode", default="all")
    p.add_argument("--out", default="./validation_report")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--visual_n", type=int, default=9)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("POND DATASET VALIDATION")
    print(f"  Data dir: {data_dir}")
    print("=" * 62)

    modes = [args.mode] if args.mode != "all" else [
        "mode_a_isolated", "mode_b_mixed", "mode_c_negatives"
    ]

    all_issues = 0
    all_warnings = 0

    for mode_name in modes:
        mode_dir = data_dir / mode_name
        if not mode_dir.is_dir():
            continue

        # ── Find the mode's annotations JSON ─────────────────────────────
        ann_path = None
        for jf in sorted(mode_dir.glob("annotations_*.json")):
            ann_path = jf
            break
        if ann_path is None:
            print(f"\n  ⚠️  {mode_name}: no annotations JSON found")
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        print(f"\n{'─' * 62}")
        print(f"  Mode: {mode_name}")
        print(f"  JSON: {ann_path.name}")
        print(f"{'─' * 62}")

        # ── Check 1 ──────────────────────────────────────────────────────
        print("\n[1/4] Checking semantic_id_map …")
        map_issues = check_semantic_map(ann)
        if map_issues:
            for i in map_issues:
                print(f"  ❌ {i}")
        else:
            n_ids = len(ann.get("semantic_id_map", {}))
            print(f"  ✅ semantic_id_map OK — {n_ids} IDs with valid labels + classes")
        all_issues += len(map_issues)

        # ── Check 2 ──────────────────────────────────────────────────────
        print("\n[2/4] Checking per-frame bbox files …")
        stats = check_frames(mode_dir, ann, args.width, args.height)

        print(f"  Total frames:       {stats['total_frames']}")
        print(f"  Frames w/ targets:  {stats['frames_with_targets']}")
        print(f"  Empty frames:       {stats['frames_no_objects']}")

        if stats["unknown_ids"]:
            for tag, sid, lbl in stats["unknown_ids"][:5]:
                print(f"  ❌ Frame {tag}: semId {sid} ('{lbl}') NOT in semantic_id_map")
            all_issues += len(stats["unknown_ids"])
        else:
            print(f"  ✅ All semantic IDs found in map")

        if stats["bbox_degenerate"]:
            for tag, lbl, box in stats["bbox_degenerate"][:5]:
                print(f"  ❌ Frame {tag}: degenerate bbox {box} for '{lbl}'")
            all_issues += len(stats["bbox_degenerate"])

        if stats["bbox_oob"]:
            for tag, lbl, box in stats["bbox_oob"][:5]:
                print(f"  ⚠️  Frame {tag}: OOB bbox {box} for '{lbl}'")
            all_warnings += len(stats["bbox_oob"])

        if stats["occlusion_flags"]:
            n = len(stats["occlusion_flags"])
            print(f"  ⚠️  {n} frames have target occlusion > {OCCLUSION_THRESHOLD} "
                  f"(filter during YOLO conversion)")
            for tag, lbl, occ in stats["occlusion_flags"][:3]:
                print(f"      frame {tag}: '{lbl}' occ={occ}")
            all_warnings += n
        else:
            print(f"  ✅ No high-occlusion target frames")

        # Mode-specific target check
        if "negatives" in mode_name:
            has_t = any(t in stats["label_counts"] for t in TARGET_LABELS)
            if has_t:
                print(f"  ❌ Mode C has targets! Should be distractor-only")
                all_issues += 1
            else:
                print(f"  ✅ No duck/swan (correct for negatives)")

        # ── Check 3 ──────────────────────────────────────────────────────
        print(f"\n[3/4] Visual overlay — first {args.visual_n} frames …")
        draw_visual_check(mode_dir, ann, out_dir / f"visual_{mode_name}",
                          n=args.visual_n)

        # ── Check 4 ──────────────────────────────────────────────────────
        print("\n[4/4] Class distribution:")
        total = sum(stats["label_counts"].values())
        for lbl, cnt in sorted(stats["label_counts"].items()):
            pct = 100 * cnt / total if total else 0
            bar = "█" * int(cnt * 40 / max(stats["label_counts"].values(), default=1))
            print(f"    {lbl:<10} {cnt:>5}  ({pct:4.1f}%)  {bar}")

        dist_issues = check_distribution(stats["label_counts"], mode_name)
        if dist_issues:
            for i in dist_issues:
                print(f"  ⚠️  {i}")
            all_warnings += len(dist_issues)
        else:
            print(f"  ✅ Distribution balanced")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print(f"SUMMARY  {all_issues} errors  |  {all_warnings} warnings")
    if all_issues == 0 and all_warnings == 0:
        print("✅ Dataset clean — ready for YOLO conversion")
    elif all_issues == 0:
        print("⚠️  No hard errors — review warnings before training")
    else:
        print("❌ Fix errors before converting to YOLO format")
    print(f"{'═' * 62}")


if __name__ == "__main__":
    main()

"""
convert_to_yolo.py — Converts pond training data from COCO to YOLO format.

Class mapping (matches generate_training_data.py):
  duck    → 0
  swan    → 1
  turtle  → 2
  lilypad → 3
  log     → 4
  rock    → 5

High-occlusion targets (occ > MAX_OCC) are dropped during conversion.
Degenerate boxes (x2 <= x1 or y2 <= y1) are skipped.
Frames with zero valid objects are excluded.

Usage:
    python convert_to_yolo.py \
        --data_root ~/isaaclab_uav/outputs/training_data \
        --out_root  ./pond_yolo \
        --width 640 --height 480
"""

import argparse, json, random, shutil
from collections import Counter
from pathlib import Path

import numpy as np

YOLO_CLASSES = {
    'duck': 0, 'swan': 1, 'turtle': 2,
    'lilypad': 3, 'log': 4, 'rock': 5,
}
CLASS_NAMES = ['duck', 'swan', 'turtle', 'lilypad', 'log', 'rock']
MAX_OCC = 0.60   # drop target instances with occlusion above this


def load_bbox(path: Path) -> list[dict]:
    """Load a bbox NPY file. Returns list of box dicts with keys:
    semId, x_min, y_min, x_max, y_max, occlusion, label."""
    item = np.load(path, allow_pickle=True)
    if hasattr(item, 'item') and item.ndim == 0:
        item = item.item()
    if not isinstance(item, dict):
        return []

    # Format: {'data': structured_array(dtype=[semanticId,x_min,y_min,x_max,y_max,occlusionRatio]),
    #          'info': {'idToLabels': {semId: {'class': label}}}}
    data_arr = item.get("data", [])
    id_to_labels = item.get("info", {}).get("idToLabels", {})

    boxes = []
    for row in data_arr:
        sem_id = int(row[0])
        label = id_to_labels.get(str(sem_id), {}).get("class", "other")
        boxes.append({
            "semId": sem_id,
            "x_min": int(row[1]), "y_min": int(row[2]),
            "x_max": int(row[3]), "y_max": int(row[4]),
            "occlusion": float(row[5]),
            "label": label,
        })
    return boxes


def bbox_to_yolo(x1, y1, x2, y2, W, H):
    """Convert pixel coords to YOLO normalized format (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh


def process_mode(mode_dir: Path, img_w: int, img_h: int,
                 max_occ: float) -> list[tuple]:
    """Process one mode directory. Returns list of (rgb_png_path, [yolo_lines])."""
    results = []

    for bbox_path in sorted(mode_dir.glob("**/bbox_*.npy")):
        tag      = bbox_path.stem.replace("bbox_", "")
        # rgb file is in the same bucket subdirectory as the bbox
        rgb_path = bbox_path.parent / f"rgb_{tag}.png"
        if not rgb_path.exists():
            continue

        try:
            boxes = load_bbox(bbox_path)
        except Exception:
            continue

        label_lines = []
        for box in boxes:
            label = box["label"]
            x1, y1 = box["x_min"], box["y_min"]
            x2, y2 = box["x_max"], box["y_max"]
            occ    = box["occlusion"]

            # Skip water plane
            if label == "water":
                continue

            # Class assignment — all 6 fine-grained classes
            if label in YOLO_CLASSES:
                if label in ('duck', 'swan') and occ > max_occ:  # skip heavily occluded targets
                    continue
                cls_id = YOLO_CLASSES[label]
            else:
                continue     # unknown label (water, …)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if label_lines:
            results.append((rgb_path, label_lines))

    return results


def split_and_write(records, out_root: Path, split=(0.70, 0.15, 0.15)):
    """Shuffle and split into train/val/test, write images + labels."""
    random.shuffle(records)
    n = len(records)
    t_end = int(n * split[0])
    v_end = t_end + int(n * split[1])

    splits = {
        "train": records[:t_end],
        "val":   records[t_end:v_end],
        "test":  records[v_end:],
    }

    for subset, items in splits.items():
        img_dir = out_root / "images" / subset
        lbl_dir = out_root / "labels" / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for rgb_path, lines in items:
            dst_img = img_dir / rgb_path.name
            dst_lbl = lbl_dir / rgb_path.with_suffix(".txt").name
            shutil.copy2(rgb_path, dst_img)
            dst_lbl.write_text("\n".join(lines))

    return {k: len(v) for k, v in splits.items()}


def write_yaml(out_root: Path, img_w: int, img_h: int):
    """Write pond.yaml dataset config for YOLO training."""
    yaml = f"""# Pond dataset — gosling camera (640×480, 24mm focal)
# Camera: static gosling viewpoint 0.35m above water
# No vertical flip — sky always at top, water at bottom
path: {out_root.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 6
names:
  0: duck
  1: swan
  2: turtle
  3: lilypad
  4: log
  5: rock
"""
    (out_root / "pond.yaml").write_text(yaml)
    print(f"  YOLO config → {out_root / 'pond.yaml'}")


def main():
    ap = argparse.ArgumentParser(description="Convert pond COCO data to YOLO format")
    ap.add_argument("--data_root", default="./training_data")
    ap.add_argument("--out_root",  default="./pond_yolo")
    ap.add_argument("--width",     type=int, default=640)
    ap.add_argument("--height",    type=int, default=480)
    ap.add_argument("--max_occlusion", type=float, default=0.60)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    MODES = [
        ("mode_a_isolated",  "annotations_mode_a.json"),
        ("mode_b_mixed",     "annotations_mode_b.json"),
        ("mode_c_negatives", "annotations_mode_c.json"),
    ]

    all_records = []
    label_counter = Counter()

    print("POND → YOLO CONVERSION")
    print(f"  Resolution: {args.width}×{args.height}  "
          f"max_occlusion: {args.max_occlusion}")
    print(f"  Source: {data_root}")
    print()

    for mode_name, ann_name in MODES:
        mode_dir = data_root / mode_name
        if not mode_dir.is_dir():
            print(f"  ⚠️  {mode_name}: not found — skipping")
            continue

        records = process_mode(mode_dir, args.width, args.height,
                               args.max_occlusion)
        all_records.extend(records)

        for _, lines in records:
            for line in lines:
                cid = int(line.split()[0])
                label_counter[CLASS_NAMES[cid]] += 1

        print(f"  {mode_name:<25} {len(records):>5} frames")

    if not all_records:
        print("ERROR: No records collected — check paths")
        return

    print(f"\n  Total: {len(all_records)} frames")
    print("  Labels:")
    for lbl, cnt in sorted(label_counter.items()):
        pct = 100 * cnt / sum(label_counter.values())
        print(f"    {lbl:<8} {cnt:>5}  ({pct:.1f}%)")

    counts = split_and_write(all_records, out_root)
    print(f"\n  Train: {counts['train']}  |  Val: {counts['val']}  |  Test: {counts['test']}")

    write_yaml(out_root, args.width, args.height)

    print(f"\n✅ YOLO dataset ready at: {out_root}")
    print("   Next: yolo detect train "
          f"data={out_root}/pond.yaml model=yolo11m.pt imgsz={args.width}"
          " flipud=0.0 mosaic=0.5")


if __name__ == "__main__":
    main()

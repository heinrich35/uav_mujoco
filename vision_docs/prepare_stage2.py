"""
prepare_stage2.py — Builds Stage 2 fine-grained classifier dataset.

Sources:
  Mode A — isolated single-object PNGs (clean, pose-diverse)
  Mode B — bbox crops from mixed pond scenes (realistic inference crops)

Output (ImageFolder-compatible):
  stage2_dataset/
    train/  duck/  swan/  other/
    val/    duck/  swan/  other/
    test/   duck/  swan/  other/

Usage:
    python prepare_stage2.py \
        --mode_a  outputs/training_data/mode_a_isolated \
        --mode_b  outputs/training_data/mode_b_mixed \
        --out     ./stage2_dataset
"""

import argparse, json, random, shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

random.seed(42)
np.random.seed(42)

TARGET_CLASSES    = ['duck', 'swan']
DISTRACTOR_CLASSES = ['rock', 'log', 'lilypad', 'turtle']

LABEL_TO_CLASS = {
    'duck': 'duck', 'swan': 'swan',
    'rock': 'other', 'log': 'other', 'lilypad': 'other', 'turtle': 'other',
}

SPLIT = (0.70, 0.15, 0.15)


def load_bbox(path: Path) -> list[dict]:
    """Load a bbox NPY file. Returns list of box dicts."""
    item = np.load(path, allow_pickle=True)
    if hasattr(item, 'item') and item.ndim == 0:
        item = item.item()
    if not isinstance(item, dict):
        return []
    data_arr = item.get("data", [])
    id_to_labels = item.get("info", {}).get("idToLabels", {})
    boxes = []
    for row in data_arr:
        sem_id = int(row[0])
        label = id_to_labels.get(str(sem_id), {}).get("class", "other")
        boxes.append({
            "x_min": int(row[1]), "y_min": int(row[2]),
            "x_max": int(row[3]), "y_max": int(row[4]),
            "occlusion": float(row[5]),
            "label": label,
        })
    return boxes


def augment_image(img: Image.Image, n_variants: int = 9) -> list[Image.Image]:
    """Generate augmented variants of a Mode A isolated image."""
    variants = [img]
    for _ in range(n_variants - 1):
        aug = img.copy()
        if random.random() > 0.5:
            aug = ImageOps.mirror(aug)
        angle = random.uniform(-25, 25)
        aug = aug.rotate(angle, expand=False,
                         fillcolor=(random.randint(80, 150),
                                    random.randint(100, 180),
                                    random.randint(120, 200)))
        aug = ImageEnhance.Brightness(aug).enhance(random.uniform(0.65, 1.45))
        aug = ImageEnhance.Contrast(aug).enhance(random.uniform(0.75, 1.35))
        aug = ImageEnhance.Color(aug).enhance(random.uniform(0.7, 1.4))
        if random.random() > 0.6:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.8)))
        W, H = aug.size
        scale = random.uniform(0.80, 1.00)
        nw, nh = int(W * scale), int(H * scale)
        x0 = random.randint(0, max(1, W - nw))
        y0 = random.randint(0, max(1, H - nh))
        aug = aug.crop((x0, y0, x0 + nw, y0 + nh)).resize((W, H), Image.BILINEAR)
        variants.append(aug)
    return variants


def collect_mode_a(mode_a_dir: Path, n_aug: int) -> list[tuple[Image.Image, str]]:
    """Collect Mode A isolated images. Label determined from bbox files.
    Each source PNG has exactly 1 object (water bbox is filtered out)."""
    records = []
    png_files = sorted(mode_a_dir.glob("rgb_*.png"))
    if not png_files:
        print("  [Mode A] No rgb_*.png files found")
        return records

    label_counts = Counter()
    for png_path in png_files:
        tag = png_path.stem.replace("rgb_", "")
        bbox_path = mode_a_dir / f"bbox_{tag}.npy"
        if not bbox_path.exists():
            continue

        boxes = load_bbox(bbox_path)
        # Find the non-water object label (Mode A has exactly 1 object per frame)
        obj_label = "other"
        for box in boxes:
            if box["label"] != "water":
                obj_label = box["label"]
                break

        cls = LABEL_TO_CLASS.get(obj_label, "other")
        label_counts[cls] += 1

        try:
            img = Image.open(png_path).convert("RGB")
        except Exception:
            continue

        for aug in augment_image(img, n_aug):
            records.append((aug, cls))

    print(f"  [Mode A] {len(png_files)} source images × {n_aug} aug = {len(records)} total")
    for cls, cnt in sorted(label_counts.items()):
        print(f"           {cls:<8} {cnt} source images")
    return records


def collect_mode_b_crops(mode_b_dir: Path, pad_frac: float = 0.15,
                          max_occlusion: float = 0.55) -> list[tuple[Image.Image, str]]:
    """Crop every labeled object from Mode B RGB images using bbox."""
    records = []
    png_files = sorted(mode_b_dir.glob("rgb_*.png"))
    if not png_files:
        print("  [Mode B] No rgb_*.png files found")
        return records

    label_counts = Counter()
    for png_path in png_files:
        tag = png_path.stem.replace("rgb_", "")
        bbox_path = mode_b_dir / f"bbox_{tag}.npy"
        if not bbox_path.exists():
            continue

        boxes = load_bbox(bbox_path)
        try:
            img = Image.open(png_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size

        for box in boxes:
            label = box["label"]
            if label == "water":
                continue
            cls = LABEL_TO_CLASS.get(label)
            if cls is None:
                continue

            occ = box["occlusion"]
            if label in TARGET_CLASSES and occ > max_occlusion:
                continue

            x1, y1 = box["x_min"], box["y_min"]
            x2, y2 = box["x_max"], box["y_max"]
            if x2 <= x1 or y2 <= y1:
                continue

            pw = int((x2 - x1) * pad_frac)
            ph = int((y2 - y1) * pad_frac)
            cx1 = max(0, x1 - pw)
            cy1 = max(0, y1 - ph)
            cx2 = min(W, x2 + pw)
            cy2 = min(H, y2 + ph)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = img.crop((cx1, cy1, cx2, cy2)).resize((224, 224), Image.BILINEAR)
            records.append((crop, cls))
            label_counts[cls] += 1

    print(f"  [Mode B] {len(records)} crops from {len(png_files)} frames")
    for cls, cnt in sorted(label_counts.items()):
        print(f"           {cls:<8} {cnt}")
    return records


def split_and_write(records: list, out_root: Path):
    """Stratified train/val/test split, write to ImageFolder layout."""
    by_class: dict[str, list] = {}
    for img, cls in records:
        by_class.setdefault(cls, []).append(img)

    for cls in by_class:
        random.shuffle(by_class[cls])

    totals = {"train": 0, "val": 0, "test": 0}
    for cls, images in by_class.items():
        n = len(images)
        t = int(n * SPLIT[0])
        v = t + int(n * SPLIT[1])
        splits = {"train": images[:t], "val": images[t:v], "test": images[v:]}
        for subset, imgs in splits.items():
            d = out_root / subset / cls
            d.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(imgs):
                img.save(d / f"{cls}_{idx:04d}.png")
            totals[subset] += len(imgs)

    print(f"\n  Stage 2 dataset → {out_root}")
    for sub, n in totals.items():
        print(f"    {sub:<6} {n} images")
    print("\n  Per-class:")
    for sub in ["train", "val", "test"]:
        parts = []
        for cls in sorted(by_class):
            cnt = len(list((out_root / sub / cls).glob("*.png")))
            parts.append(f"{cls}: {cnt}")
        print(f"    {sub:<6} " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_a",  default="outputs/training_data/mode_a_isolated")
    ap.add_argument("--mode_b",  default="outputs/training_data/mode_b_mixed")
    ap.add_argument("--out",     default="./stage2_dataset")
    ap.add_argument("--n_aug",   type=int, default=9)
    ap.add_argument("--crop_pad", type=float, default=0.15)
    args = ap.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)

    print("── Stage 2 dataset preparation ──")
    print(f"  Mode A source: {args.mode_a}")
    print(f"  Mode B source: {args.mode_b}")
    print(f"  Augmentation:  {args.n_aug}× per Mode A image")
    print()

    records_a = collect_mode_a(Path(args.mode_a), n_aug=args.n_aug)
    records_b = collect_mode_b_crops(Path(args.mode_b), pad_frac=args.crop_pad)

    all_records = records_a + records_b
    print(f"\n  Combined: {len(all_records)} images")

    cls_counts = Counter()
    for _, cls in all_records:
        cls_counts[cls] += 1
    for cls, n in sorted(cls_counts.items()):
        bar = "█" * (n * 40 // max(cls_counts.values(), default=1))
        print(f"  {cls:<8} {n:>5}  {bar}")

    split_and_write(all_records, out_root)
    print("\n✅ Ready for Stage 2 training: python train_stage2.py --data ./stage2_dataset")


if __name__ == "__main__":
    main()

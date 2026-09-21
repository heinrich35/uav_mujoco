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

Memory-efficient: streams images to disk rather than accumulating in RAM.
Parallel: uses --workers N to spread augmentation across CPU cores.

Usage:
    python prepare_stage2.py \
        --mode_a  outputs/training_data/mode_a_isolated \
        --mode_b  outputs/training_data/mode_b_mixed \
        --out     ./stage2_dataset \
        --workers 8
"""

import argparse, json, os, random, shutil, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

random.seed(42)
np.random.seed(42)

TARGET_CLASSES    = ['duck', 'swan']
DISTRACTOR_CLASSES = ['rock', 'log', 'lilypad', 'turtle']

LABEL_TO_CLASS = {
    'duck': 'duck', 'swan': 'swan',
    'rock': 'other', 'log': 'other', 'lilypad': 'other', 'turtle': 'other',
}

SPLIT = (0.70, 0.15, 0.15)
BATCH_SIZE = 100  # records per worker invocation


# ── helpers (must be importable for subprocesses) ───────────────────────────

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


# ── module-level worker functions for ProcessPoolExecutor ─────────────────

def _worker_mode_a_batch(args):
    """
    Process a batch of Mode A source images.

    Args (serializable):
        batch: list of (png_path_str, cls, out_dir_str, base_name, n_aug, worker_seed)
    Returns: (num_written, num_errors)
    """
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    batch_data, n_aug, worker_seed = args
    random.seed(worker_seed)
    np.random.seed(worker_seed + 1)

    written = 0
    errors = 0

    for png_path_str, cls, out_dir_str, base_name in batch_data:
        try:
            img = Image.open(png_path_str).convert("RGB")
        except Exception:
            errors += 1
            continue

        cls_dir = Path(out_dir_str) / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for v_idx in range(n_aug):
            if v_idx == 0:
                aug = img.copy()
            else:
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

            aug.save(cls_dir / f"{base_name}_v{v_idx:02d}.png")
            aug.close()
            written += 1

        img.close()

    return written, errors


def _worker_mode_b_batch(args):
    """
    Process a batch of Mode B frames.

    Args (serializable):
        frames: list of (png_path_str, [(x1,y1,x2,y2,cls,label,occlusion), ...], out_dir_str,
                         pad_frac, max_occlusion)
    Returns: (num_crops, num_frames_ok)
    """
    from PIL import Image

    frames, pad_frac, max_occlusion = args
    total_crops = 0
    frames_ok = 0

    TARGETS = {'duck', 'swan'}

    for png_path_str, crop_list, out_dir_str in frames:
        try:
            img = Image.open(png_path_str).convert("RGB")
        except Exception:
            continue

        W, H = img.size
        frames_ok += 1

        for x1, y1, x2, y2, cls, label, occ in crop_list:
            if label in TARGETS and occ > max_occlusion:
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
            cls_dir = Path(out_dir_str) / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            tag = Path(png_path_str).stem.replace("rgb_", "")
            crop.save(cls_dir / f"b_{tag}_{x1}_{y1}.png")
            crop.close()
            total_crops += 1

        img.close()

    return total_crops, frames_ok


# ── batched parallel dispatch ──────────────────────────────────────────────

def _chunk_list(lst, n):
    """Split list into n roughly-equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_mode_a_parallel(records: list[dict], out_root: Path, subset: str,
                            n_aug: int, n_workers: int):
    """Parallel Mode A: split records across workers, each handles a batch."""
    # Build serializable tuples
    tasks = []
    out_dir = str(out_root / subset)
    for r in records:
        base = f"a_{r['png_path'].stem}"
        tasks.append((str(r["png_path"]), r["cls"], out_dir, base))

    # Split into batches
    batch_size = max(1, len(tasks) // (n_workers * 4))  # 4 batches per worker
    batch_size = max(BATCH_SIZE, batch_size)
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    # Add n_aug and unique seed to each batch
    payloads = [(batch, n_aug, hash(batch[0][0]) % (2**31)) for batch in batches]

    total = 0
    done = 0
    n_batches = len(batches)

    if n_workers == 1:
        # Single-threaded path — no fork overhead
        for payload in payloads:
            w, e = _worker_mode_a_batch(payload)
            total += w
            done += len(payload[0])
            if done % max(1, (len(tasks) // 40)) < len(payload[0]) or done >= len(tasks):
                pct = done * 100 // len(tasks)
                print(f"  [Mode A {subset}] {done}/{len(tasks)} sources ({pct}%) → {total} images")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_worker_mode_a_batch, p): i for i, p in enumerate(payloads)}
            for fut in futures:
                try:
                    w, e = fut.result()
                    total += w
                except Exception as exc:
                    print(f"  [WARN] worker batch failed: {exc}")
                # Progress reporting from batch sizes
                batch_idx = futures[fut]
                batch_len = len(payloads[batch_idx][0])
                done += batch_len
                pct = min(done, len(tasks)) * 100 // max(len(tasks), 1)
                eff = total / max(time.time() - _start_time, 0.1)
                print(f"  [Mode A {subset}] {min(done, len(tasks))}/{len(tasks)} "
                      f"sources ({pct}%) → {total} images  {eff:.0f} img/s")

    print(f"  [Mode A {subset}] done: {len(tasks)} sources × {n_aug} = {total} images")
    return total


def process_mode_b_parallel(records: list[dict], out_root: Path, subset: str,
                            pad_frac: float, max_occlusion: float, n_workers: int):
    """Parallel Mode B: group by frame, distribute frames across workers."""
    # Group records by png_path
    by_frame: dict[str, list] = {}
    for r in records:
        key = str(r["png_path"])
        by_frame.setdefault(key, []).append(r)

    # Build serializable frame tasks
    tasks = []
    out_dir = str(out_root / subset)
    for png_path_str, crop_records in by_frame.items():
        crop_list = []
        for r in crop_records:
            crop_list.append((
                r["x1"], r["y1"], r["x2"], r["y2"],
                r["cls"], r["label"], r["occlusion"],
            ))
        tasks.append((png_path_str, crop_list, out_dir))

    # Split into batches
    batch_size = max(1, len(tasks) // (n_workers * 4))
    batch_size = max(BATCH_SIZE, batch_size)
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    payloads = [(batch, pad_frac, max_occlusion) for batch in batches]

    total_crops = 0
    n_frames = len(tasks)
    done_frames = 0

    if n_workers == 1:
        for payload in payloads:
            c, f = _worker_mode_b_batch(payload)
            total_crops += c
            done_frames += len(payload[0])
            if done_frames % max(1, (n_frames // 40)) < len(payload[0]) or done_frames >= n_frames:
                pct = done_frames * 100 // max(n_frames, 1)
                print(f"  [Mode B {subset}] {done_frames}/{n_frames} frames ({pct}%) → {total_crops} crops")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_worker_mode_b_batch, p): i for i, p in enumerate(payloads)}
            for fut in futures:
                try:
                    c, f = fut.result()
                    total_crops += c
                except Exception as exc:
                    print(f"  [WARN] worker batch failed: {exc}")
                batch_idx = futures[fut]
                done_frames += len(payloads[batch_idx][0])
                pct = min(done_frames, n_frames) * 100 // max(n_frames, 1)
                print(f"  [Mode B {subset}] {min(done_frames, n_frames)}/{n_frames} "
                      f"frames ({pct}%) → {total_crops} crops")

    print(f"  [Mode B {subset}] done: {n_frames} frames → {total_crops} crops")
    return total_crops


# Track start time for throughput reporting
_start_time = 0.0


# ── Phase 1: collect metadata (no images loaded) ───────────────────────────

def collect_mode_a_meta(mode_a_dir: Path) -> list[dict]:
    """Scan Mode A files. Returns list of {png_path, cls} dicts (lightweight)."""
    records = []
    png_files = sorted(mode_a_dir.glob("**/rgb_*.png"))
    if not png_files:
        print("  [Mode A] No rgb_*.png files found")
        return records

    label_counts = Counter()
    for png_path in png_files:
        tag = png_path.stem.replace("rgb_", "")
        bbox_path = png_path.parent / f"bbox_{tag}.npy"
        if not bbox_path.exists():
            continue

        boxes = load_bbox(bbox_path)
        obj_label = "other"
        for box in boxes:
            if box["label"] != "water":
                obj_label = box["label"]
                break

        cls = LABEL_TO_CLASS.get(obj_label, "other")
        label_counts[cls] += 1
        records.append({"png_path": png_path, "cls": cls})

    print(f"  [Mode A] {len(records)} source images")
    for cls_name, cnt in sorted(label_counts.items()):
        print(f"           {cls_name:<8} {cnt} source images")
    return records


def collect_mode_b_meta(mode_b_dir: Path) -> list[dict]:
    """Scan Mode B files. Returns list of {png_path, x1,y1,x2,y2, cls} dicts (lightweight)."""
    records = []
    png_files = sorted(mode_b_dir.glob("**/rgb_*.png"))
    if not png_files:
        print("  [Mode B] No rgb_*.png files found")
        return records

    label_counts = Counter()
    for png_path in png_files:
        tag = png_path.stem.replace("rgb_", "")
        bbox_path = png_path.parent / f"bbox_{tag}.npy"
        if not bbox_path.exists():
            continue

        boxes = load_bbox(bbox_path)
        for box in boxes:
            label = box["label"]
            if label == "water":
                continue
            cls = LABEL_TO_CLASS.get(label)
            if cls is None:
                continue

            x1, y1 = box["x_min"], box["y_min"]
            x2, y2 = box["x_max"], box["y_max"]
            if x2 <= x1 or y2 <= y1:
                continue

            records.append({
                "png_path": png_path,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cls": cls,
                "label": label,
                "occlusion": box["occlusion"],
            })
            label_counts[cls] += 1

    print(f"  [Mode B] {len(records)} crops from {len(png_files)} frames")
    for cls_name, cnt in sorted(label_counts.items()):
        print(f"           {cls_name:<8} {cnt}")
    return records


# ── Phase 2: determine split assignments ───────────────────────────────────

def assign_splits(records_a: list[dict], records_b: list[dict]
                  ) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Stratified shuffle + split. Returns train_a/val_a/test_a dicts, same for b."""
    def _split_list(items: list[dict]) -> dict[str, list[dict]]:
        shuffled = items.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        t = int(n * SPLIT[0])
        v = t + int(n * SPLIT[1])
        return {
            "train": shuffled[:t],
            "val":   shuffled[t:v],
            "test":  shuffled[v:],
        }

    # Stratify by class
    by_class_a: dict[str, list[dict]] = {}
    for r in records_a:
        by_class_a.setdefault(r["cls"], []).append(r)

    by_class_b: dict[str, list[dict]] = {}
    for r in records_b:
        by_class_b.setdefault(r["cls"], []).append(r)

    splits_a = {"train": [], "val": [], "test": []}
    splits_b = {"train": [], "val": [], "test": []}

    for cls_name, items in by_class_a.items():
        s = _split_list(items)
        for subset in ["train", "val", "test"]:
            splits_a[subset].extend(s[subset])

    for cls_name, items in by_class_b.items():
        s = _split_list(items)
        for subset in ["train", "val", "test"]:
            splits_b[subset].extend(s[subset])

    return splits_a, splits_b


# ── main ────────────────────────────────────────────────────────────────────

def main():
    global _start_time

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_a",   default="outputs/training_data/mode_a_isolated")
    ap.add_argument("--mode_b",   default="outputs/training_data/mode_b_mixed")
    ap.add_argument("--out",      default="./stage2_dataset")
    ap.add_argument("--n_aug",    type=int, default=9)
    ap.add_argument("--crop_pad", type=float, default=0.15)
    ap.add_argument("--workers",  type=int, default=0,
                    help="Number of parallel workers (default: CPU count)")
    args = ap.parse_args()

    n_workers = args.workers if args.workers > 0 else os.cpu_count() or 4
    print(f"  Workers: {n_workers}")

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)

    print("── Stage 2 dataset preparation ──")
    print(f"  Mode A source: {args.mode_a}")
    print(f"  Mode B source: {args.mode_b}")
    print(f"  Augmentation:  {args.n_aug}× per Mode A image")
    print()

    # Phase 1: collect metadata (no images loaded)
    print("Phase 1: scanning source files...")
    meta_a = collect_mode_a_meta(Path(args.mode_a))
    meta_b = collect_mode_b_meta(Path(args.mode_b))

    total_expected = len(meta_a) * args.n_aug + len(meta_b)
    print(f"\n  Expected output: ~{total_expected} images")
    cls_counts = Counter()
    for r in meta_a:
        cls_counts[r["cls"]] += args.n_aug
    for r in meta_b:
        cls_counts[r["cls"]] += 1
    for cls_name, n in sorted(cls_counts.items()):
        bar = "█" * (n * 40 // max(cls_counts.values(), default=1))
        print(f"  {cls_name:<8} {n:>5}  {bar}")

    # Phase 2: determine split assignments
    print("\nPhase 2: assigning train/val/test splits...")
    splits_a, splits_b = assign_splits(meta_a, meta_b)
    for subset in ["train", "val", "test"]:
        na = len(splits_a[subset])
        nb = len(splits_b[subset])
        print(f"  {subset:<6} {na} Mode A sources + {nb} Mode B crops "
              f"≈ {na * args.n_aug + nb} images")

    # Phase 3: process and write
    print(f"\nPhase 3: processing & writing ({n_workers} workers)...")
    _start_time = time.time()

    for subset in ["train", "val", "test"]:
        print(f"\n  ── {subset} ──")
        if splits_a[subset]:
            process_mode_a_parallel(splits_a[subset], out_root, subset,
                                    args.n_aug, n_workers)
        if splits_b[subset]:
            process_mode_b_parallel(splits_b[subset], out_root, subset,
                                    pad_frac=args.crop_pad, max_occlusion=0.55,
                                    n_workers=n_workers)

    elapsed = time.time() - _start_time
    # Final counts
    print(f"\n  Stage 2 dataset → {out_root}")
    totals = {}
    for subset in ["train", "val", "test"]:
        n = 0
        for cls_name in ["duck", "swan", "other"]:
            d = out_root / subset / cls_name
            if d.exists():
                n += len(list(d.glob("*.png")))
        totals[subset] = n
        parts = []
        for cls_name in ["duck", "swan", "other"]:
            d = out_root / subset / cls_name
            cnt = len(list(d.glob("*.png"))) if d.exists() else 0
            parts.append(f"{cls_name}: {cnt}")
        print(f"    {subset:<6} {n} images  " + "  ".join(parts))

    total_written = sum(totals.values())
    print(f"\n✅ Total written: {total_written} images  ({elapsed:.1f}s, {total_written/elapsed:.0f} img/s)")
    print(f"✅ Ready for Stage 2 training:")
    print(f"   python scripts/pond/train_stage2.py --data {args.out} --run_name pond_s2_v2 --device 0")


if __name__ == "__main__":
    main()

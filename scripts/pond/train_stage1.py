"""
train_stage1.py
───────────────
Stage 1 detector — YOLOv11 for the Gosling pond scene.
Detects 6 classes: duck(0), swan(1), turtle(2), lilypad(3), log(4), rock(5).

Prerequisites:
    pip install ultralytics

Usage:
    python train_stage1.py --data ./pond_yolo/pond.yaml --run_name pond_v2
    python train_stage1.py --eval_only runs/.../best.pt
    python train_stage1.py --preset small_object --run_name pond_v2
"""

import argparse, json, sys
from pathlib import Path

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Base parameters — shared across all presets
# ═══════════════════════════════════════════════════════════════════════════════

STAGE1_BASE = dict(
    model         = 'yolo11m.pt',     # pretrained COCO backbone
    imgsz         = 640,              # Gosling camera (640×480, 24mm focal)
    batch         = 8,                # fits RTX 5070 Ti 16 GB
    optimizer     = 'AdamW',
    lr0           = 0.001,            # initial learning rate
    lrf           = 0.01,             # final LR factor: lr0 * lrf
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 5,
    cos_lr        = True,             # cosine LR schedule
    deterministic = True,
    # ── Augmentation ──────────────────────────────────────────────────────────
    hsv_h         = 0.015,
    hsv_s         = 0.4,
    hsv_v         = 0.3,
    flipud        = 0.0,              # no vertical flip — sky must stay on top
    fliplr        = 0.5,              # horizontal flip is fine for water scenes
    mosaic        = 0.5,
    mixup         = 0.1,
    copy_paste    = 0.05,
    close_mosaic  = 10,
    scale         = 0.5,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Presets — choose with --preset
# ═══════════════════════════════════════════════════════════════════════════════

STAGE1_PRESETS = {
    "default": dict(
        description = "Balanced — original tuned parameters (v1 baseline)",
        cls          = 0.5,
        cls_pw       = 0.0,
        box          = 7.5,
        dfl          = 1.5,
        epochs       = 120,           # 20 frozen + 100 unfrozen
        patience     = 25,
    ),
    "small_object": dict(
        description = "Small/distant object focused — higher cls weight, "
                      "focal-loss positive bias, longer phase 2",
        cls          = 1.0,           # 2× classification loss weight
        cls_pw       = 1.0,           # positive-sample boost (max allowed)
        box          = 8.5,           # higher box loss for better localization
        dfl          = 2.0,           # higher DFL for edge regression
        copy_paste   = 0.15,          # 3× more copy-paste augmentation
        mosaic       = 0.7,           # more mosaic
        scale        = 0.7,           # wider scale jitter
        close_mosaic = 15,            # keep mosaic longer
        epochs       = 220,           # 20 frozen + 200 unfrozen
        patience     = 50,            # longer exploration before early stop
        lrf          = 0.005,         # gentler cosine tail
    ),
    "fast": dict(
        description = "Quick iteration — reduced epochs, aggressive early stop",
        cls          = 0.5,
        cls_pw       = 0.0,
        box          = 7.5,
        dfl          = 1.5,
        epochs       = 60,            # 20 frozen + 40 unfrozen
        patience     = 15,
    ),
}

FREEZE_BACKBONE_EPOCHS = 20

CLASS_NAMES = {0: 'duck', 1: 'swan', 2: 'turtle', 3: 'lilypad', 4: 'log', 5: 'rock'}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_params(preset_name: str) -> dict:
    """Merge base + preset. Preset keys override base keys."""
    if preset_name not in STAGE1_PRESETS:
        raise KeyError(f"Unknown preset '{preset_name}'. "
                       f"Choices: {list(STAGE1_PRESETS)}")
    p = {**STAGE1_BASE, **STAGE1_PRESETS[preset_name]}
    # Non-training keys
    p.pop('description', None)
    return p


def _compute_size_distribution(data_yaml: str, split: str = 'test'):
    """Parse YOLO-format labels and return COCO area bin counts."""
    data_root = Path(data_yaml).parent
    label_dir = data_root / 'labels' / split
    if not label_dir.exists():
        print(f"  [WARN] Label dir not found: {label_dir}")
        return {}, np.array([])

    areas = []
    for lbl_file in sorted(label_dir.glob('*.txt')):
        text = lbl_file.read_text().strip()
        if not text:
            continue
        for line in text.split('\n'):
            parts = line.split()
            if len(parts) < 5:
                continue
            w = float(parts[3]) * 640   # normalized w → px
            h = float(parts[4]) * 480   # normalized h → px
            areas.append(w * h)

    areas = np.array(areas)
    if len(areas) == 0:
        return {}, areas

    n_small  = int((areas < 32**2).sum())
    n_medium = int(((areas >= 32**2) & (areas < 96**2)).sum())
    n_large  = int((areas >= 96**2).sum())
    total = len(areas)

    return {
        "small":  n_small,
        "medium": n_medium,
        "large":  n_large,
        "total":  total,
    }, areas


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(data: str, weights: str, device: str, split: str = 'test'):
    """
    Comprehensive evaluation: mAP@50, mAP@50-95, per-class AP,
    and COCO-style area distribution.

    Returns dict with all metrics for downstream use.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.val(data=data, device=device, split=split,
                        conf=0.001, iou=0.7,  # COCO-standard thresholds
                        plots=False, verbose=False)

    # ── Extract metrics (DetMetrics API) ─────────────────────────────────
    mp, mr, map50, map50_95 = results.box.mean_results()
    ap_class_index = results.box.ap_class_index
    per_class_ap50 = results.box.ap50    # (nc,) array
    per_class_ap    = results.box.ap     # (nc,) per-class AP@50-95
    # Per-class P, R
    try:
        class_result = results.box.class_result
    except AttributeError:
        class_result = None

    # ── Print report ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVALUATION — {split} split")
    print(f"  Weights: {weights}")
    print(f"{'='*60}")
    print(f"\n  Overall:")
    print(f"    mAP@50    = {map50:.4f}")
    print(f"    mAP@75    = {getattr(results.box, 'map75', float('nan')):.4f}")
    print(f"    mAP@50-95 = {map50_95:.4f}")
    print(f"    Gap (50-95)= {float(map50 - map50_95):.4f}  "
          f"({float(map50 - map50_95)*100:.1f} pp)")
    print(f"    Precision  = {mp:.4f}")
    print(f"    Recall     = {mr:.4f}")

    # ── Per-class table ──────────────────────────────────────────────────
    print(f"\n  Per-class:")
    print(f"    {'Class':<8} {'P':>7} {'R':>7} {'mAP@50':>8} {'mAP@50-95':>10}")
    print(f"    {'-'*44}")
    per_class_metrics = {}
    for idx, cls_id in enumerate(ap_class_index):
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        ap50_val = float(per_class_ap50[idx])
        ap_val   = float(per_class_ap[idx])
        if class_result is not None:
            try:
                p, r, _, _ = class_result(idx)
                p_val = float(p)
                r_val = float(r)
            except Exception:
                p_val, r_val = 0.0, 0.0
        else:
            p_val, r_val = 0.0, 0.0
        per_class_metrics[name] = {
            "p": p_val, "r": r_val,
            "ap50": ap50_val, "ap50_95": ap_val,
        }
        print(f"    {name:<8} {p_val:7.4f} {r_val:7.4f} "
              f"{ap50_val:8.4f} {ap_val:10.4f}")

    # ── Object size distribution ─────────────────────────────────────────
    size_dist, areas = _compute_size_distribution(data, split)
    if size_dist:
        total = size_dist["total"]
        print(f"\n  Object size distribution ({split} set, COCO bins):")
        print(f"    Small  (< 1024 px²):   {size_dist['small']:>6}"
              f"  ({100*size_dist['small']/total:.1f}%)")
        print(f"    Medium (1k–9.2k px²):  {size_dist['medium']:>6}"
              f"  ({100*size_dist['medium']/total:.1f}%)")
        print(f"    Large  (> 9.2k px²):   {size_dist['large']:>6}"
              f"  ({100*size_dist['large']/total:.1f}%)")
        print(f"    NOTE: For detailed AP by size, install pycocotools and run:")
        print(f"          pip install pycocotools")
        print(f"    Then re-run evaluation for COCO-style AP-small/medium/large.")

    # ── Localization quality ─────────────────────────────────────────────
    map75 = getattr(results.box, 'map75', None)
    if map75 is not None and map50 > 0:
        loc_ratio = float(map75 / map50)
        print(f"\n  Localization quality: mAP@75/mAP@50 = {loc_ratio:.4f}"
              f"  (target > 0.95)")

    print(f"{'='*60}\n")

    return {
        "map50": float(map50),
        "map50_95": float(map50_95),
        "precision": float(mp),
        "recall": float(mr),
        "per_class": per_class_metrics,
        "size_distribution": size_dist,
        "gap_pp": float(map50 - map50_95),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(data: str, run_name: str, device: str, params: dict,
          freeze_epochs: int, resume_from: str = ""):
    """Two-phase training: frozen backbone → full fine-tune."""
    from ultralytics import YOLO

    PROJECT = 'runs/pond_stage1'
    # Exclude meta keys from YOLO training kwargs
    EXCLUDE = {'model', 'lr0', 'epochs', 'description', 'patience'}
    train_kwargs = {k: v for k, v in params.items() if k not in EXCLUDE}

    phase2_epochs = params['epochs'] - freeze_epochs

    if resume_from:
        # ── Phase 2 only ──────────────────────────────────────────────────
        phase1_best = resume_from
        if not Path(phase1_best).exists():
            raise FileNotFoundError(f"Phase-1 checkpoint not found: {phase1_best}")

        print(f"\n{'='*60}")
        print(f"STAGE 1 — Phase 2 only (resume from {resume_from})")
        print(f"  {phase2_epochs} epochs, full fine-tune")
        print(f"  cls={params['cls']}  cls_pw={params['cls_pw']}"
              f"  patience={params['patience']}")
        print(f"{'='*60}\n")

        model = YOLO(phase1_best)
        model.train(
            data    = data,
            project = PROJECT,
            name    = f'{run_name}_phase2',
            device  = device,
            freeze  = 0,
            epochs  = phase2_epochs,
            lr0     = params['lr0'] / 10,
            patience = params['patience'],
            **train_kwargs,
        )

        best_final = str(Path(model.trainer.save_dir) / 'weights' / 'best.pt')

    else:
        # ── Phase 1: frozen backbone ──────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"STAGE 1 — Phase 1: frozen backbone ({freeze_epochs} epochs)")
        print(f"  cls={params['cls']}  cls_pw={params['cls_pw']}"
              f"  lr0={params['lr0']}")
        print(f"{'='*60}\n")

        model = YOLO(params['model'])
        model.train(
            data    = data,
            project = PROJECT,
            name    = f'{run_name}_phase1',
            device  = device,
            freeze  = 10,           # freeze first 10 backbone layers
            epochs  = freeze_epochs,
            patience = params['patience'],
            **train_kwargs,
        )

        # YOLO may append -N to the name if the directory exists;
        # read back the actual save_dir from the trainer
        best_phase1 = str(Path(model.trainer.save_dir) / 'weights' / 'best.pt')
        print(f"   Phase 1 best → {best_phase1}")

        # ── Phase 2: full fine-tune ───────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"STAGE 1 — Phase 2: full fine-tune ({phase2_epochs} epochs)")
        print(f"  cls={params['cls']}  cls_pw={params['cls_pw']}"
              f"  lr0={params['lr0'] / 10}  patience={params['patience']}")
        print(f"{'='*60}\n")

        model2 = YOLO(best_phase1)
        model2.train(
            data    = data,
            project = PROJECT,
            name    = f'{run_name}_phase2',
            device  = device,
            freeze  = 0,
            epochs  = phase2_epochs,
            lr0     = params['lr0'] / 10,
            patience = params['patience'],
            **train_kwargs,
        )

        best_final = str(Path(model2.trainer.save_dir) / 'weights' / 'best.pt')

    print(f"\n✅ Stage 1 training complete")
    print(f"   Best weights → {best_final}")

    # ── Post-training evaluation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1 — Post-training evaluation (test set)")
    print(f"{'='*60}")
    test_metrics = evaluate(data, best_final, device, split='test')

    # Save metrics alongside weights
    metrics_path = (
        Path(best_final).parent.parent / 'test_metrics.json'
    )
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"   Metrics saved → {metrics_path}")
    print(f"\n   To use this model in launch_pond.py, update key 1 with:")
    print(f"   '{best_final}'")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="YOLO11 Stage 1 detector — 6-class pond object detection")
    ap.add_argument('--data',      default='./pond_yolo/pond.yaml')
    ap.add_argument('--run_name',  default='pond_v2')
    ap.add_argument('--device',    default='0', help='GPU id or "cpu"')
    ap.add_argument('--preset',    default='small_object',
                    choices=list(STAGE1_PRESETS),
                    help='Hyperparameter preset (default: small_object)')
    ap.add_argument('--resume_from', default='',
                    metavar='PHASE1_WEIGHTS',
                    help='Skip phase 1, run phase 2 only from this checkpoint')
    ap.add_argument('--eval_only', default=None,
                    metavar='WEIGHTS',
                    help='Skip training, evaluate this checkpoint')
    ap.add_argument('--eval_split', default='test',
                    choices=['val', 'test'],
                    help='Split for --eval_only (default: test)')
    args = ap.parse_args()

    # Build effective parameters
    params = build_params(args.preset)
    preset_desc = STAGE1_PRESETS[args.preset]["description"]

    # Print configuration
    print(f"\n{'='*60}")
    print(f"STAGE 1 CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Preset:      {args.preset} — {preset_desc}")
    print(f"  Model:       {params['model']}")
    print(f"  Run name:    {args.run_name}")
    print(f"  Data:        {args.data}")
    print(f"  Device:      {args.device}")
    print(f"  Epochs:      {params['epochs']}  "
          f"(Phase 1: {FREEZE_BACKBONE_EPOCHS} frozen + "
          f"Phase 2: {params['epochs'] - FREEZE_BACKBONE_EPOCHS} unfrozen)")
    print(f"  Batch:       {params['batch']}  |  imgsz: {params['imgsz']}")
    print(f"  cls:         {params['cls']}  |  cls_pw: {params['cls_pw']}")
    print(f"  box:         {params['box']}  |  dfl: {params['dfl']}")
    print(f"  Patience:    {params['patience']}")
    print(f"  copy_paste:  {params['copy_paste']}  |  mosaic: {params['mosaic']}"
          f"  |  scale: {params['scale']}")
    print(f"{'='*60}\n")

    if args.eval_only:
        evaluate(args.data, args.eval_only, args.device, split=args.eval_split)
    else:
        train(args.data, args.run_name, args.device, params,
              FREEZE_BACKBONE_EPOCHS, resume_from=args.resume_from)


if __name__ == '__main__':
    main()

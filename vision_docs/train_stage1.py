"""
train_stage1.py
───────────────
Stage 1 detector — YOLOv11 for the Gosling pond scene.
Detects duck (0), swan (1), other (2) from the robot camera stream.

Prerequisites:
    pip install ultralytics

Usage:
    python train_stage1.py --data ./pond_yolo/pond.yaml --run_name pond_v1
"""

import argparse
from pathlib import Path

# ── Training hyperparameters ──────────────────────────────────────────────────
# Tuned for a low-angle, water-surface robot camera with:
# - Objects in lower half of image (on water surface)
# - Wide distance range (1.5 m – 12 m)
# - High visual similarity between duck and swan at range
STAGE1_PARAMS = dict(
    model      = 'yolo11m.pt',     # pretrained COCO backbone
    imgsz      = 640,              # Gosling camera (640×480, 24mm focal)
    epochs     = 120,
    batch      = 8,                # adjust to GPU VRAM
    optimizer  = 'AdamW',
    lr0        = 0.001,            # initial learning rate
    lrf        = 0.01,             # final LR = lr0 * lrf
    momentum   = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 5,
    cos_lr     = True,             # cosine LR schedule
    patience   = 25,               # early stopping patience
    # ── Augmentation ──────────────────────────────────────────────────────────
    # Domain randomization already done by Replicator —
    # keep augmentation moderate to avoid fighting it
    hsv_h      = 0.015,
    hsv_s      = 0.4,
    hsv_v      = 0.3,
    flipud     = 0.0,              # no vertical flip — sky must stay on top
    fliplr     = 0.5,              # horizontal flip is fine for water scenes
    mosaic     = 0.5,              # mosaic augmentation at 50 % probability
    mixup      = 0.1,
    copy_paste = 0.05,
    # ── Pond-specific bbox anchoring ──────────────────────────────────────────
    # Objects occupy bottom half of image; aspect ratios vary widely:
    # - duck at 2 m range: large box ~200x150 px
    # - duck at 12 m range: small box ~30x20 px
    # YOLOv11 uses anchor-free detection, so no manual anchor tuning needed
    # but imgsz=960 is critical for small distant objects
)

# ── Transfer learning strategy ────────────────────────────────────────────────
# Phase 1: freeze backbone, train detection head only (fast convergence)
# Phase 2: unfreeze all, fine-tune end-to-end at low LR
FREEZE_BACKBONE_EPOCHS = 20


def train(data: str, run_name: str, device: str):
    from ultralytics import YOLO

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1 — Phase 1: frozen backbone ({FREEZE_BACKBONE_EPOCHS} epochs)")
    print(f"{'='*60}\n")

    model = YOLO(STAGE1_PARAMS['model'])
    model.train(
        data    = data,
        project = 'runs/pond_stage1',
        name    = f'{run_name}_phase1',
        device  = device,
        freeze  = 10,          # freeze first 10 backbone layers
        epochs  = FREEZE_BACKBONE_EPOCHS,
        **{k: v for k, v in STAGE1_PARAMS.items()
           if k not in ('model','epochs')},
    )

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1 — Phase 2: full fine-tune ({STAGE1_PARAMS['epochs'] - FREEZE_BACKBONE_EPOCHS} epochs)")
    print(f"{'='*60}\n")

    best_phase1 = f"runs/pond_stage1/{run_name}_phase1/weights/best.pt"
    model2 = YOLO(best_phase1)
    model2.train(
        data    = data,
        project = 'runs/pond_stage1',
        name    = f'{run_name}_phase2',
        device  = device,
        freeze  = 0,           # unfreeze everything
        epochs  = STAGE1_PARAMS['epochs'] - FREEZE_BACKBONE_EPOCHS,
        lr0     = STAGE1_PARAMS['lr0'] / 10,   # 10x lower LR for fine-tune
        **{k: v for k, v in STAGE1_PARAMS.items()
           if k not in ('model','lr0','epochs')},
    )

    best_final = f"runs/pond_stage1/{run_name}_phase2/weights/best.pt"
    print(f"\n✅ Stage 1 training complete")
    print(f"   Best weights → {best_final}")
    print(f"\n   Evaluate:")
    print(f"   yolo detect val data={data} model={best_final}")
    print(f"\n   Export for on-robot inference:")
    print(f"   yolo export model={best_final} format=engine half=True  # TensorRT")


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(data: str, weights: str, device: str):
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.val(data=data, device=device, split='test',
                        conf=0.25, iou=0.5)

    print("\n── Per-class mAP@50 ──────────────────────────────────────────")
    class_names = {0: 'duck', 1: 'swan', 2: 'other'}
    for i, name in class_names.items():
        ap = results.box.ap50[i] if hasattr(results.box, 'ap50') else '?'
        print(f"  {name:<8}  mAP@50 = {ap:.4f}" if isinstance(ap, float) else f"  {name}: {ap}")

    print("\n── Confusion matrix note ────────────────────────────────────")
    print("  Key metric: duck→swan and swan→duck off-diagonal cells")
    print("  Both should be < 5 % of their class total.")
    print("  If not → increase Stage 2 training data and contrastive loss.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',     default='./pond_yolo/pond.yaml')
    ap.add_argument('--run_name', default='pond_v1')
    ap.add_argument('--device',   default='0', help='GPU id or "cpu"')
    ap.add_argument('--eval_only', default=None,
                    metavar='WEIGHTS', help='Skip training, evaluate this checkpoint')
    args = ap.parse_args()

    if args.eval_only:
        evaluate(args.data, args.eval_only, args.device)
    else:
        train(args.data, args.run_name, args.device)


if __name__ == '__main__':
    main()

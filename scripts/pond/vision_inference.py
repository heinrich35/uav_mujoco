"""
vision_inference.py — Real-time 6-class pond object detection for gosling camera.

Integrates Stage 1 (YOLO 6-class detector) and optional Stage 2 (EfficientNet
duck/swan classifier) into the Isaac Lab pond simulation loop.

Classes: duck(0), swan(1), turtle(2), lilypad(3), log(4), rock(5)

Features:
  - SAHI (Slicing Aided Hyper Inference) — tiled inference for small/distant objects
  - WBF (Weighted Boxes Fusion) — ensemble multiple models into one result

Usage (from launch_pond.py):
    from scripts.pond.vision_inference import VisionPipeline, EnsemblePipeline

    # Single model
    detector = VisionPipeline(stage1_weights="...", stage2_weights="...", device="cuda:0")

    # Ensemble (SAHI + WBF)
    detector = EnsemblePipeline(
        pipelines=[pipeline_a, pipeline_b],
        sahi=True, sahi_rows=2, sahi_overlap=0.3,
    )

    # In simulation loop:
    detections = detector.infer(rgb_tensor)  # tensor shape (H, W, 3)
    detector.log_detections(detections)
"""

import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Detection:
    """Single detection from the vision pipeline."""
    label: str          # "duck", "swan", "turtle", "lilypad", "log", "rock"
    class_id: int       # 0-5
    confidence: float   # Stage 1 confidence
    stage2_conf: Optional[dict]  # Stage 2 duck/swan probs (only for duck/swan)
    bbox: tuple         # (x1, y1, x2, y2) in pixel coords
    size_px: int        # bbox area in pixels
    center: tuple       # (cx, cy) in pixel coords


# ═══════════════════════════════════════════════════════════════════════════════
# WBF — Weighted Boxes Fusion (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

def _box_iou(box1, box2):
    """Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def weighted_boxes_fusion(detections_list: list[list[Detection]],
                          iou_thr: float = 0.55,
                          skip_box_thr: float = 0.001) -> list[Detection]:
    """
    Fuse detections from multiple models into one set.

    Algorithm:
      1. Pool all boxes; sort by confidence descending.
      2. For each box, find all lower-confidence boxes with IoU > iou_thr
         AND same class → merge them (weighted average of coords + confidence).
      3. Skip merged boxes.
      4. Return fused set.

    Args:
        detections_list: list of detection lists, one per model.
        iou_thr: IoU threshold for merging (0.55 = standard WBF).
        skip_box_thr: drop fused boxes below this confidence.
    Returns:
        fused list of Detection objects.
    """
    # Collect all detections with model weight = 1.0
    all_boxes = []  # (x1,y1,x2,y2, conf, cls_id, label, s2_conf)
    for dets in detections_list:
        for d in dets:
            all_boxes.append({
                "bbox": list(d.bbox),
                "conf": d.confidence,
                "cls_id": d.class_id,
                "label": d.label,
                "s2_conf": d.stage2_conf,
            })

    if len(all_boxes) <= 1:
        return [d for dets in detections_list for d in dets]

    # Sort by confidence descending
    all_boxes.sort(key=lambda b: b["conf"], reverse=True)

    fused = []
    used = [False] * len(all_boxes)

    for i, box_a in enumerate(all_boxes):
        if used[i]:
            continue

        # Find all boxes of same class to merge
        cluster = [box_a]
        indices = [i]
        used[i] = True

        for j, box_b in enumerate(all_boxes):
            if used[j]:
                continue
            if box_b["cls_id"] != box_a["cls_id"]:
                continue
            iou = _box_iou(box_a["bbox"], box_b["bbox"])
            if iou > iou_thr:
                cluster.append(box_b)
                indices.append(j)
                used[j] = True

        if len(cluster) == 1:
            fused.append(box_a)
        else:
            # Weighted average: weight = confidence
            weights = [b["conf"] for b in cluster]
            total_w = sum(weights)
            if total_w == 0:
                continue

            fused_bbox = [
                sum(b["bbox"][k] * weights[bi] for bi, b in enumerate(cluster)) / total_w
                for k in range(4)
            ]
            fused_conf = sum(b["conf"] * weights[bi] for bi, b in enumerate(cluster)) / total_w
            # Scale confidence by cluster size (more models agreeing → higher confidence)
            fused_conf *= min(1.0, len(cluster) / len(detections_list) + 0.5)

            if fused_conf < skip_box_thr:
                continue

            # Take stage2_conf from the highest-confidence box in the cluster
            best = cluster[0]
            fused.append({
                "bbox": fused_bbox,
                "conf": fused_conf,
                "cls_id": best["cls_id"],
                "label": best["label"],
                "s2_conf": best["s2_conf"],
            })

    # Convert to Detection objects
    results = []
    for b in fused:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(Detection(
            label=b["label"],
            class_id=b["cls_id"],
            confidence=float(b["conf"]),
            stage2_conf=b["s2_conf"],
            bbox=(x1, y1, x2, y2),
            size_px=int((x2 - x1) * (y2 - y1)),
            center=((x1 + x2) // 2, (y1 + y2) // 2),
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VisionPipeline — single-model YOLO + Stage-2 classifier
# ═══════════════════════════════════════════════════════════════════════════════

class VisionPipeline:
    """Two-stage vision pipeline: YOLO 6-class detector → EfficientNet classifier."""

    CLASS_NAMES = {
        0: "duck", 1: "swan", 2: "turtle",
        3: "lilypad", 4: "log", 5: "rock",
    }
    CLASS_COLORS = {
        "duck":    (0, 215, 255),    # gold
        "swan":    (255, 255, 255),  # white
        "turtle":  (255, 160, 60),   # bright blue
        "lilypad": (120, 255, 60),   # lime green
        "log":     (180, 110, 0),    # deep sky blue
        "rock":    (255, 100, 160),  # violet
    }
    CLASS_EMOJI = {
        "duck": "🦆", "swan": "🦢", "turtle": "🐢",
        "lilypad": "🍀", "log": "🪵", "rock": "🪨",
    }
    S2_CLASSES = {0, 1}

    def __init__(self, stage1_weights: str, stage2_weights: str = "",
                 device: str = "cuda:0", imgsz: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 name: str = ""):
        self.device = device
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold   # mutable at runtime
        self.iou_threshold = iou_threshold     # mutable at runtime
        self.name = name or stage1_weights.split("/")[-3]
        self._has_s2 = bool(stage2_weights)

        from ultralytics import YOLO
        self.detector = YOLO(stage1_weights)
        self.detector.to(device)

        self.classifier = None
        self.norm_mean = None
        self.norm_std = None
        if self._has_s2:
            self._load_stage2(stage2_weights)

        self._last_log_time = 0.0

    def _load_stage2(self, stage2_weights: str):
        import torch.nn as nn
        try:
            import timm
        except ImportError:
            print("  [WARN] timm not installed — Stage 2 disabled")
            self._has_s2 = False
            return

        class PondClassifier(nn.Module):
            def __init__(self, backbone="efficientnet_b4", num_classes=3):
                super().__init__()
                self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
                self.head = nn.Linear(self.encoder.num_features, num_classes)
            def forward(self, x):
                return self.head(self.encoder(x))

        self.classifier = PondClassifier()
        state = torch.load(stage2_weights, map_location=self.device, weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("proj.")}
        self.classifier.load_state_dict(state)
        self.classifier.to(self.device)
        self.classifier.eval()
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.norm_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def preprocess_image(self, rgb_tensor: torch.Tensor) -> np.ndarray:
        arr = rgb_tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return arr

    def _detect_on_image(self, img: np.ndarray) -> list[Detection]:
        """Run YOLO + Stage-2 on a single image (full or tile). Returns Detection list."""
        results = self.detector(img, verbose=False, conf=self.conf_threshold,
                                iou=self.iou_threshold,
                                imgsz=self.imgsz, device=self.device)
        if not results or len(results[0].boxes) == 0:
            return []

        boxes  = results[0].boxes.xyxy.cpu().numpy()
        confs  = results[0].boxes.conf.cpu().numpy()
        clsids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Stage 2: classify duck/swan crops
        crops_for_s2, s2_indices = [], []
        if self._has_s2:
            for i, cls_id in enumerate(clsids):
                if cls_id in self.S2_CLASSES:
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_for_s2.append(crop)
                        s2_indices.append(i)

        stage2_probs = {}
        if crops_for_s2:
            stage2_probs = self._classify_crops(crops_for_s2, s2_indices)

        detections = []
        for i, cls_id in enumerate(clsids):
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            label = self.CLASS_NAMES.get(cls_id, "unknown")
            s2_conf = stage2_probs.get(i, None)
            if s2_conf is not None:
                s2_label = max(s2_conf, key=s2_conf.get)
                if s2_label in ("duck", "swan") and s2_conf[s2_label] > 0.5:
                    label = s2_label
                    cls_id = 0 if label == "duck" else 1

            detections.append(Detection(
                label=label,
                class_id=cls_id,
                confidence=float(confs[i]),
                stage2_conf=s2_conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                size_px=int((x2 - x1) * (y2 - y1)),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            ))
        return detections

    def _sahi_infer(self, img: np.ndarray, rows: int = 2, cols: int = 1,
                    overlap: float = 0.35) -> list[Detection]:
        """
        SAHI tiled inference: slice image into overlapping tiles, detect on each,
        map coordinates back to full image, then merge with NMS.

        For 640×480 portrait-mode water camera, default is 2 rows × 1 col:
          - Top tile covers horizon (distant objects)
          - Bottom tile covers near-field (close objects)
          - 35% overlap ensures objects on the boundary aren't missed
        """
        H, W = img.shape[:2]

        tile_h = int(H / (rows - overlap * (rows - 1)))
        tile_w = int(W / (cols - overlap * (cols - 1)))
        stride_h = int(tile_h * (1 - overlap))
        stride_w = int(tile_w * (1 - overlap))

        all_dets = []
        y_positions = []
        for r in range(rows):
            y0 = r * stride_h
            y0 = min(y0, H - tile_h)
            y0 = max(0, y0)
            y_positions.append(y0)

        x_positions = []
        for c in range(cols):
            x0 = c * stride_w
            x0 = min(x0, W - tile_w)
            x0 = max(0, x0)
            x_positions.append(x0)

        for y0 in sorted(set(y_positions)):
            y1 = min(y0 + tile_h, H)
            for x0 in sorted(set(x_positions)):
                x1 = min(x0 + tile_w, W)

                tile = img[y0:y1, x0:x1]
                tile_dets = self._detect_on_image(tile)

                # Map tile coords → full-image coords
                for d in tile_dets:
                    bx1, by1, bx2, by2 = d.bbox
                    d.bbox = (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                    d.center = ((d.bbox[0] + d.bbox[2]) // 2,
                                (d.bbox[1] + d.bbox[3]) // 2)
                all_dets.extend(tile_dets)

        if len(all_dets) <= 1:
            return all_dets

        # Merge overlapping detections across tiles with NMS (per-class)
        return self._nms_merge(all_dets)

    def _nms_merge(self, detections: list[Detection]) -> list[Detection]:
        """Per-class NMS merge of detections (handles tile overlaps)."""
        if len(detections) <= 1:
            return detections

        # Group by class_id
        by_class: dict[int, list[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_id, []).append(d)

        kept = []
        for cls_id, dets in by_class.items():
            # Sort by confidence descending
            dets.sort(key=lambda d: d.confidence, reverse=True)
            while dets:
                best = dets.pop(0)
                kept.append(best)
                dets = [d for d in dets
                        if _box_iou(best.bbox, d.bbox) < self.iou_threshold]

        return kept

    def infer(self, rgb_tensor: torch.Tensor) -> list[Detection]:
        """Run inference. If self.sahi is True, uses tiled SAHI inference."""
        img = self.preprocess_image(rgb_tensor)

        if getattr(self, "sahi", False):
            rows = getattr(self, "sahi_rows", 2)
            cols = getattr(self, "sahi_cols", 1)
            overlap = getattr(self, "sahi_overlap", 0.35)
            return self._sahi_infer(img, rows=rows, cols=cols, overlap=overlap)
        else:
            return self._detect_on_image(img)

    def _classify_crops(self, crops: list[np.ndarray],
                        indices: list[int]) -> dict[int, dict[str, float]]:
        import PIL.Image
        batch = []
        for crop in crops:
            pil = PIL.Image.fromarray(crop).resize((224, 224))
            arr = np.array(pil, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
            tensor = (tensor - self.norm_mean.squeeze(0)) / self.norm_std.squeeze(0)
            batch.append(tensor)
        if not batch:
            return {}
        stacked = torch.stack(batch)
        with torch.no_grad():
            logits = self.classifier(stacked)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        results = {}
        for idx, prob in zip(indices, probs):
            results[idx] = {"duck": float(prob[0]), "swan": float(prob[2]),
                            "other": float(prob[1])}
        return results

    def annotate_frame(self, img: np.ndarray, detections: list[Detection]) -> np.ndarray:
        import cv2
        annotated = img.copy()
        TARGETS = {"duck", "swan"}
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = self.CLASS_COLORS.get(d.label, (160, 160, 160))
            thickness = 3 if d.label in TARGETS else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            s2 = ""
            if d.stage2_conf:
                s2 = f" s2={max(d.stage2_conf.values()):.2f}"
            label = f"{d.label} {d.confidence:.2f}{s2}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(annotated, (x1, y1 - th - baseline - 4),
                          (x1 + tw + 6, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - baseline - 2),
                        font, font_scale, (0, 0, 0), font_thickness)
        return annotated

    def log_detections(self, detections: list[Detection], interval_ms: int = 100):
        now = time.time() * 1000
        do_print = (now - self._last_log_time) >= interval_ms
        if not do_print:
            return
        self._last_log_time = now

        if not detections:
            print("[Vision] No objects detected")
            return

        counts = Counter(d.label for d in detections)
        parts = []
        for cls_name in ["duck", "swan", "turtle", "lilypad", "log", "rock"]:
            n = counts.get(cls_name, 0)
            if n > 0:
                parts.append(f"{n}{self.CLASS_EMOJI[cls_name]}")
        summary = " ".join(parts) if parts else "0 objects"

        details = []
        for d in detections[:8]:
            emoji = self.CLASS_EMOJI.get(d.label, "?")
            s2 = f" s2={max(d.stage2_conf.values()):.2f}" if d.stage2_conf else ""
            details.append(
                f"{emoji}{d.label}({d.center[0]},{d.center[1]})"
                f"sz={d.size_px}{s2}"
            )
        extra = f" +{len(detections)-8} more" if len(detections) > 8 else ""
        print(f"[Vision] {summary}  " + " | ".join(details) + extra)


# ═══════════════════════════════════════════════════════════════════════════════
# EnsemblePipeline — SAHI + WBF across multiple models
# ═══════════════════════════════════════════════════════════════════════════════

class EnsemblePipeline:
    """
    Runs multiple VisionPipeline instances and fuses their detections with WBF.

    Supports SAHI tiled inference per-model, then WBF merges across models.

    Usage:
        p1 = VisionPipeline(s1="phase2-4.pt", ...)
        p2 = VisionPipeline(s1="phase2-5.pt", ...)
        ensemble = EnsemblePipeline([p1, p2], sahi=True)
        detections = ensemble.infer(rgb_tensor)
    """

    def __init__(self, pipelines: list[VisionPipeline],
                 sahi: bool = True, sahi_rows: int = 2, sahi_overlap: float = 0.35,
                 wbf_iou: float = 0.55):
        self.pipelines = pipelines
        self._sahi = sahi
        self._sahi_rows = sahi_rows
        self._sahi_overlap = sahi_overlap
        self._wbf_iou = wbf_iou

        # Push SAHI config into each pipeline
        for p in self.pipelines:
            p.sahi = sahi
            p.sahi_rows = sahi_rows
            p.sahi_overlap = sahi_overlap

        self._last_log_time = 0.0
        names = ", ".join(p.name for p in pipelines)
        mode = f"SAHI-{sahi_rows}r" if sahi else "full-frame"
        print(f"[Ensemble] {len(pipelines)} models [{names}]  {mode}  "
              f"WBF-iou={wbf_iou:.2f}")

    @property
    def conf_threshold(self):
        return self.pipelines[0].conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, val):
        for p in self.pipelines:
            p.conf_threshold = val

    @property
    def iou_threshold(self):
        return self.pipelines[0].iou_threshold

    @iou_threshold.setter
    def iou_threshold(self, val):
        for p in self.pipelines:
            p.iou_threshold = val

    def preprocess_image(self, rgb_tensor: torch.Tensor) -> np.ndarray:
        return self.pipelines[0].preprocess_image(rgb_tensor)

    def infer(self, rgb_tensor: torch.Tensor) -> list[Detection]:
        """Run all pipelines, fuse with WBF."""
        all_model_dets = []
        for p in self.pipelines:
            dets = p.infer(rgb_tensor)
            all_model_dets.append(dets)

        return weighted_boxes_fusion(all_model_dets, iou_thr=self._wbf_iou)

    def annotate_frame(self, img: np.ndarray, detections: list[Detection]) -> np.ndarray:
        return self.pipelines[0].annotate_frame(img, detections)

    CLASS_NAMES = VisionPipeline.CLASS_NAMES
    CLASS_COLORS = VisionPipeline.CLASS_COLORS
    CLASS_EMOJI = VisionPipeline.CLASS_EMOJI

    def log_detections(self, detections: list[Detection], interval_ms: int = 100):
        self.pipelines[0]._last_log_time = getattr(self, "_last_log_time", 0.0)
        now = time.time() * 1000
        do_print = (now - self._last_log_time) >= interval_ms
        if not do_print:
            return
        self._last_log_time = now

        if not detections:
            print("[Vision] No objects detected")
            return

        counts = Counter(d.label for d in detections)
        parts = []
        for cls_name in ["duck", "swan", "turtle", "lilypad", "log", "rock"]:
            n = counts.get(cls_name, 0)
            if n > 0:
                parts.append(f"{n}{self.CLASS_EMOJI[cls_name]}")
        summary = " ".join(parts) if parts else "0 objects"

        details = []
        for d in detections[:8]:
            emoji = self.CLASS_EMOJI.get(d.label, "?")
            s2 = f" s2={max(d.stage2_conf.values()):.2f}" if d.stage2_conf else ""
            details.append(
                f"{emoji}{d.label}({d.center[0]},{d.center[1]})"
                f"sz={d.size_px}{s2}"
            )
        extra = f" +{len(detections)-8} more" if len(detections) > 8 else ""
        print(f"[Vision] {summary}  " + " | ".join(details) + extra)

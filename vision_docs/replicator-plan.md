# Replicator Training Data Generation — Plan

## Goal
Generate training images for YOLO detection of ducks, swans, and distractor objects
(log, lilypad, turtle, rock) from a gosling-robot camera viewpoint.

## Camera (Gosling Viewpoint)
- Position: (0, -6, 0.30) — behind the pond, 0.3m above water
- Look-at: (0, 6, -0.02) — center of object zone
- Resolution: 640×480, focal length: 24mm
- Static — camera does NOT move per frame (gosling is stationary)

## Object Placement Per Frame
- 1–4 objects randomly selected from {duck, swan, log, lilypad, turtle, rock}
- Zone: x ∈ [-4, 4], y ∈ [2, 10], z ∈ [-0.03, -0.01] (on water surface)
- Minimum 0.5m clearance between any two objects (XY plane)
- Random Z-axis yaw rotation: [0°, 360°)
- Each object tagged with semantic label via `rep.create.from_usd(semantics=[("class", ...)])`

## Key API Fix
- `rep.create.from_usd(usd=..., semantics=[...], position=(x,y,z), rotation=(0,0,yaw))`
  — position/rotation passed DIRECTLY to create, NOT via `with obj: rep.modify.pose()`
  (the with-block pattern doesn't work for USD reference prims)

## Rendering
- `SimulationContext` + `sim.step()` = exactly 1 render per call
- No BasicWriter attached (prevents file explosion)
- Annotators: `rep.AnnotatorRegistry.get_annotator()` → `get_data()` after step

## Output
- 60 frames total (10 per object class)
- RGB: save as npy, convert to PNG in post-processing (pure-python zlib encoder)
- Bbox: save as npy (COCO dict format)
- annotations.json: COCO format

## Distribution
```
Frame 1-10:   duck-focused (but randomized — any object can appear)
Frame 11-20:  swan-focused
Frame 21-30:  log-focused
Frame 31-40:  lilypad-focused
Frame 41-50:  turtle-focused
Frame 51-60:  rock-focused
```
Each frame still randomly selects 1-4 objects, but the "focus" class has
higher probability of appearing (50% chance per slot).

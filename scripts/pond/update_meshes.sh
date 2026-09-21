#!/usr/bin/env bash
# ============================================================================
# update_meshes.sh — Rebuild all pond bird robots from GLB source files.
#
# Usage:
#   cd /home/heinz/isaaclab_uav
#   bash scripts/pond/update_meshes.sh
#
# What it does:
#   1. Converts gosling.glb, duck.glb, swan.glb → USD
#   2. Converts MDL shaders → UsdPreviewSurface + relative texture paths
#   3. Adds articulation APIs, mass properties, collision spheres
#   4. Fixes absolute texture paths → relative
#
# Output:
#   assets/pond/glb/usd/{name}/{name}_articulation.usd   (loaded by Isaac Lab)
#   assets/pond/glb/usd/{name}/textures/                  (extracted textures)
#
# Collision spheres:  gosling=0.2m  duck=0.3m  swan=0.4m
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ISAACLAB="$REPO_ROOT/isaaclab.sh"

echo "=============================================="
echo "  Pond Birds — Mesh Rebuild Pipeline"
echo "=============================================="

# Step 1: Convert GLB → USD
echo ""
echo "[1/3] Converting GLB → USD..."
$ISAACLAB -p "$SCRIPT_DIR/step1_convert.py" --headless
echo "  Done."

# Step 2: Fix materials + add articulation APIs + collision spheres
echo ""
echo "[2/3] Fixing materials and adding physics..."
$ISAACLAB -p "$SCRIPT_DIR/step2_fix.py"
echo "  Done."

# Step 3: Fix absolute texture paths → relative
echo ""
echo "[3/3] Fixing texture paths..."
$ISAACLAB -p "$SCRIPT_DIR/fix_paths.py"
echo "  Done."

echo ""
echo "=============================================="
echo "  All birds rebuilt successfully."
echo ""
echo "  To run the simulation:"
echo "    ./isaaclab.sh -p scripts/pond/launch_pond.py"
echo "=============================================="

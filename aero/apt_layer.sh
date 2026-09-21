#!/usr/bin/env bash
# apt layer for the aero toolstack — needs root.
# Run from the Claude session prompt:   ! sudo bash /home/heinz/isaaclab_uav/aero/apt_layer.sh
# or directly:                          sudo bash /home/heinz/isaaclab_uav/aero/apt_layer.sh
# Idempotent. ParaView deliberately runs as its own transaction (~1 GB of Qt) so an
# interruption there can't leave dpkg locked in front of the important pieces.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  xfoil gmsh zlib1g-dev software-properties-common wget ca-certificates gnupg

# OpenFOAM 14 (openfoam.org Foundation line) — repo pinned to noble main.
# NOTE: v14 has NO simpleFoam etc.; solvers run via `foamRun -solver incompressibleFluid`.
wget -qO- https://dl.openfoam.org/gpg.key | gpg --dearmor --yes -o /usr/share/keyrings/openfoam.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/openfoam.gpg] http://dl.openfoam.org/ubuntu noble main" \
  > /etc/apt/sources.list.d/openfoam.list
apt-get update
apt-cache policy openfoam14 openfoam13 || true
apt-get install -y --no-install-recommends openfoam14 \
  || { echo "openfoam14 unavailable — falling back to openfoam13"; apt-get install -y --no-install-recommends openfoam13; }

# Optional viz (skip-friendly)
apt-get install -y --no-install-recommends paraview xvfb || echo "paraview/xvfb failed (non-fatal, safe to skip)"

echo "== APT_LAYER_DONE =="
ls /opt/openfoam*/platforms/*/bin/ 2>/dev/null | grep -E 'foamRun|snappyHexMesh|decomposePar' || true
command -v xfoil gmsh || true

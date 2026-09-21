#!/usr/bin/env bash
# Build /home/heinz/env_aero — isolated CPU-only Python stack for the aero/RL loop.
# Safe to re-run; every install targets the venv interpreter ABSOLUTELY (PATH may
# point at env_isaaclab, which must never be mutated).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""

UV=/home/heinz/.local/bin/uv
VENV=/home/heinz/env_aero
PY=$VENV/bin/python

echo "== uv venv =="
"$UV" venv --python /usr/bin/python3.12 --seed "$VENV"
"$UV" pip install --python "$PY" -U pip setuptools wheel

# vtk (pulled by pyvista and PyGeM) is the only realistic path to a numpy
# downgrade — pin the numerical core.
printf 'numpy>=2.3,<3\nscipy>=1.14\n' > "$VENV/constraints.txt"
P=("$UV" pip install --python "$PY" -c "$VENV/constraints.txt")

echo "== numerical core =="
"${P[@]}" "numpy>=2.3,<3" scipy matplotlib

echo "== torch CPU-only wheel =="
"${P[@]}" --index-url https://download.pytorch.org/whl/cpu torch

echo "== RL =="
"${P[@]}" "gymnasium>=1.2" "stable-baselines3>=2.8" cloudpickle

echo "== mesh / viz / build tools =="
"${P[@]}" meshio pyvista gmsh
"${P[@]}" meson ninja

echo "== xfoil python binding (DARcorporation; the PyPI sdist is BROKEN — 12 kB tarball missing CMakeLists.txt) =="
XF_SRC=/home/heinz/src/xfoil-python
[ -d "$XF_SRC" ] || git clone -q --depth 1 https://github.com/DARcorporation/xfoil-python.git "$XF_SRC"
# Drop -fbounds-check/-ffpe-trap=invalid,zero: XFoil's legacy core writes the Kutta
# row at AIJ(N+1, ...) and hits benign FP paths; with those flags gfortran aborts at
# m_xpanel.f90:1150 on every eval. (Local patch, 2026-09-15.)
sed -i '/-fbounds-check/d; /-ffpe-trap=invalid,zero/d' "$XF_SRC/CMakeLists.txt"
"${P[@]}" "$XF_SRC" || "$VENV/bin/pip" install -c "$VENV/constraints.txt" "$XF_SRC"

echo "== PyGeM from mathLab git (PyPI 'pygem' is a GLACIER model — never install that) =="
"${P[@]}" "git+https://github.com/mathLab/PyGeM" || {
  echo "uv failed PyGeM — falling back to seeded pip"
  "$VENV/bin/pip" install -c "$VENV/constraints.txt" "git+https://github.com/mathLab/PyGeM"
}

echo "== neuralfoil surrogate (--no-deps, then its real dep: it imports aerosandbox at module level) =="
"${P[@]}" --no-deps neuralfoil
"${P[@]}" aerosandbox

echo "== installed =="
"$UV" pip list --python "$PY" | grep -Ei 'torch|gymnasium|stable|numpy|scipy|meshio|pyvista|gmsh|meson|ninja|xfoil|pygem|neuralfoil' || true
echo "SETUP_ENV_AERO_DONE"

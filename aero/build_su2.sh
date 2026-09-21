#!/usr/bin/env bash
# Build SU2 v8.5.0 from source (release binaries segfault on Ubuntu 24.04 and are serial-only).
# GPU-neutral, niced, -j8: this box also hosts the live ur5 training run.
set -euo pipefail
cd /home/heinz/src/SU2-8.5.0

# Default gcc is 11 via update-alternatives — pin 13 WITHOUT touching alternatives.
export CC=gcc-13 CXX=g++-13 OMPI_CC=gcc-13 OMPI_CXX=g++-13
export PATH="/home/heinz/env_aero/bin:$PATH"   # meson + ninja (no sudo)

/home/heinz/env_aero/bin/python meson.py setup build \
  -Dwith-mpi=enabled -Dwith-omp=false \
  -Denable-autodiff=false -Denable-directdiff=false -Denable-pywrapper=false \
  --prefix="$HOME/SU2-v8.5.0" --buildtype=release

nice -n 10 ninja -C build -j8 install

echo "== SU2 install tree =="
ls "$HOME/SU2-v8.5.0/bin" | head -20
echo "SU2_BUILD_DONE"

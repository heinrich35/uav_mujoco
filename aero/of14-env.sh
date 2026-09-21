#!/usr/bin/env bash
# OpenFOAM 14 env — source ONLY in dedicated shells/wrappers:
#   . /home/heinz/isaaclab_uav/aero/of14-env.sh
# NEVER source from ~/.bashrc and never together with env.sh: this bashrc is
# non-idempotent, rewrites PS1, adds aliases, and clobbers LD_LIBRARY_PATH.
_of_dir=$(ls -d /opt/openfoam1[34] 2>/dev/null | head -1)
if [ -z "$_of_dir" ]; then
  echo "of14-env.sh: no /opt/openfoam1[34] found — run aero/apt_layer.sh first" >&2
  return 1 2>/dev/null || exit 1
fi
. "$_of_dir/etc/bashrc"
unset _of_dir
# OpenFOAM 14 note: no simpleFoam/pimpleFoam binaries — solvers are selected at
# runtime:  foamRun -solver incompressibleFluid   (see docs/aero_stack.md)

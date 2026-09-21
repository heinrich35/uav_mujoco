#!/usr/bin/env bash
# Aero/RL environment hook — source explicitly (venvs have no activate.d):
#   . /home/heinz/isaaclab_uav/aero/env.sh
# HARD RULE: GPU is off-limits (occupied by the live training run).
# CUDA_VISIBLE_DEVICES must be set BEFORE the first `import torch` — that is why
# it lives here and not in the training scripts.
export CUDA_VISIBLE_DEVICES=""          # "" (not "-1") = no CUDA devices at all
export OMP_NUM_THREADS=1                # RL fan-out spawns N procs; each must stay single-threaded
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Isolated venv (env_isaaclab is the live training env — never pip-install into it)
source /home/heinz/env_aero/bin/activate

# SU2 (source build, v8.5.0)
export SU2_HOME="$HOME/src/SU2-8.5.0"
export SU2_RUN="$HOME/SU2-v8.5.0/bin"
if [ -d "$SU2_RUN" ]; then
  export PATH="$SU2_RUN:$PATH"
  export PYTHONPATH="$SU2_RUN${PYTHONPATH:+:$PYTHONPATH}"
fi

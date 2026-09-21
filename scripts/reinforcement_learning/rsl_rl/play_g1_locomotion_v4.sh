#!/usr/bin/env bash
# Wrapper: run PLAY_G1_LOCOMOTION_V4.sh from any cwd (repo root is detected).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${ROOT}/PLAY_G1_LOCOMOTION_V4.sh"

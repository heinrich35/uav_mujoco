#!/usr/bin/env bash
# ==============================================================================
# launch_uav.sh — Launch the MuJoCo fixed-wing UAV simulation
# ==============================================================================
#
# Usage:
#   ./launch_uav.sh [OPTIONS]
#
# All unrecognised options are forwarded to uav_sim.py.
# Run with --help to see this message, or -- --help for Python help.
#
# Presets (applied before explicit overrides):
#   ./launch_uav.sh --preset default      # default values
#   ./launch_uav.sh --preset high         # high-altitude start (z=50 m)
#   ./launch_uav.sh --preset glide        # shallow pitch, forward speed
#   ./launch_uav.sh --preset mars         # Mars gravity, thin atmosphere
#   ./launch_uav.sh --preset heavy        # heavy drone, strong forces
#   ./launch_uav.sh --preset agile        # low drag, high lift
#
# Examples:
#   ./launch_uav.sh
#   ./launch_uav.sh --pos-z 5.0 --pitch 15
#   ./launch_uav.sh --preset mars
#   ./launch_uav.sh --preset high --force-up 15000
#   ./launch_uav.sh --dry-run --preset glide
#   ./launch_uav.sh -- --help
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/scripts/mujoco/uav_sim.py"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD='\033[1m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    BOLD='' RED='' GREEN='' YELLOW='' CYAN='' NC=''
fi

# ---------------------------------------------------------------------------
# Presets  (single-line strings parsed by `read -ra`)
# ---------------------------------------------------------------------------
declare -A PRESETS

PRESETS["default"]=""

PRESETS["high"]="\
--pos-z 50.0 --pitch 5.0"

PRESETS["glide"]="\
--pos-z 5.0 --pitch 3.0 \
--force-up 5000 --force-nose 1000 \
--drag 3 20 50 --lift 50"

PRESETS["mars"]="\
--pos-z 10.0 --pitch 10.0 \
--gravity 3.71 \
--drag 0.5 3 6 --lift 4 \
--rot-damp 30 30 15 \
--force-up 3000 --force-nose 800"

PRESETS["heavy"]="\
--pos-z 2.0 --pitch 15.0 \
--force-up 20000 --force-nose 5000 \
--drag 10 60 120 --lift 80 \
--rot-damp 500 500 250"

PRESETS["agile"]="\
--pos-z 3.0 --pitch 8.0 \
--drag 2 10 20 --lift 80 \
--rot-damp 100 100 50 \
--force-up 10000 --force-nose 3000"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    echo -e "${CYAN}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     MuJoCo Fixed-Wing UAV Aerodynamics Simulation       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Usage:  $(basename "$0") [OPTIONS] [-- Python args ...]"
    echo ""
    echo "Launch Options (processed by this script):"
    echo "  --preset NAME     Apply a named preset before other args"
    echo "  --python PATH     Python interpreter (default: python3)"
    echo "  --dry-run         Print command without executing"
    echo "  --help, -h        Show this help"
    echo ""
    echo "Presets:"
    echo "  default     Default settings (z=2 m, pitch=12.8°, Earth g)"
    echo "  high        High-altitude start (z=50 m, pitch=5°)"
    echo "  glide       Shallow glide (z=5 m, pitch=3°, more lift)"
    echo "  mars        Mars gravity (3.71 m/s²), thin atmosphere"
    echo "  heavy       Heavy UAV (stronger forces, more drag)"
    echo "  agile       Low drag, high lift — very responsive"
    echo ""
    echo "Python Options:  $0 -- --help  for the full list."
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --pos-z 5.0 --pitch 15"
    echo "  $0 --preset mars"
    echo "  $0 --preset high --force-up 15000 --pos-z 60"
    echo "  $0 --dry-run --preset glide"
    exit 0
}

# ---------------------------------------------------------------------------
# Parse script-level arguments
# ---------------------------------------------------------------------------
PYTHON_BIN="python3"
PRESET=""
DRY_RUN=false
PASS_THROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)  usage ;;
        --preset)   PRESET="$2";       shift 2 ;;
        --python)   PYTHON_BIN="$2";   shift 2 ;;
        --dry-run)  DRY_RUN=true;      shift ;;
        --)         shift; PASS_THROUGH+=("$@"); break ;;
        *)          PASS_THROUGH+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo -e "${RED}Error: Python script not found: ${PYTHON_SCRIPT}${NC}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    echo -e "${RED}Error: Python interpreter not found: ${PYTHON_BIN}${NC}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build final argument list (preset args first, then user overrides)
# ---------------------------------------------------------------------------
FINAL_ARGS=()

if [[ -n "${PRESET}" ]]; then
    if [[ -z "${PRESETS[${PRESET}]+isset}" ]]; then
        echo -e "${RED}Error: unknown preset '${PRESET}'${NC}" >&2
        echo -e "Available: ${!PRESETS[*]}" >&2
        exit 1
    fi
    read -ra PRESET_ARGS <<< "${PRESETS[${PRESET}]}"
    FINAL_ARGS+=("${PRESET_ARGS[@]}")
fi

FINAL_ARGS+=("${PASS_THROUGH[@]}")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
echo -e "${CYAN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     MuJoCo Fixed-Wing UAV Aerodynamics Simulation       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

if [[ -n "${PRESET}" ]]; then
    echo -e "${YELLOW}Preset :${NC} ${BOLD}${PRESET}${NC}"
fi

# Pretty-print command: group flag+values on same line
_display_args() {
    local -a args=("$@")
    local i=0 n=${#args[@]} line="" char_count=0 max_width=70

    printf "  ${GREEN}Command:${NC} ${PYTHON_BIN} ${PYTHON_SCRIPT}"

    if [[ $n -eq 0 ]]; then
        echo ""; return
    fi

    printf " \\\\\n"
    while [[ $i -lt $n ]]; do
        local flag="${args[i]}"
        local chunk="${flag}"
        i=$((i + 1))

        # Grab value arguments that follow a flag (not starting with --)
        while [[ $i -lt $n && "${args[i]}" != --* ]]; do
            chunk+=" ${args[i]}"
            i=$((i + 1))
        done

        if [[ $i -lt $n ]]; then
            printf "    %s \\\\\n" "${chunk}"
        else
            printf "    %s\n" "${chunk}"
        fi
    done
}

_display_args "${FINAL_ARGS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Execute or dry-run
# ---------------------------------------------------------------------------
if ${DRY_RUN}; then
    echo -e "${YELLOW}[dry-run]${NC} Not executing."
    exit 0
fi

exec "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "${FINAL_ARGS[@]}"

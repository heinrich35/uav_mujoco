#!/usr/bin/env bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Exits if error occurs
set -e

# get source directory
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# Helper functions
#==

# extract the python executable
extract_python_exe() {
    if [[ -n "${CONDA_PREFIX}" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
    elif [[ -n "${VIRTUAL_ENV}" ]]; then
        echo "${VIRTUAL_ENV}/bin/python"
    else
        echo "${ISAACLAB_PATH}/_isaac_sim/python.sh"
    fi
}

# Prepend Isaac Lab extension roots to PYTHONPATH so conda/venv Python can import
# isaaclab* without a local pip editable install (bundled Isaac Sim python.sh already sets this).
# Linux: hybrid graphics (NVIDIA dGPU + AMD/Intel/iGPU) exposes multiple Vulkan ICDs.
# Isaac Sim/Omniverse RTX can crash in librtx.scenedb during Hydra/engine setup when
# Vulkan sees a non‑NVIDIA device (often early segfault/hard failure or ld.so anomalies).
# Set SKIP_ISAACLAB_VULKAN_GPU_FILTER=1 to disable. Existing VK_ICD_FILENAMES is respected.
prepend_isaaclab_linux_gpu_env() {
    case "$(uname -s)" in
        Linux) ;;
        *) return 0 ;;
    esac
    if [ "${SKIP_ISAACLAB_VULKAN_GPU_FILTER:-0}" = "1" ]; then
        return 0
    fi
    if [ -n "${VK_ICD_FILENAMES:-}" ]; then
        return 0
    fi
    local icd=""
    for icd in \
        /usr/share/vulkan/icd.d/nvidia_icd.json \
        /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
        /etc/vulkan/icd.d/nvidia_icd.json \
        /etc/vulkan/icd.d/nvidia_icd.x86_64.json \
    ; do
        if [ -f "${icd}" ]; then
            export VK_ICD_FILENAMES="${icd}"
            export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
            return 0
        fi
    done
}

# Drop one directory entry from LD_LIBRARY_PATH (colon-separated). Used so Conda/Venv lib/
# cannot shadow distro glibc resolution for Omniverse dlopen()'d plugins.
_isaaclab_ld_path_drop_dir() {
    local path="$1"
    local drop="$2"
    [ -z "$path" ] && return 0
    local -a parts
    local out=""
    local p
    IFS=: read -r -a parts <<< "${path}" || true
    for p in "${parts[@]}"; do
        [ -z "$p" ] && continue
        [ "$p" = "$drop" ] && continue
        if [ -z "$out" ]; then
            out="$p"
        else
            out="${out}:${p}"
        fi
    done
    printf '%s' "${out}"
}

# Remove any path entry whose string ends with suffix (e.g. .../site-packages/torch/lib).
_isaaclab_ld_path_drop_path_suffix() {
    local path="$1"
    local suffix="$2"
    [ -z "$path" ] && return 0
    local -a parts
    local out=""
    local p
    IFS=: read -r -a parts <<< "${path}" || true
    for p in "${parts[@]}"; do
        [ -z "$p" ] && continue
        case "$p" in
            *"${suffix}") continue ;;
        esac
        if [ -z "$out" ]; then
            out="$p"
        else
            out="${out}:${p}"
        fi
    done
    printf '%s' "${out}"
}

# Pip/Conda installs: Omniverse kit .so must resolve libc symbols (e.g. signal@@GLIBC_*) consistently.
# If CONDA_PREFIX/lib or VIRTUAL_ENV/lib is ahead in LD_LIBRARY_PATH, libc-style resolution can break for
# kit plugins (bogus "version …" text after crashing in librtx.scenedb is often follow-on corruption).
# We optionally source NVIDIA's setup script (wheel). If it exists, we do NOT prepend kit again
# (duplicate kit entries can trigger glibc _dl_find_dso_for_object during Hydra/plugin teardown).
# By default we also leave LD_LIBRARY_PATH exactly as setup_python_env.sh set it — pruning
# conda/venv/torch afterward has caused ld.so namespace assertions for some installs.
# ISAACLAB_PRUNE_AFTER_SETUP=1 re-enables conda/venv/torch pruning after sourcing setup (legacy).
# With no setup script, we prepend kit + distro libs and prune the tail.
# ISAACLAB_PRUNE_ENV_LD_LIB=0 keeps conda/venv lib in LD_LIBRARY_PATH when pruning applies.
# ISAACLAB_PRUNE_TORCH_LD_LIB=0 keeps .../site-packages/torch/lib in LD_LIBRARY_PATH.
# ISAACLAB_UNSET_LD_PRELOAD=1 clears LD_PRELOAD before Python (can fix OpenMP / plugin conflicts).
prepend_isaaclab_linux_pip_isaacsim_env() {
    local py_exe="$1"
    case "$(uname -s)" in
        Linux) ;;
        *) return 0 ;;
    esac
    if [ "${SKIP_ISAACLAB_PIP_ISAACSIM_LD:-0}" = "1" ]; then
        return 0
    fi
    if [ "${ISAACLAB_UNSET_LD_PRELOAD:-0}" = "1" ]; then
        unset LD_PRELOAD
    fi
    if [ -z "${py_exe}" ] || [ ! -x "${py_exe}" ]; then
        return 0
    fi
    if ! "${py_exe}" -c "import isaacsim" 2>/dev/null; then
        return 0
    fi

    local isaac_pkg_dir
    isaac_pkg_dir=$("${py_exe}" -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))") || return 0

    local _isaac_setup_sourced="0"
    local setup_sh=""
    for setup_sh in \
        "${isaac_pkg_dir}/setup_python_env.sh" \
        "${isaac_pkg_dir}/setup_conda_env.sh" \
        "${isaac_pkg_dir}/bin/setup_python_env.sh" \
        "${isaac_pkg_dir}/../setup_python_env.sh"; do
        if [ -f "${setup_sh}" ]; then
            # shellcheck disable=SC1090
            . "${setup_sh}"
            _isaac_setup_sourced="1"
            break
        fi
    done

    if [ "${_isaac_setup_sourced}" = "1" ] && [ "${ISAACLAB_PRUNE_AFTER_SETUP:-0}" != "1" ]; then
        return 0
    fi

    local _tail="${LD_LIBRARY_PATH:-}"
    if [ "${ISAACLAB_PRUNE_ENV_LD_LIB:-1}" = "1" ]; then
        if [ -n "${CONDA_PREFIX:-}" ]; then
            _tail=$(_isaaclab_ld_path_drop_dir "${_tail}" "${CONDA_PREFIX}/lib")
        fi
        if [ -n "${VIRTUAL_ENV:-}" ]; then
            _tail=$(_isaaclab_ld_path_drop_dir "${_tail}" "${VIRTUAL_ENV}/lib")
        fi
    fi
    if [ "${ISAACLAB_PRUNE_TORCH_LD_LIB:-1}" = "1" ]; then
        _tail=$(_isaaclab_ld_path_drop_path_suffix "${_tail}" "/site-packages/torch/lib")
    fi

    # setup_python_env.sh already prepends kit paths; prepending extra+syslib again duplicates .so load
    # paths and can trigger glibc: _dl_find_dso_for_object assertion (ns == l->l_ns) during dlopen/teardown.
    if [ "${_isaac_setup_sourced}" = "1" ]; then
        export LD_LIBRARY_PATH="${_tail}"
        return 0
    fi

    local isaac_path="${ISAAC_PATH:-}"
    if [ -z "${isaac_path}" ]; then
        isaac_path=$("${py_exe}" -c "import os; import isaacsim; print(os.environ.get('ISAAC_PATH','').strip())" 2>/dev/null) || true
    fi
    if [ -z "${isaac_path}" ]; then
        isaac_path="${isaac_pkg_dir}"
    fi

    local extra=""
    local d
    for d in \
        "${isaac_path}/kit" \
        "${isaac_path}/kit/lib64" \
        "${isaac_path}/kit/libs" \
        "${isaac_path}/kit/kernel/lib" \
        "${isaac_pkg_dir}/kit" \
        "${isaac_pkg_dir}/kit/lib64" \
        "${isaac_pkg_dir}/kit/libs" \
        "${isaac_pkg_dir}/kit/kernel/lib"; do
        if [ -d "${d}" ]; then
            if [ -z "${extra}" ]; then
                extra="${d}"
            else
                extra="${extra}:${d}"
            fi
        fi
    done

    local syslib="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
    if [ -n "${extra}" ]; then
        export LD_LIBRARY_PATH="${extra}:${syslib}${_tail:+:${_tail}}"
    else
        export LD_LIBRARY_PATH="${syslib}${_tail:+:${_tail}}"
    fi
}

prepend_isaaclab_source_pythonpath() {
    if [ ! -d "${ISAACLAB_PATH}/source" ]; then
        return 0
    fi
    local _ilp=""
    local _d
    shopt -s nullglob
    for _d in "${ISAACLAB_PATH}/source"/*; do
        if [ -d "${_d}" ]; then
            if [ -z "${_ilp}" ]; then
                _ilp="${_d}"
            else
                _ilp="${_ilp}:${_d}"
            fi
        fi
    done
    shopt -u nullglob
    if [ -n "${_ilp}" ]; then
        export PYTHONPATH="${_ilp}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
}

# extract isaac sim path
extract_isaacsim_path() {
    local isaac_path=${ISAACLAB_PATH}/_isaac_sim
    if [ ! -d "${isaac_path}" ]; then
        local python_exe=$(extract_python_exe)
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi
    echo ${isaac_path}
}

# extract the simulator exe from isaacsim
extract_isaacsim_exe() {
    local isaac_path=$(extract_isaacsim_path)
    local isaacsim_exe=${isaac_path}/isaac-sim.sh
    if [ ! -f "${isaacsim_exe}" ]; then
        if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            isaacsim_exe="isaacsim isaacsim.exp.full"
        fi
    fi
    echo ${isaacsim_exe}
}

# update the vscode settings
update_vscode_settings() {
    python_exe=$(extract_python_exe)
    setup_vscode_script="${ISAACLAB_PATH}/.vscode/tools/setup_vscode.py"
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    fi
}

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-p] [-s] [-t] [-o] [-v] [-d] [-n] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active)."
    echo -e "\t-s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
    echo -e "\t-t, --test           Run all python pytest tests."
    echo -e "\t-o, --docker         Run the docker container helper script (docker/container.sh)."
    echo -e "\t-v, --vscode         Generate the VSCode settings file from template."
    echo -e "\t-d, --docs           Build the documentation from source using sphinx."
    echo -e "\t-n, --new            Create a new external project or internal task from template."
    echo -e "\n" >&2
}

#==
# Main
#==

if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--python)
            python_exe=$(extract_python_exe)
            shift
            prepend_isaaclab_linux_gpu_env
            prepend_isaaclab_linux_pip_isaacsim_env "${python_exe}"
            prepend_isaaclab_source_pythonpath
            ${python_exe} "$@"
            break
            ;;
        -s|--sim)
            isaacsim_exe=$(extract_isaacsim_exe)
            shift
            prepend_isaaclab_linux_gpu_env
            prepend_isaaclab_linux_pip_isaacsim_env "$(extract_python_exe)"
            ${isaacsim_exe} --ext-folder ${ISAACLAB_PATH}/source $@
            break
            ;;
        -n|--new)
            python_exe=$(extract_python_exe)
            shift
            prepend_isaaclab_linux_gpu_env
            prepend_isaaclab_linux_pip_isaacsim_env "${python_exe}"
            prepend_isaaclab_source_pythonpath
            ${python_exe} ${ISAACLAB_PATH}/tools/template/cli.py $@
            break
            ;;
        -t|--test)
            python_exe=$(extract_python_exe)
            shift
            prepend_isaaclab_linux_gpu_env
            prepend_isaaclab_linux_pip_isaacsim_env "${python_exe}"
            prepend_isaaclab_source_pythonpath
            ${python_exe} -m pytest ${ISAACLAB_PATH}/tools $@
            break
            ;;
        -o|--docker)
            docker_script=${ISAACLAB_PATH}/docker/container.sh
            shift
            bash ${docker_script} $@
            break
            ;;
        -v|--vscode)
            update_vscode_settings
            shift
            break
            ;;
        -d|--docs)
            python_exe=$(extract_python_exe)
            cd ${ISAACLAB_PATH}/docs
            ${python_exe} -m pip install -q -r requirements.txt
            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/current
            echo -e "[INFO] To open documentation on default browser, run:"
            echo -e "\n\t\txdg-open $(pwd)/_build/current/index.html\n"
            cd - > /dev/null
            shift
            break
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done

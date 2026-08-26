#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf '%s\n' 'Usage: 01-Install-And-Validate.sh [--skip-engine] [--skip-api-key-check]'
}

skip_engine=false
skip_api_key_check=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-engine) skip_engine=true ;;
    --skip-api-key-check) skip_api_key_check=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
config="$project_root/configs/rtx3090ti-ubuntu-production.yaml"
runtime_base=${VISIONCORTEX_RUNTIME_BASE:-'/srv/sentinel-data/VisionCortex3090Ti'}
venv=${VISIONCORTEX_VENV:-"$runtime_base/.venv"}
python_bin=${VISIONCORTEX_PYTHON:-}
engine_root=${VISIONCORTEX_ENGINE_ROOT:-"$runtime_base/Engines"}
runtime_root="$runtime_base/Runtime"
runtime_tmp=${VISIONCORTEX_TMPDIR:-"$runtime_root/tmp"}

if [[ -z $python_bin ]]; then
  for candidate in python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin=$(command -v "$candidate")
      break
    fi
  done
fi
if [[ -z $python_bin && -x /home/x1/anaconda3/envs/gaoqing/bin/python ]]; then
  # Reuse the host's existing Python 3.12 interpreter as the venv seed. Do not
  # create a second Conda environment for this deployment.
  python_bin=/home/x1/anaconda3/envs/gaoqing/bin/python
fi
if [[ -z $python_bin || ! -x $python_bin ]]; then
  printf '%s\n' 'Python 3.11/3.12 is required. Set VISIONCORTEX_PYTHON to an executable.' >&2
  exit 1
fi
export VISIONCORTEX_PYTHON=$python_bin
"$script_dir/00-Preflight.sh" --install

first_weight="$runtime_base/Models/ClosedSetYOLO/first_person/best.pt"
third_weight="$runtime_base/Models/ClosedSetYOLO/third_person/best.pt"
expected_first='a541c59ef8b09158b9b22851dcada6231dbab0f2f1478ae824bcf609851c58ea'
expected_third='ef5a867abf21a8d790eaba054e92d114ae4567c1f867c041ed079cde0a01a36b'
[[ -f $first_weight ]] || { printf 'First-person source model is missing: %s\n' "$first_weight" >&2; exit 1; }
[[ -f $third_weight ]] || { printf 'Third-person source model is missing: %s\n' "$third_weight" >&2; exit 1; }
actual_first=$(sha256sum -- "$first_weight" | awk '{print $1}')
actual_third=$(sha256sum -- "$third_weight" | awk '{print $1}')
[[ $actual_first == "$expected_first" ]] || { printf '%s\n' 'First-person model checksum mismatch.' >&2; exit 1; }
[[ $actual_third == "$expected_third" ]] || { printf '%s\n' 'Third-person model checksum mismatch.' >&2; exit 1; }

mkdir -p -- "$runtime_root/Input-Manifests" "$runtime_tmp" "$engine_root"
export TMPDIR="$runtime_tmp"
if [[ ! -x $venv/bin/python ]]; then
  "$python_bin" -m venv --copies "$venv"
fi
venv_python="$venv/bin/python"
# Ultralytics may launch `pip` as a child process while exporting. Force every
# child command to remain inside this deployment environment rather than the
# user's base Conda environment.
export PATH="$venv/bin:$PATH"
"$venv_python" -m pip install --upgrade pip setuptools wheel
"$venv_python" -m pip install -r "$script_dir/requirements-lock.txt"
sam2_revision='2b90b9f5ceec907a1c18123530e92e794ad901a4'
# The optional connected-components CUDA extension is not required for model
# output and the host nvcc minor version may differ from the pinned Torch CUDA
# runtime. Avoid a non-reproducible local compile and reuse the already-installed
# pinned Torch wheel during SAM2's build metadata step.
SAM2_BUILD_CUDA=0 "$venv_python" -m pip install \
  --no-build-isolation --no-deps \
  "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@$sam2_revision"
"$venv_python" -m pip install --no-deps -e "$project_root"

export LABVISION_CONFIG=$config
export VISIONCORTEX_TENSORRT=required
export VISIONCORTEX_ULTRALYTICS_CONFIG_DIR="$runtime_root/ThirdParty"
export VISIONCORTEX_NAS_INDEX_CSV=${VISIONCORTEX_NAS_INDEX_CSV:-'/home/x1/桌面/nas/experiment_record_index.csv'}
export VISIONCORTEX_NAS_ARCHIVE_ROOT=${VISIONCORTEX_NAS_ARCHIVE_ROOT:-'/home/x1/桌面/nas/VisionCortexExperimentArchive'}
export VISIONCORTEX_NAS_CACHE_ROOT=${VISIONCORTEX_NAS_CACHE_ROOT:-'/home/x1/桌面/nas/VisionCortexExperimentCache'}
export VISIONCORTEX_LOCAL_INPUT_ROOT="$runtime_root/Input-Manifests"
export VISIONCORTEX_LOCAL_RUNTIME_ROOT="$runtime_root"
export VISIONCORTEX_LOCAL_CACHE_ROOT=${VISIONCORTEX_LOCAL_CACHE_ROOT:-"$VISIONCORTEX_NAS_CACHE_ROOT"}
export VISIONCORTEX_LOCAL_STAGING_ROOT=${VISIONCORTEX_LOCAL_STAGING_ROOT:-"$VISIONCORTEX_NAS_ARCHIVE_ROOT/.VisionCortex-Run-Staging"}
export VISIONCORTEX_OUTPUT_ROOT=${VISIONCORTEX_OUTPUT_ROOT:-"$VISIONCORTEX_NAS_ARCHIVE_ROOT/.VisionCortex-Run-Staging"}
export VISIONCORTEX_FIRST_PERSON_ENGINE="$engine_root/first_person.engine"
export VISIONCORTEX_THIRD_PERSON_ENGINE="$engine_root/third_person.engine"
# TensorRT export and materialized clips use the local TMPDIR exported before
# dependency installation; they never use the NAS cache as transient scratch.

cd -- "$project_root"
"$venv_python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'
name = torch.cuda.get_device_name(0)
assert 'RTX 3090 Ti' in name, f'Expected RTX 3090 Ti; detected {name}'
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (8, 6), (major, minor)
print(f'GPU={name}')
print(f'PyTorch={torch.__version__}')
print(f'PyTorch_CUDA={torch.version.cuda}')
print(f'compute_capability={major}.{minor}')
PY
"$venv_python" -c "import tensorrt as trt; print(f'TensorRT={trt.__version__}')"
"$venv_python" -m labvision_evidence prepare-public-models --config "$config"

if [[ $skip_engine == false ]]; then
  "$venv_python" -m labvision_evidence prepare-engine --config "$config"
fi
"$venv_python" -m labvision_evidence validate-models --config "$config"

if [[ $skip_api_key_check == false && -z ${ARK_API_KEY:-} ]]; then
  printf '%s\n' 'ARK_API_KEY is not present. Export it securely before starting Web or a production run.' >&2
  exit 1
fi

printf '%s\n' 'VisionCortex Ubuntu RTX 3090 Ti installation and validation completed.'
printf 'venv=%s\nconfig=%s\nruntime=%s\ncache=%s\nengines=%s\n' \
  "$venv" "$config" "$runtime_root" "$VISIONCORTEX_LOCAL_CACHE_ROOT" "$engine_root"

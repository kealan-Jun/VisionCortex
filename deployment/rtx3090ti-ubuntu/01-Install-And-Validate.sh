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
source "$script_dir/_common.sh"
deployment_config production
python_bin=$python
venv=${VISIONCORTEX_VENV:-"$project_root/.venv"}
runtime_tmp=${VISIONCORTEX_TMPDIR:-"$runtime_root/tmp"}
engine_root=${VISIONCORTEX_ENGINE_ROOT:-"$(dirname -- "$first_engine")"}
export VISIONCORTEX_PYTHON=$python_bin
"$script_dir/00-Preflight.sh" --install
# Weight identities are supplied by the prepared model registry, not a host path.
registry=${VISIONCORTEX_MODEL_REGISTRY:-"$project_root/configs/models/closed-set-yolo.json"}
mapfile -t checksums < <("$python" -B - "$registry" <<'HASH'
import json
from pathlib import Path
import sys
registry=json.loads(Path(sys.argv[1]).read_text())
for role in ('first_person', 'third_person'):
    digest=registry['models'][role]['sha256']
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise SystemExit('Invalid model registry checksum.')
    print(digest)
HASH
)
[[ ${#checksums[@]} == 2 ]] || { printf '%s\n' 'Prepared model registry is invalid.' >&2; exit 1; }
expected_first=${checksums[0]}
expected_third=${checksums[1]}
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

export VISIONCORTEX_CONFIG=$config
export VISIONCORTEX_TENSORRT=required
export VISIONCORTEX_ULTRALYTICS_CONFIG_DIR="$runtime_root/ThirdParty"
# Storage and engine identities come from the explicit prepared configuration.

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
"$venv_python" -m visioncortex prepare-public-models --config "$config"

if [[ $skip_engine == false ]]; then
  "$venv_python" -m visioncortex prepare-engine --config "$config"
fi
"$venv_python" -m visioncortex validate-models --config "$config"

if [[ $skip_api_key_check == false ]]; then deployment_credentials; fi

printf '%s\n' 'VisionCortex Ubuntu RTX 3090 Ti installation and validation completed.'
printf 'venv=%s\nconfig=%s\nruntime=%s\ncache=%s\nengines=%s\n' \
  "$venv" "$config" "$runtime_root" "$cache_root" "$engine_root"

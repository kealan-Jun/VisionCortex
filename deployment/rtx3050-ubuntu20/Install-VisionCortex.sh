#!/usr/bin/env bash
set -Eeuo pipefail

package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
install_root='/opt/visioncortex-rtx3050'
app_root="$install_root/app"
venv="$install_root/.venv"
runtime_root="$install_root/Runtime"
config="$app_root/configs/rtx3050-6gb-ubuntu20-production.yaml"

for required in sudo chmod cp getent mkdir mktemp rm rmdir sha256sum; do
  command -v "$required" >/dev/null 2>&1 || {
    printf 'Required installer command is missing: %s\n' "$required" >&2
    exit 1
  }
done
for required_path in \
  "$package_root/SHA256SUMS" \
  "$package_root/app" \
  "$package_root/vendor/python/bin/python3.12" \
  "$package_root/vendor/uv" \
  "$package_root/vendor/ffmpeg/bin/ffmpeg" \
  "$package_root/vendor/ffmpeg/bin/ffprobe" \
  "$package_root/vendor/wheelhouse" \
  "$package_root/models/ClosedSetYOLO/first_person/best.pt" \
  "$package_root/models/ClosedSetYOLO/third_person/best.pt"; do
  [[ -e $required_path ]] || { printf 'Incomplete USB package: %s\n' "$required_path" >&2; exit 1; }
done

bash "$package_root/Verify-Package.sh"
# USB filesystems may be mounted noexec or may not retain Unix mode bits. Copy
# the two static preflight binaries onto the local filesystem before executing
# them, then remove only this validated temporary directory.
preflight_bin=$(mktemp -d /tmp/visioncortex-preflight-bin.XXXXXX)
[[ -d $preflight_bin && ! -L $preflight_bin && $preflight_bin == /tmp/visioncortex-preflight-bin.* ]] || {
  printf 'Unsafe preflight temporary directory: %s\n' "$preflight_bin" >&2
  exit 1
}
cp -- "$package_root/vendor/ffmpeg/bin/ffmpeg" "$preflight_bin/ffmpeg"
cp -- "$package_root/vendor/ffmpeg/bin/ffprobe" "$preflight_bin/ffprobe"
chmod 0755 -- "$preflight_bin/ffmpeg" "$preflight_bin/ffprobe"
export PATH="$preflight_bin:$PATH"
VISIONCORTEX_RUNTIME_BASE="$install_root" \
  bash "$package_root/app/deployment/rtx3050-ubuntu20/00-Preflight.sh" --install
rm -- "$preflight_bin/ffmpeg" "$preflight_bin/ffprobe"
rmdir -- "$preflight_bin"

if [[ -e $install_root ]]; then
  printf 'Installation target already exists; refusing to overwrite: %s\n' "$install_root" >&2
  exit 1
fi

install_user=${SUDO_USER:-$USER}
install_group=$(id -gn "$install_user")
install_home=$(getent passwd "$install_user" | awk -F: 'NR==1 {print $6}')
[[ -n $install_home && -d $install_home ]] || {
  printf 'Unable to resolve the installation user home: %s\n' "$install_user" >&2
  exit 1
}
sudo install -d -m 0755 -o "$install_user" -g "$install_group" "$install_root"
mkdir -p -- "$app_root" "$install_root/python" "$install_root/bin" \
  "$install_root/Models/ClosedSetYOLO/first_person" \
  "$install_root/Models/ClosedSetYOLO/third_person" \
  "$install_root/Engines/grounding-dino-base" \
  "$runtime_root/Input-Manifests" "$runtime_root/Model-Quality" \
  "$runtime_root/PublicModels/LabPicsSemantic" \
  "$runtime_root/ThirdParty" "$runtime_root/tmp"

cp -a -- "$package_root/app/." "$app_root/"
cp -a -- "$package_root/vendor/python/." "$install_root/python/"
cp -a -- "$package_root/vendor/uv" "$install_root/bin/uv"
cp -a -- "$package_root/vendor/ffmpeg" "$install_root/ffmpeg"
chmod 0755 -- "$install_root/python/bin/python3.12" "$install_root/bin/uv" \
  "$install_root/ffmpeg/bin/ffmpeg" "$install_root/ffmpeg/bin/ffprobe"
cp -a -- "$package_root/models/ClosedSetYOLO/first_person/best.pt" \
  "$install_root/Models/ClosedSetYOLO/first_person/best.pt"
cp -a -- "$package_root/models/ClosedSetYOLO/third_person/best.pt" \
  "$install_root/Models/ClosedSetYOLO/third_person/best.pt"
cp -a -- "$package_root/models/Public/yolov8s-worldv2.pt" "$install_root/Engines/"
cp -a -- "$package_root/models/Public/ViT-B-32.pt" "$install_root/Engines/"
cp -a -- "$package_root/models/Public/sam2.1_hiera_base_plus.pt" "$install_root/Engines/"
cp -a -- "$package_root/models/Public/labpics-semantic-materials.torch" "$install_root/Engines/"
cp -a -- "$package_root/models/Public/grounding-dino-base/." \
  "$install_root/Engines/grounding-dino-base/"

"$install_root/bin/uv" venv --python "$install_root/python/bin/python3.12" \
  --no-python-downloads --link-mode copy "$venv"
uv_install_cache="$runtime_root/uv-install-cache"
mkdir -p -- "$uv_install_cache"
"$install_root/bin/uv" pip install --python "$venv/bin/python" \
  --offline --no-index --cache-dir "$uv_install_cache" --link-mode hardlink \
  --find-links "$package_root/vendor/wheelhouse" \
  --requirement "$app_root/deployment/rtx3050-ubuntu20/requirements-offline.txt"
"$install_root/bin/uv" pip install --python "$venv/bin/python" \
  --offline --no-index --cache-dir "$uv_install_cache" --link-mode hardlink \
  --no-build-isolation --no-deps "$app_root"
# The cache and venv share hard-linked package files on the target filesystem,
# avoiding a second full 13 GiB copy during installation. Once both installs
# finish, clear only this dedicated cache; venv links remain intact.
"$install_root/bin/uv" cache clean --cache-dir "$uv_install_cache"

export PATH="$install_root/ffmpeg/bin:$venv/bin:$PATH"
export TMPDIR="$runtime_root/tmp"
export LABVISION_CONFIG="$config"
export VISIONCORTEX_DEFAULT_CONFIG="$app_root/configs/default.yaml"
export VISIONCORTEX_RUNTIME_BASE="$install_root"
export VISIONCORTEX_TENSORRT=required
export VISIONCORTEX_FIRST_PERSON_ENGINE="$install_root/Engines/first_person.engine"
export VISIONCORTEX_THIRD_PERSON_ENGINE="$install_root/Engines/third_person.engine"
export VISIONCORTEX_ULTRALYTICS_CONFIG_DIR="$runtime_root/ThirdParty"

cd -- "$app_root"
"$venv/bin/python" - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"
name = torch.cuda.get_device_name(0)
assert "RTX 3050" in name, f"Expected RTX 3050; detected {name}"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (8, 6), (major, minor)
properties = torch.cuda.get_device_properties(0)
assert properties.total_memory >= 5_900 * 1024**2, properties.total_memory
print(f"GPU={name}")
print(f"PyTorch={torch.__version__}")
print(f"PyTorch_CUDA={torch.version.cuda}")
print(f"compute_capability={major}.{minor}")
PY
"$venv/bin/python" -c "import tensorrt as trt; print(f'TensorRT={trt.__version__}')"
"$venv/bin/python" -m visioncortex prepare-public-models --config "$config"
"$venv/bin/python" -m visioncortex prepare-engine --config "$config"

# TensorRT ships builder resources for many GPU architectures and Windows.
# Keep the RTX 3050 (SM86) and PTX rebuild resources, then remove only the
# irrelevant architecture files from this newly-created isolated environment.
trt_lib_dir=$(find "$venv/lib" -type d -name tensorrt_libs -print -quit)
pruned_bytes=0
pruned_files=0
if [[ -n $trt_lib_dir ]]; then
  trt_lib_dir=$(readlink -f -- "$trt_lib_dir")
  venv_lib_root=$(readlink -f -- "$venv/lib")
  [[ -d $trt_lib_dir && ! -L $trt_lib_dir && $trt_lib_dir == "$venv_lib_root/"* ]] || {
    printf 'Unsafe TensorRT pruning target: %s\n' "$trt_lib_dir" >&2
    exit 1
  }
  while IFS= read -r -d '' candidate; do
    case "$(basename -- "$candidate")" in
      *'_win_'*) ;;
      *'_sm86.'*|*'_ptx.'*) continue ;;
    esac
    resolved_candidate=$(readlink -f -- "$candidate")
    [[ -f $resolved_candidate && ! -L $candidate && $(dirname -- "$resolved_candidate") == "$trt_lib_dir" ]] || {
      printf 'Unsafe TensorRT builder resource: %s\n' "$candidate" >&2
      exit 1
    }
    bytes=$(stat -c '%s' -- "$candidate")
    rm -- "$resolved_candidate"
    pruned_bytes=$(( pruned_bytes + bytes ))
    pruned_files=$(( pruned_files + 1 ))
  done < <(find "$trt_lib_dir" -maxdepth 1 -type f -name 'libnvinfer_builder_resource_*' -print0)
fi

"$venv/bin/python" -m visioncortex validate-models --config "$config"
"$venv/bin/python" - <<'PY'
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynvml
import torch
from ultralytics import YOLO

from visioncortex.config import load_config
from visioncortex.detection import _engine_build_batch

root = Path("/opt/visioncortex-rtx3050")
config = load_config(root / "app/configs/rtx3050-6gb-ubuntu20-production.yaml")
image_size = int(config["performance"]["image_size"])
frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
records = []
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
for role in ("first_person", "third_person"):
    path = Path(config["models"][f"{role}_engine"])
    batch = int(_engine_build_batch(path) or 1)
    model = YOLO(str(path))
    model.predict([frame] * batch, imgsz=image_size, device=0, half=True, verbose=False)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    iterations = 8
    memory_used_mib = []
    gpu_compute_percent = []
    for _ in range(iterations):
        result = model.predict(
            [frame] * batch,
            imgsz=image_size,
            device=0,
            half=True,
            verbose=False,
        )
        if len(result) != batch:
            raise RuntimeError(f"{role} TensorRT smoke inference dropped frames")
        torch.cuda.synchronize(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_used_mib.append(memory.used / 1024**2)
        gpu_compute_percent.append(float(utilization.gpu))
    elapsed = time.perf_counter() - started
    build_receipt_path = path.with_suffix(path.suffix + ".build.json")
    build_receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    records.append(
        {
            "role": role,
            "batch": batch,
            "frames": batch * iterations,
            "elapsed_seconds": round(elapsed, 6),
            "images_per_second": round(batch * iterations / elapsed, 6),
            "gpu_memory_used_mib_peak": round(max(memory_used_mib), 3),
            "gpu_compute_percent_peak": max(gpu_compute_percent),
            "engine_build_receipt": str(build_receipt_path),
            "bounded_autotune": build_receipt.get("selected_benchmark"),
        }
    )
pynvml.nvmlShutdown()
(root / "Runtime/rtx3050-engine-smoke.json").write_text(
    json.dumps(
        {
            "schema_version": "visioncortex-rtx3050-engine-smoke/1",
            "status": "passed",
            "scope": "synthetic engine execution only; not real-video quality evidence",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

credential_dir="$install_home/.config/VisionCortex"
credential_file="$credential_dir/ark_api_key"
if [[ ! -f $credential_file && -t 0 ]]; then
  printf '%s' '请输入火山方舟 ARK API Key（不会回显，可直接回车稍后设置）: '
  IFS= read -r -s ark_key
  printf '\n'
  if [[ -n $ark_key ]]; then
    [[ $ark_key == ark-* && $ark_key != *$'\n'* ]] || {
      printf '%s\n' 'ARK API Key 格式不正确；未写入。' >&2
      exit 1
    }
    mkdir -p -- "$credential_dir"
    umask 077
    printf '%s\n' "$ark_key" > "$credential_file"
    chmod 600 -- "$credential_file"
    if [[ $(id -u) -eq 0 ]]; then
      chown "$install_user:$install_group" -- "$credential_dir" "$credential_file"
    fi
    unset ark_key
  fi
fi

cp -a -- "$app_root/deployment/rtx3050-ubuntu20/Start-VisionCortex.sh" \
  "$install_root/Start-VisionCortex.sh"
cp -a -- "$app_root/deployment/rtx3050-ubuntu20/00-Preflight.sh" \
  "$install_root/Preflight-VisionCortex.sh"
chmod 0755 -- "$install_root/Start-VisionCortex.sh" \
  "$install_root/Preflight-VisionCortex.sh"
package_id=$(sha256sum -- "$package_root/PACKAGE-MANIFEST.json" | awk '{print $1}')
printf '%s\n' "$package_id" > "$install_root/.package-id"
cat > "$runtime_root/install-receipt.txt" <<EOF
status=passed
package_id=$package_id
install_root=$install_root
config=$config
tensorrt_pruned_files=$pruned_files
tensorrt_pruned_bytes=$pruned_bytes
engine_smoke_receipt=$runtime_root/rtx3050-engine-smoke.json
EOF

VISIONCORTEX_RUNTIME_BASE="$install_root" \
  "$install_root/Preflight-VisionCortex.sh" --runtime
printf '%s\n' 'VisionCortex RTX 3050 offline installation completed.'
printf 'Start: %s\n' "$install_root/Start-VisionCortex.sh"
printf 'Engine smoke receipt: %s\n' "$runtime_root/rtx3050-engine-smoke.json"

#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf '%s\n' 'Usage: 00-Preflight.sh [--install|--production]'
}

mode='install'
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --install) mode='install' ;;
    --production) mode='production' ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
fi

failures=()
warnings=()

fail() { failures+=("$1"); }
warn() { warnings+=("$1"); }
have() { command -v "$1" >/dev/null 2>&1; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
runtime_root=${VISIONCORTEX_LOCAL_RUNTIME_ROOT:-'/srv/sentinel-data/VisionCortex3090Ti/Runtime'}
nas_index=${VISIONCORTEX_NAS_INDEX_CSV:-'/home/x1/桌面/nas/experiment_record_index.csv'}
nas_archive=${VISIONCORTEX_NAS_ARCHIVE_ROOT:-'/home/x1/桌面/nas/VisionCortexExperimentArchive'}
nas_cache=${VISIONCORTEX_NAS_CACHE_ROOT:-'/home/x1/桌面/nas/VisionCortexExperimentCache'}

for required in nvidia-smi ffmpeg df awk sed grep; do
  have "$required" || fail "required_command_missing=$required"
done

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ ${ID:-} == 'ubuntu' ]] || fail "unsupported_os=${ID:-unknown}; expected=ubuntu"
else
  fail 'os_release_missing=/etc/os-release'
fi

gpu_name='unknown'
gpu_vram_mib=0
gpu_compute_cap='unknown'
driver_version='unknown'
if have nvidia-smi; then
  gpu_line=$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits -i 0 2>/dev/null | sed -n '1p' || true)
  if [[ -z $gpu_line ]]; then
    fail 'gpu_query_failed=true'
  else
    IFS=',' read -r gpu_name gpu_vram_mib gpu_compute_cap driver_version <<< "$gpu_line"
    gpu_name=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<< "$gpu_name")
    gpu_vram_mib=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<< "$gpu_vram_mib")
    gpu_compute_cap=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<< "$gpu_compute_cap")
    driver_version=$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<< "$driver_version")
    [[ $gpu_name == *'RTX 3090 Ti'* ]] || fail "gpu_mismatch=$gpu_name; expected=RTX 3090 Ti"
    [[ $gpu_vram_mib =~ ^[0-9]+$ ]] && (( gpu_vram_mib >= 24000 )) || fail "gpu_vram_mib=$gpu_vram_mib; minimum=24000"
  fi
fi

python_bin=${VISIONCORTEX_PYTHON:-}
if [[ -z $python_bin ]]; then
  for candidate in python3.12 python3.11; do
    if have "$candidate"; then
      python_bin=$(command -v "$candidate")
      break
    fi
  done
fi
python_version='missing'
if [[ -z $python_bin || ! -x $python_bin ]]; then
  fail 'compatible_python_missing=Python 3.11 or 3.12 required'
else
  python_version=$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
  if ! $python_bin -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)'; then
    fail "unsupported_python=$python_version; expected=>=3.11,<3.13"
  fi
fi

if have ffmpeg; then
  ffmpeg_hwaccels=$(ffmpeg -hide_banner -hwaccels 2>&1 || true)
  ffmpeg_encoders=$(ffmpeg -hide_banner -encoders 2>&1 || true)
  ffmpeg_filters=$(ffmpeg -hide_banner -filters 2>&1 || true)
  grep -qx 'cuda' <<< "$ffmpeg_hwaccels" || fail 'ffmpeg_cuda_hwaccel=false'
  grep -q 'h264_nvenc' <<< "$ffmpeg_encoders" || fail 'ffmpeg_h264_nvenc=false'
  grep -Eq 'scale_cuda|hwupload_cuda' <<< "$ffmpeg_filters" || fail 'ffmpeg_cuda_filters=false'
fi

disk_probe=$runtime_root
while [[ ! -e $disk_probe && $disk_probe != '/' ]]; do
  disk_probe=$(dirname -- "$disk_probe")
done
free_kib=$(df -Pk -- "$disk_probe" 2>/dev/null | awk 'NR==2 {print $4}' || true)
free_gib=0
minimum_free_gib=30
if [[ $free_kib =~ ^[0-9]+$ ]]; then
  free_gib=$(( free_kib / 1024 / 1024 ))
  (( free_gib >= minimum_free_gib )) || fail "local_free_gib=$free_gib; minimum=$minimum_free_gib; mode=$mode"
else
  fail "disk_probe_failed=$disk_probe"
fi

if [[ ! -f $nas_index ]]; then
  [[ $mode == 'production' ]] && fail "nas_index_missing=$nas_index" || warn "nas_index_missing=$nas_index"
fi
if [[ ! -d $nas_archive ]]; then
  [[ $mode == 'production' ]] && fail "nas_archive_missing=$nas_archive" || warn "nas_archive_missing=$nas_archive"
fi
if [[ ! -d $nas_cache ]]; then
  [[ $mode == 'production' ]] && fail "nas_cache_missing=$nas_cache" || warn "nas_cache_missing=$nas_cache"
elif [[ ! -w $nas_cache ]]; then
  [[ $mode == 'production' ]] && fail "nas_cache_not_writable=$nas_cache" || warn "nas_cache_not_writable=$nas_cache"
fi
nas_free_gib='unavailable'
if [[ -d $nas_archive ]]; then
  nas_free_kib=$(df -Pk -- "$nas_archive" 2>/dev/null | awk 'NR==2 {print $4}' || true)
  if [[ $nas_free_kib =~ ^[0-9]+$ ]]; then
    nas_free_gib=$(( nas_free_kib / 1024 / 1024 ))
  else
    [[ $mode == 'production' ]] && fail "nas_disk_probe_failed=$nas_archive" || warn "nas_disk_probe_failed=$nas_archive"
  fi
  [[ -w $nas_archive ]] || { [[ $mode == 'production' ]] && fail "nas_archive_not_writable=$nas_archive" || warn "nas_archive_not_writable=$nas_archive"; }
fi

printf 'mode=%s\n' "$mode"
printf 'project_root=%s\n' "$project_root"
printf 'gpu_name=%s\n' "$gpu_name"
printf 'gpu_vram_mib=%s\n' "$gpu_vram_mib"
printf 'gpu_compute_cap=%s\n' "$gpu_compute_cap"
printf 'driver_version=%s\n' "$driver_version"
printf 'python=%s\n' "${python_bin:-missing}"
printf 'python_version=%s\n' "$python_version"
printf 'local_free_gib=%s\n' "$free_gib"
printf 'local_minimum_free_gib=%s\n' "$minimum_free_gib"
printf 'nas_free_gib=%s\n' "$nas_free_gib"
printf 'nas_index=%s\n' "$nas_index"
printf 'nas_archive=%s\n' "$nas_archive"
printf 'nas_cache=%s\n' "$nas_cache"
for item in "${warnings[@]}"; do printf 'warning=%s\n' "$item"; done

if (( ${#failures[@]} )); then
  for item in "${failures[@]}"; do printf 'failure=%s\n' "$item" >&2; done
  printf 'preflight=failed\n' >&2
  exit 1
fi
printf 'preflight=passed\n'

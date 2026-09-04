#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf '%s\n' 'Usage: 00-Preflight.sh [--install|--runtime|--production]'
}

mode='install'
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --install) mode='install' ;;
    --runtime) mode='runtime' ;;
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
version_at_least() {
  [[ $(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1) == "$2" ]]
}

runtime_base=${VISIONCORTEX_RUNTIME_BASE:-'/opt/visioncortex-rtx3050'}
nas_root=${VISIONCORTEX_NAS_ROOT:-'/mnt/visioncortex-nas'}
nas_index=${VISIONCORTEX_NAS_INDEX_CSV:-"$nas_root/experiment_record_index.csv"}
nas_archive=${VISIONCORTEX_NAS_ARCHIVE_ROOT:-"$nas_root/VisionCortexExperimentArchive"}
nas_cache=${VISIONCORTEX_NAS_CACHE_ROOT:-"$nas_root/VisionCortexExperimentCache"}

for required in nvidia-smi ffmpeg ffprobe df awk sed grep sort; do
  have "$required" || fail "required_command_missing=$required"
done

os_id='unknown'
os_version='unknown'
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  os_id=${ID:-unknown}
  os_version=${VERSION_ID:-unknown}
  [[ $os_id == ubuntu ]] || fail "unsupported_os=$os_id; expected=ubuntu"
  [[ $os_version == 20.04 || $os_version == 22.04 ]] || fail "unsupported_ubuntu=$os_version; expected=20.04_or_22.04"
else
  fail 'os_release_missing=/etc/os-release'
fi

kernel_version=$(uname -r | cut -d- -f1)
version_at_least "$kernel_version" '5.15.0' || fail "kernel=$kernel_version; minimum=5.15.0"

mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
swap_total_kib=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
mem_total_gib=$(( mem_total_kib / 1024 / 1024 ))
swap_total_gib=$(( swap_total_kib / 1024 / 1024 ))
(( mem_total_kib >= 14 * 1024 * 1024 )) || fail "system_memory_gib=$mem_total_gib; minimum=14"
(( swap_total_kib >= 1536 * 1024 )) || fail "swap_gib=$swap_total_gib; minimum=1.5"

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
    [[ $gpu_name == *'RTX 3050'* ]] || fail "gpu_mismatch=$gpu_name; expected=RTX 3050"
    [[ $gpu_vram_mib =~ ^[0-9]+$ ]] && (( gpu_vram_mib >= 5900 )) || fail "gpu_vram_mib=$gpu_vram_mib; minimum=5900"
    version_at_least "$driver_version" '570.0' || fail "driver=$driver_version; minimum=570.0"
    version_at_least "$gpu_compute_cap" '8.6' || fail "compute_capability=$gpu_compute_cap; minimum=8.6"
  fi
fi

if have ffmpeg; then
  ffmpeg_hwaccels=$(ffmpeg -hide_banner -hwaccels 2>&1 || true)
  ffmpeg_encoders=$(ffmpeg -hide_banner -encoders 2>&1 || true)
  grep -qx 'cuda' <<< "$ffmpeg_hwaccels" || fail 'ffmpeg_cuda_hwaccel=false'
  grep -q 'h264_nvenc' <<< "$ffmpeg_encoders" || fail 'ffmpeg_h264_nvenc=false'
fi

disk_probe=$runtime_base
while [[ ! -e $disk_probe && $disk_probe != '/' ]]; do
  disk_probe=$(dirname -- "$disk_probe")
done
free_kib=$(df -Pk -- "$disk_probe" 2>/dev/null | awk 'NR==2 {print $4}' || true)
free_gib=0
minimum_free_gib=18
[[ $mode == install ]] || minimum_free_gib=8
if [[ $free_kib =~ ^[0-9]+$ ]]; then
  free_gib=$(( free_kib / 1024 / 1024 ))
  (( free_gib >= minimum_free_gib )) || fail "local_free_gib=$free_gib; minimum=$minimum_free_gib; mode=$mode"
else
  fail "disk_probe_failed=$disk_probe"
fi

if [[ $mode == production ]]; then
  [[ -f $nas_index ]] || fail "nas_index_missing=$nas_index"
  [[ -d $nas_archive && -w $nas_archive ]] || fail "nas_archive_unavailable=$nas_archive"
  [[ -d $nas_cache && -w $nas_cache ]] || fail "nas_cache_unavailable=$nas_cache"
else
  [[ -e $nas_root ]] || warn "nas_not_checked=$nas_root"
fi

printf 'mode=%s\n' "$mode"
printf 'os=%s\nos_version=%s\nkernel=%s\n' "$os_id" "$os_version" "$kernel_version"
printf 'gpu_name=%s\ngpu_vram_mib=%s\ngpu_compute_cap=%s\ndriver_version=%s\n' \
  "$gpu_name" "$gpu_vram_mib" "$gpu_compute_cap" "$driver_version"
printf 'system_memory_gib=%s\nswap_gib=%s\n' "$mem_total_gib" "$swap_total_gib"
printf 'runtime_base=%s\nlocal_free_gib=%s\nlocal_minimum_free_gib=%s\n' \
  "$runtime_base" "$free_gib" "$minimum_free_gib"
printf 'nas_index=%s\nnas_archive=%s\nnas_cache=%s\n' "$nas_index" "$nas_archive" "$nas_cache"
for item in "${warnings[@]}"; do printf 'warning=%s\n' "$item"; done
if (( ${#failures[@]} )); then
  for item in "${failures[@]}"; do printf 'failure=%s\n' "$item" >&2; done
  printf 'preflight=failed\n' >&2
  exit 1
fi
printf 'preflight=passed\n'

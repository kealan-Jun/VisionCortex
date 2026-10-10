#!/usr/bin/env bash
set -Eeuo pipefail
mode=${VISIONCORTEX_DEPLOYMENT_MODE:-local}
while (( $# )); do
  case "$1" in
    --local) mode=local ;;
    --production) mode=production ;;
    --port) export VISIONCORTEX_WEB_PORT=${2:?missing port}; shift ;;
    --config) export VISIONCORTEX_CONFIG=${2:?missing config}; shift ;;
    -h|--help) printf '%s\n' 'Usage: Start-VisionCortex.sh [--local|--production] [--config PATH] [--port PORT]'; exit 0 ;;
    *) printf '%s\n' 'Unknown start argument.' >&2; exit 2 ;;
  esac
  shift
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -f $script_dir/app/deployment/rtx3050-ubuntu20/_environment.sh ]]; then
  source "$script_dir/app/deployment/rtx3050-ubuntu20/_environment.sh"
else
  source "$script_dir/_environment.sh"
fi
deployment_port 8003
deployment_config "$mode"
deployment_credentials
export PATH="$install_root/ffmpeg/bin:$(dirname -- "$python"):$PATH"
export TMPDIR="$runtime_root/tmp"
export VISIONCORTEX_ULTRALYTICS_CONFIG_DIR="$runtime_root/ThirdParty"
if [[ $mode == production ]]; then
  bash "$app_root/deployment/rtx3050-ubuntu20/00-Preflight.sh" --production
else
  mkdir -p -- "$archive_root" "$input_root" "$runtime_root" "$cache_root" "$staging_root"
fi
mkdir -p -- "$TMPDIR" "$VISIONCORTEX_ULTRALYTICS_CONFIG_DIR"
cd -- "$app_root"
exec "$python" -m visioncortex serve --host 127.0.0.1 --port "$port" --config "$config"

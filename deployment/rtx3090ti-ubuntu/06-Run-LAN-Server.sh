#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
mode=${VISIONCORTEX_DEPLOYMENT_MODE:-production}
[[ $mode == local || $mode == production ]] || { printf '%s\n' 'Invalid deployment mode.' >&2; exit 2; }
deployment_port 8000
deployment_config "$mode"
host=${VISIONCORTEX_WEB_HOST:-0.0.0.0}
[[ $mode != local ]] || host=127.0.0.1
case "$host" in 127.0.0.1|0.0.0.0) ;; *) printf '%s\n' 'Unsupported VisionCortex Web host.' >&2; exit 2 ;; esac
deployment_credentials
mkdir -p -- "$runtime_root/tmp"
export TMPDIR="$runtime_root/tmp"
cd -- "$project_root"
exec "$python" -m visioncortex serve --host "$host" --port "$port" --config "$config"

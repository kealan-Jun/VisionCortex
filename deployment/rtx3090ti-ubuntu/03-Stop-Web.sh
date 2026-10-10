#!/usr/bin/env bash
set -Eeuo pipefail
mode=${VISIONCORTEX_DEPLOYMENT_MODE:-local}
while (( $# )); do
  case "$1" in
    --production) mode=production ;;
    --config) export VISIONCORTEX_CONFIG=${2:?missing config}; shift ;;
    --port) export VISIONCORTEX_WEB_PORT=${2:?missing port}; shift ;;
    -h|--help) printf '%s\n' 'Usage: 03-Stop-Web.sh [--production] [--config PATH] [--port PORT]'; exit 0 ;;
    *) printf '%s\n' 'Unknown stop argument.' >&2; exit 2 ;;
  esac
  shift
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
deployment_port 8000
deployment_config "$mode"
pid_file="$runtime_root/visioncortex-web.pid"
[[ ! -L $pid_file ]] || { printf '%s\n' 'Unsafe PID file.' >&2; exit 1; }
if [[ ! -f $pid_file ]]; then printf '%s\n' 'No recorded VisionCortex Web process.'; exit 0; fi
pid=$(tr -d '[:space:]' < "$pid_file")
[[ $pid =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'Invalid PID file.' >&2; exit 1; }
if [[ ! -r /proc/$pid/cmdline ]]; then rm -- "$pid_file"; printf '%s\n' 'The recorded process has exited.'; exit 0; fi
"$python" -B "$script_dir/render_service.py" process "$project_root" "$config" "$pid"
"$python" -B "$script_dir/render_service.py" health-stop "$port" "$archive_root" "$pid"
# Only the exact checkout/configuration-owned process is signalled. SIGTERM
# lets the runtime drain its active work; no stronger signal is sent.
kill "$pid"
for _ in $(seq 1 180); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$pid" 2>/dev/null; then printf '%s\n' 'The owned process is still draining; no stronger signal was sent.' >&2; exit 1; fi
rm -- "$pid_file"
printf 'Stopped VisionCortex Web, PID=%s\n' "$pid"

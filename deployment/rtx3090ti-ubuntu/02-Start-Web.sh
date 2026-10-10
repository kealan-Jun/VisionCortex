#!/usr/bin/env bash
set -Eeuo pipefail
mode=local
open_browser=false
while (( $# )); do
  case "$1" in
    --production) mode=production ;;
    --port) export VISIONCORTEX_WEB_PORT=${2:?missing port}; shift ;;
    --config) export VISIONCORTEX_CONFIG=${2:?missing config}; shift ;;
    --open-browser) open_browser=true ;;
    -h|--help) printf '%s\n' 'Usage: 02-Start-Web.sh [--production] [--config PATH] [--port PORT] [--open-browser]'; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
deployment_port 8000
deployment_config "$mode"
deployment_credentials
pid_file="$runtime_root/visioncortex-web.pid"
stdout_log="$runtime_root/visioncortex-web.stdout.log"
stderr_log="$runtime_root/visioncortex-web.stderr.log"
url="http://127.0.0.1:$port"
if [[ -e $pid_file || -L $pid_file ]]; then
  [[ -f $pid_file && ! -L $pid_file ]] || { printf '%s\n' 'Unsafe PID file.' >&2; exit 1; }
  recorded_pid=$(tr -d '[:space:]' < "$pid_file")
  [[ $recorded_pid =~ ^[1-9][0-9]*$ && -r /proc/$recorded_pid/cmdline ]] \
    && "$python" -B "$script_dir/render_service.py" process "$project_root" "$config" "$recorded_pid" \
    && deployment_health "$recorded_pid" || { printf '%s\n' 'Stale or unsafe PID file; refusing to reuse another process.' >&2; exit 1; }
  printf 'VisionCortex Web is already running: %s/#/home (PID %s)\n' "$url" "$recorded_pid"
  exit 0
fi
"$python" -B "$script_dir/render_service.py" port "$port"
mkdir -p -- "$runtime_root/tmp" "$input_root" "$cache_root" "$staging_root" "$archive_root"
export TMPDIR="$runtime_root/tmp"
cd -- "$project_root"
nohup "$python" -m visioncortex serve --host 127.0.0.1 --port "$port" --config "$config" \
  >"$stdout_log" 2>"$stderr_log" < /dev/null &
web_pid=$!
printf '%s\n' "$web_pid" > "$pid_file.tmp"
mv -- "$pid_file.tmp" "$pid_file"
ready=false
for _ in $(seq 1 120); do
  if deployment_health "$web_pid" >/dev/null 2>&1; then ready=true; break; fi
  kill -0 "$web_pid" 2>/dev/null || break
  sleep 0.5
done
if [[ $ready != true ]]; then
  if kill -0 "$web_pid" 2>/dev/null; then kill "$web_pid"; fi
  rm -f -- "$pid_file"
  printf '%s\n' 'VisionCortex Web did not become healthy; inspect this runtime stderr log.' >&2
  exit 1
fi
printf 'VisionCortex Web: %s/#/home\nPID=%s\n' "$url" "$web_pid"
if [[ $open_browser == true ]]; then
  command -v xdg-open >/dev/null 2>&1 || { printf '%s\n' 'xdg-open is unavailable.' >&2; exit 1; }
  xdg-open "$url/#/home" >/dev/null 2>&1 &
fi

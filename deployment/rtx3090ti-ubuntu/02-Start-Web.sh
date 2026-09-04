#!/usr/bin/env bash
set -Eeuo pipefail

port=8000
open_browser=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) port=${2:?missing port}; shift ;;
    --open-browser) open_browser=true ;;
    -h|--help)
      printf '%s\n' 'Usage: 02-Start-Web.sh [--port PORT] [--open-browser]'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[[ $port =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { printf '%s\n' 'Invalid port.' >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
config="$project_root/configs/rtx3090ti-ubuntu-production.yaml"
runtime_base=${VISIONCORTEX_RUNTIME_BASE:-'/srv/sentinel-data/VisionCortex3090Ti'}
venv=${VISIONCORTEX_VENV:-"$runtime_base/.venv"}
python="$venv/bin/python"
runtime_root="$runtime_base/Runtime"
runtime_tmp=${VISIONCORTEX_TMPDIR:-"$runtime_root/tmp"}
engine_root=${VISIONCORTEX_ENGINE_ROOT:-"$runtime_base/Engines"}
ark_key_file=${VISIONCORTEX_ARK_API_KEY_FILE:-'/home/x1/.config/VisionCortex/ark_api_key'}
pid_file="$runtime_root/visioncortex-web.pid"
stdout_log="$runtime_root/visioncortex-web.stdout.log"
stderr_log="$runtime_root/visioncortex-web.stderr.log"
url="http://127.0.0.1:$port"

[[ -x $python ]] || { printf '%s\n' 'VisionCortex environment is missing; run 01-Install-And-Validate.sh first.' >&2; exit 1; }
if [[ -z ${ARK_API_KEY:-} && -f $ark_key_file ]]; then
  [[ ! -L $ark_key_file ]] || { printf '%s\n' 'ARK API key file must not be a symbolic link.' >&2; exit 1; }
  [[ $(stat -c '%u' -- "$ark_key_file") == $(id -u) ]] || { printf '%s\n' 'ARK API key file owner is invalid.' >&2; exit 1; }
  [[ $(stat -c '%a' -- "$ark_key_file") == 600 ]] || { printf '%s\n' 'ARK API key file permissions must be 600.' >&2; exit 1; }
  IFS= read -r ARK_API_KEY < "$ark_key_file"
  export ARK_API_KEY
fi
[[ -n ${ARK_API_KEY:-} ]] || { printf '%s\n' 'ARK_API_KEY is not configured in this shell.' >&2; exit 1; }
[[ $ARK_API_KEY == ark-* ]] || { printf '%s\n' 'ARK_API_KEY has an unexpected format.' >&2; exit 1; }
mkdir -p -- "$runtime_root/Input-Manifests" "$runtime_tmp" "$engine_root"
export PATH="$venv/bin:$PATH"

export VISIONCORTEX_CONFIG=$config
export VISIONCORTEX_TENSORRT=required
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
export TMPDIR="$runtime_tmp"

health_ok() {
  curl --silent --show-error --fail --max-time 3 "$url/api/health" 2>/dev/null | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"'
}

if [[ -f $pid_file ]]; then
  recorded_pid=$(tr -d '[:space:]' < "$pid_file")
  if [[ $recorded_pid =~ ^[0-9]+$ && -r /proc/$recorded_pid/cmdline ]]; then
    command_line=$(tr '\0' ' ' < "/proc/$recorded_pid/cmdline")
    if [[ $command_line == *visioncortex*serve* ]] && health_ok; then
      printf 'VisionCortex Web is already running: %s/#/home (PID %s)\n' "$url" "$recorded_pid"
      exit 0
    fi
  fi
  printf '%s\n' "Stale or unsafe PID file detected: $pid_file" >&2
  exit 1
fi

if health_ok; then
  printf '%s\n' "Port $port already hosts a healthy VisionCortex service, but it is not owned by this runtime." >&2
  exit 1
fi

cd -- "$project_root"
nohup "$python" -m visioncortex serve --host 127.0.0.1 --port "$port" --config "$config" \
  >"$stdout_log" 2>"$stderr_log" < /dev/null &
web_pid=$!
printf '%s\n' "$web_pid" > "$pid_file.tmp"
mv -- "$pid_file.tmp" "$pid_file"

ready=false
for _ in $(seq 1 120); do
  if health_ok; then ready=true; break; fi
  if ! kill -0 "$web_pid" 2>/dev/null; then break; fi
  sleep 0.5
done
if [[ $ready != true ]]; then
  if kill -0 "$web_pid" 2>/dev/null; then kill "$web_pid"; fi
  rm -f -- "$pid_file"
  printf '%s\n' 'VisionCortex Web did not become healthy.' >&2
  tail -n 30 -- "$stderr_log" >&2 || true
  exit 1
fi

printf 'VisionCortex Web: %s/#/home\nPID=%s\n' "$url" "$web_pid"
if [[ $open_browser == true ]]; then
  command -v xdg-open >/dev/null 2>&1 || { printf '%s\n' 'xdg-open is unavailable.' >&2; exit 1; }
  xdg-open "$url/#/home" >/dev/null 2>&1 &
fi

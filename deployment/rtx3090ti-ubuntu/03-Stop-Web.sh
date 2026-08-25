#!/usr/bin/env bash
set -Eeuo pipefail

runtime_base=${VISIONCORTEX_RUNTIME_BASE:-'/srv/sentinel-data/VisionCortex3090Ti'}
pid_file="$runtime_base/Runtime/visioncortex-web.pid"

if [[ ! -f $pid_file ]]; then
  printf '%s\n' 'No recorded VisionCortex Web process.'
  exit 0
fi
pid=$(tr -d '[:space:]' < "$pid_file")
[[ $pid =~ ^[0-9]+$ ]] || { printf '%s\n' "Invalid PID file: $pid_file" >&2; exit 1; }
if [[ ! -r /proc/$pid/cmdline ]]; then
  rm -- "$pid_file"
  printf '%s\n' 'The process no longer exists; the PID file was removed.'
  exit 0
fi
command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
[[ $command_line == *labvision_evidence*serve* ]] || { printf '%s\n' "PID $pid is not a VisionCortex Web process; refusing to stop it." >&2; exit 1; }

kill "$pid"
for _ in $(seq 1 50); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$pid" 2>/dev/null; then
  printf '%s\n' "VisionCortex Web PID $pid did not stop; no stronger signal was sent." >&2
  exit 1
fi
rm -- "$pid_file"
printf 'Stopped VisionCortex Web, PID=%s\n' "$pid"

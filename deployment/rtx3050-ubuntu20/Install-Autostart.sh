#!/usr/bin/env bash
set -Eeuo pipefail
mode=${VISIONCORTEX_DEPLOYMENT_MODE:-local}
while (( $# )); do
  case "$1" in
    --local) mode=local ;;
    --production) mode=production ;;
    --config) export VISIONCORTEX_CONFIG=${2:?missing private configuration}; shift ;;
    --port) export VISIONCORTEX_WEB_PORT=${2:?missing port}; shift ;;
    -h|--help) printf '%s\n' 'Usage: Install-Autostart.sh [--local|--production --config PRIVATE_SITE] [--port PORT]'; exit 0 ;;
    *) printf '%s\n' 'Unknown service installer argument.' >&2; exit 2 ;;
  esac
  shift
done
entry_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
script_dir=$entry_dir
source "$entry_dir/_environment.sh"
deployment_service visioncortex-rtx3050.service
deployment_port 8003
deployment_config "$mode"
deployment_credentials
deployment_user_bus
export VISIONCORTEX_DEPLOYMENT_RUNNER="$app_root/deployment/rtx3050-ubuntu20/Start-VisionCortex.sh"
export VISIONCORTEX_DESKTOP_RUNNER="$app_root/deployment/rtx3050-ubuntu20/Open-VisionCortex.sh"
export VISIONCORTEX_WEB_ACCESS_MODE=local
render_root=$(mktemp -d)
trap 'rm -rf -- "$render_root"' EXIT
rendered_unit="$render_root/$service_name"
"$python" -B "$shared_dir/render_service.py" render "$project_root" "$config" "$mode" \
  "$entry_dir/visioncortex.service" "$rendered_unit" "$port"
systemd-analyze --user verify "$rendered_unit"
deployment_unit_guard "$rendered_unit"
if [[ $mode == production ]]; then
  bash "$entry_dir/00-Preflight.sh" --production
fi
mkdir -p -- "$runtime_root/tmp" "$archive_root" "$input_root" "$cache_root" "$staging_root"
deployment_install_unit "$rendered_unit"
application_root="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p -- "$application_root"
"$python" -B "$shared_dir/render_service.py" desktop "$project_root" "$config" "$service_name" "$port" \
  "$application_root/${service_name%.service}.desktop"
if [[ ${VISIONCORTEX_DESKTOP_AUTOSTART:-0} == 1 ]]; then
  autostart_root="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
  mkdir -p -- "$autostart_root"
  install -m 0644 -- "$application_root/${service_name%.service}.desktop" "$autostart_root/${service_name%.service}.desktop"
fi
for _ in $(seq 1 60); do
  main_pid=$(systemctl --user show "$service_name" --property MainPID --value 2>/dev/null || true)
  if [[ $main_pid =~ ^[1-9][0-9]*$ ]] && deployment_health "$main_pid" >/dev/null 2>&1; then
    printf 'VisionCortex is healthy: http://127.0.0.1:%s/#/home\nservice=%s\n' "$port" "$service_name"
    printf 'For startup before login, an administrator may enable lingering: sudo loginctl enable-linger %s\n' "$(id -un)"
    exit 0
  fi
  sleep 0.5
done
printf 'Startup health could not be verified; inspect: systemctl --user status %s\n' "$service_name" >&2
exit 1

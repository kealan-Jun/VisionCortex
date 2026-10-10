#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
deployment_service visioncortex-local.service
deployment_port 8002
deployment_config local
deployment_user_bus
render_root=$(mktemp -d)
trap 'rm -rf -- "$render_root"' EXIT
rendered_unit="$render_root/$service_name"
"$python" -B "$script_dir/render_service.py" render "$project_root" "$config" local \
  "$script_dir/visioncortex-local.service" "$rendered_unit" "$port"
systemd-analyze --user verify "$rendered_unit"
deployment_unit_guard "$rendered_unit"
mkdir -p -- "$runtime_root/tmp" "$input_root" "$cache_root" "$staging_root" "$archive_root"
deployment_install_unit "$rendered_unit"
application_root="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p -- "$application_root"
"$python" -B "$script_dir/render_service.py" desktop "$project_root" "$config" "$service_name" "$port" \
  "$application_root/${service_name%.service}.desktop"
# Desktop autostart is an explicit per-instance preference.
if [[ ${VISIONCORTEX_DESKTOP_AUTOSTART:-0} == 1 ]]; then
  autostart_root="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
  mkdir -p -- "$autostart_root"
  install -m 0644 -- "$application_root/${service_name%.service}.desktop" "$autostart_root/${service_name%.service}.desktop"
fi
for _ in $(seq 1 60); do
  main_pid=$(systemctl --user show "$service_name" --property MainPID --value 2>/dev/null || true)
  if [[ $main_pid =~ ^[1-9][0-9]*$ ]] && deployment_health "$main_pid" >/dev/null 2>&1; then
    printf 'VisionCortex local Web is healthy: http://127.0.0.1:%s/#/home\n' "$port"
    printf 'service=%s\n' "$service_name"
    printf 'For startup before login, an administrator may enable lingering: sudo loginctl enable-linger %s\n' "$(id -un)"
    exit 0
  fi
  sleep 0.5
done
systemctl --user --no-pager --full status "$service_name" >&2 || true
exit 1

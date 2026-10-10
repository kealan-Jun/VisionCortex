#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_environment.sh"
deployment_service visioncortex-rtx3050.service
deployment_port 8003
deployment_config "${VISIONCORTEX_DEPLOYMENT_MODE:-local}"
deployment_credentials
deployment_user_bus
unit="$unit_root/$service_name"
[[ -f $unit && ! -L $unit ]] || {
  printf '%s\n' 'Install this selected VisionCortex instance service first.' >&2; exit 1;
}
"$python" -B "$shared_dir/render_service.py" owner "$project_root" "$unit"
systemctl --user start "$service_name"
for _attempt in {1..60}; do
  main_pid=$(systemctl --user show "$service_name" --property MainPID --value 2>/dev/null || true)
  if [[ $main_pid =~ ^[1-9][0-9]*$ ]] && deployment_health "$main_pid" >/dev/null 2>&1; then
    url="http://127.0.0.1:$port/#/home"
    if command -v google-chrome >/dev/null 2>&1; then
      browser_profile="${XDG_CONFIG_HOME:-$HOME/.config}/${service_name%.service}-browser"
      exec google-chrome --user-data-dir="$browser_profile" --no-first-run --no-default-browser-check --new-window "$url"
    fi
    exec xdg-open "$url"
  fi
  sleep 0.5
done
printf 'VisionCortex startup health could not be verified; inspect: systemctl --user status %s\n' "$service_name" >&2
exit 1

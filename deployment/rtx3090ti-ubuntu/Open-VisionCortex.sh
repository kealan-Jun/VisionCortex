#!/usr/bin/env bash
set -Eeuo pipefail

browser_profile="${XDG_CONFIG_HOME:-$HOME/.config}/visioncortex-browser"
browser_scale="${VISIONCORTEX_BROWSER_SCALE:-}"

if [[ -z $browser_scale ]] && command -v xrandr >/dev/null 2>&1; then
  desktop_width=$(xrandr --current 2>/dev/null \
    | sed -nE 's/.* connected primary ([0-9]+)x[0-9]+.*/\1/p' \
    | head -n 1)
  if [[ $desktop_width =~ ^[0-9]+$ ]] && (( desktop_width >= 3200 )); then
    browser_scale='1.5'
  fi
fi

if [[ -n $browser_scale && ! $browser_scale =~ ^(1([.][0-9]+)?|2([.]0+)?)$ ]]; then
  printf 'VISIONCORTEX_BROWSER_SCALE 必须是 1.0 到 2.0 之间的数字：%s\n' "$browser_scale" >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
deployment_service visioncortex-local.service
deployment_port 8002
deployment_config "${VISIONCORTEX_DEPLOYMENT_MODE:-local}"
[[ -f $unit_target && ! -L $unit_target ]] || { printf '%s\n' 'The selected VisionCortex service is not installed.' >&2; exit 1; }
"$python" -B "$script_dir/render_service.py" owner "$project_root" "$unit_target"
service=$service_name
url="http://127.0.0.1:$port/#/home"
systemctl --user start "$service"
ready=false
for _attempt in $(seq 1 60); do
  if deployment_health >/dev/null 2>&1; then ready=true; break; fi
  sleep 1
done
if [[ $ready != true ]]; then
  printf 'VisionCortex Web 启动未完成，请检查：systemctl --user status %s\n' "$service" >&2
  exit 1
fi

for browser in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$browser" >/dev/null 2>&1; then
    browser_args=(
      --user-data-dir="$browser_profile"
      --no-first-run
      --no-default-browser-check
      --disable-session-crashed-bubble
      --class=VisionCortex
    )
    if [[ -n $browser_scale ]]; then
      browser_args+=(--high-dpi-support=1 --force-device-scale-factor="$browser_scale")
    fi
    exec "$browser" "${browser_args[@]}" --app="$url"
  fi
done

command -v xdg-open >/dev/null 2>&1 || {
  printf '%s\n' '没有找到可用的浏览器启动命令。' >&2
  exit 1
}
exec xdg-open "$url"

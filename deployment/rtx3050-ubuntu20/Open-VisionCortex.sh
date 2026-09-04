#!/usr/bin/env bash
set -Eeuo pipefail

units=(visioncortex-analysis.service)
if systemctl --user cat visioncortex-local.service >/dev/null 2>&1; then
  units+=(visioncortex-local.service)
fi

systemctl --user reset-failed "${units[@]}" || true
systemctl --user start "${units[@]}"
for _attempt in {1..30}; do
  if curl --silent --fail --max-time 2 http://127.0.0.1:8001/api/health >/dev/null; then
    browser_profile="${XDG_CONFIG_HOME:-$HOME/.config}/visioncortex-browser"
    if command -v google-chrome >/dev/null 2>&1; then
      exec google-chrome \
        --app='http://127.0.0.1:8001/#/home' \
        --user-data-dir="$browser_profile" \
        --no-first-run \
        --no-default-browser-check
    fi
    exec xdg-open 'http://127.0.0.1:8001/#/home'
  fi
  sleep 1
done
printf '%s\n' 'VisionCortex 启动未完成，请检查：systemctl --user status visioncortex-analysis.service' >&2
exit 1

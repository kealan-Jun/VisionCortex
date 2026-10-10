#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
deployment_service visioncortex-lan.service
deployment_port 8000
deployment_config "${VISIONCORTEX_DEPLOYMENT_MODE:-production}"
systemctl --user --no-pager --full status "$service_name" || true
if [[ ${VISIONCORTEX_DEPLOYMENT_MODE:-production} == production ]]; then
  export VISIONCORTEX_WEB_ACCESS_MODE=lan
  export VISIONCORTEX_WEB_USERNAME=${VISIONCORTEX_WEB_USERNAME:-visioncortex}
  export VISIONCORTEX_WEB_PASSWORD_FILE=${VISIONCORTEX_WEB_PASSWORD_FILE:-"$credential_root/web_password"}
  [[ -f $VISIONCORTEX_WEB_PASSWORD_FILE && ! -L $VISIONCORTEX_WEB_PASSWORD_FILE && $(stat -c '%u' -- "$VISIONCORTEX_WEB_PASSWORD_FILE") == $(id -u) && $(stat -c '%a' -- "$VISIONCORTEX_WEB_PASSWORD_FILE") == 600 ]] || { printf '%s\n' 'Web password file is missing or unsafe.' >&2; exit 1; }
fi
deployment_health
printf 'health=ok\nurl=http://127.0.0.1:%s/#/home\n' "$port"

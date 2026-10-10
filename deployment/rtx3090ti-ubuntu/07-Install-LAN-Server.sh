#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
deployment_service visioncortex-lan.service
deployment_port 8000
deployment_config production
deployment_user_bus
password_file=${VISIONCORTEX_WEB_PASSWORD_FILE:-"$credential_root/web_password"}
username=${VISIONCORTEX_WEB_USERNAME:-visioncortex}
[[ $username =~ ^[A-Za-z0-9_.-]+$ ]] || { printf '%s\n' 'Invalid Web login username.' >&2; exit 2; }
render_root=$(mktemp -d)
password_tmp=''
trap 'rm -rf -- "$render_root"; [[ -z $password_tmp ]] || rm -f -- "$password_tmp"' EXIT
export VISIONCORTEX_WEB_ACCESS_MODE=lan VISIONCORTEX_WEB_USERNAME=$username VISIONCORTEX_WEB_PASSWORD_FILE=$password_file
export VISIONCORTEX_WEB_ALLOWED_NETWORKS=${VISIONCORTEX_WEB_ALLOWED_NETWORKS:-'127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,::1/128,fc00::/7'}
export VISIONCORTEX_ARK_API_KEY_FILE=$ark_key_file
rendered_unit="$render_root/$service_name"
"$python" -B "$script_dir/render_service.py" render "$project_root" "$config" production \
  "$script_dir/visioncortex-lan.service" "$rendered_unit" "$port"
systemd-analyze --user verify "$rendered_unit"
deployment_unit_guard "$rendered_unit"
"$script_dir/00-Preflight.sh" --production
deployment_credentials
if [[ ! -e $password_file && ! -L $password_file ]]; then
  [[ -t 0 ]] || { printf '%s\n' 'Provide a private 600-mode Web password file or run interactively.' >&2; exit 1; }
  IFS= read -r -s -p 'Web password (at least 12 characters): ' password; printf '\n'
  IFS= read -r -s -p 'Confirm password: ' password_confirm; printf '\n'
  [[ $password == "$password_confirm" && ${#password} -ge 12 && $password != *$'\r'* ]] || { printf '%s\n' 'Passwords must match and contain at least 12 characters.' >&2; exit 1; }
  mkdir -p -- "$(dirname -- "$password_file")"
  password_tmp=$(mktemp "$(dirname -- "$password_file")/.web_password.XXXXXX")
  chmod 600 -- "$password_tmp"
  printf '%s\n' "$password" > "$password_tmp"
  mv -- "$password_tmp" "$password_file"
  password_tmp=''
  unset password password_confirm
fi
[[ -f $password_file && ! -L $password_file && $(stat -c '%u' -- "$password_file") == $(id -u) && $(stat -c '%a' -- "$password_file") == 600 ]] || { printf '%s\n' 'Web password file must be owned by this user with permissions 600.' >&2; exit 1; }
mkdir -p -- "$runtime_root/tmp" "$input_root"
deployment_install_unit "$rendered_unit"
for _ in $(seq 1 120); do
  main_pid=$(systemctl --user show "$service_name" --property MainPID --value 2>/dev/null || true)
  if [[ $main_pid =~ ^[1-9][0-9]*$ ]] && deployment_health "$main_pid" >/dev/null 2>&1; then
    printf 'VisionCortex LAN Web is healthy: http://127.0.0.1:%s/#/home\nservice=%s\nusername=%s\n' "$port" "$service_name" "$username"
    exit 0
  fi
  sleep 0.5
done
systemctl --user --no-pager --full status "$service_name" >&2 || true
exit 1

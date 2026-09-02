#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
runtime_base='/srv/sentinel-data/VisionCortex3090Ti'
runtime_root="$runtime_base/Runtime"
python="$runtime_base/.venv/bin/python"
credential_root='/home/x1/.config/VisionCortex'
ark_key_file="$credential_root/ark_api_key"
password_file="$credential_root/web_password"
unit_source="$script_dir/visioncortex-lan.service"
unit_root='/home/x1/.config/systemd/user'
unit_target="$unit_root/visioncortex-lan.service"
user_runtime_dir="/run/user/$(id -u)"
password_tmp=''

cleanup() {
  if [[ -n $password_tmp && -f $password_tmp ]]; then
    rm -f -- "$password_tmp"
  fi
}
trap cleanup EXIT

[[ -d $user_runtime_dir && -S $user_runtime_dir/bus ]] || {
  printf 'User service bus is unavailable: %s/bus\n' "$user_runtime_dir" >&2
  exit 1
}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-$user_runtime_dir}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$user_runtime_dir/bus}

[[ $project_root == '/home/x1/Projects/VisionCortex' ]] || {
  printf 'Unexpected project root: %s\n' "$project_root" >&2
  exit 1
}
[[ -x $python ]] || {
  printf '%s\n' 'VisionCortex environment is missing; run 01-Install-And-Validate.sh first.' >&2
  exit 1
}
[[ -f $unit_source ]] || { printf 'Service template is missing: %s\n' "$unit_source" >&2; exit 1; }

mkdir -p -- "$credential_root" "$runtime_root/Input-Manifests" "$runtime_root/tmp" "$unit_root"
chmod 700 -- "$credential_root"

[[ -f $ark_key_file && ! -L $ark_key_file ]] || {
  printf 'Ark credential is missing: %s\n' "$ark_key_file" >&2
  printf '%s\n' 'Create it as one line, then run: chmod 600 ~/.config/VisionCortex/ark_api_key' >&2
  exit 1
}
[[ $(stat -c '%u' -- "$ark_key_file") == $(id -u) ]] || {
  printf '%s\n' 'Ark credential owner is invalid.' >&2
  exit 1
}
[[ $(stat -c '%a' -- "$ark_key_file") == 600 ]] || {
  printf '%s\n' 'Ark credential permissions must be 600.' >&2
  exit 1
}

if [[ ! -e $password_file ]]; then
  [[ -t 0 ]] || {
    printf '%s\n' 'Run this installer in an interactive terminal to create the Web login password.' >&2
    exit 1
  }
  printf '%s\n' 'Set the browser login password for user visioncortex (at least 12 characters).'
  IFS= read -r -s -p 'Password: ' password
  printf '\n'
  IFS= read -r -s -p 'Confirm password: ' password_confirm
  printf '\n'
  [[ $password == "$password_confirm" ]] || { printf '%s\n' 'Passwords do not match.' >&2; exit 1; }
  (( ${#password} >= 12 )) || { printf '%s\n' 'Password must contain at least 12 characters.' >&2; exit 1; }
  [[ $password != *$'\n'* && $password != *$'\r'* ]] || {
    printf '%s\n' 'Password must be a single line.' >&2
    exit 1
  }
  password_tmp=$(mktemp "$credential_root/.web_password.XXXXXX")
  chmod 600 -- "$password_tmp"
  printf '%s\n' "$password" > "$password_tmp"
  mv -- "$password_tmp" "$password_file"
  password_tmp=''
  unset password password_confirm
fi
[[ -f $password_file && ! -L $password_file ]] || {
  printf '%s\n' 'Web password file must be a regular non-symbolic-link file.' >&2
  exit 1
}
[[ $(stat -c '%u' -- "$password_file") == $(id -u) ]] || {
  printf '%s\n' 'Web password file owner is invalid.' >&2
  exit 1
}
[[ $(stat -c '%a' -- "$password_file") == 600 ]] || {
  printf '%s\n' 'Web password file permissions must be 600.' >&2
  exit 1
}

export VISIONCORTEX_PYTHON=$python
"$script_dir/00-Preflight.sh" --production

local_service_changed=false
if systemctl --user is-active --quiet visioncortex-local.service \
  || systemctl --user is-enabled --quiet visioncortex-local.service 2>/dev/null; then
  systemctl --user disable --now visioncortex-local.service
  local_service_changed=true
fi

install -m 0644 -- "$unit_source" "$unit_target"
systemctl --user daemon-reload
if ! systemctl --user enable --now visioncortex-lan.service; then
  if [[ $local_service_changed == true ]]; then
    systemctl --user enable --now visioncortex-local.service || true
  fi
  exit 1
fi

for _ in $(seq 1 120); do
  if VISIONCORTEX_WEB_PASSWORD_FILE="$password_file" \
    VISIONCORTEX_WEB_USERNAME='visioncortex' \
    "$python" - <<'PY' >/dev/null 2>&1
import base64
import os
import urllib.request

username = os.environ["VISIONCORTEX_WEB_USERNAME"]
password_lines = open(
    os.environ["VISIONCORTEX_WEB_PASSWORD_FILE"], encoding="utf-8"
).read().splitlines()
password = password_lines[0] if len(password_lines) == 1 else ""
authorization = base64.b64encode(f"{username}:{password}".encode()).decode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/api/health",
    headers={"Authorization": f"Basic {authorization}"},
)
with urllib.request.urlopen(request, timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    lan_ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '
      /^10\./ { print; exit }
      /^192\.168\./ { print; exit }
      /^172\.(1[6-9]|2[0-9]|3[01])\./ { print; exit }
    ')
    lan_ip=${lan_ip:-127.0.0.1}
    printf 'VisionCortex 3090 Ti LAN server is ready: http://%s:8000/#/home\n' "$lan_ip"
    printf '%s\n' 'Browser username: visioncortex'
    if [[ $(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true) != yes ]]; then
      printf 'warning=Enable boot-time startup once with: sudo loginctl enable-linger %s\n' "$(id -un)"
    fi
    exit 0
  fi
  sleep 0.5
done

systemctl --user --no-pager --full status visioncortex-lan.service >&2 || true
exit 1

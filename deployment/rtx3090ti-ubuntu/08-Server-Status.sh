#!/usr/bin/env bash
set -Eeuo pipefail

runtime_base=${VISIONCORTEX_RUNTIME_BASE:-'/srv/sentinel-data/VisionCortex3090Ti'}
python=${VISIONCORTEX_PYTHON:-"$runtime_base/.venv/bin/python"}
password_file=${VISIONCORTEX_WEB_PASSWORD_FILE:-'/home/x1/.config/VisionCortex/web_password'}
username=${VISIONCORTEX_WEB_USERNAME:-visioncortex}

systemctl --user --no-pager --full status visioncortex-lan.service || true
[[ -x $python ]] || { printf '%s\n' 'VisionCortex Python environment is missing.' >&2; exit 1; }
[[ -f $password_file && ! -L $password_file ]] || {
  printf '%s\n' 'VisionCortex Web password file is missing or unsafe.' >&2
  exit 1
}

VISIONCORTEX_WEB_PASSWORD_FILE="$password_file" \
VISIONCORTEX_WEB_USERNAME="$username" \
"$python" - <<'PY'
import base64
import json
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
with urllib.request.urlopen(request, timeout=3) as response:
    payload = json.load(response)
print(f"health={payload.get('status')}")
print(f"storage_mode={payload.get('storage_mode')}")
print(f"nas_available={payload.get('nas_available')}")
print(f"mllm_enabled={payload.get('mllm_enabled')}")
print(f"ark_key_configured={payload.get('ark_key_configured')}")
print(f"gpu_job_active={(payload.get('execution_queue') or {}).get('gpu_busy')}")
PY

lan_ip=$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '
  /^10\./ { print; exit }
  /^192\.168\./ { print; exit }
  /^172\.(1[6-9]|2[0-9]|3[01])\./ { print; exit }
')
lan_ip=${lan_ip:-127.0.0.1}
printf 'url=http://%s:8000/#/home\nusername=%s\n' "$lan_ip" "$username"

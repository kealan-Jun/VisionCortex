#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
runtime_base=${VISIONCORTEX_RUNTIME_BASE:-'/srv/sentinel-data/VisionCortex3090Ti'}
python=${VISIONCORTEX_PYTHON:-"$runtime_base/.venv/bin/python"}
config=${LABVISION_CONFIG:-"$project_root/configs/rtx3090ti-ubuntu-production.yaml"}
ark_key_file=${VISIONCORTEX_ARK_API_KEY_FILE:-'/home/x1/.config/VisionCortex/ark_api_key'}
port=${VISIONCORTEX_WEB_PORT:-8000}

[[ -x $python ]] || { printf '%s\n' 'VisionCortex Python environment is missing.' >&2; exit 1; }
[[ -f $config ]] || { printf 'VisionCortex configuration is missing: %s\n' "$config" >&2; exit 1; }
[[ -f $ark_key_file && ! -L $ark_key_file ]] || {
  printf '%s\n' 'Ark credential file is missing or unsafe.' >&2
  exit 1
}
[[ $(stat -c '%u' -- "$ark_key_file") == $(id -u) ]] || {
  printf '%s\n' 'Ark credential file owner is invalid.' >&2
  exit 1
}
[[ $(stat -c '%a' -- "$ark_key_file") == 600 ]] || {
  printf '%s\n' 'Ark credential file permissions must be 600.' >&2
  exit 1
}
ARK_API_KEY=$(<"$ark_key_file")
[[ -n $ARK_API_KEY && $ARK_API_KEY == ark-* && $ARK_API_KEY != *$'\n'* ]] || {
  printf '%s\n' 'Ark credential file has invalid contents.' >&2
  exit 1
}
export ARK_API_KEY

cd -- "$project_root"
exec "$python" -m labvision_evidence serve \
  --host 0.0.0.0 \
  --port "$port" \
  --config "$config"

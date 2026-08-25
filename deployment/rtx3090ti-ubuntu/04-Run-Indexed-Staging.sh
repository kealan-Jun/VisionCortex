#!/usr/bin/env bash
set -Eeuo pipefail

experiment_id=''
archive_name=''
cache_mode='cold'
cache_namespace=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-id) experiment_id=${2:?missing experiment id}; shift ;;
    --archive-name) archive_name=${2:?missing archive name}; shift ;;
    --cache-mode) cache_mode=${2:?missing cache mode}; shift ;;
    --cache-namespace) cache_namespace=${2:?missing cache namespace}; shift ;;
    -h|--help)
      printf '%s\n' 'Usage: 04-Run-Indexed-Staging.sh --experiment-id ID --archive-name NAME [--cache-mode cold|reuse] [--cache-namespace NAME]'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n $experiment_id && -n $archive_name ]] || { printf '%s\n' 'Experiment id and archive name are required.' >&2; exit 2; }
[[ $cache_mode == cold || $cache_mode == reuse ]] || { printf '%s\n' 'Cache mode must be cold or reuse.' >&2; exit 2; }
[[ $cache_mode == cold || -n $cache_namespace ]] || { printf '%s\n' 'Reuse mode requires a cache namespace.' >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
config="$project_root/configs/rtx3090ti-ubuntu-production.yaml"
runtime_base='/srv/sentinel-data/VisionCortex3090Ti'
python="$runtime_base/.venv/bin/python"
runtime_root="$runtime_base/Runtime"
runtime_tmp="$runtime_root/tmp"
engine_root="$runtime_base/Engines"
credential_file='/home/x1/.config/VisionCortex/ark_api_key'

[[ -x $python ]] || { printf '%s\n' 'VisionCortex Python environment is missing.' >&2; exit 1; }
[[ -d $runtime_root && -d $runtime_tmp && -d $engine_root ]] || { printf '%s\n' 'Required runtime directories are missing.' >&2; exit 1; }
if [[ -z ${ARK_API_KEY:-} ]]; then
  [[ -f $credential_file && ! -L $credential_file ]] || { printf '%s\n' 'Ark credential file is missing or unsafe.' >&2; exit 1; }
  [[ $(stat -c '%u' -- "$credential_file") == $(id -u) ]] || { printf '%s\n' 'Ark credential owner is invalid.' >&2; exit 1; }
  [[ $(stat -c '%a' -- "$credential_file") == 600 ]] || { printf '%s\n' 'Ark credential permissions must be 600.' >&2; exit 1; }
  IFS= read -r ARK_API_KEY < "$credential_file"
  export ARK_API_KEY
fi
[[ -n ${ARK_API_KEY:-} && $ARK_API_KEY == ark-* ]] || { printf '%s\n' 'Ark credential is not configured correctly.' >&2; exit 1; }

export PATH="$runtime_base/.venv/bin:$PATH"
export TMPDIR="$runtime_tmp"
export VISIONCORTEX_TENSORRT=required

args=(
  -m labvision_evidence run-index-staging
  --experiment-id "$experiment_id"
  --archive-name "$archive_name"
  --cache-mode "$cache_mode"
  --config "$config"
)
if [[ -n $cache_namespace ]]; then
  args+=(--cache-namespace "$cache_namespace")
fi
cd -- "$project_root"
exec "$python" "${args[@]}"

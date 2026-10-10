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
source "$script_dir/_common.sh"
deployment_config production
deployment_credentials
[[ -d $runtime_root && -d $runtime_root/tmp ]] || { printf '%s\n' 'Prepared runtime directories are missing.' >&2; exit 1; }
export PATH="$(dirname -- "$python"):$PATH"
export TMPDIR="$runtime_root/tmp"

args=(
  -m visioncortex run-index-staging
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

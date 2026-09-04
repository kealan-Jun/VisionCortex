#!/usr/bin/env bash
set -Eeuo pipefail

mode='production'
port=8000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) mode='local' ;;
    --production) mode='production' ;;
    --port) port=${2:?missing port}; shift ;;
    -h|--help)
      printf '%s\n' 'Usage: Start-VisionCortex.sh [--production|--local] [--port PORT]'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[[ $port =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || {
  printf '%s\n' 'Invalid port.' >&2
  exit 2
}

install_root='/opt/visioncortex-rtx3050'
app_root="$install_root/app"
python="$install_root/.venv/bin/python"
runtime_root="$install_root/Runtime"
config="$app_root/configs/rtx3050-6gb-ubuntu20-production.yaml"
[[ $mode == production ]] || config="$app_root/configs/rtx3050-6gb-ubuntu20-local.yaml"
[[ -x $python && -f $config ]] || {
  printf '%s\n' 'VisionCortex installation is incomplete.' >&2
  exit 1
}

export VISIONCORTEX_RUNTIME_BASE="$install_root"
export VISIONCORTEX_DEFAULT_CONFIG="$app_root/configs/default.yaml"
export VISIONCORTEX_TENSORRT=required
export VISIONCORTEX_FIRST_PERSON_ENGINE="$install_root/Engines/first_person.engine"
export VISIONCORTEX_THIRD_PERSON_ENGINE="$install_root/Engines/third_person.engine"
export VISIONCORTEX_ULTRALYTICS_CONFIG_DIR="$runtime_root/ThirdParty"
export TMPDIR="$runtime_root/tmp"
export PATH="$install_root/ffmpeg/bin:$install_root/.venv/bin:$PATH"
mkdir -p -- "$TMPDIR" "$VISIONCORTEX_ULTRALYTICS_CONFIG_DIR"

if [[ $mode == production ]]; then
  if [[ -f "$HOME/桌面/nas/experiment_record_index.csv" ]]; then
    nas_root="$HOME/桌面/nas"
  else
    nas_root='/mnt/visioncortex-nas'
  fi
  export VISIONCORTEX_NAS_ROOT="$nas_root"
  export VISIONCORTEX_NAS_INDEX_CSV="$nas_root/experiment_record_index.csv"
  export VISIONCORTEX_NAS_ARCHIVE_ROOT="$nas_root/VisionCortexExperimentArchive"
  export VISIONCORTEX_NAS_CACHE_ROOT="$nas_root/VisionCortexExperimentCache"
  export VISIONCORTEX_LOCAL_INPUT_ROOT="$runtime_root/Input-Manifests"
  export VISIONCORTEX_LOCAL_RUNTIME_ROOT="$runtime_root"
  export VISIONCORTEX_LOCAL_CACHE_ROOT="$VISIONCORTEX_NAS_CACHE_ROOT"
  export VISIONCORTEX_LOCAL_STAGING_ROOT="$VISIONCORTEX_NAS_ARCHIVE_ROOT/.VisionCortex-Run-Staging"
  export VISIONCORTEX_OUTPUT_ROOT="$VISIONCORTEX_LOCAL_STAGING_ROOT"
  key_file="$HOME/.config/VisionCortex/ark_api_key"
  [[ -f $key_file && ! -L $key_file ]] || {
    printf 'Secure Ark credential is missing: %s\n' "$key_file" >&2
    exit 1
  }
  [[ $(stat -c '%u' -- "$key_file") == $(id -u) ]] || {
    printf '%s\n' 'Ark credential owner is invalid.' >&2
    exit 1
  }
  [[ $(stat -c '%a' -- "$key_file") == 600 ]] || {
    printf '%s\n' 'Ark credential permissions must be 600.' >&2
    exit 1
  }
  IFS= read -r ARK_API_KEY < "$key_file"
  [[ $ARK_API_KEY == ark-* && $ARK_API_KEY != *$'\n'* ]] || {
    printf '%s\n' 'Ark credential contents are invalid.' >&2
    exit 1
  }
  export ARK_API_KEY
  "$install_root/Preflight-VisionCortex.sh" --production
else
  export VISIONCORTEX_NAS_INDEX_CSV="$runtime_root/NoNasWeb/local-experiment-index.csv"
  export VISIONCORTEX_DEVICE_REGISTRY="$runtime_root/NoNasWeb/local-device-registry.json"
  export VISIONCORTEX_NAS_ARCHIVE_ROOT="$runtime_root/NoNasWeb/Archives"
  export VISIONCORTEX_LOCAL_INPUT_ROOT="$runtime_root/NoNasWeb/Input-Manifests"
  export VISIONCORTEX_LOCAL_RUNTIME_ROOT="$runtime_root/NoNasWeb/Runtime"
  export VISIONCORTEX_LOCAL_CACHE_ROOT="$runtime_root/NoNasWeb/Cache"
  export VISIONCORTEX_LOCAL_STAGING_ROOT="$runtime_root/NoNasWeb/Archives/.VisionCortex-Run-Staging"
  export VISIONCORTEX_OUTPUT_ROOT="$VISIONCORTEX_LOCAL_STAGING_ROOT"
  mkdir -p -- "$VISIONCORTEX_NAS_ARCHIVE_ROOT" "$VISIONCORTEX_LOCAL_INPUT_ROOT" \
    "$VISIONCORTEX_LOCAL_RUNTIME_ROOT" "$VISIONCORTEX_LOCAL_CACHE_ROOT" \
    "$VISIONCORTEX_LOCAL_STAGING_ROOT"
  "$install_root/Preflight-VisionCortex.sh" --runtime
fi

cd -- "$app_root"
exec "$python" -m visioncortex serve --host 127.0.0.1 --port "$port" --config "$config"

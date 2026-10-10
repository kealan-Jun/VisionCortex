#!/usr/bin/env bash
# Shared deployment paths. Source from a launcher after determining script_dir.
project_root=${VISIONCORTEX_PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
runtime_base=${VISIONCORTEX_RUNTIME_BASE:-"${XDG_DATA_HOME:-$HOME/.local/share}/VisionCortex"}
credential_root="${XDG_CONFIG_HOME:-$HOME/.config}/VisionCortex"
ark_key_file=${VISIONCORTEX_ARK_API_KEY_FILE:-"$credential_root/ark_api_key"}
python=${VISIONCORTEX_PYTHON:-}
if [[ -z $python ]]; then
  for candidate in "${VIRTUAL_ENV:-}/bin/python" "${CONDA_PREFIX:-}/bin/python" \
    "$project_root/.venv/bin/python" "${VISIONCORTEX_VENV:-$runtime_base/.venv}/bin/python"; do
    [[ $candidate != /bin/python && -x $candidate ]] || continue
    python=$candidate
    break
  done
fi
if [[ -z $python ]]; then
  for candidate in python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then python=$(command -v "$candidate"); break; fi
  done
fi

deployment_config() {
  local mode=$1
  if [[ $mode == production ]]; then
    config=${VISIONCORTEX_CONFIG:-${VISIONCORTEX_SITE_CONFIG:-}}
    [[ -n $config ]] || {
      printf '%s\n' 'Production requires an explicit prepared VISIONCORTEX_CONFIG or VISIONCORTEX_SITE_CONFIG.' >&2
      return 1
    }
  else
    config=${VISIONCORTEX_CONFIG:-"$project_root/configs/development-local.yaml"}
  fi
  [[ -x $python ]] || { printf '%s\n' 'Python 3.11 or 3.12 required; prepare a local environment or set VISIONCORTEX_PYTHON.' >&2; return 1; }
  [[ $config == /* ]] || config="$project_root/$config"
  export VISIONCORTEX_CONFIG=$config
  export VISIONCORTEX_PROJECT_ROOT=$project_root
  export VISIONCORTEX_DEPLOYMENT_MODE=$mode
  export VISIONCORTEX_DEFAULT_CONFIG=${VISIONCORTEX_DEFAULT_CONFIG:-"$project_root/configs/default.yaml"}
  local snapshot
  snapshot=$(cd -- "$project_root" && "$python" -B "$script_dir/render_service.py" snapshot "$project_root" "$config" "$mode") || return
  # The helper emits only fixed variable names and shlex-quoted path values.
  eval "$snapshot"
  export VISIONCORTEX_PYTHON=$python
}

deployment_port() {
  port=${VISIONCORTEX_WEB_PORT-$1}
  [[ $port =~ ^[0123456789]{1,5}$ ]] || { printf '%s\n' 'VisionCortex Web port must be an ASCII decimal number from 1024 through 65535.' >&2; return 2; }
  port=$((10#$port))
  (( port >= 1024 && port <= 65535 )) || { printf '%s\n' 'VisionCortex Web port must be from 1024 through 65535.' >&2; return 2; }
  export VISIONCORTEX_WEB_PORT=$port
}

deployment_service() {
  service_name=${VISIONCORTEX_SERVICE_NAME-$1}
  local pattern='^visioncortex-[abcdefghijklmnopqrstuvwxyz0123456789][abcdefghijklmnopqrstuvwxyz0123456789-]*\.service$'
  [[ $service_name =~ $pattern && ${#service_name} -le 64 ]] || { printf '%s\n' 'Invalid VisionCortex service name; use visioncortex-<lowercase-name>.service (at most 64 characters).' >&2; return 2; }
  unit_root="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  unit_target="$unit_root/$service_name"
}

deployment_user_bus() {
  local user_runtime_dir=${XDG_RUNTIME_DIR:-"/run/user/$(id -u)"}
  [[ -d $user_runtime_dir && -S $user_runtime_dir/bus ]] || { printf 'User service bus is unavailable: %s/bus\n' "$user_runtime_dir" >&2; return 1; }
  export XDG_RUNTIME_DIR=$user_runtime_dir
  export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$user_runtime_dir/bus}
}

deployment_credentials() {
  [[ $mllm_enabled == true ]] || return 0
  [[ $api_key_env =~ ^[A-Z][A-Z0-9_]*$ ]] || { printf '%s\n' 'Invalid configured credential environment name.' >&2; return 1; }
  [[ -z ${!api_key_env:-} ]] || return 0
  local key_file=${VISIONCORTEX_MODEL_API_KEY_FILE:-}
  if [[ -z $key_file && $api_key_env == ARK_API_KEY ]]; then key_file=$ark_key_file; fi
  if [[ -z $key_file && ${VISIONCORTEX_WEB_AI_SETTINGS:-0} == 1 ]]; then return 0; fi
  [[ -n $key_file ]] || { printf '%s\n' 'The configured model provider credential is missing.' >&2; return 1; }
  [[ -f $key_file && ! -L $key_file ]] || { printf '%s\n' 'Provider credential file is missing or unsafe.' >&2; return 1; }
  [[ $(stat -c '%u' -- "$key_file") == $(id -u) ]] || { printf '%s\n' 'Provider credential file owner is invalid.' >&2; return 1; }
  [[ $(stat -c '%a' -- "$key_file") == 600 ]] || { printf '%s\n' 'Provider credential file permissions must be 600.' >&2; return 1; }
  local key
  key=$(<"$key_file")
  [[ -n $key && $key != *$'\n'* && $key != *$'\r'* ]] || { printf '%s\n' 'Provider credential file must contain one nonempty line.' >&2; return 1; }
  printf -v "$api_key_env" '%s' "$key"
  export "$api_key_env"
  unset key
}

deployment_unit_guard() {
  local rendered=$1
  [[ ! -L $unit_target ]] || { printf '%s\n' 'Refusing to replace a symbolic-link service file.' >&2; return 1; }
  if [[ -e $unit_target ]] && ! cmp -s -- "$rendered" "$unit_target"; then
    "$python" -B "$script_dir/render_service.py" owner "$project_root" "$unit_target" || return
    if systemctl --user is-active --quiet "$service_name"; then
      printf '%s\n' 'The selected service is active with different configuration; wait for tasks and stop it explicitly before reinstalling.' >&2
      return 1
    fi
  fi
  if ! systemctl --user is-active --quiet "$service_name"; then
    "$python" -B "$script_dir/render_service.py" port "$port" || return
  fi
}

deployment_install_unit() {
  local rendered=$1 unit_temp
  mkdir -p -- "$unit_root"
  if ! cmp -s -- "$rendered" "$unit_target"; then
    unit_temp=$(mktemp "$unit_root/.$service_name.XXXXXX")
    install -m 0644 -- "$rendered" "$unit_temp"
    mv -fT -- "$unit_temp" "$unit_target"
  fi
  systemctl --user daemon-reload
  systemctl --user enable --now "$service_name"
}

deployment_health() {
  "$python" -B "$script_dir/render_service.py" health "$port" "$archive_root" "${1:-}"
}

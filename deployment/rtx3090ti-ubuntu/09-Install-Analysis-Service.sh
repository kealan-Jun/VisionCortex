#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/_common.sh"
config=${VISIONCORTEX_CONFIG:-${VISIONCORTEX_SITE_CONFIG:-}}
[[ -n $config ]] || { printf '%s\n' 'Production automation requires an explicit prepared VISIONCORTEX_CONFIG or VISIONCORTEX_SITE_CONFIG.' >&2; exit 1; }
unit_source="$script_dir/visioncortex-analysis.service"
unit_root="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
service_name=${VISIONCORTEX_SERVICE_NAME-visioncortex-analysis.service}
port=${VISIONCORTEX_WEB_PORT-8001}
# ASCII-only names prevent paths/specifiers from escaping the selected unit.
service_name_pattern='^visioncortex-[abcdefghijklmnopqrstuvwxyz0123456789][abcdefghijklmnopqrstuvwxyz0123456789-]*\.service$'
if [[ ! $service_name =~ $service_name_pattern || ${#service_name} -gt 64 ]]; then
  printf '%s\n' 'Invalid VisionCortex service name; use visioncortex-<lowercase-name>.service (at most 64 characters).' >&2
  exit 2
fi
if [[ ! $port =~ ^[0123456789]{1,5}$ ]]; then
  printf '%s\n' 'VisionCortex Web port must be an ASCII decimal number from 1024 through 65535.' >&2
  exit 2
fi
port=$((10#$port))
if (( port < 1024 || port > 65535 )); then
  printf '%s\n' 'VisionCortex Web port must be from 1024 through 65535.' >&2
  exit 2
fi
unit_target="$unit_root/$service_name"
user_runtime_dir=${XDG_RUNTIME_DIR:-"/run/user/$(id -u)"}
render_root=''
unit_temp=''

cleanup() {
  if [[ -n $unit_temp ]]; then
    rm -f -- "$unit_temp"
  fi
  if [[ -n $render_root ]]; then
    rm -rf -- "$render_root"
  fi
}
trap cleanup EXIT

[[ -d $user_runtime_dir && -S $user_runtime_dir/bus ]] || {
  printf 'User service bus is unavailable: %s/bus\n' "$user_runtime_dir" >&2
  exit 1
}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-$user_runtime_dir}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-unix:path=$user_runtime_dir/bus}

[[ -x $python ]] || {
  printf '%s\n' 'VisionCortex Python environment is missing.' >&2
  exit 1
}
[[ -f $unit_source ]] || {
  printf 'Service template is missing: %s\n' "$unit_source" >&2
  exit 1
}

# Render only from the prepared configuration. Do not rewrite NAS paths, stage
# switches, models or credentials, and never install the unrendered template.
render_root=$(mktemp -d /tmp/visioncortex-analysis.XXXXXX)
rendered_unit="$render_root/$service_name"
runtime_root=$("$python" -B - "$project_root" "$python" "$config" "$ark_key_file" \
  "$unit_source" "$rendered_unit" "$unit_root" "$port" <<'PY'
import os
import json
from pathlib import Path
import sys

if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit('Prepared Python must be version 3.11 or 3.12.')

def absolute(value):
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError('Deployment paths must not contain control characters.')
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError('Deployment paths must be absolute.')
    return str(path)

project, python, config, key_file, template, rendered, _ = map(absolute, sys.argv[1:8])
port = sys.argv[8]
if '\\' in project or project.endswith(' '):
    raise ValueError('Project directory must not contain backslashes or trailing spaces.')
runner = Path(project) / 'deployment/rtx3090ti-ubuntu/06-Run-LAN-Server.sh'
default_config = absolute(os.environ.get('VISIONCORTEX_DEFAULT_CONFIG', str(Path(project) / 'configs/default.yaml')))
config = absolute(str(Path(config).resolve()))
default_config = absolute(str(Path(default_config).resolve()))
for file in (Path(config), Path(default_config), Path(project) / 'src/visioncortex/config.py'):
    if not file.is_file():
        raise ValueError(f'Prepared deployment file is missing: {file}')
if not os.access(runner, os.X_OK):
    raise ValueError(f'Prepared service launcher is missing: {runner}')
sys.path.insert(0, str(Path(project) / 'src'))
os.environ['VISIONCORTEX_DEFAULT_CONFIG'] = default_config
from visioncortex.config import load_config
from visioncortex.automation_readiness import configuration_blockers, configuration_digest
from visioncortex.runtime_process import role

settings = load_config(Path(config))
if settings.get('project', {}).get('site_configuration_required', False):
    raise ValueError('Production site configuration is unprepared.')
storage = settings['storage']
blockers = configuration_blockers(settings, role(settings))
if blockers:
    raise ValueError('Prepared NAS automation configuration is invalid: ' + ', '.join(blockers))
runtime = absolute(storage['local_runtime_root'])
paths = {
    'VISIONCORTEX_NAS_ARCHIVE_ROOT': absolute(storage['archive_root']),
    'VISIONCORTEX_LOCAL_INPUT_ROOT': absolute(storage['local_input_root']),
    'VISIONCORTEX_LOCAL_RUNTIME_ROOT': runtime,
    'VISIONCORTEX_LOCAL_CACHE_ROOT': absolute(storage['local_cache_root']),
    'VISIONCORTEX_LOCAL_STAGING_ROOT': absolute(storage['local_staging_root']),
    'VISIONCORTEX_OUTPUT_ROOT': absolute(settings['project']['output_root']),
    'VISIONCORTEX_FIRST_PERSON_ENGINE': absolute(settings['models']['first_person_engine']),
    'VISIONCORTEX_THIRD_PERSON_ENGINE': absolute(settings['models']['third_person_engine']),
}
for root in (settings['collection_ingest']['source_root'], storage['archive_root'],
             storage['local_staging_root'], storage['local_cache_root']):
    if not Path(absolute(root)).is_dir():
        raise ValueError(f'Required prepared storage root is missing: {root}')
ai_settings = os.environ.get('VISIONCORTEX_WEB_AI_SETTINGS', '1')
if ai_settings not in {'0', '1'}:
    raise ValueError('VISIONCORTEX_WEB_AI_SETTINGS must be 0 or 1.')
environment = paths | {
    'VISIONCORTEX_CONFIG': config, 'VISIONCORTEX_DEFAULT_CONFIG': default_config,
    'VISIONCORTEX_RUNTIME_ROLE': 'combined',
    'VISIONCORTEX_PROJECT_ROOT': project, 'VISIONCORTEX_DEPLOYMENT_MODE': 'production',
    'VISIONCORTEX_RUNTIME_BASE': absolute(os.environ.get('VISIONCORTEX_RUNTIME_BASE', str(Path(runtime).parent))),
    'VISIONCORTEX_PYTHON': python, 'PYTHONPATH': str(Path(project) / 'src'),
    'PYTHONDONTWRITEBYTECODE': '1',
    'VISIONCORTEX_WEB_HOST': '127.0.0.1', 'VISIONCORTEX_WEB_PORT': port,
    'VISIONCORTEX_WEB_AI_SETTINGS': ai_settings, 'VISIONCORTEX_TENSORRT': 'required',
    'VISIONCORTEX_ARK_API_KEY_FILE': key_file, 'TMPDIR': str(Path(runtime) / 'tmp'),
    'VISIONCORTEX_ULTRALYTICS_CONFIG_DIR': str(Path(runtime) / 'ThirdParty'),
}
if os.environ.get('VISIONCORTEX_MODEL_API_KEY_FILE'):
    environment['VISIONCORTEX_MODEL_API_KEY_FILE'] = absolute(os.environ['VISIONCORTEX_MODEL_API_KEY_FILE'])
# Bind the startup identity to the exact effective configuration that this
# rendered unit will load, including its forced path and TensorRT overrides.
os.environ.update(environment)
prepared_settings = load_config(Path(config))

def quoted(value, *, command_argument=False):
    value = value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
    if command_argument:
        # systemd expands dollars in arguments, but not in the executable name.
        value = value.replace('$', '$$')
    return '"' + value + '"'

unit = Path(template).read_text(encoding='utf-8')
for marker, value in {
    # WorkingDirectory takes a raw path; surrounding quotes become path bytes.
    '@PROJECT_ROOT@': project.replace('%', '%%'),
    '@ENVIRONMENT@': '\n'.join('Environment=' + quoted(name + '=' + value) for name, value in environment.items()),
    '@RUNNER@': quoted(str(runner), command_argument=True),
}.items():
    if unit.count(marker) != 1:
        raise ValueError('Analysis service template is invalid.')
    unit = unit.replace(marker, value)
Path(rendered).write_text(unit, encoding='utf-8')
Path(rendered + '.json').write_text(json.dumps({
    'config_path': config, 'default_config_path': default_config,
    'settings_sha256': configuration_digest(prepared_settings),
}), encoding='utf-8')
print(runtime)
PY
)
systemd-analyze --user verify "$rendered_unit"

if [[ -L $unit_target ]]; then
  printf '%s\n' 'Refusing to replace a symbolic-link service file.' >&2
  exit 1
fi
if [[ -e $unit_target ]] && ! cmp -s -- "$rendered_unit" "$unit_target"; then
  "$python" -B "$script_dir/render_service.py" owner "$project_root" "$unit_target"
fi

if [[ -e $unit_target || -L $unit_target ]] \
  && ! cmp -s -- "$rendered_unit" "$unit_target" \
  && systemctl --user is-active --quiet "$service_name"; then
  printf '%s\n' '正在运行的自动化服务使用不同配置。请先等待任务结束，再由管理员停止服务后重新安装；本次未替换服务文件。' >&2
  printf '停止命令：systemctl --user stop %s\n' "$service_name" >&2
  exit 1
fi

mkdir -p -- \
  "$runtime_root/Input-Manifests" \
  "$runtime_root/tmp" \
  "$unit_root"
if ! cmp -s -- "$rendered_unit" "$unit_target"; then
  unit_temp=$(mktemp "$unit_root/.$service_name.XXXXXX")
  install -m 0644 -- "$rendered_unit" "$unit_temp"
  mv -fT -- "$unit_temp" "$unit_target"
  unit_temp=''
fi
systemctl --user daemon-reload
systemctl --user enable --now "$service_name"
if [[ $(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true) != yes ]]; then
  sudo loginctl enable-linger "$(id -un)"
fi

readiness_response="$render_root/readiness.json"
last_readiness_reason='后台状态接口尚未响应。'
for _ in $(seq 1 180); do
  main_pid=$(systemctl --user show "$service_name" --property MainPID --value 2>/dev/null || true)
  if [[ $main_pid =~ ^[1-9][0-9]*$ ]]; then
    if curl --silent --show-error --fail --max-time 2 --max-filesize 65536 \
      --output "$readiness_response" "http://127.0.0.1:$port/health/automation" 2>/dev/null; then
      if readiness_check=$("$python" -B - "$readiness_response" "$rendered_unit.json" "$main_pid" "$port" <<'PY'
import json
from pathlib import Path
import sys

def reject(message):
    print(message)
    raise SystemExit(1)

try:
    response, expected, pid, port = sys.argv[1:]
    raw = Path(response).read_bytes()
    if len(raw) > 65536:
        reject('后台状态响应超过限制。')
    data = json.loads(raw)
    prepared = json.loads(Path(expected).read_bytes())
except (OSError, ValueError):
    reject('后台状态响应不是有效 JSON。')
if not isinstance(data, dict) or data.get('schema_version') != 'visioncortex-automation-readiness/1':
    reject('后台状态响应格式不匹配。')
if type(data.get('pid')) is not int or data['pid'] != int(pid):
    reject(f'后台状态来自其他进程；请检查 {port} 端口占用。')
if data.get('configuration') != prepared:
    reject('后台实际配置与本次安装配置不一致。')
if data.get('runtime_role') != 'combined' or data.get('ready') is not True:
    reject('NAS 自动处理后台尚未就绪；请检查服务日志。')
if data.get('storage_maintenance') is not False:
    reject('后台仍在存储维护状态。')
worker = data.get('worker') or {}
monitor = data.get('nas_monitor') or {}
device = data.get('device_day') or {}
if not isinstance(worker, dict) or worker.get('status') != 'running' or type(worker.get('pid')) is not int or worker['pid'] != int(pid):
    reject('后台执行者尚未就绪。')
if not isinstance(monitor, dict) or monitor.get('thread_alive') is not True or monitor.get('status') != 'watching':
    reject('NAS 监控线程尚未就绪。')
if not isinstance(device, dict) or device.get('thread_alive') is not True or device.get('runner_initialized') is not True or device.get('status') not in {'waiting_for_nas_monitor', 'running'}:
    reject('设备日自动处理调度器尚未就绪。')
paused = device.get('paused_stages', [])
if not isinstance(paused, list) or any(not isinstance(stage, str) or stage not in {'retention', 'vision', 'stt', 'understanding', 'report'} for stage in paused):
    reject('后台暂停阶段状态格式不匹配。')
if paused:
    names = {'retention': '原始文件留存', 'vision': '视频预处理', 'stt': '录音识别',
             'understanding': '视频理解', 'report': '日报'}
    print('按预配策略暂停的阶段：' + '、'.join(names[stage] for stage in paused) + '。')
PY
      ); then
        printf '%s\n' 'VisionCortex NAS 自动处理后台已启动，服务将按预配策略自动处理并发布归档。'
        [[ -z $readiness_check ]] || printf '%s\n' "$readiness_check"
        printf '查看入口：http://127.0.0.1:%s/#/home\n' "$port"
        printf '%s\n' '后台启动通过不代表真实模型执行、归档完整性或结果质量已验收。'
        exit 0
      else
        last_readiness_reason=$readiness_check
      fi
    else
      last_readiness_reason='后台状态接口尚未响应。'
    fi
  else
    last_readiness_reason='自动化服务尚无有效主进程。'
  fi
  sleep 1
done

printf 'NAS 自动处理后台未就绪：%s\n' "$last_readiness_reason" >&2
systemctl --user --no-pager --full status "$service_name" >&2 || true
exit 1

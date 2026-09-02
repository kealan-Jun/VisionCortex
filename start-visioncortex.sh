#!/usr/bin/env bash
set -Eeuo pipefail

port=8000
config=''
python_override=''
open_browser=true
check_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) port=${2:?缺少端口号}; shift ;;
    --config) config=${2:?缺少配置文件}; shift ;;
    --python) python_override=${2:?缺少Python路径}; shift ;;
    --no-browser) open_browser=false ;;
    --check-only) check_only=true ;;
    -h|--help)
      printf '%s\n' '用法：./start-visioncortex.sh [--port 8000] [--config FILE] [--python FILE] [--no-browser] [--check-only]'
      exit 0
      ;;
    *) printf '无法识别的参数：%s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
[[ $port =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || {
  printf '%s\n' '端口号必须在1到65535之间。' >&2
  exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$script_dir
if [[ -z $config ]]; then
  config="$project_root/configs/development-local.yaml"
elif [[ $config != /* ]]; then
  config="$project_root/$config"
fi
[[ -f $config ]] || { printf '找不到配置文件：%s\n' "$config" >&2; exit 1; }

if [[ -n $python_override ]]; then
  [[ -x $python_override ]] || { printf 'Python不可执行：%s\n' "$python_override" >&2; exit 1; }
  python=$python_override
elif [[ -x $project_root/.venv/bin/python ]]; then
  python="$project_root/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
  python=$(command -v python3.12)
elif command -v python3.11 >/dev/null 2>&1; then
  python=$(command -v python3.11)
elif command -v python3 >/dev/null 2>&1; then
  python=$(command -v python3)
else
  printf '%s\n' '没有找到Python。请安装Python 3.11或3.12，然后重新运行。' >&2
  exit 1
fi

"$python" "$project_root/tools/doctor.py" --project-root "$project_root" || {
  printf '%s\n' '环境检查未通过。请按照上面的“需要处理”完成修复。' >&2
  exit 1
}
[[ $check_only == true ]] && exit 0

url="http://127.0.0.1:$port"
health_ok() {
  curl --silent --show-error --fail --max-time 3 "$url/api/health" 2>/dev/null \
    | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"'
}
if health_ok; then
  printf 'VisionCortex已经启动：%s/#/home\n' "$url"
  if [[ $open_browser == true ]]; then
    if [[ $(uname -s) == Darwin ]]; then open "$url/#/home" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url/#/home" >/dev/null 2>&1 &
    fi
  fi
  exit 0
fi

runtime_root="$project_root/outputs/development-runtime"
run_root="$project_root/outputs/development-runs"
cache_root="$project_root/outputs/development-cache"
input_root="$project_root/outputs/development-input"
archive_root="$project_root/outputs/clean-local-validation"
mkdir -p -- "$runtime_root" "$run_root" "$cache_root" "$input_root" "$archive_root"
stdout_log="$runtime_root/visioncortex-web.stdout.log"
stderr_log="$runtime_root/visioncortex-web.stderr.log"
pid_file="$runtime_root/visioncortex-web.pid"

export VISIONCORTEX_CONFIG="$config"
export VISIONCORTEX_NAS_INDEX_CSV="$project_root/examples/development-index.csv"
export VISIONCORTEX_NAS_ARCHIVE_ROOT="$archive_root"
export VISIONCORTEX_NAS_CACHE_ROOT="$cache_root"
export VISIONCORTEX_LOCAL_INPUT_ROOT="$input_root"
export VISIONCORTEX_LOCAL_RUNTIME_ROOT="$runtime_root"
export VISIONCORTEX_LOCAL_CACHE_ROOT="$cache_root"
export VISIONCORTEX_LOCAL_STAGING_ROOT="$run_root"
export VISIONCORTEX_OUTPUT_ROOT="$run_root"

cd -- "$project_root"
nohup "$python" -m visioncortex serve --host 127.0.0.1 --port "$port" --config "$config" \
  >"$stdout_log" 2>"$stderr_log" < /dev/null &
web_pid=$!
printf '%s\n' "$web_pid" > "$pid_file"

ready=false
for _ in $(seq 1 120); do
  if health_ok; then ready=true; break; fi
  if ! kill -0 "$web_pid" 2>/dev/null; then break; fi
  sleep 0.5
done
if [[ $ready != true ]]; then
  if kill -0 "$web_pid" 2>/dev/null; then kill "$web_pid"; fi
  rm -f -- "$pid_file"
  printf '%s\n' 'VisionCortex启动失败，最近的错误信息：' >&2
  tail -n 30 -- "$stderr_log" >&2 || true
  exit 1
fi

printf 'VisionCortex启动成功：%s/#/home\n' "$url"
printf '%s\n' '运行模式：本地开发（不访问NAS、不自动运行模型）'
printf '日志目录：%s\n' "$runtime_root"
if [[ $open_browser == true ]]; then
  if [[ $(uname -s) == Darwin ]]; then open "$url/#/home" >/dev/null 2>&1 &
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url/#/home" >/dev/null 2>&1 &
  fi
fi

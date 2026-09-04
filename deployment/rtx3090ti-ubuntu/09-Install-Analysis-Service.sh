#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
runtime_root='/srv/sentinel-data/VisionCortex3090Ti/Runtime'
nas_root='/home/x1/桌面/nas'
unit_source="$script_dir/visioncortex-analysis.service"
unit_root="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_target="$unit_root/visioncortex-analysis.service"
user_runtime_dir="/run/user/$(id -u)"

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
[[ -x /srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python ]] || {
  printf '%s\n' 'VisionCortex Python environment is missing.' >&2
  exit 1
}
[[ -d $nas_root ]] || {
  printf 'VisionCortex NAS is not mounted: %s\n' "$nas_root" >&2
  exit 1
}
for required_nas_root in \
  "$nas_root/VisionCortexExperimentArchive" \
  "$nas_root/VisionCortexExperimentArchive/.VisionCortex-Run-Staging" \
  "$nas_root/VisionCortexExperimentCache"; do
  [[ -d $required_nas_root ]] || {
    printf 'Required VisionCortex NAS root is missing: %s\n' "$required_nas_root" >&2
    exit 1
  }
done
[[ -f $unit_source ]] || {
  printf 'Service template is missing: %s\n' "$unit_source" >&2
  exit 1
}

mkdir -p -- \
  "$runtime_root/Input-Manifests" \
  "$runtime_root/tmp" \
  "$unit_root"
install -m 0644 -- "$unit_source" "$unit_target"
systemctl --user daemon-reload
systemctl --user enable --now visioncortex-analysis.service
if [[ $(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true) != yes ]]; then
  sudo loginctl enable-linger "$(id -un)"
fi

for _ in $(seq 1 180); do
  if curl --silent --show-error --fail --max-time 2 \
    http://127.0.0.1:8001/api/health >/dev/null 2>&1; then
    printf '%s\n' 'VisionCortex NAS capture monitor is healthy at http://127.0.0.1:8001/#/home'
    printf '%s\n' 'New recorder files will appear as capture batches after the sidecars are published.'
    exit 0
  fi
  sleep 1
done

systemctl --user --no-pager --full status visioncortex-analysis.service >&2 || true
exit 1

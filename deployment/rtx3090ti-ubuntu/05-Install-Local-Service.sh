#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/../.." && pwd -P)
runtime_root='/srv/sentinel-data/VisionCortex3090Ti/Runtime'
local_root="$runtime_root/NoNasWeb"
unit_source="$script_dir/visioncortex-local.service"
unit_root='/home/x1/.config/systemd/user'
unit_target="$unit_root/visioncortex-local.service"
launcher_source="$script_dir/Open-VisionCortex.sh"
launcher_root="$runtime_root/bin"
launcher_target="$launcher_root/Open-VisionCortex.sh"
application_root="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
autostart_root="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
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
  printf '%s\n' 'VisionCortex local Python environment is missing.' >&2
  exit 1
}
[[ -f $unit_source ]] || {
  printf 'Service template is missing: %s\n' "$unit_source" >&2
  exit 1
}
[[ -f $launcher_source ]] || {
  printf 'Browser launcher is missing: %s\n' "$launcher_source" >&2
  exit 1
}

mkdir -p -- \
  "$local_root/Archives/.VisionCortex-Run-Staging" \
  "$local_root/Input-Manifests" \
  "$local_root/Runtime" \
  "$local_root/Cache" \
  "$runtime_root/tmp" \
  "$unit_root" \
  "$launcher_root" \
  "$application_root" \
  "$autostart_root"
install -m 0644 -- "$unit_source" "$unit_target"
install -m 0755 -- "$launcher_source" "$launcher_target"
cat > "$application_root/visioncortex.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=VisionCortex
Comment=打开多视角实验视频分析工作台
Exec=$launcher_target
Icon=applications-science
Terminal=false
Categories=Science;Education;
StartupNotify=true
StartupWMClass=VisionCortex
EOF
cat > "$autostart_root/visioncortex.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=VisionCortex
Comment=登录后自动打开多视角实验视频分析工作台
Exec=$launcher_target
Icon=applications-science
Terminal=false
X-GNOME-Autostart-enabled=true
StartupNotify=true
StartupWMClass=VisionCortex
EOF
chmod 0644 -- \
  "$application_root/visioncortex.desktop" \
  "$autostart_root/visioncortex.desktop"
systemctl --user daemon-reload
systemctl --user enable --now visioncortex-local.service
if [[ $(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || true) != yes ]]; then
  sudo loginctl enable-linger "$(id -un)"
fi
command -v update-desktop-database >/dev/null 2>&1 \
  && update-desktop-database "$application_root" >/dev/null 2>&1 \
  || true

for _ in $(seq 1 60); do
  if curl --silent --show-error --fail --max-time 2 \
    http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    printf '%s\n' 'VisionCortex local no-NAS service is healthy at http://127.0.0.1:8000/#/home'
    printf '%s\n' 'VisionCortex will open automatically after desktop login.'
    exit 0
  fi
  sleep 0.5
done

systemctl --user --no-pager --full status visioncortex-local.service >&2 || true
exit 1

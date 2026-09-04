#!/usr/bin/env bash
set -Eeuo pipefail

install_root=${VISIONCORTEX_INSTALL_ROOT:-/opt/visioncortex-rtx3050}
app_root="$install_root/app"
start_script="$install_root/Start-VisionCortex.sh"
launcher_source="$app_root/deployment/rtx3050-ubuntu20/Open-VisionCortex.sh"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
application_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

[[ -x $start_script && -f $launcher_source ]] || {
  printf '%s\n' 'VisionCortex 安装不完整，无法配置开机自启。' >&2
  exit 1
}

install -d -m 0755 "$unit_dir" "$application_dir"
install -m 0755 "$launcher_source" "$install_root/Open-VisionCortex.sh"
cat > "$unit_dir/visioncortex-analysis.service" <<EOF
[Unit]
Description=VisionCortex end-to-end video analysis service
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$app_root
ExecStart=$start_script --production
Restart=always
RestartSec=5
TimeoutStopSec=60
UMask=0077

[Install]
WantedBy=default.target
EOF
cat > "$application_dir/visioncortex.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=VisionCortex
Comment=打开实验视频分析工作台
Exec=$install_root/Open-VisionCortex.sh
Icon=applications-science
Terminal=false
Categories=Science;Education;
StartupNotify=true
EOF

systemctl --user daemon-reload
systemctl --user enable --now visioncortex-analysis.service
if [[ $(loginctl show-user "$(id -un)" -p Linger --value) != yes ]]; then
  sudo loginctl enable-linger "$(id -un)"
fi
command -v update-desktop-database >/dev/null 2>&1 \
  && update-desktop-database "$application_dir" >/dev/null 2>&1 \
  || true

printf '%s\n' 'VisionCortex 已配置开机自启和应用图标。'

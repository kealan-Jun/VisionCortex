#!/usr/bin/env bash
# Locate either an installed USB payload or the actual source checkout.
if [[ -d $script_dir/app/src/visioncortex ]]; then
  install_root=${VISIONCORTEX_INSTALL_ROOT:-$script_dir}
  app_root=${VISIONCORTEX_PROJECT_ROOT:-"$install_root/app"}
else
  app_root=${VISIONCORTEX_PROJECT_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
  if [[ -f $app_root/../.package-id || -x $app_root/../.venv/bin/python ]]; then
    install_root=${VISIONCORTEX_INSTALL_ROOT:-$(cd -- "$app_root/.." && pwd -P)}
  else
    install_root=${VISIONCORTEX_INSTALL_ROOT:-/opt/visioncortex-rtx3050}
  fi
fi
export VISIONCORTEX_PROJECT_ROOT=$app_root
if [[ -x $install_root/.venv/bin/python ]]; then
  export VISIONCORTEX_VENV=${VISIONCORTEX_VENV:-"$install_root/.venv"}
fi
shared_dir="$app_root/deployment/rtx3090ti-ubuntu"
script_dir=$shared_dir
source "$shared_dir/_common.sh"
export VISIONCORTEX_DEFAULT_CONFIG=${VISIONCORTEX_DEFAULT_CONFIG:-"$app_root/configs/default.yaml"}

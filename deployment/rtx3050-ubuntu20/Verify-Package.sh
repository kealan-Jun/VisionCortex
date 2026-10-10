#!/usr/bin/env bash
set -Eeuo pipefail

package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
manifest="$package_root/SHA256SUMS"
[[ -f $manifest ]] || { printf 'Package checksum manifest is missing: %s\n' "$manifest" >&2; exit 1; }
cd -- "$package_root"
sha256sum --check --strict --quiet SHA256SUMS
printf 'package_verification=passed\n'

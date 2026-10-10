#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# != 1 || $1 != --setup ]]; then
  printf '%s\n' '用法：bash install.sh --setup'
  exit 2
fi
package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
arguments=$(mktemp /tmp/visioncortex-prepared-host.XXXXXX)
trap 'rm -f -- "$arguments"' EXIT
# Only the system Python standard library is needed to verify the package.
# No shell evaluation, downloads, dependency installation or model operations.
python3 -I - "$package_root" > "$arguments" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

root = Path(sys.argv[1])
manifest_name = 'PackageManifest.json'
def checked_path(name):
    if not isinstance(name, str) or not name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError('Invalid package path')
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or '\\' in name or str(pure) != name:
        raise ValueError('Package path escapes its root')
    path = root.joinpath(*pure.parts)
    for child in (path, *path.parents):
        if child == root.parent:
            break
        if child.is_symlink():
            raise ValueError('Package contains a symlink')
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError('Package path escapes its root') from exc
    return path

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

try:
    manifest = json.loads(checked_path(manifest_name).read_text())
    if (manifest.get('schema_version') != 'visioncortex-prepared-automation-candidate/1'
            or manifest.get('release_ready') is not False or manifest.get('commit') is not None):
        raise ValueError('Expected an explicitly identified candidate service package')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Package content manifest is missing')
    for name, expected in files.items():
        path = checked_path(name)
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid package digest')
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Package content changed: ' + name)
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Package contains a symlink')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name != manifest_name:
                actual.add(name)
    if actual != set(files):
        raise ValueError('Package file set differs from its manifest')
    source_files = manifest.get('source_files')
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError('Candidate source identity is missing')
    expected_source = {name[4:] for name in files if name.startswith('App/') and name != 'App/src/visioncortex/BuildManifest.json'}
    if set(source_files) != expected_source:
        raise ValueError('Candidate source file set is inconsistent')
    for name, expected in source_files.items():
        checked_path('App/' + name)
        if files.get('App/' + name) != expected:
            raise ValueError('Candidate source identity differs from package contents')
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    identity = json.loads(checked_path('App/src/visioncortex/BuildManifest.json').read_text())
    if (manifest.get('source_digest') != source_digest or identity.get('source_digest') != source_digest
            or identity.get('commit') is not None or identity.get('release_ready') is not False
            or identity.get('lock_digest') != files.get('App/pyproject.toml')
            or identity.get('lock_digest_scope') != 'pyproject_metadata_not_dependency_lock'):
        raise ValueError('Candidate build identity is inconsistent')
    host = json.loads(checked_path('PreparedHost.json').read_text())
    python = host.get('python')
    if not isinstance(python, str) or not Path(python).is_absolute() or any(ord(c) < 32 or ord(c) == 127 for c in python):
        raise ValueError('Prepared Python path is invalid')
    service = host.get('service_name')
    if not isinstance(service, str) or len(service) > 64 or not re.fullmatch(r'visioncortex-[a-z0-9][a-z0-9-]*\.service', service, re.ASCII):
        raise ValueError('Prepared service name is invalid')
    port, ai = host.get('port'), host.get('web_ai_settings')
    if type(port) is not int or not 1024 <= port <= 65535 or type(ai) is not int or ai not in {0, 1}:
        raise ValueError('Prepared port or AI settings flag is invalid')
    config_name = host.get('config')
    if not isinstance(config_name, str) or not config_name.startswith('App/configs/') or config_name not in files:
        raise ValueError('Prepared configuration is not bound to this package')
    config = checked_path(config_name)
    launcher = 'App/deployment/rtx3090ti-ubuntu/09-Install-Analysis-Service.sh'
    default = 'App/configs/default.yaml'
    if launcher not in files or default not in files:
        raise ValueError('Package installation entry is missing')
    values = [str(root/'App'), python, str(config), str(checked_path(default)), service, str(port), str(ai)]
    sys.stdout.buffer.write(b''.join(value.encode('utf-8') + b'\0' for value in values))
except (OSError, ValueError, TypeError, AttributeError) as exc:
    print('服务包校验未通过：' + str(exc), file=sys.stderr)
    raise SystemExit(1) from exc
PY
mapfile -d '' -t prepared < "$arguments"
[[ ${#prepared[@]} == 7 ]] || { printf '%s\n' '服务包预配信息不完整。' >&2; exit 1; }
export VISIONCORTEX_PROJECT_ROOT="${prepared[0]}"
export VISIONCORTEX_PYTHON="${prepared[1]}"
export VISIONCORTEX_CONFIG="${prepared[2]}"
export VISIONCORTEX_DEFAULT_CONFIG="${prepared[3]}"
export VISIONCORTEX_SERVICE_NAME="${prepared[4]}"
export VISIONCORTEX_WEB_PORT="${prepared[5]}"
export VISIONCORTEX_WEB_AI_SETTINGS="${prepared[6]}"
rm -f -- "$arguments"
trap - EXIT
exec bash "$VISIONCORTEX_PROJECT_ROOT/deployment/rtx3090ti-ubuntu/09-Install-Analysis-Service.sh"

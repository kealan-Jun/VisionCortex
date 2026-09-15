"""Same build identity for API, workers and installed distributions."""
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path


def identity():
    try:
        package_version = version('visioncortex')
    except PackageNotFoundError:
        package_version = 'uninstalled'
    manifest = Path(os.environ.get('VISIONCORTEX_BUILD_MANIFEST') or Path(__file__).with_name('BuildManifest.json'))
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        return {key: data.get(key) for key in ('version', 'commit', 'source_digest', 'lock_digest')}
    return {'version': package_version, 'commit': None, 'source_digest': None,
            'mode': 'development_checkout'}

"""Same build identity for API, workers and installed distributions."""
from importlib.metadata import PackageNotFoundError, version
import json
import hashlib
import subprocess
from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def identity():
    try:
        package_version = version('visioncortex')
    except PackageNotFoundError:
        package_version = 'uninstalled'
    manifest = Path(os.environ.get('VISIONCORTEX_BUILD_MANIFEST') or Path(__file__).with_name('BuildManifest.json'))
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        return {key: data.get(key) for key in ('version', 'commit', 'source_digest', 'lock_digest')}
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(root.rglob('*')):
        if source.is_file() and source.suffix in {'.py', '.js', '.html', '.css'}:
            digest.update(source.relative_to(root).as_posix().encode()+b'\0'+source.read_bytes())
    try:
        commit = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True,timeout=3).stdout.strip()
        dirty = bool(subprocess.run(['git','status','--porcelain','--untracked-files=no'],cwd=root,capture_output=True,text=True,check=True,timeout=3).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {'version': package_version, 'commit': commit, 'source_digest': digest.hexdigest(),
            'tracked_worktree_dirty': dirty, 'mode': 'development_checkout'}

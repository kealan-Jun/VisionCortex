#!/usr/bin/env python3
"""Launch one role from the selected release and its verified environment."""
import os
from pathlib import Path
import sys
from Release import verify, verify_environment


def launch(role):
    if role not in {'web', 'worker'}:
        raise ValueError('Expected web or worker')
    base = Path(os.environ['VISIONCORTEX_RELEASE_ROOT']).resolve()
    bundle = (base / 'Current').resolve()
    if bundle.parent != base / 'Releases':
        raise ValueError('Selected release must remain inside Releases')
    manifest = verify(bundle)
    python = verify_environment(base, manifest)
    env = dict(os.environ, VISIONCORTEX_RUNTIME_ROLE=role, VISIONCORTEX_BUILD_MANIFEST=str(bundle / 'BuildManifest.json'),
               PYTHONDONTWRITEBYTECODE='1')
    config = os.environ['VISIONCORTEX_CONFIG']
    command = ([str(python), '-m', 'uvicorn', 'visioncortex.api:app', '--host', '127.0.0.1', '--port', '8001'] if role == 'web'
               else [str(python), '-m', 'visioncortex', 'worker', '--config', config])
    os.execve(python, command, env)


if __name__ == '__main__':
    launch(sys.argv[1])

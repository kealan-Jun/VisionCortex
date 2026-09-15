#!/usr/bin/env python3
"""Install a verified wheel bundle into its own immutable environment.

Run only in an authorized deployment. A failed environment is never launched;
retain it for diagnosis instead of overwriting another release's environment.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from Release import atomic_json, deployment_lock, verify, verify_environment


def install(base, release):
    with deployment_lock(base) as base:
        return _install(base, release)


def _install(base, release):
    if Path(release).name != release:
        raise ValueError('Invalid release directory')
    bundle = base / 'Releases' / release
    if bundle.resolve().parent != base / 'Releases':
        raise ValueError('Release escapes versioned deployment root')
    manifest = verify(bundle)
    wheels = list((bundle / 'Wheelhouse').glob('visioncortex-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Exactly one VisionCortex wheel is required')
    environment = base / 'Environments' / manifest['source_digest']
    receipt = environment / 'Installation.json'
    if receipt.is_file():
        verify_environment(base, manifest)
        return environment
    environment.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-m', 'venv', '--copies', str(environment)], check=True)
    python = environment / 'bin' / 'python'
    if not python.is_file():
        python = environment / 'Scripts' / 'python.exe'
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(bundle / 'Wheelhouse'),
                    '--require-hashes', '-r', str(bundle / 'requirements.lock')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheels[0])], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    atomic_json(receipt, {'source_digest': manifest['source_digest'], 'commit': manifest['commit'],
                          'python': str(python), 'dependency_lock_sha256': manifest['lock_digest']})
    return environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('release')
    args = parser.parse_args()
    print(install(args.base, args.release))

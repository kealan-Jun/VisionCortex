#!/usr/bin/env python3
"""Seal and atomically select an already built release; never synchronize Git.

The bundle must identify a reviewed commit from either synchronized repository. This tool does
not install dependencies, stop services, alter data or infer release approval.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

MANIFEST = 'BuildManifest.json'


@contextmanager
def deployment_lock(base):
    """Serialize installation and pointer changes using an owned local lock."""
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.Deployment.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield base
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def sync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def files(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Release bundles cannot contain symlinks')
        if path.is_file() and path != root / MANIFEST:
            relative = path.relative_to(root).as_posix()
            if path.name.startswith('.env') or path.suffix in {'.pt', '.engine', '.mp4', '.opus', '.key', '.pem'}:
                raise ValueError('Release bundle contains runtime media, models or secret material')
            result[relative] = sha(path)
    if not result:
        raise ValueError('Empty release bundle')
    return result


def atomic_json(path, data):
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temp.open('x') as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def seal(root, commit, version):
    root = Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An immutable commit is required')
    if (root / MANIFEST).exists():
        raise ValueError('Release is already sealed; build a new version directory')
    content = files(root)
    if 'requirements.lock' not in content or 'Acceptance.json' not in content:
        raise ValueError('Bundle needs requirements.lock and a release acceptance receipt')
    acceptance = json.loads((root / 'Acceptance.json').read_text())
    if accepted_commit(acceptance) != commit or acceptance.get('release_ready') is not True:
        raise ValueError('Acceptance receipt must authorize this exact commit SHA')
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    data = {'version': version, 'commit': commit, 'source_digest': digest,
            'lock_digest': content['requirements.lock'], 'files': content}
    atomic_json(root / MANIFEST, data)
    return data


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    current = files(root)
    if current != manifest['files']:
        raise ValueError('Sealed release content changed')
    if hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest() != manifest['source_digest']:
        raise ValueError('Release content digest mismatch')
    acceptance = json.loads((root / 'Acceptance.json').read_text())
    if accepted_commit(acceptance) != manifest['commit'] or acceptance.get('release_ready') is not True:
        raise ValueError('Release acceptance is missing or belongs to another SHA')
    return manifest


def accepted_commit(receipt):
    """Read neutral identities and old bundles without assigning repo roles."""
    current, legacy = receipt.get('commit_sha'), receipt.get('development_sha')
    if current is not None and legacy is not None and current != legacy:
        raise ValueError('Acceptance receipt contains conflicting commit SHAs')
    return current if current is not None else legacy


def point(path, target):
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def verify_environment(base, manifest):
    root = Path(base) / 'Environments' / manifest['source_digest']
    if root.is_symlink():
        raise ValueError('Version environment cannot be redirected')
    receipt = root / 'Installation.json'
    if not receipt.is_file():
        raise ValueError('Install the verified release before activation')
    installation = json.loads(receipt.read_text())
    if (installation.get('source_digest') != manifest['source_digest']
            or installation.get('commit') != manifest['commit']
            or installation.get('dependency_lock_sha256') != manifest['lock_digest']):
        raise ValueError('Selected environment does not match its release')
    python = Path(installation['python'])
    if (python.parent not in {root / 'bin', root / 'Scripts'} or python.is_symlink()
            or not python.is_file() or not python.resolve().is_relative_to(root.resolve())):
        raise ValueError('Interpreter is missing or escapes its version environment')
    return python


def activate(base, release=None, *, rollback=False):
    with deployment_lock(base) as base:
        return _activate(base, release, rollback=rollback)


def _activate(base, release, *, rollback):
    versions = base / 'Releases'
    current, previous = base / 'Current', base / 'Previous'
    if rollback:
        if not previous.is_symlink():
            raise ValueError('No previous release is recorded')
        target = previous.resolve()
    else:
        if not release or Path(release).name != release:
            raise ValueError('Release must name one version directory')
        target = versions / release
    if not target.resolve().is_relative_to(versions) or target.resolve().parent != versions:
        raise ValueError('Release escapes versioned deployment root')
    manifest = verify(target)
    verify_environment(base, manifest)
    if current.exists() or current.is_symlink():
        if not current.is_symlink():
            raise ValueError('Current must be an owned release pointer')
        old = current.resolve()
        if old.parent != versions:
            raise ValueError('Current points outside versioned releases')
        if old == target.resolve():
            return manifest
        verify(old)
        point(previous, old)
    atomic_json(base / 'Activation.json', {'status': 'prepared', 'target': str(target), 'commit': manifest['commit']})
    point(current, target)
    atomic_json(base / 'Activation.json', {'status': 'selected', 'target': str(target), 'commit': manifest['commit'],
                                          'service_restarted': False, 'data_migrated': False})
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('seal')
    command.add_argument('root', type=Path)
    command.add_argument('--commit', required=True)
    command.add_argument('--version', required=True)
    for name in ('activate', 'rollback'):
        command = commands.add_parser(name)
        command.add_argument('base', type=Path)
        if name == 'activate':
            command.add_argument('release')
    args = parser.parse_args()
    result = seal(args.root, args.commit, args.version) if args.command == 'seal' else activate(
        args.base, getattr(args, 'release', None), rollback=args.command == 'rollback')
    print(json.dumps({key: result[key] for key in ('version', 'commit', 'source_digest')}, indent=2))

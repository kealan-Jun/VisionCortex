"""Retain each stage's metadata before later stages update shared indexes."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from uuid import uuid4

from .archive import write_json


def save_version(root: Path, artifacts: list[Path], receipt: dict) -> Path:
    target = root / 'JSON-Config-Files/Stage-Versions' / uuid4().hex[:16]
    target.mkdir(parents=True, exist_ok=False)
    paths = set()
    excluded = {'Stage-Versions', 'Stage-Receipts', 'Stage-Refreshes', 'Retry-Attempts', 'Recovery'}
    for artifact in artifacts:
        if artifact.is_dir():
            for parent, directories, files in os.walk(artifact, followlinks=False):
                directories[:] = [d for d in directories if d not in excluded and not (Path(parent) / d).is_symlink()]
                paths.update(Path(parent) / name for name in files if not name.endswith(('.partial', '.tmp', '.lock')))
        else:
            paths.add(artifact)
    for name in ('input_manifest.yaml', 'run_manifest.json', 'cache_identity.json'):
        source = root / 'JSON-Config-Files' / name
        if source.is_file():
            paths.add(source)
    entries = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        row = {'path': relative, 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        # No duplicate raw video or extra full-media read on the critical path.
        # Metadata is immutable; media references explicitly have a weaker scope.
        if path.suffix.lower() in {'.json', '.jsonl', '.yaml', '.yml', '.csv', '.txt'}:
            saved = target / f'{len(entries):06d}{path.suffix.lower()}'
            shutil.copyfile(path, saved)
            digest = hashlib.sha256()
            with saved.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            row.update(snapshot=saved.relative_to(root).as_posix(), sha256=digest.hexdigest(), scope='immutable_metadata')
        else:
            row['scope'] = 'media_path_and_stat_only'
        entries.append(row)
    manifest = target / 'index.json'
    write_json(manifest, {**receipt, 'schema_version': 'visioncortex-stage-version/1',
                         'formal_release': False, 'media_bodies_revalidated': False,
                         'files': entries})
    return manifest

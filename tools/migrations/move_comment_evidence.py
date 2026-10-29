"""Move generated Comment evidence to its configured persistent backend.

Run in the verified production storage view after compatible readers/writers
are deployed. No model execution, queue admission or raw-media cleanup occurs.
"""
import argparse
from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import re
import sqlite3

from visioncortex.comment_evidence import PREFIXES, backend_path
from visioncortex.comment_evidence_migration import inventory, migrate
from visioncortex.config import load_config
from visioncortex.device_day import exclusive
from visioncortex.device_day_contract import DeviceDayLayout, atomic_json, file_hash, read_json, safe_child, validate_archive_name
from visioncortex.device_day_recovery_worker import verify_storage
from visioncortex.device_day_timeline import day_bounds
from visioncortex.device_day_write_guard import storage_guard, verify_before_submission


def finish_cross_mount(layout, *, max_files, max_bytes):
    """Keep a verified rollback tree on the archive's own pinned mount.

    Separate bind mounts cannot rename across each other even on one NAS.
    Active evidence stays in the backend; its recovery copy is atomically
    retired into the archive's existing private publication directory.
    """
    completed = False
    for prefix in PREFIXES:
        target = backend_path(layout.root, prefix.rstrip('/'), layout.backend_root)
        journal = target.parent/'Migration.json'
        if not journal.is_file():
            continue
        if journal.stat().st_size > 128*1024**2:
            raise ValueError('Comment retirement journal exceeds its bound')
        value = read_json(journal)
        if value.get('status') != 'verified':
            continue
        if (value.get('archive') != layout.name or value.get('source') != prefix.rstrip('/')
                or value.get('backend') != target.relative_to(layout.backend_root).as_posix()):
            raise ValueError('Comment retirement journal crossed its owner')
        source = safe_child(layout.root, value['source'])
        if not source.exists() and not value.get('retirement_copy') and (target.parent/'RetiredEvidence').is_dir():
            continue  # The original same-mount coordinator owns this recovery.
        relative = 'ProcessedClips/.Evidence/CommentRetirements/'+prefix.split('/.Evidence')[0].replace('/', '')
        backup = safe_child(layout.root, relative)
        expected = {'storage_root': 'archive', 'path': relative}
        if value.get('retirement_copy', expected) != expected:
            raise ValueError('Comment retirement copy crossed its owner')
        if source.exists() == backup.exists():
            raise ValueError('Comment retirement has ambiguous source/recovery copies')
        original = source if source.exists() else backup
        if original.is_symlink() or not original.is_dir():
            raise ValueError('Comment retirement copy is not an owned directory')
        observed = inventory(original, max_files=max_files, max_bytes=max_bytes)
        references = value['files']
        if [name for name, _ in observed] != [ref['path'] for ref in references]:
            raise ValueError('Comment retirement file membership changed')
        for ref in references:
            verify_before_submission()
            for base in (original, target):
                path = safe_child(base, ref['path'])
                if path.stat().st_size != ref['size_bytes'] or file_hash(path) != ref['sha256']:
                    raise ValueError('Comment retirement bytes differ from verified copy')
        value['retirement_copy'] = expected
        atomic_json(journal, value)
        if source.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            verify_before_submission()
            os.rename(source, backup)
        value['status'] = 'completed'
        atomic_json(journal, value)
        completed = True
    return completed


def migrate_compatible(layout, *, max_files, max_bytes):
    finish_cross_mount(layout, max_files=max_files, max_bytes=max_bytes)
    for _ in PREFIXES:
        try:
            return migrate(layout, max_files=max_files, max_bytes=max_bytes,
                           checkpoint=verify_before_submission)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            if not finish_cross_mount(layout, max_files=max_files, max_bytes=max_bytes):
                raise
    return migrate(layout, max_files=max_files, max_bytes=max_bytes,
                   checkpoint=verify_before_submission)


def run(config, archive, *, apply=False, max_files=200000, max_bytes=32*1024**3):
    archive = validate_archive_name(archive)
    storage = config['storage']
    layout = DeviceDayLayout(Path(storage['archive_root']), archive[11:], day_bounds(archive[:10])[0],
                             Path(storage['local_cache_root']), read_only_originals=True)
    runtime = Path(storage['local_runtime_root'])/'device-day'
    with storage_guard(config), ExitStack() as locks:
        locks.enter_context(exclusive(runtime/'locks'/f'{archive}.index.lock'))
        ids = set()
        projection = runtime/'receipt-projection-v2.sqlite3'
        if projection.is_file():
            with sqlite3.connect(f'file:{projection}?mode=ro', uri=True) as db:
                ids.update(row[0] for row in db.execute('SELECT id FROM receipts WHERE day=?', (archive,)))
        for prefix in PREFIXES:
            recognition = layout.root/prefix/'Recognition'
            if recognition.is_dir():
                # Historical time labels are folders, not global recording IDs.
                # Actual IDs from the authoritative projection above retain
                # their locks even when a legacy ID happens to look like time.
                time_label = re.compile(r'^\d{2}-\d{2}-\d{2}(?:\.\d{3,6})?_\d{2}-\d{2}-\d{2}(?:\.\d{3,6})?$')
                ids.update(p.name for p in recognition.iterdir()
                           if p.is_dir() and not time_label.fullmatch(p.name))
        for rid in sorted(ids):
            if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', rid):
                raise ValueError('Invalid recording lock identity')
            for stage in ('all', 'retention', 'vision', 'stt', 'understanding', 'report'):
                locks.enter_context(exclusive(runtime/'locks'/f'{rid}.{stage}.lock'))
        if apply:
            return migrate_compatible(layout, max_files=max_files, max_bytes=max_bytes)
        return [{'source': prefix.rstrip('/'), 'files': len(inventory(
            layout.root/prefix, max_files=max_files, max_bytes=max_bytes))}
            for prefix in PREFIXES if (layout.root/prefix).is_dir()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--max-files', type=int, default=200000)
    parser.add_argument('--max-bytes', type=int, default=32*1024**3)
    args = parser.parse_args()
    config = load_config(args.config)
    verify_storage(config)
    result = run(config, args.archive, apply=args.apply, max_files=args.max_files, max_bytes=args.max_bytes)
    print(json.dumps({'archive': args.archive, 'applied': args.apply,
        'results': [{k: v for k, v in row.items() if k != 'files'} | {
            'file_count': len(row['files']) if isinstance(row['files'], list) else row['files']} for row in result]}))


if __name__ == '__main__':
    main()

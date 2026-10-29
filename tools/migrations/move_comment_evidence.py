"""Move generated Comment evidence to its configured persistent backend.

Run in the verified production storage view after compatible readers/writers
are deployed. No model execution, queue admission or raw-media cleanup occurs.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import sqlite3

from visioncortex.comment_evidence import PREFIXES
from visioncortex.comment_evidence_migration import inventory, migrate
from visioncortex.config import load_config
from visioncortex.device_day import exclusive
from visioncortex.device_day_contract import DeviceDayLayout, validate_archive_name
from visioncortex.device_day_recovery_worker import verify_storage
from visioncortex.device_day_timeline import day_bounds
from visioncortex.device_day_write_guard import storage_guard, verify_before_submission


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
                ids.update(p.name for p in recognition.iterdir() if p.is_dir())
        for rid in sorted(ids):
            if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', rid):
                raise ValueError('Invalid recording lock identity')
            for stage in ('all', 'retention', 'vision', 'stt', 'understanding', 'report'):
                locks.enter_context(exclusive(runtime/'locks'/f'{rid}.{stage}.lock'))
        if apply:
            return migrate(layout, max_files=max_files, max_bytes=max_bytes,
                           checkpoint=verify_before_submission)
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

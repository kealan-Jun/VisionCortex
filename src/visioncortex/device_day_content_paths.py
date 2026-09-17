"""Explicit legacy path aliases for the user-authorized content reorganization."""
import json
from functools import lru_cache


@lru_cache(maxsize=128)
def _read(path, modified, size):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle).get('aliases', {})


def aliases(root):
    path = root / 'Comment' / 'LegacyPaths.json'
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return _read(str(path), stat.st_mtime_ns, stat.st_size)


def relocated(relative, mapping):
    for old, new in mapping.items():
        if relative == old or relative.startswith(old + '/'):
            if not isinstance(new, str) or not new.startswith('Comment/') or new.startswith('Comment/Stt/'):
                raise ValueError('Invalid legacy content destination')
            return new + relative[len(old):]
    return relative


def public_references(value, mapping):
    if isinstance(value, list):
        return [public_references(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: public_references(item, mapping) for key, item in value.items()}
    if isinstance(value, str) and value.startswith('Comment/Stt/'):
        return relocated(value, mapping)
    return value


def migrate_stt(runner, layout):
    """Copy/verify, publish aliases, then retire the old visible directory.

    Run while the owned producer is drained. Retain all execution versions in
    the time folder and a backend migration journal; never rerun recognition.
    """
    from .device_day import copy_verified, exclusive
    from .device_day_content import time_folder
    from .device_day_contract import atomic_json, digest, read_json
    source = layout.comments / 'Stt'
    if not source.is_dir():
        return []
    moved = []
    with exclusive(runner.runtime_root/'locks'/f'{layout.name}.index.lock'):
        mapping = dict(aliases(layout.root))
        for folder in sorted(source.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                raise ValueError('Unexpected legacy STT material; retained unchanged')
            retained = read_json(layout.receipts/folder.name/'retention.json')
            record = retained['recording']
            if runner.layout(record).name != layout.name or record['recording_id'] != folder.name:
                raise ValueError('Legacy STT recording does not match its device/day')
            with exclusive(runner.runtime_root/'locks'/f'{folder.name}.stt.lock'):
                relative = 'Comment/' + time_folder(record['recording_start_us'], record['recording_end_us']) + '/Recognition'
                destination = layout.root / relative
                copied = []
                for path in sorted(folder.rglob('*')):
                    if path.is_symlink():
                        raise ValueError('Legacy STT symlink cannot be relocated')
                    if path.is_file():
                        target = destination/path.relative_to(folder)
                        copied.append({'path': target.relative_to(layout.root).as_posix(), **copy_verified(path, target)})
                old = 'Comment/Stt/' + folder.name
                mapping[old] = relative
                atomic_json(layout.comments/'LegacyPaths.json', {'schema_version': 'visioncortex-content-aliases/1',
                            'aliases': mapping})
                entry = {'recording_id': folder.name, 'from': old, 'to': relative, 'files': copied,
                         'original_media_changed': False, 'new_model_calls': 0}
                journal = layout.receipts/folder.name/'history'/('content-relocation-'+digest(entry))
                journal.mkdir(parents=True, exist_ok=True)
                backup = journal/'Stt'
                if backup.exists():
                    raise ValueError('Migration backup already exists; do not discard either copy')
                folder.rename(backup)
                atomic_json(journal/'Migration.json', entry)
                moved.append(entry)
        source.rmdir()  # Only remove the now-empty container, never its content.
    return moved

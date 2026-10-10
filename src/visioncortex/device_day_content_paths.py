"""Explicit legacy path aliases for the user-authorized content reorganization."""
import json
from functools import lru_cache


@lru_cache(maxsize=128)
def _read(path, modified, size):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle).get('aliases', {})


def aliases(root):
    result = {}
    for directory in ('Comment', 'MultimodalUnderstanding'):
        path = root / directory / 'LegacyPaths.json'
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        result.update(_read(str(path), stat.st_mtime_ns, stat.st_size))
    return result


def relocated(relative, mapping):
    for old, new in mapping.items():
        if relative == old or relative.startswith(old + '/'):
            prefix = old.split('/')[0] + '/'
            if (prefix not in ('Comment/', 'MultimodalUnderstanding/') or not isinstance(new, str)
                    or not new.startswith(prefix) or new.startswith(('Comment/Stt/', 'MultimodalUnderstanding/ClipUnderstanding/'))):
                raise ValueError('Invalid legacy content destination')
            return new + relative[len(old):]
    return relative


def public_references(value, mapping):
    if isinstance(value, list):
        return [public_references(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: public_references(item, mapping) for key, item in value.items()}
    if isinstance(value, str) and value.startswith(('Comment/Stt/', 'MultimodalUnderstanding/ClipUnderstanding/')):
        return relocated(value, mapping)
    return value


def analysis_folder(layout, segment, key):
    from .device_day_content import time_folder
    from .device_day_contract import safe_child
    relative = ('MultimodalUnderstanding/' + time_folder(segment['start_us'], segment['end_us'])
                + '/Analysis/' + segment['segment_id'] + '/' + key)
    return safe_child(layout.root, relative)


def migrate_understanding(runner, layout):
    """Relocate raw execution evidence without modifying its bytes or rerunning it."""
    from .device_day import copy_verified, exclusive
    from .device_day_contract import atomic_json, digest, read_json
    source = layout.understanding / 'ClipUnderstanding'
    if not source.is_dir():
        return []
    moved = []
    with exclusive(runner.runtime_root/'locks'/f'{layout.name}.index.lock'):
        folders = sorted(source.iterdir())
        if any(not p.is_dir() or p.is_symlink() for p in folders):
            raise ValueError('Unexpected legacy understanding material; retained unchanged')
        wanted = {p.name for p in folders}
        segments = {}
        # Current and superseded CV receipts are the authority for segment
        # times. Do not derive an absolute timestamp from a hash or file mtime.
        receipts = [*layout.receipts.glob('*/vision.json'), *layout.receipts.glob('*/history/vision-*.json')]
        for receipt in receipts:
            for segment in read_json(receipt).get('segments', []):
                sid = segment['segment_id']
                if sid not in wanted:
                    continue
                identity = tuple(segment[k] for k in ('recording_id', 'start_us', 'end_us'))
                if sid in segments and identity != segments[sid][0]:
                    raise ValueError('Ambiguous legacy segment timestamps; retained unchanged')
                segments[sid] = (identity, segment)
        if wanted - segments.keys():
            raise ValueError('Legacy segment has no verified capture-time identity')
        mapping = {k: v for k, v in aliases(layout.root).items() if k.startswith('MultimodalUnderstanding/')}
        for folder in folders:
            segment = segments[folder.name][1]
            rid = segment['recording_id']
            with exclusive(runner.runtime_root/'locks'/f'{rid}.understanding.lock'):
                destination = analysis_folder(layout, segment, 'placeholder').parent
                copied = []
                for path in sorted(folder.rglob('*')):
                    if path.is_symlink():
                        raise ValueError('Legacy understanding symlink cannot be relocated')
                    if path.is_file():
                        target = destination/path.relative_to(folder)
                        copied.append({'path': target.relative_to(layout.root).as_posix(), **copy_verified(path, target)})
                old, relative = layout.relative(folder), layout.relative(destination)
                mapping[old] = relative
                atomic_json(layout.understanding/'LegacyPaths.json', {'schema_version': 'visioncortex-content-aliases/1',
                            'aliases': mapping})
                entry = {'recording_id': rid, 'segment_id': folder.name, 'from': old, 'to': relative,
                         'files': copied, 'original_media_changed': False, 'new_model_calls': 0}
                journal = layout.receipts/rid/'history'/('content-relocation-'+digest(entry))
                journal.mkdir(parents=True, exist_ok=True)
                backup = journal/'ClipUnderstanding'
                if backup.exists():
                    raise ValueError('Migration backup already exists; do not discard either copy')
                folder.rename(backup)
                atomic_json(journal/'Migration.json', entry)
                moved.append(entry)
        source.rmdir()
    return moved


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

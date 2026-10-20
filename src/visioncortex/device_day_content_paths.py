"""Explicit legacy path aliases for the user-authorized content reorganization."""
import json
import hashlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re
from zoneinfo import ZoneInfo


LAYOUT_VERSION = 'visioncortex-frozen-archive-layout/1'
FROZEN_LAYOUT_SHA256 = '1eb20d4f5058058cbd13693180f8fcad26c1c71c8d414eb85d158f9e25cbf1ec'
_TIME_DIRECTORY = re.compile(r'^\d{2}-\d{2}-\d{2}(?:\.\d{6})?_\d{2}-\d{2}-\d{2}(?:\.\d{6})?$')


@lru_cache(maxsize=1)
def layout_contract():
    path = Path(__file__).with_name('templates') / 'VC-DEVICE-DAY-LAYOUT-V1.json'
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != FROZEN_LAYOUT_SHA256:
        raise ValueError('Frozen archive layout changed without an authorized contract update')
    return json.loads(content)


def clip_name(segment):
    """One visible name shared by CV media and its readable understanding."""
    activity = layout_contract()['activities'].get(segment.get('activity'))
    if not activity:
        raise ValueError('Unknown frozen archive activity')
    relative = segment.get('json_path')
    if relative:
        parts = PurePosixPath(relative).parts
        if (len(parts) != 4 or parts[:2] != ('ProcessedClips', 'Clips')
                or parts[3] != activity + '.json'
                or not re.fullmatch(r'\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_'
                                    + activity + r'_[A-Za-z0-9_-]{1,64}', parts[2])):
            raise ValueError('Clip index violates the frozen public layout')
        return parts[2]
    identifier = segment['segment_id'].removeprefix('segment-')[-8:]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier):
        raise ValueError('Invalid clip identity')
    def label(value):
        return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).strftime('%H-%M-%S')
    return f"{label(segment['start_us'])}_{label(segment['end_us'])}_{activity}_{identifier}"


def understanding_relative(segment):
    return layout_contract()['understanding_container'] + '/' + clip_name(segment)


def private_folder(layout, directory, kind, *parts):
    from .device_day_contract import safe_child
    if directory not in {'ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment'}:
        raise ValueError('Evidence has no owning archive directory')
    return safe_child(layout.root, '/'.join((directory, layout_contract()['private_evidence_directory'], kind, *parts)))


def legacy_reference(relative):
    parts = relative.split('/')
    return len(parts) > 1 and (
        relative in {'MultimodalUnderstanding/Understanding.json', 'MultimodalUnderstanding/UnderstandingReport.html'}
        or
        (parts[0] in {'Comment', 'MultimodalUnderstanding', 'ProcessedClips'} and bool(_TIME_DIRECTORY.fullmatch(parts[1])))
        or relative.startswith(('Comment/Stt/', 'MultimodalUnderstanding/ClipUnderstanding/',
                                'MultimodalUnderstanding/RequestPlans/', 'LaboratoryDailyReport/History/',
                                'LaboratoryDailyReport/Analysis/', 'ProcessedClips/.Evidence/LegacyPresentation/')))


@lru_cache(maxsize=128)
def _read(path, modified, size):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle).get('aliases', {})


def aliases(root):
    result = {}
    for directory in ('Comment', 'MultimodalUnderstanding', 'ProcessedClips', 'LaboratoryDailyReport'):
        for path in (root/directory/'LegacyPaths.json', root/directory/'.Evidence/LegacyPaths.json'):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            result.update(_read(str(path), stat.st_mtime_ns, stat.st_size))
    return result


def _publish_aliases(layout, directory, mapping):
    from .device_day import copy_verified
    from .device_day_contract import atomic_json, file_hash
    legacy = layout.root/directory/'LegacyPaths.json'
    atomic_json(private_folder(layout, directory, 'LegacyPaths.json'), {
        'schema_version': 'visioncortex-content-aliases/1', 'aliases': mapping})
    if legacy.is_file():
        checksum = file_hash(legacy)
        backup = private_folder(layout, directory, 'LegacyAliasCopies', checksum + '.json')
        copy_verified(legacy, backup)
        if file_hash(legacy) != checksum:
            raise ValueError('Legacy alias file changed during migration')
        # Preserve its original bytes before removing it from the reading area.
        legacy.unlink()


def relocated(relative, mapping):
    visited = set()
    for _ in range(32):
        if relative in visited:
            raise ValueError('Cyclic legacy content destination')
        visited.add(relative)
        matches = [old for old in mapping if relative == old or relative.startswith(old + '/')]
        if not matches:
            return relative
        old = max(matches, key=len)
        new = mapping[old]
        prefix = old.split('/')[0] + '/'
        if (prefix not in ('Comment/', 'MultimodalUnderstanding/', 'ProcessedClips/', 'LaboratoryDailyReport/')
                or not isinstance(new, str) or not new.startswith(prefix)
                or new.startswith(('Comment/Stt/', 'MultimodalUnderstanding/ClipUnderstanding/'))
                or any(part in {'', '.', '..'} for part in new.split('/')) or '\\' in new or ':' in new):
            raise ValueError('Invalid legacy content destination')
        relative = new + relative[len(old):]
    raise ValueError('Legacy content destination chain exceeds limit')


def public_references(value, mapping):
    if isinstance(value, list):
        return [public_references(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: public_references(item, mapping) for key, item in value.items()}
    if isinstance(value, str) and value.startswith(('Comment/', 'MultimodalUnderstanding/', 'ProcessedClips/', 'LaboratoryDailyReport/')):
        return relocated(value, mapping)
    return value


def analysis_folder(layout, segment, key):
    return private_folder(layout, 'MultimodalUnderstanding', 'Analysis', segment['segment_id'], key)


def publish_selected_videos(layout, segments, *, max_bytes=536870912, bounded_batch=False,
                            max_single_artifact_bytes=8589934592, progress=None, exact_intervals_only=False):
    """Expose already selected historical bytes in the frozen clip directory.

    No extraction or selection change. Old receipts retain exact bytes through
    an explicit alias; missing or damaged media cannot be called retained.
    The current exact-selection owner must opt in before passing canonical CV
    candidates: an unmarked historical video never authorizes a new copy.
    """
    from .device_day import copy_verified
    from .device_day_contract import artifact, safe_child
    from .device_day_verification import verify_artifact_cached
    pending = []
    for segment in segments:
        if segment.get('camera_key', layout.name[11:]) != layout.name[11:]:
            raise ValueError('Selected video belongs to another device/camera')
        # The exact-interval publisher keeps a preceding whole-candidate video
        # at its historical path. Its presence does not authorize another broad
        # copy, or turn that history into a newly selected exact interval.
        if segment.get('historical_scope'):
            continue
        parent = segment.get('candidate_segment_id')
        child = bool(parent and parent != segment.get('segment_id'))
        disposition = segment.get('derived_video_disposition') or {}
        policy = disposition.get('selection_policy')
        if exact_intervals_only:
            if (not child or segment.get('video_value') != 'selected'
                    or policy != 'visioncortex-exact-value-intervals/1'
                    or disposition.get('decision') != 'retain_derivative'):
                continue
            # Use the exact owner's complete receipt/video verifier, rather
            # than accepting a child ID or a historical retain decision alone.
            from .application.retained_video import published_material
            prior = published_material(layout, segment)
            if (not prior or prior.get('segment_id') != segment['segment_id']
                    or prior.get('candidate_segment_id') != parent
                    or prior.get('selection_policy') != policy
                    or prior.get('decision') != 'retain_derivative'
                    or prior.get('video') != segment.get('video')
                    or any(disposition.get(key) != prior.get(key) for key in
                           ('segment_id', 'candidate_segment_id', 'source_ref', 'source_start_ms',
                            'source_end_ms', 'start_us', 'end_us', 'selection', 'video'))
                    or any((prior.get('selection') or {}).get(key) != segment.get(key)
                           for key in ('start_ms', 'end_ms', 'start_us', 'end_us', 'activity', 'video_value'))):
                raise ValueError('Exact selected child lacks its matching verified publication receipt')
        if not child and ('retention_intervals' in segment
                          or policy == 'visioncortex-exact-value-intervals/1'):
            continue
        video = segment.get('video')
        if not video:
            continue
        target = 'ProcessedClips/Clips/' + clip_name(segment) + '/' + layout_contract()['activities'][segment['activity']] + '.mp4'
        if video['path'] != target and safe_child(layout.root, video['path']) != safe_child(layout.root, target):
            pending.append((video, target))
    if not bounded_batch and sum(video['size_bytes'] for video, _ in pending) > max_bytes:
        raise ValueError('Selected video publication exceeds its bounded batch')
    moved, used = [], 0
    if progress is not None:
        progress['pending'] = 0
    for position, (video, relative) in enumerate(pending):
        source = safe_child(layout.root, video['path'])
        if not source.is_file() or source.stat().st_size != video['size_bytes']:
            continue
        if bounded_batch and used + video['size_bytes'] > max_bytes and (
                used or video['size_bytes'] > max_single_artifact_bytes):
            if progress is not None:
                progress['pending'] = len(pending) - position
            break
        if not verify_artifact_cached(layout.root, video):
            continue
        source = safe_child(layout.root, video['path'])
        target = safe_child(layout.root, relative)
        copy_verified(source, target)
        reference = artifact(layout.root, target)
        if any(reference[key] != video[key] for key in ('size_bytes', 'sha256')):
            raise ValueError('Selected video changed during publication')
        mapping = {k: v for k, v in aliases(layout.root).items() if k.startswith('ProcessedClips/')}
        mapping[video['path']] = relative
        _publish_aliases(layout, 'ProcessedClips', mapping)
        moved.append(reference)
        used += video['size_bytes']
    return moved


def migrate_display_layout(runner, layout, *, max_bytes=536870912, max_files=20000,
                           bounded_batch=False, max_single_artifact_bytes=8589934592):
    """Copy/verify owned display internals before publishing legacy aliases.

    The caller drains producers and owns the day index lock. Originals are
    retained in hidden recoverable copies; capture directories are excluded.
    Preflight bounds apply to the entire batch before any mutation.
    """
    from .device_day import copy_verified
    from .device_day_contract import atomic_json, digest, file_hash
    plans = []
    # Obsolete day-level templates are neither model results nor part of the
    # frozen public tree. Preserve exact bytes and receipt aliases, using a
    # content identity so a recovered older producer cannot overwrite history.
    for name in ('Understanding.json', 'UnderstandingReport.html'):
        source = layout.understanding / name
        if source.is_file():
            if source.stat().st_size > max_single_artifact_bytes:
                raise ValueError('Legacy summary exceeds its bounded artifact size')
            plans.append((source, private_folder(layout, 'MultimodalUnderstanding',
                'LegacySummaries', file_hash(source), name)))
    for directory, names in {
            'MultimodalUnderstanding': ('RequestPlans', 'ClipUnderstanding'),
            'LaboratoryDailyReport': ('Analysis', 'History'),
            'Comment': ('Stt',)}.items():
        for name in names:
            source = layout.root / directory / name
            if source.is_dir():
                kind = {'ClipUnderstanding': 'Analysis', 'Stt': 'Recognition'}.get(name, name)
                plans.append((source, private_folder(layout, directory, kind)))
    for directory in ('ProcessedClips', 'MultimodalUnderstanding'):
        parent = layout.root / directory
        if parent.is_dir():
            for source in sorted(parent.iterdir()):
                if source.is_dir() and _TIME_DIRECTORY.fullmatch(source.name):
                    plans.append((source, private_folder(layout, directory, 'LegacyPresentation', source.name)))
    parent = layout.root / 'Comment'
    if parent.is_dir():
        for folder in sorted(parent.iterdir()):
            source = folder / 'Recognition'
            if _TIME_DIRECTORY.fullmatch(folder.name) and source.is_dir():
                plans.append((source, private_folder(layout, 'Comment', 'Recognition', folder.name)))
    def units(source, destination):
        if source.is_symlink():
            raise ValueError('Display migration refuses a symlink')
        files, amount = [], 0
        paths = (source,) if source.is_file() else source.rglob('*')
        for path in paths:
            if path.is_symlink():
                raise ValueError('Display migration refuses a symlink')
            if path.is_file():
                stat = path.stat()
                files.append((path, (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)))
                amount += stat.st_size
                if not bounded_batch and (amount > max_bytes or len(files) > max_files):
                    raise ValueError('Display migration exceeds its bounded batch')
                if bounded_batch and source.is_dir() and (amount > max_bytes or len(files) > max_files):
                    break
        else:
            files.sort(key=lambda item: item[0])
            yield source, destination, files, amount
            return
        # Oversized containers are split at their own child boundaries. Keep
        # prefix aliases for whole recording/window folders whenever possible.
        files.clear()
        for child in source.iterdir():
            yield from units(child, destination/child.name)

    total, count, batches, cleanup, pending = 0, 0, [], set(), False
    for original, destination in plans:
        for source, target, files, amount in units(original, destination):
            exceeds = total + amount > max_bytes or count + len(files) > max_files
            if exceeds:
                if not bounded_batch:
                    raise ValueError('Display migration exceeds its bounded batch')
                if count or not source.is_file() or amount > max_single_artifact_bytes:
                    pending = True
                    break
            total += amount
            count += len(files)
            batches.append((source, target, files))
            cleanup.update(p for p in source.parents if p == original or p.is_relative_to(original))
        if pending:
            break
    entries = []
    for source, destination, files in batches:
        old, new = layout.relative(source), layout.relative(destination)
        copied, single_file = [], source.is_file()
        for path, before in files:
            target = destination if single_file else destination / path.relative_to(source)
            reference = copy_verified(path, target)
            copied.append({'from': layout.relative(path), 'to': layout.relative(target), **reference})
            stat = path.stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != before:
                raise ValueError('Display source changed during migration')
        # Recheck the complete source set, including files copied first.
        if ([source] if single_file else [p for p in sorted(source.rglob('*')) if p.is_file()]) != [p for p, _ in files]:
            raise ValueError('Display source membership changed during migration')
        for (path, before), reference in zip(files, copied, strict=True):
            stat = path.stat()
            if ((stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != before
                    or file_hash(path) != reference['sha256']):
                raise ValueError('Display source changed before alias publication')
        directory = old.split('/')[0]
        mapping = {k: v for k, v in aliases(layout.root).items() if k.startswith(directory + '/')}
        mapping[old] = new
        _publish_aliases(layout, directory, mapping)
        backup = private_folder(layout, directory, 'LegacyCopies', digest([old, new])[:24])
        if backup.exists() and (not single_file or not backup.is_dir() or any(backup.iterdir())):
            raise ValueError('Display migration backup exists; preserve both copies')
        backup.parent.mkdir(parents=True, exist_ok=True)
        if single_file:
            backup.mkdir(exist_ok=True)
            source.rename(backup/source.name)
        else:
            source.rename(backup)
        entry = {'from': old, 'to': new, 'backup': layout.relative(backup), 'files': copied,
                 'original_media_changed': False, 'new_model_calls': 0}
        atomic_json(backup/'DisplayMigration.json', entry)
        entries.append(entry)
    for directory in sorted(cleanup, key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()):
            directory.rmdir()
    for directory in ('ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment'):
        if (layout.root/directory/'LegacyPaths.json').is_file():
            _publish_aliases(layout, directory, {k: v for k, v in aliases(layout.root).items() if k.startswith(directory + '/')})
    return {'layout_version': LAYOUT_VERSION, 'moved': entries, 'verified_bytes': total, 'files': count,
            'pending': pending}


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
                _publish_aliases(layout, 'MultimodalUnderstanding', mapping)
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
                relative = 'Comment/.Evidence/Recognition/' + folder.name
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
                _publish_aliases(layout, 'Comment', mapping)
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

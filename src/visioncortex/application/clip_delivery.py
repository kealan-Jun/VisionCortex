"""One public folder per original capture; selected intervals are sibling files."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
from zoneinfo import ZoneInfo

from ..device_day_contract import DeviceDayLayout, atomic_json, digest, read_json, safe_child
from ..device_day_content_paths import aliases, private_folder, public_references, relocated, _publish_aliases
from ..device_day_verification import verify_artifact_cached

VERSION = 'visioncortex-source-recording-delivery/2'
LABELS = {
    'ExperimentActivity': '实验片段（模型观察）',
    'ValuableNoExperimentActivity': '有价值的非实验片段',
    'NoValueNoExperimentActivity': '无价值的非实验记录',
    'ActivityPendingClassification': '活动片段，实验类型待判断',
    'PendingValueReview': '价值待判断记录',
}


def _metadata_files(folder):
    return [folder/(name+'.json') for name in LABELS] + [folder/'Clip.json', folder/'NoExperimentActivity.json']


def enabled(config):
    return config.get('device_day', {}).get('clip_delivery') == 'source_recording_v2'


def source_unit(reference):
    """Use the recorder's capture name, not a detector or export generation."""
    path = reference['path']
    parts = PurePosixPath(path).parts
    if (len(parts) < 2 or parts[0] not in {'MetaData', 'MetaVideo'}
            or any(part in {'', '.', '..'} for part in parts) or '\\' in path or ':' in path):
        raise ValueError('Clip source is not a recorder-owned original')
    key = '/'.join(parts[:2]) if len(parts) >= 3 else path
    name = parts[1] if len(parts) >= 3 else PurePosixPath(path).stem
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
        name = 'Capture_' + digest(key)[:16]
    return key, name


def _stamp(value):
    return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).strftime('%H-%M-%S')


def _display_category(kind):
    # Keep the established filename vocabulary. Unresolved classification and
    # value are explicit JSON states, never a confirmed no-value decision.
    return 'NoExperimentActivity' if kind in {'ActivityPendingClassification', 'PendingValueReview'} else kind


def _historical_camera_binding(layout, unit, meta, value):
    """Verify the established LabVideo alias against its paired source receipts.

    These old archives intentionally have a directory/view identity distinct
    from the physical recorder. A similar name or a foreign camera alone is
    never enough: both retained source descriptors must bind the same capture,
    and the original recorder metadata must still match its recorded bytes.
    """
    alias = re.fullmatch(r'lab-video-([13])', layout.name[11:])
    if not alias:
        return None
    from ..device_day_contract import artifact

    for record in unit['recordings']:
        videos = [s for s in record.get('sources', []) if s.get('kind') == 'video'
                  and s.get('retained', {}).get('path') == unit['observed_source']]
        for video in videos:
            if any(video.get(k) != video['retained'].get(k) for k in ('sha256', 'size_bytes')):
                continue
            original = PurePosixPath(video.get('original_path', ''))
            tail = original.parts[-4:]
            if (len(tail) != 4 or tail[:2] != ('lab_video', layout.name[:10] + '-' + alias[1])
                    or tail[3] != 'rgb.mp4' or not re.fullmatch(r'\d{6}', tail[2])):
                continue
            camera = value.get('camera_key')
            if (not isinstance(camera, str)
                    or value.get('recording_relative_path') != f'{camera}/{layout.name[:10]}/{tail[2]}'):
                continue
            for sidecar in record.get('sources', []):
                reference = sidecar.get('retained', {})
                if (sidecar.get('kind') != 'sidecar' or reference.get('path') != layout.relative(meta)
                        or PurePosixPath(sidecar.get('original_path', '')) != original.with_name('meta.json')
                        or any(sidecar.get(k) != reference.get(k) for k in ('sha256', 'size_bytes'))):
                    continue
                current = artifact(layout.root, meta)
                if any(current.get(k) != reference.get(k) for k in ('path', 'size_bytes', 'sha256')):
                    raise ValueError('Historical camera binding metadata checksum differs from its source receipt')
                return {'recipe': 'lab-video-retained-source-camera-binding/1',
                        'archive_camera_key': layout.name[11:], 'recorder_camera_key': camera,
                        'metadata': current, 'recording_id': record['recording_id']}
    return None


def _original_bounds(layout, unit):
    """Actual recorder bounds, never the nominal segment end or selected action."""
    path = layout.root/unit['observed_source']
    value = {}
    parts = PurePosixPath(unit['observed_source']).parts
    if len(parts) >= 3 or parts[0] == 'MetaVideo':
        if len(parts) == 2:
            meta = path.parent/'Metadata'/path.stem/'Video/Meta.json'
            if not meta.is_file():
                # Earlier archived captures retain their recorder prefix in
                # PascalCase. Their frame extent still owns the whole file;
                # a historical processing window can cover only part of it.
                candidates = list(meta.parent.glob('*Meta.json'))
                if len(candidates) > 1:
                    raise ValueError('Archived recorder time metadata is ambiguous')
                if candidates:
                    meta = candidates[0]
        else:
            meta = path.parent/'meta.json'
            if not meta.is_file():
                candidates = list(path.parent.glob('*_rgb.mp4'))
                if len(candidates) == 1:
                    candidate = candidates[0]
                    alternate = candidate.with_name(candidate.name[:-len('_rgb.mp4')]+'_meta.json')
                    if alternate.is_file():
                        meta = alternate
                        unit['observed_source'] = layout.relative(candidate)
        if meta.is_file() and not meta.is_symlink():
            before = meta.stat()
            if before.st_size > 1024*1024:
                raise ValueError('Recorder time metadata exceeds its bounded size')
            value = read_json(meta)
            after = meta.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Recorder time metadata changed during projection')
            if value.get('camera_key') and value['camera_key'] != layout.name[11:]:
                binding = _historical_camera_binding(layout, unit, meta, value)
                if binding is None:
                    raise ValueError('Recorder time metadata belongs to another device/camera')
                unit['recorder_identity_binding'] = binding
            unit['capture_session_id'] = value.get('recording_session_id')
            unit['capture_segment_index'] = value.get('global_segment_index')
            start = value.get('recording_window_first_valid_rgb_global_us')
            end = value.get('recording_window_last_valid_rgb_global_us')
            if type(start) is int and type(end) is int and 0 < start < end:
                return start, end, 'recorded_rgb_frame_extent'
            # recording_window_* can describe the whole recording session:
            # a later file may still carry the first file's session start.
            # Keep a quality-window fallback explicitly labelled; it does not
            # assert that every point in that window has a recorded RGB frame.
            for left, right in [('recording_quality_window_start_global_us', 'recording_quality_window_end_global_us')]:
                start, end = value.get(left), value.get(right)
                if type(start) is int and type(end) is int and 0 < start < end:
                    return start, end, 'capture_quality_window'
    records = unit['recordings']
    bounds = [(r.get('start_us'), r.get('end_us')) for r in records]
    if bounds and all(type(a) is int and type(b) is int and 0 < a < b for a, b in bounds):
        return min(a for a, _ in bounds), max(b for _, b in bounds), 'verified_recording_window'
    start, end = value.get('segment_start_us'), value.get('segment_end_us')
    if type(start) is int and type(end) is int and 0 < start < end:
        return start, end, 'recorder_segment_extent'
    return None, None, 'time_pending'


def original_units(layout, index, values=None):
    """Project source inventory; creating a status folder never admits a model.

    Immediate recorder directory names include slices still being finalized.
    This does not recurse, inspect media, modify originals or enqueue work.
    """
    units = {}
    for root_name in ('MetaData', 'MetaVideo'):
        raw = layout.root/root_name
        if not raw.is_dir() or raw.is_symlink():
            continue
        children = list(raw.iterdir())
        if len(children) > 20000:
            raise ValueError('Original source inventory exceeds its bounded device-day size')
        for folder in children:
            if folder.is_symlink() or folder.name.startswith('.'):
                continue
            capture = (folder.is_dir() and (root_name == 'MetaData' or (folder/'rgb.mp4').is_file()))
            if capture:
                reference = {'path': layout.relative(folder/'rgb.mp4')}
            elif folder.is_file() and folder.suffix.lower() == '.mp4':
                reference = {'path': layout.relative(folder)}
            else:
                continue
            key, name = source_unit(reference)
            units[key] = {'key': key, 'capture_name': name, 'observed_source': reference['path'], 'recordings': []}
    for record in index.get('recordings', []):
        for source in record.get('sources', []):
            if source.get('kind') != 'video' or not source.get('retained'):
                continue
            reference = source['retained']
            key, name = source_unit(reference)
            unit = units.setdefault(key, {'key': key, 'capture_name': name, 'observed_source': reference['path'], 'recordings': []})
            if record['recording_id'] not in {row['recording_id'] for row in unit['recordings']}:
                unit['recordings'].append(record)
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    for segment in index.get('segments', []):
        key, _ = source_unit(segment['source_ref'])
        record = records.get(segment['recording_id'])
        if key in units and record and record['recording_id'] not in {
                r['recording_id'] for r in units[key]['recordings']}:
            units[key]['recordings'].append(record)
    # The full original range names the collection. Its individual files keep
    # their selected ranges and categories; the collection is not one action.
    categories = {key: set() for key in units}
    for row in views(index) if values is None else values:
        key, _ = source_unit(row['segment']['source_ref'])
        categories.setdefault(key, set()).add(row['category'])
    for unit in units.values():
        kinds = categories[unit['key']]
        kind = next((v for v in ('ExperimentActivity', 'ValuableNoExperimentActivity',
                                 'ActivityPendingClassification', 'PendingValueReview') if v in kinds),
                    'NoValueNoExperimentActivity' if kinds else 'PendingValueReview')
        start, end, basis = _original_bounds(layout, unit)
        unit.update(start_us=start, end_us=end, time_basis=basis, category=kind)
        times = f'{_stamp(start)}_{_stamp(end)}' if start and end else 'TimePending_TimePending'
        unit['name'] = f"{times}_{_display_category(kind)}_{digest(['source-recording-display/1', unit['key']])[:8]}"
    return units


def category(candidate, meaning, record, interval):
    """CV activity alone cannot name an experimental clip."""
    from ..media.retained_intervals import observed_categories
    value = interval.get('video_value')
    if value == 'confirmed_no_value':
        return 'NoValueNoExperimentActivity'
    if value != 'selected':
        return 'PendingValueReview'
    found = observed_categories(candidate, meaning, record, interval['start_ms'], interval['end_ms'])
    if 'experiment' in found:
        return 'ExperimentActivity'
    if 'non_experiment' in found:
        # A complete audited process may still have an unprocessed tail.
        # Observed work in its beginning does not classify that unseen tail.
        if (interval.get('reason') != 'existing_fine_audited_activity_process'
                or ((meaning or {}).get('status') or (record.get('stages', {}).get('understanding') or {}).get('status'))
                   == 'completed'):
            return 'ValuableNoExperimentActivity'
    return 'ActivityPendingClassification'


def views(index):
    """No candidate/child duplicate and no policy hash in the display identity."""
    from ..device_day_retention_projection import interval_views
    records = {row['recording_id']: row for row in index.get('recordings', [])}
    candidates = {row['segment_id']: row for row in index.get('segments', [])}
    meanings = {row['segment_id']: row for row in index.get('understandings', [])}
    grouped = {}
    for segment, meaning, interval in interval_views(index):
        if interval is None and segment.get('retention_intervals'):
            continue
        candidate = candidates[segment.get('candidate_segment_id', segment['segment_id'])]
        record = records.get(segment['recording_id'])
        if not record:
            continue
        interval = interval or {
            **{key: segment[key] for key in ('start_ms', 'end_ms', 'start_us', 'end_us', 'activity')},
            'video_value': 'pending', 'reason': 'value_not_determined', 'activity_category': 'unknown',
        }
        kind = category(candidate, meanings.get(candidate['segment_id']), record, interval)
        times = [segment[key] for key in ('start_ms', 'end_ms', 'start_us', 'end_us')]
        times = [int(v) if v == int(v) else v for v in times]
        key = digest([VERSION, segment['recording_id'], segment['source_ref'], *times])
        def stamp(value):
            return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).strftime('%H-%M-%S')
        name = f"{stamp(segment['start_us'])}_{stamp(segment['end_us'])}_{_display_category(kind)}_{key[:8]}"
        value = {'key': key, 'name': name, 'category': kind, 'label': LABELS[kind],
                 'segment': segment, 'meaning': meaning, 'interval': interval, 'record': record,
                 'candidate': candidate, 'source_segment_ids': [segment['segment_id']]}
        previous = grouped.get(key)
        if previous:
            # Never let a pending or omitted duplicate hide a selected result.
            if previous['category'] != kind and 'selected' == interval['video_value'] == previous['interval']['video_value']:
                raise ValueError('Conflicting classifications for one source interval')
            identifiers = previous['source_segment_ids'] + value['source_segment_ids']
            if previous['interval']['video_value'] == 'selected' or interval['video_value'] != 'selected':
                value = previous
            value['source_segment_ids'] = sorted(set(identifiers))
        grouped[key] = value
    return sorted(grouped.values(), key=lambda row: (row['segment']['start_us'], row['key']))


def _mapping(layout, directory, additions):
    mapping = {key: value for key, value in aliases(layout.root).items() if key.startswith(directory + '/')}
    mapping.update(additions)
    _publish_aliases(layout, directory, mapping)


def _group_header(folder, filename, key=None):
    path = folder/filename
    if folder.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024**2:
        return None
    value = read_json(path)
    if value.get('schema_version') == VERSION and (key is None or value.get('source_unit') == key):
        return value
    return None


def _complete_group_rename(layout, intent, journal):
    """Rename an owned group once; replay after a crash without copying video."""
    from ..device_day_write_guard import verify_before_submission
    directory = intent['directory']
    prefix = 'ProcessedClips/Clips/' if directory == 'ProcessedClips' else 'MultimodalUnderstanding/'
    filename = 'Index.json' if directory == 'ProcessedClips' else 'Understanding.json'
    old, new = intent['from'], intent['to']
    for relative in (old, new):
        if not relative.startswith(prefix) or len(relative.split('/')) != len(prefix.split('/')):
            raise ValueError('Group rename escapes its owning public directory')
    parent = layout.root/prefix
    # Resolve physical names without applying a previous generation's alias.
    source, target = (safe_child(parent, PurePosixPath(p).name) for p in (old, new))
    if source.exists():
        if target.exists() or digest(_group_header(source, filename, intent['source_unit'])) != intent['header_digest']:
            raise ValueError('Original group rename source or destination changed')
        stat = source.stat()
        if [stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns] != intent['source_identity']:
            raise ValueError('Original group changed before rename')
        verify_before_submission()
        source.rename(target)
    if digest(_group_header(target, filename, intent['source_unit'])) != intent['header_digest']:
        raise ValueError('Renamed original group failed ownership verification')
    mapping = {k: v for k, v in aliases(layout.root).items() if k.startswith(directory+'/')}
    # Flatten old destinations and remove a returning category's directory
    # alias before binding the previous display name to the new physical name.
    flattened = {}
    for key, value in mapping.items():
        value = relocated(value, mapping)
        if value == old or value.startswith(old+'/'):
            value = new+value[len(old):]
        if key != new and key != value:
            flattened[key] = value
    flattened[old] = new
    _publish_aliases(layout, directory, flattened)
    atomic_json(journal, intent | {'status': 'completed'})


def rename_original_groups(layout, units):
    """Preserve the old readable naming style for both public source groups."""
    for directory, parent, filename in [
            ('ProcessedClips', layout.processed/'Clips', 'Index.json'),
            ('MultimodalUnderstanding', layout.understanding, 'Understanding.json')]:
        journals = private_folder(layout, directory, 'OriginalGroupRenames')
        for journal in journals.glob('*.json'):
            intent = read_json(journal)
            if intent.get('status') != 'completed':
                _complete_group_rename(layout, intent, journal)
        if not parent.is_dir():
            continue
        for source in list(parent.iterdir()):
            if source.name.startswith('.') or not source.is_dir():
                continue
            header = _group_header(source, filename)
            if not header or header.get('source_unit') not in units:
                continue
            unit = units[header['source_unit']]
            if source.name == unit['name']:
                continue
            stat = source.stat()
            intent = {'schema_version': VERSION, 'directory': directory, 'from': layout.relative(source),
                      'to': layout.relative(parent/unit['name']), 'source_unit': unit['key'],
                      'header_digest': digest(header),
                      'source_identity': [stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns],
                      'original_media_changed': False, 'model_invoked': False}
            journal = journals/(digest(intent)+'.json')
            atomic_json(journal, intent)
            _complete_group_rename(layout, intent, journal)


def _retire(layout, source):
    """Durable rename intent recovers a crash before alias publication."""
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    def directory_identity(path):
        stat = path.stat()
        return [stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns]
    old = layout.relative(source)
    directory = old.split('/')[0]
    parts = old.split('/')
    if not (directory == 'ProcessedClips' and parts[:2] == ['ProcessedClips', 'Clips'] and len(parts) == 3
            or directory == 'MultimodalUnderstanding' and len(parts) == 2):
        raise ValueError('Only owned public derived clip folders may be relocated')
    if not source.exists():
        journal_root = private_folder(layout, directory, 'ClipRelocations')
        for journal in journal_root.glob('*.json'):
            intent = read_json(journal)
            target = safe_child(layout.root, intent['to'])
            if target.is_dir() and intent.get('from') == old and intent.get('status') != 'completed':
                _mapping(layout, directory, {old: intent['to']})
                atomic_json(journal, intent | {'status': 'completed'})
        return False
    before = directory_identity(source)
    target = private_folder(layout, directory, 'LegacyPresentation', source.name)
    # Mutable public views can return to an earlier category. Preserve each
    # actual directory generation rather than overwriting its earlier history.
    presented = any(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4*1024**2
                    and read_json(path).get('clip_delivery_version') == VERSION for path in _metadata_files(source))
    if presented or (source/'Understanding.json').is_file():
        target = target/digest([old, before])[:24]
    journal = private_folder(layout, directory, 'ClipRelocations', digest([old, layout.relative(target)]) + '.json')
    if source.is_symlink() or not source.is_dir() or target.exists():
        raise ValueError('Clip relocation source or destination is ambiguous')
    intent = {'schema_version': VERSION, 'from': old, 'to': layout.relative(target),
              'source_identity': before, 'operation': 'preserve_owned_derived_directory',
              'original_media_changed': False, 'model_invoked': False}
    atomic_json(journal, intent)
    guarded_mkdir(target.parent)
    verify_before_submission()
    if directory_identity(source) != before:
        raise ValueError('Clip directory changed before relocation')
    source.rename(target)
    _mapping(layout, directory, {old: layout.relative(target)})
    atomic_json(journal, intent | {'status': 'completed'})
    return True


def retire_native(layout, index, public_names, *, source_units=(), maximum=64):
    """Retire only source-matching producer/display folders, never raw media."""
    from ..device_day_content_paths import clip_name
    from ..device_day_retention_projection import interval_views
    expected = {}
    fields = ('recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us')
    def signature(value):
        normalized = {key: value.get(key) for key in fields}
        for key in ('start_ms', 'end_ms', 'start_us', 'end_us'):
            number = normalized[key]
            if type(number) in (int, float) and number == int(number):
                normalized[key] = int(number)
        return digest(normalized)
    for segment, _, _ in interval_views(index):
        expected[clip_name(segment)] = segment
    # Old generations can have different IDs but the exact same source range.
    signatures = {signature(value) for value in expected.values()}
    owners = {digest([value['recording_id'], value['source_ref']]) for value in expected.values()}
    def owned_source(value, field='source_ref'):
        reference = value.get(field) or {}
        try:
            return (bool(value.get('recording_id')) and len(reference.get('sha256', '')) == 64
                    and source_unit(reference)[0] in source_units)
        except (ValueError, KeyError, TypeError):
            return False
    count, pending = 0, False
    for directory, parent in [('ProcessedClips', layout.processed/'Clips'),
                              ('MultimodalUnderstanding', layout.understanding)]:
        if not parent.is_dir():
            continue
        for source in sorted(parent.iterdir(), reverse=True):
            if source.name in public_names or source.name.startswith('.') or not source.is_dir():
                continue
            owned = False
            if not owned:
                documents = _metadata_files(source)
                for document in documents:
                    if not document.is_symlink() and document.is_file() and document.stat().st_size <= 4*1024**2:
                        value = read_json(document)
                        if (signature(value) in signatures
                                or digest([value.get('recording_id'), value.get('source_ref')]) in owners
                                or owned_source(value)):
                            owned = True
                            break
                if directory == 'MultimodalUnderstanding':
                    document = source/'Understanding.json'
                    if not document.is_symlink() and document.is_file() and document.stat().st_size <= 16*1024**2:
                        value = read_json(document)
                        source_fields = {key: value.get(key) for key in fields}
                        source_fields.update(source_ref=value.get('source_video'), start_ms=value.get('source_start_ms'),
                                             end_ms=value.get('source_end_ms'))
                        owned = (signature(source_fields) in signatures or
                                 digest([value.get('recording_id'), value.get('source_video')]) in owners
                                 or owned_source(value, 'source_video'))
                elif not owned and source.name in expected:
                    # A newly materialized native video can precede its first
                    # readable sidecar. Its exact publication receipt and
                    # actual physical bytes, never its name alone, own it.
                    from .retained_video import published_material
                    segment = expected[source.name]
                    saved = published_material(layout, segment)
                    if saved:
                        name = 'ExperimentActivity.mp4' if segment['activity'] == 'active' else 'NoExperimentActivity.mp4'
                        reference = saved['video'] | {'path': name}
                        owned = verify_artifact_cached(source, reference)
            if not owned:
                continue
            if count >= maximum:
                pending = True
                continue
            count += _retire(layout, source)
    # Replay prepared directory moves even if the old directory disappeared.
    for directory in ('ProcessedClips', 'MultimodalUnderstanding'):
        journal_root = private_folder(layout, directory, 'ClipRelocations')
        if journal_root.is_dir():
            for journal in journal_root.glob('*.json'):
                intent = read_json(journal)
                if intent.get('status') != 'completed':
                    _retire(layout, layout.root/intent['from'])
    return {'retired': count, 'pending': pending}


def publish_asset(layout, reference, name, *, filename='Video.mp4'):
    """Publish exact verified bytes; only owned redundant derivative bytes retire."""
    from ..device_day import copy_verified
    from ..device_day_inputs import identity
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    if (not reference or not reference['path'].startswith(('ProcessedClips/Clips/',
                                                          'ProcessedClips/.Evidence/LegacyPresentation/',
                                                          'ProcessedClips/.Evidence/NativeClips/'))
            or not verify_artifact_cached(layout.root, reference)):
        return None
    source = safe_child(layout.root, reference['path'])
    parent = safe_child(layout.root, 'ProcessedClips/Clips')
    target = safe_child(parent, name+'/'+filename)  # validate physical destination, without legacy aliases
    if source == target:
        return reference | {'path': layout.relative(target)}
    before = identity(source)
    safe_child(layout.root, layout.relative(target))
    guarded_mkdir(target.parent)
    verify_before_submission()
    copied = copy_verified(source, target)
    if copied['sha256'] != reference['sha256'] or copied['size_bytes'] != reference['size_bytes']:
        raise ValueError('Public video differs from its verified selected derivative')
    result = {'path': layout.relative(target), 'sha256': copied['sha256'], 'size_bytes': copied['size_bytes']}
    if identity(source) != before:
        raise ValueError('Selected derivative changed during public delivery')
    mapping = {key: value for key, value in aliases(layout.root).items() if key.startswith('ProcessedClips/')}
    source_relative = layout.relative(source)
    for key in list(mapping):
        if relocated(key, mapping) == source_relative:
            mapping[key] = result['path']
    mapping.update({reference['path']: result['path'], source_relative: result['path']})
    mapping.pop(result['path'], None)
    _publish_aliases(layout, 'ProcessedClips', mapping)
    if not verify_artifact_cached(layout.root, result):
        raise ValueError('Public video failed independent artifact verification')
    if source_relative.startswith(('ProcessedClips/.Evidence/LegacyPresentation/', 'ProcessedClips/.Evidence/NativeClips/')):
        # The verified replacement and aliases are durable before reclaiming
        # this owned derived copy. Source video/audio/CSV never enter this path.
        verify_before_submission()
        if identity(source) != before:
            raise ValueError('Historical derived copy changed before reclaim')
        source.unlink()
    return result


def _retire_file(layout, relative, reference=None):
    """Preserve an obsolete owned interval file before retiring its public name."""
    from ..device_day_contract import artifact
    from ..device_day_inputs import identity
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    parts = relative.split('/')
    if not (parts[:2] == ['ProcessedClips', 'Clips'] and len(parts) == 4
            or parts[0] == 'MultimodalUnderstanding' and len(parts) == 3):
        raise ValueError('Interval file is outside its owned original-capture folder')
    parent = safe_child(layout.root, parts[0])
    source = safe_child(parent, '/'.join(parts[1:]))
    if not source.exists():
        return
    if source.is_symlink() or not source.is_file() or source.suffix not in {'.json', '.mp4'}:
        raise ValueError('Obsolete interval file is not regular derived data')
    before = identity(source)
    physical = artifact(source.parent, source)
    if reference and any(reference.get(key) != physical[key] for key in ('sha256', 'size_bytes')):
        raise ValueError('Obsolete video differs from its last published binding')
    # An earlier publication can alias a reclaimed historical video back to
    # this public file. A write destination must be physical, never resolved
    # through that reader alias (which could otherwise turn retirement into
    # source == destination followed by unlink).
    hidden = '/'.join(('.Evidence', 'LegacyPresentation', parts[-2], physical['sha256'], source.name))
    target = safe_child(parent, hidden)
    intent = {'schema_version': VERSION, 'from': relative, 'to': layout.relative(target),
              'source_identity': before, 'file': physical, 'original_media_changed': False, 'model_invoked': False}
    journal = private_folder(layout, parts[0], 'ClipFileRelocations', digest(intent)+'.json')
    atomic_json(journal, intent)
    guarded_mkdir(target.parent)
    verify_before_submission()
    if identity(source) != before:
        raise ValueError('Interval file changed before preserving its old name')
    duplicate = target.exists()
    if duplicate:
        if not verify_artifact_cached(target.parent, physical):
            raise ValueError('Interval history destination collision')
    else:
        source.rename(target)
    mapping = {k: v for k, v in aliases(layout.root).items() if k.startswith(parts[0]+'/')}
    mapping.pop(layout.relative(target), None)
    if relocated(layout.relative(target), mapping) != layout.relative(target):
        raise ValueError('Physical interval history is shadowed by a directory alias')
    previous = {k: v for k, v in aliases(layout.root).items() if k.startswith(parts[0]+'/')}
    if mapping != previous:
        _publish_aliases(layout, parts[0], mapping)
    _mapping(layout, parts[0], {relative: layout.relative(target)})
    if duplicate:
        verify_before_submission()
        if identity(source) != before or not verify_artifact_cached(target.parent, physical):
            raise ValueError('Interval history changed before duplicate reclaim')
        source.unlink()
    atomic_json(journal, intent | {'status': 'completed'})


def _recover_files(layout):
    for directory in ('ProcessedClips', 'MultimodalUnderstanding'):
        for journal in private_folder(layout, directory, 'ClipFileRelocations').glob('*.json'):
            intent = read_json(journal)
            if intent.get('status') == 'completed':
                continue
            if not intent['from'].startswith(directory+'/'):
                raise ValueError('Interval history intent differs from its owning evidence directory')
            target = safe_child(layout.root, intent['to'])
            if not intent['to'].startswith(directory+'/.Evidence/LegacyPresentation/'):
                raise ValueError('Interval history intent escapes its owning evidence directory')
            parent = safe_child(layout.root, directory)
            source = safe_child(parent, intent['from'].removeprefix(directory+'/'))
            if source.exists():
                _retire_file(layout, intent['from'], intent['file'])
                atomic_json(journal, intent | {'status': 'completed'})
            elif verify_artifact_cached(target.parent, intent['file']):
                _mapping(layout, directory, {intent['from']: intent['to']})
                atomic_json(journal, intent | {'status': 'completed'})
            else:
                raise ValueError('Prepared interval history has neither verified source nor destination')


def _retire_obsolete(layout, unit, rows):
    index = layout.processed/'Clips'/unit['name']/'Index.json'
    if not index.is_file():
        return
    previous = read_json(index)
    if previous.get('schema_version') != VERSION or previous.get('source_unit') != unit['key']:
        raise ValueError('Original-capture output folder belongs to another source or contract')
    expected = {row['name']+'.json' for row in rows} | {
        row['name']+'.mp4' for row in rows if row['interval']['video_value'] == 'selected'}
    for row in previous.get('intervals', []):
        for field in ('metadata', 'content', 'video'):
            reference = row.get(field)
            path = reference.get('path') if isinstance(reference, dict) else reference
            if path:
                path = relocated(path, aliases(layout.root))
            if not path or PurePosixPath(path).name in expected:
                continue
            directory = 'MultimodalUnderstanding' if field == 'content' else 'ProcessedClips'
            if path.startswith(directory+'/.Evidence/'):
                # A prior durable relocation already retired this reference.
                # Validate containment; never attempt to retire its hidden copy
                # again or treat it as a public file in the renamed group.
                safe_child(layout.root, path)
                continue
            prefix = ('ProcessedClips/Clips/' if field != 'content' else 'MultimodalUnderstanding/')+unit['name']+'/'
            if not path.startswith(prefix) or len(PurePosixPath(path).parts) != len(PurePosixPath(prefix).parts)+1:
                raise ValueError('Previous interval link escapes its original-capture folder')
            _retire_file(layout, path, reference if field == 'video' else None)


def publish(config, index, audio_entries, *, document_builder, writer, entry_builder,
            maximum_new_videos=4, maximum_video_bytes=512*1024**2, source_views=None,
            source_units=None, maximum_retired=64):
    """One original -> one directory; every selected interval is a file inside it."""
    from ..device_day_timeline import day_bounds
    from ..runtime_control import check_yield
    root = Path(config['storage']['archive_root'])
    backend = config['storage'].get('local_cache_root')
    start, _ = day_bounds(index['archive'][:10])
    layout = DeviceDayLayout(root, index['archive'][11:], max(start, 1), Path(backend) if backend else None)
    # A format-only historical migration supplies existing, source-bound
    # materializations. Normal producers retain the same selection owner.
    values = views(index) if source_views is None else source_views
    units = original_units(layout, index, values) if source_units is None else source_units
    grouped = {key: [] for key in units}
    for row in values:
        key, _ = source_unit(row['segment']['source_ref'])
        if key not in units:
            raise ValueError('Selected interval has no owning original-capture unit')
        row['unit'] = units[key]
        grouped[key].append(row)
    _recover_files(layout)
    rename_original_groups(layout, units)
    progress = retire_native(layout, index, {unit['name'] for unit in units.values()},
                             source_units=set(units), maximum=maximum_retired)
    for key, unit in units.items():
        _retire_obsolete(layout, unit, grouped[key])
    entries, copied, copied_bytes = [], 0, 0
    details = {key: [] for key in units}
    work = sorted(values, key=lambda row: row['segment']['start_us'], reverse=True) if (
        config.get('device_day', {}).get('latest_first')) else values
    for row in work:
        check_yield(checkpoint=False)
        unit = row['unit']
        segment, interval = deepcopy(row['segment']), deepcopy(row['interval'])
        relative = 'ProcessedClips/Clips/'+unit['name']+'/'+row['name']+'.json'
        content = 'MultimodalUnderstanding/'+unit['name']+'/'+row['name']+'.json'
        target = layout.root/relative
        if target.is_file() and read_json(target).get('source_interval_key') != row['key']:
            raise ValueError('Interval filename collides with another source interval')
        selected, native_video = interval['video_value'] == 'selected', segment.get('video')
        existing = layout.processed/'Clips'/unit['name']/(row['name']+'.mp4')
        new_asset = selected and native_video and not existing.is_file()
        amount = (native_video or {}).get('size_bytes', 0)
        deferred = new_asset and (copied >= maximum_new_videos
            or copied and copied_bytes+amount > maximum_video_bytes)
        if new_asset and amount > 8*1024**3:
            raise ValueError('Selected video exceeds the bounded single-artifact delivery size')
        video = publish_asset(layout, native_video, unit['name'], filename=row['name']+'.mp4') if selected and not deferred else None
        if selected and not video:
            progress['pending'] = True
            if native_video and not deferred:
                interval.update(processing_status='failed', error={
                    'reason': 'selected_native_video_missing_or_unverified',
                    'automatic_action': 'preserve_native_receipt_and_retry_verified_export'})
        if new_asset and video:
            copied += 1
            copied_bytes += amount
        segment.update(json_path=relative, video=video, display_category=row['category'], display_label=row['label'])
        interval.update(video=video)
        document, retention = document_builder(index, layout.root, segment, row['meaning'], interval,
                                               row['record'], audio_entries)
        display = {'clip_delivery_version': VERSION, 'source_unit': unit['key'], 'source_folder': unit['name'],
                   'source_interval_key': row['key'], 'display_category': row['category'], 'display_label': row['label'],
                   'source_segment_ids': row['source_segment_ids'], 'cv_activity': segment['activity'],
                   'classification_is_physical_ground_truth': False}
        document.update(display)
        retention.update(display)
        document['内容记录']['片段类型'] = row['label']
        # A category may return to an earlier filename. Public sidecars are
        # mutable views; immutable producer evidence remains in its own aliases.
        for directory, planned in [('ProcessedClips', {relative}), ('MultimodalUnderstanding', {content})]:
            previous = {key: value for key, value in aliases(layout.root).items() if key.startswith(directory+'/')}
            mapping = {key: value for key, value in previous.items() if key not in planned}
            if previous != mapping:
                _publish_aliases(layout, directory, mapping)
        mapping = aliases(layout.root)
        writer(layout.root/content, public_references(document, mapping))
        writer(target, public_references(segment | display | {'selection_interval': interval,
                                                              'retention': retention, 'model_invoked': False}, mapping))
        native_metadata = row['segment'].get('json_path')
        if (native_metadata and native_metadata != relative
                and native_metadata.startswith(('ProcessedClips/Clips/', 'ProcessedClips/.Evidence/NativeClips/'))
                and not safe_child(layout.root, native_metadata).exists()):
            # New exact children have a logical metadata address before the
            # readable view exists. Bind that missing address only after its
            # actual file is durable; never replace an immutable native JSON.
            _mapping(layout, 'ProcessedClips', {native_metadata: relative})
        entry = entry_builder(document, 'MultimodalUnderstanding/'+unit['name'])
        entry.update(display, content=content, metadata=relative, video=video,
                     source_start_ms=segment['start_ms'], source_end_ms=segment['end_ms'])
        entries.append(entry)
        details[unit['key']].append({'entry': entry, 'retention': retention, 'narrative': document['内容记录']})
    for key, unit in units.items():
        rows = sorted(details[key], key=lambda value: (value['entry']['start_us'], value['entry']['segment_id']))
        header = {'schema_version': VERSION, 'archive': index['archive'], 'source_unit': key,
            'source_folder': unit['name'], 'original_source': {'path': unit['observed_source'], 'ownership': 'recorder'},
            'original_start_us': unit['start_us'], 'original_end_us': unit['end_us'],
            'original_time_basis': unit['time_basis'],
            'original_range_is_continuous_coverage': False,
            'capture_session_id': unit.get('capture_session_id'),
            'capture_segment_index': unit.get('capture_segment_index'),
            'recorder_identity_binding': unit.get('recorder_identity_binding'),
            'processing_windows': [{'recording_id': r['recording_id'], 'start_us': r.get('start_us'),
                                    'end_us': r.get('end_us')} for r in unit['recordings']],
            'capture_name': unit['capture_name'], 'display_category': unit['category'],
            'folder_scope': 'complete_original_recording_collection',
            'folder_category_is_whole_recording_classification': False,
            'recording_ids': [row['recording_id'] for row in unit['recordings']],
            'stages': {row['recording_id']: row.get('stages', {}) for row in unit['recordings']},
            'processing_status': 'available_partial_projection' if rows else 'waiting_for_source_finalization_or_processing',
            'interval_count': len(rows), 'available_video_count': sum(bool(row['entry']['video']) for row in rows),
            'physical_action_confirmed': False, 'model_invoked_by_publication': False,
            'original_media_changed': False, 'intervals': [row['entry'] for row in rows]}
        writer(layout.processed/'Clips'/unit['name']/'Index.json', header)
        writer(layout.processed/'Clips'/unit['name']/'Retention.json', header | {
            'intervals': [row['retention'] for row in rows]})
        writer(layout.understanding/unit['name']/'Understanding.json', header | {
            '片段记录': [{'开始时间': row['entry']['开始时间'], '结束时间': row['entry']['结束时间'],
                '类别': row['entry']['display_label'], '视频': row['entry']['video'],
                '内容记录': row['narrative'], '详细记录': row['entry']['content']} for row in rows]})
    return sorted(entries, key=lambda row: (row['start_us'], row['segment_id'])), progress | {
        'original_folder_count': len(units), 'videos_published': copied, 'video_bytes_published': copied_bytes}

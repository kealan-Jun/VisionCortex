"""Read existing FieldRecognition evidence without changing its live service."""
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from threading import RLock
from zoneinfo import ZoneInfo

from ..device_day_contract import atomic_json, digest, read_json, safe_child, validate_archive_name

_CACHE = OrderedDict()
_CACHE_BYTES = 32 * 1024 * 1024
_CACHE_LOCK = RLock()


def snapshot(path):
    path = Path(path)
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if before.st_size > 8 * 1024 * 1024 or path.is_symlink():
        raise ValueError('Field archive document is not bounded regular evidence')
    with _CACHE_LOCK:
        cached = _CACHE.get(str(path))
        if cached and cached[0] == identity:
            _CACHE.move_to_end(str(path))
            return cached[1], cached[2]
    raw = path.read_bytes()
    after = path.stat()
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Field archive changed during snapshot')
    value = json.loads(raw)
    reference = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    with _CACHE_LOCK:
        _CACHE[str(path)] = (identity, value, reference)
        while sum(item[0][2] for item in _CACHE.values()) > _CACHE_BYTES:
            _CACHE.popitem(last=False)
    return value, reference


def instant(value):
    if not value:
        return None
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Field archive time requires a timezone')
    return round(stamp.timestamp() * 1e6)


def _owned_source(root, relative, camera, day):
    """Reject an explicit foreign device/day owner before reading its bytes."""
    path = safe_child(root, relative)
    for component in PurePosixPath(relative).parts[:-1]:
        try:
            owner = validate_archive_name(component)
        except ValueError:
            continue  # Older unscoped relative paths retain their readers.
        if owner != f'{day.isoformat()}_{camera}':
            raise ValueError('Field source path crossed its camera/day archive')
    return path


def _measurement_identity(result, entry, item, data, camera):
    if not isinstance(result, dict):
        raise ValueError('Field measurement result must be an object')
    if any(value.get('camera_id', camera) != camera for value in (result, entry, item, data)):
        raise ValueError('Field measurement result crossed its camera archive')
    identities = [value.get('instrument_id') for value in (result, entry, item, data)
                  if value.get('instrument_id') is not None]
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError('Field measurement instrument identity must be an explicit nonempty string')
    if identities and any(value != identities[0] for value in identities):
        raise ValueError('Field measurement result crossed its instrument identity')
    return identities[0] if identities else None


def archive_days(config):
    """Existing file publisher discovers bounded archive metadata, never media."""
    settings = config.get('device_day', {}).get('field_recognition') or {}
    if not settings.get('archive_context_enabled'):
        return {}
    root = Path(settings['archive_root'])
    if not root.is_dir():
        raise FileNotFoundError('FieldRecognition archive root is unavailable')
    zone = ZoneInfo(config.get('device_day', {}).get('timezone', 'Asia/Shanghai'))
    result = {}
    for number, folder in enumerate(root.iterdir()):
        if number >= 4096:
            raise ValueError('Field archive discovery exceeds bounded camera-day metadata')
        if not folder.is_dir() or folder.is_symlink() or not (folder/'DailyReport').is_dir():
            continue
        try:
            name = validate_archive_name(folder.name)
            start = datetime.fromisoformat(name[:10]).replace(tzinfo=zone)
        except ValueError:
            continue
        result[name] = round(start.timestamp() * 1e6)
    return result


def day_context(config, archive):
    name = validate_archive_name(archive)
    from ..device_day_timeline import day_bounds
    start, end = day_bounds(name[:10])
    return context_events(config, {'camera_key': name[11:], 'recording_start_us': start, 'recording_end_us': end})


def archive_revision(config, archive):
    """Late result/decision/binding edits invalidate only this camera-day view."""
    settings = config.get('device_day', {}).get('field_recognition') or {}
    return digest(day_context(config, archive)) if settings.get('archive_context_enabled') else None


def publish_context(config, index):
    """Publish exact producer facts and retain superseded local projections."""
    context = index.get('context') or {}
    if not (config.get('device_day', {}).get('field_recognition') or {}).get('archive_context_enabled'):
        return None
    name = validate_archive_name(index['archive'])
    value = {'schema_version': 'visioncortex-field-archive-context/1', 'archive': name,
             'source': 'field_recognition', 'model_invoked': False, 'physical_action_verified': False,
             'status': context.get('field_event_status', 'not_received'),
             'events': context.get('events', []), 'errors': context.get('field_event_errors', []),
             'source_references': context.get('field_source_references', []),
             'association_basis': 'same_camera_actual_interval_overlap_not_action_confirmation'}
    root = safe_child(Path(config['storage']['archive_root']), name + '/Comment')
    revision = digest(value)
    from ..comment_evidence import backend_path
    saved = backend_path(root.parent, 'Comment/.Evidence/FieldRecognition/'+revision+'.json',
                         Path(config['storage']['local_cache_root']))
    if saved.is_file():
        if digest(read_json(saved)) != revision:
            raise ValueError('Saved FieldRecognition context evidence identity changed')
    else:
        atomic_json(saved, value)
    from ..comment_readability import field_context_document
    atomic_json(root/'OCR/FieldEvents.json', field_context_document(value))
    return {'path': 'Comment/OCR/FieldEvents.json', 'revision': revision}


def _reading_payload(data, item, result=None):
    payload = {key: data[key] for key in ('record_scope', 'status', 'decision', 'test_only', 'withdrawn', 'unbound_readings',
        'reading_observations', 'processing_terminal', 'processing_outcome', 'processing_status',
        'skip_reasons', 'aggregation_issues') if key in data}
    payload['status'] = data.get('status', item.get('status'))
    payload['daily_index_status'] = item.get('status')
    payload['record_scope'] = data.get('record_scope', 'unknown')
    payload['normalized_values'] = result.get('values', []) if result is not None else []
    return payload


def _unbound_event(root, camera, day, start, end, item, data, refs):
    """An unbound/failed OCR event still exists without a Result.json or video."""
    left = instant(item.get('event_at') or data.get('event_at'))
    if left is None or not start <= left < end:
        return []
    photo = _owned_source(root, item['photo'], camera, day) if item.get('photo') else None
    return [{'type': 'ocr_observation', 'event_id': item['event_id'], 'revision': refs[-1]['sha256'],
        'capture_device_id': camera, 'start_us': left, 'end_us': left,
        'wearer_id': None, 'instrument_id': None, 'evidence_refs': refs,
        'payload': _reading_payload(data, item) | {'measurement_result_available': False},
        'media': {'photo': {'path': str(photo) if photo else None, 'available': bool(photo and photo.is_file())},
                  'result_file': None},
        'clock': {'basis': data.get('time_basis'), 'uncertainty_us': None, 'acquisition': data.get('acquisition')}}]


def context_events(config, recording):
    settings = config['device_day']['field_recognition']
    root = Path(settings['archive_root'])
    camera = recording.get('camera_key')
    start, end = recording.get('recording_start_us'), recording.get('recording_end_us')
    if (not isinstance(camera, str) or '/' in camera or '\\' in camera or
            type(start) is not int or type(end) is not int or end <= start or end - start > 2 * 86400 * 1e6):
        raise ValueError('Field archive requires a bounded camera interval')
    zone = ZoneInfo(config['device_day'].get('timezone', 'Asia/Shanghai'))
    day = datetime.fromtimestamp(start / 1e6, zone).date()
    last = datetime.fromtimestamp((end - 1) / 1e6, zone).date()
    events, found, errors, references = [], False, [], []

    def unavailable(path, exc, event_id=None):
        errors.append({'path': str(path), 'event_id': event_id, 'error_type': type(exc).__name__,
                       'reason': 'field_archive_evidence_unavailable', 'validation_error': str(exc)[:200],
                       'automatic_action': 'retry_source_next_publication'})

    @contextmanager
    def event_boundary(item, path, measurement_index=None):
        try:
            yield
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            errors.append({'path': str(path), 'event_id': item.get('event_id') if isinstance(item, dict) else None,
                'measurement_index': measurement_index, 'error_type': type(exc).__name__,
                'reason': 'field_archive_measurement_invalid' if measurement_index is not None else 'field_archive_event_invalid',
                'validation_error': str(exc)[:200], 'automatic_action': 'retry_source_next_publication'})

    while day <= last:
        path = safe_child(root, f'{day.isoformat()}_{camera}/DailyReport/DailyReport.json')
        if path.is_file():
            found = True
            try:
                daily, reference = snapshot(path)
            except (OSError, ValueError) as exc:
                unavailable(path, exc)
                day += timedelta(days=1)
                continue
            references.append(reference)
            if not isinstance(daily, dict) or not isinstance(daily.get('events'), list):
                unavailable(path, ValueError('Field daily report events must be an array'))
                day += timedelta(days=1)
                continue
            if daily.get('camera_id', camera) != camera:
                unavailable(path, ValueError('Field daily report crossed its camera archive'))
                day += timedelta(days=1)
                continue
            for number, item in enumerate(daily['events']):
                with event_boundary(item, path):
                    source_path = _owned_source(root, item['result'], camera, day)
                    try:
                        data, source = snapshot(source_path)
                    except (OSError, json.JSONDecodeError) as exc:
                        unavailable(source_path, exc, item.get('event_id'))
                        continue
                    references.append(source)
                    if data.get('camera_id', camera) != camera or item.get('camera_id', camera) != camera:
                        raise ValueError('Field event crossed its camera archive')
                    refs = [reference | {'json_pointer': f'/events/{number}'}, source]
                    if item['category'] == 'Bindings':
                        binding = data['binding']
                        if binding.get('camera_id') != camera:
                            raise ValueError('Field binding crossed its camera archive')
                        left = instant(binding.get('started_at'))
                        # An open historical device binding cannot extend past
                        # its supplied validity boundary or Beijing midnight.
                        midnight = datetime.combine(day + timedelta(days=1), datetime.min.time(), zone)
                        right = min(value for value in (instant(binding.get('ended_at')),
                            instant(binding.get('valid_until')), round(midnight.timestamp() * 1e6)) if value is not None)
                        if left is None or right <= left:
                            raise ValueError('Field binding has no valid effective-time interval')
                        if left >= end or right <= start:
                            continue
                        events.append({'type': 'instrument_binding' if binding.get('wearer_id') else 'scene_binding',
                            'event_id': item['event_id'], 'revision': source['sha256'], 'capture_device_id': camera,
                            'wearer_id': binding.get('wearer_id'), 'operator_id': None,
                            'start_us': max(start, left), 'end_us': min(end, right),
                            'instrument_id': (binding.get('instrument') or {}).get('id'), 'evidence_refs': refs,
                            'payload': {'role': 'wearer' if binding.get('wearer_id') else 'unknown',
                                'operator_label': binding.get('operator'), 'valid_from': binding['started_at'],
                                'valid_to': datetime.fromtimestamp(right / 1e6, zone).isoformat(),
                                'binding_id': binding.get('binding_id'), 'identity_basis': binding.get('identity_basis')},
                            'clock': {'basis': data.get('time_basis'), 'uncertainty_us': None}})
                    else:
                        available = False
                        for measurement_index, entry in enumerate(item.get('measurement_files') or []):
                            with event_boundary(item, path, measurement_index):
                                if not entry.get('result'):
                                    continue
                                result_path = _owned_source(root, entry['result'], camera, day)
                                try:
                                    result, result_ref = snapshot(result_path)
                                except (OSError, json.JSONDecodeError) as exc:
                                    unavailable(result_path, exc, item.get('event_id'))
                                    continue
                                references.append(result_ref)
                                instrument = _measurement_identity(result, entry, item, data, camera)
                                photo_time = result.get('photo_time')
                                if photo_time is not None and not isinstance(photo_time, dict):
                                    raise ValueError('Field measurement photo_time must be an object')
                                if not isinstance(result.get('values', []), list):
                                    raise ValueError('Field measurement values must be an array')
                                stamp = (photo_time or {}).get('timestamp_ms')
                                if stamp is not None:
                                    if type(stamp) not in (int, float) or not math.isfinite(stamp) or not math.isfinite(stamp * 1000):
                                        raise ValueError('Field measurement timestamp must be finite milliseconds')
                                    left = round(stamp * 1000)
                                else:
                                    left = instant(item.get('event_at'))
                                available = True
                                if left is None or not start <= left < end:
                                    continue
                                photo = _owned_source(root, item['photo'], camera, day) if item.get('photo') else None
                                events.append({'type': 'ocr_observation', 'event_id': item['event_id'] + ':' + entry['result'],
                                    'revision': result_ref['sha256'], 'capture_device_id': camera,
                                    'start_us': left, 'end_us': left, 'wearer_id': result.get('wearer_id'),
                                    'instrument_id': instrument, 'evidence_refs': [*refs, result_ref],
                                    'payload': _reading_payload(data, item, result) | {
                                        'instrument_name': entry.get('instrument_name'), 'measurement_result_available': True,
                                        'result_identity': {key: result[key] for key in ('camera_id', 'instrument_id',
                                            'wearer_id', 'device_model', 'device_no', 'qr_hash') if key in result}},
                                    'media': {'photo': {'path': str(photo) if photo else None,
                                        'available': bool(photo and photo.is_file())}, 'result_file': result_ref['path']},
                                    'clock': {'basis': 'archived_photo_time', 'uncertainty_us': None,
                                              'photo_time': photo_time, 'acquisition': data.get('acquisition')}})
                        if not available:
                            events.extend(_unbound_event(root, camera, day, start, end, item, data, refs))
        day += timedelta(days=1)
    unique, conflicts = {}, set()
    for event in events:
        if event['event_id'] in conflicts:
            continue
        previous = unique.setdefault(event['event_id'], event)
        if {k: v for k, v in previous.items() if k != 'evidence_refs'} != {k: v for k, v in event.items() if k != 'evidence_refs'}:
            unique.pop(event['event_id'])
            conflicts.add(event['event_id'])
            errors.append({'event_id': event['event_id'], 'reason': 'field_archive_conflicting_revisions',
                           'automatic_action': 'isolate_conflicting_event_and_retry_source'})
    return {'status': ('partial_archived_evidence' if errors else 'available_archived_evidence')
                if found else 'no_archive_for_camera_day',
            'events': sorted(unique.values(), key=lambda event: (event['start_us'], event['event_id'])),
            'errors': errors, 'source_references': sorted({r['path']: r for r in references}.values(), key=lambda r: r['path']),
            'source': 'field_recognition', 'association_basis': 'same_camera_actual_interval_overlap',
            'physical_action_verified': False}

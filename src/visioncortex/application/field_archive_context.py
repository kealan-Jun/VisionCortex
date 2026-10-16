"""Read existing FieldRecognition evidence without changing its live service."""
from collections import OrderedDict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

from ..device_day_contract import safe_child

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
    events, found = [], False
    while day <= last:
        path = safe_child(root, f'{day.isoformat()}_{camera}/DailyReport/DailyReport.json')
        if path.is_file():
            found = True
            daily, reference = snapshot(path)
            if daily.get('camera_id', camera) != camera:
                raise ValueError('Field daily report crossed its camera archive')
            for number, item in enumerate(daily['events']):
                data, source = snapshot(safe_child(root, item['result']))
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
                    if left is None or left >= end or right <= start:
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
                    for entry in item.get('measurement_files') or []:
                        if not entry.get('result'):
                            continue
                        result, result_ref = snapshot(safe_child(root, entry['result']))
                        stamp = (result.get('photo_time') or {}).get('timestamp_ms')
                        left = round(stamp * 1000) if type(stamp) in (int, float) else instant(item.get('event_at'))
                        if left is None or not start <= left < end:
                            continue
                        photo = safe_child(root, item['photo']) if item.get('photo') else None
                        events.append({'type': 'ocr_observation', 'event_id': item['event_id'] + ':' + entry['result'],
                            'revision': result_ref['sha256'], 'capture_device_id': camera,
                            'start_us': left, 'end_us': left, 'wearer_id': result.get('wearer_id'),
                            'instrument_id': entry.get('instrument_id'), 'evidence_refs': [*refs, result_ref],
                            'payload': {'normalized_values': result.get('values', []), 'status': item.get('status'),
                                'record_scope': data.get('record_scope', 'unknown'),
                                'instrument_name': entry.get('instrument_name'), 'decision': data.get('decision')},
                            'media': {'photo': {'path': str(photo) if photo else None,
                                'available': bool(photo and photo.is_file())}, 'result_file': result_ref['path']},
                            'clock': {'basis': 'archived_photo_time', 'uncertainty_us': None,
                                      'acquisition': data.get('acquisition')}})
        day += timedelta(days=1)
    return {'status': 'available_archived_evidence' if found else 'no_archive_for_camera_day',
            'events': sorted(events, key=lambda event: (event['start_us'], event['event_id'])),
            'source': 'field_recognition', 'association_basis': 'same_camera_actual_interval_overlap',
            'physical_action_verified': False}

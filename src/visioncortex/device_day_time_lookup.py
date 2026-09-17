"""Global-time material references; never move or copy archive/capture files."""
from datetime import datetime
import math
from pathlib import Path
import re
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .device_day_contract import safe_child
from .media_time import media_ms


def interval(start, end):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (start, end)) and 0 < start < end


def material_records(archive, index):
    from .device_day_timeline import link
    result = []
    for record in index.get('recordings', []):
        start, end = record.get('start_us'), record.get('end_us')
        if not interval(start, end):
            continue
        sources = {s['kind']: s.get('retained') for s in record.get('sources', [])}
        audio = record.get('audio') or {}
        transcript = record.get('transcription') or {}
        # Legacy indexes can carry the audio reference in either location.
        audio_ref = sources.get('audio_audio') or next((a for a in audio.get('artifacts', [])
                                                       if a.get('kind') == 'audio_audio'), None)
        audio_start, audio_end = audio.get('start_us'), audio.get('end_us')
        audio_timed = interval(audio_start, audio_end)
        result.append({'archive': archive, 'camera': archive[11:], 'recording_id': record['recording_id'],
            'start_us': start, 'end_us': end,
            'video_ref': sources.get('video'), 'video_url': link(archive, sources.get('video')),
            'clock_mapping': (record.get('processing') or {}).get('clock_mapping') or {},
            'audio_ref': audio_ref, 'audio_url': link(archive, audio_ref),
            'audio_start_us': audio_start if audio_timed else None,
            'audio_end_us': audio_end if audio_timed else None,
            'audio_time_basis': audio.get('time_basis', 'unknown'),
            'transcription_status': transcript.get('status'), 'transcription_outcome': transcript.get('outcome'),
            'transcript_url': link(archive, transcript.get('transcript_file')),
            'comments': [{**c, 'transcript_url': link(archive, c.get('transcript_path'))}
                         for c in transcript.get('comments') or [] if interval(c.get('start_us'), c.get('end_us'))]})
    return result


def query_materials(timeline, at_us, duration_seconds=60):
    from .device_day_timeline import day_bounds
    lo, hi = day_bounds(timeline['date'])
    if (isinstance(at_us, bool) or not isinstance(at_us, int) or not lo <= at_us < hi
            or not isinstance(duration_seconds, (int, float)) or not math.isfinite(duration_seconds)
            or not 0 < duration_seconds <= 3600):
        raise ValueError('Query requires a timestamp in this day and a window of at most one hour')
    end = min(hi, at_us + round(duration_seconds * 1e6))
    devices = {}
    for record in timeline.get('media_recordings', []):
        video_match = record['start_us'] < end and record['end_us'] > at_us
        audio_match = (record['audio_start_us'] is not None and record['audio_start_us'] < end
                       and record['audio_end_us'] > at_us)
        comments = [c for c in record['comments'] if c['start_us'] < end and c['end_us'] > at_us]
        if not (video_match or audio_match or comments):
            continue
        video_seconds, video_basis = media_ms(record['clock_mapping'], max(at_us, record['start_us']), record['start_us'])
        device = devices.setdefault(record['camera'], {'camera': record['camera'], 'recordings': [], 'photos': []})
        device['recordings'].append({k: v for k, v in record.items() if k not in {'clock_mapping', 'comments'}} | {
            'video_url': record['video_url'] if video_match else None,
            'video_offset_seconds': max(0, video_seconds / 1000) if video_match else None,
            'video_time_basis': video_basis, 'audio_overlaps_query': bool(audio_match),
            'audio_offset_seconds': max(0, (at_us-record['audio_start_us'])/1e6) if audio_match else None,
            'comments': comments, 'time_alignment': 'PARTIAL_EVIDENCE'})
    return {'date': timeline['date'], 'start_us': at_us, 'end_us': end, 'timezone': 'Asia/Shanghai',
            'devices': sorted(devices.values(), key=lambda d: d['camera']),
            'errors': timeline.get('errors', []), 'evidence_status': 'PARTIAL_EVIDENCE',
            'cross_camera_alignment_verified': False}


def photo_path(config, day, camera, folder, filename):
    from .device_day_timeline import day_bounds
    day_bounds(day)
    if (camera not in config.get('collection_ingest', {}).get('camera_role_map', {})
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', camera)
            or not re.fullmatch(r'\d{2}-\d{2}-\d{2}', folder)):
        raise ValueError('Unknown capture photo device or folder')
    moment = datetime.strptime(day + ' ' + folder, '%Y-%m-%d %H-%M-%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
    prefix = moment.strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(re.escape(prefix) + r'(?:_\d+)?\.(?:jpg|jpeg|png|webp)', filename, re.IGNORECASE):
        raise ValueError('Photo name does not match its capture date/time')
    root = Path(config['collection_ingest']['source_root'])
    return safe_child(root, f'voice_photos/{camera}/{day}/{folder}/{filename}'), round(moment.timestamp()*1e6)


def attach_photos(config, result):
    """Use recorder filename time explicitly; upload mtime is never capture time."""
    from .device_day_schedule import processing_cutoff
    cutoff = processing_cutoff(config.get('device_day', {}))
    if cutoff and result['end_us'] <= cutoff:
        return result | {'photo_errors': [], 'photo_history_paused_before_us': cutoff,
                         'photo_time_basis': 'capture_filename_seconds_not_verified_global_clock'}
    root = Path(config['collection_ingest']['source_root'])
    devices = {d['camera']: d for d in result['devices']}
    errors = []
    for camera in config['collection_ingest'].get('camera_role_map', {}):
        try:
            directory = safe_child(root, f'voice_photos/{camera}/{result["date"]}')
            if not directory.is_dir():
                continue
            for folder in directory.iterdir():
                try:
                    moment = datetime.strptime(result['date'] + ' ' + folder.name, '%Y-%m-%d %H-%M-%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
                except ValueError:
                    continue
                captured = round(moment.timestamp()*1e6)
                if not result['start_us'] <= captured < result['end_us'] or (cutoff and captured < cutoff):
                    continue
                if folder.is_symlink() or not folder.is_dir():
                    continue
                for child in sorted(folder.iterdir()):
                    try:
                        path, stamp = photo_path(config, result['date'], camera, folder.name, child.name)
                        stat = path.stat()
                        if not path.is_file() or stat.st_size == 0 or time.time()-stat.st_mtime < config['collection_ingest'].get('settle_seconds', 5):
                            continue
                    except (OSError, ValueError):
                        continue
                    device = devices.setdefault(camera, {'camera': camera, 'recordings': [], 'photos': []})
                    device['photos'].append({'capture_us': stamp, 'time_basis': 'capture_filename_seconds',
                        'timestamp_resolution_us': 1000000, 'capture_clock_verified': False,
                        'recording_ids': [r['recording_id'] for r in device['recordings'] if r['start_us'] <= stamp < r['end_us']],
                        'source_ref': {'storage_root': 'collection_ingest.source_root', 'path': path.relative_to(root).as_posix(),
                                       'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
                        'url': '/api/day-timeline/'+result['date']+'/photos/'+ '/'.join(quote(v, safe='') for v in (camera, folder.name, child.name))})
        except (OSError, ValueError) as exc:
            errors.append({'camera': camera, 'error_type': type(exc).__name__})
    result.update(devices=sorted(devices.values(), key=lambda d: d['camera']), photo_errors=errors,
                  photo_time_basis='capture_filename_seconds_not_verified_global_clock',
                  photo_history_paused_before_us=cutoff)
    return result


def refresh_photo_index(config, camera, day=None):
    """A per-camera background reader publishes references on local disk only."""
    from copy import deepcopy
    from .device_day_contract import atomic_json
    from .device_day_timeline import day_bounds
    day = day or datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    lo, hi = day_bounds(day)
    selected = deepcopy(config)
    selected['collection_ingest']['camera_role_map'] = {camera: config['collection_ingest']['camera_role_map'][camera]}
    value = attach_photos(selected, {'date': day, 'start_us': lo, 'end_us': hi, 'devices': []})
    value['observed_at'] = time.time()
    path = Path(config['storage']['local_runtime_root'])/'device-day'/'CapturePhotos'/f'{day}_{camera}.json'
    atomic_json(path, value)


def indexed_photos(config, result):
    """Never block an interactive time query on SMB photo discovery."""
    from .device_day_contract import read_json
    from .device_day_schedule import processing_cutoff
    cutoff = processing_cutoff(config.get('device_day', {}))
    result.update(photo_errors=[], photo_history_paused_before_us=cutoff,
                  photo_time_basis='capture_filename_seconds_not_verified_global_clock')
    if cutoff and result['end_us'] <= cutoff:
        return result
    devices = {d['camera']: d for d in result['devices']}
    root = Path(config['storage']['local_runtime_root'])/'device-day'/'CapturePhotos'
    for camera in config['collection_ingest'].get('camera_role_map', {}):
        try:
            index = read_json(root/f'{result["date"]}_{camera}.json')
            result['photo_errors'].extend(index.get('photo_errors', []))
            if time.time()-index.get('observed_at', 0) > 30:
                result['photo_errors'].append({'camera': camera, 'status': 'photo_index_stale'})
            for device in index['devices']:
                for photo in device['photos']:
                    stamp = photo['capture_us']
                    if not result['start_us'] <= stamp < result['end_us'] or (cutoff and stamp < cutoff):
                        continue
                    target = devices.setdefault(camera, {'camera': camera, 'recordings': [], 'photos': []})
                    target['photos'].append(photo | {'recording_ids': [r['recording_id'] for r in target['recordings']
                                                                      if r['start_us'] <= stamp < r['end_us']]})
        except (OSError, ValueError, KeyError):
            result['photo_errors'].append({'camera': camera, 'status': 'photo_index_initializing'})
    result['devices'] = sorted(devices.values(), key=lambda d: d['camera'])
    return result

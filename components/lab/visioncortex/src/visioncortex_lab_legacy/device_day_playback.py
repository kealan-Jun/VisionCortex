"""Read-only, same-day multi-camera playback of existing archived originals.

This presents recorder time correspondence, not a new physical-action decision
or a substitute for the offline pipeline's alignment/identity quality gates.
"""
from bisect import bisect_left
import math
from pathlib import Path
from urllib.parse import urlencode

from .media_time import media_ms

from .device_day_contract import read_json, safe_child, validate_archive_name
from .device_day_timeline import day_bounds, link


def _capture(points, milliseconds):
    i = max(1, min(len(points)-1, bisect_left([p[1]*1000 for p in points], milliseconds)))
    a, b = points[i-1:i+1]
    return round(a[0] + (milliseconds/1000-a[1])/(b[1]-a[1])*(b[0]-a[0]))




def build_playback(day, indexes, archive, segment_id, roles):
    lo, hi = day_bounds(day)
    validate_archive_name(archive)
    if archive[:10] != day:
        raise ValueError('Selected archive is outside this date')
    indexes = [(name, data) for name, data in indexes if name[:10] == day]
    selected = next((s for name, data in indexes if name == archive
                     for s in data.get('segments', []) if s['segment_id'] == segment_id), None)
    if not selected or selected.get('activity') != 'active':
        raise ValueError('Published activity segment unavailable')
    selected_index = dict(indexes)[archive]
    shared = selected_index.get('multiview_analysis') or {}
    aligned = {(r['archive'], r['recording_id']): r for r in shared.get('aligned_recordings') or []}
    def aligned_record(name, r):
        value = aligned.get((name, r['recording_id']))
        source = next((s.get('retained') for s in r.get('sources', []) if s['kind'] == 'video'), {}) or {}
        return value if value and value.get('source_sha256') == source.get('sha256') else None
    start, end = max(lo, selected['start_us']), min(hi, selected['end_us'])
    selected_record = next((r for r in selected_index.get('recordings', []) if r['recording_id'] == selected['recording_id']), {})
    selected_alignment = aligned_record(archive, selected_record) if selected_record else None
    if selected_alignment and selected.get('start_ms') is not None:
        start = max(lo, _capture(selected_alignment['seek_points'], selected['start_ms']))
        end = min(hi, _capture(selected_alignment['seek_points'], selected['end_ms']))
    experiments = [e for e in selected_index.get('aligned_experiments', [])
                   if any(s['archive'] == archive and s['recording_id'] == selected['recording_id']
                          and s['start_ms'] < selected.get('end_ms', 0) and s['end_ms'] > selected.get('start_ms', 0)
                          for s in e['sources'])]
    paired = {v for e in experiments for v in (e['group']['first_person_view'], e['group']['third_person_view'])}
    if start >= end:
        raise ValueError('Invalid activity interval')
    cameras = []
    for name, data in indexes:
        validate_archive_name(name)
        if data.get('archive') != name:
            raise ValueError('Archive/index identity mismatch')
        records, unknown_time = [], 0
        for r in data.get('recordings', []):
            if (not all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ('start_us', 'end_us'))
                    or r['end_us'] <= r['start_us']):
                unknown_time += 1
                continue
            fitted = aligned_record(name, r)
            left, right = max(start, (fitted or r)['start_us']), min(end, (fitted or r)['end_us'])
            if left >= right:
                continue
            source = next((s.get('retained') for s in r.get('sources', []) if s['kind'] == 'video'), None)
            def bounds(s):
                if fitted and s.get('start_ms') is not None:
                    return (_capture(fitted['seek_points'], s['start_ms']),
                            _capture(fitted['seek_points'], s['end_ms']))
                return s['start_us'], s['end_us']
            segments = [s for s in data.get('segments', []) if s['recording_id'] == r['recording_id']
                        and bounds(s)[0] < right and bounds(s)[1] > left]
            # Unpublished preprocessing is never classified as inactivity.
            state = ('active' if any(s['activity'] == 'active' for s in segments)
                     else 'inactive' if segments else 'pending')
            clock = r.get('processing', {}).get('clock_mapping') or {}
            first, basis = media_ms(clock, left, r['start_us'])
            last, _ = media_ms(clock, right, r['start_us'])
            knots = [[left, max(0, first)/1000]]
            if basis == 'recorder_csv_interpolation':
                knots.extend([p[1], max(0, p[0])/1000] for p in clock['points'] if left < p[1] < right)
            knots.append([right, max(0, last)/1000])
            if fitted:
                # Invert the very same offline affine/segment transform used by the shared audit.
                fitted_clock = {'basis': 'recorder_csv_interpolation',
                                'points': [[p[1]*1000, p[0]] for p in fitted['seek_points']]}
                first, _ = media_ms(fitted_clock, left, r['start_us'])
                last, _ = media_ms(fitted_clock, right, r['start_us'])
                knots = [[left, first/1000], *[p for p in fitted['seek_points'] if left < p[0] < right], [right, last/1000]]
                basis = 'shared_offline_alignment'
            records.append({'recording_id': r['recording_id'], 'start_us': left, 'end_us': right,
                            'source_url': link(name, source), 'source_ref': source,
                            'activity': state, 'time_basis': basis, 'seek_points': knots,
                            'alignment': fitted,
                            'clips': [{'start_us': max(left, bounds(s)[0]), 'end_us': min(right, bounds(s)[1]),
                                       'source_url': link(name, s['video']), 'offset_seconds': s['start_ms']/1000}
                                      for s in segments if s.get('video')]})
        cameras.append({'archive': name, 'camera': name[11:], 'role': roles.get(name[11:], 'unknown'),
                        'selected': name == archive, 'recordings': sorted(records, key=lambda r: r['start_us']),
                        'association': 'offline_group_selected' if name[11:] in paired else 'time_comparison',
                        'unknown_time_recordings': unknown_time,
                        'status': 'available' if any(r['source_url'] for r in records) else 'no_published_source'})
    selected_role = roles.get(archive[11:])
    cameras.sort(key=lambda c: (not c['selected'], c['association'] != 'offline_group_selected',
                               not bool(c['recordings']), c['role'] == selected_role, c['camera']))
    decisions = []
    origin = (selected_alignment or {}).get('global_origin_us')
    if origin is not None:
        decisions = [d for d in shared.get('formal_decisions') or []
                     if origin+d.get('global_start_ms', 0)*1000 < end
                     and origin+d.get('global_end_ms', 0)*1000 > start]
    return {'date': day, 'archive': archive, 'segment_id': segment_id, 'start_us': start, 'end_us': end,
            'cameras': cameras, 'cross_camera_alignment_verified': False,
            'alignment_ready': bool(selected_alignment and selected_alignment['state'] == 'aligned'),
            'multiview_analysis': {k: shared.get(k) for k in ('status', 'algorithms', 'alignment_quality', 'message', 'missing_sources', 'formal_decisions')},
            'grouping_decisions': decisions,
            'experiments': experiments,
            'actions': [{'action_type': f.get('action_type'), 'event_id': f.get('event_id'),
                         'url': link(archive, f)} for f in selected.get('key_frames', [])],
            'evidence_status': 'PARTIAL_EVIDENCE', 'source': 'existing_device_day_indexes',
            'missing_gates': ['measured_cross_camera_alignment', 'same_scene_and_instrument_identity']}


def load_playback(config, day, archive, segment_id):
    day_bounds(day)
    root = Path(config['storage']['archive_root'])
    indexes, errors = [], []
    for p in sorted(root.glob(day+'_*')):
        try:
            validate_archive_name(p.name)
            path = safe_child(root, p.name+'/ProcessedClips/Index.json')
            if p.is_symlink() or path.is_symlink():
                raise ValueError('Invalid index path')
            indexes.append((p.name, read_json(path)))
        except (ValueError, OSError) as exc:
            errors.append({'archive': p.name, 'reason': type(exc).__name__})
    result = build_playback(day, indexes, archive, segment_id,
                           config.get('collection_ingest', {}).get('camera_role_map', {}))
    result['errors'] = errors
    path = Path(config['storage']['local_runtime_root'])/'device-day/observed-inventory.json'
    from .observed_inventory import read_inventory
    inventory = read_inventory(path.parent)
    result['discovery_errors'] = []
    capture_root = config.get('collection_ingest', {}).get('source_root')
    for error in inventory.get('errors', []):
        if f'/{day}/' not in error.get('path', ''):
            continue
        current = dict(error)
        if capture_root:
            try:
                safe_child(Path(capture_root), error['path']).stat()
            except FileNotFoundError:
                current['current_status'] = 'source_path_missing'
                current['message'] = '原采集路径当前不存在。历史错误：'+error.get('message', '')
            except (OSError, ValueError):
                current['current_status'] = 'not_verified'
        result['discovery_errors'].append(current)
    for camera in result['cameras']:
        for r in camera['recordings']:
            r['preview_url'] = '/api/day-timeline/'+day+'/preview?'+urlencode({
                'archive': archive, 'segment_id': segment_id, 'camera_archive': camera['archive'],
                'recording_id': r['recording_id']})
            def available(reference):
                if not reference:
                    return 'missing_reference'
                try:
                    p = safe_child(root, camera['archive']+'/'+reference['path'])
                    stat = p.stat()
                    return 'available' if p.is_file() and (not reference.get('size_bytes') or stat.st_size == reference['size_bytes']) else 'size_mismatch'
                except FileNotFoundError:
                    return 'missing'
                except (OSError, ValueError):
                    return 'unreadable'
            r['source_status'] = available(r['source_ref'])
            index = next(d for n,d in indexes if n == camera['archive'])
            source_segments = {link(camera['archive'], s.get('video')): s.get('video') for s in index.get('segments', [])}
            for clip in r['clips']:
                clip['status'] = available(source_segments.get(clip['source_url']))
                if clip['status'] != 'available':
                    clip['source_url'] = None
            if r['source_status'] != 'available':
                r['preview_url'] = None
    return result


def preview(config, day, archive, segment_id, camera_archive, recording_id, window):
    """Bounded disposable preview; existing clip extraction, no new NAS output."""
    from .device_day_contract import digest
    from .web_playback_cache import cached_preview
    day_bounds(day)
    for name in (archive, camera_archive):
        validate_archive_name(name)
        if name[:10] != day:
            raise ValueError('Preview date mismatch')
    root = Path(config['storage']['archive_root'])
    indexes = [(name, read_json(safe_child(root, name+'/ProcessedClips/Index.json')))
               for name in dict.fromkeys([archive, camera_archive])]
    data = build_playback(day, indexes, archive, segment_id, {})
    camera = next(c for c in data['cameras'] if c['archive'] == camera_archive)
    r = next((r for r in camera['recordings'] if r['recording_id'] == recording_id), None)
    if not r or not r['source_url'] or window < 0:
        raise ValueError('Preview source unavailable')
    left = data['start_us'] + window * 30_000_000
    right = min(data['end_us'], r['end_us'], left + 30_000_000)
    left = max(left, r['start_us'])
    if left >= right:
        raise ValueError('Preview window outside selected activity')
    index = dict(indexes)[camera_archive]
    record = next(x for x in index['recordings'] if x['recording_id'] == recording_id)
    clock = {'basis': 'recorder_csv_interpolation', 'points': [[p[1]*1000, p[0]] for p in r['seek_points']]}
    start_ms, _ = media_ms(clock, left, record['start_us'])
    end_ms, _ = media_ms(clock, right, record['start_us'])
    source = safe_child(root, camera_archive+'/'+r['source_ref']['path'])
    return cached_preview(source, Path(config['storage']['local_runtime_root']), max(0, start_ms),
                          end_ms-start_ms, digest(r['source_ref']))

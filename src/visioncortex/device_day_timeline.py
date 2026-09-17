"""One date's observation timeline, derived from unchanged device/day indexes."""
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from .device_day_contract import DIRECTORIES, read_json, validate_archive_name, digest, safe_child


def day_bounds(day):
    parsed = date.fromisoformat(day)
    if parsed.isoformat() != day:
        raise ValueError('Use YYYY-MM-DD')
    start = datetime.combine(parsed, datetime.min.time(), ZoneInfo('Asia/Shanghai'))
    return round(start.timestamp()*1e6), round((start+timedelta(days=1)).timestamp()*1e6)


def link(archive, reference):
    value = reference.get('path') if isinstance(reference, dict) else reference
    if not isinstance(value, str):
        return None
    parts = PurePosixPath(value).parts
    if not parts or parts[0] not in DIRECTORIES or '..' in parts or '\\' in value:
        return None
    return f'/api/device-days/{quote(archive, safe="")}/files/{quote(value, safe="/")}'


def build_timeline(day, indexes):
    lo, hi = day_bounds(day)
    entries, recordings, seen, media = [], set(), set(), []
    for archive, index in indexes:
        validate_archive_name(archive)
        if archive[:10] != day:
            continue
        from .device_day_time_lookup import material_records
        media.extend(material_records(archive, index))
        record_map = {r['recording_id']: r for r in index.get('recordings', [])}
        meanings = {x['segment_id']: x for x in index.get('understandings', [])}
        recordings.update((archive, x['recording_id']) for x in index.get('recordings', []))
        for segment in index.get('segments', []):
            identity = archive + '/' + segment['segment_id']
            if identity in seen:
                continue
            seen.add(identity)
            start, end = max(lo, segment['start_us']), min(hi, segment['end_us'])
            if start >= end:
                continue
            meaning = meanings.get(segment['segment_id'], {})
            steps = [s for w in meaning.get('windows', []) for s in w.get('steps', [])
                     if isinstance(s.get('start_us'), (int, float)) and isinstance(s.get('end_us'), (int, float))
                     and start <= s['start_us'] <= s['end_us'] <= end]
            source = link(archive, segment.get('source_ref'))
            entries.append({'id': identity, 'archive': archive, 'camera': archive[11:],
                            'recording_id': segment['recording_id'], 'segment_id': segment['segment_id'],
                            'start_us': start, 'end_us': end, 'activity': segment['activity'],
                            'start_ms': segment.get('start_ms'), 'end_ms': segment.get('end_ms'),
                            'action_candidates': list({c['candidate_id']: c
                                for e in segment.get('activity_audit', {}).get('events', [])
                                for c in e.get('candidates', [])
                                if c['local_start_ms'] <= segment['end_ms'] and c['local_end_ms'] >= segment['start_ms']}.values()),
                            'clock_ref': segment.get('clock_ref'),
                            'clock_mapping': record_map.get(segment['recording_id'], {}).get('processing', {}).get('clock_mapping'), 'wall_time_basis': segment.get('wall_time_basis'),
                            'source_ref': segment.get('source_ref'), 'source_url': source,
                            'clip_url': link(archive, segment.get('video')),
                            'json_url': link(archive, segment.get('json_path')),
                            'key_frames': [{**f, 'url': link(archive, f)} for f in segment.get('key_frames', [])],
                            'scene_frames': [{**f, 'url': link(archive, f)} for f in segment.get('scene_frames', [])],
                            'steps': sorted(steps, key=lambda s: s['start_us']),
                            'summaries': [w['summary'] for w in meaning.get('windows', []) if w.get('summary')],
                            'understanding_ready': bool(meaning), 'physical_action_confirmed': False})
    entries.sort(key=lambda e: (e['start_us'], e['camera'], e['id']))
    changes = defaultdict(list)
    for n, entry in enumerate(entries):
        changes[entry['start_us']].append((n, True))
        changes[entry['end_us']].append((n, False))
    periods, current = [], set()
    points = sorted(changes)
    for position, start in enumerate(points[:-1]):
        for n, adding in changes[start]:
            current.add(n) if adding else current.discard(n)
        end = points[position+1]
        state = ('active' if any(entries[n]['activity'] == 'active' for n in current)
                 else 'inactive' if current else 'missing')
        if periods and periods[-1]['state'] == state and periods[-1]['end_us'] == start:
            periods[-1]['end_us'] = end
            periods[-1]['entry_ids'].update(entries[n]['id'] for n in current)
        else:
            periods.append({'start_us': start, 'end_us': end, 'state': state,
                            'entry_ids': {entries[n]['id'] for n in current}})
    for period in periods:
        period['entry_ids'] = sorted(period['entry_ids'])
    return {'date': day, 'entries': entries, 'periods': periods, 'media_recordings': media,
            'input_index_digests': {name: digest(index.get('segments', [])) for name, index in indexes},
            'indexed_recordings': len(recordings), 'camera_count': len({name[11:] for name, _ in indexes}),
            'covered_seconds': sum((p['end_us']-p['start_us'])/1e6 for p in periods if p['state'] != 'missing'),
            'activity_seconds': sum((p['end_us']-p['start_us'])/1e6 for p in periods if p['state'] == 'active'),
            'experiment_count': None, 'person_identity_inferred': False,
            'cross_camera_alignment_verified': False, 'evidence_status': 'PARTIAL_EVIDENCE',
            'period_semantics': 'continuous_observation_not_confirmed_experiment',
            'recording_boundary_is_experiment_boundary': False}


def load_timeline(config, day, *, audit=False):
    day_bounds(day)
    root = Path(config['storage']['archive_root'])
    inventory_path = Path(config['storage']['local_runtime_root'])/'device-day'/'observed-inventory.json'
    from .observed_inventory import read_inventory
    inventory = read_inventory(inventory_path.parent)
    lo, hi = day_bounds(day)
    records = [r for r in inventory.get('recordings', []) if lo <= r.get('recording_start_us', 0) < hi]
    names = {day+'_'+r['camera_key'] for r in records}
    # Include previously archived devices even if the capture monitor has not
    # rediscovered them; inspect index JSON only, never open source media.
    if root.is_dir():
        names.update(p.name for p in root.glob(day+'_*') if p.is_dir() and not p.is_symlink())
    indexes, errors = [], []
    for name in sorted(names):
        try:
            validate_archive_name(name)
            path = safe_child(root, name+'/'+DIRECTORIES[1]+'/Index.json')
            if (root/name).is_symlink() or path.is_symlink():
                raise ValueError('Symlink is not an archive index')
            indexes.append((name, read_json(path)))
        except (OSError, ValueError) as exc:
            errors.append({'archive': name, 'status': 'index_unavailable', 'error_type': type(exc).__name__})
    result = build_timeline(day, indexes)
    published = {(e['archive'], e['recording_id']) for e in result['entries']}
    result.update(discovered_recordings=len(records),
                  pending_recordings=sum((day+'_'+r['camera_key'], r['recording_id']) not in published for r in records),
                  errors=errors, discovery_errors=[e for e in inventory.get('errors', []) if f'/{day}/' in e.get('path', '')])
    from .device_day_cross_view import associate
    result['cross_view_links'] = associate(result['entries'],
        config.get('collection_ingest', {}).get('camera_role_map', {}), config, audit=False)
    if audit:
        from .device_day_multiview import build_device_day_multiview
        try:
            shared = build_device_day_multiview(config, day, indexes)
            result['multiview_analysis'] = {k: shared.get(k) for k in
                ('key', 'status', 'receipt_path', 'algorithms', 'alignment_quality', 'missing_sources', 'aligned_recordings', 'formal_decisions', 'capture_alignment_errors')}
            result['aligned_experiments'] = shared.get('experiments', [])
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            result['multiview_analysis'] = {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}
            result['aligned_experiments'] = []
        for edge in result['cross_view_links']:
            related = [e for e in result['aligned_experiments']
                       if all(any(s['archive'] == entry.rsplit('/', 1)[0] for s in e['sources'])
                              for entry in (edge['first_person'], edge['third_person']))
                       and any(s['start_us'] < edge['end_us'] and s['end_us'] > edge['start_us'] for s in e['sources'])]
            edge['experiment_ids'] = [e['experiment_id'] for e in related]
            if related:
                edge['status'] = 'offline_group_selected'
                edge['shared_action_audit'] = {'status': 'completed', 'algorithm': 'visioncortex.grouping.build_experiment_groups'}
    linked = {edge[k] for edge in result['cross_view_links'] for k in ('first_person', 'third_person')}
    for entry in result['entries']:
        entry['cross_view_status'] = ('candidate_available' if entry['id'] in linked
                                      else 'other_view_missing' if entry['activity'] == 'active' else 'not_requested')
        entry.pop('action_candidates', None)
        entry.pop('clock_mapping', None)
    return result

def refresh_timeline(config, day):
    from .device_day_contract import atomic_json
    from .device_day_schedule import processing_cutoff
    live_only = bool(processing_cutoff(config.get('device_day', {})))
    result = load_timeline(config, day, audit=not live_only)
    path = Path(config['storage']['local_runtime_root'])/'device-day'/'DayTimeline'/f'{day}.json'
    atomic_json(path, result)
    if live_only:
        # Publish the time index without reprocessing old experiment groups or
        # rewriting existing NAS indexes. Per-slice stages publish their own work.
        return {'date': day, 'link_count': len(result['cross_view_links']), 'publication_pending': []}
    from .device_day import exclusive
    publication_pending = []
    for name in {entry['archive'] for entry in result['entries']}:
        root = Path(config['storage']['archive_root'])
        index_path = safe_child(root, name+'/'+DIRECTORIES[1]+'/Index.json')
        try:
            with exclusive(Path(config['storage']['local_runtime_root'])/'device-day'/'locks'/f'{name}.index.lock'):
                index = read_json(index_path)
                if digest(index.get('segments', [])) != result['input_index_digests'].get(name):
                    publication_pending.append(name)
                    continue
                ids = {name+'/'+s['segment_id'] for s in index.get('segments', [])}
                index['cross_view_links'] = [edge for edge in result['cross_view_links']
                    if edge['first_person'] in ids or edge['third_person'] in ids]
                index['aligned_experiments'] = [e for e in result.get('aligned_experiments', [])
                                               if any(s['archive'] == name for s in e['sources'])]
                index['multiview_analysis'] = result.get('multiview_analysis')
                atomic_json(index_path, index)
                if result.get('multiview_analysis', {}).get('status') == 'completed':
                    from .device_day_multiview import retire_superseded_outputs
                    retire_superseded_outputs(config, name, index['aligned_experiments'])
        except (OSError, ValueError, BlockingIOError):
            publication_pending.append(name)
    return {'date': day, 'link_count': len(result['cross_view_links']), 'publication_pending': publication_pending}


def query_timeline(config, day):
    """Use the worker's atomic local index; do not reread NAS per UI poll."""
    import time
    day_bounds(day)
    path = Path(config['storage']['local_runtime_root'])/'device-day'/'DayTimeline'/f'{day}.json'
    try:
        value = read_json(path)
        if value.get('date') == day and 'media_recordings' in value:
            return value | {'index_updated_at': path.stat().st_mtime,
                            'index_age_seconds': max(0, time.time()-path.stat().st_mtime)}
    except (OSError, ValueError):
        pass
    return load_timeline(config, day)


def install_routes(app, settings_factory):
    from .device_day_progress import ProgressPoller, ProgressUnavailable
    progress_poller = ProgressPoller(settings_factory)

    @app.get('/api/day-timeline/{day}/at')
    def materials_at(day: str, at_us: int, duration_seconds: float = 60):
        from .device_day_time_lookup import query_materials, indexed_photos
        config = settings_factory()
        try:
            # Validate before any NAS reads, including malformed time windows.
            query_materials({'date': day}, at_us, duration_seconds)
            timeline = query_timeline(config, day)
            result = query_materials(timeline, at_us, duration_seconds)
            result['index_updated_at'] = timeline.get('index_updated_at')
            return indexed_photos(config, result)
        except ValueError as exc:
            raise HTTPException(400, '请使用当天的全局时间戳和不超过一小时的查询范围') from exc
        except OSError as exc:
            raise HTTPException(503, '采集索引暂不可读，请稍后重试') from exc

    @app.get('/api/day-timeline/{day}/photos/{camera}/{folder}/{filename}')
    def capture_photo(day: str, camera: str, folder: str, filename: str):
        from .device_day_time_lookup import photo_path
        from fastapi.responses import FileResponse
        try:
            path, _ = photo_path(settings_factory(), day, camera, folder, filename)
            if not path.is_file():
                raise ValueError('Missing photo')
            return FileResponse(path, headers={'Cache-Control': 'no-store'})
        except (OSError, ValueError) as exc:
            raise HTTPException(404, '该设备的采集照片不可用') from exc

    @app.get('/api/day-timeline/{day}/playback')
    def playback(day: str, archive: str, segment_id: str):
        from .device_day_playback import load_playback
        try:
            return load_playback(settings_factory(), day, archive, segment_id)
        except ValueError as exc:
            raise HTTPException(404, '该日期的实验片段不可用，请刷新总览') from exc
        except OSError as exc:
            raise HTTPException(503, 'NAS 索引暂不可读，请稍后重试') from exc

    @app.get('/api/day-timeline/{day}/preview')
    def playback_preview(day: str, archive: str, segment_id: str, camera_archive: str,
                         recording_id: str, window: int):
        from fastapi.responses import FileResponse
        from .device_day_playback import preview
        try:
            path = preview(settings_factory(), day, archive, segment_id, camera_archive, recording_id, window)
            return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'private, max-age=600'})
        except ValueError as exc:
            raise HTTPException(404, '预览区间或源文件不可用') from exc
        except (OSError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(503, '原片预览暂不可用，请稍后重试') from exc

    @app.get("/api/device-day-progress")
    async def progress():
        import asyncio
        try:
            return await progress_poller.read()
        except (asyncio.TimeoutError, ProgressUnavailable):
            from fastapi.responses import JSONResponse
            return JSONResponse({'status': 'progress_initializing', 'available': False,
                'detail': '进度快照正在生成，后台处理继续运行', 'retry_after_seconds': 5}, status_code=202)

    @app.get('/api/day-timeline/{day}')
    def timeline(day: str):
        try:
            day_bounds(day)
        except ValueError as exc:
            raise HTTPException(400, '日期应为 YYYY-MM-DD') from exc
        result = query_timeline(settings_factory(), day)
        path = Path(settings_factory()['storage']['local_runtime_root'])/'device-day'/'DayTimeline'/f'{day}.json'
        if path.is_file():
            previous = read_json(path)
            known = {edge['link_id']: edge for edge in previous.get('cross_view_links', [])}
            result['cross_view_links'] = [known.get(edge['link_id'], edge) for edge in result['cross_view_links']]
        return result

"""Small file-only time index inside the existing Comment directory."""
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from .device_day_contract import DIRECTORIES, atomic_json, read_json, safe_child, validate_archive_name
from .device_day_schedule import in_processing_scope, processing_cutoff


def build_file_index(config, index, photos):
    archive = validate_archive_name(index['archive'])
    cutoff = processing_cutoff(config.get('device_day', {}))
    def reference(value, base='device_day'):
        value = {'path': value} if isinstance(value, str) else value
        if not value or not isinstance(value.get('path'), str):
            return None
        path = PurePosixPath(value['path'])
        if not path.parts or path.is_absolute() or '..' in path.parts or '\\' in value['path']:
            return None
        if base == 'device_day' and path.parts[0] not in DIRECTORIES:
            return None
        return {k: value[k] for k in ('path', 'sha256', 'size_bytes') if k in value} | {'path_base': base}
    def time_text(value):
        return datetime.fromtimestamp(value/1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='microseconds') if value else None
    def entry(kind, identifier, start, end, **values):
        return {'kind': kind, 'recording_id': identifier, 'start_us': start, 'end_us': end,
                'start_time': time_text(start), 'end_time': time_text(end), **values}
    entries, records = [], {}
    readable = index.get('readable_content') or {}
    readable_audio = {r['recording_id']: r for r in readable.get('audio', [])}
    readable_visual = {r['segment_id']: r for r in readable.get('multimodal', [])}
    for row in index.get('recordings', []):
        if not in_processing_scope(config.get('device_day', {}),
                {'recording_start_us': row.get('start_us'), 'recording_end_us': row.get('end_us')}):
            continue
        rid = row['recording_id']
        records[rid] = row
        sources = {s['kind']: s.get('retained') for s in row.get('sources', [])}
        audio, stt = row.get('audio') or {}, row.get('transcription') or {}
        entries.append(entry('recording', rid, row.get('start_us'), row.get('end_us'),
            video=reference(sources.get('video')), stages=row.get('stages', {}),
            capture_complete=row.get('capture_complete'), capture_issues=row.get('capture_issues', []),
            transcription_status=stt.get('status'), transcription_outcome=stt.get('outcome'),
            model_invocation=stt.get('model_invocation', 'NOT_PROVEN'),
            transcript_file=reference(stt.get('transcript_file'))))
        entries.append(entry('original_video', rid, row.get('start_us'), row.get('end_us'),
            file=reference(sources.get('video')),
            clock_mapping=(row.get('processing') or {}).get('clock_mapping') or {},
            capture_complete=row.get('capture_complete')))
        audio_ref = reference(sources.get('audio_audio') or next(
            (a for a in audio.get('artifacts', []) if a.get('kind') == 'audio_audio'), None))
        if audio_ref:
            entries.append(entry('audio', rid, audio.get('start_us'), audio.get('end_us'),
                file=audio_ref, time_basis=audio.get('time_basis', 'unknown'),
                capture_complete=audio.get('capture_complete'), quality_status=audio.get('quality_status'),
                metadata_files=[reference(a) for a in audio.get('artifacts', []) if a.get('kind') != 'audio_audio'],
                audio_start_offset_us=max(0, audio['start_us']-row['start_us'])
                    if audio.get('start_us') and row.get('start_us') else None))
        for comment in stt.get('comments') or []:
            entries.append(entry('speech', rid, comment.get('start_us'), comment.get('end_us'),
                text=comment['text'], audio=reference(comment.get('audio_ref')) or audio_ref,
                audio_start_seconds=comment.get('audio_start_seconds'), audio_end_seconds=comment.get('audio_end_seconds'),
                transcript_file=reference(comment.get('transcript_path')), model=comment.get('model'),
                time_basis=comment.get('time_basis'), accuracy='NOT_PROVEN', human_reviewed=False,
                physical_action_confirmed=False))
    meanings = {m['segment_id']: m for m in index.get('understandings', [])}
    for segment in index.get('segments', []):
        rid = segment['recording_id']
        if rid not in records:
            continue
        entries.append(entry('video_segment', rid, segment['start_us'], segment['end_us'],
            activity=segment['activity'], segment_id=segment['segment_id'],
            file=reference(segment.get('video')), source_video=reference(segment.get('source_ref')),
            source_start_ms=segment.get('start_ms'), source_end_ms=segment.get('end_ms'),
            metadata_file=reference(segment.get('json_path')),
            physical_action_confirmed=False))
        meaning = meanings.get(segment['segment_id'])
        if meaning:
            # Existing understanding consumes video AND comments. A reference
            # to that output must never be labeled pure video or verbatim audio.
            entries.append(entry('multimodal_result', rid, segment['start_us'], segment['end_us'],
                segment_id=segment['segment_id'], content_source=meaning.get('content_source', 'multimodal_context'),
                status=meaning.get('status', 'completed'),
                content_file=reference(readable_visual.get(segment['segment_id'], {}).get('content')),
                files=[reference(w['model_receipt']) for w in meaning.get('windows', []) if w.get('model_receipt')],
                inputs=[reference(w['input']) for w in meaning.get('windows', []) if w.get('input')],
                physical_action_confirmed=False))
        for key, kind in [('key_frames', 'key_frame'), ('scene_frames', 'scene_frame')]:
            for frame in segment.get(key, []):
                captured = frame.get('capture_us')
                entries.append(entry(kind, rid, captured, captured+1 if captured else None,
                    file=reference(frame), frame_kind=frame.get('frame_kind'),
                    time_basis=frame.get('wall_time_basis', 'unknown'), physical_action_confirmed=False))
    for device in photos.get('devices', []):
        if device['camera'] != archive[11:]:
            continue
        for photo in device['photos']:
            captured = photo['capture_us']
            if cutoff and captured < cutoff:
                continue
            entries.append(entry('voice_photo', None, captured, captured+1000000,
                file=reference(photo['source_ref'], 'capture_root'), time_basis=photo['time_basis'],
                timestamp_resolution_us=1000000, capture_clock_verified=False))
    entries.sort(key=lambda e: (e['start_us'] or 0, e['kind']))
    recordings = [e for e in entries if e['kind'] == 'recording']
    streams = {'video': [], 'audio': [], 'images': [], 'multimodal': []}
    for item in entries:
        kind = item['kind']
        if kind in {'original_video', 'video_segment'}:
            streams['video'].append(item)
        elif kind == 'audio':
            stt = records[item['recording_id']].get('transcription') or {}
            streams['audio'].append(item | {'transcription': {
                'status': stt.get('status'), 'outcome': stt.get('outcome'),
                'model_invocation': stt.get('model_invocation', 'NOT_PROVEN'),
                'text_file': reference(stt.get('transcript_file')),
                'content_file': reference(readable_audio.get(item['recording_id'], {}).get('content')),
                'readable_text_file': reference(readable_audio.get(item['recording_id'], {}).get('text')),
                'sentences': [e for e in entries if e['kind'] == 'speech' and e['recording_id'] == item['recording_id']]}})
        elif kind in {'key_frame', 'scene_frame', 'voice_photo'}:
            streams['images'].append(item)
        elif kind == 'multimodal_result':
            streams['multimodal'].append(item)
    return {'schema_version': 'visioncortex-file-time-index/2', 'archive': archive,
            'readable_content': index.get('readable_content', {}),
            'source_index_updated_at': index.get('updated_at'),
            'source_index': {'path': 'ProcessedClips/Index.json', 'path_base': 'device_day'},
            'cross_camera_alignment_verified': False, 'evidence_status': 'PARTIAL_EVIDENCE',
            'timezone': 'Asia/Shanghai', 'timestamp_unit': 'unix_microseconds',
            'interval_semantics': 'start_inclusive_end_exclusive',
            'path_bases': {'device_day': str(Path(config['storage']['archive_root'])/archive),
                           'capture_root': config['collection_ingest']['source_root']},
            'path_resolution': 'Resolve path against path_base; device_day may be replaced by your NAS mount of this archive.',
            'process_since_us': cutoff, 'photo_index_observed_at': photos.get('observed_at'),
            'photo_index_errors': photos.get('photo_errors', []),
            'photo_index_status': 'available' if 'observed_at' in photos else 'pending',
            'recordings': recordings, 'streams': streams}


def publish_file_index(config, index, photos=None):
    """Caller holds the device/day index lock. Publish a complete atomic snapshot."""
    name = validate_archive_name(index['archive'])
    if photos is None:
        photo_path = Path(config['storage']['local_runtime_root'])/'device-day'/'CapturePhotos'/f'{name}.json'
        photos = read_json(photo_path) if photo_path.is_file() else {}
    value = build_file_index(config, index, photos)
    root = safe_child(Path(config['storage']['archive_root']), name)
    atomic_json(root/'Comment'/'TimeIndex.json', value)
    # This is a mutable projection, so do not advertise an immutable content hash.
    index['time_index'] = {'path': 'Comment/TimeIndex.json', 'schema_version': value['schema_version']}
    return value


class FileIndexPublisher:
    def __init__(self, config):
        self.config, self.versions = config, {}
        self.source_indexes = {}
        self.runtime = Path(config['storage']['local_runtime_root'])/'device-day'
        self.last_result = {'status': 'not_started'}

    def _output_revision(self, name, index):
        from .device_day_content import time_folder
        root = safe_child(Path(self.config['storage']['archive_root']), name)
        paths = [root/'LaboratoryDailyReport'/f'LaboratoryDailyReport.{suffix}' for suffix in ('json', 'html')]
        paths.extend(root/'MultimodalUnderstanding'/time_folder(s['start_us'], s['end_us'])/'Understanding.json'
                     for s in index.get('segments', []) if s.get('start_us') and s.get('end_us'))
        result = []
        for path in paths:
            try:
                stat = path.stat()
                value = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            except FileNotFoundError:
                value = None
            result.append((str(path), value))
        return tuple(result)

    def tick(self, *, archives=None):
        import logging
        from .device_day import exclusive
        from .device_day_contract import archive_name, DeviceDayLayout, VERSION
        from .observed_inventory import read_inventory
        result = {'status': 'completed', 'published': 0, 'unchanged': 0, 'busy': 0, 'errors': []}

        def failed(name, operation, exc):
            result['status'] = 'partial'
            result['errors'].append({'archive': name, 'operation': operation, 'error_type': type(exc).__name__})
            logging.getLogger(__name__).warning('File time index unavailable for %s during %s: %s',
                                               name, operation, type(exc).__name__)
        names = {archive_name(r['camera_key'], r['recording_start_us'])
                 for r in read_inventory(self.runtime)['recordings']
                 if in_processing_scope(self.config.get('device_day', {}), r)}
        requested = None if archives is None else tuple(dict.fromkeys(validate_archive_name(n) for n in archives))
        if requested is not None:
            names.intersection_update(requested)
        photo_indexes, photo_starts = {}, {}
        cutoff = processing_cutoff(self.config.get('device_day', {}))
        for path in (self.runtime/'CapturePhotos').glob('*.json'):
            if requested is not None and path.stem not in requested:
                continue
            try:
                snapshot = read_json(path)
                for device in snapshot.get('devices', []):
                    for photo in device.get('photos', []):
                        stamp = photo['capture_us']
                        if cutoff and stamp < cutoff:
                            continue
                        name = archive_name(device['camera'], stamp)
                        if path.stem == name:
                            names.add(name)
                            photo_indexes[name] = snapshot
                            photo_starts[name] = stamp
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                failed(path.stem, 'photo_index', exc)
        self.source_indexes = {name: value for name, value in self.source_indexes.items() if name in names}
        for name in sorted(names) if requested is None else (n for n in requested if n in names):
            operation = 'photo_index'
            try:
                photo_path = self.runtime/'CapturePhotos'/f'{name}.json'
                photos = photo_indexes.get(name) or (read_json(photo_path) if photo_path.is_file() else {})
                # A five-second photo heartbeat alone must not rewrite NAS indexes.
                from .device_day_contract import digest
                photo_version = digest({k: v for k, v in photos.items() if k != 'observed_at'})
                operation = 'source_index'
                with exclusive(self.runtime/'locks'/f'{name}.index.lock'):
                    path = safe_child(Path(self.config['storage']['archive_root']), name+'/ProcessedClips/Index.json')
                    if not path.is_file() and name in photo_starts:
                        # A photo does not have to wait for a video from this device/day.
                        layout = DeviceDayLayout(Path(self.config['storage']['archive_root']), name[11:], photo_starts[name])
                        layout.create()
                        atomic_json(path, {'schema_version': VERSION, 'archive': name, 'recordings': [],
                            'segments': [], 'understandings': [], 'capture_deletion': 'disabled_by_user',
                            'evidence_status': 'PARTIAL_EVIDENCE'})
                    stat = path.stat()
                    version = (stat.st_mtime_ns, stat.st_size, photo_version)
                    content_enabled = self.config.get('device_day', {}).get('readable_content_enabled', False)
                    index = None
                    if content_enabled:
                        identity = (stat.st_mtime_ns, stat.st_size)
                        cached = self.source_indexes.get(name)
                        if cached is None or cached[0] != identity:
                            from .device_day_content import readable_source_index
                            cached = (identity, readable_source_index(read_json(path)))
                            self.source_indexes[name] = cached
                        index = cached[1]
                    if content_enabled:
                        from .device_day_content import content_revision
                        operation = 'content_revision'
                        content_version = content_revision(self.config, index)
                        version += (content_version, self._output_revision(name, index))
                    if self.versions.get(name) == version:
                        result['unchanged'] += 1
                        continue
                    index = index or read_json(path)
                    before = index.get('time_index')
                    if content_enabled:
                        from .device_day_content import publish_content, with_partial_understandings
                        from .device_day_contract import DeviceDayLayout
                        from .device_day_reports import render_day
                        source = index
                        operation = 'partial_understandings'
                        index = with_partial_understandings(self.config, index)
                        operation = 'readable_content'
                        index['readable_content'] = publish_content(self.config, index)
                        if index.get('recordings'):
                            from .device_day_night_schedule import paused_stages
                            if 'report' not in paused_stages(self.config):
                                operation = 'day_report'
                                # Failed retention rows can have no start_us.
                                # The validated archive name identifies the day
                                # directory; it does not supply a media timestamp.
                                from .device_day_timeline import day_bounds
                                day_start, _ = day_bounds(validate_archive_name(name)[:10])
                                layout = DeviceDayLayout(Path(self.config['storage']['archive_root']), name[11:],
                                    day_start)
                                render_day(layout, index)
                    operation = 'time_index'
                    publish_file_index(self.config, index, photos)
                    if before != index['time_index']:
                        operation = 'source_index_pointer'
                        if content_enabled:
                            # The cache deliberately omits dense CV evidence.
                            # Under the same day lock, change only this pointer
                            # in a fresh canonical index, never save the cache.
                            original = read_json(path)
                            original['time_index'] = index['time_index']
                        atomic_json(path, original if content_enabled else index)
                    stat = path.stat()
                    self.versions[name] = (stat.st_mtime_ns, stat.st_size, photo_version)
                    if content_enabled:
                        # Remember our own completed writes. A later canonical
                        # renderer may overwrite partial meaning with pending;
                        # changed output stats trigger restoration on next tick.
                        self.versions[name] += (content_version, self._output_revision(name, index))
                        source['time_index'] = index['time_index']
                        self.source_indexes[name] = ((stat.st_mtime_ns, stat.st_size), source)
                    result['published'] += 1
            except BlockingIOError:
                result['busy'] += 1
                continue  # Producer still publishing; retry without dropping references.
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                failed(name, operation, exc)
        self.last_result = result
        return result

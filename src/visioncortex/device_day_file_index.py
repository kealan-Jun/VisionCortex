"""Small file-only time index inside the existing Comment directory."""
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from itertools import chain
from pathlib import Path, PurePosixPath
import sys
import sqlite3
from zoneinfo import ZoneInfo

from .device_day_contract import ARTIFACT_DIRECTORIES, atomic_json, read_json, safe_child, validate_archive_name
from .device_day_schedule import in_processing_scope, processing_cutoff


_SOURCE_INDEX_CACHE_DAYS = 4
_SOURCE_INDEX_CACHE_BYTES = 128 * 1024 * 1024

_READABLE_OMIT = {
    ('recordings', None, 'processing', name) for name in ('batches', 'scan_reports', 'audit_artifacts')
} | {('segments', None, 'activity_audit')}


@contextmanager
def _index_reader(config, path, mode):
    from .device_day_io import pace_copy
    from .local_storage import require_local_path
    require_local_path(Path(config['storage']['local_runtime_root']) / 'state').mkdir(parents=True, exist_ok=True)
    # The publication caller already owns its shared NAS slot. Pace reads
    # without recursively acquiring that same finite-capacity resource.
    with path.open(mode) as handle:
        class Reader:
            def read(self, size):
                block = handle.read(size)
                if block:
                    pace_copy(config, len(block))
                return block
        yield Reader()


def read_source_index(config, path):
    from .receipt_projection import projected_json
    value = projected_json(path, omit=_READABLE_OMIT,
                           opener=lambda mode: _index_reader(config, path, mode))
    for key in ('recordings', 'segments', 'understandings'):
        value.setdefault(key, [])
    return value


def _read_source_index(config, path):
    return read_source_index(config, path)


def _source_index_size(value, limit):
    """Bound retained JSON objects, not total RSS or the active publication.

    Twice the shallow Python sizes covers container overhead conservatively;
    shared children are deliberately counted again. Walk iterators rather than
    serializing/copying the index, and abandon oversized values immediately.
    The input is an acyclic JSON projection produced by readable_source_index.
    """
    total, pending = 0, [iter((value,))]
    while pending:
        try:
            item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        total += 2 * sys.getsizeof(item)
        if total > limit:
            return total
        if isinstance(item, dict):
            pending.append(chain(item.keys(), item.values()))
        elif isinstance(item, (list, tuple)):
            pending.append(iter(item))
    return total


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
        if base == 'device_day' and path.parts[0] not in ARTIFACT_DIRECTORIES:
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
        source_entries = [{'kind': source.get('kind'), 'retained': reference(source.get('retained'))}
                          for source in row.get('sources', []) if reference(source.get('retained'))]
        capture_manifest = reference(row.get('capture_manifest'))
        entries.append(entry('recording', rid, row.get('start_us'), row.get('end_us'),
            video=reference(sources.get('video')), stages=row.get('stages', {}),
            **({'sources': source_entries} if source_entries else {}),
            **({'capture_manifest': capture_manifest} if capture_manifest else {}),
            capture_complete=row.get('capture_complete'), capture_issues=row.get('capture_issues', []),
            transcription_status=stt.get('status'), transcription_outcome=stt.get('outcome'),
            audio_status=audio.get('status'), audio_verification=audio.get('input_verification'),
            audio_source_references=audio.get('collector_sources', []),
            model_invocation=stt.get('model_invocation', 'NOT_PROVEN'),
            transcript_file=reference(stt.get('transcript_file'))))
        if sources.get('video'):
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
                time_basis=comment.get('time_basis'), accuracy='NOT_PROVEN', processing_method='automatic_stt',
                physical_action_confirmed=False))
    from .device_day_retention_projection import interval_views
    from .application.clip_delivery import enabled as clip_delivery_enabled
    for segment, meaning, interval in interval_views(index):
        rid = segment['recording_id']
        if rid not in records:
            continue
        public = readable_visual.get(segment['segment_id'], {})
        if clip_delivery_enabled(config) and not public:
            continue  # Candidate evidence is retained privately, not a duplicate clip.
        entries.append(entry('video_segment', rid, segment['start_us'], segment['end_us'],
            activity=segment['activity'], segment_id=segment['segment_id'],
            file=reference(public.get('video', segment.get('video'))), source_video=reference(segment.get('source_ref')),
            source_start_ms=segment.get('start_ms'), source_end_ms=segment.get('end_ms'),
            metadata_file=reference(public.get('metadata', segment.get('json_path'))),
            **{key: public[key] for key in ('display_category', 'display_label') if key in public},
            record_scope='retention_interval' if interval else 'preprocessing_candidate',
            candidate_segment_id=segment.get('candidate_segment_id', segment['segment_id']),
            video_value=(interval or {}).get('video_value'), selection_reason=(interval or {}).get('reason'),
            activity_category=(interval or {}).get('activity_category'),
            processing_status=(interval or {}).get('processing_status'),
            historical_scope=segment.get('historical_scope'),
            source_mapping=(interval or {}).get('source_mapping'),
            physical_action_confirmed=False))
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
            'photo_index_status': ('unavailable' if photos.get('photo_errors') else
                'available' if any(d.get('photos') for d in photos.get('devices', [])) else
                'no_input' if 'observed_at' in photos else 'pending'),
            'recordings': recordings, 'streams': streams}


def publish_file_index(config, index, photos=None):
    """Caller holds the device/day index lock. Publish a complete atomic snapshot."""
    name = validate_archive_name(index['archive'])
    if photos is None:
        photo_path = Path(config['storage']['local_runtime_root'])/'device-day'/'CapturePhotos'/f'{name}.json'
        photos = read_json(photo_path) if photo_path.is_file() else {}
    value = build_file_index(config, index, photos)
    root = safe_child(Path(config['storage']['archive_root']), name)
    field_reference = verified_field_events_reference(config, index, root)
    if field_reference:
        value['field_events_file'] = field_reference
    atomic_json(root/'Comment/Audio'/'TimeIndex.json', value)
    from .comment_layout import publish_legacy_index
    publish_legacy_index(root, 'TimeIndex.json', value)
    # This is a mutable projection, so do not advertise an immutable content hash.
    index['time_index'] = {'path': 'Comment/Audio/TimeIndex.json', 'schema_version': value['schema_version']}
    return value


def verified_field_events_reference(config, index, root):
    """Point only at the current native-published, byte-verified OCR snapshot."""
    if not (config.get('device_day', {}).get('field_recognition') or {}).get('archive_context_enabled'):
        return None
    from .device_day_contract import digest
    from .device_day_verification import verify_artifact_cached
    from .application.field_archive_context import snapshot
    path = root/'Comment/OCR/FieldEvents.json'
    try:
        if not path.is_file() or path.stat().st_size > 8*1024*1024:
            return None
        value, observed = snapshot(path)
        context = index.get('context') or {}
        expected = {'schema_version': 'visioncortex-field-archive-context/1', 'archive': index['archive'],
            'source': 'field_recognition', 'model_invoked': False, 'physical_action_verified': False,
            'status': context.get('field_event_status', 'not_received'), 'events': context.get('events', []),
            'errors': context.get('field_event_errors', []), 'source_references': context.get('field_source_references', []),
            'association_basis': 'same_camera_actual_interval_overlap_not_action_confirmation'}
        canonical = {key: item for key, item in value.items() if key != '中文摘要'}
        if canonical != expected:
            return None
        from .comment_evidence import resolve
        saved_relative = 'Comment/.Evidence/FieldRecognition/'+digest(canonical)+'.json'
        backend = Path(config['storage']['local_cache_root'])
        saved = resolve(root, saved_relative, backend)
        if not saved.is_file() or saved.stat().st_size > 8*1024*1024:
            return None
        saved_value, saved_observed = snapshot(saved)
        if saved_value != canonical:
            return None
        reference = observed | {'path': 'Comment/OCR/FieldEvents.json'}
        saved_ref = saved_observed | {'path': saved_relative}
        if verify_artifact_cached(root, reference) and verify_artifact_cached(root, saved_ref, backend_root=backend):
            return reference | {'path_base': 'device_day'}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return None


def _display_migration_pending(layout):
    """Inspect native legacy names without touching a producer's evidence.

    A new legacy directory arriving after this check is deferred to the next
    tick; callers must skip migration entirely when this returns False.
    """
    from .device_day_content_paths import _TIME_DIRECTORY
    for directory in ('ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment'):
        parent = layout.root / directory
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            legacy = (directory in {'ProcessedClips', 'MultimodalUnderstanding'} and _TIME_DIRECTORY.fullmatch(child.name)
                or directory == 'MultimodalUnderstanding' and child.name in {
                    'Understanding.json', 'UnderstandingReport.html', 'ClipUnderstanding', 'RequestPlans'}
                or directory == 'LaboratoryDailyReport' and child.name in {'Analysis', 'History'}
                or directory == 'Comment' and (child.name == 'Stt' or _TIME_DIRECTORY.fullmatch(child.name)))
            if not legacy:
                continue
            # Current transcript time directories are already public. Only
            # their old nested Recognition container requires relocation.
            if directory == 'Comment' and child.name != 'Stt':
                if (child / 'Recognition').is_dir():
                    return True
            else:
                return True
    return False


class FileIndexPublisher:
    def __init__(self, config):
        self.config, self.versions = config, {}
        self.source_indexes = OrderedDict()
        self._source_index_sizes = {}
        self.source_index_bytes = 0
        self.runtime = Path(config['storage']['local_runtime_root'])/'device-day'
        self.last_result = {'status': 'not_started'}
        self._audio_runner = None
        from .device_day_bindings import CompletedExecutionBindings
        readers = CompletedExecutionBindings(config.get('device_day') or {}).historical.values()
        self.historical_readers = {}
        for entry in readers:
            self.historical_readers.setdefault(entry['archive'], []).append(entry)

    def close(self):
        if self._audio_runner is not None and hasattr(self._audio_runner, 'close'):
            self._audio_runner.close()
        self._audio_runner = None

    def _historical_publication(self, name, index=None):
        """Only explicit readers have an additional, byte-verified proof gate."""
        entries = self.historical_readers.get(name, [])
        if not entries:
            return []
        from .device_day_historical_publication import checked, publish
        results = []
        for entry in entries:
            try:
                value = checked(self.config, entry) if index is None else publish(self.config, entry, index)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                value = {'status': 'publication_verification_pending', 'error_type': type(exc).__name__,
                         'reason': str(exc)[:200], 'automatic_action': 'retry_existing_readable_publication'}
            results.append(value | {'recording_id': entry['recording_id'], 'reader_key': entry['reader_key']})
        return results

    def _independent_audio(self, name, index, records):
        """Compose verified STT into readable views, never into canonical CV data."""
        from .device_day_inplace import verified_stt_projection
        from .device_day_contract import digest
        rows = {r['recording_id']: r for r in index.get('recordings', [])}
        changes, errors = {}, []
        backend = self.config['storage'].get('local_cache_root')
        if not backend:
            return index, (), errors
        for record in records:
            rid = record['recording_id']
            prior = rows.get(rid, {})
            if (prior.get('transcription') or {}).get('status') == 'completed':
                continue  # Its canonical producer already owns this output.
            if record.get('audio', {}).get('status') != 'provided':
                continue
            path = safe_child(Path(backend), f'device-day-receipts/{name}/{rid}/stt.json')
            if not path.is_file():
                continue
            try:
                if self._audio_runner is None:
                    from .device_day_publication_worker import publication_runner
                    self._audio_runner = publication_runner(self.config)
                value = verified_stt_projection(self._audio_runner, record)
                if value is None:
                    continue
                if prior:
                    value = prior | {key: value[key] for key in
                        ('audio', 'transcription', 'independent_audio_verified', 'video_processing_complete',
                         'transcript_span') if key in value} | {
                        'stages': prior.get('stages', {}) | value['stages']}
                changes[rid] = value
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'archive': name, 'recording_id': rid,
                    'operation': 'independent_audio_verification', 'error_type': type(exc).__name__,
                    'automatic_action': 'retry_verified_stt_publication_without_model_call'})
        if not changes:
            return index, (), errors
        merged = rows | changes
        return index | {'recordings': list(merged.values())}, digest(changes), errors

    def _request_canonical_publication(self, source, index):
        """Ask the verified journal replayer to update the canonical generation.

        The reduced readable projection is never written over dense CV data.
        Only exact current inventory/source/interval matches may request replay;
        the owning runner still checks execution keys and all artifact bytes.
        """
        from .publication_journal import PublicationJournal
        from .sqlite_store import connection
        originals = {row['segment_id']: row for row in source.get('segments', [])}
        changed = {}
        for segment in index.get('segments', []):
            previous = originals.get(segment['segment_id'])
            has_published_interval = any(row.get('video') for row in [
                *segment.get('retention_intervals', []), *(previous or {}).get('retention_intervals', [])])
            if previous and ((segment.get('video') and segment['video'] != previous.get('video'))
                    or has_published_interval and segment.get('retention_intervals') != previous.get('retention_intervals')):
                changed.setdefault(segment['recording_id'], []).append(segment)
        result = {'pending': len(changed), 'requested': 0, 'errors': []}
        inventory = self.runtime/'observed-inventory.sqlite3'
        backend = self.config['storage'].get('local_cache_root')
        if not changed:
            return result
        for rid, segments in changed.items():
            try:
                if not inventory.is_file() or not backend:
                    raise ValueError('Current inventory or retained-source receipt root is unavailable')
                with connection(inventory, readonly=True) as db:
                    row = db.execute('SELECT payload FROM observations WHERE id=?', (rid,)).fetchone()
                if row is None:
                    raise ValueError('Selected-video recording is not in the current inventory')
                import json
                record = json.loads(row['payload'])
                folder = safe_child(Path(backend), 'device-day-receipts/' + index['archive'] + '/' + rid)
                retained = read_json(folder/'retention.json')
                sealed = retained.get('recording') or {}
                fields = ('recording_id', 'source_signature', 'camera_key', 'recording_start_us', 'recording_end_us')
                if (retained.get('status') != 'completed' or not record.get('source_signature')
                        or any(record.get(k) != sealed.get(k) for k in fields)
                        or record.get('camera_key') != index['archive'][11:]):
                    raise ValueError('Selected-video source differs from the current admitted recording')
                from .application.retained_video import selection_visual
                visual = selection_visual(folder/'vision.json')
                from .device_day_content_paths import aliases, public_references
                prior_segments = {s['segment_id']: s for s in public_references(visual.get('segments', []), aliases(
                    Path(self.config['storage']['archive_root'])/index['archive']))}
                source_refs = [s.get('retained') for s in retained.get('sources', []) if s.get('kind') == 'video']
                interval_fields = ('recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us', 'activity', 'json_path')
                if (visual.get('status') != 'completed' or any(
                        segment['source_ref'] not in source_refs
                        or segment['segment_id'] not in prior_segments
                        or any(segment.get(k) != prior_segments[segment['segment_id']].get(k) for k in interval_fields)
                        for segment in segments)):
                    raise ValueError('Selected-video interval is not bound to the retained source and CV receipt')
                with connection(inventory, readonly=True) as db:
                    current = db.execute('SELECT payload FROM observations WHERE id=?', (rid,)).fetchone()
                if current is None or current['payload'] != row['payload']:
                    raise ValueError('Current inventory changed during publication admission')
                journal = PublicationJournal(self.runtime)
                token = journal.ensure_pending(record)
                if token is None:
                    raise ValueError('An existing publication request owns a different source generation')
                from .device_day_contract import digest
                journal.notice_output(rid, token, digest([
                    [s['segment_id'], s.get('video'), s.get('retention_intervals', [])] for s in segments]))
                result['requested'] += 1
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                result['errors'].append({'archive': index['archive'], 'recording_id': rid,
                    'operation': 'canonical_video_publication', 'error_type': type(exc).__name__,
                    'reason': str(exc)[:200], 'automatic_action': 'retry_existing_verified_publication'})
        return result

    def _drop_source(self, name):
        self.source_indexes.pop(name, None)
        self.source_index_bytes -= self._source_index_sizes.pop(name, 0)

    def _cached_source(self, name, identity):
        cached = self.source_indexes.get(name)
        if cached is None or cached[0] != identity:
            self._drop_source(name)
            return None
        self.source_indexes.move_to_end(name)
        return cached[1]

    def _remember_source(self, name, identity, index):
        self._drop_source(name)
        size = _source_index_size(index, _SOURCE_INDEX_CACHE_BYTES)
        if size > _SOURCE_INDEX_CACHE_BYTES or _SOURCE_INDEX_CACHE_DAYS <= 0:
            return  # Publish a large day normally, without keeping it warm.
        while self.source_indexes and (len(self.source_indexes) >= _SOURCE_INDEX_CACHE_DAYS
                or self.source_index_bytes + size > _SOURCE_INDEX_CACHE_BYTES):
            self._drop_source(next(iter(self.source_indexes)))
        self.source_indexes[name] = (identity, index)
        self._source_index_sizes[name] = size
        self.source_index_bytes += size

    def _output_revision(self, name, index):
        root = safe_child(Path(self.config['storage']['archive_root']), name)
        paths = [root/'LaboratoryDailyReport'/f'LaboratoryDailyReport.{suffix}' for suffix in ('json', 'html')]
        from .device_day_content_paths import understanding_relative
        from .device_day_content import time_folder, transcript_interval
        for record in index.get('recordings', []):
            if record.get('start_us') and record.get('end_us'):
                relative = 'Comment/Audio/' + time_folder(*transcript_interval(record))
                paths.extend(root/relative/name for name in ('Transcript.json', 'Transcript.txt'))
        paths.extend(root/understanding_relative(s)/'Understanding.json'
                     for s in index.get('segments', []) if s.get('start_us') and s.get('end_us'))
        from .device_day_retention_projection import interval_views
        paths.extend(root/understanding_relative(s)/'Understanding.json'
                     for s, _meaning, interval in interval_views(index) if interval is not None)
        from .application import clip_delivery
        if clip_delivery.enabled(self.config):
            from .device_day_contract import DeviceDayLayout
            from .device_day_timeline import day_bounds
            start, _ = day_bounds(name[:10])
            layout = DeviceDayLayout(root.parent, name[11:], max(start, 1),
                                    Path(self.config['storage']['local_cache_root']))
            for unit in clip_delivery.original_units(layout, index, directory_bindings=(
                    self.config.get('collection_ingest', {}).get('directory_camera_bindings') or ())).values():
                paths.extend(root/'ProcessedClips/Clips'/unit['name']/filename
                             for filename in ('Index.json', 'Retention.json'))
                paths.append(root/'MultimodalUnderstanding'/unit['name']/'Understanding.json')
        # The readable index also includes intervals not yet in canonical
        # evidence. Inspect their actual outputs without caching another day
        # projection or treating a missing video as confirmed no value.
        readable = root/'MultimodalUnderstanding/Index.json'
        if readable.is_file():
            if readable.stat().st_size > 16*1024*1024:
                raise ValueError('Readable interval index exceeds bounded output verification')
            entries = read_json(readable).get('entries', [])
            if len(entries) > 20000:
                raise ValueError('Readable interval entry limit exceeded')
            for entry in entries:
                relative = entry.get('content')
                if not isinstance(relative, str):
                    continue
                parts = PurePosixPath(relative).parts
                if (len(parts) != 3 or parts[0] != 'MultimodalUnderstanding'
                        or not parts[2].endswith('.json')):
                    raise ValueError('Readable interval content path differs from frozen layout')
                paths.append(safe_child(root, relative))
                if entry.get('clip_delivery_version') == clip_delivery.VERSION:
                    for field in ('metadata', 'video'):
                        reference = entry.get(field)
                        path = reference.get('path') if isinstance(reference, dict) else reference
                        if path:
                            pieces = PurePosixPath(path).parts
                            if pieces[:3] != ('ProcessedClips', 'Clips', parts[1]) or len(pieces) != 4:
                                raise ValueError('Readable interval file escapes its original-capture folder')
                            paths.append(safe_child(root, path))
                    continue
                if parts[2] != 'Understanding.json':
                    raise ValueError('Legacy understanding filename differs from frozen layout')
                folder = safe_child(root, 'ProcessedClips/Clips/' + parts[1])
                activity = {'active': 'ExperimentActivity', 'inactive': 'NoExperimentActivity'}[entry['activity']]
                names = ('Retention.json', activity+'.json', activity+'.mp4')
                paths.extend(folder/name for name in names)
        result = []
        for path in sorted(set(paths)):
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
        from .receipt_projection import ProjectionDeferred, ProjectionGenerationChanged
        result = {'status': 'completed', 'published': 0, 'unchanged': 0, 'busy': 0, 'errors': []}

        def failed(name, operation, exc):
            result['status'] = 'partial'
            result['errors'].append({'archive': name, 'operation': operation, 'error_type': type(exc).__name__})
            if isinstance(exc, ProjectionDeferred):
                result['errors'][-1].update(reason='readable_projection_limit', metric=exc.metric,
                                           observed=exc.observed, limit=exc.limit)
            elif isinstance(exc, ProjectionGenerationChanged):
                result['errors'][-1].update(exc.details())
            elif operation in {'frozen_layout_migration', 'legacy_selected_video_publication'}:
                result['errors'][-1].update(reason=str(exc)[:500],
                    automatic_action='preserve_existing_bytes_and_retry_verified_publication')
            logging.getLogger(__name__).warning('File time index unavailable for %s during %s: %s',
                                               name, operation, type(exc).__name__)
        records_by_day = {}
        for record in read_inventory(self.runtime)['recordings']:
            if in_processing_scope(self.config.get('device_day', {}), record):
                name = archive_name(record['camera_key'], record['recording_start_us'])
                records_by_day.setdefault(name, []).append(record)
        names = set(records_by_day)
        from .application.field_archive_context import archive_days
        try:
            field_days = archive_days(self.config)
        except (OSError, ValueError) as exc:
            failed('FieldRecognition', 'archive_discovery', exc)
            field_days = {}
        names.update(field_days)
        # Split publication services also own photo discovery. Sparse photos
        # do not require a video, a person binding, or the HTTP companion.
        from .device_day_timeline import day_bounds
        photo_root = Path(self.config.get('collection_ingest', {}).get('source_root', self.runtime))
        photo_cameras = set(self.config.get('collection_ingest', {}).get('camera_role_map', {}))
        photo_cameras.update(name[11:] for name in names)
        for camera in photo_cameras:
            try:
                directory = safe_child(photo_root, 'voice_photos/' + camera)
                if not directory.is_dir():
                    continue
                for day in directory.iterdir():
                    if day.is_symlink() or not day.is_dir():
                        continue
                    try:
                        start, end = day_bounds(day.name)
                        if in_processing_scope(self.config.get('device_day', {}),
                                {'recording_start_us': start, 'recording_end_us': end}):
                            names.add(archive_name(camera, start))
                    except ValueError:
                        continue
            except (OSError, ValueError) as exc:
                failed(camera, 'capture_photo_discovery', exc)
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
        for name in tuple(self.source_indexes):
            if name not in names:
                self._drop_source(name)
        for name in sorted(names) if requested is None else (n for n in requested if n in names):
            operation = 'photo_index'
            try:
                photo_path = self.runtime/'CapturePhotos'/f'{name}.json'
                photos = photo_indexes.get(name) or (read_json(photo_path) if photo_path.is_file() else {})
                import time
                photo_stamp = photos.get('observed_at', 0)
                photo_due = type(photo_stamp) not in (int, float) or time.time() - photo_stamp >= 30
                if (name[11:] in photo_cameras and photo_due
                        and self.config.get('collection_ingest', {}).get('source_root')):
                    from .device_day_time_lookup import refresh_photo_index
                    # Known device/day identity authorizes photo lookup. A
                    # photo reference needs neither a model view role nor a
                    # person binding. This local catalog is not a role grant.
                    from copy import deepcopy
                    photo_config = deepcopy(self.config)
                    photo_config['collection_ingest'].setdefault('camera_role_map', {}).setdefault(name[11:], None)
                    refresh_photo_index(photo_config, name[11:], name[:10])
                    photos = read_json(photo_path)
                for device in photos.get('devices', []):
                    stamps = [p['capture_us'] for p in device.get('photos', [])]
                    if stamps:
                        photo_starts[name] = min(stamps)
                # A five-second photo heartbeat alone must not rewrite NAS indexes.
                from .device_day_contract import digest
                photo_version = digest({k: v for k, v in photos.items() if k != 'observed_at'})
                operation = 'source_index'
                with exclusive(self.runtime/'locks'/f'{name}.index.lock'):
                    path = safe_child(Path(self.config['storage']['archive_root']), name+'/ProcessedClips/Index.json')
                    audio_records = records_by_day.get(name, [])
                    audio_index, audio_version, audio_errors = self._independent_audio(name,
                        {'archive': name, 'recordings': []}, audio_records) if not path.is_file() else (None, (), [])
                    if not path.is_file() and (name in photo_starts or name in field_days or audio_version):
                        # Independent events never create recorder-owned raw directories.
                        start = photo_starts.get(name, field_days.get(name)) or audio_records[0]['recording_start_us']
                        layout = DeviceDayLayout(Path(self.config['storage']['archive_root']), name[11:], start,
                            Path(self.config['storage']['local_cache_root']))
                        from .device_day_write_guard import guarded_mkdir
                        for folder in (layout.processed, layout.understanding, layout.reports, layout.comments):
                            guarded_mkdir(folder)
                        atomic_json(path, {'schema_version': VERSION, 'archive': name, 'recordings': [],
                            'segments': [], 'understandings': [], 'capture_deletion': 'disabled_by_user',
                            'evidence_status': 'PARTIAL_EVIDENCE'})
                    stat = path.stat()
                    version = (stat.st_mtime_ns, stat.st_size, photo_version)
                    content_enabled = self.config.get('device_day', {}).get('readable_content_enabled', False)
                    index = None
                    layout_changed = False
                    if content_enabled:
                        identity = (stat.st_mtime_ns, stat.st_size)
                        index = self._cached_source(name, identity)
                        if index is None:
                            index = _read_source_index(self.config, path)
                            self._remember_source(name, identity, index)
                    # This projection remains separate from the canonical
                    # index, including while its video is absent/still writing.
                    source_index = index or _read_source_index(self.config, path)
                    if audio_index is None:
                        audio_index, audio_version, audio_errors = self._independent_audio(name, source_index, audio_records)
                    else:
                        audio_index = source_index | {'recordings': audio_index['recordings']}
                    version += (audio_version,)
                    if audio_errors:
                        result['status'] = 'partial'
                        result['errors'].extend(audio_errors)
                    if content_enabled:
                        from contextlib import ExitStack
                        from .device_day_content_paths import migrate_display_layout, publish_selected_videos
                        from .device_day_timeline import day_bounds
                        day_start, _ = day_bounds(name[:10])
                        backend = self.config['storage'].get('local_cache_root')
                        migration_layout = DeviceDayLayout(Path(self.config['storage']['archive_root']),
                            name[11:], day_start, Path(backend) if backend else None)
                        # Nonblocking stage locks let active producers finish.
                        # The existing index lock owns this publication/move.
                        with ExitStack() as producer_locks:
                            migration = {'moved': [], 'pending': False}
                            if _display_migration_pending(migration_layout):
                                for record in sorted(index.get('recordings', []), key=lambda r: r['recording_id']):
                                    for stage in ('all', 'retention', 'vision', 'stt', 'understanding', 'report'):
                                        producer_locks.enter_context(exclusive(self.runtime/'locks'/f'{record["recording_id"]}.{stage}.lock'))
                                operation = 'frozen_layout_migration'
                                migration = migrate_display_layout(None, migration_layout, bounded_batch=True)
                            result['layout_migrated'] = result.get('layout_migrated', 0) + len(migration['moved'])
                            video_progress = {}
                            operation = 'legacy_selected_video_publication'
                            published_videos = publish_selected_videos(migration_layout, index.get('segments', []),
                                bounded_batch=True, progress=video_progress, exact_intervals_only=True)
                            result['selected_videos_published'] = result.get('selected_videos_published', 0) + len(published_videos)
                            layout_changed = bool(migration['moved'] or published_videos)
                            if migration['pending'] or video_progress['pending']:
                                result['status'] = 'partial'
                                result['layout_pending'] = True
                        from .device_day_content import content_revision
                        operation = 'content_revision'
                        from .application.field_archive_context import archive_revision
                        content_version = (content_revision(self.config, index), archive_revision(self.config, name))
                        version += (content_version, self._output_revision(name, audio_index))
                    if self.versions.get(name) == version and not result.get('layout_pending') and not layout_changed:
                        historical = self._historical_publication(name)
                        if all(row['status'] == 'recovered_published' for row in historical):
                            if historical:
                                result.setdefault('historical_publications', []).extend(historical)
                            result['unchanged'] += 1
                            continue
                    index = audio_index
                    before = index.get('time_index')
                    if content_enabled:
                        from .device_day_content import publish_content, with_partial_understandings
                        from .device_day_contract import DeviceDayLayout
                        from .device_day_reports import publish_day
                        source = source_index
                        operation = 'partial_understandings'
                        index = with_partial_understandings(self.config, index)
                        from .device_day_narrative import attach_context
                        index = attach_context(self.config, index)
                        from .application.field_archive_context import publish_context
                        publish_context(self.config, index)
                        from .application.retained_video import publish_selected
                        operation = 'selected_video_publication'
                        retention_progress = publish_selected(self.config, index)
                        canonical_progress = self._request_canonical_publication(source, index)
                        if canonical_progress['pending']:
                            result['status'] = 'partial'
                            result['canonical_video_pending'] = result.get('canonical_video_pending', 0) + canonical_progress['pending']
                            result['errors'].extend(canonical_progress['errors'])
                        if retention_progress['materialized']:
                            result['selected_videos_materialized'] = result.get('selected_videos_materialized', 0) + retention_progress['materialized']
                        if retention_progress['pending']:
                            result['status'] = 'partial'
                            result['retained_video_pending'] = result.get('retained_video_pending', 0) + retention_progress['pending']
                            result['errors'].extend(retention_progress['errors'])
                        operation = 'readable_content'
                        index['readable_content'] = publish_content(self.config, index)
                        if (index['readable_content'].get('clip_delivery') or {}).get('pending'):
                            result['status'] = 'partial'
                            result['clip_delivery_pending'] = True
                        from .application.clip_delivery import enabled as clip_delivery_enabled
                        if clip_delivery_enabled(self.config):
                            from .device_day_content_paths import aliases, public_references
                            index = public_references(index, aliases(path.parent.parent))
                        if index.get('recordings') or (index.get('context') or {}).get('events'):
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
                                publish_day(self.config, layout, index)
                    operation = 'time_index'
                    publish_file_index(self.config, index, photos)
                    if before != index['time_index']:
                        operation = 'source_index_pointer'
                        from .receipt_projection import replace_json_field
                        # Preserve dense canonical evidence while replacing
                        # only the pointer. Never hydrate the full day here.
                        replace_json_field(path, 'time_index', index['time_index'],
                            opener=lambda mode: _index_reader(self.config, path, mode))
                    historical = self._historical_publication(name, index)
                    if historical:
                        result.setdefault('historical_publications', []).extend(historical)
                        if any(row['status'] != 'recovered_published' for row in historical):
                            result['status'] = 'partial'
                            result['errors'].extend({'archive': name, 'operation': 'historical_reader_publication', **row}
                                for row in historical if row['status'] != 'recovered_published')
                    stat = path.stat()
                    self.versions[name] = (stat.st_mtime_ns, stat.st_size, photo_version, audio_version)
                    if content_enabled:
                        # Remember our own completed writes. A later canonical
                        # renderer may overwrite partial meaning with pending;
                        # changed output stats trigger restoration on next tick.
                        self.versions[name] += (content_version, self._output_revision(name, index))
                        source['time_index'] = index['time_index']
                        self._remember_source(name, (stat.st_mtime_ns, stat.st_size), source)
                        if (retention_progress['pending'] or canonical_progress['pending']
                                or (index['readable_content'].get('clip_delivery') or {}).get('pending')):
                            self.versions.pop(name, None)  # Existing publisher owns retry, without rerunning models.
                    result['published'] += 1
            except (ProjectionDeferred, ProjectionGenerationChanged) as exc:
                failed(name, operation, exc)
            except BlockingIOError:
                result['busy'] += 1
                continue  # Producer still publishing; retry without dropping references.
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                failed(name, operation, exc)
        self.last_result = result
        return result

"""The existing readable publisher seals explicit historical-reader delivery.

No queue state is changed. A producer receipt proves execution; this separate
proof proves the current readable files, including exact selected video bytes.
Only explicitly configured readers enter this path.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .device_day_contract import DeviceDayLayout, artifact, atomic_json, digest, read_json, safe_child
from .device_day_historical_reader import _document, provenance, read_bundle
from .device_day_verification import verify_artifact_cached

VERSION = 'visioncortex-historical-reader-publication/1'


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _report_business(value):
    value = deepcopy(value)
    for key in ('report_generated_at', 'input_index_digest'):
        value.pop(key, None)
    # A restored report's diagnostic records the rejected previous candidate;
    # after publication current_synthesis no longer retries that candidate.
    (value.get('全天综合理解') or {}).pop('recovery_errors', None)
    return value


def folder(config, entry):
    return safe_child(Path(config['storage']['archive_root']), entry['archive']+
        '/MultimodalUnderstanding/.Evidence/HistoricalReads/'+entry['recording_id']+'/'+entry['reader_key'])


def _scope(index, recording_id):
    records = [row for row in index.get('recordings', []) if row['recording_id'] == recording_id]
    candidates = [row for row in index.get('segments', []) if row['recording_id'] == recording_id]
    identifiers = {row['segment_id'] for row in candidates}
    meanings = [row for row in index.get('understandings', []) if row['segment_id'] in identifiers]
    _require(len(records) == 1 and candidates and len(identifiers) == len(candidates)
             and len(meanings) == len(candidates), 'historical_readable_scope_is_missing_or_duplicated')
    return {'recordings': records, 'segments': candidates, 'understandings': meanings}


def _canonical_scope(index, entry):
    scope = _scope(index, entry['recording_id'])
    # Dense diagnostics and report-only timestamps do not define this output.
    record = scope['recordings'][0]
    record_keys = ('recording_id', 'start_us', 'end_us', 'camera_key', 'sources', 'audio', 'transcription')
    segment_keys = ('segment_id', 'recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us',
                    'activity', 'json_path', 'key_frames', 'scene_frames', 'retention_intervals')
    return {'recording': {key: record.get(key) for key in record_keys},
            'stages': {name: {key: record.get('stages', {}).get(name, {}).get(key) for key in ('key', 'status')}
                       for name in ('retention', 'vision', 'stt', 'understanding')},
            'segments': [{key: row.get(key) for key in segment_keys} for row in scope['segments']],
            'understandings': scope['understandings']}


def _native_scope(layout, index, entry, bundle):
    from .device_day_content_paths import aliases, public_references
    scope = _scope(index, entry['recording_id'])
    record = scope['recordings'][0]
    retained, vision, stt, semantic = (bundle[name] for name in ('retention', 'vision', 'stt', 'understanding'))
    original = retained['recording']
    _require(record['start_us'] == original['recording_start_us'] and record['end_us'] == original['recording_end_us']
             and record.get('camera_key') == original['camera_key'] and record.get('sources') == retained['sources']
             and record.get('audio') == retained.get('audio', {'status': 'not_provided', 'artifacts': []}),
             'published_record_source_or_clock_differs_from_native_producer')
    _require(all(record.get('stages', {}).get(stage, {}).get('key') == value['key']
                 and record['stages'][stage].get('status') == 'completed' for stage, value in bundle.items()),
             'published_native_keys_or_stage_status_changed')
    expected_stt = {key: stt.get(key) for key in ('status', 'outcome', 'model_invocation', 'comments', 'chunks', 'transcript_file')}
    mapping = aliases(layout.root)
    _require(record.get('transcription') == public_references(expected_stt, mapping), 'published_transcription_changed')
    native = {row['segment_id']: row for row in public_references(vision['segments'], mapping)}
    _require(set(native) == {row['segment_id'] for row in scope['segments']}, 'published_native_candidates_changed')
    for row in scope['segments']:
        saved = native[row['segment_id']]
        _require(all(row.get(key) == saved.get(key) for key in (
            'recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us', 'activity', 'json_path')),
            'published_candidate_source_or_bounds_changed')
        for field in ('key_frames', 'scene_frames'):
            frames = [{key: value for key, value in frame.items() if key != 'understanding_text'} for frame in row.get(field, [])]
            _require(frames == saved.get(field, []), 'published_candidate_frames_changed')
    expected = public_references([row | {'historical_execution_read': provenance(entry, semantic)}
                                 for row in semantic['understandings']], mapping)
    _require(scope['understandings'] == expected, 'published_understanding_or_producer_provenance_changed')
    return scope


def snapshot(config, entry, index=None, *, anchors=None):
    """Recheck real inputs, projections and exact exports; never writes or infers."""
    from .device_day import current_recording_context
    from .device_day_content_paths import clip_name, understanding_relative
    from .device_day_file_index import _read_source_index, build_file_index, verified_field_events_reference
    from .device_day_narrative import _verified_interval_video, retention_summary
    from .device_day_content import (
        content_index, segment_documents, time_folder, transcript_documents, transcript_interval, visual_entry, with_partial_understandings,
    )
    from .device_day_reports import day_document
    from .device_day_synthesis import current_synthesis
    from .device_day_retention_projection import interval_views
    from .media.retained_intervals import partition
    from .application.retained_video import validate_selection_evidence
    from .device_day_timeline import day_bounds
    root = safe_child(Path(config['storage']['archive_root']), entry['archive'])
    start, _ = day_bounds(entry['archive'][:10])
    layout = DeviceDayLayout(root.parent, entry['archive'][11:], start, Path(config['storage']['local_cache_root']),
                             read_only_originals=True)
    def context_reader(record):
        return current_recording_context(config, layout, record)
    bundle = read_bundle(config, entry, layout, context_reader=context_reader)
    # The existing streamed reader owns production byte/event/resident caps.
    # Do not hydrate or impose the legacy buffered cap on a dense canonical day.
    canonical = _read_source_index(config, layout.index)
    index = with_partial_understandings(config, canonical) if index is None else index
    _require(index['archive'] == entry['archive'] and _canonical_scope(index, entry) == _canonical_scope(canonical, entry),
             'canonical_publication_has_not_caught_up_with_readable_selection')
    scope = _native_scope(layout, index, entry, bundle)
    record = scope['recordings'][0]
    scoped = {'archive': index['archive'], **scope, 'context': index.get('context') or {}}
    meanings = {row['segment_id']: row for row in scope['understandings']}
    files = {}

    def stable_reference(relative):
        expected = (anchors or {}).get(relative)
        if expected and verify_artifact_cached(root, expected):
            return expected
        from .comment_evidence import resolve
        path = resolve(root, relative, layout.backend_root)
        before = path.stat()
        reference = layout.artifact(path)
        after = path.stat()
        _require(all(getattr(before, key) == getattr(after, key) for key in
                     ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
                 and verify_artifact_cached(root, reference, backend_root=layout.backend_root), 'historical_user_file_changed_during_hash')
        return reference

    def json_file(relative):
        from .comment_evidence import resolve
        path = resolve(root, relative, layout.backend_root)
        _require(path.stat().st_size <= 128*1024**2, 'historical_user_document_exceeds_bound')
        value, checksum = _document(path)
        reference = {'path': relative, 'sha256': checksum, 'size_bytes': path.stat().st_size}
        _require(verify_artifact_cached(root, reference, backend_root=layout.backend_root), 'historical_user_document_changed_during_read')
        files[relative] = reference
        return value

    mu_index = json_file('MultimodalUnderstanding/Index.json')
    comment_index = json_file('Comment/Audio/Index.json')
    report = json_file('LaboratoryDailyReport/LaboratoryDailyReport.json')
    time_index = json_file('Comment/Audio/TimeIndex.json')
    for directory, document in [('Comment/Audio', comment_index), ('MultimodalUnderstanding', mu_index)]:
        _require(document == content_index(entry['archive'], directory, document['entries']),
                 'readable_content_index_metadata_or_progress_changed')
    files['ProcessedClips/Index.json'] = stable_reference('ProcessedClips/Index.json')
    counts = Counter()
    videos = []
    for candidate in scope['segments']:
        expected = partition(candidate, meanings[candidate['segment_id']], record, scoped['context'], segments=scope['segments'])
        rows = candidate.get('retention_intervals') or []
        _require(len(rows) == len(expected['intervals']), 'complete_retention_partition_not_published')
        for actual, planned in zip(rows, expected['intervals'], strict=True):
            downgraded = (planned['video_value'] == 'selected' and actual.get('video_value') == 'pending'
                and actual.get('proposed_video_value') == 'selected' and actual.get('processing_status') == 'waiting_for_evidence'
                and actual.get('reason') == 'evidence_unverified')
            _require(all(actual.get(key) == value for key, value in planned.items()
                         if not downgraded or key not in {'video_value', 'reason'}), 'published_value_partition_changed')
            counts[actual['video_value']] += 1
            if actual['video_value'] == 'selected':
                validate_selection_evidence(layout, candidate, planned, meanings[candidate['segment_id']], record, scoped['context'])
                verified = _verified_interval_video(root, candidate, actual)
                _require(verified['status'] == 'PROVEN', 'selected_video_not_actually_verified')
                videos.append(verified | {'segment_id': actual['segment_id'],
                    'source_mapping': actual['source_mapping'], 'source_ref': actual['source_ref']})
                for reference in (verified['video'], verified['receipt']):
                    files[reference['path']] = reference
    readable_entries = [row for row in mu_index.get('entries', []) if row.get('recording_id') == entry['recording_id']]
    audio_entries = [row for row in comment_index.get('entries', []) if row.get('recording_id') == entry['recording_id']]
    scoped['readable_content'] = {'multimodal': readable_entries, 'audio': audio_entries}
    views = list(interval_views(scoped))
    _require(len(readable_entries) == len(views), 'readable_interval_index_missing_or_duplicated')
    for segment, meaning, interval in views:
        content = understanding_relative(segment)+'/Understanding.json'
        _require(sum(row.get('segment_id') == segment['segment_id'] and row.get('content') == content
                     for row in readable_entries) == 1, 'readable_interval_reference_mismatch')
        value = json_file(content)
        expected_value, expected_retention = segment_documents(scoped, root, segment, meaning, interval, record, audio_entries)
        _require(value == expected_value, 'readable_understanding_business_projection_changed')
        _require([row for row in readable_entries if row['segment_id'] == segment['segment_id']]
                 == [visual_entry(expected_value, understanding_relative(segment))], 'readable_content_index_entry_changed')
        retention = json_file('ProcessedClips/Clips/'+clip_name(segment)+'/Retention.json')
        _require(retention == expected_retention, 'clips_retention_business_projection_changed')
        clip = json_file(segment['json_path'])
        _require(all(clip.get(key) == segment.get(key) for key in (
            'segment_id', 'recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us')),
            'clips_json_source_or_time_mapping_changed')
        if interval is not None:
            _require(all(clip.get(key) == expected for key, expected in segment.items())
                     and clip.get('selection_interval') == interval, 'child_clips_projection_changed')
    # A no-audio status file is allowed; no raw audio or spoken text is made up.
    audio_start, audio_end = transcript_interval(record)
    audio_folder = 'Comment/Audio/'+time_folder(audio_start, audio_end)
    _require(len(audio_entries) == 1, 'readable_native_transcript_index_missing_or_duplicated')
    entry_audio, = audio_entries
    stt = record.get('transcription') or {}
    _require(entry_audio.get('content') == audio_folder+'/Transcript.json'
             and entry_audio.get('text') == audio_folder+'/Transcript.txt'
             and all(entry_audio.get(key) == expected for key, expected in {
                 'start_us': audio_start, 'end_us': audio_end, 'status': stt.get('status'), 'outcome': stt.get('outcome')}.items()),
             'readable_native_transcript_reference_or_status_changed')
    transcript = json_file(entry_audio['content'])
    expected_transcript, expected_text = transcript_documents(entry['archive'], record, scoped['context'])
    _require(transcript == expected_transcript, 'readable_transcript_differs_from_native_stt')
    _require(entry_audio == ({key: expected_transcript[key] for key in ('recording_id', 'start_us', 'end_us', 'status', 'outcome')}
             | {'content': audio_folder+'/Transcript.json', 'text': audio_folder+'/Transcript.txt'}),
             'readable_transcript_index_entry_changed')
    text_path = entry_audio['text']
    files[text_path] = stable_reference(text_path)
    _require(safe_child(root, text_path).read_bytes() == expected_text
             and verify_artifact_cached(root, files[text_path]), 'readable_transcript_text_differs_from_native_stt')
    expected_report = day_document(layout, index, current_synthesis(layout, index))
    # These two fields are publication clocks/diagnostic serialization digests,
    # not business content. No model generation time or finding is ignored.
    _require(_report_business(report) == _report_business(expected_report),
             'daily_report_business_projection_changed')
    summary = retention_summary(scoped, root)
    _require(not summary['partition_gap_duration_ms'] and all(not row['partition_overlap_detected'] for row in summary['source_coverage']),
             'published_partition_has_gap_or_overlap')
    report_summary = report.get('视频留存统计') or {}
    _require([row for row in report_summary.get('partitions', []) if row.get('recording_id') == entry['recording_id']]
             == summary['partitions'], 'daily_selected_video_accounting_differs_from_verified_intervals')
    readable = time_index.get('readable_content') or {}
    _require(set(readable) == {'audio', 'multimodal', 'digest'}
             and sorted(readable['audio'], key=lambda row: row['start_us']) == comment_index['entries']
             and sorted(readable['multimodal'], key=lambda row: row['start_us']) == mu_index['entries']
             and readable['digest'] == digest([readable['audio'], readable['multimodal']]), 'time_index_readable_content_changed')
    photo_path = Path(config['storage']['local_runtime_root'])/'device-day/CapturePhotos'/(entry['archive']+'.json')
    photos = _document(photo_path)[0] if photo_path.is_file() else {}
    expected_files = build_file_index(config, index | {'readable_content': readable}, photos)
    field_reference = verified_field_events_reference(config, index, root)
    if (config.get('device_day', {}).get('field_recognition') or {}).get('archive_context_enabled'):
        _require(field_reference is not None, 'native_field_context_not_validly_published')
    if field_reference:
        expected_files['field_events_file'] = field_reference
        field_value = json_file(field_reference['path'])
        expected_field_bytes = {key: field_reference[key] for key in ('path', 'sha256', 'size_bytes')}
        _require(files[field_reference['path']] == expected_field_bytes,
                 'native_field_context_reference_changed_during_read')
        _require(json_file('Comment/.Evidence/FieldRecognition/'+digest(field_value)+'.json') == field_value
                 and verify_artifact_cached(root, expected_field_bytes), 'native_field_context_snapshot_changed_during_read')
    # The photo consumer's heartbeat is deliberately excluded by the owning
    # FileIndexPublisher revision. Capture times, facts and refs remain exact.
    expected_files['photo_index_observed_at'] = time_index.get('photo_index_observed_at')
    _require(time_index == expected_files, 'time_index_native_source_or_selected_interval_reference_changed')
    _require(read_bundle(config, entry, layout, context_reader=context_reader) == bundle, 'native_bundle_changed_during_publication')
    _require(len(files) <= 20000 and all(verify_artifact_cached(root, value, backend_root=layout.backend_root) for value in files.values()),
             'historical_publication_artifacts_changed_or_exceeded_bound')
    return {'schema_version': VERSION, 'archive': entry['archive'], 'recording_id': entry['recording_id'],
        'reader_key': entry['reader_key'], 'reader_identity': entry['reader']['identity'],
        'producer_native_key': entry['producer']['native_key'], 'producer_source_sha': entry['producer']['source_sha'],
        'producer_receipt_digest': entry['receipt_digest'], 'producer_context_digest': entry['producer']['context_digest'],
        'current_association_digest': entry['reader']['context_digest'], 'queue_revision': entry['queue_revision'],
        'canonical_scope_digest': digest(_canonical_scope(canonical, entry)),
        'published_scope': {'archive': entry['archive'], **deepcopy(_scope(canonical, entry['recording_id']))},
        'producer_scope': provenance(entry, bundle['understanding']), 'files': [files[key] for key in sorted(files)],
        'selected_videos': videos, 'partition_counts': dict(counts),
        'selected_duration_ms': sum(row['duration_ms'] for row in videos), 'model_invoked': False,
        'publication_scope': 'verified_readable_native_results_and_exact_selected_outputs_not_action_accuracy',
        'business_comparison_exceptions': ['report_generated_at', 'input_index_digest', '全天综合理解.recovery_errors',
                                          'TimeIndex.photo_index_observed_at'],
        'pending_value_is_preserved': bool(counts['pending']), 'queue_state_modified': False}


def publish(config, entry, index):
    """Only FileIndexPublisher calls this after all its user outputs commit."""
    value = snapshot(config, entry, index)
    root = safe_child(Path(config['storage']['archive_root']), entry['archive'])
    target = folder(config, entry)/('Publication'+digest(value)+'.json')
    if target.exists():
        _require(read_json(target) == value, 'historical_publication_evidence_collision')
    else:
        atomic_json(target, value)
    reference = artifact(root, target)
    _require(verify_artifact_cached(root, reference), 'historical_publication_proof_bytes_unverified')
    atomic_json(folder(config, entry)/'Published.json', {'schema_version': VERSION, 'receipt': reference})
    return {'status': 'recovered_published', 'receipt': reference, 'reader_key': entry['reader_key'],
            'producer_native_key': entry['producer']['native_key'], 'partition_counts': value['partition_counts']}


def checked(config, entry):
    """Fresh byte/semantic validation; a saved pathname never means success."""
    try:
        root = safe_child(Path(config['storage']['archive_root']), entry['archive'])
        pointer, _ = _document(folder(config, entry)/'Published.json')
        reference = pointer['receipt']
        expected_prefix = folder(config, entry).relative_to(root).as_posix()+'/Publication'
        _require(pointer.get('schema_version') == VERSION and verify_artifact_cached(root, reference),
                 'historical_publication_pointer_or_receipt_changed')
        _require(reference['path'].startswith(expected_prefix) and reference['path'].endswith('.json'),
                 'historical_publication_proof_crossed_its_evidence_owner')
        saved, _ = _document(safe_child(root, reference['path']))
        _require(reference['path'] == expected_prefix+digest(saved)+'.json',
                 'historical_publication_proof_content_address_changed')
        anchors = {row['path']: row for row in saved.get('files', [])}
        _require(saved == snapshot(config, entry, anchors=anchors),
                 'historical_publication_no_longer_matches_current_inputs_and_outputs')
        _require(verify_artifact_cached(root, reference), 'historical_publication_proof_changed_during_verification')
        return {'status': 'recovered_published', 'receipt': reference, 'partition_counts': saved['partition_counts']}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {'status': 'publication_verification_pending', 'reason': str(exc)[:200], 'error_type': type(exc).__name__}


def previous_published_scope(config, entry, layout):
    """Retain one proved native generation, never grant changed model inputs.

    This capability requires a previous publisher-owned immutable proof and
    validates original source/receipt bytes again. Current association changes
    are deliberately excluded from the old producer input, and the caller must
    publish a visible quarantine marker. This is not current publication proof.
    """
    from .device_day_narrative import _verified_interval_video
    from .media.retained_intervals import partition
    root = layout.root
    pointer, _ = _document(folder(config, entry)/'Published.json')
    reference = pointer['receipt']
    _require(pointer.get('schema_version') == VERSION and verify_artifact_cached(root, reference),
             'previous_historical_publication_proof_missing_or_changed')
    saved, _ = _document(safe_child(root, reference['path']))
    expected_path = (folder(config, entry)/('Publication'+digest(saved)+'.json')).relative_to(root).as_posix()
    _require(reference['path'] == expected_path and saved.get('schema_version') == VERSION
             and saved.get('reader_key') == entry['reader_key']
             and saved.get('reader_identity') == entry['reader']['identity']
             and saved.get('producer_native_key') == entry['producer']['native_key']
             and saved.get('producer_receipt_digest') == entry['receipt_digest']
             and saved.get('producer_context_digest') == entry['producer']['context_digest']
             and saved.get('current_association_digest') == entry['reader']['context_digest']
             and saved.get('queue_revision') == entry['queue_revision'], 'previous_publication_identity_changed')
    frozen_context = entry['reader']['historical_context'] | entry['reader']['additions']
    bundle = read_bundle(config, entry, layout, context_reader=lambda _: frozen_context)
    index = deepcopy(saved['published_scope'])
    _require(index.get('archive') == entry['archive']
             and digest(_canonical_scope(index, entry)) == saved.get('canonical_scope_digest'),
             'previous_published_scope_digest_changed')
    scope = _native_scope(layout, index, entry, bundle)
    meanings = {row['segment_id']: row for row in scope['understandings']}
    videos = []
    for candidate in scope['segments']:
        planned = partition(candidate, meanings[candidate['segment_id']], scope['recordings'][0],
                            frozen_context, segments=scope['segments'])['intervals']
        rows = candidate.get('retention_intervals') or []
        _require(len(rows) == len(planned), 'previous_published_partition_missing')
        for actual, expected in zip(rows, planned, strict=True):
            downgraded = (expected['video_value'] == 'selected' and actual.get('video_value') == 'pending'
                and actual.get('proposed_video_value') == 'selected' and actual.get('processing_status') == 'waiting_for_evidence'
                and actual.get('reason') == 'evidence_unverified')
            _require(all(actual.get(key) == value for key, value in expected.items()
                         if not downgraded or key not in {'video_value', 'reason'}), 'previous_published_partition_changed')
            if actual['video_value'] == 'selected':
                verified = _verified_interval_video(root, candidate, actual)
                _require(verified['status'] == 'PROVEN', 'previous_selected_video_no_longer_verified')
                videos.append(verified | {'segment_id': actual['segment_id'],
                    'source_mapping': actual['source_mapping'], 'source_ref': actual['source_ref']})
    _require(videos == saved['selected_videos'] and verify_artifact_cached(root, reference),
             'previous_published_video_or_proof_changed')
    return scope


def progress_checked(config, entry):
    """One explicit reader only; use native mount and bounded shared admission."""
    from .device_day_io import slot
    from .device_day_write_guard import storage_guard
    from .runtime_control import execution_context, resource_slot
    limited = config | {'device_day': (config.get('device_day') or {}) | {'nas_io_timeout_seconds': 2},
                        'runtime': (config.get('runtime') or {}) | {'admission_timeout_seconds': 2}}
    try:
        with storage_guard(config):
            if not (folder(config, entry)/'Published.json').is_file():
                return {'status': 'publication_verification_pending', 'reason': 'historical_publication_proof_not_yet_sealed'}
            with execution_context(job_id=entry['recording_id'], source='retention_recovery', priority=20,
                    background_fair=True), resource_slot(limited, 'cpu'), slot(limited):
                # Exact original config remains the reader identity. Only the
                # resource wait budget is bounded in the private admission copy.
                return checked(config, entry)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, TimeoutError) as exc:
        return {'status': 'publication_verification_pending', 'reason': str(exc)[:200], 'error_type': type(exc).__name__}

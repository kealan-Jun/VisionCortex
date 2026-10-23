"""Materialize a new derived generation, preserving originals and old outputs."""
from pathlib import Path
import json
import math
import time
import uuid
from copy import deepcopy

from ..device_day_contract import artifact, atomic_json, digest, read_json
from ..device_day_inputs import hash_content, identity
from ..media.retained_intervals import partition, publish_disposition


class EvidenceUnverified(ValueError):
    """Keep an interval pending when its actual model evidence cannot be proven."""


def selection_visual(path):
    """Read source-bound candidates without hydrating dense detector diagnostics.

    This view only admits file publication; it never replaces the native
    receipt, its execution-key input, or its separately verified audit bytes.
    """
    from ..receipt_projection import projected_json
    return projected_json(path, omit={('batches',), ('scan_reports',), ('audit_artifacts',),
                                      ('segments', None, 'activity_audit')})


def source_stream_clip(source, destination, start_ms, duration_ms, _encoder=None):
    """Remux the selected display interval without encoding its video or audio.

    The MP4 edit list hides decoder preroll before a non-keyframe boundary.
    Stored packet counts can therefore exceed the number of displayed frames.
    Unsupported sources fail rather than silently using a lossy fallback.
    """
    from ..video_io import _run
    from ..media.retained_intervals import EXPORT_POLICY

    def streams(path):
        result = _run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)], timeout=60)
        if result.returncode:
            raise ValueError('Cannot verify retained video stream parameters')
        return [row for row in json.loads(result.stdout)['streams'] if row['codec_type'] in ('video', 'audio')]

    original = streams(source)
    original = [next(row for row in original if row['codec_type'] == 'video'),
                *(row for row in original if row['codec_type'] == 'audio')]
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-ss', f'{start_ms / 1000:.6f}',
               '-i', str(source), '-t', f'{duration_ms / 1000:.6f}', '-map', '0:v:0', '-map', '0:a?',
               '-c', 'copy', '-movflags', '+faststart', '-use_editlist', '1', str(destination)]
    result = _run(command)
    if result.returncode:
        raise ValueError('Source-preserving retained video remux failed: '
                         + result.stderr.decode('utf-8', errors='replace')[-1000:])
    exported = streams(destination)
    fields = {'video': ('codec_name', 'profile', 'width', 'height', 'pix_fmt', 'r_frame_rate',
                        'sample_aspect_ratio', 'field_order', 'color_range', 'color_space',
                        'color_transfer', 'color_primaries'),
              'audio': ('codec_name', 'profile', 'sample_fmt', 'sample_rate', 'channels', 'channel_layout')}
    if len(original) != len(exported):
        raise ValueError('Retained video lost source streams')
    verified = []
    for before, after in zip(original, exported, strict=True):
        kind = before['codec_type']
        if kind != after['codec_type'] or any(before.get(k) != after.get(k) for k in fields[kind]):
            raise ValueError('Retained video changed source stream parameters')
        verified.append({'codec_type': kind, **{k: before[k] for k in fields[kind] if k in before}})
    return {'policy': EXPORT_POLICY, 'method': 'stream_copy', 'reencoded': False,
            'display_boundary': 'mp4_edit_list', 'verified_streams': verified,
            'packet_count_includes_decoder_preroll': True}


def _validate_activity_process(layout, candidate, selection, meaning, recording, context):
    """Authenticate process selection independently of optional model/STT work."""
    from ..device_day_audit import AuditReader
    from ..device_day_content_paths import aliases, public_references
    from ..device_day_contract import safe_child
    from ..media.retained_intervals import audited_process
    try:
        if recording.get('recording_id') != candidate['recording_id']:
            raise ValueError('Activity process belongs to a different recording')
        path = layout.receipts/candidate['recording_id']/'vision.json'
        snapshots = [(path, identity(path))]
        visual = public_references(selection_visual(path), aliases(layout.root))
        expected = ((recording.get('stages') or {}).get('vision') or {}).get('key')
        if (visual.get('stage') != 'vision' or visual.get('recording_id') != candidate['recording_id']
                or visual.get('status') != 'completed' or not visual.get('key')
                or expected and visual['key'] != expected):
            raise ValueError('Activity process lacks its completed owning vision receipt')
        fields = ('segment_id', 'recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us',
                  'activity', 'classification_basis', 'activity_audit_ref')
        saved = next(row for row in visual.get('segments', [])
                     if all(row.get(key) == candidate.get(key) for key in fields))
        if not audited_process(saved):
            raise ValueError('Activity label alone cannot select a process video')
        audit_path = safe_child(layout.backend_root, saved['activity_audit_ref']['path'])
        snapshots.append((audit_path, identity(audit_path)))
        audit = AuditReader({'storage': {'local_cache_root': str(layout.backend_root)}}).read(saved)
        if (audit.get('scope') != 'single_device_activity_not_cross_view_experiment_confirmation'
                or audit.get('coarse_only_can_label_activity') is not False):
            raise ValueError('Activity process lacks actual fine-audit evidence')
        if saved['classification_basis'] == 'fine_audit_or_local_lab_work_review_not_ground_truth':
            from ..analysis.activity_recall import RECIPE
            review = audit.get('local_activity_recall') or {}
            if (review.get('recipe') != RECIPE or review.get('status') != 'completed'
                    or review.get('physical_action_confirmed') is not False
                    or review.get('formal_events_mutated') is not False
                    or any(w.get('sample_coverage') != 'complete' for w in review.get('windows', []))
                    or [saved['start_ms'], saved['end_ms']] not in audit.get('selected_activity_intervals', [])):
                raise ValueError('Local process recall lacks its complete source-bound audit')
        if selection not in _selection_partition(saved, meaning, recording, context)['intervals']:
            raise ValueError('Process selection differs from the owning fine-audited interval')
        if any(identity(p) != before for p, before in snapshots):
            raise ValueError('Activity process evidence changed during verification')
        return snapshots
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise EvidenceUnverified(str(exc) or type(exc).__name__) from exc


def validate_selection_evidence(layout, candidate, selection, meaning, recording, context=None):
    """Revalidate actual input/response/frame bytes against the owning receipt.

    This never calls a model or writes evidence. Historical incomplete anchors
    remain readable observations, but cannot authorize a new video derivative.
    """
    if selection.get('selection_owner') == 'audited_device_activity':
        return _validate_activity_process(layout, candidate, selection, meaning, recording, context)
    from ..device_day_content_paths import aliases, public_references
    from ..device_day_contract import safe_child
    from ..device_day_verification import verify_artifact_cached
    from ..media_time import capture_us
    snapshots = {}
    try:
        mapping = aliases(layout.root)
        stage_path = layout.receipts/candidate['recording_id']/'understanding.json'
        snapshots[stage_path] = identity(stage_path)
        stage = read_json(stage_path)
        expected_key = ((recording.get('stages') or {}).get('understanding') or {}).get('key')
        if (stage.get('recording_id') != candidate['recording_id'] or stage.get('stage') != 'understanding'
                or not stage.get('key') or expected_key and stage['key'] != expected_key
                or stage.get('status') != 'completed' and not stage.get('partial_observations')):
            raise ValueError('Understanding receipt identity/status is unverified')
        visual_path = stage_path.with_name('vision.json')
        snapshots[visual_path] = identity(visual_path)
        visual = public_references(selection_visual(visual_path), mapping)
        fields = ('segment_id', 'recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us', 'activity')
        if (visual.get('status') != 'completed' or stage.get('vision_key') != visual.get('key')
                or not any(all(row.get(k) == candidate.get(k) for k in fields) for row in visual.get('segments', []))):
            raise ValueError('Understanding input differs from the owning source-bound vision receipt')
        anchors = {ref['path']: ref for ref in public_references(stage.get('artifacts') or [], mapping)}
        saved = next(row for row in public_references(stage.get('understandings') or [], mapping)
                     if row.get('segment_id') == candidate['segment_id'])
        verified = deepcopy(meaning)
        if not isinstance(verified, dict) or not verified.get('windows'):
            raise ValueError('Actual completed model windows are unavailable')
        provenance = [*(selection.get('evidence') or []), *(selection.get('context_basis') or [])]
        if (not provenance or any(not isinstance(row, dict) or not row.get('input')
                                  or not row.get('model_receipt') for row in provenance)):
            raise ValueError('Selection lacks source-bound model input/result evidence')
        required = {(row['input'], row['model_receipt']) for row in provenance}
        clock = (recording.get('processing') or {}).get('clock_mapping') or {
            'origin_us': candidate['start_us'] - round(candidate['start_ms'] * 1000)}
        origin = recording.get('recording_start_us') or recording.get('start_us') or clock['origin_us']
        def anchored(relative):
            reference = anchors.get(relative)
            if not reference or not verify_artifact_cached(layout.root, reference):
                raise ValueError('Model evidence differs from the owning stage artifact: ' + relative)
            path = safe_child(layout.root, relative)
            snapshots[path] = identity(path)
            return path
        for input_name, response_name in required:
            prior = next(window for window in saved.get('windows') or []
                         if window.get('input') == input_name and window.get('model_receipt') == response_name)
            request, response = read_json(anchored(input_name)), read_json(anchored(response_name))
            metadata, raw = request.get('metadata') or {}, response.get('model_result') or {}
            if (not isinstance(request.get('prompt'), str) or not request.get('input_key')
                    or request['input_key'] != response.get('input_key')
                    or request['input_key'] != digest([metadata, request['prompt']])
                    or raw.get('status') != 'completed'):
                raise ValueError('Actual model request/response identity is unverified')
            metadata = public_references(metadata, mapping)
            frames = metadata.get('frames') or []
            if (metadata.get('source_ref') != candidate['source_ref']
                    or metadata.get('segment_id') != candidate['segment_id'] or not frames
                    or any(metadata.get(k) != prior.get(k) for k in ('start_ms', 'end_ms'))):
                raise ValueError('Actual model input source/window differs from the selected candidate')
            for frame in frames:
                anchored(frame['path'])
                if any(frame.get(k) != anchors[frame['path']].get(k) for k in ('sha256', 'size_bytes')):
                    raise ValueError('Model frame identity differs from the owning stage artifact')
            if 'experiment_steps' in raw:
                from ..device_day_steps import validate
                parsed = validate(raw, frames, metadata['start_ms'], metadata['end_ms'], origin, clock)
            else:
                from ..device_day_models import validate_understanding
                parsed = validate_understanding(raw, frames, metadata.get('comments') or [],
                    metadata['start_ms'], metadata['end_ms'], origin, clock)
            for step in parsed.get('steps') or []:
                step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
            actual = prior | parsed | {'source_frames': frames, 'coverage': metadata.get('coverage')}
            found = False
            for number, window in enumerate(verified['windows']):
                if window.get('input') == input_name and window.get('model_receipt') == response_name:
                    verified['windows'][number] = actual
                    found = True
            if not found:
                raise ValueError('Selected model window is absent from the current understanding')
        # This legacy internal port has no verified production producer. A
        # caller's bounded-context assertion cannot authenticate itself when
        # replaying the very same assertion. Only actual returned observations
        # may establish automatic bridge/readability context in this gate.
        actual_context = {key: value for key, value in (context or {}).items()
                          if key != 'video_context_intervals'}
        rebuilt = _selection_partition(candidate, verified, recording, actual_context)
        if selection not in rebuilt['intervals']:
            raise ValueError('Actual returned model steps do not reproduce the selected interval')
        if any(identity(path) != before for path, before in snapshots.items()):
            raise ValueError('Model evidence changed during selection verification')
        return list(snapshots.items())
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise EvidenceUnverified(str(exc) or type(exc).__name__) from exc


def child_segment(candidate, interval):
    """Stable exact-source interval in the existing sibling clip namespace."""
    from ..device_day_content_paths import clip_name
    from ..device_day_contract import ACTIVITY_LABELS
    fields = ('start_ms', 'end_ms', 'start_us', 'end_us', 'activity')
    values = {key: (int(interval[key]) if key != 'activity' and interval[key] == int(interval[key])
                    else interval[key]) for key in fields}
    recipe = [candidate['segment_id'], candidate['source_ref'], *[values[k] for k in fields]]
    if interval.get('export_policy'):
        recipe.append(interval['export_policy'])
    identifier = 'segment-' + digest(recipe)[:24]
    child = {'segment_id': identifier, 'candidate_segment_id': candidate['segment_id'],
             'recording_id': candidate['recording_id'], 'source_ref': candidate['source_ref'],
             **values,
             **({'export_policy': interval['export_policy']} if interval.get('export_policy') else {}),
             'activity_category': interval.get('activity_category', 'unknown'),
             'activity_label': ACTIVITY_LABELS[interval['activity']], 'key_frames': [], 'scene_frames': []}
    activity = 'ExperimentActivity' if child['activity'] == 'active' else 'NoExperimentActivity'
    child['json_path'] = 'ProcessedClips/Clips/' + clip_name(child) + '/' + activity + '.json'
    return child


def _decision(candidate, interval):
    value = interval['video_value']
    child = child_segment(candidate, interval)
    return {'schema_version': 'visioncortex-derived-video-disposition/2',
        'selection_policy': 'visioncortex-exact-value-intervals/1',
        'decision': {'selected': 'retain_derivative', 'pending': 'pending_value_review',
                     'confirmed_no_value': 'omit_static_derivative'}[value],
        'reason': interval['reason'], 'video_value': value, 'no_value_confirmed': value == 'confirmed_no_value',
        'scope': 'exact_selected_source_interval', 'original_video_unchanged': True, 'audio_unchanged': True,
        'source_ref': candidate['source_ref'], 'source_start_ms': interval['start_ms'],
        'source_end_ms': interval['end_ms'], 'start_us': interval['start_us'], 'end_us': interval['end_us'],
        'segment_id': child['segment_id'], 'json_path': child['json_path'],
        'candidate_segment_id': candidate['segment_id'], 'candidate_source_start_ms': candidate['start_ms'],
        'candidate_source_end_ms': candidate['end_ms'], 'candidate_start_us': candidate['start_us'],
        'candidate_end_us': candidate['end_us'], 'selection': interval,
        'model_evidence': interval.get('evidence', []), 'context_basis': interval.get('context_basis', []),
        'continuous_inactivity_proven': False,
        'limitation': '视频按有依据的连续过程留存，细动作是独立的语义记录；不宣称完整实验或物理动作已获确证。'}


def _selection_partition(candidate, meaning, recording, context):
    value = partition(candidate, meaning, recording, context)
    if (value.get('candidate_segment_id') != candidate['segment_id']
            or value.get('source_ref') != candidate['source_ref']
            or value.get('source_start_ms') != candidate['start_ms']
            or value.get('source_end_ms') != candidate['end_ms']):
        raise ValueError('Value partition differs from the candidate source interval')
    cursor = candidate['start_ms']
    for item in value['intervals']:
        if (any(type(item[k]) not in (int, float) or not math.isfinite(item[k])
                for k in ('start_ms', 'end_ms', 'start_us', 'end_us'))
                or item['start_ms'] != cursor or item['end_ms'] <= cursor
                or item['end_ms'] > candidate['end_ms'] or item['start_us'] >= item['end_us']
                or item['video_value'] not in ('selected', 'pending', 'confirmed_no_value')
                or item['activity'] not in ('active', 'inactive')):
            raise ValueError('Value partition must cover the exact candidate without gaps or overlaps')
        cursor = item['end_ms']
    if cursor != candidate['end_ms']:
        raise ValueError('Value partition does not cover the complete candidate')
    return value


def published_material(layout, segment):
    """Read only a byte-verified receipt for this exact source interval."""
    from ..device_day_content_paths import private_folder
    from ..device_day_contract import safe_child
    from ..device_day_verification import verify_artifact_cached
    pointer = private_folder(layout, 'ProcessedClips', 'Retention', segment['segment_id'], 'Published.json')
    if not pointer.is_file():
        return None
    try:
        reference = read_json(pointer).get('receipt')
        if not reference or not verify_artifact_cached(layout.root, reference):
            return None
        from ..device_day_content_paths import aliases, public_references
        value = public_references(read_json(safe_child(layout.root, reference['path'])), aliases(layout.root))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    expected = {'source_ref': segment['source_ref'], 'source_start_ms': segment['start_ms'],
                'source_end_ms': segment['end_ms'], 'start_us': segment['start_us'], 'end_us': segment['end_us']}
    if (not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()) or not value.get('video')
            or not verify_artifact_cached(layout.root, value['video'])):
        return None
    return value


def _read_attempt(layout, child, decision):
    from ..device_day_content_paths import private_folder
    attempt_path = private_folder(layout, 'ProcessedClips', 'Retention', child['segment_id'], 'Attempt.json')
    try:
        previous = read_json(attempt_path) if attempt_path.is_file() else {}
        if (not isinstance(previous, dict)
                or type(previous.get('attempts', 0)) is not int or previous.get('attempts', 0) < 0
                or previous.get('next_attempt_at') is not None and (
                    type(previous['next_attempt_at']) not in (int, float)
                    or not math.isfinite(previous['next_attempt_at']))):
            raise ValueError('Invalid retained-video attempt receipt')
        return attempt_path, previous, None
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        error = {'segment_id': child['segment_id'], 'recording_id': child['recording_id'],
                 'candidate_segment_id': decision['candidate_segment_id'],
                 'error_type': type(exc).__name__, 'reason': 'retention_attempt_receipt_invalid',
                 'automatic_action': 'isolate_interval_until_attempt_receipt_restored'}
        try:
            from ..device_day import copy_verified
            before = identity(attempt_path)
            checksum = hash_content(attempt_path)
            backup = private_folder(layout, 'ProcessedClips', 'Retention', child['segment_id'],
                                    'InvalidAttempts', checksum + '.json')
            copied = copy_verified(attempt_path, backup)
            if copied['sha256'] != checksum or identity(attempt_path) != before:
                raise ValueError('Invalid attempt receipt changed during preservation')
            preserved = artifact(layout.root, backup)
            recovery = {'schema_version': 'visioncortex-retention-attempt-recovery/1',
                'source_path': layout.relative(attempt_path), 'source_identity': before,
                'preserved_reference': preserved, 'expected_interval': decision,
                'evidence_status': 'NOT_PROVEN', 'model_invoked': False, 'error': error}
            record_path = backup.parent/('Recovery' + digest(recovery) + '.json')
            if not record_path.exists():
                atomic_json(record_path, recovery)
            elif read_json(record_path) != recovery:
                raise ValueError('Invalid attempt recovery record collision')
            error['preserved_reference'] = preserved
        except (OSError, ValueError, KeyError, TypeError) as preservation:
            error['preservation_error_type'] = type(preservation).__name__
        return attempt_path, {}, error


def _pending_evidence(report, row, child, candidate, decision, exc, *, error=None, emit_error=True):
    error = error or {'segment_id': child['segment_id'], 'candidate_segment_id': candidate['segment_id'],
             'recording_id': child['recording_id'], 'error_type': type(exc).__name__,
             'reason': 'evidence_unverified', 'detail': str(exc)[:500],
             'automatic_action': 'preserve_observation_and_recheck_owning_evidence'}
    if emit_error:
        report['errors'].append(error)
    report['pending'] += 1
    row.update(video_value='pending', proposed_video_value='selected', processing_status='waiting_for_evidence',
               reason='evidence_unverified', error=error,
               derived_video_disposition=decision | {'decision': 'pending_value_review', 'video_value': 'pending',
                   'reason': 'evidence_unverified', 'evidence_status': 'NOT_PROVEN'})


def _defer_attempt(path, previous, key, error):
    attempts = previous.get('attempts', 0) + 1 if previous.get('decision_digest') == key else 1
    retry_at = time.time() + min(300, 5 * 2 ** min(attempts - 1, 6))
    attempt = {'status': 'failed', 'decision_digest': key,
               'attempts': attempts, 'next_attempt_at': retry_at, 'error': error}
    atomic_json(path.parent/'Attempts'/(digest(attempt) + '.json'), attempt)
    atomic_json(path, attempt)
    return retry_at


def publish_selected(config, index, *, recording_id=None, maximum_new_videos=1, evidence_validator=None):
    """Publish selected process intervals independently of reports and identity binding.

    Every candidate contributes to the day projection. ``recording_id`` limits
    new materialization only; it must not hide other recordings' verified
    published intervals. The existing publisher owns bounded retries.
    """
    from ..device_day_contract import DeviceDayLayout
    from ..device_day_content_paths import private_folder, understanding_relative
    from ..device_day_inputs import source_path
    from ..device_day_verification import verify_artifact_cached
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    meanings = {m['segment_id']: m for m in index.get('understandings', [])}
    root = Path(config['storage']['archive_root'])
    report = {'materialized': 0, 'pending': 0, 'errors': [], 'selected_intervals': []}
    new_attempts = 0
    latest = config.get('device_day', {}).get('latest_first', False)
    candidates = index.get('segments', [])
    interval_collections = []
    if latest:
        def capture_time(row):
            value = row.get('start_us')
            return value if type(value) in (int, float) and math.isfinite(value) else float('-inf')
        candidates = sorted(candidates, key=capture_time, reverse=True)
    for candidate in candidates:
        record = records.get(candidate['recording_id'])
        if not record or not candidate.get('start_us') or not candidate.get('end_us'):
            continue
        backend = config['storage'].get('local_cache_root')
        layout = DeviceDayLayout(root, index['archive'][11:], candidate['start_us'], Path(backend) if backend else None)
        record = record | {'camera_key': index['archive'][11:]}
        meaning = meanings.get(candidate['segment_id'])
        try:
            grouped = _selection_partition(candidate, meaning, record, index.get('context'))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report['errors'].append({'segment_id': candidate['segment_id'], 'recording_id': candidate['recording_id'],
                'error_type': type(exc).__name__, 'reason': str(exc)[:500],
                'automatic_action': 'isolate_invalid_value_partition'})
            report['pending'] += 1
            continue
        # An old broad video is historical evidence, never proof that every
        # instant in this candidate was selected by the exact-interval policy.
        prior = published_material(layout, candidate)
        historical = candidate.get('video') or (prior or {}).get('video')
        if historical:
            candidate['video'] = historical
            candidate['historical_scope'] = {'source_start_ms': candidate['start_ms'],
                'source_end_ms': candidate['end_ms'], 'video': historical,
                'selection_status': 'historical_retained_not_revalidated_for_exact_selection'}
        rows = []
        candidate['retention_intervals'] = rows
        interval_collections.append(rows)
        selections = grouped['intervals']
        if latest:
            selections = sorted(selections, key=lambda row: row['start_us'], reverse=True)
        for selection in selections:
            child = child_segment(candidate, selection)
            row = selection | child | {'child_id': child['segment_id'],
                'understanding_path': understanding_relative(child) + '/Understanding.json',
                'processing_status': 'queued' if selection['video_value'] == 'selected' else
                                     'waiting_for_value_review' if selection['video_value'] == 'pending' else 'completed'}
            rows.append(row)
            decision = _decision(candidate, selection)
            row['derived_video_disposition'] = decision
            if selection['video_value'] != 'selected':
                continue
            report['selected_intervals'].append(row)
            prior = published_material(layout, child)
            if (prior and prior.get('candidate_segment_id') == candidate['segment_id']
                    and prior.get('selection_policy') == 'visioncortex-exact-value-intervals/1'
                    and prior.get('selection') == selection):
                row.update(video=prior['video'], source_mapping=prior.get('source_mapping'),
                    derived_video_disposition=prior, processing_status='completed')
                pointer = private_folder(layout, 'ProcessedClips', 'Retention', child['segment_id'], 'Published.json')
                row['receipt'] = read_json(pointer)['receipt']
                continue
            if recording_id is not None and candidate['recording_id'] != recording_id:
                continue  # Reuse verified output above, without exporting another source.
            if not config.get('device_day', {}).get('derived_video_retention', False):
                continue
            attempt_path, previous_attempt, invalid = _read_attempt(layout, child, decision)
            if invalid:
                report['errors'].append(invalid)
                report['pending'] += 1
                row.update(processing_status='failed', error=invalid)
                continue
            decision_key = digest(decision)
            if (previous_attempt.get('decision_digest') == decision_key
                    and (previous_attempt.get('next_attempt_at') or 0) > time.time()):
                error = previous_attempt.get('error') or {}
                if error.get('reason') == 'evidence_unverified':
                    _pending_evidence(report, row, child, candidate, decision, None, error=error, emit_error=False)
                else:
                    report['pending'] += 1
                    row.update(processing_status='failed', error=error)
                row['retry_at'] = previous_attempt['next_attempt_at']
                continue
            try:
                (evidence_validator or validate_selection_evidence)(layout, candidate, selection, meaning, record, index.get('context'))
            except EvidenceUnverified as exc:
                _pending_evidence(report, row, child, candidate, decision, exc)
                row['retry_at'] = _defer_attempt(attempt_path, previous_attempt, decision_key, row['error'])
                continue
            if new_attempts >= maximum_new_videos:
                report['pending'] += 1
                continue
            new_attempts += 1
            try:
                from ..runtime_control import resource_slot
                with resource_slot(config, 'cpu'), resource_slot(config, 'storage'):
                    result = materialize(layout, source_path(layout, {'retained': child['source_ref']}),
                        child, meaning, record, index.get('context'), selection=selection, candidate=candidate,
                        evidence_validator=evidence_validator,
                        private_delivery=config.get('device_day', {}).get('clip_delivery') == 'source_recording_v2')
                if not result.get('video') or not verify_artifact_cached(layout.root, result['video']):
                    raise ValueError('Selected interval lacks a verified publication artifact')
                manifest = private_folder(layout, 'ProcessedClips', 'Retention', child['segment_id'],
                                          digest(decision)[:24], 'Retention.json')
                reference = artifact(layout.root, manifest)
                atomic_json(private_folder(layout, 'ProcessedClips', 'Retention', child['segment_id'], 'Published.json'),
                            {'schema_version': 'visioncortex-retained-video-publication/1', 'receipt': reference})
                row.update(video=result['video'], derived_video_disposition=result, receipt=reference,
                           source_mapping=result['source_mapping'], processing_status='completed')
                if previous_attempt:
                    atomic_json(attempt_path, previous_attempt | {'status': 'completed', 'next_attempt_at': None})
                report['materialized'] += 1
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                from ..runtime_control import ExecutionCancelled
                if isinstance(exc, ExecutionCancelled):
                    raise
                if isinstance(exc, EvidenceUnverified):
                    _pending_evidence(report, row, child, candidate, decision, exc)
                    row['retry_at'] = _defer_attempt(attempt_path, previous_attempt, decision_key, row['error'])
                    continue
                error = {'segment_id': child['segment_id'], 'candidate_segment_id': candidate['segment_id'],
                         'recording_id': child['recording_id'], 'error_type': type(exc).__name__, 'reason': str(exc)[:500]}
                report['errors'].append(error)
                report['pending'] += 1
                retry_at = _defer_attempt(attempt_path, previous_attempt, decision_key, error)
                row.update(processing_status='failed', error=error, retry_at=retry_at)
    report['selected_intervals'] = [row for row in report['selected_intervals'] if row['video_value'] == 'selected']
    if latest:
        # Export budget follows processing priority. Persisted partitions and
        # readable results retain their ordinary chronological presentation.
        for rows in interval_collections:
            rows.sort(key=lambda row: row['start_ms'])
        report['selected_intervals'].sort(key=lambda row: row['start_us'])
    return report


def materialize(layout, source, segment, meaning, recording, context=None, *, segments=None,
                extractor=None, selection=None, candidate=None, evidence_validator=None, private_delivery=False):
    """The caller owns source resolution and verified storage admission.

    Only a positively selected interval creates a new derived video.
    This function never deletes any input or replaces a historical generation.
    """
    from ..video_io import probe_video
    from ..device_day_content_paths import clip_name, layout_contract, private_folder
    from ..device_day_verification import verify_artifact_cached
    extractor = extractor or source_stream_clip
    source = Path(source)
    candidate = candidate or segment
    grouped = _selection_partition(candidate, meaning, recording, context)
    selected = [row for row in grouped['intervals'] if row['video_value'] == 'selected']
    if selection is None:
        if not selected:
            # Compatibility for direct callers: a nonselected candidate never
            # reads its source and cannot become a full-length copy.
            value = 'confirmed_no_value' if all(row['video_value'] == 'confirmed_no_value'
                                              for row in grouped['intervals']) else 'pending'
            interval = {'start_ms': candidate['start_ms'], 'end_ms': candidate['end_ms'],
                'start_us': candidate['start_us'], 'end_us': candidate['end_us'],
                'activity': candidate['activity'], 'video_value': value,
                'reason': '候选区间没有可导出的已确认精确价值范围；完整分区与原片映射保留。',
                'evidence': [], 'context_basis': []}
            decision = _decision(candidate, interval) | {'partition': grouped}
            folder = private_folder(layout, 'ProcessedClips', 'Retention', candidate['segment_id'], digest(decision)[:24])
            return publish_disposition(folder, decision)
        if len(selected) != 1:
            raise ValueError('Materialization requires one explicit selected interval; disjoint intervals cannot be concatenated')
        selection = selected[0]
        segment = child_segment(candidate, selection)
    else:
        matches = [row for row in selected if all(selection.get(k) == v for k, v in row.items())]
        if len(matches) != 1:
            raise ValueError('Explicit selection is not a current positively selected exact source interval')
        selection = matches[0]
        expected_child = child_segment(candidate, selection)
        if any(segment.get(k) != expected_child[k] for k in (
                'segment_id', 'candidate_segment_id', 'recording_id', 'source_ref', 'start_ms', 'end_ms',
                'start_us', 'end_us', 'activity', 'json_path')):
            raise ValueError('Materialization child differs from the exact selected interval')
    decision = _decision(candidate, selection)
    folder = private_folder(layout, 'ProcessedClips', 'Retention', segment['segment_id'], digest(decision)[:24])
    manifest = folder / 'Retention.json'
    if decision['decision'] != 'retain_derivative':
        return publish_disposition(folder, decision)
    evidence = (evidence_validator or validate_selection_evidence)(layout, candidate, selection, meaning, recording, context) or []
    before = identity(source)
    if hash_content(source) != segment['source_ref']['sha256']:
        raise ValueError('Derived video source differs from the sealed original')
    if manifest.is_file():
        saved = read_json(manifest)
        if all(saved.get(k) == v for k, v in decision.items()) and saved.get('video'):
            if verify_artifact_cached(layout.root, saved['video']):
                return saved
    from ..device_day_write_guard import guarded_mkdir, verify_before_submission
    guarded_mkdir(folder)
    activity = layout_contract()['activities'][segment['activity']]
    from ..device_day_content_paths import producer_clip_folder
    target = producer_clip_folder(layout, clip_name(segment), private=private_delivery)/(activity+'.mp4')
    if target.exists():
        # Recover only our exact, durable pre-rename intent. An arbitrary
        # existing clip or an unrelated source/window must never be adopted.
        binding = ('source_ref', 'source_start_ms', 'source_end_ms', 'start_us', 'end_us',
                   'candidate_segment_id', 'candidate_source_start_ms', 'candidate_source_end_ms',
                   'candidate_start_us', 'candidate_end_us', 'segment_id', 'json_path', 'selection_policy')
        for prepared in sorted(folder.parent.glob('*/Prepared.json')):
            try:
                saved = read_json(prepared)
                video, mapping = saved.get('video') or {}, saved.get('source_mapping') or {}
                if (saved.get('decision') != 'retain_derivative'
                        or any(saved.get(k) != decision[k] for k in binding)
                        or video.get('path') != layout.relative(target)
                        or mapping.get('source_start_ms') != segment['start_ms']
                        or mapping.get('source_end_ms') != segment['end_ms']
                        or mapping.get('source_start_us') != segment['start_us']
                        or mapping.get('source_end_us') != segment['end_us']
                        or mapping.get('derived_start_ms') != 0
                        or not 0 <= mapping.get('duration_tolerance_ms', -1) <= 100
                        or abs(mapping.get('derived_end_ms', -10000) - (segment['end_ms'] - segment['start_ms']))
                           > mapping.get('duration_tolerance_ms', 1)
                        or not verify_artifact_cached(layout.root, video)):
                    continue
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
            value = publish_disposition(folder, decision, video_present=True) | {
                'video': video, 'source_mapping': mapping, 'prepared_receipt': artifact(layout.root, prepared),
                **({'video_encoding': saved['video_encoding']} if saved.get('video_encoding') else {})}
            atomic_json(manifest, value)
            return value
        raise ValueError('Unbound clip output exists; refusing to overwrite historical media')
    guarded_mkdir(target.parent)
    temporary = target.parent / ('.' + uuid.uuid4().hex + '.mp4')
    try:
        verify_before_submission()
        encoding = extractor(source, temporary, segment['start_ms'], segment['end_ms'] - segment['start_ms'], 'copy')
        info = probe_video(temporary)
        expected = segment['end_ms'] - segment['start_ms']
        fps = getattr(info, 'fps', None)
        tolerance = min(100, 1000 / fps + 1) if isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0 else 1
        if any(identity(path) != stamp for path, stamp in evidence):
            raise EvidenceUnverified('Model evidence changed during video extraction')
        if abs(info.duration_ms - expected) > tolerance or identity(source) != before:
            raise ValueError('Derived video duration or original source changed')
        verify_before_submission()
        sealed = artifact(layout.root, temporary) | {'path': layout.relative(target)}
        source_mapping = {'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
                          'source_start_us': segment['start_us'], 'source_end_us': segment['end_us'],
                          'derived_start_ms': 0, 'derived_end_ms': info.duration_ms,
                          'duration_tolerance_ms': tolerance}
        if any(identity(path) != stamp for path, stamp in evidence):
            raise EvidenceUnverified('Model evidence changed before exact video publication')
        if identity(source) != before:
            raise ValueError('Original source changed before exact video publication')
        # Publish immutable bytes/source intent before the atomic media rename.
        # A process death after rename can finish publication without extraction.
        encoded = {'video_encoding': encoding} if encoding else {}
        atomic_json(folder/'Prepared.json', decision | {'video': sealed, 'source_mapping': source_mapping, **encoded})
        temporary.replace(target)
        value = publish_disposition(folder, decision, video_present=True) | {'video': artifact(layout.root, target),
            'source_mapping': source_mapping, **encoded}
        atomic_json(manifest, value)
        return value
    finally:
        temporary.unlink(missing_ok=True)

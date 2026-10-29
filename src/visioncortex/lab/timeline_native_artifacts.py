"""Candidate Timeline activities in the existing Comment/OCR and Audio schemas.

No producer artifact is replaced. Every event directory is source/content pinned;
OCR tokens do not become attributed six-field measurements and a transcript does
not become a reviewed command or a forced word alignment.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo

from .components import pipeline_module
from .domain import clean, fingerprint, instant
from .timeline_clock import source_clock_supported


def _epoch(at):
    from datetime import datetime, timezone
    delta = at - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return delta.days * 86400000000 + delta.seconds * 1000000 + delta.microseconds


def _provenance(core, activities):
    ids = set()
    for a in activities:
        try:
            ids.add(UUID((a.get('payload') or {})['task_id']))
        except (ValueError, KeyError, TypeError):
            pass
    with core.db.transaction(readonly=True) as conn:
        tasks = conn.execute('SELECT id,kind,status,model_revision,request,result FROM hj_timeline_task WHERE customer_id=%s AND id=ANY(%s)',
                             (core.customer, list(ids))).fetchall() if ids else []
    return {str(t['id']): clean(t) for t in tasks}


def artifact_documents(core, *, archive, camera, activities, media_sources, rules):
    """Return immutable relative-path JSON/text plans; the original guard writes."""
    if len(activities) > 512:
        raise ValueError('Candidate Comment artifact plan exceeds 512 activities')
    tasks = _provenance(core, activities)
    files, references = {}, []
    for activity in activities:
        if activity['source_type'] not in {'ocr', 'speech'}:
            continue
        payload = activity.get('payload') or {}
        sources = {str(e['media_id']): media_sources[str(e['media_id'])] for e in activity.get('evidence', [])
                   if str(e['media_id']) in media_sources}
        if not sources:
            continue
        task = tasks.get(payload.get('task_id'))
        if task and (task['status'] != 'completed' or activity['id'] not in (task['result'] or {}).get('activity_ids', [])):
            raise ValueError('Comment activity differs from its completed original task')
        model = dict(model_revision=activity.get('model_revision'), task_id=payload.get('task_id'),
                     execution_verification=payload.get('execution_verification', 'supplied_not_independently_verified'),
                     accuracy='NOT_PROVEN', actual_task=task)
        identity = fingerprint(dict(activity=activity, sources=sources, model=model, publisher_rules=rules))
        stamp = 'UnknownTime'
        known_event_time = bool(activity.get('start_at')) and all(source_clock_supported(s.get('time_basis'),s.get('source_kind')) for s in sources.values())
        if known_event_time:
            stamp = instant(activity['start_at']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%H-%M-%S.%f')[:-3]
        event_id = 'timeline:' + activity['id'] + ':' + identity[:16]
        common = clean(dict(event_id=event_id, camera_id=camera, timeline_activity_id=activity['id'],
            record_scope='candidate', human_verified=False, business_confirmation=False,
            processing_rules=rules, model=model, evidence=activity.get('evidence', []), source_refs=sources,
            event_at=activity.get('start_at') if known_event_time else None,
            source_reported_event_at=(activity.get('source_reported_interval') or {}).get('start_at',activity.get('start_at')),
            capture_time_status='verified' if known_event_time else 'unknown_or_uncalibrated',
            archive_day_is_not_physical_event_time=not known_event_time,
            folder_time_basis='verified_capture_time' if known_event_time else 'unknown_capture_time_with_receipt_day_partition',
            source_time_is_not_implicitly_verified=True,
            physical_action_confirmed=False, source_text_is_untrusted_evidence=True))
        if activity['source_type'] == 'ocr':
            category = 'VideoReadings' if any(e.get('kind') == 'video' for e in activity.get('evidence', [])) else 'VoicePhotoReadings'
            folder = f'{archive}/Comment/OCR/{category}/{stamp}_timeline-{activity["id"][:8]}-{identity[:12]}'
            source_cards, photos = [], {}
            for source in sources.values():
                asset = dict(path=source.get('source_relative_path'), sha256=source.get('source_sha256'),
                    byte_size=source.get('byte_size'), kind=source.get('kind'), path_base='archive_root',
                    source_bytes_copied=False, sha256_basis='verified_media_asset_at_intake',
                    availability_status='not_rechecked_by_derived_view')
                photos[source.get('source_sha256')] = asset
                source_cards.append(dict(source, camera_id=camera, captured_at=source.get('source_reported_at'),
                    capture_time_basis=source.get('time_basis'), received_at=source.get('capture_received_at'),
                    original=asset, image=asset if source.get('kind') == 'photo' else None,
                    operator=None, panel_regions=[]))
            readings = []
            for index, field in enumerate(payload.get('fields', [])):
                # Preserve per-field physical identity only when its own source
                # registration/context supports it; current equipment is absent.
                bound = field.get('association_status') == 'spatial_identity_supported'
                iid = (field.get('dynamic_context') or {}).get('instrument_id') if bound else None
                readings.append(dict(reading_id=activity['id'] + ':' + str(index + 1),
                    value=field.get('value'), display_value=field.get('value_text', field.get('value')),
                    unit=field.get('unit'), instrument_id=iid, candidate_instrument_id=activity.get('instrument_id'),
                    measurement_name=field.get('field_key'), status='needs_review', result_stage='candidate',
                    human_verified=False, field_status=field.get('status'), raw_field=field,
                    quality_issues=field.get('quality_issues', []), identity_basis='per_field_source_context' if iid else 'unbound'))
            for index, token in enumerate(payload.get('raw_tokens', [])):
                readings.append(dict(reading_id=activity['id'] + ':raw:' + str(index + 1),
                    text=token.get('text'), value=None, display_value=token.get('text'), unit=None,
                    instrument_id=None, measurement_name=None, status='needs_review', result_stage='unmapped_raw_ocr',
                    human_verified=False, raw_token=token, quality_issues=['unmapped_raw_token_not_a_confirmed_numeric_field']))
            for reading in readings:
                reading.update(job_id=payload.get('task_id'), capture_id=next(iter(sources.values())).get('capture_id') if len(sources) == 1 else None,
                    raw_text=reading.get('text', activity.get('text_content')), raw_value=reading.get('value'),
                    raw_unit=reading.get('unit'))
            evidence = dict(common, schema='field-recognition-event/2', derived_view=True, category=category,
                status='observations_retained', sources=source_cards, photos=photos, measurements=[], measurement_files=[],
                observations=[dict(activity=activity, task=task)], decision=None,
                six_field_measurements_generated=False, current_equipment_used_for_panel_identity=False)
            files[folder + '/Readings.json'] = dict(common, schema='field-numeric-observations/1',
                aggregation='per_source_field_or_raw_token_not_merged', readings=readings, evidence='Evidence.json')
            files[folder + '/Evidence.json'] = evidence
            unbound = [r for r in readings if r['instrument_id'] is None]
            if unbound or not readings:
                files[folder + '/UnboundReadings.json'] = dict(common, schema='field-unbound-readings/1',
                    derived_view=True, title='未绑定读数候选（非正式测量结果）', instrument_id=None, qr_hash=None,
                    status='needs_review', aggregation='per_source_candidates_not_merged', readings=unbound,
                    evidence=folder + '/Evidence.json')
            references.append(dict(activity_id=activity['id'], source_type='ocr', readings=folder + '/Readings.json',
                evidence=folder + '/Evidence.json', unbound_readings=folder + '/UnboundReadings.json' if unbound or not readings else None))
        else:
            folder = f'{archive}/Comment/Audio/{stamp}_timeline-{activity["id"][:8]}-{identity[:12]}'
            units = payload.get('aligned_units') or []
            aligned = bool(units)
            if len(units) > 10000:
                raise ValueError('Transcript candidate exceeds bounded original alignment units')
            if not units:
                refs = activity.get('evidence') or []
                units = [dict(text=activity.get('text_content', ''), start_seconds=refs[0].get('start_seconds') if refs else None,
                              end_seconds=refs[0].get('end_seconds') if refs else None)]
            source_id = next(iter(sources)) if len(sources) == 1 else None
            source = sources.get(source_id, {})
            known = source_id is not None and source_clock_supported(source.get('time_basis'),source.get('source_kind')) and source.get('source_reported_at') is not None
            sentences = []
            for index, unit in enumerate(units):
                left, right = unit.get('start_seconds'), unit.get('end_seconds')
                start_us = end_us = None
                if known and left is not None and right is not None:
                    base = instant(source['source_reported_at'])
                    start_us = _epoch(base + timedelta(microseconds=int(Decimal(str(left)) * 1000000)))
                    end_us = _epoch(base + timedelta(microseconds=int(Decimal(str(right)) * 1000000)))
                sentences.append(dict(comment_id=activity['id'] + ':' + str(index), text=unit['text'],
                    start_us=start_us, end_us=end_us, source_start_seconds=left, source_end_seconds=right,
                    source_media_id=source_id, source_sha256=source.get('source_sha256'), speaker=None))
            start_us, end_us = sentences[0]['start_us'], sentences[-1]['end_us']
            if known and start_us is not None and end_us is not None:
                value, text = pipeline_module('device_day_content').transcript_documents(archive,
                    dict(recording_id=source_id, start_us=start_us, end_us=end_us,
                         audio=dict(status='provided', source_ref=source), transcription=dict(status='completed',
                         outcome='transcribed_candidate', comments=sentences, model_invocation='PROVEN' if task and payload.get('execution_verification', '').startswith('local_') else 'NOT_PROVEN')))
            else:
                value = dict(schema_version='visioncortex-readable-transcript/1', archive=archive, device=camera,
                    date=archive[:10], timezone='Asia/Shanghai', timestamp_unit='unix_microseconds', path_base='device_day',
                    recording_id=source_id, start_us=None, end_us=None, sentences=sentences, audio=source,
                    status='completed', outcome='transcribed_candidate_source_clock_unknown', accuracy='NOT_PROVEN',
                    content_source='machine_transcribed_speech', processing_method='automatic_stt',
                    model_invocation='PROVEN' if task and payload.get('execution_verification', '').startswith('local_') else 'NOT_PROVEN')
                text = ('源时钟未校准；以下为机器转写候选，非实验指令或真值。\n' +
                    '\n'.join(f'[{u["source_start_seconds"]}..{u["source_end_seconds"]} source seconds] {u["text"]}' for u in sentences) + '\n').encode()
            value.update(common, alignment_status='source_relative_aligned_candidate' if aligned else 'alignment_unknown',
                original_alignment=payload.get('asr_alignment'), aligned_units=payload.get('aligned_units'),
                word_alignment_is_forced_alignment=False, speaker_attribution='unknown_not_inferred',
                comment_interpretation=payload.get('comment_candidate'), source_sample_coverage=payload.get('source_sample_coverage'))
            files[folder + '/Transcript.json'] = value
            files[folder + '/Transcript.txt'] = text
            references.append(dict(activity_id=activity['id'], source_type='speech', transcript=folder + '/Transcript.json',
                                   text=folder + '/Transcript.txt', alignment_status=value['alignment_status']))
    return files, references

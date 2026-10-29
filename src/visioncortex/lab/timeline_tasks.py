"""Model-neutral local worker boundary, with immutable requests and leased results.

This module schedules work; it never fabricates inference or calls an external API.
The chosen local model runner obtains tasks and supplies structured observations.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import json

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident, instant
from .timeline import Timeline, VERSION, offset_time, seconds
from .rules import Rules
from .retained_clips import RetainedClips,original_spans


class TimelineTasks:
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.core, self.customer = timeline.core, timeline.customer

    def register_video_qr(self, key, media_id, registration, *, actor, reason):
        """Explicit model-free original-video small windows; never auto-admit a source."""
        from .qr_video import validate_registration
        if not actor or not reason:
            raise ValueError('Video QR registration author and reason required')

        def action(conn):
            media = conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, ident(media_id))).fetchone()
            if not media:
                raise Missing('Video QR outside customer scope')
            asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, ident(media_id))).fetchone()
            validate_registration(registration, media=media, asset=asset)
            revision = 'video-qr:' + registration['revision'] + ':' + fingerprint(registration)[:16]
            pending = dict(schema_version='vc-lab-video-qr-registration/1', video_qr=registration,
                           registration_provenance=dict(actor=actor, reason=reason))
            task_id = ident()
            conn.execute("INSERT INTO hj_timeline_task(customer_id,id,media_id,kind,analysis_revision,request) VALUES(%s,%s,%s,'qr',%s,%s)",
                         (self.customer, task_id, ident(media_id), revision, Jsonb(clean(pending))))
            return clean(dict(task_id=task_id, media_id=media_id, analysis_revision=revision,
                              status='awaiting_provider', input_mode='video_qr', model_required=False,
                              coverage_scope='explicit_registered_small_windows_only',
                              registration_fingerprint=fingerprint(registration)))
        return self.timeline._command(key, 'register_video_qr', dict(media_id=media_id,registration=registration,actor=actor,reason=reason), action)

    def register_temporal_ocr(self, key, media_id, registration, *, actor, reason):
        """Opt in one source/field/profile revision; no new source or model is enabled."""
        from .temporal_video import validate_registration
        if not actor or not reason:
            raise ValueError('Temporal OCR registration author and reason required')

        def action(conn):
            media = conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, ident(media_id))).fetchone()
            if not media:
                raise Missing('Temporal video outside customer scope')
            asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, ident(media_id))).fetchone()
            validate_registration(registration, media=media, asset=asset)
            for spec in registration['fields']:
                if spec.get('instrument_id'):
                    self.core._entity(conn, spec['instrument_id'], 'instrument')
            revision = 'temporal-display:' + registration['revision'] + ':' + fingerprint(registration)[:16]
            pending = dict(schema_version='vc-lab-temporal-ocr-registration/1', temporal_display=registration,
                           registration_provenance=dict(actor=actor, reason=reason))
            task_id = ident()
            conn.execute("INSERT INTO hj_timeline_task(customer_id,id,media_id,kind,analysis_revision,request) VALUES(%s,%s,%s,'ocr',%s,%s)",
                         (self.customer, task_id, ident(media_id), revision, Jsonb(clean(pending))))
            return clean(dict(task_id=task_id, media_id=media_id, analysis_revision=revision,
                              status='awaiting_provider', input_mode='temporal_display', registration_fingerprint=fingerprint(registration)))
        return self.timeline._command(key, 'register_temporal_ocr', dict(media_id=media_id, registration=registration, actor=actor, reason=reason), action)

    def plan(self, conn, media, kind, *, temporal_registration=None, qr_registration=None):
        duration = media['duration_seconds']
        temporal = kind == 'ocr' and media['kind'] == 'video'
        qr = kind == 'qr'
        if qr:
            from .qr_video import validate_registration
            asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, ident(media['media_id']))).fetchone()
            validate_registration(qr_registration, media=media, asset=asset)
        if temporal:
            from .temporal_video import validate_registration
            asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, ident(media['media_id']))).fetchone()
            validate_registration(temporal_registration, media=media, asset=asset)
        complete=kind=='video_understanding' and self.timeline.policies.video_input_unit=='complete_retained_clip'
        clip=None
        if complete:
            clip=RetainedClips(self.timeline,None).get(conn,media['media_id'])
            if not clip:raise Conflict('Whole-video understanding requires selected clip provenance, not raw packets')
            RetainedClips.validate(conn,self.customer,clip)
        width = Decimal(self.timeline.policies.window_seconds)
        context = Decimal(0 if complete or temporal or qr else self.timeline.policies.context_seconds)
        if complete:width=duration
        cores = [(Decimal(0), Decimal(0))] if media['kind'] == 'photo' else []
        cursor = Decimal(0)
        if temporal or qr:
            registration = qr_registration if qr else temporal_registration
            cores = [(seconds(i['start_seconds']), seconds(i['end_seconds'])) for i in registration['source_intervals']]
            cursor = duration
        while cursor < duration:
            end = min(duration, cursor + width)
            cores.append((cursor, end))
            cursor = end
        if not cores:
            raise ValueError('Unknown duration cannot be scheduled as understood video/audio')
        windows = []
        for index, (start, end) in enumerate(cores):
            refs = []
            left, right = max(Decimal(0), start - context), min(duration, end + context)
            refs.append(dict(media_id=str(media['media_id']), start_seconds=str(left), end_seconds=str(right)))
            media_by_id = {str(media['media_id']): media}
            if not complete and not temporal and not qr and media['start_at'] and media['session_key'] and media['kind'] != 'photo':
                begin = offset_time(media['start_at'], start - context)
                finish = offset_time(media['start_at'], end + context)
                neighbors = conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id<>%s AND kind=%s AND stream_key=%s AND session_key=%s AND start_at<%s AND end_at>%s ORDER BY start_at,media_id LIMIT 32', (self.customer, media['media_id'], media['kind'], media['stream_key'], media['session_key'], finish, begin)).fetchall()
                for other in neighbors:
                    media_by_id[str(other['media_id'])] = other
                    lo, hi = max(begin, other['start_at']), min(finish, other['end_at'])
                    if lo < hi:
                        def exact_offset(value):
                            delta = value - other['start_at']
                            return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1000000
                        refs.append(dict(media_id=str(other['media_id']), start_seconds=str(exact_offset(lo)), end_seconds=str(exact_offset(hi))))
            for ref in refs:
                source = media_by_id[ref['media_id']]
                ref.update(start_at=offset_time(source['start_at'], seconds(ref['start_seconds'])),
                           end_at=offset_time(source['start_at'], seconds(ref['end_seconds'])))
            assets = conn.execute('SELECT a.id,a.version,a.verified_sha256,a.evidence_status,c.source_kind FROM media_asset a JOIN capture c ON c.customer_id=a.customer_id AND c.id=a.capture_id WHERE a.customer_id=%s AND a.id=ANY(%s)', (self.customer, [ident(ref['media_id']) for ref in refs])).fetchall()
            by_id = {str(a['id']): a for a in assets}
            for ref in refs:
                asset = by_id[ref['media_id']]
                if asset['evidence_status'] != 'available':
                    raise Conflict('Model input evidence is unavailable or failed integrity checks')
                ref.update(sha256=asset['verified_sha256'], media_version=asset['version'], source_kind=asset['source_kind'], media_url='/internal/media/' + ref['media_id'])
                if complete:ref['input_unit']='complete_retained_clip'
            refs.sort(key=lambda r: (r['start_at'] is None, r['start_at'] or datetime.min.replace(tzinfo=timezone.utc), r['media_id'], seconds(r['start_seconds'])))
            issues, previous_end = [], None
            for ref in refs:
                if ref['start_at'] is None:
                    issues.append(dict(kind='unknown_clock', media_id=ref['media_id']))
                    continue
                if previous_end and ref['start_at'] != previous_end:
                    issues.append(dict(kind='gap' if ref['start_at'] > previous_end else 'overlap', start_at=min(previous_end, ref['start_at']), end_at=max(previous_end, ref['start_at'])))
                previous_end = max(previous_end, ref['end_at']) if previous_end else ref['end_at']
            windows.append(dict(index=index, core=dict(media_id=str(media['media_id']), start_seconds=str(start), end_seconds=str(end)), input_spans=refs, continuity_issues=issues))
        request=dict(schema_version='vc-lab-timeline-model-request/1', kind=kind,
                          timeline_version=VERSION, processing_rules=Rules().identity(),
                          timeline_policy_fingerprint=fingerprint(self.timeline.policies.document),
                          primary_media_id=media['media_id'], media_kind=media['kind'], stream_key=media['stream_key'], session_key=media['session_key'],
                          start_at=media['start_at'], duration_seconds=duration, time_basis=media['time_basis'],
                          windows=windows, preserve_source_order=True,
                          context_is_not_continuous_recording_proof=True,
                          output_schema='vc-lab-timeline-model-result/1')
        if temporal:
            request.update(input_mode='temporal_display', temporal_display=temporal_registration,
                           coverage_scope='explicit_registered_source_intervals_only', state_scope='independent_window_no_cross_gap_memory')
        if qr:
            request.update(input_mode='video_qr', video_qr=qr_registration,
                           coverage_scope='explicit_registered_small_windows_only_not_whole_clip',
                           state_scope='raw_qr_only_no_panel_operation_or_business_identity')
        if complete:
            request.update(input_unit='complete_retained_clip',retained_clip=clip,preceding_context=[])
            if clip['run_ids'] and media['start_at']:
                rows=conn.execute("SELECT DISTINCT a.id,a.text_content,a.evidence_ids,a.end_at FROM hj_activity a JOIN hj_activity_run r ON r.customer_id=a.customer_id AND r.activity_id=a.id WHERE a.customer_id=%s AND r.run_id=ANY(%s) AND a.end_at<=%s AND a.end_at>=%s AND a.source_type='video' ORDER BY a.end_at DESC,a.id LIMIT 8",
                    (self.customer,[ident(r) for r in clip['run_ids']],media['start_at'],media['start_at']-timedelta(seconds=self.timeline.policies.preceding_context_seconds))).fetchall()
                prior=[]
                for row in rows:
                    valid=conn.execute("SELECT count(*) AS n FROM evidence_ref e JOIN media_asset a ON a.customer_id=e.customer_id AND a.id=e.media_id WHERE e.customer_id=%s AND e.id=ANY(%s) AND a.version=e.media_version AND a.evidence_status='available'",(self.customer,row['evidence_ids'])).fetchone()['n']
                    if not row['evidence_ids'] or valid!=len(row['evidence_ids']):
                        continue
                    candidate=clean(dict(activity_id=row['id'],text=row['text_content'],evidence_ids=row['evidence_ids'],
                                         status='prior_candidate_not_current_clip_observation'))
                    if len(json.dumps([candidate,*prior],ensure_ascii=False).encode())>4096:
                        continue  # Do not truncate a fact or exceed the reserved context.
                    prior.insert(0,candidate)
                request['preceding_context']=prior
        return clean(request)

    def claim(self, key, kind, model_revision, *, lease_seconds=300, now=None):
        if kind not in {'video_understanding', 'transcription', 'ocr', 'qr'} or not model_revision:
            raise ValueError('Explicit model task kind and revision required')
        if kind == 'qr':
            from .qr_video import REVISION
            if model_revision != REVISION:
                raise ValueError('Video QR requires the installed model-free executor revision')
        if type(lease_seconds) is not int or not 10 <= lease_seconds <= 3600:
            raise ValueError('Lease must be 10..3600 seconds')
        at = instant(now) if now else datetime.now(timezone.utc)

        def action(conn):
            expired = conn.execute("UPDATE hj_timeline_task SET status=CASE WHEN attempts>=3 THEN 'needs_attention' ELSE 'awaiting_provider' END,error_code='lease_expired',lease_token=NULL,lease_until=NULL WHERE customer_id=%s AND status='leased' AND lease_until<=%s RETURNING id", (self.customer, at)).fetchall()
            for row in expired:
                conn.execute("UPDATE hj_timeline_task_attempt SET outcome='lease_expired',finished_at=%s WHERE customer_id=%s AND task_id=%s AND outcome='leased'", (at, self.customer, row['id']))
            selected_only=kind=='video_understanding' and self.timeline.policies.video_input_unit=='complete_retained_clip'
            prior_enabled = kind == 'video_understanding' and self.timeline.policies.document.get('multimodal_prior_context', {}).get('enabled') is True
            candidates = conn.execute("SELECT t.*,m.kind AS media_kind,m.capture_id,m.stream_key,m.session_key,m.start_at,m.end_at,m.duration_seconds,m.time_basis FROM hj_timeline_task t JOIN hj_timeline_media m USING(customer_id,media_id) WHERE t.customer_id=%s AND t.kind=%s AND t.status='awaiting_provider' AND t.next_attempt_at<=%s AND t.attempts<3 AND (m.kind='photo' OR m.duration_seconds>0) AND (NOT %s OR EXISTS(SELECT 1 FROM media_derivation d WHERE d.customer_id=t.customer_id AND d.child_id=t.media_id AND d.transform->>'schema_version'='vc-lab-retained-clip/1')) ORDER BY t.created_at,t.id FOR UPDATE OF t SKIP LOCKED LIMIT %s", (self.customer, kind, at,selected_only,32 if prior_enabled else 1)).fetchall()
            task, blocked = None, []
            for candidate in candidates:
                waiting = self._pending_priors(conn, candidate) if prior_enabled else []
                if waiting:
                    blocked.append(dict(task_id=str(candidate['id']), pending_prior_task_ids=[str(r['id']) for r in waiting]))
                else:
                    task = candidate
                    break
            if not task:
                return dict(task=None, **({'pending_prior_channels': blocked} if blocked else {}))
            media = dict(task, kind=task['media_kind'])
            pending = task.get('request') or {}
            request = self.plan(conn, media, kind, temporal_registration=pending.get('temporal_display'), qr_registration=pending.get('video_qr'))
            if pending.get('registration_provenance'):
                request['registration_provenance'] = pending['registration_provenance']
            request['task_id'] = str(task['id'])
            request['analysis_revision'] = task['analysis_revision']
            if kind == 'ocr' and task['media_kind'] == 'photo':
                # A specific revision remains pinned even if a newer registration arrives.
                registration = conn.execute('SELECT * FROM hj_ocr_frame_registration WHERE customer_id=%s AND media_id=%s AND (%s=\'initial\' OR %s=\'frame-registration:\'||revision::text) ORDER BY revision DESC LIMIT 1', (self.customer, task['media_id'], task['analysis_revision'], task['analysis_revision'])).fetchone()
                request['frame_registration'] = clean(registration) if registration else None
            request['input_fingerprint'] = fingerprint(request)
            self.timeline._dirty(conn, media['start_at'], media['end_at'])
            token, until = ident(), at + timedelta(seconds=lease_seconds)
            conn.execute("UPDATE hj_timeline_task SET status='leased',attempts=attempts+1,error_code=NULL,lease_token=%s,lease_until=%s,model_revision=%s,request=%s WHERE customer_id=%s AND id=%s", (token, until, model_revision, Jsonb(request), self.customer, task['id']))
            conn.execute('INSERT INTO hj_timeline_task_attempt(customer_id,token,task_id,model_revision,request,started_at,lease_until) VALUES(%s,%s,%s,%s,%s,%s,%s)', (self.customer, token, task['id'], model_revision, Jsonb(request), at, until))
            wait=max(0,(at-task['created_at']).total_seconds())
            conn.execute("INSERT INTO hj_timeline_timing(customer_id,attempt_token,stage,started_at,finished_at,elapsed_seconds) VALUES(%s,%s,'queue_wait',%s,%s,%s)",(self.customer,token,task['created_at'],at,Decimal(str(wait))))
            return dict(task_id=task['id'], lease_token=token, lease_until=until, model_revision=model_revision, request=request)
        return self.timeline._command(key, 'claim_model_task', dict(kind=kind, model_revision=model_revision, lease_seconds=lease_seconds, now=now), action)

    def _pending_priors(self, conn, task):
        """Wait for existing overlapping channels; never invent missing channels.

        The original receipt camera (not an audio/photo channel prefix) scopes
        dependencies. A failed channel is explicit missing evidence, not a
        reason to retry a successful video model or fabricate a transcript.
        """
        return conn.execute('''SELECT d.id FROM hj_timeline_task d
            JOIN hj_timeline_media dm ON dm.customer_id=d.customer_id AND dm.media_id=d.media_id
            JOIN capture dc ON dc.customer_id=dm.customer_id AND dc.id=dm.capture_id
            JOIN inbox di ON di.customer_id=dc.customer_id AND di.source_id=dc.source_id
              AND di.event_id=dc.source_capture_id AND di.revision=dc.source_revision
            JOIN capture vc ON vc.customer_id=d.customer_id AND vc.id=%s
            JOIN inbox vi ON vi.customer_id=vc.customer_id AND vi.source_id=vc.source_id
              AND vi.event_id=vc.source_capture_id AND vi.revision=vc.source_revision
            WHERE d.customer_id=%s AND d.kind IN ('transcription','ocr','qr')
              AND d.status IN ('awaiting_provider','leased') AND d.id<>%s
              AND di.payload->>'camera_id'=vi.payload->>'camera_id'
              AND di.payload->>'camera_id' IS NOT NULL
              AND dm.time_basis->'uncertainty_us'='0'::jsonb
              AND dm.time_basis->'clock_sync_verified' IS DISTINCT FROM 'false'::jsonb
              AND dm.time_basis->'clock_sync_valid' IS DISTINCT FROM 'false'::jsonb
              AND (dm.time_basis->'clock_sync_verified'='true'::jsonb
                OR dm.time_basis->'clock_sync_valid'='true'::jsonb
                OR (di.payload->>'source_kind'='fixture'
                    AND dm.time_basis->>'basis' IN ('synthetic_fixture_clock','fixture_receipt')))
              AND dm.time_basis=di.payload->'time_basis'
              AND vi.payload#>'{time_basis,uncertainty_us}'='0'::jsonb
              AND vi.payload#>'{time_basis,clock_sync_verified}' IS DISTINCT FROM 'false'::jsonb
              AND vi.payload#>'{time_basis,clock_sync_valid}' IS DISTINCT FROM 'false'::jsonb
              AND (vi.payload#>'{time_basis,clock_sync_verified}'='true'::jsonb
                OR vi.payload#>'{time_basis,clock_sync_valid}'='true'::jsonb
                OR (vi.payload->>'source_kind'='fixture'
                    AND vi.payload#>>'{time_basis,basis}' IN ('synthetic_fixture_clock','fixture_receipt')))
              AND dm.start_at<%s AND (dm.end_at>%s OR (dm.start_at=dm.end_at AND dm.start_at>=%s))
            ORDER BY d.created_at,d.id LIMIT 101''',
            (task['capture_id'], self.customer, task['id'],
             task['end_at'], task['start_at'], task['start_at'])).fetchall()

    def submit(self, key, task_id, lease_token, model_revision, result, *, now=None):
        return self._submit(key, task_id, lease_token, model_revision, result, now=now)

    def _submit(self, key, task_id, lease_token, model_revision, result, *, now=None, local_provider_fingerprint=None):
        at = instant(now) if now else datetime.now(timezone.utc)
        if not isinstance(result, dict) or result.get('schema_version') != 'vc-lab-timeline-model-result/1':
            raise ValueError('Versioned structured model result required')
        events = result.get('events')
        if not isinstance(events, list) or len(events) > 500:
            raise ValueError('Bounded structured observations required')

        def action(conn):
            task = conn.execute('SELECT * FROM hj_timeline_task WHERE customer_id=%s AND id=%s FOR UPDATE', (self.customer, ident(task_id))).fetchone()
            if not task:
                raise Missing('Task outside customer scope')
            if task['status'] != 'leased' or task['lease_token'] != ident(lease_token) or task['lease_until'] <= at:
                raise Conflict('Expired or superseded model worker lease')
            if model_revision != task['model_revision'] or result.get('input_fingerprint') != task['request']['input_fingerprint']:
                raise Conflict('Result model or input fingerprint differs from the claimed request')
            temporal = task['request'].get('input_mode') == 'temporal_display'
            qr = task['kind'] == 'qr'
            if qr:
                from .qr_video import REVISION, validate_registration as validate_qr, validate_execution
                if not local_provider_fingerprint or model_revision != REVISION:
                    raise Conflict('Video QR completion requires stored local pixel execution evidence')
                media = conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, task['media_id'])).fetchone()
                asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, task['media_id'])).fetchone()
                validate_qr(task['request']['video_qr'], media=media, asset=asset)
            if temporal:
                from .temporal_video import validate_registration
                media = conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, task['media_id'])).fetchone()
                asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s', (self.customer, task['media_id'])).fetchone()
                validate_registration(task['request']['temporal_display'], media=media, asset=asset)
            covered = result.get('covered_window_indices')
            expected = set(range(len(task['request']['windows'])))
            if not isinstance(covered, list) or any(type(i) is not int for i in covered) or len(covered) != len(set(covered)) or not set(covered) <= expected:
                raise ValueError('Invalid declared input coverage')
            execution, receipts = 'supplied_not_independently_verified', []
            if local_provider_fingerprint:
                rows = conn.execute('SELECT * FROM hj_timeline_window_result WHERE customer_id=%s AND task_id=%s AND input_fingerprint=%s AND provider_fingerprint=%s ORDER BY window_index', (self.customer, task['id'], task['request']['input_fingerprint'], local_provider_fingerprint)).fetchall()
                stored_events = [event for row in rows for event in row['normalized_result']['events']]
                if any(fingerprint(row['normalized_result']) != row['result_fingerprint'] for row in rows):
                    raise Conflict('Stored local execution result differs from its fingerprint')
                if [row['window_index'] for row in rows] != sorted(expected) or covered != sorted(expected) or fingerprint(stored_events) != fingerprint(events):
                    raise Conflict('Local submission differs from its stored window executions')
                execution = 'local_transport_and_payload_verified'
                if not any(row['execution_receipt']['invocations'] for row in rows):
                    execution = 'local_geometry_inputs_verified'
                if temporal and not any(i.get('execution') for row in rows for i in row['execution_receipt']['invocations']):
                    execution = 'local_temporal_pixels_verified_without_ocr'
                receipts = [row['execution_receipt'] for row in rows]
                if qr:
                    frames = 0
                    for row in rows:
                        window = task['request']['windows'][row['window_index']]
                        if len(row['execution_receipt']['invocations']) != 1 or len(row['normalized_result']['events']) != 1:
                            raise Conflict('Video QR requires one bounded pixel receipt/event per window')
                        receipt = row['execution_receipt']['invocations'][0].get('video_qr')
                        validate_execution(receipt, window['input_spans'][0], task['request']['video_qr'], conn, self.core)
                        frames += receipt['original_decoded_frame_count']
                        if row['normalized_result']['events'][0].get('payload',{}).get('video_qr') != receipt:
                            raise Conflict('QR summary differs from its stored frame execution receipt')
                    if frames > 600:
                        raise Conflict('Video QR result exceeded the task original-frame bound')
                    execution = 'local_model_free_qr_pixels_verified'
            ids = []
            for index, item in enumerate(events):
                if qr and (set(item) != {'kind','text','spans','payload'} or item['kind'] != 'unknown'
                           or any(item.get(k) for k in ('fields','run_ids','sample_id','instrument_id','person_id','tag_id'))):
                    raise Conflict('Video QR observations cannot supply fields, actions or business attribution')
                spans = item.get('spans', [])
                if not spans or str(spans[0].get('media_id')) != str(task['media_id']):
                    raise ValueError('Each event must anchor in the primary media; context cannot duplicate other files')
                # All submitted evidence must be in the immutable claimed input.
                allowed = [s for w in task['request']['windows'] if w['index'] in covered for s in w['input_spans']]
                for span in spans:
                    if not any(str(s['media_id']) == str(span['media_id']) and seconds(s['start_seconds']) <= seconds(span['start_seconds']) <= seconds(span['end_seconds']) <= seconds(s['end_seconds']) for s in allowed):
                        raise ValueError('Model evidence escaped its declared input coverage')
                fields = item.get('fields', [])
                for field in fields:
                    if not isinstance(field.get('field_key'), str) or not field['field_key'] or isinstance(field.get('value'), (float, dict, list, bool)):
                        raise ValueError('OCR fields need an explicit key and exact scalar candidate')
                    if temporal and (field.get('status') not in {'confirmed','uncertain','unreadable'} or
                                     (field.get('status') != 'confirmed' and (field.get('value') is not None or field.get('value_text') is not None)) or
                                     field.get('auto_confirmed') is not False):
                        raise ValueError('Temporal visual candidates cannot create business confirmations or uncertain values')
                    if temporal and field.get('status') == 'confirmed' and not local_provider_fingerprint:
                        raise Conflict('Confirmed temporal visual output requires stored local execution evidence')
                    if temporal:
                        matches = [spec for spec in task['request']['temporal_display']['fields']
                                   if all(field.get(k) == spec.get(k) for k in ('screen_id','field_key','profile_id','role','unit'))
                                   and field.get('channel','') == spec.get('channel','')]
                        if len(matches) != 1:
                            raise Conflict('Temporal result differs from registered screen/field/profile identity')
                        registration = task['request']['temporal_display']
                        from .temporal_video import resolve_dynamic_context
                        from .dynamic_context import DynamicContext
                        refs = field.get('temporal_display',{}).get('source_refs',[])
                        if registration.get('dynamic_context'):
                            offsets=[]
                            for ref in refs:
                                offset=Fraction(ref['pts'])*Fraction(ref['time_base'])-Fraction(ref['stream_origin'])
                                if (ref.get('source_sha256')!=registration['source_sha256'] or
                                    str(ref.get('media_id'))!=str(task['media_id']) or
                                    offset!=Fraction(ref['media_offset_seconds']) or
                                    not any(Fraction(s['start_seconds'])<=offset<=Fraction(s['end_seconds']) for s in spans)):
                                    raise Conflict('Dynamic association source support differs from original PTS')
                                offsets.append(offset)
                            if not offsets:
                                raise Conflict('Dynamic association requires actual source support PTS')
                            context=resolve_dynamic_context(DynamicContext(self.core).query_window,matches[0],
                                dict(media_id=str(task['media_id'])),registration,min(offsets),max(offsets),
                                revision=field.get('dynamic_context',{}).get('revision'))
                        else:
                            context=resolve_dynamic_context(None,matches[0],{},registration,0,0)
                        if item.get('instrument_id')!=context.get('instrument_id'):
                            raise Conflict('Temporal result differs from its support-range physical association')
                        field.update(reading_status=field['status'],association_status=context['association_status'],dynamic_context=context)
                payload = dict(item.get('payload', {}), fields=fields, task_id=str(task['id']),
                               accuracy='NOT_PROVEN', execution_verification=execution)
                if task['kind'] == 'video_understanding':
                    from .value_candidates import candidate
                    payload['value_candidate'] = candidate(payload.get('value_candidate'), spans,
                        allowed_spans=allowed, model_revision=model_revision,
                        input_fingerprint=task['request']['input_fingerprint'], execution_verification=execution)
                elif 'value_candidate' in payload:
                    raise ValueError('Video value candidates require a video understanding task')
                if temporal:
                    # Output-supplied flags cannot upgrade the source clock or
                    # cross the registration's semantic/physical boundary.
                    payload.update(field_mapping='temporal_registered_video',
                                   temporal_world_clock_status=task['request']['temporal_display']['world_clock_status'],
                                   observation_time_basis='original_media_pts', business_confirmation=False,
                                   external_confirmation='not_provided')
                    if fields:
                        payload.update(dynamic_context=fields[0]['dynamic_context'],
                                       association_status=fields[0]['association_status'])
                if task['request'].get('retained_clip'):
                    clip=task['request']['retained_clip'];RetainedClips.validate(conn,self.customer,clip)
                    mapped=original_spans(clip,spans[0]['start_seconds'],spans[0]['end_seconds'])
                    spans=[*spans,*mapped]
                    payload.update(original_source_spans=mapped,retained_clip=clip,context_is_new_observation=False)
                value = self.timeline.add_activity(f'timeline-result:{task["id"]}:{index}', item['kind'],
                            {'video_understanding': 'video', 'transcription': 'speech', 'ocr': 'ocr', 'qr': 'video'}[task['kind']],
                            item.get('text', ''), spans, run_ids=item.get('run_ids'), sample_id=item.get('sample_id'),
                            instrument_id=item.get('instrument_id'), person_id=item.get('person_id'), tag_id=item.get('tag_id'),
                            model_revision=model_revision, payload=payload)
                ids.append(value['id'])
            value_only = events and all((event.get('payload') or {}).get('value_candidate', {}).get('value_assessment_only') is True for event in events)
            outcome = ('partial_declared_coverage' if set(covered) != expected else
                       'empty_observations_not_verified_absence' if not events else
                       'pending_value_assessments_not_physical_actions' if value_only else 'candidate_observations')
            stored = dict(result, activity_ids=ids, outcome=outcome, accuracy='NOT_PROVEN', execution_verification=execution, execution_receipts=receipts)
            conn.execute("UPDATE hj_timeline_task SET status='completed',result=%s,completed_at=%s WHERE customer_id=%s AND id=%s", (Jsonb(clean(stored)), at, self.customer, task['id']))
            conn.execute("UPDATE hj_timeline_task_attempt SET outcome='completed',finished_at=%s WHERE customer_id=%s AND token=%s", (at, self.customer, ident(lease_token)))
            media = conn.execute('SELECT start_at,end_at FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, task['media_id'])).fetchone()
            self.timeline._dirty(conn, media['start_at'], media['end_at'])
            return dict(task_id=task_id, activity_ids=ids, status='completed', outcome=outcome)
        return self.timeline._command(key, 'submit_model_result', dict(task_id=task_id, lease_token=lease_token, model_revision=model_revision, result=result, now=now, local_provider_fingerprint=local_provider_fingerprint), action)

    def renew(self, task_id, lease_token, lease_seconds=300):
        if type(lease_seconds) is not int or not 10 <= lease_seconds <= 3600:
            raise ValueError('Invalid lease extension')
        with self.core.db.transaction() as conn:
            row = conn.execute("UPDATE hj_timeline_task SET lease_until=now()+(%s*interval '1 second') WHERE customer_id=%s AND id=%s AND lease_token=%s AND status='leased' AND lease_until>now() RETURNING lease_until", (lease_seconds, self.customer, ident(task_id), ident(lease_token))).fetchone()
            if not row:
                raise Conflict('Cannot renew an expired or superseded lease')
            conn.execute('UPDATE hj_timeline_task_attempt SET lease_until=%s WHERE customer_id=%s AND token=%s', (row['lease_until'], self.customer, ident(lease_token)))

    def fail(self, task_id, lease_token, error_code,*,retryable=True):
        # Only error class names enter persistent logs; endpoints/tokens/stderr do not.
        if not error_code.isidentifier() or len(error_code) > 100:
            raise ValueError('Bounded error code required')
        with self.core.db.transaction() as conn:
            row = conn.execute("UPDATE hj_timeline_task SET status=CASE WHEN attempts>=3 OR NOT %s THEN 'needs_attention' ELSE 'awaiting_provider' END,error_code=%s,next_attempt_at=now()+(LEAST(300,5*power(2,attempts-1))*interval '1 second'),lease_token=NULL,lease_until=NULL WHERE customer_id=%s AND id=%s AND lease_token=%s AND status='leased' AND lease_until>now() RETURNING status,media_id", (retryable,error_code, self.customer, ident(task_id), ident(lease_token))).fetchone()
            if not row:
                raise Conflict('Failed worker no longer owns an active lease')
            conn.execute("UPDATE hj_timeline_task_attempt SET outcome=%s,finished_at=now() WHERE customer_id=%s AND token=%s", ('failed:' + error_code, self.customer, ident(lease_token)))
            media = conn.execute('SELECT start_at,end_at FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, row['media_id'])).fetchone()
            self.timeline._dirty(conn, media['start_at'], media['end_at'])
            return dict(task_id=str(task_id), status=row['status'], error_code=error_code)

    def reanalyze(self, key, media_id, kind, analysis_revision):
        if kind not in {'video_understanding', 'transcription', 'ocr'} or not isinstance(analysis_revision, str) or not 1 <= len(analysis_revision) <= 100 or analysis_revision.startswith('frame-registration:'):
            raise ValueError('Explicit analysis revision and kind required')
        def action(conn):
            media = conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (self.customer, ident(media_id))).fetchone()
            if not media:
                raise Missing('Media outside customer scope')
            if (kind == 'ocr') != (media['kind'] == 'photo') or (kind == 'video_understanding' and media['kind'] != 'video'):
                raise ValueError('Analysis kind does not match media kind')
            old = conn.execute('SELECT id FROM hj_timeline_task WHERE customer_id=%s AND media_id=%s AND kind=%s AND analysis_revision=%s', (self.customer, ident(media_id), kind, analysis_revision)).fetchone()
            if old:
                raise Conflict('Analysis revision already exists; use a new revision')
            task_id = ident()
            conn.execute('INSERT INTO hj_timeline_task(customer_id,id,media_id,kind,analysis_revision) VALUES(%s,%s,%s,%s,%s)', (self.customer, task_id, ident(media_id), kind, analysis_revision))
            return dict(task_id=task_id, status='awaiting_provider')
        return self.timeline._command(key, 'reanalyze_media', dict(media_id=media_id, kind=kind, analysis_revision=analysis_revision), action)

"""Server-authorized generic capture QR intake, independent of recorder naming.

Original receipts remain immutable. A private TimelinePolicy authorizes this
customer's captures through append-only command receipts. Decoded text remains
untrusted; only a unique active equipment identity and verified source clock can
produce a camera-scoped presence. No person, run, material usage or quantity is
inferred. Raw-video tasks keep native pixels/PTS and have bounded independent
raw-only persistence, not the research geometry projection's whole-media limit.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident
from .dynamic_context import DynamicContext, _source_time
from .media import Media, hash_file
from .operator_scan import CAMERA_INTENT, OperatorScans, declaration_activity
from .rules import Rules
from .subprocesses import run_bounded
from .timeline import Timeline
from .timeline_policy import TimelinePolicies

SCHEMA = 'vc-lab-capture-equipment-intake/1'
AUTH_SCHEMA = 'vc-lab-capture-equipment-authorization/1'
RAW_SCHEMA = 'vc-lab-video-qr/2'
AUTH_PREFIX = 'capture-equipment-policy:'
OBS_PREFIX = 'capture-equipment-observation:'


def policy(settings):
    return TimelinePolicies.load(settings.timeline_policy_path).document.get('automatic_equipment_qr', {})


def _auth(conn, core, media_id):
    row = conn.execute('SELECT fingerprint,result FROM command_receipt WHERE customer_id=%s AND key=%s',
                       (core.customer, AUTH_PREFIX + str(media_id))).fetchone()
    if row and (fingerprint(row['result']) != row['fingerprint'] or row['result'].get('schema_version') != AUTH_SCHEMA):
        raise Conflict('Server capture QR authorization fingerprint differs')
    return row['result'] if row else None


def partition(duration, config):
    width, cursor = Decimal(config['video_window_seconds']), Decimal(0)
    if not 0 < duration <= config['max_packet_seconds']:
        raise ValueError('Native packet exceeds its explicit private duration limit')
    intervals = []
    while cursor < duration:
        end = min(duration, cursor + width)
        intervals.append(dict(start_seconds=str(cursor), end_seconds=str(end)))
        cursor = end
    if len(intervals) > config['max_packet_tasks']:
        raise ValueError('Native packet exceeds its explicit QR task count limit')
    return intervals


def _probe(path, settings, expected_sha):
    from .temporal_video import source_guard

    with source_guard(path, settings):
        before = path.stat()
        raw = run_bounded(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=index,time_base,start_pts,duration_ts,nb_frames', '-of', 'json', str(path)], '',
            env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')}, timeout=10, max_bytes=16384)
        document = json.loads(raw)
        streams = document.get('streams', [])
        if len(streams) != 1:
            raise ValueError('One bounded original video stream required')
        stream = streams[0]
        duration = Fraction(int(stream['duration_ts'])) * Fraction(stream['time_base'])
        count = int(stream['nb_frames'])
        if not 0 < duration <= 900 or not 1 <= count <= 54000:
            raise ValueError('Original packet exceeds 900 seconds/54000 frames or has unknown coverage')
        if hash_file(path, settings.max_native_bytes)[0] != expected_sha:
            raise Conflict('Original packet changed during QR technical probe')
        after = path.stat()
        def stat(s):
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if stat(before) != stat(after):
            raise Conflict('Original packet changed across QR technical probe/hash')
        return dict(method='ffprobe_stream_metadata_not_model', stream=stream,
                    native_duration_fraction=str(duration), declared_frame_count=count,
                    stream_metadata_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    source_sha256=expected_sha, source_stable=True)


def admit(conn, core, settings, media, receipt, *, verified_path=None):
    """Called inside the original Ingest transaction; no nested DB connection.

    Capture, source authorization and all QR tasks either commit together or roll
    back together. An unsupported packet is retained with needs_attention; the
    original capture and its ordinary understanding/OCR policy are unchanged.
    """
    config = policy(settings)
    if not config.get('enabled') or media['kind'] not in {'photo', 'video'}:
        return None
    if not config[media['kind']] or receipt.get('scan_intent') is not None:
        return None
    if receipt['source_kind'] not in config['source_kinds']:
        return None
    old = _auth(conn, core, media['media_id'])
    if old:
        return old
    Timeline(core).lock(conn)
    old = _auth(conn, core, media['media_id'])
    if old:
        return old
    asset = conn.execute('SELECT * FROM media_asset WHERE customer_id=%s AND id=%s',
                         (core.customer, media['media_id'])).fetchone()
    if asset is None or asset['evidence_status'] != 'available':
        raise Conflict('Generic QR admission requires verified original media')
    camera = receipt.get('camera_id')
    if isinstance(camera, str):
        from .archive_layout import day_directory
        try:
            day_directory(camera, '2000-01-01T00:00:00+08:00')
        except (ValueError, TypeError):
            camera = None
        if camera and not conn.execute("SELECT 1 FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s", (core.customer, camera)).fetchone():
            core._register(conn, 'capture_device', camera, dict(registration_basis='validated_capture_protocol_camera_key',
                physical_hardware_pairing_verified=False, wearer_or_person_assigned=False))
    document = clean(dict(schema_version=AUTH_SCHEMA, media_id=media['media_id'], capture_id=media['capture_id'],
        source_sha256=asset['verified_sha256'], media_version=asset['version'], kind=media['kind'],
        camera_code=receipt.get('camera_id'), source_stream_key=media['stream_key'],
        original_source_session_key=media['session_key'], source_kind=receipt['source_kind'],
        receipt_sha256=fingerprint(receipt), declared_sha256_present=receipt.get('sha256') is not None,
        policy=config, policy_sha256=fingerprint(config), processing_rules=Rules().identity(),
        authority='server_pinned_automatic_equipment_qr_policy', status='authorized', task_ids=[],
        manual_intent_fabricated=False, person_or_run_inferred=False, quantity_written=False))
    if media['kind'] == 'video':
        try:
            if verified_path is None:
                raise ValueError('Original packet location unavailable for bounded technical probe')
            probe = _probe(verified_path, settings, asset['verified_sha256'])
            actual = Fraction(probe['native_duration_fraction'])
            if actual != Fraction(media['duration_seconds']):
                raise ValueError('Native stream duration differs from admitted original-media range')
            if actual > config['max_packet_seconds']:
                raise ValueError('Native packet exceeds its explicit private duration limit')
            document['technical_probe'] = probe
            if not media['session_key']:
                namespace = 'capture:' + str(media['capture_id'])
                conn.execute('UPDATE hj_timeline_media SET session_key=%s WHERE customer_id=%s AND media_id=%s AND session_key IS NULL',
                             (namespace, core.customer, media['media_id']))
                media = dict(media, session_key=namespace)
                document['state_namespace_basis'] = 'technical_namespace_not_physical_session'
            else:
                document['state_namespace_basis'] = 'provided_source_session_not_experiment_identity'
            document['source_session_key'] = media['session_key']
            intervals = partition(media['duration_seconds'], config)
            from .qr_video import validate_registration
            from .scan_intake import verified_clock

            known = media['start_at'] is not None and verified_clock(media['time_basis'])
            tasks = []
            for index, interval in enumerate(intervals):
                registration = dict(schema_version=RAW_SCHEMA, enabled=True,
                    revision='automatic:' + fingerprint(dict(policy=document['policy_sha256'], media=str(media['media_id']), index=index))[:32],
                    source_sha256=asset['verified_sha256'], media_version=asset['version'],
                    source_stream_key=media['stream_key'], source_session_key=media['session_key'],
                    rotation_degrees_clockwise=config['rotation_degrees_clockwise'], orientation_basis=config['orientation_basis'],
                    time_basis='original_media_pts', world_clock_status='verified' if known else 'uncalibrated',
                    source_intervals=[interval], persistence='bounded_raw_qr_only',
                    authorization_key=AUTH_PREFIX + str(media['media_id']))
                validate_registration(registration, media=media, asset=asset)
                task_id = ident()
                pending = dict(schema_version='vc-lab-video-qr-registration/1', video_qr=registration,
                    registration_provenance=dict(actor='configured generic capture QR policy',
                        reason='native packet full-frame bounded QR plan', authorization_key=registration['authorization_key'],
                        policy_sha256=document['policy_sha256']))
                revision = 'video-qr:' + registration['revision'] + ':' + fingerprint(registration)[:16]
                tasks.append((task_id, revision, pending))
            for task_id, revision, pending in tasks:
                conn.execute("INSERT INTO hj_timeline_task(customer_id,id,media_id,kind,analysis_revision,request) VALUES(%s,%s,%s,'qr',%s,%s)",
                             (core.customer, task_id, media['media_id'], revision, Jsonb(clean(pending))))
            document.update(task_ids=[str(t[0]) for t in tasks], source_intervals=intervals,
                coverage_plan='entire_verified_stream_duration_partition_no_sampling',
                whole_media_decoded=False, throughput_or_quality_proven=False)
        except (ValueError, KeyError, OSError, RuntimeError, TimeoutError) as exc:
            document.update(status='needs_attention', reason=str(exc)[:512], error_code=type(exc).__name__,
                            automatic_retry=False, original_capture_retained=True)
    conn.execute('INSERT INTO command_receipt(customer_id,key,fingerprint,result) VALUES(%s,%s,%s,%s)',
                 (core.customer, AUTH_PREFIX + str(media['media_id']), fingerprint(document), Jsonb(document)))
    return document


def newest_candidates(root, excluded, *, max_items=2000, include_bare_photos=True):
    """Bounded recent-day discovery, preserving every existing producer format.

    Enumerates only a bounded top-level catalogue before descending recent days;
    overlarge/unknown catalogues return partial coverage rather than scanning old
    archives and presenting the first arbitrary walk prefix as current input.
    """
    from .archive_layout import day_directory
    from .input_layout import photo_location

    top, other, root_receipts, truncated = [], [], [], False
    for index, path in enumerate(root.iterdir()):
        if index >= 8192:
            return [], True
        if path.is_symlink() or path.name in excluded:
            continue
        if not path.is_dir():
            if path.name in {'CaptureReceipt.json', 'PhotoReceipt.json'} or path.name.endswith('recording_ready.json'):
                root_receipts.append(path)
            continue
        day, separator, camera = path.name.partition('_')
        try:
            valid = separator and day_directory(camera, day + 'T00:00:00+08:00') == path.name
        except (ValueError, TypeError):
            valid = False
        if valid:
            top.append(path)
        else:
            other.append(path)
    top.sort(key=lambda p: p.name, reverse=True)
    if len(top) > 64:
        truncated = True
    files, visited = root_receipts, 0
    for base in top[:64] + sorted(other, key=lambda p: p.name, reverse=True)[:64]:
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = sorted((d for d in dirs if d not in excluded and not Path(directory, d).is_symlink()), reverse=True)
            for name in sorted(names, reverse=True):
                visited += 1
                if visited > max_items:
                    return files, True
                path = Path(directory, name)
                if path.is_symlink():
                    continue
                candidate = name in {'CaptureReceipt.json', 'PhotoReceipt.json'} or name.endswith('recording_ready.json')
                if not candidate and include_bare_photos and path.suffix.lower() in {'.png', '.jpg', '.jpeg'} and 'PhotoReceipt.json' not in names:
                    try:
                        photo_location(path.relative_to(root).as_posix())
                        candidate = True
                    except ValueError:
                        pass
                if candidate:
                    files.append(path)
    # Explicit receipt inboxes outside device/day paths remain supported, but
    # are not recursively walked ahead of fresh native days.
    return files, truncated


class BoundedQRContext(DynamicContext):
    """Existing frame validators/storage, bounded per QR task; no geometry replay.

    This explicit raw-only mode may retain many independent tasks for a packet.
    It does not enlarge or call the research DynamicContext projection limits.
    """
    def __init__(self, core, ref, registration):
        super().__init__(core)
        self.ref, self.registration = ref, registration

    def append_batch(self, key, media_id, source_session_key, frame_events):
        if not isinstance(frame_events, list) or not 1 <= len(frame_events) <= 128:
            raise ValueError('Bounded raw QR frame batch required')
        request = clean(dict(schema_version=SCHEMA, action='bounded_raw_qr_batch', media_id=media_id,
                             session=source_session_key, registration=self.registration, frames=frame_events))
        if len(json.dumps(request).encode()) > 1024**2:
            raise ValueError('Raw QR batch exceeds 1 MiB')

        def action(conn):
            from .scan_intake import verified_clock

            media = self._media(conn, media_id, source_session_key)
            authorization = _auth(conn, self.core, media_id)
            if (not authorization or authorization['status'] != 'authorized'
                    or str(media_id) != authorization['media_id']
                    or authorization['source_sha256'] != media['verified_sha256']
                    or authorization['media_version'] != media['version']):
                raise Conflict('Server source authorization unavailable for raw QR persistence')
            interval = self.registration['source_intervals'][0]
            if (self.registration['authorization_key'] != AUTH_PREFIX + str(media_id)
                    or interval not in authorization['source_intervals']
                    or self.registration['rotation_degrees_clockwise'] != authorization['policy']['rotation_degrees_clockwise']
                    or self.registration['orientation_basis'] != authorization['policy']['orientation_basis']):
                raise Conflict('Raw QR window/orientation differs from its pinned server plan')
            lo, hi = Fraction(interval['start_seconds']), Fraction(interval['end_seconds'])
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                         (f'raw-qr:{self.core.customer}:{media_id}:{lo}:{hi}',))
            existing = conn.execute('SELECT frame_key,payload_sha256 FROM hj_dynamic_frame_event WHERE customer_id=%s AND media_id=%s AND source_session_key=%s AND source_seconds>=%s AND source_seconds<%s',
                (self.core.customer, ident(media_id), source_session_key, Decimal(lo.numerator) / Decimal(lo.denominator), Decimal(hi.numerator) / Decimal(hi.denominator))).fetchall()
            by_key = {r['frame_key']: r for r in existing}
            keys = [e.get('frame_key') for e in frame_events]
            if len(set(keys)) != len(keys) or len(existing) + len(set(keys) - set(by_key)) > 600:
                raise Conflict('One raw QR window exceeds 600 original frames')
            accepted = 0
            for event in frame_events:
                self._validate_event(conn, event, media)
                at = _source_time(event)
                if not lo <= at < hi or event.get('tracks') or event.get('supersedes_frame_key'):
                    raise Conflict('Raw-only QR cannot supply geometry/corrections or escape its original task window')
                digest = fingerprint(event)
                previous = by_key.get(event['frame_key'])
                if previous:
                    if previous['payload_sha256'] != digest:
                        raise Conflict('Raw QR frame id changed immutable pixels/metadata')
                    continue
                world = media['start_at'] + timedelta(microseconds=round(at * 1000000)) if verified_clock(media['time_basis']) and media['start_at'] else None
                conn.execute('INSERT INTO hj_dynamic_frame_event(customer_id,id,media_id,source_session_key,frame_key,source_seconds,payload,payload_sha256,event_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                    (self.core.customer, ident(), ident(media_id), source_session_key, event['frame_key'],
                     Decimal(at.numerator) / Decimal(at.denominator), Jsonb(clean(event)), digest, world))
                accepted += 1
            return dict(status='raw_qr_committed_no_geometry_projection', accepted=accepted, revision=None,
                        snapshot=dict(processing_rules=Rules().identity(), geometry_projection_created=False))
        return self.core.command(key, request, action)


def _media(conn, core, media_id):
    return conn.execute('''SELECT m.*,a.version,a.verified_sha256,a.evidence_status,i.payload AS receipt
        FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
        JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
        JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
          AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
        WHERE m.customer_id=%s AND m.media_id=%s''', (core.customer, ident(media_id))).fetchone()


def _source_matches(core, media, authorization):
    return (authorization is not None and authorization.get('status') == 'authorized'
            and authorization.get('authority') == 'server_pinned_automatic_equipment_qr_policy'
            and authorization.get('media_id') == str(media['media_id'])
            and authorization.get('source_sha256') == media['verified_sha256']
            and authorization.get('media_version') == media['version']
            and authorization.get('receipt_sha256') == fingerprint(media['receipt'])
            and authorization.get('source_stream_key') == media['stream_key']
            and authorization.get('source_session_key', authorization.get('original_source_session_key')) == media['session_key']
            and media['time_basis'] == media['receipt']['time_basis']
            and authorization.get('policy_sha256') == fingerprint(authorization.get('policy'))
            and media['receipt'].get('scan_intent') is None
            and media['evidence_status'] == 'available'
            and (media['receipt'].get('sha256') is None or media['receipt']['sha256'] == media['verified_sha256']))


def _photo(conn, core, job_id):
    rows = conn.execute('''SELECT j.*,a.id AS media_id,o.evidence_ids,o.id AS observation_id
        FROM processing_job j JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
        JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id
        JOIN observation o ON o.customer_id=j.customer_id AND o.job_id=j.id AND o.kind='qr'
        WHERE j.customer_id=%s AND j.id=%s AND j.worker_kind='qr' AND c.kind='photo' ''',
        (core.customer, ident(job_id))).fetchall()
    if len(rows) != 1:
        return None
    row = rows[0]
    result = row['result'] or {}
    rules = result.get('processing_identity', {}).get('processing_rules')
    if row['status'] != 'completed' or rules != row['rule_identity'] or not rules or rules.get('sha256') != row['rule_fingerprint']:
        raise Conflict('Generic photo QR result differs from its completed original rules')
    return row


def _video(conn, core, frame_id):
    frame = conn.execute('SELECT * FROM hj_dynamic_frame_event WHERE customer_id=%s AND id=%s',
                         (core.customer, ident(frame_id))).fetchone()
    if frame is None:
        return None
    rows = conn.execute('''SELECT t.id,t.request,t.result,w.normalized_result,w.result_fingerprint,w.execution_receipt
        FROM hj_timeline_task t JOIN hj_timeline_window_result w ON w.customer_id=t.customer_id AND w.task_id=t.id
          AND w.input_fingerprint=t.request->>'input_fingerprint'
        WHERE t.customer_id=%s AND t.media_id=%s AND t.kind='qr' AND t.status='completed'
          AND t.request->'video_qr'->>'schema_version'='vc-lab-video-qr/2'
          AND w.execution_receipt->'invocations'->0->'video_qr'->'original_frames' @> %s::jsonb LIMIT 2''',
        (core.customer, frame['media_id'], Jsonb([dict(frame_event_id=str(frame_id))]))).fetchall()
    if len(rows) != 1:
        raise Conflict('Generic raw QR frame requires one completed local task checkpoint')
    row = rows[0]
    if (fingerprint(row['normalized_result']) != row['result_fingerprint']
            or row['result'].get('execution_verification') != 'local_model_free_qr_pixels_verified'
            or fingerprint(frame['payload']) != frame['payload_sha256']):
        raise Conflict('Completed raw QR checkpoint or immutable frame fingerprint differs')
    receipt = row['execution_receipt']['invocations'][0]['video_qr']
    points = [p for p in receipt['original_frames'] if p['frame_event_id'] == str(frame_id)]
    if len(points) != 1:
        raise Conflict('One actual raw QR frame reference required')
    point = points[0]
    if (point['frame_event_sha256'] != frame['payload_sha256']
            or point['qr_occurrences'] != frame['payload']['qr_occurrences']
            or point['pts'] != frame['payload']['source_pts']
            or Fraction(point['time_base']) != Fraction(frame['payload']['time_base'])
            or Fraction(point['stream_origin']) != Fraction(frame['payload']['stream_origin'])
            or receipt['source_sha256'] != frame['payload']['source_sha256']
            or receipt['media_version'] != frame['payload']['media_version']
            or receipt['processing_rules'] != row['request']['processing_rules']):
        raise Conflict('Raw QR original PTS/pixels/text differ from completed local evidence')
    return dict(frame=frame, task=row, receipt=receipt, point=point)


def source_proof(conn, core, observation_id, camera_id, qr_value, at, evidence):
    """Optional proof port used by existing OperatorScans.record, never a bypass.

    None means this is an older explicit source, whose original proof still runs.
    A generic authorization that exists but fails validation is rejected.
    """
    from .scan_intake import verified_clock

    photo = _photo(conn, core, observation_id)
    video = None if photo else _video(conn, core, observation_id)
    if photo is None and video is None:
        return None
    media_id = photo['media_id'] if photo else video['frame']['media_id']
    media = _media(conn, core, media_id)
    authorization = _auth(conn, core, media_id)
    if authorization is None:
        return None
    if not _source_matches(core, media, authorization):
        raise Conflict('Generic capture source no longer matches its server-pinned policy')
    camera = core._entity(conn, camera_id, 'capture_device')
    values = photo['result'].get('decoded_values', []) if photo else [q['value'] for q in video['point']['qr_occurrences']]
    original_evidence = photo['evidence_ids'] if photo else video['frame']['payload']['evidence_ids']
    when = media['start_at'] if photo else video['frame']['event_at']
    if (camera['code'] != media['receipt'].get('camera_id') or at is None or at != when
            or not verified_clock(media['time_basis']) or sorted(set(values)) != [qr_value]
            or not evidence or set(map(str, evidence)) != set(map(str, original_evidence))):
        raise Conflict('Generic camera scope, source clock, QR text or evidence does not match')
    if video and str(video['task']['id']) not in authorization['task_ids']:
        raise Conflict('Raw QR task was not authorized by the original admission')
    return clean(dict(qr_job_id=photo['id'] if photo else video['task']['id'],
        source_observation_id=observation_id, source_frame_event_id=None if photo else video['frame']['id'],
        source_sha256=media['verified_sha256'], media_version=media['version'],
        authorization_key=AUTH_PREFIX + str(media_id), authorization_sha256=fingerprint(authorization),
        policy_sha256=authorization['policy_sha256'], original_time_basis=media['time_basis'],
        processing_rules=photo['rule_identity'] if photo else video['receipt']['processing_rules'],
        authority='server_pinned_automatic_equipment_qr_policy', declared_source_sha_present=media['receipt'].get('sha256') is not None))


class CaptureEquipmentIntake:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    def _admit_existing(self, config):
        with self.core.db.transaction(readonly=True) as conn:
            sources = conn.execute('''SELECT m.media_id FROM hj_timeline_media m
                JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                  AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                WHERE m.customer_id=%s AND m.kind=ANY(%s)
                  AND COALESCE(i.payload->'scan_intent','null'::jsonb)='null'::jsonb AND c.source_kind=ANY(%s)
                  AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=m.customer_id
                    AND r.key=%s||m.media_id::text)
                ORDER BY c.occurred_at DESC NULLS LAST,m.media_id LIMIT %s''',
                (self.core.customer, [kind for kind in ('photo','video') if config[kind]],
                 config['source_kinds'], AUTH_PREFIX, config['max_admissions_per_tick'])).fetchall()
        outcomes = []
        for source in sources:
            located = Media(self.core, self.settings).locate(source['media_id'])
            if located['status'] != 'available':
                outcomes.append(dict(media_id=str(source['media_id']), status='source_unavailable'))
                continue

            def action(conn, media_id=source['media_id'], path=located['path']):
                media = _media(conn, self.core, media_id)
                return admit(conn, self.core, self.settings, media, media['receipt'], verified_path=path)
            outcome = self.core.command('capture-equipment-admit:' + str(source['media_id']),
                dict(media_id=str(source['media_id']), policy_sha256=fingerprint(config)), action)
            if outcome:
                outcomes.append(outcome)
        return outcomes

    def _observation(self, conn, authorization, media, observation_id, values, at, evidence, *, video=False):
        from .scan_intake import verified_clock

        if (not isinstance(values, list) or len(values) > 64
                or any(type(v) is not str or not v or len(v.encode()) > 4096 for v in values)):
            raise Conflict('Generic decoded QR values exceed their original bounds')
        unique = sorted(set(values))
        camera = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s",
                              (self.core.customer, media['receipt'].get('camera_id'))).fetchone()
        when = at if verified_clock(media['time_basis']) else None
        ops = OperatorScans(self.core)
        resolution = None
        status = 'no_qr_decoded' if not unique else 'ambiguous_multiple_qr' if len(unique) > 1 else 'unknown_source_camera' if camera is None else 'unknown_time' if when is None else None
        if status is None:
            resolution = ops._camera_resolution(conn, unique[0], when)
            status = resolution['status']
        pins = clean(dict(schema_version=SCHEMA, source_observation_id=observation_id, media_id=media['media_id'],
            source_sha256=media['verified_sha256'], media_version=media['version'],
            authorization_key=AUTH_PREFIX + str(media['media_id']), authorization_sha256=fingerprint(authorization),
            policy_sha256=authorization['policy_sha256'], camera_id=camera['id'] if camera else None,
            original_camera_code=media['receipt'].get('camera_id'), original_time_basis=media['time_basis'],
            decoded_values=unique, source_reported_at=at, event_time_verified=when is not None,
            evidence_ids=evidence, person_inferred=False, material_quantity_written=False))
        if status == 'declared_equipment':
            anchored = conn.execute("SELECT 1 FROM hj_operator_scan_event WHERE customer_id=%s AND payload->'source_evidence'->>'authorization_key'=%s LIMIT 1",
                (self.core.customer, AUTH_PREFIX + str(media['media_id']))).fetchone()
            previous = conn.execute("""SELECT occurred_at,payload FROM hj_operator_scan_event WHERE customer_id=%s
                AND person_id IS NULL AND camera_id=%s AND occurred_at<=%s AND scan_kind='equipment'
                ORDER BY occurred_at DESC,id DESC LIMIT 1""", (self.core.customer, camera['id'], when)).fetchone()
            if (anchored and previous and previous['payload']['resolution'].get('target_id') == resolution['target_id']
                    and 0 <= (when - previous['occurred_at']).total_seconds() < authorization['policy']['presence_interval_seconds']):
                return dict(status='same_equipment_presence_rate_limited', evidence=pins,
                            operator_context_changed=False, original_pixel_observation_retained=True)
            scan = ops.record('capture-equipment-operator:' + str(observation_id), intent=CAMERA_INTENT,
                scan_kind='equipment', qr_value=unique[0], occurred_at=clean(when), evidence_ids=list(map(str, evidence)),
                camera_id=str(camera['id']), actor='configured automatic equipment QR policy',
                reason='source-pinned generic capture equipment observation', source_job_id=str(observation_id))
            return dict(status=scan['status'], scan=scan, evidence=pins)
        if video:
            # The existing video activity already contains every raw frame/QR.
            # A candidate consumer receipt does not duplicate 30 activities/sec.
            return dict(status=status, evidence=pins, resolution=resolution,
                        operator_context_changed=False, original_video_activity_retained=True)
        activity = declaration_activity(conn, self.core, text='通用采集二维码观察：' + status,
            at=when, evidence_ids=evidence, payload=dict(capture_equipment_intake=pins,
                status=status, resolution=resolution, operator_context_changed=False,
                origin='generic_capture_machine_qr_evidence', human_review_performed=False,
                source_type_presentation='machine_qr_observation', experiment_end_implied=False))
        return dict(status=status, activity_id=activity, evidence=pins, operator_context_changed=False)

    def _consume_photo(self, job_id):
        with self.core.db.transaction(readonly=True) as conn:
            photo = _photo(conn, self.core, job_id)
        located = Media(self.core, self.settings).locate(photo['media_id'])
        if located['status'] != 'available':
            raise Conflict('Generic QR original media is unavailable or changed')

        def action(conn):
            Timeline(self.core).lock(conn)
            photo = _photo(conn, self.core, job_id)
            media = _media(conn, self.core, photo['media_id'])
            authorization = _auth(conn, self.core, media['media_id'])
            if not _source_matches(self.core, media, authorization):
                raise Conflict('Generic photo differs from its server source authorization')
            return self._observation(conn, authorization, media, job_id, photo['result']['decoded_values'],
                                     media['start_at'], photo['evidence_ids'])
        return self.core.command(OBS_PREFIX + str(job_id), dict(schema_version=SCHEMA, photo_job_id=str(job_id)), action)

    def _consume_video(self, task_id, budget):
        with self.core.db.transaction(readonly=True) as conn:
            task = conn.execute('SELECT * FROM hj_timeline_task WHERE customer_id=%s AND id=%s AND status=\'completed\' AND kind=\'qr\'',
                                (self.core.customer, ident(task_id))).fetchone()
            media_id = task['media_id']
            authorization = _auth(conn, self.core, media_id)
            position = authorization['task_ids'].index(str(task_id))
            earlier = authorization['task_ids'][:position]
            if earlier:
                finished = conn.execute('SELECT key FROM command_receipt WHERE customer_id=%s AND key=ANY(%s)',
                    (self.core.customer, ['capture-equipment-task-consumed:' + v for v in earlier])).fetchall()
                if len(finished) != len(earlier):
                    return dict(task_id=str(task_id), processed=0, status='deferred',
                        reason='earlier_original_packet_qr_window_not_consumed', outcomes=[],
                        source_order_preserved=True)
            checkpoints = conn.execute('SELECT * FROM hj_timeline_window_result WHERE customer_id=%s AND task_id=%s AND input_fingerprint=%s ORDER BY window_index',
                (self.core.customer, task['id'], task['request']['input_fingerprint'])).fetchall()
            if len(checkpoints) != 1 or fingerprint(checkpoints[0]['normalized_result']) != checkpoints[0]['result_fingerprint']:
                raise Conflict('One intact bounded generic video checkpoint required')
            points = checkpoints[0]['execution_receipt']['invocations'][0]['video_qr']['original_frames']
            if not 1 <= len(points) <= 600:
                raise Conflict('Generic QR original frame count exceeds its task limit')
            done = conn.execute('SELECT key FROM command_receipt WHERE customer_id=%s AND key=ANY(%s)',
                (self.core.customer, [OBS_PREFIX + p['frame_event_id'] for p in points])).fetchall()
        keys = {r['key'] for r in done}
        pending = [p for p in points if OBS_PREFIX + p['frame_event_id'] not in keys]
        selected = pending[:budget]
        if selected:
            located = Media(self.core, self.settings).locate(media_id)
            if located['status'] != 'available':
                raise Conflict('Generic video original source is unavailable or changed')
        outcomes = []
        for point in selected:
            frame_id = point['frame_event_id']

            def action(conn, frame_id=frame_id):
                Timeline(self.core).lock(conn)
                original = _video(conn, self.core, frame_id)
                frame = original['frame']
                media = _media(conn, self.core, media_id)
                authorization = _auth(conn, self.core, media_id)
                if (not _source_matches(self.core, media, authorization)
                        or str(task_id) not in authorization['task_ids']):
                    raise Conflict('Generic QR task differs from server-authorized original media')
                return self._observation(conn, authorization, media, frame_id,
                    [v['value'] for v in frame['payload']['qr_occurrences']], frame['event_at'],
                    frame['payload']['evidence_ids'], video=True)
            outcomes.append(self.core.command(OBS_PREFIX + frame_id,
                dict(schema_version=SCHEMA, video_task_id=str(task_id), frame_event_id=frame_id), action))
        remaining = len(pending) - len(selected)
        if remaining == 0:
            self.core.command('capture-equipment-task-consumed:' + str(task_id),
                dict(task_id=str(task_id), source_frame_event_ids=[p['frame_event_id'] for p in points]),
                lambda conn: dict(status='all_original_decoded_frames_consumed', task_id=str(task_id),
                                  frame_count=len(points), whole_packet_completed=False))
        counts = {}
        for outcome in outcomes:
            counts[outcome['status']] = counts.get(outcome['status'], 0) + 1
        return dict(task_id=str(task_id), processed=len(selected), remaining_frames=remaining,
                    outcomes=outcomes, status_counts=counts, packet_progress=self.packet_progress(media_id))

    def packet_progress(self, media_id):
        with self.core.db.transaction(readonly=True) as conn:
            authorization = _auth(conn, self.core, media_id)
            rows = conn.execute('''SELECT t.id,t.status,r.result AS consumption FROM hj_timeline_task t
                LEFT JOIN command_receipt r ON r.customer_id=t.customer_id
                    AND r.key='capture-equipment-task-consumed:'||t.id::text
                WHERE t.customer_id=%s AND t.id=ANY(%s)''',
                (self.core.customer, [ident(v) for v in authorization['task_ids']])).fetchall()
        consumed = [r for r in rows if r['consumption'] and r['consumption']['status'] == 'all_original_decoded_frames_consumed']
        count = sum(r['consumption']['frame_count'] for r in consumed)
        expected = authorization['technical_probe']['declared_frame_count']
        complete = len(rows) == len(consumed) == len(authorization['task_ids']) and count == expected
        return dict(media_id=str(media_id), planned_tasks=len(authorization['task_ids']),
            completed_tasks=sum(r['status'] == 'completed' for r in rows), consumed_tasks=len(consumed),
            actual_decoded_and_consumed_frames=count, independently_probed_container_frame_count=expected,
            whole_packet_frame_coverage='PROVEN' if complete else 'NOT_PROVEN',
            source_policy='all_decoded_native_frames_no_sampling', model_or_field_accuracy='NOT_PROVEN',
            throughput_or_quality_accepted=False)

    def _failure(self, source_id, exc):
        return self.core.command('capture-equipment-failed:' + str(source_id),
            dict(source_id=str(source_id)), lambda conn: dict(status='needs_attention',
                source_id=str(source_id), error_code=type(exc).__name__, reason=str(exc)[:512],
                automatic_retry=False, original_media_retained=True))

    def tick(self):
        config = policy(self.settings)
        if self.settings.processing_profile != 'timeline-v1' or not config.get('enabled'):
            return dict(enabled=False)
        admissions = self._admit_existing(config)
        budget = config['max_observations_per_tick']
        with self.core.db.transaction(readonly=True) as conn:
            photos = conn.execute('''SELECT j.id FROM processing_job j JOIN media_asset a
                ON a.customer_id=j.customer_id AND a.capture_id=j.capture_id
                JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
                JOIN command_receipt p ON p.customer_id=a.customer_id AND p.key=%s||a.id::text
                WHERE j.customer_id=%s AND j.worker_kind='qr' AND j.status='completed'
                  AND p.result->>'status'='authorized'
                  AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=j.customer_id AND r.key=%s||j.id::text)
                  AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=j.customer_id AND r.key='capture-equipment-failed:'||j.id::text)
                ORDER BY c.occurred_at DESC NULLS LAST,j.created_at DESC,j.id LIMIT %s''', (AUTH_PREFIX, self.core.customer, OBS_PREFIX, budget)).fetchall()
        photo_outcomes = []
        for row in photos:
            try:
                result = self._consume_photo(row['id'])
            except (ValueError, Missing, OSError, RuntimeError) as exc:
                result = self._failure(row['id'], exc)
            photo_outcomes.append(dict(job_id=str(row['id']), **result))
        budget -= len(photos)
        video_outcomes = []
        if budget:
            with self.core.db.transaction(readonly=True) as conn:
                tasks = conn.execute('''SELECT t.id FROM hj_timeline_task t JOIN command_receipt p
                    ON p.customer_id=t.customer_id AND p.key=%s||t.media_id::text
                    JOIN hj_timeline_media m ON m.customer_id=t.customer_id AND m.media_id=t.media_id
                    WHERE t.customer_id=%s AND t.kind='qr' AND t.status='completed'
                      AND t.request->'video_qr'->>'schema_version'='vc-lab-video-qr/2'
                      AND p.result->>'status'='authorized'
                      AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=t.customer_id
                          AND r.key='capture-equipment-task-consumed:'||t.id::text)
                      AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=t.customer_id
                          AND r.key='capture-equipment-failed:'||t.id::text)
                    ORDER BY m.start_at DESC NULLS LAST,t.media_id,
                        (t.request->'video_qr'->'source_intervals'->0->>'start_seconds')::numeric,t.id LIMIT %s''',
                    (AUTH_PREFIX, self.core.customer, config['max_admissions_per_tick'])).fetchall()
            for task in tasks:
                if budget <= 0:
                    break
                try:
                    outcome = self._consume_video(task['id'], budget)
                except (ValueError, Missing, OSError, RuntimeError) as exc:
                    outcome = dict(self._failure(task['id'], exc), processed=0)
                video_outcomes.append(outcome)
                budget -= outcome['processed']
        return dict(enabled=True, schema_version=SCHEMA, admissions=admissions,
                    photo_outcomes=photo_outcomes, video_outcomes=video_outcomes,
                    processed=len(photos) + sum(v['processed'] for v in video_outcomes),
                    source_default_changed=False, model_calls=0, material_quantity_written=False)

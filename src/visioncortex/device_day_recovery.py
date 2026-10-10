"""Recover sealed originals without depending on a removed capture pathname.

This revalidates retention only. Historical model receipts are never promoted.
The source snapshot must be identical; every archived byte is checked again.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time

from .device_day import exclusive, now
from .device_day_contract import VERSION, atomic_json, digest, read_json, safe_child
from .device_day_verification import verify_artifact_cached


def source_identity(record):
    return {k: v for k, v in record.items()
            if k not in {'processing_priority', 'archive_date', 'updated_at'}}


def verified_original(runner, record, receipt):
    """Require the complete sealed snapshot, not just a video with the same name."""
    if (receipt.get('schema_version') != VERSION or receipt.get('stage') != 'retention'
            or receipt.get('status') != 'completed'
            or receipt.get('recording_id') != record['recording_id']
            or source_identity(receipt.get('recording', {})) != source_identity(record)
            or not record.get('source_signature')):
        return False
    configured_root = Path(runner.config['collection_ingest']['source_root'])
    capture_root = configured_root.resolve()
    if Path(receipt.get('capture_root', '')).resolve() != capture_root:
        return False
    sources, artifacts = receipt.get('sources', []), receipt.get('artifacts', [])
    if not sources or not artifacts or receipt.get('audit_artifacts'):
        return False
    if sorted(s.get('kind') for s in sources if s.get('kind') in {'video', 'clock'}) != ['clock', 'video']:
        return False
    if sorted(digest(s['retained']) for s in sources) != sorted(digest(a) for a in artifacts):
        return False
    layout = runner.layout(record)
    video = Path(record['video_path'])
    prefix = video.name[:-len('rgb.mp4')] if video.name.endswith('rgb.mp4') else ''
    by_path = {s['original_path']: s for s in sources}
    if len(by_path) != len(sources):
        return False
    snapshots = []
    for original in (video, Path(record['frames_path']), video.with_name(prefix + 'meta.json'),
                     video.with_name(prefix + 'recording_ready.json')):
        source = by_path.get(str(original))
        if source:
            snapshots.append((original.relative_to(configured_root).as_posix(), source['size_bytes'], source['mtime_ns']))
    if hashlib.sha256(json.dumps(snapshots).encode()).hexdigest() != record['source_signature']:
        return False
    expected_audio_sources = sorted(('audio_' + s['kind'], s['path']) for s in record.get('audio', {}).get('files', []))
    if sorted((s['kind'], s['original_path']) for s in sources if s['kind'].startswith('audio_')) != expected_audio_sources:
        return False
    for source in sources:
        original = Path(source['original_path'])
        reference = source['retained']
        if original.parent != video.parent or not original.resolve().is_relative_to(capture_root):
            return False
        if source['kind'] in {'video', 'clock'}:
            field = 'video_path' if source['kind'] == 'video' else 'frames_path'
            if str(original) != record[field]:
                return False
        if (source['sha256'] != reference['sha256'] or source['size_bytes'] != reference['size_bytes']
                or reference['path'] != layout.relative(layout.source_path(record, source['kind'], original))
                or not reference['path'].startswith('MetaVideo/')):
            return False
        if source['kind'] == 'video' and source['size_bytes'] != record['size_bytes']:
            return False
        if not verify_artifact_cached(layout.root, reference):
            return False
    audio = receipt.get('audio', {}).get('artifacts', [])
    expected_audio = [s['retained'] | {'kind': s['kind']} for s in sources if s['kind'].startswith('audio_')]
    return sorted(map(digest, audio)) == sorted(map(digest, expected_audio))


def recover_one(runner, row):
    record = json.loads(row['payload'])
    rid = record['recording_id']
    layout = runner.layout(record)
    path = runner._receipt(layout, record, 'retention')
    # No overwrite of a live or replaced capture source, even with an old seal.
    try:
        Path(record['video_path']).lstat()
        return {'status': 'capture_present', 'recording_id': rid}
    except FileNotFoundError:
        pass
    queue = runner.queues['retention']
    with ExitStack() as locks:
        for stage in ('all', 'retention'):
            locks.enter_context(exclusive(runner.runtime_root / 'locks' / f'{rid}.{stage}.lock'))
        candidates = [path, *sorted((path.parent / 'history').glob('retention-*.json'), reverse=True)[:64]]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                receipt = read_json(candidate)
                if not verified_original(runner, record, receipt):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            key = runner._key('retention', record, record)
            revision = runner._key('retention', record, [record, None])
            proof = {'status': 'verified_archived_original', 'recording_id': rid,
                     'verified_at': now(), 'historical_receipt': candidate.relative_to(runner.backend_root).as_posix(),
                     'historical_receipt_digest': digest(receipt), 'previous_key': receipt.get('key'),
                     'current_key': key, 'artifact_count': len(receipt['artifacts']),
                     'source_snapshot_unchanged': True, 'capture_source_reinspected': False,
                     'capture_source_changed': False, 'model_invoked': False}
            recovered = receipt | {'key': key, 'recording': record, 'recovery': proof}
            result = {'schema_version': VERSION, 'archive': layout.name, 'recording_id': rid,
                      'stage': 'retention', 'status': 'completed', 'recovered_from_archive': True}
            try:
                Path(record['video_path']).lstat()
                return {'status': 'capture_present', 'recording_id': rid}
            except FileNotFoundError:
                pass
            # Publish before the queue commit. A crash between them is recoverable
            # from the current receipt. CAS leaves new inputs/owners untouched.
            with queue.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                current = db.execute('SELECT * FROM recordings WHERE recording_id=?', (rid,)).fetchone()
                if (not current or current['status'] not in {'failed', 'queued'} or current['revision'] != row['revision']
                        or current['payload'] != row['payload'] or current['updated_at'] != row['updated_at']):
                    return {'status': 'queue_changed', 'recording_id': rid}
                if path.is_file():
                    previous = read_json(path)
                    atomic_json(path.parent / 'history' / f'retention-{digest(previous)}.json', previous)
                atomic_json(path, recovered)
                atomic_json(safe_child(runner.runtime_root, f'RetentionRecovery/{rid}.json'), proof)
                db.execute("UPDATE recordings SET status='completed',revision=?,result=?,completed_at=?,"
                           "wall_seconds=0,lease_owner=NULL,lease_until=NULL,updated_at=? WHERE recording_id=?",
                           (revision, json.dumps(result), time.time(), time.time(), rid))
            return result
    return {'status': 'no_verified_archive', 'recording_id': rid}


class RetentionRecovery:
    """One bounded archive verification on its own worker; live intake stays free."""

    def __init__(self):
        self.checked = {}

    def tick(self, runner):
        # Admission only posts one local-receipt snapshot. All NAS metadata and
        # byte checks for a waiting stage run on this existing bounded worker.
        verified = runner._prerequisite_checks.verify_pending(runner)
        if verified and verified['prerequisite_artifacts_verified']:
            return verified
        # Unblock already waiting preprocessing before investigating capture
        # paths which have never supplied a downstream task.
        with runner.queues['vision'].connect() as db:
            waiting = {row[0] for row in db.execute("SELECT recording_id FROM recordings WHERE status='queued' "
                "OR (status='running' AND COALESCE(lease_until,0)<?)", (time.time(),))}
        blocked = set()
        for stage in ('vision', 'stt', 'understanding', 'report'):
            with runner.queues[stage].connect() as db:
                blocked.update(row[0] for row in db.execute(
                    "SELECT recording_id FROM recordings WHERE status='waiting_for_prerequisite' "
                    "AND json_extract(result,'$.prerequisite_stage')='retention'"))
        waiting.update(blocked)
        with runner.queues['retention'].connect() as db:
            rows = list(db.execute("SELECT * FROM recordings WHERE (status='failed' AND json_extract(result,'$.error_type')='FileNotFoundError') "
                                   "OR (status='queued' AND input_status='missing') "
                                   "OR (status='completed' AND recording_id IN (SELECT value FROM json_each(?))) "
                                   "ORDER BY COALESCE(json_extract(payload,'$.processing_priority'),0),"
                                   "json_extract(payload,'$.recording_start_us')", (json.dumps(sorted(blocked)),)))
        rows.sort(key=lambda row: row['recording_id'] not in waiting)
        from .device_day_schedule import in_processing_scope
        for row in rows:
            if not in_processing_scope(runner.settings, json.loads(row['payload'])):
                continue
            identity = (row['revision'], row['updated_at'])
            previous = self.checked.get(row['recording_id'])
            if previous and previous[0] == identity and time.monotonic() < previous[1]:
                continue
            self.checked[row['recording_id']] = (identity, time.monotonic() + 900)
            try:
                result = (repair_completed_retention(runner, row) if row['status'] == 'completed'
                          else recover_one(runner, row))
                if result.get('status') == 'capture_present':
                    result = resume_reappeared_input(runner, row)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {'status': 'verification_unavailable', 'recording_id': row['recording_id'],
                          'error_type': type(exc).__name__}
            status_path = runner.runtime_root / 'InputAvailability' / f"{row['recording_id']}.json"
            previous_status = read_json(status_path) if status_path.is_file() else {}
            atomic_json(status_path, previous_status | result | {'checked_at': now(),
                        'queue_revision': row['revision'], 'queue_updated_at': row['updated_at']})
            return result
        return {'status': 'idle'}


def repair_completed_retention(runner, row):
    """Requeue one damaged archive only after its unchanged source is ready.

    A downstream metadata rejection alone never revokes historical completion.
    Recheck the exact receipt and artifacts on this bounded recovery worker,
    before and after inspecting the source. Transient/unreadable sources stay
    blocked; a repair identity can be scheduled only once, durably and by CAS.
    """
    from .nas_recordings import _inspect
    record = json.loads(row['payload'])
    rid = record['recording_id']
    layout = runner.layout(record)
    path = runner._receipt(layout, record, 'retention')
    try:
        receipt = read_json(path)
    except FileNotFoundError:
        receipt = None
    key = runner._key('retention', record, record)
    if receipt is not None and (receipt.get('status') != 'completed' or not runner._accepts_receipt(receipt, key)):
        return {'status': 'retention_identity_changed', 'recording_id': rid}
    if runner._load(path, key, layout) is not None:
        return {'status': 'prerequisite_restored', 'recording_id': rid}
    root = Path(runner.config['collection_ingest']['source_root'])
    video = Path(record['video_path'])
    try:
        video.stat()
        if not layout.raw.is_dir():
            return {'status': 'archive_unavailable', 'recording_id': rid}
        source_observed_at = time.time()
        fresh = _inspect(root, video, source_observed_at,
                         float(runner.config['collection_ingest'].get('settle_seconds', 120)))
        identities = []
        for source in (video, Path(record['frames_path'])):
            info = source.stat()
            identities.append((str(source), info.st_dev, info.st_ino, info.st_size,
                               info.st_mtime_ns, info.st_ctime_ns))
    except OSError as exc:
        return {'status': 'waiting_for_retention_source', 'recording_id': rid,
                'error_type': type(exc).__name__}
    if not fresh.get('processable', fresh.get('available')):
        return {'status': 'capture_not_ready', 'recording_id': rid}
    if (fresh.get('source_signature') != record.get('source_signature')
            or fresh.get('audio', {}).get('source_signature') != record.get('audio', {}).get('source_signature')):
        return {'status': 'changed_input_waiting_for_monitor', 'recording_id': rid}
    try:
        current_receipt = read_json(path)
    except FileNotFoundError:
        current_receipt = None
    if current_receipt != receipt:
        return {'status': 'retention_identity_changed', 'recording_id': rid}
    if runner._load(path, key, layout) is not None:
        return {'status': 'prerequisite_restored', 'recording_id': rid}
    identity = digest([identities, digest(receipt)])
    result = {'status': 'retention_repair_queued', 'recording_id': rid,
              'repair_identity': identity, 'receipt_digest': digest(receipt),
              'receipt_missing': receipt is None,
              'source_signature_unchanged': True, 'model_invoked': False,
              'previous_attempts': row['attempts']}
    with runner.queues['retention'].connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM task_events WHERE task_id=? AND state='retention_repair_queued' "
                      "AND json_extract(data,'$.repair_identity')=? LIMIT 1", (rid, identity)).fetchone():
            return {'status': 'unchanged_retention_repair_already_attempted', 'recording_id': rid}
        changed = db.execute("UPDATE recordings SET status='queued',attempts=0,input_status='ready',"
                             "result=?,queued_at=?,updated_at=?,lease_owner=NULL,lease_until=NULL "
                             "WHERE recording_id=? AND status='completed' AND revision=? AND updated_at=? AND payload=?",
                             (json.dumps(result), time.time(), time.time(), rid, row['revision'],
                              row['updated_at'], row['payload'])).rowcount
        if not changed:
            return {'status': 'queue_changed', 'recording_id': rid}
        from .task_events import append
        append(db, rid, 'retention_repair_queued', revision=row['revision'], data=result)
    from .input_availability import Availability
    Availability(runner.runtime_root).mark(record, 'ready', observed=source_observed_at,
                                          reason='verified_retention_repair_source')
    return result


def resume_reappeared_input(runner, row):
    """Retry an actually restored source once per file identity, never paid stages."""
    from .nas_recordings import _inspect
    record = json.loads(row['payload'])
    rid = record['recording_id']
    root = Path(runner.config['collection_ingest']['source_root'])
    video = Path(record['video_path'])
    fresh = _inspect(root, video, time.time(),
                     float(runner.config['collection_ingest'].get('settle_seconds', 120)))
    if not fresh.get('processable', fresh.get('available')):
        return {'status': 'capture_not_ready', 'recording_id': rid}
    if fresh.get('source_signature') != record.get('source_signature'):
        return {'status': 'changed_input_waiting_for_monitor', 'recording_id': rid}
    identity = []
    for path in (video, Path(record['frames_path'])):
        stat = path.stat()
        identity.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    fingerprint = digest(identity)
    path = runner.runtime_root / 'InputAvailability' / f'{rid}.json'
    previous = read_json(path) if path.is_file() else {}
    if previous.get('retried_file_identity') == fingerprint:
        return {'status': 'unchanged_input_retry_already_attempted', 'recording_id': rid}
    with runner.queues['retention'].connect() as db:
        changed = db.execute("UPDATE recordings SET status='queued',attempts=0,queued_at=?,updated_at=?,"
                             "lease_owner=NULL,lease_until=NULL WHERE recording_id=? AND status='failed' "
                             "AND revision=? AND updated_at=?",
                             (time.time(), time.time(), rid, row['revision'], row['updated_at'])).rowcount
    return {'status': 'restored_input_queued' if changed else 'queue_changed', 'recording_id': rid,
            **({'retried_file_identity': fingerprint} if changed else {})}


def repair_missing_index(runner, name):
    """Regenerate a missing user index through the existing current-receipt filter."""
    from .device_day_contract import validate_archive_name
    validate_archive_name(name)
    index = safe_child(runner.archive_root, name + '/ProcessedClips/Index.json')
    if index.exists():
        return {'status': 'already_present', 'archive': name}
    root = safe_child(runner.backend_root, 'device-day-receipts/' + name)
    for path in sorted(root.glob('*/retention.json')):
        try:
            receipt = read_json(path)
            record = receipt.get('recording')
            if not record or runner.layout(record).name != name:
                continue
            runner.refresh_index(runner.layout(record))
            if not index.is_file():
                return {'status': 'waiting_for_index_lock', 'archive': name}
            proof = {'status': 'rebuilt_from_stage_receipts', 'archive': name, 'completed_at': now(),
                     'index_digest': digest(read_json(index)), 'model_invoked': False,
                     'source_media_changed': False, 'selection': 'canonical_current_receipt_filter'}
            atomic_json(runner.runtime_root / 'IndexRecovery' / (name + '.json'), proof)
            return proof
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {'status': 'verification_unavailable', 'archive': name, 'error_type': type(exc).__name__}
    return {'status': 'no_source_receipt', 'archive': name}

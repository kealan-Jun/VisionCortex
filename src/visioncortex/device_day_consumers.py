"""Local consumer ownership evidence; no NAS, model startup or queue mutation."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time

from .device_day_contract import DEPENDENCIES, STAGES, atomic_json
from .device_day_night_schedule import paused_stages, stage_admitted
from .device_day_schedule import in_processing_scope
from .sqlite_store import connection
from .device_day_retry import retry_limit


VERSION = 'visioncortex-device-day-consumer/1'


def process_start_ticks(pid, proc_root=Path('/proc')):
    """A PID alone cannot identify an owner after process ID reuse."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        value = (Path(proc_root) / str(pid) / 'stat').read_text()
        return int(value.rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class WorkerStatus:
    """Keep ownership fresh while a single worker spends minutes in process()."""
    def __init__(self, config, stage, *, recent_seconds=14400, backfill_enabled=False,
                 interval=2):
        if stage not in STAGES:
            raise ValueError('Unknown device/day stage')
        from .build_identity import identity
        pid = os.getpid()
        ticks = process_start_ticks(pid)
        root = Path(config['storage']['local_runtime_root']) / 'device-day' / 'Consumers'
        self.path = root / f'{stage}-{pid}-{ticks}.json'
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        scopes = ['recent', 'history'] if recent_seconds is None or backfill_enabled else ['recent']
        build = identity()
        self.state = {'schema_version': VERSION, 'stage': stage, 'pid': pid,
                      'process_start_ticks': ticks, 'build': build,
                      'supported_scopes': scopes, 'recent_seconds': recent_seconds,
                      'backfill_enabled': bool(backfill_enabled), 'status': 'starting',
                      'scope': 'recent', 'started_at': time.time()}

    def _publish(self):
        with self.lock:
            atomic_json(self.path, self.state | {'at': time.time()})

    def update(self, *, status='running', scope=None, last_result=None, **fields):
        # Persist only operational fields, never arbitrary result messages or
        # configuration/provider payloads that may contain credentials.
        safe = {key: value for key, value in fields.items() if key in
                {'recording_id', 'backfill_status', 'error_type', 'last_success_at'}}
        if last_result is not None:
            safe['last_result'] = {key: last_result[key] for key in
                                  ('status', 'recording_id', 'wall_seconds', 'admitted', 'reason', 'error_type')
                                  if key in last_result}
        if scope is not None:
            if scope not in {'recent', 'history', 'all'}:
                raise ValueError('Unknown consumer scope')
            safe['scope'] = scope
        with self.lock:
            self.state.update(safe, status=status)
        self._publish()

    def __enter__(self):
        self.update(status='running')
        def heartbeat():
            while not self.stop.wait(self.interval):
                try:
                    self._publish()
                except OSError:
                    # A failed local status write expires naturally. It must
                    # not kill an already leased model or archive operation.
                    pass
        self.thread = threading.Thread(target=heartbeat, name='consumer-heartbeat', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5, self.interval * 2))
        self.update(status='stopped', error_type=exc_type.__name__ if exc_type else None)


def _owners(root, now, proc_root):
    owners, errors = {stage: [] for stage in STAGES}, []
    folder = root / 'Consumers'
    if not folder.is_dir():
        return owners, errors
    for path in sorted(folder.glob('*.json')):
        try:
            data = json.loads(path.read_text())
            if data.get('schema_version') != VERSION or data.get('stage') not in STAGES:
                raise ValueError('Invalid consumer record')
            stamp = data.get('at')
            ticks = data.get('process_start_ticks')
            scopes = data.get('supported_scopes')
            commit = (data.get('build') or {}).get('commit')
            age = now - stamp if type(stamp) in (int, float) else float('inf')
            if not math.isfinite(age) or age < -5 or age > 15:
                state = 'stale'
            elif data.get('status') in {'stopped', 'failed'}:
                state = data['status']
            elif (type(ticks) is not int or ticks <= 0 or not isinstance(scopes, list)
                  or not scopes or set(scopes) - {'recent', 'history'}
                  or not isinstance(commit, str) or re.fullmatch(r'[0-9a-f]{40}', commit) is None):
                state = 'identity_unverified'
            elif process_start_ticks(data.get('pid'), proc_root) != ticks:
                state = 'owner_absent_or_replaced'
            else:
                state = 'verified'
            owners[data['stage']].append({key: data.get(key) for key in
                ('pid', 'process_start_ticks', 'build', 'status', 'scope', 'supported_scopes',
                 'backfill_enabled', 'last_result', 'backfill_status')} | {
                    'owner_state': state, 'heartbeat_age_seconds': max(0, age) if math.isfinite(age) else None})
        except (OSError, ValueError, TypeError, AttributeError):
            errors.append('consumer_status_unavailable')
    return owners, errors


def _local_rows(root):
    queues, errors = {}, []
    for stage in STAGES:
        path = root / f'queue-{stage}.sqlite3'
        queues[stage] = {}
        if not path.is_file():
            continue
        try:
            with connection(path, readonly=True, timeout=.25) as db:
                columns = {row[1] for row in db.execute('PRAGMA table_info(recordings)')}
                limit = 'retry_limit' if 'retry_limit' in columns else '0 AS retry_limit'
                queues[stage] = {row['recording_id']: dict(row) for row in db.execute(
                    'SELECT recording_id,payload,status,input_status,lease_until,attempts,updated_at,'
                    + limit + ' FROM recordings')}
        except (OSError, sqlite3.Error):
            errors.append(f'{stage}_queue_unavailable')
    inputs = {}
    path = root / 'InputAvailability.sqlite3'
    if path.is_file():
        try:
            with connection(path, readonly=True, timeout=.25) as db:
                inputs = {row['id']: dict(row) for row in db.execute('SELECT id,signature,state FROM inputs')}
        except (OSError, sqlite3.Error):
            errors.append('input_availability_unavailable')
    return queues, inputs, errors


def consumer_snapshot(config, *, now=None, proc_root=Path('/proc')):
    """Expose missing scope ownership from local hints, not runtime proof.

    Queue metadata can identify a possible ready backlog. Receipt identity and
    media validation remain execution gates; this reader never performs them.
    """
    current = time.time() if now is None else now
    settings = config.get('device_day') or {}
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    owners, errors = _owners(root, current, proc_root)
    queues, inputs, queue_errors = _local_rows(root)
    errors.extend(queue_errors)
    cutoff = (current - settings.get('live_priority_seconds', 14400)) * 1_000_000
    paused = paused_stages(config)
    result = {}
    for stage in STAGES:
        counts = {scope: {'pending': 0, 'locally_ready': 0, 'running': 0, 'blocked': 0}
                  for scope in ('recent', 'history')}
        for rid, row in queues[stage].items():
            if row['status'] == 'completed':
                continue
            try:
                record = json.loads(row['payload'])
                if not in_processing_scope(settings, record):
                    continue
                captured = max(record.get('recording_start_us') or 0, record.get('recording_end_us') or 0)
                scope = 'recent' if captured >= cutoff else 'history'
                count = counts[scope]
                if row['status'] == 'running' and (row['lease_until'] or 0) >= current:
                    count['running'] += 1
                    continue
                count['pending'] += 1
                available = inputs.get(rid, {})
                input_ready = (available.get('signature') != record.get('source_signature')
                               or available.get('state', 'ready') == 'ready')
                retry_ready = (row['status'] != 'failed' or
                               row['attempts'] < retry_limit(row, settings.get('failure_retry_limit') or 3)
                               and current - row['updated_at'] >= 60)
                parents = DEPENDENCIES[stage]
                # Inplace vision/STT can run before the archival queue finishes.
                if settings.get('inplace_preprocessing') and stage in {'vision', 'stt'}:
                    parents = ()
                parents_ready = all(queues[parent].get(rid, {}).get('status') == 'completed' for parent in parents)
                if (row['status'] in {'queued', 'failed', 'running'} and row['input_status'] == 'ready'
                        and record.get('processable', record.get('available', False)) and input_ready
                        and retry_ready and parents_ready
                        and record.get('configured_role') in {'first_person', 'third_person'}):
                    count['locally_ready'] += 1
                else:
                    count['blocked'] += 1
            except (ValueError, TypeError, AttributeError):
                errors.append(f'{stage}_queue_payload_unavailable')
        verified = [owner for owner in owners[stage] if owner['owner_state'] == 'verified']
        coverage = {}
        for scope, count in counts.items():
            covered = any(scope in owner['supported_scopes'] for owner in verified)
            if not settings.get('enabled'):
                state = 'disabled'
            elif stage in paused:
                state = 'paused_by_user'
            elif not stage_admitted(config, stage, datetime.fromtimestamp(current).astimezone()):
                state = 'waiting_for_night_window'
            elif covered:
                state = 'covered'
            elif any(error == f'{name}_queue_unavailable' for name in (stage, *DEPENDENCIES[stage])
                     for error in queue_errors) or 'input_availability_unavailable' in queue_errors:
                state = 'evidence_unavailable'
            elif count['locally_ready']:
                state = 'no_consumer'
            else:
                state = 'no_ready_work'
            coverage[scope] = count | {'status': state, 'verified_owner_count': sum(
                scope in owner['supported_scopes'] for owner in verified)}
        result[stage] = {'scopes': coverage, 'owners': owners[stage],
                         'duplicate_owner_scopes': [scope for scope in counts if
                            sum(scope in owner['supported_scopes'] for owner in verified) > 1]}
    return {'schema_version': VERSION, 'observed_at': current, 'stages': result,
            'errors': sorted(set(errors)), 'local_metadata_only': True,
            'readiness_boundary': 'receipt_and_media_validation_required'}

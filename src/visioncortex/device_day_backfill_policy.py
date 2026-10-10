"""Read-only local admission observations for opportunistic device/day backfill.

These observations are scheduling hints, never input or artifact acceptance.
They read the existing monitor and SQLite catalogs only; a worker must still
acquire its leases and verify the selected recording before execution.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from threading import Lock
import time

from .device_day_contract import DEPENDENCIES, STAGES
from .device_day_night_schedule import paused_stages, stage_admitted


BACKFILL_SOURCE = 'device_day_backfill'
_MONITOR_TAIL_BYTES = 256 * 1024
_MONITOR_ROOT_LIMIT = 32
_MONITOR_LANE_LIMIT = 128
_MONITOR_LOCK = Lock()
_MONITOR_SUCCESSES = OrderedDict()
_SHARED = {
    'retention': {'nas', 'storage', 'cpu'},
    'vision': {'nas', 'vision', 'cpu'},
    'stt': {'nas', 'stt', 'cpu', 'cloud'},
    'understanding': {'nas', 'cpu', 'cloud'},
    'report': {'nas', 'cpu', 'storage'},
}
_RESOURCES = {
    'nas-io': 'nas', 'nas-io-read': 'nas', 'nas-live-vision': 'nas',
    'nas-copy': 'nas', 'storage': 'storage', 'vision': 'vision',
    'cpu': 'cpu', 'cloud': 'cloud', 'stt': 'stt',
}


class PolicyUnavailable(ValueError):
    """Local observations cannot safely establish idle capacity."""


def _options(config, now):
    root = Path(config['storage']['local_runtime_root'])
    if not root.is_absolute() or str(root).startswith(('//', '\\\\')):
        raise PolicyUnavailable('local_runtime_root_required')
    current = time.time() if now is None else float(now)
    settings = config.get('device_day') or {}
    policy = settings.get('backfill') or {}
    horizon = float(settings.get('live_priority_seconds', 14400))
    cooldown = float(policy.get('failure_cooldown_seconds', 60))
    if (not math.isfinite(current) or current < 0 or not math.isfinite(horizon)
            or horizon <= 0 or not math.isfinite(cooldown) or cooldown < 0):
        raise PolicyUnavailable('invalid_backfill_clock_or_window')
    return root, current, int((current - horizon) * 1_000_000), cooldown


def _monitor(config, root, now):
    # Only complete, verified observations can authorize the brief scanning
    # state which the existing discovery publisher emits on every normal poll.
    # Keep this evidence process-local, bounded and isolated by runtime root.
    root_key = os.path.abspath(root)
    with _MONITOR_LOCK:
        try:
            return _monitor_snapshot(config, root, now, root_key)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError):
            _MONITOR_SUCCESSES.pop(root_key, None)
            raise


def _monitor_snapshot(config, root, now, root_key):
    if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
        raise PolicyUnavailable('invalid_backfill_clock_or_window')
    # A caller may capture now and then lose a race to another reader. Only
    # the wall clock sampled under our lock can prove actual clock reversal.
    clock_at = time.time()
    if not math.isfinite(clock_at) or clock_at < 0:
        raise PolicyUnavailable('invalid_backfill_clock_or_window')
    path = root / 'state' / 'nas-recording-monitor.json'
    # The canonical publisher puts the top-level monitor object last. Read its
    # bounded tail rather than decoding every historical recording on each tick.
    # Small and legacy compact snapshots remain supported within the same bound.
    with path.open('rb') as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - _MONITOR_TAIL_BYTES))
        raw = handle.read(_MONITOR_TAIL_BYTES)
        if size > _MONITOR_TAIL_BYTES:
            # The bounded seek may land inside an unrelated UTF-8 character.
            while raw and raw[0] & 0xC0 == 0x80:
                raw = raw[1:]
        tail = raw.decode('utf-8')
    if size <= _MONITOR_TAIL_BYTES:
        data = json.loads(tail)
    else:
        data = {}
        decoder = json.JSONDecoder()
        for metadata_key in ('monitor', 'truncated', 'discovery_lanes', 'pending_publication_count',
                    'pending_live_publication_count'):
            marker = '\n  ' + json.dumps(metadata_key) + ':'
            at = tail.rfind(marker)
            if at < 0:
                if metadata_key == 'monitor':
                    raise PolicyUnavailable('monitor_metadata_outside_bound')
                continue
            value = tail[at + len(marker):].lstrip()
            parsed, end = decoder.raw_decode(value)
            data[metadata_key] = parsed
            if metadata_key == 'monitor' and value[end:].strip() != '}':
                raise PolicyUnavailable('monitor_snapshot_incomplete')
    if not isinstance(data, dict):
        raise PolicyUnavailable('monitor_snapshot_invalid')
    monitor = data.get('monitor')
    if not isinstance(monitor, dict) or monitor.get('status') != 'watching':
        raise PolicyUnavailable('monitor_not_watching')
    stamp = datetime.fromisoformat(monitor['observed_at'])
    if stamp.tzinfo is None:
        raise PolicyUnavailable('monitor_time_has_no_timezone')
    poll = float(monitor.get('poll_seconds', config.get('collection_ingest', {}).get('poll_seconds', 5)))
    limit = float((config.get('device_day', {}).get('backfill') or {}).get(
        'monitor_max_age_seconds', max(30, 3 * poll)))
    age = now - stamp.timestamp()
    if (not math.isfinite(poll) or poll <= 0 or not math.isfinite(limit) or limit <= 0
            or not math.isfinite(age) or age < -5 or age > limit):
        raise PolicyUnavailable('monitor_stale')
    previous = _MONITOR_SUCCESSES.get(root_key)
    if previous and (clock_at < previous['clock_at'] or stamp.timestamp() < previous['snapshot_at']):
        raise PolicyUnavailable('monitor_time_reversed')
    pending_live = data.get('pending_live_publication_count', data.get('pending_publication_count', 0))
    if type(pending_live) is not int or pending_live < 0:
        raise PolicyUnavailable('monitor_pending_count_invalid')
    if data.get('truncated') or pending_live:
        raise PolicyUnavailable('monitor_inventory_incomplete')
    reusable_snapshot = (data.get('truncated') is False
                         and ('pending_live_publication_count' in data or 'pending_publication_count' in data))
    if not reusable_snapshot:
        # Preserve the legacy watching gate, but unknown global completeness
        # cannot seed or inherit successful evidence for a later normal scan.
        _MONITOR_SUCCESSES.pop(root_key, None)
        previous = None
    lanes = data.get('discovery_lanes')
    if lanes is None and config.get('device_day', {}).get('camera_lanes'):
        raise PolicyUnavailable('monitor_live_lanes_missing')
    completed_lanes = {}
    if lanes is not None:
        if not isinstance(lanes, list) or any(not isinstance(lane, dict) for lane in lanes):
            raise PolicyUnavailable('monitor_lanes_invalid')
        live = [lane for lane in lanes if lane.get('mode') == 'live']
        if not live or len(live) > _MONITOR_LANE_LIMIT:
            raise PolicyUnavailable('monitor_live_lanes_missing')
        cameras = set()
        for lane in live:
            camera = lane.get('camera_key')
            if not isinstance(camera, str) or not camera or len(camera) > 256 or camera in cameras:
                raise PolicyUnavailable('monitor_lanes_invalid')
            cameras.add(camera)
            if (lane.get('thread_alive') is not True or lane.get('errors') or lane.get('truncated')):
                raise PolicyUnavailable('monitor_live_lane_unavailable')
            if lane.get('status') == 'watching':
                completed = lane.get('completed_at')
                if (type(completed) not in (int, float) or not math.isfinite(completed)
                        or completed <= 0 or not -5 <= now - completed <= limit):
                    raise PolicyUnavailable('monitor_live_lane_stale')
                old_completed = (previous or {}).get('lanes', {}).get(camera)
                if old_completed is not None and completed < old_completed:
                    raise PolicyUnavailable('monitor_time_reversed')
                # Missing legacy fields do not establish reusable success
                # evidence, even if the original watching gate can accept it.
                if lane.get('errors') == [] and lane.get('truncated') is False:
                    completed_lanes[camera] = completed
            elif lane.get('status') == 'scanning':
                if not reusable_snapshot:
                    raise PolicyUnavailable('monitor_inventory_incomplete')
                completed = (previous or {}).get('lanes', {}).get(camera)
                started = lane.get('started_at')
                if ('errors' in lane and lane['errors'] != []
                        or 'truncated' in lane and lane['truncated'] is not False
                        or type(started) not in (int, float) or not math.isfinite(started)
                        or completed is None or started < completed
                        or not -5 <= now - started <= limit
                        or not -5 <= now - completed <= limit):
                    raise PolicyUnavailable('monitor_live_lane_unavailable')
                completed_lanes[camera] = completed
            else:
                raise PolicyUnavailable('monitor_live_lane_unavailable')
    if reusable_snapshot:
        _MONITOR_SUCCESSES[root_key] = {'clock_at': clock_at, 'snapshot_at': stamp.timestamp(),
                                      'lanes': completed_lanes}
        _MONITOR_SUCCESSES.move_to_end(root_key)
        while len(_MONITOR_SUCCESSES) > _MONITOR_ROOT_LIMIT:
            _MONITOR_SUCCESSES.popitem(last=False)
    return max(0.0, age)


@contextmanager
def _snapshot(root):
    base = root / 'device-day'
    connection = sqlite3.connect((base / 'observed-inventory.sqlite3').as_uri() + '?mode=ro',
                                 uri=True, timeout=.2)
    connection.row_factory = sqlite3.Row
    attached = set()
    try:
        for alias, path, required in [
            *((f'q_{stage}', base / f'queue-{stage}.sqlite3', True) for stage in STAGES),
            ('availability', base / 'InputAvailability.sqlite3', False),
            ('resources', root / 'state' / 'resources.sqlite3', False),
            ('publication', base / 'publication-journal.sqlite3', False),
            ('provider', root / 'state' / 'provider-circuits.sqlite3', False),
        ]:
            if not required and not path.is_file():
                continue
            connection.execute(f'ATTACH DATABASE ? AS {alias}', (path.as_uri() + '?mode=ro',))
            attached.add(alias)
        yield connection, attached
    finally:
        connection.close()


def _parents(stage):
    needed = set(DEPENDENCIES[stage])
    for _ in STAGES:
        needed.update(parent for name in tuple(needed) for parent in DEPENDENCIES[name])
    return [name for name in STAGES if name in needed]


def _signature_matches(queue, record, stage):
    source = (f"json_extract({queue}.payload,'$.source_signature') IS "
              f"json_extract({record}.payload,'$.source_signature')")
    if stage != 'vision':
        source += (f" AND json_extract({queue}.payload,'$.audio.source_signature') IS "
                   f"json_extract({record}.payload,'$.audio.source_signature')")
    return '(' + source + ')'


def _record_conditions(config, attached):
    role_map = config.get('collection_ingest', {}).get('camera_role_map') or {}
    # Explicit configured roles override the stored observation after metadata
    # reinspection. Unbound additional directories must stay fail-closed.
    clauses = [
        "COALESCE(json_extract(o.payload,'$.processable'),json_extract(o.payload,'$.available'),0)=1",
        "COALESCE((SELECT value FROM json_each(:roles) WHERE key=json_extract(o.payload,'$.camera_key')),"
        "json_extract(o.payload,'$.configured_role')) IN ('first_person','third_person')",
        "COALESCE(json_extract(o.payload,'$.camera_binding_status'),'')!='needs_directory_binding'",
    ]
    cutoff = (config.get('device_day') or {}).get('process_since_us')
    if cutoff is not None:
        if type(cutoff) is not int or cutoff <= 0:
            raise PolicyUnavailable('invalid_processing_cutoff')
        clauses.append("MAX(COALESCE(json_extract(o.payload,'$.recording_start_us'),0),"
                       "COALESCE(json_extract(o.payload,'$.recording_end_us'),0))>=:scope")
    if 'availability' in attached:
        clauses.append("NOT EXISTS(SELECT 1 FROM availability.inputs a WHERE a.id=o.id "
                       "AND a.signature IS json_extract(o.payload,'$.source_signature') AND a.state!='ready')")
    return clauses, {'roles': json.dumps(role_map), 'scope': cutoff}


def _eligible_conditions(stage, *, historical=False, retry_budget=':attempts'):
    matches = _signature_matches('q', 'o', stage)
    clauses = [
        "(q.recording_id IS NULL OR q.status!='running' OR COALESCE(q.lease_until,0)<:now)",
        f"(q.recording_id IS NULL OR NOT {matches} OR (q.status!='completed' AND "
        "NOT(q.status='running' AND COALESCE(q.lease_until,0)>=:now) AND "
        f"NOT(q.status='failed' AND (q.attempts>={retry_budget} OR q.updated_at>:retry_before))))",
        f"(q.recording_id IS NULL OR NOT {matches} OR q.input_status='ready')",
    ]
    if stage == 'stt':
        clauses.append("json_extract(o.payload,'$.audio.status') IN ('provided','no_input','not_provided')")
    # In-place primary stages are independent: vision/STT must not wait for a
    # full original copy. Downstream still needs every current transitive parent.
    if stage in {'understanding', 'report'}:
        for parent in _parents(stage):
            clauses.append(f"EXISTS(SELECT 1 FROM q_{parent}.recordings p WHERE p.recording_id=o.id "
                           f"AND p.status='completed' AND {_signature_matches('p', 'o', parent)})")
    return clauses


def _retry_budget(db, stage):
    # Read old immutable deployments without a schema migration; queue owners
    # add the optional, audited recovery ceiling when opening their own store.
    columns = {row['name'] for row in db.execute(f'PRAGMA q_{stage}.table_info(recordings)')}
    return 'MAX(:attempts,COALESCE(q.retry_limit,0))' if 'retry_limit' in columns else ':attempts'


def _provider_blocked(config, stage, db, attached, root, now):
    if stage not in {'stt', 'understanding'}:
        return False
    speech = config.get('speech_recognition') or {}
    if stage == 'stt' and speech.get('provider') != 'aliyun_qwen':
        return False
    binding = speech.get('connection') if stage == 'stt' else config.get('mllm')
    binding = binding or config.get('mllm') or {}
    if config.get('runtime', {}).get('provider_circuit_enabled') and 'provider' in attached:
        def scope(service):
            values = [binding.get(k) for k in ('provider', 'base_url', 'credential_ref', 'api_key_env')]
            return hashlib.sha256(json.dumps([*values, service], sort_keys=True).encode()).hexdigest()
        if db.execute('SELECT 1 FROM provider.circuits WHERE key IN (?,?) '
                      'AND MAX(retry_at,probe_until)>? LIMIT 1',
                      (scope(stage), scope('account'), now)).fetchone():
            return True
    legacy = root / 'device-day' / 'provider-blocks' / 'Aliyun.json'
    if (config.get('mllm', {}).get('provider') == 'aliyun' and legacy.is_file()):
        return json.loads(legacy.read_text(encoding='utf-8')).get('active') is True
    return False


def live_demand(config, stage, now=None):
    """Observe current live demand and relevant occupied resources, without writes.

    Only ``status == 'idle'`` permits a backfill admission attempt. Missing,
    stale or unreadable discovery never means zero demand. Resource leases from
    this backfill source are excluded so a worker can recheck while executing.
    """
    counts = {'eligible': 0, 'running': 0, 'resource_waiting': 0,
              'resource_running': 0, 'pending_publication': 0}
    current = time.time() if now is None else now
    result = {'status': 'status_unavailable', 'blocked': True, 'reasons': [],
              'counts': counts, 'observed_at': current, 'monitor_age_seconds': None}
    try:
        if stage not in STAGES:
            raise PolicyUnavailable('unknown_stage')
        root, current, cutoff, cooldown = _options(config, now)
        result['observed_at'] = current
        try:
            result['monitor_age_seconds'] = _monitor(config, root, current)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, AttributeError) as exc:
            result.update(status='waiting_for_monitor', reasons=[
                str(exc) if isinstance(exc, PolicyUnavailable) else 'monitor_snapshot_unavailable'])
            return result
        with _snapshot(root) as (db, attached):
            fair = ((config.get('device_day') or {}).get('backfill') or {}).get('mode', 'idle') == 'fair'
            if fair:
                if stage in paused_stages(config):
                    result.update(status='paused_by_user', reasons=['paused_by_user'])
                    return result
                if not stage_admitted(config, stage, datetime.fromtimestamp(current).astimezone()):
                    result.update(status='waiting_for_night_window', reasons=['waiting_for_night_window'])
                    return result
                if stage != 'stt' and _provider_blocked(config, stage, db, attached, root, current):
                    result.update(status='waiting_for_provider', reasons=['waiting_for_provider'])
                    return result
            clauses, parameters = _record_conditions(config, attached)
            parameters.update(now=current, cutoff=cutoff, retry_before=current - cooldown,
                              attempts=(config.get('device_day') or {}).get('failure_retry_limit') or 3)
            recent = ("MAX(COALESCE(json_extract(o.payload,'$.recording_start_us'),0),"
                      "COALESCE(json_extract(o.payload,'$.recording_end_us'),0))>=:cutoff")
            paused = paused_stages(config)
            for live_stage in STAGES:
                if not _SHARED[stage].intersection(_SHARED[live_stage]):
                    continue
                # Existing valid leases occupy real resources even when a new
                # pause, source revision or unavailable-input observation appears.
                running = db.execute(f"SELECT COUNT(*) FROM q_{live_stage}.recordings "
                    "WHERE status='running' AND COALESCE(lease_until,0)>=:now AND "
                    "MAX(COALESCE(json_extract(payload,'$.recording_start_us'),0),"
                    "COALESCE(json_extract(payload,'$.recording_end_us'),0))>=:cutoff", parameters).fetchone()[0]
                counts['running'] += running
                if live_stage in paused or not stage_admitted(
                        config, live_stage, datetime.fromtimestamp(current).astimezone()):
                    continue
                if _provider_blocked(config, live_stage, db, attached, root, current):
                    continue
                eligible = clauses + [recent] + _eligible_conditions(
                    live_stage, retry_budget=_retry_budget(db, live_stage))
                counts['eligible'] += db.execute(
                    f"SELECT COUNT(*) FROM observations o LEFT JOIN q_{live_stage}.recordings q "
                    "ON q.recording_id=o.id WHERE " + ' AND '.join(eligible), parameters).fetchone()[0]
            if 'resources' in attached:
                relevant = [name for name, group in _RESOURCES.items() if group in _SHARED[stage]]
                for row in db.execute("SELECT state,COUNT(*) n FROM resources.leases WHERE expires>? "
                        "AND source!=? AND resource IN (SELECT value FROM json_each(?)) GROUP BY state",
                        (current, BACKFILL_SOURCE, json.dumps(relevant))):
                    if row['state'] in {'running', 'waiting'}:
                        counts['resource_' + row['state']] += row['n']
            if 'publication' in attached and 'nas' in _SHARED[stage]:
                counts['pending_publication'] = db.execute(
                    "SELECT COUNT(*) FROM publication.dirty WHERE "
                    "MAX(COALESCE(json_extract(record,'$.recording_start_us'),0),"
                    "COALESCE(json_extract(record,'$.recording_end_us'),0))>=?", (cutoff,)).fetchone()[0]
        result['reasons'] = [name for name, count in counts.items() if count]
        fair = ((config.get('device_day') or {}).get('backfill') or {}).get('mode', 'idle') == 'fair'
        # Fair history joins the same bounded resource queues, with aging.
        # Live demand remains visible, but it cannot veto history indefinitely.
        result.update(status='fair_capacity' if fair else 'waiting_for_live' if result['reasons'] else 'idle',
                      blocked=False if fair else bool(result['reasons']))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, OverflowError) as exc:
        result['reasons'] = [str(exc) if isinstance(exc, PolicyUnavailable) else 'local_snapshot_unavailable']
    return result


def historical_candidates(config, stage, *, limit=16, now=None, exclude=()):
    """Return a bounded newest-first shortlist; execution verifies its artifacts.

    An unreadable policy observation raises instead of silently reporting an
    empty backlog. A stale monitor safely returns no candidates.
    """
    if stage not in STAGES or type(limit) is not int or not 1 <= limit <= 128:
        raise PolicyUnavailable('invalid_candidate_selection')
    if not isinstance(exclude, (tuple, list, set, frozenset)) or any(not isinstance(x, str) for x in exclude):
        raise PolicyUnavailable('invalid_candidate_exclusions')
    root, current, cutoff, cooldown = _options(config, now)
    try:
        _monitor(config, root, current)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, AttributeError):
        return []
    if not stage_admitted(config, stage, datetime.fromtimestamp(current).astimezone()):
        return []
    try:
        with _snapshot(root) as (db, attached):
            provider_blocked = _provider_blocked(config, stage, db, attached, root, current)
            if provider_blocked and stage != 'stt':
                return []
            clauses, parameters = _record_conditions(config, attached)
            parameters.update(now=current, cutoff=cutoff, retry_before=current - cooldown, limit=limit,
                              attempts=(config.get('device_day') or {}).get('failure_retry_limit') or 3,
                              excluded=json.dumps(list(exclude)))
            clauses += ["MAX(COALESCE(json_extract(o.payload,'$.recording_start_us'),0),"
                        "COALESCE(json_extract(o.payload,'$.recording_end_us'),0))<:cutoff"]
            clauses += _eligible_conditions(stage, historical=True, retry_budget=_retry_budget(db, stage))
            if provider_blocked:
                # A video with no separately published audio still needs its
                # actual track probe. Transport gates protect any embedded ASR.
                clauses.append("json_extract(o.payload,'$.audio.status') IN ('no_input','not_provided')")
            clauses.append('o.id NOT IN (SELECT value FROM json_each(:excluded))')
            fair = ((config.get('device_day') or {}).get('backfill') or {}).get('mode', 'idle') == 'fair'
            order = 'ASC' if fair else 'DESC'
            rows = db.execute(f"SELECT o.payload FROM observations o LEFT JOIN q_{stage}.recordings q "
                              "ON q.recording_id=o.id WHERE " + ' AND '.join(clauses) +
                              f" ORDER BY json_extract(o.payload,'$.recording_start_us') {order},o.id LIMIT :limit",
                              parameters)
            from .input_availability import configured_record
            return [configured_record(config, json.loads(row['payload'])) for row in rows]
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        raise PolicyUnavailable('historical_catalog_unavailable') from exc


def historical_candidate_ids(config, stage, *, limit=16, now=None, exclude=()):
    return [record['recording_id'] for record in historical_candidates(
        config, stage, limit=limit, now=now, exclude=exclude)]

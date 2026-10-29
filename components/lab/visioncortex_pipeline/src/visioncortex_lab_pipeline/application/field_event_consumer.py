"""Bounded FieldRecognition feed consumption and recoverable local personnel views."""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from ..device_day_contract import atomic_bytes, atomic_json, digest, file_hash
from ..evidence.field_events import FieldEventInbox
from ..evidence.person_day import person_days
from ..local_storage import require_local_path


class PersonDayCapacityExceeded(OSError):
    """Keep inbox progress and old reports; require verified archival before retry."""
    reason = 'person_day_capacity_requires_archival'


def encoded_size(value):
    return sum(len(piece.encode('utf-8')) for piece in json.JSONEncoder(
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).iterencode(value)) + 1


def publish_person_json(path, value):
    # Same facts and content identity, without repeated presentation whitespace.
    # Capacity checks use exactly this wire encoding before allocating it.
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8'))


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('device_day.field_recognition must be an object')
    if type(settings.get('enabled', False)) is not bool:
        raise ValueError('Field event enabled must be boolean')
    archive = settings.get('archive_context_enabled', False)
    if type(archive) is not bool:
        raise ValueError('Field archive context enabled must be boolean')
    if archive and (settings.get('enabled') or not isinstance(settings.get('archive_root'), str)
                    or not Path(settings['archive_root']).is_absolute()):
        raise ValueError('Field archive context requires an explicit root and independent feed disabled')
    if not settings.get('enabled', False):
        return
    endpoint = settings.get('endpoint')
    if not isinstance(endpoint, str):
        raise ValueError('Field event endpoint is required')
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {'', '/'}):
        raise ValueError('Field event endpoint must be a service origin without credentials')
    cameras = settings.get('camera_ids')
    if (not isinstance(cameras, list) or not 1 <= len(cameras) <= 64
            or any(not isinstance(camera, str) or not camera.strip() or len(camera) > 200 for camera in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError('Field event camera_ids must list distinct explicitly configured cameras')
    for key, default, lower, upper in [('pages_per_tick', 2, 1, 20), ('page_size', 100, 1, 100)]:
        value = settings.get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError('Field event page bound is invalid')
    timeout = settings.get('timeout_seconds', 5)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise ValueError('Field event timeout must be bounded to 30 seconds')
    tenant = settings.get('tenant')
    if tenant is not None and (not isinstance(tenant, str) or not tenant.strip() or len(tenant) > 200):
        raise ValueError('Field event tenant is invalid')
    capacity = settings.get('person_day_max_bytes', 64 * 1024 * 1024)
    if type(capacity) is not int or not 4096 <= capacity <= 1024 * 1024 * 1024:
        raise ValueError('PersonDay capacity must be an explicit integer between 4 KiB and 1 GiB')


def fetch_json(url, timeout):
    request = Request(url, headers={'Accept': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as reply:
        # A bounded page must not become an unbounded response or another service.
        if urlsplit(reply.url).netloc != urlsplit(url).netloc:
            raise ValueError('Field feed redirected to a different service')
        raw = reply.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('Field event response exceeded the bounded page size')
        return json.loads(raw)


class FieldEventConsumer:
    def __init__(self, runtime_root, settings, *, fetch_json=fetch_json, timezone='Asia/Shanghai'):
        validate_settings(settings)
        self.root, self.settings = require_local_path(Path(runtime_root)), settings
        self.fetch, self.timezone = fetch_json, timezone
        self.inbox = FieldEventInbox(self.root)

    def publish(self):
        from ..device_day import exclusive
        try:
            with exclusive(require_local_path(self.root / 'locks' / 'person-day-publication.lock')):
                # Obtain the source snapshot only after global ownership. Every
                # publisher (feed replay and device index) uses the same lock.
                return self._publish_locked()
        except BlockingIOError:
            return {'published': False, 'status': 'running_elsewhere'}

    def _publish_locked(self):
        events, state = self.inbox.snapshot()
        from ..evidence import person_day
        render_identity = digest({'renderer': file_hash(Path(__file__)), 'person_day': file_hash(Path(person_day.__file__)),
                                  'timezone': self.timezone})
        root = require_local_path(self.root / 'PersonDay')
        manifest_path = root / 'Manifest.json'
        if state['generation'] == state['published_generation'] and manifest_path.is_file():
            prior = json.loads(manifest_path.read_text())
            if prior.get('render_policy_identity') == render_identity and digest(prior) == state['manifest_digest']:
                return {'published': False, 'generation': state['generation']}
        collection = person_days(events, timezone=self.timezone)
        tenants = sorted({event.get('tenant') for event in events}, key=lambda value: value or '')
        outputs, plans = [], []
        for tenant in tenants:
            directory = root / digest({'tenant': tenant})[:24]
            content = {'schema_version': 'person-day-publication/1', 'generation': state['generation'],
                       'render_policy_identity': render_identity,
                       'tenant': tenant, 'reports': [report for report in collection['items'] if report['tenant'] == tenant],
                       'unassociated': [event for event in collection['unassociated'] if event.get('tenant') == tenant],
                       'withdrawn': [event for event in collection['withdrawn']
                                     if any(e['event_id'] == event['event_id'] and e.get('tenant') == tenant for e in events)],
                       'model_invoked': False, 'business_accepted': False, 'accuracy_status': 'NOT_PROVEN'}
            content['content_revision'] = digest(content)
            version = directory / f"Revision{state['generation']}-{render_identity[:12]}.json"
            if version.exists() and json.loads(version.read_text()) != content:
                raise ValueError('Personnel publication revision already has different content')
            latest = directory / 'Latest.json'
            plans.append((directory, version, latest, content))
            outputs.append({'tenant': tenant, 'path': latest.relative_to(root).as_posix(),
                            'revision_path': version.relative_to(root).as_posix(),
                            'content_revision': content['content_revision']})
        manifest = {'schema_version': 'person-day-manifest/1', 'generation': state['generation'],
                    'render_policy_identity': render_identity,
                    'outputs': outputs, 'publication_status': 'published', 'model_invoked': False,
                    'evidence_status': 'PARTIAL_EVIDENCE'}
        existing = sum(require_local_path(path).stat().st_size for path in root.rglob('*') if path.is_file()) if root.exists() else 0
        growth, largest = 0, encoded_size(manifest)
        for _, version, latest, content in plans:
            size = encoded_size(content)
            growth += (0 if version.exists() else size) + size - (latest.stat().st_size if latest.exists() else 0)
            largest = max(largest, size)
        growth += encoded_size(manifest) - (manifest_path.stat().st_size if manifest_path.exists() else 0)
        if existing + growth + largest > self.settings.get('person_day_max_bytes', 64 * 1024 * 1024):
            raise PersonDayCapacityExceeded('PersonDay local capacity reached; archive verified revisions before replay')
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory, version, latest, content in plans:
            directory.mkdir(mode=0o700, exist_ok=True)
            if not version.exists():
                publish_person_json(version, content)
                version.chmod(0o600)
            if json.loads(version.read_text()) != content:
                raise ValueError('Personnel publication verification failed')
            publish_person_json(latest, content)
            latest.chmod(0o600)
        publish_person_json(root / 'Manifest.json', manifest)
        (root / 'Manifest.json').chmod(0o600)
        if json.loads((root / 'Manifest.json').read_text()) != manifest:
            raise ValueError('Personnel publication manifest verification failed')
        self.inbox.published(state['generation'], digest(manifest))
        return {'published': True, 'generation': state['generation'], 'person_days': len(collection['items']),
                'unassociated': len(collection['unassociated']), 'manifest': str(root / 'Manifest.json')}

    def tick(self):
        inserted, pages, results, errors = 0, 0, [], []
        endpoint = self.settings['endpoint'].rstrip('/')
        # One bounded share per device; backlog cannot starve another configured device.
        for camera in self.settings['camera_ids']:
            pending = False
            try:
                for _ in range(self.settings.get('pages_per_tick', 2)):
                    cursor = self.inbox.cursor(endpoint, camera)
                    params = {'camera_id': camera, 'after_seq': cursor['cursor'],
                              'limit': self.settings.get('page_size', 100)}
                    if cursor['source_instance']:
                        params['source_instance'] = cursor['source_instance']
                    page = self.fetch(endpoint + '/api/events/feed?' + urlencode(params), self.settings.get('timeout_seconds', 5))
                    result = self.inbox.ingest(endpoint, camera, page, tenant=self.settings.get('tenant'))
                    inserted += result['inserted']
                    pages += 1
                    pending = page['has_more']
                    if not pending:
                        break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                status = getattr(exc, 'code', None)
                reason = ('field_auth_required' if status in {401, 403} else
                          'event_feed_not_deployed' if status == 404 else
                          'source_identity_or_cursor_conflict' if status == 409 else
                          'field_feed_unavailable' if isinstance(exc, OSError) else 'field_feed_contract_invalid')
                errors.append({'camera_id': camera, 'error_type': type(exc).__name__, 'reason': reason})
                pending = True
            finally:
                cursor = self.inbox.cursor(endpoint, camera)
                results.append({'camera_id': camera, 'cursor': cursor['cursor'], 'pending': pending})
        publication = self.publish()
        return {'status': 'partial' if errors else 'consumed', 'inserted': inserted, 'pages': pages, 'cameras': results,
                'errors': errors,
                'publication': publication, 'model_invoked': False, 'consumer_wired': True,
                'business_accepted': False}


def context_events(config, recording):
    """Read only actual overlapping events; no media or nearest-time association."""
    settings = (config.get('device_day') or {}).get('field_recognition') or {}
    if settings.get('archive_context_enabled', False):
        validate_settings(settings)
        from .field_archive_context import context_events as archived_context
        return archived_context(config, recording)
    if not settings.get('enabled', False):
        return {'status': 'no_input', 'events': [], 'source': 'field_recognition'}
    validate_settings(settings)
    camera = recording.get('camera_key')
    if camera not in settings['camera_ids']:
        return {'status': 'camera_not_configured', 'events': [], 'source': 'field_recognition'}
    path = require_local_path(Path(config['storage']['local_runtime_root']) / 'device-day' / 'field-events.sqlite3')
    if not path.is_file():
        return {'status': 'not_received', 'events': [], 'source': 'field_recognition'}
    start, end = recording.get('recording_start_us'), recording.get('recording_end_us')
    if type(start) is not int or type(end) is not int or end <= start:
        raise ValueError('Field context requires a verified recording interval')
    from ..evidence.field_events import instant
    from ..sqlite_store import connection
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    events, unknown_time = [], 0
    with connection(path, readonly=True) as db:
        for row in db.execute("SELECT document FROM field_current WHERE json_extract(document,'$.capture_device_id')=?", (camera,)):
            event = json.loads(row['document'])
            if event.get('withdrawn') or event.get('tenant') != settings.get('tenant'):
                continue
            occurred = instant(event.get('occurred_start'))
            if occurred is None:
                unknown_time += 1
                continue
            finished = instant(event.get('occurred_end'))
            event_start = (occurred - epoch) // timedelta(microseconds=1)
            event_end = ((finished - epoch) // timedelta(microseconds=1)) if finished else event_start
            # An unclosed binding is a valid-time relation, not a point event.
            if event.get('type') == 'instrument_binding' and finished is None:
                event_end = end
            if not (event_start < end and (event_end > start or event_end == event_start and start <= event_start)):
                continue
            events.append({'event_id': event['event_id'], 'revision': event['revision'], 'type': event['type'],
                           'start_us': max(start, event_start), 'end_us': min(end, max(event_end, event_start)),
                           'evidence_refs': event['evidence_refs'], 'capture_id': event.get('capture_id'),
                           'instrument_id': event.get('instrument_id'), 'wearer_id': event.get('wearer_id'),
                           'operator_id': event.get('operator_id'), 'associations': event.get('associations'),
                           'clock': event.get('clock'), 'physical_action_verified': False})
            # Readings and original media are evidence, not just opaque event IDs.
            # Keep recognition/review status and units exactly as received.
            events[-1].update(payload=event.get('payload', {}), media=event.get('media', []),
                              capture_device_id=camera, source_instance=event.get('source_instance'),
                              occurred_start=event.get('occurred_start'), occurred_end=event.get('occurred_end'))
    events.sort(key=lambda event: (event['start_us'], event['event_id'], event['revision']))
    return {'status': 'available', 'events': events, 'unknown_time_events': unknown_time,
            'source': 'field_recognition', 'association_basis': 'actual_interval_overlap',
            'physical_action_verified': False}


def consume_device_index(config, index, archive_id):
    """Port for refresh_index after its existing receipt/key/artifact gates.

    This port does not accept paths or read receipts itself. The source index is
    a verified publication snapshot, and remains the authority for its facts.
    """
    settings = (config.get('device_day') or {}).get('field_recognition') or {}
    if not settings.get('enabled', False):
        return {'status': 'no_input', 'model_invoked': False}
    validate_settings(settings)
    from ..evidence.device_events import device_index_events
    events, revision = device_index_events(index, archive_id, tenant=settings.get('tenant'))
    runtime = Path(config['storage']['local_runtime_root']) / 'device-day'
    consumer = FieldEventConsumer(runtime, settings, timezone=(config.get('device_day') or {}).get('timezone', 'Asia/Shanghai'))
    result = consumer.inbox.replace_device_index(archive_id, revision, events)
    return {'status': 'projected', 'events': result, 'publication': consumer.publish(),
            'model_invoked': False, 'validation_basis': 'caller_verified_device_index',
            'business_accepted': False}


def archive_person_history(config, runtime_root, destination):
    """Explicit verified history offload; preserve latest, inbox and path mapping.

    The caller supplies its authorized archive destination and storage checks.
    Local/no-NAS callers cannot implicitly inherit a network archive. The same
    publication lock excludes an active writer; only superseded JSON revisions
    are eligible. No live database, media or current pointer is relocated.
    """
    from ..device_day import exclusive
    from ..device_day_write_guard import storage_guard
    runtime = require_local_path(Path(runtime_root))
    root = runtime / 'PersonDay'
    destination = Path(destination) / digest({'runtime_root': str(runtime)})[:24]
    with exclusive(runtime / 'locks' / 'person-day-publication.lock'), storage_guard(config):
        manifest = json.loads((root / 'Manifest.json').read_text())
        current = {output['revision_path'] for output in manifest['outputs']}
        receipt_path = root / 'HistoryArchives.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {
            'schema_version': 'person-day-history-archives/1', 'entries': {}}
        moved, reclaimed = 0, 0
        for path in sorted(root.glob('*/Revision*.json')):
            relative = path.relative_to(root).as_posix()
            if relative in current:
                continue
            require_local_path(path)
            initial = path.stat()
            data = json.loads(path.read_text())
            if (data.get('schema_version') != 'person-day-publication/1'
                    or data.get('content_revision') != digest({k: v for k, v in data.items() if k != 'content_revision'})):
                raise ValueError('Person history revision content identity is invalid')
            checksum = file_hash(path)
            target = destination / relative
            if target.exists():
                if file_hash(target) != checksum:
                    raise ValueError('Person history archive collision; original retained')
            else:
                atomic_bytes(target, path.read_bytes())
            if file_hash(target) != checksum or file_hash(path) != checksum or any(getattr(path.stat(), k) != getattr(initial, k)
                    for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
                raise ValueError('Person history archive verification failed; original retained')
            receipt['entries'][relative] = {'archive_path': str(target), 'sha256': checksum,
                'size_bytes': initial.st_size, 'content_revision': data['content_revision'],
                'generation': data['generation'], 'original_relative_path': relative}
            # Durable verified mapping precedes reclaim, including crash recovery.
            atomic_json(receipt_path, receipt)
            if json.loads(receipt_path.read_text()) != receipt:
                raise ValueError('Person history archive mapping verification failed')
            from ..device_day_write_guard import CURRENT
            CURRENT.get().validate()
            path.unlink()
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            moved += 1
            reclaimed += initial.st_size
        return {'archived_revisions': moved, 'reclaimed_bytes': reclaimed,
                'history_mapping': str(receipt_path), 'latest_preserved': True, 'inbox_unchanged': True}


def read_person_revision(config, runtime_root, relative):
    """Read an original revision identity from SSD or its verified archive map."""
    from ..device_day_contract import safe_child
    root = require_local_path(Path(runtime_root)) / 'PersonDay'
    path = safe_child(root, relative)
    if path.is_file():
        return json.loads(path.read_text())
    reference = json.loads((root / 'HistoryArchives.json').read_text())['entries'][relative]
    from ..device_day_write_guard import storage_guard, CURRENT
    with storage_guard(config):
        target = Path(reference['archive_path'])
        CURRENT.get().mount_for(target)
        if target.stat().st_size != reference['size_bytes'] or file_hash(target) != reference['sha256']:
            raise ValueError('Archived person revision integrity check failed')
        return json.loads(target.read_text())

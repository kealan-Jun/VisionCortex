"""Progressive camera-day synthesis owned by the existing report consumer."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import sqlite3
import time

from .device_day_contract import DeviceDayLayout, archive_name, atomic_json, digest, read_json


def current_day_index(config, layout, *, source_index=None):
    """Use the same verified partial windows as the published day timeline."""
    from .device_day_file_index import read_source_index
    from .device_day_content import with_partial_understandings
    from .device_day_io import slot
    with slot(config):
        index = read_source_index(config, layout.index) if source_index is None else source_index
        if index.get('archive') != layout.name:
            raise ValueError('Day index crossed a camera/date boundary')
        index = with_partial_understandings(config, index)
        if (config.get('device_day', {}).get('field_recognition') or {}).get('archive_context_enabled'):
            from .device_day_narrative import attach_context
            index = attach_context(config, index)
    return index


def _index_version(path):
    if path.is_symlink():
        raise ValueError('Day index must not be a symlink')
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class DaySynthesisWorker:
    def __init__(self):
        self.checked = {}
        self.state = {'days': {}, 'schedule': {}}
        self.lane = 'historical_fair_share'
        self.hint_error = None
        self.hints = {}

    def candidates(self, config):
        from .device_day_schedule import in_processing_scope
        from .device_day_publication_worker import local_change_hints
        from .sqlite_store import connection
        root = Path(config['storage']['local_runtime_root']) / 'device-day'
        path = root / 'observed-inventory.sqlite3'
        if not path.is_file():
            return []
        days, records = {}, []
        with connection(path, readonly=True) as db:
            for row in db.execute('SELECT payload FROM observations'):
                record = json.loads(row[0])
                if not in_processing_scope(config['device_day'], record):
                    continue
                if not record.get('camera_key') or not record.get('recording_start_us'):
                    continue
                records.append(record)
                name = archive_name(record['camera_key'], record['recording_start_us'])
                days[name] = record
        # Group locally before constructing layouts: resolving a NAS layout
        # for every recording delayed admission across thousands of captures.
        settings = config.get('device_day') or {}
        now = time.time()
        minimum = (settings.get('day_synthesis') or {}).get('minimum_update_seconds', 60)
        pending = {n for n in days if self.state['days'].get(n, {}).get('status') == 'publication_pending'}
        eligible = {n for n in days if n in pending or
                    now - self.state['days'].get(n, {}).get('attempted_at', 0) >= minimum}
        schedule = self.state.setdefault('schedule', {})
        self.hint_error = None
        try:
            hints = local_change_hints(root, records, days)
        except (OSError, ValueError, sqlite3.Error) as exc:
            hints = {}
            self.hint_error = type(exc).__name__
        changed = {n for n in eligible if n in hints and hints[n] != schedule.get('observed_hints', {}).get(n)}
        cutoff = (now - settings.get('live_priority_seconds', 14400)) * 1e6
        recent = {archive_name(r['camera_key'], r['recording_start_us']) for r in records
                  if max(r.get('recording_start_us') or 0, r.get('recording_end_us') or 0) >= cutoff}
        today = datetime.fromtimestamp(now, ZoneInfo('Asia/Shanghai')).date().isoformat()
        recent.update(n for n in days if n[:10] >= today)
        priority = (changed & recent) or (eligible & recent) or changed
        turn = schedule.get('turn', 0) + 1
        schedule.update(turn=turn)
        self.hints = hints
        self.lane = 'changed_or_live' if priority and turn % 3 else 'historical_fair_share'
        def order(name):
            return self.checked.get(name, 0), -int(name[:10].replace('-', '')), name
        names = sorted(pending, key=order)
        names += sorted(priority - set(names), key=order) if self.lane == 'changed_or_live' else []
        names += sorted(eligible - set(names), key=order)
        return [DeviceDayLayout(Path(config['storage']['archive_root']), days[n]['camera_key'],
                    days[n]['recording_start_us'], Path(config['storage']['local_cache_root'])) for n in names[:16]]

    def tick(self, config, stop, *, analyzer_factory=None):
        settings = config.get('device_day', {}).get('day_synthesis') or {}
        if not settings.get('enabled') or stop.is_set():
            return {'status': 'disabled' if not settings.get('enabled') else 'stopped'}
        from .device_day import exclusive
        from .device_day_synthesis import current_synthesis, evidence_bundle, synthesize
        from .device_day_reports import preserve_prior_synthesis, render_day
        from .device_day_write_guard import storage_guard
        from .mllm import ArkAnalyzer
        from .runtime_control import execution_context
        root = Path(config['storage']['local_runtime_root']) / 'device-day'
        state_path = root / 'DaySynthesis' / 'State.json'
        state = read_json(state_path) if state_path.is_file() else {'days': {}}
        self.state = state
        self.checked = state.setdefault('schedule', {}).setdefault('checked', {})
        minimum = settings.get('minimum_update_seconds', 60)
        if type(minimum) not in (int, float) or not 60 <= minimum <= 86400:
            raise ValueError('day_synthesis.minimum_update_seconds must be between 60 and 86400')
        with storage_guard(config):
            for layout in self.candidates(config)[:16]:
                self.checked[layout.name] = time.time()
                if stop.is_set():
                    return {'status': 'stopped'}
                try:
                    source_version = _index_version(layout.index)
                    index = current_day_index(config, layout)
                    bundle = evidence_bundle(index)
                    if layout.name in self.hints:
                        state['schedule'].setdefault('observed_hints', {})[layout.name] = self.hints[layout.name]
                    if not any(row['kind'] in {'video', 'speech', 'field'} for row in bundle['evidence']):
                        continue
                    key = digest(bundle)
                    previous = current_synthesis(layout, index)
                    saved = state['days'].get(layout.name, {})
                    ready = previous if previous.get('input_digest') == key else None
                    if ready and saved.get('input_digest') == key and saved.get('status') == 'completed_for_available_inputs':
                        continue
                    if not ready and time.time() - saved.get('attempted_at', 0) < minimum:
                        continue
                    # This local lock prevents overlapping paid synthesis for
                    # the same camera-day across a graceful service handover.
                    with exclusive(root / 'locks' / ('day-synthesis-' + layout.name + '.lock')):
                        result = ready
                        if result is None:
                            state['days'][layout.name] = {'input_digest': key, 'attempted_at': time.time(),
                                                         'status': 'running'}
                            atomic_json(state_path, state)
                            analyzer = (analyzer_factory or ArkAnalyzer)(config)
                            try:
                                # No stop signal can interrupt an already issued
                                # request before its durable provider receipt.
                                with execution_context(job_id='day_' + layout.name, source='device_day_backfill',
                                                       priority=20, background_fair=True, stop=None):
                                    result = synthesize(config, layout, index, analyzer,
                                        retry_output_tokens=settings.get('retry_output_tokens'))
                            finally:
                                analyzer.close()
                        if result.get('context_archives') != [layout.name]:
                            raise ValueError('Day output crossed a camera/date boundary')
                        # Save the complete native result before a busy publisher
                        # can defer it. The normal verified history reader also
                        # recovers this snapshot after a graceful handover.
                        snapshot = {'全天综合理解': result}
                        atomic_json(layout.reports / '.Evidence/History' / ('Report' + digest(snapshot) + '.json'), snapshot)
                        state['days'][layout.name] = saved | state['days'].get(layout.name, {}) | {
                            'input_digest': key, 'status': 'publication_pending'}
                        atomic_json(state_path, state)
                        while True:
                            with ExitStack() as acquired:
                                try:
                                    acquired.enter_context(exclusive(root / 'locks' / (layout.name + '.index.lock')))
                                except BlockingIOError:
                                    if stop.wait(1):
                                        return {'status': 'publication_pending', 'archive': layout.name,
                                                'input_digest': key}
                                    continue
                                path = layout.reports / 'LaboratoryDailyReport.json'
                                old = read_json(path) if path.is_file() else {}
                                preserve_prior_synthesis(layout, result)
                                atomic_json(path, old | {'全天综合理解': result})
                                # Recheck windows/context after the request, but
                                # avoid rescanning an unchanged multi-GB index.
                                source = index if _index_version(layout.index) == source_version else None
                                latest = current_day_index(config, layout, source_index=source)
                                render_day(layout, latest)
                                break
                        state['days'][layout.name] |= {'status': 'completed_for_available_inputs',
                            'generated_at': result.get('generated_at'), 'finished_at': time.time()}
                        atomic_json(state_path, state)
                        return {'status': 'processed', 'archive': layout.name, 'input_digest': key,
                                'scheduling_lane': self.lane,
                                'result': {'status': 'completed_for_available_inputs'}}
                except BlockingIOError:
                    continue
                except Exception as exc:
                    from .multimodal_usage import SubmittedUnknown
                    from .runtime_control import ExecutionYielded
                    from .provider_control import ProviderUnavailable
                    status = ('waiting_for_provider_receipt' if isinstance(exc, SubmittedUnknown) else
                              'resource_wait' if isinstance(exc, (ExecutionYielded, ProviderUnavailable)) else 'failed')
                    state['days'][layout.name] = {'attempted_at': time.time(), 'status': status,
                                                 'error_type': type(exc).__name__}
                    atomic_json(state_path, state)
                    return {'status': status, 'archive': layout.name, 'error_type': type(exc).__name__}
            atomic_json(state_path, state)
        return {'status': 'waiting', 'scheduling_lane': self.lane, 'hint_error': self.hint_error}


class DaySynthesisSupervisor:
    """One background unit; report admission and live heartbeats keep polling."""
    def __init__(self, stop):
        self.stop, self.worker = stop, DaySynthesisWorker()
        self.pool, self.future = ThreadPoolExecutor(max_workers=1, thread_name_prefix='day-synthesis'), None

    def poll(self, config):
        result = None
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
            except Exception as exc:
                result = {'status': 'failed', 'error_type': type(exc).__name__}
            finally:
                self.future = None
        if self.future is None and not self.stop.is_set():
            self.future = self.pool.submit(self.worker.tick, config, self.stop)
        return result or {'status': 'running'}

    def close(self):
        # Finish the bounded issued request and save its receipt before exit.
        self.pool.shutdown(wait=True)

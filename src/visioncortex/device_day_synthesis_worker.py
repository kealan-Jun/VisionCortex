"""Progressive camera-day synthesis owned by the existing report consumer."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time

from .device_day_contract import DeviceDayLayout, atomic_json, digest, read_json


class DaySynthesisWorker:
    def __init__(self):
        self.checked = {}

    def candidates(self, config):
        from .device_day_schedule import in_processing_scope
        from .sqlite_store import connection
        root = Path(config['storage']['local_runtime_root']) / 'device-day'
        path = root / 'observed-inventory.sqlite3'
        if not path.is_file():
            return []
        days = {}
        with connection(path, readonly=True) as db:
            for row in db.execute('SELECT payload FROM observations'):
                record = json.loads(row[0])
                if not in_processing_scope(config['device_day'], record):
                    continue
                if not record.get('camera_key') or not record.get('recording_start_us'):
                    continue
                layout = DeviceDayLayout(Path(config['storage']['archive_root']), record['camera_key'],
                    record['recording_start_us'], Path(config['storage']['local_cache_root']))
                days[layout.name] = layout
        # Every camera-day eventually gets a turn, including late audio/OCR.
        return sorted(days.values(), key=lambda layout: (self.checked.get(layout.name, 0),
                      -int(layout.name[:10].replace('-', '')), layout.name))

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
        minimum = settings.get('minimum_update_seconds', 60)
        if type(minimum) not in (int, float) or not 60 <= minimum <= 86400:
            raise ValueError('day_synthesis.minimum_update_seconds must be between 60 and 86400')
        with storage_guard(config):
            for layout in self.candidates(config)[:16]:
                self.checked[layout.name] = time.time()
                if stop.is_set():
                    return {'status': 'stopped'}
                try:
                    index = read_json(layout.index)
                    if index.get('archive') != layout.name:
                        raise ValueError('Day index crossed a camera/date boundary')
                    bundle = evidence_bundle(index)
                    if not any(row['kind'] in {'video', 'speech', 'field'} for row in bundle['evidence']):
                        continue
                    key = digest(bundle)
                    previous = current_synthesis(layout, index)
                    if previous.get('input_digest') == key:
                        continue
                    saved = state['days'].get(layout.name, {})
                    if time.time() - saved.get('attempted_at', 0) < minimum:
                        continue
                    # This local lock prevents overlapping paid synthesis for
                    # the same camera-day across a graceful service handover.
                    with exclusive(root / 'locks' / ('day-synthesis-' + layout.name + '.lock')):
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
                        with exclusive(root / 'locks' / (layout.name + '.index.lock')):
                            path = layout.reports / 'LaboratoryDailyReport.json'
                            old = read_json(path) if path.is_file() else {}
                            preserve_prior_synthesis(layout, result)
                            atomic_json(path, old | {'全天综合理解': result})
                            # A new clip may have arrived during the request.
                            # Rendering retains the verified as-of conclusion.
                            render_day(layout, read_json(layout.index))
                        state['days'][layout.name] |= {'status': 'completed_for_available_inputs',
                            'generated_at': result.get('generated_at'), 'finished_at': time.time()}
                        atomic_json(state_path, state)
                        return {'status': 'processed', 'archive': layout.name, 'input_digest': key,
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
        return {'status': 'waiting'}


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

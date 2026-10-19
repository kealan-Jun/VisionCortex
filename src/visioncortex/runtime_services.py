"""Independent operational workers; model-free progress and input integrity."""

from contextlib import contextmanager
import logging
import multiprocessing
import os
from pathlib import Path
import threading
import time

from .device_day_contract import atomic_json


def _publish_progress(config, control, interval):
    """Only local read models; never share the inference process's Python locks."""
    from .device_day_progress import snapshot
    path = Path(config['storage']['local_runtime_root']) / 'device-day' / 'ProgressSnapshot.json'
    try:
        while not control.poll():
            try:
                started = time.monotonic()
                value = snapshot(config)
                value.update(snapshot_producer_pid=os.getpid(),
                             snapshot_collection_seconds=time.monotonic()-started)
                atomic_json(path, value)
            except Exception as exc:
                logging.getLogger(__name__).warning('Progress publication unavailable: %s', type(exc).__name__)
            if control.poll(interval):
                break
    finally:
        control.close()


class ProgressPublisher:
    """Supervised spawn process: no fork of a loaded CUDA runtime."""
    def __init__(self, config, *, interval=5):
        self.config = config
        self.interval = interval
        self.context = multiprocessing.get_context('spawn')
        # A killed child can strand a multiprocessing.Event's semaphore. Give
        # each generation a fresh pipe, with no shared synchronization locks.
        self._lock = threading.Lock()
        self._closed = False
        self._control = None
        self.process = None

    def ensure(self):
        with self._lock:
            if self._closed or self.process is not None and self.process.is_alive():
                return
            if self.process is not None:
                self.process.join(timeout=0)
                self.process.close()
                self.process = None
            if self._control is not None:
                self._control.close()
            reader, self._control = self.context.Pipe(duplex=False)
            process = self.context.Process(target=_publish_progress,
                args=(self.config, reader, self.interval), name='progress-publisher', daemon=True)
            try:
                process.start()
                self.process = process
            except BaseException:
                self._control.close()
                self._control = None
                process.close()
                raise
            finally:
                reader.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._control is not None:
                try:
                    self._control.send_bytes(b'\x00')
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    self._control.close()
                    self._control = None
            if self.process is None:
                return
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if not self.process.is_alive():
                self.process.close()
                self.process = None


@contextmanager
def services(config):
    stop = threading.Event()
    root = Path(config["storage"]["local_runtime_root"])

    def loop(name, interval, action):
        while not stop.is_set():
            try:
                action()
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "%s unavailable: %s", name, type(exc).__name__
                )
            stop.wait(interval)

    from .task_events import EventStore
    from .knowledge import Knowledge
    from .input_availability import Reconciler

    events = EventStore(root)
    knowledge = Knowledge(config) if config["storage"].get("archive_root") else None
    availability = Reconciler(config)

    progress = ProgressPublisher(config)

    tasks = [
        ("progress-supervisor", 5, progress.ensure),
        ("task-outbox", 2, events.harvest),
        (
            "knowledge-index",
            5,
            lambda: knowledge.tick(stop=stop) if knowledge else None,
        ),
        ("input-availability", 5, availability.tick),
    ]
    if config.get('device_day', {}).get('enabled'):
        from .device_day_time_lookup import refresh_photo_index
        from .device_day_file_index import FileIndexPublisher
        file_indexes = FileIndexPublisher(config)
        tasks.append(('file-time-index', 5, file_indexes.tick))
        for camera in config.get('collection_ingest', {}).get('camera_role_map', {}):
            tasks.append(('capture-photos-'+camera, 5,
                          lambda camera=camera: refresh_photo_index(config, camera)))
    subscriptions = config.get("runtime", {}).get("result_callbacks", [])
    if subscriptions:
        tasks.append(("result-callbacks", 5, lambda: events.dispatch(subscriptions)))
    threads = [
        threading.Thread(target=loop, args=task, name=task[0], daemon=True)
        for task in tasks
    ]
    for thread in threads:
        thread.start()
    try:
        yield
    finally:
        stop.set()
        deadline = time.monotonic() + 5
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        progress.close()


def serve_integrity(config_path, stop):
    """Independent guarded checker; it never instantiates a model backend."""
    from .build_identity import identity
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner, exclusive
    from .device_day_consumers import process_start_ticks
    from .input_availability import IntegrityMonitor
    from .local_storage import require_local_path
    from .runtime_control import ExecutionCancelled, ExecutionYielded, execution_context, resource_slot
    config = apply_active(load_config(Path(config_path)))
    root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'
    runner = DeviceDayRunner(config, backend=object())
    monitor = IntegrityMonitor(config, runner)
    state = {'schema_version': 'visioncortex-input-integrity-service/1', 'pid': os.getpid(),
             'process_start_ticks': process_start_ticks(os.getpid()), 'build': identity(),
             'status': 'starting', 'started_at': time.time(), 'tick_count': 0, 'completed_ticks': 0,
             'model_invoked': False, 'capture_modified': False, 'settings': monitor.options}
    lock, heartbeat_stop = threading.Lock(), threading.Event()
    path = root / 'InputIntegrity/Service.json'

    def publish(**fields):
        with lock:
            state.update(fields)
            atomic_json(path, state | {'updated_at': time.time()})

    def heartbeat():
        while not heartbeat_stop.wait(2):
            try:
                publish()
            except OSError:
                pass

    with exclusive(root / 'locks/input-integrity-service.lock'):
        thread = threading.Thread(target=heartbeat, name='integrity-heartbeat', daemon=True)
        thread.start()
        last_error = None
        try:
            while not stop.is_set():
                try:
                    publish(status=('disabled' if not monitor.options['enabled'] else 'failed' if last_error else 'running'),
                            phase='checking', tick_started_at=time.time(),
                            tick_count=state['tick_count'] + 1)
                    with execution_context(job_id='input-integrity', source='retention_recovery',
                                           priority=1, stop=stop, yield_signal=stop):
                        with resource_slot(config, 'cpu'):
                            result = monitor.tick(stop=stop)
                    if last_error:
                        logging.getLogger(__name__).info('Input integrity service recovered')
                    last_error = None
                    publish(status='running' if monitor.options['enabled'] else 'disabled',
                            phase='waiting', last_completed_at=time.time(), last_result=result,
                            completed_ticks=state['completed_ticks'] + 1)
                except ExecutionYielded:
                    publish(phase='waiting', last_result={'status': 'yielded_to_foreground'})
                except ExecutionCancelled:
                    break
                except Exception as exc:
                    if last_error != type(exc).__name__:
                        logging.getLogger(__name__).error('Input integrity service check failed: %s', type(exc).__name__)
                    last_error = type(exc).__name__
                    publish(status='failed', phase='waiting', error_type=type(exc).__name__)
                stop.wait(monitor.options['interval_seconds'])
        finally:
            heartbeat_stop.set()
            thread.join(timeout=5)
            publish(status='stopped', phase='stopped')


def serve_progress(config_path, stop, *, interval=5):
    """Publish local queue observations when inference owners run separately."""
    from .config import load_config
    from .device_day import exclusive
    from .local_storage import require_local_path
    config = load_config(Path(config_path))
    root = require_local_path(Path(config['storage']['local_runtime_root'])) / 'device-day'

    class Control:
        def poll(self, timeout=None):
            return stop.is_set() if timeout is None else stop.wait(timeout)

        def close(self):
            pass

    with exclusive(root / 'locks' / 'progress-publisher.lock'):
        _publish_progress(config, Control(), interval)


def main(argv=None):
    import argparse
    import signal
    parser = argparse.ArgumentParser(description='Independent model-free operational services')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--input-integrity', action='store_true')
    modes.add_argument('--progress', action='store_true')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    logging.basicConfig(level=logging.INFO)
    if args.progress:
        serve_progress(args.config, stop)
    else:
        serve_integrity(args.config, stop)


if __name__ == '__main__':
    main()

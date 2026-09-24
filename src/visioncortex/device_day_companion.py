"""Take over device/day dispatch while the existing HTTP/discovery host stays up.

The host's disabled dispatcher and an explicit drained-thread receipt are the
handoff fence. This process never acquires Worker.lock or starts a NAS scanner.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import threading
import time

from .device_day import exclusive
from .device_day_contract import atomic_json


def process_start_ticks(pid, proc_root=Path('/proc')):
    try:
        value = (proc_root / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    # comm can contain spaces and parentheses; field 22 follows its final ')'.
    return int(value.rsplit(')', 1)[1].split()[19])


def validate_handoff(config, config_path, receipt, *, legacy_config=None,
                     legacy_config_path=None, proc_root=Path('/proc')):
    legacy_config = config if legacy_config is None else legacy_config
    legacy_config_path = config_path if legacy_config_path is None else legacy_config_path
    if legacy_config.get('device_day', {}).get('enabled') is not False:
        raise ValueError('The legacy device/day dispatcher must be disabled before takeover')
    if (receipt.get('schema_version') != 'visioncortex-device-day-handoff/1'
            or receipt.get('legacy_dispatch_drained') is not True
            or receipt.get('legacy_restart_configuration_verified') is not True
            or receipt.get('config_path') != str(Path(config_path).resolve())
            or receipt.get('legacy_config_path', receipt.get('config_path')) != str(Path(legacy_config_path).resolve())
            or receipt.get('runtime_root') != str(Path(config['storage']['local_runtime_root']).resolve())):
        raise ValueError('A matching, verified device/day handoff receipt is required')
    pid, ticks = receipt.get('legacy_pid'), receipt.get('legacy_start_ticks')
    threads = receipt.get('legacy_dispatch_thread_ids')
    if (type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0
            or not isinstance(threads, list) or not threads
            or any(type(tid) is not int or tid <= 0 for tid in threads)):
        raise ValueError('Handoff must identify the old process and its dispatcher/executor native threads')
    if process_start_ticks(pid, proc_root) == ticks:
        if any((proc_root / str(pid) / 'task' / str(tid)).exists() for tid in threads):
            raise ValueError('Legacy device/day dispatcher or executor has not drained')


class MonitorBridge:
    """Forward the host's existing local monitor snapshot, including after restart."""
    def __init__(self, runtime_root):
        self.path = Path(runtime_root) / 'state' / 'nas-recording-monitor.json'
        self.generation = None

    def poll(self, service, settings):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        generation = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if generation == self.generation:
            return False
        inventory = json.loads(self.path.read_text())
        if not isinstance(inventory.get('recordings'), list):
            raise ValueError('NAS monitor snapshot has no recording inventory')
        # A retry snapshot is a previous successful snapshot, not a fresh input
        # readiness observation. Keep its old catalog without refreshing it.
        if inventory.get('monitor', {}).get('status') != 'watching':
            return False
        service.observe(settings, inventory)
        self.generation = generation  # A failed write must be retried.
        return True


class AncillaryIndexes:
    """Replace only index/photo workers omitted by a restarted disabled host."""
    def __init__(self, handoff, settings_factory, *, process_identity=process_start_ticks):
        self.handoff, self.settings_factory = handoff, settings_factory
        self.process_identity = process_identity
        self.stop = threading.Event()
        self.threads = {}
        self.errors = {}
        self.file_index = None

    def ensure(self):
        if (self.process_identity(self.handoff['legacy_pid']) == self.handoff['legacy_start_ticks']
                or self.stop.is_set()):
            return
        from .device_day_file_index import FileIndexPublisher
        from .device_day_time_lookup import refresh_photo_index
        settings = self.settings_factory()
        if not settings['device_day']['enabled']:
            return
        def file_index():
            settings = self.settings_factory()
            if self.file_index is None:
                self.file_index = FileIndexPublisher(settings)
            else:
                self.file_index.config = settings
            self.file_index.tick()
        actions = {'file-time-index': file_index}
        for camera in settings.get('collection_ingest', {}).get('camera_role_map', {}):
            actions['capture-photos-' + camera] = lambda camera=camera: refresh_photo_index(
                self.settings_factory(), camera)
        def work(name, action):
            while not self.stop.is_set():
                try:
                    action()
                    self.errors.pop(name, None)
                except Exception as exc:
                    self.errors[name] = type(exc).__name__
                self.stop.wait(5)
        for name, action in actions.items():
            if name not in self.threads:
                thread = threading.Thread(target=work, args=(name, action), name=name, daemon=True)
                self.threads[name] = thread
                thread.start()

    def drain(self):
        self.stop.set()
        for thread in self.threads.values():
            thread.join()


class Companion:
    def __init__(self, config_path, handoff_path, *, legacy_config_path=None,
                 loader=None, service_factory=None):
        from .config import load_config
        from .device_day_service import DeviceDayService
        self.config_path, self.handoff_path = Path(config_path), Path(handoff_path)
        self.legacy_config_path = Path(legacy_config_path or config_path)
        self.loader = loader or load_config
        self.draining = threading.Event()
        self.last_settings = None
        self.error = None
        config = self.loader(self.config_path)
        self.handoff = json.loads(self.handoff_path.read_text())
        validate_handoff(config, self.config_path, self.handoff,
                         legacy_config=self.loader(self.legacy_config_path),
                         legacy_config_path=self.legacy_config_path)
        self.roots = self._roots(config)
        self.root = Path(config['storage']['local_runtime_root'])
        self.bridge = MonitorBridge(self.root)
        self.settings()
        self.service = (service_factory or DeviceDayService)(self.settings, threading.Lock())
        self.ancillaries = AncillaryIndexes(self.handoff, self.settings)

    @staticmethod
    def _roots(config):
        return {key: config['storage'].get(key) for key in
                ('local_runtime_root', 'local_cache_root', 'archive_root')} | {
                    'capture_root': config.get('collection_ingest', {}).get('source_root')}

    def settings(self):
        if self.draining.is_set() and self.last_settings is not None:
            value = deepcopy(self.last_settings)
            value['device_day']['enabled'] = False
            return value
        try:
            config = self.loader(self.config_path)
            legacy = self.loader(self.legacy_config_path)
            if legacy.get('device_day', {}).get('enabled') is not False:
                raise ValueError('Legacy device/day dispatch was re-enabled during companion ownership')
            if self._roots(config) != self.roots or self._roots(legacy) != self.roots:
                raise ValueError('Storage roots changed during companion ownership')
            from .ai_settings import apply_active
            config = apply_active(config)
            config = deepcopy(config)
            config.setdefault('device_day', {})['enabled'] = True
            self.last_settings = config
            return config
        except Exception as exc:
            if self.last_settings is None:
                raise
            # Configuration parser errors may quote secret-bearing YAML.
            self.error = 'configuration:' + type(exc).__name__
            self.draining.set()
            return self.settings()

    def request_drain(self, *_):
        # No stop_event: currently leased model/copy work finishes normally.
        self.draining.set()
        self.service.wakeup.set()

    def run(self, *, interval=1):
        from .build_identity import identity
        from .shared_inference import close_pools
        status_path = self.root / 'state' / 'DeviceDayCompanionStatus.json'
        build = identity()
        def publish(state, **extra):
            atomic_json(status_path, {'schema_version': 'visioncortex-device-day-companion/1',
                'pid': os.getpid(), 'build': build, 'status': state, 'at': time.time(),
                'config_path': str(self.config_path.resolve()),
                'legacy_config_path': str(self.legacy_config_path.resolve()),
                'handoff_receipt': str(self.handoff_path.resolve()),
                'legacy_pid': self.handoff['legacy_pid'],
                'nas_scanner': 'existing_host_local_snapshot',
                'ancillary_index_owner': 'companion' if self.ancillaries.threads else 'legacy_host',
                'ancillary_errors': dict(self.ancillaries.errors),
                'pipeline': self.service.last_result, **extra})
        with exclusive(self.root / 'state' / 'DeviceDayCompanion.lock'):
            # Recheck inside singleton ownership; no DB/model work before it.
            validate_handoff(self.loader(self.config_path), self.config_path, self.handoff,
                             legacy_config=self.loader(self.legacy_config_path),
                             legacy_config_path=self.legacy_config_path)
            self.service.start()
            try:
                while self.service.thread is not None and self.service.thread.is_alive():
                    settings = self.settings()
                    if not self.draining.is_set():
                        self.ancillaries.ensure()
                        try:
                            self.bridge.poll(self.service, settings)
                            if self.error and self.error.startswith('monitor_bridge:'):
                                self.error = None
                        except (OSError, ValueError) as exc:
                            self.error = 'monitor_bridge:' + type(exc).__name__
                    publish('draining' if self.draining.is_set() else 'running', error=self.error)
                    time.sleep(interval)
                if not self.draining.is_set():
                    self.error = 'device_day_dispatcher_exited'
            except BaseException:
                self.request_drain()
                raise
            finally:
                self.request_drain()
                while self.service.thread is not None and self.service.thread.is_alive():
                    publish('draining', error=self.error)
                    self.service.thread.join(timeout=interval)
                self.ancillaries.drain()
                close_pools()
                publish('stopped', error=self.error)
        if self.error and not self.error.startswith('monitor_bridge:'):
            raise RuntimeError(self.error)


def run_companion(config_path, handoff_path, *, legacy_config_path=None):
    os.environ['VISIONCORTEX_RUNTIME_ROLE'] = 'worker'
    companion = Companion(config_path, handoff_path, legacy_config_path=legacy_config_path)
    previous = {signum: signal.signal(signum, companion.request_drain)
                for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        companion.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

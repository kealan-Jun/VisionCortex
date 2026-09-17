"""Independent live/history discovery lanes; one blocked NAS read stays local."""
from __future__ import annotations

import threading
import time
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .device_day_contract import digest
from .nas_recordings import _camera_directories, _recording_batches, _root, scan_recordings


class CameraMonitor:
    def __init__(self, settings, stop, observe):
        self.settings, self.stop, self.observe = settings, stop, observe
        self.lock = threading.Lock()
        self.flush_lock = threading.Lock()
        self.threads, self.states, self.records, self.seen = {}, {}, {}, {}
        self.restarts = {}
        self.dirty = {}
        self.discovery = {}
        self.completed_lanes = set()
        self.paused_folders = set()
        self.sealed_folders = {}
        from .device_day_schedule import in_processing_scope, processing_cutoff
        if processing_cutoff(settings.get('device_day', {})):
            from .observed_inventory import read_inventory
            root = Path(settings['storage']['local_runtime_root']) / 'device-day'
            for record in read_inventory(root)['recordings']:
                if record.get('processable') and not in_processing_scope(settings['device_day'], record):
                    self.paused_folders.add(str(Path(record['video_path']).parent))

    def _run_lane(self, camera, mode):
        try:
            self._lane(camera, mode)
        except Exception as exc:
            # Unexpected failures are visible and the next poll restarts this
            # lane. Never leave a dead thread permanently registered as live.
            logging.getLogger(__name__).exception('NAS monitor lane stopped: %s/%s', camera, mode)
            with self.lock:
                self.states[camera, mode] = {
                    'camera_key': camera, 'mode': mode, 'status': 'failed',
                    'errors': [{'path': camera, 'error_type': type(exc).__name__,
                                'message': '监控线程异常退出，下次检查自动恢复'}]}

    def _lane(self, camera, mode):
        key = (camera, mode)
        interval = max(1, float(self.settings['collection_ingest'].get('poll_seconds', 5))) if mode == 'live' else 300
        while not self.stop.is_set():
            started = time.time()
            with self.lock:
                self.states[key] = {'camera_key': camera, 'mode': mode, 'status': 'scanning', 'started_at': started}
            config = deepcopy(self.settings)
            config['collection_ingest'].update(camera_directories=[camera], capture_date=None,
                                               max_recordings_per_camera=0)
            config['collection_ingest']['capture_since_date'] = (
                datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat() if mode == 'live'
                else self.settings.get('device_day', {}).get('start_date'))
            pending, last_emit = [], [0.0]
            def flush():
                if pending:
                    self.flush()
                    pending.clear()
                    last_emit[0] = time.monotonic()
            def forward(record):
                from .device_day_schedule import in_processing_scope
                if record.get('processable') and not in_processing_scope(self.settings.get('device_day', {}), record):
                    with self.lock:
                        self.paused_folders.add(str(Path(record['video_path']).parent))
                    return
                identity = digest(record)
                with self.lock:
                    recheck = float(self.settings['collection_ingest'].get('sealed_recheck_seconds', 0))
                    if (mode == 'live' and recheck > 0 and record.get('processable')
                            and record.get('audio', {}).get('status') in {'provided', 'no_input'}):
                        self.sealed_folders[str(Path(record['video_path']).parent)] = time.monotonic() + recheck
                    previous = self.records.get(record['recording_id'], {})
                    if previous.get('updated_at', '') > record.get('updated_at', ''):
                        return
                    if self.seen.get(record['recording_id']) == identity:
                        return
                    self.seen[record['recording_id']] = identity
                    self.records[record['recording_id']] = record
                    self.dirty[record['recording_id']] = record
                    self.discovery[record['recording_id']] = {
                        'observed_at': time.time(),
                        'cohort': 'historical_backfill' if mode == 'history' else
                                  'live_observation' if key in self.completed_lanes else 'startup_inventory'}
                pending.append(record)
                if len(pending) >= 25 or time.monotonic() - last_emit[0] >= 1:
                    flush()
            try:
                with self.lock:
                    self.sealed_folders = {p: deadline for p, deadline in self.sealed_folders.items()
                                           if deadline > time.monotonic()}
                    skipped = set(self.paused_folders) | set(self.sealed_folders)
                inventory = scan_recordings(config, on_record=forward, **({'skip_folders': skipped} if skipped else {}))
                flush()
                self.observe(self.settings, {'recordings': [], 'errors': inventory.get('errors', [])})
                state = {'camera_key': camera, 'mode': mode, 'status': 'watching', 'started_at': started,
                         'completed_at': time.time(), 'errors': inventory.get('errors', []),
                         'truncated': inventory.get('truncated', False)}
            except (OSError, ValueError, TypeError) as exc:
                # dirty is acknowledged only after a successful write. Do not
                # repeat a failing write inside the exception handler itself.
                state = {'camera_key': camera, 'mode': mode, 'status': 'retrying', 'started_at': started,
                         'errors': [{'path': camera, 'error_type': type(exc).__name__, 'message': '采集读取失败，其他相机继续'}]}
            with self.lock:
                self.states[key] = state
                if state['status'] == 'watching':
                    self.completed_lanes.add(key)
            self.stop.wait(min(interval, 5) if state['status'] == 'retrying' else interval)

    def flush(self):
        # Other scanners may keep discovering/updating records while a single
        # writer publishes a snapshot. A busy writer is not a failed scan.
        if not self.flush_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                pending = dict(self.dirty)
                discovery = {key: self.discovery[key] for key in pending if key in self.discovery}
            if not pending:
                return
            self.observe(self.settings, {"recordings": list(pending.values()), "discovery": discovery})
            with self.lock:
                for key, record in pending.items():
                    # A newer revision discovered during publication must be
                    # published later, not erased by this older snapshot.
                    if self.dirty.get(key) is record:
                        self.dirty.pop(key)
                        self.discovery.pop(key, None)
        finally:
            self.flush_lock.release()

    def poll(self):
        root = _root(self.settings)
        cameras = _camera_directories(root, self.settings['collection_ingest'], False)
        from .device_day_schedule import processing_cutoff
        modes = ('live',) if processing_cutoff(self.settings.get('device_day', {})) else ('live', 'history')
        for camera in cameras:
            for mode in modes:
                key = (camera.name, mode)
                previous = self.threads.get(key)
                if not self.stop.is_set() and (previous is None or not previous.is_alive()):
                    self.restarts[key] = self.restarts.get(key, 0) + int(previous is not None)
                    with self.lock:
                        self.states[key] = {'camera_key': camera.name, 'mode': mode, 'status': 'starting',
                                            'started_at': time.time()}
                    thread = threading.Thread(target=self._run_lane, args=key, daemon=True,
                                              name=f'nas-{mode}-{camera.name}')
                    self.threads[key] = thread
                    thread.start()
        publication_errors = []
        try:
            self.flush()
        except (OSError, ValueError, TypeError) as exc:
            publication_errors.append({'path': 'observed-inventory.json', 'error_type': type(exc).__name__,
                                       'message': '采集清单暂未写入，保留发现记录并自动重试'})
        with self.lock:
            records = list(self.records.values())
            states = [dict(value, thread_alive=self.threads[key].is_alive(),
                           restart_count=self.restarts.get(key, 0)) for key, value in self.states.items()]
            pending_count = len(self.dirty)
        for state in states:
            if state['status'] == 'scanning' and time.time() - state['started_at'] > 120:
                state['status'] = 'slow_or_unavailable'
        return {'mode': 'directory_metadata', 'recordings': records, 'recording_count': len(records),
                'batches': _recording_batches(records, self.settings['collection_ingest']),
                'errors': publication_errors + [error for state in states for error in state.get('errors', [])],
                'truncated': any(state.get('truncated') for state in states),
                'camera_directories': [camera.name for camera in cameras],
                'camera_directory_count': len(cameras), 'discovery_lanes': states,
                'pending_publication_count': pending_count,
                'discovery_in_progress': any(state['status'] in {'scanning', 'slow_or_unavailable'} for state in states)}

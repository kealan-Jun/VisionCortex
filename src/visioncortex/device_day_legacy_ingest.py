"""Explicit legacy-source intake into the existing device/day pipeline.

The native metadata-only monitor remains unchanged. Originals are retained by
the pinned existing retention implementation before these observations enter
the shared model queues. This process has no model execution port.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .device_day_contract import atomic_json, digest, read_json


def intake_config(config):
    from .capture_layout import validate

    result = deepcopy(config)
    options = result.get('device_day', {}).get('legacy_ingest') or {}
    directories = options.get('directories')
    ingest = result['collection_ingest']
    allowed = ingest.get('additional_camera_directories') or []
    if (options.get('enabled') is not True or not isinstance(directories, list) or not directories
            or any(not isinstance(name, str) or name not in allowed for name in directories)
            or len(set(directories)) != len(directories)):
        raise ValueError('Legacy intake requires explicit additional source directories')
    limit = options.get('max_imports_per_turn', 1)
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('Legacy intake requires a bounded import turn')
    ingest.update(discovery_mode='legacy_and_metadata', discover_recorder_metadata=False,
                  discover_plain_video_csv=False, camera_directories=directories,
                  max_scan_depth=3, max_recordings_per_camera=0)
    validate(ingest)
    return result


def check_record(config, record):
    """Never grant the legacy-copy exception to a native or unrelated source."""
    from .input_availability import configured_record

    options = config['device_day']['legacy_ingest']
    root = Path(config['collection_ingest']['source_root'])
    video = Path(record['video_path'])
    relative = video.relative_to(root)
    if (record.get('input_layout') or video.is_symlink() or '..' in relative.parts
            or not 4 <= len(relative.parts) <= 5 or relative.parts[0] not in options['directories']
            or relative.name != 'rgb.mp4' or relative.as_posix() != record.get('relative_path')
            or not video.resolve().is_relative_to((root / relative.parts[0]).resolve())):
        raise ValueError('Legacy intake record escaped its explicit source')
    bound = configured_record(config, record)
    if (bound.get('camera_key') != record.get('camera_key')
            or bound.get('configured_role') not in {'first_person', 'third_person'}
            or bound.get('camera_binding_status') != 'configured'):
        raise ValueError('Legacy intake requires an explicit directory/view binding')


class FrozenRetention:
    """One retention-only request to the actual unchanged production executor."""

    def __init__(self, config, config_path):
        from .source_identity import source_digest, source_manifest

        options = config['device_day']['legacy_ingest']
        self.source = Path(options['retention_source_root'])
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.source, text=True).strip()
        if actual != options['retention_source_sha'] or source_digest(source_manifest(
                self.source / 'src/visioncortex')) != options['retention_source_digest']:
            raise ValueError('Frozen legacy retention source identity mismatch')
        self.config_path = Path(config_path)
        self.config_sha = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        self.endpoint = Path(__file__).resolve().parents[2] / 'tools/migrations/retain_legacy_capture.py'
        self.timeout = float(config.get('device_day', {}).get('nas_io_timeout_seconds', 3600)) * 2

    def __call__(self, record):
        environment = dict(os.environ, PYTHONPATH=str(self.source / 'src'), PYTHONDONTWRITEBYTECODE='1')
        # Build identity must describe the actual executor, not this scanner.
        environment.pop('VISIONCORTEX_BUILD_MANIFEST', None)
        message = json.dumps(record, separators=(',', ':')).encode()
        if len(message) > 1024**2:
            raise ValueError('Legacy retention request exceeds its bound')
        result = subprocess.run(
            [sys.executable, str(self.endpoint), str(self.config_path), self.config_sha],
            input=message, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.source, env=environment, timeout=self.timeout, check=True)
        if len(result.stdout) > 1024**2:
            raise ValueError('Legacy retention response exceeds its bound')
        return json.loads(result.stdout)


class LegacyIntake:
    def __init__(self, config, retain, observe):
        self.config = intake_config(config)
        self.retain, self.observe = retain, observe
        self.root = Path(config['storage']['local_runtime_root']) / 'device-day/LegacyIntake'
        self.path = self.root / 'Checkpoint.json'
        self.saved = read_json(self.path) if self.path.is_file() else {'records': {}}
        self.deferred = {}

    def tick(self, *, stop=None, scan=None):
        from .nas_recordings import scan_recordings
        from .device_day_recovery_worker import verify_storage

        verify_storage(self.config)
        inventory = (scan or scan_recordings)(self.config)
        errors = list(inventory.get('errors') or [])
        retained = reused = waiting = attempted = 0
        limit = self.config['device_day']['legacy_ingest'].get('max_imports_per_turn', 1)
        for record in sorted(inventory['recordings'], key=lambda r: (r['recording_start_us'], r['relative_path']), reverse=True):
            if stop is not None and stop.is_set():
                break
            signature = digest([record['source_signature'], record.get('audio', {}).get('source_signature'),
                                record.get('camera_key'), record.get('configured_role')])
            rid = record['recording_id']
            if not record.get('processable'):
                waiting += 1
                continue
            try:
                check_record(self.config, record)
                if self.saved['records'].get(rid, {}).get('signature') == signature:
                    reused += 1
                else:
                    if self.deferred.get(rid, 0) > time.time():
                        continue
                    if attempted >= limit:
                        break
                    attempted += 1
                    self.status('retaining', inventory, errors, current=record['relative_path'])
                    result = self.retain(record)
                    if result.get('status') != 'completed' or result.get('stage') != 'retention':
                        raise ValueError('Legacy retention has not completed')
                    self.saved['records'][rid] = {'signature': signature, 'result': result, 'at': time.time()}
                    atomic_json(self.path, self.saved)
                    self.deferred.pop(rid, None)
                    retained += 1
                # Existing queue admission, leases, inference and publication
                # remain the sole owners after a verified original is present.
                self.observe(self.config, {'recordings': [record], 'discovery': {rid: {
                    'observed_at': time.time(), 'cohort': 'historical_backfill'}}})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.deferred[rid] = time.time() + 60
                errors.append({'path': record['relative_path'], 'error_type': type(exc).__name__,
                               'message': 'Legacy intake deferred; original source preserved'})
        return self.status('degraded' if errors or self.deferred or inventory.get('truncated') else 'watching', inventory, errors,
                           retained_this_turn=retained, reused_this_turn=reused, waiting_for_capture=waiting,
                           imports_attempted=attempted, deferred_count=len(self.deferred))

    def status(self, status, inventory, errors, **extra):
        value = {'status': status, 'updated_at': time.time(), 'source_directories':
                 self.config['device_day']['legacy_ingest']['directories'],
                 'discovered': len(inventory['recordings']), 'retained': len(self.saved['records']),
                 'errors': errors, 'truncated': inventory.get('truncated', False),
                 'model_calls': 0, **extra}
        atomic_json(self.root / 'Service.json', value)
        return value


def serve(config_path, stop):
    from .config import load_config
    from .device_day import exclusive
    from .device_day_service import DeviceDayService
    from .device_day_recovery_worker import verify_storage

    config = load_config(Path(config_path))
    intake_config(config)
    verify_storage(config)
    service = DeviceDayService(lambda: config, threading.Lock())
    worker = LegacyIntake(config, FrozenRetention(config, config_path), service.observe)
    with exclusive(worker.root.parent / 'locks/legacy-intake.lock'):
        while not stop.is_set():
            try:
                result = worker.tick(stop=stop)
            except (OSError, ValueError) as exc:
                worker.status('retrying', {'recordings': []}, [{'error_type': type(exc).__name__}])
                result = {}
            stop.wait(1 if result.get('imports_attempted') else 60)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    serve(args.config, stop)


if __name__ == '__main__':
    main()

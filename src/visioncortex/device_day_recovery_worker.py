"""Run archive recovery independently of HTTP and model stage consumers."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time

from .device_day_contract import atomic_json, digest, read_json


def verify_storage(config, *, mountinfo=Path('/proc/self/mountinfo')):
    """Fail closed before NAS access; expected identities stay in site config."""
    settings = (config.get('device_day') or {}).get('recovery') or {}
    checks = settings.get('storage_checks') or []
    if not checks:
        if (config.get('runtime') or {}).get('local_only'):
            return
        raise ValueError('Archive recovery requires verified storage identities')
    mounts = {}
    for line in mountinfo.read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, details = left.split(), right.split()
        # mountinfo escapes whitespace and backslashes using octal sequences.
        path = fields[4]
        for escaped, literal in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            path = path.replace(escaped, literal)
        mounts[Path(path)] = (details[0], details[1])
    for check in checks:
        expected = (Path(check['mount_path']).absolute(), check['filesystem'], check['source'])
        if mounts.get(expected[0]) != expected[1:]:
            raise ValueError('Archive recovery storage mount identity mismatch')
        marker = Path(check['marker_path'])
        if not marker.absolute().is_relative_to(expected[0]):
            raise ValueError('Storage marker must remain under the verified mount')
        if marker.is_symlink() or not marker.resolve().is_relative_to(expected[0].resolve()):
            raise ValueError('Storage marker escaped the verified mount')
        actual = read_json(marker)
        if actual.get(check.get('marker_id_field', 'volume_id')) != check['marker_id']:
            raise ValueError('Archive recovery storage volume identity mismatch')
        if any(actual.get(key) != value for key, value in (check.get('marker_fields') or {}).items()):
            raise ValueError('Archive recovery storage marker identity mismatch')


def provider_probe(config):
    """The scoped circuit owns modern half-open requests; keep legacy recovery."""
    from .device_day import exclusive
    from .device_day_provider_gate import ProviderGate
    gate = ProviderGate(config)
    if (config.get('runtime') or {}).get('provider_circuit_enabled'):
        # blocks() still honors an applicable legacy marker after checking the
        # modern circuit. Only that active legacy gate needs its old recovery
        # owner; absent/inactive or another provider's marker stays transport-only.
        if (config.get('mllm') or {}).get('provider') != 'aliyun' or not gate.path.is_file():
            return {'status': 'transport_managed'}
        if not read_json(gate.path).get('active'):
            return {'status': 'transport_managed'}
    root = Path(config['storage']['local_runtime_root']) / 'device-day'
    try:
        with exclusive(root / 'locks' / 'legacy-provider-probe.lock'):
            state = gate.probe_if_due()
    except BlockingIOError:
        return {'status': 'running_elsewhere'}
    return {'status': 'blocked' if state.get('active') else 'available',
            **{key: state[key] for key in ('last_probe_at', 'next_probe_at', 'last_probe_status') if key in state}}


class RecoveryStatus:
    """Fresh ownership plus tick progress, including during long byte checks."""

    def __init__(self, root, *, interval=2):
        from .build_identity import identity
        from .device_day_consumers import process_start_ticks
        self.path = Path(root) / 'RetentionRecovery' / 'Service.json'
        self.interval = interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.state = {'schema_version': 'visioncortex-retention-recovery-worker/1',
                      'pid': os.getpid(), 'process_start_ticks': process_start_ticks(os.getpid()),
                      'build': identity(), 'status': 'starting', 'started_at': time.time(),
                      'tick_count': 0, 'completed_tick_count': 0, 'recovered_count': 0}

    def publish(self, **fields):
        with self.lock:
            self.state.update(fields)
            atomic_json(self.path, self.state | {'updated_at': time.time()})

    def __enter__(self):
        self.publish(status='running')
        def heartbeat():
            while not self.stop.wait(self.interval):
                try:
                    self.publish()
                except OSError:
                    pass
        self.thread = threading.Thread(target=heartbeat, name='recovery-heartbeat', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=max(5, self.interval * 2))
        self.publish(status='stopped')


def tick(runner, recovery, stop):
    from .device_day_io import slot
    from .runtime_control import execution_context, resource_slot
    verify_storage(runner.config)
    with execution_context(job_id='retention-recovery', source='retention_recovery',
                           priority=1, stop=stop, yield_signal=stop):
        with resource_slot(runner.config, 'cpu'), slot(runner.config):
            return recovery.tick(runner)


def serve(config_path, stop):
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner
    from .device_day_recovery import RetentionRecovery
    from .local_storage import require_local_path
    from .runtime_control import ExecutionCancelled
    runner, generation, roots, status = None, None, None, None
    recovery = None
    try:
        while not stop.is_set():
            try:
                config = apply_active(load_config(Path(config_path)))
                current_roots = {key: config['storage'].get(key) for key in
                                 ('local_runtime_root', 'local_cache_root', 'archive_root')}
                if roots is not None and roots != current_roots:
                    raise ValueError('Archive recovery storage roots changed')
                # SQLite queues/checkpoints must never use a network filesystem.
                require_local_path(Path(current_roots['local_runtime_root']))
                roots = current_roots
                if status is None:
                    status = RecoveryStatus(Path(roots['local_runtime_root']) / 'device-day').__enter__()
                if not (config.get('device_day') or {}).get('enabled'):
                    status.publish(status='disabled')
                    stop.wait(5)
                    continue
                key = digest(config)
                if runner is None or generation != key:
                    # Recovery never calls a model backend, even accidentally.
                    runner = DeviceDayRunner(config, backend=object())
                    settings = runner.settings.get('recovery') or {}
                    recovery = RetentionRecovery(batch_size=settings.get('batch_size', 128),
                                                 cooldown_seconds=settings.get('cooldown_seconds', 900))
                    generation = key
                status.publish(status='running', phase='checking', tick_started_at=time.time(),
                               tick_count=status.state['tick_count'] + 1)
                probe = provider_probe(config)
                result = tick(runner, recovery, stop)
                status.publish(phase='waiting', last_tick_completed_at=time.time(),
                               completed_tick_count=status.state['completed_tick_count'] + 1,
                               recovered_count=status.state['recovered_count'] + int(result.get('status') == 'completed'),
                               last_result={name: result[name] for name in
                                            ('status', 'recording_id', 'error_type') if name in result},
                               provider_probe=probe)
            except ExecutionCancelled:
                if status is not None:
                    status.publish(status='stopping', phase='cancelled')
                break
            except Exception as exc:
                if status is None:
                    # There is no safe local status destination yet. Let the
                    # supervisor restart, with a redacted startup failure.
                    raise RuntimeError('Archive recovery startup failed: ' + type(exc).__name__) from None
                if status is not None:
                    status.publish(status='failed', phase='waiting', error_type=type(exc).__name__)
            stop.wait(5)
    finally:
        if status is not None:
            status.__exit__(None, None, None)


def main(argv=None):
    import argparse
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        serve(args.config, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()

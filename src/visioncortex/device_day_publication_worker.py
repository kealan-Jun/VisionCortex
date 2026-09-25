"""Replay bounded standard publication journals without starting model work."""
from contextlib import ExitStack
from pathlib import Path
import time

from .device_day_contract import atomic_json, digest


class PublicationReplay:
    def tick(self, runner):
        from .device_day import exclusive
        from .publication_journal import PublicationJournal, reconcile
        from .device_day_schedule import processing_cutoff
        with ExitStack() as locks:
            try:
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / 'publication-replay.lock'))
            except BlockingIOError:
                return {'status': 'running_elsewhere', 'published': 0}
            # The standard replay owns per-record stage locks, validates every
            # receipt/artifact, prefers recent records and bounds each round.
            published = reconcile(runner)
            pending = PublicationJournal(runner.runtime_root).pending(1, since_us=processing_cutoff(runner.settings))
            return {'status': 'completed', 'published': published, 'pending_publications': bool(pending)}


def serve(config_path, stop):
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner
    replay = PublicationReplay()
    runner, generation, roots, state_path = None, None, None, None
    while not stop.is_set():
        started = time.time()
        try:
            config = apply_active(load_config(Path(config_path)))
            current_roots = {key: config['storage'].get(key) for key in
                             ('local_runtime_root', 'local_cache_root', 'archive_root')}
            if roots is not None and roots != current_roots:
                raise ValueError('Publication replay storage roots changed')
            roots = current_roots
            key = digest(config)
            if runner is None or key != generation:
                runner, generation = DeviceDayRunner(config), key
            state_path = runner.runtime_root / 'PublicationReplay' / 'Service.json'
            atomic_json(state_path, {'status': 'running', 'started_at': started,
                                    'scope': 'standard_publication_journal', 'model_invoked': False})
            result = replay.tick(runner)
            status = {'status': 'watching', 'started_at': started, 'updated_at': time.time(),
                      'last_result': result, 'scope': 'standard_publication_journal', 'model_invoked': False}
        except Exception as exc:
            status = {'status': 'failed', 'started_at': started, 'updated_at': time.time(),
                      'error_type': type(exc).__name__, 'scope': 'standard_publication_journal', 'model_invoked': False}
        if state_path is not None:
            atomic_json(state_path, status)
        stop.wait(5)
    if state_path is not None:
        atomic_json(state_path, {'status': 'stopped', 'updated_at': time.time(),
                                'scope': 'standard_publication_journal', 'model_invoked': False})


def main(argv=None):
    import argparse
    import signal
    import threading
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

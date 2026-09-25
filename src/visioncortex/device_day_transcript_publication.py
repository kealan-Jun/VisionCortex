"""Publish verified speech promptly, independently of video/day publication."""
from contextlib import ExitStack
import json
import time

from .device_day_contract import atomic_json, digest, read_json


def publish_one(runner, record, *, queue_token=None):
    from .device_day import exclusive
    from .device_day_content import publish_transcript
    from .device_day_inputs import binding
    layout = runner.layout(record)
    with ExitStack() as locks:
        for stage in ('all', 'stt'):
            locks.enter_context(exclusive(runner.runtime_root / 'locks' /
                                         f"{record['recording_id']}.{stage}.lock"))
        if queue_token is not None:
            from .sqlite_store import connection
            with connection(runner.queues['stt'].path, readonly=True) as db:
                row = db.execute('SELECT status,revision,updated_at,payload FROM recordings WHERE recording_id=?',
                                 (record['recording_id'],)).fetchone()
            if (not row or row['status'] != 'completed' or json.loads(row['payload']) != record
                    or digest([row['revision'], row['updated_at']]) != queue_token):
                raise ValueError('Completed speech queue generation changed')
        seal_path = runner._receipt(layout, record, 'input-stt')
        if seal_path.is_file():
            seal = read_json(seal_path)
            saved = seal['recording']
            fields = ('recording_id', 'camera_key', 'recording_start_us', 'recording_end_us', 'configured_role')
            if (seal.get('status') != 'ready' or seal.get('recording_id') != record['recording_id']
                    or any(saved.get(k) != record.get(k) for k in fields)
                    or seal.get('source_signature') != record.get('audio', {}).get('source_signature')):
                raise ValueError('Completed speech input no longer matches recording')
            # Reconstruct the exact sealed recipe without reopening source media.
            # The completed receipt and all its outputs are verified below.
            audio = dict(seal['audio'])
            audio['artifacts'] = [s['retained'] | {'kind': s['kind']} for s in seal['sources']
                                  if s['kind'].startswith('audio_')]
            inputs = {'input_binding_version': 1, 'recording': saved, 'sources': seal['sources'],
                      'audio': audio, 'artifacts': [], 'input_binding': binding(seal)}
        else:
            inputs = read_json(runner._receipt(layout, record, 'retention'))
            if (inputs.get('status') != 'completed'
                    or inputs.get('recording', {}).get('source_signature') != record.get('source_signature')):
                raise ValueError('Completed speech has no matching retained input')
        receipt = runner._load(runner._receipt(layout, record, 'stt'), runner._key('stt', record, inputs), layout)
        if (not receipt or receipt.get('stage') != 'stt' or receipt.get('recording_id') != record['recording_id']
                or (seal_path.is_file() and receipt.get('input_binding') != inputs['input_binding'])):
            raise ValueError('Completed speech receipt or artifacts failed verification')
        return publish_transcript(runner.config, layout.name, {
            'recording_id': record['recording_id'], 'start_us': record['recording_start_us'],
            'end_us': record['recording_end_us'], 'audio': inputs.get('audio'), 'transcription': receipt})


class TranscriptPublication:
    """Replay completed queue rows after restart; failures never rerun ASR."""

    def tick(self, runner, *, limit=4):
        from .device_day import exclusive
        # The optional standalone publisher and a future upgraded companion
        # share both the projection checkpoint and this OS-owned lock.
        with ExitStack() as locks:
            try:
                locks.enter_context(exclusive(runner.runtime_root / 'locks' / 'transcript-publication.lock'))
            except BlockingIOError:
                return {'status': 'running_elsewhere', 'results': []}
            return self._tick(runner, limit=limit)

    def _tick(self, runner, *, limit):
        from .device_day_schedule import in_processing_scope
        from .sqlite_store import connection
        state_path = runner.runtime_root / 'TranscriptProjection' / 'State.json'
        try:
            state = read_json(state_path) if state_path.is_file() else {}
        except (OSError, ValueError):
            state = {}  # Lost projection bookkeeping cannot confer receipt trust.
        current = time.time()
        with connection(runner.queues['stt'].path, readonly=True) as db:
            rows = db.execute("SELECT recording_id,revision,updated_at FROM recordings "
                              "WHERE status='completed' ORDER BY "
                              "COALESCE(json_extract(payload,'$.recording_start_us'),0) DESC").fetchall()
            selected = []
            for row in rows:
                token = digest([row['revision'], row['updated_at']])
                previous = state.get(row['recording_id'], {})
                if previous.get('token') == token and (previous.get('status') == 'completed'
                                                       or previous.get('retry_at', 0) > current):
                    continue
                record = json.loads(db.execute('SELECT payload FROM recordings WHERE recording_id=?',
                                               (row['recording_id'],)).fetchone()['payload'])
                if in_processing_scope(runner.settings, record):
                    selected.append((token, record))
                if len(selected) >= limit:
                    break
        results = []
        for token, record in selected:
            try:
                result = {'status': 'completed', 'projection': publish_one(runner, record, queue_token=token)}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {'status': 'failed', 'error_type': type(exc).__name__, 'retry_at': current + 30}
            result |= {'recording_id': record['recording_id'], 'token': token, 'updated_at': time.time(),
                       'scope': 'readable_transcript_only', 'model_invoked': False}
            state[record['recording_id']] = result
            results.append(result)
            atomic_json(state_path, state)
        return {'results': results}


def serve(config_path, stop):
    """No dispatcher, discovery, backend startup or inference is started here."""
    from pathlib import Path
    from .ai_settings import apply_active
    from .config import load_config
    from .device_day import DeviceDayRunner
    publisher = TranscriptPublication()
    runner, generation, roots = None, None, None
    status_path = None
    while not stop.is_set():
        try:
            config = apply_active(load_config(Path(config_path)))
            current_roots = {key: config['storage'].get(key) for key in
                             ('local_runtime_root', 'local_cache_root', 'archive_root')}
            if roots is not None and current_roots != roots:
                raise ValueError('Transcript publisher storage roots changed')
            roots = current_roots
            key = digest(config)
            if runner is None or generation != key:
                runner = DeviceDayRunner(config)
                generation = key
            status_path = runner.runtime_root / 'TranscriptProjection' / 'Service.json'
            result = publisher.tick(runner)
            status = {'status': 'running', 'updated_at': time.time(), 'last_result': result,
                      'scope': 'readable_transcript_only', 'model_invoked': False}
        except Exception as exc:
            status = {'status': 'failed', 'updated_at': time.time(), 'error_type': type(exc).__name__,
                      'scope': 'readable_transcript_only', 'model_invoked': False}
        if status_path is not None:
            atomic_json(status_path, status)
        stop.wait(5)
    if status_path is not None:
        atomic_json(status_path, {'status': 'stopped', 'updated_at': time.time(),
                                 'scope': 'readable_transcript_only', 'model_invoked': False})


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

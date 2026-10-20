"""Split STT replays native speech publication without another service or ASR."""
from contextlib import nullcontext
import json
import threading
import time

import pytest

from test_device_day_transcript_publication import speech  # noqa: F401
from test_device_day import device_config  # noqa: F401
from visioncortex.device_day_contract import atomic_json, digest, read_json
from visioncortex.device_day_transcript_publication import TranscriptPublication, TranscriptPublicationSupervisor


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, 'bounded background task did not progress'
        time.sleep(.01)


@pytest.fixture
def publication_ports(monkeypatch):
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())


def test_split_stt_serve_publishes_existing_completed_while_next_asr_is_blocked(speech, publication_ports, monkeypatch):  # noqa: F811
    from visioncortex import device_day_retention_worker as module
    runner, record, layout, calls = speech
    runner.config['device_day'].update(enabled=True, latest_first=True, inplace_preprocessing=True)
    blocked, release, stop = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr('visioncortex.config.load_config', lambda path: runner.config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda config: config)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', lambda config: runner)
    monkeypatch.setattr(module, 'DeferredPrimaryPublicationRunner', lambda config: runner)
    def long_stt(self, current, signal):
        assert self.stage == 'stt'
        blocked.set()
        assert release.wait(5)
        return {'status': 'waiting', 'admitted': 0}
    monkeypatch.setattr(module.RetentionWorker, 'tick', long_stt)
    before = read_json(runner._receipt(layout, record, 'stt'))
    service = threading.Thread(target=module.serve, args=('owned.yaml', stop), kwargs={'stage': 'stt'})
    service.start()
    try:
        assert blocked.wait(5)
        wait_for(lambda: bool(list(layout.comments.glob('*/Transcript.json'))))
        value = read_json(next(layout.comments.glob('*/Transcript.json')))
        assert value['sentences'][0]['text'] == 'owned fixture words'
        assert not release.is_set() and len(calls) == 1
        assert read_json(runner._receipt(layout, record, 'stt')) == before
        assert not layout.index.exists()
    finally:
        stop.set()
        release.set()
        service.join(5)
    assert not service.is_alive()
    assert read_json(runner.runtime_root/'TranscriptProjection/Service.json')['status'] == 'stopped'


def test_supervisor_waits_for_old_generation_and_atomic_write_before_close(speech, publication_ports, monkeypatch):  # noqa: F811
    from visioncortex import device_day_transcript_publication as module
    runner, _, _, calls = speech
    started, release, stop = threading.Event(), threading.Event(), threading.Event()
    generations = []
    def tick(self, current, **kwargs):
        generations.append(current.config['revision'])
        started.set()
        assert release.wait(5)
        return {'results': []}
    runner.config['revision'] = 'old'
    monkeypatch.setattr(module.TranscriptPublication, 'tick', tick)
    publisher = TranscriptPublicationSupervisor(stop, interval=.01)
    publisher.poll(runner)
    assert started.wait(5)
    from types import SimpleNamespace
    changed = SimpleNamespace(config=runner.config | {'revision': 'new'}, runtime_root=runner.runtime_root)
    assert publisher.poll(changed)['status'] == 'configuration_handover'
    assert generations == ['old']
    closer = threading.Thread(target=publisher.close)
    closer.start()
    assert closer.is_alive()
    release.set()
    closer.join(5)
    assert not closer.is_alive() and generations == ['old'] and len(calls) == 1


def test_storage_failure_is_separate_and_restart_replays_native_completed(speech, publication_ports, monkeypatch):  # noqa: F811
    runner, _, layout, calls = speech
    stop = threading.Event()
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage',
                        lambda config: (_ for _ in ()).throw(OSError('offline')))
    publisher = TranscriptPublicationSupervisor(stop, interval=.01)
    try:
        publisher.poll(runner)
        path = runner.runtime_root/'TranscriptProjection/Service.json'
        wait_for(lambda: path.is_file() and read_json(path)['status'] == 'failed')
        assert not list(layout.comments.glob('*/Transcript.json'))
        with runner.queues['stt'].connect() as db:
            assert db.execute('SELECT status FROM recordings').fetchone()['status'] == 'completed'
    finally:
        publisher.close()
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    resumed = TranscriptPublicationSupervisor(stop, interval=.01)
    try:
        resumed.poll(runner)
        wait_for(lambda: bool(list(layout.comments.glob('*/Transcript.json'))))
        assert len(calls) == 1
    finally:
        resumed.close()


def test_continuous_new_completed_speech_keeps_an_oldest_share_and_backoff(speech, monkeypatch):  # noqa: F811
    from visioncortex import device_day_transcript_publication as module
    runner, record, _, calls = speech
    queue = runner.queues['stt']
    # Only selection is mocked here; native artifact/binding checks are covered
    # by the integration above and the original transcript-publication suite.
    published = []
    monkeypatch.setattr(module, 'publish_one', lambda current, value, **kw: published.append(value['recording_id']) or {})
    with queue.connect() as db:
        original = dict(db.execute('SELECT * FROM recordings').fetchone())
        db.execute('DELETE FROM recordings')
        def insert(identifier, stamp):
            value = original | {'recording_id': identifier,
                'payload': json.dumps(record | {'recording_id': identifier, 'recording_start_us': stamp}),
                'revision': digest(identifier), 'updated_at': stamp}
            columns = ','.join(value)
            db.execute(f'INSERT INTO recordings({columns}) VALUES({",".join("?" for _ in value)})', tuple(value.values()))
        insert('oldest', 1)
        for i in range(8):
            insert('recent'+str(i), i+100)
    publisher = TranscriptPublication()
    assert len(publisher.tick(runner, limit=4)['results']) == 4
    assert 'oldest' in published and published[:3] == ['recent7', 'recent6', 'recent5']
    assert len(calls) == 1
    # A failed projection keeps its own token/backoff, never rewrites ASR.
    with queue.connect() as db:
        row = db.execute("SELECT revision,updated_at FROM recordings WHERE recording_id='recent4'").fetchone()
        token = digest([row['revision'], row['updated_at']])
    failed = {'token': token, 'status': 'failed', 'retry_at': time.time()+60}
    path = runner.runtime_root/'TranscriptProjection/State.json'
    state = read_json(path)
    state['recent4'] = failed
    atomic_json(path, state)
    with queue.connect() as db:
        for i in range(8):
            insert('arrived'+str(i), 1000+i)
    publisher.tick(runner, limit=4)
    assert 'recent4' not in published
    assert published[-1] == 'recent0' and published[-4:-1] == ['arrived7', 'arrived6', 'arrived5']

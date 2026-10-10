"""Owned fixtures only: independent speech delivery, not ASR quality evidence."""
from pathlib import Path

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner, exclusive
from visioncortex.device_day_contract import artifact, atomic_json, read_json
from visioncortex.device_day_transcript_publication import TranscriptPublication, publish_one


@pytest.fixture
def speech(device_config):  # noqa: F811
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    calls = []
    def transcribe(layout, inputs, key):
        calls.append(key)
        path = layout.comments / 'Recognition' / key / 'Transcript.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('owned fixture words')
        return {'outcome': 'transcribed', 'comments': [{'text': 'owned fixture words',
                'start_us': record['recording_start_us']}], 'artifacts': [artifact(layout.root, path)],
                'transcript_file': layout.relative(path), 'model_invocation': 'NOT_PROVEN'}
    backend.transcribe = transcribe
    runner = DeviceDayRunner(device_config, backend)
    result = runner.run_once({'recordings': [record]}, stage='stt')
    assert result['results'][0]['status'] == 'completed', result
    return runner, record, layout, calls


def test_transcript_publishes_before_video_retention_without_source_media_reads(speech, monkeypatch):
    runner, record, layout, calls = speech
    original = Path(record['video_path']).read_bytes()
    monkeypatch.setattr('visioncortex.device_day_inputs.hash_content', lambda *a: pytest.fail('source media read'))
    result = publish_one(runner, record)
    data = read_json(layout.root / result['content'])
    assert data['sentences'][0]['text'] == 'owned fixture words'
    assert data['status'] == 'completed' and data['physical_action_confirmed'] is False
    assert data['model_invocation'] == 'NOT_PROVEN'
    assert 'owned fixture words' in (layout.root / result['text']).read_text()
    assert not layout.index.exists()
    assert not runner._receipt(layout, record, 'retention').exists()
    assert Path(record['video_path']).read_bytes() == original
    assert len(calls) == 1


@pytest.mark.parametrize('damage', ['artifact', 'binding', 'key', 'source'])
def test_invalid_receipt_never_becomes_readable(speech, damage):
    runner, record, layout, _ = speech
    path = runner._receipt(layout, record, 'stt')
    value = read_json(path)
    if damage == 'artifact':
        (layout.root / value['artifacts'][0]['path']).write_text('tampered')
    elif damage == 'source':
        record = record | {'audio': {'source_signature': 'changed'}}
    else:
        value['input_binding' if damage == 'binding' else 'key'] = 'incorrect'
        atomic_json(path, value)
    with pytest.raises(ValueError):
        publish_one(runner, record)
    assert not list(layout.comments.glob('*/Transcript.json'))


def test_failed_projection_retries_after_restart_without_retranscription(speech, monkeypatch):
    runner, record, layout, calls = speech
    from visioncortex import device_day_content
    original = device_day_content.publish_transcript
    monkeypatch.setattr(device_day_content, 'publish_transcript', lambda *a: (_ for _ in ()).throw(OSError('offline')))
    publisher = TranscriptPublication()
    failed = publisher.tick(runner)['results'][0]
    assert failed['status'] == 'failed'
    monkeypatch.setattr(device_day_content, 'publish_transcript', original)
    monkeypatch.setattr('visioncortex.device_day_transcript_publication.time.time', lambda: failed['retry_at'] + 1)
    result = TranscriptPublication().tick(runner)['results'][0]
    assert result['status'] == 'completed'
    assert (layout.root / result['projection']['text']).is_file()
    assert TranscriptPublication().tick(runner)['results'] == []
    assert len(calls) == 1 and not layout.index.exists()


def test_live_stt_lock_defers_projection(speech):
    runner, record, layout, _ = speech
    with exclusive(runner.runtime_root / 'locks' / f"{record['recording_id']}.stt.lock"):
        result = TranscriptPublication().tick(runner)['results'][0]
    assert result['status'] == 'failed'
    assert not list(layout.comments.glob('*/Transcript.json'))


def test_queue_generation_changed_rejects_projection(speech):
    runner, record, layout, _ = speech
    with pytest.raises(ValueError, match='queue generation'):
        publish_one(runner, record, queue_token='stale')
    assert not list(layout.comments.glob('*/Transcript.json'))


def test_service_publishes_speech_while_day_index_worker_is_blocked(speech, monkeypatch):
    import threading
    import time
    from visioncortex import device_day_service
    runner, record, layout, calls = speech
    blocked, release = threading.Event(), threading.Event()
    def slow_index(*args):
        blocked.set()
        release.wait(10)
        return 0
    monkeypatch.setattr('visioncortex.publication_journal.reconcile', slow_index)
    monkeypatch.setattr(runner, 'run_once', lambda *a, **k: {'results': []})
    monkeypatch.setattr(device_day_service, 'DeviceDayRunner', lambda _: runner)
    monkeypatch.setattr('visioncortex.device_day_overview.ArchiveOverview.publish', lambda *a: {})
    monkeypatch.setattr('visioncortex.device_day_recovery.RetentionRecovery.tick', lambda *a: {})
    monkeypatch.setattr('visioncortex.device_day_timeline.refresh_timeline', lambda *a, **k: {})
    service = device_day_service.DeviceDayService(lambda: runner.config, threading.Lock())
    service.observe(runner.config, {'recordings': [record]})
    service.start()
    try:
        assert blocked.wait(5)
        deadline = time.monotonic() + 5
        while not list(layout.comments.glob('*/Transcript.json')):
            assert time.monotonic() < deadline, service.last_result
            time.sleep(.02)
        assert not release.is_set() and not layout.index.exists()
        assert len(calls) == 1
    finally:
        release.set()
        service.stop()


def test_global_projection_lock_excludes_other_process(speech):
    import subprocess
    import sys
    runner, _, _, calls = speech
    with exclusive(runner.runtime_root / 'locks' / 'transcript-publication.lock'):
        script = ("from pathlib import Path; from types import SimpleNamespace; "
                  "from visioncortex.device_day_transcript_publication import TranscriptPublication; "
                  f"r=SimpleNamespace(runtime_root=Path({str(runner.runtime_root)!r})); "
                  "assert TranscriptPublication().tick(r)['status']=='running_elsewhere'")
        subprocess.run([sys.executable, '-c', script], check=True, timeout=5)
    assert TranscriptPublication().tick(runner)['results'][0]['status'] == 'completed'
    assert len(calls) == 1


def test_standalone_finishes_current_projection_after_stop_signal(speech, monkeypatch):
    import threading
    from visioncortex import device_day_transcript_publication as module
    runner, _, layout, calls = speech
    stop = threading.Event()
    events = []
    monkeypatch.setattr('visioncortex.config.load_config', lambda path: runner.config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda config: events.append('active') or config)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', lambda config: runner)
    original = module.publish_one
    def publish(*args, **kwargs):
        stop.set()  # SIGTERM requests no new round, not cancellation of this one.
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'publish_one', publish)
    module.serve('owned-test-config.yaml', stop)
    assert events == ['active'] and len(calls) == 1
    assert list(layout.comments.glob('*/Transcript.json'))
    assert read_json(runner.runtime_root / 'TranscriptProjection' / 'Service.json')['status'] == 'stopped'
    assert not layout.index.exists()


def test_cli_accepts_config_and_installs_graceful_signals(monkeypatch):
    import signal
    from visioncortex import device_day_transcript_publication as module
    before = signal.getsignal(signal.SIGTERM)
    def serve(config, stop):
        assert config == 'owned-test.yaml'
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert stop.is_set()
    monkeypatch.setattr(module, 'serve', serve)
    module.main(['--config', 'owned-test.yaml'])
    assert signal.getsignal(signal.SIGTERM) == before

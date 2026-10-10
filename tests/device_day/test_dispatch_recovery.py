"""Split-worker contracts with owned files and fake external boundaries only."""
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from test_device_day_retention_worker import record, setup
from visioncortex import device_day_backfill, device_day_retention_worker, device_day_stt, speech
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe


def speech_runner(config, monkeypatch, status, *, embedded=False):
    config['device_day'].update(inplace_preprocessing=True, latest_first=True)
    folder = capture(config)
    if status == 'no_input':
        path = folder / 'audio_meta.json'
        atomic_json(path, {'quality_status': 'no_input'})
        os.utime(path, (1, 1))
    item, layout = item_and_layout(config)
    assert item['audio']['status'] == status
    models = FakeModels()
    models.transcribe = lambda layout, retained, key: device_day_stt.transcribe(config, layout, retained, key)
    probed = []

    def probe(path):
        probed.append(Path(path))
        return {'duration_seconds': 3} if embedded else None

    monkeypatch.setattr(speech, 'probe_audio', probe)
    monkeypatch.setattr(speech, 'runtime_request', lambda *_: pytest.fail('Unexpected real ASR setup'))
    monkeypatch.setattr(speech, 'invoke', lambda *_a, **_kw: pytest.fail('Unexpected real ASR invocation'))
    runner = DeviceDayRunner(config, models)
    observe(runner.runtime_root, {'recordings': [item]})
    return runner, item, layout, models, probed


@pytest.mark.parametrize('status', ['no_input', 'not_provided'])
def test_no_recorder_audio_is_probed_and_releases_downstream(device_config, monkeypatch, status):  # noqa: F811
    runner, item, layout, models, probed = speech_runner(device_config, monkeypatch, status)
    stop = threading.Event()
    # Dispatch through the actual split lanes, not an all-stage runner shortcut.
    for stage in ('vision', 'stt', 'retention', 'understanding', 'report'):
        result = RetentionWorker(stage=stage).tick(runner, stop)
        assert result['result']['status'] == 'completed', result
    saved = read_json(runner._receipt(layout, item, 'stt'))
    assert probed == [Path(item['video_path'])]
    assert saved['outcome'] == 'no_audio'
    assert saved['source_status'] == status
    assert saved['comments'] == saved['artifacts'] == []
    assert saved['model_invocation'] == 'NOT_PROVEN'
    assert read_json(runner._receipt(layout, item, 'publication'))['status'] == 'completed'
    assert (layout.reports / 'LaboratoryDailyReport.html').is_file()
    assert models.vision_calls == models.semantic_calls == 1
    original = runner._receipt(layout, item, 'stt').read_bytes()
    assert RetentionWorker(stage='stt').tick(runner, stop)['status'] == 'waiting'
    assert runner._receipt(layout, item, 'stt').read_bytes() == original
    assert len(probed) == 1


@pytest.mark.parametrize('status', ['no_input', 'not_provided'])
def test_embedded_track_cannot_be_discarded_as_no_audio(device_config, monkeypatch, status):  # noqa: F811
    device_config['speech_recognition']['enabled'] = False
    runner, item, layout, models, probed = speech_runner(device_config, monkeypatch, status, embedded=True)
    result = RetentionWorker(stage='stt').tick(runner, threading.Event())
    assert result['result']['status'] == 'failed', result
    saved = read_json(runner._receipt(layout, item, 'stt'))
    assert saved['status'] == 'failed'
    assert saved.get('outcome') != 'no_audio'
    assert probed == [Path(item['video_path'])]
    assert models.vision_calls == models.semantic_calls == 0
    assert RetentionWorker(stage='understanding').tick(runner, threading.Event())['status'] == 'waiting'


def test_audio_probe_error_is_not_a_no_audio_completion(device_config, monkeypatch):  # noqa: F811
    runner, item, layout, _, _ = speech_runner(device_config, monkeypatch, 'not_provided')

    def unreadable(_path):
        raise subprocess.CalledProcessError(1, ['ffprobe', 'owned-synthetic-input'])

    monkeypatch.setattr(speech, 'probe_audio', unreadable)
    result = RetentionWorker(stage='stt').tick(runner, threading.Event())
    assert result['result']['status'] == 'failed', result
    saved = read_json(runner._receipt(layout, item, 'stt'))
    assert saved['status'] == 'failed' and saved.get('outcome') != 'no_audio'


def test_late_recorder_audio_reopens_owned_stages_without_repeating_vision(device_config, monkeypatch):  # noqa: F811
    runner, item, layout, models, probed = speech_runner(device_config, monkeypatch, 'not_provided')
    stop = threading.Event()
    for stage in ('vision', 'stt', 'retention', 'understanding', 'report'):
        assert RetentionWorker(stage=stage).tick(runner, stop)['result']['status'] == 'completed'
    previous = {stage: read_json(runner._receipt(layout, item, stage))
                for stage in ('vision', 'stt', 'retention', 'understanding', 'report')}
    folder = Path(item['video_path']).parent
    (folder / 'audio.opus').write_bytes(b'owned-late-recorder-audio')
    atomic_json(folder / 'audio_meta.json', {'recording_session_id': 'fixture-1',
                'first_audio_global_us': item['recording_start_us']})
    for path in folder.glob('audio*'):
        os.utime(path, (time.time() - 1000, time.time() - 1000))
    pending, _ = item_and_layout(device_config)
    assert pending['audio']['status'] == 'pending_publication'
    observe(runner.runtime_root, {'recordings': [pending]})
    assert RetentionWorker(stage='stt').candidates(runner) == []
    assert read_json(runner._receipt(layout, item, 'stt')) == previous['stt']
    atomic_json(folder / 'audio_ready.json', {'recording_session_id': 'fixture-1', 'ready': True})
    os.utime(folder / 'audio_ready.json', (time.time() - 1000, time.time() - 1000))
    updated, _ = item_and_layout(device_config)
    assert updated['audio']['status'] == 'provided'
    assert updated['source_signature'] == item['source_signature']
    assert updated['audio']['source_signature'] != item['audio']['source_signature']
    observe(runner.runtime_root, {'recordings': [updated]})
    assert RetentionWorker(stage='vision').candidates(runner) == []
    assert len(probed) == 1
    calls = []

    def transcribe(_layout, retained, _key):
        # Deterministic late-audio model port. This is not real ASR evidence.
        assert retained['audio']['status'] == 'provided'
        assert any(source['kind'] == 'audio_audio' for source in retained['sources'])
        calls.append(True)
        return {'outcome': 'no_speech', 'comments': [], 'artifacts': [], 'model_invocation': 'NOT_PROVEN'}

    models.transcribe = transcribe
    for stage in ('stt', 'retention', 'understanding', 'report'):
        result = RetentionWorker(stage=stage).tick(runner, stop)
        assert result['result']['status'] == 'completed', result
        assert read_json(runner._receipt(layout, updated, stage))['key'] != previous[stage]['key']
    assert calls == [True] and models.vision_calls == 1 and models.semantic_calls == 2
    current_visual = read_json(runner._receipt(layout, updated, 'vision'))
    assert {key: current_visual[key] for key in ('key', 'segments', 'artifacts', 'input_binding')} == {
        key: previous['vision'][key] for key in ('key', 'segments', 'artifacts', 'input_binding')}
    histories = [read_json(path) for path in (layout.receipts / item['recording_id'] / 'history').glob('*.json')]
    assert previous['stt'] in histories
    archived = read_json(runner._receipt(layout, updated, 'retention'))
    source = next(source for source in archived['sources'] if source['kind'] == 'audio_audio')
    assert (layout.root / source['retained']['path']).read_bytes() == b'owned-late-recorder-audio'
    for stage in ('vision', 'stt', 'retention', 'understanding', 'report'):
        assert RetentionWorker(stage=stage).tick(runner, stop)['status'] == 'waiting'


@pytest.mark.parametrize('status', ['pending_publication', 'association_mismatch', 'unknown'])
def test_unready_audio_never_claims_or_probes(tmp_path, monkeypatch, status):
    runner = setup(tmp_path, monkeypatch, [record('blocked', 1_000_000, audio={'status': status})])
    runner.process = lambda *_a, **_kw: pytest.fail('Unready audio must not execute')
    assert RetentionWorker(stage='stt').tick(runner, threading.Event()) == {'status': 'waiting', 'admitted': 0}
    with runner.queues['stt'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 0


def prepared_report(config):
    config['device_day'].update(inplace_preprocessing=True, latest_first=True, camera_lanes=True,
                                backfill={'enabled': True, 'stages': ['report'], 'idle_seconds': 0})
    capture(config)
    item, layout = item_and_layout(config)
    models = FakeModels()
    runner = DeviceDayRunner(config, models)
    for stage in ('vision', 'stt', 'retention', 'understanding'):
        result = runner.run_once({'recordings': [item]}, stage=stage)['results'][0]
        assert result['status'] == 'completed', result
    observe(runner.runtime_root, {'recordings': [item]})
    return runner, item, layout, models


def report_worker(runner, item, monkeypatch):
    # Local discovery/queue shortlist is a hint only. admit() and process()
    # below still verify the actual publication, parent keys and file bytes.
    monkeypatch.setattr(device_day_backfill, 'live_demand',
                        lambda *_a: {'status': 'idle', 'blocked': False, 'reasons': []})
    monkeypatch.setattr(device_day_backfill, 'historical_candidates', lambda *_a, **_kw: [item])
    return device_day_backfill.BackfillWorker('report')


def test_historical_report_has_consumer_and_keeps_parent_execution(device_config, monkeypatch):  # noqa: F811
    runner, item, layout, models = prepared_report(device_config)
    worker = report_worker(runner, item, monkeypatch)
    before = {stage: runner._receipt(layout, item, stage).read_bytes() for stage in worker.admitter.parents()}
    result = worker.tick(runner, threading.Event())
    assert result['result']['status'] == 'completed', result
    assert (layout.reports / 'LaboratoryDailyReport.html').is_file()
    assert {stage: runner._receipt(layout, item, stage).read_bytes() for stage in before} == before
    assert models.vision_calls == models.semantic_calls == 1


@pytest.mark.parametrize('damage', ['publication', 'parent_key', 'artifact'])
def test_historical_report_cannot_bypass_invalid_parent(device_config, monkeypatch, damage):  # noqa: F811
    runner, item, layout, models = prepared_report(device_config)
    worker = report_worker(runner, item, monkeypatch)
    if damage == 'publication':
        atomic_json(runner._receipt(layout, item, 'publication'), {'status': 'publishing'})
    else:
        path = runner._receipt(layout, item, 'vision')
        saved = read_json(path)
        if damage == 'parent_key':
            atomic_json(path, saved | {'key': 'unknown-execution-identity'})
        else:
            (layout.root / saved['artifacts'][0]['path']).write_bytes(b'changed-owned-test-frame')
    result = worker.tick(runner, threading.Event())
    assert result['status'] == 'waiting', result
    assert models.vision_calls == models.semantic_calls == 1
    with runner.queues['report'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM recordings').fetchone()[0] == 0


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt', 'understanding', 'report'])
@pytest.mark.parametrize('mode', ['idle', 'fair'])
def test_fair_lane_is_polled_before_live_work_and_covers_primary_history(tmp_path, monkeypatch, stage, mode):
    calls, scopes = [], []
    stop = threading.Event()
    settings = {'enabled': True, 'inplace_preprocessing': True, 'latest_first': True, 'camera_lanes': True,
                'backfill': {'enabled': True, 'mode': mode}}
    config = {'device_day': settings, 'storage': {'local_runtime_root': str(tmp_path)}}
    runner = SimpleNamespace(config=config, settings=settings, runtime_root=tmp_path / 'device-day')
    monkeypatch.setattr('visioncortex.config.load_config', lambda *_: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda config: config)
    backends, creations = [], []
    backend = object()

    def create_backend(current_config):
        assert current_config is config
        backends.append(backend)
        return backend

    def create_runner(current_config, **kwargs):
        assert current_config is config
        creations.append(kwargs)
        return runner

    monkeypatch.setattr('visioncortex.device_day_fair_backend.create_backend', create_backend)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', create_runner)

    class Supervisor:
        def __init__(self, current_stage, current_stop):
            assert current_stage == stage and current_stop is stop
        def request_yield(self):
            pass
        def poll(self, current_runner, *, allow_start):
            assert current_runner is runner
            calls.append(('history', allow_start))
            # A completed unit releases this process to the next live tick.
            return {'status': 'processed' if allow_start else 'waiting_for_live'}
        def close(self):
            pass

    class Status:
        def __init__(self, config, stage, *, recent_seconds, backfill_enabled):
            scopes.append((recent_seconds, backfill_enabled))
        def __enter__(self):
            return self
        def update(self, **_kw):
            pass
        def __exit__(self, *_):
            pass

    def tick(worker, current_runner, current_stop):
        assert current_runner is runner
        calls.append(('live', worker.recent_seconds))
        current_stop.set()
        return {'status': 'processed', 'admitted': 1}

    monkeypatch.setattr(device_day_backfill, 'BackfillSupervisor', Supervisor)
    monkeypatch.setattr('visioncortex.device_day_consumers.WorkerStatus', Status)
    monkeypatch.setattr(RetentionWorker, 'tick', tick)
    device_day_retention_worker.serve('owned-test-config.yaml', stop, stage=stage)
    recent = 14400 if mode == 'fair' or stage in {'vision', 'understanding', 'report'} else None
    assert scopes == [(recent, True)]
    assert calls == ([('history', True), ('live', recent)] if mode == 'fair' else
                     [('live', recent), ('history', False)])
    assert backends == ([backend] if mode == 'fair' and stage == 'vision' else [])
    assert creations == ([{'backend': backend}] if mode == 'fair' and stage == 'vision' else [{}])


def test_active_fair_history_keeps_live_admission_running(tmp_path, monkeypatch):
    stop, calls, scopes = threading.Event(), [], []
    settings = {'enabled': True, 'inplace_preprocessing': True, 'latest_first': True, 'camera_lanes': True,
                'backfill': {'enabled': True, 'mode': 'fair'}}
    config = {'device_day': settings, 'storage': {'local_runtime_root': str(tmp_path)}}
    runner = SimpleNamespace(config=config, settings=settings, runtime_root=tmp_path / 'device-day')
    monkeypatch.setattr('visioncortex.config.load_config', lambda *_: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda config: config)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', lambda *_a, **_kw: runner)
    monkeypatch.setattr('visioncortex.device_day_fair_backend.create_backend', lambda *_: object())
    monkeypatch.setattr(stop, 'wait', lambda _delay: stop.is_set())

    class Supervisor:
        def __init__(self, *_):
            pass
        def request_yield(self):
            pass
        def poll(self, _runner, *, allow_start):
            assert allow_start
            calls.append('history')
            return {'status': 'running'}
        def close(self):
            pass

    class Status:
        def __init__(self, *_a, **_kw):
            pass
        def __enter__(self):
            return self
        def update(self, **fields):
            if fields.get('scope'):
                scopes.append(fields['scope'])
        def __exit__(self, *_):
            pass

    def tick(_worker, _runner, current_stop):
        calls.append('live')
        current_stop.set()
        return {'status': 'processed', 'admitted': 1}

    monkeypatch.setattr(device_day_backfill, 'BackfillSupervisor', Supervisor)
    monkeypatch.setattr('visioncortex.device_day_consumers.WorkerStatus', Status)
    monkeypatch.setattr(RetentionWorker, 'tick', tick)
    device_day_retention_worker.serve('owned-test-config.yaml', stop, stage='vision')
    assert calls == ['history', 'live']
    assert 'history' in scopes

from types import SimpleNamespace

from test_device_day import device_config  # noqa: F401
from test_device_day_inplace import setup  # noqa: F401
from visioncortex.device_day import exclusive
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_publication_worker import PublicationReplay
from visioncortex.publication_journal import PublicationJournal, reconcile


def test_standard_replay_recovers_publication_without_model_execution(setup, monkeypatch):  # noqa: F811
    _, record, layout, backend, runner = setup
    assert runner.process(record, stage='vision')['status'] == 'completed'
    assert runner.process(record, stage='stt')['status'] == 'completed'
    original = runner.refresh_index
    monkeypatch.setattr(runner, 'refresh_index', lambda *a, **k: (_ for _ in ()).throw(OSError('offline')))
    result = runner.process(record, stage='retention')
    assert result['status'] == 'completed' and result['formal_publication'] == 'pending'
    monkeypatch.setattr(runner, 'refresh_index', original)
    monkeypatch.setattr(runner, '_backend', lambda: (_ for _ in ()).throw(AssertionError('model startup')))
    assert PublicationReplay().tick(runner)['published'] == 1
    assert read_json(layout.index)['recordings']
    assert not PublicationJournal(runner.runtime_root).pending()
    assert backend.vision_calls == 1


def test_malformed_record_does_not_block_later_publication(tmp_path):
    journal = PublicationJournal(tmp_path)
    bad = journal.begin({'recording_id': 'bad'})
    good = journal.begin({'recording_id': 'good', 'camera_key': 'owned_cam01',
                          'recording_start_us': 1789005600000000})
    def layout(record):
        if record['recording_id'] == 'bad':
            raise KeyError('recording')
        return 'owned-layout'
    runner = SimpleNamespace(runtime_root=tmp_path, backend_root=tmp_path, settings={}, layout=layout,
                             refresh_index=lambda *a, **k: True)
    assert reconcile(runner) == 1
    assert journal.pending() == [(bad, {'recording_id': 'bad'})]
    assert good != bad


def test_not_yet_publishable_rows_rotate_behind_other_pending_work(tmp_path):
    journal = PublicationJournal(tmp_path)
    clock = {'camera_key': 'owned_cam01', 'recording_start_us': 1789005600000000}
    first = journal.begin(clock | {'recording_id': 'first'})
    second = journal.begin(clock | {'recording_id': 'second'})
    runner = SimpleNamespace(runtime_root=tmp_path, backend_root=tmp_path, settings={}, layout=lambda record: 'layout',
                             refresh_index=lambda *a, **k: False)
    # Observe the bounded first round directly to prove its deferred item moves.
    original = journal.pending
    from unittest.mock import patch
    with patch.object(PublicationJournal, 'pending', lambda *a, **k: original(1)):
        assert reconcile(runner) == 0
    assert [token for token, _ in journal.pending()] == [second, first]


def test_global_replay_lock_prevents_duplicate_worker(tmp_path):
    runner = SimpleNamespace(runtime_root=tmp_path)
    with exclusive(tmp_path / 'locks' / 'publication-replay.lock'):
        assert PublicationReplay().tick(runner)['status'] == 'running_elsewhere'


def test_signal_finishes_current_standard_round_and_stops_before_next(tmp_path, monkeypatch):
    import threading
    from visioncortex import device_day_publication_worker as module
    stop = threading.Event()
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    runner = SimpleNamespace(runtime_root=tmp_path)
    monkeypatch.setattr('visioncortex.config.load_config', lambda path: config)
    monkeypatch.setattr('visioncortex.ai_settings.apply_active', lambda value: value)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', lambda value: runner)
    calls = []
    def tick(self, current):
        calls.append(current)
        stop.set()
        return {'status': 'completed', 'published': 2}
    monkeypatch.setattr(module.PublicationReplay, 'tick', tick)
    module.serve('owned-config.yaml', stop)
    assert calls == [runner]
    assert read_json(tmp_path / 'PublicationReplay/Service.json')['status'] == 'stopped'

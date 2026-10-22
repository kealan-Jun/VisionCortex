"""Retention policy upgrades invalidate persistent readable publication hints."""
from types import SimpleNamespace

import pytest

from visioncortex.device_day_publication_worker import ReadableReplay


@pytest.mark.parametrize('version', ['POLICY', 'EXPORT_POLICY'])
def test_publication_policy_change_invalidates_persisted_input_hint(tmp_path, monkeypatch, version):
    from visioncortex import device_day_publication_worker as worker
    from visioncortex.media import retained_intervals
    monkeypatch.setattr(worker, 'local_change_hints', lambda *args: {'day': 'same-source-hint'})
    runner = SimpleNamespace(runtime_root=tmp_path)
    old = ReadableReplay._local_hints(runner, [], {'day'})
    assert old != {'day': 'same-source-hint'}
    assert ReadableReplay._local_hints(runner, [], {'day'}) == old
    monkeypatch.setattr(retained_intervals, version, 'explicit-new-policy-version')
    assert ReadableReplay._local_hints(runner, [], {'day'}) != old


def test_saved_completed_day_is_revisited_after_policy_upgrade_latest_first(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from threading import Event
    from visioncortex.device_day_contract import archive_name, atomic_json
    from visioncortex.device_day_publication_worker import local_change_hints
    from visioncortex.observed_inventory import observe
    from visioncortex.timeline_invalidation import TimelineInvalidations
    from visioncortex.media import retained_intervals

    records = [{'recording_id': str(i), 'camera_key': f'camera{i}_cam01',
                'recording_start_us': 1800000000000000-i*86400000000} for i in range(2)]
    observe(tmp_path, {'recordings': records})
    names = [archive_name(r['camera_key'], r['recording_start_us']) for r in records]
    for name in names:
        TimelineInvalidations(tmp_path).changed(name, 'unchanged-model-result')
    config = {'storage': {'local_runtime_root': str(tmp_path.parent)},
              'runtime': {'local_only': True}, 'device_day': {'latest_first': True}}
    runner = SimpleNamespace(runtime_root=tmp_path, config=config)
    saved = {'checked': {names[0]: 999},
             'published_hints': local_change_hints(tmp_path, records, names)}
    state = tmp_path/'ReadablePublication/Service.json'
    atomic_json(state, saved)
    calls = []
    class Publisher:
        def __init__(self, config):
            pass
        def tick(self, *, archives):
            calls.extend(archives)
            return {'status': 'completed', 'published': 1}
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', Publisher)
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())
    worker = ReadableReplay(Event())
    assert worker.tick(runner)['archive'] == names[0]
    assert worker.published_hints[names[0]] != saved['published_hints'][names[0]]
    atomic_json(state, {'checked': worker.checked, 'published_hints': worker.published_hints})
    monkeypatch.setattr(retained_intervals, 'POLICY', 'next-reviewed-retention-policy')
    assert ReadableReplay(Event()).tick(runner)['archive'] == names[0]
    assert calls == [names[0], names[0]]


def test_naming_migration_invalidates_only_source_group_hints(tmp_path, monkeypatch):
    from visioncortex import device_day_publication_worker as worker
    monkeypatch.setattr(worker, 'local_change_hints', lambda *args: {'day': 'unchanged-model-receipts'})
    grouped = SimpleNamespace(runtime_root=tmp_path,
                              config={'device_day': {'clip_delivery': 'source_recording_v2'}})
    legacy = SimpleNamespace(runtime_root=tmp_path, config={})
    old_grouped = ReadableReplay._local_hints(grouped, [], {'day'})
    old_legacy = ReadableReplay._local_hints(legacy, [], {'day'})
    monkeypatch.setattr(worker, 'READABLE_LAYOUT_POLICY', 'next-reviewed-naming-recipe')
    assert ReadableReplay._local_hints(grouped, [], {'day'}) != old_grouped
    assert ReadableReplay._local_hints(legacy, [], {'day'}) == old_legacy


def test_continuously_changing_latest_day_cannot_starve_completed_history(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from threading import Event
    from visioncortex.device_day_contract import archive_name
    from visioncortex.observed_inventory import observe
    from visioncortex import device_day_publication_worker as module
    records = [{'recording_id': str(i), 'camera_key': f'camera{i}_cam01',
                'recording_start_us': 1800000000000000-i*86400000000} for i in range(2)]
    observe(tmp_path, {'recordings': records})
    names = [archive_name(r['camera_key'], r['recording_start_us']) for r in records]
    runner = SimpleNamespace(runtime_root=tmp_path, config={
        'storage': {'local_runtime_root': str(tmp_path.parent)},
        'runtime': {'local_only': True}, 'device_day': {'latest_first': True}})
    calls = []
    class Publisher:
        def __init__(self, config):
            pass
        def tick(self, *, archives):
            calls.extend(archives)
            return {'status': 'completed', 'published': 1}
    monkeypatch.setattr('visioncortex.device_day_file_index.FileIndexPublisher', Publisher)
    monkeypatch.setattr('visioncortex.device_day_recovery_worker.verify_storage', lambda config: None)
    monkeypatch.setattr('visioncortex.device_day_io.slot', lambda config: nullcontext())
    monkeypatch.setattr('visioncortex.runtime_control.resource_slot', lambda *a, **k: nullcontext())
    monkeypatch.setattr(module, 'local_change_hints',
                        lambda *args: {names[0]: str(len(calls)), names[1]: 'completed-old-day'})
    replay = ReadableReplay(Event())
    assert replay.tick(runner)['archive'] == names[0]
    assert replay.tick(runner)['archive'] == names[0]
    third = replay.tick(runner)
    assert third['archive'] == names[1]
    assert third['scheduling_lane'] == 'historical_fair_share'


def test_standard_replay_has_a_resumable_resource_scope(tmp_path, monkeypatch):
    from threading import Event
    from visioncortex.device_day_publication_worker import PublicationReplay
    from visioncortex.runtime_control import CURRENT
    from visioncortex.publication_journal import PublicationJournal
    stop = Event()
    runner = SimpleNamespace(runtime_root=tmp_path, config={}, settings={})
    seen = []
    def reconcile(current, *, stop, limit):
        context = CURRENT.get()
        seen.append(context)
        assert current is runner and limit == 4
        assert context.job_id == 'publication-replay'
        assert context.yield_signal is stop and context.stop is stop
        return 0
    monkeypatch.setattr('visioncortex.publication_journal.reconcile', reconcile)
    monkeypatch.setattr(PublicationJournal, 'pending', lambda *a, **kw: [])
    assert PublicationReplay(stop=stop).tick(runner)['status'] == 'completed'
    assert len(seen) == 1 and CURRENT.get().job_id != 'publication-replay'

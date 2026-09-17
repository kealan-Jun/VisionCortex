from copy import deepcopy
import threading
import time

import pytest

from test_device_day import device_config as _device_config, capture, item_and_layout, FakeModels
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import STAGES
from visioncortex.device_day_schedule import in_processing_scope, processing_cutoff
from visioncortex.publication_journal import PublicationJournal


@pytest.fixture
def device_config(default_config, tmp_path):
    return _device_config.__wrapped__(default_config, tmp_path)


@pytest.mark.parametrize('start,end,allowed', [(10, 99, False), (10, 100, True), (100, 120, True), (110, 120, True), (0, 0, False)])
def test_fixed_capture_cutoff_includes_active_slice_not_late_upload(start, end, allowed):
    record = {'recording_start_us': start, 'recording_end_us': end, 'updated_at': '2099-01-01'}
    assert in_processing_scope({'process_since_us': 100}, record) is allowed
    assert in_processing_scope({}, record)


@pytest.mark.parametrize('value', [True, -1, 0, 1.5, 'now'])
def test_invalid_cutoff_cannot_silently_enable_backfill(value):
    with pytest.raises(ValueError):
        processing_cutoff({'process_since_us': value})


@pytest.mark.parametrize('stage', STAGES)
def test_old_durable_queue_is_preserved_but_cannot_be_admitted(device_config, monkeypatch, stage):
    capture(device_config)
    record, _ = item_and_layout(device_config)
    device_config['device_day']['process_since_us'] = record['recording_end_us'] + 1
    runner = DeviceDayRunner(device_config, FakeModels())
    queue = runner.queues[stage]
    queue.enqueue(record, 'original-revision')
    monkeypatch.setattr(runner, 'layout', lambda *a: pytest.fail('Historical NAS must not be inspected'))
    assert runner._prepare_stage({'recordings': [record]}, stage, None) == set()
    assert runner.run_once({'recordings': []}, stage=stage)['results'] == []
    with queue.connect() as db:
        row = db.execute('SELECT status,revision,attempts FROM recordings').fetchone()
        assert tuple(row) == ('queued', 'original-revision', 0)


def test_cutover_slice_is_automatically_eligible_without_rekeying_media(device_config):
    capture(device_config)
    record, _ = item_and_layout(device_config)
    before = DeviceDayRunner(device_config, FakeModels())
    expected = before._key('retention', record, record)
    config = deepcopy(device_config)
    config['device_day']['process_since_us'] = record['recording_start_us'] + 1
    runner = DeviceDayRunner(config, FakeModels())
    assert runner._key('retention', record, record) == expected
    assert runner._prepare_stage({'recordings': [record]}, 'retention', None) == {record['recording_id']}


def test_publication_cutoff_does_not_starve_new_work_behind_old_entries(tmp_path):
    journal = PublicationJournal(tmp_path)
    for index in range(40):
        journal.begin({'recording_id': str(index), 'recording_end_us': 99})
    journal.begin({'recording_id': 'live', 'recording_start_us': 90, 'recording_end_us': 101})
    assert [r['recording_id'] for _, r in journal.pending(1, since_us=100)] == ['live']
    assert len(journal.pending(100)) == 41


def test_background_pipeline_processes_new_capture_without_replaying_old_request(device_config, monkeypatch):
    from visioncortex import device_day_service
    from visioncortex.device_day_contract import atomic_json, read_json
    from visioncortex.nas_recordings import scan_recordings
    start = 1789005600000000
    capture(device_config, start=start, duration=10, label='old')
    capture(device_config, start=start + 20000000, duration=10, label='new')
    inventory = scan_recordings(device_config)
    old, new = sorted(inventory['recordings'], key=lambda r: r['recording_start_us'])
    device_config['device_day']['process_since_us'] = start + 15000000
    device_config['device_day']['night_processing'] = {'enabled': False}
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    monkeypatch.setattr(device_day_service, 'DeviceDayRunner', lambda _: runner)
    monkeypatch.setattr('visioncortex.device_day_overview.ArchiveOverview.publish', lambda *a: {})
    service = device_day_service.DeviceDayService(lambda: device_config, threading.Lock())
    monkeypatch.setattr(service, '_storage_available', lambda *a: {'ready': True})
    request = runner.runtime_root / 'requests' / 'old.json'
    atomic_json(request, {'status': 'queued', 'recordings': [old]})
    for queue in runner.queues.values():
        queue.enqueue(old, 'preserved-history')
    service.observe(device_config, inventory)
    service.start()
    try:
        deadline = time.monotonic() + 15
        while True:
            states = []
            for queue in runner.queues.values():
                with queue.connect() as db:
                    row = db.execute('SELECT status FROM recordings WHERE recording_id=?', (new['recording_id'],)).fetchone()
                    states.append(row[0] if row else None)
            if states == ['completed'] * len(STAGES):
                break
            assert time.monotonic() < deadline, states
            time.sleep(.05)
    finally:
        service.stop()
    assert read_json(request)['status'] == 'queued'
    assert backend.vision_calls == backend.semantic_calls == 1
    assert new['recording_id'] in {r['recording_id'] for r in read_json(runner.layout(new).index)['recordings']}
    for queue in runner.queues.values():
        with queue.connect() as db:
            assert db.execute('SELECT status,attempts FROM recordings WHERE recording_id=?',
                              (old['recording_id'],)).fetchone()['status'] == 'queued'

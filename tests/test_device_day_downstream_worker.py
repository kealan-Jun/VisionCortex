"""Latest downstream lanes preserve standard receipts and incremental reports."""
import threading

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from test_device_day_retention_worker import record, setup
from visioncortex.device_day import DeviceDayRunner, load_context
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.observed_inventory import observe


def prepared(config):
    config['device_day'].update(inplace_preprocessing=True, latest_first=True)
    capture(config)
    item, layout = item_and_layout(config)
    models = FakeModels()
    runner = DeviceDayRunner(config, models)
    for stage in ('vision', 'stt', 'retention'):
        result = runner.run_once({'recordings': [item]}, stage=stage)['results'][0]
        assert result['status'] == 'completed', result
    observe(runner.runtime_root, {'recordings': [item]})
    assert read_json(runner._receipt(layout, item, 'publication'))['status'] == 'completed'
    return runner, item, layout, models


def test_real_receipts_feed_understanding_then_incremental_report_without_parent_rerun(device_config):  # noqa: F811
    runner, item, layout, models = prepared(device_config)
    # Another not-yet-preprocessed recording must not delay this partial day.
    observe(runner.runtime_root, {'recordings': [record('later-open', item['recording_start_us'] + 1_000_000)]})
    stop = threading.Event()
    worker = RetentionWorker(stage='understanding')
    result = worker.tick(runner, stop)
    assert result['result']['status'] == 'completed', result
    receipts = {stage: read_json(runner._receipt(layout, item, stage)) for stage in ('vision', 'stt')}
    context = load_context(layout, item)
    expected = runner._key('understanding', item | {'archive_date': layout.name[:10]}, [receipts, context])
    with runner.queues['understanding'].connect() as db:
        assert db.execute('SELECT revision FROM recordings WHERE recording_id=?',
                          (item['recording_id'],)).fetchone()[0] == expected
    report = RetentionWorker(stage='report').tick(runner, stop)
    assert report['result']['status'] == 'completed', report
    assert (layout.reports / 'LaboratoryDailyReport.html').is_file()
    assert read_json(layout.index)['understandings']
    assert worker.tick(runner, stop)['status'] == 'waiting'
    assert RetentionWorker(stage='report').tick(runner, stop)['status'] == 'waiting'
    assert models.vision_calls == models.semantic_calls == 1


@pytest.mark.parametrize('damage', ['publication', 'artifact', 'parent_key'])
def test_downstream_rejects_unpublished_or_invalid_parent_without_queue_or_model(device_config, damage):  # noqa: F811
    runner, item, layout, models = prepared(device_config)
    from visioncortex.device_day_contract import atomic_json, safe_child
    if damage == 'publication':
        atomic_json(runner._receipt(layout, item, 'publication'), {'status': 'publishing'})
    else:
        path = runner._receipt(layout, item, 'vision')
        saved = read_json(path)
        if damage == 'artifact':
            safe_child(layout.root, saved['artifacts'][0]['path']).write_bytes(b'changed-artifact')
        else:
            atomic_json(path, saved | {'key': 'different-model-or-source'})
    result = RetentionWorker(stage='understanding').tick(runner, threading.Event())
    assert result['status'] == 'waiting', result
    with runner.queues['understanding'].connect() as db:
        assert db.execute('SELECT count(*) FROM recordings').fetchone()[0] == 0
    assert models.semantic_calls == 0


@pytest.mark.parametrize('stage', ['understanding', 'report'])
def test_local_parent_prefilter_and_latest_order_ignore_unready_newer_records(tmp_path, monkeypatch, stage):
    rows = [record('ready-old', 1_000_000), record('ready-new', 2_000_000), record('unready', 3_000_000)]
    runner = setup(tmp_path, monkeypatch, rows)
    runner.config['storage'] = {'local_runtime_root': str(runner.runtime_root)}
    for name in ('understanding', 'report'):
        runner.queues[name] = DeviceDayQueue(runner.runtime_root / f'queue-{name}.sqlite3', latest_first=True)
    worker = RetentionWorker(stage=stage)
    for parent in worker.parents():
        for row in rows[:2]:
            queue = runner.queues[parent]
            queue.enqueue(row, row['source_signature'])
            assert queue.claim(row['recording_id'], allowed={row['recording_id']})
            queue.finish(row['recording_id'], row['recording_id'], {'status': 'completed'}, 1)
    assert [r['recording_id'] for r in worker.candidates(runner)] == ['ready-new']
    calls = []
    def admit(runner, row):
        calls.append(row['recording_id'])
        runner.queues[stage].enqueue(row, row['source_signature'])
        return True
    monkeypatch.setattr(worker, 'admit', admit)
    assert worker.tick(runner, threading.Event())['recording_id'] == 'ready-new'
    assert worker.tick(runner, threading.Event())['recording_id'] == 'ready-old'
    assert calls == ['ready-new', 'ready-old']


@pytest.mark.parametrize('stage', ['understanding', 'report'])
def test_downstream_pause_schedule_and_cli_keep_stage_independence(tmp_path, monkeypatch, stage):
    from visioncortex import device_day_retention_worker as module
    runner = setup(tmp_path, monkeypatch, [])
    runner.config['storage'] = {'local_runtime_root': str(runner.runtime_root)}
    runner.config['device_day'] = {'paused_stages': [stage]}
    assert RetentionWorker(stage=stage).tick(runner, threading.Event())['status'] == 'paused_by_user'
    runner.config['device_day'] = {}
    monkeypatch.setattr('visioncortex.device_day_night_schedule.stage_admitted', lambda *_: False)
    assert RetentionWorker(stage=stage).tick(runner, threading.Event())['status'] == 'waiting_for_night_window'
    calls = []
    monkeypatch.setattr(module, 'serve', lambda config, stop, stage: calls.append(stage))
    module.main(['--config', 'owned-fixture.yaml', '--stage', stage])
    assert calls == [stage]


@pytest.mark.parametrize('stage,capacity,process_new', [
    ('understanding', 4, True), ('understanding', 1, False), ('report', 4, False),
])
def test_latest_understanding_can_pass_one_main_slice_but_not_two(tmp_path, monkeypatch, stage, capacity, process_new):
    busy, latest = record('busy', 1_000_000), record('latest', 2_000_000)
    runner = setup(tmp_path, monkeypatch, [busy, latest])
    runner.config.update(storage={'local_runtime_root': str(runner.runtime_root)},
                         runtime={'resource_limits': {'cloud': capacity}})
    for name in ('understanding', 'report'):
        runner.queues[name] = DeviceDayQueue(runner.runtime_root / f'queue-{name}.sqlite3', latest_first=True)
    worker = RetentionWorker(stage=stage)
    def complete_parents(row):
        for parent in worker.parents():
            queue = runner.queues[parent]
            queue.enqueue(row, row['source_signature'])
            assert queue.claim('parent', allowed={row['recording_id']})
            queue.finish('parent', row['recording_id'], {'status': 'completed'}, 1)
    for row in (busy, latest):
        complete_parents(row)
    queue = runner.queues[stage]
    queue.enqueue(busy, busy['source_signature'])
    assert queue.claim('main', allowed={'busy'})
    with queue.connect() as db:
        original = dict(db.execute("SELECT * FROM recordings WHERE recording_id='busy'").fetchone())
    def admit(runner, row):
        runner.queues[stage].enqueue(row, row['source_signature'])
        return True
    monkeypatch.setattr(worker, 'admit', admit)
    calls = []
    def process(row, **kwargs):
        # The extra camera lease never substitutes for provider-wide admission.
        from visioncortex.provider_control import provider_request
        from visioncortex.runtime_control import ResourceCoordinator
        with provider_request(runner.config, 'understanding', {}):
            active = ResourceCoordinator(runner.runtime_root / 'state/resources.sqlite3').snapshot()
            assert sum(r['units'] for r in active if r['resource'] == 'cloud' and r['state'] == 'running') <= capacity
            calls.append(row['recording_id'])
        return {'status': 'completed'}
    runner.process = process
    result = worker.tick(runner, threading.Event())
    assert calls == (['latest'] if process_new else [])
    assert result['status'] == ('processed' if process_new else 'waiting')
    with queue.connect() as db:
        assert original == dict(db.execute("SELECT * FROM recordings WHERE recording_id='busy'").fetchone())
    if process_new:
        second, third = record('second', 3_000_000), record('third', 4_000_000)
        queue.enqueue(second, second['source_signature'])
        assert queue.claim('another-owner', allowed={'second'}, camera_serial=True, camera_limit=2)
        complete_parents(third)
        observe(runner.runtime_root, {'recordings': [third]})
        assert worker.tick(runner, threading.Event())['status'] == 'waiting'
        assert calls == ['latest']

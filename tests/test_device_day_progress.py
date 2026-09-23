import json
import time

from visioncortex.device_day_progress import snapshot
from visioncortex.device_day_queue import DeviceDayQueue


def test_progress_distinguishes_missing_stage_expired_lease_and_running(tmp_path):
    root = tmp_path / 'device-day'
    root.mkdir()
    rows = [{'recording_id': str(n), 'camera_key': f'camera{n}', 'configured_role': 'first_person',
             'recording_start_us': 1787792400000000, 'recording_end_us': 1787792410000000} for n in range(3)]
    (root/'observed-inventory.json').write_text(json.dumps({'recordings':rows}))
    q = DeviceDayQueue(root/'queue-vision.sqlite3')
    for row in rows[:2]:
        q.enqueue(row, 'v1')
    q.claim('expired')
    q.claim('live')
    with q.connect() as db:
        db.execute("update recordings set lease_until=? where lease_owner='expired'", (time.time()-1,))
    data = snapshot({'storage':{'local_runtime_root':str(tmp_path)}})
    assert len(data['running']) == 1
    assert data['running'][0]['recording_id'] == '1'
    day = next(iter(data['days'].values()))
    assert day['total'] == 3
    assert day['stages']['vision'] == {'completed':0,'queued':0,'running':1,'failed':0,'expired':1,'not_enqueued':1}
    assert day['stages']['retention']['not_enqueued'] == 3
    assert not (root/'queue-retention.sqlite3').exists()


def test_uninitialized_progress_does_not_create_runtime(tmp_path):
    config = {'storage':{'local_runtime_root':str(tmp_path/'absent')}}
    assert snapshot(config)['days'] == {}
    assert not (tmp_path/'absent').exists()


def test_prerequisite_wait_is_visible_without_live_lease_or_model_failure(tmp_path):
    root = tmp_path / 'device-day'
    record = {'recording_id': 'blocked', 'camera_key': 'camera', 'configured_role': 'first_person',
              'recording_start_us': 1787792400000000, 'recording_end_us': 1787792410000000}
    queue = DeviceDayQueue(root/'queue-vision.sqlite3')
    queue.enqueue(record, 'v1')
    queue.wait_for_prerequisite('blocked', 'v1', 'retention')
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    value = snapshot(config)
    day = next(iter(value['days']))
    counts = value['days'][day]['stages']['vision']
    assert counts['waiting_for_prerequisite'] == 1
    assert counts['failed'] == counts['running'] == counts['queued'] == 0
    assert value['running'] == []
    assert value['waiting'][day]['vision'] == {'prerequisite_not_verified': 1}
    config['device_day'] = {'paused_stages': ['vision']}
    assert snapshot(config)['waiting'][day]['vision'] == {'paused_by_user': 1}


def test_waiting_counts_distinguish_failed_upstream_from_compute_backlog(tmp_path):
    root = tmp_path / 'device-day'
    rows = [{'recording_id': str(n), 'camera_key': f'camera{n}', 'configured_role': 'first_person',
             'recording_start_us': 1787792400000000, 'recording_end_us': 1787792410000000} for n in range(3)]
    retention = DeviceDayQueue(root/'queue-retention.sqlite3')
    vision = DeviceDayQueue(root/'queue-vision.sqlite3')
    for row in rows:
        retention.enqueue(row, 'r1')
        vision.enqueue(row, 'v1')
    retention.claim('r0')
    retention.finish('r0','0',{'status':'failed','error_type':'FileNotFoundError'},1)
    retention.claim('r1')
    retention.finish('r1','1',{'status':'completed'},1)
    result = snapshot({'storage':{'local_runtime_root':str(tmp_path)}})
    waiting = next(iter(result['waiting'].values()))['vision']
    assert waiting == {'pending_validation':1,'upstream_pending':1}
    assert next(iter(result['days'].values()))['stages']['vision']['queued'] == sum(waiting.values())
    assert next(iter(result['days'].values()))['stages']['vision']['input_missing'] == 1


def test_progress_polls_share_work_and_timeout_does_not_cancel_it():
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from visioncortex.device_day_progress import ProgressPoller
    started, release = Event(), Event()
    calls = []
    def reader(config):
        calls.append(config)
        started.set()
        assert release.wait(3)
        return {'observed_at': 123, 'days': {}}
    async def run(executor):
        poller = ProgressPoller(lambda: {'local': True}, executor=executor, reader=reader)
        first = asyncio.create_task(poller.read(timeout=0.02))
        others = [asyncio.create_task(poller.read()) for _ in range(12)]
        try:
            await first
            assert False, 'slow read must time out'
        except asyncio.TimeoutError:
            pass
        assert started.is_set()
        assert len(calls) == 1
        release.set()
        results = await asyncio.gather(*others)
        assert all(r['observed_at'] == 123 and r['days'] == {} for r in results)
        assert (await poller.read())['observed_at'] == results[0]['observed_at']
        assert len(calls) == 1
    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_published_progress_refreshes_first_poll_even_when_settings_are_busy(tmp_path):
    import asyncio
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.device_day_progress import ProgressPoller

    path = tmp_path / 'device-day/ProgressSnapshot.json'
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    busy, release = Event(), Event()
    calls = []

    def settings():
        calls.append(True)
        if len(calls) > 1:
            busy.set()
            assert release.wait(5)
        return config

    async def run(executor):
        poller = ProgressPoller(settings, executor=executor)
        now = time.time()
        atomic_json(path, {'observed_at': now - 100, 'days': {}})
        assert (await poller.read())['snapshot_stale'] is True
        atomic_json(path, {'observed_at': now, 'days': {'new': {} }})
        poller.started = 0
        try:
            # No extra browser poll is needed to see what the publisher wrote.
            assert (await poller.read())['observed_at'] == now
            assert await asyncio.to_thread(busy.wait, 1)
            atomic_json(path, {'observed_at': now + .001, 'days': {'newer': {}}})
            newest = await poller.read()
            assert newest['days'] == {'newer': {}}
            assert newest['snapshot_stale'] is False
            # A late result cannot replace a newer observed state.
            poller._remember({'observed_at': now - 100, 'days': {'old': {}}})
            assert poller.last_good['days'] == {'newer': {}}
        finally:
            release.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


def test_failure_categories_and_component_percentiles_are_visible(tmp_path):
    root = tmp_path / 'device-day'
    q = DeviceDayQueue(root / 'queue-vision.sqlite3')
    for n in range(3):
        row = {'recording_id': str(n), 'camera_key': 'cam', 'recording_start_us': 1787792400000000,
               'configured_role': 'first_person'}
        q.enqueue(row, 'v1')
        q.claim(str(n))
        q.finish(str(n), str(n), {'status': 'completed', 'component_timings': {'coarse_scan_seconds': n + 1}}
                 if n < 2 else {'status': 'failed', 'error_type': 'FileNotFoundError'}, 1)
    data = snapshot({'storage': {'local_runtime_root': str(tmp_path)}})
    day = next(iter(data['days']))
    assert data['failure_categories'][day]['vision']['FileNotFoundError'] == 1
    assert data['vision_component_timings'][day]['coarse_scan_seconds'] == {
        'samples': 2, 'median_seconds': 1.5, 'p95_seconds': 2}
    assert '不能据此认定已删除' in data['latency_html']
    assert '粗扫' in data['latency_html']


def test_missing_inputs_excluded_from_processing_denominator(tmp_path):
    q = DeviceDayQueue(tmp_path / 'device-day/queue-retention.sqlite3')
    for n in range(2):
        row = {'recording_id': str(n), 'camera_key': 'cam', 'recording_start_us': 1787792400000000,
               'configured_role': 'first_person'}
        q.enqueue(row, 'v1')
        q.claim(str(n))
        q.finish(str(n), str(n), {'status': 'failed', 'error_type': 'FileNotFoundError'}
                 if n == 0 else {'status': 'completed'}, 1)
    d = next(iter(snapshot({'storage': {'local_runtime_root': str(tmp_path)}})['days'].values()))
    assert d['total'] == 2
    assert d['processing_total'] == 1
    assert d['missing_input_count'] == 1


def test_unreadable_queue_never_becomes_zero_completed_progress(tmp_path):
    import pytest
    from visioncortex.device_day_progress import ProgressUnavailable
    root = tmp_path / 'device-day'
    root.mkdir()
    (root / 'queue-vision.sqlite3').write_bytes(b'not a readable SQLite database')
    with pytest.raises(ProgressUnavailable, match='vision'):
        snapshot({'storage': {'local_runtime_root': str(tmp_path)}})


def test_stop_does_not_start_another_nas_preparation_walk():
    from threading import Event
    from visioncortex.device_day import DeviceDayRunner
    stopped = Event()
    stopped.set()
    # No runner initialization or NAS access is necessary once stopping.
    runner = object.__new__(DeviceDayRunner)
    assert runner._prepare_stage(None, 'vision', None, stopped) == set()

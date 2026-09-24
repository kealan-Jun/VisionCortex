"""Ready sources remain claimable while later NAS metadata is still loading."""
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from test_device_day import FakeModels, capture
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_inplace import marker
from visioncortex.nas_recordings import scan_recordings


@pytest.fixture
def device_config(default_config, tmp_path):
    from test_device_day import device_config as fixture
    return fixture.__wrapped__(default_config, tmp_path)


def inventory_with_recent_first(config, monkeypatch):
    capture(config, start=1789005660000000, label='100100')
    capture(config, camera='b_cam02')
    inventory = scan_recordings(config)
    inventory['recordings'].sort(key=lambda record: record['recording_start_us'])
    older, recent = inventory['recordings']
    # Fix the live-lane classification independently of the test's wall date.
    # The builder and queue still perform their actual ordering and claiming.
    monkeypatch.setattr('visioncortex.device_day_schedule.scheduling_record',
                        lambda record, **kwargs: record | {'processing_priority': -1})
    return inventory, recent, older


@pytest.mark.parametrize('legacy', [False, True])
def test_ready_source_is_claimable_before_later_source_mode_check_finishes(
        device_config, monkeypatch, legacy):
    inventory, recent, older = inventory_with_recent_first(device_config, monkeypatch)
    backend = FakeModels()
    if legacy:
        previous = DeviceDayRunner(device_config, backend)
        previous.run_once({'recordings': [recent]}, stage='retention')
        checkpoint = previous._receipt(previous.layout(recent), recent, 'retention')
        original_checkpoint = checkpoint.read_bytes()
    device_config['device_day'].update(inplace_preprocessing=True, camera_lanes=True)
    runner = DeviceDayRunner(device_config, backend)
    blocked, release = threading.Event(), threading.Event()
    from visioncortex import device_day_inplace
    original_active = device_day_inplace.active

    def read_mode(current, record, **kwargs):
        if record['recording_id'] == older['recording_id']:
            blocked.set()
            assert release.wait(5), 'Test did not release the later source'
        return original_active(current, record, **kwargs)

    monkeypatch.setattr(device_day_inplace, 'active', read_mode)
    with ThreadPoolExecutor(max_workers=1) as pool:
        preparing = pool.submit(runner._prepare_stage, inventory, 'vision', None)
        try:
            assert blocked.wait(3)
            admitted = runner._prepare_stage(inventory, 'vision', None)
            assert admitted == {recent['recording_id']}
            claimed = runner.queues['vision'].claim('other-camera-slot', allowed=admitted)
            assert claimed['recording_id'] == recent['recording_id']
            assert runner.queues['vision'].claim('duplicate', allowed=admitted) is None
        finally:
            release.set()
        assert preparing.result(timeout=5) == {recent['recording_id'], older['recording_id']}
    assert backend.vision_calls == 0
    assert marker(runner, recent).exists() is not legacy
    if legacy:
        assert checkpoint.read_bytes() == original_checkpoint


def test_legacy_source_still_requires_completed_parent(device_config):
    capture(device_config)
    inventory = scan_recordings(device_config)
    record = inventory['recordings'][0]
    device_config['device_day'].update(inplace_preprocessing=True, camera_lanes=True)
    runner = DeviceDayRunner(device_config, FakeModels())
    # An incomplete old model checkpoint must remain on its original route.
    old_checkpoint = runner._receipt(runner.layout(record), record, 'vision')
    atomic_json(old_checkpoint, {'status': 'failed', 'recording_id': record['recording_id']})
    before = old_checkpoint.read_bytes()
    assert runner._prepare_stage(inventory, 'vision', None) == set()
    assert runner.queues['vision'].claim('worker') is None
    assert not marker(runner, record).exists()
    assert old_checkpoint.read_bytes() == before


def test_cancellation_stops_before_next_source_remote_mode_check(device_config, monkeypatch):
    inventory, recent, older = inventory_with_recent_first(device_config, monkeypatch)
    device_config['device_day']['inplace_preprocessing'] = True
    runner = DeviceDayRunner(device_config, FakeModels())
    stopped = threading.Event()
    from visioncortex import device_day_inplace
    original_active = device_day_inplace.active
    original_enqueue = device_day_inplace.enqueue

    def read_mode(current, record, **kwargs):
        assert record['recording_id'] != older['recording_id'], 'Read NAS metadata after cancellation'
        return original_active(current, record, **kwargs)

    def enqueue_then_stop(*args):
        admitted = original_enqueue(*args)
        stopped.set()
        return admitted

    monkeypatch.setattr(device_day_inplace, 'active', read_mode)
    monkeypatch.setattr(device_day_inplace, 'enqueue', enqueue_then_stop)
    assert runner._prepare_stage(inventory, 'vision', None, stopped) == {recent['recording_id']}
    assert runner._admitted['vision'] == {recent['recording_id']}


def test_unavailable_source_is_filtered_before_remote_mode_check(device_config, monkeypatch):
    capture(device_config)
    inventory = scan_recordings(device_config)
    inventory = deepcopy(inventory)
    inventory['recordings'][0]['processable'] = False
    device_config['device_day']['inplace_preprocessing'] = True
    runner = DeviceDayRunner(device_config, FakeModels())
    monkeypatch.setattr('visioncortex.device_day_inplace.active',
                        lambda *args, **kwargs: pytest.fail('Unavailable source reached remote mode check'))
    assert runner._prepare_stage(inventory, 'vision', None) == set()

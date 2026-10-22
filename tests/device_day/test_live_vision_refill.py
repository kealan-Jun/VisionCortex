"""Continuous admission refills while an older slice is running."""
from threading import Event
import time

import pytest

from test_live_vision_parallel import runner_for, statuses
from visioncortex.device_day import exclusive
from visioncortex.runtime_control import CURRENT


def test_new_camera_finishes_while_first_recording_is_still_running(tmp_path, default_config):
    first_started, second_started, release = Event(), Event(), Event()
    completed = []

    def process(record, **kwargs):
        assert CURRENT.get().job_id == record['recording_id']
        if record['recording_id'] == 'newest':
            first_started.set()
            assert release.wait(5)
        else:
            second_started.set()
        completed.append(record['recording_id'])
        return {'status': 'completed'}

    worker, runner, queue = runner_for(tmp_path, default_config, process)
    worker.continuous_vision = True
    records = worker.candidates(runner)
    worker.candidates = lambda _: records[:2]
    try:
        start = time.monotonic()
        assert worker.tick(runner, Event())['inflight'] == 1
        assert time.monotonic()-start < 1 and first_started.wait(1)
        with pytest.raises(BlockingIOError):
            with exclusive(runner.runtime_root/'locks/vision-worker.lock'):
                pass
        worker.candidates = lambda _: records[:3]
        assert worker.tick(runner, Event())['inflight'] == 2
        assert second_started.wait(1)
        # Queue completion is owned by _execute_claimed after process returns.
        deadline = time.monotonic()+1
        while statuses(queue)['second']['status'] != 'completed' and time.monotonic() < deadline:
            time.sleep(.01)
        assert 'second' in completed and 'newest' not in completed
        saved = statuses(queue)
        assert saved['newest']['status'] == 'running'
        assert saved['second']['status'] == 'completed'
        assert saved['older']['status'] == 'queued'
        assert saved['newest']['lease_owner'] != saved['second']['lease_owner']
    finally:
        release.set()
        worker.close()
    with exclusive(runner.runtime_root/'locks/vision-worker.lock'):
        pass
    assert statuses(queue)['newest']['status'] == 'completed'


def test_saturated_pool_enqueues_later_arrivals_and_retains_capacity(tmp_path, default_config):
    release, started = Event(), Event()

    def process(record, **kwargs):
        started.set()
        assert release.wait(5)
        return {'status': 'completed'}

    worker, runner, queue = runner_for(tmp_path, default_config, process, capacity=1, jobs=1)
    worker.continuous_vision = True
    records = worker.candidates(runner)
    worker.candidates = lambda _: records[:2]
    try:
        assert worker.tick(runner, Event())['inflight'] == 1
        assert started.wait(1)
        worker.candidates = lambda _: records
        assert worker.tick(runner, Event())['inflight'] == 1
        saved = statuses(queue)
        assert saved['second']['status'] == 'queued'
        assert sum(row['status'] == 'running' for row in saved.values()) == 1
    finally:
        release.set()
        worker.close()


def test_native_completion_wakes_publication_without_reading_nas(tmp_path, default_config):
    from visioncortex.publication_journal import PublicationJournal
    worker, runner, queue = runner_for(tmp_path, default_config,
        lambda *args, **kwargs: {'status': 'completed', 'formal_publication': 'pending'}, capacity=1, jobs=1)
    record = worker.candidates(runner)[1]
    journal = PublicationJournal(runner.runtime_root)
    token = journal.begin(record)
    journal.defer(record['recording_id'], token, reason='vision_not_finished')
    assert journal.pending(ready_only=True) == []
    assert worker.tick(runner, Event())['recording_id'] == record['recording_id']
    assert journal.pending(ready_only=True) == [(token, record)]
    assert statuses(queue)[record['recording_id']]['status'] == 'completed'


def test_failed_hint_does_not_revoke_actual_completed_stage(tmp_path, default_config, monkeypatch):
    from visioncortex.publication_journal import PublicationJournal
    worker, runner, queue = runner_for(tmp_path, default_config,
        lambda *args, **kwargs: {'status': 'completed', 'formal_publication': 'pending'}, capacity=1, jobs=1)
    def unavailable(*args):
        raise OSError('local hint unavailable')
    monkeypatch.setattr(PublicationJournal, 'ensure_pending', unavailable)
    result = worker.tick(runner, Event())
    assert result['result']['status'] == 'completed'
    assert statuses(queue)['newest']['status'] == 'completed'

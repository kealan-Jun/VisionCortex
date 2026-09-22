from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from visioncortex.config import load_config
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_schedule import (
    in_processing_scope,
    refresh_queue_priorities,
    scheduling_record,
)


def test_production_drains_live_then_preserved_history_and_prioritizes_new_arrivals(tmp_path):
    config = load_config(Path('configs/rtx3090ti-ubuntu-production.yaml'))
    settings = config['device_day']
    current = datetime.now(ZoneInfo('Asia/Shanghai'))
    # Noon keeps the recent-live and older-same-day cases distinct at any test time.
    current = current.replace(hour=12, minute=0, second=0, microsecond=0)
    queue = DeviceDayQueue(tmp_path / 'queue.sqlite3')

    def enqueue(name, start):
        record = {
            'recording_id': name, 'camera_key': 'camera',
            'configured_role': 'first_person',
            'recording_start_us': round(start.timestamp() * 1e6),
            'recording_end_us': round((start + timedelta(minutes=1)).timestamp() * 1e6),
        }
        assert in_processing_scope(settings, record)
        queue.enqueue(scheduling_record(
            record, today=current.date().isoformat(),
            live_priority_seconds=settings['live_priority_seconds'],
            now_us=round(current.timestamp() * 1e6),
        ), name + '-original-revision')

    enqueue('history', current - timedelta(days=30))
    enqueue('earlier_today', current - timedelta(hours=5))
    enqueue('recent', current - timedelta(minutes=30))
    enqueue('newest', current - timedelta(minutes=15))

    def finish_next(expected):
        record = queue.claim('worker', camera_serial=True)
        assert record['recording_id'] == expected
        queue.finish('worker', expected, {'status': 'completed'}, 1)

    finish_next('newest')
    finish_next('recent')
    finish_next('earlier_today')
    enqueue('arrival', current - timedelta(minutes=1))
    finish_next('arrival')
    finish_next('history')
    assert queue.claim('worker', camera_serial=True) is None
    # Reclassification must preserve completed work, provenance and retry history.
    refresh_queue_priorities(queue, live_priority_seconds=settings['live_priority_seconds'])
    with queue.connect() as db:
        rows = list(db.execute('SELECT recording_id,revision,status,attempts FROM recordings'))
    assert len(rows) == 5
    assert all(row['status'] == 'completed' and row['attempts'] == 1
               and row['revision'] == row['recording_id'] + '-original-revision' for row in rows)


@pytest.fixture
def device_config(default_config, tmp_path):
    from test_device_day import device_config as fixture
    return fixture.__wrapped__(default_config, tmp_path)


def test_camera_concurrency_is_bounded_and_claims_remain_exclusive(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    queue = DeviceDayQueue(tmp_path / 'queue.sqlite3')
    for camera in ('a', 'b'):
        for n in range(3):
            queue.enqueue({'recording_id': f'{camera}-{n}', 'camera_key': camera,
                           'configured_role': 'first_person', 'recording_start_us': n + 1}, 'v1')
    def claim(n):
        return queue.claim(str(n), camera_serial=True, camera_limit=2)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = [row for row in pool.map(claim, range(6)) if row]
    assert len({row['recording_id'] for row in rows}) == 4
    assert [row['camera_key'] for row in rows].count('a') == 2
    assert [row['camera_key'] for row in rows].count('b') == 2
    assert queue.claim('full', camera_serial=True, camera_limit=2) is None
    # An expired worker releases only its slot; other leases remain exclusive.
    with queue.connect() as db:
        db.execute('UPDATE recordings SET lease_until=0 WHERE recording_id=?', (rows[0]['recording_id'],))
    assert queue.claim('replacement', camera_serial=True, camera_limit=2)['recording_id'] == rows[0]['recording_id']
    assert queue.claim('full-again', camera_serial=True, camera_limit=2) is None


def test_lease_heartbeat_does_not_depend_on_result_publication(tmp_path, monkeypatch):
    from threading import Event
    import time
    queue = DeviceDayQueue(tmp_path / 'queue.sqlite3')
    queue.enqueue({'recording_id': 'slice', 'camera_key': 'a', 'configured_role': 'first_person'}, 'v1')
    queue.claim('owner')
    with queue.connect() as db:
        db.execute('UPDATE recordings SET lease_until=0')
    renewed = Event()
    original = queue.renew
    def renew(owner):
        original(owner)
        renewed.set()
    monkeypatch.setattr(queue, 'renew', renew)
    with queue.heartbeat('owner', interval=.01):
        # The publisher remains blocked here while the independent heartbeat runs.
        assert renewed.wait(2)
        with queue.connect() as db:
            assert db.execute('SELECT lease_until FROM recordings').fetchone()[0] > time.time() + 60
        assert queue.claim('duplicate') is None


def test_pausing_semantics_keeps_video_and_speech_running_without_report_publication(device_config, monkeypatch):
    from test_device_day import FakeModels, capture
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.nas_recordings import scan_recordings

    class Backend(FakeModels):
        speech_calls = 0

        def transcribe(self, *args):
            self.speech_calls += 1
            return super().transcribe(*args)

    capture(device_config)
    device_config['device_day']['paused_stages'] = ['understanding', 'report']
    inventory = scan_recordings(device_config)
    record = inventory['recordings'][0]
    backend = Backend()
    runner = DeviceDayRunner(device_config, backend)
    layout = runner.layout(record)
    layout.reports.mkdir(parents=True)
    old_report = layout.reports / 'LaboratoryDailyReport.html'
    old_report.write_text('previous report remains readable')
    monkeypatch.setattr('visioncortex.device_day_reports.render_day',
                        lambda *args: pytest.fail('Report publication is paused'))
    for stage in ('understanding', 'report'):
        runner.queues[stage].enqueue(record, 'preserved-revision')
        assert runner.run_once(stage=stage)['status'] == 'paused_by_user'

    runner.run_once(inventory)
    assert backend.vision_calls == backend.speech_calls == 1
    assert backend.semantic_calls == 0
    assert old_report.read_text() == 'previous report remains readable'
    assert layout.index.is_file()
    for stage in ('retention', 'vision', 'stt'):
        assert runner.queues[stage].snapshot()['counts'] == {'completed': 1}
    for stage in ('understanding', 'report'):
        with runner.queues[stage].connect() as db:
            assert tuple(db.execute('SELECT status,revision,attempts FROM recordings').fetchone()) == (
                'queued', 'preserved-revision', 0)

    # A restart retains the policy and full requests remain resumable.
    from visioncortex.device_day_service import DeviceDayService
    service = DeviceDayService(lambda: device_config, None)
    service._runner = DeviceDayRunner(device_config, backend)
    assert not service._request_complete({'recordings': []})
    from visioncortex.device_day_progress import snapshot
    progress = snapshot(device_config)
    assert progress['night_schedule']['paused_stages'] == ['report', 'understanding']
    waiting = next(iter(progress['waiting'].values()))
    assert waiting['understanding'] == waiting['report'] == {'paused_by_user': 1}


def test_background_audit_waits_for_speech_and_video_then_resumes(tmp_path):
    from types import SimpleNamespace
    from visioncortex.device_day_timeline import defer_multiview_audit

    settings = {'failure_retry_limit': 3, 'paused_stages': ['understanding', 'report']}
    queues = {stage: DeviceDayQueue(tmp_path / f'{stage}.sqlite3')
              for stage in ('retention', 'vision', 'stt', 'understanding', 'report')}
    runner = SimpleNamespace(settings=settings, config={'device_day': settings}, queues=queues)
    record = {'recording_id': 'slice', 'configured_role': 'first_person'}
    queues['understanding'].enqueue(record, 'revision')
    assert not defer_multiview_audit(runner)
    for stage in ('retention', 'vision', 'stt'):
        queue = queues[stage]
        queue.enqueue(record, 'revision')
        assert defer_multiview_audit(runner)
        queue.claim('worker')
        assert defer_multiview_audit(runner)
        queue.finish('worker', 'slice', {'status': 'failed'}, 1)
        assert defer_multiview_audit(runner), 'A retryable failure is still priority work'
        with queue.connect() as db:
            db.execute("UPDATE recordings SET attempts=3 WHERE recording_id='slice'")
        assert not defer_multiview_audit(runner)
        queue.enqueue(record, 'new-revision')
        with queue.connect() as db:
            db.execute("UPDATE recordings SET input_status='missing' WHERE recording_id='slice'")
        assert not defer_multiview_audit(runner), 'Unavailable input must not starve background work'


def test_publication_prefers_recent_results_then_drains_history_without_losing_writes(tmp_path):
    from visioncortex.publication_journal import PublicationJournal
    journal = PublicationJournal(tmp_path)
    old = journal.begin({'recording_id': 'history', 'recording_start_us': 10})
    recent = journal.begin({'recording_id': 'recent', 'recording_start_us': 101})
    assert journal.pending(1, prefer_since_us=100)[0][0] == recent
    journal.complete('recent', recent)
    assert journal.pending(1, prefer_since_us=100)[0][0] == old
    arrival = journal.begin({'recording_id': 'arrival', 'recording_start_us': 102})
    assert journal.pending(1, prefer_since_us=100)[0][0] == arrival
    journal.complete('arrival', arrival)
    assert journal.pending(1, prefer_since_us=100)[0][0] == old


def test_speech_index_publishes_while_historical_auxiliary_sweep_is_busy(device_config, monkeypatch):
    import threading
    import time
    from test_device_day import FakeModels, capture
    from visioncortex import device_day_service
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.device_day_contract import read_json
    from visioncortex.nas_recordings import scan_recordings

    capture(device_config)
    device_config['device_day'].update(paused_stages=['vision', 'understanding', 'report'],
                                      defer_audio_day_publication=True)
    inventory = scan_recordings(device_config)
    record = inventory['recordings'][0]
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.run_once(inventory, stage='retention')
    runner.run_once(inventory, stage='stt')
    busy, release = threading.Event(), threading.Event()

    def slow_sweep(*args):
        busy.set()
        release.wait(15)
        return []

    monkeypatch.setattr(device_day_service, 'DeviceDayRunner', lambda _: runner)
    monkeypatch.setattr('visioncortex.device_day_capture_files.reconcile_capture_files', slow_sweep)
    monkeypatch.setattr('visioncortex.device_day_publication.reconcile_outputs', lambda *args: {})
    monkeypatch.setattr('visioncortex.device_day_overview.ArchiveOverview.publish', lambda *args: {})
    from visioncortex.publication_journal import reconcile
    monkeypatch.setattr('visioncortex.publication_journal.reconcile',
                        lambda current: reconcile(current) if busy.is_set() else 0)
    service = device_day_service.DeviceDayService(lambda: device_config, threading.Lock())
    monkeypatch.setattr(service, '_storage_available', lambda *args: {'ready': True})
    service.observe(device_config, inventory)
    service.start()
    try:
        assert busy.wait(8)
        deadline = time.monotonic() + 10
        while True:
            path = runner.layout(record).index
            published = read_json(path).get('recordings', []) if path.is_file() else []
            if any(r.get('transcription', {}).get('status') == 'completed' for r in published):
                break
            assert time.monotonic() < deadline, 'Completed speech waited for historical auxiliary files'
            time.sleep(.05)
        assert not release.is_set()
    finally:
        release.set()
        service.stop()

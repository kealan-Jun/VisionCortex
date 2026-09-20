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

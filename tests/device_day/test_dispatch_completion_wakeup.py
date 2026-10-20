"""Controlled stage completions wake the real dispatcher without poll timing."""
from collections import Counter
import queue
import threading
import time

from test_device_day import FakeModels, capture, device_config
from visioncortex.device_day import DeviceDayRunner
from visioncortex.nas_recordings import scan_recordings

__all__ = ['device_config']


class DispatcherWait:
    def __init__(self, service, monkeypatch):
        self.service = service
        self.original = service.wakeup.wait
        self.turns = queue.Queue()
        self.expired = 0
        monkeypatch.setattr(service.wakeup, 'wait', self.wait)

    def wait(self, _timeout):
        stages = {stage: result.get('results')
                  for stage, result in self.service.last_result.get('stages', {}).items()}
        self.turns.put((stages, self.service.wakeup.is_set()))
        # Disable the one-second fallback during the assertion. Only an actual
        # notification can advance the dispatcher within the test's wait.
        notified = self.original(5)
        self.expired += not notified
        return notified

    def parked(self, predicate):
        deadline = time.monotonic() + 2
        while True:
            stages, notified = self.turns.get(timeout=max(0, deadline - time.monotonic()))
            if not notified and predicate(stages):
                return


def service_for(config, monkeypatch, run_once):
    from visioncortex import device_day_service as module
    config['device_day'].update(retention_workers=1, vision_workers=1,
                                paused_stages=['stt', 'understanding', 'report'])
    capture(config)
    runner = DeviceDayRunner(config, FakeModels())
    monkeypatch.setattr(runner, 'run_once', run_once)
    monkeypatch.setattr(module, 'DeviceDayRunner', lambda _settings: runner)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _path: type('Disk', (), {'free': 10**12})())
    monkeypatch.setattr('visioncortex.device_day_recovery.RetentionRecovery.tick', lambda *_: {'status': 'idle'})
    monkeypatch.setattr('visioncortex.device_day_overview.ArchiveOverview.publish', lambda *_: {'status': 'idle'})
    monkeypatch.setattr('visioncortex.device_day_transcript_publication.TranscriptPublication.tick',
                        lambda *_: {'results': []})
    monkeypatch.setattr('visioncortex.device_day_timeline.refresh_timeline', lambda *_, **__: {})
    monkeypatch.setattr('visioncortex.publication_journal.reconcile', lambda *_: 0)
    service = module.DeviceDayService(lambda: config, threading.Lock())
    service.observe(config, scan_recordings(config))
    return service


def test_stage_completion_wakes_sleeping_dispatcher_and_admits_dependency(device_config, monkeypatch):
    release, upstream_done, vision_seen, child_done = (threading.Event() for _ in range(4))
    calls = Counter()

    def run_once(_inventory, *, stage, **_kwargs):
        calls[stage] += 1
        if stage == 'retention' and calls[stage] == 1:
            assert release.wait(5)
            upstream_done.set()
            return {'results': [{'status': 'completed', 'recording_id': 'owned'}]}
        if stage == 'vision':
            vision_seen.set()
            if upstream_done.is_set() and not child_done.is_set():
                child_done.set()
                return {'results': [{'status': 'completed', 'recording_id': 'owned'}]}
        return {'results': []}

    service = service_for(device_config, monkeypatch, run_once)
    waiter = DispatcherWait(service, monkeypatch)
    service.start()
    try:
        assert vision_seen.wait(2)
        service.wakeup.set()  # Let the initial empty vision result enter backoff.
        waiter.parked(lambda stages: stages.get('vision') == [] and 'retention' not in stages)
        assert calls == {'retention': 1, 'vision': 1}
        release.set()
        assert child_done.wait(2), service.last_result
        waiter.parked(lambda stages: stages.get('retention') == stages.get('vision') == [])
        assert calls == {'retention': 2, 'vision': 3}
        assert waiter.expired == 0
    finally:
        release.set()
        service.stop()


def test_empty_stage_completion_keeps_backoff_without_duplicate_dispatch(device_config, monkeypatch):
    calls = Counter()

    def run_once(_inventory, *, stage, **_kwargs):
        calls[stage] += 1
        return {'results': []}

    service = service_for(device_config, monkeypatch, run_once)
    waiter = DispatcherWait(service, monkeypatch)
    service.start()
    try:
        def idle(stages):
            return stages.get('retention') == stages.get('vision') == []
        waiter.parked(idle)
        assert calls == {'retention': 1, 'vision': 1}
        for _ in range(3):
            service.wakeup.set()
            waiter.parked(idle)
            assert calls == {'retention': 1, 'vision': 1}
        assert waiter.expired == 0
    finally:
        service.stop()

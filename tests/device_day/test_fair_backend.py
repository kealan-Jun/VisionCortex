from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from test_background_model_preparation import Scanner, config, shared as _shared
from visioncortex.device_day_fair_backend import FairVisionBackend
from visioncortex.runtime_control import CURRENT, execution_context


@pytest.fixture
def shared(monkeypatch):
    yield from _shared.__wrapped__(monkeypatch)


def test_fair_vision_adapter_preserves_all_arguments_result_and_control_context():
    observed = []
    result = {'immutable': 'original-model-result'}
    def vision(*args):
        observed.append((args, CURRENT.get()))
        return result
    backend = FairVisionBackend(SimpleNamespace(vision=vision))
    layout, retention, key = object(), object(), object()
    stop, signal = Event(), Event()
    with execution_context(source='device_day_backfill', background_fair=True,
                           job_id='history-recording', priority=2, stop=stop, yield_signal=signal):
        parent = CURRENT.get()
        assert backend.vision(layout, retention, key) is result
        assert CURRENT.get() is parent
    args, context = observed[0]
    assert args == (layout, retention, key)
    assert context.source == 'device_day_fair_vision'
    assert context.background_fair and context.priority == 2
    assert context.job_id == 'history-recording' and context.stop is stop and context.yield_signal is signal


@pytest.mark.parametrize('source,fair', [('nas', False), ('device_day_backfill', False), ('nas', True)])
def test_adapter_keeps_live_and_idle_history_admission_unchanged(source, fair):
    seen = []
    backend = FairVisionBackend(SimpleNamespace(vision=lambda *args: seen.append(CURRENT.get())))
    with execution_context(source=source, background_fair=fair):
        parent = CURRENT.get()
        backend.vision(None, None, None)
    assert seen == [parent]


def test_paid_and_transcription_ports_keep_original_owner_and_source():
    seen = []
    owner = SimpleNamespace(
        transcribe=lambda *args: seen.append(('stt', args, CURRENT.get())) or 'speech',
        understand=lambda *args: seen.append(('understanding', args, CURRENT.get())) or 'paid-response',
        report=lambda *args: seen.append(('report', args, CURRENT.get())) or 'report',
        config={'original': True})
    backend = FairVisionBackend(owner)
    with execution_context(source='device_day_backfill', background_fair=True):
        parent = CURRENT.get()
        assert backend.transcribe(1, 2, 3) == 'speech'
        assert backend.understand(1, 2, 3, 4, 5) == 'paid-response'
        assert backend.report(6) == 'report'
    assert seen == [('stt', (1, 2, 3), parent), ('understanding', (1, 2, 3, 4, 5), parent),
                    ('report', (6,), parent)]
    assert backend.config is owner.config


def test_fair_adapter_uses_existing_bounded_pool_outputs_while_live_is_busy(shared):
    settings = config()
    created = []
    def factory(*args, **kwargs):
        created.append(1)
        return Scanner()
    class Owner:
        def vision(self, layout, retention, key):
            lease = shared.acquire_scanner(factory, 'first_person', settings, 640, 4)
            try:
                return lease.infer(retention)
            finally:
                lease.close()
    backend = FairVisionBackend(Owner())
    first = shared.acquire_scanner(factory, 'first_person', settings, 640, 4)
    second = shared.acquire_scanner(factory, 'first_person', settings, 640, 4)
    try:
        with execution_context(source='device_day_backfill', background_fair=True, priority=2):
            assert backend.vision(None, ['history-1', 'history-2'], 'key') == ['history-1', 'history-2']
        assert first.infer(['live-1']) == ['live-1']
        assert second.infer(['live-2']) == ['live-2']
        assert len(created) == len(next(iter(shared._POOLS.values()))) == 2
        assert shared._BACKGROUND_WARM is None
    finally:
        first.close()
        second.close()


def test_bounded_fair_initialization_does_not_stop_existing_foreground_inference(shared):
    settings = config()
    entered, release = Event(), Event()
    created = []
    class Preparing(Scanner):
        def prepare(self):
            entered.set()
            assert release.wait(5)
    def factory(*args, **kwargs):
        created.append(1)
        return Preparing() if len(created) == 2 else Scanner()
    class Owner:
        def vision(self, layout, retention, key):
            lease = shared.acquire_scanner(factory, 'first_person', settings, 640, 4)
            try:
                return lease.infer(retention)
            finally:
                lease.close()
    backend = FairVisionBackend(Owner())
    live = shared.acquire_scanner(factory, 'first_person', settings, 640, 4)
    def history():
        with execution_context(source='device_day_backfill', background_fair=True, priority=2):
            return backend.vision(None, ['history'], 'key')
    try:
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(history)
            try:
                assert entered.wait(2)
                assert len(created) == 2
                assert live.infer(['live-during-initialization']) == ['live-during-initialization']
                assert not pending.done()
                release.set()
                assert pending.result(timeout=2) == ['history']
                assert len(created) == len(next(iter(shared._POOLS.values()))) == 2
                assert shared._BACKGROUND_WARM is None
            finally:
                release.set()
    finally:
        live.close()

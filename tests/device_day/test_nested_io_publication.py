"""Composed publication shares one live job lease without evading I/O bounds."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event

import pytest

from visioncortex.device_day_io import slot
from visioncortex.runtime_control import ExecutionCancelled, ResourceCoordinator, execution_context
from visioncortex.sqlite_store import connection


def config(root, capacity=3):
    return {'storage': {'local_runtime_root': str(root)},
            'device_day': {'nas_io_slots': capacity, 'archive_copy_workers': 1,
                           'nas_io_timeout_seconds': 2}}


@pytest.mark.parametrize('capacity', [1, 2, 3])
def test_composed_publication_reuses_reader_and_total_leases(tmp_path, capacity):
    settings = config(tmp_path, capacity)
    coordinator = ResourceCoordinator(tmp_path/'state/resources.sqlite3')
    def leases():
        with connection(coordinator.path, readonly=True) as db:
            return {r['id'] for r in db.execute('SELECT id FROM leases')}
    with execution_context(job_id='publication', yield_signal=Event()), slot(settings):
        original = leases()
        with slot(settings), slot(settings, copy=True):
            assert leases() == original
        assert leases() == original
    assert coordinator.snapshot() == []


def test_nested_copy_keeps_its_independent_whole_copy_limit_and_releases_on_error(tmp_path):
    settings = config(tmp_path)
    coordinator = ResourceCoordinator(tmp_path/'state/resources.sqlite3')
    with pytest.raises(ValueError, match='publication'):
        with execution_context(job_id='publication', yield_signal=Event()), slot(settings):
            with slot(settings, copy=True, whole_copy=True), slot(settings, copy=True):
                leases = coordinator.snapshot()
                assert sorted(r['resource'] for r in leases) == ['nas-copy', 'nas-io', 'nas-io-read']
                raise ValueError('publication interrupted')
    assert coordinator.snapshot() == []


def test_copied_thread_context_cannot_borrow_the_parents_slot(tmp_path):
    settings = config(tmp_path, 1)
    coordinator = ResourceCoordinator(tmp_path/'state/resources.sqlite3')
    stop, started = Event(), Event()
    with execution_context(job_id='publication', stop=stop, yield_signal=stop), slot(settings):
        context = copy_context()

        def nested():
            started.set()
            with slot(settings):
                pytest.fail('Concurrent thread bypassed the total I/O limit')

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(context.run, nested)
            assert started.wait(1)
            stop.set()
            with pytest.raises(ExecutionCancelled):
                future.result(2)
    assert coordinator.snapshot() == []


def test_reuse_is_scoped_to_runtime_job_and_cancellation(tmp_path):
    first, second = config(tmp_path/'first'), config(tmp_path/'second')
    stop = Event()
    with execution_context(job_id='publication', stop=stop, yield_signal=stop), slot(first):
        with slot(second):
            assert len(ResourceCoordinator(tmp_path/'second/state/resources.sqlite3').snapshot()) == 2
        stop.set()
        with pytest.raises(ExecutionCancelled), slot(first):
            pytest.fail('Nested admission ignored cancellation')

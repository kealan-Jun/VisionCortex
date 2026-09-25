"""Owned local resource leases: no NAS, media or models are accessed."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
import sqlite3

import pytest

from visioncortex import device_day_io as io
from visioncortex import runtime_control as control
from visioncortex.runtime_control import ExecutionCancelled, ExecutionContext, ResourceCoordinator, execution_context
from visioncortex.sqlite_store import connection


def settings(tmp_path, slots=3):
    return {'storage': {'local_runtime_root': str(tmp_path)},
            'device_day': {'nas_io_slots': slots, 'archive_copy_workers': 1, 'nas_io_timeout_seconds': 3}}


def test_waiting_probe_is_readonly_until_lease_needs_renewal(tmp_path, monkeypatch):
    coordinator = ResourceCoordinator(tmp_path/'resources.sqlite3')
    ctx = ExecutionContext()
    assert coordinator.try_claim('holder', 'nas-io', 1, 1, ctx, now=0)
    assert not coordinator.try_claim('waiter', 'nas-io', 1, 1, ctx, now=1)
    writes = []
    @contextmanager
    def counted(path, **kwargs):
        if not kwargs.get('readonly'):
            writes.append(path)
        with connection(path, **kwargs) as db:
            yield db
    monkeypatch.setattr(control, 'connection', counted)
    for now in range(2, 30):
        assert not coordinator.try_claim('waiter', 'nas-io', 1, 1, ctx, now=now)
    assert writes == []
    assert not coordinator.try_claim('waiter', 'nas-io', 1, 1, ctx, now=62)
    assert len(writes) == 1
    with connection(coordinator.path, readonly=True) as db:
        row = db.execute("SELECT created,expires FROM leases WHERE id='waiter'").fetchone()
        assert tuple(row) == (1, 152)
    coordinator.release('holder')
    assert coordinator.try_claim('waiter', 'nas-io', 1, 1, ctx, now=63)


def test_copy_blocks_keep_original_place_without_exceeding_capacity(tmp_path):
    coordinator = ResourceCoordinator(tmp_path/'resources.sqlite3')
    ctx = ExecutionContext(priority=0)
    assert coordinator.try_claim('holder', 'nas-io', 1, 1, ctx, now=100)
    assert not coordinator.try_claim('later-reader', 'nas-io', 1, 1, ctx, now=105)
    assert not coordinator.try_claim('block-1', 'nas-io', 1, 1, ctx, now=106, queued_at=101)
    coordinator.release('holder')
    assert coordinator.try_claim('block-1', 'nas-io', 1, 1, ctx, now=107, queued_at=101)
    assert not coordinator.try_claim('later-reader', 'nas-io', 1, 1, ctx, now=107)
    coordinator.release('block-1')
    assert coordinator.try_claim('block-2', 'nas-io', 1, 1, ctx, now=108, queued_at=101)
    coordinator.release('block-2')
    assert coordinator.try_claim('later-reader', 'nas-io', 1, 1, ctx, now=109)


def test_copy_age_is_shared_only_inside_whole_copy_context(tmp_path, monkeypatch):
    calls = []
    class Coordinator:
        def __init__(self, path):
            pass
        @contextmanager
        def acquire(self, resource, **kwargs):
            calls.append((resource, kwargs))
            yield
    monkeypatch.setattr(io, 'ResourceCoordinator', Coordinator)
    clock = iter((100, 200))
    monkeypatch.setattr(io.time, 'time', lambda: next(clock))
    with pytest.raises(RuntimeError, match='fixture'):
        with io.slot(settings(tmp_path), copy=True, whole_copy=True):
            with io.slot(settings(tmp_path), copy=True):
                pass
            with io.slot(settings(tmp_path), copy=True):
                raise RuntimeError('fixture')
    with io.slot(settings(tmp_path), copy=True):
        pass
    assert [options['queued_at'] for _, options in calls] == [100, 100, 100, 200]
    assert all(options['context'].priority == 0 for _, options in calls)


def test_archive_can_run_while_read_capacity_is_full_and_cancel_releases_only_waiter(tmp_path):
    config = settings(tmp_path)
    coordinator = ResourceCoordinator(tmp_path/'state/resources.sqlite3')
    stop, started = Event(), Event()
    with io.slot(config), io.slot(config):
        with ThreadPoolExecutor(1) as pool:
            def reader():
                with execution_context(job_id='third-reader', stop=stop):
                    started.set()
                    with io.slot(config):
                        pytest.fail('Read capacity exceeded')
            waiting = pool.submit(reader)
            assert started.wait(2)
            with io.slot(config, copy=True):
                leases = coordinator.snapshot()
                assert sum(r['state'] == 'running' and r['resource'] == 'nas-io' for r in leases) == 3
                assert sum(r['state'] == 'running' and r['resource'] == 'nas-io-read' for r in leases) == 2
            stop.set()
            with pytest.raises(ExecutionCancelled):
                waiting.result(3)
    assert coordinator.snapshot() == []


def test_single_io_slot_does_not_create_zero_capacity_reader_gate(tmp_path):
    config = settings(tmp_path, slots=1)
    with io.slot(config):
        with connection(Path(config['storage']['local_runtime_root'])/'state/resources.sqlite3', readonly=True) as db:
            assert [r[0] for r in db.execute('SELECT name FROM resources')] == ['nas-io']


def test_transient_sqlite_busy_retries_but_other_database_errors_propagate(tmp_path, monkeypatch):
    coordinator = ResourceCoordinator(tmp_path/'resources.sqlite3')
    original = coordinator.try_claim
    calls = []
    def busy_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            exc = sqlite3.OperationalError('database is locked')
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc
        return original(*args, **kwargs)
    monkeypatch.setattr(coordinator, 'try_claim', busy_once)
    with coordinator.acquire('nas-io', capacity=1, timeout=2):
        assert len(calls) == 2
    assert coordinator.snapshot() == []
    def invalid(*args, **kwargs):
        exc = sqlite3.OperationalError('not a busy failure')
        exc.sqlite_errorcode = sqlite3.SQLITE_ERROR
        raise exc
    monkeypatch.setattr(coordinator, 'try_claim', invalid)
    with pytest.raises(sqlite3.OperationalError, match='not a busy'):
        with coordinator.acquire('nas-io', capacity=1):
            pytest.fail('Invalid database accepted')

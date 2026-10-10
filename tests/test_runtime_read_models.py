import json
import time

import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.input_availability import Availability
from visioncortex.device_day_activity import job, phase, observations, counted
from visioncortex.device_day_latency import observe, upload_completed, snapshot


def test_input_publication_continues_during_read_snapshot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex.sqlite_store import connection

    state = Availability(tmp_path)
    record = {'recording_id': 'one', 'source_signature': 'v1'}
    state.mark(record, 'waiting', observed=1)
    with ThreadPoolExecutor(max_workers=1) as workers:
        with connection(state.path, readonly=True) as reader:
            assert reader.execute('SELECT state FROM inputs').fetchone()[0] == 'waiting'

            def publish():
                Availability(tmp_path).mark(record, 'ready', observed=2)

            workers.submit(publish).result(timeout=3)
            # The held snapshot stays coherent while the new version commits.
            assert reader.execute('SELECT state FROM inputs').fetchone()[0] == 'waiting'
    assert state.states()['one']['state'] == 'ready'


def test_unavailable_inputs_do_not_block_other_cameras(tmp_path):
    q = DeviceDayQueue(tmp_path / "queue-vision.sqlite3")
    a = Availability(tmp_path)
    record = {
        "recording_id": "missing",
        "source_signature": "v1",
        "configured_role": "first_person",
        "camera_key": "cam1",
    }
    q.enqueue(record, "revision1")
    q.enqueue(record | {"recording_id": "good", "camera_key": "cam2"}, "revision1")
    a.mark(record, "missing")
    q.sync_availability(a.states())
    assert q.claim("worker")["recording_id"] == "good"
    assert q.snapshot()["counts"]["input_missing"] == 1
    a.mark(record, "ready")
    q.sync_availability(a.states())
    assert q.claim("worker2")["recording_id"] == "missing"


def test_old_missing_version_does_not_block_new_source(tmp_path):
    q = DeviceDayQueue(tmp_path / "queue-vision.sqlite3")
    a = Availability(tmp_path)
    record = {
        "recording_id": "one",
        "source_signature": "v1",
        "configured_role": "first_person",
        "camera_key": "cam",
    }
    q.enqueue(record, "rev1")
    a.mark(record, "missing")
    q.sync_availability(a.states())
    q.enqueue(record | {"source_signature": "v2"}, "rev2")
    q.sync_availability(a.states())
    assert q.claim("worker")["source_signature"] == "v2"


def test_durable_phase_and_counts_cleaned_after_completion(tmp_path):
    import sqlite3

    with job("vision", "one", root=tmp_path):
        with phase("coarse_scan_seconds"):
            counted("coarse", 8)
            with sqlite3.connect(tmp_path / "WorkPhases.sqlite3") as db:
                payload = json.loads(
                    db.execute("SELECT payload FROM phases").fetchone()[0]
                )
                assert payload["phase"] == "coarse_scan_seconds"
            assert "粗扫" in observations(tmp_path)[("vision", "one")]["phase"]
    assert observations(tmp_path) == {}


def test_receiver_receipt_revision_scoped_and_not_reconstructed(tmp_path):
    from visioncortex.observed_inventory import observe as inventory

    now = time.time()
    config = {"storage": {"local_runtime_root": str(tmp_path)}}
    record = {
        "recording_id": "r1",
        "camera_key": "cam",
        "source_signature": "sig",
        "processable": True,
        "recording_start_us": round((now - 600) * 1e6),
        "recording_end_us": round(now * 1e6),
    }
    inventory(tmp_path / "device-day", {"recordings": [record]})
    observe(config, [record], now=now)
    assert snapshot(config)["recent"][0]["upload_completed_at"] is None
    payload = {"recording_id": "r1", "source_signature": "sig", "completed_at": now - 3}
    upload_completed(config, payload)
    upload_completed(config, payload)
    data = snapshot(config)
    assert data["upload_completion_time_available"]
    assert data["recent"][0]["discovery_delay_seconds"] == 3
    with pytest.raises(ValueError):
        upload_completed(config, payload | {"source_signature": "wrong"})
    with pytest.raises(ValueError):
        upload_completed(config, payload | {"completed_at": now - 5})
    with pytest.raises(ValueError):
        upload_completed(config, payload | {"completed_at": now + 50})


def test_camera_binding_releases_only_blocked_jobs_without_changing_success(tmp_path):
    from visioncortex.input_availability import configured_record

    config = {'collection_ingest': {'camera_role_map': {'a': 'first_person'}}}
    q = DeviceDayQueue(tmp_path / 'queue-retention.sqlite3')
    item = {'recording_id': 'one', 'camera_key': 'a', 'source_signature': 'same'}
    q.enqueue(item, 'same')
    a = Availability(tmp_path)
    a.mark(item, 'missing')
    q.sync_availability(a.states())
    bound = configured_record(config, item)
    assert 'configured_role' not in item
    q.enqueue(bound, 'same')
    assert q.claim('worker') is None  # Role repair cannot override missing input.
    a.mark(item, 'ready')
    q.sync_availability(a.states())
    assert q.claim('worker')['configured_role'] == 'first_person'
    q.enqueue(item, 'same')  # An active lease keeps its own bound snapshot.
    with q.connect() as db:
        assert json.loads(db.execute('SELECT payload FROM recordings').fetchone()[0]) == bound
    q.finish('worker', 'one', {'status': 'completed', 'proof': 'original'}, 1)
    q.enqueue(bound, 'same')
    with q.connect() as db:
        row = db.execute('SELECT * FROM recordings').fetchone()
        assert row['status'] == 'completed' and 'original' in row['result']
    q.enqueue(configured_record(config, item | {'recording_id': 'unknown', 'camera_key': 'b'}), 'same')
    assert q.snapshot()['counts']['needs_camera_role'] == 1


def reconciler_fixture(tmp_path):
    from visioncortex.input_availability import Reconciler

    source = tmp_path / 'capture'
    source.mkdir()
    video = source / 'rgb.mp4'
    video.write_bytes(b'synthetic-not-video')
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'collection_ingest': {
        'source_root': str(source), 'camera_role_map': {'cam': 'first_person'}}}
    record = {'recording_id': 'r', 'camera_key': 'cam', 'video_path': str(video),
              'recording_start_us': 1789005600000000, 'source_signature': 'same', 'processable': True}
    q = DeviceDayQueue(tmp_path / 'device-day/queue-retention.sqlite3')
    q.enqueue(record, 'same')
    return Reconciler(config), record, source, video


def test_input_reinspection_preserves_explicit_view_role(tmp_path, monkeypatch):
    from visioncortex.observed_inventory import read_inventory
    reconciler, record, source, video = reconciler_fixture(tmp_path)
    monkeypatch.setattr('visioncortex.nas_recordings._inspect', lambda *args: record.copy())
    reconciler.tick()
    fresh = read_inventory(reconciler.root)['recordings'][0]
    assert fresh['configured_role'] == 'first_person'
    assert Availability(reconciler.root).states()['r']['state'] == 'ready'


def test_input_reinspection_skips_history_before_bounded_selection(tmp_path, monkeypatch):
    reconciler, record, source, video = reconciler_fixture(tmp_path)
    cutoff = record['recording_start_us']
    reconciler.config['device_day'] = {'process_since_us': cutoff}
    q = DeviceDayQueue(reconciler.root/'queue-retention.sqlite3')
    for number in range(30):
        q.enqueue(record | {'recording_id': f'old-{number:02d}', 'recording_start_us': cutoff-100,
                            'recording_end_us': cutoff-1, 'video_path': str(source/f'old-{number}.mp4')}, 'old')
    calls = []
    monkeypatch.setattr('visioncortex.nas_recordings._inspect', lambda *args: calls.append(args[1]) or record.copy())
    reconciler.tick()
    assert calls == [video]
    assert set(Availability(reconciler.root).states()) == {'r'}


def test_input_sidecar_race_is_not_video_deletion_and_outage_preserves_state(tmp_path, monkeypatch):
    reconciler, record, source, video = reconciler_fixture(tmp_path)
    def missing_sidecar(*args):
        raise FileNotFoundError('metadata publication raced the inspection')
    monkeypatch.setattr('visioncortex.nas_recordings._inspect', missing_sidecar)
    reconciler.tick()
    a = Availability(reconciler.root)
    assert a.states()['r']['state'] == 'waiting'
    assert a.states()['r']['reason'] == 'capture_metadata_missing'
    video.unlink()
    reconciler.cursor = ''
    reconciler.tick()
    assert a.states()['r']['state'] == 'missing'
    a.mark(record, 'ready')
    source.rmdir()  # Unmounted/unavailable capture root.
    reconciler.cursor = ''
    reconciler.tick()
    assert a.states()['r']['state'] == 'ready'


def test_transient_failures_rechecked_without_waiting_for_historical_sweep(tmp_path, monkeypatch):
    reconciler, record, source, video = reconciler_fixture(tmp_path)
    a = Availability(reconciler.root)
    a.mark(record, 'unavailable')
    reconciler.cursor = 'z'
    monkeypatch.setattr('visioncortex.nas_recordings._inspect', lambda *args: record.copy())
    reconciler.tick()
    assert a.states()['r']['state'] == 'ready'


def test_input_initialization_retries_only_busy_errors(monkeypatch, tmp_path):
    from contextlib import contextmanager
    import sqlite3
    import visioncortex.input_availability as module

    real_connection = module.connection
    attempts = []

    @contextmanager
    def locked_once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        with real_connection(*args, **kwargs) as db:
            yield db

    monkeypatch.setattr(module, "connection", locked_once)
    state = Availability(tmp_path)
    state.mark({"recording_id": "one"}, "ready")
    assert state.states()["one"]["state"] == "ready"
    assert len(attempts) == 4

    @contextmanager
    def invalid_database(*args, **kwargs):
        raise sqlite3.OperationalError("invalid database")
        yield

    monkeypatch.setattr(module, "connection", invalid_database)
    with pytest.raises(sqlite3.OperationalError, match="invalid database"):
        Availability(tmp_path)

import io
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from test_receipt_streaming import fixture
from test_device_day import device_config  # noqa: F401
from visioncortex import receipt_projection as module
from visioncortex.device_day_contract import read_json


def deferred(projection):
    return read_json(next((projection.path.parent / 'ReceiptProjectionDeferred').glob('*.json')))


def test_oversized_source_defers_before_read_and_preserves_cache(tmp_path, monkeypatch):
    layout, projection = fixture(tmp_path)
    before = projection.records(layout, ['retention'])
    source = layout.receipts / 'a' / 'retention.json'
    source.write_text(json.dumps({'audit': 'x' * 600}))
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 512)
    original = Path.open

    def opened(path, *args, **kwargs):
        assert path != source, 'Oversized authoritative receipt must not be read'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(module.ProjectionDeferred):
        projection.records(layout, ['retention'], recording_id='a')
    with sqlite3.connect(projection.path) as db:
        assert [(r[0], json.loads(r[1])) for r in db.execute('SELECT id,payload FROM receipts ORDER BY id')] == before
    state = deferred(projection)
    assert state['status'] == 'deferred' and state['recording_id'] == 'a'
    assert state['metric'] == 'encoded_bytes' and state['limit'] == 512
    assert not list(projection.path.parent.glob('receipt-projection-*/'))


def test_growing_source_cannot_bypass_stat_budget(tmp_path, monkeypatch):
    layout, projection = fixture(tmp_path)
    source = layout.receipts / 'a' / 'retention.json'
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 512)
    monkeypatch.setattr(module, '_CHUNK_BYTES', 64)
    original = Path.open

    def opened(path, *args, **kwargs):
        if path == source:
            return io.BytesIO(json.dumps({'grown': 'x' * 2000}).encode())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(module.ProjectionDeferred):
        projection.records(layout, ['retention'])
    assert deferred(projection)['recording_id'] == 'a'
    with sqlite3.connect(projection.path) as db:
        assert db.execute('SELECT count(*) FROM receipts').fetchone()[0] == 0


def test_existing_huge_cache_is_rejected_before_any_row_decode_or_nas_read(tmp_path, monkeypatch):
    layout, projection = fixture(tmp_path)
    projection.records(layout, ['retention'])
    with sqlite3.connect(projection.path) as db:
        db.execute("UPDATE receipts SET payload='' WHERE id='c'")
        for ordinal in range(20):
            db.execute('INSERT INTO receipt_chunks VALUES(?,?,?,?)', ('day', 'c', ordinal, b'x' * 64))
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 512)
    monkeypatch.setattr(module, '_read_chunks', lambda *a: pytest.fail('Oversized cached row decoded'))
    original = Path.open

    def opened(path, *args, **kwargs):
        assert not path.is_relative_to(layout.receipts), 'Cache guard must precede NAS reads'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(module.ProjectionDeferred):
        next(projection.iter_records(layout, ['retention'], recording_id='a'))
    assert deferred(projection)['recording_id'] == 'c'


@pytest.mark.parametrize('kind', ['events', 'depth', 'bytes'])
def test_direct_chunk_hydration_is_bounded(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(module, '_MAX_JSON_EVENTS', 30 if kind == 'events' else 1000)
    monkeypatch.setattr(module, '_MAX_JSON_DEPTH', 4 if kind == 'depth' else 128)
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 32 if kind == 'bytes' else 4096)
    raw = ('[' * 5 + '0' + ']' * 5 if kind == 'depth' else json.dumps(list(range(100)))).encode()
    rows = [(raw[i:i+16],) for i in range(0, len(raw), 16)]
    with pytest.raises(module.ProjectionDeferred) as error:
        module._read_chunks(rows)
    assert error.value.metric == {'events': 'json_events', 'depth': 'json_depth', 'bytes': 'encoded_bytes'}[kind]


def test_event_limit_defers_without_replacing_generation_then_recovers(tmp_path, monkeypatch):
    layout, projection = fixture(tmp_path)
    projection.records(layout, ['retention'])
    source = layout.receipts / 'a' / 'retention.json'
    source.write_text(json.dumps({'rows': list(range(100))}))
    monkeypatch.setattr(module, '_CHUNK_BYTES', 32)
    monkeypatch.setattr(module, '_MAX_JSON_EVENTS', 30)
    with pytest.raises(module.ProjectionDeferred):
        projection.records(layout, ['retention'], recording_id='a')
    assert deferred(projection)['metric'] == 'json_events'
    source.write_text('{"status":"completed","replacement":true}')
    assert dict(projection.records(layout, ['retention'], recording_id='a'))['a']['retention']['replacement']
    assert deferred(projection)['status'] == 'ready'


def test_deferred_day_keeps_published_index_and_journal_but_other_day_progresses(tmp_path, monkeypatch):
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.publication_journal import PublicationJournal, reconcile
    root = tmp_path / 'runtime'
    root.mkdir()
    receipts = tmp_path / 'receipts'
    (receipts / 'bad').mkdir(parents=True)
    (receipts / 'bad' / 'retention.json').write_text(json.dumps({'too_large': 'x' * 1000}))
    index = tmp_path / 'Index.json'
    index.write_text('{"previous":"published"}')
    layout = SimpleNamespace(name='bad-day', receipts=receipts, index=index)
    journal = PublicationJournal(root)
    base = {'camera_key': 'owned_cam01', 'recording_start_us': 1789005600000000}
    bad = journal.begin(base | {'recording_id': 'bad'})
    journal.begin(base | {'recording_id': 'good'})
    runner = SimpleNamespace(runtime_root=root, backend_root=tmp_path, settings={}, _index_locks={},
                             layout=lambda record: layout if record['recording_id'] == 'bad' else None)
    runner.refresh_index = lambda target, **kw: (DeviceDayRunner.refresh_index(runner, target, **kw)
                                                if target is not None else True)
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 512)
    assert reconcile(runner) == 1
    assert index.read_text() == '{"previous":"published"}'
    assert journal.pending() == [(bad, base | {'recording_id': 'bad'})]
    assert deferred(module.ReceiptProjection(root))['status'] == 'deferred'


def test_report_projection_deferral_never_completes_queue_or_acknowledges_journal(device_config, monkeypatch):  # noqa: F811
    from test_device_day_prerequisites import prepared
    from visioncortex.publication_journal import PublicationJournal
    runner, record, layout, inventory, _, backend = prepared(device_config)
    assert runner.process(record, stage='all')['status'] == 'completed'
    before = layout.index.read_bytes()
    report_before = (layout.reports / 'LaboratoryDailyReport.html').read_bytes()
    report_receipt = runner._receipt(layout, record, 'report')
    report_receipt.unlink()
    calls = backend.vision_calls, backend.semantic_calls
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 512)
    result = runner.run_once(inventory, stage='report')
    assert result['results'][0]['status'] == 'waiting_for_publication'
    with runner.queues['report'].connect() as db:
        status = db.execute('SELECT status FROM recordings WHERE recording_id=?',
                            (record['recording_id'],)).fetchone()[0]
    assert status != 'completed'
    assert PublicationJournal(runner.runtime_root).pending()[0][1]['recording_id'] == record['recording_id']
    assert layout.index.read_bytes() == before
    assert (layout.reports / 'LaboratoryDailyReport.html').read_bytes() == report_before
    assert (backend.vision_calls, backend.semantic_calls) == calls

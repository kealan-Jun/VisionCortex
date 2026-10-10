"""Frozen JSON bytes, atomic failure behavior, and bounded receipt residency."""
import hashlib
import json
import math
import io
import sqlite3
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex import device_day_contract as contract
from visioncortex.receipt_projection import ReceiptProjection


@pytest.mark.parametrize('value', [None, True, -1, 1.25e-30, -0.0, '中文\n"\\\t🧪',
    {'b': [0, None, {}, []], 'a': {'x': 1.3e25, '负值': -0.0}},
    {'rows': [{'index': i, 'unicode': '实验/🧪', 'nested': [True, None, 1.2]} for i in range(3000)]}])
def test_streaming_preserves_exact_frozen_bytes_and_digest(tmp_path, value):
    target = tmp_path / 'receipt.json'
    expected = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    contract.atomic_json(target, value)
    assert target.read_bytes() == expected
    expected_hash = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert contract.digest(value) == expected_hash


@pytest.mark.parametrize('failure', ['nan', 'cycle', 'unsupported', 'fsync'])
def test_partial_json_never_replaces_previous_receipt(tmp_path, monkeypatch, failure):
    target=tmp_path/'receipt.json'
    original=b'{"previous":"verified"}\n'
    target.write_bytes(original)
    value={'large_prefix':['x'*80]*20000}
    if failure=='nan':
        value['invalid']=math.nan
    elif failure=='cycle':
        value['invalid']=value
    elif failure=='unsupported':
        value['invalid']=object()
    else:
        def fail(_fd):
            raise OSError('injected fsync error')
        monkeypatch.setattr(contract.os,'fsync',fail)
    with pytest.raises((ValueError,TypeError,OSError)):
        contract.atomic_json(target,value)
    assert target.read_bytes()==original
    assert list(tmp_path.iterdir())==[target]


def fixture(tmp_path):
    root=tmp_path/'receipts'
    for identifier in ('a','b','c'):
        (root/identifier).mkdir(parents=True)
        (root/identifier/'retention.json').write_text(json.dumps({'status':'completed','id':identifier}))
    return SimpleNamespace(name='day',receipts=root),ReceiptProjection(tmp_path/'runtime')


def test_receipts_are_released_between_nas_reads_and_output_rows(tmp_path,monkeypatch):
    layout,projection=fixture(tmp_path)
    refs=[]
    original=json.loads
    class Tracked(dict):
        pass
    def read(*args, **kwargs):
        assert not any(ref() is not None for ref in refs), 'Prior NAS receipt still retained'
        row=Tracked(original(*args, **kwargs))
        refs.append(weakref.ref(row))
        return row
    monkeypatch.setattr(json,'loads',read)
    rows=projection.iter_records(layout,['retention'])
    assert next(rows)[0]=='a'
    assert not any(ref() is not None for ref in refs)
    assert not [p for p in projection.path.parent.glob('receipt-projection-*') if p.is_dir()]
    # Only one JSON parse per next(), rather than a list of all decoded rows.
    parse=original
    calls=[]
    def loads(*args,**kwargs):
        calls.append(1)
        return parse(*args,**kwargs)
    monkeypatch.setattr(json,'loads',loads)
    assert next(rows)[0]=='b' and len(calls)==1
    rows.close()


@pytest.mark.parametrize('failure', ['read','invalid_json','sql'])
def test_full_rebuild_failure_keeps_previous_generation_and_cleans_spool(tmp_path,monkeypatch,failure):
    layout,projection=fixture(tmp_path)
    before=projection.records(layout,['retention'])
    original=Path.open
    def read(path, *args, **kwargs):
        if path.parent.name=='b':
            if failure=='read':
                raise OSError('injected NAS failure')
            if failure=='invalid_json':
                return io.BytesIO(b'{"incomplete":')
        if path.name=='retention.json':
            return io.BytesIO(b'{"version":"replacement"}')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',read)
    if failure=='sql':
        with sqlite3.connect(projection.path) as db:
            db.execute("CREATE TRIGGER fail_insert BEFORE INSERT ON receipts WHEN NEW.id='b' BEGIN SELECT RAISE(ABORT,'injected SQL failure'); END")
    with pytest.raises((OSError,ValueError,sqlite3.DatabaseError)):
        list(projection.iter_records(layout,['retention']))
    with sqlite3.connect(projection.path) as db:
        saved=[(x[0],json.loads(x[1])) for x in db.execute('SELECT id,payload FROM receipts ORDER BY id')]
    assert saved==before
    assert not [p for p in projection.path.parent.glob('receipt-projection-*') if p.is_dir()]


def test_nas_reads_do_not_hold_projection_writer_lock(tmp_path,monkeypatch):
    layout,projection=fixture(tmp_path)
    original=Path.open
    def read(path,*args,**kwargs):
        if path.is_relative_to(layout.receipts):
            with sqlite3.connect(projection.path,timeout=.1) as db:
                db.execute('BEGIN IMMEDIATE')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',read)
    assert len(projection.records(layout,['retention']))==3


def test_large_receipt_uses_bounded_values_and_preserves_complete_evidence(tmp_path, monkeypatch):
    """Exercise SQLite's real bind limit with small fixtures, without a 2GB test."""
    from contextlib import contextmanager
    from visioncortex import receipt_projection as module
    layout, projection = fixture(tmp_path)
    monkeypatch.setattr(module, '_CHUNK_BYTES', 256)
    value = {'status': 'completed', 'dense_audit': [
        {'frame': i, 'text': '实验🧪' * 12, 'score': i / 7, 'large_integer': 2**80}
        for i in range(100)]}
    source = layout.receipts/'a'/'retention.json'
    source.write_text(json.dumps(value, ensure_ascii=False))
    original_connection = module.connection
    @contextmanager
    def limited(*args, **kwargs):
        with original_connection(*args, **kwargs) as db:
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
            yield db
    monkeypatch.setattr(module, 'connection', limited)
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        assert path != source and path.stat().st_size <= 256, 'Large receipt read as one string'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert dict(projection.records(layout, ['retention']))['a'] == {'retention': value}
    with sqlite3.connect(projection.path) as db:
        count, maximum = db.execute('SELECT count(*),max(length(payload)) FROM receipt_chunks').fetchone()
    assert count > 1 and maximum <= 256
    assert not [p for p in projection.path.parent.glob('receipt-projection-*') if p.is_dir()]


def test_large_projection_failure_keeps_complete_previous_generation(tmp_path, monkeypatch):
    from visioncortex import receipt_projection as module
    layout, projection = fixture(tmp_path)
    monkeypatch.setattr(module, '_CHUNK_BYTES', 64)
    source = layout.receipts/'a'/'retention.json'
    original = {'evidence': 'original' * 100}
    source.write_text(json.dumps(original))
    before = projection.records(layout, ['retention'])
    with sqlite3.connect(projection.path) as db:
        db.execute("CREATE TRIGGER fail_chunk BEFORE INSERT ON receipt_chunks WHEN NEW.ordinal=1 "
                   "BEGIN SELECT RAISE(ABORT,'injected chunk failure'); END")
    source.write_text(json.dumps({'evidence': 'replacement' * 100}))
    with pytest.raises(sqlite3.DatabaseError):
        projection.records(layout, ['retention'])
    with module.connection(projection.path, readonly=True) as db:
        saved = module._read_chunks(db.execute("SELECT payload FROM receipt_chunks WHERE id='a' ORDER BY ordinal"))
    assert saved == dict(before)['a'] == {'retention': original}
    assert not [p for p in projection.path.parent.glob('receipt-projection-*') if p.is_dir()]


def test_incremental_projection_clears_old_chunks_and_rejects_invalid_large_json(tmp_path, monkeypatch):
    from visioncortex import receipt_projection as module
    layout, projection = fixture(tmp_path)
    monkeypatch.setattr(module, '_CHUNK_BYTES', 64)
    source = layout.receipts/'a'/'retention.json'
    source.write_text(json.dumps({'evidence': 'x' * 1000}))
    projection.records(layout, ['retention'])
    source.write_text('{"unfinished":' + ' ' * 1000)
    with pytest.raises(ValueError, match='Invalid receipt JSON'):
        projection.records(layout, ['retention'], recording_id='a')
    source.write_text('{"status":"replacement"}')
    assert dict(projection.records(layout, ['retention'], recording_id='a'))['a'] == {
        'retention': {'status': 'replacement'}}
    with sqlite3.connect(projection.path) as db:
        assert db.execute("SELECT count(*) FROM receipt_chunks WHERE id='a'").fetchone()[0] == 0


def test_chunked_projection_does_not_modify_legacy_worker_database(tmp_path):
    root = tmp_path/'runtime'
    root.mkdir()
    legacy = root/'receipt-projection.sqlite3'
    legacy.write_bytes(b'owned by the previous worker')
    projection = ReceiptProjection(root)
    assert projection.path != legacy
    assert legacy.read_bytes() == b'owned by the previous worker'

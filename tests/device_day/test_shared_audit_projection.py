"""Large repeated audit JSON keeps complete values within a resident bound."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex import receipt_projection as module
from visioncortex.device_day_contract import digest


def projection(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIONCORTEX_RECEIPT_PROJECTION_SHARED_AUDITS', '1')
    monkeypatch.setattr(module, '_CHUNK_BYTES', 512)
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 128 * 1024)
    folder = tmp_path / 'receipts' / 'a'
    folder.mkdir(parents=True)
    layout = SimpleNamespace(name='owned-day', receipts=folder.parent)
    return layout, module.ReceiptProjection(tmp_path / 'runtime'), folder / 'vision.json'


def test_repeated_audits_restore_exact_receipt_and_share_only_equal_values(tmp_path, monkeypatch):
    layout, cache, source = projection(tmp_path, monkeypatch)
    audit = {'events': [{'frame': n, 'description': '实验🧪' * 30, 'large_integer': 2**80}
                        for n in range(100)]}
    value = {'status': 'completed', 'segments': [
        {'ordinal': i, 'activity_audit': audit} for i in range(20)]}
    raw = json.dumps(value, ensure_ascii=False).encode()
    assert len(raw) > module._MAX_RECEIPT_BYTES
    source.write_bytes(raw)
    checksum = source.read_bytes()
    restored = dict(cache.records(layout, ['vision']))['a']['vision']
    assert restored == value and digest(restored) == digest(value)
    assert restored['segments'][0]['activity_audit'] is restored['segments'][19]['activity_audit']
    assert source.read_bytes() == checksum
    changed = json.loads(raw)
    changed['segments'][0]['activity_audit']['events'][0]['frame'] = -1
    # Distinct objects remain distinct; hash interning never compares counts alone.
    monkeypatch.setattr(module, '_MAX_RECEIPT_BYTES', 256 * 1024)
    result = module._deduplicated_json(io.BytesIO(json.dumps(changed).encode()))
    assert result == changed
    assert result['segments'][0]['activity_audit'] != result['segments'][1]['activity_audit']


def test_unique_oversized_values_defer_without_replacing_previous_cache(tmp_path, monkeypatch):
    layout, cache, source = projection(tmp_path, monkeypatch)
    source.write_text('{"status":"original"}')
    assert dict(cache.records(layout, ['vision']))['a']['vision']['status'] == 'original'
    source.write_text(json.dumps({'activity_audit': {'unique': list(range(20000))}}))
    with pytest.raises(module.ProjectionDeferred, match='resident_bytes'):
        cache.records(layout, ['vision'])
    with module.connection(cache.path, readonly=True) as db:
        saved = db.execute("SELECT payload FROM receipts WHERE id='a'").fetchone()[0]
    assert json.loads(saved)['vision']['status'] == 'original'
    assert not list(cache.path.parent.glob('receipt-projection-*/'))


def test_unchanged_verified_receipts_skip_source_reads_and_cache_writes(tmp_path, monkeypatch):
    layout, cache, source = projection(tmp_path, monkeypatch)
    source.write_text('{"status":"old"}')
    first = cache.records(layout, ['vision'])
    with module.connection(cache.path) as db:
        db.execute("CREATE TRIGGER no_rewrite BEFORE INSERT ON receipts BEGIN SELECT RAISE(ABORT,'unnecessary write'); END")
    opened = Path.open
    def guard(path, *args, **kwargs):
        assert path != source, 'Unchanged verified source was reread'
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    assert cache.records(layout, ['vision']) == first


def test_receipt_change_during_stream_preserves_prior_generation(tmp_path, monkeypatch):
    layout, cache, source = projection(tmp_path, monkeypatch)
    source.write_text('{"status":"old"}')
    cache.records(layout, ['vision'])
    source.write_text('{"status":"new"}')
    opened = Path.open
    def raced(path, *args, **kwargs):
        if path == source:
            with opened(source, 'w') as handle:
                handle.write('{"status":"concurrently-replaced"}')
            return io.BytesIO(b'{"status":"new"}')
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', raced)
    with pytest.raises(module.ProjectionGenerationChanged, match='source changed'):
        cache.records(layout, ['vision'])
    with module.connection(cache.path, readonly=True) as db:
        assert 'old' in db.execute("SELECT payload FROM receipts WHERE id='a'").fetchone()[0]


@pytest.mark.parametrize('value', [None, 2**80, {'x': [1.5, True, None], 'activity_audit': []}])
def test_streaming_numbers_and_empty_containers_keep_legacy_values(value):
    assert module._deduplicated_json(io.BytesIO(json.dumps(value).encode())) == value

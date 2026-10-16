"""Reviewed native checkpoints retain their exact original model receipts."""
import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, digest, file_hash, read_json
from visioncortex.device_day_engine_rollover import CompletedStageReceipts
from visioncortex.device_day_inputs import input_path, processing_input


@pytest.mark.parametrize('stage', ['vision', 'stt'])
def test_verified_native_checkpoint_is_not_relabelled(device_config, tmp_path, monkeypatch, stage):  # noqa: F811
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    record, layout = item_and_layout(device_config)
    old = DeviceDayRunner(device_config, FakeModels())
    for name in ('vision', 'stt', 'retention'):
        assert old.process(record, stage=name)['status'] == 'completed'
    path = old._receipt(layout, record, stage)
    original_bytes = path.read_bytes()
    saved = read_json(path)
    calls = []
    backend = FakeModels()
    for name in ('vision', 'transcribe'):
        original = getattr(backend, name)
        def tracked(*args, _original=original, _name=name):
            calls.append(_name)
            return _original(*args)
        setattr(backend, name, tracked)
    current = DeviceDayRunner(device_config, backend)
    original_key = current._key
    def migrated(name, *args, **kwargs):
        key = original_key(name, *args, **kwargs)
        return digest(['new-source-verification', key]) if name == stage else key
    monkeypatch.setattr(current, '_key', migrated)
    archived = read_json(path.with_name('retention.json'))
    value = processing_input(layout, read_json(input_path(current, record, stage)), archived)
    from visioncortex.device_day import visual_input
    key = current._key(stage, record, visual_input(value) if stage == 'vision' else value)
    proof = tmp_path / 'Checkpoints.json'
    atomic_json(proof, {'schema_version': 'visioncortex-resume-checkpoints/1', 'entries': [{
        'stage': stage, 'recording_id': record['recording_id'], 'expected_key': key,
        'receipt_key': saved['key'], 'receipt_digest': digest(saved), 'queue_revision': '1' * 64}]})
    current._completed_stage_receipts = CompletedStageReceipts({'completed_stage_receipts': {
        'path': str(proof), 'sha256': file_hash(proof)}})
    result = current.process(record, stage=stage)
    assert result['status'] == 'completed', result
    assert calls == []
    assert path.read_bytes() == original_bytes
    assert read_json(path)['key'] == saved['key'] != key
    assert current._completed_stage_receipts.accepts(read_json(path), key)
    assert not current._completed_stage_receipts.accepts(saved | {'input_binding': 'changed'}, key)
    if stage == 'vision':
        # A new archival attestation of the same verified bytes must not
        # mutate the original detector receipt or invalidate its checkpoint.
        ret_path = path.with_name('retention.json')
        atomic_json(ret_path, read_json(ret_path) | {'fresh_source_validation': 'new-execution'})
        assert current.process(record, stage=stage)['status'] == 'completed'
        assert path.read_bytes() == original_bytes
        publication = read_json(path.with_name('publication.json'))
        assert publication['status'] == 'completed'
        assert publication['retention_digest'] == digest(read_json(ret_path))
        assert publication['vision_digest'] == digest(saved)

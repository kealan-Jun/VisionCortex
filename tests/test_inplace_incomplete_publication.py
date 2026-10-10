"""Partial sibling receipt isolation; fixture publications do not prove quality."""
from pathlib import Path

import pytest

from test_device_day import device_config  # noqa: F401
from test_device_day_inplace import setup  # noqa: F401
from visioncortex import device_day_inplace as inplace
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json
from device_day_identity_baseline_v1 import PUBLICATION_AST_SHA256_V1, PUBLICATION_IDENTITY_V1


@pytest.mark.parametrize('status', ['running', 'failed', 'completed'])
def test_unsealed_sibling_cannot_block_completed_slice_publication(setup, status):  # noqa: F811
    _, record, layout, backend, runner = setup
    assert runner.process(record, stage='vision')['status'] == 'completed'
    bad = layout.receipts / 'sibling-without-sealed-recording'
    atomic_json(bad / 'retention.json', {'input_binding_version': 1, 'status': status,
                                       'error_type': 'InputChanged', 'recording_id': bad.name})
    atomic_json(bad / 'vision.json', {'status': 'completed', 'segments': [{'segment_id': 'must-not-publish'}]})
    result = runner.process(record, stage='retention')
    assert result['status'] == 'completed', result
    assert result['formal_publication'] == 'completed', result
    index = read_json(layout.index)
    assert index['segments']
    assert {segment['recording_id'] for segment in index['segments']} == {record['recording_id']}
    sibling = next(row for row in index['recordings'] if row['recording_id'] == bad.name)
    assert 'vision' not in sibling['stages']
    assert sibling['stages']['retention']['status'] == status
    assert backend.vision_calls == 1


@pytest.mark.parametrize('recording', [None, {}, {'recording_id': 'incomplete'}])
def test_malformed_retention_has_no_downstream_publication_and_no_receipt_lookup(monkeypatch, recording):
    monkeypatch.setattr(inplace, 'receipt', lambda *args: pytest.fail('unsealed record looked up'))
    stages = {'retention': {'input_binding_version': 1, 'status': 'completed', 'recording': recording},
              **{stage: {'status': 'completed'} for stage in ('vision', 'stt', 'understanding', 'report')}}
    filtered = inplace.publication_filter(None, None, stages)
    assert filtered == {'retention': stages['retention']}
    assert len(stages) == 5


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt', 'understanding', 'report'])
def test_new_execution_does_not_accept_the_old_publication_stage_key(setup, stage):  # noqa: F811
    config, record, _, backend, runner = setup
    assert inplace.active(runner, record, initialize=True)
    historical = '3bb8ceaea6c91571b0f6304c697ab870319ab6805125988ae8bde79d3e819c3a'
    assert inplace.execution_identity() != historical
    before = runner._key(stage, record, {})
    parent = DeviceDayRunner(config, backend)
    assert parent._key(stage, record, {}) == before
    parent._inplace_execution_identity = historical
    assert parent._key(stage, record, {}) != before
    parent._inplace_execution_identity = 'unknown-execution-change'
    assert parent._key(stage, record, {}) != before


@pytest.mark.parametrize('fingerprint,expected', [
    (PUBLICATION_AST_SHA256_V1, PUBLICATION_IDENTITY_V1),
    ('f' * 64, 'f' * 64),
])
def test_fixed_reviewed_publication_fingerprint_keeps_its_historical_alias(monkeypatch, fingerprint, expected):
    # Inject the independently verified historical AST witness at the digest
    # port. Keep actual-source mutation coverage below; no old worker is copied.
    inputs = []

    def witnessed_digest(parts):
        assert len(parts) == 6
        assert all(part.startswith('FunctionDef(') for part in parts)
        inputs.append(parts)
        return fingerprint

    monkeypatch.setattr(inplace, 'digest', witnessed_digest)
    assert inplace.execution_identity() == expected
    assert len(inputs) == 1


def test_unknown_execution_edit_is_not_aliased(monkeypatch):
    actual = inplace.execution_identity()
    source = Path(inplace.__file__).read_text()
    read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *args, **kwargs:
                        source.replace('def _archive(', 'def _archive_unknown(')
                        if path == Path(inplace.__file__) else read(path, *args, **kwargs))
    assert inplace.execution_identity() != actual

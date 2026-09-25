"""Presentation-only compatibility is exact; unknown code still invalidates."""
import hashlib
from pathlib import Path

import pytest

from test_device_day import FakeModels, capture, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import file_hash
from visioncortex.device_day_runtime_identity import compatible_runtime_hash


@pytest.fixture
def device_config(default_config, tmp_path):
    from test_device_day import device_config as fixture
    return fixture.__wrapped__(default_config, tmp_path)


def test_reviewed_json_encoding_requires_the_exact_contract(tmp_path):
    from visioncortex import device_day_contract
    path = Path(device_day_contract.__file__)
    checksum = file_hash(path)
    assert compatible_runtime_hash(path, checksum) == (
        '619f42061083ce5fe3c1d5b0d6637e1b98453f781c161439b1198de73a68f7e8')
    changed = hashlib.sha256(path.read_bytes() + b'\n# unknown contract change').hexdigest()
    assert compatible_runtime_hash(path, changed) == changed
    assert compatible_runtime_hash(tmp_path / path.name, checksum) == checksum


def test_reviewed_audit_backend_requires_the_exact_helper(tmp_path, monkeypatch):
    from visioncortex import device_day_audit, device_day_models
    backend = Path(device_day_models.__file__)
    helper = Path(device_day_audit.__file__)
    backend_hash, helper_hash = file_hash(backend), file_hash(helper)
    canonical = '7c04a7a02843f482f30e5524a1e84cbbaea21278b03fbe8d494789fd24e1b3cd'
    assert compatible_runtime_hash(backend, backend_hash) == canonical
    assert compatible_runtime_hash(helper, helper_hash) is None
    for path, checksum in ((backend, backend_hash), (helper, helper_hash)):
        changed = hashlib.sha256(path.read_bytes() + b'\n# unknown edit').hexdigest()
        assert compatible_runtime_hash(path, changed) == changed
        assert compatible_runtime_hash(tmp_path / path.name, checksum) == checksum
    read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes',
                        lambda path: read(path) + (b'\n# changed helper' if path == helper else b''))
    assert compatible_runtime_hash(backend, backend_hash) == backend_hash


@pytest.mark.parametrize('stage', ['vision', 'understanding'])
def test_unknown_audit_helper_invalidates_the_stage(device_config, monkeypatch, stage):
    capture(device_config)
    record, _ = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    before = runner._key(stage, record, {})
    retained = runner._key('retention', record, record)
    original = runner._hash
    monkeypatch.setattr(runner, '_hash',
                        lambda path: 'f' * 64 if Path(path).name == 'device_day_audit.py' else original(path))
    assert runner._key(stage, record, {}) != before
    assert runner._key('retention', record, record) == retained


@pytest.mark.parametrize('coverage', [False, True])
@pytest.mark.parametrize('stage', ['vision', 'understanding'])
def test_cache_lifetime_revision_keeps_exact_parent_stage_keys(device_config, monkeypatch, coverage, stage):
    from visioncortex import device_day_models
    capture(device_config)
    record, _ = item_and_layout(device_config)
    device_config['device_day']['understanding_coverage_since_us'] = (
        record['recording_start_us'] if coverage else None)
    runner = DeviceDayRunner(device_config, FakeModels())
    actual_key = runner._key(stage, record, {})
    original = runner._hash
    parent = 'de18f0bfdf127e0c3f1b5b732d150e53cca4c01a95b1146540d4b23ea0adaba5'
    monkeypatch.setattr(runner, '_hash',
                        lambda path: parent if Path(path).name == 'device_day_models.py' else original(path))
    assert runner._key(stage, record, {}) == actual_key
    unknown = hashlib.sha256(Path(device_day_models.__file__).read_bytes() + b'\n# unknown edit').hexdigest()
    monkeypatch.setattr(runner, '_hash',
                        lambda path: unknown if Path(path).name == 'device_day_models.py' else original(path))
    assert runner._key(stage, record, {}) != actual_key


def test_completed_legacy_vision_is_reused_without_rewriting_its_receipt(device_config, monkeypatch):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    previous = DeviceDayRunner(device_config, FakeModels())
    # Exact pre-publication source hashes; the old recipe had no helper entry.
    # Its reviewed current helper maps to None and is removed from the code list.
    old_hashes = {
        'device_day_models.py': '47b7460f97d5c671894b49dfdf5a629813799fa7b9dde73b6ab5a02ad7303fd6',
        'device_day_contract.py': '619f42061083ce5fe3c1d5b0d6637e1b98453f781c161439b1198de73a68f7e8',
    }
    original = previous._hash
    monkeypatch.setattr(previous, '_hash', lambda path: old_hashes.get(Path(path).name) or original(path))
    assert previous.process(record, stage='retention')['status'] == 'completed'
    assert previous.process(record, stage='vision')['status'] == 'completed'
    path = previous._receipt(layout, record, 'vision')
    receipt = path.read_bytes()
    backend = FakeModels()
    current = DeviceDayRunner(device_config, backend)
    assert current.process(record, stage='vision')['status'] == 'completed'
    assert backend.vision_calls == 0
    assert path.read_bytes() == receipt


@pytest.mark.parametrize('phase', ['coarse', 'fine'])
def test_audit_publication_and_json_serializer_do_not_enter_model_scan_identity(
        default_config, tmp_path, monkeypatch, phase):
    from test_scan_dependencies import setup as fixture
    from visioncortex.device_day_models import DeviceDayModels
    config, retention, info = fixture.__wrapped__(default_config, tmp_path)
    backend = DeviceDayModels(config)
    before = backend.scan_identity(retention, info, phase, [(0, 1000)], 2, 640)
    original = backend._identity_hash
    presentation = {'device_day_contract.py', 'device_day_models.py', 'device_day_audit.py'}
    monkeypatch.setattr(backend, '_identity_hash',
                        lambda path: 'changed-presentation' if path.name in presentation else original(path))
    assert backend.scan_identity(retention, info, phase, [(0, 1000)], 2, 640) == before

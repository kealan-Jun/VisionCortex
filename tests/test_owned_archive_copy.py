"""Owned local fixtures: SMB timestamp-refresh simulation, no NAS/model run."""
import hashlib
from contextlib import nullcontext
from pathlib import Path

import pytest

from visioncortex import device_day_inputs as inputs


@pytest.fixture
def owned(tmp_path, monkeypatch):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'sealed bytes' * 100)
    reference = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'size_bytes': source.stat().st_size}
    sealed = {'original_path': str(source), 'resolved_path': str(source.resolve()),
              'retained': reference, 'identity': inputs.identity(source)}
    monkeypatch.setattr(inputs, 'slot', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(inputs, 'pace_copy', lambda *args: None)
    return source, sealed, tmp_path / 'archive' / 'copy.bin'


def test_owned_copy_retries_metadata_refresh_and_publishes_exact_bytes(owned, monkeypatch):
    source, sealed, target = owned
    identity, opened = inputs.identity, Path.open
    observations, reads = [], []

    def refreshed(path):
        value = identity(path)
        if Path(path).suffix == '.partial':
            observations.append(path)
            if len(observations) <= 2:
                value[-2:] = [n - 1000 for n in value[-2:]]
        return value

    def counted(path, *args, **kwargs):
        if path.suffix == '.partial' and args and args[0] == 'rb':
            reads.append(path)
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(inputs, 'identity', refreshed)
    monkeypatch.setattr(Path, 'open', counted)
    inputs.copy_sealed({}, sealed, target)
    assert len(reads) == 2
    assert target.read_bytes() == source.read_bytes()
    assert not list(target.parent.glob('*.partial'))


@pytest.mark.parametrize('failure', ['timestamps', 'inode', 'digest', 'size'])
def test_owned_copy_rejects_unstable_or_modified_temporary(owned, monkeypatch, failure):
    _, sealed, target = owned
    identity, opened = inputs.identity, Path.open
    calls, reads = [], []

    def changed(path):
        value = identity(path)
        if Path(path).suffix == '.partial':
            calls.append(path)
            if failure == 'timestamps':
                value[-1] += len(calls)
            elif failure == 'inode' and len(calls) >= 3:
                value[1] += 1
            elif failure == 'size':
                value[2] += 1
        return value

    def counted(path, *args, **kwargs):
        if path.suffix == '.partial' and args and args[0] == 'rb':
            reads.append(path)
            if failure == 'digest':
                with opened(path, 'r+b') as handle:
                    handle.write(b'X')
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(inputs, 'identity', changed)
    monkeypatch.setattr(Path, 'open', counted)
    with pytest.raises(inputs.InputChanged):
        inputs.copy_sealed({}, sealed, target)
    assert len(reads) == {'timestamps': 3, 'inode': 1, 'digest': 1, 'size': 0}[failure]
    assert not target.exists()
    assert not list(target.parent.glob('*.partial'))


def test_source_metadata_change_still_rejected_without_owned_retry(owned, monkeypatch):
    source, sealed, target = owned
    identity, calls = inputs.identity, []

    def changed(path):
        value = identity(path)
        if Path(path) == source:
            calls.append(path)
            if len(calls) >= 2:
                value[-1] += 1
        return value

    monkeypatch.setattr(inputs, 'identity', changed)
    monkeypatch.setattr(inputs, '_verify_owned_copy', lambda *args: pytest.fail('source must stay strict'))
    with pytest.raises(inputs.InputChanged, match='Capture input changed during archival copy'):
        inputs.copy_sealed({}, sealed, target)
    assert not target.exists()


def test_preexisting_target_verification_is_not_relaxed(owned, monkeypatch):
    source, sealed, target = owned
    target.parent.mkdir()
    target.write_bytes(source.read_bytes())
    monkeypatch.setattr(inputs, '_verify_owned_copy', lambda *args: pytest.fail('not our temporary copy'))
    monkeypatch.setattr(inputs, 'verify_content', lambda *args: False)
    with pytest.raises(inputs.InputChanged, match='Immutable MetaVideo destination collision'):
        inputs.copy_sealed({}, sealed, target)
    assert target.read_bytes() == source.read_bytes()


def test_global_content_hash_still_rejects_metadata_refresh(owned, monkeypatch):
    source, _, _ = owned
    identity, calls = inputs.identity, []

    def changed(path):
        value = identity(path)
        calls.append(path)
        value[-1] += len(calls)
        return value

    monkeypatch.setattr(inputs, 'identity', changed)
    with pytest.raises(inputs.InputChanged, match='Input changed during content verification'):
        inputs.hash_content(source)


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt', 'understanding', 'report'])
def test_owned_copy_fix_preserves_exact_completed_stage_key(default_config, tmp_path, monkeypatch, stage):
    from test_device_day import device_config, capture, item_and_layout, FakeModels
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.device_day_inplace import active
    from visioncortex.device_day_runtime_identity import compatible_runtime_hash
    config = device_config.__wrapped__(default_config, tmp_path)
    config['device_day']['inplace_preprocessing'] = True
    capture(config)
    record, _ = item_and_layout(config)
    runner = DeviceDayRunner(config, FakeModels())
    assert active(runner, record, initialize=True)
    before = runner._key(stage, record, {})
    original = runner._hash
    parent = '6a92d972cc2721f852d5774be6a2ea771eaf1a594f389778ebab0d691e6def20'
    monkeypatch.setattr(runner, '_hash', lambda path: parent if Path(path).name == 'device_day_inputs.py' else original(path))
    assert runner._key(stage, record, {}) == before
    path = Path(inputs.__file__)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert compatible_runtime_hash(path, checksum) == parent
    assert compatible_runtime_hash(tmp_path / path.name, checksum) == checksum
    unknown = hashlib.sha256(path.read_bytes() + b'\n# unrelated edit').hexdigest()
    monkeypatch.setattr(runner, '_hash', lambda path: unknown if Path(path).name == 'device_day_inputs.py' else original(path))
    assert runner._key(stage, record, {}) != before

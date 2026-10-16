"""Owned local fixtures model remounts; no NAS or model-quality claims."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_device_day import FakeModels, capture, device_config as device_config_fixture, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, digest, read_json
from visioncortex.device_day_inputs import binding, input_path
from visioncortex.publication_journal import PublicationJournal, reconcile, revalidate_remounted_seals

device_config = device_config_fixture


@pytest.fixture
def remounted(device_config):
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    for stage in ('vision', 'stt', 'retention'):
        assert runner.process(record, stage=stage)['status'] == 'completed'
    previous = {}
    for stage in ('vision', 'stt'):
        path = input_path(runner, record, stage)
        seal = read_json(path)
        for source in seal['sources']:
            source['identity'][0] += 100  # Different mount number only.
        atomic_json(path, seal)
        previous[stage] = seal
    return runner, record, layout, backend, previous


def test_replay_revalidates_content_and_keeps_execution_receipts(remounted):
    runner, record, layout, backend, previous = remounted
    receipt_paths = [runner._receipt(layout, record, s) for s in ('retention', 'vision', 'stt')]
    before = [p.read_bytes() for p in receipt_paths]
    PublicationJournal(runner.runtime_root).begin(record)
    assert reconcile(runner) == 1
    assert [p.read_bytes() for p in receipt_paths] == before
    assert backend.vision_calls == 1 and backend.semantic_calls == 0
    for stage, old in previous.items():
        path = input_path(runner, record, stage)
        updated = read_json(path)
        assert binding(updated) == binding(old)
        assert updated['mount_revalidation']['previous_seal_digest'] == digest(old)
        assert read_json(path.parent / 'history' / f'{path.stem}-{digest(old)}.json') == old
    assert revalidate_remounted_seals(runner, record) == 0


@pytest.mark.parametrize('change', ['inode', 'timestamp', 'hash', 'signature', 'camera', 'resolved_path'])
def test_remount_recovery_refuses_other_changes_without_rewriting_seals(remounted, change):
    runner, record, _, _, _ = remounted
    path = input_path(runner, record, 'vision')
    seal = read_json(path)
    if change == 'inode':
        seal['sources'][0]['identity'][1] += 1
    elif change == 'timestamp':
        seal['sources'][0]['identity'][3] += 1
    elif change == 'hash':
        seal['sources'][0]['sha256'] = '0' * 64
    elif change == 'signature':
        seal['source_signature'] = 'other-source'
    elif change == 'camera':
        seal['recording']['camera_key'] = 'other-camera'
    else:
        seal['sources'][0]['resolved_path'] = str(Path('/other/source'))
    atomic_json(path, seal)
    paths = [input_path(runner, record, s) for s in ('vision', 'stt')]
    before = [p.read_bytes() for p in paths]
    with pytest.raises(ValueError):
        revalidate_remounted_seals(runner, record)
    assert [p.read_bytes() for p in paths] == before


def test_changed_source_during_hash_prevents_any_seal_update(remounted, monkeypatch):
    runner, record, _, _, _ = remounted
    from visioncortex import device_day_inputs
    original = device_day_inputs.verify_content
    paths = [input_path(runner, record, s) for s in ('vision', 'stt')]
    before = [p.read_bytes() for p in paths]
    changed = False

    def verify(path, reference):
        nonlocal changed
        result = original(path, reference)
        if not changed:
            changed = True
            path.write_bytes(path.read_bytes() + b'changed-during-verification')
        return result

    monkeypatch.setattr(device_day_inputs, 'verify_content', verify)
    with pytest.raises(ValueError):
        revalidate_remounted_seals(runner, record)
    assert [p.read_bytes() for p in paths] == before


def test_invalid_second_seal_cannot_partially_rebind_the_first(remounted):
    runner, record, _, _, _ = remounted
    path = input_path(runner, record, 'stt')
    seal = deepcopy(read_json(path))
    seal['sources'][0]['identity'][1] += 1
    atomic_json(path, seal)
    first = input_path(runner, record, 'vision').read_bytes()
    with pytest.raises(ValueError):
        revalidate_remounted_seals(runner, record)
    assert input_path(runner, record, 'vision').read_bytes() == first

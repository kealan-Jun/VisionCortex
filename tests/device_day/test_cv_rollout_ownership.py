"""CV rollout reads pinned parents while retaining the actual vision producer."""
from types import SimpleNamespace
import json

import pytest

from visioncortex.device_day_publication_worker import publication_runner
from visioncortex.device_day_retention_worker import vision_runner


class ParentOwner:
    def __init__(self, setting):
        self.value, self.closed = setting, False

    def key(self, stage, recording, inputs, **kwargs):
        assert stage in {'retention', 'vision', 'stt', 'understanding', 'report'}
        return self.value

    def close(self):
        self.closed = True


class Base:
    def __init__(self, config, backend=None):
        self.settings, self.root, self.backend = config['device_day'], config['root'], backend
        self._completed_execution_bindings = SimpleNamespace(historical_candidate=lambda *args: None)

    def _key(self, *args, **kwargs):
        return 'b'*64

    def process(self, recording, *, stage, **kwargs):
        return {'stage': stage}

    def layout(self, recording):
        return self.root

    def _receipt(self, layout, recording, stage):
        return layout/(stage+'.json')


def test_vision_only_runner_cannot_execute_a_frozen_parent_or_paid_stage(tmp_path, monkeypatch):
    monkeypatch.setattr('visioncortex.device_day_publication_worker.FrozenReceiptKeyOwner', ParentOwner)
    config = {'root': tmp_path, 'device_day': {'vision_prerequisite_key_owner': 'a'*64}}
    runner = vision_runner(config, Base)
    assert runner._key('retention', {}, {}) == 'a'*64
    assert runner._key('vision', {}, {}) == 'b'*64
    assert runner.process({}, stage='vision') == {'stage': 'vision'}
    for stage in ('all', 'retention', 'stt', 'understanding', 'report'):
        with pytest.raises(ValueError, match='Vision-only'):
            runner.process({}, stage=stage)
    runner.close()
    assert runner.prerequisite_owner.closed


@pytest.mark.parametrize('observed,expected', [(None, 'a'), ('a', 'a'), ('b', 'b'), ('c', 'a')])
def test_publication_never_accepts_an_arbitrary_observed_vision_key(tmp_path, monkeypatch, observed, expected):
    monkeypatch.setattr('visioncortex.device_day_publication_worker.FrozenReceiptKeyOwner', ParentOwner)
    monkeypatch.setattr('visioncortex.device_day.DeviceDayRunner', Base)
    config = {'root': tmp_path, 'device_day': {'publication_receipt_key_owner': 'a'*64,
                                              'publication_vision_key_owner': 'b'*64}}
    if observed is not None:
        (tmp_path/'vision.json').write_text(json.dumps({'key': observed*64}))
    runner = publication_runner(config)
    assert type(runner.backend) is object
    assert runner._key('vision', {}, {}) == expected*64
    assert runner._key('retention', {}, {}) == 'a'*64
    with pytest.raises(RuntimeError, match='cannot invoke models'):
        runner._backend()
    with pytest.raises(RuntimeError, match='cannot execute stages'):
        runner.process({}, stage='vision')
    runner.close()
    assert runner.receipt_key_owner.closed and runner.vision_key_owner.closed

import json

import pytest

from visioncortex.device_day_bindings import CompletedBindingInvalid
from visioncortex.device_day_retention_worker import stt_runner, storage_runner
from visioncortex.device_day_contract import digest
from test_cv_rollout_ownership import Base


class Owner:
    def __init__(self, setting):
        self.closed = False

    def key(self, stage, record, inputs, **kwargs):
        return 'a'*64 if stage == 'retention' or inputs == {'original': True} else 'c'*64

    def close(self):
        self.closed = True


class Reader(Base):
    def _accepts_receipt(self, receipt, key):
        return receipt.get('key') == key

    def _load_verified(self, path, key, layout):
        value = json.loads(path.read_text())
        return value if value.get('files_valid') else None


def test_storage_rollout_reads_old_completed_key_but_never_executes_under_it(tmp_path, monkeypatch):
    monkeypatch.setattr('visioncortex.device_day_publication_worker.FrozenReceiptKeyOwner', Owner)
    runner = stt_runner({'root': tmp_path, 'device_day': {'stt_completed_key_owner': {}}}, Reader)
    # Explicit configuration is required; an empty setting keeps the base runner.
    assert type(runner) is Reader
    runner = stt_runner({'root': tmp_path, 'device_day': {'stt_completed_key_owner': {'pinned': True}}}, Reader)
    assert runner._key('retention', {}, {}) == 'a'*64
    assert runner._key('stt', {}, {}) == 'b'*64  # fresh build owns new calls
    saved = {'status': 'completed', 'key': 'a'*64, 'files_valid': True}
    path = tmp_path/'stt.json'
    path.write_text(json.dumps(saved))
    assert runner._key('stt', {}, {'original': True}) == 'a'*64
    assert runner._load_verified(path, 'a'*64, tmp_path) == saved
    assert runner.reserved_completions[(str(path), 'a'*64)] == digest(saved)
    assert runner._key('stt', {}, {'changed': True}) == 'b'*64
    path.write_text(json.dumps(saved | {'files_valid': False}))
    with pytest.raises(CompletedBindingInvalid, match='no provider fallback'):
        runner._load_verified(path, 'a'*64, tmp_path)
    for stage in ('all', 'retention', 'vision', 'understanding', 'report'):
        with pytest.raises(ValueError, match='cannot execute another producer'):
            runner.process({}, stage=stage)
    runner.close()
    assert runner.previous_owner.closed


def test_report_parent_reads_completed_new_stt_without_executing_that_stage(tmp_path, monkeypatch):
    monkeypatch.setattr('visioncortex.device_day_publication_worker.FrozenReceiptKeyOwner', Owner)
    settings = {'report_completed_key_owner': {'pinned': True},
                'storage_parent_key_owners': {'stt': {'pinned_new_stt': True}}}
    runner = storage_runner({'root': tmp_path, 'device_day': settings}, Reader, 'report')
    (tmp_path/'stt.json').write_text(json.dumps({'key': 'a'*64}))
    assert runner._key('stt', {}, {'original': True}) == 'a'*64
    assert runner._key('report', {}, {}) == 'b'*64
    with pytest.raises(ValueError, match='cannot execute another producer'):
        runner.process({}, stage='stt')
    runner.close()

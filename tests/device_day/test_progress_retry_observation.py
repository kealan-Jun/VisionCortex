"""Repair owner status is local observation, not completed model evidence."""
import json

import pytest

from visioncortex import device_day_progress as progress


def observation(**changes):
    return {'available': True, 'owner_state': 'verified', 'status': 'running', 'phase': 'checking',
            'configured_enabled': True, 'allowed_classes': ['provider_binding', 'vision_nms'],
            'pid': 123, 'granted_count': 2, 'rejected_count': 1, 'completed_tick_count': 3,
            'last_result': {'status': 'verification_rejected', 'error_type': 'ValueError',
                            'scanned': 7, 'outside_scope': 2, 'unknown_or_disabled': 1, 'cooling': 3},
            'proof_scope': 'verified_retry_budget_grants_not_stage_completion', **changes}


def test_verified_owner_shows_phase_budget_and_narrow_rejection_scope():
    html = progress.render_retry_observation(observation())
    for expected in ('已启用', '已核验进程正在核验修复条件', '进程 123', '授权 2 次',
                     '核验未通过 1 次', 'ValueError', '2 不在自动修复范围',
                     '1 原因未识别或未授权', '3 等待冷却', '不代表模型已执行'):
        assert expected in html


@pytest.mark.parametrize('owner', ['stale', 'unverified', 'stopped', 'failed'])
def test_unverified_or_stale_owner_never_claims_currently_running(owner):
    html = progress.render_retry_observation(observation(owner_state=owner))
    assert '已核验进程正在' not in html
    assert '进程 123' not in html
    assert '最近服务快照累计' in html


def test_verified_disabled_service_preserves_actual_configuration_and_escapes_labels():
    html = progress.render_retry_observation(observation(
        status='disabled', configured_enabled=False,
        allowed_classes=['<custom>'], last_result={'status': 'verification_rejected', 'error_type': '<unsafe>'}))
    assert '自动修复未启用' in html and '正在核验' not in html
    assert '&lt;unsafe&gt;' in html and '&lt;custom&gt;' in html
    assert '<unsafe>' not in html


@pytest.mark.parametrize(('reason', 'expected'), [
    ('parent_queue_not_completed_stt', '录音识别：前序队列尚未完成'),
    ('parent_receipt_or_artifact_unverified_retention', '归档：前序回执或产物尚未通过核验'),
    ('archived_source_snapshot_or_stable_bytes_unverified', '归档原片快照或稳定字节核验未通过'),
    ('current_queue_recipe_changed_normal_admission_required', '当前执行配方已变化，须走正常任务准入'),
])
def test_fixed_rejection_reasons_show_the_unmet_gate(reason, expected):
    html = progress.render_retry_observation(observation(
        last_result={'status': 'verification_rejected', 'reason': reason}))
    assert '阻塞原因：' + expected in html
    stale = progress.render_retry_observation(observation(last_result={'status': 'waiting', 'reason': reason}))
    assert '阻塞原因：' not in stale, 'a previous rejection cannot describe a new waiting round'


def test_snapshot_with_absent_service_does_not_claim_enabled_config_is_running(tmp_path):
    config = {'storage': {'local_runtime_root': str(tmp_path)},
              'device_day': {'retry_repair': {'enabled': True}}}
    value = progress.snapshot(config)
    assert value['retry_repair']['available'] is False
    assert '运行状态未核实' in value['retry_repair_html']
    assert not (tmp_path / 'device-day').exists(), 'observation cannot create queues/service output'


def test_cached_progress_refreshes_repair_owner_without_collecting_queues(tmp_path, monkeypatch):
    config = {'storage': {'local_runtime_root': str(tmp_path)}}
    path = tmp_path / 'device-day/ProgressSnapshot.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'days': {}, 'observed_at': 1, 'retry_repair': {'status': 'old'}}))
    from visioncortex import device_day_consumers, device_day_retry_worker
    monkeypatch.setattr(device_day_consumers, 'consumer_snapshot', lambda config: {'observed_at': 2, 'stages': {}})
    calls = []
    fresh = observation()
    def read(config, *, now=None):
        calls.append(config)
        return fresh
    monkeypatch.setattr(device_day_retry_worker, 'retry_service_snapshot', read)
    monkeypatch.setattr(progress, 'render_consumers', lambda value: 'local owners')
    poller = progress.ProgressPoller(lambda: config)
    value = poller._collect()
    assert calls == [config]
    assert value['retry_repair'] == fresh
    assert '授权 2 次' in value['retry_repair_html']
    assert path.read_text() == json.dumps({'days': {}, 'observed_at': 1, 'retry_repair': {'status': 'old'}})
    assert not list(path.parent.glob('*.sqlite3'))

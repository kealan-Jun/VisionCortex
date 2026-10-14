"""Paid request recovery uses local receipts and mock transports only."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from visioncortex import scene_requests
from visioncortex.device_day_contract import read_json
from visioncortex.mllm import ArkAnalyzer
from visioncortex.multimodal_usage import UsageLedger
from visioncortex.provider_control import ProviderUnavailable
from visioncortex.sqlite_store import connection


def setup_window(default_config, tmp_path):
    config = deepcopy(default_config)
    config['storage'].update(local_runtime_root=str(tmp_path/'runtime'), local_cache_root=str(tmp_path/'cache'))
    config['mllm'].update(provider='synthetic', model='vision', max_retries=3, api_protocol='chat_completions',
                          base_url='https://example.test/v1', enabled=True)
    metadata = {'schema_version': 'visioncortex-device-day/1', 'segment_id': 'slice', 'activity': 'inactive',
                'start_ms': 0, 'end_ms': 1000, 'source_ref': {'sha256': 'a'*64},
                'frames': [], 'comments': [], 'protocol': None}
    return config, metadata, tmp_path/'Input.json', tmp_path/'Result.json'


def invoke(window, analyzer):
    config, metadata, request, result = window
    return scene_requests.window(config, analyzer, 'saved-test-prompt', metadata, [], request, result,
                                 lambda raw: {'validated': raw['status'] == 'completed'})


def known_result():
    return {'status': 'completed', 'attempts': 1, 'request_id': 'synthetic-request',
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30, 'cached_input_tokens': 0}}


class FakeAnalyzer:
    def __init__(self, result=None):
        self.calls = 0
        self.result = result

    def _call(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return deepcopy(self.result or known_result())


def test_process_loss_does_not_release_same_input_after_restart(default_config, tmp_path):
    window = setup_window(default_config, tmp_path)
    with pytest.raises(RuntimeError, match='process loss'):
        invoke(window, FakeAnalyzer(RuntimeError('process loss after submission')))
    restarted = FakeAnalyzer()
    with pytest.raises(ProviderUnavailable, match='submitted_unknown'):
        invoke(window, restarted)
    assert restarted.calls == 0
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    assert UsageLedger(window[0]).summary(day)['calls'] == 1


def test_unknown_transport_outcome_remains_blocked_on_replay(default_config, tmp_path):
    window = setup_window(default_config, tmp_path)
    failed = {'status': 'submitted_unknown', 'attempts': 1, 'usage': {'total_tokens': None},
              'attempt_receipts': [{'status': 'submitted_unknown', 'request_submitted': True}]}
    with pytest.raises(ProviderUnavailable, match='submitted_unknown'):
        invoke(window, FakeAnalyzer(failed))
    restarted = FakeAnalyzer()
    with pytest.raises(ProviderUnavailable, match='submitted_unknown'):
        invoke(window, restarted)
    assert restarted.calls == 0
    assert read_json(window[3])['model_result']['status'] == 'submitted_unknown'


def test_local_receipt_recovers_response_when_archive_publication_fails(default_config, tmp_path, monkeypatch):
    window = setup_window(default_config, tmp_path)
    original = scene_requests.atomic_json
    def unavailable(path, value):
        if path == window[3]:
            raise OSError('archive unavailable')
        return original(path, value)
    monkeypatch.setattr(scene_requests, 'atomic_json', unavailable)
    first = FakeAnalyzer()
    with pytest.raises(OSError, match='archive unavailable'):
        invoke(window, first)
    monkeypatch.setattr(scene_requests, 'atomic_json', original)
    second = FakeAnalyzer()
    assert invoke(window, second)[2]
    assert first.calls == 1 and second.calls == 0
    assert read_json(window[3])['model_result']['request_id'] == 'synthetic-request'


def test_read_timeout_never_starts_another_billable_attempt(default_config, tmp_path, monkeypatch):
    config = setup_window(default_config, tmp_path)[0]
    monkeypatch.setattr('visioncortex.mllm.model_api_key', lambda *args, **kwargs: 'synthetic-only')
    requests = []
    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout('synthetic timeout', request=request)
    analyzer = ArkAnalyzer(config)
    analyzer.client.close()
    analyzer.client = httpx.Client(transport=httpx.MockTransport(timeout))
    try:
        result = analyzer._call('prompt', {}, [])
    finally:
        analyzer.close()
    assert result['status'] == 'submitted_unknown'
    assert result['attempts'] == len(requests) == 1
    assert result['attempt_receipts'][0]['request_submitted']


def test_receipt_capacity_blocks_before_submission(default_config, tmp_path):
    window = setup_window(default_config, tmp_path)
    window[0]['mllm'].update(local_receipt_max_bytes=100, local_receipt_total_bytes=50)
    analyzer = FakeAnalyzer()
    with pytest.raises(ProviderUnavailable, match='receipt'):
        invoke(window, analyzer)
    assert analyzer.calls == 0


def test_claim_is_atomic_without_any_daily_budget(default_config, tmp_path):
    config = setup_window(default_config, tmp_path)[0]
    ledger = UsageLedger(config)
    def claim(_):
        try:
            return ledger.reserve('same-input')
        except ProviderUnavailable:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert len([result for result in pool.map(claim, range(4)) if result]) == 1


def test_legacy_pending_reservation_never_expires_into_paid_retry(default_config, tmp_path):
    config = setup_window(default_config, tmp_path)[0]
    ledger = UsageLedger(config)
    reservation = ledger.reserve('old-input')
    with connection(ledger.path) as db:
        db.execute('UPDATE calls SET started=0,day=?,status=? WHERE id=?', ('1970-01-01', 'pending', reservation))
        db.execute('DELETE FROM request_claims')
    with pytest.raises(ProviderUnavailable, match='submitted_unknown'):
        UsageLedger(config).reserve('old-input')


def test_incomplete_stream_never_starts_format_or_transport_retry(default_config, tmp_path, monkeypatch):
    config = setup_window(default_config, tmp_path)[0]
    config['mllm'].update(provider='aliyun', quality_mode='quality')
    monkeypatch.setattr('visioncortex.mllm.model_api_key', lambda *args, **kwargs: 'synthetic-only')
    requests = []
    def truncated(request):
        requests.append(request)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text='data: {"id":"incomplete","choices":[]}\n\n')
    analyzer = ArkAnalyzer(config)
    analyzer.client.close()
    analyzer.client = httpx.Client(transport=httpx.MockTransport(truncated))
    try:
        result = analyzer._call('prompt', {}, [])
    finally:
        analyzer.close()
    assert result['status'] == 'submitted_unknown'
    assert result['attempts'] == len(requests) == 1


def test_failed_http_response_preserves_billed_usage_in_the_actual_ledger(default_config, tmp_path, monkeypatch):
    window = setup_window(default_config, tmp_path)
    config = window[0]
    monkeypatch.setattr('visioncortex.mllm.model_api_key', lambda *args, **kwargs: 'synthetic-only')
    def billed_error(request):
        return httpx.Response(400, json={'id': 'billed-error', 'model': 'vision', 'error': 'synthetic failure',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                      'prompt_tokens_details': {'cached_tokens': 40}}})
    analyzer = ArkAnalyzer(config)
    analyzer.client.close()
    analyzer.client = httpx.Client(transport=httpx.MockTransport(billed_error))
    try:
        with pytest.raises(ValueError, match='failed'):
            invoke(window, analyzer)
    finally:
        analyzer.close()
    ledger = UsageLedger(config)
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    stats = ledger.summary(day)
    assert stats['known_tokens']['total_tokens'] == 120
    assert stats['failed_with_reported_usage'] == 1
    assert stats['request_ids'] == ['billed-error']
    with connection(ledger.path, readonly=True) as db:
        assert db.execute('SELECT input_key FROM calls').fetchone()[0] == scene_requests.request_key(config, 'saved-test-prompt', window[1])

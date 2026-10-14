"""Explicit synthetic prices and saved receipts; no paid requests."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from visioncortex.multimodal_usage import UsageLedger
from visioncortex.pricing_table import PricingTable
from visioncortex.provider_control import scope as account_scope
from visioncortex.usage_reconciliation import UsageReconciliationManager


def settings(tmp_path):
    return {'storage': {'local_runtime_root': str(tmp_path)},
            'mllm': {'provider': 'synthetic', 'model': 'vision', 'max_retries': 1, 'api_key_env': 'SYNTHETIC_KEY'},
            'pricing': {'providers': {'synthetic': {'models': {'vision': {
                'input_token_price': 0.001, 'output_token_price': 0.002,
                'cached_input_token_price': 0.0005, 'currency': 'CNY'}}}}}}


def known_usage(**extra):
    return {'provider': 'synthetic', 'model': 'vision',
            'known_tokens': {'input_tokens': 100, 'output_tokens': 20, 'cached_input_tokens': 40}, **extra}


def saved_call(config, *, status='completed'):
    ledger = UsageLedger(config)
    reservation = ledger.reserve('synthetic-window')
    ledger.finish(reservation, {'status': status, 'attempts': 1, 'request_id': 'synthetic-request',
        'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'cached_input_tokens': 40}})
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date()
    selected = {'account_key': account_scope(config['mllm'], 'account'), 'provider': 'synthetic', 'model': 'vision',
                'start_at': f'{day}T00:00:00+08:00', 'end_at': f'{day+timedelta(days=1)}T00:00:00+08:00'}
    vendor = {'scope': selected, 'request_ids': ['synthetic-request'], 'input_tokens': 100,
              'output_tokens': 20, 'cached_input_tokens': 40, 'total_tokens': 120,
              'attempts': 1, 'cost': 0.12, 'currency': 'CNY', 'multiplier': 1}
    return day.isoformat(), selected, vendor


def test_cached_input_is_counted_once_for_recorded_actual_model(tmp_path):
    cost = PricingTable(settings(tmp_path)).calculate_cost(known_usage())
    assert cost['input_cost'] == pytest.approx(0.060)
    assert cost['output_cost'] == pytest.approx(0.040)
    assert cost['cached_input_cost'] == pytest.approx(0.020)
    assert cost['total_cost'] == pytest.approx(0.120)


@pytest.mark.parametrize('usage', [known_usage(model='unpriced'), known_usage(unknown_attempts=1),
    known_usage(pending_calls=1), {'known_tokens': {'input_tokens': 1000}},
    known_usage(known_tokens={'input_tokens': 10, 'output_tokens': 20, 'cached_input_tokens': 11})])
def test_unknown_or_invalid_cost_is_never_zero_or_default_claude(tmp_path, usage):
    pricing = PricingTable(settings(tmp_path))
    assert pricing.calculate_cost(usage)['total_cost'] is None
    assert pricing.get_model_price('anthropic', 'claude-sonnet-4') is None


def test_missing_cached_rate_is_unknown_and_multiplier_is_applied(tmp_path):
    config = settings(tmp_path)
    rates = config['pricing']['providers']['synthetic']['models']['vision']
    rates.pop('cached_input_token_price')
    assert PricingTable(config).calculate_cost(known_usage())['total_cost'] is None
    rates.update(cached_input_token_price=0.0005, multiplier=2)
    assert PricingTable(config).calculate_cost(known_usage())['total_cost'] == pytest.approx(0.24)


def test_ledger_uses_historical_model_and_accounts_for_failed_billed_usage(tmp_path):
    config = settings(tmp_path)
    day, _, _ = saved_call(config, status='failed')
    config['mllm'].update(provider='different-provider', model='different-model')
    usage = UsageLedger(config, readonly=True).summary(day)
    assert usage['per_model'][0]['provider'] == 'synthetic'
    assert usage['known_tokens']['total_tokens'] == 120
    assert PricingTable(config).calculate_cost(usage)['total_cost'] == pytest.approx(0.120)


def test_same_tokens_and_different_money_is_mismatch(tmp_path):
    config = settings(tmp_path)
    day, selected, vendor = saved_call(config)
    manager = UsageReconciliationManager(tmp_path, config)
    assert manager.reconcile_daily(day, vendor, scope=selected).status == 'MATCH'
    vendor['cost'] = 0.20
    report = manager.reconcile_daily(day, vendor, scope=selected)
    assert report.status == 'MISMATCH'
    assert report.differences.token_diff == 0
    assert report.differences.cost_diff == pytest.approx(0.08)
    vendor['cost'] = 0.12000001
    assert manager.reconcile_daily(day, vendor, scope=selected).status == 'MISMATCH'


@pytest.mark.parametrize('change', ['request', 'account', 'end', 'cached', 'multiplier'])
def test_scope_and_rate_evidence_is_required_for_match(tmp_path, change):
    config = settings(tmp_path)
    day, selected, vendor = saved_call(config)
    if change == 'request':
        vendor['request_ids'] = ['unrelated-request']
    elif change == 'account':
        vendor['scope'] = {**selected, 'account_key': 'unrelated-account'}
    elif change == 'end':
        vendor['scope'] = {**selected, 'end_at': f'{day}T13:53:04+08:00'}
    elif change == 'cached':
        vendor.pop('cached_input_tokens')
    else:
        vendor.pop('multiplier')
    report = UsageReconciliationManager(tmp_path, config).reconcile_daily(day, vendor, scope=selected)
    assert report.status == 'PARTIAL'


def test_unknown_price_and_pending_vendor_are_separate(tmp_path):
    config = settings(tmp_path)
    day, selected, vendor = saved_call(config)
    config['pricing'] = {}
    manager = UsageReconciliationManager(tmp_path, config)
    assert manager.reconcile_daily(day, vendor, scope=selected).status == 'UNKNOWN_PRICE'
    assert manager.reconcile_daily(day).status == 'PENDING_VENDOR'


def test_readonly_empty_report_does_not_create_a_database(tmp_path):
    config = settings(tmp_path)
    manager = UsageReconciliationManager(tmp_path, config)
    assert manager.reconcile_daily('2026-10-14').status == 'PENDING_VENDOR'
    assert not manager.usage_ledger.path.exists()


def test_per_attempt_missing_token_fields_cannot_be_hidden_in_aggregated_usage(tmp_path):
    config = settings(tmp_path)
    ledger = UsageLedger(config)
    reservation = ledger.reserve('partial-attempt-fields')
    ledger.finish(reservation, {'status': 'failed', 'attempts': 2,
        'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 200, 'cached_input_tokens': 40},
        'attempt_receipts': [{'usage': {'total_tokens': 80}}, {'usage': {
            'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'cached_input_tokens': 40}}]})
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    usage = ledger.summary(day)
    assert usage['unknown_attempts'] == 0
    assert usage['per_model'][0]['missing_token_counts'] == 1
    assert PricingTable(config).calculate_cost(usage)['total_cost'] is None


def test_request_scope_refuses_timezone_ambiguous_timestamps(tmp_path):
    config = settings(tmp_path)
    day, selected, _ = saved_call(config)
    selected['end_at'] = f'{day}T13:53:04'
    with pytest.raises(ValueError, match='timezone'):
        UsageLedger(config, readonly=True).summary(day, scope=selected)


def test_billing_uses_response_model_instead_of_unrelated_selected_price(tmp_path):
    config = settings(tmp_path)
    ledger = UsageLedger(config)
    reservation = ledger.reserve('gateway-returned-different-model')
    ledger.finish(reservation, {'status': 'completed', 'attempts': 1, 'response_model': 'unpriced-model',
        'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'cached_input_tokens': 40}})
    day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    usage = ledger.summary(day)
    assert usage['per_model'][0]['model'] == 'unpriced-model'
    assert PricingTable(config).calculate_cost(usage)['total_cost'] is None

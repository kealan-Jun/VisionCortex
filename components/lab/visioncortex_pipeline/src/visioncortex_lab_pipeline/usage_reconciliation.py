"""Offline billing reconciliation of explicit matching account/request scopes."""
from dataclasses import dataclass, field

from .multimodal_usage import UsageLedger
from .pricing_table import PricingTable, amount


@dataclass
class UsageDifference:
    token_diff: int | None = None
    attempts_diff: int | None = None
    cost_diff: float | None = None
    percentage_diff: float | None = None


@dataclass
class ReconciliationReport:
    date: str
    status: str = 'PENDING_VENDOR'
    local_usage: dict = field(default_factory=dict)
    vendor_usage: dict | None = None
    cost: dict = field(default_factory=dict)
    differences: UsageDifference | None = None
    anomalies: list = field(default_factory=list)


class UsageReconciliationManager:
    def __init__(self, runtime_root, config):
        if str(runtime_root) != str(config['storage']['local_runtime_root']):
            raise ValueError('Billing runtime root differs from the configured local ledger')
        self.usage_ledger = UsageLedger(config, readonly=True)
        self.pricing = PricingTable(config)

    def _compare_usage(self, local, vendor, cost=None):
        costs = cost or {}
        local_total = local.get('known_tokens', {}).get('total_tokens')
        vendor_total = vendor.get('total_tokens')
        token_diff = vendor_total-local_total if type(local_total) is int and type(vendor_total) is int else None
        attempts = vendor.get('attempts')
        local_attempts = local.get('attempts_or_reservations')
        attempts_diff = attempts-local_attempts if type(attempts) is int and type(local_attempts) is int else None
        cost_diff = None
        if costs.get('total_cost') is not None and vendor.get('cost') is not None and vendor.get('currency') == costs.get('currency'):
            cost_diff = float(amount(vendor['cost'])-amount(costs['total_cost']))
        return UsageDifference(token_diff, attempts_diff, cost_diff,
                               100*token_diff/local_total if token_diff is not None and local_total else None)

    def reconcile_daily(self, target_date, vendor_usage=None, *, scope=None):
        local = self.usage_ledger.summary(target_date, scope=scope)
        report = ReconciliationReport(target_date, local_usage=local, vendor_usage=vendor_usage,
                                      cost=self.pricing.calculate_cost(local))
        if vendor_usage is None:
            return report
        report.differences = self._compare_usage(local, vendor_usage, report.cost)
        reasons = {row.get('reason') for row in report.cost['breakdown']}
        if reasons.intersection({'actual_model_price_not_configured', 'cached_input_price_not_configured', 'invalid_price_or_token_counts'}):
            report.status = 'UNKNOWN_PRICE'
            return report
        required = {'account_key', 'provider', 'model', 'start_at', 'end_at'}
        vendor_scope = vendor_usage.get('scope')
        scope_matches = (isinstance(scope, dict) and required.issubset(scope)
                         and all(scope[k] is not None for k in required) and vendor_scope == scope)
        ids = vendor_usage.get('request_ids')
        ids_match = (isinstance(ids, list) and all(isinstance(item, str) and item for item in ids)
                     and len(ids) == len(set(ids))
                     and sorted(ids) == local['request_ids']
                     and len(ids) == local['attempts_or_reservations'])
        diff = report.differences
        comparable = (scope_matches and ids_match and not local['pending_calls'] and not local['unknown_attempts']
                      and report.cost['total_cost'] is not None and diff.token_diff is not None
                      and diff.attempts_diff is not None and diff.cost_diff is not None)
        fields = ('input_tokens', 'output_tokens', 'cached_input_tokens')
        if not comparable or any(type(vendor_usage.get(name)) is not int for name in fields):
            report.status = 'PARTIAL'
            report.anomalies.append({'reason': 'account_model_time_request_or_usage_scope_unresolved'})
            return report
        multiplier = {row['multiplier'] for row in report.cost['breakdown']}
        if vendor_usage.get('multiplier') is None:
            report.status = 'PARTIAL'
            report.anomalies.append({'reason': 'vendor_multiplier_missing'})
            return report
        exact_tokens = all(vendor_usage[name] == local['known_tokens'][name] for name in fields)
        report.status = ('MATCH' if exact_tokens and diff.token_diff == 0 and diff.attempts_diff == 0
                         and amount(vendor_usage['cost']) == amount(report.cost['total_cost'])
                         and multiplier == {float(amount(vendor_usage['multiplier']))} else 'MISMATCH')
        return report

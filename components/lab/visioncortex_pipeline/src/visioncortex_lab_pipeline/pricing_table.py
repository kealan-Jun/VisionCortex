"""Price recorded provider/model usage with explicitly supplied rates only."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    model: str
    input_token_price: Decimal
    output_token_price: Decimal
    cached_input_token_price: Decimal | None
    currency: str
    multiplier: Decimal


def amount(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('A rate or amount must be an explicit finite nonnegative number')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid monetary value') from exc
    if not result.is_finite() or result < 0:
        raise ValueError('Invalid monetary value')
    return result


class PricingTable:
    def __init__(self, config):
        self.config = config

    def get_model_price(self, provider, model):
        rates = self.config.get('pricing', {}).get('providers', {}).get(provider, {}).get('models', {}).get(model)
        if rates is None:
            return None
        currency = rates.get('currency')
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isupper():
            raise ValueError('Configured rates require an explicit currency')
        inp = amount(rates['input_token_price'])
        cached = rates.get('cached_input_token_price')
        if cached is None and 'cached_input_discount' in rates:
            discount = amount(rates['cached_input_discount'])
            if discount > 1:
                raise ValueError('Invalid explicit cached input discount')
            cached = inp * discount
        return ModelPricing(provider, model, inp, amount(rates['output_token_price']),
                            amount(cached) if cached is not None else None, currency,
                            amount(rates.get('multiplier', 1)))

    def calculate_cost(self, usage):
        groups = usage.get('per_model')
        if groups is None:
            groups = [usage]
        breakdown = []
        for group in groups:
            row = {'provider': group.get('provider'), 'model': group.get('model'),
                   'account_key': group.get('account_key'), 'cost': None, 'currency': None,
                   'reason': 'actual_model_price_not_configured'}
            try:
                rate = self.get_model_price(row['provider'], row['model'])
                if rate is None:
                    breakdown.append(row)
                    continue
                tokens = group.get('known_tokens') or {}
                inp, out, cached = (tokens.get(k) for k in ('input_tokens', 'output_tokens', 'cached_input_tokens'))
                if (any(type(v) is not int or v < 0 for v in (inp, out, cached)) or cached > inp
                        or group.get('unknown_attempts') or group.get('pending_calls')
                        or group.get('missing_token_counts') or group.get('cached_input_usage_unknown')
                        or group.get('model_identity_unresolved')):
                    row['reason'] = 'usage_unresolved'
                elif cached and rate.cached_input_token_price is None:
                    row['reason'] = 'cached_input_price_not_configured'
                else:
                    costs = ((inp-cached)*rate.input_token_price*rate.multiplier,
                             out*rate.output_token_price*rate.multiplier,
                             cached*(rate.cached_input_token_price or Decimal(0))*rate.multiplier)
                    row.update(input_cost=float(costs[0]), output_cost=float(costs[1]),
                               cached_input_cost=float(costs[2]), cost=float(sum(costs)),
                               currency=rate.currency, multiplier=float(rate.multiplier), reason=None,
                               price_source='explicit_configuration')
            except (ValueError, KeyError, TypeError):
                row['reason'] = 'invalid_price_or_token_counts'
            breakdown.append(row)
        currencies = {row['currency'] for row in breakdown if row['currency']}
        complete = (all(row['cost'] is not None for row in breakdown) and len(currencies) <= 1
                    and not usage.get('unknown_attempts') and not usage.get('pending_calls'))
        result = {name: float(sum((amount(row.get(name, 0)) for row in breakdown), Decimal(0))) if complete else None
                  for name in ('input_cost', 'output_cost', 'cached_input_cost')}
        result.update(total_cost=float(sum((amount(row['cost']) for row in breakdown), Decimal(0))) if complete else None,
                      currency=next(iter(currencies)) if len(currencies) == 1 else None,
                      breakdown=breakdown, status='estimated' if complete else 'unknown',
                      billing_reconciliation='NOT_PROVEN')
        return result

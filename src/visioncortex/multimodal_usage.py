"""Local per-request accounting; reused outputs never create billable entries."""
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import time
import uuid
import json

from .provider_control import ProviderUnavailable
from .sqlite_store import connection


class SubmittedUnknown(ProviderUnavailable):
    """An existing submission needs receipt reconciliation before execution."""
    reason = 'submitted_unknown'


def validate(settings):
    for name in ('scene_daily_attempt_limit', 'scene_daily_token_limit', 'local_receipt_max_bytes', 'local_receipt_total_bytes'):
        value = settings.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'mllm.{name} must be a positive integer or null')
    if not isinstance(settings.get('compact_scene_metadata', False), bool):
        raise ValueError('mllm.compact_scene_metadata must be a boolean')


class UsageLedger:
    def __init__(self, config, *, readonly=False):
        self.config = config
        self.settings = config.get('mllm') or {}
        validate(self.settings)
        self.path = Path(config['storage']['local_runtime_root'])/'state'/'MultimodalUsage.sqlite3'
        if readonly:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY, day TEXT NOT NULL,
                  input_key TEXT NOT NULL, provider TEXT, model TEXT, started REAL NOT NULL,
                  ended REAL, reserved_attempts INTEGER NOT NULL, attempts INTEGER,
                  status TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
                  total_tokens INTEGER, cached_input_tokens INTEGER, unknown_attempts INTEGER);
                CREATE INDEX IF NOT EXISTS calls_by_day ON calls(day);''')
            db.execute('BEGIN IMMEDIATE')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(calls)')}
            for name in ('account_key', 'receipt_json', 'response_model'):
                if name not in columns:
                    db.execute(f'ALTER TABLE calls ADD COLUMN {name} TEXT')
            db.executescript('''CREATE INDEX IF NOT EXISTS calls_by_input ON calls(input_key);
                CREATE TABLE IF NOT EXISTS request_claims(input_key TEXT PRIMARY KEY,
                  reservation_id TEXT NOT NULL UNIQUE REFERENCES calls(id));
                CREATE TABLE IF NOT EXISTS local_receipts(reservation_id TEXT PRIMARY KEY REFERENCES calls(id),
                  input_key TEXT NOT NULL, payload TEXT NOT NULL, saved REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS receipts_by_input ON local_receipts(input_key);''')

    def reserve(self, input_key):
        today = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
        attempts = max(1, int(self.settings.get('max_retries', 1)))
        identifier = uuid.uuid4().hex
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            # The database claim survives process death and is independent of
            # the OS window lock. Old pending reservations stay unresolved;
            # migration never guesses that a paid request was not submitted.
            unresolved = db.execute('SELECT id FROM calls WHERE input_key=? AND '
                '(ended IS NULL OR status=?) LIMIT 1', (input_key, 'submitted_unknown')).fetchone()
            claim = db.execute('SELECT reservation_id FROM request_claims WHERE input_key=?', (input_key,)).fetchone()
            if unresolved or claim:
                raise SubmittedUnknown('submitted_unknown: reconcile the existing request receipt before another submission')
            maximum = self.settings.get('local_receipt_max_bytes', 2*1024*1024)
            capacity = self.settings.get('local_receipt_total_bytes', 64*1024*1024)
            occupied = db.execute('SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM local_receipts').fetchone()[0]
            outstanding = db.execute('SELECT COUNT(*) FROM request_claims').fetchone()[0]
            if occupied + (outstanding+1)*maximum > capacity:
                raise ProviderUnavailable('Local paid-response receipt recovery capacity exhausted; work remains queued')
            used = db.execute('SELECT COALESCE(SUM(COALESCE(attempts,reserved_attempts)),0), '
                'COALESCE(SUM(total_tokens),0), COALESCE(SUM(COALESCE(unknown_attempts,reserved_attempts)),0) '
                'FROM calls WHERE day=?', (today,)).fetchone()
            cap = self.settings.get('scene_daily_attempt_limit')
            tokens = self.settings.get('scene_daily_token_limit')
            if cap is not None and used[0] + attempts > cap:
                raise ProviderUnavailable('Video understanding daily request budget reached; queued until next day')
            if tokens is not None and (used[1] >= tokens or used[2]):
                # Unknown/in-flight spend cannot be assumed free. With a token
                # threshold enabled only one outstanding call is admitted.
                raise ProviderUnavailable('Video understanding token budget reached or usage unresolved; work remains queued')
            from .provider_control import scope
            db.execute('INSERT INTO calls(id,day,input_key,provider,model,started,reserved_attempts,status,account_key) '
                       'VALUES(?,?,?,?,?,?,?,?,?)', (identifier,today,input_key,self.settings.get('provider'),
                       self.settings.get('model'),time.time(),attempts,'pending',scope(self.settings, 'account')))
            db.execute('INSERT INTO request_claims VALUES(?,?)', (input_key, identifier))
        return identifier

    def recover(self, input_key):
        """A locally received result can be published after storage recovers."""
        with connection(self.path, readonly=True) as db:
            row = db.execute('SELECT reservation_id,payload FROM local_receipts WHERE input_key=? '
                             'ORDER BY saved DESC LIMIT 1', (input_key,)).fetchone()
            return (row['reservation_id'], json.loads(row['payload'])) if row else None

    def published(self, identifier):
        # Only the temporary recovery payload is reclaimed. Calls and attempt
        # receipts remain in the accounting history.
        with connection(self.path) as db:
            db.execute('DELETE FROM local_receipts WHERE reservation_id=?', (identifier,))

    def submitted_unknown(self, identifier):
        with connection(self.path) as db:
            db.execute('UPDATE calls SET status=? WHERE id=? AND ended IS NULL', ('submitted_unknown', identifier))

    def received(self, identifier, result):
        """Persist the transport result on SSD before archive publication."""
        payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
        if len(payload.encode('utf-8')) > self.settings.get('local_receipt_max_bytes', 2*1024*1024):
            raise SubmittedUnknown('submitted_unknown: paid-response receipt exceeds the declared local recovery bound')
        with connection(self.path) as db:
            row = db.execute('SELECT input_key FROM calls WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise ValueError('Unknown paid-request reservation')
            db.execute('INSERT OR REPLACE INTO local_receipts VALUES(?,?,?,?)',
                       (identifier, row['input_key'], payload, time.time()))

    def finish(self, identifier, result):
        usage = result.get('usage') or {}
        reported = result.get('attempt_receipts') or []
        count = result.get('attempts')
        # A killed process or an exception without a transport receipt stays
        # unresolved. The reservation is never reclaimed on a guess.
        if count is None and result.get('status') in {'disabled', 'skipped_missing_api_key'}:
            count = 0  # The adapter explicitly returns before any submission.
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return
        unknown = (max(0, count - len(reported)) + sum((r.get('usage') or {}).get('total_tokens') is None for r in reported)
                   if reported else usage.get('unknown_attempt_count', count if usage.get('total_tokens') is None else max(0,count-1)))
        with connection(self.path) as db:
            db.execute('UPDATE calls SET ended=?,attempts=?,status=?,input_tokens=?,output_tokens=?,total_tokens=?, '
                       'cached_input_tokens=?,unknown_attempts=?,receipt_json=?,response_model=? WHERE id=? AND ended IS NULL',
                       (time.time(),count,result.get('status','unknown'),
                        *[usage.get(k) for k in ('input_tokens','output_tokens','total_tokens','cached_input_tokens')],unknown,
                        json.dumps({'request_id': result.get('request_id'), 'attempt_receipts': reported}, separators=(',', ':')),
                        result.get('response_model'),identifier))
            if result.get('status') != 'submitted_unknown':
                db.execute('DELETE FROM request_claims WHERE reservation_id=?', (identifier,))

    def summary(self, day, *, scope=None):
        date.fromisoformat(day)
        rows = []
        if self.path.is_file():
            with connection(self.path, readonly=True) as db:
                rows = db.execute('SELECT * FROM calls WHERE day=?', (day,)).fetchall()
        if scope:
            dates = [datetime.fromisoformat(scope[k]) for k in ('start_at', 'end_at')]
            if any(value.tzinfo is None for value in dates):
                raise ValueError('Billing scope timestamps require explicit timezone offsets')
            start, end = (value.timestamp() for value in dates)
            if end <= start:
                raise ValueError('Billing scope must use a nonempty [start_at,end_at) interval')
            rows = [row for row in rows if start <= row['started'] < end
                    and all(key in row.keys() and row[key] == scope[key] for key in ('provider', 'account_key'))
                    and (row['response_model'] if 'response_model' in row.keys() and row['response_model'] else row['model']) == scope['model']]
        fields = ('input_tokens','output_tokens','total_tokens','cached_input_tokens')
        by_model, request_ids = {}, []
        for row in rows:
            receipt = json.loads(row['receipt_json'] or '{}') if 'receipt_json' in row.keys() else {}
            attempt_receipts = receipt.get('attempt_receipts', [])
            account = row['account_key'] if 'account_key' in row.keys() else None
            response_model = row['response_model'] if 'response_model' in row.keys() else None
            identity = (row['provider'], response_model or row['model'], account)
            group = by_model.setdefault(identity, {'provider': identity[0], 'model': identity[1], 'account_key': account,
                'calls': 0, 'pending_calls': 0, 'unknown_attempts': 0, 'missing_token_counts': 0,
                'cached_input_usage_unknown': 0, 'model_identity_unresolved': 0,
                'known_tokens': dict.fromkeys(fields, 0)})
            group['calls'] += 1
            group['pending_calls'] += row['ended'] is None
            group['unknown_attempts'] += row['unknown_attempts'] if row['unknown_attempts'] is not None else row['reserved_attempts']
            submitted = row['attempts'] is None or row['attempts'] > 0
            missing = row['input_tokens'] is None or row['output_tokens'] is None
            missing_cache = row['cached_input_tokens'] is None
            if attempt_receipts:
                missing |= any((item.get('usage') or {}).get('input_tokens') is None
                               or (item.get('usage') or {}).get('output_tokens') is None for item in attempt_receipts)
                missing_cache |= any((item.get('usage') or {}).get('cached_input_tokens') is None for item in attempt_receipts)
                models = {item['response_model'] for item in attempt_receipts if item.get('response_model')}
                group['model_identity_unresolved'] += bool(models and models != {identity[1]})
            group['missing_token_counts'] += submitted and missing
            group['cached_input_usage_unknown'] += submitted and missing_cache
            for field in fields:
                group['known_tokens'][field] += row[field] or 0
            ids = [item.get('request_id') for item in attempt_receipts]
            request_ids.extend(item for item in ids or [receipt.get('request_id')] if item)
        return {'date': day, 'timezone': 'Asia/Shanghai', 'scope': 'device_day_video_understanding_only',
                'coverage': 'requests_reserved_by_this_ledger; historical_calls_not_imported',
                'day_basis': 'reservation_start; retries_stay_with_original_reservation',
                'calls': len(rows), 'attempts_or_reservations': sum(r['attempts'] if r['attempts'] is not None else r['reserved_attempts'] for r in rows),
                'pending_calls': sum(r['ended'] is None for r in rows),
                'known_tokens': {k: sum(r[k] or 0 for r in rows) for k in fields},
                'per_model': list(by_model.values()), 'request_ids': sorted(request_ids),
                'reconciliation_scope': scope,
                'submitted_unknown_calls': sum(r['status'] == 'submitted_unknown' or r['ended'] is None for r in rows),
                'completed_calls': sum(r['status'] == 'completed' for r in rows),
                'failed_calls': sum(r['status'] == 'failed' for r in rows),
                'failed_with_reported_usage': sum(r['status'] == 'failed' and r['total_tokens'] is not None for r in rows),
                'unknown_attempts': sum(r['unknown_attempts'] if r['unknown_attempts'] is not None else r['reserved_attempts'] for r in rows),
                'billing_reconciliation': 'NOT_PROVEN', 'currency_cost': None,
                'limits': {k: self.settings.get(k) for k in ('scene_daily_attempt_limit','scene_daily_token_limit')},
                'token_limit_semantics': 'stop_new_calls_after_observed_threshold; one_call_can_exceed_threshold'}


def main():
    """Read local usage without loading a model, credentials, or NAS media."""
    import argparse
    import json
    from .config import load_config
    parser = argparse.ArgumentParser(description='Read local device/day video API usage (no API calls)')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--date', default=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat())
    parser.add_argument('--vendor-usage', type=Path, help='Saved vendor JSON, never fetched from the provider')
    parser.add_argument('--scope', type=Path, help='Explicit local account/model/time scope JSON')
    args = parser.parse_args()
    from dataclasses import asdict
    from .usage_reconciliation import UsageReconciliationManager
    config = load_config(args.config)
    vendor = json.loads(args.vendor_usage.read_text()) if args.vendor_usage else None
    selected_scope = json.loads(args.scope.read_text()) if args.scope else None
    if vendor is None and selected_scope is None:
        from .pricing_table import PricingTable
        result = UsageLedger(config, readonly=True).summary(args.date)
        result['estimated_cost'] = PricingTable(config).calculate_cost(result)
    else:
        report = UsageReconciliationManager(config['storage']['local_runtime_root'], config).reconcile_daily(args.date, vendor, scope=selected_scope)
        result = asdict(report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

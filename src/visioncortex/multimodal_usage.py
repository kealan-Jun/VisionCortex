"""Local per-request accounting; reused outputs never create billable entries."""
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import time
import uuid

from .provider_control import ProviderUnavailable
from .sqlite_store import connection


def validate(settings):
    for name in ('scene_daily_attempt_limit', 'scene_daily_token_limit'):
        value = settings.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'mllm.{name} must be a positive integer or null')
    if not isinstance(settings.get('compact_scene_metadata', False), bool):
        raise ValueError('mllm.compact_scene_metadata must be a boolean')


class UsageLedger:
    def __init__(self, config, *, readonly=False):
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

    def reserve(self, input_key):
        today = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
        attempts = max(1, int(self.settings.get('max_retries', 1)))
        identifier = uuid.uuid4().hex
        with connection(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
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
            db.execute('INSERT INTO calls(id,day,input_key,provider,model,started,reserved_attempts,status) '
                       'VALUES(?,?,?,?,?,?,?,?)', (identifier,today,input_key,self.settings.get('provider'),
                       self.settings.get('model'),time.time(),attempts,'pending'))
        return identifier

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
                       'cached_input_tokens=?,unknown_attempts=? WHERE id=? AND ended IS NULL',
                       (time.time(),count,result.get('status','unknown'),
                        *[usage.get(k) for k in ('input_tokens','output_tokens','total_tokens','cached_input_tokens')],unknown,identifier))

    def summary(self, day):
        date.fromisoformat(day)
        rows = []
        if self.path.is_file():
            with connection(self.path, readonly=True) as db:
                rows = db.execute('SELECT * FROM calls WHERE day=?', (day,)).fetchall()
        fields = ('input_tokens','output_tokens','total_tokens','cached_input_tokens')
        return {'date': day, 'timezone': 'Asia/Shanghai', 'scope': 'device_day_video_understanding_only',
                'coverage': 'requests_reserved_by_this_ledger; historical_calls_not_imported',
                'day_basis': 'reservation_start; retries_stay_with_original_reservation',
                'calls': len(rows), 'attempts_or_reservations': sum(r['attempts'] if r['attempts'] is not None else r['reserved_attempts'] for r in rows),
                'pending_calls': sum(r['ended'] is None for r in rows),
                'known_tokens': {k: sum(r[k] or 0 for r in rows) for k in fields},
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
    args = parser.parse_args()
    print(json.dumps(UsageLedger(load_config(args.config), readonly=True).summary(args.date), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

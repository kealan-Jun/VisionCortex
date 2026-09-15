"""Provider/service/credential-scoped admission, without persisting secrets."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import time

from .sqlite_store import connection as connect
from .runtime_control import resource_slot


class ProviderUnavailable(RuntimeError):
    pass


def scope(binding, service):
    # Reference identifiers only: never hash/store the credential value.
    values = [binding.get(k) for k in ('provider', 'base_url', 'credential_ref', 'api_key_env')]
    return hashlib.sha256(json.dumps([*values, service], sort_keys=True).encode()).hexdigest()


def classify(error):
    message = str(error).lower()
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if status is None:
        match = re.search(r'\bhttp (\d{3})\b', message)
        status = int(match[1]) if match else None
    if 'arrearage' in message:
        return 'account_unavailable', 900
    if status in {401, 403}:
        return 'credential_unavailable', 900
    if status == 429:
        return 'rate_limited', 60
    if status in {500, 502, 503, 504} or any(name in type(error).__name__.lower() for name in ('timeout', 'connecterror', 'networkerror')):
        return 'transport_unavailable', 30
    return None


class Circuit:
    def __init__(self, root, binding, service):
        self.path = Path(root) / 'state' / 'provider-circuits.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = scope(binding, service)
        self.account = scope(binding, 'account')
        with connect(self.path) as db:
            db.executescript('''PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS circuits(key TEXT PRIMARY KEY, reason TEXT,
                    retry_at REAL, probe_until REAL NOT NULL DEFAULT 0);''')

    def admit(self, now=None):
        now = time.time() if now is None else now
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute('SELECT * FROM circuits WHERE key IN (?,?)', (self.key, self.account)).fetchall()
            if any(row['retry_at'] > now or row['probe_until'] > now for row in rows):
                raise ProviderUnavailable('Provider circuit open; other stages continue independently')
            for row in rows:
                # One real request is the half-open probe across all processes.
                db.execute('UPDATE circuits SET probe_until=? WHERE key=?', (now+900, row['key']))
            return [(row['key'], now+900) for row in rows]

    def failed(self, error, *, now=None):
        classified = classify(error)
        if classified is None:
            return False
        reason, delay = classified
        key = self.account if reason in {'account_unavailable', 'credential_unavailable'} else self.key
        with connect(self.path) as db:
            db.execute('INSERT OR REPLACE INTO circuits VALUES(?,?,?,0)',
                       (key, reason, (time.time() if now is None else now)+delay))
        return True

    def succeeded(self, probes):
        with connect(self.path) as db:
            # A concurrent request may have reopened the circuit. Its newer
            # failure must survive an earlier probe's late successful response.
            db.executemany('DELETE FROM circuits WHERE key=? AND probe_until=?', probes)

    def state(self):
        with connect(self.path, readonly=True) as db:
            return [dict(row) for row in db.execute('SELECT reason,retry_at,probe_until FROM circuits WHERE key IN (?,?)',
                                                   (self.key, self.account))]


@contextmanager
def provider_request(config, service, binding):
    root = config.get('storage', {}).get('local_runtime_root')
    active = bool(root and config.get('runtime', {}).get('provider_circuit_enabled', False))
    circuit = Circuit(root, binding, service) if active else None
    with resource_slot(config, 'cloud'):
        probes = circuit.admit() if circuit else []
        try:
            yield
        except Exception as exc:
            if circuit:
                circuit.failed(exc)
                # Delete only our old probe generations. A new failure has
                # probe_until=0 and survives this acknowledgement.
                circuit.succeeded(probes)
            raise
        else:
            if circuit:
                circuit.succeeded(probes)

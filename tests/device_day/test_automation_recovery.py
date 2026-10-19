"""Recovery capacity, durable waits and progress observations use owned data."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3

import httpx
import pytest

from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_progress import processing_health, render_processing_health
from visioncortex.multimodal_usage import UsageLedger, SubmittedUnknown, ReceiptCapacityUnavailable
from visioncortex.provider_control import Circuit, ProviderUnavailable
from visioncortex.publication_journal import PublicationJournal
from visioncortex.sqlite_store import connection


def config(default_config, tmp_path):
    value = deepcopy(default_config)
    value['storage']['local_runtime_root'] = str(tmp_path / 'runtime')
    value['mllm'].update(local_receipt_max_bytes=1024, local_receipt_total_bytes=2048)
    value['device_day'].update(enabled=True, inplace_preprocessing=True)
    return value


def test_unknown_claims_preserve_deduplication_without_reserving_empty_buffers(default_config, tmp_path):
    ledger = UsageLedger(config(default_config, tmp_path))
    for key in ('old-a', 'old-b'):
        reservation = ledger.reserve(key)
        ledger.finish(reservation, {'status': 'submitted_unknown', 'attempts': 1})
    assert ledger.recovery_status()['reserved_bytes'] == 0
    assert ledger.recovery_status()['unresolved_requests'] == 2
    with pytest.raises(SubmittedUnknown):
        ledger.reserve('old-a')
    assert ledger.reserve('new-input')
    with connection(ledger.path, readonly=True) as db:
        assert db.execute('SELECT COUNT(*) FROM request_claims').fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM calls WHERE status='submitted_unknown'").fetchone()[0] == 2


def test_received_payload_is_not_counted_again_as_an_empty_reservation(default_config, tmp_path):
    value = config(default_config, tmp_path)
    ledger = UsageLedger(value)
    first = ledger.reserve('first')
    ledger.received(first, {'status': 'completed', 'data': 'saved'})
    assert ledger.recovery_status()['active_reservations'] == 0
    assert ledger.reserve('second')
    with pytest.raises(ReceiptCapacityUnavailable):
        ledger.reserve('third')
    assert ledger.recover('first')[0] == first


def test_active_capacity_is_transactional_and_pending_requests_never_expire(default_config, tmp_path):
    value = config(default_config, tmp_path)
    ledger = UsageLedger(value)
    def reserve(i):
        try:
            return ledger.reserve(str(i))
        except ProviderUnavailable:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(bool(x) for x in pool.map(reserve, range(8))) == 2
    with connection(ledger.path) as db:
        db.execute('UPDATE calls SET started=0')
    assert ledger.recovery_status()['admission_available'] is False
    with pytest.raises(ReceiptCapacityUnavailable):
        ledger.reserve('another')


def test_known_unknown_exception_frees_only_payload_reservation(default_config, tmp_path):
    ledger = UsageLedger(config(default_config, tmp_path))
    item = ledger.reserve('ambiguous')
    ledger.submitted_unknown(item)
    assert ledger.recovery_status()['active_reservations'] == 0
    with pytest.raises(SubmittedUnknown):
        ledger.reserve('ambiguous')


def test_protocol_failures_open_shared_transport_circuit(tmp_path):
    binding = {'provider': 'synthetic', 'base_url': 'https://example.test/v1'}
    circuit = Circuit(tmp_path, binding, 'understanding')
    circuit.failed(httpx.RemoteProtocolError('owned test'), now=100)
    with pytest.raises(ProviderUnavailable):
        circuit.admit(now=101)


def test_publication_backoff_survives_restart_and_new_receipt_wakes_it(tmp_path):
    journal = PublicationJournal(tmp_path)
    token = journal.begin({'recording_id': 'a'})
    journal.defer('a', token)
    assert PublicationJournal(tmp_path).pending(ready_only=True) == []
    assert journal.pending()[0][0] == token  # Visibility includes deferred work.
    replacement = journal.begin({'recording_id': 'a', 'revision': 'new'})
    assert replacement != token
    assert journal.pending(ready_only=True)[0][0] == replacement
    journal.defer('a', token)  # Stale producer cannot delay the new generation.
    assert journal.pending(ready_only=True)[0][0] == replacement


def test_stage_wait_cannot_be_claimed_early_and_new_revision_bypasses_it(tmp_path, monkeypatch):
    from visioncortex import device_day_queue as module
    clock = [100.0]
    monkeypatch.setattr(module.time, 'time', lambda: clock[0])
    queue = DeviceDayQueue(tmp_path / 'queue-understanding.sqlite3')
    record = {'recording_id': 'a', 'configured_role': 'first_person'}
    queue.enqueue(record, 'v1')
    assert queue.claim('worker')
    queue.finish('worker', 'a', {'status': 'waiting_for_provider', 'error_type': 'ReceiptCapacityUnavailable'}, 1)
    assert queue.claim('early') is None
    clock[0] += 31
    assert queue.claim('later')
    queue.finish('later', 'a', {'status': 'waiting_for_provider'}, 1)
    assert queue.claim('early-again') is None
    queue.enqueue(record, 'v2')
    assert queue.claim('new-revision')


def test_progress_distinguishes_unobserved_inputs_and_reports_stall(default_config, tmp_path):
    value = config(default_config, tmp_path)
    root = tmp_path / 'runtime' / 'device-day'
    queue = DeviceDayQueue(root / 'queue-vision.sqlite3')
    record = {'configured_role': 'first_person', 'source_signature': 'sealed', 'processable': True,
              'recording_start_us': 1_800_000_000_000_000}
    for rid in ('ready', 'unobserved', 'missing'):
        queue.enqueue(record | {'recording_id': rid}, rid)
    with queue.connect() as db:
        db.execute('UPDATE recordings SET queued_at=100,updated_at=100')
        db.execute("UPDATE recordings SET input_status='missing' WHERE recording_id='missing'")
    with sqlite3.connect(root / 'observed-inventory.sqlite3') as db:
        db.execute('CREATE TABLE observations(id TEXT PRIMARY KEY,payload TEXT)')
        db.execute('INSERT INTO observations VALUES(?,?)', ('ready', json.dumps(record)))
    before = queue.path.read_bytes()
    health = processing_health(value, now=2000)
    categories = health['stages']['vision']['categories']
    assert categories == {'locally_ready': 1, 'inventory_unobserved': 1, 'input_missing': 1}
    assert {'stage': 'vision', 'reason': 'no_completed_progress'} in health['alerts']
    assert '当前清单未观察到 1' in render_processing_health(health)
    assert queue.path.read_bytes() == before


def test_empty_health_is_read_only_and_never_calls_storage_or_models(tmp_path):
    value = {'storage': {'local_runtime_root': str(tmp_path / 'absent')}}
    assert processing_health(value)['alerts'] == []
    assert not (tmp_path / 'absent').exists()

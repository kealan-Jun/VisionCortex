"""Missing inputs are dependencies, while source/access faults remain our work."""
from copy import deepcopy
import json
import sqlite3

from visioncortex.device_day_progress import processing_health, render_consumers, render_processing_health
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.input_availability import Availability


def setup_queue(default_config, tmp_path, stage='vision'):
    config = deepcopy(default_config)
    config['storage']['local_runtime_root'] = str(tmp_path)
    config['device_day'].update(enabled=True, inplace_preprocessing=True)
    queue = DeviceDayQueue(tmp_path / 'device-day' / f'queue-{stage}.sqlite3')
    return config, queue


def record(rid, signature='source'):
    return {'recording_id': rid, 'source_signature': signature, 'configured_role': 'first_person',
            'processable': True, 'recording_start_us': 1_800_000_000_000_000}


def test_missing_only_is_not_our_backlog_or_a_stall_and_completed_work_is_preserved(default_config, tmp_path):
    config, queue = setup_queue(default_config, tmp_path)
    for rid in ('missing', 'done', 'active'):
        queue.enqueue(record(rid), rid)
    with queue.connect() as db:
        db.execute("UPDATE recordings SET input_status='missing',queued_at=1,updated_at=1")
        db.execute("UPDATE recordings SET status='completed',completed_at=10 WHERE recording_id='done'")
        db.execute("UPDATE recordings SET status='running',lease_until=5000 WHERE recording_id='active'")
    state = processing_health(config, now=100)['stages']['vision']
    assert state['workload'] == {'completed': 1, 'running': 1, 'processing_backlog': 0, 'system_attention': 0,
                                 'waiting_for_input': 1, 'scheduled_wait': 0, 'outside_processing_scope': 0,
                                 'failed': 0, 'result_unknown': 0,
                                 'publication_pending': 0, 'recovered_published': 0}
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='completed',completed_at=10 WHERE recording_id='active'")
    health = processing_health(config, now=5000)
    assert health['stages']['vision']['status'] == 'waiting_for_input'
    assert health['alerts'] == []
    assert health['input_loss_evidence'] == 'NOT_PROVEN'
    assert '本系统待处理 0' in render_processing_health(health)
    assert '输入等待 1（不计处理积压）' in render_processing_health(health)


def test_current_availability_splits_missing_from_access_faults_without_accepting_stale_flags(default_config, tmp_path):
    config, queue = setup_queue(default_config, tmp_path)
    availability = Availability(queue.path.parent)
    for rid, state in [('missing', 'missing'), ('waiting', 'waiting'), ('access', 'unavailable'), ('stale', 'missing')]:
        queue.enqueue(record(rid), rid)
        availability.mark(record(rid, 'old' if rid == 'stale' else 'source'), state)
    health = processing_health(config)
    state = health['stages']['vision']
    assert state['categories'] == {'input_missing': 1, 'input_waiting': 1, 'input_unavailable': 1, 'locally_ready': 1}
    assert state['workload']['waiting_for_input'] == 2
    assert state['workload']['processing_backlog'] == 2
    assert state['workload']['system_attention'] == 1
    assert health['workload_complete'] is True


def test_downstream_input_wait_does_not_become_our_backlog_but_access_faults_and_other_revisions_do(default_config, tmp_path):
    config, queue = setup_queue(default_config, tmp_path, 'retention')
    config['device_day']['inplace_preprocessing'] = False
    vision = DeviceDayQueue(queue.path.parent / 'queue-vision.sqlite3')
    for rid, state in [('missing', 'missing'), ('access', 'unavailable'), ('other-revision', 'missing')]:
        queue.enqueue(record(rid, 'old' if rid == 'other-revision' else 'source'), rid)
        vision.enqueue(record(rid), rid)
        with queue.connect() as db:
            db.execute('UPDATE recordings SET input_status=? WHERE recording_id=?', (state, rid))
    state = processing_health(config)['stages']['vision']
    assert state['categories'] == {'upstream_input_waiting': 1, 'waiting_for_prerequisite': 2}
    assert state['workload']['waiting_for_input'] == 1
    assert state['workload']['processing_backlog'] == 2


def test_unobserved_inventory_and_version_mismatches_remain_our_backlog(default_config, tmp_path):
    config, queue = setup_queue(default_config, tmp_path)
    for rid in ('unobserved', 'mismatch'):
        queue.enqueue(record(rid), rid)
    with sqlite3.connect(queue.path.parent / 'observed-inventory.sqlite3') as db:
        db.execute('CREATE TABLE observations(id TEXT PRIMARY KEY,payload TEXT)')
        db.execute('INSERT INTO observations VALUES(?,?)', ('mismatch', json.dumps(record('mismatch', 'new'))))
    state = processing_health(config)['stages']['vision']
    assert state['categories'] == {'inventory_unobserved': 1, 'source_version_mismatch': 1}
    assert state['workload']['processing_backlog'] == state['workload']['system_attention'] == 2
    assert state['workload']['waiting_for_input'] == 0


def test_unreadable_input_catalog_never_reports_complete_metrics(default_config, tmp_path):
    config, queue = setup_queue(default_config, tmp_path)
    (queue.path.parent / 'InputAvailability.sqlite3').write_bytes(b'owned invalid database')
    health = processing_health(config)
    assert 'input_availability_unavailable' in health['errors']
    assert health['workload_complete'] is False
    assert health['status'] == 'evidence_unavailable'


def test_legacy_raw_pending_counts_are_not_rendered_as_processing_backlog():
    consumers = {'stages': {'vision': {'scopes': {
        'history': {'status': 'no_ready_work', 'locally_ready': 0, 'running': 0, 'pending': 999, 'blocked': 999}},
        'owners': []}}}
    text = render_consumers(consumers)
    assert '999' not in text
    assert '自动处理消费者与积压' not in text
    assert '输入等待、系统待排查和处理积压见处理进度核验' in text


def test_unknown_failures_and_publication_are_separate_from_ready_backlog(default_config, tmp_path):
    from visioncortex.publication_journal import PublicationJournal
    config, queue = setup_queue(default_config, tmp_path, 'understanding')
    for rid in ('unknown', 'expired', 'failed', 'ready', 'generated'):
        queue.enqueue(record(rid), rid)
    with queue.connect() as db:
        for rid, reason in [('unknown', 'submitted_unknown'), ('expired', 'result_unknown')]:
            db.execute("UPDATE recordings SET status='blocked',result=? WHERE recording_id=?",
                       (json.dumps({'reason': reason}), rid))
        db.execute("UPDATE recordings SET status='failed',attempts=3 WHERE recording_id='failed'")
        db.execute("UPDATE recordings SET status='completed',completed_at=10 WHERE recording_id='generated'")
    # Completed primary stages make remaining understanding work locally ready.
    for stage in ('retention', 'vision', 'stt'):
        parent = DeviceDayQueue(queue.path.with_name(f'queue-{stage}.sqlite3'))
        for rid in ('failed', 'ready'):
            parent.enqueue(record(rid), rid)
        with parent.connect() as db:
            db.execute("UPDATE recordings SET status='completed'")
    journal = PublicationJournal(queue.path.parent)
    journal.begin(record('generated'))
    journal.begin(record('generated'))  # Replacement is one recording, not two tasks.
    health = processing_health(config, now=100)
    state = health['stages']['understanding']
    assert state['workload']['result_unknown'] == 2
    assert state['workload']['failed'] == 1
    assert state['workload']['processing_backlog'] == state['ready_backlog'] == 1
    assert state['queue_status_counts'] == {'blocked': 2, 'failed': 1, 'queued': 1, 'completed': 1}
    assert health['publication']['completed_output_stage_pending_recordings'] == 1
    assert health['publication']['generated_pending_recordings'] is None
    assert health['publication']['pending_recordings'] == 1
    assert '结果未知 2' in render_processing_health(health)


def test_expired_paid_lease_is_unknown_before_recovery_and_retention_is_not_generated_output(default_config, tmp_path):
    from visioncortex.publication_journal import PublicationJournal
    config, queue = setup_queue(default_config, tmp_path, 'understanding')
    queue.enqueue(record('expired'), 'old')
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='running',lease_until=1")
    parent = DeviceDayQueue(queue.path.with_name('queue-retention.sqlite3'))
    parent.enqueue(record('only-originals'), 'originals')
    with parent.connect() as db:
        db.execute("UPDATE recordings SET status='completed'")
    PublicationJournal(queue.path.parent).begin(record('only-originals'))
    health = processing_health(config, now=100)
    state = health['stages']['understanding']
    assert state['workload']['result_unknown'] == 1
    assert state['ready_backlog'] == state['workload']['processing_backlog'] == 0
    assert health['publication']['completed_output_stage_pending_recordings'] == 0

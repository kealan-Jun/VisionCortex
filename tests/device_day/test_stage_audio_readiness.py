"""Independent recorder audio only releases STT, never video or unknown requests."""
import json
from pathlib import Path
import time

import pytest

from test_device_day_backfill_policy import LocalState, NOW, recording
from test_device_day_retention_worker import setup as worker_setup
from visioncortex.device_day_backfill_policy import historical_candidates
from visioncortex.device_day_progress import processing_health, snapshot
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.device_day_retention_worker import RetentionWorker
from visioncortex.input_availability import Availability, Reconciler, stage_input_state
from visioncortex.observed_inventory import observe


def audio_record(**overrides):
    return {'recording_id': 'audio-first', 'camera_key': 'camera', 'configured_role': None,
            'recording_start_us': 1_790_000_000_000_000, 'recording_end_us': 1_790_000_010_000_000,
            'source_signature': 'video-not-ready', 'recording_session_id': 'stable-session',
            'video_path': '/forbidden-nas/expected.mp4', 'available': False, 'processable': False,
            'stt_processable': True, 'stt_source_signature': 'stable-audio-input',
            'audio': {'status': 'provided', 'source_signature': 'stable-audio-bytes',
                      'start_us': 1_790_000_000_000_000, 'end_us': 1_790_000_010_000_000}, **overrides}


@pytest.mark.parametrize('state,reason', [('waiting', 'monitor_observation'),
    ('missing', 'capture_missing_pending_archive_verification'), ('waiting', 'capture_not_closed')])
def test_audio_projection_is_stage_specific_and_read_only(tmp_path, state, reason):
    root, record = tmp_path / 'device-day', audio_record()
    observe(root, {'recordings': [record]})
    availability = Availability(root)
    availability.mark(record, state, reason=reason)
    before = availability.states()
    assert availability.states_for_stage('stt')[record['recording_id']]['state'] == 'ready'
    assert availability.states_for_stage('vision') == before
    assert availability.states_for_stage('retention') == before
    assert availability.states() == before


@pytest.mark.parametrize('reason', ['content_identity_conflict', 'input_inspection_unavailable:OSError:errno=5',
                                    'unverified_archive'])
def test_audio_does_not_release_unrelated_availability_fault(reason):
    state = {'state': 'unavailable', 'reason': reason}
    assert stage_input_state(audio_record(), 'stt', state) == state


def test_stale_audio_observation_cannot_release_current_version(tmp_path):
    root, record = tmp_path / 'device-day', audio_record()
    observe(root, {'recordings': [record]})
    availability = Availability(root)
    availability.mark(record | {'source_signature': 'another-source'}, 'missing', reason='monitor_observation')
    assert availability.states_for_stage('stt') == availability.states()


def test_failed_inspection_revokes_only_its_exact_observation(tmp_path):
    from visioncortex.input_availability import _revoke_stt_observation
    from visioncortex.observed_inventory import read_inventory, revision
    root, record = tmp_path / 'device-day', audio_record()
    observe(root, {'recordings': [record]})
    later = record | {'source_signature': 'new-video-observation', 'processable': True}
    observe(root, {'recordings': [later]})
    previous_revision = revision(root)
    _revoke_stt_observation(root, record)
    assert read_inventory(root)['recordings'] == [later]
    assert revision(root) == previous_revision
    _revoke_stt_observation(root, later)
    assert read_inventory(root)['recordings'] == [later | {'stt_processable': False}]
    assert revision(root)[1] == previous_revision[1] + 1


def test_reconciler_inspects_independent_audio_even_when_expected_video_is_absent(tmp_path, monkeypatch):
    from visioncortex import input_availability, nas_recordings
    from visioncortex.observed_inventory import read_inventory
    root, source = tmp_path / 'runtime/device-day', tmp_path / 'capture'
    source.mkdir()
    record = audio_record(video_path=str(source / 'expected.mp4'), stt_processable=False)
    observe(root, {'recordings': [record]})
    queue = DeviceDayQueue(root / 'queue-retention.sqlite3')
    queue.enqueue(record, 'waiting-for-video')
    config = {'storage': {'local_runtime_root': str(root.parent)},
              'collection_ingest': {'source_root': str(source)}}
    fresh = record | {'stt_processable': True, 'size_bytes': None}
    calls = []
    def inspect(*args):
        calls.append(args[1])
        return fresh
    monkeypatch.setattr(nas_recordings, '_inspect', inspect)
    monkeypatch.setattr(input_availability, 'configured_record', lambda config, item: item)
    Reconciler(config).tick()
    availability = Availability(root)
    assert calls == [Path(record['video_path'])]
    assert read_inventory(root)['recordings'] == [fresh]
    assert availability.states()[record['recording_id']]['state'] == 'missing'
    assert availability.states_for_stage('stt')[record['recording_id']]['state'] == 'ready'
    def missing_audio(*args):
        raise FileNotFoundError('independent audio sidecar disappeared')
    monkeypatch.setattr(nas_recordings, '_inspect', missing_audio)
    Reconciler(config).tick()
    assert read_inventory(root)['recordings'][0]['stt_processable'] is False
    assert availability.states_for_stage('stt')[record['recording_id']]['state'] != 'ready'
    assert not Path(record['video_path']).exists()


@pytest.mark.parametrize('extra', [{'stt_processable': False}, {'audio': {'status': 'pending_publication'}},
    {'audio': {'status': 'association_mismatch'}}, {'audio': {'status': 'not_provided'}}])
def test_not_ready_audio_or_embedded_probe_requires_its_own_ready_inputs(tmp_path, monkeypatch, extra):
    record = audio_record(**extra)
    runner = worker_setup(tmp_path, monkeypatch, [record])
    worker = RetentionWorker(stage='stt')
    assert worker.candidates(runner) == []
    assert worker.admit(runner, record) is False


def test_audio_first_candidates_and_completed_audio_do_not_reopen_when_video_arrives(tmp_path, monkeypatch):
    record = audio_record()
    runner = worker_setup(tmp_path, monkeypatch, [record])
    worker = RetentionWorker(stage='stt')
    assert worker.candidates(runner) == [record]
    assert RetentionWorker(stage='retention').candidates(runner) == []
    assert RetentionWorker(stage='vision').candidates(runner) == []
    queue = runner.queues['stt']
    queue.enqueue(record, 'audio-recipe')
    queue.claim('owner')
    queue.finish('owner', record['recording_id'], {'status': 'completed'}, 1)
    later = record | {'source_signature': 'closed-video', 'processable': True, 'available': True}
    observe(runner.runtime_root, {'recordings': [later]})
    assert worker.candidates(runner) == []


def test_worker_admission_keeps_unrelated_source_conflicts_closed(tmp_path, monkeypatch):
    record = audio_record()
    runner = worker_setup(tmp_path, monkeypatch, [record])
    availability = Availability(runner.runtime_root)
    worker = RetentionWorker(stage='stt')
    availability.mark(record, 'missing', reason='capture_missing_pending_archive_verification')
    assert worker.candidates(runner) == [record]
    availability.mark(record, 'waiting', reason='content_identity_conflict')
    assert worker.candidates(runner) == []
    assert worker.admit(runner, record) is False


@pytest.mark.parametrize('status,reason', [('blocked', 'submitted_unknown'), ('blocked', 'result_unknown'),
                                        ('running', None)])
def test_late_video_cannot_replace_unknown_or_expired_audio_request(tmp_path, status, reason):
    record = audio_record()
    # Legacy payloads did not carry the new stage readiness fields.
    previous = {key: value for key, value in record.items() if not key.startswith('stt_')}
    queue = DeviceDayQueue(tmp_path / 'queue-stt.sqlite3')
    queue.enqueue(previous, 'old-paid-request')
    with queue.connect() as db:
        db.execute('UPDATE recordings SET status=?,blocked_reason=?,lease_until=?,result=?',
                   (status, reason, time.time() - 10, json.dumps({'reason': reason or 'in_flight'})))
        before = dict(db.execute('SELECT * FROM recordings').fetchone())
    later = record | {'source_signature': 'closed-video', 'processable': True, 'available': True}
    assert queue.enqueue(later, 'new-code-recipe') is False
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM recordings').fetchone()) == before
        assert db.execute('SELECT COUNT(*) FROM source_versions').fetchone()[0] == 0


def test_history_uses_audio_readiness_and_same_audio_completion_without_video_dependency(tmp_path):
    local = LocalState(tmp_path / 'runtime')
    record = recording('history', age=500, **{key: value for key, value in audio_record().items()
                       if key not in {'recording_id', 'recording_start_us', 'recording_end_us'}})
    local.observe(record)
    availability = Availability(local.base)
    availability.mark(record, 'missing', reason='capture_missing_pending_archive_verification')
    assert historical_candidates(local.config, 'stt', now=NOW) == [record]
    assert historical_candidates(local.config, 'vision', now=NOW) == []
    previous = record | {'source_signature': 'earlier-video-state'}
    local.queue('stt', previous, status='completed')
    assert historical_candidates(local.config, 'stt', now=NOW) == []
    local.queue('stt', previous, status='failed', attempts=3)
    assert historical_candidates(local.config, 'stt', now=NOW) == []
    availability.mark(record, 'unavailable', reason='content_identity_conflict')
    local.queue('stt', previous, attempts=0)
    assert historical_candidates(local.config, 'stt', now=NOW) == []


def test_unknown_same_audio_cannot_fill_bounded_history_share_when_video_arrives(tmp_path):
    local = LocalState(tmp_path / 'runtime')
    common = {key: value for key, value in audio_record().items()
              if key not in {'recording_id', 'recording_start_us', 'recording_end_us'}}
    unknown = recording('unknown', age=300, **common)
    ready = recording('ready', age=500, **common)
    local.observe(unknown, ready)
    # Its video changed while the exact original paid audio request remains
    # unresolved. The later video cannot turn it into a new audio admission.
    local.queue('stt', unknown | {'source_signature': 'earlier-video'}, status='blocked')
    assert historical_candidates(local.config, 'stt', now=NOW, limit=1) == [ready]


def test_stage_progress_keeps_audio_runnable_and_video_missing_separate(tmp_path):
    root, record = tmp_path / 'device-day', audio_record(configured_role='first_person')
    observe(root, {'recordings': [record]})
    queues = {stage: DeviceDayQueue(root / f'queue-{stage}.sqlite3')
              for stage in ('retention', 'vision', 'stt', 'understanding', 'report')}
    for queue in queues.values():
        queue.enqueue(record, 'recipe')
    availability = Availability(root)
    availability.mark(record, 'missing', reason='capture_missing_pending_archive_verification')
    for stage, queue in queues.items():
        queue.sync_availability(availability.states_for_stage(stage))
    config = {'storage': {'local_runtime_root': str(tmp_path)}, 'device_day': {'inplace_preprocessing': True}}
    health = processing_health(config)
    assert health['stages']['stt']['categories'] == {'locally_ready': 1}
    assert health['stages']['stt']['queue_status_counts'] == {'queued': 1}
    assert health['stages']['vision']['categories'] == {'input_missing': 1}
    day = next(iter(snapshot(config)['days'].values()))
    assert day['missing_input_count'] == 1 and day['total'] == 1
    assert day['stages']['stt']['queued'] == day['stages']['stt']['processing_total'] == 1
    assert day['stages']['stt']['missing_input_count'] == 0
    assert day['stages']['vision']['input_missing'] == day['stages']['vision']['missing_input_count'] == 1
    assert Path(record['video_path']).exists() is False

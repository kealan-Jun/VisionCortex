"""Lane failures never authorize missing input or block healthy device history."""
from copy import deepcopy
import json

import pytest

from test_device_day_backfill_policy import LocalState, NOW, recording, scanning_lane, successful_lane
from visioncortex.device_day_backfill_policy import historical_candidate_ids, live_demand
from visioncortex.device_day_live_discovery import discovery_status, publish_status


@pytest.fixture
def local(tmp_path):
    state = LocalState(tmp_path / 'runtime')
    state.config['device_day']['camera_lanes'] = True
    state.config['collection_ingest'].update(discovery_mode='metadata_only', camera_role_map={'a': 'first_person'})
    return state


def publish(local, lanes, *, records=(), errors=(), **extra):
    snapshot = {'recordings': [], 'discovery_mode': 'metadata_only', 'discovery_lanes': lanes,
                'camera_directories': [lane['camera_key'] for lane in lanes],
                'errors': [error for lane in lanes for error in lane.get('errors', [])] + list(errors),
                'truncated': any(lane.get('truncated') for lane in lanes),
                'pending_publication_count': 0, 'pending_live_publication_count': 0, **extra}
    status = discovery_status(local.config, snapshot, recordings=records, observed_at=NOW)
    publish_status(local.base, status)
    return status


def test_missing_role_keeps_alert_and_other_camera_history_advances(local):
    row = recording('history', age=200)
    unknown = recording('unregistered', age=200, camera='new-hardware', configured_role=None)
    local.observe(row, unknown)
    status = publish(local, [successful_lane('a'), successful_lane('new-hardware')])
    assert status['status'] == 'degraded'
    assert status['alerts'][0]['type'] == 'camera_role_unconfigured'
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == ['history']
    assert set(historical_candidate_ids(local.config, 'retention', now=NOW)) == {'history', 'unregistered'}
    unknown['audio'] = {'status': 'provided', 'source_signature': 'own-audio'}
    local.observe(unknown)
    assert historical_candidate_ids(local.config, 'stt', now=NOW) == ['unregistered']
    # Registration can release the dependent stage; names never supply a role.
    local.config['collection_ingest']['camera_role_map']['new-hardware'] = 'third_person'
    assert set(historical_candidate_ids(local.config, 'vision', now=NOW)) == {'history', 'unregistered'}


@pytest.mark.parametrize('failure', [
    {'status': 'failed', 'thread_alive': False}, {'status': 'retrying'},
    {'errors': [{'error_type': 'OSError', 'path': 'broken'}]}, {'truncated': True},
    {'status': 'starting'}, {'status': 'scanning', 'started_at': NOW, 'initial_scan_completed': False},
    {'completed_at': NOW - 31}, {'status': 'scanning', 'started_at': NOW - 31},
])
def test_known_failed_camera_is_excluded_without_blocking_healthy_history(local, failure):
    local.observe(recording('healthy', age=200), recording('broken', age=300, camera='broken'))
    local.config['collection_ingest']['camera_role_map']['broken'] = 'third_person'
    publish(local, [successful_lane('a'), successful_lane('broken', **failure)])
    state = live_demand(local.config, 'vision', NOW)
    assert state['status'] == 'idle'
    assert state['isolated_camera_keys'] == ['broken']
    for stage in ('retention', 'vision'):
        assert historical_candidate_ids(local.config, stage, now=NOW) == ['healthy']


@pytest.mark.parametrize('change', ['scope_absent', 'wrong_schema', 'wrong_source', 'hidden_lane_failure',
                                    'global_error', 'pending_live', 'all_failed', 'stale'])
def test_unknown_global_or_unverified_lane_conditions_still_fail_closed(local, change):
    lanes = [successful_lane('a'), successful_lane('broken', status='failed', thread_alive=False)]
    options = {}
    if change == 'global_error':
        options['errors'] = [{'path': 'observed-inventory.json', 'error_type': 'OSError'}]
    elif change == 'pending_live':
        options['pending_live_publication_count'] = 1
    elif change == 'all_failed':
        lanes[0].update(status='retrying')
    publish(local, lanes, **options)
    path = local.root / 'state' / 'nas-recording-monitor.json'
    data = json.loads(path.read_text())
    if change == 'scope_absent':
        data.pop('admission_scope')
    elif change == 'wrong_schema':
        data['admission_scope']['schema_version'] = 'unknown'
    elif change == 'wrong_source':
        data['monitor_source'] = 'unverified'
    elif change == 'hidden_lane_failure':
        data['admission_scope']['unavailable_camera_keys'] = []
    elif change == 'stale':
        data['monitor']['observed_at'] = '1970-01-01T00:00:00+00:00'
    path.write_text(json.dumps(data))
    local.observe(recording('healthy', age=200))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == []


def test_scoped_large_monitor_and_normal_scan_reuse_only_healthy_completion(local):
    rows = {'recordings': [{'opaque': 'x' * 300000}]}
    broken = successful_lane('broken', status='retrying')
    publish(local, [successful_lane('a'), broken], **rows)
    assert live_demand(local.config, 'vision', NOW)['status'] == 'idle'
    publish(local, [scanning_lane('a', errors=[], truncated=False), broken], **rows)
    assert live_demand(local.config, 'vision', NOW + 1)['status'] == 'idle'
    # A formerly failed lane cannot borrow a different camera's completion.
    publish(local, [scanning_lane('a', errors=[], truncated=False),
                    scanning_lane('broken', errors=[], truncated=False)], **rows)
    assert live_demand(local.config, 'vision', NOW + 2)['status'] == 'waiting_for_monitor'


def test_verified_hardware_rename_role_is_not_reported_as_unconfigured(local):
    record = recording('renamed', camera='renamed-host', configured_role='first_person',
                       camera_binding_status='configured_hardware_serial')
    before = deepcopy(local.config)
    status = publish(local, [successful_lane('a'), successful_lane('renamed-host')], records=[record])
    assert status['unconfigured_camera_directories'] == []
    assert status['status'] == 'watching'
    assert local.config == before


@pytest.mark.parametrize('field,value', [
    ('errors', 'missing'), ('errors', None), ('errors', ''),
    ('truncated', 'missing'), ('truncated', None), ('truncated', 0),
])
def test_scoped_degraded_requires_explicit_complete_health_fields(local, field, value):
    publish(local, [successful_lane('a'), successful_lane('unregistered')])
    path = local.root / 'state' / 'nas-recording-monitor.json'
    snapshot = json.loads(path.read_text())
    lane = snapshot['discovery_lanes'][0]
    if value == 'missing':
        lane.pop(field)
    else:
        lane[field] = value
    path.write_text(json.dumps(snapshot))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'waiting_for_monitor'
    assert historical_candidate_ids(local.config, 'vision', now=NOW) == []


def test_fair_history_reuses_durable_camera_turn_before_bounded_limit(local):
    local.config['device_day']['backfill'] = {'mode': 'fair'}
    publish(local, [successful_lane('a'), successful_lane('b')])
    local.observe(*(recording(str(i), age=500 + i) for i in range(30)),
                  recording('other-camera', age=200, camera='b'))
    path = local.base / 'queue-vision.sqlite3'
    local.write(path, 'CREATE TABLE camera_dispatch(camera_key TEXT PRIMARY KEY,last_claim REAL)')
    local.write(path, 'INSERT INTO camera_dispatch VALUES(?,?)', ('a', NOW - 1))
    # Recent arrivals still reserve resources, but fair history retains a turn.
    local.observe(recording('live', age=1))
    assert live_demand(local.config, 'vision', NOW)['status'] == 'fair_capacity'
    assert historical_candidate_ids(local.config, 'vision', limit=1, now=NOW) == ['other-camera']
    local.write(path, 'INSERT INTO camera_dispatch VALUES(?,?)', ('b', NOW))
    assert historical_candidate_ids(local.config, 'vision', limit=1, now=NOW) == ['29']

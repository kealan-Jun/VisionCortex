"""Local receipts only: polling and input arrivals are independent clocks."""
from datetime import datetime, timezone
import json

import pytest

from visioncortex.device_day_latency import discovery_health, observe, render, snapshot


NOW = 1_800_000_000


def receipt(tmp_path, *, at=NOW, **changes):
    path = tmp_path/'state/nas-recording-monitor.json'
    path.parent.mkdir(exist_ok=True)
    data = {'monitor': {'observed_at': datetime.fromtimestamp(at, timezone.utc).isoformat(),
                        'status': 'watching', 'poll_seconds': 5},
            'discovery_lanes': [{'camera_key': 'c', 'mode': 'live', 'status': 'watching',
                                 'thread_alive': True, 'completed_at': NOW-1}],
            'errors': [], 'pending_publication_count': 0, 'truncated': False}
    data.update(changes)
    path.write_text(json.dumps(data), encoding='utf-8')
    return {'storage': {'local_runtime_root': str(tmp_path)}}


def test_fresh_poll_with_unchanged_arrival_is_not_a_monitor_outage(tmp_path):
    cfg = receipt(tmp_path)
    row = {'recording_id': 'r', 'camera_key': 'c', 'recording_start_us': int((NOW-18*3600)*1e6),
           'source_signature': 'original', 'processable': True}
    observe(cfg, [row], now=NOW-18*3600)
    observe(cfg, [row], now=NOW)
    value = snapshot(cfg, now=NOW+2)
    assert value['discovery']['poll_status'] == 'watching'
    assert value['discovery']['coverage_status'] == 'healthy'
    assert value['discovery']['poll_age_seconds'] == 2
    assert value['latest_revision_observed_at'] == NOW-18*3600
    assert '无新版本不等于轮询停止' in render(value)


def test_new_arrival_cannot_hide_a_stale_poll(tmp_path):
    cfg = receipt(tmp_path, at=NOW-100)
    value = snapshot(cfg, now=NOW)
    assert value['discovery']['poll_status'] == 'stale'
    assert value['discovery']['coverage_status'] == 'degraded'


@pytest.mark.parametrize('lane', [
    {'status': 'failed'}, {'thread_alive': False}, {'completed_at': NOW-121},
    {'status': 'slow_or_unavailable'}, {'errors': [{'error_type': 'OSError'}]}, {'truncated': True},
    {'completed_at': None}, {'status': 'starting', 'started_at': NOW-121},
])
def test_coordinator_heartbeat_does_not_hide_lane_failures(tmp_path, lane):
    base = {'camera_key': 'c', 'mode': 'live', 'status': 'watching',
            'thread_alive': True, 'completed_at': NOW-1}
    cfg = receipt(tmp_path, discovery_lanes=[base | lane])
    health = discovery_health(cfg, now=NOW)
    assert health['poll_status'] == 'watching'
    assert health['coverage_status'] == 'degraded'
    assert len(health['lane_issues']) == 1


def test_history_scan_uses_its_own_interval_and_incomplete_scan_is_explicit(tmp_path):
    cfg = receipt(tmp_path, discovery_lanes=[{'camera_key': 'c', 'mode': 'history', 'status': 'watching',
                                            'thread_alive': True, 'completed_at': NOW-290}])
    assert discovery_health(cfg, now=NOW)['coverage_status'] == 'healthy'
    receipt(tmp_path, discovery_lanes=[{'camera_key': 'c', 'mode': 'live', 'status': 'scanning',
                                      'thread_alive': True, 'started_at': NOW-1}])
    assert discovery_health(cfg, now=NOW)['coverage_status'] == 'in_progress'
    receipt(tmp_path, discovery_lanes=[])
    assert discovery_health(cfg, now=NOW)['coverage_status'] == 'unavailable'


def test_pending_publication_and_truncation_are_not_complete_coverage(tmp_path):
    cfg = receipt(tmp_path, pending_publication_count=1)
    assert discovery_health(cfg, now=NOW)['coverage_status'] == 'in_progress'
    receipt(tmp_path, truncated=True)
    assert discovery_health(cfg, now=NOW)['coverage_status'] == 'degraded'


def test_missing_or_invalid_receipt_never_claims_healthy(tmp_path):
    cfg = {'storage': {'local_runtime_root': str(tmp_path)}}
    assert discovery_health(cfg, now=NOW)['poll_status'] == 'unavailable'
    receipt(tmp_path, monitor={'observed_at': 'not-a-date'})
    assert discovery_health(cfg, now=NOW)['poll_status'] == 'unavailable'
    receipt(tmp_path, at=NOW+100)
    assert discovery_health(cfg, now=NOW)['poll_status'] == 'clock_invalid'

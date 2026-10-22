"""Capture time is the primary order when latest_first is enabled."""
from visioncortex.device_day_queue import DeviceDayQueue


def test_capture_time_precedes_camera_rotation_wait_age_and_class_cost(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-vision.sqlite3', latest_first=True,
                           scheduling={'enabled': True, 'max_ready_wait_seconds': 1})
    old = {'recording_id': 'old', 'camera_key': 'old-camera', 'recording_start_us': 100,
           'configured_role': 'first_person', 'processable': True, 'activity_kind': 'unresolved'}
    new = {'recording_id': 'new', 'camera_key': 'new-camera', 'recording_start_us': 200,
           'configured_role': 'first_person', 'processable': True, 'activity_kind': 'experiment_candidate'}
    for row in (old, new):
        queue.enqueue(row, row['recording_id'])
    with queue.connect() as db:
        db.execute("UPDATE recordings SET queued_at=0,first_ready_at=0 WHERE recording_id='old'")
        db.execute("INSERT INTO camera_dispatch VALUES('new-camera',999999)")
        db.execute("INSERT INTO dispatch_costs VALUES('class','experiment_candidate',999999)")
    assert queue.claim('owner', camera_serial=True)['recording_id'] == 'new'


def test_latest_history_is_selected_before_oldest_history_when_already_admitted(tmp_path):
    queue = DeviceDayQueue(tmp_path/'queue-understanding.sqlite3', latest_first=True,
                           scheduling={'enabled': True})
    for identity, stamp in [('old', 1), ('new', 2)]:
        row = {'recording_id': identity, 'camera_key': identity, 'recording_start_us': stamp,
               'source_signature': identity, 'configured_role': 'first_person', 'processable': True}
        queue.enqueue(row, identity)
        queue.grant_admission(row, cohort='historical_backfill', reason='verified-ready-history')
    assert queue.claim('owner', camera_serial=True)['recording_id'] == 'new'

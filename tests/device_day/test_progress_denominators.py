"""Stage ratios preserve successful historical/no-audio work independently."""
from visioncortex.device_day_progress import snapshot
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.input_availability import Availability


def test_missing_vision_input_does_not_remove_completed_stt_from_its_denominator(tmp_path):
    root = tmp_path / 'device-day'
    queues = {stage: DeviceDayQueue(root / f'queue-{stage}.sqlite3')
              for stage in ('retention', 'vision', 'stt', 'understanding', 'report')}
    records = [{'recording_id': str(n), 'camera_key': 'cam', 'configured_role': 'first_person',
                'recording_start_us': 1787792400000000 + n, 'source_signature': str(n), 'processable': True}
               for n in range(3)]
    for queue in queues.values():
        for record in records:
            queue.enqueue(record, 'v1')
    for n in range(3):
        queues['stt'].claim('stt')
        queues['stt'].finish('stt', str(n), {'status': 'completed', 'transcription_outcome': 'no_audio'}, 1)
    # One completed historical vision receipt remains counted even if the
    # original is currently unavailable to unfinished retention/other stages.
    queues['vision'].claim('vision')
    queues['vision'].finish('vision', '0', {'status': 'completed'}, 1)
    Availability(root).mark(records[0], 'missing')
    Availability(root).mark(records[1], 'missing')
    day = next(iter(snapshot({'storage': {'local_runtime_root': str(tmp_path)}})['days'].values()))
    assert day['total'] == 3 and day['processing_total'] == 2
    assert day['stages']['stt']['completed'] == day['stages']['stt']['processing_total'] == 3
    assert day['stages']['vision']['completed'] == 1
    assert day['stages']['vision']['processing_total'] == 2
    assert day['stages']['retention']['processing_total'] == 1
    assert day['stages']['retention']['missing_input_count'] == 2
    for stage in day['stages'].values():
        assert stage['completed'] <= stage['processing_total'] <= day['total']


def test_current_running_lease_remains_in_its_stage_denominator(tmp_path):
    root = tmp_path / 'device-day'
    queue = DeviceDayQueue(root / 'queue-vision.sqlite3')
    record = {'recording_id': 'running', 'camera_key': 'cam', 'configured_role': 'first_person',
              'recording_start_us': 1787792400000000, 'source_signature': 'running', 'processable': True}
    queue.enqueue(record, 'v1')
    queue.claim('owner')
    Availability(root).mark(record, 'missing')
    day = next(iter(snapshot({'storage': {'local_runtime_root': str(tmp_path)}})['days'].values()))
    assert day['stages']['vision']['running'] == day['stages']['vision']['processing_total'] == 1
    assert day['stages']['retention']['processing_total'] == 0

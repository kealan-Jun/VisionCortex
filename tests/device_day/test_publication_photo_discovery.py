"""Split file publication observes sparse photos without an HTTP companion."""
import os
import time
from pathlib import Path

from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher, build_file_index
from visioncortex.device_day_timeline import day_bounds
from visioncortex.observed_inventory import observe

DAY = '2026-10-21'
CAMERA = 'camera_cam01'
NAME = DAY + '_' + CAMERA
START = day_bounds(DAY)[0]


def settings(tmp_path):
    return {'storage': {'local_runtime_root': str(tmp_path/'runtime'), 'archive_root': str(tmp_path/'archive')},
            'collection_ingest': {'source_root': str(tmp_path/'capture'),
                                  'camera_role_map': {CAMERA: 'first_person'}, 'settle_seconds': 5},
            'device_day': {'process_since_us': START}}


def test_no_photos_is_a_checked_no_input_state_and_late_photo_is_discovered(tmp_path):
    config = settings(tmp_path)
    runtime = Path(config['storage']['local_runtime_root'])/'device-day'
    observe(runtime, {'recordings': [{'recording_id': 'slice', 'camera_key': CAMERA,
        'recording_start_us': START, 'recording_end_us': START+10000000}]})
    source = {'archive': NAME, 'recordings': [], 'segments': []}
    root = Path(config['storage']['archive_root'])/NAME
    atomic_json(root/'ProcessedClips/Index.json', source)
    publisher = FileIndexPublisher(config)
    assert not publisher.tick(archives=[NAME])['errors']
    target = root/'Comment/TimeIndex.json'
    assert read_json(target)['photo_index_status'] == 'no_input'
    photo = Path(config['collection_ingest']['source_root'])/f'voice_photos/{CAMERA}/{DAY}/12-00-00/20261021_120000.jpg'
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b'owned-reference-test')
    os.utime(photo, (time.time()-60, time.time()-60))
    index = runtime/'CapturePhotos'/f'{NAME}.json'
    atomic_json(index, read_json(index) | {'observed_at': 0})
    assert not publisher.tick(archives=[NAME])['errors']
    value = read_json(target)
    assert value['photo_index_status'] == 'available'
    assert value['streams']['images'][0]['file']['path'].endswith('20261021_120000.jpg')
    assert photo.read_bytes() == b'owned-reference-test'


def test_photo_only_day_is_discovered_without_inventory_or_prebuilt_index(tmp_path):
    config = settings(tmp_path)
    photo = Path(config['collection_ingest']['source_root'])/f'voice_photos/{CAMERA}/{DAY}/12-00-00/20261021_120000.jpg'
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b'owned-reference-test')
    os.utime(photo, (time.time()-60, time.time()-60))
    result = FileIndexPublisher(config).tick(archives=[NAME])
    assert not result['errors'] and result['published'] == 1
    root = Path(config['storage']['archive_root'])/NAME
    assert read_json(root/'Comment/TimeIndex.json')['photo_index_status'] == 'available'
    assert not (root/'MetaVideo').exists()


def test_photo_scan_error_is_not_presented_as_missing_input(tmp_path):
    config = settings(tmp_path)
    value = build_file_index(config, {'archive': NAME, 'recordings': [], 'segments': []},
                             {'observed_at': 1, 'devices': [], 'photo_errors': [{'error_type': 'OSError'}]})
    assert value['photo_index_status'] == 'unavailable'

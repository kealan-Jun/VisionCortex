"""Optional TimeIndex provenance without copying or guessing capture facts."""
from copy import deepcopy

from visioncortex.application.field_archive_context import publish_context
from visioncortex.device_day_contract import atomic_json, digest, read_json
from visioncortex.device_day_file_index import build_file_index, publish_file_index


def fixture(tmp_path):
    name = '2026-10-19_camera'
    root = tmp_path/name
    root.mkdir()
    config = {'storage': {'archive_root': str(tmp_path), 'local_runtime_root': str(tmp_path/'Runtime')},
        'collection_ingest': {'source_root': str(tmp_path/'Captures')},
        'device_day': {'field_recognition': {'enabled': False, 'archive_context_enabled': True}}}
    index = {'archive': name, 'recordings': [{'recording_id': 'r', 'start_us': 1, 'end_us': 2,
        'sources': [{'kind': 'video_frame_timestamps', 'retained': {'path': 'MetaData/time.csv', 'sha256': 'a'*64, 'size_bytes': 3}},
                    {'kind': 'audio_audio', 'retained': {'path': 'MetaData/audio.wav', 'sha256': 'b'*64, 'size_bytes': 4}}],
        'capture_manifest': {'path': 'MetaData/Capture.json', 'sha256': 'c'*64, 'size_bytes': 5}}],
        'segments': [], 'understandings': [],
        'context': {'events': [{'event_id': 'draft', 'payload': {'status': 'draft', 'unbound_readings': [{'value': '1'}]}}],
                    'field_event_status': 'available_archived_evidence', 'field_source_references': []}}
    return config, index, root


def test_original_csv_audio_and_capture_manifest_references_are_preserved(tmp_path):
    config, index, _ = fixture(tmp_path)
    record = build_file_index(config, index, {})['recordings'][0]
    assert [row['kind'] for row in record['sources']] == ['video_frame_timestamps', 'audio_audio']
    for expected, actual in zip(index['recordings'][0]['sources'], record['sources'], strict=True):
        assert actual == {'kind': expected['kind'], 'retained': expected['retained'] | {'path_base': 'device_day'}}
    assert record['capture_manifest'] == index['recordings'][0]['capture_manifest'] | {'path_base': 'device_day'}


def test_field_pointer_requires_current_native_publication_evidence_and_real_bytes(tmp_path):
    config, index, root = fixture(tmp_path)
    before = publish_file_index(config, index, {})
    assert 'field_events_file' not in before
    publish_context(config, index)
    result = publish_file_index(config, index, {})
    reference = result['field_events_file']
    assert reference['path'] == 'Comment/FieldEvents.json' and reference['path_base'] == 'device_day'
    assert reference['sha256'] and reference['size_bytes'] > 0
    original = deepcopy(index)
    index['context']['events'][0]['payload']['status'] = 'revised'
    assert 'field_events_file' not in publish_file_index(config, index, {})
    published = read_json(root/'Comment/FieldEvents.json')
    saved = root/'Comment/.Evidence/FieldRecognition'/(digest(published)+'.json')
    saved.unlink()
    assert 'field_events_file' not in publish_file_index(config, original, {})
    atomic_json(saved, published)
    raw = (root/'Comment/FieldEvents.json').read_bytes()
    (root/'Comment/FieldEvents.json').write_bytes(bytes([raw[0] ^ 1])+raw[1:])
    assert 'field_events_file' not in publish_file_index(config, original, {})

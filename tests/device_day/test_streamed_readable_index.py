"""Readable publication bounds peak memory, including canonical pointer writes."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from visioncortex.device_day_content import readable_source_index
from visioncortex.device_day_file_index import _READABLE_OMIT
from visioncortex.receipt_projection import projected_json, replace_json_field, ProjectionDeferred


def test_large_stream_policy_keeps_buffered_receipts_within_their_separate_limits():
    package = Path(__import__('visioncortex').__file__).resolve().parent.parent
    env = os.environ | {'PYTHONPATH': str(package),
        'VISIONCORTEX_RECEIPT_PROJECTION_MAX_BYTES': str(256 * 1024**2),
        'VISIONCORTEX_RECEIPT_PROJECTION_MAX_JSON_EVENTS': '8000000',
        'VISIONCORTEX_RECEIPT_PROJECTION_STREAM_MAX_BYTES': str(2 * 1024**3),
        'VISIONCORTEX_RECEIPT_PROJECTION_STREAM_MAX_JSON_EVENTS': '268435456'}
    script = '''
from visioncortex.receipt_projection import (
    _MAX_RECEIPT_BYTES, _MAX_JSON_EVENTS, _MAX_STREAM_BYTES, _MAX_STREAM_EVENTS)
assert _MAX_RECEIPT_BYTES == 256 * 1024**2
assert _MAX_JSON_EVENTS == 8000000
assert _MAX_STREAM_BYTES == 2 * 1024**3
assert _MAX_STREAM_EVENTS == 268435456
'''
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    env['VISIONCORTEX_RECEIPT_PROJECTION_STREAM_MAX_BYTES'] = str(2 * 1024**3 + 1)
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'must be between 1' in result.stderr


def test_projection_matches_legacy_fields_without_decoding_dense_subtrees(tmp_path):
    value = {'archive': 'owned', 'recordings': [{'processing': {
        'batches': [{'huge': 'x' * 1000}], 'scan_reports': [], 'audit_artifacts': [],
        'clock_mapping': {'origin_us': 2**80}}}],
        'segments': [{'activity_audit': {'heavy': True}, 'scene_frames': [{'path': '场景.jpg'}]}],
        'understandings': [{'text': '保留原文', 'score': 1.25}],
        'segments.item.activity_audit': {'must': 'remain'},
        'unknown': {'item': {'activity_audit': 'remain'}}}
    source = tmp_path / 'Index.json'
    source.write_text(json.dumps(value, ensure_ascii=False))
    before = source.read_bytes()
    assert projected_json(source, omit=_READABLE_OMIT) == readable_source_index(value)
    assert source.read_bytes() == before
    with pytest.raises(ProjectionDeferred, match='projected_resident_bytes'):
        projected_json(source, omit=_READABLE_OMIT, max_resident_bytes=1)


@pytest.mark.parametrize('old', [None, 1, {}, {'nested': [1, {'time_index': 'untouched'}]}])
def test_pointer_replacement_preserves_all_other_values_and_has_one_key(tmp_path, old):
    source = tmp_path / 'Index.json'
    original = {'number': 2**80, 'text': '湿实验🧪', 'float': .125,
                'nested': {'time_index': [False, None]}, 'time_index': old}
    source.write_text(json.dumps(original, ensure_ascii=False))
    pointer = {'path': 'Comment/TimeIndex.json', 'schema_version': 'owned/1'}
    replace_json_field(source, 'time_index', pointer)
    actual = json.loads(source.read_bytes(), object_pairs_hook=lambda pairs: pairs)
    assert [k for k, _ in actual].count('time_index') == 1
    assert json.loads(source.read_bytes()) == original | {'time_index': pointer}
    source.write_text('{}')
    replace_json_field(source, 'time_index', pointer)
    assert json.loads(source.read_bytes()) == {'time_index': pointer}


def test_bad_or_replaced_source_cannot_publish_a_partial_pointer(tmp_path):
    source = tmp_path / 'Index.json'
    source.write_text('{"kept": [')
    with pytest.raises(ValueError):
        replace_json_field(source, 'time_index', {})
    assert source.read_text() == '{"kept": ['
    source.write_text('{"kept": 1}')
    def raced(mode):
        source.write_text('{"concurrent": true}')
        return io.BytesIO(b'{"kept": 1}')
    with pytest.raises(ValueError, match='source changed'):
        replace_json_field(source, 'time_index', {}, opener=raced)
    assert source.read_text() == '{"concurrent": true}'
    assert not list(tmp_path.glob('*.partial'))


def test_projection_limit_is_reported_as_an_error_instead_of_a_busy_producer(tmp_path, monkeypatch):
    from test_device_day_file_index import config
    from test_device_day_time_lookup import ARCHIVE, BASE, index
    from visioncortex import receipt_projection
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.device_day_file_index import FileIndexPublisher
    from visioncortex.observed_inventory import observe
    cfg = config(tmp_path)
    path = Path(cfg['storage']['archive_root']) / ARCHIVE / 'ProcessedClips/Index.json'
    atomic_json(path, index())
    before = path.read_bytes()
    observe(Path(cfg['storage']['local_runtime_root']) / 'device-day', {'recordings': [
        {'recording_id': 'slice', 'camera_key': ARCHIVE[11:], 'recording_start_us': BASE}]})
    monkeypatch.setattr(receipt_projection, '_MAX_STREAM_BYTES', 1)
    publisher = FileIndexPublisher(cfg)
    result = publisher.tick()
    assert result['status'] == 'partial' and result['busy'] == result['published'] == 0
    assert result['errors'][0]['reason'] == 'readable_projection_limit'
    assert not publisher.versions and path.read_bytes() == before


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux address-space regression guard')
def test_dense_source_projection_and_pointer_write_fit_a_small_process(tmp_path):
    source = tmp_path / 'Index.json'
    item = json.dumps({'payload': 'x' * 4096}).encode()
    with source.open('wb') as handle:
        handle.write(b'{"segments":[{"activity_audit":[')
        for i in range(16000):
            if i:
                handle.write(b',')
            handle.write(item)
        handle.write(b'],"scene_frames":[]}],"kept":"original"}')
    script = '''
import resource,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(128*1024**2,128*1024**2))
from visioncortex.receipt_projection import projected_json,replace_json_field
p=Path(sys.argv[1])
v=projected_json(p,omit={('segments',None,'activity_audit')})
assert v=={'segments':[{'scene_frames':[]}],'kept':'original'}
replace_json_field(p,'time_index',{'path':'Comment/TimeIndex.json'})
assert projected_json(p,omit={('segments',None,'activity_audit')})==v|{'time_index':{'path':'Comment/TimeIndex.json'}}
print(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')))
'''
    package = Path(__import__('visioncortex').__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, '-c', script, str(source)],
                            env=os.environ | {'PYTHONPATH': str(package)}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) < 96 * 1024

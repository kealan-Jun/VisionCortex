"""Cache eviction changes residence only; publication remains authoritative."""
import sys

from test_device_day_file_index import config
from test_device_day_time_lookup import ARCHIVE, BASE, index
from visioncortex import device_day_content, device_day_file_index
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher, _source_index_size
from visioncortex.observed_inventory import observe


def test_size_counts_nested_unicode_and_shared_children_without_json_copy():
    shared = ['实验员', 123, None]
    value = {'a': shared, 'b': shared}
    one = sys.getsizeof(shared) + sum(map(sys.getsizeof, shared))
    expected = 2 * (sys.getsizeof(value) + sum(map(sys.getsizeof, value)) + 2 * one)
    assert _source_index_size(value, expected) == expected

    class MustNotTraverse(list):
        def __iter__(self):
            raise AssertionError('oversized branch must not be traversed')

    assert _source_index_size(MustNotTraverse(), 1) > 1


def test_lru_keeps_recent_days_and_versions_survive_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(device_day_file_index, '_SOURCE_INDEX_CACHE_DAYS', 2)
    publisher = FileIndexPublisher(config(tmp_path))
    for day in ('one', 'two'):
        publisher._remember_source(day, (1, 2), {'archive': day})
        publisher.versions[day] = ('published', day)
    assert publisher._cached_source('one', (1, 2)) == {'archive': 'one'}
    publisher._remember_source('three', (1, 2), {'archive': 'three'})
    assert list(publisher.source_indexes) == ['one', 'three']
    assert publisher.versions['two'] == ('published', 'two')
    assert publisher.source_index_bytes == sum(publisher._source_index_sizes.values())


def test_byte_budget_and_oversized_revision_drop_stale_cache(tmp_path, monkeypatch):
    value = {'text': 'x' * 300}
    cost = _source_index_size(value, 100000)
    monkeypatch.setattr(device_day_file_index, '_SOURCE_INDEX_CACHE_BYTES', cost + 20)
    publisher = FileIndexPublisher(config(tmp_path))
    publisher._remember_source('one', (1, 2), value)
    publisher._remember_source('two', (1, 2), value)
    assert list(publisher.source_indexes) == ['two']
    assert publisher.source_index_bytes <= cost + 20
    publisher._remember_source('two', (2, 3), {'text': 'x' * 10000})
    assert not publisher.source_indexes and publisher.source_index_bytes == 0
    publisher._remember_source('one', (1, 2), value)
    assert publisher._cached_source('one', (3, 4)) is None
    assert publisher.source_index_bytes == 0


def test_evicted_days_stay_unchanged_and_external_content_overwrite_restores(tmp_path, monkeypatch):
    monkeypatch.setattr(device_day_file_index, '_SOURCE_INDEX_CACHE_DAYS', 1)
    cfg = config(tmp_path)
    cfg['device_day'].update(readable_content_enabled=True, paused_stages=['report'])
    names = [ARCHIVE, ARCHIVE.replace('camera_cam01', 'camera_cam02')]
    from pathlib import Path
    root = Path(cfg['storage']['archive_root'])
    observe(Path(cfg['storage']['local_runtime_root'])/'device-day', {'recordings': [
        {'recording_id': str(i), 'camera_key': name[11:], 'recording_start_us': BASE}
        for i, name in enumerate(names)]})
    for name in names:
        atomic_json(root/name/'ProcessedClips/Index.json', index() | {'archive': name})
    monkeypatch.setattr(device_day_content, 'content_revision', lambda *args: ())
    output_names = ('LaboratoryDailyReport.json', 'LaboratoryDailyReport.html')

    def content(config, source):
        for suffix in output_names:
            path = root/source['archive']/'LaboratoryDailyReport'/suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('verified published content')
        return {}

    monkeypatch.setattr(device_day_content, 'publish_content', content)
    publisher = FileIndexPublisher(cfg)
    assert publisher.tick()['published'] == 2
    assert list(publisher.source_indexes) == [names[1]]
    times = {name: (root/name/'Comment/TimeIndex.json').stat().st_mtime_ns for name in names}
    assert publisher.tick()['unchanged'] == 2
    assert {name: (root/name/'Comment/TimeIndex.json').stat().st_mtime_ns for name in names} == times
    changed = root/names[0]/'LaboratoryDailyReport'/output_names[0]
    changed.write_text('external pending replacement')
    result = publisher.tick()
    assert result['published'] == 1 and result['unchanged'] == 1
    assert changed.read_text() == 'verified published content'
    assert list(publisher.source_indexes) == [names[1]]
    canonical = root/names[0]/'ProcessedClips/Index.json'
    source = read_json(canonical)
    source['recordings'][0]['transcription']['comments'][0]['text'] = '新来源转写'
    atomic_json(canonical, source)
    assert publisher.tick()['published'] == 1
    assert any(s['text'] == '新来源转写' for audio in read_json(
        root/names[0]/'Comment/TimeIndex.json')['streams']['audio']
        for s in audio['transcription']['sentences'])


def test_oversized_day_publishes_normally_without_retaining_source(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(device_day_file_index, '_SOURCE_INDEX_CACHE_BYTES', 1)
    cfg = config(tmp_path)
    cfg['device_day'].update(readable_content_enabled=True, paused_stages=['report'])
    observe(Path(cfg['storage']['local_runtime_root'])/'device-day', {'recordings': [
        {'recording_id': 'slice', 'camera_key': ARCHIVE[11:], 'recording_start_us': BASE}]})
    root = Path(cfg['storage']['archive_root'])/ARCHIVE
    atomic_json(root/'ProcessedClips/Index.json', index())
    publisher = FileIndexPublisher(cfg)
    assert publisher.tick()['published'] == 1
    assert not publisher.source_indexes and publisher.source_index_bytes == 0
    assert read_json(root/'Comment/TimeIndex.json')['archive'] == ARCHIVE
    assert publisher.tick()['unchanged'] == 1

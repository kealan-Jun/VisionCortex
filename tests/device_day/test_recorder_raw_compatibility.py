"""Both recorder raw roots remain native; fixture bytes are synthetic media."""
from pathlib import Path

import pytest

from test_device_day import FakeModels, device_config  # noqa: F401
from test_recorder_metadata import audio, config, incoming, stable  # noqa: F401
from visioncortex.capture_layout import METADATA_LAYOUT, metadata_directories
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import DIRECTORIES, DeviceDayLayout, read_json, safe_child
from visioncortex.nas_recordings import scan_recordings


def native(settings, raw, label='session'):
    folder = incoming(settings, label=label)
    if raw == 'MetaVideo':
        folder.parent.rename(folder.parent.with_name(raw))
        folder = folder.parent.with_name(raw) / folder.name
    audio(folder)
    (folder / 'native-camera-note.txt').write_bytes(b'preserve-native-name-and-content')
    stable(folder)
    return folder


@pytest.mark.parametrize('mode', ['metadata_only', 'legacy_and_metadata'])
@pytest.mark.parametrize('raw', ['MetaData', 'MetaVideo'])
def test_native_root_is_discovered_without_rewriting_sources(config, mode, raw):  # noqa: F811
    config['collection_ingest']['discovery_mode'] = mode
    folder = native(config, raw)
    before = {path.relative_to(folder): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
              for path in folder.rglob('*') if path.is_file()}
    result = scan_recordings(config)
    assert not result['errors']
    record, = result['recordings']
    assert result['camera_directories'] == ['a_cam01']
    assert record['input_layout'] == METADATA_LAYOUT
    assert record['camera_key'] == 'a_cam01' and record['processable']
    assert record['audio']['status'] == 'provided'
    assert record['video_path'] == str(folder / 'rgb.mp4')
    if raw == 'MetaData':
        assert 'source_raw_directory' not in record
    else:
        assert record['source_raw_directory'] == raw
    assert metadata_directories(config) == [folder.parent]
    assert before == {path.relative_to(folder): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                      for path in folder.rglob('*') if path.is_file()}
    assert list(folder.parent.parent.iterdir()) == [folder.parent]


@pytest.mark.parametrize('inplace', [False, True])
def test_metavideo_retention_references_native_bytes_and_creates_no_second_raw_root(
        config, monkeypatch, inplace):  # noqa: F811
    config['device_day']['inplace_preprocessing'] = inplace
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    folder = native(config, 'MetaVideo')
    before = {path.relative_to(folder): (path.read_bytes(), path.stat().st_ino) for path in folder.rglob('*')
              if path.is_file()}
    record, = scan_recordings(config)['recordings']
    runner = DeviceDayRunner(config, FakeModels())
    layout = runner.layout(record)
    assert layout.raw == folder.parent
    original_open = Path.open
    def checked(path, mode='r', *args, **kwargs):
        if path.is_relative_to(folder.parent) and any(flag in mode for flag in 'wa+x'):
            pytest.fail(f'Pipeline attempted to write capture original: {path}')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', checked)
    result = runner.process(record, stage='retention')
    assert result['status'] == 'completed', result
    receipt = read_json(runner._receipt(layout, record, 'retention'))
    assert all(safe_child(layout.root, source['retained']['path']) == Path(source['original_path'])
               for source in receipt['sources'])
    assert before == {path.relative_to(folder): (path.read_bytes(), path.stat().st_ino)
                      for path in folder.rglob('*') if path.is_file()}
    assert {path.name for path in layout.root.iterdir()} == {'MetaVideo', *DIRECTORIES[1:]}


def test_mixed_native_roots_are_both_discovered_and_no_readiness_is_invented(config):  # noqa: F811
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    historical = native(config, 'MetaVideo', label='first')
    current = native(config, 'MetaData', label='second')
    (historical / 'recording_ready.json').unlink()
    before = {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
              for folder in (historical, current) for path in folder.rglob('*') if path.is_file()}
    result = scan_recordings(config)
    assert not result['errors']
    assert result['camera_directories'] == ['a_cam01']
    rows = {Path(row['video_path']).parent: row for row in result['recordings']}
    assert set(rows) == {historical, current}
    assert not rows[historical]['processable']
    assert rows[current]['processable']
    assert {root.name for root in metadata_directories(config)} == {'MetaData', 'MetaVideo'}
    assert {path.name for path in historical.parent.parent.iterdir()} == {'MetaData', 'MetaVideo'}
    assert all((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == expected
               for path, expected in before.items())


def test_missing_explicit_metavideo_is_not_created_as_any_raw_spelling(tmp_path):
    layout = DeviceDayLayout(tmp_path, 'a_cam01', 1789005600000000, raw_directory='MetaVideo')
    with pytest.raises(ValueError, match='Recorder-owned MetaVideo is unavailable'):
        layout.create()
    assert not layout.root.exists()


def test_historical_retained_video_without_native_handoff_is_not_new_input(config):  # noqa: F811
    config['collection_ingest']['discovery_mode'] = 'metadata_only'
    folder = native(config, 'MetaData')
    historical = folder.parent.with_name('MetaVideo')
    historical.mkdir()
    (historical / '10-00-00_10-15-00.mp4').write_bytes(b'historical-retained-unknown-media')
    (historical / 'rgb.mp4').write_bytes(b'historical-unknown-media')
    records = scan_recordings(config)['recordings']
    assert [row['video_path'] for row in records] == [str(folder / 'rgb.mp4')]

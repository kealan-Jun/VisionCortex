from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from repo_paths import ROOT
from visioncortex.device_day_legacy_ingest import LegacyIntake, check_record, intake_config
from visioncortex.nas_recordings import scan_recordings


@pytest.fixture
def settings(default_config, tmp_path):
    config = deepcopy(default_config)
    config['runtime']['local_only'] = True
    config['device_day'].update(enabled=True, start_date=None, raw_originals_read_only=True,
                               inplace_preprocessing=True, camera_lanes=True,
                               legacy_ingest={'enabled': True, 'directories': ['lab_video'], 'max_imports_per_turn': 2})
    source = tmp_path / 'source'
    config['storage'].update(archive_root=str(source / 'archive'), local_cache_root=str(tmp_path / 'cache'),
                             local_runtime_root=str(tmp_path / 'runtime'))
    (source / 'archive').mkdir(parents=True)
    config['collection_ingest'].update(enabled=True, mode='directory_metadata', source_root=str(source), discovery_mode='metadata_only',
        additional_camera_directories=['lab_video'], camera_directories=[], settle_seconds=0,
        directory_camera_bindings=[
            {'directory_glob': 'lab_video/20??-??-??-1/*', 'camera_key': 'lab-video-1'},
            {'directory_glob': 'lab_video/20??-??-??-3/*', 'camera_key': 'lab-video-3'},
            {'directory_glob': 'lab_video/20??-??-??-3/20??-??-??/*', 'camera_key': 'lab-video-3'},
            {'directory_glob': 'lab_video/2026-10-20/*', 'camera_key': 'lab-video-3'}],
        camera_role_map={'lab-video-1': 'first_person', 'lab-video-3': 'third_person'},
        camera_group_map={'lab-video-1': 'lab-video', 'lab-video-3': 'lab-video'})
    return config


def capture(config, relative, start=1792480000000000):
    path = Path(config['collection_ingest']['source_root']) / relative
    path.mkdir(parents=True)
    (path / 'rgb.mp4').write_bytes(b'not-decoded-retention-source')
    (path / 'frames.csv').write_text(f'global_timestamp_us,rgb_video_frame_index\n{start},0\n')
    meta = {'camera_key': 'physical_cam01', 'closed': True, 'frames_publish_state': 'finalized',
            'rgb_file': 'rgb.mp4', 'frames_file': 'frames.csv', 'segment_start_us': start,
            'segment_end_us': start + 1000000, 'recording_session_id': start - 1000000}
    (path / 'meta.json').write_text(json.dumps(meta))
    (path / 'recording_ready.json').write_text(json.dumps(meta | {'ready': True, 'recording_complete': False,
                                                               'recording_quality_status': 'partial'}))
    return path


def test_explicit_legacy_intake_keeps_native_monitor_scope_and_finds_nested_date(settings):
    capture(settings, 'lab_video/2026-10-18-3/2026-10-18/175944')
    capture(settings, 'lab_video/2026-10-20/113833')
    capture(settings, 'unrelated_cam01/2026-10-20/113833')
    before = deepcopy(settings)
    assert scan_recordings(settings)['recordings'] == []
    rows = scan_recordings(intake_config(settings))['recordings']
    assert len(rows) == 2
    assert all(row['processable'] for row in rows), rows
    assert all(row['camera_key'] == 'lab-video-3' and row['configured_role'] == 'third_person' for row in rows)
    assert settings == before
    scoped = intake_config(settings)
    scoped['collection_ingest']['capture_date'] = '2026-10-20'
    assert [row['relative_path'] for row in scan_recordings(scoped)['recordings']] == ['lab_video/2026-10-20/113833/rgb.mp4']


@pytest.mark.parametrize('directories', [[], ['../elsewhere'], ['unrelated_cam01'], ['lab_video', 'lab_video']])
def test_intake_rejects_unapproved_roots(settings, directories):
    settings['device_day']['legacy_ingest']['directories'] = directories
    with pytest.raises(ValueError):
        intake_config(settings)


def test_retention_precedes_observation_is_latest_first_and_resumes_without_paid_work(settings):
    capture(settings, 'lab_video/2026-10-19-1/113833', start=1792390000000000)
    capture(settings, 'lab_video/2026-10-20/113833')
    events = []

    def retain(record):
        events.append(('retention', record['relative_path']))
        return {'status': 'completed', 'stage': 'retention'}

    def observe(config, inventory):
        events.append(('observe', inventory['recordings'][0]['relative_path']))

    worker = LegacyIntake(settings, retain, observe)
    result = worker.tick()
    assert [kind for kind, _ in events] == ['retention', 'observe', 'retention', 'observe']
    assert '2026-10-20/' in events[0][1]
    assert result['retained'] == 2 and result['model_calls'] == 0
    events.clear()
    result = LegacyIntake(settings, retain, observe).tick()
    assert [kind for kind, _ in events] == ['observe', 'observe']
    assert result['reused_this_turn'] == 2


def test_failed_retention_does_not_admit_models_or_block_other_inputs(settings):
    capture(settings, 'lab_video/2026-10-19-1/113833', start=1792390000000000)
    capture(settings, 'lab_video/2026-10-20/113833')
    observed = []

    def retain(record):
        if '/2026-10-20/' in record['video_path']:
            raise OSError('NAS unavailable')
        return {'status': 'completed', 'stage': 'retention'}

    result = LegacyIntake(settings, retain, lambda c, i: observed.extend(i['recordings'])).tick()
    assert result['status'] == 'degraded' and len(result['errors']) == 1
    assert len(observed) == 1 and '2026-10-19-1' in observed[0]['relative_path']


def test_bounded_turn_rescans_newest_arrival_before_older_backlog(settings):
    settings['device_day']['legacy_ingest']['max_imports_per_turn'] = 1
    capture(settings, 'lab_video/2026-10-19-1/113833', start=1792390000000000)
    capture(settings, 'lab_video/2026-10-20/113833')
    imported = []

    def retain(record):
        imported.append(record['relative_path'])
        return {'status': 'completed', 'stage': 'retention'}

    worker = LegacyIntake(settings, retain, lambda c, i: None)
    first = worker.tick()
    assert first['imports_attempted'] == 1 and first['retained'] == 1
    capture(settings, 'lab_video/2026-10-21-1/113833', start=1792570000000000)
    worker.tick()
    assert imported == ['lab_video/2026-10-20/113833/rgb.mp4', 'lab_video/2026-10-21-1/113833/rgb.mp4']


def test_retention_port_copies_only_its_bound_legacy_source_and_keeps_stage_key(settings):
    from visioncortex.device_day import DeviceDayRunner

    source = capture(settings, 'lab_video/2026-10-20/113833')
    originals = {p: p.read_bytes() for p in source.iterdir()}
    record = scan_recordings(intake_config(settings))['recordings'][0]
    assert record['processable'], record
    spec = importlib.util.spec_from_file_location('retain_legacy_capture', ROOT / 'tools/migrations/retain_legacy_capture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = deepcopy(settings)
    result = module.retain(settings, record)
    assert result['status'] == 'completed' and result['stage'] == 'retention'
    assert result['vision_admitted'] is True
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert settings == before and settings['device_day']['raw_originals_read_only'] is True
    scoped = deepcopy(settings)
    scoped['device_day']['raw_originals_read_only'] = False
    assert DeviceDayRunner(settings, backend=object())._key('retention', record, {}) == DeviceDayRunner(
        scoped, backend=object())._key('retention', record, {})
    runner = DeviceDayRunner(settings, backend=object())
    assert runner.queues['vision'].admissions()[record['recording_id']] == record['source_signature']
    assert runner.queues['understanding'].pending() == []
    for invalid in [record | {'input_layout': 'visioncortex-recorder-metadata/1'},
                    record | {'relative_path': 'other/rgb.mp4'}, record | {'camera_key': 'physical_cam01'}]:
        with pytest.raises(ValueError):
            module.retain(settings, invalid)
        with pytest.raises(ValueError):
            check_record(intake_config(settings), invalid)

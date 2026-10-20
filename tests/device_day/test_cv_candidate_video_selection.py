"""Actual CV adapter dispatch with synthetic scan/decoder ports, no real model."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex.device_day import DeviceDayRunner, visual_input
from visioncortex.device_day_contract import (
    DeviceDayLayout, artifact, atomic_json, digest, file_hash, read_json, verify_artifact,
)
from visioncortex.device_day_models import DeviceDayModels
from visioncortex.schemas import FrameEvidence, ViewRole


@pytest.fixture
def scan_case(default_config, tmp_path, monkeypatch):
    from visioncortex import device_day_models as module
    start = 1792376825000000
    layout = DeviceDayLayout(tmp_path / 'Archive', 'camera_cam01', start, tmp_path / 'Cache',
                             raw_directory='MetaData', read_only_originals=True)
    raw = layout.root / 'MetaData' / '102705'
    raw.mkdir(parents=True)
    video, clock = raw / 'rgb.mp4', raw / 'frames.csv'
    video.write_bytes(b'synthetic source; never decoded as a real recording')
    clock.write_bytes(b'original collector clock bytes')
    record = {'recording_id': 'recording', 'camera_key': 'camera_cam01', 'configured_role': 'first_person',
              'recording_start_us': start, 'recording_end_us': start + 1200000,
              'source_signature': 'f' * 64, 'capture_complete': True}
    config = deepcopy(default_config)
    config['storage'].update(archive_root=str(layout.root.parent), local_cache_root=str(layout.backend_root),
                             local_runtime_root=str(tmp_path / 'Runtime'))
    config['collection_ingest'].update(source_root=str(layout.root), enabled=True,
                                       camera_role_map={'camera_cam01': 'first_person'})
    config['device_day'].update(enabled=True, inactive_frames=3, inplace_preprocessing=False)
    config['runtime']['local_only'] = True
    config['performance']['coarse_detection_fps'] = 2
    refs = [artifact(layout.root, p) for p in (video, clock)]
    retained = {'status': 'completed', 'recording': record, 'artifacts': refs,
                'sources': [{'kind': kind, 'original_path': str(path), 'retained': ref}
                            for kind, path, ref in zip(('video', 'clock'), (video, clock), refs, strict=True)]}
    model = DeviceDayModels(config)
    info = SimpleNamespace(fps=30, frame_count=30, duration_ms=1000)
    monkeypatch.setattr(model, '_probe', lambda _: info)
    mapping = {'basis': 'recorder_csv_interpolation', 'origin_us': start,
               'points': [[0, start], [500, start + 500000], [1000, start + 1200000]]}
    monkeypatch.setattr(module, 'capture_clock', lambda *_: mapping)
    native = {'native_frame_count': 30, 'available_windows': [(0, 1000)]}
    monkeypatch.setattr('visioncortex.source_frames.SourceFrameTrace', lambda *_: object())
    monkeypatch.setattr('visioncortex.source_frames.source_observation_windows', lambda *_: native)
    validations, scans, reads, exports = [], [], [], []
    monkeypatch.setattr('visioncortex.detection.validate_models',
                        lambda _: validations.append('validated') or {'fixture': True, 'real_model_invoked': False})
    frame = FrameEvidence(view_id='camera_cam01', role=ViewRole.FIRST_PERSON,
                          frame_index=15, local_ms=500, width=16, height=16)
    monkeypatch.setattr('visioncortex.detection.iter_frame_evidence',
                        lambda _: iter([frame.model_copy(update={'local_ms': 0}), frame]))
    caches = {}
    for phase in ('coarse', 'fine'):
        directory = layout.backend_root / 'ScanFixture' / phase
        directory.mkdir(parents=True)
        ledger = directory / 'camera_cam01.detections.jsonl'
        ledger.write_bytes(b'synthetic detector ledger')
        report = directory / 'runtime_fixture.json'
        atomic_json(report, {'phase': phase, 'deterministic_fixture': True})
        atomic_json(directory / 'scan-receipt.json', {'status': 'completed',
            'artifacts': [artifact(directory, p) for p in (ledger, report)]})
        caches[phase] = directory, ledger
    case = SimpleNamespace(layout=layout, record=record, retained=retained, model=model, config=config,
        video=video, clock=clock, mapping=mapping, native=native, scans=scans, validations=validations,
        reads=reads, exports=exports, active=True, reused=False)
    def scan(view, _info, _transform, _retention, phase, _windows, _fps, _image_size):
        scans.append(phase)
        directory, ledger = caches[phase]
        return {view.view_id: ledger}, directory, {'current_wall_seconds': 0, 'reused': case.reused}
    monkeypatch.setattr(model, '_scan', scan)
    monkeypatch.setattr(model, '_plan', lambda *_: ([], [(0, 1000)], {'fixture': True}, {'reused': case.reused}))
    monkeypatch.setattr(model, '_index_fine', lambda *args: (args[-2], {'enabled': True, 'fixture': True}, []))
    event = {'event_id': 'event', 'action_type': 'object_movement', 'key_global_ms': 500}
    def audit(*_):
        return [], [(0, 1000)] if case.active else [], {
            'status': 'completed', 'selected_key_events': [event] if case.active else [],
            'physical_action_confirmed': False}, {'event': {'local_ms': 500,
                                                           'frame_evidence': frame.model_dump(mode='json')}}
    monkeypatch.setattr(model, '_audit_fine', audit)
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    class Reader:
        def __init__(self, **_):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            reads.append('closed')
        def read_path(self, path, timestamp):
            reads.append((path, timestamp))
            return image
    monkeypatch.setattr('visioncortex.video_io.ViewFrameReader', Reader)
    monkeypatch.setattr('visioncortex.source_frames.read_evidence_frame', lambda *_: (
        image, {'status': 'verified', 'view_local_ms': 500, 'test_only': True}))
    def forbidden_export(*args, **kwargs):
        exports.append((args, kwargs))
        pytest.fail('CV activity classification must never export a whole candidate video')
    monkeypatch.setattr('visioncortex.video_io.extract_clip', forbidden_export)
    return case


@pytest.mark.parametrize('active,reused', [(True, False), (True, True), (False, False)])
def test_cv_classification_keeps_candidates_frames_and_audit_without_video_export(scan_case, active, reused):
    case = scan_case
    case.active, case.reused = active, reused
    originals = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                 for p in (case.video, case.clock)}
    result = case.model.vision(case.layout, case.retained, 'new-execution-key')
    assert case.scans == ['coarse', 'fine'] and case.validations == ['validated']
    assert case.exports == [] and case.reads[-1] == 'closed'
    assert len(result['segments']) == 1
    segment = result['segments'][0]
    assert segment['activity'] == ('active' if active else 'inactive')
    assert segment['video'] is None and 'historical_scope' not in segment
    assert len(segment['scene_frames']) == 3 and len(segment['key_frames']) == int(active)
    assert segment['source_ref'] == case.retained['sources'][0]['retained']
    assert segment['clock_ref'] == case.retained['sources'][1]['retained']
    assert (segment['start_us'], segment['end_us']) == (case.record['recording_start_us'],
                                                     case.record['recording_end_us'])
    assert result['clock_mapping'] == case.mapping and result['media_coverage'] == case.native
    assert result['batches'][0]['coarse_timing']['reused'] == reused
    assert result['batches'][0]['fine_timing']['reused'] == reused
    assert {report['phase'] for report in result['scan_reports']} == {'coarse', 'fine'}
    assert read_json(case.layout.root / segment['json_path']) == segment
    assert all(verify_artifact(case.layout.root, ref) for ref in result['artifacts'])
    assert all(verify_artifact(case.layout.backend_root, ref) for ref in result['audit_artifacts'])
    audit = read_json(case.layout.backend_root / segment['activity_audit_ref']['path'])
    assert bool(audit['selected_key_events']) == active
    assert not list(case.layout.processed.rglob('*.mp4'))
    for path, snapshot in originals.items():
        assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == snapshot


def test_exact_historical_checkpoint_retains_whole_video_and_receipt_bytes_without_rescan(scan_case):
    case = scan_case
    folder = case.layout.processed / 'Clips' / '10-27-05_10-27-06_ExperimentActivity_legacy'
    folder.mkdir(parents=True)
    video = folder / 'ExperimentActivity.mp4'
    video.write_bytes(b'previously generated whole candidate; historical evidence')
    candidate = {'segment_id': 'segment-legacy', 'recording_id': case.record['recording_id'],
        'activity': 'active', 'start_ms': 0, 'end_ms': 1000,
        'start_us': case.record['recording_start_us'], 'end_us': case.record['recording_end_us'],
        'source_ref': case.retained['sources'][0]['retained'], 'video': artifact(case.layout.root, video),
        'json_path': case.layout.relative(folder / 'ExperimentActivity.json')}
    atomic_json(case.layout.root / candidate['json_path'], candidate)
    receipt = {'stage': 'vision', 'status': 'completed', 'key': 'a' * 64,
        'recording_id': case.record['recording_id'], 'segments': [candidate],
        'artifacts': [candidate['video'], artifact(case.layout.root, case.layout.root / candidate['json_path'])]}
    path = case.layout.receipts / case.record['recording_id'] / 'vision.json'
    atomic_json(path, receipt)
    expected = DeviceDayRunner(case.config, case.model)._key('vision', case.record, visual_input(case.retained))
    manifest = case.layout.backend_root / 'ReviewedHistoricalCheckpoint.json'
    atomic_json(manifest, {'schema_version': 'visioncortex-resume-checkpoints/1', 'entries': [{
        'stage': 'vision', 'recording_id': case.record['recording_id'], 'expected_key': expected,
        'receipt_key': receipt['key'], 'receipt_digest': digest(receipt), 'queue_revision': 'c' * 64}]})
    config = deepcopy(case.config)
    config['device_day']['completed_stage_receipts'] = {'path': str(manifest), 'sha256': file_hash(manifest)}
    original_bytes = {p: p.read_bytes() for p in (path, video, case.layout.root / candidate['json_path'])}
    runner = DeviceDayRunner(config, case.model)
    loaded = runner._load(path, expected, case.layout)
    assert loaded == receipt and case.scans == case.exports == case.validations == []
    assert loaded['segments'][0]['video'] == candidate['video']
    assert 'historical_scope' not in read_json(path)['segments'][0]
    from visioncortex.application.retained_video import publish_selected
    index = {'archive': case.layout.name, 'segments': deepcopy(loaded['segments']), 'understandings': [],
             'recordings': [case.record | {'start_us': case.record['recording_start_us'],
                                         'end_us': case.record['recording_end_us']}]}
    assert publish_selected(config, index)['materialized'] == 0
    assert index['segments'][0]['historical_scope']['selection_status'] == 'historical_retained_not_revalidated_for_exact_selection'
    assert case.exports == []
    for filename, original in original_bytes.items():
        assert filename.read_bytes() == original

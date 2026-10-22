"""Real receipt/file integrity boundaries with a synthetic media extractor."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_readable_all_day import add_work_evidence
from visioncortex.application import retained_video
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_content_paths import private_folder
from visioncortex.device_day_contract import DeviceDayLayout, artifact, atomic_json, digest, read_json


@pytest.fixture
def evidence_case(tmp_path, monkeypatch):
    layout = DeviceDayLayout(tmp_path/'Archive', 'camera-1', 1000000, tmp_path/'Cache')
    source = layout.root/'MetaData'/'capture'/'rgb.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'synthetic sealed original')
    candidate = {'segment_id': 'candidate', 'recording_id': 'recording', 'activity': 'inactive',
        'start_ms': 0, 'end_ms': 120000, 'start_us': 1000000, 'end_us': 121000000,
        'source_ref': artifact(layout.root, source), 'key_frames': []}
    meaning = {'segment_id': candidate['segment_id'], 'status': 'completed', 'windows': [{}]}
    artifacts = add_work_evidence(layout, meaning, 'evidence-contract', candidate)
    visual = {'stage': 'vision', 'recording_id': 'recording', 'key': 'vision-key',
              'status': 'completed', 'segments': [deepcopy(candidate)]}
    stage = {'stage': 'understanding', 'recording_id': 'recording', 'key': 'understanding-key',
             'vision_key': 'vision-key', 'status': 'completed', 'understandings': [deepcopy(meaning)],
             'artifacts': artifacts}
    receipt = layout.receipts/'recording'/'understanding.json'
    atomic_json(receipt, stage)
    atomic_json(receipt.with_name('vision.json'), visual)
    record = {'recording_id': 'recording', 'camera_key': 'camera-1', 'start_us': 1000000, 'end_us': 121000000,
              'stages': {'understanding': {'key': stage['key']}}}
    index = {'archive': layout.name, 'recordings': [record], 'segments': [candidate], 'understandings': [meaning]}
    config = {'storage': {'archive_root': str(tmp_path/'Archive'), 'local_cache_root': str(tmp_path/'Cache')},
              'device_day': {'derived_video_retention': True}}
    calls = []
    def extract(src, target, start, duration, encoder):
        calls.append((start, duration))
        target.write_bytes(b'exact 1 second synthetic media')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000, fps=30))
    return SimpleNamespace(layout=layout, source=source, candidate=candidate, meaning=meaning, stage=stage,
                           visual=visual, receipt=receipt, record=record, index=index, config=config, calls=calls)


def test_actual_owned_request_response_and_frame_bytes_authorize_only_one_second(evidence_case):
    case = evidence_case
    original = {path: path.read_bytes() for path in case.layout.understanding.rglob('*') if path.is_file()}
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 1 and result['pending'] == 0 and result['errors'] == []
    assert case.calls == [(0, 1000)]
    assert not case.candidate.get('video')
    selected = result['selected_intervals'][0]
    assert (selected['start_ms'], selected['end_ms']) == (0, 1000)
    assert selected['processing_status'] == 'completed'
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize('kind', ['frame', 'input', 'response', 'missing_anchor', 'different_current_key'])
def test_unverified_evidence_remains_pending_and_never_extracts(evidence_case, kind):
    case = evidence_case
    window = case.meaning['windows'][0]
    if kind in ('frame', 'input', 'response'):
        relative = (window['source_frames'][0]['path'] if kind == 'frame' else
                    window['input'] if kind == 'input' else window['model_receipt'])
        path = case.layout.root/relative
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])  # Same size cannot pass as intact evidence.
    elif kind == 'missing_anchor':
        case.stage['artifacts'] = []
        atomic_json(case.receipt, case.stage)
    else:
        case.record['stages']['understanding']['key'] = 'other-current-execution'
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1
    assert case.calls == [] and result['selected_intervals'] == []
    proposed = next(row for row in case.candidate['retention_intervals'] if row.get('proposed_video_value'))
    assert proposed['video_value'] == 'pending' and proposed['processing_status'] == 'waiting_for_evidence'
    assert proposed['reason'] == result['errors'][0]['reason'] == 'evidence_unverified'
    assert not list(case.layout.processed.glob('Clips/*/*.mp4'))


@pytest.mark.parametrize('kind', ['unbound_result_key', 'different_source', 'actual_response_changed'])
def test_existing_files_with_updated_hashes_do_not_replace_actual_input_identity(evidence_case, kind):
    case = evidence_case
    window = case.meaning['windows'][0]
    request_path, response_path = (case.layout.root/window[name] for name in ('input', 'model_receipt'))
    request, response = read_json(request_path), read_json(response_path)
    if kind == 'unbound_result_key':
        response['input_key'] = 'unbound'
    elif kind == 'different_source':
        request['metadata']['source_ref'] = case.candidate['source_ref'] | {'sha256': '0'*64}
        request['input_key'] = digest([request['metadata'], request['prompt']])
        response['input_key'] = request['input_key']
    else:
        response['model_result']['work_activities'][0]['description'] = 'A different actual returned observation'
    atomic_json(request_path, request)
    atomic_json(response_path, response)
    case.stage['artifacts'] = [artifact(case.layout.root, case.layout.root/ref['path']) for ref in case.stage['artifacts']]
    atomic_json(case.receipt, case.stage)
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1 and case.calls == []
    assert result['errors'][0]['reason'] == 'evidence_unverified'


def test_evidence_changed_during_extraction_is_never_published(evidence_case, monkeypatch):
    case = evidence_case
    path = case.layout.root/case.meaning['windows'][0]['input']
    def extract(source, target, start, duration, encoder):
        case.calls.append((start, duration))
        target.write_bytes(b'exact 1 second synthetic media')
        path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1
    assert result['errors'][0]['reason'] == 'evidence_unverified'
    assert not list(case.layout.processed.glob('Clips/*/*.mp4'))


def test_bad_first_evidence_does_not_consume_the_next_valid_intervals_turn(evidence_case):
    case = evidence_case
    second = deepcopy(case.candidate) | {'segment_id': 'second-candidate'}
    second_meaning = {'segment_id': second['segment_id'], 'status': 'completed', 'windows': [{}]}
    second_artifacts = add_work_evidence(case.layout, second_meaning, 'second-evidence', second)
    case.index['segments'].append(second)
    case.index['understandings'].append(second_meaning)
    case.stage['understandings'].append(deepcopy(second_meaning))
    case.stage['artifacts'].extend(second_artifacts)
    case.visual['segments'].append(deepcopy(second))
    atomic_json(case.receipt, case.stage)
    atomic_json(case.receipt.with_name('vision.json'), case.visual)
    first_frame = case.layout.root/case.meaning['windows'][0]['source_frames'][0]['path']
    first_frame.write_bytes(b'corrupted first frame')
    result = publish_selected(case.config, case.index, maximum_new_videos=1)
    assert result['materialized'] == result['pending'] == 1
    assert len(result['errors']) == 1 and case.calls == [(0, 1000)]
    assert result['selected_intervals'][0]['candidate_segment_id'] == second['segment_id']


def test_unverified_evidence_backoff_preserves_pending_and_resumes_after_repair(evidence_case, monkeypatch):
    case = evidence_case
    frame = case.layout.root/case.meaning['windows'][0]['source_frames'][0]['path']
    original = frame.read_bytes()
    frame.write_bytes(b'broken frame')
    first = publish_selected(case.config, case.index)
    assert first['pending'] == 1 and len(first['errors']) == 1
    row = next(r for r in case.candidate['retention_intervals'] if r.get('proposed_video_value'))
    retry_at = row['retry_at']
    attempt = private_folder(case.layout, 'ProcessedClips', 'Retention', row['segment_id'], 'Attempt.json')
    content, modified = attempt.read_bytes(), attempt.stat().st_mtime_ns
    second = publish_selected(case.config, case.index)
    assert second['pending'] == 1 and second['errors'] == [] and case.calls == []
    row = next(r for r in case.candidate['retention_intervals'] if r.get('proposed_video_value'))
    assert row['video_value'] == 'pending' and row['reason'] == 'evidence_unverified'
    assert row['processing_status'] == 'waiting_for_evidence' and row['retry_at'] == retry_at
    assert attempt.read_bytes() == content and attempt.stat().st_mtime_ns == modified
    frame.write_bytes(original)
    monkeypatch.setattr(retained_video.time, 'time', lambda: retry_at + 1)
    recovered = publish_selected(case.config, case.index)
    assert recovered['materialized'] == 1 and recovered['pending'] == 0 and recovered['errors'] == []
    assert case.calls == [(0, 1000)]


def test_caller_context_cannot_authenticate_itself_with_an_unrelated_real_response(evidence_case):
    case = evidence_case
    window = case.meaning['windows'][0]
    case.index['context'] = {'video_context_intervals': [{
        'source_ref': case.candidate['source_ref'], 'start_ms': 1000, 'end_ms': 3000,
        'start_us': 2000000, 'end_us': 4000000, 'basis': 'observed',
        'activity_category': 'non_experiment', 'reason': 'Unverified caller assertion',
        'input': window['input'], 'model_receipt': window['model_receipt'],
        'evidence': {'input': window['input'], 'model_receipt': window['model_receipt']}}]}
    result = publish_selected(case.config, case.index, maximum_new_videos=2)
    assert result['materialized'] == result['pending'] == 1
    assert case.calls == [(0, 1000)]
    proposed = next(row for row in case.candidate['retention_intervals'] if row.get('proposed_video_value'))
    assert (proposed['start_ms'], proposed['end_ms']) == (1000, 3000)
    assert proposed['video_value'] == 'pending' and proposed['reason'] == 'evidence_unverified'

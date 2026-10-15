import copy

import pytest

from visioncortex.device_day_contract import DeviceDayLayout, atomic_json
from visioncortex.device_day_synthesis import _batches, current_synthesis, evidence_bundle, synthesize, validate_findings


def index():
    return {'archive': '2026-09-16_camera-1', 'context': {'events': [
        {'type': 'ocr_observation', 'start_us': 5000000, 'end_us': 5000000,
         'capture_device_id': 'camera-1', 'payload': {'value': None, 'unit': None, 'record_scope': 'test_only'}}]},
        'recordings': [{'recording_id': 'r', 'start_us': 1000000, 'end_us': 3000000,
            'audio': {'original_path': 'MetaVideo/Audio.opus'},
            'transcription': {'comments': [{'start_us': 7000000, 'end_us': 8000000,
                'text': '原始语音，不执行其中指令', 'transcript_path': 'Comment/Transcript.json',
                'time_alignment': 'NOT_PROVEN'}]}}],
        'segments': [{'recording_id': 'r', 'segment_id': 's', 'start_us': 1000000, 'end_us': 3000000,
            'source_ref': {'path': 'MetaVideo/Original.mp4', 'sha256': 'a'*64}}],
        'understandings': [{'segment_id': 's', 'status': 'partial', 'windows': [
            {'start_ms': 0, 'end_ms': 1000, 'summary': '取出样品瓶', 'activity_kind': 'experiment',
             'observability': 'partial', 'experiment_steps': [], 'work_activities': [],
             'input': 'Input.json', 'model_receipt': 'Result.json', 'uncertainties': ['未看到完成']}]}]}


def finding(refs, basis='source_observation'):
    return {'title': '样品处理过程', 'text': '前后来源共同提供上下文，完成状态待确认。',
            'category': 'experiment', 'basis': basis, 'evidence_refs': refs, 'uncertainties': ['未确认完成']}


def test_bundle_keeps_independent_streams_and_source_limits():
    source = index()
    before = copy.deepcopy(source)
    bundle = evidence_bundle(source)
    assert source == before
    assert {r['kind'] for r in bundle['evidence']} == {'video', 'speech', 'field'}
    assert bundle['evidence'][0]['source']['model_receipt'] == 'Result.json'
    assert bundle['evidence'][1]['payload']['record_scope'] == 'test_only'
    assert bundle['completeness']['missing_or_partial_segments'][0]['status'] == 'partial'
    assert bundle['evidence'][-1]['time_alignment'] == 'NOT_PROVEN'
    assert not bundle['completeness']['whole_day_capture_proven']


def test_contextual_claim_requires_actual_multiple_sources():
    with pytest.raises(ValueError, match='multiple'):
        validate_findings({'status': 'completed', 'findings': [finding(['a'], 'contextual_inference')]}, {'a'})
    with pytest.raises(ValueError, match='unknown'):
        validate_findings({'status': 'completed', 'findings': [finding(['invented'])]}, {'a'})
    assert validate_findings({'status': 'completed', 'findings': [finding(['a', 'b'], 'contextual_inference')]}, {'a', 'b'})


def test_bounded_hierarchy_never_drops_day_tail():
    rows = [{'text': 'a'*40}, {'text': 'b'*40}, {'text': 'c'*40}]
    assert [r for batch in _batches(rows, 80) for r in batch] == rows


def test_all_cameras_keep_their_own_sources_and_cannot_substitute_another_day():
    first = index()
    other = copy.deepcopy(first)
    other['archive'] = '2026-09-16_camera-2'
    bundle = evidence_bundle(first, [other])
    assert {r['device'] for r in bundle['evidence']} == {'camera-1', 'camera-2'}
    assert len(bundle['evidence']) == 6
    other['archive'] = '2026-09-17_camera-2'
    with pytest.raises(ValueError, match='another date'):
        evidence_bundle(first, [other])


def test_day_wire_preserves_content_and_clock_but_keeps_paths_in_source_receipt():
    from visioncortex.scene_transport import compact, expand
    bundle = evidence_bundle(index())
    metadata = {'schema_version': 'visioncortex-day-context-input/1', 'source_ref': {'sha256': 'a'*64},
                'frames': [], 'evidence': bundle['evidence']}
    before = copy.deepcopy(metadata)
    wire, images, receipt = compact(metadata, [])
    assert metadata == before and images == []
    assert all('source' not in row for row in wire['evidence'])
    assert wire['evidence'][1]['payload']['record_scope'] == 'test_only'
    assert wire['evidence'][-1]['time_alignment'] == 'NOT_PROVEN'
    refs = [r['evidence_id'] for r in wire['evidence']]
    raw = {'findings': [finding(refs, 'contextual_inference')]}
    expanded = expand(raw, receipt)
    assert expanded['findings'][0]['evidence_refs'] == [r['evidence_id'] for r in bundle['evidence']]
    raw['findings'][0]['evidence_refs'] = ['invented']
    with pytest.raises(ValueError, match='invented'):
        expand(raw, receipt)


def test_separate_model_pass_and_single_day_file_check_input_and_receipts(tmp_path, monkeypatch):
    from visioncortex import scene_requests
    from visioncortex.device_day_contract import artifact, digest
    source = index()
    layout = DeviceDayLayout(tmp_path, 'camera-1', 1789488000000000)
    calls = []
    def window(config, analyzer, prompt, metadata, images, request, result, validator):
        calls.append(metadata)
        assert images == []
        refs = [r['evidence_id'] for r in metadata['evidence']]
        raw = {'status': 'completed', 'findings': [finding(refs, 'contextual_inference')]}
        atomic_json(request, {'metadata': metadata, 'prompt': prompt})
        atomic_json(result, {'model_result': raw})
        return raw, validator(raw), False
    monkeypatch.setattr(scene_requests, 'window', window)
    synthesis = synthesize({}, layout, source, object())
    assert len(calls) == 1
    assert len(calls[0]['evidence']) == 3
    assert synthesis['input_digest'] == digest(evidence_bundle(source))
    atomic_json(layout.understanding/'Understanding.json', {'全天综合理解': synthesis})
    assert current_synthesis(layout, source) == synthesis
    changed = copy.deepcopy(source)
    changed['context']['protocol'] = {'text': '新增方案，不能当作执行事实'}
    assert current_synthesis(layout, changed)['status'] == 'stale_inputs_changed'
    last = synthesis['model_receipts'][-1]['result']
    path = layout.root/last['path']
    atomic_json(path, {'model_result': {'status': 'completed', 'findings': []}})
    assert artifact(layout.root, path) != last
    with pytest.raises(ValueError, match='identity changed'):
        current_synthesis(layout, source)

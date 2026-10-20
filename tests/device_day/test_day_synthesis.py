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
    assert {r['kind'] for r in bundle['evidence']} == {'video', 'speech', 'field', 'coverage'}
    assert bundle['evidence'][0]['source']['model_receipt'] == 'Result.json'
    assert bundle['evidence'][1]['payload']['record_scope'] == 'test_only'
    assert bundle['completeness']['missing_or_partial_segments'][0]['status'] == 'partial'
    assert next(r for r in bundle['evidence'] if r['kind'] == 'speech')['time_alignment'] == 'NOT_PROVEN'
    assert not bundle['completeness']['whole_day_capture_proven']


def test_day_context_excludes_explicit_placeholder_work_but_keeps_real_uncertain_waiting():
    source = index()
    window = source['understandings'][0]['windows'][0]
    placeholder = {'description': '仅为占位，不承载动作', 'basis': 'uncertain',
                   'outcome': 'incomplete_or_uncertain', 'before_state': '未知', 'after_state': '未知'}
    waiting = placeholder | {'description': '等待仪器读数变化', 'category': 'waiting'}
    window['work_activities'] = [placeholder, waiting]
    before = copy.deepcopy(source)
    video = next(row for row in evidence_bundle(source)['evidence'] if row['kind'] == 'video')
    assert video['work_activities'] == [waiting]
    assert any('占位' in reason for reason in video['uncertainties'])
    assert video['source']['model_receipt'] == 'Result.json' and source == before


def test_contextual_claim_requires_actual_multiple_sources():
    with pytest.raises(ValueError, match='multiple'):
        validate_findings({'status': 'completed', 'findings': [finding(['a'], 'contextual_inference')]}, {'a'})
    with pytest.raises(ValueError, match='unknown'):
        validate_findings({'status': 'completed', 'findings': [finding(['invented'])]}, {'a'})
    assert validate_findings({'status': 'completed', 'findings': [finding(['a', 'b'], 'contextual_inference')]}, {'a', 'b'})


def test_missing_processed_video_is_pending_value_review_never_no_value():
    from visioncortex.scene_transport import _day_completeness
    source = index()
    source.update(processing_scope='selected_processed_clips_only', selected_clip_count=2,
        pending_value_review_intervals=[{'start_us': 9000000, 'end_us': 12000000,
            'status': 'pending_value_review_missing_processed_video', 'original_available': True,
            'no_value_proven': False, 'historical_folder': '/private/source'}])
    completeness = evidence_bundle(source)['completeness']
    assert completeness['selected_clips_expected'] == 2
    pending = completeness['pending_value_review_intervals'][0]
    assert pending['no_value_proven'] is False
    assert pending['status'] == 'pending_value_review_missing_processed_video'
    assert 'historical_folder' not in pending
    compact = _day_completeness(completeness)
    assert compact['pending_value_review_intervals']['record_count'] == 1
    assert len(compact['pending_value_review_intervals']['ranges']) == 1


def test_completed_daily_report_is_recoverable_when_later_inputs_arrive(tmp_path):
    from visioncortex.device_day_reports import preserve_prior_synthesis
    from visioncortex.device_day_contract import read_json
    layout = DeviceDayLayout(tmp_path, 'camera-1', 1789488000000000)
    previous = {'全天综合理解': {'status': 'completed_for_available_inputs', 'findings': [finding(['a'])]},
                '全天记录': {'说明': '先前已保存的版本'}}
    atomic_json(layout.reports/'LaboratoryDailyReport.json', previous)
    preserve_prior_synthesis(layout, previous['全天综合理解'])
    assert not list((layout.reports/'.Evidence/History').glob('*.json'))
    preserve_prior_synthesis(layout, {'status': 'stale_inputs_changed'})
    saved = list((layout.reports/'.Evidence/History').glob('*.json'))
    assert len(saved) == 1 and read_json(saved[0]) == previous
    assert not (layout.reports/'History').exists()


def test_bounded_hierarchy_never_drops_day_tail():
    rows = [{'text': 'a'*40}, {'text': 'b'*40}, {'text': 'c'*40}]
    assert [r for batch in _batches(rows, 80) for r in batch] == rows


def test_all_cameras_keep_their_own_sources_and_cannot_substitute_another_day():
    first = index()
    other = copy.deepcopy(first)
    other['archive'] = '2026-09-16_camera-2'
    bundle = evidence_bundle(first, [other])
    assert {r['device'] for r in bundle['evidence'] if 'device' in r} == {'camera-1', 'camera-2'}
    assert len(bundle['evidence']) == 7
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
    assert next(r for r in wire['evidence'] if r['kind'] == 'speech')['time_alignment'] == 'NOT_PROVEN'
    refs = [r['evidence_id'] for r in wire['evidence']]
    raw = {'findings': [finding(refs, 'contextual_inference')]}
    expanded = expand(raw, receipt)
    assert expanded['findings'][0]['evidence_refs'] == [r['evidence_id'] for r in bundle['evidence']]
    raw['findings'][0]['evidence_refs'] = ['invented']
    with pytest.raises(ValueError, match='invented'):
        expand(raw, receipt)


@pytest.mark.parametrize('history_directory', ['.Evidence/History', 'History'])
def test_separate_model_pass_and_single_day_file_check_input_and_receipts(tmp_path, monkeypatch, history_directory):
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
    assert len(calls[0]['evidence']) == 4
    assert synthesis['input_digest'] == digest(evidence_bundle(source))
    atomic_json(layout.reports/'LaboratoryDailyReport.json', {'全天综合理解': synthesis})
    assert current_synthesis(layout, source) == synthesis
    changed = copy.deepcopy(source)
    changed['context']['protocol'] = {'text': '新增方案，不能当作执行事实'}
    updating = current_synthesis(layout, changed)
    assert updating['status'] == 'updating_for_new_inputs'
    assert updating['findings'] == synthesis['findings']
    assert updating['input_digest'] == synthesis['input_digest']
    assert updating['current_input_digest'] == digest(evidence_bundle(changed))
    assert updating['input_coverage']['clip_ranges'][0]['status'] == 'partial'
    # Recover the last successful conclusion even if an older publisher replaces
    # the canonical view with a pending marker. No new model call is implicit.
    from visioncortex.device_day_reports import preserve_prior_synthesis
    preserve_prior_synthesis(layout, {'status': 'pending'})
    hidden_history = next((layout.reports/'.Evidence/History').glob('Report*.json'))
    original_history_bytes = hidden_history.read_bytes()
    if history_directory == 'History':
        legacy_history = layout.reports/'History'/hidden_history.name
        legacy_history.parent.mkdir()
        hidden_history.rename(legacy_history)
    expected_history = layout.reports/history_directory/hidden_history.name
    atomic_json(layout.reports/'LaboratoryDailyReport.json', {'全天综合理解': {'status': 'pending'}})
    changed['understandings'][0]['status'] = 'completed'
    recovered = current_synthesis(layout, changed)
    assert recovered['findings'] == synthesis['findings']
    assert recovered['current_coverage']['completed_clips'] == 1
    assert recovered['input_coverage']['completed_clips'] == 0
    assert len(calls) == 1
    assert expected_history.read_bytes() == original_history_bytes
    another_camera = copy.deepcopy(changed)
    another_camera['archive'] = '2026-09-16_camera-2'
    assert current_synthesis(layout, another_camera)['status'] == 'stale_inputs_changed'
    last = synthesis['model_receipts'][-1]['result']
    path = layout.root/last['path']
    atomic_json(path, {'model_result': {'status': 'completed', 'findings': []}})
    assert artifact(layout.root, path) != last
    rejected = current_synthesis(layout, changed)
    assert rejected['status'] == 'unverified_saved_synthesis'
    assert rejected['evidence_status'] == 'NOT_PROVEN' and rejected['findings'] == []
    assert any('identity changed' in row['reason'] for row in rejected['recovery_errors'])


def saved_synthesis_fixture(tmp_path, monkeypatch):
    from visioncortex import scene_requests
    source = index()
    source['segments'][0].update(activity='inactive', activity_label='无实验活动',
        start_ms=0, end_ms=2000, key_frames=[], scene_frames=[],
        json_path='ProcessedClips/Clips/00-00-01_00-00-03_NoExperimentActivity_s/NoExperimentActivity.json')
    source['understandings'][0]['windows'][0].update(input=None, model_receipt=None)
    layout = DeviceDayLayout(tmp_path, 'camera-1', 1789488000000000)
    calls = []
    def window(config, analyzer, prompt, metadata, images, request, result, validator):
        calls.append(metadata)
        raw = {'status': 'completed', 'findings': [finding([r['evidence_id'] for r in metadata['evidence']])]}
        atomic_json(request, {'metadata': metadata, 'prompt': prompt})
        atomic_json(result, {'model_result': raw})
        return raw, validator(raw), False
    monkeypatch.setattr(scene_requests, 'window', window)
    return layout, source, synthesize({}, layout, source, object()), calls


@pytest.mark.parametrize('current_kind,newest_kind', [
    ('malformed', 'malformed'), ('bad_receipt', 'malformed'), ('pending', 'bad_receipt'),
])
def test_recovery_checks_every_candidate_before_falling_back_to_older_verified_history(tmp_path, monkeypatch,
        current_kind, newest_kind):
    import os
    layout, source, saved, calls = saved_synthesis_fixture(tmp_path, monkeypatch)
    older = layout.reports/'.Evidence/History/ReportOld.json'
    newest = older.with_name('ReportNew.json')
    current = layout.reports/'LaboratoryDailyReport.json'
    bad = copy.deepcopy(saved)
    bad['model_receipts'][0]['result']['sha256'] = 'f' * 64
    atomic_json(older, {'全天综合理解': saved})
    for path, kind in ((current, current_kind), (newest, newest_kind)):
        if kind == 'malformed':
            path.write_bytes(b'{broken original report bytes')
        else:
            atomic_json(path, {'全天综合理解': bad if kind == 'bad_receipt' else {'status': 'pending'}})
    os.utime(older, (1000, 1000))
    os.utime(newest, (2000, 2000))
    before = {path: path.read_bytes() for path in (older, newest, current)}
    result = current_synthesis(layout, source)
    assert result['status'] == 'completed_for_available_inputs'
    assert result['findings'] == saved['findings'] and result['input_digest'] == saved['input_digest']
    assert result['recovery_errors'] and len(calls) == 1
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('valid_older', [True, False])
def test_render_recovers_from_corrupt_current_and_preserves_exact_unverified_bytes(tmp_path, monkeypatch, valid_older):
    import hashlib
    from visioncortex.device_day_contract import read_json
    from visioncortex.device_day_reports import render_day
    layout, source, saved, calls = saved_synthesis_fixture(tmp_path, monkeypatch)
    older = layout.reports/'.Evidence/History/ReportOld.json'
    bad = copy.deepcopy(saved)
    bad['model_receipts'][0]['result']['sha256'] = 'f' * 64
    atomic_json(older, {'全天综合理解': saved if valid_older else bad})
    original_history = older.read_bytes()
    current = layout.reports/'LaboratoryDailyReport.json'
    original_current = b'{unreadable current with historical evidence'
    current.write_bytes(original_current)
    render_day(layout, source)
    result = read_json(current)['全天综合理解']
    if valid_older:
        assert result['status'] == 'completed_for_available_inputs'
        assert result['findings'] == saved['findings']
    else:
        assert result['status'] == 'unverified_saved_synthesis'
        assert result['evidence_status'] == 'NOT_PROVEN' and result['findings'] == []
    assert result['recovery_errors'] and len(calls) == 1
    preserved = layout.reports/'.Evidence/History'/('UnverifiedReport' + hashlib.sha256(original_current).hexdigest() + '.json')
    assert preserved.read_bytes() == original_current and older.read_bytes() == original_history
    record = read_json(next((layout.reports/'.Evidence/Recovery').glob('Report*.json')))
    assert record['source_path'] == layout.relative(current)
    assert record['source_stat']['st_size'] == len(original_current)
    assert record['preserved_reference']['sha256'] == hashlib.sha256(original_current).hexdigest()
    assert record['evidence_status'] == 'NOT_PROVEN' and record['model_invoked'] is False


def test_output_upgrade_uses_matching_adapter_preserves_success_and_cannot_bypass_unknown(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex import scene_requests
    from visioncortex.device_day_contract import digest, read_json
    from visioncortex.multimodal_usage import UsageLedger, SubmittedUnknown
    cfg = {'mllm': {'provider': 'aliyun', 'model': 'synthetic', 'base_url': 'https://example.test/v1',
        'api_protocol': 'chat_completions', 'max_output_tokens': 16384},
        'storage': {'local_runtime_root': str(tmp_path/'Runtime'), 'local_cache_root': str(tmp_path/'Cache')}}
    layout = DeviceDayLayout(tmp_path/'Archive', 'camera-1', 1789488000000000)
    budgets, paths = [], []
    def analyzer(budget):
        return SimpleNamespace(budget=budget, close=lambda: None)
    def window(config, adapter, prompt, metadata, images, request, result, validator):
        assert adapter.budget == config['mllm']['max_output_tokens']
        budgets.append(adapter.budget)
        policy = scene_requests.request_identity(config, prompt, metadata)['request_policy']
        key = digest([metadata, prompt])
        raw = {'status': 'completed', 'findings': [finding([r['evidence_id'] for r in metadata['evidence']])]}
        atomic_json(request, {'prompt': prompt, 'metadata': metadata, 'input_key': key})
        atomic_json(result, {'model_result': raw, 'input_key': key, 'request_policy': policy})
        paths[:] = [request, result]
        return raw, validator(raw), False
    monkeypatch.setattr(scene_requests, 'window', window)
    options = {'retry_output_tokens': 32768, 'analyzer_factory': lambda c: analyzer(c['mllm']['max_output_tokens'])}
    for _ in range(2):
        synthesize(cfg, layout, index(), analyzer(16384), **options)
    assert budgets == [16384, 16384]
    failed = read_json(paths[1])
    failed['model_result'] = {'status': 'failed', 'error': '模型响应被截断'}
    atomic_json(paths[1], failed)
    request = read_json(paths[0])
    ledger = UsageLedger(cfg)
    reservation = ledger.reserve(scene_requests.request_key(cfg, request['prompt'], request['metadata']))
    with pytest.raises(SubmittedUnknown):
        synthesize(cfg, layout, index(), analyzer(16384), **options)
    assert budgets == [16384, 16384]
    ledger.finish(reservation, {'status': 'failed', 'attempts': 0, 'usage': {'total_tokens': 0}})
    synthesize(cfg, layout, index(), analyzer(16384), **options)
    synthesize(cfg, layout, index(), analyzer(16384), **options)
    assert budgets == [16384, 16384, 32768, 32768]

import copy

import pytest
from test_device_day import FakeModels, capture, device_config, item_and_layout

from visioncortex.device_day_steps import validate
from visioncortex.device_day_narrative import day_record, segment_record
from visioncortex.media.retained_intervals import disposition, publish_disposition
from visioncortex.scene_transport import expand

__all__ = ['device_config']


def model(kind='non_experiment'):
    step = {'start_ms': 0, 'end_ms': 1000, 'action': '操作电脑', 'objects': ['键盘'],
        'before_state': '手在键盘上方', 'after_state': '手指按在键盘上', 'visible_evidence': '按键姿态变化',
        'outcome': 'visible_change', 'description': '实验员操作键盘', 'frame_ids': ['a', 'b'],
        'comment_ids': [], 'basis': 'observed', 'category': 'computer'}
    return {'summary': '可见操作电脑', 'activity_observed': 'inactive', 'experiment_steps': [],
        'activity_kind': kind, 'observability': 'clear', 'work_activities': [step] if kind != 'static' else [],
        'frame_observations': [{'frame_id': 'a', 'text': '键盘上方'}, {'frame_id': 'b', 'text': '按键'}],
        'uncertainties': []}


def fixtures():
    segment = {'segment_id': 's', 'recording_id': 'r', 'activity': 'inactive', 'start_ms': 0,
        'end_ms': 120000, 'start_us': 1000000, 'end_us': 121000000,
        'source_ref': {'path': 'MetaVideo/Original.mp4', 'sha256': 'a' * 64}, 'key_frames': []}
    recording = {'recording_id': 'r', 'camera_key': 'camera-1', 'start_us': 1000000, 'end_us': 121000000}
    meaning = {'status': 'completed', 'windows': [model('static') | {'start_ms': 0, 'end_ms': 120000,
        'input': 'Input.json', 'model_receipt': 'Result.json',
        'source_frames': [{'frame_id': name, 'local_ms': stamp, 'path': name + '.jpg',
                           'sha256': 'b'*64, 'size_bytes': 10} for name, stamp in (('a', 0), ('b', 1000))]}]}
    return segment, recording, meaning


def add_work_evidence(layout, item, key, segment):
    """Synthetic model-port evidence with two byte-bound 0/1s source frames."""
    from visioncortex.device_day_contract import artifact, atomic_json, digest
    folder = layout.understanding/'ClipUnderstanding'
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_id, local_ms in (('a', 0), ('b', 1000)):
        path = folder/f'{key}-{frame_id}.jpg'
        path.write_bytes(('synthetic contract frame ' + frame_id).encode())
        frames.append(artifact(layout.root, path) | {'frame_id': frame_id, 'local_ms': local_ms})
    request, response = folder/f'{key}.Input.json', folder/f'{key}.Result.json'
    metadata = {'start_ms': segment['start_ms'], 'end_ms': segment['end_ms'], 'frames': frames,
                'segment_id': segment['segment_id'], 'source_ref': segment['source_ref'], 'test_only': True}
    prompt = 'Synthetic model-port contract test; no real inference.'
    input_key = digest([metadata, prompt])
    raw = model() | {'status': 'completed'}
    atomic_json(request, {'input_key': input_key, 'prompt': prompt, 'metadata': metadata})
    atomic_json(response, {'input_key': input_key, 'model_result': raw})
    parsed = validate(raw, frames, segment['start_ms'], segment['end_ms'], segment['start_us'],
                      {'origin_us': segment['start_us'] - round(segment['start_ms']*1000)})
    item['windows'][0].update(parsed, start_ms=segment['start_ms'], end_ms=segment['end_ms'],
        source_frames=frames, coverage=None, input=layout.relative(request), model_receipt=layout.relative(response))
    return [*[artifact(layout.root, layout.root/f['path']) for f in frames],
            artifact(layout.root, request), artifact(layout.root, response)]


def test_non_experiment_work_is_structured_even_when_cv_reports_inactive():
    parsed = validate(model(), [{'frame_id': 'a', 'local_ms': 0}, {'frame_id': 'b', 'local_ms': 1000}],
                      0, 1000, 1000000, {'origin_us': 1000000})
    assert parsed['experiment_steps'] == []
    assert parsed['work_activities'][0]['start_us'] == 1000000
    assert parsed['steps'][0]['description'] == '实验员操作键盘'
    wrong = model('static')
    wrong['work_activities'] = model()['work_activities']
    original = copy.deepcopy(wrong)
    conservative = validate(wrong, [{'frame_id': 'a', 'local_ms': 0}, {'frame_id': 'b', 'local_ms': 1000}],
                            0, 1000, 1000000, {'origin_us': 1000000})
    assert conservative['activity_kind'] == 'uncertain'
    assert conservative['work_activities']
    assert conservative['conservative_normalization']['raw_response_unchanged']
    assert wrong == original


def test_explicit_static_placeholder_never_becomes_work_or_video_value(tmp_path):
    from visioncortex.device_day_content import publish_content
    from visioncortex.device_day_contract import DeviceDayLayout, read_json
    segment, recording, meaning = fixtures()
    placeholder = model()['work_activities'][0] | {
        'action': '无可见动作，实验台静置',
        'description': '本数组按契约不承载动作；此处仅为占位说明该段无任何可观察的非实验工作',
        'before_state': '未知', 'after_state': '未知', 'basis': 'uncertain',
        'outcome': 'incomplete_or_uncertain', 'category': 'waiting'}
    meaning['segment_id'] = segment['segment_id']
    meaning['windows'][0].update(activity_kind='uncertain', work_activities=[placeholder], steps=[placeholder],
        conservative_normalization={'model_activity_kind': 'static', 'raw_response_unchanged': True})
    original = copy.deepcopy(meaning)
    assert disposition(segment, meaning, recording)['decision'] == 'pending_value_review'
    view = segment_record(segment, meaning, recording)
    assert view['非实验工作'] == []
    assert view['分窗口观察'][0]['无效结构化内容'][0]['row'] == placeholder
    layout = DeviceDayLayout(tmp_path, 'camera-1', segment['start_us'])
    result = publish_content({'storage': {'archive_root': str(tmp_path)}},
        {'archive': layout.name, 'recordings': [recording], 'segments': [segment], 'understandings': [meaning]})
    public = read_json(layout.root/result['multimodal'][0]['content'])
    window = public['understanding']['windows'][0]
    assert window['work_activities'] == [] and window['steps'] == []
    assert window['invalid_structured_content'][0]['row'] == placeholder
    assert meaning == original  # The actual response/receipt is not rewritten.


def test_real_scene_step_projection_excludes_placeholder_and_preserves_uncertain_waiting_in_report(tmp_path):
    from visioncortex.device_day_contract import DeviceDayLayout, read_json
    from visioncortex.device_day_narrative import effective_meaning
    from visioncortex.device_day_reports import render_day
    payload = model()
    placeholder = payload['work_activities'][0] | {
        'description': '仅为占位，不承载动作', 'before_state': '未知', 'after_state': '未知',
        'basis': 'uncertain', 'outcome': 'incomplete_or_uncertain', 'category': 'waiting'}
    waiting = placeholder | {'description': '实验员在仪器前等待，结果无法确认', 'action': '等待仪器读数变化'}
    payload['work_activities'] = [placeholder, waiting]
    parsed = validate(payload, [{'frame_id': 'a', 'local_ms': 0}, {'frame_id': 'b', 'local_ms': 1000}],
                      0, 1000, 1000000, {'origin_us': 1000000})
    assert set(parsed['steps'][0]) < set(parsed['work_activities'][0])
    segment, recording, _ = fixtures()
    segment.update(scene_frames=[], activity_label='无实验活动',
        json_path='ProcessedClips/Clips/Fixture/NoExperimentActivity.json')
    meaning = {'segment_id': segment['segment_id'], 'status': 'completed',
               'windows': [parsed | {'start_ms': 0, 'end_ms': 1000}]}
    before = copy.deepcopy(meaning)
    effective = effective_meaning(meaning)['windows'][0]
    assert [row['description'] for row in effective['steps']] == [waiting['description']]
    assert [row['description'] for row in effective['work_activities']] == [waiting['description']]
    assert effective['invalid_structured_content'][0]['row']['description'] == placeholder['description']
    layout = DeviceDayLayout(tmp_path, 'camera-1', segment['start_us'])
    render_day(layout, {'archive': layout.name, 'recordings': [recording], 'segments': [segment],
                        'understandings': [meaning]})
    report = read_json(layout.reports/'LaboratoryDailyReport.json')
    visible = report['entries'][0]['understanding']['windows'][0]
    assert [row['description'] for row in visible['steps']] == [waiting['description']]
    html = (layout.reports/'LaboratoryDailyReport.html').read_text()
    assert waiting['description'] in html and placeholder['description'] not in html
    assert meaning == before


@pytest.mark.parametrize('description', [
    '实验员在实验台前等待，等待结果无法确认',
    '实验员在电脑中输入placeholder变量，界面结果无法确认',
    '实验员输入占位符文本，字符内容可见，界面结果无法确认',
])
def test_uncertain_real_waiting_keeps_content_and_waits_for_video_boundaries(description):
    segment, recording, meaning = fixtures()
    real = model()['work_activities'][0] | {
        'action': '实验台前等待反应', 'description': description,
        'before_state': '未知', 'after_state': '未知', 'basis': 'uncertain',
        'outcome': 'incomplete_or_uncertain', 'category': 'waiting'}
    meaning['windows'][0].update(activity_kind='uncertain', work_activities=[real])
    assert disposition(segment, meaning, recording)['decision'] == 'pending_value_review'
    assert segment_record(segment, meaning, recording)['非实验工作'][0]['动作'] == real['action']


def test_compact_transport_restores_non_experiment_frame_references():
    raw = model()
    raw['work_activities'][0]['frame_ids'] = ['f1', 'f2']
    raw['frame_observations'] = []
    result = expand(raw, {'frame_aliases': {'f1': 'a', 'f2': 'b'}})
    assert result['work_activities'][0]['frame_ids'] == ['a', 'b']


def test_static_derivative_omission_explains_why_and_keeps_original_and_audio(tmp_path):
    segment, recording, meaning = fixtures()
    meaning['windows'][0]['coverage'] = {'all_decoded_frames_submitted': True,
        'native_pts_to_pixels_verified': True, 'input_frames': 2}
    meaning['windows'][0]['source_frames'][-1]['local_ms'] = segment['end_ms'] - 1
    original = tmp_path / 'Original.mp4'
    original.write_bytes(b'untouched')
    decision = disposition(segment, meaning, recording)
    assert decision['decision'] == 'omit_static_derivative'
    assert not decision['continuous_inactivity_proven']
    publish_disposition(tmp_path / 'TimeFolder', decision)
    assert (tmp_path / 'TimeFolder/VideoNotRetained.txt').is_file()
    assert original.read_bytes() == b'untouched'
    assert decision['audio_unchanged']


@pytest.mark.parametrize('protection', ['speech', 'ocr', 'waiting', 'obscured', 'pending', 'gap', 'audio_pending'])
def test_uncertainty_or_protected_context_cannot_be_omitted(protection):
    segment, recording, meaning = fixtures()
    context = {}
    if protection == 'speech':
        recording['transcription'] = {'comments': [{'start_us': 2000000, 'end_us': 3000000, 'text': '等待反应'}]}
    elif protection == 'ocr':
        context['events'] = [{'start_us': 2000000, 'end_us': 2000000, 'type': 'ocr_observation'}]
    elif protection == 'waiting':
        segment['activity_semantics'] = 'experiment_waiting'
    elif protection == 'obscured':
        meaning['windows'][0]['observability'] = 'obscured'
    elif protection == 'pending':
        meaning['status'] = 'partial'
    elif protection == 'gap':
        meaning['windows'][0]['start_ms'] = 10000
    else:
        recording['audio'] = {'status': 'provided'}
    assert disposition(segment, meaning, recording, context)['decision'] == 'pending_value_review'


def test_short_static_samples_remain_pending_instead_of_becoming_no_value():
    segment, recording, meaning = fixtures()
    segment['end_ms'] = meaning['windows'][0]['end_ms'] = 8000
    assert disposition(segment, meaning, recording)['decision'] == 'pending_value_review'


@pytest.mark.parametrize('seconds', [8, 120, 900])
def test_sparse_static_windows_cannot_confirm_no_value(seconds):
    segment, recording, meaning = fixtures()
    segment['end_ms'] = meaning['windows'][0]['end_ms'] = seconds * 1000
    meaning['windows'][0]['coverage'] = {'all_decoded_frames_submitted': False,
        'native_pts_to_pixels_verified': True, 'input_frames': 3}
    value = disposition(segment, meaning, recording)
    assert value['video_value'] == 'pending'
    assert not value['no_value_confirmed']


@pytest.mark.parametrize('kind', ['pending', 'static', 'uncertain'])
def test_unselected_intervals_do_not_read_source_or_extract_a_video(tmp_path, kind):
    from visioncortex.application.retained_video import materialize
    from visioncortex.device_day_contract import DeviceDayLayout
    segment, recording, meaning = fixtures()
    if kind == 'pending':
        meaning = None
    else:
        meaning['windows'][0]['activity_kind'] = kind
    def forbidden(*args):
        raise AssertionError('An unselected interval must not extract media')
    layout = DeviceDayLayout(tmp_path/'Output', 'camera-1', 1000000)
    result = materialize(layout, tmp_path/'AbsentOriginal.mp4', segment, meaning, recording, extractor=forbidden)
    assert result['decision'] == 'pending_value_review'
    assert not result.get('video') and not list(layout.root.rglob('*.mp4'))


def test_readable_timeline_keeps_exact_device_time_ocr_values_and_unknown_units():
    segment, recording, meaning = fixtures()
    event = {'start_us': 2000000, 'end_us': 2000000, 'type': 'ocr_observation',
             'capture_device_id': 'camera-1', 'payload': {'normalized_values': [{'value': 5, 'unit': None}],
             'review_status': 'unreviewed'}}
    other = copy.deepcopy(event) | {'capture_device_id': 'camera-2'}
    outside = copy.deepcopy(event) | {'start_us': 500000000, 'end_us': 500000000}
    value = segment_record(segment, meaning, recording, {'events': [event, other, outside]})
    assert len(value['仪器与现场记录']) == 1
    assert value['仪器与现场记录'][0]['识别内容']['normalized_values'][0]['unit'] is None
    assert '尚未证明' in value['仪器与现场记录'][0]['关联依据']
    index = {'archive': '2026-09-16_camera-1', 'segments': [segment], 'recordings': [recording,
        recording | {'recording_id': 'r2', 'start_us': 131000000, 'end_us': 141000000}], 'understandings': [meaning | {'segment_id': 's'}]}
    assert len(day_record(index)['采集间隙']) == 1


def test_materializer_uses_new_generation_and_verifies_original(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.application.retained_video import materialize
    from visioncortex.device_day_contract import artifact, DeviceDayLayout
    segment, recording, meaning = fixtures()
    source = tmp_path / 'Original.mp4'
    source.write_bytes(b'original')
    segment['source_ref'] = artifact(tmp_path, source)
    layout = DeviceDayLayout(tmp_path / 'Output', 'camera-1', 1000000)
    # Non-static work must obtain a playable derivative, never a missing-video note.
    meaning['windows'][0].update(activity_kind='non_experiment', work_activities=model()['work_activities'])
    calls = []
    def extract(src, dst, left, width, encoder):
        calls.append((src, left, width))
        dst.write_bytes(b'derivative')
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda path: SimpleNamespace(duration_ms=1000))
    result = materialize(layout, source, segment, meaning, recording, extractor=extract, evidence_validator=lambda *args: [])
    assert result['video']['path'].endswith('Activity.mp4')
    assert calls == [(source, 0, 1000)]
    assert result['source_end_ms'] == 1000 and result['candidate_source_end_ms'] == 120000
    assert source.read_bytes() == b'original'
    materialize(layout, source, segment, meaning, recording, extractor=extract, evidence_validator=lambda *args: [])
    assert len(calls) == 1
    source.write_bytes(b'replaced')
    with pytest.raises(ValueError):
        materialize(layout, source, segment, meaning, recording, extractor=extract, evidence_validator=lambda *args: [])


def test_crash_after_video_rename_replays_verified_prepared_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.application import retained_video
    from visioncortex.device_day_contract import artifact, DeviceDayLayout, read_json
    segment, recording, meaning = fixtures()
    source = tmp_path/'Original.mp4'
    source.write_bytes(b'original')
    segment['source_ref'] = artifact(tmp_path, source)
    meaning['windows'][0].update(activity_kind='non_experiment', work_activities=model()['work_activities'])
    layout = DeviceDayLayout(tmp_path/'Output', 'camera-1', segment['start_us'])
    calls = []
    def extract(src, target, *args):
        calls.append(src)
        target.write_bytes(b'derivative')
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000))
    original_publish = retained_video.publish_disposition
    def crash(*args, **kwargs):
        raise RuntimeError('simulated process death after rename')
    monkeypatch.setattr(retained_video, 'publish_disposition', crash)
    with pytest.raises(RuntimeError, match='after rename'):
        retained_video.materialize(layout, source, segment, meaning, recording, extractor=extract, evidence_validator=lambda *args: [])
    prepared = next(layout.root.glob('ProcessedClips/.Evidence/Retention/*/*/Prepared.json'))
    saved = read_json(prepared)
    assert (layout.root/saved['video']['path']).read_bytes() == b'derivative'
    assert not prepared.with_name('Retention.json').exists()
    monkeypatch.setattr(retained_video, 'publish_disposition', original_publish)
    restored = retained_video.materialize(layout, source, segment, meaning, recording, extractor=extract, evidence_validator=lambda *args: [])
    assert restored['prepared_receipt'] == artifact(layout.root, prepared)
    assert restored['video'] == saved['video'] and len(calls) == 1
    assert source.read_bytes() == b'original'
    # A different source interval must not adopt that prepared publication.
    other = segment | {'end_ms': segment['end_ms'] - 1000, 'end_us': segment['end_us'] - 1000000}
    other_meaning = copy.deepcopy(meaning)
    other_meaning['windows'][0]['end_ms'] = other['end_ms']
    with pytest.raises(ValueError, match='Unbound clip output'):
        retained_video.materialize(layout, source, other, other_meaning, recording, extractor=extract, evidence_validator=lambda *args: [])


def test_understanding_publishes_non_experiment_video_without_report_stage(device_config, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.device_day_contract import artifact, atomic_json, read_json
    capture(device_config)
    record, layout = item_and_layout(device_config)
    device_config['device_day']['derived_video_retention'] = True
    def extract(source, target, start, duration, encoder):
        target.write_bytes(b'derived')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda path: SimpleNamespace(duration_ms=1000))
    class WorkModels(FakeModels):
        def understand(self, layout, recording, vision, context, key):
            result = super().understand(layout, recording, vision, context, key)
            item = result['understandings'][0]
            evidence = add_work_evidence(layout, item, key, vision['segments'][0])
            path = layout.understanding/'ClipUnderstanding'/f'{key}.json'
            atomic_json(path, item)
            result['artifacts'] = [artifact(layout.root, path), *evidence]
            return result
    runner = DeviceDayRunner(device_config, WorkModels())
    for stage in ('retention', 'vision', 'stt', 'understanding'):
        result = runner.process(record, stage=stage)
        assert result['status'] == 'completed'
    assert result['status'] == 'completed'
    index = read_json(layout.index)
    candidate = index['segments'][0]
    assert not candidate.get('video')
    selected = next(row for row in candidate['retention_intervals'] if row['video_value'] == 'selected')
    assert selected['video']['path'].endswith('NoExperimentActivity.mp4')
    assert selected['derived_video_disposition']['original_video_unchanged']
    assert (selected['start_ms'], selected['end_ms']) == (0, 1000)
    assert not (layout.receipts/record['recording_id']/'report.json').exists()
    assert (layout.root/selected['video']['path']).read_bytes() == b'derived'


def test_missing_camera_role_only_holds_visual_stages(device_config):
    from visioncortex.device_day import DeviceDayRunner
    capture(device_config)
    record, layout = item_and_layout(device_config)
    record['configured_role'] = None
    runner = DeviceDayRunner(device_config, FakeModels())
    assert runner.process(record, stage='retention')['status'] == 'completed'
    assert runner.process(record, stage='stt')['status'] == 'completed'
    for stage in ('vision', 'understanding', 'report'):
        assert runner.process(record, stage=stage)['status'] == 'needs_camera_role'


@pytest.mark.parametrize('error_kind', ['failed', 'submitted_unknown'])
def test_later_window_failure_keeps_completed_observations_and_selected_video(device_config, monkeypatch, error_kind):
    from types import SimpleNamespace
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.device_day_contract import artifact, atomic_json, read_json
    from visioncortex.multimodal_usage import SubmittedUnknown
    capture(device_config)
    record, layout = item_and_layout(device_config)
    device_config['device_day'].update(derived_video_retention=True, understanding_coverage_since_us=1,
                                       readable_content_enabled=True)
    def extract(source, target, *args):
        target.write_bytes(b'derived')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000))
    class PartialModels(FakeModels):
        def understand(self, layout, recording, vision, context, key):
            result = super().understand(layout, recording, vision, context, key)
            value = result['understandings'][0]
            value.update(status='partial')
            evidence = add_work_evidence(layout, value, key, vision['segments'][0])
            path = layout.understanding/'ClipUnderstanding'/f'{key}.json'
            atomic_json(path, value)
            result.update(partial_observations=True, artifacts=[artifact(layout.root, path), *evidence])
            error = SubmittedUnknown('submitted_unknown') if error_kind == 'submitted_unknown' else ValueError('later window failed')
            error.partial_receipt = result
            raise error
    models = PartialModels()
    runner = DeviceDayRunner(device_config, models)
    for stage in ('retention', 'vision'):
        assert runner.process(record, stage=stage)['status'] == 'completed'
    assert runner.process(record, stage='understanding')['status'] == error_kind
    index = read_json(layout.index)
    assert index['understandings'][0]['status'] == 'partial'
    assert index['understandings'][0]['windows'][0]['work_activities']
    candidate = index['segments'][0]
    assert not candidate.get('video')
    selected = next(row for row in candidate['retention_intervals'] if row['video_value'] == 'selected')
    assert (layout.root/selected['video']['path']).read_bytes() == b'derived'
    assert (selected['start_ms'], selected['end_ms']) == (0, 1000)
    stage = read_json(layout.receipts/record['recording_id']/'understanding.json')
    assert stage['status'] == error_kind and stage['partial_observations']
    assert models.semantic_calls == 1
    # No FileIndexPublisher scan or report synthesis is needed to make this
    # actual completed model window directly readable after the later failure.
    entries = read_json(layout.understanding/'Index.json')['entries']
    entry = next(row for row in entries if row['segment_id'] == candidate['segment_id'])
    readable = read_json(layout.root/entry['content'])
    assert readable['status'] == 'partial'
    assert readable['understanding']['windows'][0]['work_activities']
    assert readable['video'] is None  # A candidate is never the selected export.
    child = next(row for row in entries if row['segment_id'] == selected['segment_id'])
    assert read_json(layout.root/child['content'])['video'] == selected['video']
    before = (layout.receipts/record['recording_id']/'understanding.json').read_bytes()
    def aggregate_unavailable(*_args, **_kwargs):
        raise RuntimeError('owned aggregate unavailable')
    monkeypatch.setattr('visioncortex.device_day_reports.render_day', aggregate_unavailable)
    with pytest.raises(RuntimeError, match='aggregate unavailable'):
        runner.refresh_index(layout, recording_id=record['recording_id'])
    assert read_json(layout.root/entry['content'])['understanding'] == readable['understanding']
    assert (layout.receipts/record['recording_id']/'understanding.json').read_bytes() == before
    assert models.semantic_calls == 1


def test_verified_publication_replays_without_model_or_report_and_backs_off_failures(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.application.retained_video import child_segment, publish_selected, published_material
    from visioncortex.device_day_contract import DeviceDayLayout, artifact
    segment, record, meaning = fixtures()
    meaning['status'] = 'partial'
    meaning['segment_id'] = segment['segment_id']
    meaning['windows'][0].update(activity_kind='non_experiment', work_activities=model()['work_activities'])
    layout = DeviceDayLayout(tmp_path, 'camera-1', segment['start_us'])
    source = layout.root/'MetaVideo/Original.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original')
    segment['source_ref'] = artifact(layout.root, source)
    config = {'storage': {'archive_root': str(tmp_path)}, 'device_day': {'derived_video_retention': True}}
    index = {'archive': layout.name, 'recordings': [record], 'segments': [segment], 'understandings': [meaning]}
    calls = []
    def extract(src, dst, *args):
        calls.append(src)
        dst.write_bytes(b'selected')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000))
    # The pure materialization/retry port has explicit synthetic evidence;
    # strict actual request/response validation is covered separately.
    def verified_fixture(*args):
        return []
    assert publish_selected(config, index, evidence_validator=verified_fixture)['materialized'] == 1
    selected = next(row for row in segment['retention_intervals'] if row['video_value'] == 'selected')
    child = child_segment(segment, selected)
    saved = published_material(layout, child)
    assert saved['video'] == selected['video'] and not segment.get('video')
    segment.pop('retention_intervals')
    assert publish_selected(config, index, evidence_validator=verified_fixture)['materialized'] == 0
    assert len(calls) == 1 and source.read_bytes() == b'original'
    # A corrupted same-size video is never accepted as a complete publication.
    (layout.root/saved['video']['path']).write_bytes(b'corrupt!')
    assert published_material(layout, child) is None
    first = publish_selected(config, index, evidence_validator=verified_fixture)
    assert first['pending'] == 1 and first['errors']
    second = publish_selected(config, index, evidence_validator=verified_fixture)
    assert second['pending'] == 1 and not second['errors']
    assert len(calls) == 1


@pytest.mark.parametrize('bad_bytes', [
    b'{"attempts":1,"unknown_previous_submission":\xff',
    b'{"attempts":"unknown previous submission"}',
    ('{"attempts":1,"next_attempt_at":' + str(10**400) + '}').encode(),
])
def test_bad_attempt_is_preserved_and_isolated_while_another_selected_interval_publishes(tmp_path, monkeypatch, bad_bytes):
    from hashlib import sha256
    from types import SimpleNamespace
    from visioncortex.application.retained_video import child_segment, publish_selected, published_material
    from visioncortex.device_day_content_paths import clip_name, private_folder
    from visioncortex.device_day_contract import DeviceDayLayout, artifact, read_json
    segment, record, meaning = fixtures()
    layout = DeviceDayLayout(tmp_path, 'camera-1', segment['start_us'])
    source = layout.root/'MetaVideo/Original.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original')
    segment['source_ref'] = artifact(layout.root, source)
    second = copy.deepcopy(segment) | {'segment_id': 'second'}
    meaning['segment_id'] = segment['segment_id']
    meaning['windows'][0].update(activity_kind='non_experiment', work_activities=model()['work_activities'])
    second_meaning = copy.deepcopy(meaning) | {'segment_id': second['segment_id']}
    index = {'archive': layout.name, 'recordings': [record], 'segments': [segment, second],
             'understandings': [meaning, second_meaning]}
    chosen = next(row for row in disposition(segment, meaning, record)['partition']['intervals']
                  if row['video_value'] == 'selected')
    child = child_segment(segment, chosen)
    attempt = private_folder(layout, 'ProcessedClips', 'Retention', child['segment_id'], 'Attempt.json')
    attempt.parent.mkdir(parents=True)
    attempt.write_bytes(bad_bytes)
    unbound = layout.root/'ProcessedClips'/'Clips'/clip_name(child)/'NoExperimentActivity.mp4'
    unbound.parent.mkdir(parents=True)
    unbound.write_bytes(b'unbound historical video')
    calls = []
    def extract(src, dst, *args):
        calls.append(src)
        dst.write_bytes(b'selected second interval')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000))
    config = {'storage': {'archive_root': str(tmp_path)}, 'device_day': {'derived_video_retention': True}}
    def verified_fixture(*args):
        return []
    result = publish_selected(config, index, evidence_validator=verified_fixture)
    assert result['materialized'] == result['pending'] == 1
    assert len(result['errors']) == 1
    assert result['errors'][0]['reason'] == 'retention_attempt_receipt_invalid'
    assert 'preservation_error_type' not in result['errors'][0]
    failed = next(row for row in segment['retention_intervals'] if row['video_value'] == 'selected')
    assert failed['processing_status'] == 'failed'
    assert not segment.get('video') and published_material(layout, child) is None
    selected_second = next(row for row in second['retention_intervals'] if row['video_value'] == 'selected')
    assert (layout.root/selected_second['video']['path']).read_bytes() == b'selected second interval'
    assert len(calls) == 1
    assert source.read_bytes() == b'original' and unbound.read_bytes() == b'unbound historical video'
    assert attempt.read_bytes() == bad_bytes
    backup = attempt.parent/'InvalidAttempts'/(sha256(bad_bytes).hexdigest() + '.json')
    assert backup.read_bytes() == bad_bytes
    recovered = list(backup.parent.glob('Recovery*.json'))
    assert len(recovered) == 1
    saved = read_json(recovered[0])
    assert saved['source_path'] == layout.relative(attempt)
    assert saved['preserved_reference'] == artifact(layout.root, backup)
    assert saved['evidence_status'] == 'NOT_PROVEN' and saved['model_invoked'] is False
    assert saved['expected_interval']['source_ref'] == segment['source_ref']
    before = backup.stat().st_mtime_ns
    again = publish_selected(config, index, evidence_validator=verified_fixture)
    assert again['materialized'] == 0 and again['pending'] == 1
    assert len(again['errors']) == 1 and 'preservation_error_type' not in again['errors'][0]
    assert len(list(backup.parent.glob('Recovery*.json'))) == 1
    assert backup.stat().st_mtime_ns == before
    assert len(calls) == 1 and attempt.read_bytes() == bad_bytes


def test_experimenter_name_follows_binding_handover_and_never_crosses_devices():
    from visioncortex.device_day_narrative import experimenter
    first = {'type': 'instrument_binding', 'capture_device_id': 'camera-1', 'wearer_id': 'p1',
        'payload': {'role': 'wearer', 'operator_label': '实验员甲', 'valid_from': '2026-09-16T09:00:00+08:00',
                    'valid_to': '2026-09-16T12:00:00+08:00'}}
    second = copy.deepcopy(first) | {'wearer_id': 'p2'}
    second['payload'].update(operator_label='实验员乙', valid_from='2026-09-16T12:00:00+08:00', valid_to=None)
    from datetime import datetime
    def us(stamp):
        return round(datetime.fromisoformat('2026-09-16T'+stamp+'+08:00').timestamp()*1e6)
    context = {'events': [first, second]}
    bound = experimenter(context, 'camera-1', us('11:00:00'), us('12:00:00'))
    assert bound['姓名'] == '实验员甲'
    assert bound['关联方式'] == 'FieldRecognition设备自动绑定'
    assert '物理操作者已确认' not in bound
    assert experimenter(context, 'camera-1', us('12:00:00'), us('13:00:00'))['姓名'] == '实验员乙'
    assert experimenter(context, 'camera-1', us('11:59:00'), us('12:01:00'))['人员ID'] is None
    assert experimenter(context, 'camera-2', us('11:00:00'), us('12:00:00'))['人员ID'] is None
    assert experimenter(context, 'camera-2', us('11:00:00'), us('12:00:00'))['状态'] == '未找到有效绑定'
    assert experimenter(context, 'camera-1', us('12:00:00'), us('12:00:00'))['姓名'] == '实验员乙'


def test_automatic_readable_records_preserve_uncertainty_without_manual_review_tasks():
    segment, recording, meaning = fixtures()
    recording['transcription'] = {'comments': [{'start_us': 2000000, 'end_us': 3000000,
        'text': '换一个', 'transcript_path': 'Comment/Transcript.json'}]}
    context = {'events': [{'type': 'ocr_observation', 'capture_device_id': 'camera-1',
        'start_us': 2000000, 'end_us': 3000000,
        'payload': {'normalized_values': [{'value': None, 'unit': None}],
                    'record_scope': 'test_only', 'review_status': 'unreviewed'}}]}
    original = copy.deepcopy((recording, context))
    clip = segment_record(segment, meaning, recording, context)
    day = day_record({'archive': '2026-09-16_camera-1', 'recordings': [recording],
        'segments': [segment], 'understandings': [meaning | {'segment_id': segment['segment_id']}], 'context': context})
    for speech in (clip['录音记录'][0], day['录音时间线'][0]):
        assert speech['原话'] == '换一个'
        assert speech['识别方式'] == '自动转写'
        assert '人工复核' not in speech
    for payload in (clip['仪器与现场记录'][0]['识别内容'], day['现场事件时间线'][0]['内容']):
        assert payload['record_scope'] == 'test_only'
        assert payload['normalized_values'] == [{'value': None, 'unit': None}]
        assert 'review_status' not in payload
    assert (recording, context) == original
    assert '实验方案' not in clip
    assert '实验方案' not in day

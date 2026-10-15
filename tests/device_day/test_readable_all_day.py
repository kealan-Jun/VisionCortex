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
        'input': 'Input.json', 'model_receipt': 'Result.json'}]}
    return segment, recording, meaning


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


def test_compact_transport_restores_non_experiment_frame_references():
    raw = model()
    raw['work_activities'][0]['frame_ids'] = ['f1', 'f2']
    raw['frame_observations'] = []
    result = expand(raw, {'frame_aliases': {'f1': 'a', 'f2': 'b'}})
    assert result['work_activities'][0]['frame_ids'] == ['a', 'b']


def test_static_derivative_omission_explains_why_and_keeps_original_and_audio(tmp_path):
    segment, recording, meaning = fixtures()
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
    assert disposition(segment, meaning, recording, context)['decision'] == 'retain_derivative'


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
    meaning['windows'][0]['activity_kind'] = 'uncertain'
    calls = []
    def extract(src, dst, left, width, encoder):
        calls.append((src, left, width))
        dst.write_bytes(b'derivative')
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda path: SimpleNamespace(duration_ms=120000))
    result = materialize(layout, source, segment, meaning, recording, extractor=extract)
    assert result['video']['path'].endswith('Activity.mp4')
    assert calls == [(source, 0, 120000)]
    assert source.read_bytes() == b'original'
    materialize(layout, source, segment, meaning, recording, extractor=extract)
    assert len(calls) == 1
    source.write_bytes(b'replaced')
    with pytest.raises(ValueError):
        materialize(layout, source, segment, meaning, recording, extractor=extract)


def test_report_stage_publishes_retained_media_without_modifying_original(device_config, monkeypatch):
    from types import SimpleNamespace
    from visioncortex.device_day import DeviceDayRunner
    from visioncortex.device_day_contract import read_json
    capture(device_config)
    record, layout = item_and_layout(device_config)
    device_config['device_day']['derived_video_retention'] = True
    def extract(source, target, start, duration, encoder):
        target.write_bytes(b'derived')
    monkeypatch.setattr('visioncortex.video_io.extract_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda path: SimpleNamespace(duration_ms=900000))
    runner = DeviceDayRunner(device_config, FakeModels())
    result = runner.process(record)
    assert result['status'] == 'completed'
    index = read_json(layout.index)
    assert index['segments'][0]['video']['path'].endswith('Activity.mp4')
    assert index['segments'][0]['derived_video_disposition']['original_video_unchanged']


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
    assert experimenter(context, 'camera-1', us('11:00:00'), us('12:00:00'))['姓名'] == '实验员甲'
    assert experimenter(context, 'camera-1', us('12:00:00'), us('13:00:00'))['姓名'] == '实验员乙'
    assert experimenter(context, 'camera-1', us('11:59:00'), us('12:01:00'))['人员ID'] is None
    assert experimenter(context, 'camera-2', us('11:00:00'), us('12:00:00'))['人员ID'] is None
    assert experimenter(context, 'camera-1', us('12:00:00'), us('12:00:00'))['姓名'] == '实验员乙'

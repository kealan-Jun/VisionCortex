"""Exact interval materialization, independently of the value inference port."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from visioncortex.application import retained_video
from visioncortex.device_day_content_paths import clip_name, private_folder
from visioncortex.device_day_contract import DeviceDayLayout, artifact, read_json


@pytest.fixture
def exact_case(tmp_path, monkeypatch):
    layout = DeviceDayLayout(tmp_path, 'camera-1', 1000000)
    source = layout.root/'MetaData'/'capture'/'rgb.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'sealed original; no actual video model in this contract test')
    candidate = {'segment_id': 'candidate-15min', 'recording_id': 'recording', 'activity': 'active',
        'start_ms': 0, 'end_ms': 900000, 'start_us': 1000000, 'end_us': 901000000,
        'source_ref': artifact(layout.root, source), 'key_frames': [], 'scene_frames': []}
    record = {'recording_id': 'recording', 'camera_key': 'camera-1', 'start_us': 1000000, 'end_us': 901000000}
    def interval(left, right, value, activity='inactive'):
        return {'start_ms': left, 'end_ms': right, 'start_us': 1000000 + left*1000,
            'end_us': 1000000 + right*1000, 'video_value': value, 'activity': activity,
            'activity_category': 'experiment' if activity == 'active' else 'non_experiment',
            'reason': 'actual bounded value evidence' if value == 'selected' else 'value remains unknown',
            'evidence': [{'input': 'Input.json', 'response': 'Result.json', 'frames': ['left', 'right']}]
                        if value == 'selected' else [], 'context_basis': []}
    intervals = [interval(0, 130000, 'pending'), interval(130000, 155000, 'selected', 'active'),
                 interval(155000, 540000, 'pending'), interval(540000, 560000, 'selected'),
                 interval(560000, 900000, 'pending')]
    grouped = {'candidate_segment_id': candidate['segment_id'], 'source_ref': candidate['source_ref'],
               'source_start_ms': 0, 'source_end_ms': 900000, 'intervals': intervals}
    monkeypatch.setattr(retained_video, 'partition', lambda *args: deepcopy(grouped))
    calls, durations = [], {}
    def extract(src, dst, left, duration, encoder):
        calls.append((src, left, duration, encoder))
        durations[dst] = duration
        dst.write_bytes(f'exact interval: {left} to {left+duration}'.encode())
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video',
                        lambda path: SimpleNamespace(duration_ms=durations[path], fps=30))
    config = {'storage': {'archive_root': str(tmp_path)}, 'device_day': {'derived_video_retention': True}}
    index = {'archive': layout.name, 'segments': [candidate], 'recordings': [record], 'understandings': []}
    return SimpleNamespace(layout=layout, source=source, candidate=candidate, record=record, grouped=grouped, validator=lambda *args: [],
                           intervals=intervals, calls=calls, config=config, index=index)


def test_two_disjoint_values_export_25_and_20_seconds_only(exact_case):
    case = exact_case
    result = retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    assert result['materialized'] == 2 and result['pending'] == 0 and result['errors'] == []
    assert [(row[1], row[2]) for row in case.calls] == [(130000, 25000), (540000, 20000)]
    assert not case.candidate.get('video')
    assert len(case.candidate['retention_intervals']) == 5
    selected = result['selected_intervals']
    assert [(row['start_ms'], row['end_ms'], row['activity']) for row in selected] == [
        (130000, 155000, 'active'), (540000, 560000, 'inactive')]
    for row in selected:
        assert row['processing_status'] == 'completed'
        child = retained_video.child_segment(case.candidate, row)
        assert row['json_path'] == child['json_path']
        assert row['understanding_path'] == 'MultimodalUnderstanding/' + clip_name(child) + '/Understanding.json'
        assert row['video']['path'].rsplit('/', 1)[0] == row['json_path'].rsplit('/', 1)[0]
        saved = read_json(case.layout.root/row['receipt']['path'])
        assert saved['candidate_segment_id'] == case.candidate['segment_id']
        assert saved['candidate_source_start_ms'] == 0 and saved['candidate_source_end_ms'] == 900000
        assert saved['source_ref'] == case.candidate['source_ref']
        assert saved['source_start_ms'] == row['start_ms'] and saved['source_end_ms'] == row['end_ms']
        assert saved['video'] == artifact(case.layout.root, case.layout.root/row['video']['path'])
    assert all(not row.get('video') for row in case.candidate['retention_intervals'] if row['video_value'] != 'selected')
    assert len(list(case.layout.processed.glob('Clips/*/*.mp4'))) == 2
    assert case.source.read_bytes().startswith(b'sealed original')
    assert not case.layout.root.joinpath('MetaVideo').exists()


def test_bounded_turn_resumes_next_child_without_reextracting_published_one(exact_case):
    case = exact_case
    first = retained_video.publish_selected(case.config, case.index, evidence_validator=case.validator)
    assert first['materialized'] == first['pending'] == 1
    first_row = deepcopy(first['selected_intervals'][0])
    assert retained_video.publish_selected(case.config, case.index, evidence_validator=case.validator)['materialized'] == 1
    third = retained_video.publish_selected(case.config, case.index, evidence_validator=case.validator)
    assert third['materialized'] == third['pending'] == 0 and third['errors'] == []
    assert len(case.calls) == 2
    assert third['selected_intervals'][0]['video'] == first_row['video']
    assert third['selected_intervals'][0]['segment_id'] == first_row['segment_id']
    numeric = first_row | {'start_ms': float(first_row['start_ms']), 'end_ms': float(first_row['end_ms'])}
    assert retained_video.child_segment(case.candidate, numeric)['segment_id'] == first_row['segment_id']


@pytest.mark.parametrize('change', [{'start_ms': 0}, {'end_ms': 900000}, {'evidence': []}, {'video_value': 'pending'}])
def test_explicit_selection_cannot_forge_or_expand_a_positive_interval(exact_case, change):
    case = exact_case
    chosen = deepcopy(case.intervals[1])
    forged = chosen | change
    child = retained_video.child_segment(case.candidate, chosen)
    with pytest.raises(ValueError, match='not a current positively selected exact'):
        retained_video.materialize(case.layout, case.source, child, None, case.record,
                                  selection=forged, candidate=case.candidate, evidence_validator=case.validator)
    assert case.calls == [] and not list(case.layout.processed.glob('Clips/*/*.mp4'))


def test_candidate_cannot_impersonate_its_exact_child_or_concatenate_selections(exact_case):
    case = exact_case
    with pytest.raises(ValueError, match='child differs'):
        retained_video.materialize(case.layout, case.source, case.candidate, None, case.record,
                                  selection=case.intervals[1], candidate=case.candidate, evidence_validator=case.validator)
    with pytest.raises(ValueError, match='disjoint intervals cannot be concatenated'):
        retained_video.materialize(case.layout, case.source, case.candidate, None, case.record)
    assert case.calls == []


def test_historical_whole_candidate_is_retained_without_becoming_new_selection(exact_case):
    case = exact_case
    old = case.layout.root/'ProcessedClips'/'HistoricalWhole.mp4'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'historical whole candidate bytes')
    reference = artifact(case.layout.root, old)
    case.candidate['video'] = reference
    result = retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    assert result['materialized'] == 2
    assert case.candidate['video'] == reference and old.read_bytes() == b'historical whole candidate bytes'
    assert case.candidate['historical_scope']['selection_status'] == 'historical_retained_not_revalidated_for_exact_selection'
    assert all(row['video'] != reference for row in result['selected_intervals'])
    assert [(row[1], row[2]) for row in case.calls] == [(130000, 25000), (540000, 20000)]


def test_prepared_exact_child_recovers_after_rename_without_extraction(exact_case, monkeypatch):
    case = exact_case
    selection = case.intervals[1]
    child = retained_video.child_segment(case.candidate, selection)
    real_publish = retained_video.publish_disposition
    def crash(*args, **kwargs):
        raise RuntimeError('crash after exact rename')
    monkeypatch.setattr(retained_video, 'publish_disposition', crash)
    with pytest.raises(RuntimeError, match='exact rename'):
        retained_video.materialize(case.layout, case.source, child, None, case.record,
                                  selection=selection, candidate=case.candidate, evidence_validator=case.validator)
    prepared = next(private_folder(case.layout, 'ProcessedClips', 'Retention', child['segment_id']).glob('*/Prepared.json'))
    saved = read_json(prepared)
    assert saved['source_start_ms'] == 130000 and saved['source_end_ms'] == 155000
    monkeypatch.setattr(retained_video, 'publish_disposition', real_publish)
    result = retained_video.materialize(case.layout, case.source, child, None, case.record,
                                      selection=selection, candidate=case.candidate, evidence_validator=case.validator)
    assert result['video'] == saved['video'] and result['prepared_receipt'] == artifact(case.layout.root, prepared)
    assert len(case.calls) == 1


def test_extraction_uses_source_ms_and_preserves_independent_capture_clock(exact_case):
    case = exact_case
    selection = case.intervals[1]
    selection['end_us'] += 5000000  # Actual capture-clock mapping is not media duration.
    child = retained_video.child_segment(case.candidate, selection)
    result = retained_video.materialize(case.layout, case.source, child, None, case.record,
                                      selection=selection, candidate=case.candidate, evidence_validator=case.validator)
    assert [(row[1], row[2]) for row in case.calls] == [(130000, 25000)]
    assert result['source_mapping']['source_end_us'] - result['source_mapping']['source_start_us'] == 30000000
    assert result['source_mapping']['derived_end_ms'] == 25000


def test_full_candidate_output_is_rejected_for_a_25_second_selection(exact_case, monkeypatch):
    case = exact_case
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=900000, fps=30))
    result = retained_video.publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    assert result['materialized'] == 0 and result['pending'] == 2
    assert len(result['errors']) == 2
    assert all('duration' in row['reason'] for row in result['errors'])
    assert not list(case.layout.processed.glob('Clips/*/*.mp4'))
    assert not case.candidate.get('video')

"""Daily exact-output accounting uses owned temporary files, never real models."""
from copy import deepcopy

import pytest

from test_retained_video_subintervals import exact_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_contract import artifact, atomic_json, read_json
from visioncortex.device_day_narrative import day_record, retention_summary
from visioncortex.device_day_reports import render_day

__all__ = ['exact_case']


def populated(case):
    publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    return [row for row in case.candidate['retention_intervals'] if row['video_value'] == 'selected']


def test_two_exact_outputs_total_45_seconds_and_every_pending_partition_is_reported(exact_case):
    case = exact_case
    selected = populated(case)
    before = deepcopy(case.index)
    value = retention_summary(case.index, case.layout.root)
    assert value['original_recording_count'] == value['original_video_count'] == 1
    assert value['preprocessing_candidate_count'] == 1 and value['partition_count'] == 5
    assert value['selected_interval_count'] == value['available_selected_interval_count'] == 2
    assert value['actual_video_duration_ms'] == value['planned_selected_duration_ms'] == 45000
    assert value['actual_video_duration_seconds'] == 45
    assert value['pending_interval_count'] == 3 and value['pending_duration_ms'] == 855000
    assert value['confirmed_no_value_interval_count'] == 0
    assert value['evidence_status'] == 'PROVEN'
    assert [r['video']['path'] for r in value['partitions'] if r['video_value'] == 'selected'] == [r['video']['path'] for r in selected]
    assert value['source_coverage'][0]['partition_gaps_ms'] == []
    assert value['source_coverage'][0]['model_window_gaps_ms'] == [[0, 900000]]
    assert not value['whole_day_capture_proven'] and case.index == before


def test_no_root_is_planned_only_and_never_claims_files_or_actual_duration(exact_case):
    case = exact_case
    populated(case)
    value = retention_summary(case.index)
    assert value['selected_interval_count'] == 2
    assert value['available_selected_interval_count'] is None
    assert value['actual_video_duration_ms'] is None and value['actual_video_duration_seconds'] is None
    assert value['evidence_status'] == 'NOT_PROVEN' and not value['actual_files_checked']


@pytest.mark.parametrize('failure', ['missing_video', 'corrupt_video', 'missing_receipt',
    'corrupt_receipt', 'wrong_source', 'wrong_interval', 'missing_duration', 'wrong_duration'])
def test_unreadable_or_unbound_selected_file_does_not_contribute_available_count_or_duration(exact_case, failure):
    case = exact_case
    first, second = populated(case)
    if failure == 'missing_video':
        (case.layout.root/first['video']['path']).unlink()
    elif failure == 'corrupt_video':
        target = case.layout.root/first['video']['path']
        original = target.read_bytes()
        target.write_bytes(b'X' + original[1:])  # same length; hash, not size alone, must reject
    elif failure == 'missing_receipt':
        (case.layout.root/first['receipt']['path']).unlink()
    elif failure == 'corrupt_receipt':
        (case.layout.root/first['receipt']['path']).write_bytes(b'not JSON')
    else:
        target = case.layout.root/first['receipt']['path']
        receipt = read_json(target)
        if failure == 'wrong_source':
            receipt['source_ref'] = receipt['source_ref'] | {'sha256': 'b' * 64}
        elif failure == 'wrong_interval':
            receipt['source_start_ms'] -= 1000
        elif failure == 'missing_duration':
            receipt['source_mapping'].pop('derived_end_ms')
            first['source_mapping'] = receipt['source_mapping']
        else:
            receipt['source_mapping']['derived_end_ms'] = 900000
            first['source_mapping'] = receipt['source_mapping']
        atomic_json(target, receipt)
        first['receipt'] = artifact(case.layout.root, target)  # even an updated mutable index cannot hide bad binding
    value = retention_summary(case.index, case.layout.root)
    assert value['selected_interval_count'] == 2
    assert value['available_selected_interval_count'] == value['unverified_selected_interval_count'] == 1
    assert value['actual_video_duration_ms'] == second['end_ms'] - second['start_ms'] == 20000
    assert value['evidence_status'] == 'PARTIAL_EVIDENCE'
    assert next(row for row in value['partitions'] if row['segment_id'] == first['segment_id'])['actual_video_verification']['status'] == 'NOT_PROVEN'


def test_historical_whole_and_child_views_are_never_counted_as_third_selected_clip(exact_case):
    case = exact_case
    selected = populated(case)
    historical = case.layout.processed/'HistoricalWhole.mp4'
    historical.write_bytes(b'historical whole candidate remains readable')
    case.candidate['video'] = artifact(case.layout.root, historical)
    case.index['segments'].extend(deepcopy(selected))
    value = retention_summary(case.index, case.layout.root)
    assert value['preprocessing_candidate_count'] == 1
    assert value['selected_interval_count'] == value['available_selected_interval_count'] == 2
    assert value['actual_video_duration_ms'] == 45000
    case.candidate.pop('retention_intervals')
    value = retention_summary(case.index, case.layout.root)
    assert value['selected_interval_count'] == value['available_selected_interval_count'] == 0
    assert value['actual_video_duration_ms'] == 0 and value['pending_interval_count'] == 1
    assert not value['source_coverage'][0]['partition_available']
    assert historical.read_bytes().startswith(b'historical whole')


def test_no_value_pending_and_unrepresented_source_gaps_stay_distinct(exact_case):
    case = exact_case
    populated(case)
    rows = case.candidate['retention_intervals']
    rows[-1].update(video_value='confirmed_no_value', reason='verified continuous static interval')
    rows.pop(2)  # source 155000--540000 is genuinely unrepresented
    value = retention_summary(case.index, case.layout.root)
    assert value['confirmed_no_value_interval_count'] == 1
    assert value['confirmed_no_value_duration_ms'] == 340000
    assert value['pending_interval_count'] == 1 and value['pending_duration_ms'] == 130000
    assert value['source_coverage'][0]['partition_gaps_ms'] == [[155000, 540000]]
    assert next(row for row in value['partitions'] if row['video_value'] == 'confirmed_no_value')['reason'] == 'verified continuous static interval'


def test_report_json_directly_exposes_verified_counts_without_repeating_parent_process(exact_case):
    case = exact_case
    selected = populated(case)
    case.candidate.update(activity_label='实验活动', json_path='ProcessedClips/Clips/Original/ExperimentActivity.json')
    # The report consumes the original process once, even if a caller also supplies interval views.
    case.index['segments'].extend(deepcopy(selected))
    case.record['transcription'] = {'comments': [{'start_us': 1000000, 'end_us': 2000000,
        'text': '独立语音', 'transcript_path': 'Comment/Transcript.json'}]}
    case.index['context'] = {'events': [{'type': 'ocr_observation', 'start_us': 3000000, 'end_us': 3000000,
        'capture_device_id': 'camera-1', 'payload': {'value': 7, 'unit': None, 'test_only': True}}]}
    render_day(case.layout, case.index)
    report = read_json(case.layout.reports/'LaboratoryDailyReport.json')
    summary = report['视频留存统计']
    assert report['全天记录']['视频留存统计'] == summary
    assert summary['selected_interval_count'] == summary['available_selected_interval_count'] == 2
    assert summary['actual_video_duration_seconds'] == 45
    assert len(report['entries']) == len(report['全天记录']['全天时间线']) == 1
    assert len(report['全天记录']['录音时间线']) == len(report['全天记录']['现场事件时间线']) == 1
    assert report['全天记录']['现场事件时间线'][0]['内容'] == {'value': 7, 'unit': None, 'test_only': True}
    assert report['report_generated_at'] and report['logical_report_count'] == 1
    assert not report['additional_mllm_calls']
    assert day_record(case.index)['视频留存统计']['evidence_status'] == 'NOT_PROVEN'

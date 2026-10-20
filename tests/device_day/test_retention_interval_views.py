"""Exact selection views retain candidate compatibility without broad exports."""
from copy import deepcopy

from test_retained_video_subintervals import exact_case

from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_content import publish_content
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_file_index import build_file_index
from visioncortex.device_day_retention_projection import project_meaning

__all__ = ['exact_case']


def test_two_exact_exports_and_all_gaps_have_matching_readable_sibling_views(exact_case):
    case = exact_case
    publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    case.config['collection_ingest'] = {'source_root': str(case.layout.root)}
    result = publish_content(case.config, case.index)
    case.index['readable_content'] = result
    assert len(result['multimodal']) == 6  # candidate plus five full partition records
    for row in case.candidate['retention_intervals']:
        assert (case.layout.root/row['json_path']).is_file()
        content = read_json(case.layout.root/row['understanding_path'])
        assert (content['source_start_ms'], content['source_end_ms']) == (row['start_ms'], row['end_ms'])
        assert content['candidate_segment_id'] == case.candidate['segment_id']
        assert content['source_video'] == case.candidate['source_ref']
        assert content['selection_interval']['video_value'] == row['video_value']
        decision = content['处理视频留存']
        assert decision['partition']['intervals'][0]['video_value'] == row['video_value']
        assert decision['selected_duration_ms'] == (row['end_ms']-row['start_ms'] if row['video_value'] == 'selected' else 0)
        if row['video_value'] == 'selected':
            assert content['video_status'] == 'available'
            assert content['video'] == row['video']
        else:
            assert not content['video'] and content['video_status'] == 'pending_selection'
    time_index = build_file_index(case.config, case.index, {})
    selected = [r for r in time_index['streams']['video'] if r['kind'] == 'video_segment' and r.get('file')]
    assert [(r['source_start_ms'], r['source_end_ms']) for r in selected] == [(130000, 155000), (540000, 560000)]
    assert all(r['record_scope'] == 'retention_interval' for r in selected)
    assert sum(r['source_end_ms']-r['source_start_ms'] for r in selected) == 45000
    assert len(time_index['recordings']) == 1  # intervals are not extra source recordings
    summary = read_json(case.layout.understanding/'Index.json')['理解进度']
    assert summary['片段总数'] == 1 and summary['留存区间记录数'] == 5
    assert not case.candidate.get('video')


def test_actual_child_sidecars_and_video_invalidate_publication_output_revision(exact_case):
    from visioncortex.device_day_file_index import FileIndexPublisher
    case = exact_case
    publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=case.validator)
    publish_content(case.config, case.index)
    case.config['storage']['local_runtime_root'] = str(case.layout.root/'LocalRuntime')
    publisher = FileIndexPublisher(case.config)
    selected = next(row for row in case.candidate['retention_intervals'] if row.get('video'))
    for relative in (selected['json_path'], selected['understanding_path'], selected['video']['path']):
        before = publisher._output_revision(case.layout.name, case.index)
        target = case.layout.root/relative
        original = target.read_bytes()
        target.unlink()  # owned test outputs only; original source is untouched
        assert publisher._output_revision(case.layout.name, case.index) != before
        target.write_bytes(original)


def test_cropped_view_does_not_promote_cross_boundary_parent_action_to_complete_step():
    full = {'start_ms': 130000, 'end_ms': 155000, 'frame_ids': ['left', 'middle', 'right'],
            'action': '移动物体', 'objects': ['物体'], 'description': '移动物体到另一位置',
            'before_state': '左侧', 'after_state': '右侧', 'visible_evidence': '位置改变',
            'basis': 'observed', 'outcome': 'visible_change', 'comment_ids': []}
    meaning = {'segment_id': 'parent', 'status': 'completed', 'windows': [{'start_ms': 0, 'end_ms': 900000,
        'input': 'Input.json', 'model_receipt': 'Result.json', 'experiment_steps': [full],
        'work_activities': [], 'steps': [full], 'source_frames': [
            {'frame_id': 'left', 'local_ms': 130000}, {'frame_id': 'middle', 'local_ms': 140000},
            {'frame_id': 'right', 'local_ms': 155000}], 'frame_observations': [], 'uncertainties': []}]}
    before = deepcopy(meaning)
    child = {'segment_id': 'child', 'candidate_segment_id': 'parent', 'start_ms': 135000, 'end_ms': 145000}
    projected = project_meaning(meaning, child)
    window = projected['windows'][0]
    assert not window['experiment_steps'] and not window['steps']
    assert window['overlapping_parent_observations'][0]['original_observation'] == full
    assert window['parent_model_window']['start_ms'] == 0 and window['parent_model_window']['end_ms'] == 900000
    assert [f['frame_id'] for f in window['source_frames']] == ['middle']
    assert projected['projection_from_parent'] and not projected['physical_action_confirmed']
    assert meaning == before


def test_missing_model_evidence_keeps_parent_and_child_pending_consistently(exact_case):
    from visioncortex.application.retained_video import EvidenceUnverified
    case = exact_case
    def missing(*_args):
        raise EvidenceUnverified('Actual result evidence is unavailable')
    publish_selected(case.config, case.index, maximum_new_videos=2, evidence_validator=missing)
    result = publish_content(case.config, case.index)
    for entry in result['multimodal']:
        content = read_json(case.layout.root/entry['content'])
        decision = content['处理视频留存']
        assert decision['video_value'] == 'pending'
        assert decision['selected_duration_ms'] == 0
        assert all(row['video_value'] == 'pending' for row in decision['partition']['intervals'])
        assert not content['video']
        if content['selection_interval'] and content['selection_interval']['reason'] == 'evidence_unverified':
            assert content['video_status'] == decision['processing_status'] == 'waiting_for_evidence'
    assert case.calls == []

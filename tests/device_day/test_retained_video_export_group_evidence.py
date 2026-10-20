"""Connected model spans remain one export under the actual artifact owner gate."""
from copy import deepcopy
from types import SimpleNamespace

from test_readable_all_day import model
from test_retained_video_evidence import evidence_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_contract import artifact, atomic_json, digest, read_json
from visioncortex.device_day_steps import validate

__all__ = ['evidence_case']
END = 7766.666666666667
OVERLAP_END = 5266.666666666667


def install_overlapping_responses(case):
    """Owned request/response/frame files; synthetic extractor, no model calls."""
    old_input = case.layout.root/case.meaning['windows'][0]['input']
    old_request = read_json(old_input)
    base = model()['work_activities'][0]
    base.pop('category')
    windows, references, originals = [], [], []
    specs = [(0, 5500, 2600, 5266.67, (2700, 4000, 5000, OVERLAP_END)),
             (5000, 10000, 5000, END, (5000, 6000, END, 9000))]
    for number, (left, right, start, end, times) in enumerate(specs):
        frames = []
        for index, stamp in enumerate(times):
            path = old_input.parent/f'export-group-{number}-{index}.jpg'
            path.write_bytes(f'Synthetic model input frame {number} {stamp}'.encode())
            frames.append(artifact(case.layout.root, path) | {'frame_id': f'w{number}f{index}', 'local_ms': stamp})
        ids = [frame['frame_id'] for frame in frames if start <= frame['local_ms'] <= end]
        step = base | {'start_ms': start, 'end_ms': end, 'frame_ids': ids,
            'action': '旋拧瓶盖', 'objects': ['蓝色瓶盖', '试剂瓶'],
            'description': '双手在试剂瓶与瓶盖处旋拧，开闭结果无法确认',
            'before_state': '双手位于瓶盖与瓶身', 'after_state': '瓶盖仍在瓶口，开闭结果无法确认',
            'visible_evidence': '所引帧中两手握持瓶盖和瓶身、握姿改变',
            'outcome': 'incomplete_or_uncertain'}
        raw = model() | {'status': 'completed', 'activity_observed': 'active', 'activity_kind': 'experiment',
            'experiment_steps': [step], 'work_activities': [],
            'frame_observations': [{'frame_id': frame['frame_id'], 'text': '手位于瓶盖和瓶身附近'} for frame in frames]}
        metadata = old_request['metadata'] | {'frames': frames, 'start_ms': left, 'end_ms': right}
        request = old_request | {'metadata': metadata, 'input_key': digest([metadata, old_request['prompt']])}
        input_path, result_path = (old_input.parent/f'export-group-{number}.{kind}.json' for kind in ('Input', 'Result'))
        atomic_json(input_path, request)
        atomic_json(result_path, {'input_key': request['input_key'], 'model_result': raw})
        parsed = validate(raw, frames, left, right, case.candidate['start_us'], {'origin_us': case.candidate['start_us']})
        windows.append(parsed | {'start_ms': left, 'end_ms': right, 'source_frames': frames, 'coverage': None,
            'input': case.layout.relative(input_path), 'model_receipt': case.layout.relative(result_path)})
        references.extend([artifact(case.layout.root, p) for p in (input_path, result_path)])
        references.extend({key: frame[key] for key in ('path', 'size_bytes', 'sha256')} for frame in frames)
        originals.append((result_path, result_path.read_bytes()))
    case.meaning['windows'] = windows
    case.stage.update(understandings=[deepcopy(case.meaning)], artifacts=references)
    atomic_json(case.receipt, case.stage)
    return originals


def test_actual_overlapping_window_responses_materialize_one_continuous_export(evidence_case, monkeypatch):
    case = evidence_case
    original_responses = install_overlapping_responses(case)
    monkeypatch.setattr('visioncortex.video_io.probe_video',
                        lambda _: SimpleNamespace(duration_ms=END-2700, fps=30))
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 1 and result['pending'] == 0 and result['errors'] == []
    assert case.calls == [(2700, END-2700)]
    group, = result['selected_intervals']
    assert (group['start_ms'], group['end_ms']) == (2700, END)
    assert [(r['start_ms'], r['end_ms']) for r in group['selected_subranges']] == [
        (2700, 5000), (5000, OVERLAP_END), (OVERLAP_END, END)]
    assert len(group['evidence']) == len(group['core_action_ranges']) == 2
    assert all(r['outcome'] == 'incomplete_or_uncertain' for r in group['evidence'])
    assert group['context_subranges'] == [] and group['physical_action_confirmed'] is False
    assert all(path.read_bytes() == raw for path, raw in original_responses)
    assert not case.candidate.get('video')
    from visioncortex.application.retained_video import child_segment
    from visioncortex.device_day_retention_projection import project_meaning
    child = child_segment(case.candidate, group)
    presented = project_meaning(case.meaning, child)
    assert any(window['overlapping_parent_observations'] for window in presented['windows'])
    assert all(child['start_ms'] <= step['start_ms'] < step['end_ms'] <= child['end_ms']
               for window in presented['windows'] for step in window['experiment_steps'])
    assert any(window['experiment_steps'] for window in presented['windows'])


def test_forged_group_extent_cannot_replace_actual_returned_steps(evidence_case):
    case = evidence_case
    original_responses = install_overlapping_responses(case)
    step = case.meaning['windows'][1]['experiment_steps'][0]
    step.update(end_ms=9000, end_us=case.candidate['start_us']+9000000, frame_ids=['w1f0', 'w1f3'])
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1 and case.calls == []
    assert result['errors'][0]['reason'] == 'evidence_unverified'
    assert 'Actual returned model steps' in result['errors'][0]['detail']
    assert all(path.read_bytes() == raw for path, raw in original_responses)
    assert not list(case.layout.processed.glob('Clips/*/*.mp4'))

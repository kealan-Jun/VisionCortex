"""Continuous viewing groups must reproduce actual returned gap/context evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_readable_all_day import model
from test_retained_video_evidence import evidence_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_contract import artifact, atomic_json, digest, read_json
from visioncortex.device_day_steps import validate

__all__ = ['evidence_case']


def install_response(case, *, single=False, omit_gap=False, omit_pre=False, forged=False):
    """All input/response/frame artifacts are real owned temporary files."""
    window = case.meaning['windows'][0]
    input_path, result_path = (case.layout.root/window[k] for k in ('input', 'model_receipt'))
    request = read_json(input_path)
    stamps = (0, 1000, 2000, 4000) if single else (1000, 2000, 4000, 6000, 7000)
    frames = []
    for stamp in stamps:
        path = input_path.parent/f'continuity-{stamp}.jpg'
        path.write_bytes(f'Synthetic source frame {stamp}'.encode())
        frames.append(artifact(case.layout.root, path) | {'frame_id': f'f{stamp}', 'local_ms': stamp})
    first = model()['work_activities'][0] | {
        'start_ms': 1000, 'end_ms': 2000, 'frame_ids': ['f1000', 'f2000'],
        'before_state': '手位于键盘上方', 'after_state': '手离开后键盘仍在桌面',
        'action': '按下键盘按键', 'description': '实验员第一次按键'}
    second = first | {'start_ms': 6000, 'end_ms': 7000, 'frame_ids': ['f6000', 'f7000'],
                      'before_state': '间隙内未见手，键盘可见', 'after_state': '手重新放在键盘上',
                      'description': '实验员第二次按键'}
    raw = model() | {'status': 'completed', 'work_activities': [first] if single else [first, second],
        'frame_observations': [{'frame_id': f'f{stamp}', 'text': '键盘与双手仍在同一操作位置'}
                               for stamp in stamps if not (omit_gap and stamp == 4000)
                               and not (omit_pre and stamp == 0)]}
    # The parser requires a frame observation per input frame. An actual
    # unknown observation carries no basis for a continuity/readability span.
    raw['frame_observations'].extend({'frame_id': f'f{stamp}', 'text': '无法确认场景对象'}
        for stamp in stamps if not any(row['frame_id'] == f'f{stamp}' for row in raw['frame_observations']))
    metadata = request['metadata'] | {'frames': frames}
    request.update(metadata=metadata, input_key=digest([metadata, request['prompt']]))
    atomic_json(input_path, request)
    atomic_json(result_path, {'input_key': request['input_key'], 'model_result': raw})
    parsed = validate(raw, frames, metadata['start_ms'], metadata['end_ms'], case.candidate['start_us'],
                      {'origin_us': case.candidate['start_us']})
    actual_window = {'start_ms': metadata['start_ms'], 'end_ms': metadata['end_ms'], **parsed,
                    'source_frames': frames, 'coverage': None,
                    'input': window['input'], 'model_receipt': window['model_receipt']}
    case.meaning['windows'] = [deepcopy(actual_window)]
    case.stage['understandings'] = [deepcopy(case.meaning)]
    case.stage['artifacts'] = [artifact(case.layout.root, path) for path in (input_path, result_path)] + [
        {key: frame[key] for key in ('path', 'size_bytes', 'sha256')} for frame in frames]
    atomic_json(case.receipt, case.stage)
    if forged:
        stamp = 0 if single else 4000
        for row in case.meaning['windows'][0]['frame_observations']:
            if row['frame_id'] == f'f{stamp}':
                row['text'] = '键盘与双手仍在同一操作位置'
    return result_path, raw


def test_actual_nearby_work_and_gap_export_one_continuous_group(evidence_case, monkeypatch):
    case = evidence_case
    result_path, raw = install_response(case)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=6000, fps=30))
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 1 and result['pending'] == 0 and result['errors'] == []
    assert case.calls == [(1000, 6000)]
    group = result['selected_intervals'][0]
    assert [(row['start_ms'], row['end_ms']) for row in group['core_action_ranges']] == [(1000, 2000), (6000, 7000)]
    assert any(row['start_ms'] == 2000 and row['end_ms'] == 6000 for row in group['context_subranges'])
    assert len([row for row in group['context_basis'] if row['kind'] == 'continuity_context']) == 1
    assert read_json(result_path)['model_result'] == raw
    assert raw['work_activities'][0]['after_state'] != raw['work_activities'][1]['before_state']
    assert not case.candidate.get('video')


def test_actual_short_work_has_only_observed_bounded_readability_context(evidence_case, monkeypatch):
    case = evidence_case
    result_path, raw = install_response(case, single=True)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=4000, fps=30))
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 1 and result['pending'] == 0 and result['errors'] == []
    assert case.calls == [(0, 4000)]
    group = result['selected_intervals'][0]
    assert [(row['start_ms'], row['end_ms']) for row in group['core_action_ranges']] == [(1000, 2000)]
    assert [(row['start_ms'], row['end_ms']) for row in group['context_subranges']] == [(0, 1000), (2000, 4000)]
    assert all(row['kind'] == 'readability_context' for row in group['context_basis'])
    assert read_json(result_path)['model_result'] == raw


@pytest.mark.parametrize('single', [False, True])
def test_forged_canonical_gap_or_context_never_replaces_actual_response(evidence_case, single):
    case = evidence_case
    install_response(case, single=single, omit_gap=not single, omit_pre=single, forged=True)
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1 and case.calls == []
    assert result['errors'][0]['reason'] == 'evidence_unverified'
    assert 'Actual returned model steps' in result['errors'][0]['detail']
    proposed = next(row for row in case.candidate['retention_intervals'] if row.get('proposed_video_value'))
    assert proposed['video_value'] == 'pending' and proposed['processing_status'] == 'waiting_for_evidence'
    assert not list(case.layout.processed.glob('Clips/*/*.mp4'))


def test_corrupt_actual_gap_frame_blocks_group_without_weakening_frame_integrity(evidence_case):
    case = evidence_case
    install_response(case)
    frame = next(row for row in case.meaning['windows'][0]['source_frames'] if row['local_ms'] == 4000)
    path = case.layout.root/frame['path']
    before = path.read_bytes()
    path.write_bytes(bytes([before[0] ^ 1]) + before[1:])
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1 and case.calls == []
    assert result['errors'][0]['reason'] == 'evidence_unverified'
    assert 'owning stage artifact' in result['errors'][0]['detail']

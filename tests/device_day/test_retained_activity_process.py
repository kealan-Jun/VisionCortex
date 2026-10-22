"""Fine-audited process selection must survive sparse/missing semantic steps."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_retained_video_evidence import evidence_case
from visioncortex.application.retained_video import publish_selected
from visioncortex.device_day_audit import persist
from visioncortex.device_day_contract import atomic_json

__all__ = ['evidence_case']


def install_process(case, monkeypatch):
    audit = {'scope': 'single_device_activity_not_cross_view_experiment_confirmation',
             'coarse_only_can_label_activity': False, 'physical_action_confirmed': False,
             'device_activity_decisions': [{'basis': 'fine_anchored_operation_continuity',
                                           'start_ms': 0, 'end_ms': 50100}]}
    summary, reference = persist(case.layout, 'recording', 'vision-key', 0, audit)
    case.candidate.update(activity='active', end_ms=50100, end_us=51100000,
        classification_basis='existing_fine_audit_and_device_boundaries_not_ground_truth',
        activity_audit=summary, activity_audit_ref=reference)
    case.visual['segments'] = [deepcopy(case.candidate)]
    atomic_json(case.receipt.with_name('vision.json'), case.visual)
    case.record['stages']['vision'] = {'key': 'vision-key'}
    monkeypatch.setattr('visioncortex.video_io.probe_video',
                        lambda _: SimpleNamespace(duration_ms=50100, fps=30))


@pytest.mark.parametrize('understanding', ['absent', 'partial', 'failed'])
def test_process_exports_50_seconds_without_waiting_for_semantics_audio_or_person(evidence_case, monkeypatch, understanding):
    case = evidence_case
    install_process(case, monkeypatch)
    case.candidate.pop('activity_audit')  # Bounded read projections omit dense audit summaries.
    case.receipt.unlink()  # A vision-owned export must not read this receipt.
    case.record.update(audio={'status': 'provided'}, transcription={'status': 'pending'})
    case.index['context'] = {'events': []}  # No person binding.
    if understanding == 'absent':
        case.index['understandings'] = []
    else:
        case.meaning['status'] = understanding
    original = deepcopy(case.index['understandings'])
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 1 and result['errors'] == []
    assert case.calls == [(0, 50100)]
    row, = result['selected_intervals']
    assert row['selection_owner'] == 'audited_device_activity'
    assert row['physical_action_confirmed'] is False
    assert row['semantic_steps_define_video_boundaries'] is False
    assert row['source_mapping']['source_end_ms'] == 50100
    assert case.index['understandings'] == original


@pytest.mark.parametrize('damage', ['bounds', 'audit_bytes', 'current_key', 'receipt_missing'])
def test_label_and_reference_cannot_replace_actual_owned_fine_receipt(evidence_case, monkeypatch, damage):
    case = evidence_case
    install_process(case, monkeypatch)
    if damage == 'bounds':
        case.candidate.update(end_ms=60100, end_us=61100000)
    elif damage == 'audit_bytes':
        path = case.layout.backend_root/case.candidate['activity_audit_ref']['path']
        path.write_bytes(path.read_bytes() + b' ')
    elif damage == 'current_key':
        case.record['stages']['vision']['key'] = 'different-key'
    else:
        case.receipt.with_name('vision.json').unlink()
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1
    assert case.calls == [] and result['selected_intervals'] == []
    assert result['errors'][0]['reason'] == 'evidence_unverified'


def test_corrupt_audit_during_extraction_does_not_publish_process(evidence_case, monkeypatch):
    case = evidence_case
    install_process(case, monkeypatch)
    path = case.layout.backend_root/case.candidate['activity_audit_ref']['path']
    def extract(source, target, start, duration, encoder):
        target.write_bytes(b'synthetic process video')
        path.write_bytes(path.read_bytes() + b' ')
    monkeypatch.setattr('visioncortex.application.retained_video.source_stream_clip', extract)
    result = publish_selected(case.config, case.index)
    assert result['materialized'] == 0 and result['pending'] == 1
    assert not list(case.layout.processed.glob('Clips/*/*.mp4'))

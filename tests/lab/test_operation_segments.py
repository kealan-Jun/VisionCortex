"""Operation envelopes, synthetic fixtures; no claim of real field accuracy."""
from datetime import timedelta

import pytest

from visioncortex.lab.domain import fingerprint, instant
from visioncortex.lab.operation_segments import project_anchors
from visioncortex.lab.timeline_policy import TimelinePolicies
from .test_operator_scan import at, scene as operator_scene, scan
from .test_comment_transition import source, reviewed, apply


START, END = instant(at('00:00:00')), instant('2026-10-30T00:00:00+08:00')


def qr(key, clock, target='A', person='P', camera='C', status='declared_equipment'):
    payload = dict(resolution=dict(target_id=target), status=status, run_id=None, batch_id=None)
    return dict(id=key, occurred_at=at(clock), person_id=person, camera_id=camera,
                scan_kind='equipment', payload=payload, payload_sha256=fingerprint(payload), evidence_ids=[])


def comment(key, clock, kind, person='P', camera='C', status='applied_reviewed_interpretation'):
    return dict(id=key, start_at=at(clock), evidence_ids=[], payload=dict(
        comment_transition=kind, person_id=person, camera_id=camera, status=status))


def project(scans, comments=(), start=START, end=END):
    return project_anchors(scans, comments, start, end, 120)


def test_packet_boundary_and_repeated_code_do_not_end_open_operation():
    result = project([qr('q1', '09:00:00'), qr('q2', '09:15:00'), qr('q3', '09:30:00')])
    phase, = result['segments']
    assert phase['end_at'] is None and len(phase['qr_anchors']) == 3
    assert phase['continues_after_day'] and not result['packet_boundary_closes_segment']
    assert phase['missing_anchors'] == ['start_comment', 'end_comment_and_qr']


def test_switch_to_B_and_back_preserves_phase_and_experiment_distinction():
    phases = project([qr('q1', '09:00:00'), qr('q2', '09:15:00', 'B'), qr('q3', '09:30:00')])['segments']
    assert [p['equipment_id'] for p in phases] == ['A', 'B', 'A']
    assert all(not p['experiment_end_implied'] and not p['quantity_ledger_written'] for p in phases)
    assert phases[0]['status'] == 'switched_without_paired_end_comment'


def test_only_paired_reviewed_comment_closes_envelope_not_unrelated_speech():
    scans = [qr('q1', '09:00:00'), qr('q2', '09:20:00')]
    comments = [comment('s', '09:00:10', 'stage_start'),
                comment('wrong-person', '09:19:00', 'stage_complete', person='someone-else'),
                comment('pending', '09:19:20', 'stage_complete', status='needs_attention'),
                comment('end', '09:20:10', 'stage_complete')]
    phase, = project(scans, comments)['segments']
    assert phase['status'] == 'paired_end' and not phase['missing_anchors']
    assert instant(phase['end_at']) == instant(at('09:20:10'))
    assert [a['activity_id'] for a in phase['comment_anchors']] == ['s', 'end']
    assert not phase['continuous_activity_proven']


def test_missing_end_QR_keeps_comment_unpaired_instead_of_inventing_boundary():
    result = project([qr('q1', '09:00:00')], [comment('end', '09:20:00', 'stage_complete')])
    assert result['segments'][0]['end_at'] is None
    assert result['unpaired_anchors'][0]['reason'] == 'unpaired_or_ambiguous_comment'


def test_same_device_new_paired_start_reopens_without_presence_toggle():
    scans = [qr('q1', '09:00:00'), qr('q2', '09:20:00'), qr('q3', '09:30:00')]
    end = comment('end', '09:20:00', 'stage_complete')
    assert len(project(scans, [end])['segments']) == 1
    phases = project(scans, [end, comment('next', '09:30:00', 'stage_start')])['segments']
    assert len(phases) == 2 and phases[1]['status'] == 'open_paired_start'
    assert phases[0]['segment_id'] != phases[1]['segment_id']


def test_end_comment_cannot_close_next_device_before_its_start():
    result = project([qr('a', '09:00:00'), qr('b', '10:00:00', 'B')],
                     [comment('end-A', '09:59:50', 'stage_complete')])
    assert result['segments'][1]['status'] == 'open_missing_start_comment'
    assert result['segments'][1]['end_at'] is None


def test_batch_only_declaration_cannot_mix_new_batch_with_previous_run():
    first, second = qr('a', '09:00:00'), qr('b', '09:30:00')
    first['payload'].update(run_explicit=True, run_id='R1', batch_id='B1')
    second['payload'].update(batch_explicit=True, batch_id='B2')
    phases = project([first, second])['segments']
    assert phases[1]['run_id'] is None and phases[1]['target_batch_id'] == 'B2'


def test_conflicting_run_start_does_not_create_phantom_reopened_phase():
    scans = [qr('q1', '09:00:00'), qr('q2', '09:20:00'), qr('q3', '09:30:00')]
    scans[0]['payload'].update(run_explicit=True, run_id='R1')
    end, start = comment('end', '09:20:00', 'stage_complete'), comment('start', '09:30:00', 'stage_start')
    end['payload']['run_id'] = 'R1'
    start['payload']['run_id'] = 'R2'
    assert len(project(scans, [end, start])['segments']) == 1


def test_two_codes_at_same_time_and_other_camera_never_choose_arbitrary_device():
    result = project([qr('q1', '09:00:00'), qr('q2', '09:10:00', 'A'),
                      qr('q3', '09:10:00', 'B'), qr('q4', '09:00:00', 'B', camera='D')])
    assert len(result['segments']) == 2
    assert result['segments'][0]['status'] == 'ambiguous_equipment_boundary'
    assert len(result['unpaired_anchors']) == 1


def test_open_stage_carries_to_next_day_without_new_activity_or_packet():
    result = project([qr('q1', '23:55:00')], start=END, end=END+timedelta(days=1))
    phase, = result['segments']
    assert phase['continues_from_previous_day'] and phase['end_at'] is None


def test_policy_bounds_reject_boolean_limits_and_unknown_settings():
    TimelinePolicies(dict(schema_version='vc-lab-timeline-policy/2',
        operation_segments=dict(enabled=True, anchor_pairing_seconds=120),
        multimodal_prior_context=dict(enabled=True, max_items=16, max_utf8_bytes=8192, lookback_seconds=1800)))
    for config in (dict(enabled=True, anchor_pairing_seconds=True),
                   dict(enabled=True, anchor_pairing_seconds=0), dict(enabled=True, typo=120)):
        with pytest.raises(ValueError):
            TimelinePolicies(dict(schema_version='vc-lab-timeline-policy/2', operation_segments=config))


@pytest.fixture
def scene(core, settings):
    return operator_scene.__wrapped__(core, settings)


def test_stage_start_uses_existing_reviewed_audio_path_and_daily_revision(scene, settings):
    runtime = scene['runtime']
    runtime.timeline.policies = TimelinePolicies(dict(schema_version='vc-lab-timeline-policy/2',
        operation_segments=dict(enabled=True, anchor_pairing_seconds=120)))
    scan(scene, 'start-qr', run_id=scene['run'])
    src = source(scene, settings, clock='09:00:01')
    applied = apply(scene, 'reviewed-start', reviewed(scene, src, 'stage_start'))
    assert applied['snapshot']['declared_stage_started']
    assert not applied['snapshot']['whole_experiment_completed']
    snapshot = runtime.views.snapshot('2026-10-29')
    phase, = snapshot['document']['operation_segments']['segments']
    assert phase['status'] == 'open_paired_start' and phase['run_id'] == scene['run']
    assert runtime.views.snapshot('2026-10-29')['revision'] == snapshot['revision']
    scan(scene, 'switch-qr', value=scene['devices'][1], when='09:20:00')
    revised = runtime.views.snapshot('2026-10-29')
    assert revised['revision'] == snapshot['revision'] + 1
    assert len(revised['document']['operation_segments']['segments']) == 2
    assert runtime.views.snapshot('2026-10-30')['document']['operation_segments']['segments'][0]['continues_from_previous_day']
    with scene['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',
                            (scene['core'].customer,)).fetchone()['n'] == 0


def test_late_prior_day_end_revises_existing_next_day_carryover_without_overwrite(scene, settings):
    runtime = scene['runtime']
    runtime.timeline.policies = TimelinePolicies(dict(schema_version='vc-lab-timeline-policy/2',
        operation_segments=dict(enabled=True, anchor_pairing_seconds=120)))
    scan(scene, 'late-open', when='23:50:00', run_id=scene['run'])
    before = runtime.views.snapshot('2026-10-30')
    assert len(before['document']['operation_segments']['segments']) == 1
    scan(scene, 'late-end-qr', when='23:55:00')
    src = source(scene, settings, clock='23:55:00')
    apply(scene, 'late-end-comment', reviewed(scene, src))
    refreshed = runtime.views.refresh_dirty()
    assert any(r['day'] == '2026-10-30' for r in refreshed['reports'])
    after = runtime.views.snapshot('2026-10-30')
    assert after['revision'] == before['revision'] + 1
    assert not after['document']['operation_segments']['segments']
    with scene['core'].db.transaction(readonly=True) as conn:
        original = conn.execute('SELECT document FROM hj_daily_report WHERE customer_id=%s AND day=%s AND revision=%s',
                                (scene['core'].customer, before['day'], before['revision'])).fetchone()
        assert original['document'] == before['document']

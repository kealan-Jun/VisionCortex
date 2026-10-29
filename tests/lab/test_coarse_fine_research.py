"""Research-only contract tests; no customer source, GPU, model or business DB."""
from ._support import ROOT
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from visioncortex.lab.coarse_fine_native import NativeBudget, NativeCapacityError, audit_native
from visioncortex.lab.coarse_fine_research import (
    Anchor, Claim, ConsumedFrame, Hint, Packet, Policy, Revision, RevisionLedger,
    Span, StageClocks, StreamMemoryProgress, StreamScheduler, build_plan, nightly_budget,
)

ZERO = datetime(2026, 10, 27, tzinfo=timezone.utc)
SHA = 'a' * 64


def packet(name='p1', *, at=0, duration=900000000, stream='cam1', session='recording1', sha=None):
    return Packet(name, sha or hashlib.sha256(name.encode()).hexdigest(), stream, session,
                  (ZERO + timedelta(seconds=at)).isoformat(), duration, 'synthetic_clock', 0)


def span(p, lo=1000000, hi=2000000):
    return Span(p.media_id, p.source_sha256, lo, hi)


def hint(p, name='hint1', *, lo=1000000, hi=2000000, kind='transfer', **kwargs):
    return Hint(name, span(p, lo, hi), kind, '合成动作候选', 'fixture-small/1', SHA, **kwargs)


def test_variable_packets_keep_context_across_file_boundary_and_one_physical_timeline():
    first, second = packet(), packet('short-final', at=900, duration=120000000)
    plan = build_plan([second, first], [hint(second, lo=0, hi=1000000)],
                      policy=Policy(before_us=2000000, after_us=1000000))
    task = plan.tasks[0]
    assert task.status == 'pending' and not plan.continuity
    assert task.source_spans == (span(first, 898000000, 900000000), span(second, 0, 2000000))
    assert task.request()['input_policy']['spatial'] == 'native_pixels'
    assert not task.run_ids  # Transport continuity cannot invent experiment membership.
    assert sum(c['end_us'] - c['start_us'] for c in plan.coverage) == 1020000000
    assert all(c['semantic_completeness'] == 'NOT_PROVEN' and c['raw_evidence_retained'] for c in plan.coverage)


@pytest.mark.parametrize('delta,kind', [(2, 'gap'), (-2, 'overlap')])
def test_context_gaps_and_overlaps_are_explicit_not_concatenated(delta, kind):
    first, second = packet(), packet('p2', at=900 + delta)
    plan = build_plan([first, second], [hint(second, lo=0, hi=1000000)])
    task = plan.tasks[0]
    assert kind in [c['kind'] for c in plan.continuity]
    assert task.status == 'needs_attention'
    assert task.context_gaps if delta > 0 else 'overlapping_source_context' in task.attention_reasons


def test_nested_packet_overlap_does_not_create_a_fictitious_gap_or_hide_time_mapping_changes():
    outer, inner, tail = packet(duration=20000000), packet('inner', at=5, duration=5000000), packet('tail', at=15, duration=10000000)
    plan = build_plan([outer, inner, tail], [hint(tail)])
    assert [v['kind'] for v in plan.continuity] == ['overlap', 'overlap']
    assert plan.continuity[-1]['previous_media_id'] == outer.media_id
    assert not plan.tasks[0].context_gaps
    changed = replace(tail, captured_at=(ZERO + timedelta(seconds=16)).isoformat())
    assert build_plan([outer, inner, changed], [hint(changed)]).tasks[0].task_id != plan.tasks[0].task_id


@pytest.mark.parametrize('uncertainty,status', [(None, 'needs_attention'), (1000000, 'needs_attention'), (0, 'pending')])
def test_source_clock_uncertainty_remains_explicit_in_tasks_and_each_coverage_interval(uncertainty, status):
    p = replace(packet(), clock_basis='filename_label_not_independently_calibrated', clock_uncertainty_us=uncertainty)
    plan = build_plan([p], [hint(p)])
    assert plan.tasks[0].status == status
    assert ('uncertain_source_clock' in plan.tasks[0].attention_reasons) == (uncertainty != 0)
    assert all(v['clock_basis'] == p.clock_basis and v['clock_uncertainty_us'] == uncertainty for v in plan.coverage)
    if uncertainty is None:
        known = replace(p, clock_uncertainty_us=0)
        assert plan.tasks[0].source_packets_sha256 != build_plan([known], [hint(known)]).tasks[0].source_packets_sha256


def test_unspecified_clock_uncertainty_defaults_to_unknown_even_for_unselected_coverage():
    fields = asdict(packet())
    del fields['clock_uncertainty_us']
    p = Packet(**{**fields, 'clock_basis': 'filename_label_not_independently_calibrated'})
    assert p.clock_uncertainty_us is None
    plan = build_plan([p])
    assert not plan.tasks
    assert plan.coverage[0]['clock_uncertainty_us'] is None
    assert plan.coverage[0]['clock_basis'] == p.clock_basis
    assert plan.coverage[0]['semantic_completeness'] == 'NOT_PROVEN'


def test_repeated_bytes_are_recorded_and_not_silently_removed():
    first, second = packet(), packet('p2', at=900, sha=packet().source_sha256)
    plan = build_plan([first, second])
    assert plan.continuity[0]['kind'] == 'repeated_source_bytes'
    assert len(plan.packets) == 2 and len(plan.coverage) == 2


@pytest.mark.parametrize('kind', ['audio', 'ocr', 'binding', 'uncertain', 'unfinished', 'conflict'])
def test_external_anchors_trigger_fine_review_even_when_coarse_says_waiting(kind):
    p = packet()
    anchor = Anchor('anchor', span(p), kind, 'immutable-' + kind, ('run-a', 'run-b'), ('label-history-1',))
    plan = build_plan([p], [hint(p, kind='waiting')], [anchor])
    assert len(plan.tasks) == 1
    assert plan.tasks[0].external_evidence_ids == ('immutable-' + kind,)
    assert plan.tasks[0].run_ids == ('run-a', 'run-b')
    assert plan.tasks[0].binding_revision_ids == ('label-history-1',)
    assert plan.hints[0].kind == 'waiting'  # No rewrite of coarse observation.


def test_uncertain_unfinished_conflicting_hints_trigger_and_are_never_confirmed():
    p = packet()
    plan = build_plan([p], [hint(p, kind='unknown', uncertain=True, unfinished=True, conflicting=True)])
    assert set(plan.tasks[0].trigger_kinds) == {'uncertain', 'unfinished', 'conflict'}
    assert plan.tasks[0].request()['output_role'] == 'candidate_revision_only'


def test_blind_audits_are_deterministic_and_not_dependent_on_importance_output():
    p = packet(duration=120000000)
    policy = Policy(before_us=0, after_us=0, audit_period_us=30000000)
    a, b = build_plan([p], policy=policy), build_plan([p], [hint(p, kind='rest')], policy=policy)
    assert len(a.tasks) == 4 and a.tasks == b.tasks
    assert all(t.trigger_kinds == ('blind_audit',) for t in a.tasks)
    assert any(c['fine_status'] == 'not_selected_for_fine_still_unanalysed' for c in a.coverage)


def test_capacity_failure_retains_full_evidence_and_does_not_schedule_auto_retry():
    p = packet()
    plan = build_plan([p], [hint(p, lo=0, hi=60000000)])
    task = plan.tasks[0]
    assert task.status == 'needs_attention' and task.source_spans[0].end_us == 63000000
    scheduler = StreamScheduler(plan)
    assert scheduler.claim('worker') is None
    assert scheduler.backlog()['needs_attention'] == 1


def test_stream_scheduler_supports_one_then_four_streams_without_sharing_context():
    packets = [packet('p' + str(i), stream='cam' + str(i)) for i in range(4)]
    hints = [hint(p, 'h' + str(i)) for i, p in enumerate(packets)]
    plan = build_plan(packets, hints)
    serial, parallel = StreamScheduler(plan), StreamScheduler(plan, max_concurrent=4)
    assert serial.claim('first', now=1) is not None and serial.claim('second', now=1) is None
    leases = [parallel.claim('worker' + str(i), now=1) for i in range(4)]
    assert len({r['stream_state_key'] for r in leases}) == 4
    assert parallel.claim('fifth', now=1) is None
    for i, lease in enumerate(leases):
        parallel.finish(lease['token'], 'worker' + str(i), outcome='candidate_returned', result_sha256=SHA, now=2)
    assert parallel.backlog() == dict(pending=0, running=0, candidate_returned=4, needs_attention=0)


def test_same_stream_and_session_are_never_concurrently_mutated_and_expiry_needs_attention():
    p = packet()
    plan = build_plan([p], [hint(p), hint(p, 'second', lo=5000000, hi=6000000)])
    scheduler = StreamScheduler(plan, max_concurrent=4)
    first = scheduler.claim('one', now=1, lease_seconds=2)
    assert scheduler.claim('two', now=1) is None
    second = scheduler.claim('two', now=4)
    assert second['task'].task_id != first['task'].task_id
    assert scheduler.backlog()['needs_attention'] == 1
    with pytest.raises(ValueError, match='Stale'):
        scheduler.finish(first['token'], 'one', outcome='candidate_returned', result_sha256=SHA, now=4)


def test_actual_memory_counts_advance_only_on_success_and_keep_cross_file_state():
    p, q = packet(), packet('p2', at=900)
    memory = StreamMemoryProgress(*p.key, 'fixture-small/1', SHA)
    f = ConsumedFrame(span(p), 30, 1000000)
    one = memory.commit(p, [f, f], actual_update_count_after=1, state_sha256=SHA, successful=True)
    assert one['cumulative_input_frames'] == 2 and one['cumulative_unique_source_frames'] == 1
    two = memory.commit(q, [ConsumedFrame(span(q), 30, 1000000)], actual_update_count_after=2,
                        state_sha256='b' * 64, successful=True)
    assert two['state_changed'] and two['latest_processed_epoch_us'] == q.start_us + 1000000
    with pytest.raises(ValueError, match='successful'):
        memory.commit(q, [f], actual_update_count_after=3, state_sha256=SHA, successful=False)
    with pytest.raises(ValueError, match='stream'):
        memory.commit(replace(q, stream_id='other-camera'), [f], actual_update_count_after=3, state_sha256=SHA, successful=True)
    with pytest.raises(ValueError, match='rewind'):
        memory.commit(p, [f], actual_update_count_after=3, state_sha256=SHA, successful=True)
    assert len(memory.updates) == 2


def test_memory_rejects_reverse_input_and_changed_capture_clock_without_advancing_counters():
    p = packet()
    memory = StreamMemoryProgress(*p.key, 'fixture-small/1', SHA)
    early, later = ConsumedFrame(span(p, 0, 5000000), 30, 1000000), ConsumedFrame(span(p, 0, 5000000), 60, 2000000)
    with pytest.raises(ValueError, match='chronological'):
        memory.commit(p, [later, early], actual_update_count_after=1, state_sha256=SHA, successful=True)
    assert not memory.updates and memory.total_input_frames == 0
    memory.commit(p, [early], actual_update_count_after=1, state_sha256=SHA, successful=True)
    with pytest.raises(ValueError, match='time mapping'):
        memory.commit(replace(p, captured_at=(ZERO + timedelta(seconds=10)).isoformat()), [later],
                      actual_update_count_after=2, state_sha256=SHA, successful=True)
    # Chronological rolling-context re-encoding remains explicitly supported.
    result = memory.commit(p, [early, later], actual_update_count_after=2, state_sha256=SHA, successful=True)
    assert result['cumulative_input_frames'] == 3 and result['cumulative_unique_source_frames'] == 2


@pytest.fixture
def native_fixture(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / 'native.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 6, (32, 24))
    if not writer.isOpened():
        pytest.fail('Actual FFV1 codec is required for native evidence verification')
    originals = []
    for i in range(12):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, :] = (i * 10, i, 255 - i)
        frame[i % 24, :, :] = (20, 80, 140)
        originals.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        writer.write(frame)
    writer.release()
    p = packet(duration=2000000, sha=hashlib.sha256(path.read_bytes()).hexdigest())
    task = build_plan([p], [hint(p)], policy=Policy(before_us=0, after_us=0)).tasks[0]
    guards = []

    def guard(given, packet):
        assert given.resolve().is_relative_to(tmp_path) and given.is_file() and packet == p
        guards.append(str(given))

    return p, task, path, originals, guard, guards


def test_actual_codec_delivers_every_selected_native_frame_and_exact_time_provenance(native_fixture):
    p, task, path, originals, guard, guards = native_fixture
    received = []

    def consume(trace, rgb):
        assert not rgb.flags.writeable
        assert (rgb == originals[trace.source_frame_index]).all()
        received.append(trace)

    clocks = StageClocks()
    receipt = audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard,
                           consume_frame=consume, clocks=clocks)
    assert receipt.delivered_frame_count == 6 and receipt.decoded_frame_count == 12
    assert [f.source_frame_index for f in received] == list(range(6, 12))
    assert all((f.width, f.height) == (32, 24) and f.source_sha256 == p.source_sha256 for f in received)
    assert received[0].source_pts_seconds == '1'
    assert len(guards) == 2 and receipt.model_consumption == 'NOT_ATTESTED_BY_LOCAL_DECODE'
    assert {'read', 'decode', 'encode'} <= {r['stage'] for r in clocks.records}


def test_actual_codec_cross_packet_context_keeps_order_and_both_original_source_coordinates(native_fixture):
    import cv2

    p, _, path, originals, _, _ = native_fixture
    other = path.with_name('short-final.avi')
    writer = cv2.VideoWriter(str(other), cv2.VideoWriter_fourcc(*'FFV1'), 6, (32, 24))
    assert writer.isOpened()
    for frame in reversed(originals[:6]):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    q = packet('p2', at=2, duration=1000000, sha=hashlib.sha256(other.read_bytes()).hexdigest())
    plan = build_plan([p, q], [hint(q, lo=0, hi=500000)], policy=Policy(before_us=500000, after_us=500000))
    received = []

    def guard(given, source):
        assert given.resolve().is_relative_to(path.parent) and source in (p, q)

    receipt = audit_native(plan.tasks[0], {p.media_id: p, q.media_id: q}, {p.media_id: path, q.media_id: other},
                           source_guard=guard, consume_frame=lambda f, rgb: received.append(f))
    assert receipt.delivered_frame_count == 9
    assert [(f.media_id, f.source_frame_index) for f in received] == [(p.media_id, i) for i in range(9, 12)] + [(q.media_id, i) for i in range(6)]
    assert received[2].source_pts_seconds == '11/6' and received[3].source_pts_seconds == '0'


@pytest.mark.parametrize('budget', [NativeBudget(max_frames=5), NativeBudget(max_pixels_per_frame=700), NativeBudget(max_total_pixels=4000)])
def test_native_capacity_rejects_before_any_frame_delivery_without_resizing_or_sampling(native_fixture, budget):
    p, task, path, _, guard, _ = native_fixture
    received = []
    with pytest.raises(NativeCapacityError):
        audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard,
                     consume_frame=lambda *args: received.append(args), budget=budget)
    assert not received


def test_no_source_access_before_explicit_guard_and_changed_sha_rejected(native_fixture):
    p, task, path, _, guard, _ = native_fixture
    with pytest.raises(ValueError, match='guard'):
        audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=None, consume_frame=lambda *args: None)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA'):
        audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard, consume_frame=lambda *args: None)


@pytest.mark.parametrize('change', [dict(captured_at=(ZERO + timedelta(seconds=10)).isoformat()),
                                   dict(stream_id='other-camera'), dict(clock_uncertainty_us=1000000)])
def test_native_rejects_changed_time_identity_before_source_access(native_fixture, change):
    p, task, path, _, _, _ = native_fixture
    accessed = []
    with pytest.raises(ValueError, match='time mapping'):
        audit_native(task, {p.media_id: replace(p, **change)}, {p.media_id: path},
                     source_guard=lambda *args: accessed.append(args), consume_frame=lambda *args: accessed.append(args))
    assert not accessed


def test_candidate_revisions_append_original_evidence_without_overwriting_observations(native_fixture):
    p, task, path, _, guard, _ = native_fixture
    receipt = audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard, consume_frame=lambda *args: None)
    ledger = RevisionLedger()
    original = Claim('c1', 'visible_action', '原片中可见的合成动作候选', (span(p),))
    first = Revision(task.task_id, 'r1', None, 'fixture-fine/1', SHA, (original,))
    ledger.append(task, first, native_receipt=receipt)
    corrected = Claim('c2', 'uncertainty', '纠正：证据不足以确定方向', (span(p),), ('c1',))
    second = Revision(task.task_id, 'r2', 'r1', 'fixture-fine/2', SHA, (corrected,))
    ledger.append(task, second, native_receipt=receipt)
    assert ledger.items == (first, second)
    assert ledger.items[0].claims[0] == original and ledger.items[1].status == 'candidate_not_confirmed'
    assert task.request()['coarse_hint_role'] == 'unverified_leads_not_observations_or_answers'


@pytest.mark.parametrize('kind,evidence', [('physical_completion', True), ('confirmed_reading', True), ('visible_action', False)])
def test_unsupported_or_business_fact_claims_rejected(native_fixture, kind, evidence):
    p, task, path, _, guard, _ = native_fixture
    receipt = audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard, consume_frame=lambda *args: None)
    claim = Claim('c', kind, '不得凭空确认为客户事实', (span(p),) if evidence else ())
    with pytest.raises(ValueError):
        RevisionLedger().append(task, Revision(task.task_id, 'r1', None, 'fixture-fine/1', SHA, (claim,)), native_receipt=receipt)


def test_fine_text_alone_and_foreign_evidence_are_rejected(native_fixture):
    p, task, path, _, guard, _ = native_fixture
    receipt = audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard, consume_frame=lambda *args: None)
    claim = Claim('c', 'visible_action', '引用错误来源', (replace(span(p), source_sha256='b' * 64),))
    revision = Revision(task.task_id, 'r1', None, 'fixture-fine/1', SHA, (claim,))
    with pytest.raises(ValueError, match='receipt'):
        RevisionLedger().append(task, revision, native_receipt=None)
    with pytest.raises(ValueError, match='outside'):
        RevisionLedger().append(task, revision, native_receipt=receipt)
    with pytest.raises(ValueError, match='modified'):
        replace(receipt, input_sha256='f' * 64).validate(task)


def test_decode_receipt_cannot_be_relabelled_as_model_consumption_attestation(native_fixture):
    p, task, path, _, guard, _ = native_fixture
    receipt = audit_native(task, {p.media_id: p}, {p.media_id: path}, source_guard=guard, consume_frame=lambda *args: None)
    with pytest.raises(ValueError, match='attestation'):
        replace(receipt, model_consumption='ALL_FRAMES_MODEL_CONSUMED').validate(task)
    with pytest.raises(ValueError, match='decoded count'):
        replace(receipt, decoded_frame_count=0).validate(task)


def test_unknown_nightly_stages_remain_null_and_small_trial_cannot_pass():
    clocks, calls = StageClocks(), []
    with clocks.stage('fine_model', sync=lambda: calls.append('sync')):
        pass
    with clocks.stage('coarse_memory', mode='reused_memory'):
        pass
    resource = clocks.resource(stream_id='fixture-camera', gpu_used_mib=12000, gpu_total_mib=49140, cache_bytes=12)
    result = nightly_budget(clocks, dict(pending=3, running=1, needs_attention=0), whole_wall_seconds=2, source_seconds=10)
    assert calls == ['sync', 'sync']
    assert result['stages_seconds']['cold']['read'] is None
    assert result['stages_seconds']['cold']['fine_model'] >= 0
    assert result['stages_seconds']['reused_memory']['coarse_memory'] >= 0
    assert result['acceptance'] == 'NOT_DEMONSTRATED' and result['next_shift_eta_seconds'] is None
    assert all(w['fits'] is None for w in result['budgets'])
    assert result['resource_curve'][0] == resource and resource['process_rss_bytes'] is None


def test_nightly_success_requires_actual_all_stage_clocks_quality_input_and_publication():
    from visioncortex.lab.coarse_fine_research import STAGES

    clocks = StageClocks()
    for stage in STAGES:
        with clocks.stage(stage):
            pass
    result = nightly_budget(clocks, dict(pending=0, running=0, needs_attention=0), whole_wall_seconds=9 * 3600,
                            source_seconds=4 * 3600, complete_input=True, accepted_quality=True, publication_verified=True)
    assert result['acceptance'] == 'DEMONSTRATED'
    assert [v['fits'] for v in result['budgets']] == [False, True, True]


@pytest.mark.parametrize('missing', ['failed_stage', 'missing_backlog', 'unknown_backlog', 'unknown_status', 'source_duration'])
def test_failed_or_unknown_actual_work_cannot_become_a_nightly_pass(missing):
    from visioncortex.lab.coarse_fine_research import STAGES

    clocks = StageClocks()
    for stage in STAGES:
        try:
            with clocks.stage(stage):
                if missing == 'failed_stage' and stage == 'fine_model':
                    raise RuntimeError('real attempt failed before a result')
        except RuntimeError:
            pass
    backlog = dict(pending=0, running=0, needs_attention=0)
    if missing == 'missing_backlog':
        del backlog['pending']
    if missing == 'unknown_backlog':
        backlog['pending'] = None
    if missing == 'unknown_status':
        backlog['failed'] = 1
    result = nightly_budget(clocks, backlog, whole_wall_seconds=5,
                            source_seconds=None if missing == 'source_duration' else 10,
                            complete_input=True, accepted_quality=True, publication_verified=True)
    assert result['acceptance'] == 'NOT_DEMONSTRATED' and all(v['fits'] is None for v in result['budgets'])
    # The failed attempt's actual resource time is retained, never replaced by zero/null.
    assert result['stages_seconds']['cold']['fine_model'] is not None


@pytest.mark.parametrize('update', [dict(duration_us=0), dict(clock_uncertainty_us=-1), dict(captured_at='2026-10-27T00:00:00'), dict(source_sha256='unknown')])
def test_unknown_source_identity_clock_and_invalid_interval_cannot_be_silently_used(update):
    with pytest.raises(ValueError):
        Packet(**{**asdict(packet()), **update})


def test_module_does_not_enable_default_runtime_or_import_model_frameworks():
    source = (ROOT / 'src/visioncortex/lab/timeline_runtime.py').read_text()
    assert 'coarse_fine' not in source

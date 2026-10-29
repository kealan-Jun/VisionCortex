"""Explicit synthetic transport/model fixtures; no real inference or business writes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
import qrcode

from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Conflict, Core, Missing
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import Worker
from visioncortex.lab.timeline import Timeline
from visioncortex.lab.timeline_policy import TimelinePolicies
from visioncortex.lab.timeline_runtime import TimelineRuntime
from visioncortex.lab.timeline_tasks import TimelineTasks
from visioncortex.lab.timeline_views import TimelineViews

DAY = '2026-10-27'


def at(clock):
    return DAY + 'T' + clock + '+08:00'


@pytest.fixture
def hj(core, settings):
    settings = replace(settings, processing_profile='timeline-v1')
    timeline = Timeline(core)
    samples = [core.register('sample-' + str(i), 'material_unit', 'fixture-sample-' + str(i))['id'] for i in range(3)]
    instruments = [core.register('instrument-' + str(i), 'instrument', 'fixture-instrument-' + str(i))['id'] for i in range(2)]
    person = core.register('person', 'person', 'fixture-person')['id']
    tag = core.register('tag', 'tag', 'fixture-sample-tag', {'mode': 'reusable'})['id']
    device_tag = core.register('device-tag', 'tag', 'fixture-equipment-qr', {'mode': 'permanent'})['id']
    core.assign_tag('device-identity', device_tag, instruments[0], at('08:00:00'), actor='fixture', reason='fixture-only identity')
    return dict(core=core, settings=settings, timeline=timeline, samples=samples, instruments=instruments, person=person, tag=tag, device_tag=device_tag)


def run(hj, code='A', sample=0, timeline=None):
    return (timeline or hj['timeline']).create_run('run-' + code, 'fixture-run-' + code, 'fixture-test', [hj['samples'][sample]], person_id=hj['person'])['id']


def media(hj, name, clock='09:00:00', duration=900, kind='video', session='fixture-session', stream='fixture-camera', captured=True, **extra):
    # Intentionally not a real video: tests receipt/time logic, not codec/model quality.
    path = hj['settings'].incoming / (name + {'video': '.mp4', 'audio': '.wav', 'photo': '.png'}[kind])
    path.write_bytes(('synthetic fixture payload ' + name).encode())
    receipt = dict(capture_id=name, source_instance=stream, source_revision='fixture/1', kind=kind,
                   session_id=session, relative_path=path.name, captured_at=at(clock) if captured else None,
                   time_basis={'basis': 'synthetic_fixture_clock', 'uncertainty_us': 0},
                   start_seconds='0', end_seconds=str(duration), source_kind='fixture',
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)
    return Ingest(hj['core'], hj['settings']).accept(receipt)


def span(m, start=0, end=1):
    return dict(media_id=m['media_id'], start_seconds=str(start), end_seconds=str(end))


def activity(hj, key, m, *, start=0, end=1, kind='start', source_type='video', **kwargs):
    return hj['timeline'].add_activity(key, kind, source_type, 'explicit synthetic fixture observation', [span(m, start, end)], model_revision='fixture-model-not-real-inference/1', **kwargs)


def bind(hj, rid, *, sample=0, instrument=0, clock='09:00:00', key=None, timeline=None, **kwargs):
    return (timeline or hj['timeline']).bind(key or 'bind-' + str(uuid4()), rid, hj['instruments'][instrument], hj['samples'][sample], at(clock), actor='fixture', **kwargs)


def record(hj, rid):
    with hj['core'].db.transaction(readonly=True) as conn:
        return hj['timeline']._run(conn, rid)


def test_variable_files_share_experiment_without_ending_at_file_boundary(hj):
    rid = run(hj)
    a, b = media(hj, 'first'), media(hj, 'short', '09:15:00', 180)
    value = hj['timeline'].add_activity('cross-boundary', 'transfer', 'video', 'fixture crossing boundary', [span(a, 895, 900), span(b, 0, 5)], run_ids=[rid], model_revision='fixture/1')
    assert value['start_at'].startswith(DAY + 'T01:14:55')
    clips = TimelineViews(hj['timeline']).clip_index(rid)['clips']
    assert len(clips) == 2 and {x['media_id'] for x in clips} == {a['media_id'], b['media_id']}
    assert record(hj, rid)['record_status'] == 'open'
    with hj['core'].db.transaction(readonly=True) as conn:
        assert conn.execute("SELECT count(*) AS n FROM processing_job WHERE customer_id=%s AND worker_kind='vision'", (hj['core'].customer,)).fetchone()['n'] == 0
        assert conn.execute('SELECT count(*) AS n FROM hj_timeline_task WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 2


def test_reused_label_uses_capture_time_and_handles_late_history(hj):
    t, tag = hj['timeline'], hj['tag']
    t.label_event('assign-B-first-arrival', tag, at('09:30:00'), target_id=hj['samples'][1], actor='fixture', reason='recycle label')
    t.label_event('assign-A-late-arrival', tag, at('09:00:00'), target_id=hj['samples'][0], actor='fixture', reason='historical assignment')
    a, b = run(hj), run(hj, 'B', 1)
    bind(hj, a)
    bind(hj, b, sample=1, clock='09:30:00')
    late = media(hj, 'arrived-after-label-reuse', '09:05:00', 180)
    result = activity(hj, 'late-activity', late, tag_id=tag, instrument_id=hj['instruments'][0])
    assert result['sample_id'] == hj['samples'][0] and result['run_ids'] == [a]
    assert t.resolve_label(tag, at('09:31:00'))['target_id'] == hj['samples'][1]


def test_label_release_and_conflicting_event_are_explicit(hj):
    t, tag = hj['timeline'], hj['tag']
    t.label_event('A', tag, at('09:00:00'), target_id=hj['samples'][0], actor='fixture', reason='bind')
    t.label_event('release', tag, at('09:10:00'), actor='fixture', reason='label recovered')
    assert t.resolve_label(tag, at('09:12:00'))['status'] == 'unassigned'
    assert t.resolve_label(tag, at('09:09:59'), at('09:10:01'))['status'] == 'crosses_rebinding'
    with pytest.raises(Conflict):
        t.label_event('conflict', tag, at('09:00:00'), target_id=hj['samples'][1], actor='fixture', reason='conflicting fixture')
    with pytest.raises(Conflict):
        t.label_event('wrong-equipment-reuse', hj['device_tag'], at('10:00:00'), target_id=hj['samples'][1], actor='fixture', reason='must reject')


def test_new_label_events_preserve_existing_earlier_assignment_history(hj):
    hj['core'].assign_tag('legacy-history', hj['tag'], hj['samples'][0], at('09:00:00'), actor='fixture', reason='existing history')
    hj['timeline'].label_event('new-history', hj['tag'], at('09:30:00'), target_id=hj['samples'][1], actor='fixture', reason='new reuse')
    assert hj['timeline'].resolve_label(hj['tag'], at('09:05:00'))['target_id'] == hj['samples'][0]
    assert hj['timeline'].resolve_label(hj['tag'], at('09:31:00'))['target_id'] == hj['samples'][1]


def test_our_qr_business_binding_contains_experiment_and_sample_information(hj):
    t = hj['timeline']
    t.label_event('sample-assignment', hj['tag'], at('09:00:00'), target_id=hj['samples'][0], actor='fixture', reason='fixture sample assignment')
    rid = run(hj)
    result = t.bind_qr('qr-bind', rid, 'fixture-equipment-qr', 'fixture-sample-tag', at('09:01:00'), actor='fixture')
    assert result['sample_id'] == hj['samples'][0] and result['run_id'] == rid
    assert result['person_id'] == hj['person'] and result['label_assignment_id']
    assert result['business_snapshot']['instrument']['id'] == hj['instruments'][0]
    assert result['business_snapshot']['sample']['id'] == hj['samples'][0]
    assert result['business_snapshot']['experiment_type'] == 'fixture-test'
    assert t.bind_qr('qr-bind', rid, 'fixture-equipment-qr', 'fixture-sample-tag', at('09:01:00'), actor='fixture') == result


def test_unknown_capture_clock_does_not_inherit_latest_binding(hj):
    rid = run(hj)
    bind(hj, rid)
    m = media(hj, 'unknown-clock', captured=False)
    value = activity(hj, 'unknown-clock-event', m, sample_id=hj['samples'][0], instrument_id=hj['instruments'][0])
    assert value['run_ids'] == [] and value['start_at'] is None


def test_same_sample_parallel_experiments_are_ambiguous_without_run_identity(hj):
    a, b = run(hj), run(hj, 'B')
    bind(hj, a)
    bind(hj, b, clock='09:00:01')
    m = media(hj, 'ambiguous', '09:01:00')
    value = activity(hj, 'ambiguous-event', m, sample_id=hj['samples'][0], instrument_id=hj['instruments'][0])
    assert value['association_status'] == 'ambiguous' and value['run_ids'] == []


def test_preparation_does_not_close_previous_run_even_on_exclusive_device(hj):
    policies = TimelinePolicies({'schema_version': 'vc-lab-timeline-policy/1', 'defaults': {'handover': 'exclusive_position'}})
    t = Timeline(hj['core'], policies)
    a, b = run(hj, timeline=t), run(hj, 'B', 1, t)
    bind(hj, a, timeline=t)
    bind(hj, b, sample=1, clock='09:05:00', semantics='preparation', timeline=t)
    assert record(hj, a)['record_status'] == 'open'


def test_verified_position_handover_only_closes_affected_record(hj):
    t = Timeline(hj['core'], TimelinePolicies({'schema_version': 'vc-lab-timeline-policy/1', 'defaults': {'handover': 'exclusive_position'}}))
    a, b, c = run(hj, timeline=t), run(hj, 'B', 1, t), run(hj, 'C', 2, t)
    bind(hj, a, position='test-position', timeline=t)
    bind(hj, c, sample=2, position='other-position', timeline=t)
    bind(hj, b, sample=1, clock='09:20:00', position='test-position', semantics='handover', timeline=t)
    assert record(hj, a)['closure_reason'] == 'verified_position_handover'
    assert record(hj, b)['record_status'] == record(hj, c)['record_status'] == 'open'


def test_multi_sample_device_and_other_instrument_binding_remain_independent(hj):
    a, b = run(hj), run(hj, 'B', 1)
    bind(hj, a)
    bind(hj, b, sample=1, clock='09:05:00', semantics='handover')
    bind(hj, b, sample=1, instrument=1, clock='09:10:00')
    assert record(hj, a)['record_status'] == 'open'


def test_parallel_shared_action_and_serial_lineage_without_inventory_writes(hj):
    a, b = run(hj), run(hj, 'B', 1)
    hj['timeline'].link_runs('serial', a, b, relation='sample_derivation')
    with pytest.raises(Conflict):
        hj['timeline'].link_runs('cycle', b, a)
    m = media(hj, 'shared')
    value = activity(hj, 'shared-action', m, run_ids=[a, b], kind='settings')
    assert set(value['run_ids']) == {a, b}
    assert all(len(TimelineViews(hj['timeline']).clip_index(r)['clips']) == 1 for r in (a, b))
    with hj['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 0


def test_low_priority_keeps_original_evidence_and_unknown_is_not_low_value(hj):
    rid, m = run(hj), media(hj, 'priority')
    activity(hj, 'chat', m, kind='chat', run_ids=[rid])
    activity(hj, 'unknown', m, start=2, end=3, kind='unknown', run_ids=[rid])
    views = TimelineViews(hj['timeline'])
    assert len(views.clip_index(rid)['clips']) == 1
    assert len(views.clip_index(rid, include_low_priority=True)['clips']) == 2
    assert len(views.trace(at('09:00:00'), at('09:15:00'))['activities']) == 2


def test_trace_identifies_recording_gaps_and_overlaps(hj):
    media(hj, 'one', '09:00:00', 60)
    media(hj, 'overlap', '09:00:50', 20)
    media(hj, 'after-gap', '09:02:00', 60)
    trace = TimelineViews(hj['timeline']).trace(at('09:00:00'), at('09:03:00'))
    coverage = trace['indexed_coverage'][0]
    assert len(coverage['gaps']) == 1 and len(coverage['overlaps']) == 1
    assert trace['coverage'] != 'complete'


def test_window_plans_carry_context_only_within_same_source_session(hj):
    a = media(hj, 'previous', '09:00:00', 60)
    b = media(hj, 'next', '09:01:00', 60)
    c = media(hj, 'unrelated-session', '09:02:00', 60, session='different-session')
    tasks = TimelineTasks(hj['timeline'])
    with hj['core'].db.transaction(readonly=True) as conn:
        get = lambda m: conn.execute('SELECT * FROM hj_timeline_media WHERE customer_id=%s AND media_id=%s', (hj['core'].customer, m['media_id'])).fetchone()
        plan = tasks.plan(conn, get(b), 'video_understanding')
        assert a['media_id'] in {x['media_id'] for x in plan['windows'][0]['input_spans']}
        assert c['media_id'] not in {x['media_id'] for w in plan['windows'] for x in w['input_spans']}
        assert sum(float(w['core']['end_seconds']) - float(w['core']['start_seconds']) for w in plan['windows']) == 60


def result_for(claim, events=None):
    return dict(schema_version='vc-lab-timeline-model-result/1', input_fingerprint=claim['request']['input_fingerprint'],
                covered_window_indices=[w['index'] for w in claim['request']['windows']], events=events or [])


def test_model_task_leases_and_exact_result_identity(hj):
    m = media(hj, 'model-input', duration=30)
    tasks = TimelineTasks(hj['timeline'])
    claim = tasks.claim('claim', 'video_understanding', 'fixture/1', now=at('09:01:00'))
    with pytest.raises(Conflict):
        tasks.submit('wrong-model', claim['task_id'], claim['lease_token'], 'fixture/2', result_for(claim), now=at('09:01:01'))
    bad = result_for(claim, [dict(kind='transfer', spans=[span(m, 0, 31)])])
    with pytest.raises(ValueError):
        tasks.submit('bad-range', claim['task_id'], claim['lease_token'], 'fixture/1', bad, now=at('09:01:01'))
    good = result_for(claim, [dict(kind='transfer', text='fixture transfer', spans=[span(m, 1, 2)])])
    done = tasks.submit('submit', claim['task_id'], claim['lease_token'], 'fixture/1', good, now=at('09:01:02'))
    assert done['status'] == 'completed'
    assert tasks.submit('submit', claim['task_id'], claim['lease_token'], 'fixture/1', good, now=at('09:01:02')) == done


def test_expired_task_cannot_commit_and_empty_json_is_not_quality_success(hj):
    media(hj, 'lease-input', duration=30)
    tasks = TimelineTasks(hj['timeline'])
    old = tasks.claim('old', 'video_understanding', 'fixture/1', lease_seconds=10, now=at('09:00:00'))
    new = tasks.claim('new', 'video_understanding', 'fixture/1', now=at('09:00:11'))
    with pytest.raises(Conflict):
        tasks.submit('expired-result', old['task_id'], old['lease_token'], 'fixture/1', result_for(old), now=at('09:00:12'))
    done = tasks.submit('empty-result', new['task_id'], new['lease_token'], 'fixture/1', result_for(new), now=at('09:00:12'))
    assert done['outcome'] == 'empty_observations_not_verified_absence'


def test_triplet_is_one_readout_and_conflicting_numbers_stay_candidates(hj):
    photos = [media(hj, 'photo-' + str(i), '09:00:0' + str(i), 0, 'photo', session='fixture-burst', stream='fixture-photo-camera', group_expected_photos=3) for i in range(3)]
    tasks = TimelineTasks(hj['timeline'])
    for i, value in enumerate(('1130', '1130', '1138')):
        claim = tasks.claim('claim-ocr-' + str(i), 'ocr', 'fixture-ocr/1', now=at('09:01:00'))
        m = {'media_id': claim['request']['primary_media_id']}
        result = result_for(claim, [dict(kind='reading', spans=[span(m, 0, 0)], fields=[dict(field_key='display', value=value, unit=None)])])
        tasks.submit('submit-ocr-' + str(i), claim['task_id'], claim['lease_token'], 'fixture-ocr/1', result, now=at('09:01:01'))
    with hj['core'].db.transaction(readonly=True) as conn:
        session = conn.execute('SELECT session_id FROM capture WHERE customer_id=%s AND id=%s', (hj['core'].customer, photos[0]['capture_id'])).fetchone()['session_id']
    readout = TimelineViews(hj['timeline']).readout(session)
    assert readout['received_photos'] == readout['ocr_completed_photos'] == 3
    assert readout['burst_status'] == 'complete' and readout['fields'][0]['status'] == 'conflicting_candidates'
    assert not readout['business_write_performed'] and readout['external_confirmation'] == 'not_provided'


def test_auto_archive_produces_report_without_asserting_instrument_completion(hj):
    rid = run(hj)
    bind(hj, rid)
    result = hj['timeline'].close_due('2026-10-28T00:00:00+08:00')
    assert result['archived'] == [rid] and not result['physical_completion_inferred']
    views = TimelineViews(hj['timeline'])
    old = views.snapshot(DAY)
    assert old == views.snapshot(DAY)
    entry = old['document']['experiments'][0]
    assert entry['record']['record_status'] == 'archived' and entry['instrument_completion_time'] is None
    late = media(hj, 'late-return', '09:20:00', 180)
    activity(hj, 'late-return-event', late, kind='reading', run_ids=[rid])
    new = views.snapshot(DAY)
    assert new['revision'] > old['revision'] and new['document']['activity_count'] == 1
    assert record(hj, rid)['record_status'] == 'archived'


def test_midnight_event_boundaries_and_cross_day_continuation(hj):
    rid = run(hj)
    m = media(hj, 'midnight', '23:59:00', 120)
    activity(hj, 'before-midnight', m, end=60, run_ids=[rid])
    activity(hj, 'after-midnight', m, start=60, end=120, run_ids=[rid])
    views = TimelineViews(hj['timeline'])
    assert views.snapshot(DAY)['document']['activity_count'] == 1
    assert views.snapshot('2026-10-28')['document']['activity_count'] == 1


def test_internal_api_auth_and_customer_isolation(hj, db):
    rid = run(hj)
    bind(hj, rid)
    client = TestClient(create_app(hj['core'], hj['settings']))
    assert client.get('/internal/timeline/reports/' + DAY).status_code == 401
    headers = {'Authorization': 'Bearer ' + hj['settings'].token}
    assert client.get('/internal/timeline/reports/' + DAY, headers=headers).status_code == 200
    other = Core(db, Core.customer_instance(db, 'other-' + str(uuid4()), scope='fixture')['id'])
    with pytest.raises(Missing):
        TimelineViews(Timeline(other)).clip_index(rid)


def test_concurrent_retry_creates_one_binding(hj):
    rid = run(hj)
    def work(_):
        return bind(hj, rid, key='same-intent')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(8)))
    assert len({r['id'] for r in results}) == 1


def test_timeline_runtime_is_opt_in_and_refreshes_dirty_reports(hj):
    runtime = TimelineRuntime(hj['core'], replace(hj['settings'], processing_profile='native'))
    assert not runtime.tick()['enabled']
    bind(hj, run(hj))
    tick = TimelineRuntime(hj['core'], hj['settings']).tick(now=at('12:00:00'))
    assert tick['enabled'] and tick['reports'][0]['day'] == DAY


def test_closure_policy_does_not_allow_infinite_wait_or_15_minute_assumption():
    with pytest.raises(ValueError):
        TimelinePolicies({'schema_version': 'vc-lab-timeline-policy/1', 'defaults': {'daily_archive': False}})
    assert TimelinePolicies().record()['inactivity_seconds'] is None


def test_late_binding_association_is_revised_without_rewriting_observation(hj):
    rid = run(hj)
    m = media(hj, 'before-binding-arrives', '09:05:00', 30)
    original = activity(hj, 'before-binding', m, sample_id=hj['samples'][0], instrument_id=hj['instruments'][0])
    assert original['run_ids'] == []
    bind(hj, rid)
    assert hj['timeline'].reconcile_pending()['revised_activities'] == [original['id']]
    trace = TimelineViews(hj['timeline']).trace(at('09:00:00'), at('09:10:00'))
    revised = trace['activities'][0]
    assert revised['run_ids'] == [rid] and revised['association_status'] == 'unassigned'
    assert revised['current_association_status'] == 'bound_identity'
    assert len(revised['association_revisions']) == 1


def test_clock_uncertainty_crossing_recycled_label_is_not_guessed(hj):
    t = hj['timeline']
    t.label_event('A', hj['tag'], at('09:00:00'), target_id=hj['samples'][0], actor='fixture', reason='fixture')
    t.label_event('B', hj['tag'], at('09:05:01'), target_id=hj['samples'][1], actor='fixture', reason='fixture')
    m = media(hj, 'uncertain-clock', '09:05:00', 30)
    with hj['core'].db.transaction() as conn:
        conn.execute("UPDATE hj_timeline_media SET time_basis=time_basis||'{\"uncertainty_us\":2000000}'::jsonb WHERE customer_id=%s AND media_id=%s", (hj['core'].customer, m['media_id']))
    value = activity(hj, 'uncertain-event', m, tag_id=hj['tag'], instrument_id=hj['instruments'][0])
    assert value['run_ids'] == [] and value['association_status'] == 'unresolved_label'


def test_existing_qr_decoder_reads_our_reusable_sample_assignment(hj):
    hj['timeline'].label_event('label', hj['tag'], at('09:00:00'), target_id=hj['samples'][0], actor='fixture', reason='fixture')
    path = hj['settings'].incoming / 'actual-fixture-qr.png'
    qrcode.make('fixture-sample-tag').save(path)
    Ingest(hj['core'], hj['settings']).accept(dict(capture_id='qr-photo', source_instance='fixture-camera', source_revision='fixture/1', kind='photo', relative_path=path.name, captured_at=at('09:05:00'), end_seconds='0', time_basis={'basis': 'fixture'}, source_kind='fixture', sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = Worker(hj['core'], hj['settings']).run_once()
    assert result['status'] == 'completed'
    with hj['core'].db.transaction(readonly=True) as conn:
        payload = conn.execute("SELECT payload FROM observation WHERE customer_id=%s AND kind='qr'", (hj['core'].customer,)).fetchone()['payload']
    assert payload['tag_associations'][0]['target_id'] == hj['samples'][0]
    assert payload['tag_associations'][0]['identity_only']


def test_closure_due_filter_does_not_starve_short_policy_after_old_long_record(hj):
    t = Timeline(hj['core'], TimelinePolicies({'schema_version': 'vc-lab-timeline-policy/1', 'defaults': {'daily_archive': False, 'inactivity_seconds': 864000}}))
    old = run(hj, 'long', timeline=t)
    bind(hj, old, timeline=t)
    quick = Timeline(hj['core'], TimelinePolicies({'schema_version': 'vc-lab-timeline-policy/1', 'defaults': {'daily_archive': False, 'inactivity_seconds': 60}}))
    recent = run(hj, 'short', timeline=quick)
    bind(hj, recent, clock='09:10:00', timeline=quick)
    result = quick.close_due(at('09:12:00'), limit=1)
    assert result['archived'] == [recent] and record(hj, old)['record_status'] == 'open'


def test_failed_activity_transaction_leaves_no_half_evidence_or_record_update(hj, monkeypatch):
    rid, m = run(hj), media(hj, 'rollback', duration=30)
    original = hj['core']._emit
    def fail(conn, kind, *args):
        if kind == 'experiment_activity':
            raise RuntimeError('synthetic interruption')
        return original(conn, kind, *args)
    monkeypatch.setattr(hj['core'], '_emit', fail)
    with pytest.raises(RuntimeError):
        activity(hj, 'interrupted', m, run_ids=[rid])
    assert record(hj, rid)['last_observed_at'] is None
    with hj['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM hj_activity WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 0


def test_label_history_and_binding_can_both_arrive_after_the_observation(hj):
    rid = run(hj)
    m = media(hj, 'before-all-business-events', '09:05:00', 30)
    observed = activity(hj, 'before-all-events', m, tag_id=hj['tag'], instrument_id=hj['instruments'][0])
    assert observed['association_status'] == 'unresolved_label'
    hj['timeline'].label_event('late-label', hj['tag'], at('09:00:00'), target_id=hj['samples'][0], actor='fixture', reason='late assignment')
    hj['timeline'].reconcile_pending()
    bind(hj, rid)
    hj['timeline'].reconcile_pending()
    trace = TimelineViews(hj['timeline']).trace(at('09:00:00'), at('09:10:00'))
    assert trace['activities'][0]['run_ids'] == [rid]
    assert len(trace['activities'][0]['association_revisions']) == 2


def test_timeline_worker_does_not_claim_old_detector_jobs(hj):
    legacy = dict(hj, settings=replace(hj['settings'], processing_profile='native'))
    media(legacy, 'old-native-video', duration=30)
    assert Worker(hj['core'], hj['settings']).claim() is None
    with hj['core'].db.transaction(readonly=True) as conn:
        row = conn.execute('SELECT status FROM processing_job WHERE customer_id=%s', (hj['core'].customer,)).fetchone()
    assert row['status'] == 'queued'


def test_upload_marker_prevents_premature_timeline_admission(hj):
    marker = hj['settings'].incoming / '.uploading'
    marker.write_text('fixture upload in progress')
    with pytest.raises(ValueError):
        media(hj, 'unfinished-upload', duration=180)
    with hj['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM hj_timeline_media WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 0
    marker.unlink()
    media(hj, 'unfinished-upload', duration=180)

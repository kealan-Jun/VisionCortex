"""Model-free value consumption: candidates are observable, never retention authority."""
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from visioncortex.lab.native_timeline_links import build_sidecar
from visioncortex.lab.timeline_report import render_report
from visioncortex.lab.timeline_runner import TimelineRunner
from visioncortex.lab.timeline_tasks import TimelineTasks
from visioncortex.lab.timeline_views import TimelineViews
from visioncortex.lab.value_candidates import attach, candidate, pending_event
from .test_timeline import DAY, at, hj as timeline_fixture, media, result_for, span


@pytest.fixture(name='hj')
def context(core, settings):
    return timeline_fixture.__wrapped__(core, settings)


def source(mid='fixture-media', lo='0', hi='10'):
    return [dict(media_id=mid, start_seconds=lo, end_seconds=hi)]


@pytest.mark.parametrize(('experiment', 'value', 'category'), [
    ('experimental', 'valuable', 'ExperimentActivity'),
    ('non_experimental', 'valuable', 'ValuableNoExperimentActivity'),
    ('non_experimental', 'no_value', 'NoValueNoExperimentActivity'),
    ('experimental', 'no_value', 'PendingValueReview'),
    ('experimental', 'unknown', 'PendingValueReview'),
    ('non_experimental', 'unknown', 'PendingValueReview'),
    ('unknown', 'valuable', 'PendingValueReview'),
    ('unknown', 'no_value', 'PendingValueReview'),
    ('unknown', 'unknown', 'PendingValueReview'),
])
def test_axes_are_independent_and_categories_have_no_retention_authority(experiment, value, category):
    out = candidate(dict(experiment_status=experiment, value_status=value, reason='fixture observed evidence'), source())
    assert out['category_candidate'] == category
    assert not out['classification_is_physical_ground_truth'] and not out['confirmed_classification']
    assert not out['confirmed_no_value'] and out['retention_effect'] == 'none'
    assert out['selection_decision'] == 'unchanged'


def test_supplied_confirmation_and_source_pins_cannot_upgrade_a_value_claim():
    raw = dict(experiment_status='non_experimental', value_status='no_value', reason='fixture only',
               confirmed_classification=True, confirmed_no_value=True, retention_effect='discard',
               source_spans=source('foreign-media'), model_revision='invented', input_fingerprint='fake')
    out = candidate(raw, source(), model_revision='actual/1', input_fingerprint='a' * 64,
                    execution_verification='supplied_not_independently_verified')
    assert out['model_revision'] == 'actual/1' and out['input_fingerprint'] == 'a' * 64
    assert out['source_spans'] == source() and not out['confirmed_no_value']
    assert out['execution_verification'] == 'supplied_not_independently_verified'


def test_missing_value_is_unknown_even_when_kind_says_rest_or_transfer():
    for kind in ('rest', 'transfer', 'waiting'):
        event = attach(dict(kind=kind, text='fixture', spans=source()))
        assert event['payload']['value_candidate']['category_candidate'] == 'PendingValueReview'
        assert event['kind'] == kind and event['spans'] == source()


@pytest.mark.parametrize('change', [
    {'experiment_status': 'confirmed'}, {'value_status': True}, {'value_status': {}},
    {'reason': ''}, {'boundary_uncertainty': {'start': ['-1', '0'], 'end': ['10', '10']}},
    {'boundary_uncertainty': {'start': ['0', '0'], 'end': ['10', '11']}},
    {'boundary_uncertainty': {'start': ['1', '2'], 'end': ['9', '10']}},
])
def test_invalid_or_escaped_value_evidence_is_rejected(change):
    with pytest.raises(ValueError):
        candidate(dict(reason='fixture', **change) if 'reason' not in change else change, source())


def test_empty_answer_is_only_pending_assessment_with_original_extent():
    event = pending_event(source())
    assert event['kind'] == 'unknown' and event['spans'] == source()
    value = event['payload']['value_candidate']
    assert value['value_assessment_only'] and value['category_candidate'] == 'PendingValueReview'
    assert not value['confirmed_no_value']


def video_result(rows):
    body = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'events': rows})}}],
            'model': 'fixture-model'}
    return body


def invoke_video(rows):
    runner = TimelineRunner.__new__(TimelineRunner)
    runner.media = SimpleNamespace(prepare=lambda *args: (b'fixture-not-a-real-video', 'video/mp4', {'fixture': True}))
    calls = []
    def call(endpoint, body):
        calls.append(body)
        return video_result(rows), {'fixture_transport': True}
    ref = dict(media_id='fixture-media', start_seconds='5', end_seconds='25', sha256='a' * 64, media_version=1)
    window = dict(core=dict(media_id='fixture-media', start_seconds='10', end_seconds='20'),
                  input_spans=[ref], continuity_issues=[])
    events, receipts = runner._video(SimpleNamespace(call=call), {'served_model': 'fixture-model'}, window, None)
    return events, receipts, calls


def test_actual_video_prompt_accepts_rich_value_and_maps_uncertainty_without_recutting():
    raw = dict(kind='waiting', text='fixture visible waiting', start_seconds='6', end_seconds='8',
               experiment_status='unknown', value_status='valuable', reason='purpose unknown',
               boundary_uncertainty=dict(start=['5', '7'], end=['8', '9']))
    events, _, calls = invoke_video([raw])
    value = events[0]['payload']['value_candidate']
    assert value['category_candidate'] == 'PendingValueReview'
    assert value['source_spans'] == source(lo='11', hi='13')
    assert value['boundary_uncertainty'] == {'start': ['10', '12'], 'end': ['13', '14']}
    prompt = calls[0]['messages'][0]['content'][-1]['text']
    assert 'experiment_status' in prompt and 'value_status' in prompt and 'unknown' in prompt
    assert events[0]['kind'] == 'waiting'


def test_legacy_video_reply_and_empty_reply_remain_explicit_pending():
    legacy = dict(kind='other_work', text='fixture action', start_seconds='6', end_seconds='8')
    events, _, _ = invoke_video([legacy])
    assert events[0]['payload']['value_candidate']['category_candidate'] == 'PendingValueReview'
    empty, _, _ = invoke_video([])
    assert empty[0]['spans'] == source(lo='10', hi='20')
    assert empty[0]['payload']['value_candidate']['value_assessment_only']


def test_value_boundary_cannot_claim_context_outside_primary_core():
    raw = dict(kind='other_work', text='fixture', start_seconds='6', end_seconds='8',
               boundary_uncertainty=dict(start=['0', '6'], end=['8', '9']))
    with pytest.raises(ValueError, match='owned core'):
        invoke_video([raw])


def claim_value(hj, name='value-input', *, captured=True):
    m = media(hj, name, duration=30, captured=captured)
    tasks = TimelineTasks(hj['timeline'])
    claim = tasks.claim('claim-' + name, 'video_understanding', 'fixture-value-model/1', now=at('09:01:00'))
    return m, tasks, claim


def test_db_four_pending_results_are_consumed_queried_reported_and_retried_idempotently(hj):
    m, tasks, claim = claim_value(hj)
    sentinel = hj['settings'].runtime_root / 'existing-native-retention.json'
    sentinel.write_text('{"video_value":"selected","owner":"native"}')
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    events = [pending_event([span(m, i, i + 1)], reason='fixture missing experiment/value evidence') for i in range(4)]
    result = result_for(claim, events)
    done = tasks.submit('submit-four-pending', claim['task_id'], claim['lease_token'], claim['model_revision'], result, now=at('09:01:01'))
    assert tasks.submit('submit-four-pending', claim['task_id'], claim['lease_token'], claim['model_revision'], result, now=at('09:01:01')) == done
    assert len(done['activity_ids']) == 4 and done['outcome'] == 'pending_value_assessments_not_physical_actions'
    views = TimelineViews(hj['timeline'])
    trace = views.trace(at('09:00:00'), at('09:02:00'))
    assert len(trace['video_value_candidates']) == 4
    assert all(v['candidate']['category_candidate'] == 'PendingValueReview' for v in trace['video_value_candidates'])
    report = views.snapshot(DAY)['document']
    assert len(report['video_value_candidates']) == 4
    assert '不能计为已观察到的物理动作' in render_report(report)
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before
    with hj['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 0
        assert conn.execute('SELECT count(*) AS n FROM media_derivation WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 0
    sidecar = build_sidecar(archive=DAY + '_fixture-camera', customer_id=hj['core'].customer, revision=1,
                           snapshot_fingerprint='b' * 64, rules={'sha256': 'c' * 64}, canonical_report='fixture.json',
                           activities=trace['activities'], person_projection={})
    assert len(sidecar['observations']) == 4
    assert all(x['value_candidate']['category_candidate'] == 'PendingValueReview' for x in sidecar['observations'])
    assert not sidecar['interpretation']['physical_clip_generation']


def test_db_no_value_model_claim_never_discards_source_or_becomes_confirmation(hj):
    m, tasks, claim = claim_value(hj)
    event = attach(dict(kind='waiting', text='fixture only', spans=[span(m, 1, 2)]),
                   dict(experiment_status='non_experimental', value_status='no_value', reason='fixture assertion',
                        confirmed_no_value=True, retention_effect='discard'))
    tasks.submit('no-value', claim['task_id'], claim['lease_token'], claim['model_revision'], result_for(claim, [event]), now=at('09:01:01'))
    value = TimelineViews(hj['timeline']).trace(at('09:00:00'), at('09:02:00'))['video_value_candidates'][0]['candidate']
    assert value['category_candidate'] == 'NoValueNoExperimentActivity' and value['status'] == 'candidate'
    assert value['retention_effect'] == 'none' and not value['confirmed_no_value']
    assert value['model_revision'] == claim['model_revision']
    assert value['input_fingerprint'] == claim['request']['input_fingerprint']
    assert value['source_inputs'][0]['sha256'] == claim['request']['windows'][0]['input_spans'][0]['sha256']
    assert value['execution_verification'] == 'supplied_not_independently_verified'


def test_db_unknown_time_pending_has_no_invented_day_or_run(hj):
    m, tasks, claim = claim_value(hj, captured=False)
    event = pending_event([span(m, 0, 1)])
    tasks.submit('unknown-time-value', claim['task_id'], claim['lease_token'], claim['model_revision'], result_for(claim, [event]), now=at('09:01:01'))
    views = TimelineViews(hj['timeline'])
    trace = views.trace(at('09:00:00'), at('09:02:00'))
    assert trace['video_value_candidates'] == []
    unknown = trace['unknown_time_video_value_candidates']['candidates']
    assert len(unknown) == 1 and unknown[0]['start_at'] is None
    assert unknown[0]['range_assignment'] == 'unknown_time_not_assigned_to_requested_range'
    report = views.snapshot(DAY)['document']
    assert report['video_value_candidates'] == []
    assert len(report['unknown_time_video_value_candidates']['candidates']) == 1
    assert report['experiments'] == []


def test_db_foreign_or_escaped_candidate_evidence_rolls_back(hj):
    m, tasks, claim = claim_value(hj)
    event = pending_event([span(m, 0, 1)])
    event['spans'][0]['media_id'] = str(uuid4())
    with pytest.raises(ValueError, match='primary media'):
        tasks.submit('foreign-value-source', claim['task_id'], claim['lease_token'], claim['model_revision'], result_for(claim, [event]), now=at('09:01:01'))
    with hj['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM hj_activity WHERE customer_id=%s', (hj['core'].customer,)).fetchone()['n'] == 0

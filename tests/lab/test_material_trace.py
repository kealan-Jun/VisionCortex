"""Actual PostgreSQL/Core histories; synthetic business identities, no real gold."""
from copy import deepcopy

import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.material_trace import MaterialTrace
from ._fixtures import db as evidence_db, settings as evidence_settings


@pytest.fixture(scope='session', name='db')
def fixture_db():
    return evidence_db.__wrapped__()


@pytest.fixture
def settings(core, tmp_path):
    return evidence_settings.__wrapped__(core, tmp_path)


def at(minute):
    return f'2026-10-28T10:{minute:02d}:00+08:00'


@pytest.fixture
def sample(core):
    source1 = core.register('source1', 'source_lot', 'RAW-01')['id']
    source2 = core.register('source2', 'source_lot', 'RAW-02')['id']
    target = core.register('target', 'process_batch', 'PREP-07')['id']
    m1 = core.register_material('m1', 'M1', 'g', '20', metadata={'source_lot_id': source1}, occurred_at=at(0))['material_id']
    m2 = core.register_material('m2', 'M2', 'g', None, metadata={'source_lot_id': source2}, occurred_at=at(0))['material_id']
    c1 = core.register('c1', 'container', 'BOTTLE-1')['id']
    tag = core.register('tag', 'tag', 'LABEL-1', {'mode': 'reusable'})['id']
    core.assign_tag('assign1', tag, c1, at(0), actor='fixture-registrar', reason='fixture physical registration')
    core.bind_container('content1', c1, m1, at(0), actor='fixture-registrar')
    return dict(trace=MaterialTrace(core), m1=m1, m2=m2, c1=c1, tag=tag, target=target, source1=source1, source2=source2)


def observe(sample, key='observation', occurrence='action-1', **kwargs):
    return sample['trace'].append_observation(key, occurrence, ['LABEL-1'], actor='fixture-observer', reason='fixture evidence', event_start=at(5), clock_status='verified', **kwargs)


def test_old_video_uses_original_container_contents_after_reuse(core, sample):
    old = observe(sample)
    core.bind_container('content2', sample['c1'], sample['m2'], at(10), actor='fixture-registrar')
    late = observe(sample, 'late', 'late-action')
    assert late['snapshot']['resolutions'][0]['material_id'] == sample['m1']
    assert late['snapshot']['resolutions'][0]['source_lot_id'] == sample['source1']
    new = sample['trace'].resolve_label('LABEL-1', event_start=at(15), clock_status='verified')
    assert new['material_id'] == sample['m2'] and new['source_lot_id'] == sample['source2']
    assert sample['trace'].history('action-1')['current']['snapshot'] == old['snapshot']


def test_reusable_label_binding_preserves_original_material(core, sample):
    core.assign_tag('assign2', sample['tag'], sample['m2'], at(10), actor='fixture-registrar', reason='reuse')
    first = sample['trace'].resolve_label('LABEL-1', event_start=at(5), clock_status='verified')
    last = sample['trace'].resolve_label('LABEL-1', event_start=at(15), clock_status='verified')
    assert first['material_id'] == sample['m1'] and last['material_id'] == sample['m2']
    assert first['tag_assignment']['version'] == 1 and last['tag_assignment']['version'] == 2


@pytest.mark.parametrize('history', ['tag', 'content'])
def test_uncertain_time_crossing_binding_boundary_is_not_latest(core, sample, history):
    if history == 'tag':
        core.assign_tag('next', sample['tag'], sample['m2'], at(10), actor='fixture-registrar', reason='reuse')
    else:
        core.bind_container('next', sample['c1'], sample['m2'], at(10), actor='fixture-registrar')
    result = sample['trace'].resolve_label('LABEL-1', event_start=at(9), event_end=at(11), clock_status='verified')
    assert result['association_status'] == 'needs_review' and result['material_id'] is None


def test_unverified_clock_never_reads_current_relation(sample):
    result = sample['trace'].resolve_label('LABEL-1', event_start=at(5), clock_status='uncalibrated')
    assert result['material_id'] is None and result['reasons'] == ['event_clock_unverified_no_current_binding_fallback']


def test_seen_and_handled_never_consume_or_clear_unknown_balance(core, sample):
    before = sample['trace'].material_history(sample['m1'])
    for index, relation in enumerate(['seen', 'handled', 'usage_candidate']):
        row = observe(sample, f'seen-{index}', f'action-{index}', relation=relation)
        assert row['snapshot']['quantity_ledger_written'] is False
    after = sample['trace'].material_history(sample['m1'])
    assert after['state'] == before['state'] and after['operations'] == before['operations']
    assert sample['trace'].material_history(sample['m2'])['state']['balance'] is None


def test_same_action_merges_evidence_distinct_actions_same_value_do_not(sample):
    q = [dict(label='LABEL-1', value='2.000', unit='g', semantics='spoken_usage', source='speech')]
    one = observe(sample, 'one', 'take-1', quantities=q, relation='usage_candidate', target_batch_id=sample['target'])
    assert observe(sample, 'one', 'take-1', quantities=q, relation='usage_candidate', target_batch_id=sample['target']) == one
    two = observe(sample, 'two', 'take-1', quantities=deepcopy(q), relation='usage_candidate', target_batch_id=sample['target'])
    assert two['revision'] == 2 and len(two['snapshot']['quantities']) == 1
    separate = observe(sample, 'three', 'take-2', quantities=q, relation='usage_candidate', target_batch_id=sample['target'])
    assert separate['revision'] == 1
    assert two['snapshot']['resolutions'][0]['source_lot_id'] != two['snapshot']['event_context']['target_batch_id']


def test_candidate_identity_cannot_be_replaced_by_another_target(sample, core):
    observe(sample)
    other = core.register('target-other', 'process_batch', 'OTHER')['id']
    with pytest.raises(Conflict):
        observe(sample, 'mutate', target_batch_id=other)
    corrected = observe(sample, 'correct', target_batch_id=other, correction=True, expected_revision=1)
    history = sample['trace'].history('action-1')
    assert len(history['revisions']) == 2 and history['revisions'][0]['snapshot']['event_context']['target_batch_id'] is None
    assert corrected['snapshot']['supersedes_id'] == history['revisions'][0]['id']
    before = sample['trace'].history('action-1', known_at=history['revisions'][0]['processed_at'])
    assert len(before['revisions']) == 1


def test_duplicate_physical_label_never_assigns_inventory(sample):
    row = observe(sample, visual_label_conflicts=['LABEL-1'])
    result = row['snapshot']['resolutions'][0]
    assert result['material_id'] is None and result['association_status'] == 'needs_review'


def test_gross_reading_and_conflicting_speech_are_explicit_candidates(sample):
    q = [dict(label='LABEL-1', value='10', unit='g', semantics='gross')]
    one = observe(sample, quantities=q, relation='usage_candidate')
    assert 'screen_quantity_is_not_verified_usage' in one['snapshot']['reasons']
    q2 = [dict(label='LABEL-1', value='2', unit='g', semantics='spoken_usage')]
    two = observe(sample, 'speech', quantities=q2, relation='usage_candidate')
    assert 'quantity_or_semantics_conflict' in two['snapshot']['reasons']


def test_unknown_label_and_cross_customer_reference_are_not_registered(sample, core, db):
    assert sample['trace'].resolve_label('MISSING', event_start=at(5), clock_status='verified')['reasons'] == ['unregistered_label']
    with pytest.raises(ValueError):
        observe(sample, quantities=[dict(label='LABEL-1', value=0.1, unit='g', semantics='net')])
    with pytest.raises(ValueError):
        observe(sample, relation='confirmed')


def test_existing_authoritative_operation_is_visible_without_second_ledger(core, sample):
    core.operate('confirmed-use', 'consume', [dict(material_id=sample['m1'], quantity='2', unit='g', quantity_basis='verified_usage')], occurred_at=at(6), actor='fixture-authorized-confirmation', batch_id=sample['target'])
    result = sample['trace'].material_history(sample['m1'])
    assert result['state']['balance'] == '18.000000000'
    assert result['operations'][-1]['batch_id'] == sample['target']


def test_actual_authenticated_command_trace_and_report_share_core(core, sample, settings):
    from fastapi.testclient import TestClient
    from visioncortex.lab.api import create_app
    from visioncortex.lab.timeline_report import render_report
    from visioncortex.lab.timeline_runtime import TimelineRuntime
    client = TestClient(create_app(core, settings))
    headers = {'Authorization': 'Bearer ' + settings.token}
    body = dict(key='actual-entry', occurrence_key='real-entry-fixture-action', labels=['LABEL-1'],
                actor='fixture-observer', reason='fixture-localhost-test', event_start=at(5),
                clock_status='verified', relation='usage_candidate', target_batch_id=sample['target'])
    assert client.post('/internal/commands/timeline-observe-material', json=body).status_code == 401
    response = client.post('/internal/commands/timeline-observe-material', json=body, headers=headers)
    assert response.status_code == 200
    assert client.post('/internal/commands/timeline-observe-material', json=body, headers=headers).json() == response.json()
    history = client.post('/internal/commands/timeline-material-observation-history', json={'occurrence_key':body['occurrence_key']}, headers=headers).json()
    assert len(history['revisions']) == 1
    document = client.get('/internal/timeline/reports/2026-10-28', headers=headers).json()['document']
    assert document['context_evidence']['material_observations'][0]['id'] == response.json()['id']
    rendered = render_report(document)
    assert '来源批次' in rendered and '未扣库存' in rendered
    runtime = TimelineRuntime(core, settings)
    trace = runtime.views.trace(at(0), at(59))
    assert trace['context_evidence']['material_observations'][0]['snapshot']['quantity_ledger_written'] is False
    assert sample['trace'].material_history(sample['m1'])['state']['balance'] == '20.000000000'


def test_equivalent_decimal_and_units_preserve_evidence_without_false_conflict(sample):
    values = [dict(label='LABEL-1', value=value, unit=unit, semantics='spoken_usage') for value, unit in [('2', 'g'), ('2.000', 'g'), ('2000', 'mg')]]
    for index, item in enumerate(values):
        result = observe(sample, f'equivalent-{index}', 'same-action', relation='usage_candidate', quantities=[item])
        assert 'quantity_or_semantics_conflict' not in result['snapshot']['reasons']
    assert len(result['snapshot']['quantities']) == 3


def test_cross_day_observation_refreshes_middle_report_and_correction_old_range(core, sample, settings):
    from visioncortex.lab.timeline_runtime import TimelineRuntime
    runtime = TimelineRuntime(core, settings)
    before = runtime.views.snapshot('2026-10-27')
    row = sample['trace'].append_observation('cross-day', 'three-days', ['LABEL-1'], actor='fixture-observer', reason='fixture multi-day time uncertainty', clock_status='verified', event_start='2026-10-26T10:00:00+08:00', event_end='2026-10-28T10:00:00+08:00')
    runtime.views.refresh_dirty()
    after = runtime.views.snapshot('2026-10-27')
    assert after['revision'] == before['revision'] + 1
    assert after['document']['context_evidence']['material_observations'][0]['id'] == row['id']
    sample['trace'].append_observation('correct-cross-day', 'three-days', ['LABEL-1'], actor='fixture-observer', reason='fixture late timestamp correction', clock_status='verified', event_start=at(5), expected_revision=1, correction=True)
    runtime.views.refresh_dirty()
    corrected = runtime.views.snapshot('2026-10-27')
    assert corrected['revision'] == after['revision'] + 1
    assert corrected['document']['context_evidence']['material_observations'] == []

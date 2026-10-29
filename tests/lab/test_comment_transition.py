"""Structured reviewed speech fixtures, not real ASR semantic acceptance."""
from dataclasses import replace
import hashlib
from uuid import UUID
import wave

import pytest

from visioncortex.lab.domain import Conflict
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.timeline_commands import command_methods

from .test_operator_scan import scene as operator_scene, scan, context, at, rows


@pytest.fixture
def scene(core, settings):
    return operator_scene.__wrapped__(core, settings)


def source(s, settings, name='comment', clock='09:01:00', *, captured=True):
    path = settings.incoming / (name + '.wav')
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
        f.writeframes(b'\0\0' * 32000)
    receipt = dict(capture_id=name, source_instance='fixture-camera-0', camera_id='fixture-camera-0',
        source_revision='synthetic/1', kind='audio', session_id='fixture-audio-session',
        relative_path=path.name, captured_at=at(clock) if captured else None,
        time_basis={'basis':'synthetic_fixture_clock','uncertainty_us':0}, start_seconds='0', end_seconds='4',
        source_kind='fixture', sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    m = Ingest(s['core'], replace(settings, processing_profile='timeline-v1')).accept(receipt)
    spans = [dict(media_id=m['media_id'], start_seconds='0', end_seconds='1')]
    speech = s['runtime'].timeline.add_activity(name, 'speech', 'speech', '原音转写候选：阶段已完成，计划去另一设备。', spans,
        run_ids=[s['run']], sample_id=s['materials'][0], person_id=s['people'][0], model_revision='synthetic-asr-not-real/1')
    return dict(activity=speech, spans=spans, hash=hashlib.sha256(speech['text_content'].encode()).hexdigest())


def body(s, src, interpretation='stage_complete', **extra):
    result = dict(interpretation=interpretation, run_id=s['run'], sample_id=s['materials'][0], batch_id=s['batches'][0],
        source_activity_id=src['activity']['id'], source_text_sha256=src['hash'],
        source_evidence_ids=src['activity']['evidence_ids'], text=src['activity']['text_content'],
        occurred_at=src['activity']['end_at'], actor='synthetic-review-application', reason='structured fixture interpretation',
        person_id=s['people'][0], camera_id=s['cameras'][0])
    result.update(extra)
    return result


def reviewed(s, src, interpretation='stage_complete', *, status='reviewed'):
    data=body(s, src, interpretation)
    pin={k:data[k] for k in ('source_activity_id','source_text_sha256','interpretation','run_id','sample_id','batch_id','person_id','camera_id')}
    pin.update(source_evidence_ids=sorted(data['source_evidence_ids']), status=status, actor='synthetic reviewer', reason='fixture review, not real acceptance')
    review=s['runtime'].timeline.add_activity('review-'+interpretation+'-'+str(src['activity']['id']), 'settings', 'reviewed_note',
        'Explicit structured fixture review', src['spans'], payload={'comment_interpretation_review':pin})
    return dict(data, review_activity_id=review['id'])


def apply(s, key, data):
    return command_methods(s['runtime'])['timeline-comment-transition'](key, **data)


def run_row(s):
    with s['core'].db.transaction(readonly=True) as conn:
        return s['runtime'].timeline._run(conn, s['run'])


def test_stage_done_and_B_plan_preserve_same_batch_and_run_until_active_scan(scene, settings):
    scan(scene, 'A', run_id=scene['run'])
    scan(scene, 'material', kind='material', value='fixture-reusable-label', run_id=scene['run'], when='09:00:30')
    src=source(scene, settings)
    done=apply(scene, 'stage-done', reviewed(scene, src))
    plan=apply(scene, 'B-plan', reviewed(scene, src, 'next_stage_plan'))
    assert done['status']==plan['status']=='applied_reviewed_interpretation'
    assert context(scene)['equipment_id']==scene['devices'][0]
    assert run_row(scene)['record_status']=='open'
    switched=scan(scene, 'B', value=scene['devices'][1], when='09:02:00')['context']
    assert (switched['run_id'],switched['batch_id'])==(scene['run'],scene['batches'][0])
    assert switched['materials'][0]['resolution']['source_lot_id']==scene['lots'][0]
    assert run_row(scene)['record_status']=='open' and run_row(scene)['closed_at'] is None
    trace=scene['runtime'].views.trace(at('09:00:00'),at('09:03:00'))
    assert 'stage_complete' in str(trace) and 'next_stage_plan' in str(trace) and 'operator_scan_event' in str(trace)
    assert rows(scene,'quantity_ledger_entry')==[]


def test_only_explicit_reviewed_whole_completion_archives_once_without_release_or_usage(scene, settings):
    scan(scene, 'A', run_id=scene['run'])
    src=source(scene, settings, clock='09:04:00')
    data=reviewed(scene, src, 'experiment_complete')
    result=apply(scene, 'whole-done', data)
    assert result['status']=='applied_reviewed_interpretation' and result['snapshot']['whole_experiment_completed']
    assert apply(scene, 'whole-done', data)==result
    record=run_row(scene)
    assert record['record_status']=='archived' and record['closure_reason']=='explicit_reviewed_whole_experiment_comment'
    assert not result['snapshot']['equipment_released'] and not result['snapshot']['quantity_ledger_written']
    assert context(scene)['equipment_id']==scene['devices'][0]
    activities=[r for r in rows(scene,'hj_activity') if r['row']['payload'].get('comment_transition')=='experiment_complete']
    assert len(activities)==1 and activities[0]['row']['kind']=='end'
    assert rows(scene,'quantity_ledger_entry')==[]
    scene['runtime'].views.refresh_dirty()
    assert 'experiment_complete' in str(scene['runtime'].views.snapshot('2026-10-29'))


def test_unreviewed_or_ambiguous_ASR_interpretation_remains_attention(scene, settings):
    src=source(scene, settings)
    result=apply(scene, 'unreviewed', body(scene, src, 'experiment_complete'))
    ambiguous=apply(scene, 'ambiguous', reviewed(scene, src, 'experiment_complete', status='ambiguous'))
    assert result['status']==ambiguous['status']=='needs_attention'
    assert run_row(scene)['record_status']=='open'
    assert not result['snapshot']['whole_experiment_completed']


def test_text_hash_scope_or_source_mismatch_cannot_close_run(scene, settings):
    src=source(scene, settings)
    data=reviewed(scene, src, 'experiment_complete')
    for changed in (dict(text='篡改候选原文'),dict(source_text_sha256='0'*64),dict(camera_id=scene['cameras'][1])):
        with pytest.raises(Conflict):apply(scene, 'bad-'+str(changed), dict(data, **changed))
    assert run_row(scene)['record_status']=='open'
    assert not [r for r in rows(scene,'hj_activity') if 'comment_transition' in r['row']['payload']]


def test_late_whole_completion_conflicting_with_later_run_work_stays_attention(scene, settings):
    src=source(scene, settings)
    data=reviewed(scene, src, 'experiment_complete')
    scan(scene, 'later-equipment', run_id=scene['run'], when='09:03:00')
    result=apply(scene, 'late-whole', data)
    assert result['status']=='needs_attention'
    assert 'late_completion_conflicts_with_later_run_activity' in result['snapshot']['attention_reasons']
    assert run_row(scene)['record_status']=='open'


def test_unknown_audio_clock_never_inherits_current_context_or_closes_run(scene, settings):
    scan(scene, 'current-A', run_id=scene['run'])
    src=source(scene, settings, captured=False)
    data=reviewed(scene, src, 'experiment_complete')
    result=apply(scene, 'unknown-time', data)
    assert result['status']=='needs_attention' and not result['snapshot']['whole_experiment_completed']
    with scene['core'].db.transaction(readonly=True) as conn:
        event=conn.execute('SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s',(scene['core'].customer,UUID(result['activity_id']))).fetchone()
    assert event['start_at'] is None and event['instrument_id'] is None
    assert run_row(scene)['record_status']=='open'

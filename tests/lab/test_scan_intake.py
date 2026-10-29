"""Generated QR evidence across intake/worker/command/report, not field quality."""
from dataclasses import replace
import hashlib
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import qrcode

from visioncortex.lab.api import create_app
from visioncortex.lab.domain import Conflict, Missing
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import Worker
from visioncortex.lab.scan_intake import ScanIntake, verified_clock
from visioncortex.lab.timeline_runtime import TimelineRuntime

from .test_operator_scan import scene, context, rows, at

__all__ = ['scene']


def source(s, name, values, *, clock=True, declaration=None, when='09:00:01', camera='fixture-camera-0', process=True):
    settings = replace(s['runtime'].settings, processing_profile='timeline-v1')
    images = [qrcode.make(v).convert('RGB') for v in values]
    image = Image.new('RGB', (sum(p.width for p in images) or 400, 400), 'white')
    offset = 0
    for part in images:
        image.paste(part, (offset, 0))
        offset += part.width
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    path = settings.incoming / name / 'frame.png'
    path.parent.mkdir()
    path.write_bytes(buffer.getvalue())
    receipt = dict(capture_id=name, source_instance=camera, source_revision='1',
        camera_id=camera, kind='photo', session_id=name,
        relative_path=path.relative_to(settings.incoming).as_posix(), captured_at=at(when),
        source_kind='fixture', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        time_basis=dict(basis='generated_fixture_clock', uncertainty_us=0, clock_sync_verified=clock))
    if declaration is not None:
        receipt['scan_intent'] = declaration
    accepted = Ingest(s['core'], settings).accept(receipt)
    if process:
        completed = Worker(s['core'], settings).run_once()
        assert completed['status'] == 'completed'
        assert completed['job_id'] in accepted['job_ids']
    else:
        completed = dict(job_id=accepted['job_ids'][0], status='queued')
    return settings, completed, path


def consume(s, name, job, **changes):
    args = dict(job_id=job['job_id'], intent='operator_scan', scan_kind='equipment',
                actor='generated fixture scanner', reason='explicit scan fixture', person_id=s['people'][0])
    args.update(changes)
    return ScanIntake(s['core'], s['runtime'].settings).consume(name, **args)


def test_unique_pixel_uuid_to_existing_operator_context_api_and_report(scene):
    s = scene
    settings, job, _ = source(s, 'first', [s['devices'][0]])
    args = dict(key='api-scan', job_id=job['job_id'], intent='operator_scan', scan_kind='equipment',
                actor='fixture', reason='explicit fixture scan', person_id=s['people'][0], run_id=s['run'])
    with TestClient(create_app(s['core'], settings)) as client:
        url = '/internal/commands/timeline-consume-scan'
        assert client.post(url, json=args).status_code == 401
        response = client.post(url, json=args, headers={'Authorization': 'Bearer ' + settings.token})
        assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'declared_equipment'
    assert result['scan']['context']['equipment_id'] == s['devices'][0]
    assert result['scan']['context']['run_id'] == s['run']
    assert result['evidence']['source_camera_key'] == 'fixture-camera-0'
    report = TimelineRuntime(s['core'], settings).views.snapshot('2026-10-29')
    assert any(a['id'] == result['scan']['activity_id'] for e in report['document']['experiments'] for a in e['activities'])
    assert rows(s, 'quantity_ledger_entry') == []


def test_retry_with_new_command_key_reuses_same_physical_scan(scene):
    s = scene
    _, job, _ = source(s, 'retry', [s['devices'][0]])
    first = consume(s, 'caller-one', job, run_id=s['run'])
    second = consume(s, 'caller-two', job, run_id=s['run'])
    assert first == second and len(rows(s, 'hj_operator_scan_event')) == 1
    with pytest.raises(Conflict, match='Idempotency'):
        consume(s, 'caller-three', job, batch_id=s['batches'][1])


@pytest.mark.parametrize('values,status', [(None, 'no_qr_decoded'), ('multiple', 'ambiguous_multiple_qr')])
def test_failed_or_multiple_pixels_never_declare_operation(scene, values, status):
    s = scene
    _, job, _ = source(s, status, s['devices'] if values else [])
    result = consume(s, status, job)
    assert result['status'] == status and not result['operator_context_changed']
    assert rows(s, 'hj_operator_scan_event') == [] and context(s)['equipment_id'] is None


def test_background_visibility_keeps_evidence_without_operation(scene):
    s = scene
    _, job, _ = source(s, 'passive', [s['devices'][0]])
    result = consume(s, 'passive', job, intent='background_qr')
    assert result['status'] == 'passive_observation'
    assert rows(s, 'hj_operator_scan_event') == []
    assert not result['evidence']['equipment_activation_performed']


def test_unknown_camera_clock_does_not_inherit_current_equipment(scene):
    s = scene
    _, job, _ = source(s, 'unknown-clock', [s['devices'][0]], clock=False)
    result = consume(s, 'unknown-clock', job, run_id=s['run'])
    assert result['status'] == 'unknown_time'
    assert result['scan']['context']['equipment_id'] is None
    assert result['evidence']['source_reported_at'] is not None
    assert not result['evidence']['event_time_verified']


def test_changed_source_or_foreign_job_cannot_be_scanned(scene):
    s = scene
    _, job, path = source(s, 'changed', [s['devices'][0]])
    # Both the producer source and admitted cache are changed: neither can satisfy its SHA.
    path.write_bytes(b'changed')
    with s['core'].db.transaction(readonly=True) as conn:
        locations = conn.execute("SELECT relative_path FROM media_location WHERE customer_id=%s AND storage='cache'",
                                 (s['core'].customer,)).fetchall()
    for location in locations:
        (s['runtime'].settings.cache / location['relative_path']).write_bytes(b'changed')
    with pytest.raises(Conflict, match='unavailable|changed'):
        consume(s, 'changed', job)
    with pytest.raises(Missing):
        consume(s, 'foreign', {'job_id': s['devices'][0]})
    assert rows(s, 'hj_operator_scan_event') == []


def test_automatic_receipt_intent_consumes_completed_qr_once(scene):
    s = scene
    declaration = dict(intent='operator_scan', scan_kind='equipment', actor='fixture producer',
                       reason='explicit scan event', person_id=s['people'][0], run_id=s['run'])
    settings, _, _ = source(s, 'automatic', [s['devices'][1]], declaration=declaration)
    runtime = TimelineRuntime(s['core'], settings)
    assert runtime.tick()['scans']['consumed'] == 1
    assert runtime.tick()['scans']['consumed'] == 0
    assert context(s)['equipment_id'] == s['devices'][1]


def test_source_scans_keep_A_B_material_context_and_historical_time(scene):
    s = scene
    _, first, _ = source(s, 'device-A', [s['devices'][0]], when='09:00:01')
    a = consume(s, 'scan-A', first, run_id=s['run'])
    _, material, _ = source(s, 'material', ['fixture-reusable-label'], when='09:01:00')
    m = consume(s, 'scan-material', material, scan_kind='material')
    _, second, _ = source(s, 'device-B', [s['devices'][1]], when='09:02:00')
    b = consume(s, 'scan-B', second)
    assert a['scan']['context']['equipment_id'] == s['devices'][0]
    assert m['scan']['snapshot']['run_id'] == s['run']
    assert b['scan']['context']['equipment_id'] == s['devices'][1]
    assert b['scan']['context']['materials'][0]['resolution']['source_lot_id'] == s['lots'][0]
    assert context(s, '09:01:30')['equipment_id'] == s['devices'][0]
    assert not b['scan']['context']['switches'][-1]['ends_previous_experiment']
    assert rows(s, 'quantity_ledger_entry') == []


def test_malformed_receipt_does_not_prevent_next_valid_scan(scene):
    s = scene
    bad = dict(intent='operator_scan', scan_kind='equipment')
    settings, _, _ = source(s, 'malformed', [s['devices'][0]], declaration=bad)
    good = dict(bad, actor='fixture producer', reason='explicit scan event', person_id=s['people'][0])
    source(s, 'valid-after-malformed', [s['devices'][1]], declaration=good, when='09:00:02')
    outcomes = TimelineRuntime(s['core'], settings).scan_intake.tick()['outcomes']
    assert {x['status'] for x in outcomes} == {'declared_equipment', 'needs_attention'}
    assert context(s)['equipment_id'] == s['devices'][1]


def test_clock_requires_positive_verification_and_rejects_conflicting_flags():
    assert not verified_clock(dict(basis='filename', uncertainty_us=0))
    assert not verified_clock(dict(basis='fixture', uncertainty_us=False, clock_sync_verified=True))
    assert not verified_clock(dict(basis='fixture', uncertainty_us=0, clock_sync_verified=True, clock_sync_valid=False))
    assert verified_clock(dict(basis='fixture', uncertainty_us=0, clock_sync_verified=True))


def declaration(s, *, kind='equipment', person=0, **extra):
    return dict(intent='operator_scan', scan_kind=kind, actor='fixture producer',
                reason='explicit fixture scan', person_id=s['people'][person], **extra)


def test_new_backlog_material_consumes_earlier_target_before_freezing_snapshot(scene):
    s = scene
    settings, first, _ = source(s, 'backlog-A', [s['devices'][0]], when='09:00:00',
        declaration=declaration(s, run_id=s['run']))
    _, last, _ = source(s, 'backlog-M', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material'))
    intake = ScanIntake(s['core'], settings)
    result = intake.tick()
    assert [r['job_id'] for r in result['outcomes']] == [first['job_id'], last['job_id']]
    assert result['scheduling'] == 'latest_first_with_exact_scope_causal_dependencies'
    assert result['outcomes'][-1]['scan']['snapshot']['run_id'] == s['run']
    assert context(s)['materials'][0]['run_id'] == s['run']


def test_new_dependency_chain_crosses_limit_without_global_oldest_first(scene):
    s = scene
    settings, _, _ = source(s, 'chain-A', [s['devices'][0]], when='09:00:00',
        declaration=declaration(s, run_id=s['run']))
    source(s, 'chain-M1', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material', batch_id=s['batches'][1]))
    _, newest, _ = source(s, 'chain-M2', ['fixture-reusable-label'], when='09:02:00',
        declaration=declaration(s, kind='material'))
    intake = ScanIntake(s['core'], settings)
    for _ in range(2):
        result = intake.tick(limit=1)
        assert result['consumed'] == 1
        assert result['outcomes'][-1]['job_id'] == newest['job_id']
        assert result['outcomes'][-1]['status'] == 'deferred'
    final = intake.tick(limit=1)['outcomes'][0]
    assert final['job_id'] == newest['job_id']
    assert final['scan']['snapshot']['batch_id'] == s['batches'][1]
    assert final['scan']['snapshot']['run_id'] is None
    assert final['scan']['snapshot']['target_selection']['fixed_at_recording']


@pytest.mark.parametrize('scope', ['camera', 'person', 'unknown_clock', 'simultaneous'])
def test_new_dependency_scope_and_time_cannot_be_borrowed(scene, scope):
    s = scene
    settings, earlier, _ = source(s, 'scope-A', [s['devices'][0]],
        when='09:01:00' if scope == 'simultaneous' else '09:00:00',
        camera='fixture-camera-1' if scope == 'camera' else 'fixture-camera-0',
        clock=scope != 'unknown_clock', declaration=declaration(s,
            person=1 if scope == 'person' else 0, run_id=s['run']))
    _, newest, _ = source(s, 'scope-M', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material'))
    result = ScanIntake(s['core'], settings).tick(limit=1)
    assert result['outcomes'][0]['job_id'] == newest['job_id']
    assert result['outcomes'][0]['scan']['snapshot']['run_id'] is None
    assert result['outcomes'][0]['status'] == 'needs_material_context'
    assert all(r['job_id'] != earlier['job_id'] for r in result['outcomes'])


def test_new_latest_material_unknown_time_never_uses_verified_prior_target(scene):
    s = scene
    settings, _, _ = source(s, 'unknown-M-A', [s['devices'][0]], when='09:00:00',
        declaration=declaration(s, run_id=s['run']))
    _, newest, _ = source(s, 'unknown-M', ['fixture-reusable-label'], when='09:01:00', clock=False,
        declaration=declaration(s, kind='material'))
    result = ScanIntake(s['core'], settings).tick(limit=1)['outcomes'][0]
    assert result['job_id'] == newest['job_id'] and result['status'] == 'unknown_time'
    assert result['scan']['snapshot']['run_id'] is None
    assert not result['evidence']['event_time_verified']


def test_new_future_declaration_keeps_latest_first_but_cannot_change_older_material(scene):
    s = scene
    settings, old, _ = source(s, 'physical-old-A', [s['devices'][0]], when='09:00:00',
        declaration=declaration(s, run_id=s['run']))
    _, material, _ = source(s, 'physical-M', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material'))
    _, future, _ = source(s, 'physical-future-B', [s['devices'][1]], when='09:02:00',
        declaration=declaration(s, batch_id=s['batches'][1]))
    intake = ScanIntake(s['core'], settings)
    first = intake.tick(limit=2)
    assert [r['job_id'] for r in first['outcomes']] == [future['job_id'], old['job_id'], material['job_id']]
    assert first['outcomes'][-1]['status'] == 'deferred'
    result = intake.tick(limit=1)['outcomes'][0]
    assert result['scan']['snapshot']['run_id'] == s['run']
    assert result['scan']['snapshot']['batch_id'] == s['batches'][0]
    assert context(s)['batch_id'] == s['batches'][1]


@pytest.mark.parametrize('transient', [False, True])
def test_new_twenty_failed_newest_items_do_not_starve_twenty_first_valid(scene, transient, monkeypatch):
    s = scene
    settings, valid, _ = source(s, 'older-valid', [s['devices'][0]], when='09:00:00',
        declaration=declaration(s, run_id=s['run']))
    failed = set()
    for index in range(20):
        _, job, _ = source(s, 'newer-failed-' + str(index), [s['devices'][1]], when='09:01:%02d' % index,
            declaration=declaration(s) if transient else dict(intent='operator_scan', scan_kind='equipment'))
        failed.add(job['job_id'])
    intake = ScanIntake(s['core'], settings)
    original = intake.consume

    def intermittent(key, **body):
        if body['job_id'] in failed:
            raise Conflict('Temporary original media unavailable')
        return original(key, **body)
    if transient:
        monkeypatch.setattr(intake, 'consume', intermittent)
    first = intake.tick()
    assert len(first['outcomes']) == 20 and all(r['status'] == 'needs_attention' for r in first['outcomes'])
    assert all(bool(r['next_retry_at']) == transient for r in first['outcomes'])
    second = intake.tick()
    assert second['consumed'] == 1 and second['outcomes'][0]['job_id'] == valid['job_id']
    assert second['outcomes'][0]['status'] == 'declared_equipment'
    assert len(rows(s, 'hj_operator_scan_event')) == 1
    receipts = rows(s, 'command_receipt')
    failures = [r['row'] for r in receipts if r['row']['key'].startswith('scan-intake-failure:')]
    assert len(failures) == 20


def test_new_material_waits_for_temporarily_unavailable_causal_source(scene, monkeypatch):
    s = scene
    settings, first, _ = source(s, 'blocked-A', [s['devices'][0]], when='09:00:00',
        declaration=declaration(s, run_id=s['run']))
    _, latest, _ = source(s, 'blocked-M', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material'))
    intake = ScanIntake(s['core'], settings)
    original = intake.consume

    def unavailable(key, **body):
        if body['job_id'] == first['job_id']:
            raise Conflict('Temporary source unavailable')
        return original(key, **body)
    monkeypatch.setattr(intake, 'consume', unavailable)
    assert intake.tick()['outcomes'][-1]['status'] == 'deferred'
    again = intake.tick()
    assert again['consumed'] == 0
    assert again['outcomes'] == [dict(job_id=latest['job_id'], status='deferred',
        reason='earlier_exact_scope_declarations_pending', source_or_scan_snapshot_changed=False)]
    assert rows(s, 'hj_operator_scan_event') == []


def test_new_retry_receipts_are_bounded_and_require_explicit_resume_after_exhaustion(scene):
    s = scene
    settings, completed, _ = source(s, 'bounded-retry', [s['devices'][0]],
        declaration=declaration(s))
    intake = ScanIntake(s['core'], settings)
    job = dict(id=completed['job_id'], declaration=declaration(s), retry=None)
    # Exercise three distinct recorded attempts; tick's real deadline gate is
    # covered by the unavailable-source test without sleeping or editing clocks.
    for attempt in range(1, 4):
        result = intake._failure(job, Conflict('Temporary source unavailable'), permanent=False)
        assert result['attempt'] == attempt
        job['retry'] = result
    assert result['automatic_retry_exhausted'] and result['explicit_resume_required']
    assert result['next_retry_at'] is None and intake.tick()['consumed'] == 0
    assert len([r for r in rows(s, 'command_receipt')
                if r['row']['key'].startswith('scan-intake-failure:')]) == 3
    assert rows(s, 'hj_operator_scan_event') == []


def test_new_machine_empty_or_passive_record_never_claims_human_review(scene):
    s = scene
    _, job, _ = source(s, 'machine-only', [])
    result = consume(s, 'machine-only', job)
    activity = next(r['row'] for r in rows(s, 'hj_activity') if r['row']['id'] == result['activity_id'])
    assert activity['payload']['origin'] == 'saved_qr_machine_evidence'
    assert activity['payload']['human_review_performed'] is False
    assert activity['payload']['source_type_presentation'] == 'machine_qr_observation'


@pytest.mark.parametrize('scope', ['same', 'camera', 'person', 'unknown_clock'])
def test_new_unfinished_explicit_target_defers_only_exact_known_scope(scene, scope):
    s = scene
    settings, earlier, _ = source(s, 'queued-target', [s['devices'][0]], when='09:00:00', process=False,
        camera='fixture-camera-1' if scope == 'camera' else 'fixture-camera-0', clock=scope != 'unknown_clock',
        declaration=declaration(s, person=1 if scope == 'person' else 0, run_id=s['run']))
    _, material, _ = source(s, 'ready-material', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material'))
    intake = ScanIntake(s['core'], settings)
    result = intake.tick(limit=1)['outcomes'][0]
    assert result['job_id'] == material['job_id']
    if scope == 'same':
        assert result['status'] == 'deferred' and result['dependency_job_id'] == earlier['job_id']
        assert rows(s, 'hj_operator_scan_event') == []
        # Only this test's explicit worker call completes the queued fixture;
        # ScanIntake itself does not start or repeat decoding.
        assert Worker(s['core'], settings).run_once()['job_id'] == earlier['job_id']
        result = intake.tick()['outcomes'][-1]
        assert result['scan']['snapshot']['run_id'] == s['run']
    else:
        assert result['status'] == 'needs_material_context'
        assert result['scan']['snapshot']['run_id'] is None


def test_new_failed_explicit_target_requires_attention_without_freezing_material(scene):
    s = scene
    settings, earlier, _ = source(s, 'failed-target', [s['devices'][0]], when='09:00:00', process=False,
        declaration=declaration(s, run_id=s['run']))
    _, material, _ = source(s, 'material-after-failed-target', ['fixture-reusable-label'], when='09:01:00',
        declaration=declaration(s, kind='material'))
    with s['core'].db.transaction() as conn:
        conn.execute("UPDATE processing_job SET status='failed',attempts=3,error_code='fixture_decode_failure' WHERE customer_id=%s AND id=%s",
                     (s['core'].customer, earlier['job_id']))
    intake = ScanIntake(s['core'], settings)
    result = intake.tick()['outcomes'][0]
    assert result['job_id'] == material['job_id'] and result['status'] == 'needs_attention'
    assert result['causal_dependency_failed'] and result['explicit_resume_required']
    assert not result['permanent_declaration_error']
    assert intake.tick()['consumed'] == 0 and rows(s, 'hj_operator_scan_event') == []

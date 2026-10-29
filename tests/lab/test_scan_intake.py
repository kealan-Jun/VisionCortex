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


def source(s, name, values, *, clock=True, declaration=None, when='09:00:01'):
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
    receipt = dict(capture_id=name, source_instance='fixture-camera-0', source_revision='1',
        camera_id='fixture-camera-0', kind='photo', session_id=name,
        relative_path=path.relative_to(settings.incoming).as_posix(), captured_at=at(when),
        source_kind='fixture', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        time_basis=dict(basis='generated_fixture_clock', uncertainty_us=0, clock_sync_verified=clock))
    if declaration is not None:
        receipt['scan_intent'] = declaration
    accepted = Ingest(s['core'], settings).accept(receipt)
    completed = Worker(s['core'], settings).run_once()
    assert completed['status'] == 'completed'
    assert completed['job_id'] in accepted['job_ids']
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

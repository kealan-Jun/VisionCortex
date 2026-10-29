"""Natural-camera source is opt-in; all cameras/QRs below are generated fixtures.

No model, physical camera, legacy SQLite binding or customer activation is used.
"""
import asyncio
from dataclasses import replace
import hashlib
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import cv2
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from field_recognition.capture.qr_decode import decode_qr
from visioncortex.lab.domain import Conflict
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.operator_scan import CAMERA_INTENT
from visioncortex.lab.processing import Worker
from visioncortex.lab.scan_source import CAMERA_SCHEMA, ENV, ScanSourcePublisher
from visioncortex.lab.scan_source_scanner import CameraEquipmentObserver, attach_to_runtime, app_factory
from visioncortex.lab.timeline_runtime import TimelineRuntime

from .test_operator_scan import at, rows, scene
from .test_scan_source import configuration as manual_configuration, lossless_codec, metadata, pixels, receipts
from .test_scan_source_launcher import launcher

__all__ = ['scene', 'launcher']


def configuration(root, cameras=('fixture-camera-0',)):
    config = manual_configuration(root, cameras)
    config.update(schema_version=CAMERA_SCHEMA, mode='camera_equipment', poll_interval_ms=50,
                  presence_interval_seconds=15)
    return config


def publisher_frame(publisher, value, index, *, camera='fixture-camera-0', clock=True, epoch='package-1'):
    image = pixels(value)
    meta = dict(metadata(index, camera=camera, clock=clock), media_session_id=epoch, device_boot_id='fixture-boot')
    return publisher.publish(image, meta, None, decode_qr(image))


def test_new_mode_without_manual_owner_session_and_presence_cross_package(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / 'runtime'))
    value = str(uuid4())
    assert publisher_frame(publisher, [value], 1)['published']
    assert publisher_frame(publisher, [value], 1)['status'] == 'duplicate'
    assert not publisher_frame(publisher, [value], 2)['published']
    assert publisher_frame(publisher, [value], 16)['published']
    assert publisher_frame(publisher, [value], 901, epoch='package-2')['published']
    saved = receipts(publisher.root)
    assert len(saved) == 3
    assert {r['scan_intent']['intent'] for r in saved} == {CAMERA_INTENT}
    assert len({r['session_id'] for r in saved}) == 2  # source epochs, not business experiment IDs
    for receipt in saved:
        source = receipt['scan_source']
        assert source['session_owner'] is source['operator_label_unverified'] is source['person_id'] is None
        assert source['manual_session_required'] is False
        assert source['visible_QR_is_not_panel_geometry'] and not source['experiment_end_implied']
    bad = metadata(20)
    del bad['decoded_frame_id']
    with pytest.raises(ValueError, match='stable native frame'):
        publisher.publish(pixels([value]), bad, None, ([value], None, {}))


def test_thin_observer_original_decoder_pixels_and_native_frame_identity(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / 'runtime'))
    value = str(uuid4())
    frame = pixels([value])
    camera = SimpleNamespace(target='fixture-camera-0', frame=lambda: (frame, metadata()))
    observer = CameraEquipmentObserver(camera, publisher)
    assert observer.worker is None and not receipts(publisher.root)
    first = observer.step()
    assert first['published'] and observer.step()['status'] == 'same_native_frame'
    receipt = receipts(publisher.root)[0]
    assert receipt['scan_source']['decoded_values'] == [value]
    assert receipt['scan_source']['original_rgb_sha256'] == hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
    assert observer.frames == 1 and observer.worker is None


def test_new_clock_recovery_does_not_suppress_first_verified_same_code(tmp_path):
    publisher = ScanSourcePublisher(configuration(tmp_path / 'runtime'))
    value = str(uuid4())
    assert publisher_frame(publisher, [value], 1, clock=False)['published']
    assert publisher_frame(publisher, [value], 2, clock=True)['published']
    assert {r['time_basis']['clock_sync_valid'] for r in receipts(publisher.root)} == {False, True}


@pytest.mark.parametrize('age', [None, -1, float('nan'), 1001])
def test_observer_stale_or_unknown_age_never_publishes(tmp_path, age):
    publisher = ScanSourcePublisher(configuration(tmp_path / 'runtime'))
    camera = SimpleNamespace(target='fixture-camera-0', frame=lambda: (pixels([str(uuid4())]), dict(metadata(), frame_age_ms=age)))
    observer = CameraEquipmentObserver(camera, publisher)
    assert not observer.step()['published']
    assert not receipts(publisher.root)


class FakeRuntime(dict):
    def close(self):
        self['calls'].append('close')


def fake_runtime():
    calls = []
    camera = SimpleNamespace(target='fixture-camera-0', start=lambda: calls.append('camera.start'))
    return FakeRuntime(calls=calls, receiver_camera=camera, RUNTIME_ENABLED=False,
                       SERVICE_ROLE='capture', stopping=threading.Event(),
                       live_scanner=SimpleNamespace(session=None),
                       db=lambda: (_ for _ in ()).throw(AssertionError('legacy SQLite binding touched')))


def test_new_lifecycle_starts_only_receiver_observer_and_does_not_call_legacy_binding(tmp_path, monkeypatch):
    manifest = tmp_path / 'source.json'
    manifest.write_text(json.dumps(configuration(tmp_path / 'runtime')))
    monkeypatch.setenv(ENV, str(manifest))
    runtime = fake_runtime()
    attach_to_runtime(runtime)
    observer = runtime['camera_equipment_observer']
    assert runtime['calls'] == [] and observer.worker is None
    monkeypatch.setattr(observer, 'start', lambda: runtime['calls'].append('observer.start'))
    monkeypatch.setattr(observer, 'close', lambda: runtime['calls'].append('observer.close'))

    async def exercise():
        async with runtime['lifespan'](None):
            runtime['receiver_camera'].on_service_status(dict(online=False))
            assert observer.service_status == dict(online=False)
            assert runtime['calls'] == ['camera.start', 'observer.start']
    asyncio.run(exercise())
    assert runtime['calls'] == ['camera.start', 'observer.start', 'observer.close', 'close']
    assert runtime['stopping'].is_set()


def test_source_only_http_does_not_expose_legacy_binding_or_state_sweep(launcher, tmp_path, monkeypatch):
    from visioncortex.lab import scan_source_scanner
    from field_recognition.api import application

    document, manifest = launcher
    document.update(source_enabled=True, field_config=dict(receiver_url='http://fixture.invalid/never-connect', camera_id='fixture-camera-0'))
    manifest.write_text(json.dumps(document))
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(configuration(tmp_path / 'publisher-runtime')))
    monkeypatch.setenv(ENV, str(source))
    runtime = fake_runtime()
    monkeypatch.setattr(scan_source_scanner, '_resource_factory', lambda assets: lambda *a, **kw: runtime)

    def create_app(runtime):
        app = FastAPI(lifespan=runtime['lifespan'])

        @app.get('/api/state')
        def forbidden_state():
            raise AssertionError('Original dashboard sweep must not run')

        @app.post('/api/bindings')
        def forbidden_binding():
            raise AssertionError('Original binder must not run')

        @app.get('/health/ready')
        def ready():
            return dict(status='ready')
        return app
    monkeypatch.setattr(application, 'create_app', create_app)
    app = app_factory()
    observer = runtime['camera_equipment_observer']
    monkeypatch.setattr(observer, 'start', lambda: None)
    with TestClient(app) as client:
        assert client.post('/api/bindings', json={}).status_code == 409
        assert client.get('/api/state').status_code == 409
        assert client.get('/health/ready').json()['status'] == 'ready'
    assert runtime['calls'] == ['camera.start', 'close']


def drain(s, settings, count):
    assert Ingest(s['core'], settings).discover_once()['processed'] == count
    completed = [Worker(s['core'], settings).run_once() for _ in range(count)]
    assert all(j['status'] == 'completed' for j in completed)
    runtime = TimelineRuntime(s['core'], settings)
    result = runtime.scan_intake.tick(limit=100)
    assert result['consumed'] == count
    assert runtime.scan_intake.tick()['consumed'] == 0
    return runtime, result['outcomes']


def test_new_native_codec_auto_observe_queue_qr_operation_cross_package_two_cameras(scene, tmp_path):
    s = scene
    settings = replace(s['runtime'].settings, processing_profile='timeline-v1')
    publisher = ScanSourcePublisher(configuration(settings.runtime_root, ('fixture-camera-0', 'fixture-camera-1')))
    a, b = s['devices']
    values = [a, a, b, a, b]
    originals = [pixels([v]) for v in values]
    decoded = lossless_codec(tmp_path / 'generated-natural-camera.mkv', originals)
    for image, index, camera, package in zip(decoded, [1, 901, 902, 903, 1],
            ['fixture-camera-0'] * 4 + ['fixture-camera-1'], ['first', 'second', 'second', 'second', 'first'], strict=True):
        port = SimpleNamespace(target=camera, frame=lambda p=image, n=index, c=camera, e=package:
            (p, dict(metadata(n, camera=c), media_session_id=e, device_boot_id='fixture-boot')))
        assert CameraEquipmentObserver(port, publisher).step()['published']
    runtime, outcomes = drain(s, settings, 5)
    assert all(o['status'] == 'declared_equipment' for o in outcomes)
    current = runtime.operator_scans.context(as_of=at('09:30:00'), camera_id=s['cameras'][0])
    other = runtime.operator_scans.context(as_of=at('09:30:00'), camera_id=s['cameras'][1])
    assert current['equipment_id'] == a and other['equipment_id'] == b
    assert len(current['switches']) == 3 and len(current['equipment_presences']) == 4
    assert [v['to_equipment_id'] for v in current['switches']] == [a, b, a]
    assert all(not v['ends_previous_experiment'] for v in current['switches'])
    assert current['run_id'] is current['batch_id'] is current['person_id'] is None
    assert not current['experiment_end_implied']
    snapshot = runtime.views.snapshot('2026-10-29')
    events = [r['row'] for r in rows(s, 'hj_operator_scan_event')]
    assert len(events) == 5
    assert all(e['payload']['authority'] == 'configured_camera_equipment_presence_not_manual_scan' for e in events)
    assert all(e['payload']['source_qr_job_id'] and not e['payload']['human_review_performed'] for e in events)
    assert rows(s, 'quantity_ledger_entry') == rows(s, 'fact_revision') == []
    with s['core'].db.transaction(readonly=True) as conn:
        sources = conn.execute('SELECT payload FROM inbox WHERE customer_id=%s', (s['core'].customer,)).fetchall()
    proof = dict(fixture_only=True, cameras_are_synthetic=True, source_receipts=[x['payload'] for x in sources],
        codec='lossless_ffv1_bgr0', native_source_rgb_sha256=[hashlib.sha256(cv2.cvtColor(p, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() for p in originals],
        outcomes=outcomes, context=current, second_camera_context=other, snapshot=snapshot,
        model_calls=0, legacy_sqlite_binding_calls=0, fact_count=0, ledger_count=0)
    (tmp_path / 'NaturalCameraChain.json').write_text(json.dumps(proof, indent=2) + '\n')


def test_new_natural_unknown_material_reserved_multiple_empty_are_observations_only(scene):
    s = scene
    settings = replace(s['runtime'].settings, processing_profile='timeline-v1')
    publisher = ScanSourcePublisher(configuration(settings.runtime_root))
    reserved = s['core'].issue_equipment('natural-reserved', actor='synthetic fixture')['device_id']
    cases = [([s['devices'][0]], False), (['fixture-reusable-label'], True),
             ([reserved], True), (s['devices'], True), ([], True), ([str(uuid4())], True)]
    for index, (values, clock) in enumerate(cases, 1):
        assert publisher_frame(publisher, values, index, clock=clock)['published']
    runtime, outcomes = drain(s, settings, len(cases))
    assert {o['status'] for o in outcomes} == {'unknown_time', 'wrong_identity_kind',
        'equipment_not_active_at_event', 'ambiguous_multiple_qr', 'no_qr_decoded', 'unregistered_label'}
    assert all(not o['operator_context_changed'] for o in outcomes)
    current = runtime.operator_scans.context(as_of=at('09:30:00'), camera_id=s['cameras'][0])
    assert current['equipment_id'] is None
    assert rows(s, 'hj_operator_scan_event') == []
    assert rows(s, 'quantity_ledger_entry') == rows(s, 'fact_revision') == []
    saved = receipts(publisher.root)
    assert next(r for r in saved if r['time_basis']['clock_sync_valid'] is False)['captured_at'] is None


def test_new_camera_authority_cannot_be_claimed_without_persisted_source(scene):
    s = scene
    with pytest.raises(Conflict, match='saved QR source'):
        s['ops'].record('forged-natural', intent=CAMERA_INTENT, scan_kind='equipment',
            qr_value=s['devices'][0], occurred_at=at('09:00:01'), evidence_ids=[], camera_id=s['cameras'][0],
            actor='fixture', reason='intent alone must not be authority')
    with pytest.raises(ValueError, match='no inferred person'):
        s['ops'].record('forged-person', intent=CAMERA_INTENT, scan_kind='equipment',
            qr_value=s['devices'][0], occurred_at=at('09:00:01'), evidence_ids=[],
            camera_id=s['cameras'][0], person_id=s['people'][0], actor='fixture', reason='not permitted')
    assert rows(s, 'hj_operator_scan_event') == []


def test_new_camera_source_retry_same_job_once_late_event_replayed_by_event_time(scene):
    s = scene
    settings = replace(s['runtime'].settings, processing_profile='timeline-v1')
    publisher = ScanSourcePublisher(configuration(settings.runtime_root))
    for value, index in [(s['devices'][0], 1), (s['devices'][1], 20), (s['devices'][0], 15)]:
        assert publisher_frame(publisher, [value], index)['published']
    runtime, outcomes = drain(s, settings, 3)
    first = outcomes[0]
    declaration = receipts(publisher.root)[0]['scan_intent']
    repeated = runtime.scan_intake.consume('new-caller-same-job', job_id=first['job_id'], **declaration)
    assert repeated == {k: v for k, v in first.items() if k != 'job_id'}
    assert len(rows(s, 'hj_operator_scan_event')) == 3
    current = runtime.operator_scans.context(as_of=at('09:00:30'), camera_id=s['cameras'][0])
    assert current['equipment_id'] == s['devices'][1]
    assert [v['to_equipment_id'] for v in current['switches']] == s['devices']
    assert len(current['equipment_presences']) == 3


def test_new_camera_intent_rejects_v1_manual_receipt_as_machine_authority(scene):
    from .test_scan_intake import source

    s = scene
    declaration = dict(intent=CAMERA_INTENT, scan_kind='equipment', actor='fixture', reason='not a v2 source')
    settings, job, _ = source(s, 'manual-forged-as-natural', [s['devices'][0]], declaration=declaration)
    runtime = TimelineRuntime(s['core'], settings)
    with pytest.raises(Conflict, match='source receipt'):
        runtime.scan_intake.consume('forged-manual', job_id=job['job_id'], **declaration)
    assert rows(s, 'hj_operator_scan_event') == []

"""Generic incoming protocol; generated media only, no camera/model services."""
from dataclasses import replace
from decimal import Decimal
import hashlib
from io import BytesIO
import json
import os
import subprocess
from uuid import uuid4

import cv2
import numpy as np
import pytest
import qrcode

from visioncortex.lab.capture_equipment_intake import AUTH_PREFIX, BoundedQRContext, CaptureEquipmentIntake, newest_candidates, partition
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.processing import Worker
from visioncortex.lab.timeline_runtime import TimelineRuntime

from .test_operator_scan import at, rows, scene
from .test_scan_intake import source

__all__ = ['scene']


def config(**changes):
    result = dict(enabled=True, revision='generated-generic-capture/1', photo=True, video=True,
        video_window_seconds=10, max_packet_seconds=900, max_packet_tasks=90,
        max_admissions_per_tick=20, max_observations_per_tick=600,
        rotation_degrees_clockwise=0, orientation_basis='generated original native upright pixels',
        source_kinds=['fixture'], presence_interval_seconds=60)
    result.update(changes)
    return result


def enable(s, tmp_path, **changes):
    path = tmp_path / 'generic-policy.json'
    path.write_text(json.dumps(dict(schema_version='vc-lab-timeline-policy/1', automatic_equipment_qr=config(**changes))))
    settings = replace(s['runtime'].settings, processing_profile='timeline-v1', timeline_policy_path=path)
    s['runtime'] = TimelineRuntime(s['core'], settings)
    return settings


def authorization(s, media_id):
    with s['core'].db.transaction(readonly=True) as conn:
        return conn.execute('SELECT result FROM command_receipt WHERE customer_id=%s AND key=%s',
            (s['core'].customer, AUTH_PREFIX + str(media_id))).fetchone()['result']


def test_new_partition_15_minute_packet_exactly_90_nonoverlapping_tasks():
    plan = partition(Decimal(900), config())
    assert len(plan) == 90 and plan[0] == dict(start_seconds='0', end_seconds='10')
    assert plan[-1] == dict(start_seconds='890', end_seconds='900')
    assert all(a['end_seconds'] == b['start_seconds'] for a, b in zip(plan, plan[1:], strict=False))
    assert sum(Decimal(p['end_seconds']) - Decimal(p['start_seconds']) for p in plan) == 900
    with pytest.raises(ValueError):
        partition(Decimal('900.000001'), config())


def test_new_day_desc_discovery_visits_recent_before_old_archive_prefix(tmp_path):
    for day in ['2020-01-01', '2026-10-29']:
        root = tmp_path / (day + '_fixture-any-camera') / 'MetaData' / '09-00-00'
        root.mkdir(parents=True)
        (root / (day.replace('-', '') + '_090000.jpg')).write_bytes(b'path-only fixture')
    found, partial = newest_candidates(tmp_path, set(), max_items=1)
    assert partial and len(found) == 1 and found[0].parts[-4].startswith('2026-10-29_')


def test_new_received_photo_without_manual_intent_registers_only_logical_camera_and_consumes(scene, tmp_path):
    s = scene
    settings = enable(s, tmp_path)
    camera = 'fixture-unlisted-' + str(uuid4())
    _, job, path = source(s, 'ordinary-photo', [s['devices'][0]], camera=camera)
    outcome = s['runtime'].scan_intake.tick()['automatic_equipment_qr']
    assert outcome['processed'] == 1 and outcome['photo_outcomes'][0]['status'] == 'declared_equipment'
    scan = outcome['photo_outcomes'][0]['scan']
    assert scan['snapshot']['person_id'] is None
    assert scan['snapshot']['source_authority'] == 'server_pinned_automatic_equipment_qr_policy'
    assert scan['snapshot']['source_qr_job_id'] == job['job_id']
    with s['core'].db.transaction(readonly=True) as conn:
        record = conn.execute('SELECT i.payload,a.id FROM inbox i JOIN capture c ON c.customer_id=i.customer_id AND c.source_id=i.source_id AND c.source_capture_id=i.event_id AND c.source_revision=i.revision JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id WHERE i.customer_id=%s', (s['core'].customer,)).fetchone()
        logical = conn.execute("SELECT metadata FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s", (s['core'].customer, camera)).fetchone()
    assert 'scan_intent' not in record['payload']
    assert not logical['metadata']['physical_hardware_pairing_verified']
    assert authorization(s, record['id'])['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert s['runtime'].scan_intake.tick()['automatic_equipment_qr']['processed'] == 0
    accepted = Ingest(s['core'], settings).accept(record['payload'])
    assert accepted['duplicate']
    assert len(rows(s, 'processing_job')) == len(rows(s, 'hj_operator_scan_event')) == 1
    assert rows(s, 'fact_revision') == rows(s, 'quantity_ledger_entry') == []


def test_new_default_off_keeps_received_photo_passive_until_explicit_server_policy(scene, tmp_path):
    s = scene
    settings, job, _ = source(s, 'previously-received', [s['devices'][0]])
    runtime = TimelineRuntime(s['core'], settings)
    assert not runtime.scan_intake.tick()['automatic_equipment_qr']['enabled']
    assert rows(s, 'hj_operator_scan_event') == []
    enable(s, tmp_path)
    result = s['runtime'].scan_intake.tick()['automatic_equipment_qr']
    assert len(result['admissions']) == 1 and result['photo_outcomes'][0]['status'] == 'declared_equipment'
    assert result['photo_outcomes'][0]['job_id'] == job['job_id']
    assert len(rows(s, 'processing_job')) == 1  # completed QR reused, not rerun


@pytest.mark.parametrize('case,expected', [('unknown_clock', 'unknown_time'), ('material', 'wrong_identity_kind'),
    ('multiple', 'ambiguous_multiple_qr'), ('empty', 'no_qr_decoded'), ('reserved', 'equipment_not_active_at_event')])
def test_new_generic_photo_candidates_do_not_create_operation_or_quantities(scene, tmp_path, case, expected):
    s = scene
    enable(s, tmp_path)
    value = s['devices'] if case == 'multiple' else [] if case == 'empty' else ['fixture-reusable-label'] if case == 'material' else [s['core'].issue_equipment('unused', actor='fixture')['device_id']] if case == 'reserved' else [s['devices'][0]]
    source(s, case, value, clock=case != 'unknown_clock')
    outcome = s['runtime'].scan_intake.tick()['automatic_equipment_qr']['photo_outcomes'][0]
    assert outcome['status'] == expected and not outcome['operator_context_changed']
    assert rows(s, 'hj_operator_scan_event') == rows(s, 'quantity_ledger_entry') == rows(s, 'fact_revision') == []


def test_new_native_receipt_without_declared_sha_uses_verified_bytes_not_fabricated_hash(scene, tmp_path):
    s = scene
    settings = enable(s, tmp_path)
    image = qrcode.make(s['devices'][0]).convert('RGB')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    path = settings.incoming / 'ordinary.png'
    path.write_bytes(buffer.getvalue())
    receipt = dict(capture_id='ordinary-no-declared-sha', source_instance='generic-input', source_revision='1',
        camera_id='generic-new-camera', kind='photo', relative_path=path.name, captured_at=at('09:00:00'),
        source_kind='fixture', time_basis=dict(basis='explicit generated fixture source clock', uncertainty_us=0, clock_sync_valid=True))
    admitted = Ingest(s['core'], settings).accept(receipt)
    assert not authorization(s, admitted['media_id'])['declared_sha256_present']
    assert Worker(s['core'], settings).run_once()['status'] == 'completed'
    outcome = s['runtime'].scan_intake.tick()['automatic_equipment_qr']['photo_outcomes'][0]
    assert outcome['status'] == 'declared_equipment'
    assert not outcome['scan']['snapshot']['source_evidence']['declared_source_sha_present']


def test_new_bare_photos_any_camera_no_three_shot_requirement_keep_unknown_clock(scene, tmp_path):
    s = scene
    settings = enable(s, tmp_path)
    camera = 'fixture-new-bare-camera'
    folder = settings.incoming / ('2026-10-29_' + camera) / 'MetaData' / '09-00-00'
    folder.mkdir(parents=True)
    # Four members cannot be assigned an invented original three-shot burst.
    for i in range(4):
        path = folder / ('20261029_090000_' + str(i) + '.jpg')
        qrcode.make(s['devices'][0]).convert('RGB').save(path)
        os.utime(path, (1, 1))
    result = Ingest(s['core'], settings).discover_once()
    assert result['processed'] == 4 and not result['errors']
    assert len(rows(s, 'capture')) == 4 and len(rows(s, 'processing_job')) == 4
    assert all(Worker(s['core'], settings).run_once()['status'] == 'completed' for _ in range(4))
    outcomes = s['runtime'].scan_intake.tick()['automatic_equipment_qr']['photo_outcomes']
    assert len(outcomes) == 4 and all(o['status'] == 'unknown_time' for o in outcomes)
    assert rows(s, 'hj_operator_scan_event') == []
    receipts = [r['row']['payload'] for r in rows(s, 'inbox')]
    assert all(p['session_id'] is None and p['group_expected_photos'] is None for p in receipts)
    assert all(p['time_basis']['basis'] == 'nas_filename_local_time_not_hardware_verified' for p in receipts)


def native_video(s, settings, values, *, session=True, clock=True):
    parts = []
    for value in values:
        qr = np.asarray(qrcode.make(value).convert('RGB'))[:, :, ::-1].copy()
        image = np.full((400, max(400, qr.shape[1] + 40), 3), 230, np.uint8)
        image[20:20 + qr.shape[0], 20:20 + qr.shape[1]] = qr
        parts.append(image)
    path = settings.incoming / 'MetaData' / 'generic-native-video' / 'rgb.mp4'
    path.parent.mkdir(parents=True)
    h, w = parts[0].shape[:2]
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}', '-r', '2', '-i', 'pipe:0', '-c:v', 'libx264rgb', '-crf', '0',
        '-preset', 'ultrafast', '-threads', '1', str(path)], input=b''.join(p.tobytes() for p in parts), check=True, timeout=10)
    receipt = dict(capture_id='native-arbitrary-device-packet', source_instance='native:any-generic-source',
        source_revision='1', camera_id='generic-protocol-camera', kind='video',
        relative_path=path.relative_to(settings.incoming).as_posix(), captured_at=at('09:00:00'),
        source_kind='fixture', time_basis=dict(basis='generated verified video clock', uncertainty_us=0, clock_sync_valid=clock),
        native_record=dict(fixture_only=True), start_seconds='0', end_seconds=str(Decimal(len(parts)) / 2))
    if session:
        receipt['session_id'] = 'native-source-session'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    admitted = Ingest(s['core'], settings).accept(receipt, native_original=True)
    return path, before, receipt, admitted, parts


def test_new_generic_video_all_frames_automatic_task_db_presence_report_no_manual_registration(scene, tmp_path, monkeypatch):
    s = scene
    settings = enable(s, tmp_path, video_window_seconds=1, max_packet_seconds=90)
    path, sha, receipt, accepted, parts = native_video(s, settings,
        [s['devices'][0]] * 2 + [s['devices'][1]] * 2 + [s['devices'][0]] * 2, session=False)
    auth = authorization(s, accepted['media_id'])
    assert auth['status'] == 'authorized' and len(auth['task_ids']) == 3
    assert auth['state_namespace_basis'] == 'technical_namespace_not_physical_session'
    assert auth['original_source_session_key'] is None
    assert all(float(i['end_seconds']) - float(i['start_seconds']) <= 1 for i in auth['source_intervals'])
    from .test_video_qr import forbid_provider
    forbid_provider(monkeypatch)
    assert all(s['runtime'].runner.run_once()['status'] == 'completed' for _ in range(3))
    deferred = CaptureEquipmentIntake(s['core'], settings)._consume_video(auth['task_ids'][-1], 600)
    assert deferred['status'] == 'deferred' and deferred['processed'] == 0
    outcome = s['runtime'].scan_intake.tick()['automatic_equipment_qr']
    assert outcome['processed'] == 6
    progress = CaptureEquipmentIntake(s['core'], settings).packet_progress(accepted['media_id'])
    assert progress['whole_packet_frame_coverage'] == 'PROVEN'
    assert progress['actual_decoded_and_consumed_frames'] == progress['independently_probed_container_frame_count'] == 6
    assert len(rows(s, 'hj_dynamic_frame_event')) == 6
    assert len(rows(s, 'hj_operator_scan_event')) == 3
    with s['core'].db.transaction(readonly=True) as conn:
        camera = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND code='generic-protocol-camera' AND kind='capture_device'", (s['core'].customer,)).fetchone()['id']
    context = s['runtime'].operator_scans.context(as_of=at('09:05:00'), camera_id=str(camera))
    assert [p['to_equipment_id'] for p in context['switches']] == [s['devices'][0], s['devices'][1], s['devices'][0]]
    assert context['person_id'] is None and not context['experiment_end_implied']
    snapshot = s['runtime'].views.snapshot('2026-10-29')
    assert s['devices'][0] in str(snapshot)
    duplicate = Ingest(s['core'], settings).accept(receipt, native_original=True)
    assert duplicate['duplicate'] and len([r for r in rows(s, 'hj_timeline_task') if r['row']['kind'] == 'qr']) == 3
    assert s['runtime'].scan_intake.tick()['automatic_equipment_qr']['processed'] == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    assert rows(s, 'fact_revision') == rows(s, 'quantity_ledger_entry') == []
    proof = dict(fixture_only=True, actual_video_sha256=sha, policy_authorization=auth,
        input_rgb_sha256=[hashlib.sha256(cv2.cvtColor(p, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() for p in parts],
        result=outcome, packet_progress=progress, operator_context=context, report=snapshot,
        original_video_unchanged=True, model_calls=0)
    (tmp_path / 'GenericNativeChain.json').write_text(json.dumps(proof, indent=2) + '\n')


def test_new_bounded_raw_context_exceeds_old_total_media_limit_without_raising_research_limit(scene, tmp_path):
    # Storage-bound counterexample: seed >2048 prior-window row-count fixtures,
    # then append an actual new-window frame through the real validator/SQL path.
    # The seeded rows are not pixel evidence and are never read as such. This is
    # a generated DB counterexample, not a claim that a real 15min codec ran.
    from visioncortex.lab.dynamic_context import DynamicPolicy
    from visioncortex.lab.dynamic_video import DynamicVideo

    s = scene
    settings = enable(s, tmp_path)
    path, _, _, accepted, _ = native_video(s, settings, [s['devices'][0]] * 22)
    auth = authorization(s, accepted['media_id'])
    with s['core'].db.transaction(readonly=True) as conn:
        registration = conn.execute('SELECT request FROM hj_timeline_task WHERE customer_id=%s AND id=%s',
            (s['core'].customer, auth['task_ids'][1])).fetchone()['request']['video_qr']
    ref = dict(media_id=accepted['media_id'], sha256=auth['source_sha256'], media_version=1, start_seconds='10', end_seconds='11')
    context = BoundedQRContext(s['core'], ref, registration)
    assert DynamicPolicy().max_events == 2048
    with s['core'].db.transaction() as conn:
        conn.execute('''INSERT INTO hj_dynamic_frame_event(customer_id,id,media_id,source_session_key,frame_key,source_seconds,payload,payload_sha256)
            SELECT %s,gen_random_uuid(),%s,%s,'prior-window:'||n::text,n/1000.0,'{}','seed-bounded-counterexample'
            FROM generate_series(1,2049) n''', (s['core'].customer, accepted['media_id'], auth['source_session_key']))
    # Actual new raw pixels go through original QR extraction and source checks.
    image = np.full((192, 192, 3), 128, np.uint8)
    producer = DynamicVideo(context, accepted['media_id'], auth['source_session_key'],
        dict(sha256=auth['source_sha256'], media_version=1, start_seconds='10', end_seconds='11'),
        dict(schema_version='vc-lab-dynamic-video/1', enabled=True, revision='new-raw-window', max_frames=600,
             batch_frames=1, max_buffer_bytes=32*1024**2, max_seconds=120, panels=[]))
    meta = dict(pts=10000, time_base='1/1000', stream_origin='0', media_offset='10',
        source_sha256=auth['source_sha256'], original_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        source_to_upright=[[1,0,0],[0,1,0],[0,0,1]])
    assert producer.observe(image, meta)['status'] == 'committed'
    assert len(rows(s, 'hj_dynamic_frame_event')) == 2050
    assert rows(s, 'hj_dynamic_context_revision') == []
    assert producer.receipt()['confirmed_operations'] == 0


def test_new_generic_video_unknown_clock_retains_all_pts_without_operation(scene, tmp_path, monkeypatch):
    from .test_video_qr import forbid_provider

    s = scene
    settings = enable(s, tmp_path)
    _, _, _, accepted, _ = native_video(s, settings, [s['devices'][0]] * 2, clock=False)
    forbid_provider(monkeypatch)
    assert s['runtime'].runner.run_once()['status'] == 'completed'
    result = s['runtime'].scan_intake.tick()['automatic_equipment_qr']
    assert result['processed'] == 2
    assert result['video_outcomes'][0]['status_counts'] == {'unknown_time': 2}
    assert all(r['row']['event_at'] is None for r in rows(s, 'hj_dynamic_frame_event'))
    assert rows(s, 'hj_operator_scan_event') == []
    assert CaptureEquipmentIntake(s['core'], settings).packet_progress(accepted['media_id'])['whole_packet_frame_coverage'] == 'PROVEN'


def test_new_probe_failure_retains_capture_and_does_not_queue_partial_plan(scene, tmp_path):
    s = scene
    settings = enable(s, tmp_path)
    path = settings.incoming / 'MetaData' / 'unreadable' / 'rgb.mp4'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'not a video - generated rejection fixture')
    receipt = dict(capture_id='bad-video', source_instance='arbitrary-input', source_revision='1', camera_id='bad-video-camera',
        kind='video', relative_path=path.relative_to(settings.incoming).as_posix(), source_kind='fixture',
        time_basis=dict(basis='unknown'), end_seconds='900', native_record=dict(fixture_only=True))
    result = Ingest(s['core'], settings).accept(receipt, native_original=True)
    assert result['automatic_equipment_qr']['status'] == 'needs_attention'
    assert len(rows(s, 'capture')) == 1
    assert not any(r['row']['kind'] == 'qr' for r in rows(s, 'hj_timeline_task'))
    assert path.exists() and rows(s, 'hj_operator_scan_event') == []

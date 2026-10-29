"""One new dual-source native-codec QR integration fixture, never field acceptance.

Uses generated QR UUIDs/pixels as untrusted fixture evidence. No permanent
equipment is registered and no model/provider is called. This is not field acceptance.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import shutil
import subprocess
import traceback

from fastapi.testclient import TestClient
import numpy as np
import qrcode
from uuid import uuid4
import pytest

from visioncortex.lab.api import create_app
from visioncortex.lab.domain import clean, fingerprint
from visioncortex.lab.dynamic_context import DynamicContext
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.local_provider import LocalHTTP, LocalProviders
from visioncortex.lab.rules import Rules
from visioncortex.lab.timeline_report import render_report
from visioncortex.lab.timeline_runtime import TimelineRuntime
from ._fixtures import db as evidence_db, settings as evidence_settings


DAY = '2026-10-29'
CLOCK = DAY + 'T09:00:00+08:00'
SESSION = 'fixture-identical-session'
CAMERAS = ('fixture-wearable-a', 'fixture-wearable-b')


@pytest.fixture(scope='session', name='db')
def isolated_db():
    # Explicit existing owned Unix-socket fixture. Never loads runtime.env.
    return evidence_db.__wrapped__()


@pytest.fixture
def settings(core, tmp_path):
    return replace(evidence_settings.__wrapped__(core, tmp_path),
                   processing_profile='timeline-v1', timeline_providers_path=None,
                   model_manifest=None, live_sources_manifest=None,
                   cache_max_bytes=32 * 1024**2, max_capture_bytes=32 * 1024**2,
                   max_native_bytes=32 * 1024**2, allow_external_inference=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved(root, name, value):
    path = root / name
    assert not path.exists(), 'Keep every attempt in a new proof directory'
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + '\n')
    return path


def proof_directory(tmp_path):
    result = tmp_path / 'CombinedReceipt'
    result.mkdir(parents=True, exist_ok=False)
    return result


def codec(settings, camera, label, lane):
    # Paste the complete generated 645x430 fixture label at native pixels. Same filename,
    # packet capture ID, session text and nonzero PTS occur in both source roots.
    path = settings.incoming / f'{DAY}_{camera}' / 'MetaData/090000/rgb.mp4'
    path.parent.mkdir(parents=True)
    height, width = 540, 960
    rgb_frames, frame_refs = [], []
    for n in range(8):
        rgb = np.full((height, width, 3), 240, dtype=np.uint8)
        rgb[:16, :16] = [lane * 70, n * 27, 31]
        visible = lane == 1 or n >= 4
        if visible:
            rgb[40:40 + label.shape[0], 40:40 + label.shape[1]] = label
        rgb_frames.append(rgb)
        frame_refs.append(dict(frame_index=n, pts=200000 + n * 100000,
                               time_base='1/1000000', stream_origin='1/5',
                               media_offset_seconds=str(Fraction(n, 10)),
                               original_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                               qr_pattern_visible=visible))
    # libx264rgb -crf0 preserves native RGB and actual PTS. Generated fixture only.
    subprocess.run([
        'ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', '10', '-i', 'pipe:0', '-vf',
        'settb=1/1000000,setpts=200000+N*100000', '-vsync', '0',
        '-enc_time_base', '1/1000000', '-c:v', 'libx264rgb', '-crf', '0',
        '-preset', 'ultrafast', '-threads', '1', '-video_track_timescale', '1000000',
        str(path),
    ], input=b''.join(frame.tobytes() for frame in rgb_frames), check=True)
    receipt = dict(capture_id='packet-001', source_instance=camera,
                   source_revision='explicit-synthetic-two-wearable/1', camera_id=camera,
                   kind='video', session_id=SESSION,
                   relative_path=path.relative_to(settings.incoming).as_posix(),
                   captured_at=CLOCK,
                   time_basis={'basis': 'generated_fixture_clock_not_field_calibration',
                               'uncertainty_us': 0, 'clock_sync_valid': True},
                   start_seconds='0', end_seconds='0.8', source_kind='fixture',
                   sha256=sha(path), native_record={'fixture_only': True,
                       'no_sender_seal_fabricated': True, 'codec_generated_for_this_test': True})
    return path, receipt, frame_refs


def registration(receipt, start, end, revision):
    return dict(schema_version='vc-lab-video-qr/1', enabled=True, revision=revision,
                source_sha256=receipt['sha256'], media_version=1,
                source_stream_key=receipt['source_instance'], source_session_key=SESSION,
                rotation_degrees_clockwise=0,
                orientation_basis='upright generated fixture with native generated QR pixels',
                time_basis='original_media_pts', world_clock_status='verified',
                source_intervals=[dict(start_seconds=start, end_seconds=end)])


def count_rows(core, table, where='', params=()):
    # Table identifiers are fixed by the test, never external data.
    with core.db.transaction(readonly=True) as conn:
        return conn.execute(f'SELECT count(*) AS n FROM {table} WHERE customer_id=%s' + where,
                            (core.customer, *params)).fetchone()['n']


def test_two_native_wearables_generated_uuid_qr_isolation_late_and_replay(
        core, settings, tmp_path, monkeypatch):
    root = proof_directory(tmp_path)
    stage = 'generated_label_preparation'
    model_calls = []

    def forbidden(*args, **kwargs):
        model_calls.append('forbidden_model_or_provider_access')
        raise AssertionError('QR-only fixture must not access model/provider/config/token')

    monkeypatch.setattr(LocalProviders, 'load', forbidden)
    monkeypatch.setattr(LocalHTTP, 'preflight', forbidden)
    monkeypatch.setattr(LocalHTTP, 'infer', forbidden)
    try:
        assert settings.timeline_providers_path is settings.model_manifest is None
        assert settings.source_root is None and not settings.timeline_publish
        labels, patterns = [], []
        for n in range(2):
            value = str(uuid4())
            qr = np.asarray(qrcode.make(value).convert('RGB'))
            pixels = np.full((430, 645, 3), 255, dtype=np.uint8)
            pixels[:qr.shape[0], :qr.shape[1]] = qr
            labels.append(pixels)
            patterns.append(dict(device_code=f'FIXTURE-{n + 1}', decoded_uuid=value,
                                 rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                                 generated_fixture=True, no_customer_identity_activation=True))
        stage = 'generate_two_native_fixtures'
        sources = [codec(settings, camera, labels[n], n) for n, camera in enumerate(CAMERAS)]
        assert sum(path.stat().st_size for path, _, _ in sources) < 32 * 1024**2
        # Recoverable copies of newly generated fixture inputs only. Customer
        # originals are never copied; the actual registered source stays intact.
        fixture_copies = []
        for n, (path, _, _) in enumerate(sources):
            target = root / 'NativeFixtures' / f'{DAY}_{CAMERAS[n]}' / 'MetaData/090000/rgb.mp4'
            target.parent.mkdir(parents=True)
            shutil.copyfile(path, target)
            assert sha(target) == sha(path)
            fixture_copies.append(str(target))
        saved(root, 'OriginalSources.json', dict(
            synthetic_fixture_not_field_acceptance=True, generated_qr_patterns=patterns,
            customer_originals_modified=False, fixture_world_clock_is_not_field_calibration=True,
            sources=[dict(camera_id=CAMERAS[n], path=str(path), filename=path.name,
                          sha256=sha(path), bytes=path.stat().st_size, receipt=receipt,
                          recoverable_generated_fixture_copy=fixture_copies[n],
                          exact_original_rgb_frames=refs)
                     for n, (path, receipt, refs) in enumerate(sources)]))
        stage = 'ingest_colliding_identifiers'
        with ThreadPoolExecutor(max_workers=2) as pool:
            captured = list(pool.map(lambda source: Ingest(core, settings).accept(
                source[1], native_original=True), sources))
        assert len({m['capture_id'] for m in captured}) == 2
        assert len({m['media_id'] for m in captured}) == 2
        # Raw generated UUID strings are observed only: no global UUID reservation.
        assert count_rows(core, 'equipment_identity') == count_rows(core, 'entity') == 0
        runtime = TimelineRuntime(core, settings)
        docs = [registration(sources[0][1], '0.4', '0.8', 'two-lane-later/1'),
                registration(sources[1][1], '0', '0.8', 'two-lane-whole/1')]
        jobs = [runtime.tasks.register_video_qr(
            'explicit-native-qr-' + str(n), m['media_id'], docs[n],
            actor='generated fixture', reason='new two-source no-model integration')
                for n, m in enumerate(captured)]
        saved(root, 'Registered.json', dict(captured=captured, tasks=jobs, registrations=docs,
                                          provider_configuration=None, real_model_calls=0))
        stage = 'parallel_original_decode_qr_checkpoint_submit'
        # Each independent runner leases one QR job from the same customer queue.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: TimelineRuntime(core, settings).runner.run_once(), range(2)))
        saved(root, 'InitialRunResults.json', results)
        assert all(result['status'] == 'completed' for result in results), results
        assert {r['task_id'] for r in results} == {j['task_id'] for j in jobs}
        assert not model_calls
        dynamic = DynamicContext(core)
        a_before = dynamic.query_window(captured[0]['media_id'], SESSION, '0', '0.8')
        b_before = dynamic.query_window(captured[1]['media_id'], SESSION, '0', '0.8')
        for n, context in enumerate([a_before, b_before]):
            assert context['source_stream_key'] == CAMERAS[n]
            assert context['source_session_key'] == SESSION
            assert context['media_id'] == captured[n]['media_id']
            assert context['per_panel'] == [] and context['current_focus'] is None
        initial_trace = runtime.views.trace(CLOCK, DAY + 'T09:00:02+08:00')
        assert len(initial_trace['activities']) == 2
        first_report = runtime.views.snapshot(DAY)
        saved(root, 'BeforeLate.json', dict(contexts=[a_before, b_before],
                                          trace=initial_trace, report=first_report))
        stage = 'duplicate_delivery_and_registration'
        before_duplicate = {table: count_rows(core, table) for table in
                            ['capture', 'media_asset', 'hj_dynamic_frame_event',
                             'hj_timeline_window_result', 'hj_activity']}
        duplicate = Ingest(core, settings).accept(sources[1][1], native_original=True)
        assert duplicate['duplicate'] and duplicate['capture_id'] == captured[1]['capture_id']
        assert runtime.tasks.register_video_qr(
            'explicit-native-qr-1', captured[1]['media_id'], docs[1],
            actor='generated fixture', reason='new two-source no-model integration') == jobs[1]
        assert {table: count_rows(core, table) for table in before_duplicate} == before_duplicate
        saved(root, 'DuplicateIdempotency.json', dict(duplicate=duplicate,
              row_counts_unchanged=before_duplicate, qr_task_count=count_rows(core, 'hj_timeline_task', " AND kind='qr'")))
        stage = 'late_earlier_source_interval_no_other_lane_change'
        early = registration(sources[0][1], '0', '0.4', 'two-lane-earlier-late/1')
        late_job = runtime.tasks.register_video_qr(
            'late-earlier-native-qr', captured[0]['media_id'], early,
            actor='generated fixture', reason='late explicit earlier window, not a retry')
        late_result = runtime.runner.run_once()
        saved(root, 'LateResult.json', dict(registration=early, task=late_job, result=late_result))
        assert late_result['status'] == 'completed', late_result
        a_after = dynamic.query_window(captured[0]['media_id'], SESSION, '0', '0.8')
        b_after = dynamic.query_window(captured[1]['media_id'], SESSION, '0', '0.8')
        assert a_after['revision'] > a_before['revision']
        assert fingerprint(b_before) == fingerprint(b_after)
        historical_a = dynamic.query_window(captured[0]['media_id'], SESSION, '0', '0.8',
                                             revision=a_before['revision'])
        assert historical_a == a_before
        saved(root, 'SourceContexts.json', dict(a_before=a_before, a_after=a_after,
              historical_a=historical_a, b_before=b_before, b_after=b_after))
        stage = 'persisted_native_frame_and_checkpoint_readback'
        with core.db.transaction(readonly=True) as conn:
            sql_read_only = conn.execute('SHOW transaction_read_only').fetchone()['transaction_read_only']
            source_rows = conn.execute('''SELECT s.name AS source_name,c.id AS capture_id,
                c.source_capture_id,cs.id AS capture_session_id,cs.source_session_id,
                a.id AS media_id,a.verified_sha256
                FROM capture c JOIN source_instance s ON s.customer_id=c.customer_id AND s.id=c.source_id
                JOIN capture_session cs ON cs.customer_id=c.customer_id AND cs.id=c.session_id
                JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id
                WHERE c.customer_id=%s ORDER BY s.name''', (core.customer,)).fetchall()
            frames = conn.execute('SELECT * FROM hj_dynamic_frame_event WHERE customer_id=%s ORDER BY media_id,source_seconds',
                                  (core.customer,)).fetchall()
            checkpoints = conn.execute('SELECT * FROM hj_timeline_window_result WHERE customer_id=%s ORDER BY task_id,window_index',
                                       (core.customer,)).fetchall()
            tasks = conn.execute("SELECT * FROM hj_timeline_task WHERE customer_id=%s AND kind='qr' ORDER BY created_at,id",
                                 (core.customer,)).fetchall()
        assert len(frames) == 16 and len(checkpoints) == len(tasks) == 3
        assert all(t['status'] == 'completed' for t in tasks)
        assert sql_read_only == 'on'
        assert len(source_rows) == 2 and len({row['capture_session_id'] for row in source_rows}) == 2
        assert {row['source_name'] for row in source_rows} == set(CAMERAS)
        assert {row['source_capture_id'] for row in source_rows} == {'packet-001'}
        assert {row['source_session_id'] for row in source_rows} == {SESSION}
        saved(root, 'SQLReadback.json', dict(transaction_read_only=sql_read_only,
              all_queries_customer_scoped=True, sources=source_rows,
              dynamic_frames=len(frames), qr_tasks=len(tasks), checkpoints=len(checkpoints)))
        saved(root, 'PersistedFrames.json', frames)
        saved(root, 'Checkpoints.json', checkpoints)
        for checkpoint in checkpoints:
            assert fingerprint(checkpoint['normalized_result']) == checkpoint['result_fingerprint']
            qr = checkpoint['execution_receipt']['invocations'][0]['video_qr']
            lane = CAMERAS.index(qr['source_stream_key'])
            assert qr['media_id'] == captured[lane]['media_id']
            assert qr['source_session_key'] == SESSION
            assert qr['source_sha256'] == sources[lane][1]['sha256']
            assert qr['identity_status'] == 'unknown_no_verified_panel_link'
            assert qr['operation_status'] == 'not_inferred'
            assert qr['fields'] == [] and not qr['business_confirmation']
            assert qr['model_invoked'] is False
            assert qr['producer']['qr_calls'] == len(qr['original_frames'])
            for frame in qr['original_frames']:
                index = int(Fraction(frame['media_offset_seconds']) * 10)
                expected = sources[lane][2][index]
                assert frame['pts'] == expected['pts']
                assert Fraction(frame['time_base']) == Fraction(expected['time_base'])
                assert Fraction(frame['stream_origin']) == Fraction(expected['stream_origin'])
                assert frame['original_rgb_sha256'] == expected['original_rgb_sha256']
                assert frame['upright_rgb_sha256'] == expected['original_rgb_sha256']
                values = {item['value'] for item in frame['qr_occurrences']}
                assert values == ({patterns[lane]['decoded_uuid']} if expected['qr_pattern_visible'] else set())
        stage = 'restart_trace_daily_report_and_authorized_media_ranges'
        # Recreate runtime over persisted PG. No repeat decode/submit/model call.
        restarted = TimelineRuntime(core, settings)
        trace = restarted.views.trace(CLOCK, DAY + 'T09:00:02+08:00')
        assert len(trace['activities']) == 3
        assert {a['payload']['video_qr']['source_stream_key'] for a in trace['activities']} == set(CAMERAS)
        assert all(a['instrument_id'] is a['sample_id'] is a['person_id'] is None
                   and a['run_ids'] == [] for a in trace['activities'])
        report = restarted.views.snapshot(DAY)
        assert report['revision'] > first_report['revision']
        assert restarted.views.snapshot(DAY)['revision'] == report['revision']
        text = render_report(report['document'])
        for m in captured:
            assert m['media_id'] in text
        saved(root, 'Trace.json', trace)
        saved(root, 'Report.json', report)
        (root / 'Report.md').write_text(text)
        ranges = []
        with TestClient(create_app(core, settings)) as client:
            assert client.get('/internal/media/' + captured[0]['media_id']).status_code == 401
            for n, m in enumerate(captured):
                evidence = client.get('/internal/evidence/' + m['evidence_id'],
                                      headers={'Authorization': 'Bearer ' + settings.token})
                assert evidence.status_code == 200
                response = client.get('/internal/media/' + m['media_id'], headers={
                    'Authorization': 'Bearer ' + settings.token, 'Range': 'bytes=0-15'})
                with sources[n][0].open('rb') as stream:
                    expected = stream.read(16)
                assert response.status_code == 206 and response.content == expected
                ranges.append(dict(camera_id=CAMERAS[n], media_id=m['media_id'],
                    evidence_id=m['evidence_id'], evidence_status=evidence.status_code,
                    range_status=response.status_code, returned_bytes=len(response.content),
                    returned_sha256=hashlib.sha256(response.content).hexdigest(),
                    content_range=response.headers['content-range']))
        saved(root, 'MediaRangeReadback.json', ranges)
        for path, receipt, _ in sources:
            assert path.name == 'rgb.mp4' and sha(path) == receipt['sha256']
        zero_counts = {table: count_rows(core, table) for table in
                       ['field_candidate', 'fact_revision', 'quantity_ledger_entry',
                        'equipment_identity', 'hj_experiment_binding']}
        assert all(n == 0 for n in zero_counts.values()) and not model_calls
        saved(root, 'Completed.json', dict(status='completed',
            evidence_class='synthetic_native_codec_with_generated_QR_pixels_not_field_acceptance',
            processing_rules=Rules().identity(), customer_id=str(core.customer),
            source_count=2, distinct_capture_ids=len({m['capture_id'] for m in captured}),
            distinct_media_ids=len({m['media_id'] for m in captured}),
            dynamic_frame_count=len(frames), qr_tasks=len(tasks), checkpoints=len(checkpoints),
            activities=len(trace['activities']), zero_counts=zero_counts,
            original_sources_unchanged=True, new_equipment_ids_issued=0,
            recoverable_generated_fixture_copies=fixture_copies, copied_customer_originals=0,
            model_calls=0, provider_calls=model_calls, QR_panels_verified=False,
            current_customer_sources_activated=False, formal_NAS_publication=False,
            media_ranges=ranges, generated_fixture_only=True))
    except BaseException as exc:
        saved(root, 'Failure.json', dict(stage=stage, error=repr(exc),
              traceback=traceback.format_exc(limit=12), model_or_provider_attempts=model_calls,
              original_customer_media_or_uuids_changed=False))
        raise

import json
from pathlib import Path
import threading
from types import SimpleNamespace
import hashlib
import time

import pytest

from visioncortex.application import native_machine as service
from visioncortex.device_day_contract import artifact, atomic_json, archive_name
from visioncortex.device_day_queue import DeviceDayQueue
from visioncortex.run_queue import DurableRunQueue, QueuedRunJob
from fastapi.testclient import TestClient
from visioncortex.machine_api import MachineConfig, create_app
from visioncortex.schemas import ViewRole


@pytest.fixture
def published(tmp_path):
    inputs = tmp_path/'camera/2026-10-16/120000'
    inputs.mkdir(parents=True)
    video, clock = inputs/'rgb.mp4', inputs/'frames.csv'
    video.write_bytes(b'owned deterministic source fixture')
    clock.write_bytes(b'owned deterministic clock fixture')
    record = {'recording_id': 'r1', 'camera_key': 'camera', 'configured_role': 'third_person',
              'relative_path': 'camera/2026-10-16/120000/rgb.mp4', 'video_path': str(video),
              'frames_path': str(clock), 'source_signature': 'v1', 'size_bytes': video.stat().st_size,
              'recording_start_us': 1792123200000000, 'recording_end_us': 1792123210000000}
    config = service.NativeDeviceDayConfig(runtime_root=tmp_path/'runtime', archive_root=tmp_path/'archives',
                                         input_roots={str(tmp_path/'camera'): 'camera'})
    for stage in ('retention', 'vision', 'stt', 'understanding', 'report'):
        queue = DeviceDayQueue(config.runtime_root/'device-day'/f'queue-{stage}.sqlite3')
        queue.enqueue(record, stage+'-revision')
        queue.claim('native-owner')
        queue.finish('native-owner', 'r1', {'status': 'completed'}, .1)
    name = archive_name('camera', record['recording_start_us'])
    root = config.archive_root/name
    root.mkdir(parents=True)
    for name, content in [('Source.mp4', video.read_bytes()), ('Clock.csv', clock.read_bytes()),
                          ('Frame.jpg', b'owned image fixture'), ('Segment.json', b'{}'),
                          ('Input.json', b'{}'), ('Result.json', b'{}')]:
        (root/name).write_bytes(content)
    frame = artifact(root, root/'Frame.jpg') | {'frame_id': 'frame-1', 'frame_kind': 'scene_sample',
                                               'local_ms': 500, 'capture_us': record['recording_start_us']+500000}
    segment = {'recording_id': 'r1', 'segment_id': 's1', 'key_frames': [], 'scene_frames': [frame],
               'source_ref': artifact(root, root/'Source.mp4'), 'clock_ref': artifact(root, root/'Clock.csv'),
               'json_path': 'Segment.json'}
    meaning = {'segment_id': 's1', 'status': 'completed', 'windows': [{'summary': '场景观察',
                'source_frames': [frame], 'input': 'Input.json', 'model_receipt': 'Result.json'}]}
    atomic_json(root/'ProcessedClips/Index.json', {'recordings': [record, {'recording_id': 'unselected'}],
                'segments': [segment], 'understandings': [meaning]})
    atomic_json(root/'LaboratoryDailyReport/LaboratoryDailyReport.json', {'entries': []})
    (root/'LaboratoryDailyReport/LaboratoryDailyReport.html').write_text('native report')
    manifest = service.NativePathManifest(experiment_id='machine-one', views=[{
        'view_id': 'external-camera', 'role': 'third_person', 'video': video, 'timestamps_csv': clock}])
    return config, manifest, root


def test_single_camera_reuses_native_results_without_changing_queues(published):
    config, manifest, root = published
    before = {s: service._rows(config, s, recording_id='r1') for s in ('vision', 'stt', 'understanding', 'report')}
    requests = service.bind_inputs(manifest, config)
    result = service.inspect(config, requests)
    assert result['ready'] and result['additional_model_invocations'] == 0
    assert [r['recording_id'] for r in result['device_days'][0]['recordings']] == ['r1']
    assert result['physical_action_confirmed'] is False
    for stage, rows in before.items():
        assert service._rows(config, stage, recording_id='r1') == rows
    evidence = service.evidence(result, 'frame-1')
    assert evidence['frame_kind'] == 'scene_sample'
    assert evidence['source']['virtual_local_ms'] == 500
    assert evidence['source']['source_segment_path'] == str(manifest.views[0].video)


@pytest.mark.parametrize('change', ['failed', 'changed_source', 'corrupt_frame'])
def test_native_failure_version_change_and_corruption_fail_closed(published, change):
    config, manifest, root = published
    requests = service.bind_inputs(manifest, config)
    if change == 'corrupt_frame':
        (root/'Frame.jpg').write_bytes(b'changed')
    else:
        with DeviceDayQueue(service._database(config, 'vision')).connect() as db:
            if change == 'failed':
                db.execute("UPDATE recordings SET status='failed'")
            else:
                db.execute("UPDATE recordings SET payload=json_set(payload,'$.source_signature','v2')")
    with pytest.raises((ValueError, RuntimeError)):
        service.inspect(config, requests)


def test_queue_completion_waits_for_native_publication(published):
    config, manifest, root = published
    requests = service.bind_inputs(manifest, config)
    index = json.loads((root/'ProcessedClips/Index.json').read_text())
    index['understandings'] = []
    atomic_json(root/'ProcessedClips/Index.json', index)
    assert not service.inspect(config, requests)['ready']


def test_video_only_delivery_follows_native_report_without_waiting_for_stt(published):
    config, manifest, root = published
    requests = service.bind_inputs(manifest, config)
    with DeviceDayQueue(service._database(config, 'stt')).connect() as db:
        db.execute("UPDATE recordings SET status='queued'")
    assert service.inspect(config, requests)['ready']


def test_wrong_role_and_missing_discovery_are_not_faked(published):
    config, manifest, _ = published
    manifest.views[0].role = ViewRole.FIRST_PERSON
    with pytest.raises(ValueError):
        service.bind_inputs(manifest, config)
    manifest.views[0].role = ViewRole.THIRD_PERSON
    with DeviceDayQueue(service._database(config, 'vision')).connect() as db:
        db.execute('DELETE FROM recordings')
    with pytest.raises(LookupError):
        service.bind_inputs(manifest, config)


def test_machine_lease_and_restart_link_same_native_ids(published, tmp_path, monkeypatch):
    config, manifest, _ = published
    native = SimpleNamespace(queue=DurableRunQueue(tmp_path/'machine.sqlite3'))
    service.enqueue(native, manifest, config, 'run1', 'archive1')
    job = native.queue.claim_next('owner1', lease_seconds=1, now=10, kind='native_device_day')
    assert job.kind == 'native_device_day'
    assert native.queue.claim_next('owner2', lease_seconds=1, now=10.5) is None
    recovered = native.queue.claim_next('owner2', lease_seconds=1, now=12)
    assert recovered.reclaimed and recovered.payload['requests'] == job.payload['requests']
    updates = []
    monkeypatch.setattr('visioncortex.application.runtime_host._update', lambda *args, **kwargs: updates.append(kwargs))
    service.execute(QueuedRunJob('run1', job.kind, job.payload, 1, 2, True),
                    {'storage': {'local_runtime_root': str(tmp_path/'machine-runtime')}}, threading.Event())
    assert updates[-1]['state'] == 'completed'
    receipt = json.loads(Path(updates[-1]['native_result_receipt']).read_text())
    assert receipt['requests'][0]['recording_id'] == 'r1'


def test_reference_claim_cannot_bypass_gpu_admission_for_legacy_jobs(tmp_path):
    queue = DurableRunQueue(tmp_path/'q.sqlite3')
    queue.enqueue('legacy', 'run', {})
    assert queue.claim_next('owner', lease_seconds=1, kind='native_device_day') is None
    queue.enqueue('native', 'native_device_day', {})
    assert queue.claim_next('owner', lease_seconds=1, kind='native_device_day').run_id == 'native'
    assert queue.get_job('legacy')['status'] == 'queued'


def test_machine_single_camera_api_idempotency_scope_and_native_evidence(published, tmp_path, monkeypatch):
    config, manifest, root = published
    database = tmp_path/'machine/state/queue.sqlite3'
    native = SimpleNamespace(queue=DurableRunQueue(database))
    settings = {'storage': {'local_runtime_root': str(tmp_path/'machine'), 'archive_root': str(tmp_path/'machine-archives')}}
    native.settings = lambda: settings
    native.maintenance_active = lambda: False
    native.initialize_queue = lambda _: None
    native.queue_database_path = lambda _: database
    native.get_run = lambda run: native.queue.load_run(run) | {'run_id': run}
    config = MachineConfig(native_device_day=config, clients=[{
        'client_id': 'gateway', 'token_sha256': hashlib.sha256(b'owned-token').hexdigest(),
        'default_scope': 'camera', 'scopes': {'camera': list(config.input_roots), 'other': list(config.input_roots)}}])
    header = {'Authorization': 'Bearer owned-token', 'Idempotency-Key': 'once'}
    def update(run_id, **fields):
        native.queue.patch_run(run_id, fields)
    monkeypatch.setattr('visioncortex.application.runtime_host._update', update)
    with TestClient(create_app(config, native)) as client:
        reply = client.post('/api/runs/from-paths', json=manifest.model_dump(mode='json'), headers=header)
        assert reply.status_code == 202
        run_id = reply.json()['run_id']
        deadline = time.monotonic()+5
        while native.queue.get_job(run_id) is None and time.monotonic() < deadline:
            time.sleep(.02)
        job = native.queue.claim_next('machine', lease_seconds=60)
        assert job is not None and job.kind == 'native_device_day'
        service.execute(job, settings, threading.Event())
        assert client.get('/api/runs/'+run_id, headers=header).json()['state'] == 'completed'
        again = client.post('/api/runs/from-paths', json=manifest.model_dump(mode='json'), headers=header)
        assert again.json()['run_id'] == run_id
        archive = reply.json()['experiment_id']
        result = client.get('/api/archives/'+archive, headers=header)
        assert result.status_code == 200 and result.json()['workflow'] == 'device_day'
        evidence = client.get('/api/evidence/frame-1', params={'archive': archive}, headers=header)
        assert evidence.status_code == 200
        url = evidence.json()['artifact_url']
        assert client.get(url, headers=header).content == (root/'Frame.jpg').read_bytes()
        assert client.get(url).status_code == 401
        assert client.get(url, headers=header | {'X-Execution-Scope': 'other'}).status_code == 404
        forbidden = url.replace('Frame.jpg', 'LaboratoryDailyReport/LaboratoryDailyReport.html')
        assert client.get(forbidden, headers=header).status_code == 404
        assert service._rows(config.native_device_day, 'vision', recording_id='r1')[0]['attempts'] == 1

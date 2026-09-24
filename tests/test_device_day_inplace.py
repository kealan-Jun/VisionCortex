"""Local owned byte fixtures only; no model, NAS or accuracy evidence."""
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_inplace import active, archive_priority, receipt, retention_policy
from visioncortex.device_day_inputs import source_path
from visioncortex.publication_journal import PublicationJournal, reconcile


@pytest.fixture
def setup(device_config):  # noqa: F811
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    return device_config, record, layout, backend, DeviceDayRunner(device_config, backend)


def test_preprocess_before_archive_and_publish_without_reinference(setup):
    config, record, layout, backend, runner = setup
    original = backend.vision
    def vision(layout, value, key):
        assert not list(layout.raw.glob('*.mp4'))
        assert source_path(layout, value['sources'][0]) == Path(record['video_path'])
        return original(layout, value, key)
    backend.vision = vision
    result = runner.process(record, stage='vision')
    assert result['status'] == 'completed', result
    assert result['message'] == '预处理完成，等待归档'
    assert not layout.index.exists()
    assert receipt(runner, record, 'input-vision')['status'] == 'ready'
    assert runner.process(record, stage='stt')['status'] == 'completed'
    result = runner.process(record, stage='retention')
    assert result['status'] == 'completed', result
    assert result['formal_publication'] == 'completed', result
    assert read_json(layout.index)['segments']
    key = receipt(runner, record, 'vision')['key']
    Path(record['video_path']).unlink()  # Owned fixture simulates external expiry.
    resumed = DeviceDayRunner(config, backend)
    assert resumed.process(record, stage='vision')['status'] == 'completed'
    assert receipt(resumed, record, 'vision')['key'] == key
    assert backend.vision_calls == 1
    assert reconcile(resumed) == 1
    assert not PublicationJournal(runner.runtime_root).pending()


def test_same_size_replacement_with_restored_mtime_blocks_publication(setup):
    _, record, layout, backend, runner = setup
    original = backend.vision
    def changed(layout, value, key):
        result = original(layout, value, key)
        path = Path(record['video_path'])
        before = path.stat()
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(b'X' * before.st_size)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, path)
        return result
    backend.vision = changed
    result = runner.process(record, stage='vision')
    assert result['status'] == 'failed', result
    assert result['error_type'] == 'InputChanged'
    assert not layout.index.exists()
    assert receipt(runner, record, 'publication')['status'] == 'blocked'
    assert PublicationJournal(runner.runtime_root).pending()


def test_archive_retry_does_not_run_models_again(setup, monkeypatch):
    _, record, layout, backend, runner = setup
    assert runner.process(record, stage='vision')['status'] == 'completed'
    import visioncortex.device_day_inplace as module
    original = module.copy_sealed
    calls = []
    def fail_once(config, source, target, **kwargs):
        calls.append(source['kind'])
        if len(calls) == 2:
            raise OSError('owned fixture interruption')
        return original(config, source, target, **kwargs)
    monkeypatch.setattr(module, 'copy_sealed', fail_once)
    assert runner.process(record, stage='retention')['status'] == 'failed'
    assert receipt(runner, record, 'vision')['status'] == 'completed'
    assert runner.process(record, stage='retention', retry=True)['status'] == 'completed'
    assert backend.vision_calls == 1
    assert read_json(layout.index)['segments']


def test_preprocess_failure_does_not_recopy_archive(setup, monkeypatch):
    _, record, _, backend, runner = setup
    assert runner.process(record, stage='retention')['status'] == 'completed'
    backend.fail = True
    assert runner.process(record, stage='vision')['status'] == 'failed'
    import visioncortex.device_day_inplace as module
    monkeypatch.setattr(module, 'copy_sealed', lambda *a, **k: pytest.fail('recopied verified original'))
    backend.fail = False
    assert runner.process(record, stage='vision', retry=True)['status'] == 'completed'
    assert runner.process(record, stage='retention', retry=True)['status'] == 'completed'


def test_publication_interruption_recovers_after_restart(setup, monkeypatch):
    config, record, layout, backend, runner = setup
    assert runner.process(record, stage='vision')['status'] == 'completed'
    original = runner.refresh_index
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('crash after atomic index replacement')
    monkeypatch.setattr(runner, 'refresh_index', interrupted)
    result = runner.process(record, stage='retention')
    assert result['status'] == 'completed'
    assert result['formal_publication'] == 'pending'
    assert receipt(runner, record, 'publication')['status'] == 'blocked'
    resumed = DeviceDayRunner(config, backend)
    assert reconcile(resumed) == 1
    assert receipt(resumed, record, 'publication')['status'] == 'completed'
    assert read_json(layout.index)['segments']
    assert backend.vision_calls == 1


def test_independent_queue_admission_and_duplicate_claims(setup):
    _, record, _, backend, runner = setup
    inventory = {'recordings': [record]}
    assert runner._prepare_stage(inventory, 'retention', None) == set()
    assert runner._prepare_stage(inventory, 'vision', None) == {record['recording_id']}
    assert runner._prepare_stage(inventory, 'stt', None) == {record['recording_id']}
    queue = runner.queues['vision']
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(queue.claim, ['one', 'two']))
    assert sum(x is not None for x in claims) == 1
    assert runner.process(record, stage='vision')['status'] == 'completed'
    runner._prepared.clear()
    assert runner._prepare_stage(inventory, 'retention', None) == {record['recording_id']}
    assert backend.vision_calls == 1


def test_os_lock_prevents_duplicate_inference(setup):
    _, record, _, backend, runner = setup
    begun, release = threading.Event(), threading.Event()
    original = backend.vision
    def slow(*args):
        begun.set()
        assert release.wait(5)
        return original(*args)
    backend.vision = slow
    with ThreadPoolExecutor(max_workers=2) as pool:
        job = pool.submit(runner.process, record, stage='vision')
        assert begun.wait(5)
        duplicate = runner.process(record, stage='vision')
        release.set()
        assert duplicate['status'] == 'running_elsewhere'
        assert job.result()['status'] == 'completed'
    assert backend.vision_calls == 1


def test_unknown_retention_is_explicit_and_risk_or_age_preempts(setup):
    config, record, _, _, runner = setup
    assert retention_policy(runner, record)['status'] == 'unknown'
    assert archive_priority(runner, record, time.time()) == (False, False)
    assert archive_priority(runner, record, time.time() - 901) == (False, True)
    runner.settings['capture_retention'] = {'cleanup_risk': True}
    assert archive_priority(runner, record, time.time()) == (True, False)
    assert runner._prepare_stage({'recordings': [record]}, 'retention', None) == {record['recording_id']}
    runner.settings['capture_retention'] = {'seconds': 60}
    assert retention_policy(runner, record)['deadline'] == record['recording_end_us'] / 1e6 + 60


def test_old_completed_receipts_and_old_queue_survive(setup):
    config, record, layout, backend, _ = setup
    config['device_day']['inplace_preprocessing'] = False
    legacy = DeviceDayRunner(config, backend)
    assert legacy.process(record)['status'] == 'completed'
    before = (layout.receipts / record['recording_id'] / 'vision.json').read_bytes()
    legacy.queues['vision'].enqueue(record, 'old-revision')
    claimed = legacy.queues['vision'].claim('old-owner')
    legacy.queues['vision'].finish('old-owner', claimed['recording_id'], {'status': 'completed'}, 1)
    config['device_day']['inplace_preprocessing'] = True
    current = DeviceDayRunner(config, backend)
    assert not active(current, record)
    assert current.process(record, stage='vision')['status'] == 'completed'
    assert (layout.receipts / record['recording_id'] / 'vision.json').read_bytes() == before
    assert current.queues['vision'].snapshot()['counts']['completed'] == 1
    assert backend.vision_calls == 1


def test_old_unstarted_queue_migrates_without_loss(setup):
    config, record, _, _, runner = setup
    runner.queues['vision'].enqueue(record, 'old-revision')
    current = DeviceDayRunner(config, FakeModels())
    assert current._prepare_stage({'recordings': []}, 'vision', None) == {record['recording_id']}
    assert current.queues['vision'].claim('new-owner')['recording_id'] == record['recording_id']


def test_input_disappears_during_inference(setup):
    _, record, layout, backend, runner = setup
    original = backend.vision
    def disappear(*args):
        result = original(*args)
        Path(record['video_path']).unlink()
        return result
    backend.vision = disappear
    assert runner.process(record, stage='vision')['status'] == 'failed'
    assert not layout.index.exists()


def test_all_stages_and_service_queues_finish(setup):
    _, record, layout, backend, runner = setup
    result = runner.run_once({'recordings': [record]}, stage='all')
    assert all(r['status'] == 'completed' for r in result['results']), result
    assert read_json(layout.index)['segments']
    assert backend.vision_calls == 1 and backend.semantic_calls == 1
    assert runner.queues['retention'].snapshot()['counts']['completed'] == 1
    assert runner.run_once({'recordings': [record]}, stage='all')['results'] == []


def test_cli_all_stages(setup):
    _, record, layout, backend, runner = setup
    result = runner.process(record)
    assert result['status'] == 'completed', result
    assert read_json(layout.index)['segments']
    assert backend.vision_calls == 1 and backend.semantic_calls == 1


def test_lifecycle_status_and_timings_are_durable_local_metadata(setup):
    config, record, _, _, runner = setup
    assert runner.run_once({'recordings': [record]}, stage='vision')['results'][0]['status'] == 'completed'
    from visioncortex.device_day_progress import snapshot
    result = snapshot(config)
    assert '预处理完成，等待归档' in result['latency_html']
    state = result['input_lifecycle'][0]
    assert state['input_ready']['vision'] == 'ready'
    assert state['archive_status'] == 'pending'
    assert state['publication_status'] == 'pending'
    assert state['timings']['vision']['input_hash_seconds'] >= 0
    assert state['timings']['vision']['queue_wait_seconds'] >= 0


def test_restarts_resume_expired_lease_without_duplicate_model(setup):
    config, record, _, backend, runner = setup
    runner._prepare_stage({'recordings': [record]}, 'vision', None)
    queue = runner.queues['vision']
    queue.claim('dead-worker')
    assert runner.process(record, stage='vision')['status'] == 'completed'
    with queue.connect() as db:
        db.execute("UPDATE recordings SET lease_until=0 WHERE lease_owner='dead-worker'")
    restarted = DeviceDayRunner(config, backend)
    assert restarted.run_once({'recordings': []}, stage='vision')['results'][0]['status'] == 'completed'
    assert backend.vision_calls == 1


def test_shared_io_copy_limit_and_aging_are_global(setup):
    config, _, _, _, _ = setup
    from visioncortex.runtime_control import ExecutionContext, ResourceCoordinator
    root = Path(config['storage']['local_runtime_root']) / 'state/resources.sqlite3'
    one, two = ResourceCoordinator(root), ResourceCoordinator(root)
    assert one.try_claim('decode', 'nas-io', 1, 1, ExecutionContext(priority=0), now=0)
    assert not two.try_claim('copy', 'nas-io', 1, 1, ExecutionContext(priority=10), now=1)
    # Keep the waiting copy lease alive while long-running decoders occupy IO.
    for now in range(50, 351, 50):
        from visioncortex.sqlite_store import connection
        with connection(root) as db:
            db.execute("UPDATE leases SET expires=? WHERE id IN ('decode','copy')", (now + 90,))
        assert not two.try_claim('copy', 'nas-io', 1, 1, ExecutionContext(priority=10), now=now)
    one.release('decode')
    assert not one.try_claim('new-live', 'nas-io', 1, 1, ExecutionContext(priority=0), now=351)
    assert two.try_claim('copy', 'nas-io', 1, 1, ExecutionContext(priority=10), now=351)


def test_completed_input_hash_is_not_read_twice(setup, monkeypatch):
    _, record, _, _, runner = setup
    path = Path(record['video_path'])
    original = Path.open
    reads = []
    def counted(self, *args, **kwargs):
        if self == path and args and args[0] == 'rb':
            reads.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    assert runner.process(record, stage='vision')['status'] == 'completed'
    assert len(reads) == 1


def test_mismatched_content_locator_rejected(tmp_path):
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.speech_worker import resolve_source
    locator = tmp_path / 'Location.json'
    atomic_json(locator, {'binding': 'wrong', 'files': {}})
    with pytest.raises(ValueError, match='content binding'):
        resolve_source({'content_locator': str(locator), 'binding': 'right'})


def test_stt_chunk_resume_after_archive_path_change_without_repeat_calls(setup, monkeypatch):
    config, record, layout, backend, runner = setup
    from visioncortex import speech, speech_qwen as qwen, speech_worker
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.device_day_stt import transcribe
    from test_speech_qwen import event
    folder = Path(record['video_path']).parent
    (folder / 'audio.opus').write_bytes(b'owned-synthetic-audio')
    atomic_json(folder / 'audio_meta.json', {'recording_session_id': 'fixture-1',
                'first_audio_global_us': record['recording_start_us'], 'audio_duration_us': 360000000})
    atomic_json(folder / 'audio_ready.json', {'recording_session_id': 'fixture-1', 'ready': True, 'audio_valid': True})
    for file in folder.glob('audio*'):
        os.utime(file, (time.time() - 1000, time.time() - 1000))
    record, _ = item_and_layout(config)
    config['speech_recognition']['enabled'] = True
    runner = DeviceDayRunner(config, backend)
    runtime = {'provider': 'aliyun_qwen', 'worker_sha256': speech_worker.sha256(Path(qwen.__file__)),
               'max_audio_seconds': 180, 'model': {'model': qwen.MODEL},
               'endpoint': 'https://dashscope.aliyuncs.com/api/v1/test'}
    monkeypatch.setattr(speech, 'probe_audio', lambda path: {'duration_seconds': 360})
    monkeypatch.setattr(speech, 'runtime_request', lambda config: runtime)
    monkeypatch.setattr(qwen.subprocess, 'run', lambda cmd, **kwargs: Path(cmd[-1]).write_bytes(b'derived-test-audio'))
    calls = []
    def recognize(*args):
        calls.append(True)
        if len(calls) == 2:
            raise OSError('owned second chunk interruption')
        return {'http_status': 200, 'events': [event()], 'wall_seconds': .1}
    monkeypatch.setattr(qwen, 'recognize', recognize)
    backend.transcribe = lambda layout, retained, key: transcribe(config, layout, retained, key)
    assert runner.process(record, stage='stt')['status'] == 'failed'
    assert len(calls) == 2
    assert not list(layout.raw.glob('*.mp4'))
    requests = {p: p.read_bytes() for p in layout.receipts.glob('*/speech/*/*/request.json')}
    assert runner.process(record, stage='retention')['status'] == 'completed'
    (folder / 'audio.opus').unlink()  # External expiry simulated only on owned fixture.
    result = DeviceDayRunner(config, backend).process(record, stage='stt', retry=True)
    assert result['status'] == 'completed', result
    assert len(calls) == 3  # First successful chunk was not billed/executed again.
    assert all(p.read_bytes() == content for p, content in requests.items())


def test_late_audio_does_not_repeat_video(setup):
    config, record, _, backend, runner = setup
    from visioncortex.device_day_contract import atomic_json
    assert runner.process(record, stage='vision')['status'] == 'completed'
    assert runner.process(record, stage='retention')['status'] == 'completed'
    folder = Path(record['video_path']).parent
    (folder / 'audio.opus').write_bytes(b'owned-late-audio')
    atomic_json(folder / 'audio_ready.json', {'ready': True})
    for file in folder.glob('audio*'):
        os.utime(file, (time.time() - 1000, time.time() - 1000))
    updated, _ = item_and_layout(config)
    assert runner.process(updated, stage='stt')['status'] == 'completed'
    assert runner.process(updated, stage='retention')['status'] == 'completed'
    assert runner.process(updated, stage='vision')['status'] == 'completed'
    assert backend.vision_calls == 1


@pytest.mark.parametrize('setting,value', [('nas_io_slots', 0), ('archive_copy_workers', True),
                                          ('archive_max_wait_seconds', float('inf'))])
def test_invalid_scheduling_config_fails_closed(setup, setting, value):
    config, *_ = setup
    config['device_day'][setting] = value
    with pytest.raises(ValueError):
        DeviceDayRunner(config, FakeModels())


def test_failed_legacy_archive_can_migrate_without_clearing_queue(setup):
    config, record, _, _, runner = setup
    from visioncortex.device_day_contract import atomic_json
    path = runner._receipt(runner.layout(record), record, 'retention')
    atomic_json(path, {'status': 'failed', 'stage': 'retention', 'message': 'old copy interrupted'})
    runner.queues['vision'].enqueue(record, 'old-queue-key')
    assert runner.run_once({'recordings': [record]}, stage='vision')['results'][0]['status'] == 'completed'
    assert list(path.parent.glob('history/legacy-retention-*.json'))
    assert runner.queues['vision'].snapshot()['counts']['completed'] == 1


def test_concurrent_preprocessing_keeps_publication_journal_until_all_locks_release(setup):
    _, record, _, backend, runner = setup
    assert runner.process(record, stage='retention')['status'] == 'completed'
    started, release = threading.Event(), threading.Event()
    original = backend.transcribe
    def pending(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    backend.transcribe = pending
    with ThreadPoolExecutor(max_workers=2) as pool:
        stt = pool.submit(runner.process, record, stage='stt')
        assert started.wait(5)
        assert runner.process(record, stage='vision')['status'] == 'completed'
        assert PublicationJournal(runner.runtime_root).pending()
        assert reconcile(runner) == 0
        release.set()
        assert stt.result()['status'] == 'completed'
    assert reconcile(runner) == 1


def test_corrupted_archive_blocks_publication_preserves_completed_preprocessing(setup):
    _, record, layout, backend, runner = setup
    assert runner.process(record, stage='vision')['status'] == 'completed'
    assert runner.process(record, stage='retention')['status'] == 'completed'
    archived = next(layout.raw.glob('*.mp4'))
    archived.write_bytes(b'corrupted owned original')
    result = runner.process(record, stage='vision')
    assert result['status'] == 'failed' and result['error_type'] == 'InputChanged'
    assert receipt(runner, record, 'vision')['status'] == 'completed'
    assert receipt(runner, record, 'publication')['status'] == 'blocked'
    assert backend.vision_calls == 1


def test_native_frame_relocation_and_plan_identity_are_content_bound(setup):
    _, record, layout, _, runner = setup
    from visioncortex.device_day_inputs import input_context, plan_sources, relocate_evidence
    from visioncortex.schemas import FrameEvidence, SourceFrameIdentity, VideoInfo, ViewInput
    assert runner.process(record, stage='vision')['status'] == 'completed'
    seal = receipt(runner, record, 'input-vision')
    video = next(s for s in seal['sources'] if s['kind'] == 'video')
    view = ViewInput(view_id=record['camera_key'], role=record['configured_role'],
                     video=record['video_path'], timestamps_csv=record['frames_path'])
    info = VideoInfo(path=view.video, duration_ms=1000, fps=10, width=16, height=16, frame_count=10)
    native = SourceFrameIdentity(source_path=view.video.resolve(), source_size_bytes=video['size_bytes'],
             source_mtime_ns=video['mtime_ns'], status='resolved', packet_position=1, source_pts=0,
             time_base='1/1000', source_frame_index=0, decoded_pixels_sha256='a' * 64)
    evidence = FrameEvidence(view_id=view.view_id, role=view.role, frame_index=0, local_ms=0,
                             width=16, height=16, source_frame=native)
    assert runner.process(record, stage='retention')['status'] == 'completed'
    archived = layout.root / video['retained']['path']
    moved = view.model_copy(update={'video': archived})
    with input_context(seal):
        assert plan_sources(view, info) == plan_sources(moved, info.model_copy(update={'path': archived}))
        rebased, proof = relocate_evidence(moved, evidence)
        assert rebased.source_frame.source_path == archived
        assert rebased.source_frame.decoded_pixels_sha256 == native.decoded_pixels_sha256
        assert rebased.source_frame.source_pts == native.source_pts
        assert proof['ledger_rewritten'] is False
        assert evidence.source_frame.source_path == view.video
        archived.write_bytes(b'changed owned bytes')
        with pytest.raises(ValueError):
            relocate_evidence(moved, evidence)


def test_service_uses_inplace_preprocessing(setup, monkeypatch):
    config, record, _, backend, runner = setup
    from visioncortex import device_day_service as module
    from visioncortex.nas_recordings import scan_recordings
    begun, release = threading.Event(), threading.Event()
    original = backend.vision
    def processing(layout, value, key):
        assert not list(layout.raw.glob('*.mp4'))
        begun.set()
        assert release.wait(8)
        return original(layout, value, key)
    backend.vision = processing
    monkeypatch.setattr(module, 'DeviceDayRunner', lambda settings: runner)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 10**12})())
    service = module.DeviceDayService(lambda: config, threading.Lock())
    service.observe(config, scan_recordings(config))
    service.start()
    try:
        assert begun.wait(8), service.last_result
        assert receipt(runner, record, 'retention').get('status') != 'completed'
        release.set()
        deadline = time.monotonic() + 10
        while receipt(runner, record, 'publication').get('status') != 'completed':
            assert time.monotonic() < deadline, service.last_result
            time.sleep(.05)
    finally:
        release.set()
        service.stop()


def test_queue_can_verify_archive_after_external_capture_expiry(setup):
    _, record, _, backend, runner = setup
    assert runner.process(record, stage='retention')['status'] == 'completed'
    from visioncortex.input_availability import Availability
    runner._prepare_stage({'recordings': [record]}, 'vision', None)
    Path(record['video_path']).unlink()
    Availability(runner.runtime_root).mark(record, 'missing', reason='owned fixture expired')
    runner._prepared.clear()
    result = runner.run_once({'recordings': []}, stage='vision')
    assert result['results'][0]['status'] == 'completed', result
    assert backend.vision_calls == 1


def test_audio_deadline_can_preempt_later_video_retention(setup):
    _, record, _, _, runner = setup
    ended = time.time()
    record = record | {'recording_end_us': round(ended * 1e6),
                       'audio': {'status': 'provided', 'end_us': round(ended * 1e6)}}
    runner.settings['capture_retention'] = {'by_kind': {'video': {'seconds': 86400}, 'audio': {'seconds': 60}},
                                             'safety_margin_seconds': 120}
    policy = retention_policy(runner, record)
    assert policy['status'] == 'known'
    assert policy['deadline'] == record['audio']['end_us'] / 1e6 + 60
    assert archive_priority(runner, record, time.time())[0]
    record['audio']['end_us'] = None
    assert retention_policy(runner, record)['status'] == 'unknown'


def test_restart_migrates_retention_only_backlog_and_preserves_age(setup):
    config, record, _, backend, runner = setup
    queue = runner.queues['retention']
    queue.enqueue(record, 'legacy')
    with queue.connect() as db:
        db.execute('UPDATE recordings SET queued_at=?', (time.time() - 2000,))
    restarted = DeviceDayRunner(config, backend)
    assert restarted._prepare_stage({'recordings': []}, 'retention', None) == {record['recording_id']}
    assert restarted._prepare_stage({'recordings': []}, 'vision', None) == {record['recording_id']}


def test_audio_validation_failure_does_not_prevent_saving_video(setup, monkeypatch):
    _, record, layout, _, runner = setup
    import visioncortex.device_day_inplace as module
    original = module.seal_input
    def seal(runner, record, stage):
        if stage == 'stt':
            raise OSError('owned audio metadata unavailable')
        return original(runner, record, stage)
    monkeypatch.setattr(module, 'seal_input', seal)
    assert runner.process(record, stage='retention')['status'] == 'failed'
    assert next(layout.raw.glob('*.mp4')).read_bytes() == Path(record['video_path']).read_bytes()
    assert not layout.index.exists()


def test_archive_input_error_does_not_mark_other_stage_inputs_unavailable(setup, monkeypatch):
    _, record, _, _, runner = setup
    runner.settings['capture_retention'] = {'cleanup_risk': True}
    import visioncortex.device_day_inplace as module
    monkeypatch.setattr(module, 'copy_sealed', lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError('owned archive failure')))
    result = runner.run_once({'recordings': [record]}, stage='retention')
    assert result['results'][0]['status'] == 'failed'
    from visioncortex.input_availability import Availability
    assert record['recording_id'] not in Availability(runner.runtime_root).states()
    result = runner.run_once({'recordings': []}, stage='vision')
    assert result['results'][0]['status'] == 'completed'


def test_concurrent_input_hashing_shares_actual_byte_read(tmp_path, monkeypatch):
    from visioncortex.device_day_inputs import hash_content
    path = tmp_path / 'owned.bin'
    path.write_bytes(b'owned content')
    original = Path.open
    reads = []
    def counted(self, *args, **kwargs):
        if self == path and args and args[0] == 'rb':
            reads.append(self)
            time.sleep(.03)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    with ThreadPoolExecutor(max_workers=2) as pool:
        checksums = list(pool.map(hash_content, [path, path]))
    assert checksums[0] == checksums[1]
    assert len(reads) == 1

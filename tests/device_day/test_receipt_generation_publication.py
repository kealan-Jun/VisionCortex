"""Concurrent owned-file receipt generations defer publication, never inference."""
import json
import threading
import time

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import atomic_json, read_json, verify_artifact
from visioncortex.publication_journal import PublicationJournal, reconcile
from visioncortex.receipt_projection import ProjectionGenerationChanged, ReceiptProjection

__all__ = ['device_config']


@pytest.mark.parametrize(('stage', 'checkpoint'), [('vision', 1), ('stt', 2)])
def test_parallel_receipt_commit_preserves_completed_stage_and_replays_without_any_model(
        device_config, monkeypatch, stage, checkpoint):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    stt_calls = []
    transcribe = backend.transcribe
    def count_transcribe(*args):
        stt_calls.append(True)
        return transcribe(*args)
    backend.transcribe = count_transcribe
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(record, stage='retention')['status'] == 'completed'
    if stage == 'stt':
        assert runner.process(record, stage='vision')['status'] == 'completed'
    before_index = layout.index.read_bytes()
    sibling = runner._receipt(layout, record, 'retention' if stage == 'vision' else 'vision')
    sibling_bytes = sibling.read_bytes()
    sibling_value = read_json(sibling)
    started, finished = threading.Event(), threading.Event()
    failures, completed_receipt = [], []
    def writer():
        try:
            assert started.wait(5), 'Publication never reached its real generation read'
            # A sibling stage atomically recommits its same verified output.
            # Bytes/native key stay exact, but the inode generation changes.
            atomic_json(sibling, sibling_value)
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()
    worker = threading.Thread(target=writer)
    worker.start()
    version = ReceiptProjection._source_version
    reads = []
    def generation(folder, stages):
        current = version(folder, stages)
        if folder.name == record['recording_id']:
            reads.append(current)
            if len(reads) == checkpoint:
                path = runner._receipt(layout, record, stage)
                completed_receipt.append(path.read_bytes())
                assert read_json(path)['status'] == 'completed'
                started.set()
                assert finished.wait(5), 'Independent receipt writer did not finish'
        return current
    monkeypatch.setattr(ReceiptProjection, '_source_version', staticmethod(generation))
    try:
        result = runner.run_once({'recordings': [record]}, stage=stage)['results'][0]
    finally:
        started.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert result['status'] == 'completed'
    assert result['formal_publication'] == 'pending'
    assert result['reason'] == 'receipt_generation_changed'
    assert result['error_type'] == 'ProjectionGenerationChanged'
    assert result['expected_generation'] != result['observed_generation']
    path = runner._receipt(layout, record, stage)
    assert path.read_bytes() == completed_receipt[0]
    native = read_json(path)
    assert native['status'] == 'completed' and 'reason' not in native
    for receipt in (native, sibling_value):
        assert all(verify_artifact(layout.root, ref) for ref in receipt['artifacts'])
    assert sibling.read_bytes() == sibling_bytes
    assert layout.index.read_bytes() == before_index  # No mixed generation was accepted.
    with runner.queues[stage].connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert row['status'] == 'completed' and row['attempts'] == 1
    assert json.loads(row['result'])['reason'] == 'receipt_generation_changed'
    assert (backend.vision_calls, len(stt_calls), backend.semantic_calls) == (1, int(stage == 'stt'), 0)
    journal = PublicationJournal(runner.runtime_root)
    token, pending = journal.pending()[0]
    assert pending['recording_id'] == record['recording_id']
    with runner.queues[stage].connect() as db:
        assert db.execute('SELECT status FROM recordings').fetchone()[0] == 'completed'
    # Exercise the existing replay owner after its bounded backoff, using the
    # journal's public scheduling port. No queue state or completion is edited.
    journal.defer(record['recording_id'], token, now=time.time()-3601)
    assert reconcile(runner) == 1
    assert journal.pending() == []
    assert path.read_bytes() == completed_receipt[0] and sibling.read_bytes() == sibling_bytes
    assert (backend.vision_calls, len(stt_calls), backend.semantic_calls) == (1, int(stage == 'stt'), 0)
    assert read_json(layout.index)['recordings'][0]['stages'][stage]['status'] == 'completed'
    assert runner.run_once({'recordings': [record]}, stage=stage)['results'] == []
    with runner.queues[stage].connect() as db:
        assert db.execute('SELECT attempts FROM recordings').fetchone()[0] == 1
    assert (backend.vision_calls, len(stt_calls), backend.semantic_calls) == (1, int(stage == 'stt'), 0)


def test_report_generation_change_is_pending_and_never_a_completed_or_failed_report(device_config, monkeypatch):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(record)['status'] == 'completed'
    path = runner._receipt(layout, record, 'report')
    path.unlink()
    before = layout.index.read_bytes()
    def changed(*args, **kwargs):
        raise ProjectionGenerationChanged(record['recording_id'], 'old verified generation', 'new concurrent generation')
    monkeypatch.setattr(runner, 'refresh_index', changed)
    result = runner.run_once({'recordings': [record]}, stage='report')['results'][0]
    assert result['status'] == read_json(path)['status'] == 'waiting_for_publication'
    assert result['reason'] == 'receipt_generation_changed'
    assert result['error_type'] == 'ProjectionGenerationChanged'
    with runner.queues['report'].connect() as db:
        assert db.execute('SELECT status FROM recordings').fetchone()[0] not in {'completed', 'failed'}
    assert layout.index.read_bytes() == before
    assert PublicationJournal(runner.runtime_root).pending()
    assert (backend.vision_calls, backend.semantic_calls) == (1, 1)


@pytest.mark.parametrize('error_kind', ['submitted_unknown', 'failed'])
def test_generation_race_never_hides_unknown_claim_or_real_partial_failure_and_replays_prior_window(
        device_config, monkeypatch, error_kind):
    from types import SimpleNamespace
    from test_readable_all_day import add_work_evidence
    from visioncortex.device_day_contract import artifact
    from visioncortex.multimodal_usage import SubmittedUnknown, UsageLedger
    from visioncortex.sqlite_store import connection
    capture(device_config)
    record, layout = item_and_layout(device_config)
    device_config['device_day'].update(readable_content_enabled=True, derived_video_retention=True,
                                       understanding_coverage_since_us=1)
    ledger = UsageLedger(device_config)
    claims = []
    class PartialModels(FakeModels):
        def understand(self, layout, recording, vision, context, key):
            result = super().understand(layout, recording, vision, context, key)
            value = result['understandings'][0]
            value['status'] = 'partial'
            evidence = add_work_evidence(layout, value, key, vision['segments'][0])
            path = layout.understanding/'ClipUnderstanding'/f'{key}.json'
            atomic_json(path, value)
            result.update(partial_observations=True, artifacts=[artifact(layout.root, path), *evidence])
            if error_kind == 'submitted_unknown':
                reservation = ledger.reserve(key)
                ledger.submitted_unknown(reservation)
                claims.append((key, reservation))
                error = SubmittedUnknown('submitted_unknown: owned deterministic request, never resend')
            else:
                error = ValueError('owned deterministic later window failure')
            error.partial_receipt = result
            raise error
    def extract(_source, target, *_args):
        target.write_bytes(b'owned exact selected partial window')
    monkeypatch.setattr('visioncortex.video_io.extract_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000))
    backend = PartialModels()
    runner = DeviceDayRunner(device_config, backend)
    for stage in ('retention', 'vision', 'stt'):
        assert runner.process(record, stage=stage)['status'] == 'completed'
    sibling = runner._receipt(layout, record, 'vision')
    sibling_bytes, value = sibling.read_bytes(), read_json(sibling)
    started, finished = threading.Event(), threading.Event()
    failures = []
    def writer():
        try:
            assert started.wait(5)
            atomic_json(sibling, value)
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()
    worker = threading.Thread(target=writer)
    worker.start()
    version = ReceiptProjection._source_version
    injected = []
    def generation(folder, stages):
        original = version(folder, stages)
        if folder.name == record['recording_id'] and not injected:
            injected.append(True)
            assert read_json(folder/'understanding.json')['status'] == error_kind
            started.set()
            assert finished.wait(5)
        return original
    monkeypatch.setattr(ReceiptProjection, '_source_version', staticmethod(generation))
    try:
        result = runner.run_once({'recordings': [record]}, stage='understanding')['results'][0]
    finally:
        started.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert result['status'] == error_kind
    path = runner._receipt(layout, record, 'understanding')
    native_bytes, native = path.read_bytes(), read_json(path)
    assert native['status'] == error_kind and native['partial_observations']
    assert all(verify_artifact(layout.root, ref) for ref in native['artifacts'])
    with runner.queues['understanding'].connect() as db:
        row = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert row['status'] == ('blocked' if error_kind == 'submitted_unknown' else 'failed')
    assert row['blocked_reason'] == ('submitted_unknown' if error_kind == 'submitted_unknown' else None)
    assert json.loads(row['result'])['status'] == error_kind
    assert backend.semantic_calls == backend.vision_calls == 1
    journal = PublicationJournal(runner.runtime_root)
    token, _ = journal.pending()[0]
    journal.defer(record['recording_id'], token, now=time.time()-3601)
    assert reconcile(runner) == 1
    assert journal.pending() == []
    assert path.read_bytes() == native_bytes and sibling.read_bytes() == sibling_bytes
    index = read_json(layout.index)
    assert index['understandings'][0]['status'] == 'partial'
    candidate = index['segments'][0]
    selected = next(item for item in candidate['retention_intervals'] if item['video_value'] == 'selected')
    assert (layout.root/selected['video']['path']).read_bytes() == b'owned exact selected partial window'
    entries = read_json(layout.understanding/'Index.json')['entries']
    child = next(row for row in entries if row['segment_id'] == selected['segment_id'])
    assert read_json(layout.root/child['content'])['understanding']['windows'][0]['work_activities']
    assert runner.run_once({'recordings': [record]}, stage='understanding')['results'] == []
    assert backend.semantic_calls == backend.vision_calls == 1
    with runner.queues['understanding'].connect() as db:
        after = dict(db.execute('SELECT * FROM recordings').fetchone())
    assert after == row
    if error_kind == 'submitted_unknown':
        key, reservation = claims[0]
        with connection(ledger.path, readonly=True) as db:
            assert db.execute('SELECT count(*) FROM calls').fetchone()[0] == 1
            assert tuple(db.execute('SELECT input_key,reservation_id FROM request_claims').fetchone()) == (key, reservation)
            assert db.execute('SELECT status FROM calls').fetchone()[0] == 'submitted_unknown'
        with pytest.raises(SubmittedUnknown):
            ledger.reserve(key)
        assert len(claims) == 1


def test_generation_defer_is_explicit_at_process_and_refresh_ports_not_flock_busy(device_config, monkeypatch):
    from contextlib import contextmanager
    capture(device_config)
    record, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    changed = ProjectionGenerationChanged(record['recording_id'], 'old', 'new')
    @contextmanager
    def race(*_args, **_kwargs):
        raise changed
        yield
    monkeypatch.setattr('visioncortex.device_day.exclusive', race)
    with pytest.raises(ProjectionGenerationChanged) as error:
        runner._guarded_refresh_index(layout)
    assert error.value is changed
    result = runner._guarded_process(record, stage='stt', retry=False)
    assert result == {'recording_id': record['recording_id'], 'status': 'waiting_for_publication', **changed.details()}


def test_existing_journal_preserves_exact_generation_reason_for_retry(device_config, monkeypatch):
    from visioncortex.sqlite_store import connection
    capture(device_config)
    record, _layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    journal = PublicationJournal(runner.runtime_root)
    journal.begin(record)
    changed = ProjectionGenerationChanged(record['recording_id'], 'old', 'new')
    def race(*_args, **_kwargs):
        raise changed
    monkeypatch.setattr(runner, 'refresh_index', race)
    assert reconcile(runner) == 0
    assert len(journal.pending()) == 1
    with connection(journal.path, readonly=True) as db:
        reason = json.loads(db.execute('SELECT reason FROM retries').fetchone()[0])
    assert reason == changed.details()


def test_inplace_previous_publication_becomes_pending_for_new_stage_generation_then_recovers(device_config, monkeypatch):
    from visioncortex.device_day_inplace import marker, progress, receipt
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    record, layout = item_and_layout(device_config)
    backend = FakeModels()
    stt_calls = []
    transcribe = backend.transcribe
    def count_transcribe(*args):
        stt_calls.append(True)
        return transcribe(*args)
    backend.transcribe = count_transcribe
    runner = DeviceDayRunner(device_config, backend)
    assert runner.process(record, stage='vision')['status'] == 'completed'
    prior = runner.process(record, stage='retention')
    assert prior['formal_publication'] == 'completed'
    assert read_json(marker(runner, record))['publication'] == 'completed'
    assert progress(device_config)[0]['label'] == '正式索引已发布'
    old_files = {path: path.read_bytes() for path in (layout.index, layout.comments/'TimeIndex.json',
        layout.reports/'LaboratoryDailyReport.json', layout.reports/'LaboratoryDailyReport.html')}
    old_native = {stage: runner._receipt(layout, record, stage).read_bytes() for stage in ('vision', 'retention')}
    sibling = runner._receipt(layout, record, 'vision')
    vision = read_json(sibling)
    started, finished = threading.Event(), threading.Event()
    failures, stt_bytes, injected = [], [], []
    def writer():
        try:
            assert started.wait(5)
            atomic_json(sibling, vision)
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()
    worker = threading.Thread(target=writer)
    worker.start()
    version = ReceiptProjection._source_version
    def generation(folder, stages):
        observed = version(folder, stages)
        if folder.name == record['recording_id'] and not injected:
            injected.append(True)
            assert receipt(runner, record, 'publication')['status'] == 'publishing'
            assert read_json(marker(runner, record))['publication'] == 'completed'
            stt_bytes.append(runner._receipt(layout, record, 'stt').read_bytes())
            assert receipt(runner, record, 'stt')['status'] == 'completed'
            started.set()
            assert finished.wait(5)
        return observed
    monkeypatch.setattr(ReceiptProjection, '_source_version', staticmethod(generation))
    try:
        result = runner.process(record, stage='stt')
    finally:
        started.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert result['status'] == 'completed' and result['formal_publication'] == 'pending', result
    assert result['reason'] == 'receipt_generation_changed'
    assert runner._receipt(layout, record, 'stt').read_bytes() == stt_bytes[0]
    assert receipt(runner, record, 'stt')['status'] == 'completed'
    # Preserve the existing verified-producer publication_filter semantics;
    # current lifecycle/progress is pending, not a stale completed label.
    assert receipt(runner, record, 'publication')['status'] == 'publishing'
    current = read_json(marker(runner, record))
    assert current['publication'] == 'pending'
    assert current['publication_defer']['reason'] == 'receipt_generation_changed'
    assert current['publication_defer']['expected_generation'] != current['publication_defer']['observed_generation']
    visible = progress(device_config)[0]
    assert visible['publication_status'] == 'pending' and visible['label'] != '正式索引已发布'
    assert visible['publication_defer'] == current['publication_defer']
    assert {path: path.read_bytes() for path in old_files} == old_files
    for stage, raw in old_native.items():
        assert runner._receipt(layout, record, stage).read_bytes() == raw
    for stage in ('vision', 'retention', 'stt'):
        assert all(verify_artifact(layout.root, ref) for ref in receipt(runner, record, stage)['artifacts'])
    journal = PublicationJournal(runner.runtime_root)
    token, _ = journal.pending()[0]
    journal.defer(record['recording_id'], token, now=time.time()-3601)
    assert reconcile(runner) == 1
    assert journal.pending() == []
    assert receipt(runner, record, 'publication')['status'] == 'completed'
    assert read_json(marker(runner, record))['publication'] == 'completed'
    assert read_json(marker(runner, record))['publication_defer'] is None
    visible = progress(device_config)[0]
    assert visible['label'] == '正式索引已发布' and 'publication_defer' not in visible
    assert runner._receipt(layout, record, 'stt').read_bytes() == stt_bytes[0]
    assert (backend.vision_calls, backend.semantic_calls, len(stt_calls)) == (1, 0, 1)
    # Repeat with the cached completed native STT: None -> dict -> None ->
    # dict must remain a lifecycle roundtrip, never a metadata merge failure.
    latest_files = {path: path.read_bytes() for path in old_files}
    started, finished = threading.Event(), threading.Event()
    injected.clear()
    worker = threading.Thread(target=writer)
    worker.start()
    try:
        repeated = runner.process(record, stage='stt')
    finally:
        started.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert repeated['status'] == 'completed' and repeated['formal_publication'] == 'pending', repeated
    assert read_json(marker(runner, record))['publication_defer']['reason'] == 'receipt_generation_changed'
    assert {path: path.read_bytes() for path in latest_files} == latest_files
    assert progress(device_config)[0]['publication_status'] == 'pending'
    token, _ = journal.pending()[0]
    journal.defer(record['recording_id'], token, now=time.time()-3601)
    assert reconcile(runner) == 1 and journal.pending() == []
    assert read_json(marker(runner, record))['publication_defer'] is None
    assert progress(device_config)[0]['label'] == '正式索引已发布'
    assert stt_bytes == [stt_bytes[0], stt_bytes[0]]
    assert runner._receipt(layout, record, 'stt').read_bytes() == stt_bytes[0]
    assert (backend.vision_calls, backend.semantic_calls, len(stt_calls)) == (1, 0, 1)

"""New clock/snapshot recovery counterexamples; no model, real camera or NAS."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

from PIL import Image
import pytest

from visioncortex.lab.archive_layout import contract_bytes
from visioncortex.lab.domain import instant
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.timeline_policy import ZONE
from visioncortex.lab.timeline_publication import TimelinePublication

from .test_person_report import scene, media
from .test_timeline_native_artifacts import submit

__all__ = ['scene']


def publisher(s):
    archive = s['settings'].incoming
    (archive / '.System').mkdir()
    (archive / '.System/ArchiveContract.json').write_bytes(contract_bytes())
    settings = replace(s['settings'], source_root=archive, unified_archive_required=True, timeline_publish=True)
    return TimelinePublication(s['runtime'], settings)


def test_rollout_cutoff_keeps_old_snapshots_and_publishes_new_revision_only(scene):
    s=scene
    source=media(s,'rollout-photo',kind='photo')
    submit(s,source,'ocr',[dict(kind='reading',text='first candidate',
        spans=[dict(media_id=source['media_id'],start_seconds='0',end_seconds='0')],
        payload=dict(raw_tokens=[dict(text='12.3')],fields=[]))])
    old=s['runtime'].views.snapshot('2026-10-29')
    configured=publisher(s)
    cutoff=datetime.now(timezone.utc)
    rollout=TimelinePublication(s['runtime'],configured.settings,not_before=cutoff)
    assert rollout.tick() is None
    second=media(s,'rollout-late-photo',kind='photo')
    submit(s,second,'ocr',[dict(kind='reading',text='second candidate',
        spans=[dict(media_id=second['media_id'],start_seconds='0',end_seconds='0')],
        payload=dict(raw_tokens=[dict(text='12.4')],fields=[]))])
    new=s['runtime'].views.snapshot('2026-10-29')
    assert new['revision']>old['revision']
    actual=rollout.tick();assert actual['revision']==new['revision']
    assert rollout.tick() is None  # Older unpublished revisions remain excluded.
    with s['core'].db.transaction(readonly=True) as conn:
        retained=conn.execute('SELECT document FROM hj_daily_report WHERE customer_id=%s AND day=%s AND revision=%s',
            (s['core'].customer,'2026-10-29',old['revision'])).fetchone()['document']
    assert retained==old['document']


def test_rollout_cutoff_requires_explicit_timezone(scene):
    with pytest.raises(ValueError,match='timezone'):
        TimelinePublication(scene['runtime'],scene['settings'],not_before='2026-10-30T00:00:00')


def test_new_reported_but_unverified_legacy_time_publishes_only_received_partition(scene, tmp_path):
    s = scene
    path = s['settings'].incoming / 'legacy-reported-clock.png'
    Image.new('RGB', (4, 4), 'white').save(path)
    reported = '2001-02-03T09:00:00+08:00'
    source = Ingest(s['core'], s['settings']).accept(dict(capture_id='reported-unverified-clock',
        source_instance='fixture-source-clock', source_revision='fixture/1', kind='photo',
        relative_path=path.name, camera_id='fixture-camera-0', captured_at=reported,
        time_basis=dict(basis='filename', uncertainty_us=0), source_kind='fixture',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Reproduce a persisted row from before clock support was checked at admit.
    # The immutable producer receipt and its reported timestamp stay unchanged.
    with s['core'].db.transaction() as conn:
        conn.execute('UPDATE hj_timeline_media SET start_at=%s,end_at=%s WHERE customer_id=%s AND media_id=%s',
            (reported, reported, s['core'].customer, source['media_id']))
    result = submit(s, source, 'ocr', [dict(kind='reading', text='24.2 candidate',
        spans=[dict(media_id=source['media_id'], start_seconds='0', end_seconds='0')],
        payload=dict(raw_tokens=[dict(text='24.2')], fields=[]))])
    with s['core'].db.transaction(readonly=True) as conn:
        received = conn.execute('SELECT received_at FROM capture WHERE customer_id=%s AND id=%s',
            (s['core'].customer, source['capture_id'])).fetchone()['received_at']
    day = str(received.astimezone(ZONE).date())
    snapshot = s['runtime'].views.snapshot(day)
    assert snapshot['document']['publication_media_ids'] == [source['media_id']]
    publication = publisher(s)
    actual = publication.tick()
    archive = s['settings'].incoming
    bindings = json.loads((archive / f'{day}_fixture-camera-0/Comment/Bindings.json').read_text())
    original = bindings['media_sources'][source['media_id']]
    assert original['source_reported_at'] == reported
    assert original['capture_time_status'] == 'unknown'
    assert original['archive_day_basis'] == 'capture_received_at_partition_only'
    assert original['receipt_day_is_not_physical_event_time']
    assert {o['activity_id'] for o in bindings['observations']} == set(result['activity_ids'])
    ref = bindings['comment_artifacts'][0]
    evidence = json.loads((archive / ref['evidence']).read_text())
    assert evidence['event_at'] is None and evidence['archive_day_is_not_physical_event_time']
    assert '/09-00-00.000_' in ref['evidence']
    assert evidence['folder_time_basis']=='source_reported_display_only_uncalibrated'
    assert not (archive / '2001-02-03_fixture-camera-0').exists()
    (tmp_path / 'UnsupportedReportedClockProof.json').write_text(json.dumps(dict(
        snapshot=snapshot, publication=actual, bindings=bindings, actual_model_calls=0), indent=2) + '\n')


def test_new_interrupted_snapshot_publish_keeps_source_set_after_older_transaction_commits(scene, monkeypatch, tmp_path):
    s = scene
    first = media(s, 'snapshot-initial-photo', kind='photo')
    submit(s, first, 'ocr', [dict(kind='reading', text='initial candidate',
        spans=[dict(media_id=first['media_id'], start_seconds='0', end_seconds='0')],
        payload=dict(raw_tokens=[dict(text='24.2')], fields=[]))])
    publication = publisher(s)
    archive = s['settings'].incoming
    original_transaction = s['core'].db.transaction
    with original_transaction() as late_transaction:
        transaction_started = late_transaction.execute('SELECT now() AS at').fetchone()['at']
        snapshot = s['runtime'].views.snapshot('2026-10-29')
        assert snapshot['document']['publication_media_ids'] == [first['media_id']]
        assert transaction_started <= instant(snapshot['created_at'])

        def interrupt(*args, **kwargs):
            raise OSError('fixture I/O interruption after immutable artifact writes')

        with monkeypatch.context() as patch:
            patch.setattr(publication, '_write_latest', interrupt)
            with pytest.raises(OSError, match='I/O interruption'):
                publication.tick()
        before = {str(p.relative_to(archive)): p.read_bytes() for p in archive.rglob('*') if p.is_file()}
        assert any('/Associations/' in p for p in before)

        @contextmanager
        def same_started_transaction(**kwargs):
            yield late_transaction

        # Actual intake uses a transaction begun before the report but commits
        # after its failed publish. PostgreSQL now()/received_at is therefore
        # older than the snapshot: a timestamp cutoff alone cannot isolate it.
        with monkeypatch.context() as patch:
            patch.setattr(s['core'].db, 'transaction', same_started_transaction)
            late = media(s, 'snapshot-late-camera-photo', kind='photo', camera=1)
    with original_transaction(readonly=True) as conn:
        received = conn.execute('SELECT received_at FROM capture WHERE customer_id=%s AND id=%s',
            (s['core'].customer, late['capture_id'])).fetchone()['received_at']
    assert received <= instant(snapshot['created_at'])
    recovered = publication.tick()
    assert recovered['revision'] == snapshot['revision']
    assert all((archive / p).read_bytes() == raw for p, raw in before.items())
    assert not (archive / '2026-10-29_fixture-camera-1').exists()
    first_bindings = json.loads((archive / '2026-10-29_fixture-camera-0/Comment/Bindings.json').read_text())
    assert set(first_bindings['media_sources']) == {first['media_id']}
    assert first_bindings['source_set_basis'] == 'snapshot_publication_media_ids'
    revised = s['runtime'].views.snapshot('2026-10-29')
    assert revised['revision'] > snapshot['revision']
    assert set(revised['document']['publication_media_ids']) == {first['media_id'], late['media_id']}
    later = publication.tick()
    assert later['revision'] == revised['revision']
    assert (archive / '2026-10-29_fixture-camera-1/Comment/Bindings.json').exists()
    (tmp_path / 'InterruptedPublishSourceSetProof.json').write_text(json.dumps(dict(
        snapshot=snapshot, new_snapshot=revised, recovered=recovered, later=later,
        transaction_started=transaction_started.isoformat(), late_capture_received_at=received.isoformat(),
        previously_written_sha256={p:hashlib.sha256(raw).hexdigest() for p, raw in before.items()},
        original_snapshot_bytes_unchanged=True, actual_model_calls=0), indent=2) + '\n')

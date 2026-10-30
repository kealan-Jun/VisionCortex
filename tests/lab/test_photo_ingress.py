"""Photos-only source admission, generated local images; no models or NAS."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import time
from uuid import uuid4

from PIL import Image
import pytest

from visioncortex.lab.components import pipeline_module
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.photo_ingress import PhotoIngress, SCHEMA, candidates
from visioncortex.lab.storage import publication_config

from .test_capture_equipment_intake import enable
from .test_operator_scan import scene

__all__ = ['scene']


def configuration(settings, tmp_path):
    path = tmp_path/'photo-ingress.json'
    path.write_text(json.dumps(dict(schema_version=SCHEMA, enabled=True, revision='generated-fixture/1',
        source_root=str(settings.incoming), not_before_day='2026-10-29', max_photos_per_tick=4)))
    path.chmod(0o600)
    return path


def photos(root, camera, moment, *, legacy=False, day='2026-10-29'):
    folder = root/(day+'_'+camera)/'MetaData'
    folder = folder/moment if legacy else folder/'Photos'/moment
    folder.mkdir(parents=True)
    for suffix in ['', '_001', '_002']:
        path = folder/(day.replace('-','')+'_'+moment.replace('-','')+suffix+'.jpg')
        Image.new('RGB',(8,8),'white').save(path)
        os.utime(path,(time.time()-2,time.time()-2))
    return folder


def test_photo_only_discovery_keeps_original_camera_newest_scope_and_default_off(tmp_path, monkeypatch):
    monkeypatch.delenv('VC_LAB_PHOTO_INGRESS_MANIFEST', raising=False)
    assert PhotoIngress(None, None).step() == dict(enabled=False)
    root=tmp_path/'runtime'
    root.mkdir()
    settings=SimpleNamespace(runtime_root=root,source_root=None,nas_marker_sha256=None,
        incoming=root,cache=root/'cache')
    photos(root,'arbitrary.camera_cam07','17-58-08')
    photos(root,'unlisted-camera_cam03','17-59-01')
    photos(root,'do-not-scan-old','09-00-00',legacy=True)
    photos(root,'do-not-backfill','10-00-00',day='2026-10-28')
    doc=dict(not_before_day='2026-10-29')
    with pipeline_module('device_day_write_guard').storage_guard(publication_config(settings)) as guard:
        found=candidates(root,doc,guard)
    assert len(found)==2
    assert found[0]['camera']=='unlisted-camera_cam03'
    assert {v['camera'] for v in found}=={'arbitrary.camera_cam07','unlisted-camera_cam03'}
    assert all('/MetaData/Photos/' in str(v['path']) and v['cost']==3 for v in found)
    assert not any('do-not-' in str(v['path']) for v in found)


def test_photo_ingress_actual_pg_is_bounded_idempotent_and_waits_upload_completion(scene,tmp_path):
    s=scene
    settings=enable(s,tmp_path,video=False)
    # source_root must remain the existing local runtime root for this fixture.
    settings=replace(settings,source_root=settings.incoming)
    newer=photos(settings.incoming,'generic-new-camera','17-58-08')
    older=photos(settings.incoming,'generic-new-camera','17-47-37')
    legacy=photos(settings.incoming,'generic-new-camera','17-59-00',legacy=True)
    snapshots={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in settings.incoming.rglob('*.jpg')}
    marker=newer/'.uploading'
    marker.write_text('generated upload gate')
    ingress=PhotoIngress(s['core'],settings,configuration(settings,tmp_path))
    first=ingress.tick()
    assert first['processed_photos']==3 and first['attempted_photos']==3
    assert first['errors'][0]['relative_path'].startswith(str(newer.relative_to(settings.incoming)))
    marker.unlink()
    second=ingress.step()
    assert second['processed_photos']==3 and second['attempted_photos']==3
    assert ingress.tick()['processed_photos']==0
    with s['core'].db.transaction(readonly=True) as conn:
        rows=conn.execute('SELECT payload FROM inbox WHERE customer_id=%s',(s['core'].customer,)).fetchall()
        assert len(rows)==6
        assert all('/MetaData/Photos/' in r['payload']['relative_path'] for r in rows)
        assert all(r['payload']['camera_id']=='generic-new-camera' for r in rows)
        assert all(r['payload']['time_basis']['clock_sync_valid'] is False for r in rows)
        assert all(r['payload']['group_expected_photos'] is None for r in rows)
        assert conn.execute('SELECT count(*) AS n FROM processing_job WHERE customer_id=%s',(s['core'].customer,)).fetchone()['n']==6
    assert Path(settings.runtime_root/'photo-ingress/Checkpoint.sqlite3').is_file()
    for path,snapshot in snapshots.items():
        assert (path.read_bytes(),path.stat().st_mtime_ns)==snapshot
    assert not (older/'PhotoReceipt.json').exists() and not (legacy/'PhotoReceipt.json').exists()


def test_six_member_receipt_resumes_four_plus_two_and_replays_without_duplicate_pg_work(scene, tmp_path):
    """Real local bytes/PG admission; no QR decoder, OCR or NAS is invoked."""
    settings = enable(scene, tmp_path, video=False)
    settings = replace(settings, source_root=settings.incoming)
    camera = 'fixture-six-member-camera'
    folder = settings.incoming / ('2026-10-29_' + camera) / 'MetaData/Photos/12-00-00'
    folder.mkdir(parents=True)
    members = []
    for index in range(6):
        filename = '20261029_120000' + (f'_{index:03d}' if index else '') + '.jpg'
        path = folder / filename
        Image.new('RGB', (8, 8), (index * 30, 40, 90)).save(path)
        os.utime(path, (time.time() - 2, time.time() - 2))
        members.append(dict(filename=filename, capture_id=str(uuid4()), revision='1',
            captured_at='2026-10-29T12:00:00+08:00',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    burst_id = str(uuid4())
    receipt = folder / 'PhotoReceipt.json'
    document = dict(schema_version='field-photo-receipt/1', camera_id=camera,
        measurement=dict(expected_photos=6, burst_id=burst_id), photos=members, source_kind='fixture')
    # The complete receipt is validated before taking even its first four.
    bad_members = [dict(member) for member in members]
    bad_members[-1]['filename'] = '../escape.jpg'
    oversized = (members * 3)[:17]
    for invalid, reason in (
        (dict(document, photos=bad_members), 'own capture directory'),
        (dict(document, photos=oversized,
              measurement=dict(expected_photos=17, burst_id=burst_id)), 'membership/count mismatch'),
    ):
        receipt.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=reason):
            Ingest(scene['core'], settings).photo_receipt(receipt, member_limit=4)
    with scene['core'].db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM capture WHERE customer_id=%s',
            (scene['core'].customer,)).fetchone()['n'] == 0
    receipt.write_text(json.dumps(document))
    os.utime(receipt, (time.time() - 2, time.time() - 2))
    originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    manifest = configuration(settings, tmp_path)
    checkpoint = settings.runtime_root / 'photo-ingress/Checkpoint.sqlite3'
    checkpoint.parent.mkdir()
    # Exercise upgrade of the pre-existing local four-column checkpoint.
    with sqlite3.connect(checkpoint) as db:
        db.execute('CREATE TABLE items(path TEXT PRIMARY KEY, signature TEXT NOT NULL, status TEXT NOT NULL, retry_after REAL NOT NULL)')

    def progress():
        with sqlite3.connect(checkpoint) as db:
            return db.execute('SELECT status,member_offset FROM items').fetchone()

    first = PhotoIngress(scene['core'], settings, manifest).tick()
    assert (first['processed_photos'], first['attempted_photos'], first['duplicates']) == (4, 4, 0)
    assert not first['errors'] and progress() == ('partial', 4)
    assert first['pending'][0]['expected_photos'] == 6
    # A new instance models a process restart between ticks.
    second = PhotoIngress(scene['core'], settings, manifest).tick()
    assert (second['processed_photos'], second['attempted_photos'], second['duplicates']) == (2, 2, 0)
    assert not second['errors'] and progress() == ('accepted', 6)

    # Simulate committed Core members with an older durable local cursor after
    # a crash. Replaying the last four consumes the tick budget, not new jobs.
    with sqlite3.connect(checkpoint) as db:
        db.execute("UPDATE items SET member_offset=2,status='partial'")
    replay = PhotoIngress(scene['core'], settings, manifest).tick()
    assert (replay['processed_photos'], replay['attempted_photos'], replay['duplicates']) == (0, 4, 4)
    assert not replay['errors'] and progress() == ('accepted', 6)
    # A changed source signature starts at zero, preserving identical original
    # bytes and capture identities rather than trusting the former offset.
    os.utime(receipt, (time.time() - 1, time.time() - 1))
    reset = PhotoIngress(scene['core'], settings, manifest).tick()
    assert (reset['processed_photos'], reset['attempted_photos'], reset['duplicates']) == (0, 4, 4)
    assert not reset['errors'] and progress() == ('partial', 4)
    finish = PhotoIngress(scene['core'], settings, manifest).tick()
    assert (finish['processed_photos'], finish['attempted_photos'], finish['duplicates']) == (0, 2, 2)
    assert not finish['errors'] and progress() == ('accepted', 6)
    assert PhotoIngress(scene['core'], settings, manifest).tick()['attempted_photos'] == 0
    with scene['core'].db.transaction(readonly=True) as conn:
        customer = (scene['core'].customer,)
        rows = conn.execute('SELECT payload FROM inbox WHERE customer_id=%s', customer).fetchall()
        assert len(rows) == 6
        assert {row['payload']['capture_id'] for row in rows} == {m['capture_id'] for m in members}
        assert all(row['payload']['group_expected_photos'] == 6
            and row['payload']['session_id'] == burst_id
            and row['payload']['batch_basis'] == 'producer_photo_receipt' for row in rows)
        for table in ('capture', 'processing_job', 'hj_timeline_task'):
            assert conn.execute(f'SELECT count(*) AS n FROM {table} WHERE customer_id=%s', customer).fetchone()['n'] == 6
        assert conn.execute("SELECT count(*) AS n FROM processing_job WHERE customer_id=%s AND worker_kind='qr'", customer).fetchone()['n'] == 6
        assert conn.execute("SELECT count(*) AS n FROM hj_timeline_task WHERE customer_id=%s AND kind='ocr' AND status='awaiting_provider'", customer).fetchone()['n'] == 6
        assert conn.execute('SELECT count(*) AS n FROM capture_session WHERE customer_id=%s', customer).fetchone()['n'] == 1
    for path, (raw, modified) in originals.items():
        assert path.read_bytes() == raw
        if path != receipt:
            assert path.stat().st_mtime_ns == modified

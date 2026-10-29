"""Photos-only source admission, generated local images; no models or NAS."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from PIL import Image

from visioncortex.lab.components import pipeline_module
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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
import time
from uuid import uuid4

import pytest
import qrcode

from visioncortex.lab.domain import Conflict
from visioncortex.lab.equipment import validate_catalogue
from visioncortex.lab.ingest import Ingest
from visioncortex.lab.media import Media
from visioncortex.lab.cache import prune
from visioncortex.lab.processing import Worker


def fixture_catalogue():
    # Generated identities test allocation/history without any site catalogue.
    equipment = []
    for n in range(1, 33):
        device_id = str(uuid4())
        equipment.append(dict(device_id=device_id, qr_payload=device_id,
                              device_code=f'EQ-{n:06d}', name=f'fixture-instrument-{n}',
                              physical_verification='pending', ordinal=n))
    return dict(schema_version='vc-lab-equipment-catalogue/1',
                source={'sha256': hashlib.sha256(b'generated catalogue fixture').hexdigest()},
                equipment=equipment)


def test_private_site_scope_source_contract():
    # This optional check is real supplied source validation, never synthetic acceptance.
    from ._support import private_fixture
    data = json.loads(private_fixture('site-equipment-catalogue.json').read_text())
    expectations = json.loads(private_fixture('site-expectations.json').read_text())
    assert len(validate_catalogue(data)) == expectations['equipment_count']
    assert all(r['physical_verification'] == 'pending' for r in data['equipment'])
    scope = json.loads(private_fixture('site-scope.json').read_text())
    assert scope['phase_one'] == expectations['phase_one']
    assert 'network_design' in scope['responsibility_exclusions']


def test_supplied_uuid_reservation_activation_retirement_preserve_history(core, settings):
    catalogue = fixture_catalogue()
    reserved = core.reserve_equipment_catalogue('reserve', catalogue)
    assert reserved['activated'] == 0
    row = catalogue['equipment'][0]
    assert core.resolve_qr(row['device_id'], '2026-10-23T12:00:00+08:00')['reason'] == 'physical_verification_pending'
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM entity WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 0
    active = core.activate_equipment('activate', row['device_id'], verified_at='2026-10-23T10:00:00+08:00', actor='fixture-only', evidence={'fixture':'simulated_physical_check'})
    assert active['device_id'] == row['device_id'] and active['device_code']=='EQ-000001'
    assert core.activate_equipment('activate', row['device_id'], verified_at='2026-10-23T10:00:00+08:00', actor='fixture-only', evidence={'fixture':'simulated_physical_check'}) == active
    assert core.resolve_qr(row['device_id'], '2026-10-23T09:00:00+08:00')['association_status']=='needs_review'
    assert core.resolve_qr(row['device_id'], '2026-10-23T11:00:00+08:00')['target_id'] == row['device_id']
    with ThreadPoolExecutor(max_workers=5) as pool:
        new = list(pool.map(lambda n:core.issue_equipment(f'new-{n}', actor='fixture'),range(5)))
    assert sorted(r['device_code'] for r in new) == [f'EQ-{n:06d}' for n in range(33,38)]
    core.retire_equipment('retire', row['device_id'], at='2026-10-23T12:00:00+08:00', actor='fixture', reason='fixture replacement')
    assert core.resolve_qr(row['device_id'], '2026-10-23T11:00:00+08:00')['target_id'] == row['device_id']
    assert core.resolve_qr(row['device_id'], '2026-10-23T13:00:00+08:00')['association_status']=='needs_review'
    with pytest.raises(Conflict):
        core.assign_tag('reactivate', active['label_id'], row['device_id'], '2026-10-23T13:00:00+08:00', actor='fixture', reason='forbidden reuse')


def test_five_parallel_sources_read_generated_uuid_qr_and_do_not_open_material_stock(core, settings):
    rows = fixture_catalogue()
    core.reserve_equipment_catalogue('reserve',rows)
    expected = []
    for n, row in enumerate(rows['equipment'][:5]):
        core.activate_equipment(f'act-{n}',row['device_id'],verified_at='2026-10-23T09:00:00+08:00',actor='fixture',evidence={'fixture':True})
        folder=settings.incoming/f'camera-{n}';folder.mkdir()
        qrcode.make(row['device_id']).save(folder/'image.png')
        expected.append(dict(capture_id='photo',source_instance=f'camera-{n}',source_revision='1',kind='photo',relative_path=f'camera-{n}/image.png',captured_at='2026-10-23T10:00:00+08:00',time_basis={'basis':'fixture'},source_kind='fixture'))
    with ThreadPoolExecutor(max_workers=5) as pool:
        captured=list(pool.map(lambda r:Ingest(core,settings).accept(r),expected))
    assert len({r['capture_id'] for r in captured}) == 5
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(lambda _:Worker(core,settings).run_once(),range(10)))
    with core.db.transaction(readonly=True) as conn:
        observations=conn.execute("SELECT payload FROM observation WHERE customer_id=%s AND kind='qr'",(core.customer,)).fetchall()
        assert len(observations)==5
        assert {r['payload']['tag_associations'][0]['target_id'] for r in observations} == {r['device_id'] for r in rows['equipment'][:5]}
        assert conn.execute('SELECT count(*) AS n FROM quantity_ledger_entry WHERE customer_id=%s',(core.customer,)).fetchone()['n']==0


def test_ttl_only_removes_verified_working_copy_and_pins_running_reader(core, settings, photo):
    captured=Ingest(core,settings).accept(photo)
    cached=Media(core,settings).locate(captured['media_id'])['path']
    old=time.time()-90000;os.utime(cached,(old,old))
    job=Worker(core,settings).claim()
    assert prune(core,settings)['pinned']==1 and cached.exists()
    with core.db.transaction() as conn:
        conn.execute("UPDATE processing_job SET status='queued',lease_token=NULL,lease_until=NULL WHERE customer_id=%s AND id=%s",(core.customer,job['id']))
    original=settings.incoming/photo['relative_path'];raw=original.read_bytes();original.unlink()
    assert prune(core,settings)['original_unverified']==1 and cached.exists()
    original.write_bytes(raw)
    result=prune(core,settings)
    assert result['removed']==1 and not cached.exists()
    assert Media(core,settings).locate(captured['media_id'])['path']==original


def test_directory_triplet_partial_arrival_is_incremental_and_never_mutates_originals(core, settings):
    settings=replace(settings,photo_batch_mode='directory_triplet',photo_cameras=('camera-one',))
    folder=settings.incoming/'2026-10-23_camera-one/MetaData/10-00-00';folder.mkdir(parents=True)
    def add(n):
        p=folder/f'20261023_100000_{n}.png';qrcode.make('fixture-'+str(n)).save(p)
        os.utime(p,(time.time()-2,time.time()-2))
        return p,hashlib.sha256(p.read_bytes()).hexdigest()
    first,digest=add(1)
    assert Ingest(core,settings).discover_once()['processed']==1
    add(2);add(3)
    result=Ingest(core,settings).discover_once()
    assert not result['errors'],result
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM capture WHERE customer_id=%s',(core.customer,)).fetchone()['n']==3
        assert conn.execute('SELECT count(*) AS n FROM capture_session WHERE customer_id=%s',(core.customer,)).fetchone()['n']==1
    assert hashlib.sha256(first.read_bytes()).hexdigest()==digest
    assert not (folder/'PhotoReceipt.json').exists()
    add(4)
    assert Ingest(core,settings).discover_once()['errors']


def test_native_original_bypasses_only_copy_bound_not_verified_source_identity(core, settings):
    settings=replace(settings,max_capture_bytes=10,max_native_bytes=1024)
    folder=settings.incoming/'2026-10-23_camera/MetaData/100000';folder.mkdir(parents=True)
    path=folder/'rgb.mp4';path.write_bytes(b'explicit-fixture-not-real-video'*8)
    receipt=dict(capture_id='sealed-fixture',source_instance='fixture-recorder',source_revision='1',kind='video',relative_path=path.relative_to(settings.incoming).as_posix(),captured_at='2026-10-23T10:00:00+08:00',time_basis={'basis':'fixture'},source_kind='fixture',native_record={'fixture_only':True})
    with pytest.raises(ValueError,match='bound'):
        Ingest(core,settings).accept(receipt)
    value=Ingest(core,settings).accept(receipt,native_original=True)
    assert Media(core,settings).locate(value['media_id'])['path']==path
    assert not list(settings.cache.rglob('*.mp4'))

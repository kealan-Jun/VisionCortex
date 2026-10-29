from dataclasses import replace
import json

import pytest

from visioncortex.lab.archive_layout import CONTRACT_PATH, day_directory, prepare_archive, verify_archive_contract
from visioncortex.lab.domain import Conflict
from visioncortex.lab.rules import Rules


def test_shared_day_boundary_camera_identity_and_contract():
    camera='lab.cam_'+('x'*105)
    assert day_directory(camera,'2026-10-25T16:00:00+00:00')=='2026-10-26_'+camera
    assert day_directory(camera,'2026-10-25T15:59:59.999999+00:00')=='2026-10-25_'+camera
    assert day_directory('lab_cam_1','2026-10-26T00:00:00+08:00')=='2026-10-26_lab_cam_1'
    with pytest.raises(ValueError):day_directory('../camera','2026-10-26T00:00:00+08:00')
    with pytest.raises(ValueError):day_directory('camera','2026-10-26T00:00:00')
    profile=Rules().profile('archive')
    assert profile['partition_by_processing_project'] is False
    assert profile['result_contract']['instrument_fields']==['wearer_id','device_model','device_no','qr_hash','photo_time','values']
    assert not any(name in profile['directories'] for name in ('VC','FR','VisionCortex','FieldRecognition'))


def test_prepare_archive_is_idempotent_and_contract_mismatch_blocks_writes(settings):
    configured=replace(settings,source_root=settings.runtime_root/'Archive',unified_archive_required=True)
    with pytest.raises(FileNotFoundError):configured.prepare()
    result=prepare_archive(configured)
    again=prepare_archive(configured)
    assert result==again
    assert result['customer_sources_started']==0 and result['data_created'] is False
    configured.prepare()
    root=configured.source_root
    assert verify_archive_contract(root)==root/CONTRACT_PATH
    assert sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())==[CONTRACT_PATH,'README.txt']
    contract=json.loads((root/CONTRACT_PATH).read_text())
    contract['partition_by_processing_project']=True
    (root/CONTRACT_PATH).write_text(json.dumps(contract))
    with pytest.raises(Conflict):configured.prepare()
    with pytest.raises(Conflict):prepare_archive(configured)
    assert not list(root.glob('20*'))


def _previous_archive(settings):
    import hashlib
    from visioncortex.lab.components import ROOT
    root=settings.runtime_root/'Archive'
    (root/'.System').mkdir(parents=True)
    old=(ROOT/'contracts/lab/archive-layout.json').read_bytes()
    (root/CONTRACT_PATH).write_bytes(old)
    (root/'README.txt').write_text('previous fixture instructions\n')
    return replace(settings,source_root=root,unified_archive_required=True),old,hashlib.sha256(old).hexdigest()


def test_contract_upgrade_keeps_history_and_refuses_existing_day_data(settings):
    from visioncortex.lab.archive_layout import upgrade_archive
    configured,old,digest=_previous_archive(settings)
    root=configured.source_root
    result=upgrade_archive(configured,digest)
    assert result['contract']=='vc-lab-unified-archive/2'
    assert result['historical_data_moved'] is False
    assert (root/'.System/ArchiveContracts'/digest/'ArchiveContract.json').read_bytes()==old
    assert (root/'.System/ArchiveContracts'/digest/'README.txt').read_text()=='previous fixture instructions\n'
    assert upgrade_archive(configured,digest)==result
    assert prepare_archive(configured)['customer_sources_started']==0
    (root/'2026-10-26_camera-one').mkdir()
    with pytest.raises(Conflict,match='already contains data'):
        upgrade_archive(configured,digest)


def test_contract_upgrade_resumes_after_metadata_write_interruption(settings,monkeypatch):
    from visioncortex.lab.archive_layout import upgrade_archive
    from visioncortex.lab.components import pipeline_module
    configured,old,digest=_previous_archive(settings)
    root=configured.source_root
    guard=pipeline_module('device_day_write_guard')
    original=guard.guarded_atomic
    fail=[True]
    def interrupted(path,blocks):
        if path==root/'README.txt' and fail[0]:
            fail[0]=False
            raise OSError('fixture interrupted after contract pointer update')
        return original(path,blocks)
    monkeypatch.setattr(guard,'guarded_atomic',interrupted)
    with pytest.raises(OSError):upgrade_archive(configured,digest)
    assert json.loads((root/CONTRACT_PATH).read_text())['schema_version']=='vc-lab-unified-archive/2'
    assert (root/'README.txt').read_text()=='previous fixture instructions\n'
    upgrade_archive(configured,digest)
    assert 'Comment/Audio' in (root/'README.txt').read_text()
    assert (root/'.System/ArchiveContracts'/digest/'ArchiveContract.json').read_bytes()==old


def test_native_comment_binding_is_isolated_and_rejects_source_drift():
    from visioncortex.lab.components import pipeline_module
    from visioncortex.lab.native_comment_layout import module,bind_source
    assert pipeline_module('device_day_contract').DIRECTORIES[-1]=='Comment'
    assert module('device_day_contract').DIRECTORIES[-1]=='Comment/Audio'
    assert pipeline_module('device_day_contract').DIRECTORIES[-1]=='Comment'
    with pytest.raises(ValueError,match='binding no longer matches'):
        bind_source('device_day_contract','drifted fixture source')


def test_format_examples_with_real_receipts_are_excluded_from_auto_intake(core, settings):
    from visioncortex.lab.ingest import Ingest
    for name in ('示例_留存格式', '示例_真实设备一天'):
        folder = settings.incoming/name/'2026-10-14_fixture-camera-02'/'MetaData'
        folder.mkdir(parents=True)
        (folder/'CaptureReceipt.json').write_text('invalid archive-only receipt')
        (folder/'example_recording_ready.json').write_text('invalid archive-only seal')
    result = Ingest(core, settings).discover_once()
    assert result['processed'] == 0 and result['errors'] == []
    with core.db.transaction(readonly=True) as conn:
        assert conn.execute('SELECT count(*) AS n FROM intake_item WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 0
        assert conn.execute('SELECT count(*) AS n FROM capture WHERE customer_id=%s', (core.customer,)).fetchone()['n'] == 0
    # Ordinary intake remains active outside the explicitly named examples.
    ordinary = settings.incoming/'ordinary'
    ordinary.mkdir()
    (ordinary/'CaptureReceipt.json').write_text('invalid live receipt')
    result = Ingest(core, settings).discover_once()
    assert len(result['errors']) == 1
    assert result['errors'][0]['relative_path'] == 'ordinary/CaptureReceipt.json'

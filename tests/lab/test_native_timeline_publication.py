"""Actual owned PG publication into a local synthetic archive, no native/model job."""
from dataclasses import replace
import json

from visioncortex.lab.archive_layout import contract_bytes
from visioncortex.lab.delivery import NativeDelivery
from visioncortex.lab.native_linked_layout import module
from visioncortex.lab.timeline_publication import TimelinePublication

from .test_person_report import scene,media,event,report
from .test_operator_scan import scan

__all__=['scene']


def test_camera_sidecars_from_real_snapshot_preserve_native_main_and_late_revision(scene,tmp_path):
    s=scene
    a=event(s,'camera0-observation',media(s,'camera0-observation',camera=0))
    other=event(s,'camera1-observation',media(s,'camera1-observation',camera=1))
    active=scan(s,'explicit-A',run_id=s['run'])
    snapshot=report(s)
    archive=s['settings'].incoming
    (archive/'.System').mkdir();(archive/'.System/ArchiveContract.json').write_bytes(contract_bytes())
    settings=replace(s['settings'],source_root=archive,unified_archive_required=True,timeline_publish=True)
    publication=TimelinePublication(s['runtime'],settings)
    first=publication.tick();assert first['revision']==snapshot['revision']
    paths=[archive/r['relative_path'] for r in first['files'] if '/Associations/' in r['relative_path']]
    assert len(paths)==2
    before={str(p):p.read_bytes() for p in paths}
    index={json.loads(p.read_text())['camera_key']:json.loads(p.read_text()) for p in paths}
    assert {o['activity_id'] for o in index['fixture-camera-0']['observations']}=={a['id'],active['activity_id']}
    assert {o['activity_id'] for o in index['fixture-camera-1']['observations']}=={other['id']}
    assert all(v['native_entry_refresh_status']=='pending_native_publication' for v in index.values())
    assert NativeDelivery(s['core'],settings).tick()=={'created':0,'dependencies_ready':0,'profile':'timeline-v1'}
    assert not list(archive.glob('*/LaboratoryDailyReport/LaboratoryDailyReport.json'))
    assert not list(archive.glob('*/MultimodalUnderstanding/*/Understanding.json'))
    assert publication.tick() is None
    late=scan(s,'late-B',value=s['devices'][1],when='09:00:02')
    revised=report(s);assert revised['revision']>snapshot['revision']
    second=publication.tick();assert second['revision']==revised['revision']
    assert all(p.read_bytes()==before[str(p)] for p in paths)
    from types import SimpleNamespace
    root=archive/'2026-10-29_fixture-camera-0'
    document=module('device_day_reports').day_document(SimpleNamespace(root=root,name=root.name),
        {'archive':root.name,'recordings':[],'segments':[],'understandings':[]},
        {'status':'pending'},readable={'视频留存统计':{}},entries=[],report_generated_at='fixture')
    assert document['关联时间线']['snapshot_revision']==revised['revision']
    assert document['关联时间线']['native_entry_refresh_status']=='included_by_native_builder'
    assert document['全天综合理解']['status']=='pending'
    (tmp_path/'publication-receipt.json').write_text(json.dumps({'first':first,'late':second,'late_scan':late,
        'main_refresh_requires_original_native_publisher':True},indent=2))

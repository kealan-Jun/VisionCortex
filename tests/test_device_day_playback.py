from copy import deepcopy

import pytest

from visioncortex.device_day_playback import build_playback, media_ms
from visioncortex.device_day_timeline import day_bounds

DAY = '2026-09-11'
BASE = day_bounds(DAY)[0]


def record(name, start, end):
    return {'recording_id': name, 'start_us': BASE+start*1_000_000, 'end_us': BASE+end*1_000_000,
            'sources': [{'kind': 'video', 'retained': {'path': 'MetaVideo/'+name+'.mp4'}}]}


def fixture():
    fp = {'archive': DAY+'_fp', 'recordings': [record('f', 100, 200)], 'segments': [
        {'segment_id': 'active', 'recording_id': 'f', 'start_us': BASE+120_000_000,
         'end_us': BASE+180_000_000, 'activity': 'active'}]}
    tp = {'archive': DAY+'_tp', 'recordings': [record('t1', 100, 150), record('t2', 150, 200)],
          'segments': [{'segment_id': 'inactive', 'recording_id': 't1', 'activity': 'inactive',
                        'start_us': BASE+100_000_000, 'end_us': BASE+150_000_000}]}
    return [(fp['archive'], fp), (tp['archive'], tp)]


def test_playback_keeps_inactive_view_and_unprocessed_next_slice():
    indexes = fixture()
    indexes[1][1]['recordings'].append({'recording_id':'unretained','start_us':None,'end_us':None})
    before = deepcopy(indexes)
    data = build_playback(DAY, indexes, DAY+'_fp', 'active', {'fp': 'first_person', 'tp': 'third_person'})
    records = data['cameras'][1]['recordings']
    assert data['cameras'][1]['unknown_time_recordings'] == 1
    assert [r['activity'] for r in records] == ['inactive', 'pending']
    assert records[0]['seek_points'] == [[BASE+120_000_000, 20], [BASE+150_000_000, 50]]
    assert records[1]['seek_points'] == [[BASE+150_000_000, 0], [BASE+180_000_000, 30]]
    assert not data['cross_camera_alignment_verified']
    assert indexes == before


def test_inverse_uses_actual_csv_clock_knots():
    from visioncortex.device_day_models import capture_us
    clock = {'basis': 'recorder_csv_interpolation', 'origin_us': BASE,
             'points': [[0, BASE], [1000, BASE+1_001_000], [2000, BASE+2_001_500]]}
    for ms in (0, 100, 999, 1001, 1800, 2200):
        actual, basis = media_ms(clock, capture_us(clock, ms), BASE)
        assert actual == pytest.approx(ms, abs=.001)
        assert basis == 'recorder_csv_interpolation'
    broken = dict(clock, points=[[0, BASE], [100, BASE]])
    assert media_ms(broken, BASE+500_000, BASE) == (500, 'capture_start_plus_media_time_estimate')


def test_shared_alignment_moves_clip_boundaries_and_rejects_stale_source():
    indexes=fixture()
    fp=indexes[0][1]
    fp['recordings'][0]['sources'][0]['retained']['sha256']='sealed'
    fp['segments'][0].update(start_ms=20_000,end_ms=80_000,video={'path':'ProcessedClips/Clips/A/ExperimentActivity.mp4'})
    fp['multiview_analysis']={'aligned_recordings':[{
        'archive':DAY+'_fp','recording_id':'f','source_sha256':'sealed',
        'start_us':BASE+101_000_000,'end_us':BASE+201_000_000,'state':'aligned',
        'seek_points':[[BASE+101_000_000,0],[BASE+201_000_000,100]],'global_origin_us':BASE}],
        'formal_decisions':[{'decision':'quarantined_missing_dual_view','global_start_ms':121_000,'global_end_ms':181_000}]}
    result=build_playback(DAY,indexes,DAY+'_fp','active',{'fp':'first_person','tp':'third_person'})
    assert result['start_us']==BASE+121_000_000
    assert result['end_us']==BASE+181_000_000
    rec=result['cameras'][0]['recordings'][0]
    assert rec['clips'][0]['start_us']==result['start_us']
    assert rec['clips'][0]['end_us']==result['end_us']
    assert rec['seek_points'][0][1]==20
    assert result['alignment_ready'] and len(result['grouping_decisions'])==1
    fp['recordings'][0]['sources'][0]['retained']['sha256']='new-source'
    stale=build_playback(DAY,indexes,DAY+'_fp','active',{})
    assert not stale['alignment_ready']
    assert stale['start_us']==BASE+120_000_000


def test_date_scope_path_safety_and_missing_source():
    indexes = fixture()
    other = deepcopy(indexes[1][1])
    other['archive']='2026-09-10_old'
    indexes.append((other['archive'], other))
    indexes[1][1]['recordings'][0]['sources'][0]['retained']['path']='../../outside.mp4'
    result=build_playback(DAY,indexes,DAY+'_fp','active',{})
    assert len(result['cameras']) == 2
    tp=next(c for c in result['cameras'] if c['camera']=='tp')
    assert tp['recordings'][0]['source_url'] is None
    with pytest.raises(ValueError):
        build_playback(DAY,indexes,'2026-09-10_fp','active',{})
    with pytest.raises(ValueError):
        build_playback(DAY,indexes,DAY+'_tp','inactive',{})


def test_api_reports_missing_index_and_discovery_errors(default_config, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from visioncortex.device_day_contract import atomic_json
    from visioncortex.device_day_timeline import install_routes
    cfg=deepcopy(default_config)
    cfg['storage']={'archive_root':str(tmp_path/'archive'),'local_runtime_root':str(tmp_path/'runtime')}
    for name,index in fixture():
        atomic_json(tmp_path/'archive'/name/'ProcessedClips/Index.json',index)
    (tmp_path/'archive'/(DAY+'_broken')).mkdir()
    atomic_json(tmp_path/'runtime/device-day/observed-inventory.json',{
        'errors':[{'path':'fp/'+DAY+'/182433/rgb.mp4','message':'metadata busy'},
                  {'path':'fp/2026-09-10/182433/rgb.mp4','message':'other day'}]})
    app=FastAPI()
    install_routes(app,lambda:cfg)
    client=TestClient(app)
    r=client.get('/api/day-timeline/'+DAY+'/playback',params={'archive':DAY+'_fp','segment_id':'active'})
    assert r.status_code==200
    assert r.json()['errors']==[{'archive':DAY+'_broken','reason':'FileNotFoundError'}]
    assert len(r.json()['discovery_errors'])==1


def test_preview_reuses_exporter_and_never_modifies_original(tmp_path, monkeypatch):
    from visioncortex.web_playback_cache import cached_preview
    from visioncortex import video_io
    source=tmp_path/'original.mp4'
    source.write_bytes(b'fixture-source')
    calls=[]
    def extract(src, dest, start, duration):
        calls.append((src,start,duration))
        dest.write_bytes(b'fixture-preview')
    monkeypatch.setattr(video_io,'extract_clip',extract)
    a=cached_preview(source,tmp_path/'runtime',5000,30000,'source-digest')
    assert a.read_bytes()==b'fixture-preview'
    assert cached_preview(source,tmp_path/'runtime',5000,30000,'source-digest')==a
    assert len(calls)==1
    assert source.read_bytes()==b'fixture-source'
    assert a.is_relative_to(tmp_path/'runtime/cache/WebPlayback')
    with pytest.raises(ValueError):
        cached_preview(source,tmp_path/'runtime',0,900000,'digest')

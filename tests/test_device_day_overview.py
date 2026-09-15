from copy import deepcopy
from types import SimpleNamespace

import pytest

from visioncortex.device_day_contract import atomic_json
from visioncortex.device_day_overview import ArchiveOverview, assemble, summarize, render, union_seconds
from visioncortex.device_day_progress import snapshot


def index(name, role_start=0):
    start = 1789005600000000 + role_start
    return {'archive':name, 'updated_at':'2026-09-10T12:00:00+08:00',
            'recordings':[{'recording_id':'r1','sources':[], 'transcription':{'status':'completed','outcome':'no_audio'}},
                          {'recording_id':'r2','sources':[]}],
            'segments':[{'segment_id':'s1','recording_id':'r1','activity':'active','start_us':start,
                         'end_us':start+10_000_000,'start_ms':0,'end_ms':10_000,
                         'source_ref':{'path':'MetaVideo/10-00.mp4'},'video':None,
                         'json_path':'ProcessedClips/Clips/One/ExperimentActivity.json',
                         'key_frames':[{'path':'ProcessedClips/Clips/One/KeyFrames/One.jpg'}]},
                        {'segment_id':'s2','recording_id':'r1','activity':'inactive','start_us':start+10_000_000,
                         'end_us':start+20_000_000,'key_frames':[],
                         'scene_frames':[{'path':'sample.jpg'}]}],
            'understandings':[]}


def test_overview_preserves_grains_role_time_union_and_unprocessed_gap(tmp_path):
    cfg={'storage':{'local_runtime_root':str(tmp_path)},'collection_ingest':{'camera_role_map':{
        'a_cam01':'first_person','b_cam02':'third_person'}}}
    a=summarize('2026-09-10_a_cam01',index('2026-09-10_a_cam01'),cfg)
    b=summarize('2026-09-10_b_cam02',index('2026-09-10_b_cam02',5_000_000),cfg)
    progress=snapshot(cfg)
    progress['days']['2026-09-10']={'total':4,'camera_count':2,'stages':{}}
    data=assemble(progress,[a,b],[])
    day=data['days'][0]
    assert day['active_count']==2
    assert day['keyframes']==2  # inactive scene sample is never an action keyframe
    assert day['activity_seconds']==15  # 10 seconds per view, 5-second overlap
    assert day['preprocessed_recordings']==2
    assert day['indexed_recordings']==4
    assert day['stt_transcribed_recordings']==0  # completed/no-audio is not STT
    assert a['role']=='first_person' and b['role']=='third_person'
    page=render(data)
    assert '2 尚无发布结果' in page
    assert '#t=0.000,10.000' in page
    assert '打开实验视频' in page
    assert '原录音 / 已转写分片' in page
    assert page.index('实验片段，从这里看') < page.index('当前运行情况')
    assert page.index('观看实验片段') < page.index('每天处理到哪里了')
    assert 'data-multiview' in page and '在 NAS 中的位置' in page
    assert 'VisionCortexExperimentArchive/2026-09-10_a_cam01/MetaVideo/10-00.mp4' in page


def test_overview_rejects_cross_archive_and_unassociated_segment():
    with pytest.raises(ValueError):
        summarize('2026-09-10_b_cam02',index('2026-09-10_a_cam01'),{})
    data=index('2026-09-10_a_cam01')
    data['segments'][0]['recording_id']='missing'
    with pytest.raises(ValueError):
        summarize(data['archive'],data,{})
    assert union_seconds([(5,4)])==0


def test_publication_updates_same_readme_and_drops_unreadable_cache(tmp_path, monkeypatch):
    root=tmp_path/'archive'
    name='2026-09-10_a_cam01'
    path=root/name/'ProcessedClips/Index.json'
    atomic_json(path,index(name))
    cfg={'storage':{'local_runtime_root':str(tmp_path/'runtime')}}
    runner=SimpleNamespace(config=cfg,archive_root=root,runtime_root=tmp_path/'runtime/device-day')
    worker=ArchiveOverview()
    assert worker.publish(runner)['active_segments']==1
    assert (root/'Readme.html').read_bytes()==(runner.runtime_root/'ArchiveOverview.html').read_bytes()
    first=(root/'Readme.html').read_text()
    assert name in first
    import visioncortex.device_day_overview as module
    original=module.read_json
    monkeypatch.setattr(module,'read_json',lambda p: (_ for _ in ()).throw(AssertionError('unchanged index reread')))
    assert worker.publish(runner)['active_segments']==1
    monkeypatch.setattr(module,'read_json',original)
    path.write_text('broken')
    result=worker.publish(runner)
    assert len(result['errors'])==1 and result['active_segments']==0
    assert '总索引读取异常' in (root/'Readme.html').read_text()


def test_no_source_link_escapes_archive_and_no_fabricated_days(tmp_path):
    cfg={'storage':{'local_runtime_root':str(tmp_path)}}
    empty=assemble(snapshot(cfg),[],[])
    assert empty['days']==[]
    assert '尚无可读取的采集数据' in render(empty)
    data=index('2026-09-10_a_cam01')
    data['segments'][0]['source_ref']['path']='../../secret'
    row=summarize(data['archive'],data,cfg)
    page=render(assemble(snapshot(cfg),[row],[]))
    assert '../../secret' not in page
    changed=deepcopy(data)
    changed['segments'][0]['key_frames'].append({'path':'ProcessedClips/<script>alert(1)</script>.jpg'})
    page=render(assemble(snapshot(cfg),[summarize(data['archive'],changed,cfg)],[]))
    assert '<script>alert(1)</script>' not in page

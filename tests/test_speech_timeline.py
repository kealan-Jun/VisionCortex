from types import SimpleNamespace

import pytest

from visioncortex import speech_worker
from visioncortex.input_seal import build_input_seal, write_input_seal
from visioncortex.schemas import AlignmentTransform, RunManifest, ViewInput
from visioncortex.speech_timeline import audio_map, build, load, video_path
from visioncortex.storage import _bounded_source_fingerprint


def test_clock_map_preserves_retiming_gaps_and_unknown_offset(tmp_path):
    clock = tmp_path / 'frames.csv'
    clock.write_text('frame_index,global_timestamp_ms,clock_sync_valid\n0,10000,1\n10,11000,1\n20,12000,1\n30,16000,1\n')
    source = {'alignment_basis':'recorder_shared_clock', 'video_clock_sha256':speech_worker.sha256(clock),
              '_request':{'source':{'start_global_us':10000000}}}
    chunk = {'start_seconds':0, 'end_seconds':6}
    transform = AlignmentTransform(view_id='fp',reference_view_id='fp',state='aligned')
    result = audio_map(source,chunk,SimpleNamespace(timestamps_csv=clock),SimpleNamespace(fps=20),transform,0)
    assert result['anchors'] == [[0,0],[1,500],[2,1000],[6,1500]]
    assert result['max_gap_seconds'] == 2
    unknown = audio_map({},chunk,None,None,transform,0)
    assert unknown['state'] == 'offset_unknown' and unknown['anchors'] == []
    offset = audio_map({'audio_offset_ms':250},chunk,None,None,transform,1000)
    assert offset['anchors'] == [[0,1250],[6,7250]]
    clock.write_text('tampered')
    with pytest.raises(ValueError,match='时钟'):
        audio_map(source,chunk,SimpleNamespace(timestamps_csv=clock),SimpleNamespace(fps=20),transform,0)


def timeline_fixture(tmp_path, monkeypatch):
    root = tmp_path/'archive'
    video = tmp_path/'source.mp4'
    video.write_bytes(b'original video')
    clock = tmp_path/'frames.csv'
    clock.write_text('frame_index,timestamp_ms\n0,0\n30,1000\n')
    manifest = RunManifest(experiment_id='test',views=[ViewInput(view_id=view,role=role,video=video,timestamps_csv=clock) for view,role in [('fp','first_person'),('tp','third_person')]])
    source = {'kind':'video','view_id':'fp','path':str(video),'size_bytes':video.stat().st_size,
              'mtime_ns':video.stat().st_mtime_ns,**_bounded_source_fingerprint(video)}
    seal = build_input_seal(manifest,source_mode='test',sources=[source],role_resolution={},copied_source_bytes=0)
    json_root = root/'JSON-Config-Files'
    write_input_seal(json_root/'Input-Manifests/input_seal.json',seal)
    alignment = json_root/'time_alignment.json'
    speech_worker.atomic_json(alignment,[{'view_id':view,'reference_view_id':'fp','state':'aligned'} for view in ('fp','tp')])
    speech_worker.atomic_json(json_root/'video_probe.json',{view:{'path':str(video),'duration_ms':1000,'fps':30,'width':10,'height':10,'frame_count':30} for view in ('fp','tp')})
    speech_worker.atomic_json(json_root/'speech.json',{'alignment_file_sha256':speech_worker.sha256(alignment),'sources':[]})
    def preview(root, source, duration):
        target = root/'Key-Materials/preview.mp4'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(b'derived browser preview')
        return {'path':str(target.relative_to(root)),**speech_worker.file_record(target)}
    monkeypatch.setattr('visioncortex.speech_timeline.make_preview',preview)
    build(root)
    return root,video


def test_source_preview_is_pinned_to_seal_and_timeline_version(tmp_path, monkeypatch):
    root,video = timeline_fixture(tmp_path, monkeypatch)
    timeline = load(root)
    assert video_path(root,timeline['sha256'],'fp',0) == root/'Key-Materials/preview.mp4'
    with pytest.raises(ValueError,match='不存在'):
        video_path(root,timeline['sha256'],'another-view',0)
    with pytest.raises(ValueError,match='更新'):
        video_path(root,'old-version','fp',0)
    video.write_bytes(b'changed content')
    with pytest.raises(ValueError,match='身份'):
        video_path(root,timeline['sha256'],'fp',0)


def test_changed_alignment_cannot_reuse_timeline(tmp_path, monkeypatch):
    root,_ = timeline_fixture(tmp_path, monkeypatch)
    speech_worker.atomic_json(root/'JSON-Config-Files/time_alignment.json',[])
    with pytest.raises(ValueError,match='版本'):
        load(root)


def test_preview_stream_supports_ranges_and_rejects_tampering(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from visioncortex import api
    root,_ = timeline_fixture(tmp_path, monkeypatch)
    timeline = load(root)
    monkeypatch.setattr(api,'_resolve_staging_run',lambda _:root)
    client = TestClient(api.app)
    url = f"/api/staging-runs/retained/speech-video?view=fp&part=0&timeline={timeline['sha256']}"
    response = client.get(url,headers={'Range':'bytes=0-6'})
    assert response.status_code == 206 and response.content == b'derived'
    (root/'Key-Materials/preview.mp4').write_bytes(b'changed')
    assert client.get(url,headers={'Range':'bytes=0-6'}).status_code == 409

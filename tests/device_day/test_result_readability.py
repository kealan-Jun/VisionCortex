"""Human JSON and old machine reads agree; no successful action is invented."""
from copy import deepcopy
import json

import pytest

from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.result_readability import KEY, canonical, kind, present


def pending():
    return dict(schema_version='visioncortex-source-recording-delivery/2',
        original_start_us=1793268966388025, original_end_us=1793268996298006,
        folder_scope='complete_original_recording_collection', display_category='PendingValueReview',
        available_video_count=0, source_folder='NoExperimentActivity_legacy',
        original_source={'path': 'MetaData/capture/rgb.mp4'}, intervals=[dict(
            start_us=1793268966388025, end_us=1793268996298006,
            selection_interval=dict(video_value='pending', reason='value_not_determined'), video=None)])


def test_pending_filename_does_not_become_confirmed_no_value():
    value = pending(); before = deepcopy(value); projected = present(value, 'retention')
    assert projected[KEY]['留存结论'] == '价值待判断'
    assert projected[KEY]['区间'][0]['留存结论'] == '价值待判断'
    assert '18:16:06.388' in projected[KEY]['开始']
    assert projected[KEY]['关联']['实验设备'] is None
    assert projected[KEY]['关联']['材料来源批次'] is None
    assert canonical(projected) == before == value
    assert present(projected, 'retention') == projected


@pytest.mark.parametrize('loader', ['ordinary', 'frozen_result'])
def test_actual_guarded_writer_and_machine_reader_keep_old_contents(tmp_path, monkeypatch, loader):
    monkeypatch.setenv('VISIONCORTEX_READABLE_RESULTS', '1')
    if loader == 'ordinary':
        writer, reader = atomic_json, read_json
    else:
        from visioncortex.lab.native_result_layout import module
        owner = module('device_day_contract'); writer, reader = owner.atomic_json, owner.read_json
    root = tmp_path/'2026-10-29_fixture-camera'
    path = root/'ProcessedClips/Clips/existing/Retention.json'; value = pending()
    writer(path, value)
    raw = json.loads(path.read_text()); assert next(iter(raw)) == KEY
    assert reader(path) == value and canonical(raw) == value
    before = set(root.rglob('*')); writer(path, reader(path))
    assert set(root.rglob('*')) == before and reader(path) == value
    for relative in ['ProcessedClips/Index.json', 'MultimodalUnderstanding/Index.json',
                     'LaboratoryDailyReport/Timeline/r000001.json',
                     'ProcessedClips/.Evidence/Original/Retention.json', 'MetaData/Retention.json']:
        writer(root/relative, value)
        assert KEY not in json.loads((root/relative).read_text())


def test_tampering_fails_before_machine_data_can_be_read():
    value = present(pending(), 'retention'); value[KEY]['留存结论'] = '确认实验结束'
    with pytest.raises(ValueError, match='source projection'): canonical(value)
    value = present(pending(), 'retention'); value['intervals'][0]['video'] = {'path': 'made-up.mp4'}
    with pytest.raises(ValueError, match='saved machine data'): canonical(value)


def test_daily_sorts_saved_speech_video_and_ocr_without_inventing_links():
    value = {'archive':'2026-10-29_fixture-camera','全天综合理解':{'status':'pending'},
        '全天记录':{'日期':'2026-10-29','设备':'fixture-camera','全天时间线':[
            {'开始时间':'2026-10-29T18:00:00','结束时间':'2026-10-29T18:00:15','状态说明':'理解未完成'}],
            '录音时间线':[{'开始时间':'2026-10-29T13:05:45.760','原话':'原话。','转写来源':'Comment/Audio/existing/Transcript.json'}],
            '现场事件时间线':[{'记录时间':'2026-10-29T14:00:00','类型':'ocr_observation','内容':{'value':None}}]}}
    result = present(value, 'day')[KEY]
    assert [r['类型'] for r in result['按时间查看']] == ['自动录音转写','ocr_observation','视频记录']
    assert result['按时间查看'][0]['原话'] == '原话。'
    assert not result['完整一天已验收'] and result['跨模态关联']['目标制备批次'] is None
    assert canonical(present(value, 'day')) == value


def test_legacy_daily_entries_are_readable_without_hydrating_their_audit_trees():
    value={'archive':'2026-10-29_fixture-camera','entries':[{
        'start_us':1793268966388025,'end_us':1793268996298006,'label':'old candidate',
        'understanding_status':'completed','understanding':{'windows':[{'summary':'已有模型原文。'}]},
        'transcription':{'comments':[{'text':'原话。'}]}}]}
    summary=present(value,'day')[KEY]
    assert summary['按时间查看'][0]['模型原概述']==['已有模型原文。']
    assert summary['按时间查看'][0]['原话'][0]['内容']=='原话。'
    assert summary['时间线记录总数']==1 and not summary['完整一天已验收']
    assert canonical(present(value,'day'))==value


def test_pending_understanding_is_readable_and_exact_receipts_remain_exact(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIONCORTEX_READABLE_RESULTS','1')
    value = {'status':'pending','understanding':None,'video':None,'内容记录':{
        '状态说明':'模型理解尚未完成','实验步骤':[],'录音记录':[],'分窗口观察':[]}}
    path=tmp_path/'MultimodalUnderstanding/existing/Understanding.json';atomic_json(path,value)
    assert json.loads(path.read_text())[KEY]['理解状态']=='模型理解尚未完成'
    from visioncortex.device_day_contract import artifact
    from visioncortex.device_day_verification import ArtifactVerifier
    verifier=ArtifactVerifier();ref=artifact(tmp_path,path)
    assert verifier.verify(tmp_path,ref)
    changed=deepcopy(ref);changed['sha256']='0'*64
    assert not verifier.verify(tmp_path,changed)
    assert read_json(path)==value
    assert kind(path)=='understanding'


def test_optional_refresh_bridge_uses_verified_canonical_reads(tmp_path, monkeypatch):
    monkeypatch.setenv('VISIONCORTEX_READABLE_RESULTS','1')
    from visioncortex.lab import result_delivery
    from visioncortex.lab.native_result_layout import module
    cls = result_delivery.refresh_class()
    assert cls is result_delivery.refresh_class()
    root=tmp_path/'2026-10-29_fixture-camera'
    relative='MultimodalUnderstanding/existing/Understanding.json'
    original={'archive':root.name,'status':'pending','understanding':None}
    owner=module('device_day_contract');owner.atomic_json(root/relative,original)
    with module('device_day_write_guard').storage_guard({'runtime':{'local_only':True}}) as guard:
        saved=result_delivery._BRIDGE._read(root,relative,guard)
    assert saved[0]==original and KEY in json.loads((root/relative).read_text())
    assert cls.__module__=='visioncortex.lab._readable_native_refresh'


def test_legacy_narrative_string_keeps_its_actual_words():
    value={'内容记录':'旧版模型原文。','status':'completed'}
    projected=present(value,'understanding')
    assert projected[KEY]['旧版内容原文']=='旧版模型原文。'
    assert canonical(projected)==value

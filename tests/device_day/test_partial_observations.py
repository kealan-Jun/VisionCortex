"""Rescheduling keeps validated partial observations visible without completing a run."""
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import read_json
from visioncortex.device_day_understanding import UnderstandingCheckpointYield
from test_device_day import FakeModels, capture, item_and_layout, device_config as device_config_fixture

device_config = device_config_fixture


def test_checkpoint_report_keeps_verified_window_and_rejects_changed_artifact(device_config):
    capture(device_config)
    item, layout = item_and_layout(device_config)
    device_config['device_day']['understanding_coverage_since_us'] = item['recording_start_us']-1
    class CheckpointModels(FakeModels):
        def understand(self, layout, recording, vision, context, key):
            saved = super().understand(layout, recording, vision, context, key)
            saved.update(content_source='video_only',context_digest=None,partial_observations=True)
            for meaning in saved['understandings']:
                meaning['status']='partial'
            raise UnderstandingCheckpointYield('owned-plan.json',1,partial_receipt=saved)
    runner=DeviceDayRunner(device_config,CheckpointModels())
    assert runner.process(item,stage='retention')['status']=='completed'
    assert runner.process(item,stage='vision')['status']=='completed'
    result=runner.process(item,stage='understanding')
    assert result['status']=='scheduled_checkpoint'
    assert runner.refresh_index(layout,recording_id=item['recording_id'])
    index=read_json(layout.index)
    assert index['understandings'][0]['status']=='partial'
    assert '测试关键帧文字' in (layout.reports/'LaboratoryDailyReport.html').read_text()
    assert read_json(runner._receipt(layout,item,'understanding'))['status']=='scheduled_checkpoint'
    assert runner._load(runner._receipt(layout,item,'understanding'),result['key'],layout) is None
    reference=result['artifacts'][0]
    path=layout.root/reference['path']
    original=path.read_bytes()
    path.write_bytes(b'x'*len(original))
    assert runner.refresh_index(layout,recording_id=item['recording_id'])
    assert read_json(layout.index)['understandings']==[]

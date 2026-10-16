"""Publish independent video understanding without changing its paid identity."""
import pytest

from test_device_day import device_config  # noqa: F401
from test_device_day_downstream_worker import prepared
from visioncortex.device_day_contract import read_json


@pytest.mark.parametrize('video_only', [True, False])
def test_completed_understanding_appears_in_index_and_daily_report(device_config, monkeypatch, video_only):  # noqa: F811
    if video_only:
        device_config['device_day']['understanding_coverage_since_us'] = 1
    runner, item, layout, models = prepared(device_config)
    assert runner.process(item, stage='understanding')['status'] == 'completed'
    assert runner.process(item, stage='report')['status'] == 'completed'
    assert read_json(layout.index)['understandings']
    entries = read_json(layout.reports / 'LaboratoryDailyReport.json')['entries']
    assert entries[0]['understanding']['windows'][0]['summary'].startswith('抽样画面')
    monkeypatch.setattr(runner, '_backend', lambda: pytest.fail('publication must not invoke models'))
    assert runner.refresh_index(layout, recording_id=item['recording_id'])
    assert read_json(layout.index)['understandings']
    assert models.vision_calls == models.semantic_calls == 1


@pytest.mark.parametrize('video_only', [True, False])
def test_missing_speech_blocks_only_legacy_semantic_publication(device_config, monkeypatch, video_only):  # noqa: F811
    if video_only:
        device_config['device_day']['understanding_coverage_since_us'] = 1
    runner, item, layout, models = prepared(device_config)
    assert runner.process(item, stage='understanding')['status'] == 'completed'
    runner._receipt(layout, item, 'stt').unlink()  # Owned synthetic test receipt only.
    monkeypatch.setattr(runner, '_backend', lambda: pytest.fail('publication must not invoke models'))
    assert runner.refresh_index(layout, recording_id=item['recording_id'])
    assert bool(read_json(layout.index)['understandings']) is video_only
    assert models.vision_calls == models.semantic_calls == 1

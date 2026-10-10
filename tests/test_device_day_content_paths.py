from pathlib import Path

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_content_paths import aliases, migrate_stt, migrate_understanding, public_references
from visioncortex.device_day_contract import artifact, atomic_json, read_json, safe_child, verify_artifact

__all__ = ['device_config']


def test_relocation_keeps_old_receipt_valid_and_only_one_visible_entry(device_config):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(record, stage='retention')
    old = layout.comments/'Stt'/record['recording_id']/'version'/'Transcript.txt'
    old.parent.mkdir(parents=True)
    old.write_text('原始机器转写\n')
    ref = artifact(layout.root, old)
    result = migrate_stt(runner, layout)
    assert len(result) == 1 and not (layout.comments/'Stt').exists()
    assert verify_artifact(layout.root, ref)
    new = safe_child(layout.root, ref['path'])
    assert 'Recognition/version' in str(new)
    assert new.read_text() == '原始机器转写\n'
    assert migrate_stt(runner, layout) == []
    mapped = public_references({'transcript_file': ref}, read_json(layout.comments/'LegacyPaths.json')['aliases'])
    assert Path(mapped['transcript_file']['path']).parts[1].startswith('10-00-00.')
    assert verify_artifact(layout.root, mapped['transcript_file'])


def test_alias_cannot_escape_archive_or_recurse(tmp_path):
    for target in ('Comment/../../outside', 'Comment/Stt/again'):
        atomic_json(tmp_path/'Comment/LegacyPaths.json', {'aliases': {'Comment/Stt/a': target}})
        with pytest.raises(ValueError):
            safe_child(tmp_path, 'Comment/Stt/a/file.txt')


def test_understanding_relocation_preserves_results_and_historical_segment_times(device_config):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    runner = DeviceDayRunner(device_config, FakeModels())
    runner.process(record, stage='retention')
    segment = {'segment_id': 'segment-test', 'recording_id': record['recording_id'],
               'start_us': record['recording_start_us']+1000000, 'end_us': record['recording_end_us']}
    atomic_json(layout.receipts/record['recording_id']/'history/vision-old.json', {'segments': [segment]})
    old = layout.understanding/'ClipUnderstanding/segment-test/version/window/Result.json'
    atomic_json(old, {'model_result': {'summary': '原始模型结果'}})
    ref = artifact(layout.root, old)
    result = migrate_understanding(runner, layout)
    assert len(result) == 1 and not (layout.understanding/'ClipUnderstanding').exists()
    assert verify_artifact(layout.root, ref)
    mapped = public_references({'model_receipt': ref['path']}, aliases(layout.root))
    assert '/10-00-01.' in mapped['model_receipt']
    assert '/Analysis/segment-test/version/window/Result.json' in mapped['model_receipt']
    assert migrate_understanding(runner, layout) == []


def test_understanding_alias_rejects_other_stream_and_recursive_target(tmp_path):
    for target in ('Comment/wrong', 'MultimodalUnderstanding/ClipUnderstanding/again',
                   'MultimodalUnderstanding/../../outside'):
        atomic_json(tmp_path/'MultimodalUnderstanding/LegacyPaths.json',
                    {'aliases': {'MultimodalUnderstanding/ClipUnderstanding/a': target}})
        with pytest.raises(ValueError):
            safe_child(tmp_path, 'MultimodalUnderstanding/ClipUnderstanding/a/Result.json')

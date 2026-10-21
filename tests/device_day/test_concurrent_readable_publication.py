"""Current immutable results publish while independent native producers run."""
import pytest

from test_frozen_archive_layout import fixture as content_fixture
from visioncortex.device_day import exclusive
from visioncortex.device_day_contract import VERSION, atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher, _display_migration_pending
from visioncortex.observed_inventory import observe


@pytest.mark.parametrize('legacy', [False, True])
def test_only_legacy_relocation_needs_every_producer_drained(tmp_path, legacy):
    config, index = content_fixture(tmp_path)
    index['schema_version'] = VERSION
    index['segments'][0]['activity_label'] = '无实验活动'
    runtime = tmp_path/'OwnedRuntime'
    config['storage']['local_runtime_root'] = str(runtime)
    config['collection_ingest'] = {'source_root': str(tmp_path/'Capture')}
    config['device_day']['readable_content_enabled'] = True
    root = tmp_path/index['archive']
    atomic_json(root/'ProcessedClips/Index.json', index)
    observe(runtime/'device-day', {'recordings': [{
        'recording_id': 'recording', 'camera_key': index['archive'][11:],
        'recording_start_us': index['segments'][0]['start_us']}]})
    old = root/'Comment/Stt/Transcript.txt'
    if legacy:
        old.parent.mkdir(parents=True)
        old.write_text('preserve owned legacy transcript')
    with exclusive(runtime/'device-day/locks/recording.understanding.lock'):
        result = FileIndexPublisher(config).tick(archives=[index['archive']])
    assert result['busy'] == int(legacy), result
    assert result['published'] == int(not legacy), result
    if legacy:
        assert old.read_text() == 'preserve owned legacy transcript'
        assert not (root/'Comment/.Evidence/Recognition').exists()
    else:
        assert (root/'MultimodalUnderstanding/Index.json').is_file()
        assert read_json(root/'ProcessedClips/Index.json')['segments'] == index['segments']


@pytest.mark.parametrize('relative', [
    'MultimodalUnderstanding/Understanding.json',
    'MultimodalUnderstanding/UnderstandingReport.html',
    'MultimodalUnderstanding/RequestPlans/Plan.json',
    'MultimodalUnderstanding/ClipUnderstanding/Result.json',
    'LaboratoryDailyReport/Analysis/Result.json',
    'LaboratoryDailyReport/History/Result.json',
    'ProcessedClips/13-00-00_13-00-01/Activity.json',
    'MultimodalUnderstanding/13-00-00_13-00-01/Understanding.json',
    'Comment/13-00-00_13-00-01/Recognition/Transcript.json',
])
def test_native_legacy_names_still_require_drain(tmp_path, relative):
    path = tmp_path/relative
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    from types import SimpleNamespace
    assert _display_migration_pending(SimpleNamespace(root=tmp_path))


def test_current_transcript_and_clip_directories_do_not_require_drain(tmp_path):
    from types import SimpleNamespace
    for relative in ['Comment/13-00-00_13-00-01/Transcript.json',
                     'ProcessedClips/Clips/13-00-00_13-00-01_ExperimentActivity_abcd/ExperimentActivity.json',
                     'MultimodalUnderstanding/13-00-00_13-00-01_ExperimentActivity_abcd/Understanding.json']:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    assert not _display_migration_pending(SimpleNamespace(root=tmp_path))

"""The existing verified replayer owns canonical selected-video references."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout
from test_readable_all_day import add_work_evidence
from visioncortex.device_day import DeviceDayRunner
from visioncortex.device_day_contract import VERSION, artifact, atomic_json, read_json
from visioncortex.device_day_file_index import FileIndexPublisher
from visioncortex.observed_inventory import observe
from visioncortex.publication_journal import PublicationJournal, reconcile
from visioncortex.sqlite_store import connection

__all__ = ['device_config']


@pytest.fixture
def selected_case(device_config, monkeypatch):
    capture(device_config)
    record, layout = item_and_layout(device_config)
    device_config['device_day'].update(derived_video_retention=False, readable_content_enabled=True,
                                      paused_stages=['report'])
    class WorkModels(FakeModels):
        def vision(self, *args):
            result = super().vision(*args)
            result['batches'] = [{'dense_cv_evidence': 'preserve every byte' * 1000}]
            return result

        def understand(self, layout, recording, vision, context, key):
            result = super().understand(layout, recording, vision, context, key)
            item = result['understandings'][0]
            evidence = add_work_evidence(layout, item, key, vision['segments'][0])
            path = layout.understanding/'ClipUnderstanding'/f'{key}.json'
            atomic_json(path, item)
            result['artifacts'] = [artifact(layout.root, path), *evidence]
            return result
    models = WorkModels()
    runner = DeviceDayRunner(device_config, models)
    assert runner.process(record)['status'] == 'completed'
    original = read_json(layout.index)
    assert not original['segments'][0].get('video')
    observe(runner.runtime_root, {'recordings': [record]})
    device_config['device_day']['derived_video_retention'] = True
    def extract(source, target, start, duration, *args):
        assert start == 0 and duration == 1000  # A 15-minute candidate contains one second of observed work.
        target.write_bytes(b'actual derived file in synthetic contract test')
    monkeypatch.setattr('visioncortex.video_io.extract_clip', extract)
    monkeypatch.setattr('visioncortex.video_io.probe_video', lambda _: SimpleNamespace(duration_ms=1000))
    return device_config, runner, models, record, layout, original


def test_independent_video_publication_replays_canonical_without_losing_dense_cv(selected_case):
    config, runner, models, record, layout, original = selected_case
    publisher = FileIndexPublisher(config)
    result = publisher.tick()
    assert result['selected_videos_materialized'] == result['canonical_video_pending'] == 1
    interim = read_json(layout.index)
    assert not interim['segments'][0].get('video')
    assert interim['recordings'][0]['processing']['batches'] == original['recordings'][0]['processing']['batches']
    cached = publisher.source_indexes[layout.name][1]
    assert 'batches' not in cached['recordings'][0]['processing']  # It cannot replace the canonical index.
    time_index = read_json(layout.comments/'TimeIndex.json')
    segment_file = next(row['file'] for row in time_index['streams']['video']
                        if row['kind'] == 'video_segment' and row.get('file'))
    assert (layout.root/segment_file['path']).is_file()
    journal = PublicationJournal(runner.runtime_root)
    token, pending = journal.pending()[0]
    journal.defer(record['recording_id'], token, reason='lock_busy', now=0)
    assert publisher.tick()['canonical_video_pending'] == 1
    assert journal.pending() == [(token, pending)]
    with connection(journal.path, readonly=True) as db:
        assert db.execute('SELECT attempts FROM retries WHERE id=?', (record['recording_id'],)).fetchone()[0] == 1
        assert [row[1] for row in db.execute('PRAGMA table_info(dirty)')] == ['id', 'token', 'record']
    calls = models.vision_calls, models.semantic_calls
    assert reconcile(runner) == 1
    canonical = read_json(layout.index)
    assert canonical['schema_version'] == VERSION
    parent = canonical['segments'][0]
    assert not parent.get('video')
    selected = next(row for row in parent['retention_intervals'] if row.get('video'))
    assert selected['video']['path'] == segment_file['path']
    assert (selected['start_ms'], selected['end_ms']) == (0, 1000)
    assert canonical['recordings'][0]['processing']['batches'] == original['recordings'][0]['processing']['batches']
    assert (models.vision_calls, models.semantic_calls) == calls
    assert not journal.pending()
    assert not publisher.tick().get('canonical_video_pending')


def test_changed_inventory_source_cannot_request_or_ack_canonical_publication(selected_case):
    config, runner, models, record, layout, original = selected_case
    publisher = FileIndexPublisher(config)
    assert publisher.tick()['canonical_video_pending'] == 1
    journal = PublicationJournal(runner.runtime_root)
    before = journal.pending()
    changed = deepcopy(record)
    changed['source_signature'] = 'different-current-source'
    observe(runner.runtime_root, {'recordings': [changed]})
    result = publisher.tick()
    assert any('current admitted recording' in row.get('reason', '') for row in result['errors'])
    assert journal.pending() == before
    assert reconcile(runner) == 0
    assert journal.pending() == before
    assert not read_json(layout.index)['segments'][0].get('video')
    assert models.vision_calls == models.semantic_calls == 1


def test_joining_publication_preserves_old_schema_token_and_other_source_generation(tmp_path):
    journal = PublicationJournal(tmp_path)
    record = {'recording_id': 'owned', 'source_signature': 'source-a', 'camera_key': 'camera-1',
              'recording_start_us': 1000000, 'recording_end_us': 2000000}
    token = journal.begin(record)
    assert journal.ensure_pending(record) == token
    assert journal.ensure_pending(record | {'source_signature': 'source-b'}) is None
    assert journal.pending() == [(token, record)]

"""Exact interval siblings preserve history through legacy publication ports."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from visioncortex.device_day_content_paths import clip_name, publish_selected_videos
from visioncortex.device_day_contract import DeviceDayLayout, artifact, atomic_json, read_json
from visioncortex.device_day_publication import reconcile_outputs


def archive(tmp_path):
    start = 1792376825000000
    layout = DeviceDayLayout(tmp_path / 'Archive', 'camera_cam01', start, tmp_path / 'Cache')
    source = layout.root / 'MetaData' / '102705' / 'rgb.mp4'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'collector-original-bytes')
    parent = {'recording_id': 'recording', 'segment_id': 'segment-parent', 'activity': 'inactive',
              'source_ref': artifact(layout.root, source), 'start_ms': 0, 'end_ms': 120000,
              'start_us': start, 'end_us': start + 120000000}
    parent['json_path'] = 'ProcessedClips/Clips/' + clip_name(parent) + '/NoExperimentActivity.json'
    child = deepcopy(parent)
    child.update(segment_id='segment-child', candidate_segment_id=parent['segment_id'],
                 start_ms=10000, end_ms=20000, start_us=start + 10000000,
                 end_us=start + 20000000, json_path=None, video_value='selected')
    child['json_path'] = 'ProcessedClips/Clips/' + clip_name(child) + '/NoExperimentActivity.json'
    return layout, parent, child, source


@pytest.mark.parametrize('marker', [
    {'historical_scope': 'historical_only_not_revalidated_for_exact_selection'},
    {'historical_scope': {'selection_status': 'historical_retained_not_revalidated_for_exact_selection'}},
    {'retention_intervals': []},
    {'derived_video_disposition': {'selection_policy': 'visioncortex-exact-value-intervals/1'}},
])
def test_broad_parent_is_not_copied_as_a_new_exact_selection(tmp_path, marker):
    layout, parent, _, original = archive(tmp_path)
    old = layout.processed / '.Evidence' / 'Historical' / 'Whole.mp4'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'historical-whole-candidate')
    parent.update(video=artifact(layout.root, old), **marker)
    before = deepcopy(parent)
    raw = original.read_bytes()
    progress = {}
    assert publish_selected_videos(layout, [parent], progress=progress) == []
    assert progress == {'pending': 0}
    assert old.read_bytes() == b'historical-whole-candidate'
    assert original.read_bytes() == raw and parent == before
    assert not (layout.processed / 'Clips').exists()
    assert not (layout.processed / '.Evidence' / 'LegacyPaths.json').exists()


def test_exact_child_can_use_legacy_copy_port_without_republishing_parent(tmp_path):
    layout, parent, child, original = archive(tmp_path)
    old = layout.processed / '.Evidence' / 'Historical' / 'Exact.mp4'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'exact-child-bytes')
    child['video'] = artifact(layout.root, old)
    parent.update(video=child['video'], retention_intervals=[child])
    result = publish_selected_videos(layout, [parent, child])
    target = layout.root / result[0]['path']
    assert len(result) == 1 and target.parent.name == clip_name(child)
    assert target.read_bytes() == old.read_bytes() == b'exact-child-bytes'
    assert not (layout.processed / 'Clips' / clip_name(parent)).exists()
    assert original.read_bytes() == b'collector-original-bytes'


def reconciler(layout, parent, monkeypatch):
    from visioncortex import device_day_capture_files
    monkeypatch.setattr(device_day_capture_files, 'publish_capture_catalog', lambda *_: None)
    record = {'recording_id': parent['recording_id'], 'camera_key': 'camera_cam01',
              'recording_start_us': parent['start_us'], 'recording_end_us': parent['end_us']}
    folder = layout.receipts / record['recording_id']
    atomic_json(folder / 'retention.json', {'status': 'completed', 'recording': record, 'sources': []})
    atomic_json(folder / 'vision.json', {'status': 'completed', 'key': 'current', 'segments': [parent]})
    atomic_json(layout.root / parent['json_path'], parent)
    return SimpleNamespace(archive_root=layout.root.parent, backend_root=layout.backend_root,
                           runtime_root=layout.backend_root / 'Runtime',
                           layout=lambda _: layout, _key=lambda *_: 'current')


def test_reconciler_keeps_canonical_child_outside_the_cv_receipt(tmp_path, monkeypatch):
    layout, parent, child, original = archive(tmp_path)
    runner = reconciler(layout, parent, monkeypatch)
    projected = deepcopy(parent)
    projected['retention_intervals'] = [child]
    atomic_json(layout.index, {'segments': [projected]})
    folder = (layout.root / child['json_path']).parent
    # Prove that the canonical reference protects the folder independently of
    # whether its first metadata document has the new child schema yet.
    atomic_json(folder / 'Legacy.json', {'recording_id': parent['recording_id']})
    video = folder / 'NoExperimentActivity.mp4'
    video.write_bytes(b'completed-child')
    assert reconcile_outputs(runner)['relocated'] == []
    assert video.read_bytes() == b'completed-child'
    assert original.read_bytes() == b'collector-original-bytes'
    assert read_json(layout.receipts / parent['recording_id'] / 'vision.json')['segments'] == [parent]


@pytest.mark.parametrize('metadata', [
    {'record_scope': 'retention_interval'},
    {'schema_version': 'visioncortex-retained-interval/1'},
    {'candidate_segment_id': 'segment-parent'},
])
def test_reconciler_keeps_child_before_canonical_replay_and_after_history(tmp_path, monkeypatch, metadata):
    layout, parent, child, original = archive(tmp_path)
    runner = reconciler(layout, parent, monkeypatch)
    folder = (layout.root / child['json_path']).parent
    atomic_json(folder / 'Retention.json', {'recording_id': parent['recording_id'],
                                          'segment_id': child['segment_id'], **metadata})
    video = folder / 'NoExperimentActivity.mp4'
    video.write_bytes(b'durable-child-before-index-replay')
    assert reconcile_outputs(runner)['relocated'] == []
    assert reconcile_outputs(runner)['relocated'] == []
    assert video.read_bytes() == b'durable-child-before-index-replay'
    assert original.read_bytes() == b'collector-original-bytes'


def test_large_canonical_index_defers_all_optional_legacy_moves(tmp_path):
    from visioncortex.device_day_contract import DeviceDayLayout
    from visioncortex.device_day_publication import _kept_clip_folders
    layout = DeviceDayLayout(tmp_path, 'camera-1', 1000000)
    layout.index.parent.mkdir(parents=True)
    with layout.index.open('wb') as handle:
        handle.truncate(8*1024*1024+1)  # owned sparse fixture, never parsed
    assert _kept_clip_folders(layout, {'segments': []}, {'recording_id': 'r'}) is None

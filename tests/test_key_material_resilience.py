import json

import numpy as np
import pytest

from visioncortex import archive
from visioncortex.schemas import AlignmentTransform, EvidenceEvent, ExperimentGroup, VideoInfo, ViewInput


@pytest.mark.parametrize('failure,expected_counts', [
    ('frame', (1, 3)), ('clip', (3, 1)), ('both', (1, 1)),
    ('aligned_frame', (2, 3)), ('aligned_clip', (3, 2)), ('source', (1, 1)),
])
def test_media_failures_keep_independent_outputs_and_retry_only_selected_event(tmp_path, monkeypatch, failure, expected_counts):
    layout = archive.ArchiveLayout(tmp_path / 'archive')
    layout.create()
    views = [ViewInput(view_id='fp', role='first_person', video=tmp_path/'fp.mp4'),
             ViewInput(view_id='tp', role='third_person', video=tmp_path/'tp.mp4')]
    infos = {view.view_id: VideoInfo(path=view.video, duration_ms=10000, fps=30,
                                   width=32, height=24, frame_count=300) for view in views}
    transforms = {view.view_id: AlignmentTransform(view_id=view.view_id, reference_view_id='fp') for view in views}
    events = [EvidenceEvent(event_id=name, action_type='object_movement', global_start_ms=start,
                            global_end_ms=start+1000, key_global_ms=start+500, objects=['pipette'],
                            accepted=True, formal_admission_status='provisional', confidence=.5,
                            audit_reason='fixture', supporting_views=[], supporting_roles=[], candidates=[])
              for name, start in [('EVT-A', 1000), ('EVT-B', 6000)]]
    group = ExperimentGroup(group_id='GROUP', continuity_type='independent', atomic_experiment_ids=['EXP'],
                            global_start_ms=0, global_end_ms=10000, participating_views=['fp', 'tp'],
                            first_person_view='fp', third_person_view='tp', continuity_reason='fixture',
                            key_event_ids=[event.event_id for event in events])
    enabled = [True]
    closed = []
    source_selection = archive._select_key_material_media_source

    def select_source(view, info, transform, event, *args, **kwargs):
        if enabled[0] and failure == 'source' and event.event_id == 'EVT-A' and view.view_id == 'fp':
            raise ValueError('source unavailable')
        return source_selection(view, info, transform, event, *args, **kwargs)

    class Reader:
        def __init__(self, **kwargs):
            pass

        def read(self, view, info, ms):
            if enabled[0] and failure in {'frame', 'both'} and view.view_id == 'fp' and ms < 3000:
                return None
            return np.full((24, 32, 3), 128, dtype=np.uint8)

        def close(self):
            closed.append(True)

    def clip(view, info, destination, *args):
        if enabled[0] and failure in {'clip', 'both'} and view.view_id == 'fp' and 'EVT-A' in str(destination):
            raise RuntimeError('encoder unavailable')
        destination.write_bytes(b'clip fixture')

    def aligned_frame(first, third, destination, labels):
        if enabled[0] and failure == 'aligned_frame' and 'EVT-A' in str(destination):
            raise RuntimeError('image composition unavailable')
        destination.write_bytes(b'aligned frame fixture')

    def aligned_clip(inputs, destination, encoder):
        if enabled[0] and failure == 'aligned_clip' and 'EVT-A' in str(destination):
            raise RuntimeError('video composition unavailable')
        destination.write_bytes(b'aligned clip fixture')

    monkeypatch.setattr(archive, 'ViewFrameReader', Reader)
    monkeypatch.setattr(archive, '_select_key_material_media_source', select_source)
    monkeypatch.setattr(archive, 'nearest_frame_evidence_many', lambda path, timestamps: dict.fromkeys(timestamps))
    monkeypatch.setattr(archive, 'extract_view_clip', clip)
    monkeypatch.setattr(archive, '_write_aligned_frame', aligned_frame)
    monkeypatch.setattr(archive, 'create_grid_video', aligned_clip)
    old_files = []
    for field in ['key_frames', 'key_clips']:
        for view in ['fp', 'tp', 'aligned_first_third']:
            path = layout.root / f'old-{field}-{view}.dat'
            path.write_bytes(b'previous result')
            getattr(events[0], field)[view] = path.name
            old_files.append(path)
    config = {'segmentation': {'key_clip_pre_seconds': .5, 'key_clip_post_seconds': .5},
              'performance': {'ffmpeg_video_encoder': 'libx264', 'materialization_workers': 2,
                              'overlap_aligned_key_materials': True}}
    paths = {view.view_id: tmp_path/f'{view.view_id}.jsonl' for view in views}
    archive.materialize_key_materials(layout, events, [group], views, infos, transforms, paths, config)
    assert len(events[0].key_frames) == expected_counts[0]
    assert len(events[0].key_clips) == expected_counts[1]
    assert len(events[1].key_frames) == len(events[1].key_clips) == 3
    assert all(path.read_bytes() == b'previous result' for path in old_files)
    assert all(not path.startswith('old-') for field in ['key_frames', 'key_clips'] for path in getattr(events[0], field).values())
    assert len(closed) == 2
    assert events[0].observability['key_material_materialization']['status'] == 'partial'
    for item in events:
        expected = item.observability['key_material_materialization']['status']
        for relative in [*item.key_frames.values(), *item.key_clips.values()]:
            sidecar = json.loads((layout.root/relative).with_suffix('.json').read_text())
            assert sidecar['provenance']['cv']['observability']['key_material_materialization']['status'] == expected
    runtime_path = layout.json_config / 'key_material_materialization_runtime.json'
    runtime = json.loads(runtime_path.read_text())
    assert runtime['status'] == 'partial'
    assert runtime['retry_event_ids'] == ['EVT-A']
    assert runtime['incomplete_artifacts']
    assert {item['event_id'] for item in runtime['incomplete_artifacts']} == {'EVT-A'}
    previous_b = {p: (layout.root/p).read_bytes() for p in [*events[1].key_frames.values(), *events[1].key_clips.values()]}
    enabled[0] = False
    archive.materialize_key_materials(layout, events, [group], views, infos, transforms, paths, config,
                                     materialize_event_ids={'EVT-A'})
    assert len(events[0].key_frames) == len(events[0].key_clips) == 3
    assert events[0].observability['key_material_materialization']['status'] == 'completed'
    assert all((layout.root/p).read_bytes() == content for p, content in previous_b.items())
    assert events[0].formal_admission_status == 'provisional'
    restored = json.loads(runtime_path.read_text())
    assert restored['status'] == 'completed'
    assert restored['incomplete_artifacts'] == restored['retry_event_ids'] == []
    assert restored['materialization_passes'][0]['incomplete_artifacts'] == runtime['incomplete_artifacts']
    assert len(restored['records']) == 6


def test_programming_errors_are_not_disguised_as_missing_media():
    def invalid():
        raise TypeError('programming error')
    with pytest.raises(TypeError, match='programming error'):
        archive._attempt_key_material('key_frame', invalid)

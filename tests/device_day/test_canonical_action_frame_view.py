"""Materialization keeps configured references and canonical native readers."""
from types import SimpleNamespace

import numpy as np

from visioncortex.device_day_models import DeviceDayModels
from visioncortex.schemas import FrameEvidence, SourceFrameIdentity, VideoInfo, ViewInput


def test_action_frame_adapts_the_original_input_port_without_rewriting_ledger(tmp_path, monkeypatch):
    from visioncortex import device_day_models, source_frames
    physical = tmp_path / 'physical'
    physical.mkdir()
    video = physical / 'video.mp4'
    video.write_bytes(b'owned source fixture')
    alias = tmp_path / 'nas'
    alias.symlink_to(physical, target_is_directory=True)
    view = ViewInput(view_id='camera', role='third_person', video=alias / video.name)
    stat = video.stat()
    native = SourceFrameIdentity(source_path=video, source_size_bytes=stat.st_size,
                                 source_mtime_ns=stat.st_mtime_ns, status='resolved',
                                 packet_position=1, source_pts=0, time_base='1/1000',
                                 source_frame_index=0, decoded_pixels_sha256='a' * 64)
    evidence = FrameEvidence(view_id=view.view_id, role=view.role, frame_index=0, local_ms=0,
                             width=16, height=16, source_frame=native)
    before = evidence.model_dump(mode='json')
    calls = []

    def original_input_port(selected, item):
        # The original port's same-path branch uses literal Path equality.
        assert selected.video == item.source_frame.source_path
        calls.append('relocate')
        return item, None

    def read_native(selected, info, item):
        assert selected.video == video
        assert item.model_dump(mode='json') == before
        calls.append('read')
        return np.zeros((16, 16, 3), dtype=np.uint8), {'status': 'verified', 'view_local_ms': 0}

    monkeypatch.setattr(device_day_models, 'relocate_evidence', original_input_port)
    monkeypatch.setattr(source_frames, 'read_evidence_frame', read_native)
    layout = SimpleNamespace(root=tmp_path, capture_sources={'retained/video.mp4': view.video})
    info = VideoInfo(path=view.video, duration_ms=1000, fps=10, width=16, height=16, frame_count=10)
    directory = tmp_path / 'frames'
    directory.mkdir()
    result = DeviceDayModels._action_frame(None, layout, view, info, {'frame_evidence': before}, directory)
    assert calls == ['relocate', 'read']
    assert result[0]['source_path'] == 'retained/video.mp4'
    assert result[0]['source_frame_verification']['status'] == 'verified'
    assert view.video == alias / video.name

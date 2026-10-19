"""NAS aliases must not be mistaken for a different native-frame source."""
from pathlib import Path
import os

import pytest

from visioncortex.device_day_inputs import InputChanged, input_context, relocate_evidence
from visioncortex.schemas import FrameEvidence, SourceFrameIdentity, VideoInfo, ViewInput
from visioncortex.source_frames import read_evidence_frame


@pytest.fixture
def aliased_frame(tmp_path):
    original = tmp_path / 'capture.mp4'
    original.write_bytes(b'original sealed bytes')
    physical = tmp_path / 'archive'
    physical.mkdir()
    retained = physical / 'video.mp4'
    retained.write_bytes(original.read_bytes())
    alias = tmp_path / 'nas'
    alias.symlink_to(physical, target_is_directory=True)
    view = ViewInput(view_id='camera', role='third_person', video=alias / 'video.mp4')
    stat = retained.stat()
    frame = SourceFrameIdentity(source_path=retained, source_size_bytes=stat.st_size,
                                source_mtime_ns=stat.st_mtime_ns, status='resolved',
                                packet_position=1, source_pts=0, time_base='1/1000',
                                source_frame_index=0, decoded_pixels_sha256='a' * 64)
    evidence = FrameEvidence(view_id=view.view_id, role=view.role, frame_index=0,
                             local_ms=0, width=16, height=16, source_frame=frame)
    # The capture is a different path and can have a different timestamp from
    # its retained copy. Neither is the native ledger's source to relocate.
    seal = {'sources': [{'kind': 'video', 'original_path': str(original),
                         'resolved_path': str(original), 'size_bytes': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns - 1}]}
    return view, evidence, seal, retained


def test_archive_alias_preserves_the_original_native_ledger(aliased_frame):
    view, evidence, seal, _ = aliased_frame
    before = evidence.model_dump(mode='json')
    with input_context(seal):
        result, proof = relocate_evidence(view, evidence)
    assert result is evidence and proof is None
    assert result.model_dump(mode='json') == before
    assert result.source_frame.source_path != view.video
    assert result.source_frame.source_path == view.video.resolve()


@pytest.mark.parametrize('replacement', [b'changed', b'original sealed byteX'])
def test_alias_does_not_bypass_native_source_change_checks(aliased_frame, monkeypatch, replacement):
    view, evidence, seal, retained = aliased_frame
    retained.write_bytes(replacement)
    os.utime(retained, ns=(retained.stat().st_atime_ns,
                          evidence.source_frame.source_mtime_ns + 1_000_000_000))
    import visioncortex.source_frames as frames

    def unexpected_decode(*args, **kwargs):
        pytest.fail('A changed source must be rejected before decoding')

    monkeypatch.setattr(frames, '_run', unexpected_decode)
    with input_context(seal):
        result, _ = relocate_evidence(view, evidence)
    info = VideoInfo(path=view.video, duration_ms=1000, fps=10, width=16, height=16, frame_count=10)
    image, proof = read_evidence_frame(view, info, result)
    assert image is None and proof['reason'] == 'source_file_changed'


def test_same_basename_in_another_archive_is_not_the_same_source(aliased_frame, tmp_path):
    view, evidence, seal, retained = aliased_frame
    other = tmp_path / 'other'
    other.mkdir()
    other_path = other / retained.name
    other_path.write_bytes(retained.read_bytes())
    changed = view.model_copy(update={'video': Path(other_path)})
    with input_context(seal), pytest.raises(InputChanged, match='does not bind'):
        relocate_evidence(changed, evidence)

import hashlib
import io
import json
import queue
import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from visioncortex.detection import FramePacket, _producer, _read_checkpoint, _write_checkpoint
from visioncortex.schemas import FrameEvidence, SourceFrameIdentity, VideoSegmentInput, ViewInput, ViewRole
from visioncortex.source_frames import SOURCE_FRAME_CONTRACT, SampledFrame, SourceFrameTrace, read_evidence_frame, retime_sampled_frame
from visioncortex.video_io import _ffmpeg_frame_iterator, _ffmpeg_multi_window_iterator, _ffmpeg_passthrough_arguments, probe_video


@pytest.fixture
def source_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires FFmpeg and ffprobe")
    path = tmp_path / "variable-timestamps.mp4"
    generated = subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=30:d=2",
        "-vf", "select=not(eq(mod(n\\,5)\\,1)),setpts=PTS+4/TB",
        *_ffmpeg_passthrough_arguments(), "-c:v", "libx264", "-bf", "3", str(path),
    ], capture_output=True)
    assert generated.returncode == 0, generated.stderr.decode(errors="replace")
    return path


@pytest.mark.parametrize("start_ms", [0, 375])
def test_native_pts_survive_vfr_b_frames_nonzero_origin_and_seek(source_video, start_ms):
    info = probe_video(source_video)
    frames = list(_ffmpeg_frame_iterator(source_video, info, start_ms, 1400, 8, 96, None, False, 1))
    native = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp,pkt_pos", "-of", "json", str(source_video),
    ]))["frames"]
    pts = {int(row["pkt_pos"]): int(row["best_effort_timestamp"]) for row in native}
    assert len(frames) >= 8
    for item in frames:
        identity = item.source_frame
        assert identity.status == "resolved", identity.model_dump()
        assert identity.source_pts == pts[identity.packet_position]
        assert float(identity.source_pts * Fraction(identity.time_base)) >= 4
        assert identity.decoded_pixels_sha256 == hashlib.sha256(item[2].tobytes()).hexdigest()
        if start_ms:
            assert identity.source_frame_index is None
    assert frames[0][1] == start_ms  # Grid time stays separate from native PTS.


@pytest.mark.parametrize("start_ms", [0, 1125])
def test_material_frame_matches_native_pts_and_exact_model_pixels(source_video, start_ms):
    info = probe_video(source_video)
    frame = list(_ffmpeg_frame_iterator(source_video, info, start_ms, start_ms + 125, 8, 64, None, False, 1))[0]
    evidence = FrameEvidence(
        view_id="fp", role=ViewRole.FIRST_PERSON, frame_index=frame[0], local_ms=frame[1],
        width=frame[2].shape[1], height=frame[2].shape[0], source_frame=frame.source_frame,
    )
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=source_video)
    actual, receipt = read_evidence_frame(view, info, evidence)
    assert receipt["status"] == "verified", receipt
    assert np.array_equal(actual, frame[2])
    assert receipt["source_format_start_ms"] == 4000
    assert receipt["view_local_ms"] == pytest.approx(float(frame.source_frame.source_pts * Fraction(frame.source_frame.time_base) * 1000) - 4000)
    assert receipt["detections_bound_to_pixels"] is True


@pytest.mark.parametrize("change,reason", [
    ("role", "ledger_view_or_role_mismatch"),
    ("path", "source_not_unique_in_view"),
    ("stat", "source_file_changed"),
    ("position", "decoded_native_identity_mismatch"),
    ("pixels", "decoded_pixels_do_not_match_ledger"),
    ("pts", "decoded_native_identity_mismatch"),
])
def test_material_frame_rejects_changed_or_wrong_source_identity(source_video, tmp_path, change, reason):
    info = probe_video(source_video)
    sampled = list(_ffmpeg_frame_iterator(source_video, info, 0, 125, 8, 64, None, False, 1))[0]
    evidence = FrameEvidence(
        view_id="fp", role=ViewRole.FIRST_PERSON, frame_index=sampled[0], local_ms=sampled[1],
        width=sampled[2].shape[1], height=sampled[2].shape[0], source_frame=sampled.source_frame,
    )
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=source_video)
    if change == "role":
        evidence.role = ViewRole.THIRD_PERSON
    else:
        updates = {
            "path": {"source_path": tmp_path / "another-camera.mp4"},
            "stat": {"source_size_bytes": 0},
            "position": {"packet_position": sampled.source_frame.packet_position + 1},
            "pixels": {"decoded_pixels_sha256": "0" * 64},
            "pts": {"source_pts": sampled.source_frame.source_pts + 1},
        }
        evidence.source_frame = evidence.source_frame.model_copy(update=updates[change])
    actual, receipt = read_evidence_frame(view, info, evidence)
    assert actual is None
    assert receipt["reason"] == reason
    assert receipt["detections_bound_to_pixels"] is False


def test_material_frame_virtual_segment_origin_does_not_change_native_identity(source_video):
    from visioncortex.schemas import VideoSegmentInfo
    info = probe_video(source_video)
    sampled = list(_ffmpeg_frame_iterator(source_video, info, 0, 125, 8, 64, None, False, 1))[0]
    evidence = FrameEvidence(
        view_id="fp", role=ViewRole.FIRST_PERSON, frame_index=999, local_ms=30000,
        width=sampled[2].shape[1], height=sampled[2].shape[0], source_frame=sampled.source_frame,
    )
    info.segments = [VideoSegmentInfo(
        **info.model_dump(exclude={"segments", "source_clock_duration_ms", "media_timing_source"}),
        virtual_start_ms=30000, virtual_end_ms=32000, frame_start_index=900,
    )]
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, segments=[VideoSegmentInput(video=source_video)])
    actual, receipt = read_evidence_frame(view, info, evidence)
    assert np.array_equal(actual, sampled[2])
    assert receipt["view_local_ms"] == 30000 + receipt["source_media_ms"]


def test_disjoint_windows_keep_physical_identity_when_virtual_time_changes(source_video):
    info = probe_video(source_video)
    frames = list(_ffmpeg_multi_window_iterator(
        source_video, info, 0, 1500, [(0, 500), (1000, 1500)], 8, 96, None, 1,
    ))
    assert [item[1] for item in frames] == [0, 125, 250, 375, 1000, 1125, 1250, 1375]
    for item in frames:
        assert item.source_frame.status == "resolved"
        shifted = retime_sampled_frame(item, item[0] + 900, item[1] + 30000)
        assert shifted.source_frame == item.source_frame
        assert shifted[1] == item[1] + 30000


def test_terminal_hold_reuses_original_identity_and_is_explicit(source_video):
    info = probe_video(source_video)
    receipt = {}
    frames = list(_ffmpeg_multi_window_iterator(
        source_video, info, 0, 2250, [(0, 2250)], 8, 96, None, 1, receipt=receipt,
    ))
    assert receipt["terminal_eof_fill_frames"] > 0
    first_hold = next(i for i, frame in enumerate(frames) if frame.source_frame.terminal_frame_hold)
    original = frames[first_hold - 1].source_frame
    for frame in frames[first_hold:]:
        assert frame.source_frame.model_copy(update={"terminal_frame_hold": False}) == original


def test_identical_images_keep_distinct_source_timestamps(source_video, tmp_path):
    path = tmp_path / "static.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=96x64:r=30:d=1",
        "-c:v", "libx264", str(path),
    ], capture_output=True, check=True)
    frames = list(_ffmpeg_frame_iterator(path, probe_video(path), 0, 750, 8, 96, None, False, 1))
    assert len(frames) == 6
    assert len({frame.source_frame.decoded_pixels_sha256 for frame in frames}) == 1
    assert len({frame.source_frame.source_pts for frame in frames}) == 6
    assert all(frame.source_frame.status == "resolved" for frame in frames)


def test_ambiguous_or_missing_positions_never_acquire_invented_timestamps(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"only a parser fixture")
    monkeypatch.setattr("visioncortex.source_frames.shutil.which", lambda _name: "ffprobe")
    monkeypatch.setattr(SourceFrameTrace, "_probe", lambda *_: {
        "streams": [{"time_base": "1/30"}],
    })
    monkeypatch.setattr("visioncortex.source_frames._native_rows", lambda *_: [
        {"pkt_pos": "20", "best_effort_timestamp": 1},
        {"pkt_pos": "20", "best_effort_timestamp": 2},
    ])
    monkeypatch.setattr("visioncortex.source_frames._encoder_stats_supported", lambda _: False)
    from visioncortex import source_frames as sf
    metadata_probe = SourceFrameTrace._probe
    monkeypatch.setattr(SourceFrameTrace, "_probe", lambda self, args: {**metadata_probe(self, args), "frames": sf._native_rows(self.path, 0, 1)})
    trace = SourceFrameTrace(source, 0, 1000, 8)
    log = io.BytesIO(b"[showinfo@source_identity] n: 0 pts: 0 pos: 20 fmt:yuv420p\n"
                     b"[showinfo@source_identity] n: 1 pts: 1 pos: -1 fmt:yuv420p\n")
    trace.start(log)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    first, second = trace.identity(0, frame), trace.identity(1, frame)
    assert first.status == "ambiguous"
    assert second.status == "unavailable"
    assert first.source_pts is second.source_pts is None
    assert first.source_frame_index is second.source_frame_index is None
    trace.finish(log)


def test_metadata_probe_failure_keeps_pixels_without_claiming_native_identity(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"only a parser fixture")
    monkeypatch.setattr("visioncortex.source_frames.shutil.which", lambda _name: "ffprobe")

    def fail(*_):
        raise OSError("probe unavailable")

    monkeypatch.setattr(SourceFrameTrace, "_probe", fail)
    monkeypatch.setattr("visioncortex.source_frames._encoder_stats_supported", lambda _: False)
    trace = SourceFrameTrace(source, 0, 1000, 8)
    trace.start(io.BytesIO(b"[showinfo@source_identity] n: 0 pts: 0 pos: 20 fmt:yuv420p\n"))
    identity = trace.identity(0, np.zeros((8, 8, 3), dtype=np.uint8))
    assert identity.status == "unavailable"
    assert identity.reason == "native_frame_probe_unavailable"
    assert identity.source_pts is None
    trace.finish(None)


def test_showinfo_without_packet_position_reports_missing_identity_without_waiting(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"parser fixture")
    monkeypatch.setattr("visioncortex.source_frames.shutil.which", lambda _name: "ffprobe")
    monkeypatch.setattr(SourceFrameTrace, "_probe", lambda *_: {
        "streams": [{"time_base": "1/30"}], "frames": [],
    })
    monkeypatch.setattr("visioncortex.source_frames._native_rows", lambda *_: [])
    monkeypatch.setattr("visioncortex.source_frames._encoder_stats_supported", lambda _: False)
    from visioncortex import source_frames as sf
    metadata_probe = SourceFrameTrace._probe
    monkeypatch.setattr(SourceFrameTrace, "_probe", lambda self, args: {**metadata_probe(self, args), "frames": sf._native_rows(self.path, 0, 1)})
    trace = SourceFrameTrace(source, 0, 1000, 8)
    trace.start(io.BytesIO(b"[showinfo@source_identity] n: 0 pts: 0 pts_time:0 duration:1 fmt:yuv420p\n"))
    trace.thread.join()
    assert trace.records.queue[0] == (0, -1, None)
    identity = trace.identity(0, np.zeros((8, 8, 3), dtype=np.uint8))
    assert identity.status == "unavailable"
    assert identity.reason == "decoder_frame_position_unavailable"
    assert identity.source_pts is None
    trace.finish(None)


@pytest.mark.parametrize("case", [
    "match", "progress_prefix", "crlf", "progress_crlf", "duplicate_pts", "missing_pts", "wrong_timebase", "invalid_timebase",
    "unknown_pts", "missing_stats", "wrong_index", "probe_failure",
])
def test_mux_identity_requires_unique_native_pts_and_matching_timebase(tmp_path, monkeypatch, case):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"mux parser fixture")
    monkeypatch.setattr("visioncortex.source_frames.shutil.which", lambda name: name)
    monkeypatch.setattr("visioncortex.source_frames._encoder_stats_supported", lambda _: True)

    def metadata(*_):
        if case == "probe_failure":
            raise OSError("probe unavailable")
        return {"streams": [{"time_base": "1/30"}]}

    monkeypatch.setattr(SourceFrameTrace, "_probe", metadata)
    rows = [{"pkt_pos": "20", "best_effort_timestamp": 1}]
    if case == "duplicate_pts":
        rows.append({"pkt_pos": "21", "best_effort_timestamp": 1})
    monkeypatch.setattr("visioncortex.source_frames._native_rows", lambda *_: rows)
    from visioncortex import source_frames as sf
    metadata_probe = SourceFrameTrace._probe
    monkeypatch.setattr(SourceFrameTrace, "_probe", lambda self, args: {**metadata_probe(self, args), "frames": sf._native_rows(self.path, 0, 1)})
    trace = SourceFrameTrace(source, 0, 1000, 8)
    pts = {"missing_pts": 2, "unknown_pts": 9223372036854775807}.get(case, 1)
    time_base = {"wrong_timebase": "1/60", "invalid_timebase": "1/0"}.get(case, "1/30")
    index = 1 if case == "wrong_index" else 0
    log = f"VC_SOURCE {index} {pts} {time_base}\n" if case != "missing_stats" else "unrelated log\n"
    if case in ("progress_prefix", "progress_crlf"):
        log = "frame=8 fps=8.0 time=00:00:01.00\r" + log
    if case in ("crlf", "progress_crlf"):
        log = log.replace("\n", "\r\n")
    trace.start(io.BytesIO(log.encode()))
    trace.thread.join(timeout=2)
    assert not trace.thread.is_alive()
    identity = trace.identity(0, np.zeros((8, 8, 3), dtype=np.uint8))
    resolved = case in ("match", "progress_prefix", "crlf", "progress_crlf")
    assert identity.status == ("resolved" if resolved else "unavailable")
    if resolved:
        assert (identity.source_pts, identity.packet_position, identity.time_base) == (1, 20, "1/30")
    else:
        assert identity.source_pts is identity.time_base is None
    trace.finish(None)


def test_streaming_decoder_progress_does_not_hide_frame_identity(source_video, monkeypatch):
    original_popen = subprocess.Popen

    def streaming(command, **kwargs):
        if command[0] == "ffmpeg" and "rawvideo" in command:
            command = [command[0], "-re", "-stats_period", "0.01", *command[1:]]
        return original_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", streaming)
    frames = list(_ffmpeg_frame_iterator(source_video, probe_video(source_video), 0, 1000, 8, 96, None, False, 1))
    assert len(frames) == 8
    assert all(frame.source_frame.status == "resolved" for frame in frames)


@pytest.mark.parametrize("start_ms", [111.345, 375])
@pytest.mark.parametrize("multi_window", [False, True])
def test_mux_trace_preserves_existing_sampling_with_fractional_origin(tmp_path, monkeypatch, start_ms, multi_window):
    from visioncortex import source_frames
    if not source_frames._encoder_stats_supported(shutil.which("ffmpeg")):
        pytest.skip("requires FFmpeg 6+ pre-encoding statistics")
    path = tmp_path / "fractional-origin.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=30:d=2",
        "-vf", "select=not(eq(mod(n\\,5)\\,1)),setpts=PTS+4.037/TB",
        *_ffmpeg_passthrough_arguments(), "-c:v", "libx264", "-bf", "3", str(path),
    ], capture_output=True, check=True)
    info = probe_video(path)

    def sample():
        if multi_window:
            return list(_ffmpeg_multi_window_iterator(
                path, info, start_ms, start_ms + 1250,
                [(start_ms, start_ms + 375), (start_ms + 875, start_ms + 1250)], 8, 96, None, 1,
            ))
        return list(_ffmpeg_frame_iterator(path, info, start_ms, start_ms + 1000, 8, 96, None, False, 1))

    current = sample()
    monkeypatch.setattr(source_frames, "_encoder_stats_supported", lambda _: False)
    reference = sample()
    assert current and len(current) == len(reference)
    for actual, expected in zip(current, reference, strict=True):
        assert actual[:2] == expected[:2]
        assert np.array_equal(actual[2], expected[2])
        assert actual.source_frame.status == "resolved"


def test_frame_identity_reaches_model_packet_without_retiming(default_config, monkeypatch, tmp_path):
    path = tmp_path / "source.mp4"
    view = ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=path)
    from visioncortex.schemas import VideoInfo
    info = VideoInfo(path=path, duration_ms=125, fps=30, width=8, height=8, frame_count=4)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    identity = SourceFrameIdentity(
        source_path=path, source_size_bytes=1, source_mtime_ns=1, status="resolved",
        packet_position=40, source_pts=512, time_base="1/15360", source_frame_index=1,
        decoded_pixels_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
    )
    monkeypatch.setattr("visioncortex.detection.iter_view_sampled_frames",
                        lambda *_: iter([SampledFrame(0, 0, frame, identity)]))
    output = queue.Queue()
    _producer(view, info, output, set(), default_config, None, 8, 8, False, "cpu", 1, (8, 8), None)
    packets = []
    while not output.empty():
        item = output.get()
        if isinstance(item, FramePacket):
            packets.append(item)
    assert len(packets) == 1
    assert packets[0].local_ms == 0
    assert packets[0].source_frame == identity


def test_old_checkpoint_cannot_skip_source_identity_work_or_truncate_evidence(tmp_path):
    ledger, checkpoint = tmp_path / "frames.jsonl", tmp_path / "checkpoint.json"
    ledger.write_bytes(b"durable\n")
    _write_checkpoint(checkpoint, {0}, ledger)
    ledger.write_bytes(ledger.read_bytes() + b"partial-tail\n")
    before = ledger.read_bytes()
    with pytest.raises(RuntimeError, match="Source frame identity contract changed"):
        _read_checkpoint(checkpoint, ledger, source_frame_contract=SOURCE_FRAME_CONTRACT)
    assert ledger.read_bytes() == before


def test_partial_decode_failure_does_not_replay_a_different_decoder(monkeypatch, tmp_path):
    from visioncortex import video_io
    from visioncortex.schemas import VideoInfo
    path = tmp_path / "missing.mp4"
    info = VideoInfo(path=path, duration_ms=1000, fps=30, width=8, height=8, frame_count=30)
    calls = []

    def fail_after_one(*_args):
        calls.append(1)
        yield (0, 0, np.zeros((8, 8, 3), dtype=np.uint8))
        raise RuntimeError("decoder failed after delivering one source frame")

    monkeypatch.setattr(video_io.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(video_io, "_ffmpeg_frame_iterator", fail_after_one)
    frames = video_io.iter_sampled_frames(path, info, 0, 1000, 8, 8, hwaccel="cuda")
    assert next(frames)[1] == 0
    with pytest.raises(RuntimeError, match="decoder failed after delivering"):
        next(frames)
    assert calls == [1]


def test_material_frame_reproduces_native_size_nv12_conversion(source_video):
    info = probe_video(source_video)
    sampled = next(_ffmpeg_frame_iterator(source_video, info, 0, 125, 8, 96, None, False, 1))
    identity = sampled.source_frame
    nv12 = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts",
        "-threads", "2", "-i", str(source_video), "-vf",
        f"select=eq(pts\\,{identity.source_pts}),format=nv12,scale=96:64",
        "-an", "-sn", *_ffmpeg_passthrough_arguments(), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ], capture_output=True, check=True).stdout
    evidence = FrameEvidence(
        view_id="tp", role=ViewRole.THIRD_PERSON, frame_index=sampled[0], local_ms=sampled[1],
        width=96, height=64,
        source_frame=identity.model_copy(update={"decoded_pixels_sha256": hashlib.sha256(nv12).hexdigest()}),
    )
    view = ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=source_video)
    actual, receipt = read_evidence_frame(view, info, evidence)
    assert receipt["status"] == "verified", receipt
    assert actual.tobytes() == nv12
    assert receipt["source_frame"]["source_pts"] == identity.source_pts
    assert receipt["source_frame"]["packet_position"] == identity.packet_position
    expected_format = "decoder_default" if sampled[2].tobytes() == nv12 else "nv12"
    assert receipt["reproduction_pixel_format"] == expected_format


@pytest.mark.parametrize("encoder_stats", [False, True])
@pytest.mark.parametrize("second_result,reason", [
    ("match", None),
    ("wrong_identity", "decoded_native_identity_mismatch"),
    ("wrong_pixels", "decoded_pixels_do_not_match_ledger"),
    ("changed_source", "source_file_changed_during_decode"),
])
def test_nv12_retry_is_bounded_and_preserves_source_checks(tmp_path, monkeypatch, second_result, reason, encoder_stats):
    monkeypatch.setattr("visioncortex.source_frames._encoder_stats_supported", lambda _: encoder_stats)
    monkeypatch.setattr("visioncortex.video_io._ffmpeg_passthrough_arguments", lambda: ("-fps_mode", "passthrough"))
    from visioncortex.schemas import VideoInfo
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"decoder contract fixture")
    target = np.full((8, 8, 3), 127, dtype=np.uint8).tobytes()
    wrong = bytes(len(target))
    stat = path.stat()
    identity = SourceFrameIdentity(
        source_path=path, source_size_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
        status="resolved", packet_position=42, source_pts=1, time_base="1/30",
        decoded_pixels_sha256=hashlib.sha256(target).hexdigest(),
    )
    evidence = FrameEvidence(
        view_id="tp", role=ViewRole.THIRD_PERSON, frame_index=0, local_ms=0,
        width=8, height=8, source_frame=identity,
    )
    view = ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=path)
    info = VideoInfo(path=path, duration_ms=1000, fps=30, width=8, height=8, frame_count=30)
    calls = []

    def decode(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "streams": [{"time_base": "1/30"}], "format": {"start_time": "0"},
                "frames": [{"pkt_pos": "42", "best_effort_timestamp": 1}],
            }).encode(), b"")
        calls.append(command)
        assert len(calls) <= 2
        assert kwargs["timeout"] == 20
        is_retry = len(calls) == 2
        assert ("format=nv12," in command[command.index("-vf") + 1]) == is_retry
        wrong_identity = is_retry and second_result == "wrong_identity"
        position = 43 if wrong_identity else 42
        pixels = target if is_retry and second_result != "wrong_pixels" else wrong
        if is_retry and second_result == "changed_source":
            path.write_bytes(b"changed file")
        log = ("[showinfo@source_identity] config in time_base: 1/30\n"
               f"[showinfo@source_identity] n: 0 pts: 1 pos: {position}\n")
        if encoder_stats:
            assert "-stats_enc_pre" in command
            log += f"VC_SOURCE 0 {2 if wrong_identity else 1} 1/30\n"
        return subprocess.CompletedProcess(command, 0, pixels, log.encode())

    monkeypatch.setattr("visioncortex.source_frames.subprocess.run", decode)
    actual, receipt = read_evidence_frame(view, info, evidence)
    assert len(calls) == 2
    if reason is None:
        assert actual.tobytes() == target
        assert receipt["reproduction_pixel_format"] == "nv12"
        assert receipt["detections_bound_to_pixels"] is True
    else:
        assert actual is None
        assert receipt["reason"] == reason
        assert receipt["detections_bound_to_pixels"] is False


def test_native_index_reuse_preserves_ambiguity_and_invalidates_source(tmp_path, monkeypatch):
    source = tmp_path / 'native.mp4'
    source.write_bytes(b'original')
    calls = []
    monkeypatch.setattr('visioncortex.source_frames.shutil.which', lambda _: 'ffprobe')
    def probe(self, arguments):
        calls.append(arguments)
        return {'streams': [{'time_base': '1/1000'}], 'frames': [
            {'pkt_pos': '10', 'best_effort_timestamp': 0},
            {'pkt_pos': '10', 'best_effort_timestamp': 1},
        ]}
    monkeypatch.setattr(SourceFrameTrace, '_probe', probe)
    first = SourceFrameTrace(source, 0, 1000, 2)
    second = SourceFrameTrace(source, 0, 1000, 20)
    assert len(calls) == 2
    assert second.positions == first.positions == {10: [(0, 0), (1, 1)]}
    subwindow = SourceFrameTrace(source, 100, 500, 20)
    assert subwindow.positions == {10: [(0, None), (1, None)]}
    assert len(calls) == 2
    second.positions.clear()
    assert SourceFrameTrace(source, 0, 1000, 20).positions == first.positions
    SourceFrameTrace(source, 0, 2000, 20)
    assert len(calls) == 4
    source.write_bytes(b'changed source')
    SourceFrameTrace(source, 0, 1000, 20)
    assert len(calls) == 6


def test_native_index_cache_never_retains_probe_failure(tmp_path, monkeypatch):
    source = tmp_path / 'retry.mp4'
    source.write_bytes(b'original')
    monkeypatch.setattr('visioncortex.source_frames.shutil.which', lambda _: 'ffprobe')
    calls = []
    def probe(self, arguments):
        calls.append(arguments)
        raise ValueError('probe unavailable')
    monkeypatch.setattr(SourceFrameTrace, '_probe', probe)
    for _ in range(2):
        assert SourceFrameTrace(source, 0, 1000, 2).failure == 'native_frame_probe_unavailable'
    assert len(calls) == 2


def test_native_index_retains_many_camera_lanes_with_bounded_rows(tmp_path, monkeypatch):
    from collections import OrderedDict
    from visioncortex import source_frames as module
    monkeypatch.setattr(module, "_NATIVE_INDEX_CACHE", OrderedDict())
    monkeypatch.setattr(module, "_NATIVE_INDEX_TOTAL_ROWS", 12)
    monkeypatch.setattr(module.shutil, "which", lambda _: "ffprobe")
    calls = []
    def probe(self, args):
        calls.append(self.path)
        return {"streams": [{"time_base": "1/1000"}], "frames": [
            {"pkt_pos": "10", "best_effort_timestamp": 1}]}
    monkeypatch.setattr(SourceFrameTrace, "_probe", probe)
    paths = [tmp_path / f"Camera{i}.mp4" for i in range(12)]
    for path in paths:
        path.write_bytes(b"fixture")
        SourceFrameTrace(path, 0, 1000, 2)
    assert len(calls) == 24
    for path in paths:
        assert SourceFrameTrace(path, 0, 1000, 20).positions
    assert len(calls) == 24
    extra = tmp_path / "Extra.mp4"
    extra.write_bytes(b"fixture")
    SourceFrameTrace(extra, 0, 1000, 2)
    assert len(module._NATIVE_INDEX_CACHE) == 12
    SourceFrameTrace(paths[0], 0, 1000, 2)
    assert len(calls) == 28


@pytest.fixture
def seek_gap_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires FFmpeg and ffprobe")
    path = tmp_path / "seek-gap.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=30:d=4",
        "-vf", "select=not(between(t\\,1\\,2.5))", "-vsync", "0",
        "-c:v", "libx264", "-g", "30", str(path),
    ], capture_output=True, check=True)
    return path


def test_seek_into_gap_uses_output_pts_not_emitted_frame_ordinal(seek_gap_video):
    info = probe_video(seek_gap_video)
    frames = list(_ffmpeg_frame_iterator(seek_gap_video, info, 1500, 3500, 10, 96, None, False, 1))
    assert frames
    assert frames[0][1] >= 2500
    assert frames[-1][1] < 3500
    assert all(frame.source_frame.status == "resolved" for frame in frames)
    assert [frame[1] for frame in frames] == list(range(2500, 3500, 100))


def test_persistent_seek_gap_cannot_publish_retimed_frames(seek_gap_video):
    info = probe_video(seek_gap_video)
    with pytest.raises(RuntimeError, match="Source timestamp gap"):
        list(_ffmpeg_multi_window_iterator(
            seek_gap_video, info, 1500, 3500, [(1500, 3500)], 10, 96, None, 1,
        ))


def test_native_gap_is_not_an_inactive_interval(seek_gap_video):
    from visioncortex.source_frames import source_observation_windows
    info = probe_video(seek_gap_video)
    trace = SourceFrameTrace(seek_gap_video, 0, info.duration_ms, 2)
    coverage = source_observation_windows(trace, info.duration_ms, 30)
    assert coverage["gap_activity"] == "not_assessed"
    assert coverage["unavailable_intervals"] == [pytest.approx([1000, 2533.333333])]
    assert len(coverage["available_windows"]) == 2
    assert coverage["available_windows"][0][1] == pytest.approx(1000)
    assert coverage["available_windows"][1][0] == pytest.approx(2533.333333)


def test_normal_vfr_nonzero_origin_keeps_full_observation_window(source_video):
    from visioncortex.source_frames import source_observation_windows
    info = probe_video(source_video)
    coverage = source_observation_windows(SourceFrameTrace(source_video, 0, info.duration_ms, 2), info.duration_ms, info.fps)
    assert coverage["unavailable_intervals"] == []
    assert coverage["available_windows"] == [[0, info.duration_ms]]


def test_recorder_clock_uses_native_pts_across_gap(seek_gap_video, tmp_path):
    from visioncortex.alignment import read_video_timestamp_csv
    import csv
    info = probe_video(seek_gap_video)
    native = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(seek_gap_video),
    ]))["frames"]
    path = tmp_path / "Frames.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rgb_video_frame_index", "rgb_recorded", "local_time_us", "global_timestamp_us", "clock_sync_valid"])
        for index, frame in enumerate(native):
            instant = 1789358150000000 + round(float(frame['best_effort_timestamp_time']) * 1e6)
            writer.writerow([index, 1, instant, instant, 1])
    points = read_video_timestamp_csv(path, info, sample_count=4096)
    assert len(points) == len(native)
    assert [p.local_ms for p in points] == pytest.approx([float(f['best_effort_timestamp_time']) * 1000 for f in native], abs=.001)
    assert max(b.local_ms - a.local_ms for a, b in zip(points, points[1:], strict=False)) > 1500


def test_device_action_material_uses_verified_model_pixels(source_video, tmp_path, default_config):
    from types import SimpleNamespace
    from visioncortex.device_day_models import DeviceDayModels
    info = probe_video(source_video)
    frame = list(_ffmpeg_frame_iterator(source_video, info, 1125, 1250, 8, 64, None, False, 1))[0]
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=source_video)
    evidence = FrameEvidence(view_id="fp", role=view.role, frame_index=frame[0], local_ms=frame[1],
                             width=frame[2].shape[1], height=frame[2].shape[0], source_frame=frame.source_frame)
    layout = SimpleNamespace(root=tmp_path, relative=lambda path: path.relative_to(tmp_path).as_posix())
    refs = DeviceDayModels(default_config)._action_frame(layout, view, info, {"frame_evidence": evidence.model_dump(mode="json")}, tmp_path / "KeyFrames")
    assert len(refs) == 1
    assert refs[0]["source_time_basis"] == "verified_native_source_frame"
    assert refs[0]["source_frame_verification"]["detections_bound_to_pixels"] is True
    assert refs[0]["source_frame_verification"]["source_frame"]["decoded_pixels_sha256"] == frame.source_frame.decoded_pixels_sha256

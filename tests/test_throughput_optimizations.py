import base64
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import httpx
import cv2

from visioncortex import storage, video_io
from visioncortex.archive import _run_bounded_semantic_waves
from visioncortex.mllm import ArkAnalyzer, _image_data_url
from visioncortex.schemas import VideoInfo, ViewInput, ViewRole
from visioncortex.storage import IncrementalArchivePublisher
from visioncortex.telemetry import ResourceMonitor, _NvmlSampler
from visioncortex.video_io import ViewFrameReader


def test_mllm_reuses_one_http_connection_pool(monkeypatch, default_config):
    clients = []

    class FakeResponse:
        is_error = False

        def json(self):
            return {
                "choices": [{"message": {"content": json.dumps({"confidence": 0.9})}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

    class FakeClient:
        def __init__(self, **_kwargs):
            self.posts = 0
            self.closed = False
            clients.append(self)

        def post(self, *_args, **_kwargs):
            self.posts += 1
            return FakeResponse()

        def close(self):
            self.closed = True

    monkeypatch.setenv("ARK_API_KEY", "configured-for-test")
    monkeypatch.setattr("visioncortex.mllm.httpx.Client", FakeClient)
    analyzer = ArkAnalyzer(default_config)
    first = analyzer._call("system", {"event": 1}, [])
    second = analyzer._call("system", {"event": 2}, [])
    analyzer.close()

    assert first["status"] == second["status"] == "completed"
    assert len(clients) == 1
    assert clients[0].posts == 2
    assert clients[0].closed is True


def test_mllm_bounds_wire_image_without_mutating_evidence(tmp_path):
    path = tmp_path / "evidence.jpg"
    source = np.random.default_rng(7).integers(
        0, 256, size=(720, 1280, 3), dtype=np.uint8
    )
    assert cv2.imwrite(
        str(path), source, [cv2.IMWRITE_JPEG_QUALITY, 96]
    )
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    data_url = _image_data_url(path, max_edge=640, jpeg_quality=75)

    encoded = data_url.split(",", 1)[1]
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert max(decoded.shape[:2]) == 640
    assert len(base64.b64decode(encoded)) < path.stat().st_size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_digest


def test_mllm_transport_failure_circuit_skips_later_calls(
    monkeypatch, default_config
):
    class FailingClient:
        def __init__(self, **_kwargs):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            self.posts += 1
            raise httpx.ReadTimeout("timed out")

        def close(self):
            return None

    default_config["mllm"]["max_retries"] = 1
    default_config["mllm"]["failure_circuit_breaker_threshold"] = 2
    monkeypatch.setenv("ARK_API_KEY", "configured-for-test")
    monkeypatch.setattr("visioncortex.mllm.httpx.Client", FailingClient)
    analyzer = ArkAnalyzer(default_config)

    first = analyzer._call("system", {"event": 1}, [])
    second = analyzer._call("system", {"event": 2}, [])
    third = analyzer._call("system", {"event": 3}, [])

    assert first["status"] == second["status"] == "failed"
    assert second["failure_circuit_open"] is True
    assert second["request_image_transport"] == {
        "maximum_edge_pixels": 0,
        "jpeg_quality": 85,
        "source_artifacts_mutated": False,
    }
    assert third["status"] == "skipped_failure_circuit_open"
    assert third["attempts"] == 0
    assert third["request_image_transport"] == second["request_image_transport"]
    assert analyzer.client.posts == 2


def test_semantic_queue_waits_for_failure_bounded_wave_before_scheduling_more():
    failure_threshold = 2
    lock = threading.Lock()
    transport_calls = 0
    circuit_open = False

    def analyze(item):
        nonlocal transport_calls, circuit_open
        with lock:
            if circuit_open:
                return item, "skipped"
            transport_calls += 1
        # Keep both members of the first wave in flight before either opens the
        # circuit, matching concurrent HTTP failures.
        time.sleep(0.01)
        with lock:
            if transport_calls >= failure_threshold:
                circuit_open = True
        return item, "failed"

    results = _run_bounded_semantic_waves(
        list(range(7)),
        analyze,
        workers=8,
        failure_threshold=failure_threshold,
    )

    assert transport_calls == failure_threshold
    assert sum(status == "failed" for _item, status in results) == failure_threshold
    assert sum(status == "skipped" for _item, status in results) == 5


def test_mllm_retries_schema_invalid_response_before_accepting(monkeypatch, default_config):
    valid = {
        "current_step": "抓取离心管",
        "next_step": "移动离心管",
        "next_step_evidence": {
            "status": "inferred",
            "reason": "当前抓取姿态支持谨慎预测",
            "evidence_event_ids": ["EVT-1"],
        },
        "action_type_confirmed": "hand_object_contact",
        "objects": ["gloved_hand", "tube"],
        "hand_object_interactions": [
            {"hand": "right", "object": "tube", "contact": "grasp"}
        ],
        "physical_change": {"before": "未抓取", "after": "已抓取"},
        "per_view_observations": [
            {"view_id": "fp", "observation": "手接触离心管"}
        ],
        "candidate_action_support_by_view": [
            {
                "view_id": "fp",
                "supports_candidate_action": True,
                "confidence": 0.9,
                "reason": "接触清晰",
            }
        ],
        "confirmed_action_support_by_view": [
            {
                "view_id": "fp",
                "supports_confirmed_action": True,
                "confidence": 0.9,
                "reason": "接触清晰",
            }
        ],
        "action_proof": {
            "proof_type": "direct_other",
            "visible_liquid_or_level_change": False,
            "source_contact_visible": False,
            "withdrawal_or_transport_visible": False,
            "target_contact_visible": False,
            "release_or_plunger_change_visible": False,
            "dual_role_cv_sequence_verified": False,
            "container_before_state_visible": False,
            "container_after_state_visible": False,
            "container_state_transition_completed": False,
            "reason": "可见抓取",
        },
        "cross_view_consistency": "single_view",
        "evidence_verdict": "confirmed",
        "temporal_support": {
            "before": "未接触",
            "peak": "抓取",
            "after": "保持抓取",
        },
        "confidence": 0.9,
        "uncertainties": [],
    }
    responses = [{"confidence": 0.9}, valid]

    class FakeResponse:
        is_error = False

        def __init__(self, result):
            self.result = result

        def json(self):
            return {
                "choices": [
                    {"message": {"content": json.dumps(self.result)}}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

    class FakeClient:
        def __init__(self, **_kwargs):
            self.posts = 0

        def post(self, *_args, **_kwargs):
            response = FakeResponse(responses[self.posts])
            self.posts += 1
            return response

        def close(self):
            return None

    default_config["mllm"]["max_retries"] = 2
    monkeypatch.setenv("ARK_API_KEY", "configured-for-test")
    monkeypatch.setattr("visioncortex.mllm.httpx.Client", FakeClient)
    monkeypatch.setattr("visioncortex.mllm.time.sleep", lambda _seconds: None)
    analyzer = ArkAnalyzer(default_config)

    result = analyzer._call(
        "system", {"event_id": "EVT-1"}, [], response_kind="event"
    )

    assert result["status"] == "completed"
    assert result["attempts"] == 2
    assert result["response_contract"] == (
        "visioncortex-event-mllm-response/1"
    )
    assert analyzer.client.posts == 2


def test_mllm_refuses_to_silently_truncate_evidence(monkeypatch, default_config):
    monkeypatch.setenv("ARK_API_KEY", "configured-for-test")
    analyzer = ArkAnalyzer(default_config)
    try:
        with pytest.raises(ValueError, match="silently discard images"):
            analyzer._call(
                "system",
                {"event_id": "EVT-1"},
                [("one", Path("one.jpg")), ("two", Path("two.jpg"))],
                max_images=1,
            )
    finally:
        analyzer.close()


def test_encoder_capability_is_probed_once(monkeypatch):
    calls = []

    def fake_run(command, timeout=None):
        calls.append((command, timeout))
        return SimpleNamespace(returncode=0, stdout=b"h264_nvenc", stderr=b"")

    video_io._encoder_available.cache_clear()
    monkeypatch.setattr(video_io, "_run", fake_run)
    assert video_io._encoder_available("h264_nvenc") is True
    assert video_io._encoder_available("h264_nvenc") is True
    assert len(calls) == 1


def test_encoder_runtime_probe_rejects_nvenc_driver_mismatch(monkeypatch):
    calls = []

    def fake_run(command, timeout=None):
        calls.append((command, timeout))
        if "-encoders" in command:
            return SimpleNamespace(returncode=0, stdout=b"h264_nvenc libx264", stderr=b"")
        encoder = command[command.index("-c:v") + 1]
        if encoder == "h264_nvenc":
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=b"Driver does not support the required nvenc API version 13.0",
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    video_io._encoder_available.cache_clear()
    video_io._encoder_usable.cache_clear()
    monkeypatch.setattr(video_io, "_run", fake_run)

    report = video_io.video_encoder_preflight("h264_nvenc")

    assert report["requested_encoder_listed"] is True
    assert report["requested_encoder_usable"] is False
    assert report["selected_encoder"] == "libx264"
    assert report["software_fallback_active"] is True
    assert sum("-encoders" in command for command, _ in calls) == 2


def test_encoder_runtime_probe_uses_supported_production_dimensions(monkeypatch):
    commands = []

    def fake_run(command, timeout=None):
        commands.append((command, timeout))
        if "-encoders" in command:
            return SimpleNamespace(returncode=0, stdout=b"h264_nvenc", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    video_io._encoder_available.cache_clear()
    video_io._encoder_usable.cache_clear()
    monkeypatch.setattr(video_io, "_run", fake_run)

    assert video_io._encoder_usable("h264_nvenc") is True
    probe = next(command for command, _ in commands if "-f" in command and "lavfi" in command)
    source = probe[probe.index("-i") + 1]
    assert "s=640x360" in source


def test_grid_video_retries_software_encoder_after_runtime_nvenc_failure(
    monkeypatch, tmp_path
):
    encoders = []
    commands = []
    timeouts = []

    monkeypatch.setattr(video_io, "select_video_encoder", lambda _preferred: "h264_nvenc")

    def fake_run(command, timeout=None):
        commands.append(command.copy())
        timeouts.append(timeout)
        encoder = command[command.index("-c:v") + 1]
        encoders.append(encoder)
        return SimpleNamespace(
            returncode=1 if encoder == "h264_nvenc" else 0,
            stdout=b"",
            stderr=b"nvenc mismatch" if encoder == "h264_nvenc" else b"",
        )

    monkeypatch.setattr(video_io, "_run", fake_run)
    video_io.create_grid_video(
        [("first", tmp_path / "first.mp4"), ("third", tmp_path / "third.mp4")],
        tmp_path / "aligned.mp4",
    )

    assert encoders == ["h264_nvenc", "libx264"]
    assert "-cq" in commands[0] and commands[0][commands[0].index("-cq") + 1] == "26"
    assert "-rc" in commands[0] and "-b:v" in commands[0]
    assert "-cq" not in commands[1] and "-rc" not in commands[1]
    assert commands[1][commands[1].index("-crf") + 1] == "23"
    assert all(timeout is not None and 0.0 < timeout <= 120.0 for timeout in timeouts)
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert filter_graph.count("fps=30") == 2
    assert filter_graph.count("settb=AVTB") == 2
    assert filter_graph.count("setpts=N/(30*TB)") == 2
    assert "shortest=1" in filter_graph
    sync_option, sync_value = video_io._ffmpeg_cfr_arguments()
    assert commands[0][commands[0].index(sync_option) + 1] == sync_value


def test_grid_video_timeout_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(video_io, "select_video_encoder", lambda _preferred: "libx264")

    def fake_run(command, timeout=None):
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(video_io, "_run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 0.050 seconds"):
        video_io.create_grid_video(
            [("first", tmp_path / "first.mp4"), ("third", tmp_path / "third.mp4")],
            tmp_path / "aligned.mp4",
            preferred_encoder="libx264",
            timeout_seconds=0.05,
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is optional")
def test_grid_video_bounds_near_matching_fractional_frame_rates(tmp_path):
    first = tmp_path / "first-29.97fps.mp4"
    third = tmp_path / "third-30.04fps.mp4"
    destination = tmp_path / "aligned.mp4"
    for path, rate, duration in (
        (first, "30000/1001", "1.034"),
        (third, "751/25", "1.032"),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size=96x64:rate={rate}",
                "-t",
                duration,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            timeout=10.0,
        )

    started = time.monotonic()
    video_io.create_grid_video(
        [("first", first), ("third", third)],
        destination,
        preferred_encoder="libx264",
        timeout_seconds=10.0,
    )

    rendered = video_io.probe_video(destination)
    assert time.monotonic() - started < 10.0
    assert rendered.duration_ms == pytest.approx(1033.0, abs=80.0)
    assert rendered.fps == pytest.approx(30.0, abs=0.05)
    assert rendered.frame_count <= 32


def test_frame_reader_reuses_decoder_for_same_segment(monkeypatch):
    opened = []

    class FakeCapture:
        def __init__(self, path):
            self.path = path
            self.released = False
            opened.append(self)

        def isOpened(self):
            return not self.released

        def set(self, *_args):
            return True

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(video_io.cv2, "VideoCapture", FakeCapture)
    path = Path("same.mp4")
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=path)
    info = VideoInfo(
        path=path,
        duration_ms=10_000.0,
        fps=30.0,
        width=8,
        height=8,
        frame_count=300,
        size_bytes=100,
    )
    with ViewFrameReader(max_open=2) as reader:
        assert reader.read(view, info, 1_000.0) is not None
        assert reader.read(view, info, 2_000.0) is not None
    assert len(opened) == 1
    assert opened[0].released is True


def test_frame_reader_falls_back_to_bounded_ffmpeg_seek(monkeypatch):
    class FailedCapture:
        def __init__(self, _path):
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            return False, None

        def release(self):
            self.released = True

    encoded, jpeg = video_io.cv2.imencode(
        ".jpg", np.full((8, 12, 3), 127, dtype=np.uint8)
    )
    assert encoded
    commands = []

    def fake_run(command, timeout=None):
        commands.append((command, timeout))
        return subprocess.CompletedProcess(command, 0, jpeg.tobytes(), b"")

    monkeypatch.setattr(video_io.cv2, "VideoCapture", FailedCapture)
    monkeypatch.setattr(video_io, "_run", fake_run)
    path = Path("vfr-derived.mp4")
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=path)
    info = VideoInfo(
        path=path,
        duration_ms=10_000.0,
        fps=30.0,
        width=12,
        height=8,
        frame_count=300,
        size_bytes=100,
    )

    with ViewFrameReader(max_open=1) as reader:
        frame = reader.read(view, info, 2_500.0)

    assert frame is not None
    assert frame.shape == (8, 12, 3)
    assert commands[0][0][:4] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    assert commands[0][1] == 30.0


def test_nvml_sampler_uses_persistent_driver_handle(monkeypatch):
    calls = {"init": 0, "shutdown": 0}
    fake = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        NVML_CLOCK_SM=1,
        nvmlInit=lambda: calls.__setitem__("init", calls["init"] + 1),
        nvmlShutdown=lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
        nvmlDeviceGetHandleByIndex=lambda _index: "gpu0",
        nvmlDeviceGetUtilizationRates=lambda _handle: SimpleNamespace(gpu=87),
        nvmlDeviceGetMemoryInfo=lambda _handle: SimpleNamespace(
            used=4 * 1024**3, total=8 * 1024**3
        ),
        nvmlDeviceGetDecoderUtilization=lambda _handle: (63, 1_000),
        nvmlDeviceGetEncoderUtilization=lambda _handle: (11, 1_000),
        nvmlDeviceGetTemperature=lambda _handle, _sensor: 72,
        nvmlDeviceGetPowerUsage=lambda _handle: 101_500,
        nvmlDeviceGetClockInfo=lambda _handle, _clock: 2_100,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = _NvmlSampler.create()
    assert sampler is not None
    sample = sampler.sample()
    sampler.close()
    assert sample["utilization.gpu"] == 87.0
    assert sample["utilization.decoder"] == 63.0
    assert sample["memory.used"] == 4096.0
    assert sample["power.draw"] == 101.5
    assert calls == {"init": 1, "shutdown": 1}


def test_resource_monitor_survives_one_sampling_failure(monkeypatch, tmp_path):
    monitor = ResourceMonitor(tmp_path / "resource_telemetry.json", 0.25)
    calls = {"gpu": 0}

    def flaky_gpu():
        calls["gpu"] += 1
        if calls["gpu"] == 1:
            raise RuntimeError("transient NVML failure")
        return {}

    monkeypatch.setattr(monitor, "_gpu", flaky_gpu)
    monkeypatch.setattr(monitor, "_smb_connections", lambda: [])
    monitor.start()
    time.sleep(0.8)
    report = monitor.stop()

    assert report["sample_count"] >= 1
    journal = [json.loads(line) for line in monitor.journal_destination.read_text().splitlines()]
    assert journal == report["samples"]
    assert report["monitor_health"]["sampling_error_count"] >= 1
    assert report["monitor_health"]["thread_ended_unexpectedly"] is False
    assert report["monitor_health"]["thread_alive_after_stop"] is False
    assert any(
        item["component"] == "sample_loop"
        for item in report["monitor_health"]["sampling_errors"]
    )


def test_nvml_field_failure_is_retained_in_monitor_health(monkeypatch, tmp_path):
    monitor = ResourceMonitor(tmp_path / "resource_telemetry.json", 0.25)

    class PartialSampler:
        def sample(self):
            return {"utilization.gpu": 1.0}

        def drain_errors(self):
            return [("memory", RuntimeError("memory query failed"))]

        def close(self):
            return None

    monitor._nvml = PartialSampler()
    assert monitor._gpu()["utilization.gpu"] == 1.0
    report = monitor.report()

    assert any(
        item["component"] == "nvml_memory"
        for item in report["monitor_health"]["sampling_errors"]
    )


def test_publisher_ledger_skips_rehashing_verified_file(monkeypatch, tmp_path):
    local = tmp_path / "local"
    nas = tmp_path / "nas"
    source = local / "JSON-Config-Files" / "result.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"passed":true}', encoding="utf-8")
    hash_calls = []
    original = storage._sha256_file

    def counted(path):
        hash_calls.append(path)
        return original(path)

    monkeypatch.setattr(storage, "_sha256_file", counted)
    publisher = IncrementalArchivePublisher(local, nas)
    publisher.publish_file(source)
    first_count = len(hash_calls)
    publisher.publish_file(source)
    assert first_count > 0
    assert len(hash_calls) == first_count

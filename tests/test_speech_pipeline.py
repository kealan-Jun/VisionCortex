from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from visioncortex import api, speech, speech_worker
from visioncortex.archive import ArchiveLayout
from visioncortex.config import load_manifest
from visioncortex.schemas import RunManifest, ViewInput
from visioncortex.upload_sessions import UploadSessionStore
from test_resumable_uploads import _payload, _reset_api_stores, _upload_all


def recorder(tmp_path):
    folder = tmp_path / "device_cam01" / "2026-08-27" / "090000"
    folder.mkdir(parents=True)
    common = dict(
        recording_session_id=123,
        sender_id="device",
        segment_window_start_global_us=1000000,
        segment_window_end_global_us=4000000,
    )
    refs = dict(
        task_audio_file="audio.opus",
        task_audio_meta_file="audio_meta.json",
        task_audio_ready_file="audio_ready.json",
        task_audio_timing_file="audio_timing.csv",
    )
    (folder / "audio.opus").write_bytes(b"recorder-audio")
    (folder / "audio_timing.csv").write_text("global_timestamp_us\n1000000\n")
    speech_worker.atomic_json(
        folder / "meta.json", {**common, **refs, "closed": True, "rgb_file": "rgb.mp4"}
    )
    speech_worker.atomic_json(
        folder / "recording_ready.json", {**common, **refs, "ready": True}
    )
    speech_worker.atomic_json(
        folder / "audio_meta.json",
        {**common, "audio_valid": True, "first_audio_global_us": 1020000},
    )
    speech_worker.atomic_json(
        folder / "audio_ready.json",
        {
            **common,
            "ready": True,
            "audio_valid": True,
            "quality_status": "complete",
            "files": {
                name: speech_worker.file_record(folder / name)
                for name in ("audio.opus", "audio_meta.json", "audio_timing.csv")
            },
        },
    )
    config = {
        "speech_recognition": {"enabled": True},
        "collection_ingest": {
            "enabled": True,
            "mode": "directory_metadata",
            "source_root": str(tmp_path),
        },
    }
    return folder, config


def test_recorder_pairing_and_fail_closed_identity(tmp_path):
    folder, config = recorder(tmp_path)
    relative = folder.relative_to(tmp_path).as_posix()
    item = speech.inspect_source(config, relative)
    assert item["available"]
    assert item["_sealed"]["start_global_us"] == 1020000
    speech_worker.verify_files(folder, item["_sealed"]["files"])
    (folder / "audio.opus").write_bytes(b"changed--audio")
    with pytest.raises(ValueError, match="冻结清单"):
        speech_worker.verify_files(folder, item["_sealed"]["files"])
    metadata = speech_worker.read_json(folder / "recording_ready.json")
    metadata["sender_id"] = "different-device"
    speech_worker.atomic_json(folder / "recording_ready.json", metadata)
    with pytest.raises(ValueError, match="会话或设备"):
        speech.inspect_source(config, relative)


def test_no_input_is_visible_and_sidecar_escape_rejected(tmp_path):
    folder, config = recorder(tmp_path)
    relative = folder.relative_to(tmp_path).as_posix()
    ready = speech_worker.read_json(folder / "audio_ready.json")
    ready.update(audio_valid=False, quality_status="no_input")
    speech_worker.atomic_json(folder / "audio_ready.json", ready)
    (folder / "audio.opus").unlink()
    assert speech.inspect_source(config, relative)["status"] == "no_input"
    metadata = speech_worker.read_json(folder / "meta.json")
    metadata["task_audio_file"] = "../elsewhere.opus"
    speech_worker.atomic_json(folder / "meta.json", metadata)
    with pytest.raises(ValueError, match="文件名"):
        speech.inspect_source(config, relative)


def test_disabled_discovery_never_touches_inherited_nas(monkeypatch):
    monkeypatch.setattr(
        Path, "resolve", lambda *_args, **_kwargs: pytest.fail("NAS resolution")
    )
    monkeypatch.setattr(
        Path, "is_dir", lambda *_args, **_kwargs: pytest.fail("NAS access")
    )
    config = {
        "speech_recognition": {"enabled": False},
        "collection_ingest": {"enabled": True, "source_root": "/unavailable/nas"},
    }
    assert speech.discover(config, None) == []


def test_local_audio_has_no_inferred_video_sync(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    view = ViewInput(view_id="fp", role="first_person", video=video, audio=audio)
    monkeypatch.setattr(speech, "probe_audio", lambda _: {"duration_seconds": 3})
    monkeypatch.setattr(
        speech, "_source_root", lambda _: pytest.fail("inherited NAS root accessed")
    )
    sources = speech.discover(
        {
            "speech_recognition": {"enabled": True},
            "collection_ingest": {"enabled": False, "source_root": "/NAS"},
        },
        SimpleNamespace(views=[view]),
    )
    assert sources[0]["alignment"] == "NOT_PROVEN"
    assert sources[0]["audio_offset_ms"] is None
    assert sources[0]["_sealed"]["start_global_us"] is None


@pytest.mark.parametrize("relative", [
    "VisionCortexExperimentArchive/experiment/Original-Experiment-Videos/first/video.mp4",
    "archive/experiment/video.mp4",
    "device_cam01/not-a-date/segment/video.mp4",
])
@pytest.mark.parametrize("embedded_audio", [False, True])
def test_uploaded_video_under_nas_root_is_not_a_recorder_folder(
    tmp_path, monkeypatch, relative, embedded_audio,
):
    video = tmp_path / relative
    video.parent.mkdir(parents=True)
    video.write_bytes(b"uploaded-video")
    config = {
        "speech_recognition": {"enabled": True},
        "collection_ingest": {
            "enabled": True, "mode": "directory_metadata",
            "source_root": str(tmp_path),
        },
    }
    monkeypatch.setattr(
        speech, "inspect_source", lambda *_: pytest.fail("upload treated as recorder")
    )
    monkeypatch.setattr(
        speech, "probe_audio",
        lambda _: {"duration_seconds": 3} if embedded_audio else None,
    )
    view = ViewInput(view_id="fp", role="first_person", video=video)
    sources = speech.discover(config, SimpleNamespace(views=[view]))
    assert sources[0]["available"] is embedded_audio
    assert sources[0]["status"] == ("complete" if embedded_audio else "no_audio")
    if embedded_audio:
        assert sources[0]["_sealed"]["audio_file"] == "video.mp4"


def test_subtitles_use_listening_chunk_timeline_and_escape_cues(tmp_path):
    request = {"source": {}, "model": {}, "start_seconds": 60, "end_seconds": 90}
    rows = [
        {
            "start_seconds": 62.2,
            "end_seconds": 63.4,
            "text": "<拍照>\n\n00:00:00 --> bad",
        }
    ]
    speech_worker.write_transcript(tmp_path, request, rows)
    subtitle = (tmp_path / "transcript.vtt").read_text()
    assert "00:00:02.200 --> 00:00:03.400" in subtitle
    assert "&lt;拍照&gt;" in subtitle
    assert "\n\n00:00:00" not in subtitle
    with pytest.raises(ValueError, match="时间戳"):
        speech_worker.write_transcript(
            tmp_path, request, [{**rows[0], "end_seconds": float("nan")}]
        )


def test_stage_archives_results_search_and_integrity(tmp_path, monkeypatch):
    layout = ArchiveLayout(tmp_path / "experiment")
    layout.create()
    manifest = SimpleNamespace(experiment_id="experiment", views=[SimpleNamespace(view_id="fp", segments=[], video=tmp_path / "video.mp4")])
    original = tmp_path / "original.wav"
    original.write_bytes(b"fixture-original-audio")
    source = {
        "id": "fp-0001",
        "view_id": "fp",
        "segment_ordinal": 0,
        "available": True,
        "duration_seconds": 5,
        "audio_offset_ms": 250,
        "alignment": "PARTIAL_EVIDENCE",
        "_sealed": {"folder": str(tmp_path), "resolved_folder": str(tmp_path.resolve()),
                    "audio_file": original.name, "files": {original.name: speech_worker.file_record(original)}},
    }
    monkeypatch.setattr(speech, "discover", lambda *_: [source])
    monkeypatch.setattr(speech, "runtime_request", lambda _: {"max_audio_seconds": 3})

    def fake_invoke(work, request, **kwargs):
        work.mkdir(parents=True, exist_ok=True)
        start = request["start_seconds"]
        rows = [
                {
                    "id": "segment-0001",
                    "start_seconds": start + 0.2,
                "end_seconds": start + 0.8,
                "text": "加入样品 拍照",
                "human_reviewed": False,
                "nearby_device_playback": [],
            }
        ]
        speech_worker.write_transcript(work, {**request, "model": {}}, rows)
        (work / "audio.m4a").write_bytes(b"deterministic-contract-only")
        speech_worker.atomic_json(work / "request.json", request)
        receipt = {"outcome": "transcribed", "actual_asr_invocation": "NOT_PROVEN"}
        speech_worker.atomic_json(work / "receipt.json", receipt)
        return receipt

    monkeypatch.setattr(speech, "invoke", fake_invoke)
    result = speech.run_stage(
        {"speech_recognition": {"enabled": True}},
        manifest,
        layout,
        {"fp": SimpleNamespace(segments=[])},
        {"fp": SimpleNamespace(to_global=lambda value: value + 1000)},
    )
    assert result["status"] == "completed"
    assert len(result["sources"][0]["chunks"]) == 2
    assert not list(layout.experiment_clips.iterdir()), (
        "audio must not be counted as video experiments"
    )
    page = speech.archive_result(layout.root, "拍照", limit=1)
    assert page["total"] == 2 and page["next_offset"] == 1
    assert page["segments"][0]["aligned_start_ms"] == 1450
    second = speech.archive_result(layout.root, "拍照", offset=1, limit=1)["segments"][
        0
    ]
    assert second["playback_start_seconds"] == pytest.approx(0.2)
    assert second["start_seconds"] == pytest.approx(3.2)
    assert result["physical_action_confirmation"] is False
    # Formal and in-progress endpoints stay scoped to this experiment.
    monkeypatch.setattr(api, "_resolve_archive", lambda _: layout.root)
    monkeypatch.setattr(api, "_resolve_staging_run", lambda _: layout.root)
    client = TestClient(api.app)
    response = client.get("/api/archives/experiment/speech?q=拍照&limit=1")
    assert response.status_code == 200
    assert response.json()["sources"][0]["chunks"][0]["files"]["audio.m4a"][
        "url"
    ].startswith("/api/archive-file?archive=experiment&")
    assert (
        client.get("/api/archives/experiment/speech?release=stale").status_code == 409
    )
    assert (
        client.get("/api/staging-runs/run-1/speech")
        .json()["sources"][0]["chunks"][0]["files"]["audio.m4a"]["url"]
        .startswith("/api/staging-file?run_id=run-1&")
    )
    path = (
        layout.root
        / result["sources"][0]["chunks"][0]["files"]["aligned-transcript.json"]["path"]
    )
    path.write_text("{}")
    assert client.get("/api/archives/experiment/speech").status_code == 409


def test_failed_speech_cannot_publish_completed_receipt(tmp_path, monkeypatch):
    layout = ArchiveLayout(tmp_path)
    layout.create()
    monkeypatch.setattr(
        speech, "discover", lambda *_: (_ for _ in ()).throw(ValueError("mismatch"))
    )
    with pytest.raises(ValueError):
        speech.run_stage(
            {"speech_recognition": {"enabled": True}},
            SimpleNamespace(experiment_id="test", views=[SimpleNamespace(view_id="fp", segments=[], video=tmp_path / "video.mp4")]),
            layout,
            {},
            {},
        )
    assert (
        speech_worker.read_json(layout.json_config / "speech.json")["status"]
        == "failed"
    )


def test_companion_upload_survives_restart_and_is_sealed(monkeypatch, tmp_path):
    _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    payload = _payload(b"first", b"third")
    payload["files"].append(
        dict(
            file_id="audio-0", kind="audio", file_index=0, name="recording.opus", size=5
        )
    )
    payload["view_specs"][0].update(audio_index=0, audio_offset_ms=1250)
    response = client.post("/api/upload-sessions", json=payload)
    assert response.status_code == 201, response.text
    session_id = response.json()["session_id"]
    _upload_all(
        client,
        session_id,
        {"video-0": b"first", "video-1": b"third", "audio-0": b"audio"},
    )
    api._upload_sessions = UploadSessionStore(api._upload_sessions.database)
    response = client.post(f"/api/upload-sessions/{session_id}/finalize")
    assert response.status_code == 202, response.text
    session = api._upload_sessions.get(session_id)
    root = Path(session["archive_root"])
    manifest = load_manifest(root / "JSON-Config-Files/input_manifest.yaml")
    assert manifest.views[0].audio.read_bytes() == b"audio"
    assert manifest.views[0].audio_offset_ms == 1250
    receipts = api._manifest_source_receipts(manifest)
    assert [row["kind"] for row in receipts].count("audio") == 1
    seal = json.loads(
        (root / "JSON-Config-Files/Input-Manifests/input_seal.json").read_text()
    )
    assert "recording.opus" in json.dumps(seal)
    assert (
        client.post(f"/api/upload-sessions/{session_id}/finalize").json()["run_id"]
        == response.json()["run_id"]
    )


@pytest.mark.parametrize("change", ["unused", "duplicate", "nonfinite", "missing"])
def test_upload_rejects_bad_audio_mappings(tmp_path, change):
    payload = _payload(b"f", b"t")
    payload["files"].append(
        dict(
            file_id="audio-0", kind="audio", file_index=0, name="recording.opus", size=5
        )
    )
    payload["view_specs"][0]["audio_index"] = 0
    if change == "unused":
        payload["view_specs"][0].pop("audio_index")
    if change == "duplicate":
        payload["view_specs"][1]["audio_index"] = 0
    if change == "nonfinite":
        payload["view_specs"][0]["audio_offset_ms"] = float("nan")
    if change == "missing":
        payload["view_specs"][0]["audio_index"] = 1
    with pytest.raises(api.HTTPException):
        api._parse_upload_session_files(payload, tmp_path / "exp", "session")


def test_audio_schema_upgrade_preserves_partial_upload_and_chunks(
    monkeypatch, tmp_path
):
    _reset_api_stores(monkeypatch, tmp_path)
    client = TestClient(api.app)
    session_id = client.post(
        "/api/upload-sessions", json=_payload(b"first", b"third")
    ).json()["session_id"]
    _upload_all(client, session_id, {"video-0": b"fi"})
    with sqlite3.connect(api._upload_sessions.database) as connection:
        dump = "\n".join(connection.iterdump()).replace(
            "'video', 'timestamp_csv', 'audio'", "'video', 'timestamp_csv'"
        )
    old_database = tmp_path / "legacy.sqlite"
    with sqlite3.connect(old_database) as connection:
        connection.executescript(dump)
    upgraded = UploadSessionStore(old_database)
    assert upgraded.get(session_id)["files"][0]["uploaded_bytes"] == 2
    with sqlite3.connect(old_database) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM upload_chunks").fetchone()[0] == 1
        )
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        assert (
            "'audio'"
            in connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='upload_files'"
            ).fetchone()[0]
        )
    UploadSessionStore(old_database)


def test_manifest_resolves_companion_audio_paths(tmp_path):
    payload = {
        "experiment_id": "test",
        "views": [
            {
                "view_id": "fp",
                "role": "first_person",
                "video": "first.mp4",
                "audio": "first.wav",
            },
            {
                "view_id": "tp",
                "role": "third_person",
                "segments": [{"video": "third.mp4", "audio": "third.wav"}],
            },
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload))
    manifest = load_manifest(path)
    assert manifest.views[0].audio == tmp_path / "first.wav"
    assert manifest.views[1].segments[0].audio == tmp_path / "third.wav"
    assert RunManifest.model_validate(manifest.model_dump()) == manifest


def test_speech_runtime_preserves_virtual_environment_executable(tmp_path):
    base = tmp_path / "python-base"
    base.write_bytes(b"interpreter")
    venv = tmp_path / "venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(base)
    config = {
        "speech_recognition": {
            "python_executable": str(venv),
            "model_directory": str(tmp_path),
        }
    }
    request = speech.runtime_request(config)
    assert request["python_executable"] == str(venv)
    assert request["python_executable"] != str(base)


def test_failed_speech_does_not_prevent_visual_scan_submission(tmp_path, monkeypatch):
    from visioncortex import pipeline as pipeline_module
    from visioncortex.config import load_config
    from visioncortex.schemas import AlignmentTransform, VideoInfo
    from test_pipeline_preflight import _NoopResourceMonitor

    views = []
    for name, role in [("fp", "first_person"), ("tp", "third_person")]:
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(b"video")
        views.append(ViewInput(view_id=name, role=role, video=path))
    manifest = RunManifest(experiment_id="speech-gate", views=views)
    config = load_config(Path("configs/development-local.yaml"))
    config["project"]["output_root"] = str(tmp_path / "output")
    config["storage"]["local_cache_root"] = str(tmp_path / "cache")
    config["performance"]["media_pipeline_preflight_enabled"] = False
    config["speech_recognition"]["enabled"] = True
    config["capture_quality"]["enabled"] = False  # This test isolates the speech-stage failure boundary.
    infos = {
        view.view_id: VideoInfo(
            path=view.video, duration_ms=1000, fps=1, width=16, height=16, frame_count=1
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(view_id=view.view_id, reference_view_id="fp")
        for view in views
    }
    monkeypatch.setattr(pipeline_module, "ResourceMonitor", _NoopResourceMonitor)
    monkeypatch.setattr(pipeline_module, "validate_models", lambda _: {})
    monkeypatch.setattr(
        pipeline_module,
        "video_encoder_preflight",
        lambda _: {"selected_encoder": "libx264"},
    )
    monkeypatch.setattr(pipeline_module, "probe_views", lambda *_args, **_kwargs: infos)
    monkeypatch.setattr(
        pipeline_module, "build_alignments", lambda *_args: (transforms, [])
    )
    monkeypatch.setattr(
        pipeline_module,
        "alignment_quality_report",
        lambda *_args: {"formal_evidence_ready": True},
    )
    monkeypatch.setattr(
        pipeline_module,
        "write_aligned_csv",
        lambda path, *_args: path.write_text("time\n"),
    )
    calls = []

    def reject_speech(
        received_config,
        received_manifest,
        layout,
        received_infos,
        received_transforms,
        progress,
        publisher=None,
    ):
        calls.append(received_manifest.experiment_id)
        assert received_config["speech_recognition"]["enabled"]
        assert received_infos == infos and received_transforms == transforms
        assert (layout.json_config / "time_alignment.json").is_file()
        progress("录音哈希验证失败")
        raise ValueError("audio hash mismatch")

    monkeypatch.setattr(speech, "run_stage", reject_speech)
    def visual_submitted(*args, **kwargs):
        raise RuntimeError("visual scan was submitted independently")
    monkeypatch.setattr(pipeline_module, "scan_videos", visual_submitted)
    # No real decoder/model: the scanner boundary proves independent admission.
    with pytest.raises(RuntimeError, match="visual scan was submitted independently"):
        pipeline_module.EvidencePipeline(config).run(manifest)
    assert calls == ["speech-gate"]
    status = json.loads(
        next((tmp_path / "output").rglob("run_status.json")).read_text()
    )
    assert status["stage"] == "failed" and status["failed_stage"] != "speech"
    assert "alignment" in status["completed_stages"]
    assert "speech" not in status["completed_stages"]


def test_formal_speech_index_is_bound_to_release_manifest(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    config_root = root / "JSON-Config-Files"
    config_root.mkdir(parents=True)
    index = config_root / "speech.json"
    speech_worker.atomic_json(index, {"status": "completed", "sources": []})
    record = speech_worker.file_record(index)
    speech_worker.atomic_json(
        root / "release.json",
        {
            "manifests": {
                "JSON-Config-Files": [
                    {
                        "path": "speech.json",
                        "size_bytes": record["size"],
                        "sha256": record["sha256"],
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(api, "_resolve_archive", lambda _: root)
    monkeypatch.setattr(
        api,
        "read_current_release_pointer",
        lambda _: {"release_id": "accepted", "release_manifest": "release.json"},
    )
    monkeypatch.setattr(
        api, "lightweight_release_integrity", lambda *_: "release_manifest_verified"
    )
    client = TestClient(api.app)
    assert client.get("/api/archives/exp/speech?release=accepted").status_code == 200
    speech_worker.atomic_json(index, {"status": "failed", "sources": []})
    assert client.get("/api/archives/exp/speech?release=accepted").status_code == 409


def test_complete_asr_cache_checks_runtime_request_and_every_artifact(tmp_path):
    files = {}
    for name in ('transcript.json', 'transcript.txt', 'transcript.srt', 'transcript.vtt', 'audio.m4a'):
        path = tmp_path / name
        path.write_bytes(b'unchanged')
        files[name] = speech_worker.file_record(path)
    receipt = {'status':'completed', 'request_sha256':'request', 'runtime_sha256':'runtime', 'artifacts':files}
    speech_worker.atomic_json(tmp_path / 'receipt.json', receipt)
    assert speech_worker.completed_cache(tmp_path, 'request', 'runtime') == receipt
    assert speech_worker.completed_cache(tmp_path, 'changed', 'runtime') is None
    assert speech_worker.completed_cache(tmp_path, 'request', 'changed') is None
    (tmp_path / 'audio.m4a').write_bytes(b'tampered')
    assert speech_worker.completed_cache(tmp_path, 'request', 'runtime') is None


@pytest.mark.parametrize("enabled", [True, False])
def test_missing_audio_skips_model_start_and_keeps_video_inputs(tmp_path, monkeypatch, enabled):
    from visioncortex.archive import ArchiveLayout
    from visioncortex.speech_semantics import SpeechContext
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"source-identity-only")
    manifest = RunManifest(experiment_id="no-audio", views=[ViewInput(view_id=name, role=role, video=video) for name, role in [("fp", "first_person"), ("tp", "third_person")]])
    config = {"speech_recognition": {"enabled": enabled}, "collection_ingest": {"enabled": False}}
    monkeypatch.setattr(speech, "probe_audio", lambda _: None)
    monkeypatch.setattr(speech, "runtime_request", lambda _: pytest.fail("No audio must not start an ASR runtime"))
    result = speech.run_stage(config, manifest, layout, {}, {})
    assert result["status"] == ("completed" if enabled else "disabled")
    assert not any(item["available"] for item in result["sources"])
    assert SpeechContext(layout.root, config).rows == []
    assert video.read_bytes() == b"source-identity-only"

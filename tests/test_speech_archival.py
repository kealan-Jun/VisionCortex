"""Archival/index contracts only; fixtures do not prove ASR or speaker accuracy."""
from types import SimpleNamespace

import pytest

from visioncortex import speech, speech_worker
from visioncortex.archive import ArchiveLayout
from visioncortex.report_references import export_reference_index
from visioncortex.schemas import ViewInput
from test_speech_pipeline import recorder
from test_speech_semantics import context_fixture


def test_nas_original_saved_without_starting_asr_and_published(tmp_path, monkeypatch):
    folder, config = recorder(tmp_path / "nas")
    config["speech_recognition"]["enabled"] = False
    (folder / "rgb.mp4").write_bytes(b"video-stays-in-place")
    view = ViewInput(view_id="fp", role="first_person", video=folder / "rgb.mp4")
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    calls = []
    publisher = SimpleNamespace(publish_directory=lambda p: calls.append(str(p)), publish_file=lambda p: calls.append(str(p)))
    monkeypatch.setattr(speech, "runtime_request", lambda _: pytest.fail("ASR initialized while disabled"))
    result = speech.run_stage(config, SimpleNamespace(experiment_id="exp", views=[view]), layout, {}, {}, publisher=publisher)
    original = result["sources"][0]["original"]
    assert result["status"] == "disabled" and result["archive_status"] == "saved"
    assert original["speaker_identity"]["status"] == "unknown"
    assert (layout.root / original["file"]["path"]).read_bytes() == (folder / "audio.opus").read_bytes()
    assert len(original["source_files"]) == 6
    assert "Key-Materials/Experiment-Audio" in calls
    assert str(layout.json_config / "speech.json") in calls
    assert not list(layout.root.rglob("*.mp4"))
    exported = export_reference_index(layout.root, [])
    assert exported["speech"]["sources"][0]["original"]["file"] == original["file"]
    assert exported["speech"]["utterances"] == []


def test_local_attachment_archived_before_asr_setup_failure(tmp_path, monkeypatch):
    video, audio = tmp_path / "clip.mp4", tmp_path / "voice.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio-source")
    view = ViewInput(view_id="fp", role="first_person", video=video, audio=audio)
    layout = ArchiveLayout(tmp_path / "archive")
    layout.create()
    monkeypatch.setattr(speech, "probe_audio", lambda _: {"duration_seconds": 3})
    monkeypatch.setattr(speech, "_source_root", lambda *_args, **_kwargs: pytest.fail("disabled NAS was touched"))
    monkeypatch.setattr(speech, "runtime_request", lambda _: (_ for _ in ()).throw(ValueError("ASR unavailable")))
    with pytest.raises(ValueError, match="ASR unavailable"):
        speech.run_stage({"speech_recognition": {"enabled": True}, "collection_ingest": {"enabled": False}},
                         SimpleNamespace(experiment_id="exp", views=[view]), layout, {}, {})
    saved = speech_worker.read_json(layout.json_config / "speech.json")
    assert saved["status"] == "failed"
    assert (layout.root / saved["sources"][0]["original"]["file"]["path"]).read_bytes() == b"audio-source"


def test_source_changes_are_rejected_and_existing_archive_is_not_replaced(tmp_path, monkeypatch):
    from visioncortex.speech_archive import preserve
    audio = tmp_path / "voice.opus"
    audio.write_bytes(b"original")
    source = {"_sealed": {"folder": str(tmp_path), "resolved_folder": str(tmp_path.resolve()),
                          "audio_file": audio.name, "files": {audio.name: speech_worker.file_record(audio)}}}
    root = tmp_path / "archive"
    result = preserve(root, source)
    target = root / result["file"]["path"]
    audio.write_bytes(b"modified")
    # The already preserved original remains addressable on an idempotent retry.
    assert preserve(root, source) == result
    assert target.read_bytes() == b"original"
    other = tmp_path / "other-archive"
    with pytest.raises(ValueError, match="不一致"):
        preserve(other, source)
    assert not list(other.rglob("*.partial"))


def test_speech_export_keeps_occurrence_identity_unknown_speaker_and_step_refs(tmp_path):
    context_fixture(tmp_path)
    groups = [{"group_id": "G1", "model_understanding": {"steps": [
        {"speech_segment_ids": ["chunk:one", "missing"]}]}}]
    result = export_reference_index(tmp_path, groups)["speech"]
    assert result["status"] == "references_incomplete"
    assert result["step_links"][0]["speech_reference_ids"] == ["chunk:one", "missing"]
    row = result["utterances"][0]
    assert row["reference_id"] == "chunk:one" and row["text"]
    assert row["speaker_identity"] == {"status": "unknown", "person_id": None, "name": None}
    assert row["audio"]["path"].endswith("audio.m4a")
    assert row["aligned_start_ms"] is not None
    assert result["physical_action_confirmation"] is False
    assert not (tmp_path / "JSON-Config-Files/evidence_index.sqlite").exists()

from pathlib import Path

import pytest

from visioncortex import speech_qwen as qwen, speech_worker


@pytest.mark.parametrize("duration", [0.0065, 163.4, 180, 180.0065, 900.0065])
def test_audio_windows_preserve_short_recordings_without_millisecond_tail_jobs(duration):
    from visioncortex.device_day_stt import audio_windows
    windows = audio_windows(duration, 180)
    assert windows[0][0] == 0 and windows[-1][1] == duration
    assert all(0 < end - start <= 180 for start, end in windows)
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:], strict=False))
    if duration > 1:
        assert all(end - start >= 1 for start, end in windows)


def event(text="你好。", begin=200, end=800, number=1, cumulative=None):
    return {"request_id": "provider-request", "output": {
        "text": text if cumulative is None else cumulative,
        "sentence": {"sentence_id": number, "sentence_end": True, "channel_id": 0,
                     "begin_time": begin, "end_time": end, "text": text,
                     "words": [{"begin_time": begin, "end_time": end, "text": text}]}},
        "usage": {"duration": 1}}


def test_absolute_source_times_and_chunk_local_subtitles(tmp_path):
    request = {"start_seconds": 120, "end_seconds": 123, "model": {"model": qwen.MODEL}, "source": {}}
    rows = qwen.parse_segments({"http_status": 200, "events": [event()]}, request)
    assert rows[0]["start_seconds"] == 120.2
    assert rows[0]["words"][0]["end_seconds"] == 120.8
    speech_worker.write_transcript(tmp_path, request, rows)
    assert "00:00:00.200 --> 00:00:00.800" in (tmp_path / "transcript.vtt").read_text()
    assert speech_worker.read_json(tmp_path / "transcript.json")["physical_action_confirmation"] is False


def test_cumulative_snapshots_do_not_duplicate_prior_text():
    old = event()
    new = event("你好。再见。", end=1800, number=2)
    new["output"]["sentence"]["words"] = [old["output"]["sentence"]["words"][0],
                                            {"begin_time": 1000, "end_time": 1800, "text": "再见。"}]
    rows = qwen.parse_segments({"http_status": 200, "events": [old, new]},
                              {"start_seconds": 0, "end_seconds": 2})
    assert [r["text"] for r in rows] == ["你好。", "再见。"]
    assert rows[1]["start_seconds"] == 1


def test_discrete_stream_preserves_every_sentence():
    events = [event(), event("再见。", begin=1000, end=1800, number=2, cumulative="你好。再见。")]
    rows = qwen.parse_segments({"http_status": 200, "events": events}, {"start_seconds": 0, "end_seconds": 2})
    assert len(rows) == 2


@pytest.mark.parametrize("kind", ["missing_text", "out_of_range", "unfinished", "unauthorized"])
def test_incomplete_or_failed_recognition_is_not_published(kind):
    response = {"http_status": 200, "events": [event()]}
    if kind == "missing_text":
        response["events"][0]["output"]["text"] += "漏掉的一句。"
    elif kind == "out_of_range":
        response["events"][0]["output"]["sentence"]["end_time"] = 5000
    elif kind == "unfinished":
        response["events"][0]["output"]["sentence"]["sentence_end"] = False
    else:
        response = {"http_status": 401, "events": [{"code": "InvalidApiKey"}]}
    with pytest.raises(ValueError):
        qwen.parse_segments(response, {"start_seconds": 0, "end_seconds": 2})


@pytest.mark.parametrize("status", [200, 400])
def test_explicit_provider_empty_result_is_preserved_without_invented_text(status):
    assert qwen.parse_segments({"http_status": status, "events": [{"message": qwen.NO_WORDS}]},
                               {"start_seconds": 0, "end_seconds": 2}) == []


def test_verified_receipt_reuse_does_not_claim_a_new_api_call(tmp_path, monkeypatch):
    original = tmp_path / "original.opus"
    original.write_bytes(b"test-only-audio")
    request = {"worker_sha256": speech_worker.sha256(Path(qwen.__file__)),
               "start_seconds": 0, "end_seconds": 2, "max_audio_seconds": 180,
               "model": {"model": qwen.MODEL}, "endpoint": "https://dashscope.aliyuncs.com/api/v1/test",
               "source": {"folder": str(tmp_path), "resolved_folder": str(tmp_path.resolve()),
                          "files": {original.name: speech_worker.file_record(original)}, "audio_file": original.name}}
    calls = []
    monkeypatch.setattr(qwen.subprocess, "run", lambda cmd, **kwargs: Path(cmd[-1]).write_bytes(b"derived-test-audio"))
    def recognize(*args):
        calls.append(True)
        return {"http_status": 200, "events": [event()], "wall_seconds": .1}
    monkeypatch.setattr(qwen, "recognize", recognize)
    output = tmp_path / "job"
    first = qwen.invoke(output, request)
    assert first["actual_asr_invocation"] == "PROVEN"
    second = qwen.invoke(output, request)
    assert second["execution"]["actual_asr_invocation"] == "NOT_PROVEN"
    assert second["execution"]["model_invocations"] == 0
    assert len(calls) == 1
    (output / "transcript.json").write_text("tampered")
    with pytest.raises(ValueError):
        qwen.invoke(output, request)
    assert len(calls) == 1


def test_provider_binding_is_not_replaced_by_asr_model(monkeypatch):
    from visioncortex import provider_credentials
    binding = {"provider": "aliyun", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
               "model": "already-verified-vision-model", "api_protocol": "chat_completions",
               "credential_ref": "a" * 32}
    observed = []
    monkeypatch.setattr(provider_credentials, "model_api_key", lambda c: observed.append(c) or "test-only-key")
    config = {"mllm": binding, "speech_recognition": {"model": qwen.MODEL,
              "adapter_sha256": speech_worker.sha256(Path(qwen.__file__)), "max_audio_seconds": 180}}
    runtime = qwen.runtime_request(config)
    assert observed[0]["model"] == binding["model"]
    assert runtime["model"]["model"] == qwen.MODEL
    assert "test-only-key" not in str(runtime)
    config["speech_recognition"]["model"] = "older-model"
    with pytest.raises(ValueError):
        qwen.runtime_request(config)


def test_archive_revision_reuses_exact_asr_request_without_mutating_history(tmp_path, monkeypatch):
    from visioncortex import speech
    from visioncortex.device_day_contract import DeviceDayLayout, artifact
    from visioncortex.device_day_stt import transcribe
    start = 1788949185000000
    layout = DeviceDayLayout(tmp_path / "archive", "fixture_cam01", start, tmp_path / "backend")
    original = layout.raw / "Audio" / "Fixture.opus"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"synthetic-audio-not-runtime-evidence")
    retention = {"recording": {"recording_id": "fixture", "recording_start_us": start},
                 "audio": {"status": "complete", "start_us": start},
                 "sources": [{"kind": "audio_audio", "retained": artifact(layout.root, original)}]}
    runtime = {"provider": "aliyun_qwen", "worker_sha256": speech_worker.sha256(Path(qwen.__file__)),
               "max_audio_seconds": 180, "model": {"model": qwen.MODEL},
               "endpoint": "https://dashscope.aliyuncs.com/api/v1/test"}
    monkeypatch.setattr(speech, "probe_audio", lambda path: {"duration_seconds": 2})
    monkeypatch.setattr(speech, "runtime_request", lambda config: runtime)
    monkeypatch.setattr(qwen.subprocess, "run", lambda cmd, **kwargs: Path(cmd[-1]).write_bytes(b"derived-test-audio"))
    calls = []
    def recognize(*args):
        calls.append(True)
        return {"http_status": 200, "events": [event()], "wall_seconds": .1}
    monkeypatch.setattr(qwen, "recognize", recognize)
    config = {"speech_recognition": {"enabled": True}}
    transcribe(config, layout, retention, "old")
    previous = layout.receipts / "fixture/speech/old/0000"
    history = {p.name: p.read_bytes() for p in previous.iterdir() if p.is_file()}
    result = transcribe(config, layout, retention, "new")
    assert len(calls) == 1 and len(result["comments"]) == 1
    assert result["chunks"][0]["current_execution"]["model_invocations"] == 0
    assert history == {p.name: p.read_bytes() for p in previous.iterdir() if p.is_file()}
    assert (layout.receipts / "fixture/speech/new/0000/audio.wav").is_file()

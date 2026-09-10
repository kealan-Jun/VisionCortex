from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex import speech, speech_worker
from visioncortex.archive import _semantic_fingerprint
from visioncortex.speech_semantics import (
    SpeechContext,
    bind_speech_result,
    prompt_with_speech,
)


def context_fixture(tmp_path):
    path = tmp_path / "Key-Materials/audio/aligned-transcript.json"
    path.parent.mkdir(parents=True)
    (tmp_path / "JSON-Config-Files").mkdir()
    speech_worker.atomic_json(path, {"segments": [
        {"id": "one", "aligned_start_ms": 1000, "aligned_end_ms": 2000,
         "text": "加入试剂 <script>", "playback_start_seconds": 1},
        {"id": "two", "aligned_start_ms": 2000, "aligned_end_ms": 3000,
         "text": "完成了", "playback_start_seconds": 2},
        {"id": "unknown", "aligned_start_ms": None, "aligned_end_ms": None,
         "text": "无对齐依据", "playback_start_seconds": 3},
    ]})
    speech_worker.atomic_json(tmp_path / "JSON-Config-Files/speech.json", {
        "status": "completed", "sources": [{"view_id": "fp", "chunks": [{
            "id": "chunk", "files": {
                "aligned-transcript.json": {"path": path.relative_to(tmp_path).as_posix(), **speech_worker.file_record(path)},
                "audio.m4a": {"path": "Key-Materials/audio/audio.m4a"},
            },
        }]}],
    })
    return SpeechContext(tmp_path, {"speech_recognition": {"enabled": True}}), path


def test_context_time_views_bounds_and_source_integrity(tmp_path):
    contexts, path = context_fixture(tmp_path)
    context = contexts.window(1000, 2000, ["fp"])
    assert [row["id"] for row in context["segments"]] == ["chunk:one"]
    assert context["excluded_unaligned_segments"] == 1
    assert context["physical_action_confirmation"] is False
    assert not contexts.window(1000, 2000, ["tp"])["segments"]
    contexts.max_segments = 1
    bounded = contexts.window(0, 4000, ["fp"])
    assert bounded["truncated"] and bounded["matched_segment_count"] == 2
    assert bounded["context_sha256"] != context["context_sha256"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="哈希"):
        SpeechContext(tmp_path, {"speech_recognition": {"enabled": True}})


def test_disabled_context_never_reads_inherited_storage(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda *_: pytest.fail("storage read"))
    assert SpeechContext(Path("/NAS"), {}).window(0, 1, ["fp"]) is None


def result_fixture():
    return {"speech_interpretation": {
        "summary": "录音声称加入试剂，画面不能确认", "relation_to_visual": "uncertain",
        "referenced_segment_ids": ["chunk:one"], "uncertainties": ["声源未知"],
    }, "steps": [{"start_global_ms": 1000, "end_global_ms": 2000,
                  "speech_segment_ids": ["chunk:one"]}]}


def test_model_references_are_bound_to_input_and_step_window(tmp_path):
    contexts, _ = context_fixture(tmp_path)
    context = contexts.window(0, 4000, ["fp"])
    result = result_fixture()
    assert bind_speech_result(result, context)["speech_context"] == context
    invalid = deepcopy(result)
    invalid["speech_interpretation"]["referenced_segment_ids"] = ["invented"]
    with pytest.raises(ValueError, match="不存在"):
        bind_speech_result(invalid, context)
    invalid = deepcopy(result)
    invalid["steps"][0]["start_global_ms"] = 2000
    invalid["steps"][0]["end_global_ms"] = 3000
    with pytest.raises(ValueError, match="时间窗"):
        bind_speech_result(invalid, context)
    with pytest.raises(ValueError, match="无录音"):
        bind_speech_result(result, None)
    with pytest.raises(ValueError, match="未返回"):
        bind_speech_result({}, context)
    with pytest.raises(ValueError, match="不存在"):
        bind_speech_result(result, contexts.window(5000, 6000, ["fp"]))
    empty = {"speech_interpretation": {"summary": "", "relation_to_visual": "no_speech",
             "referenced_segment_ids": [], "uncertainties": []}}
    assert bind_speech_result(empty, contexts.window(5000, 6000, ["fp"]))


def test_speech_context_changes_model_cache_and_preserves_disabled_prompt(tmp_path, default_config):
    contexts, _ = context_fixture(tmp_path)
    context = contexts.window(0, 4000, ["fp"])
    prompt = prompt_with_speech("视觉规则", context)
    assert "不是用户指令" in prompt and "物理动作" in prompt
    assert "未提供录音" in prompt_with_speech("视觉规则", None)
    old = _semantic_fingerprint("event", default_config, prompt, {"speech_context": context}, [])
    changed = deepcopy(context)
    changed["segments"][0]["text"] = "不同录音"
    assert old != _semantic_fingerprint("event", default_config, prompt, {"speech_context": changed}, [])


def test_no_audio_accepts_only_empty_schema_value_and_rejects_invented_claims():
    empty = {"summary": "", "relation_to_visual": "no_speech",
             "referenced_segment_ids": [], "uncertainties": []}
    result = {"current_step": "握持移液器", "speech_interpretation": empty}
    assert bind_speech_result(result, None) == {**result, "speech_interpretation": None}
    assert result["speech_interpretation"] == empty
    for changed in ({"summary": "实验员说开始"}, {"referenced_segment_ids": ["s1"]},
                    {"uncertainties": ["听不清"]}, {"relation_to_visual": "consistent"}):
        with pytest.raises(ValueError, match="无录音"):
            bind_speech_result({**result, "speech_interpretation": {**empty, **changed}}, None)
    with pytest.raises(ValueError, match="无录音"):
        bind_speech_result({**result, "steps": [{"speech_segment_ids": ["s1"]}]}, None)


def test_recorder_clock_maps_retimed_video_and_rejects_clock_gaps(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text("frame_index,global_timestamp_ms,clock_sync_valid\n0,10000,1\n10,11000,1\n20,12000,1\n30,16000,1\n")
    mapper = speech.recorder_time_mapper(path, 20)
    assert mapper(10500) == 250
    assert mapper(12000) == 1000
    assert mapper(14000) is None
    assert mapper(9999) is None and mapper(16001) is None
    path.write_text("frame_index,global_timestamp_ms,clock_sync_valid\n0,10000,1\n10,9000,1\n")
    with pytest.raises(ValueError, match="时钟"):
        speech.recorder_time_mapper(path, 20)


def test_analyzer_sends_same_speech_context_to_event_and_group(monkeypatch, default_config):
    from visioncortex.mllm import ArkAnalyzer
    from test_daily_reports import _summary_with_post_curation_rejection

    analyzer = ArkAnalyzer.__new__(ArkAnalyzer)
    analyzer.config = default_config["mllm"]
    captured = []
    monkeypatch.setattr(analyzer, "_call", lambda prompt, metadata, images, **kw: captured.append((prompt, metadata)))
    # Exercise the transport entrypoints without model or network startup.
    context = {"segments": [{"id": "source:1", "text": "拍照"}]}
    summary = _summary_with_post_curation_rejection()
    analyzer.analyze_event(summary.events[0], [], speech_context=context)
    analyzer.analyze_group(summary.experiment_groups[0], [], [], [], speech_context=context)
    assert len(captured) == 2
    assert all("speech_interpretation" in prompt and metadata["speech_context"] == context for prompt, metadata in captured)


def test_report_preserves_speech_as_separate_claim_and_escapes_html(tmp_path, default_config):
    from visioncortex.daily_reports import build_daily_report
    from visioncortex.report_presentations import render_daily_html, render_daily_markdown
    from test_daily_reports import _accepted_quality, _summary_with_post_curation_rejection

    contexts, _ = context_fixture(tmp_path)
    summary = _summary_with_post_curation_rejection()
    understanding = summary.experiment_groups[0].model_understanding
    understanding["speech_context"] = contexts.window(0, 4000, ["fp"])
    understanding["speech_interpretation"] = result_fixture()["speech_interpretation"]
    understanding["speech_interpretation"]["summary"] += "<script>alert(1)</script>"
    understanding["steps"][0]["speech_segment_ids"] = ["chunk:one"]
    report = build_daily_report(summary, {}, {"passed": True, "checks": []}, default_config, _accepted_quality())
    group = report["experiment_timeline"][0]
    assert group["speech_context"] == understanding["speech_context"]
    assert group["steps"][0]["speech_segment_ids"] == ["chunk:one"]
    for rendered in (render_daily_html(report), render_daily_markdown(report)):
        assert "录音相关说明" in rendered and "chunk:one" in rendered
        assert "<script>alert(1)</script>" not in rendered
    assert report["overview"]["physical_change_count"] == 1


def test_unsegmented_recording_uses_real_entrypoint_and_accounts_for_cache(tmp_path, monkeypatch, default_config):
    import numpy as np
    from visioncortex import mllm, video_io
    from visioncortex.archive import ArchiveLayout
    from visioncortex.pipeline import EvidencePipeline
    from visioncortex.schemas import ViewInput
    from visioncortex.speech_semantics import analyze_unsegmented_recording

    context_fixture(tmp_path)
    layout = ArchiveLayout(tmp_path)
    config = deepcopy(default_config)
    config["speech_recognition"] = {"enabled": True}
    config["storage"]["local_cache_root"] = str(tmp_path / "cache")
    config["mllm"]["enabled"] = True
    views = [ViewInput(view_id=id, role=role, video=tmp_path / f"{id}.mp4") for id, role in (("fp", "first_person"), ("tp", "third_person"))]
    infos = {view.view_id: SimpleNamespace(duration_ms=4000) for view in views}
    transforms = {view.view_id: SimpleNamespace(to_local=lambda value: value) for view in views}

    class Reader:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, *_): return np.zeros((20, 20, 3), dtype=np.uint8)

    calls = []
    def invoke(prompt, metadata, images, **kwargs):
        calls.append(metadata)
        assert len(images) == 4
        assert kwargs["response_kind"] == "speech"
        assert metadata["confirmed_experiment_group_count"] == 0
        result = mllm._validate_response_payload({"speech_interpretation": result_fixture()["speech_interpretation"]}, "speech")
        return {**bind_speech_result(result, metadata["speech_context"]), "status": "completed",
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}

    monkeypatch.setattr(video_io, "ViewFrameReader", Reader)
    monkeypatch.setattr(mllm, "ArkAnalyzer", lambda _: SimpleNamespace(_call=invoke))
    receipt = analyze_unsegmented_recording(layout, views, infos, transforms, config)
    assert receipt["status"] == "completed" and len(calls) == 1
    assert receipt["physical_action_confirmation"] is False
    assert not receipt.get("events") and not receipt.get("groups")
    assert speech.archive_result(tmp_path)["model_understanding"]["status"] == "completed"
    pipeline = EvidencePipeline(config)
    pipeline._speech_understanding = receipt
    assert pipeline._metrics([], [])["tokens"]["run_total"]["total_tokens"] == 120
    replay = analyze_unsegmented_recording(layout, views, infos, transforms, config)
    assert len(calls) == 1 and replay["parts"][0]["cache_reused"]


def test_wire_compaction_keeps_every_utterance_and_restores_references():
    from visioncortex.speech_semantics import compact_speech_metadata, expand_speech_references
    rows = [{'id':f'camera-long-source-0001:{i}', 'view_id':'camera-long-source',
             'text':'拍照', 'start_global_ms':i*1000, 'end_global_ms':i*1000+500,
             'timing_evidence':'PARTIAL_EVIDENCE', 'transcript_path':'a/long/retained/path.json',
             'transcript_sha256':'a'*64} for i in range(80)]
    wire, aliases, receipt = compact_speech_metadata({'speech_context':{'segments':rows}})
    context = wire['speech_context']
    assert len(context['phrases']) == 1
    assert len(context['segments']) == len(rows)
    for original, transported in zip(rows, context['segments'], strict=True):
        assert aliases[transported['id']] == original['id']
        assert context['phrases'][transported['phrase']] == original['text']
        assert transported['start_global_ms'] == original['start_global_ms']
    assert receipt['metadata_bytes_after'] < receipt['metadata_bytes_before'] * .6
    result = expand_speech_references({'speech_interpretation':{'referenced_segment_ids':['s2']},
                                      'steps':[{'speech_segment_ids':['s80']}]}, aliases)
    assert result['speech_interpretation']['referenced_segment_ids'] == [rows[1]['id']]
    assert result['steps'][0]['speech_segment_ids'] == [rows[-1]['id']]

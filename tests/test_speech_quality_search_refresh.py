"""Deterministic contracts, not real speech/model accuracy evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from visioncortex import api, capture_quality, mllm, speech, speech_search, speech_worker
from visioncortex import speech_refresh, speech_semantics, stage_refresh
from test_speech_semantics import context_fixture, result_fixture


def search_fixture(root):
    contexts, path = context_fixture(root)
    rows = speech_worker.read_json(path)["segments"]
    rows[0]["text"] = "加样"
    rows += [{**rows[0], "id": f"repeat-{i}", "playback_start_seconds": i+4,
              "nearby_device_playback": [1] if i == 0 else []} for i in range(3)]
    rows += [{**rows[0], "id": "english", "text": "Pipette"}]
    speech_worker.atomic_json(path, {"segments": rows})
    index = root/"JSON-Config-Files/speech.json"
    payload = speech_worker.read_json(index)
    payload["sources"][0]["chunks"][0]["files"]["aligned-transcript.json"].update(speech_worker.file_record(path))
    speech_worker.atomic_json(index, payload)
    speech_search.build(root)
    return path


def test_index_aliases_folding_all_occurrences_and_provenance(tmp_path):
    search_fixture(tmp_path)
    result = speech.archive_result(tmp_path, "移液", fold=True)
    assert result["total"] == 2 and result["occurrence_total"] == 5
    repeated = result["segments"][0]
    assert repeated["occurrence_count"] == 4
    page1 = speech.archive_result(tmp_path, phrase=repeated["phrase_id"], limit=2)
    page2 = speech.archive_result(tmp_path, phrase=repeated["phrase_id"], offset=page1["next_offset"], limit=2)
    assert len({row["reference_id"] for row in page1["segments"]+page2["segments"]}) == 4
    assert page2["next_offset"] is None
    assert all(not row["physical_action_confirmation"] for row in result["segments"])
    assert speech.archive_result(tmp_path, "移液", aliases=False)["total"] == 0
    assert speech.archive_result(tmp_path, hint="possible_device_playback")["total"] == 1
    assert speech.archive_result(tmp_path, "pipette", aliases=False)["segments"][0]["text"] == "Pipette"


def test_cached_index_does_not_parse_transcripts_again_but_rejects_tampering(tmp_path, monkeypatch):
    path = search_fixture(tmp_path)
    speech.archive_result(tmp_path)
    original = speech_worker.read_json
    monkeypatch.setattr(speech_worker, "read_json", lambda p, *args: pytest.fail("transcript reparsed") if p == path else original(p, *args))
    assert speech.archive_result(tmp_path, "加样")["total"] == 5
    path.write_text("{}")
    with pytest.raises(ValueError, match="完整性"):
        speech.archive_result(tmp_path)


def test_derived_index_tampering_rejected(tmp_path):
    search_fixture(tmp_path)
    path = tmp_path/speech_search.CONTROL
    payload = speech_worker.read_json(path)
    payload["rows"][0]["text"] = "invented"
    speech_worker.atomic_json(path, payload)
    with pytest.raises(ValueError, match="完整性"):
        speech.archive_result(tmp_path)


def test_evaluation_requires_human_reference_and_reports_subset_cer(tmp_path):
    search_fixture(tmp_path)
    reference = {"speech_sha256": speech_worker.sha256(tmp_path/"JSON-Config-Files/speech.json"),
                 "human_reviewed": True, "reviewer": "deterministic-test-human-attestation-only",
                 "segments": [{"reference_id": "chunk:one", "text": "加液"}]}
    result = speech_search.evaluate(tmp_path, reference)
    assert result["character_error_rate"] == .5
    assert result["evaluated_segments"] == 1 and result["total_segments"] == 7
    for changes in ({"human_reviewed": False}, {"speech_sha256": "x"}, {"reviewer": ""}):
        with pytest.raises(ValueError):
            speech_search.evaluate(tmp_path, {**reference, **changes})
    with pytest.raises(ValueError, match="重复"):
        speech_search.evaluate(tmp_path, {**reference, "segments": reference["segments"]*2})


def test_audio_samples_detect_silence_and_clipping_without_accuracy_claim():
    silence = capture_quality.audio_metrics(np.zeros(16000, dtype="<f4").tobytes())
    assert silence["silent_fraction"] == 1 and silence["longest_sampled_silence_seconds"] == 1
    clipped = capture_quality.audio_metrics(np.ones(16000, dtype="<f4").tobytes())
    assert clipped["clipped_sample_fraction"] == 1 and clipped["silent_fraction"] == 0
    with pytest.raises(ValueError):
        capture_quality.audio_metrics(b"")


def test_bounded_capture_diagnostics_show_sample_coverage_and_do_not_confirm_quality(tmp_path, monkeypatch):
    video = tmp_path/"video.mp4"
    video.write_bytes(b"fixture")
    view = SimpleNamespace(view_id="fp", segments=[], video=video, audio=None)
    manifest = SimpleNamespace(views=[view])
    infos = {"fp": SimpleNamespace(segments=[], duration_ms=900000)}
    calls = []
    def decode(path, start, seconds, audio):
        calls.append((start, seconds, audio))
        return np.zeros(int(seconds*16000), dtype="<f4").tobytes() if audio else bytes(160*90)
    monkeypatch.setattr(capture_quality, "_decode", decode)
    monkeypatch.setattr(speech, "probe_audio", lambda _: {"duration_seconds":900})
    result = capture_quality.inspect(manifest, {"capture_quality":{"enabled":True}}, infos)
    row = result["records"][0]
    assert len(calls) == 8 and row["sampled_audio_seconds"] == 12
    assert row["audio_sample_coverage"] == pytest.approx(12/900)
    assert row["dark_sample_fraction"] == 1 and len(row["warnings"]) == 2
    assert result["accuracy_evidence"] == "NOT_PROVEN" and not result["full_media_quality_proven"]
    monkeypatch.setattr(capture_quality, "_decode", lambda *_: pytest.fail("decoded while disabled"))
    assert capture_quality.inspect(manifest, {})["status"] == "disabled"


def group_fixture(root):
    context_fixture(root)
    group = {"group_id":"G1", "global_start_ms":0, "global_end_ms":4000, "participating_views":["fp"],
             "model_understanding":{"steps":[{"start_global_ms":1000,"end_global_ms":2000,"current_step":"retained visual action"},
                                                 {"start_global_ms":3000,"end_global_ms":4000,"current_step":"other action"}]}}
    speech_worker.atomic_json(root/speech_refresh.GROUPS, {"groups":[group, {**deepcopy(group),"group_id":"G2"}]})
    image = root/"sample.jpg"
    image.write_bytes(b"deterministic-retained-image")
    speech_refresh.input_path(root,"G1").parent.mkdir(parents=True,exist_ok=True)
    speech_worker.atomic_json(speech_refresh.input_path(root,"G1"), {"group_id":"G1", "start_global_ms":0,"end_global_ms":4000,
        "visual_samples":[{"path":"sample.jpg","label":"paired views",**speech_worker.file_record(image)}]})
    return group


def test_group_refresh_only_changes_selected_speech_refs_and_invalidates_report(tmp_path, monkeypatch, default_config):
    group = group_fixture(tmp_path)
    config = deepcopy(default_config)
    config["speech_recognition"]["enabled"]=True
    config["storage"]["local_cache_root"]=str(tmp_path/"cache")
    config["project"]["semantic_cache_mode"]="reuse"
    calls=[]
    def invoke(*args, **kwargs):
        calls.append(args[1])
        return {"status":"completed", "speech_interpretation":result_fixture()["speech_interpretation"]}
    monkeypatch.setattr(mllm,"ArkAnalyzer",lambda _:SimpleNamespace(_call=invoke,close=lambda:None))
    speech_worker.atomic_json(tmp_path/"JSON-Config-Files/quality_acceptance.json",{"passed":False})
    quality = speech_worker.sha256(tmp_path/"JSON-Config-Files/quality_acceptance.json")
    speech_worker.atomic_json(tmp_path/"JSON-Config-Files/daily_report_manifest.json",{"passed":True,"html":"old.html"})
    receipt = stage_refresh.refresh(tmp_path,"understanding",config,target="group:G1",revision=speech_worker.sha256(tmp_path/speech_refresh.GROUPS))
    assert len(calls)==1 and receipt["cv_invocations"]==receipt["asr_invocations"]==receipt["source_media_decodes"]==0
    groups = speech_worker.read_json(tmp_path/speech_refresh.GROUPS)["groups"]
    assert groups[0]==group
    updated = speech_refresh.apply(tmp_path,groups)
    assert updated[1]==groups[1]
    assert updated[0]["model_understanding"]["steps"][0]["speech_segment_ids"]==["chunk:one"]
    assert updated[0]["model_understanding"]["steps"][1]["speech_segment_ids"]==[]
    assert updated[0]["model_understanding"]["steps"][0]["current_step"]=="retained visual action"
    assert speech_worker.sha256(tmp_path/"JSON-Config-Files/quality_acceptance.json")==quality
    assert speech_worker.read_json(tmp_path/"JSON-Config-Files/daily_report_manifest.json")["status"]=="stale"
    assert "关联口述记录：1 条" in (tmp_path/"Partial-Results/Partial-Evidence-Report.html").read_text()
    exported = speech_worker.read_json(tmp_path/"Partial-Results/Analysis-Result.json")
    assert any(row["reference_id"] == "chunk:one" for row in exported["reference_index"]["speech"]["utterances"])
    replay=stage_refresh.refresh(tmp_path,"understanding",config,target="group:G1")
    assert len(calls)==1 and replay["model_invocations"]==0
    groups[0]["global_end_ms"]=5000
    speech_worker.atomic_json(tmp_path/speech_refresh.GROUPS,{"groups":groups})
    with pytest.raises(ValueError,match="过期"):
        speech_refresh.apply(tmp_path,groups)


def test_group_refresh_rejects_changed_images_before_model(tmp_path,monkeypatch,default_config):
    group_fixture(tmp_path)
    (tmp_path/"sample.jpg").write_bytes(b"tampered")
    monkeypatch.setattr(mllm,"ArkAnalyzer",lambda _:pytest.fail("model invoked"))
    with pytest.raises(ValueError,match="完整性"):
        speech_refresh.refresh_group(tmp_path,default_config,"group:G1")


def test_failed_model_preserves_group_and_existing_report(tmp_path, monkeypatch, default_config):
    group_fixture(tmp_path)
    config = deepcopy(default_config)
    config["speech_recognition"]["enabled"] = True
    config["storage"]["local_cache_root"] = str(tmp_path/"cache")
    monkeypatch.setattr(mllm, "ArkAnalyzer", lambda _: SimpleNamespace(_call=lambda *_args, **_kw: {"status":"failed"}, close=lambda:None))
    report = tmp_path/"JSON-Config-Files/daily_report_manifest.json"
    speech_worker.atomic_json(report, {"passed":True, "html":"original.html"})
    before = report.read_bytes()
    with pytest.raises(ValueError, match="原成果已保留"):
        speech_refresh.refresh_group(tmp_path, config, "group:G1")
    assert report.read_bytes() == before
    assert not (tmp_path/speech_refresh.CONTROL).exists()


def test_report_entrypoint_uses_group_speech_revision(tmp_path, monkeypatch, default_config):
    from visioncortex import daily_reports
    from test_daily_reports import _summary_with_post_curation_rejection
    contexts, _ = context_fixture(tmp_path)
    summary = _summary_with_post_curation_rejection()
    groups = [group.model_dump(mode="json") for group in summary.experiment_groups]
    speech_worker.atomic_json(tmp_path/"JSON-Config-Files/evidence_package.json", summary.model_dump(mode="json"))
    speech_worker.atomic_json(tmp_path/speech_refresh.GROUPS, {"groups":groups})
    speech_worker.atomic_json(tmp_path/"JSON-Config-Files/run_metrics.json", {})
    interpretation = result_fixture()["speech_interpretation"]
    speech_worker.atomic_json(tmp_path/speech_refresh.CONTROL, {"bindings":speech_refresh._base(tmp_path),
        "groups":{groups[0]["group_id"]:{"base_group_sha256":speech_refresh.digest(groups[0]),
            "result":{"speech_context":contexts.window(0,4000,["fp"]), "speech_interpretation":interpretation}}}})
    captured = []
    monkeypatch.setattr(daily_reports, "generate_daily_report_archive", lambda layout, model, *args: captured.append(model))
    daily_reports.generate_daily_report_from_archive(tmp_path, default_config)
    assert captured[0].experiment_groups[0].model_understanding["speech_interpretation"] == interpretation
    assert captured[0].events == summary.events


def test_recording_refresh_only_invokes_selected_part(tmp_path,monkeypatch,default_config):
    contexts,_=context_fixture(tmp_path)
    context=contexts.window(0,4000,["fp"])
    image=tmp_path/"sample.jpg"
    image.write_bytes(b"retained")
    part={"speech_context":context,"visual_samples":[{"label":"sample","path":"sample.jpg",**speech_worker.file_record(image)}],
          "status":"completed", "speech_interpretation":result_fixture()["speech_interpretation"]}
    path=tmp_path/"JSON-Config-Files/speech_understanding.json"
    speech_worker.atomic_json(path,{"status":"completed","parts":[part,deepcopy(part)],"index_sha256":contexts.identity})
    calls=[]
    def invoke(prompt,metadata,*args,**kwargs):
        calls.append(metadata)
        return {**speech_semantics.bind_speech_result({"speech_interpretation":result_fixture()["speech_interpretation"]},context),"status":"completed"}
    monkeypatch.setattr(mllm,"ArkAnalyzer",lambda _:SimpleNamespace(_call=invoke,close=lambda:None))
    config=deepcopy(default_config)
    config["speech_recognition"]["enabled"]=True
    config["storage"]["local_cache_root"]=str(tmp_path/"cache")
    result=speech_semantics.refresh_recording_understanding(tmp_path,config,"recording:1")
    assert len(calls)==1 and result["parts"][0]==part
    with pytest.raises(ValueError,match="不存在"):
        speech_semantics.refresh_recording_understanding(tmp_path,config,"recording:99")


def test_cross_experiment_search_returns_replay_links_and_surfaces_unavailable_sources(tmp_path,monkeypatch):
    left=tmp_path/"left"
    left.mkdir()
    search_fixture(left)
    right=tmp_path/"right"
    right.mkdir()
    search_fixture(right)
    monkeypatch.setattr(api,"_search_archive_roots",lambda _: [("one",left),("two",right)])
    monkeypatch.setattr(api,"_resolve_archive",lambda name: {"one":left,"two":right}[name])
    monkeypatch.setattr(api,"_runs",{})
    monkeypatch.setattr(api,"read_current_release_pointer",lambda _:None)
    client=TestClient(api.app)
    result=client.get("/api/speech-search?q=移液").json()
    assert len(result["segments"])==10
    assert {row["experiment"] for row in result["segments"]}=={"one","two"}
    assert all("/speech?chunk=chunk&t=" in row["href"] for row in result["segments"])
    (right/"Key-Materials/audio/aligned-transcript.json").write_text("{}")
    result=client.get("/api/speech-search?q=移液").json()
    assert len(result["segments"])==5 and result["unavailable"][0]["name"]=="two"


@pytest.mark.parametrize('required', [False, True])
def test_missing_optional_audio_is_not_a_capture_warning(tmp_path, monkeypatch, required):
    video = tmp_path/'video.mp4'
    video.write_bytes(b'fixture')
    view = SimpleNamespace(view_id='fp', segments=[], video=video, audio=None)
    monkeypatch.setattr(capture_quality, '_decode', lambda *_: bytes([127]) * (160*90))
    monkeypatch.setattr(speech, 'probe_audio', lambda _: None)
    result = capture_quality.inspect(SimpleNamespace(views=[view]),
        {'capture_quality':{'enabled':True},'speech_recognition':{'required':required}},
        {'fp':SimpleNamespace(segments=[], duration_ms=27000)})
    assert bool(result['records'][0]['warnings']) is required
    assert result['records'][0]['audio_status'] == 'no_audio'
    assert result['records'][0]['notes']

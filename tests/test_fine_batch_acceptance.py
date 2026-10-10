from __future__ import annotations

import json

import pytest

from visioncortex import fine_batch_acceptance as acceptance


SHA = "1" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


@pytest.fixture
def scans(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "_checkout_identity", lambda: (SHA, False))
    roots = [tmp_path / "baseline", tmp_path / "candidate"]
    for root, batch in zip(roots, (4, 16), strict=True):
        rows = []
        for role in sorted(acceptance.ROLES):
            directory = root / role
            ledger = directory / f"{role}.detections.jsonl"
            source = {"source": f"/unread-media/{role}.mp4", "source_stat": [10, 20],
                      "windows": [[0, 1000]], "role": role}
            frame = {"view_id": role, "role": role, "frame_index": 0, "local_ms": 0,
                     "global_ms": None, "detections": [], "source_frame": {
                         "source_path": source["source"], "source_size_bytes": 10, "source_mtime_ns": 20,
                         "status": "resolved", "packet_position": 1, "source_pts": 0, "time_base": "1/30",
                         "decoded_pixels_sha256": "2" * 64}}
            event = {"action_type": "pipetting", "global_start_ms": 0, "global_end_ms": 1000}
            write(ledger, frame)
            write(directory / "ActivityAudit.json", {"intervals": [[0, 1000]],
                  "selected_key_events": [event], "events": [event]})
            write(directory / "Candidates.json", [event])
            write(directory / f"runtime_fine_{role}.json", {
                "role": role, "phase": "fine", "backend": "TensorRT", "engine_build_batch": batch,
                "model_path": f"/unread-engines/{root.name}/{role}.engine",
                "final_effective_batch_size": batch, "actual_batch_size_max": batch,
                "oom_batch_contractions": []})
            rows.append({"view_id": role, "ledger": str(ledger), **source})
        write(root / "result.json", {"phase": "fine", "rows": rows, "wall_seconds": 10 / batch,
              "comparison_identity": {"code_sha": SHA, "config_sha256": "3" * 64,
                                      "manifest_sha256": "4" * 64, "git_dirty": False},
              "models": {**{role: {"sha256": "5" * 64} for role in acceptance.ROLES},
                         **{f"{role}_engine": {"path": f"/unread-engines/{root.name}/{role}.engine",
                                               "sha256": "6" * 64,
                                               **({"build_receipt": {"engine_sha256": "6" * 64, "weights_sha256": "5" * 64}}
                                                  if root.name == "candidate" else {})} for role in acceptance.ROLES}}})
    return roots


def test_matching_real_frame_receipts_prove_bounded_comparison_only(scans):
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "passed"
    assert result["evidence_status"] == "PARTIAL_EVIDENCE"
    assert result["promotion_ready"] is False
    assert result["role_batches"] == {role: 16 for role in acceptance.ROLES}
    assert result["bounded_scan_speedup"] == 4
    assert result["remaining_gates"]


@pytest.mark.parametrize("field", ["intervals", "selected_key_events", "events", "candidates"])
def test_action_changes_are_rejected_with_exact_differences(scans, field):
    path = scans[1] / "first_person" / ("Candidates.json" if field == "candidates" else "ActivityAudit.json")
    mutate(path, lambda data: data.clear() if field == "candidates" else data[field].clear())
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "rejected"
    assert any(d["gate"] == "same_actions" and d["field"] == field and d["removed"] for d in result["differences"])


@pytest.mark.parametrize("field,value", [("source", "/another.mp4"), ("source_stat", [11, 20]),
                                         ("windows", [[1, 1000]]), ("role", "third_person")])
def test_input_drift_is_rejected(scans, field, value):
    mutate(scans[1] / "result.json", lambda r: r["rows"][0].update({field: value}))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "rejected"
    assert any(d["gate"] == "same_input" for d in result["differences"])


@pytest.mark.parametrize("key", ["config_sha256", "manifest_sha256", "code_sha"])
def test_changed_comparison_identity_is_rejected(scans, key):
    mutate(scans[1] / "result.json", lambda r: r["comparison_identity"].update({key: "6" * len(r["comparison_identity"][key])}))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "rejected"


@pytest.mark.parametrize("relative", ["result.json", "first_person/ActivityAudit.json", "first_person/Candidates.json",
                                      "first_person/runtime_fine_first_person.json", "first_person/first_person.detections.jsonl"])
def test_missing_receipts_are_insufficient(scans, relative):
    (scans[1] / relative).unlink()
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "insufficient_evidence"
    assert result["missing_evidence"]


def test_legacy_receipts_cannot_pass_without_identity(scans):
    for root in scans:
        mutate(root / "result.json", lambda r: r.pop("comparison_identity"))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "insufficient_evidence"


@pytest.mark.parametrize("dirty,current_sha", [(True, SHA), (False, "9" * 40)])
def test_current_source_must_be_clean_and_match(scans, monkeypatch, dirty, current_sha):
    monkeypatch.setattr(acceptance, "_checkout_identity", lambda: (current_sha, dirty))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "insufficient_evidence"


@pytest.mark.parametrize("physical", [False, True])
def test_time_or_pixel_changes_reject_comparison(scans, physical):
    path = scans[1] / "first_person/first_person.detections.jsonl"
    mutate(path, lambda f: f["source_frame"].update(decoded_pixels_sha256="6" * 64) if physical else f.update(local_ms=1))
    result = acceptance.compare_fine_scans(*scans)
    assert any(d["gate"] == "same_sampled_frames" for d in result["differences"])


@pytest.mark.parametrize("change,gate", [({"actual_batch_size_max": 4}, "target_batch_not_observed"),
                                       ({"oom_batch_contractions": [{"from_batch_size": 16, "to_batch_size": 8}]}, "no_batch_contractions"),
                                       ({"engine_build_batch": 4}, "batch_capacity")])
def test_missing_target_batch_or_contraction_rejects(scans, change, gate):
    mutate(scans[1] / "first_person/runtime_fine_first_person.json", lambda r: r.update(change))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "rejected"
    assert any(d["gate"] == gate for d in result["differences"])


def test_shared_caller_batches_cannot_substitute_for_physical_batches(scans):
    path = scans[1] / "first_person/runtime_fine_first_person.json"
    mutate(path, lambda r: r.update(shared_inference={"last_engine_batch_sizes": [16]}))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "insufficient_evidence"
    mutate(path, lambda r: r["shared_inference"].update(engine_batch_size_max=16, oom_batch_contractions=[],
           scope="process_role_pool_cumulative_not_per_video", engine_batch_size_counts={"16": 2}, effective_batch_size=16))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "passed"
    mutate(path, lambda r: r["shared_inference"]["oom_batch_contractions"].append({"from_batch_size": 32, "to_batch_size": 16}))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "rejected"


def test_both_roles_are_required(scans):
    for root in scans:
        mutate(root / "result.json", lambda r: r["rows"].pop())
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "insufficient_evidence"
    assert any(g["gate"] == "both_view_roles" for g in result["missing_evidence"])


def test_comparator_never_reads_an_external_ledger(scans, tmp_path):
    outside = tmp_path / "external-ledger.jsonl"
    outside.write_text("never read this")
    mutate(scans[1] / "result.json", lambda r: r["rows"][0].update(ledger=str(outside)))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "insufficient_evidence"
    assert "escapes" in result["missing_evidence"][0]["reason"]


def test_runtime_must_belong_to_receipted_engine(scans):
    mutate(scans[1] / "first_person/runtime_fine_first_person.json", lambda r: r.update(model_path="/other.engine"))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "insufficient_evidence"
    assert any(g["gate"] == "runtime_engine_identity" for g in result["missing_evidence"])


def test_source_weight_drift_rejects(scans):
    mutate(scans[1] / "result.json", lambda r: r["models"]["first_person"].update(sha256="7" * 64))
    assert acceptance.compare_fine_scans(*scans)["comparison_status"] == "rejected"


def test_same_empty_actions_report_coverage_limit(scans):
    for root in scans:
        mutate(root / "first_person/ActivityAudit.json", lambda r: r["selected_key_events"].clear())
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "passed"
    assert "NOT_PROVEN" in result["views"]["first_person"]["quality_limit"]
    assert result["promotion_ready"] is False


@pytest.mark.parametrize("shared", [None, {}])
def test_shared_scope_requires_physical_receipt_even_when_caller_batch_is_sixteen(scans, shared):
    path = scans[1] / "first_person/runtime_fine_first_person.json"
    mutate(path, lambda r: r.update(inference_statistics_scope="caller_submissions_not_physical_gpu_batches",
                                    shared_inference=shared))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "insufficient_evidence"
    assert any(g["gate"] == "physical_batch_evidence" for g in result["missing_evidence"])


def test_candidate_requires_build_receipt_but_reviewed_baseline_does_not(scans):
    mutate(scans[1] / "result.json", lambda r: r["models"]["first_person_engine"].pop("build_receipt"))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "insufficient_evidence"
    assert any(g["gate"] == "candidate_build_identity" for g in result["missing_evidence"])


@pytest.mark.parametrize("field", ["engine_sha256", "weights_sha256"])
def test_candidate_build_must_bind_engine_to_configured_weights(scans, field):
    mutate(scans[1] / "result.json", lambda r: r["models"]["first_person_engine"]["build_receipt"].update({field: "7" * 64}))
    result = acceptance.compare_fine_scans(*scans)
    assert result["comparison_status"] == "rejected"
    assert any(g["gate"] == "candidate_build_identity" for g in result["differences"])


def test_failed_comparison_retains_raw_times_without_claiming_speedup(scans):
    mutate(scans[1] / "result.json", lambda r: r["comparison_identity"].update(config_sha256="7" * 64))
    result = acceptance.compare_fine_scans(*scans)
    assert result["scan_wall_seconds"] == {"baseline": 2.5, "candidate": .625}
    assert "bounded_scan_speedup" not in result
    assert "NOT_PROVEN" in result["speed_evidence"]

import json
from pathlib import Path

from visioncortex.collection_curation import (
    CONTENT_PROBE_DISTRIBUTION_STRATEGY,
    CONTENT_PROBE_SCHEMA_VERSION,
    build_curation_catalog,
    write_probe_source_readiness,
    write_curation_catalog,
)


def _config(tmp_path: Path) -> dict:
    return {
        "storage": {"local_cache_root": str(tmp_path / "cache")},
        "collection_curation": {
            "content_probe_max_seconds": 300,
            "content_probe_max_views": 2,
        },
    }


def _collection(experiment_id: str, name: str, duration: float = 60.0) -> dict:
    return {
        "experiment_id": experiment_id,
        "display_name": name,
        "status": "ready",
        "duration_seconds": duration,
        "camera_count": 5,
        "resolved_view_counts": {"first_person": 1, "third_person": 4},
        "processing": {"state": "not_processed"},
        "fingerprint_sha256": experiment_id,
    }


def test_probe_cap_never_truncates_long_production(tmp_path):
    payload = build_curation_catalog(
        {
            "collections": [
                _collection(
                    "exp-long",
                    "CustomFlow_standard_correct_12_ABCFA_0001",
                    11966.364,
                )
            ]
        },
        _config(tmp_path),
    )
    item = payload["experiments"][0]
    assert item["content_probe"]["maximum_timeline_seconds"] == 300
    assert item["content_probe"]["may_satisfy_production_completion"] is False
    assert item["production_run"]["full_timeline"] is True
    assert item["production_run"]["timeline_duration_seconds"] == 11966.364
    assert item["disposition"] == "eligible_full_timeline_production"


def test_hardware_test_excluded_and_ambiguous_requires_probe(tmp_path):
    payload = build_curation_catalog(
        {
            "collections": [
                _collection("hardware", "recording_matrix_20260720_01_1min_0001"),
                _collection("ambiguous", "test_Short_0053"),
            ]
        },
        _config(tmp_path),
    )
    assert payload["experiments"][0]["disposition"] == "excluded_test"
    assert (
        payload["experiments"][1]["disposition"]
        == "requires_bounded_content_probe"
    )


def test_accepted_formal_archive_is_authoritative_for_ambiguous_name(tmp_path):
    collection = _collection("archived", "test_Short_0044")
    collection["processing"] = {"state": "archived"}

    payload = build_curation_catalog(
        {"collections": [collection]},
        _config(tmp_path),
    )

    item = payload["experiments"][0]
    assert item["classification"] == "real_experiment"
    assert item["classification_source"] == "accepted_formal_archive"
    assert item["disposition"] == "already_formally_archived"


def test_source_readiness_blocks_before_ambiguous_probe(tmp_path):
    collection = _collection("attention", "undefined_0001")
    collection["status"] = "attention"

    payload = build_curation_catalog(
        {"collections": [collection]},
        _config(tmp_path),
    )

    assert payload["experiments"][0]["disposition"] == "blocked_source_readiness"


def test_probe_missing_source_receipt_overrides_stale_ready_status(tmp_path):
    config = _config(tmp_path)
    write_probe_source_readiness(
        config,
        "missing",
        status="blocked",
        error="NAS source missing: /nas/missing.mp4",
    )

    payload = build_curation_catalog(
        {"collections": [_collection("missing", "fz_0005")]},
        config,
    )

    item = payload["experiments"][0]
    assert item["collection_status"] == "attention"
    assert item["disposition"] == "blocked_source_readiness"
    assert item["blocking_issues"][-1]["code"] == "probe_source_file_missing"


def test_valid_probe_enables_full_timeline_and_overlong_fails_closed(tmp_path):
    config = _config(tmp_path)
    receipt_dir = (
        tmp_path / "cache" / "Collection-Catalog" / "Content-Probe-Receipts"
    )
    receipt_dir.mkdir(parents=True)
    base = {
        "schema_version": CONTENT_PROBE_SCHEMA_VERSION,
        "scope": "classification_only",
        "production_completion_eligible": False,
        "source_copy_bytes": 0,
        "distribution_strategy": CONTENT_PROBE_DISTRIBUTION_STRATEGY,
        "views": [
            {"view_id": "fp", "role": "first_person"},
            {"view_id": "tp", "role": "third_person"},
        ],
        "verdict": "real_experiment",
    }
    (receipt_dir / "good.json").write_text(
        json.dumps(
            {
                **base,
                "experiment_id": "good",
                "decoded_timeline_seconds": 300,
                "timeline_duration_seconds": 900,
                "sampled_intervals": [
                    {"start_seconds": 0, "end_seconds": 50, "selection_reason": "uniform_anchor"},
                    {"start_seconds": 200, "end_seconds": 250, "selection_reason": "uniform_anchor"},
                    {"start_seconds": 425, "end_seconds": 475, "selection_reason": "uniform_anchor"},
                    {"start_seconds": 650, "end_seconds": 700, "selection_reason": "uniform_anchor"},
                    {"start_seconds": 850, "end_seconds": 900, "selection_reason": "uniform_anchor"},
                    {"start_seconds": 500, "end_seconds": 550, "selection_reason": "motion_peak"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (receipt_dir / "overlong.json").write_text(
        json.dumps(
            {
                **base,
                "experiment_id": "overlong",
                "decoded_timeline_seconds": 300.001,
                "timeline_duration_seconds": 1105,
                "sampled_intervals": [
                    {"start_seconds": 0, "end_seconds": 300.001, "selection_reason": "uniform_anchor"}
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = build_curation_catalog(
        {
            "collections": [
                _collection("good", "fz_0001", 900),
                _collection("overlong", "test_long_0004", 1105),
            ]
        },
        config,
    )
    good, overlong = payload["experiments"]
    assert good["disposition"] == "eligible_full_timeline_production"
    assert good["production_run"]["timeline_duration_seconds"] == 900
    assert overlong["disposition"] == "requires_bounded_content_probe"
    assert "decoded_timeline_exceeds_probe_limit" in overlong["content_probe"][
        "receipt"
    ]["errors"]


def test_full_short_negative_probe_overrides_failed_empty_real_prefix(tmp_path):
    config = _config(tmp_path)
    receipt_dir = (
        tmp_path / "cache" / "Collection-Catalog" / "Content-Probe-Receipts"
    )
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "empty-real-prefix.json").write_text(
        json.dumps(
            {
                "schema_version": CONTENT_PROBE_SCHEMA_VERSION,
                "experiment_id": "empty-real-prefix",
                "scope": "classification_only",
                "production_completion_eligible": False,
                "source_copy_bytes": 0,
                "decoded_timeline_seconds": 7.8,
                "timeline_duration_seconds": 7.8,
                "distribution_strategy": CONTENT_PROBE_DISTRIBUTION_STRATEGY,
                "sampled_intervals": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 7.8,
                        "selection_reason": "uniform_anchor",
                    }
                ],
                "views": [
                    {"view_id": "fp", "role": "first_person"},
                    {"view_id": "tp", "role": "third_person"},
                ],
                "verdict": "recording_or_hardware_test",
            }
        ),
        encoding="utf-8",
    )
    collection = _collection(
        "empty-real-prefix", "TubeCleanDry_standard_correct_1_0001", 7.8
    )
    collection["processing"] = {"state": "failed"}

    payload = build_curation_catalog({"collections": [collection]}, config)

    item = payload["experiments"][0]
    assert item["metadata_classification"] == "real_experiment"
    assert item["classification"] == "recording_or_hardware_test"
    assert item["classification_source"] == "full_short_probe_after_empty_production"
    assert item["disposition"] == "excluded_test"


def test_long_probe_must_span_full_timeline_and_include_motion_peak(tmp_path):
    config = _config(tmp_path)
    receipt_dir = (
        tmp_path / "cache" / "Collection-Catalog" / "Content-Probe-Receipts"
    )
    receipt_dir.mkdir(parents=True)
    receipt = {
        "schema_version": CONTENT_PROBE_SCHEMA_VERSION,
        "experiment_id": "front-only",
        "scope": "classification_only",
        "production_completion_eligible": False,
        "source_copy_bytes": 0,
        "decoded_timeline_seconds": 300,
        "timeline_duration_seconds": 3600,
        "distribution_strategy": CONTENT_PROBE_DISTRIBUTION_STRATEGY,
        "sampled_intervals": [
            {"start_seconds": 0, "end_seconds": 300, "selection_reason": "uniform_anchor"}
        ],
        "views": [
            {"view_id": "fp", "role": "first_person"},
            {"view_id": "tp", "role": "third_person"},
        ],
        "verdict": "recording_or_hardware_test",
    }
    (receipt_dir / "front-only.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    payload = build_curation_catalog(
        {"collections": [_collection("front-only", "unknown_0001", 3600)]},
        config,
    )
    item = payload["experiments"][0]
    assert item["disposition"] == "requires_bounded_content_probe"
    assert "full_timeline_anchor_bands_missing" in item["content_probe"]["receipt"][
        "errors"
    ]
    assert "motion_peak_interval_required" in item["content_probe"]["receipt"][
        "errors"
    ]


def test_writes_json_and_csv(tmp_path):
    config = _config(tmp_path)
    payload = build_curation_catalog(
        {"collections": [_collection("exp-1", "Pipetting_standard_correct_13_0004")]},
        config,
    )
    paths = write_curation_catalog(config, payload)
    assert Path(paths["json"]).is_file()
    assert "production_full_timeline" in Path(paths["csv"]).read_text()

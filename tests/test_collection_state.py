import json

from labvision_evidence.collection_state import (
    collection_state_path,
    read_collection_states,
    record_collection_state,
)


def test_collection_processing_state_is_durable_and_bounded(tmp_path):
    config = {"storage": {"archive_root": str(tmp_path / "archive")}}

    record_collection_state(
        config,
        "exp-1",
        archive_name="Experiment-01",
        run_id="run-1",
        state="queued",
        details={"source_copy_bytes": 0},
    )
    record_collection_state(
        config,
        "exp-1",
        archive_name="Experiment-01",
        run_id="run-1",
        state="archived",
        details={"formal_archive": "archive/Experiment-01"},
    )

    payload = read_collection_states(config)
    state = payload["collections"]["exp-1"]
    assert payload["available"] is True
    assert state["state"] == "archived"
    assert state["archive_name"] == "Experiment-01"
    assert [item["state"] for item in state["history"]] == ["queued", "archived"]
    assert json.loads(collection_state_path(config).read_text(encoding="utf-8"))[
        "schema_version"
    ].endswith("/1")


def test_failed_retry_does_not_demote_still_valid_formal_archive(tmp_path):
    archive_root = tmp_path / "archive"
    formal = archive_root / "Experiment-01"
    (formal / "JSON-Config-Files").mkdir(parents=True)
    (formal / "Lab-Daily-Reports" / "2026-08-25").mkdir(parents=True)
    for path in (
        formal / "JSON-Config-Files" / "quality_acceptance.json",
        formal / "JSON-Config-Files" / "evidence_package_eval.json",
        formal
        / "Lab-Daily-Reports"
        / "2026-08-25"
        / "Daily-Report-Eval.json",
    ):
        path.write_text('{"passed": true}', encoding="utf-8")
    config = {"storage": {"archive_root": str(archive_root)}}

    record_collection_state(
        config,
        "exp-1",
        archive_name="Experiment-01",
        run_id="accepted-run",
        state="archived",
        details={"formal_archive": str(formal)},
    )
    record_collection_state(
        config,
        "exp-1",
        archive_name="Experiment-01",
        run_id="retry-run",
        state="failed",
        details={
            "formal_archive": str(formal),
            "formal_archive_preserved": True,
        },
    )

    effective = read_collection_states(config)["collections"]["exp-1"]
    raw = json.loads(
        collection_state_path(config).read_text(encoding="utf-8")
    )["collections"]["exp-1"]

    assert effective["state"] == "archived"
    assert effective["latest_run_state"] == "failed"
    assert effective["effective_state_reason"] == (
        "accepted_formal_archive_preserved_across_retry"
    )
    assert raw["state"] == "failed"

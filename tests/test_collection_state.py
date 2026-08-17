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

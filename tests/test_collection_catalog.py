import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from labvision_evidence.collection_catalog import (
    clear_collection_catalog_cache,
    discover_collections,
    get_collection,
)
from labvision_evidence.device_registry import (
    DEVICE_REGISTRY_SCHEMA_VERSION,
    load_device_registry,
    resolve_view_role,
)


def _write_registry(path: Path, *, approved_override: bool = False) -> Path:
    payload = {
        "schema_version": DEVICE_REGISTRY_SCHEMA_VERSION,
        "devices": {
            "fp": {"display_name": "Wearable FP", "expected_role": "first_person"},
            "tp": {"display_name": "Fixed TP", "expected_role": "third_person"},
        },
        "experiment_role_overrides": {},
    }
    if approved_override:
        payload["experiment_role_overrides"] = {
            "exp-1": {
                "fp": {
                    "role": "third_person",
                    "status": "approved",
                    "reason": "Approved validation recording role policy.",
                    "source": "test",
                }
            }
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_index(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    experiment_id: str,
    camera_key: str,
    camera_view: str,
    *,
    ended: bool = True,
    segment_count: int = 2,
) -> dict[str, str]:
    updated = datetime(2026, 8, 10, 10, tzinfo=timezone.utc)
    return {
        "experiment_id": experiment_id,
        "experiment_prefix": "Collection Test",
        "camera_key": camera_key,
        "camera_id": "cam01",
        "camera_view": camera_view,
        "recording_start_time": "2026-08-10T08:00:00+00:00",
        "recording_end_time": "2026-08-10T09:00:00+00:00" if ended else "",
        "segment_count": str(segment_count),
        "rgb_file": f"Z:/{camera_key}/001.mp4;Z:/{camera_key}/002.mp4",
        "frames_file": f"Z:/{camera_key}/001.csv;Z:/{camera_key}/002.csv",
        "updated_at": updated.isoformat(),
        "sync_error": "",
    }


def test_device_registry_conflict_requires_explicit_approved_override(tmp_path):
    registry = load_device_registry(_write_registry(tmp_path / "registry.json"))

    conflict = resolve_view_role(registry, "exp-1", "fp", "side")

    assert conflict["status"] == "blocking_conflict"
    assert conflict["index_role"] == "third_person"
    assert conflict["registry_expected_role"] == "first_person"
    assert "index_registry_role_mismatch" in conflict["blocking_reasons"]

    approved = load_device_registry(
        _write_registry(tmp_path / "registry-approved.json", approved_override=True)
    )
    receipt = resolve_view_role(approved, "exp-1", "fp", "first")

    assert receipt["status"] == "approved_override"
    assert receipt["resolved_role"] == "third_person"
    assert receipt["resolution_source"] == "approved_experiment_override"
    assert receipt["blocking_reasons"] == []


def test_collection_catalog_groups_index_rows_without_opening_media(default_config, tmp_path):
    registry_path = _write_registry(tmp_path / "registry.json")
    index_path = _write_index(
        tmp_path / "experiment_record_index.csv",
        [_row("exp-ready", "fp", "first"), _row("exp-ready", "tp", "side")],
    )
    default_config["storage"].update(
        {
            "index_csv": str(index_path),
            "device_registry_path": str(registry_path),
            "local_cache_root": str(tmp_path / "cache"),
            "archive_root": str(tmp_path / "archive"),
        }
    )
    default_config["collection_ingest"].update(
        {"settle_seconds": 120, "persist_snapshot": True}
    )
    clear_collection_catalog_cache()
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)

    first = discover_collections(default_config, now=now)
    second = discover_collections(default_config, now=now + timedelta(seconds=1))
    collection = get_collection(default_config, "exp-ready")

    assert first["index"]["cache_hit"] is False
    assert second["index"]["cache_hit"] is True
    assert first["monitoring_policy"]["recursive_nas_scan"] is False
    assert first["monitoring_policy"]["video_decode"] is False
    assert collection["status"] == "ready"
    assert collection["ready_to_analyze"] is True
    assert collection["camera_count"] == 2
    assert collection["video_segment_count"] == 4
    assert collection["clock_segment_count"] == 4
    assert collection["resolved_view_counts"] == {
        "first_person": 1,
        "third_person": 1,
    }
    assert collection["source_policy"]["video_files_opened"] == 0
    assert collection["source_policy"]["source_path_stats"] == 0
    assert collection["processing"]["state"] == "not_processed"
    assert not (tmp_path / "fp" / "001.mp4").exists()
    snapshot = (
        tmp_path
        / "cache"
        / "Collection-Catalog"
        / "experiment_record_index.snapshot.json"
    )
    assert snapshot.is_file()


def test_collection_catalog_keeps_open_recordings_out_of_ready_queue(
    default_config, tmp_path
):
    registry_path = _write_registry(tmp_path / "registry.json")
    index_path = _write_index(
        tmp_path / "experiment_record_index.csv",
        [_row("exp-open", "fp", "first"), _row("exp-open", "tp", "side", ended=False)],
    )
    default_config["storage"].update(
        {
            "index_csv": str(index_path),
            "device_registry_path": str(registry_path),
            "local_cache_root": str(tmp_path / "cache"),
        }
    )
    default_config["collection_ingest"]["persist_snapshot"] = False
    clear_collection_catalog_cache()

    collection = discover_collections(
        default_config, now=datetime(2026, 8, 11, tzinfo=timezone.utc)
    )["collections"][0]

    assert collection["status"] == "recording"
    assert collection["sealed"] is False
    assert collection["ready_to_analyze"] is False

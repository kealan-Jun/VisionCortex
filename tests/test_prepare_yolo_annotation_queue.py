from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_yolo_annotation_queue.py"
SPEC = importlib.util.spec_from_file_location("prepare_yolo_annotation_queue", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_annotation_queue_prioritizes_confirmed_and_small_targets(tmp_path: Path):
    source = tmp_path / "source.csv"
    rows = [
        {"event_id": "EVT-000578", "class_name": "bottle_cap", "issue_type": "prior_cv_expected_class_persistent_miss", "image_path": "a.jpg", "severity_score": "100"},
        {"event_id": "EVT-2", "class_name": "paper", "issue_type": "cross_view_class_support_gap", "image_path": "b.jpg", "severity_score": "20"},
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_csv = tmp_path / "queue.csv"
    output_json = tmp_path / "queue.json"

    summary = MODULE.build_queue(source, output_csv, output_json)

    assert summary["item_count"] == 2
    assert summary["items"][0]["current_disposition"] == "confirmed_false_negative"
    assert summary["priority_counts"] == {"P0": 1, "P1": 0, "P2": 1}
    assert output_csv.is_file() and output_json.is_file()

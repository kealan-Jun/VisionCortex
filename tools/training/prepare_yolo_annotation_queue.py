from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SMALL_TARGETS = {"bottle_cap", "tube_cap", "magnetic_stir_bar", "spatula"}
ONTOLOGY_CONFLICTS = {
    "beaker|container",
    "container|beaker",
    "sample_bottle|reagent_bottle",
    "reagent_bottle|sample_bottle",
    "hand|gloved_hand",
    "gloved_hand|hand",
    "tube|tube_cap",
    "tube_cap|tube",
}
CONFIRMED_FALSE_NEGATIVE = {("EVT-000578", "bottle_cap"), ("EVT-000579", "bottle_cap")}
KNOWN_LABEL_OR_THRESHOLD = {
    ("EVT-000154", "hand"),
    ("EVT-000175", "hand"),
    ("EVT-000304", "hand"),
    ("EVT-000349", "hand"),
}
KNOWN_NOT_PURE_MISS = {
    ("EVT-000183", "hand"),
    ("EVT-000256", "hand"),
    ("EVT-000367", "hand"),
}


def _task(issue_type: str) -> str:
    return {
        "prior_cv_expected_class_persistent_miss": "box_false_negative_review",
        "expected_class_below_production_threshold": "confidence_calibration_review",
        "overlapping_class_ambiguity": "competing_class_adjudication",
        "duplicate_overlapping_detection": "duplicate_instance_review",
        "cross_view_class_support_gap": "view_visibility_or_occlusion_review",
        "expected_class_temporal_flicker": "temporal_track_review",
    }.get(issue_type, "manual_visual_review")


def enrich(row: dict[str, str]) -> dict[str, Any]:
    event_class = (row.get("event_id", ""), row.get("class_name", ""))
    class_name = row.get("class_name", "")
    issue_type = row.get("issue_type", "")
    if event_class in CONFIRMED_FALSE_NEGATIVE:
        priority, disposition = "P0", "confirmed_false_negative"
    elif event_class in KNOWN_LABEL_OR_THRESHOLD:
        priority, disposition = "P0", "label_or_threshold_conflict"
    elif event_class in KNOWN_NOT_PURE_MISS:
        priority, disposition = "P1", "not_pure_detector_miss"
    elif class_name in SMALL_TARGETS or class_name in ONTOLOGY_CONFLICTS:
        priority, disposition = "P0", "requires_independent_box_gt"
    elif issue_type in {
        "overlapping_class_ambiguity",
        "duplicate_overlapping_detection",
    }:
        priority, disposition = "P1", "requires_independent_box_gt"
    else:
        priority, disposition = "P2", "review_visibility_before_box_label"
    output: dict[str, Any] = dict(row)
    output.update(
        {
            "priority": priority,
            "annotation_task": _task(issue_type),
            "current_disposition": disposition,
            "required_output": (
                "class_name, xyxy, visibility, occlusion, reviewer_1, reviewer_2, adjudication"
            ),
            "review_status": "pending" if disposition != "confirmed_false_negative" else "confirmed",
        }
    )
    return output


def build_queue(source_csv: Path, output_csv: Path, output_json: Path) -> dict[str, Any]:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [enrich(row) for row in csv.DictReader(handle) if row.get("image_path")]
    rows.sort(
        key=lambda item: (
            {"P0": 0, "P1": 1, "P2": 2}.get(str(item["priority"]), 9),
            -float(item.get("severity_score") or 0),
            str(item.get("event_id")),
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "schema_version": "visioncortex-yolo-annotation-queue/1.0.0",
        "source": str(source_csv),
        "item_count": len(rows),
        "priority_counts": {
            priority: sum(item["priority"] == priority for item in rows)
            for priority in ("P0", "P1", "P2")
        },
        "confirmed_false_negative_count": sum(
            item["current_disposition"] == "confirmed_false_negative" for item in rows
        ),
        "items": rows,
    }
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a reviewed YOLO annotation queue.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    summary = build_queue(args.source, args.output_csv, args.output_json)
    print(json.dumps({key: summary[key] for key in ("item_count", "priority_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

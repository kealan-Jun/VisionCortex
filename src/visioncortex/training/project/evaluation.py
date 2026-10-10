"""Evaluation responsibilities for explicitly authorized training."""
from __future__ import annotations

from ...yolo_evaluation import evaluate_yolo_predictions



def project_evaluation(
    predictions: list[dict],
    rows: list[dict],
    names: list[str],
    *,
    confidence: float = 0.25,
) -> dict:
    images, annotations = [], []
    for index, row in enumerate(rows):
        a, source = row["annotation"], row["source"]
        images.append(
            dict(
                image_id=row["id"],
                event_id=source["source_group"],
                role=a["role"],
                frame_index=index,
            )
        )
        for box in a["boxes"]:
            annotations.append(
                dict(
                    image_id=row["id"],
                    annotation_id=row["id"] + ":" + box["id"],
                    class_name=names[box["class_id"]],
                    xyxy=box["xyxy_px"],
                )
            )
    truth = dict(
        schema_version="visioncortex-yolo-project-annotations/1",
        dataset_id="frozen-project-cohort",
        images=images,
        annotations=annotations,
    )
    report = evaluate_yolo_predictions(
        predictions, truth, confidence_threshold=confidence
    )
    report.update(
        schema_version="visioncortex-yolo-project-evaluation/1",
        truth_status="project_annotations",
        independent_ground_truth=False,
        production_ready=False,
        metric_scope="internal_project_comparison_only",
    )
    report["insufficient_positive_classes"] = [
        name
        for name in names
        if report["per_class"].get(name, {}).get("gt_count", 0) < 20
    ]
    report["crop_images"] = [
        r["id"]
        for r in rows
        if r["source"].get("provenance", {}).get("source_kind") == "derived_crop"
        or r["id"].startswith("awcrop-")
    ]
    return report

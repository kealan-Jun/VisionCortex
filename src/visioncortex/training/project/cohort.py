"""Cohort responsibilities for explicitly authorized training."""
from __future__ import annotations

import json
from pathlib import Path

from .common import sha256, canonical_sha

def validate_export(
    root: Path, receipt_sha: str, role: str, *, supervision: str = "complete"
) -> tuple[dict, list[dict]]:
    if supervision not in {"complete", "outside_ignore"}:
        raise ValueError("Unknown supervision contract")
    if role not in {"first_person", "third_person"}:
        raise ValueError(
            "A detector experiment requires an explicit first/third person role"
        )
    if sha256(root / "receipt.json") != receipt_sha:
        raise ValueError("Dataset receipt SHA256 mismatch")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    augmented_partial = (
        supervision == "outside_ignore"
        and receipt.get("schema_version") == "annotation-workbench-partial-training-export/2"
    )
    if (
        receipt.get("schema_version") != (
            "annotation-workbench-yolo-export/1" if supervision == "complete"
            else ("annotation-workbench-partial-training-export/2" if augmented_partial
                  else "annotation-workbench-partial-training-export/1")
        )
        or receipt.get("truth_status") != "project_annotations"
        or receipt.get("independent_ground_truth") is not False
        or receipt.get("check", {}).get("export_ready") is not True
    ):
        raise ValueError("Expected a passing project-annotation export receipt")
    if supervision == "outside_ignore" and (
        receipt.get("supervision") != dict(
            mode="outside_ignore", complete_annotations=False,
            required_trainer_contract="ignore_negative_classification_at_anchor_centers/1",
            known_positive_terms="retained", validation_and_test="complete_annotations_only",
            augmentation="offline_affine_ignore_regions/1" if augmented_partial else "disabled",
            production_ready=False,
        )
        or receipt["check"].get("supervision") != "outside_ignore"
        or ("augmentation" in receipt) != augmented_partial
    ):
        raise ValueError("Partial-supervision receipt requires the exact ignore contract")
    for relative, expected in receipt["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or sha256(root / path) != expected:
            raise ValueError(f"Dataset file identity mismatch: {relative}")
    for part in ["images", "labels"]:
        expected = {
            name for name in receipt["files"] if name.startswith(f"{role}/{part}/")
        }
        actual = {
            str(path.relative_to(root))
            for path in (root / role / part).rglob("*")
            if path.is_file() and (part == "images" or path.suffix == ".txt")
        }
        if not expected or actual != expected:
            raise ValueError(f"Unregistered or missing {part} in training directory")
    snapshot = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    indexed = {r["id"]: r for r in snapshot["images"]}
    rows = []
    expected_ignores = {}
    seen = set()
    groups: dict[str, set[str]] = {}
    for pin in receipt["records"]:
        iid = pin["image_id"]
        if iid in seen:
            raise ValueError("Duplicate exported image identity")
        seen.add(iid)
        row = indexed[iid]
        a, source = row["annotation"], row["source"]
        complete = (
            a["status"] == "reviewed"
            and a["completeness"] == "all_visible_instances"
            and not a["ignore_regions"]
        )
        partial_train = (
            supervision == "outside_ignore" and a["split"] == "train"
            and a["status"] == "reviewed_partial"
            and a["completeness"] == "all_visible_outside_ignore"
            and bool(a["ignore_regions"])
            and bool(a.get("review_notes", "").strip())
        )
        if (
            pin["revision"] != row["revision"]
            or pin["annotation_sha256"] != canonical_sha(a)
            or pin["source_sha256"] != source["sha256"]
            or sha256(Path(source["path"])) != source["sha256"]
            or not (complete or partial_train)
            or a["truth_status"] != "project_annotations"
            or a["independent_ground_truth"] is not False
            or pin["role"] != a["role"]
            or pin["split"] != a["split"]
            or a["split"] not in {"train", "val", "test"}
        ):
            raise ValueError(
                f"Unreviewed, changed or incompatible project record: {iid}"
            )
        if supervision == "outside_ignore":
            relative = f"{a['role']}/images/{a['split']}/{iid}{Path(source['path']).suffix.lower()}"
            if receipt["files"].get(relative) != source["sha256"]:
                raise ValueError("Missing pinned image for partial-supervision metadata")
            expected_ignores[relative] = dict(
                image_id=iid, source_size=[source["width"], source["height"]],
                annotation_sha256=canonical_sha(a),
                xyxy_px=[r["xyxy_px"] for r in a["ignore_regions"]],
            )
        groups.setdefault(source["source_group"], set()).add(a["split"])
        if a["role"] == role:
            rows.append(row)
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("Source group crosses partitions")
    if supervision == "outside_ignore":
        if augmented_partial:
            from ...project_augmentation import validate_partial_augmentation

            expected_ignores.update(validate_partial_augmentation(
                root, receipt, {iid: indexed[iid] for iid in seen}, canonical_sha,
            ))
        if {p for p in receipt["files"] if "/images/" in p} != set(expected_ignores):
            raise ValueError("Partial training images differ from registered originals and derivatives")
        if "ignore-regions.json" not in receipt["files"] or json.loads(
            (root / "ignore-regions.json").read_text(encoding="utf-8")
        ) != expected_ignores:
            raise ValueError("Ignore metadata differs from frozen reviewed annotations")
    if not {"train", "val"}.issubset({r["annotation"]["split"] for r in rows}):
        raise ValueError("Role needs both train and validation images")
    return receipt, rows



def training_sampling_sources(root: Path, role: str, receipt: dict, rows: list[dict]) -> dict:
    training = {r["id"]: r for r in rows if r["annotation"]["split"] == "train"}
    # This function consumes the receipt only after validate_export has verified v2.
    derived = {
        d["id"]: d for d in receipt.get("augmentation", {}).get("records", [])
        if receipt.get("schema_version") == "annotation-workbench-partial-training-export/2"
        and d["role"] == role and d["split"] == "train"
    }
    sources = {}
    for relative in receipt["files"]:
        if not relative.startswith(f"{role}/images/train/"):
            continue
        row = training.get(Path(relative).stem)
        if row is None:
            d = derived.get(Path(relative).stem)
            if d is None or relative != d["image_file"] or d["source_image_id"] not in training:
                raise ValueError("Sampling cannot include unregistered or derived training images")
            sources[str((root / relative).absolute())] = dict(
                image_id=d["id"], source_group=d["source_group"], revision=d["source_revision"],
                source_sha256=d["image_sha256"], annotation_sha256=d["derived_annotation_sha256"],
                parent_image_id=d["source_image_id"], parent_source_sha256=d["source_sha256"],
                counts_as_new_source=False,
            )
            continue
        sources[str((root / relative).absolute())] = dict(
            image_id=row["id"], source_group=row["source"]["source_group"],
            revision=row["revision"], source_sha256=row["source"]["sha256"],
            annotation_sha256=canonical_sha(row["annotation"]),
        )
    if (len(sources) != len(training) + len(derived)
            or {r["image_id"] for r in sources.values()} != set(training) | set(derived)):
        raise ValueError("Sampling membership differs from the frozen training images")
    return sources

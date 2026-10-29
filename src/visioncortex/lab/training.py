"""Customer dataset checks and adapters to the existing training/evaluation owners."""
from __future__ import annotations

import json
from pathlib import Path

from .components import enable_components
from .media import hash_file, safe_path
from .rules import Rules


def validate_dataset(manifest: Path, image_root: Path, *, customer_code: str, max_bytes=2 * 1024**3):
    if str(image_root.absolute()).startswith(("/mnt/", "/media/")):
        raise ValueError("Training entry uses a bounded local working subset; prepare a verified subset first")
    data = json.loads(manifest.read_text())
    if data.get("customer_code") != customer_code or data.get("truth_status") != "reviewed_ground_truth" or not data.get("reviewed_by"):
        raise ValueError("Explicit customer ownership and reviewed human annotations are required")
    if not data.get("schema_version", "").startswith("visioncortex-yolo-"):
        raise ValueError("Use the existing VisionCortex reviewed-annotation schema")
    if not data.get("classes") or len(set(data["classes"])) != len(data["classes"]):
        raise ValueError("Explicit unique customer classes are required")
    groups, hashes, ids, counts = {}, {}, set(), {"train": 0, "val": 0, "test": 0}
    size = 0
    for item in data.get("images", []):
        image_id, split, group = item["image_id"], item["split"], item.get("capture_group_id")
        if image_id in ids or split not in counts or not group or not item.get("sha256"):
            raise ValueError("Each image needs a unique identity, reviewed split, capture group and SHA256")
        ids.add(image_id)
        path = safe_path(image_root, item.get("image_path") or item["file_name"])
        digest, bytes_ = hash_file(path, max_bytes)
        if digest != item["sha256"]:
            raise ValueError("Customer dataset image fingerprint mismatch")
        for mapping, key in ((groups, group), (hashes, digest)):
            if key in mapping and mapping[key] != split:
                raise ValueError("Capture groups or duplicate image bytes leak across dataset splits")
            mapping[key] = split
        size += bytes_
        if size > max_bytes:
            raise ValueError("Customer training working subset exceeds its SSD bound")
        counts[split] += 1
    if not counts["train"] or not counts["val"]:
        raise ValueError("Independent non-empty training and validation groups are required")
    for item in data.get("annotations", []):
        if item.get("image_id") not in ids or item.get("review_status") not in {"reviewed", "double_checked", "adjudicated", "confirmed"}:
            raise ValueError("Annotations need explicit human review and a valid image identity")
    return dict(customer_code=customer_code, image_count=len(ids), bytes=size, split_counts=counts, capture_groups=len(groups), processing_rules=Rules().identity())


def prepare_dataset(manifest, image_root, output, *, customer_code):
    validation = validate_dataset(manifest, image_root, customer_code=customer_code)
    enable_components()
    from visioncortex_lab_legacy.training.reviewed_dataset import build_yolo_training_dataset
    result = build_yolo_training_dataset(manifest, image_root, output)
    return dict(validation=validation, existing_dataset_receipt=result)


def evaluate(ground_truth, predictions, output):
    enable_components()
    from visioncortex_lab_legacy.yolo_evaluation import evaluate_files
    return evaluate_files(predictions, ground_truth, output)

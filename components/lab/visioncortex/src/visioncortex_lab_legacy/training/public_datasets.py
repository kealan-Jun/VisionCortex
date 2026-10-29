"""Public datasets responsibilities for explicitly authorized training."""
from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

from .dataset_common import _sha256, _safe_name, _dataset_yaml_classes, PUBLIC_DATASET_SCHEMA, PUBLIC_UNION_SCHEMA, _IMAGE_SUFFIXES

def _find_single_directory(root: Path, name: str) -> Path:
    matches = sorted(
        item
        for item in root.rglob("*")
        if item.is_dir() and item.name.casefold() == name.casefold()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one public YOLO {name} directory under {root}, got {len(matches)}"
        )
    return matches[0]



def _find_child_directory(root: Path, name: str) -> Path:
    matches = [
        item
        for item in root.iterdir()
        if item.is_dir() and item.name.casefold() == name.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {name} directory under public split {root}"
        )
    return matches[0]



def _public_yolo_classes(source_root: Path) -> tuple[Path, list[str]]:
    candidates = sorted(
        item
        for item in source_root.rglob("*")
        if item.is_file()
        and item.name.casefold() in {"classes.names", "classes.txt"}
    )
    if len(candidates) == 1:
        authority = candidates[0]
        classes = [
            line.strip()
            for line in authority.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif not candidates:
        yaml_candidates = sorted(
            item
            for item in source_root.rglob("*")
            if item.is_file() and item.name.casefold() in {"data.yaml", "dataset.yaml"}
        )
        if len(yaml_candidates) != 1:
            raise RuntimeError(
                "Expected one public YOLO class file or data.yaml"
            )
        authority = yaml_candidates[0]
        payload = yaml.safe_load(authority.read_text(encoding="utf-8-sig")) or {}
        names = payload.get("names")
        if isinstance(names, list):
            classes = [str(item).strip() for item in names]
        elif isinstance(names, dict):
            indexed = {int(key): str(value).strip() for key, value in names.items()}
            if sorted(indexed) != list(range(len(indexed))):
                raise RuntimeError("Public YOLO class ids must be contiguous")
            classes = [indexed[index] for index in range(len(indexed))]
        else:
            raise RuntimeError("Public YOLO data.yaml does not declare names")
    else:
        raise RuntimeError(
            "Expected one Classes.names or classes.txt in public YOLO dataset"
        )
    if not classes or len(classes) != len(set(classes)):
        raise RuntimeError("Public YOLO class list is empty or contains duplicates")
    return authority, classes



def _validate_public_yolo_split(
    split_root: Path,
    classes: list[str],
) -> dict[str, Any]:
    images_root = _find_child_directory(split_root, "Images")
    labels_root = _find_child_directory(split_root, "Labels")
    images = sorted(
        item
        for item in images_root.rglob("*")
        if item.is_file() and item.suffix.casefold() in _IMAGE_SUFFIXES
    )
    labels = sorted(item for item in labels_root.rglob("*.txt") if item.is_file())
    images_by_stem = {
        item.relative_to(images_root).with_suffix(""): item for item in images
    }
    labels_by_stem = {
        item.relative_to(labels_root).with_suffix(""): item for item in labels
    }
    if not images or images_by_stem.keys() != labels_by_stem.keys():
        raise RuntimeError(
            f"Public YOLO image/label mismatch in {split_root}: "
            f"images={len(images_by_stem)} labels={len(labels_by_stem)}"
        )
    annotation_count = 0
    box_annotation_count = 0
    polygon_annotation_count = 0
    class_counts = {name: 0 for name in classes}
    normalized_labels: dict[Path, str] = {}
    for label in labels:
        normalized_rows = []
        for line_number, row in enumerate(
            label.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            values = row.split()
            if len(values) != 5 and (
                len(values) < 7 or (len(values) - 1) % 2 != 0
            ):
                raise RuntimeError(
                    f"Invalid public YOLO row: {label}:{line_number}"
                )
            try:
                class_id = int(values[0])
                coordinates = [float(item) for item in values[1:]]
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid public YOLO value: {label}:{line_number}"
                ) from exc
            if class_id < 0 or class_id >= len(classes):
                raise RuntimeError(
                    f"Public YOLO class id is out of range: {label}:{line_number}"
                )
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in coordinates
            ):
                raise RuntimeError(
                    f"Public YOLO box is invalid: {label}:{line_number}"
                )
            if len(values) == 5:
                x_center, y_center, width, height = coordinates
                box_annotation_count += 1
            else:
                xs, ys = coordinates[0::2], coordinates[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                width, height = x_max - x_min, y_max - y_min
                x_center, y_center = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                polygon_annotation_count += 1
            if width <= 0.0 or height <= 0.0:
                raise RuntimeError(
                    f"Public YOLO box is invalid: {label}:{line_number}"
                )
            normalized_rows.append(
                " ".join(
                    (
                        str(class_id),
                        f"{x_center:.8f}",
                        f"{y_center:.8f}",
                        f"{width:.8f}",
                        f"{height:.8f}",
                    )
                )
            )
            annotation_count += 1
            class_counts[classes[class_id]] += 1
        normalized_labels[label.relative_to(labels_root)] = (
            "\n".join(normalized_rows) + ("\n" if normalized_rows else "")
        )
    return {
        "images_root": images_root,
        "labels_root": labels_root,
        "image_count": len(images),
        "annotation_count": annotation_count,
        "box_annotation_count": box_annotation_count,
        "polygon_annotation_count": polygon_annotation_count,
        "class_annotation_counts": class_counts,
        "file_pairs": [
            {
                "image": images_by_stem[stem],
                "image_relative": images_by_stem[stem].relative_to(images_root),
                "label": labels_by_stem[stem],
                "label_relative": labels_by_stem[stem].relative_to(labels_root),
                "normalized_label": normalized_labels[
                    labels_by_stem[stem].relative_to(labels_root)
                ],
            }
            for stem in sorted(images_by_stem)
        ],
    }



def build_public_yolo_training_view(
    source_root: Path,
    dataset_receipt_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate public human YOLO labels and expose a zero-copy training view."""

    source_root = source_root.resolve()
    dataset_receipt_path = dataset_receipt_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Public YOLO training output already exists: {output}")
    if not source_root.is_dir() or not dataset_receipt_path.is_file():
        raise FileNotFoundError("Public YOLO source or acquisition receipt is missing")
    acquisition = json.loads(dataset_receipt_path.read_text(encoding="utf-8-sig"))
    if (
        acquisition.get("schema_version") != "visioncortex-public-dataset-receipt/1"
        or acquisition.get("status") != "completed"
        or acquisition.get("nas_accessed") is not False
    ):
        raise RuntimeError("Public dataset acquisition receipt is not trusted")
    if str(acquisition.get("license") or "") != "CC-BY-4.0":
        raise RuntimeError("Public YOLO training requires the pinned CC-BY-4.0 source")
    _, classes = _public_yolo_classes(source_root)
    source_splits = {
        "train": _find_single_directory(source_root, "Train"),
        "val": _find_single_directory(source_root, "Valid"),
        "test": _find_single_directory(source_root, "Test"),
    }
    split_receipts = {
        split: _validate_public_yolo_split(split_root, classes)
        for split, split_root in source_splits.items()
    }
    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    label_materialization_bytes = 0
    try:
        for split, split_receipt in split_receipts.items():
            image_destination_root = temporary / "images" / split
            label_destination_root = temporary / "labels" / split
            for pair in split_receipt["file_pairs"]:
                image_destination = image_destination_root / pair["image_relative"]
                label_destination = label_destination_root / pair["label_relative"]
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(pair["image"], image_destination)
                label_destination.write_text(
                    pair["normalized_label"], encoding="utf-8"
                )
                label_materialization_bytes += len(
                    pair["normalized_label"].encode("utf-8")
                )
        data_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(classes)},
        }
        (temporary / "dataset.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        receipt = {
            "schema_version": PUBLIC_DATASET_SCHEMA,
            "status": "completed",
            "truth_status": "public_human_annotations",
            "dataset_id": acquisition.get("dataset_id"),
            "title": acquisition.get("title"),
            "license": acquisition.get("license"),
            "source_record": acquisition.get("source_record"),
            "source_dataset_receipt": str(dataset_receipt_path),
            "source_dataset_receipt_sha256": _sha256(dataset_receipt_path),
            "class_count": len(classes),
            "classes": classes,
            "split_image_counts": {
                split: item["image_count"] for split, item in split_receipts.items()
            },
            "split_annotation_counts": {
                split: item["annotation_count"] for split, item in split_receipts.items()
            },
            "source_annotation_formats": {
                "box": sum(
                    item["box_annotation_count"] for item in split_receipts.values()
                ),
                "polygon_converted_to_bounding_box": sum(
                    item["polygon_annotation_count"]
                    for item in split_receipts.values()
                ),
            },
            "class_annotation_counts": {
                class_name: sum(
                    item["class_annotation_counts"][class_name]
                    for item in split_receipts.values()
                )
                for class_name in classes
            },
            "link_mode": "image_file_symlink",
            "label_mode": "validated_box_materialization",
            "source_copy_bytes": 0,
            "label_materialization_bytes": label_materialization_bytes,
            "nas_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "dataset-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        return {
            **receipt,
            "output": str(output),
            "dataset_yaml": str(output / "dataset.yaml"),
        }
    except Exception:
        raise



def build_mapped_public_yolo_union(
    source_roots: dict[str, Path],
    mapping_path: Path,
    target_registry_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Map public human boxes into the exact production ontology without image copies."""

    output = output.resolve()
    mapping_path = mapping_path.resolve()
    target_registry_path = target_registry_path.resolve()
    if output.exists():
        raise FileExistsError(f"Mapped public YOLO output already exists: {output}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("schema_version") != "visioncortex-public-yolo-ontology-map/1":
        raise RuntimeError("Unsupported public YOLO ontology mapping schema")
    registry = json.loads(target_registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "visioncortex-closed-set-model-registry/1":
        raise RuntimeError("Unsupported target YOLO ontology registry")
    target_classes = [str(item) for item in registry.get("ontology", {}).get("classes") or []]
    if len(target_classes) != 21 or len(target_classes) != len(set(target_classes)):
        raise RuntimeError("Mapped public training requires the exact 21-class ontology")
    target_ids = {name: index for index, name in enumerate(target_classes)}
    source_mappings = mapping.get("datasets")
    if (
        not source_roots
        or not isinstance(source_mappings, dict)
        or not set(source_roots).issubset(source_mappings)
    ):
        raise RuntimeError("Every public YOLO source must exist in the mapping registry")
    oversample = {
        str(key): int(value)
        for key, value in (mapping.get("train_oversample_factors") or {}).items()
    }
    if any(key not in target_ids or value < 1 or value > 8 for key, value in oversample.items()):
        raise RuntimeError("Public YOLO oversampling policy is invalid")

    temporary = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:8]}")
    split_image_counts = {split: 0 for split in ("train", "val", "test")}
    underlying_image_counts = {split: 0 for split in ("train", "val", "test")}
    raw_mapped_image_counts = {split: 0 for split in ("train", "val", "test")}
    split_annotation_counts = {split: 0 for split in ("train", "val", "test")}
    class_annotation_counts = {name: 0 for name in target_classes}
    excluded_unmapped_images = 0
    label_materialization_bytes = 0
    source_receipts = []
    destination_names: set[tuple[str, str]] = set()
    records_by_content_hash: dict[str, list[dict[str, Any]]] = {}
    source_hash_cache: dict[Path, str] = {}
    split_priority = {"train": 0, "val": 1, "test": 2}
    try:
        for dataset_id, raw_root in sorted(source_roots.items()):
            root = raw_root.resolve()
            receipt_path = root / "dataset-receipt.json"
            yaml_path = root / "dataset.yaml"
            if not receipt_path.is_file() or not yaml_path.is_file():
                raise FileNotFoundError(
                    f"Standardized public YOLO source is incomplete: {dataset_id}"
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            if (
                receipt.get("schema_version") != PUBLIC_DATASET_SCHEMA
                or receipt.get("status") != "completed"
                or receipt.get("truth_status") != "public_human_annotations"
                or receipt.get("source_copy_bytes") != 0
                or receipt.get("nas_accessed") is not False
            ):
                raise RuntimeError(f"Untrusted standardized public source: {dataset_id}")
            classes = _dataset_yaml_classes(yaml_path)
            raw_class_map = source_mappings[dataset_id].get("class_mapping")
            if not isinstance(raw_class_map, dict) or set(raw_class_map) != set(classes):
                raise RuntimeError(
                    f"Every public class requires an explicit mapping: {dataset_id}"
                )
            class_map = {
                name: (str(raw_class_map[name]) if raw_class_map[name] is not None else None)
                for name in classes
            }
            if any(value not in target_ids for value in class_map.values() if value is not None):
                raise RuntimeError(f"Public class maps outside target ontology: {dataset_id}")
            source_receipts.append(
                {
                    "dataset_id": dataset_id,
                    "root": str(root),
                    "receipt": str(receipt_path),
                    "receipt_sha256": _sha256(receipt_path),
                    "source_classes": classes,
                    "class_mapping": class_map,
                }
            )
            for split in ("train", "val", "test"):
                images_root = root / "images" / split
                labels_root = root / "labels" / split
                images = {
                    path.relative_to(images_root).with_suffix(""): path
                    for path in images_root.rglob("*")
                    if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
                }
                labels = {
                    path.relative_to(labels_root).with_suffix(""): path
                    for path in labels_root.rglob("*.txt")
                    if path.is_file()
                }
                if not images or images.keys() != labels.keys():
                    raise RuntimeError(
                        f"Mapped public image/label mismatch: {dataset_id}/{split}"
                    )
                for relative_stem in sorted(images):
                    mapped_rows: list[str] = []
                    mapped_classes: set[str] = set()
                    for line_number, row in enumerate(
                        labels[relative_stem].read_text(encoding="utf-8-sig").splitlines(),
                        start=1,
                    ):
                        values = row.split()
                        if len(values) != 5:
                            raise RuntimeError(
                                f"Invalid public YOLO row: {labels[relative_stem]}:{line_number}"
                            )
                        try:
                            source_id = int(values[0])
                            coordinates = [float(item) for item in values[1:]]
                        except ValueError as exc:
                            raise RuntimeError("Invalid public YOLO mapping value") from exc
                        if source_id < 0 or source_id >= len(classes):
                            raise RuntimeError("Public YOLO mapping class id is out of range")
                        if (
                            not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates)
                            or coordinates[2] <= 0.0
                            or coordinates[3] <= 0.0
                        ):
                            raise RuntimeError("Public YOLO mapping box is invalid")
                        target_name = class_map[classes[source_id]]
                        if target_name is None:
                            continue
                        mapped_classes.add(target_name)
                        mapped_rows.append(
                            " ".join(
                                [str(target_ids[target_name]), *[f"{value:.8f}" for value in coordinates]]
                            )
                        )
                    if not mapped_rows:
                        excluded_unmapped_images += 1
                        continue
                    raw_mapped_image_counts[split] += 1
                    source_image = images[relative_stem].resolve(strict=True)
                    digest = source_hash_cache.get(source_image)
                    if digest is None:
                        digest = _sha256(source_image)
                        source_hash_cache[source_image] = digest
                    records_by_content_hash.setdefault(digest, []).append(
                        {
                            "dataset_id": dataset_id,
                            "split": split,
                            "relative_stem": relative_stem.as_posix(),
                            "source_image": source_image,
                            "mapped_rows": mapped_rows,
                            "mapped_classes": mapped_classes,
                        }
                    )
        excluded_cross_split_duplicate_images = 0
        merged_same_split_duplicate_images = 0
        for _digest, records in sorted(records_by_content_hash.items()):
            winning_split = max(
                (str(record["split"]) for record in records),
                key=lambda split: split_priority[split],
            )
            winning = sorted(
                (record for record in records if record["split"] == winning_split),
                key=lambda record: (
                    str(record["dataset_id"]),
                    str(record["relative_stem"]),
                ),
            )
            excluded_cross_split_duplicate_images += len(records) - len(winning)
            merged_same_split_duplicate_images += max(0, len(winning) - 1)
            authority = winning[0]
            mapped_rows = sorted(
                {
                    str(row)
                    for record in winning
                    for row in record["mapped_rows"]
                },
                key=lambda row: (
                    int(row.split()[0]),
                    tuple(float(value) for value in row.split()[1:]),
                ),
            )
            mapped_classes = {
                target_classes[int(row.split()[0])] for row in mapped_rows
            }
            underlying_image_counts[winning_split] += 1
            factor = (
                max(
                    [oversample.get(name, 1) for name in mapped_classes],
                    default=1,
                )
                if winning_split == "train"
                else 1
            )
            source_image = Path(authority["source_image"])
            base_name = _safe_name(
                f"{authority['dataset_id']}__{authority['relative_stem']}"
            )
            for repeat in range(factor):
                suffix = "" if repeat == 0 else f"__repeat{repeat}"
                destination_name = (
                    f"{base_name}{suffix}{source_image.suffix.casefold()}"
                )
                identity = (winning_split, destination_name)
                if identity in destination_names:
                    raise RuntimeError(f"Mapped public filename collision: {identity}")
                destination_names.add(identity)
                image_destination = (
                    temporary / "images" / winning_split / destination_name
                )
                label_destination = (
                    temporary
                    / "labels"
                    / winning_split
                    / f"{Path(destination_name).stem}.txt"
                )
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(source_image, image_destination)
                label_payload = "\n".join(mapped_rows) + "\n"
                label_destination.write_text(label_payload, encoding="utf-8")
                label_materialization_bytes += len(label_payload.encode("utf-8"))
                split_image_counts[winning_split] += 1
                split_annotation_counts[winning_split] += len(mapped_rows)
                for target_name in mapped_classes:
                    class_annotation_counts[target_name] += sum(
                        1
                        for mapped_row in mapped_rows
                        if int(mapped_row.split()[0]) == target_ids[target_name]
                    )
        if not split_image_counts["train"] or not split_image_counts["val"] or not split_image_counts["test"]:
            raise RuntimeError("Mapped public union requires non-empty train/val/test splits")
        data_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(target_classes)},
        }
        (temporary / "dataset.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        receipt = {
            "schema_version": PUBLIC_UNION_SCHEMA,
            "status": "completed",
            "truth_status": "public_human_annotations",
            "target_ontology_registry": str(target_registry_path),
            "target_ontology_registry_sha256": _sha256(target_registry_path),
            "class_count": len(target_classes),
            "classes": target_classes,
            "source_receipts": source_receipts,
            "raw_mapped_image_counts": raw_mapped_image_counts,
            "split_image_counts": split_image_counts,
            "underlying_image_counts": underlying_image_counts,
            "split_annotation_counts": split_annotation_counts,
            "class_annotation_counts": class_annotation_counts,
            "excluded_unmapped_image_count": excluded_unmapped_images,
            "cross_split_content_policy": "keep_test_then_val_then_train",
            "excluded_cross_split_duplicate_image_count": (
                excluded_cross_split_duplicate_images
            ),
            "merged_same_split_duplicate_image_count": (
                merged_same_split_duplicate_images
            ),
            "unique_content_hash_count": len(records_by_content_hash),
            "train_oversample_factors": oversample,
            "link_mode": "file_symlink",
            "source_copy_bytes": 0,
            "label_materialization_bytes": label_materialization_bytes,
            "nas_accessed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "dataset-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, output)
        return {**receipt, "output": str(output), "dataset_yaml": str(output / "dataset.yaml")}
    except Exception:
        raise

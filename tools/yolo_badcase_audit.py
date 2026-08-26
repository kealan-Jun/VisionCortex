from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


SAMPLE_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
ROLE_FILES = {"first_person": "First-Person.mp4", "third_person": "Third-Person.mp4"}
MODEL_PATHS = {
    "first_person": Path(r"D:\LabModels\yolo\first_person\current\best.pt"),
    "third_person": Path(r"D:\LabModels\yolo\third_person\current\best.pt"),
}
CLASS_ALIASES = {"tube-cap": "tube_cap"}
ACTOR_CLASSES = {"hand", "gloved_hand", "lab_coat"}


@dataclass(frozen=True)
class FrameRequest:
    event_id: str
    action_type: str
    role: str
    clip_path: Path
    sidecar_path: Path
    phase_index: int
    fraction: float
    frame_index: int
    frame_count: int
    expected_objects: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_class(value: Any) -> str:
    raw = str(value or "").strip().replace("-", "_")
    return CLASS_ALIASES.get(raw, raw)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_objects(sidecar: dict[str, Any]) -> tuple[str, ...]:
    cv_objects = (
        ((sidecar.get("provenance") or {}).get("cv") or {}).get("objects") or []
    )
    known = {
        "balance",
        "beaker",
        "gloved_hand",
        "lab_coat",
        "paper",
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
        "spatula",
        "tube",
        "tube_cap",
        "spearhead",
        "pipette",
        "container",
        "PPE_Storage",
        "hand",
        "reagent_bottle_open",
        "bottle_cap",
        "magnetic_stirrer",
        "tube_rack",
        "magnetic_stir_bar",
    }
    return tuple(
        sorted(
            {
                normalized
                for item in cv_objects
                if (normalized := _normalize_class(item)) in known
            }
        )
    )


def _read_frame(clip_path: Path, frame_index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(clip_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _clip_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()


def discover_requests(archive_root: Path) -> tuple[list[FrameRequest], dict[str, Any]]:
    clip_root = archive_root / "Key-Materials" / "Key-Clips"
    requests: list[FrameRequest] = []
    event_metadata: dict[str, Any] = {}
    for first_clip in sorted(clip_root.rglob("First-Person.mp4")):
        sidecar_path = first_clip.with_suffix(".json")
        if not sidecar_path.is_file():
            continue
        sidecar = _json(sidecar_path)
        event_id = str(sidecar.get("event_id") or first_clip.parent.name)
        action_type = str(sidecar.get("action_type") or "unknown")
        expected = _expected_objects(sidecar)
        event_metadata[event_id] = {
            "event_id": event_id,
            "action_type": action_type,
            "parent_event_id": sidecar.get("parent_event_id"),
            "expected_objects_from_prior_cv_sidecar": list(expected),
            "observed_facts": (sidecar.get("decision") or {}).get("observed_facts") or [],
            "prior_cv": ((sidecar.get("provenance") or {}).get("cv") or {}),
        }
        for role, filename in ROLE_FILES.items():
            clip_path = first_clip.parent / filename
            role_sidecar = clip_path.with_suffix(".json")
            if not clip_path.is_file():
                continue
            frame_count = _clip_frame_count(clip_path)
            if frame_count <= 0:
                continue
            for phase_index, fraction in enumerate(SAMPLE_FRACTIONS):
                frame_index = min(
                    frame_count - 1,
                    max(0, round((frame_count - 1) * fraction)),
                )
                requests.append(
                    FrameRequest(
                        event_id=event_id,
                        action_type=action_type,
                        role=role,
                        clip_path=clip_path,
                        sidecar_path=role_sidecar,
                        phase_index=phase_index,
                        fraction=fraction,
                        frame_index=frame_index,
                        frame_count=frame_count,
                        expected_objects=expected,
                    )
                )
    return requests, event_metadata


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _prediction_record(request: FrameRequest, result: Any) -> dict[str, Any]:
    height, width = result.orig_shape
    detections: list[dict[str, Any]] = []
    for cls, confidence, box in zip(
        result.boxes.cls.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
        result.boxes.xyxy.detach().cpu().tolist(),
        strict=True,
    ):
        detections.append(
            {
                "class_id": int(cls),
                "class_name": _normalize_class(result.names[int(cls)]),
                "confidence": round(float(confidence), 6),
                "xyxy": [round(float(value), 3) for value in box],
                "xyxy_norm": [
                    round(float(box[0]) / width, 6),
                    round(float(box[1]) / height, 6),
                    round(float(box[2]) / width, 6),
                    round(float(box[3]) / height, 6),
                ],
            }
        )
    ambiguous_pairs = []
    duplicate_pairs = []
    for index, left in enumerate(detections):
        for right in detections[index + 1 :]:
            overlap = _iou(left["xyxy"], right["xyxy"])
            if overlap < 0.65:
                continue
            pair = {
                "left_class": left["class_name"],
                "right_class": right["class_name"],
                "left_confidence": left["confidence"],
                "right_confidence": right["confidence"],
                "iou": round(overlap, 6),
            }
            if left["class_name"] == right["class_name"]:
                duplicate_pairs.append(pair)
            else:
                ambiguous_pairs.append(pair)
    return {
        "event_id": request.event_id,
        "action_type": request.action_type,
        "role": request.role,
        "clip_path": str(request.clip_path),
        "sidecar_path": str(request.sidecar_path),
        "phase_index": request.phase_index,
        "sample_fraction": request.fraction,
        "frame_index": request.frame_index,
        "frame_count": request.frame_count,
        "expected_objects_from_prior_cv_sidecar": list(request.expected_objects),
        "detections": detections,
        "class_counts": dict(Counter(item["class_name"] for item in detections)),
        "ambiguous_overlapping_class_pairs": ambiguous_pairs,
        "duplicate_overlapping_boxes": duplicate_pairs,
    }


def run_inference(
    requests: list[FrameRequest],
    *,
    batch_size: int,
    image_size: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    for role in ("first_person", "third_person"):
        role_requests = [item for item in requests if item.role == role]
        model_path = MODEL_PATHS[role]
        model = YOLO(str(model_path))
        role_started = time.perf_counter()
        decoded = 0
        failed = 0
        for offset in range(0, len(role_requests), batch_size):
            batch_requests = role_requests[offset : offset + batch_size]
            images: list[np.ndarray] = []
            valid_requests: list[FrameRequest] = []
            for request in batch_requests:
                frame = _read_frame(request.clip_path, request.frame_index)
                if frame is None:
                    failed += 1
                    continue
                decoded += 1
                images.append(frame)
                valid_requests.append(request)
            if not images:
                continue
            results = model.predict(
                source=images,
                imgsz=image_size,
                conf=confidence,
                iou=0.70,
                max_det=300,
                batch=len(images),
                device=0 if torch.cuda.is_available() else "cpu",
                half=torch.cuda.is_available(),
                verbose=False,
            )
            records.extend(
                _prediction_record(request, result)
                for request, result in zip(valid_requests, results, strict=True)
            )
        runtime[role] = {
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "requested_frames": len(role_requests),
            "decoded_frames": decoded,
            "decode_failures": failed,
            "wall_seconds": round(time.perf_counter() - role_started, 6),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records, runtime


def mine_badcases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_event_role[(record["event_id"], record["role"])].append(record)
    badcases: list[dict[str, Any]] = []
    presence_by_event: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for (event_id, role), items in sorted(by_event_role.items()):
        items.sort(key=lambda item: item["phase_index"])
        expected = sorted(
            {
                value
                for item in items
                for value in item["expected_objects_from_prior_cv_sidecar"]
            }
        )
        presence = Counter()
        confidence_by_class: dict[str, list[float]] = defaultdict(list)
        ambiguity_frames = []
        duplicate_frames = []
        for item in items:
            present = {det["class_name"] for det in item["detections"]}
            presence.update(present)
            for det in item["detections"]:
                confidence_by_class[det["class_name"]].append(det["confidence"])
            if item["ambiguous_overlapping_class_pairs"]:
                ambiguity_frames.append(item)
            if item["duplicate_overlapping_boxes"]:
                duplicate_frames.append(item)
        presence_by_event[event_id][role] = dict(presence)
        for class_name in expected:
            count = int(presence[class_name])
            confidence_values = confidence_by_class.get(class_name) or []
            if count == 0:
                badcases.append(
                    {
                        "event_id": event_id,
                        "action_type": items[0]["action_type"],
                        "role": role,
                        "issue_type": "prior_cv_expected_class_persistent_miss",
                        "class_name": class_name,
                        "severity_score": 100,
                        "phase_presence_count": 0,
                        "phase_count": len(items),
                        "max_confidence": None,
                        "representative_phase_index": 2,
                        "evidence_status": "suspected_until_visual_review",
                    }
                )
            elif count <= 2:
                badcases.append(
                    {
                        "event_id": event_id,
                        "action_type": items[0]["action_type"],
                        "role": role,
                        "issue_type": "expected_class_temporal_flicker",
                        "class_name": class_name,
                        "severity_score": 80 - count * 5,
                        "phase_presence_count": count,
                        "phase_count": len(items),
                        "max_confidence": max(confidence_values),
                        "representative_phase_index": next(
                            (
                                item["phase_index"]
                                for item in items
                                if class_name
                                not in {det["class_name"] for det in item["detections"]}
                            ),
                            2,
                        ),
                        "evidence_status": "suspected_until_visual_review",
                    }
                )
            elif max(confidence_values) < 0.25:
                badcases.append(
                    {
                        "event_id": event_id,
                        "action_type": items[0]["action_type"],
                        "role": role,
                        "issue_type": "expected_class_below_production_threshold",
                        "class_name": class_name,
                        "severity_score": 65,
                        "phase_presence_count": count,
                        "phase_count": len(items),
                        "max_confidence": max(confidence_values),
                        "representative_phase_index": 2,
                        "evidence_status": "confirmed_threshold_failure",
                    }
                )
        for item in ambiguity_frames:
            pair = max(
                item["ambiguous_overlapping_class_pairs"],
                key=lambda value: value["iou"],
            )
            badcases.append(
                {
                    "event_id": event_id,
                    "action_type": item["action_type"],
                    "role": role,
                    "issue_type": "overlapping_class_ambiguity",
                    "class_name": f"{pair['left_class']}|{pair['right_class']}",
                    "severity_score": round(45 + 40 * pair["iou"], 3),
                    "phase_presence_count": None,
                    "phase_count": len(items),
                    "max_confidence": max(
                        pair["left_confidence"], pair["right_confidence"]
                    ),
                    "representative_phase_index": item["phase_index"],
                    "overlap_iou": pair["iou"],
                    "evidence_status": "model_internal_conflict",
                }
            )
        for item in duplicate_frames:
            pair = max(
                item["duplicate_overlapping_boxes"], key=lambda value: value["iou"]
            )
            badcases.append(
                {
                    "event_id": event_id,
                    "action_type": item["action_type"],
                    "role": role,
                    "issue_type": "duplicate_overlapping_detection",
                    "class_name": pair["left_class"],
                    "severity_score": round(35 + 30 * pair["iou"], 3),
                    "phase_presence_count": None,
                    "phase_count": len(items),
                    "max_confidence": max(
                        pair["left_confidence"], pair["right_confidence"]
                    ),
                    "representative_phase_index": item["phase_index"],
                    "overlap_iou": pair["iou"],
                    "evidence_status": "model_internal_conflict",
                }
            )

    for event_id, by_role in presence_by_event.items():
        if not {"first_person", "third_person"}.issubset(by_role):
            continue
        classes = set(by_role["first_person"]) | set(by_role["third_person"])
        action_type = next(
            item["action_type"] for item in records if item["event_id"] == event_id
        )
        for class_name in sorted(classes - ACTOR_CLASSES):
            fp_count = int(by_role["first_person"].get(class_name, 0))
            tp_count = int(by_role["third_person"].get(class_name, 0))
            if max(fp_count, tp_count) >= 4 and min(fp_count, tp_count) == 0:
                missing_role = "first_person" if fp_count == 0 else "third_person"
                badcases.append(
                    {
                        "event_id": event_id,
                        "action_type": action_type,
                        "role": missing_role,
                        "issue_type": "cross_view_class_support_gap",
                        "class_name": class_name,
                        "severity_score": 55,
                        "phase_presence_count": 0,
                        "phase_count": 5,
                        "max_confidence": None,
                        "representative_phase_index": 2,
                        "other_role_presence_count": max(fp_count, tp_count),
                        "evidence_status": "viewpoint_or_detection_gap",
                    }
                )
    return sorted(
        badcases,
        key=lambda item: (
            -float(item["severity_score"]),
            item["event_id"],
            item["role"],
            item["issue_type"],
            item["class_name"],
        ),
    )


def _draw_detections(frame: np.ndarray, detections: Iterable[dict[str, Any]]) -> np.ndarray:
    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["xyxy"])
        confidence = float(detection["confidence"])
        color = (50, 220, 50) if confidence >= 0.25 else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{detection['class_name']} {confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def materialize_badcase_images(
    badcases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    output_root: Path,
    limit: int,
) -> list[dict[str, Any]]:
    by_key = {
        (item["event_id"], item["role"], item["phase_index"]): item
        for item in records
    }
    image_dir = output_root / "Badcase-Images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    seen: set[tuple[str, str, str, str]] = set()
    for badcase in badcases:
        identity = (
            badcase["event_id"],
            badcase["role"],
            badcase["issue_type"],
            badcase["class_name"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        record = by_key.get(
            (
                badcase["event_id"],
                badcase["role"],
                int(badcase["representative_phase_index"]),
            )
        )
        if record is None:
            continue
        frame = _read_frame(Path(record["clip_path"]), int(record["frame_index"]))
        if frame is None:
            continue
        predicted = _draw_detections(frame, record["detections"])
        header = np.full((80, frame.shape[1] * 2, 3), 25, dtype=np.uint8)
        cv2.putText(
            header,
            f"{badcase['event_id']} | {badcase['role']} | {badcase['issue_type']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"target={badcase['class_name']} | expected={','.join(record['expected_objects_from_prior_cv_sidecar']) or 'none'} | left=raw right=prediction",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        combined = np.vstack([header, np.hstack([frame, predicted])])
        rank = len(selected) + 1
        safe_class = badcase["class_name"].replace("|", "-")
        path = image_dir / (
            f"{rank:03d}_{badcase['event_id']}_{badcase['role']}_"
            f"{badcase['issue_type']}_{safe_class}.jpg"
        )
        cv2.imwrite(str(path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
        badcase["image_path"] = str(path)
        selected.append(badcase)
        if len(selected) >= limit:
            break
    return selected


def write_contact_sheets(
    selected: list[dict[str, Any]], output_root: Path, page_size: int = 12
) -> list[Path]:
    sheet_dir = output_root / "Contact-Sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    tile_width, tile_height = 640, 360
    columns = 3
    for page_index, offset in enumerate(range(0, len(selected), page_size), 1):
        page = selected[offset : offset + page_size]
        rows = math.ceil(len(page) / columns)
        canvas = np.full((rows * tile_height, columns * tile_width, 3), 18, dtype=np.uint8)
        for index, item in enumerate(page):
            image = cv2.imread(str(item["image_path"]))
            if image is None:
                continue
            image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            row, column = divmod(index, columns)
            canvas[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = image
        path = sheet_dir / f"YOLO-Badcases-{page_index:02d}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pages.append(path)
    return pages


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--badcase-image-limit", type=int, default=72)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    requests, event_metadata = discover_requests(args.archive_root.resolve())
    records, runtime = run_inference(
        requests,
        batch_size=max(1, args.batch_size),
        image_size=args.image_size,
        confidence=args.confidence,
    )
    badcases = mine_badcases(records)
    selected = materialize_badcase_images(
        badcases, records, output_root, max(1, args.badcase_image_limit)
    )
    contact_sheets = write_contact_sheets(selected, output_root)

    with (output_root / "Predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_root / "Event-Metadata.json").write_text(
        json.dumps(event_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "Badcase-Candidates.json").write_text(
        json.dumps(badcases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_root / "Badcase-Candidates.csv", badcases)

    issue_counts = Counter(item["issue_type"] for item in badcases)
    class_counts = Counter(item["class_name"] for item in badcases)
    role_counts = Counter(item["role"] for item in badcases)
    action_counts = Counter(item["action_type"] for item in badcases)
    summary = {
        "schema_version": "visioncortex-yolo-badcase-audit/1",
        "status": "completed_candidate_mining_pending_manual_box_gt",
        "archive_root": str(args.archive_root.resolve()),
        "event_count": len(event_metadata),
        "clip_count": len({str(item.clip_path) for item in requests}),
        "requested_frame_count": len(requests),
        "predicted_frame_count": len(records),
        "badcase_candidate_count": len(badcases),
        "materialized_badcase_image_count": len(selected),
        "contact_sheets": [str(path) for path in contact_sheets],
        "issue_counts": dict(issue_counts.most_common()),
        "class_counts": dict(class_counts.most_common()),
        "role_counts": dict(role_counts.most_common()),
        "action_type_counts": dict(action_counts.most_common()),
        "sample_fractions": SAMPLE_FRACTIONS,
        "inference": {
            "image_size": args.image_size,
            "confidence_floor": args.confidence,
            "production_threshold_reference": 0.25,
            "batch_size": args.batch_size,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runtime_by_role": runtime,
        },
        "host": platform.node(),
        "wall_seconds": round(time.perf_counter() - started, 6),
        "limitations": [
            "No independent per-box ground-truth annotations were available.",
            "Expected objects come from the prior CV sidecars and are not independent ground truth.",
            "Cross-view gaps may be caused by occlusion or viewpoint rather than detector failure.",
            "Only visually reviewed examples may be called confirmed false positives or false negatives.",
            "This PyTorch audit measures learned weights, not TensorRT scheduling performance.",
        ],
    }
    (output_root / "Summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

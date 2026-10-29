"""Bounded image checks for box-derived movement candidates.

Optical flow is observation evidence, never semantic action confirmation.
Missing/ambiguous images retain the candidate with an unverified receipt.
"""
from __future__ import annotations

import hashlib
import math
import time
from bisect import bisect_left, bisect_right
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .detection import iter_frame_evidence
from .schemas import ActionCandidate, ActionType, FrameEvidence, VideoInfo, ViewInput
from .video_io import ViewFrameReader


def _mask(shape: tuple[int, int], box: Sequence[float], padding: float = 0.0) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    left, top = max(0, int((x1 - dx) * width)), max(0, int((y1 - dy) * height))
    right, bottom = min(width, math.ceil((x2 + dx) * width)), min(height, math.ceil((y2 + dy) * height))
    result = np.zeros(shape, dtype=np.uint8)
    if right > left and bottom > top:
        result[top:bottom, left:right] = 255
    return result


def _flow(previous: np.ndarray, current: np.ndarray, mask: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray, int]:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=maximum, qualityLevel=0.01, minDistance=4, mask=mask)
    empty = np.empty((0, 2), dtype=np.float32)
    if points is None:
        return empty, empty, 0
    forward, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=3)
    if forward is None or status is None:
        return empty, empty, len(points)
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, winSize=(21, 21), maxLevel=3)
    if backward is None or back_status is None:
        return empty, empty, len(points)
    before, after = points.reshape(-1, 2), forward.reshape(-1, 2)
    valid = ((status.ravel() == 1) & (back_status.ravel() == 1)
             & np.isfinite(after).all(axis=1)
             & (np.linalg.norm(backward.reshape(-1, 2) - before, axis=1) <= 1.0))
    if error is not None:
        valid &= error.ravel() <= 20.0
    height, width = previous.shape
    valid &= (after[:, 0] >= 0) & (after[:, 0] < width) & (after[:, 1] >= 0) & (after[:, 1] < height)
    return before[valid], after[valid], len(points)


def verify_image_motion(
    previous: np.ndarray,
    current: np.ndarray,
    previous_box: Sequence[float],
    current_box: Sequence[float],
    *,
    excluded_boxes: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """Compare object features with nearby background over the same time pair."""
    receipt: dict[str, Any] = {"status": "unverified", "reason": "insufficient_image_features"}
    if previous.shape != current.shape or previous.ndim != 2:
        return {**receipt, "reason": "incompatible_frames"}
    before, after, detected = _flow(previous, current, _mask(previous.shape, previous_box), 100)
    receipt.update(feature_count=len(before), detected_feature_count=detected)
    if len(before) < 8 or len(before) / max(detected, 1) < 0.50:
        return receipt
    raw = np.linalg.norm(after - before, axis=1)
    receipt["raw_median_displacement_px"] = round(float(np.median(raw)), 5)
    # A valid, nearly stationary image contradicts a large jumping box even
    # when the surrounding scene has too little texture for camera estimation.
    if float(np.quantile(raw, 0.80)) <= 0.65:
        return {**receipt, "status": "contradicted", "reason": "stationary_image_features"}

    target_mask = _mask(previous.shape, previous_box, 0.25) | _mask(previous.shape, current_box, 0.25)
    exclusions = target_mask.copy()
    for box in excluded_boxes:
        exclusions |= _mask(previous.shape, box, 0.1)
    local_mask = _mask(previous.shape, previous_box, 1.5) & cv2.bitwise_not(exclusions)
    global_mask = cv2.bitwise_not(exclusions)
    for scope, mask in (("local_background", local_mask), ("global_background", global_mask)):
        source, target, _ = _flow(previous, current, mask, 300)
        if len(source) < 16:
            continue
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            continue
        accepted = inliers.ravel() == 1
        if accepted.sum() < 12 or accepted.mean() < 0.65:
            continue
        extent = np.ptp(source[accepted], axis=0)
        if min(extent) < 20 or (scope == "global_background" and np.prod(extent) < previous.size * 0.15):
            continue
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        if not 0.80 <= scale <= 1.25:
            continue
        background_error = np.linalg.norm(target - (source @ matrix[:, :2].T + matrix[:, 2]), axis=1)
        noise = float(np.quantile(background_error[accepted], 0.80))
        residual_vectors = after - (before @ matrix[:, :2].T + matrix[:, 2])
        residual = np.linalg.norm(residual_vectors, axis=1)
        stationary_limit, moving_limit = max(0.65, noise * 2), max(1.2, noise * 3)
        receipt.update(
            camera_model=scope, camera_affine=matrix.tolist(),
            background_feature_count=len(source), background_inlier_count=int(accepted.sum()),
            background_noise_px=round(noise, 5),
            compensated_median_displacement_px=round(float(np.median(residual)), 5),
            moving_feature_ratio=round(float(np.mean(residual >= moving_limit)), 5),
        )
        if float(np.quantile(residual, 0.80)) <= stationary_limit:
            return {**receipt, "status": "contradicted", "reason": "camera_motion_only"}
        median_vector = np.median(residual_vectors, axis=0)
        coherent = np.mean(np.linalg.norm(residual_vectors - median_vector, axis=1) <= max(1.5, np.linalg.norm(median_vector)))
        if np.median(residual) >= moving_limit and coherent >= 0.65:
            return {**receipt, "status": "supported", "reason": "object_motion_relative_to_background"}
        return {**receipt, "reason": "ambiguous_relative_motion"}
    return {**receipt, "reason": "camera_motion_unresolved"}


def _sample_pairs(candidate: ActionCandidate, maximum: int) -> list[dict[str, Any]]:
    observations = [item for item in candidate.evidence
                    if isinstance(item.get("observation_local_ms"), (int, float))
                    and item.get("track_id") is not None]
    if not observations:
        return []
    indices = set(np.linspace(0, len(observations) - 1, min(maximum, len(observations)), dtype=int).tolist())
    if maximum >= 3:
        peak = max(range(len(observations)), key=lambda i: observations[i].get("displacement_norm", 0))
        if peak not in indices and len(indices) == maximum:
            indices.remove(sorted(indices)[1])
        indices.add(peak)
    pairs = []
    for index in sorted(indices):
        item = observations[index]
        end = float(item["observation_local_ms"])
        delta = float(item.get("delta_ms") or 150.0)
        if not math.isfinite(end) or not math.isfinite(delta) or delta < 33 or end < delta or delta > 2000:
            continue
        pairs.append({"start_ms": end - delta, "end_ms": end, "track_id": item["track_id"],
                      "class_name": (item.get("object_instance") or {}).get("class_name", candidate.objects[0])})
    return pairs


def _selected_frames(path: Path, requests: set[float]) -> dict[float, FrameEvidence]:
    targets = sorted(requests)
    selected: dict[float, FrameEvidence] = {}
    for frame in iter_frame_evidence(path):
        lo, hi = bisect_left(targets, frame.local_ms - 26.0), bisect_right(targets, frame.local_ms + 26.0)
        for target in targets[lo:hi]:
            previous = selected.get(target)
            if previous is None or abs(frame.local_ms - target) < abs(previous.local_ms - target):
                selected[target] = frame
    return selected


def verify_movement_candidates(
    candidates: Sequence[ActionCandidate], views: Sequence[ViewInput], infos: dict[str, VideoInfo],
    detection_paths: dict[str, Path], config: dict[str, Any],
    *, progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Parallelize independent cameras while preserving global budget and order."""
    settings = config.get("segmentation", {}).get("movement_visual_verification", {})
    workers = max(1, min(8, int(settings.get("workers", 1))))
    movement = [c for c in candidates if c.action_type == ActionType.OBJECT_MOVEMENT]
    by_view: dict[str, list[ActionCandidate]] = defaultdict(list)
    for candidate in movement:
        by_view[candidate.view_id].append(candidate)
    if workers == 1 or len(by_view) < 2 or not settings.get("enabled", True):
        return _verify_movement_candidates_serial(
            candidates, views, infos, detection_paths, config, progress=progress)
    started = time.perf_counter()
    deadline = started + max(1.0, float(settings.get("max_wall_seconds", 600)))
    limit = max(1, int(settings.get("max_candidates", 2048)))
    allowed = frozenset(id(c) for c in movement[:limit])
    reports = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(by_view))) as executor:
        futures = {executor.submit(
            _verify_movement_candidates_serial, group, views, infos, detection_paths, config,
            _deadline=deadline, _allowed_ids=allowed): view_id
            for view_id, group in by_view.items()}
        completed = 0
        for future in as_completed(futures):
            report = future.result()
            reports[futures[future]] = report
            completed += len(report["candidates"])
            if progress:
                progress(completed, len(movement))
    first = next(iter(reports.values()))
    result = {key: first[key] for key in ("schema_version", "enabled", "policy")}
    rows = {view_id: iter(report["candidates"]) for view_id, report in reports.items()}
    result["candidates"] = [next(rows[c.view_id]) for c in movement]
    result.update(
        counts=dict(Counter(c["status"] for c in result["candidates"])),
        duration_seconds=round(time.perf_counter() - started, 6),
        decoded_frames=sum(r["decoded_frames"] for r in reports.values()),
        frame_cache_hits=sum(r["frame_cache_hits"] for r in reports.values()),
        ledger_failures={k: v for r in reports.values() for k, v in r["ledger_failures"].items()},
        model_calls=0, deleted_candidates=0, workers=min(workers, len(by_view)),
    )
    return result


def _verify_movement_candidates_serial(
    candidates: Sequence[ActionCandidate], views: Sequence[ViewInput], infos: dict[str, VideoInfo],
    detection_paths: dict[str, Path], config: dict[str, Any],
    *, progress: Callable[[int, int], None] | None = None,
    _deadline: float | None = None, _allowed_ids: frozenset[int] | None = None,
) -> dict[str, Any]:
    """Attach receipts without deleting candidates or letting one failure abort peers."""
    settings = config.get("segmentation", {}).get("movement_visual_verification", {})
    report: dict[str, Any] = {"schema_version": "visioncortex-movement-visual-verification/1",
                              "enabled": bool(settings.get("enabled", True)), "candidates": [],
                              "policy": "candidate_observation_only_not_action_confirmation"}
    if not report["enabled"]:
        return report
    started = time.perf_counter()
    maximum = max(1, min(5, int(settings.get("max_pairs_per_candidate", 3))))
    limit = max(1, int(settings.get("max_candidates", 2048)))
    budget = max(1.0, float(settings.get("max_wall_seconds", 600)))
    deadline = started + budget if _deadline is None else _deadline
    cache_size = max(1, min(4096, int(settings.get("frame_cache_size", 96))))
    movement = [c for c in candidates if c.action_type == ActionType.OBJECT_MOVEMENT]
    plans = {id(c): _sample_pairs(c, maximum) for c in movement[:limit]}
    requests: dict[str, set[float]] = defaultdict(set)
    for c in movement[:limit]:
        for pair in plans[id(c)]:
            requests[c.view_id].update((pair["start_ms"], pair["end_ms"]))
    frames, failures = {}, {}
    for view_id, targets in requests.items():
        try:
            frames[view_id] = _selected_frames(detection_paths[view_id], targets)
        except (OSError, ValueError, KeyError) as exc:
            failures[view_id] = type(exc).__name__
    by_view = {v.view_id: v for v in views}
    cache: OrderedDict[tuple[str, float], np.ndarray | None] = OrderedDict()
    decoded, hits = 0, 0
    with ViewFrameReader(max_open=2) as reader:
        def read(view_id: str, local_ms: float) -> np.ndarray | None:
            nonlocal decoded, hits
            key = (view_id, local_ms)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
                return cache[key]
            image = reader.read(by_view[view_id], infos[view_id], local_ms)
            decoded += 1
            if image is not None:
                width = min(640, image.shape[1])
                image = cv2.cvtColor(cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1]))), cv2.COLOR_BGR2GRAY)
            cache[key] = image
            while len(cache) > cache_size:
                cache.popitem(last=False)
            return image

        for index, candidate in enumerate(movement):
            checks: list[dict[str, Any]] = []
            reason = "no_timed_track_observations"
            if (index >= limit or time.perf_counter() >= deadline
                    or (_allowed_ids is not None and id(candidate) not in _allowed_ids)):
                reason = "verification_budget_exhausted"
            elif candidate.view_id in failures:
                reason = "source_ledger_unavailable"
            else:
                for pair in plans.get(id(candidate), []):
                    check = {**pair, "status": "unverified", "reason": "track_or_frame_unavailable"}
                    try:
                        ledger = frames.get(candidate.view_id, {})
                        first, last = ledger.get(pair["start_ms"]), ledger.get(pair["end_ms"])
                        if first is not None and last is not None:
                            boxes = [[b for b in f.detections if b.track_id == pair["track_id"] and b.class_name == pair["class_name"]]
                                     for f in (first, last)]
                            if all(len(bs) == 1 for bs in boxes) and last.local_ms > first.local_ms:
                                before, after = read(candidate.view_id, first.local_ms), read(candidate.view_id, last.local_ms)
                                if before is not None and after is not None:
                                    check.update(verify_image_motion(before, after, boxes[0][0].xyxy_norm, boxes[1][0].xyxy_norm,
                                        excluded_boxes=[b.xyxy_norm for f in (first, last) for b in f.detections if b.class_name in {"hand", "gloved_hand"}]))
                                    check.update(start_ms=first.local_ms, end_ms=last.local_ms,
                                                 previous_box=list(boxes[0][0].xyxy_norm), current_box=list(boxes[1][0].xyxy_norm),
                                                 input_gray_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in (before, after)])
                    except (OSError, ValueError, KeyError, cv2.error) as exc:
                        check.update(reason="image_verification_error", error_type=type(exc).__name__)
                    checks.append(check)
            statuses = Counter(c["status"] for c in checks)
            status = "supported" if statuses["supported"] >= 2 else (
                "contradicted" if len(checks) >= 2 and statuses["contradicted"] == len(checks) else "unverified")
            receipt = {"status": status, "reason": reason if not checks else "bounded_temporal_image_checks",
                       "checks": checks, "physical_action_confirmed": False}
            candidate.provenance["movement_visual_verification"] = receipt
            if status != "supported":
                candidate.uncertainty.append("画面尚未支持该物体移动候选；保留记录，不作为移动动作证据")
            report["candidates"].append({"candidate_id": candidate.candidate_id, "view_id": candidate.view_id,
                                         "start_ms": candidate.global_start_ms, "end_ms": candidate.global_end_ms,
                                         "objects": list(candidate.objects), **receipt})
            if progress is not None and (index % 20 == 0 or index + 1 == len(movement)):
                progress(index + 1, len(movement))
    report.update(counts=dict(Counter(c["status"] for c in report["candidates"])),
                  duration_seconds=round(time.perf_counter() - started, 6), decoded_frames=decoded,
                  frame_cache_hits=hits, ledger_failures=failures, model_calls=0, deleted_candidates=0)
    return report

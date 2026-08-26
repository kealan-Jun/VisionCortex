from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .collection_curation import (
    CONTENT_PROBE_ANCHOR_FRACTIONS,
    CONTENT_PROBE_DISTRIBUTION_STRATEGY,
    CONTENT_PROBE_SCHEMA_VERSION,
    FULL_TIMELINE_SWEEP_SCHEMA_VERSION,
    _receipt_path,
    evaluate_full_timeline_semantic_consensus,
)
from .mllm import ArkStepAnalyzer
from .schemas import RunManifest, VideoInfo, ViewInput, ViewRole
from .video_io import ViewFrameReader, probe_view, probe_views


CONTENT_PROBE_SYSTEM_PROMPT = """你是实验室录像内容分类审计模型。输入来自同一条完整时间轴上的分布式抽样，标签给出视角、角色、时间点、区间选择原因和运动分数。
只判断录像是否包含真实实验操作，不判断实验是否成功，不补写试剂、剂量或目的。

real_experiment 必须至少有一段画面直接显示实验人员对实验对象执行有目的的实验动作链，例如称量、移液器源到目标操作、容器开合、清洗、加样、搅拌或设备控制。仅有人经过、摆拍、拿起相机、空台面、色卡/测试板、镜头遮挡、设备录制测试、静态器材或单次无上下文触碰都不够。
仅拿起、移动或重新摆放未开合的瓶子、瓶盖、烧杯或其他器具，不是实验动作链；手放在设备外壳上也不是设备控制。除非画面直接显示按键/旋钮/参数/运行状态改变、样品进出，或容器/液体/样品状态发生可见变化，否则这些行为只能算准备、整理或录制测试线索。
recording_or_hardware_test 仅在全时间轴抽样均未见真实实验动作，且画面明确是录制/硬件/机位测试，或完整短录像只有准备、走动、遮挡、静态场景时使用。
证据不足、关键区间看不清或不同区间冲突时必须 inconclusive。不能用文件名中的 test/fz/undefined 代替画面证据。

输出单个 JSON 对象：
{
  "verdict":"real_experiment/recording_or_hardware_test/inconclusive",
  "confidence":0.0,
  "visible_experimental_action":false,
  "purposeful_experimental_action_chain":false,
  "action_chain_steps":[{"image_label":"精确输入标签", "time_seconds":0.0, "observable_step":"直接可见步骤", "experimental_change":"直接可见的对象/设备/样品变化；没有则为空字符串"}],
  "action_evidence":[{"image_label":"精确输入标签", "observable_action":"直接可见动作", "objects":["可见对象"]}],
  "recording_test_cues":["直接可见测试线索"],
  "no_operation_across_all_intervals":false,
  "interval_assessments":[{"selection_reason":"uniform_anchor/motion_peak", "time_seconds":0.0, "assessment":"可见事实"}],
  "rationale":"保守分类理由",
  "uncertainties":["不确定项"]
}
禁止输出 Markdown。"""

@dataclass(frozen=True)
class ProbeFrame:
    view_id: str
    role: str
    interval_index: int
    selection_reason: str
    time_seconds: float
    motion_score: float
    jpeg: bytes


def distributed_probe_intervals(
    timeline_seconds: float,
    maximum_seconds: float = 300.0,
) -> list[dict[str, Any]]:
    """Return non-overlapping full-timeline anchors within the hard budget."""

    duration = float(timeline_seconds)
    budget = float(maximum_seconds)
    if duration <= 0.0 or budget <= 0.0:
        raise ValueError("probe timeline and budget must be positive")
    if duration <= budget:
        return [
            {
                "start_seconds": 0.0,
                "end_seconds": duration,
                "selection_reason": "uniform_anchor",
                "anchor_fraction": None,
            }
        ]
    width = min(40.0, budget / 7.0)
    intervals: list[dict[str, Any]] = []
    for fraction in CONTENT_PROBE_ANCHOR_FRACTIONS:
        center = duration * fraction
        start = min(max(0.0, center - width / 2.0), duration - width)
        intervals.append(
            {
                "start_seconds": start,
                "end_seconds": start + width,
                "selection_reason": "uniform_anchor",
                "anchor_fraction": fraction,
            }
        )
    return sorted(intervals, key=lambda item: float(item["start_seconds"]))


def motion_followup_interval(
    timeline_seconds: float,
    anchor_intervals: Sequence[dict[str, Any]],
    observed_peak_seconds: float,
    remaining_budget_seconds: float,
) -> dict[str, Any]:
    """Choose a bounded non-overlapping gap nearest observed anchor motion."""

    ordered = sorted(
        (
            float(item["start_seconds"]),
            float(item["end_seconds"]),
        )
        for item in anchor_intervals
    )
    gaps = [
        (left[1], right[0])
        for left, right in zip(ordered, ordered[1:], strict=False)
        if right[0] - left[1] > 0.05
    ]
    if not gaps:
        raise ValueError("no non-overlapping gap remains for motion follow-up")
    peak = float(observed_peak_seconds)
    gap_start, gap_end = min(
        gaps,
        key=lambda gap: (
            0.0
            if gap[0] <= peak <= gap[1]
            else min(abs(peak - gap[0]), abs(peak - gap[1])),
            gap[0],
        ),
    )
    length = min(50.0, float(remaining_budget_seconds), gap_end - gap_start)
    if length <= 0.0:
        raise ValueError("no probe budget remains for motion follow-up")
    center = min(max(peak, gap_start + length / 2.0), gap_end - length / 2.0)
    start = center - length / 2.0
    return {
        "start_seconds": max(0.0, start),
        "end_seconds": min(float(timeline_seconds), start + length),
        "selection_reason": "motion_peak",
        "anchor_fraction": None,
        "selection_basis": "nearest_non_overlapping_gap_to_highest_anchor_motion",
        "observed_anchor_peak_seconds": peak,
    }


def gate_probe_verdict(model_result: dict[str, Any]) -> tuple[str, list[str]]:
    """Convert the model proposal into a fail-closed classification."""

    proposed = str(model_result.get("verdict") or "inconclusive")
    confidence = float(model_result.get("confidence") or 0.0)
    reasons: list[str] = []
    if proposed == "real_experiment":
        evidence = [
            item
            for item in (model_result.get("action_evidence") or [])
            if isinstance(item, dict)
            and str(item.get("observable_action") or "").strip()
            and str(item.get("image_label") or "").strip()
        ]
        if (
            confidence >= 0.72
            and model_result.get("visible_experimental_action") is True
            and model_result.get("purposeful_experimental_action_chain") is True
            and len(
                [
                    item
                    for item in (model_result.get("action_chain_steps") or [])
                    if isinstance(item, dict)
                    and str(item.get("image_label") or "").strip()
                    and str(item.get("observable_step") or "").strip()
                ]
            )
            >= 2
            and any(
                str(item.get("experimental_change") or "").strip()
                for item in (model_result.get("action_chain_steps") or [])
                if isinstance(item, dict)
            )
            and evidence
        ):
            return proposed, reasons
        reasons.append("real_experiment_purposeful_action_chain_gate_not_met")
    elif proposed == "recording_or_hardware_test":
        if (
            confidence >= 0.85
            and model_result.get("visible_experimental_action") is False
            and model_result.get("no_operation_across_all_intervals") is True
        ):
            return proposed, reasons
        reasons.append("recording_test_all_intervals_gate_not_met")
    else:
        reasons.append("model_inconclusive")
    return "inconclusive", reasons


def _resize_for_review(frame: np.ndarray, maximum_width: int = 960) -> np.ndarray:
    if frame.shape[1] <= maximum_width:
        return frame
    scale = maximum_width / frame.shape[1]
    return cv2.resize(
        frame,
        (maximum_width, max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _decode_probe_frames(
    selected_views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    intervals: Sequence[dict[str, Any]],
    sampling_fps: float,
) -> list[ProbeFrame]:
    frames: list[ProbeFrame] = []
    with ViewFrameReader(max_open=2) as reader:
        for view in selected_views:
            previous_by_interval: dict[int, np.ndarray] = {}
            info = infos[view.view_id]
            for interval_index, interval in enumerate(intervals, start=1):
                start = float(interval["start_seconds"])
                end = float(interval["end_seconds"])
                count = max(1, int(np.ceil((end - start) * sampling_fps)))
                times = np.linspace(
                    start + min(0.5 / sampling_fps, (end - start) / 2.0),
                    max(start, end - min(0.5 / sampling_fps, (end - start) / 2.0)),
                    count,
                )
                for timestamp in times:
                    local_ms = max(0.0, float(timestamp) * 1000.0)
                    # Never relabel the last available frame as evidence from
                    # an uncovered tail.  A shorter view simply contributes no
                    # frame outside its physical media timeline.
                    if local_ms >= max(0.0, info.duration_ms - 1.0):
                        continue
                    frame = reader.read(view, info, local_ms)
                    if frame is None:
                        continue
                    review = _resize_for_review(frame)
                    gray = cv2.cvtColor(
                        cv2.resize(review, (160, 90)), cv2.COLOR_BGR2GRAY
                    )
                    previous = previous_by_interval.get(interval_index)
                    score = (
                        float(cv2.absdiff(gray, previous).mean())
                        if previous is not None
                        else 0.0
                    )
                    previous_by_interval[interval_index] = gray
                    ok, encoded = cv2.imencode(
                        ".jpg", review, [cv2.IMWRITE_JPEG_QUALITY, 88]
                    )
                    if not ok:
                        continue
                    frames.append(
                        ProbeFrame(
                            view_id=view.view_id,
                            role=view.role.value,
                            interval_index=interval_index,
                            selection_reason=str(interval["selection_reason"]),
                            time_seconds=float(timestamp),
                            motion_score=score,
                            jpeg=encoded.tobytes(),
                        )
                    )
    return frames


def _select_review_frames(
    frames: Sequence[ProbeFrame], maximum_images: int = 24
) -> list[ProbeFrame]:
    """Select a motion-rich storyboard while preserving per-view balance."""

    coverage: list[ProbeFrame] = []
    keys = sorted(
        {
            (item.selection_reason, item.interval_index, item.view_id)
            for item in frames
        },
        key=lambda item: (item[0], item[1], item[2]),
    )
    for key in keys:
        candidates = [
            item
            for item in frames
            if (item.selection_reason, item.interval_index, item.view_id) == key
        ]
        coverage.append(
            max(candidates, key=lambda item: (item.motion_score, -item.time_seconds))
        )
    selected = list(dict.fromkeys(coverage))
    view_ids = sorted({item.view_id for item in frames})
    per_view_quota = max(1, maximum_images // max(1, len(view_ids)))
    for view_id in view_ids:
        view_selected = sum(item.view_id == view_id for item in selected)
        for item in sorted(
            (frame for frame in frames if frame.view_id == view_id),
            key=lambda frame: (-frame.motion_score, frame.time_seconds),
        ):
            if item not in selected:
                selected.append(item)
                view_selected += 1
            if view_selected >= per_view_quota or len(selected) >= maximum_images:
                break
    for item in sorted(
        frames,
        key=lambda frame: (-frame.motion_score, frame.time_seconds, frame.view_id),
    ):
        if item not in selected:
            selected.append(item)
        if len(selected) >= maximum_images:
            break
    return sorted(
        selected[:maximum_images],
        key=lambda item: (item.time_seconds, item.role, item.view_id),
    )


def run_bounded_content_probe(
    config: dict[str, Any],
    manifest: RunManifest,
    *,
    display_name: str,
    expected_timeline_seconds: float,
    third_view_id: str | None = None,
) -> dict[str, Any]:
    settings = config.get("collection_curation") or {}
    maximum_seconds = float(settings.get("content_probe_max_seconds", 300.0))
    maximum_views = int(settings.get("content_probe_max_views", 2))
    if maximum_views < 2:
        raise ValueError("content probe requires one first- and one third-person view")
    first = next(
        (view for view in manifest.views if view.role == ViewRole.FIRST_PERSON), None
    )
    third_views = [
        view for view in manifest.views if view.role == ViewRole.THIRD_PERSON
    ]
    third = (
        next((view for view in third_views if view.view_id == third_view_id), None)
        if third_view_id
        else next(iter(third_views), None)
    )
    if first is None or third is None:
        available = [view.view_id for view in third_views]
        raise ValueError(
            "content probe requires a resolved cross-view pair; "
            f"requested_third_view={third_view_id!r} available={available}"
        )
    selected_views = [first, third]
    infos = {view.view_id: probe_view(view) for view in selected_views}
    duration = float(expected_timeline_seconds)
    if duration <= 0.0:
        raise ValueError("catalog timeline duration is missing")
    anchors = distributed_probe_intervals(duration, maximum_seconds)
    frames = _decode_probe_frames(selected_views, infos, anchors, sampling_fps=1.0)
    intervals = list(anchors)
    if duration > maximum_seconds:
        if not frames:
            raise RuntimeError("motion anchor decode produced no frames")
        peak = max(frames, key=lambda item: item.motion_score)
        used = sum(
            float(item["end_seconds"]) - float(item["start_seconds"])
            for item in intervals
        )
        followup = motion_followup_interval(
            duration,
            intervals,
            peak.time_seconds,
            maximum_seconds - used,
        )
        intervals.append(followup)
        intervals.sort(key=lambda item: float(item["start_seconds"]))
        frames.extend(
            _decode_probe_frames(
                selected_views, infos, [followup], sampling_fps=1.0
            )
        )
    decoded_seconds = sum(
        float(item["end_seconds"]) - float(item["start_seconds"])
        for item in intervals
    )
    if decoded_seconds > maximum_seconds + 1e-6:
        raise RuntimeError("content probe interval budget exceeded")
    if not frames:
        raise RuntimeError("content probe decoded no review frames")

    run_id = f"probe-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    artifact_root = (
        Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "Content-Probe-Artifacts"
        / manifest.experiment_id
        / run_id
    )
    artifact_root.mkdir(parents=True, exist_ok=False)
    selected = _select_review_frames(frames)
    images: list[tuple[str, Path]] = []
    image_records: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        path = artifact_root / f"frame-{index:02d}.jpg"
        temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
        temporary.write_bytes(item.jpeg)
        os.replace(temporary, path)
        label = (
            f"image={index:02d}; view_id={item.view_id}; role={item.role}; "
            f"selection_reason={item.selection_reason}; "
            f"timeline_seconds={item.time_seconds:.3f}; "
            f"motion_score={item.motion_score:.3f}"
        )
        images.append((label, path))
        image_records.append(
            {
                "label": label,
                "path": str(path),
                "sha256": hashlib.sha256(item.jpeg).hexdigest(),
            }
        )

    analyzer = ArkStepAnalyzer(config)
    try:
        result = analyzer._call(
            CONTENT_PROBE_SYSTEM_PROMPT,
            {
                "experiment_id": manifest.experiment_id,
                "display_name": display_name,
                "timeline_duration_seconds": duration,
                "sampled_intervals": intervals,
                "view_pair": [
                    {"view_id": view.view_id, "role": view.role.value}
                    for view in selected_views
                ],
                "task": "仅按可见内容分类真实实验、录制/硬件测试或不确定",
            },
            images,
            max_images=24,
        )
    finally:
        analyzer.close()
    if result.get("status") != "completed":
        verdict, gate_reasons = "inconclusive", ["ark_probe_not_completed"]
    else:
        verdict, gate_reasons = gate_probe_verdict(result)
    full_timeline_observed = (
        duration <= maximum_seconds
        and abs(decoded_seconds - duration) <= 0.1
    )
    # A distributed sample can prove that a long recording contains a real
    # experiment when it captures a direct action, but absence inside the
    # sampled windows cannot prove that an experiment is absent elsewhere on
    # the timeline.  Keep the negative path fail-closed so a late or narrow
    # experiment is never discarded as a recording/hardware test.
    if verdict == "recording_or_hardware_test" and not full_timeline_observed:
        verdict = "inconclusive"
        gate_reasons.append(
            "negative_long_timeline_probe_cannot_exclude_unsampled_experiment"
        )
    receipt = {
        "schema_version": CONTENT_PROBE_SCHEMA_VERSION,
        "experiment_id": manifest.experiment_id,
        "display_name": display_name,
        "scope": "classification_only",
        "production_completion_eligible": False,
        "negative_exclusion_eligible": full_timeline_observed,
        "source_copy_bytes": 0,
        "decoded_timeline_seconds": round(decoded_seconds, 6),
        "timeline_duration_seconds": duration,
        "distribution_strategy": CONTENT_PROBE_DISTRIBUTION_STRATEGY,
        "sampled_intervals": [
            {
                "start_seconds": round(float(item["start_seconds"]), 6),
                "end_seconds": round(float(item["end_seconds"]), 6),
                "selection_reason": item["selection_reason"],
                **(
                    {"anchor_fraction": item.get("anchor_fraction")}
                    if item.get("anchor_fraction") is not None
                    else {}
                ),
                **(
                    {"selection_basis": item.get("selection_basis")}
                    if item.get("selection_basis")
                    else {}
                ),
            }
            for item in intervals
        ],
        "views": [
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "source_segment_count": len(view.segments) or 1,
                "probed_duration_seconds": round(
                    infos[view.view_id].duration_ms / 1000.0, 6
                ),
            }
            for view in selected_views
        ],
        "verdict": verdict,
        "model_proposed_verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "gate_reasons": gate_reasons,
        "review_backend": result.get("model") or "ark_unavailable",
        "ark_status": result.get("status"),
        "ark_latency_seconds": result.get("latency_seconds"),
        "ark_usage": result.get("usage") or {},
        "model_result": result,
        "selected_review_frames": image_records,
        "decoded_frame_count": len(frames),
        "selected_review_frame_count": len(selected),
        "artifact_root": str(artifact_root),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    receipt_path = _receipt_path(config, manifest.experiment_id)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.is_file():
        previous_bytes = receipt_path.read_bytes()
        previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
        history_root = (
            receipt_path.parent
            / "History"
            / manifest.experiment_id
        )
        history_root.mkdir(parents=True, exist_ok=True)
        history_path = history_root / (
            f"{run_id}--previous-{previous_sha256[:12]}.json"
        )
        shutil.copy2(receipt_path, history_path)
        receipt["previous_receipt"] = {
            "path": str(history_path),
            "sha256": previous_sha256,
        }
    if receipt["verdict"] != "real_experiment":
        # Direct positive evidence is monotonic.  A later camera may be
        # inconclusive, but it cannot erase an earlier independently gated
        # action chain.  Keep the newer observation for audit while restoring
        # the strongest valid positive receipt as the catalog decision.
        positive_candidates: list[tuple[Path, dict[str, Any]]] = []
        history_root = receipt_path.parent / "History" / manifest.experiment_id
        for candidate_path in (
            sorted(history_root.glob("*.json"))
            if history_root.is_dir()
            else []
        ):
            try:
                candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            model_result = candidate.get("model_result") or {}
            if (
                candidate.get("schema_version") == CONTENT_PROBE_SCHEMA_VERSION
                and candidate.get("experiment_id") == manifest.experiment_id
                and int(candidate.get("source_copy_bytes", -1)) == 0
                and candidate.get("verdict") == "real_experiment"
                and gate_probe_verdict(model_result)[0] == "real_experiment"
            ):
                positive_candidates.append((candidate_path, candidate))
        if positive_candidates:
            positive_path, positive = positive_candidates[-1]
            receipt["additional_probe_observation"] = {
                "verdict": receipt["verdict"],
                "model_proposed_verdict": receipt.get("model_proposed_verdict"),
                "confidence": receipt.get("confidence"),
                "gate_reasons": receipt.get("gate_reasons") or [],
                "ark_status": receipt.get("ark_status"),
                "ark_usage": receipt.get("ark_usage") or {},
                "artifact_root": receipt.get("artifact_root"),
                "selected_review_frames": receipt.get("selected_review_frames")
                or [],
            }
            receipt.update(
                {
                    "verdict": "real_experiment",
                    "model_proposed_verdict": (
                        "retained_prior_direct_positive_action_chain"
                    ),
                    "confidence": positive.get("confidence"),
                    "gate_reasons": [],
                    "review_backend": positive.get("review_backend"),
                    "model_result": positive.get("model_result") or {},
                    "retained_positive_evidence": {
                        "path": str(positive_path),
                        "sha256": hashlib.sha256(
                            positive_path.read_bytes()
                        ).hexdigest(),
                        "artifact_root": positive.get("artifact_root"),
                        "selected_review_frames": positive.get(
                            "selected_review_frames"
                        )
                        or [],
                        "views": positive.get("views") or [],
                        "sampled_intervals": positive.get("sampled_intervals")
                        or [],
                    },
                }
            )
    temporary = receipt_path.with_name(
        f".{receipt_path.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    return {**receipt, "receipt_path": str(receipt_path)}


def run_full_timeline_content_sweep(
    config: dict[str, Any],
    manifest: RunManifest,
    *,
    display_name: str,
    expected_timeline_seconds: float,
) -> dict[str, Any]:
    """Semantically review every contiguous <=5 minute band of a long video.

    This is deliberately separate from the bounded classification probe.  It
    is a production-grade negative adjudication input: the sum of its chunks
    covers the complete timeline, while each individual Ark request remains
    bounded and auditable.
    """

    settings = config.get("collection_curation") or {}
    chunk_seconds = float(settings.get("content_probe_max_seconds", 300.0))
    if chunk_seconds <= 0.0 or chunk_seconds > 300.0:
        raise ValueError("full-timeline sweep chunks must be in (0, 300] seconds")
    recording_duration = float(expected_timeline_seconds)
    if recording_duration <= chunk_seconds:
        raise ValueError("full-timeline sweep is only for timelines over five minutes")
    first = next(
        (view for view in manifest.views if view.role == ViewRole.FIRST_PERSON), None
    )
    third_views = [
        view for view in manifest.views if view.role == ViewRole.THIRD_PERSON
    ]
    if first is None or len(third_views) < 2:
        raise ValueError(
            "full-timeline sweep requires one first-person and two distinct "
            "third-person views"
        )
    infos = probe_views(manifest.views)
    media_duration = max(
        float(info.duration_ms) / 1000.0 for info in infos.values()
    )
    if media_duration <= chunk_seconds:
        raise ValueError(
            "full-timeline source media is not longer than five minutes"
        )
    if media_duration > recording_duration + 1.0:
        raise RuntimeError(
            "source media duration exceeds indexed recording timeline: "
            f"media={media_duration:.3f}s index={recording_duration:.3f}s"
        )
    sweep_id = f"full-sweep-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    artifact_root = (
        Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "Full-Timeline-Sweep-Artifacts"
        / manifest.experiment_id
        / sweep_id
    )
    artifact_root.mkdir(parents=True, exist_ok=False)
    analyzer = ArkStepAnalyzer(config)
    chunks: list[dict[str, Any]] = []
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }
    try:
        start = 0.0
        chunk_index = 0
        while start < media_duration - 1e-6:
            chunk_index += 1
            end = min(media_duration, start + chunk_seconds)
            # Review every source view that contains media in this band.  The
            # full CV pass remains the high-rate detector; this storyboard is
            # the independent semantic audit and must not alternate views in a
            # way that could hide an action visible in only one camera.
            selected_views = [
                view
                for view in manifest.views
                if float(infos[view.view_id].duration_ms) / 1000.0
                > start + 0.001
            ]
            interval = {
                "start_seconds": start,
                "end_seconds": end,
                "selection_reason": "uniform_anchor",
                "anchor_fraction": None,
            }
            frames = _decode_probe_frames(
                selected_views, infos, [interval], sampling_fps=1.0
            )
            if not frames:
                raise RuntimeError(
                    f"full-timeline sweep decoded no frames for {start:.3f}-{end:.3f}"
                )
            selected = _select_review_frames(frames)
            chunk_root = artifact_root / f"chunk-{chunk_index:03d}"
            chunk_root.mkdir(parents=True, exist_ok=False)
            images: list[tuple[str, Path]] = []
            image_records: list[dict[str, Any]] = []
            for image_index, item in enumerate(selected, start=1):
                path = chunk_root / f"frame-{image_index:02d}.jpg"
                temporary = path.with_name(
                    f".{path.name}.partial-{uuid.uuid4().hex[:8]}"
                )
                temporary.write_bytes(item.jpeg)
                os.replace(temporary, path)
                label = (
                    f"image={image_index:02d}; chunk={chunk_index:03d}; "
                    f"view_id={item.view_id}; role={item.role}; "
                    f"selection_reason=full_timeline_chunk; "
                    f"timeline_seconds={item.time_seconds:.3f}; "
                    f"motion_score={item.motion_score:.3f}"
                )
                images.append((label, path))
                image_records.append(
                    {
                        "label": label,
                        "path": str(path),
                        "sha256": hashlib.sha256(item.jpeg).hexdigest(),
                    }
                )
            result = analyzer._call(
                CONTENT_PROBE_SYSTEM_PROMPT,
                {
                    "experiment_id": manifest.experiment_id,
                    "display_name": display_name,
                    "recording_timeline_duration_seconds": recording_duration,
                    "media_timeline_duration_seconds": media_duration,
                    "chunk_index": chunk_index,
                    "chunk_start_seconds": start,
                    "chunk_end_seconds": end,
                    "chunk_contiguously_covers_full_timeline": True,
                    "active_views": [
                        {"view_id": view.view_id, "role": view.role.value}
                        for view in selected_views
                    ],
                    "task": (
                        "仅按本连续区块内直接可见内容分类；不得用其他区块或文件名推断"
                    ),
                },
                images,
                max_images=24,
            )
            gated_verdict, gate_reasons = (
                gate_probe_verdict(result)
                if result.get("status") == "completed"
                else ("inconclusive", ["ark_probe_not_completed"])
            )
            usage = result.get("usage") or {}
            for key in total_usage:
                total_usage[key] += int(usage.get(key) or 0)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_seconds": round(start, 6),
                    "end_seconds": round(end, 6),
                    "decoded_timeline_seconds": round(end - start, 6),
                    "views": [
                        {
                            "view_id": view.view_id,
                            "role": view.role.value,
                            "media_duration_seconds": round(
                                float(infos[view.view_id].duration_ms) / 1000.0,
                                6,
                            ),
                            "decoded_frame_count": sum(
                                frame.view_id == view.view_id for frame in frames
                            ),
                            "selected_review_frame_count": sum(
                                frame.view_id == view.view_id for frame in selected
                            ),
                        }
                        for view in selected_views
                    ],
                    "decoded_frame_count": len(frames),
                    "selected_review_frame_count": len(selected),
                    "selected_review_frames": image_records,
                    "gated_verdict": gated_verdict,
                    "gate_reasons": gate_reasons,
                    "ark_status": result.get("status"),
                    "ark_model": result.get("model"),
                    "ark_latency_seconds": result.get("latency_seconds"),
                    "ark_usage": usage,
                    "model_result": result,
                }
            )
            start = end
    finally:
        analyzer.close()
    all_completed = all(item["ark_status"] == "completed" for item in chunks)
    all_visible_false = all(
        (item.get("model_result") or {}).get("visible_experimental_action") is False
        for item in chunks
    )
    all_chunks_negative = all(
        item["gated_verdict"] == "recording_or_hardware_test" for item in chunks
    )
    source_view_coverage = {
        view.view_id: {
            "role": view.role.value,
            "media_duration_seconds": round(
                float(infos[view.view_id].duration_ms) / 1000.0, 6
            ),
            "semantic_sampling_fps": 1.0,
            "decoded_frame_count": sum(
                int(view_item.get("decoded_frame_count") or 0)
                for item in chunks
                for view_item in item["views"]
                if view_item["view_id"] == view.view_id
            ),
            "selected_review_frame_count": sum(
                int(view_item.get("selected_review_frame_count") or 0)
                for item in chunks
                for view_item in item["views"]
                if view_item["view_id"] == view.view_id
            ),
        }
        for view in manifest.views
    }
    all_source_views_sampled = all(
        int(item["decoded_frame_count"]) > 0
        and int(item["selected_review_frame_count"]) > 0
        for item in source_view_coverage.values()
    )
    any_real = any(item["gated_verdict"] == "real_experiment" for item in chunks)
    status = (
        "real_experiment_visible"
        if any_real
        else "no_visible_experimental_action_across_full_timeline"
        if all_completed
        and all_visible_false
        and all_chunks_negative
        and all_source_views_sampled
        else "inconclusive"
    )
    receipt = {
        "schema_version": FULL_TIMELINE_SWEEP_SCHEMA_VERSION,
        "sweep_id": sweep_id,
        "experiment_id": manifest.experiment_id,
        "display_name": display_name,
        "scope": "full_timeline_classification_adjudication",
        "production_completion_eligible": False,
        "source_copy_bytes": 0,
        "timeline_duration_seconds": recording_duration,
        "recording_timeline_duration_seconds": recording_duration,
        "media_timeline_duration_seconds": round(media_duration, 6),
        "indexed_capture_gap_seconds": round(
            max(0.0, recording_duration - media_duration), 6
        ),
        "semantic_sampling_fps": 1.0,
        "decoded_timeline_seconds": round(
            sum(float(item["decoded_timeline_seconds"]) for item in chunks), 6
        ),
        "maximum_chunk_seconds": chunk_seconds,
        "coverage_start_seconds": 0.0,
        "coverage_end_seconds": round(media_duration, 6),
        "coverage_gap_seconds": 0.0,
        "chunk_count": len(chunks),
        "distinct_third_person_views": sorted(
            {
                str(view["view_id"])
                for item in chunks
                for view in item["views"]
                if view["role"] == ViewRole.THIRD_PERSON.value
            }
        ),
        "all_ark_completed": all_completed,
        "all_chunks_visible_experimental_action_false": all_visible_false,
        "all_chunks_negative": all_chunks_negative,
        "all_source_views_sampled": all_source_views_sampled,
        "source_view_coverage": source_view_coverage,
        "status": status,
        "ark_usage": total_usage,
        "chunks": chunks,
        "artifact_root": str(artifact_root),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    receipt_path = (
        Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "Full-Timeline-Sweep-Receipts"
        / f"{manifest.experiment_id}.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(
        f".{receipt_path.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    return {**receipt, "receipt_path": str(receipt_path)}


def adjudicate_full_timeline_sweep_with_cv(
    config: dict[str, Any],
    experiment_id: str,
    cv_staging_root: Path,
) -> dict[str, Any]:
    """Exclude one long ambiguous recording only after exhaustive dual audit."""

    receipt_path = _receipt_path(config, experiment_id)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"content probe receipt is missing: {receipt_path}")
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    sweep_path = (
        Path(config["storage"]["local_cache_root"])
        / "Collection-Catalog"
        / "Full-Timeline-Sweep-Receipts"
        / f"{experiment_id}.json"
    ).resolve(strict=True)
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    if sweep.get("schema_version") != FULL_TIMELINE_SWEEP_SCHEMA_VERSION:
        raise RuntimeError("full-timeline semantic sweep schema mismatch")
    if sweep.get("experiment_id") != experiment_id:
        raise RuntimeError("full-timeline semantic sweep experiment mismatch")
    if int(sweep.get("source_copy_bytes", -1)) != 0:
        raise RuntimeError("full-timeline semantic sweep copied source media")
    semantic_consensus = evaluate_full_timeline_semantic_consensus(sweep)
    if semantic_consensus.get("eligible") is not True:
        raise RuntimeError(
            "full-timeline semantic sweep did not establish absence of a "
            f"purposeful action chain: {semantic_consensus.get('reason')}"
        )

    visual_audit_reference: dict[str, str] | None = None
    if semantic_consensus.get("mode") == "purposeful_action_chain_absent":
        visual_audit_path = (
            Path(config["storage"]["local_cache_root"])
            / "Collection-Catalog"
            / "Full-Timeline-Visual-Audit-Receipts"
            / f"{experiment_id}.json"
        ).resolve(strict=True)
        visual_audit = json.loads(visual_audit_path.read_text(encoding="utf-8"))
        sweep_sha256 = hashlib.sha256(sweep_path.read_bytes()).hexdigest()
        if (
            visual_audit.get("schema_version")
            != "visioncortex-full-timeline-visual-audit/1"
            or visual_audit.get("experiment_id") != experiment_id
            or int(visual_audit.get("source_copy_bytes", -1)) != 0
            or visual_audit.get("sweep_receipt_sha256") != sweep_sha256
            or visual_audit.get("verdict") != "no_purposeful_action_chain"
            or sorted(
                int(item)
                for item in visual_audit.get("reviewed_inconclusive_chunk_indices")
                or []
            )
            != sorted(
                int(item)
                for item in semantic_consensus.get("inconclusive_chunk_indices")
                or []
            )
        ):
            raise RuntimeError("independent visual audit receipt is invalid or incomplete")
        visual_audit_reference = {
            "path": str(visual_audit_path),
            "sha256": hashlib.sha256(visual_audit_path.read_bytes()).hexdigest(),
        }

    staging_root = cv_staging_root.resolve(strict=True)
    configured_staging = Path(config["storage"]["local_staging_root"]).resolve(
        strict=True
    )
    if not staging_root.is_relative_to(configured_staging):
        raise ValueError("CV adjudication root must remain below configured staging")
    json_root = staging_root / "JSON-Config-Files"
    evidence_paths = {
        "boundary_precheck": json_root / "boundary_precheck.json",
        "fine_scan_windows": json_root / "fine_scan_windows.json",
        "scan_runtime_fine": json_root / "scan_runtime_fine.json",
        "run_metrics": json_root / "run_metrics.json",
        "original_ingest": json_root / "Stage-Receipts" / "original_ingest.json",
    }
    evidence = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in evidence_paths.items()
    }
    boundary = evidence["boundary_precheck"]
    fine = evidence["fine_scan_windows"]
    scan_runtime = evidence["scan_runtime_fine"]
    metrics = evidence["run_metrics"]
    ingest = evidence["original_ingest"]
    predicted_groups = int(
        (boundary.get("experiment_boundaries") or {}).get(
            "predicted_experiment_count", -1
        )
    )
    unresolved = [
        str(item)
        for item in (
            (boundary.get("cross_view_cluster_completeness") or {}).get(
                "unresolved_candidate_ids"
            )
            or []
        )
    ]
    if predicted_groups != 0:
        raise RuntimeError("long negative adjudication requires zero CV groups")
    if any(not item.startswith("MOTION-") for item in unresolved):
        raise RuntimeError("long negative adjudication has non-motion unresolved CV evidence")
    if fine.get("strategy") != "exhaustive_all_view_full_timeline_negative_audit":
        raise RuntimeError("CV run was not configured as an exhaustive negative audit")
    eligible = {str(item) for item in fine.get("eligible_view_ids") or []}
    scanned = {str(item) for item in fine.get("selected_view_ids") or []}
    source_views = {
        str(item) for item in (sweep.get("source_view_coverage") or {}).keys()
    }
    if not source_views or eligible != source_views or scanned != source_views:
        raise RuntimeError(
            "exhaustive CV must scan every source view represented in the semantic sweep"
        )
    coverage = fine.get("coverage") or {}
    if any(
        abs(float((coverage.get(view_id) or {}).get("coverage_ratio") or 0.0) - 1.0)
        > 0.001
        for view_id in source_views
    ):
        raise RuntimeError("exhaustive CV did not cover every full physical media timeline")
    minimum_fps = float(config["performance"]["detection_fps"])
    if float(fine.get("sample_fps") or 0.0) + 1e-6 < minimum_fps:
        raise RuntimeError("exhaustive CV sample FPS is below the production setting")
    active_runtime_views = {
        str(item)
        for report in scan_runtime.get("role_reports") or []
        for item in (report.get("unique_output_timestamps_by_view") or {}).keys()
    }
    if active_runtime_views != source_views:
        raise RuntimeError("fine scan runtime lacks one or more source views")
    backends = {
        str(report.get("backend") or "")
        for report in scan_runtime.get("role_reports") or []
    }
    if backends != {"TensorRT"}:
        raise RuntimeError(f"exhaustive CV backend mismatch: {sorted(backends)}")
    if metrics.get("mllm_calls"):
        raise RuntimeError("negative CV adjudication must precede production model calls")
    if int(ingest.get("source_copy_bytes", -1)) != 0:
        raise RuntimeError("long negative adjudication source copy invariant failed")

    def reference(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    previous_bytes = receipt_path.read_bytes()
    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
    adjudication_id = (
        f"full-timeline-adjudication-{datetime.now():%Y%m%d-%H%M%S}-"
        f"{uuid.uuid4().hex[:4]}"
    )
    history_root = receipt_path.parent / "History" / experiment_id
    history_root.mkdir(parents=True, exist_ok=True)
    previous_path = history_root / (
        f"{adjudication_id}--previous-{previous_sha256[:12]}.json"
    )
    shutil.copy2(receipt_path, previous_path)
    adjudication = {
        "schema_version": "visioncortex-full-timeline-cv-adjudication/1",
        "adjudication_id": adjudication_id,
        "experiment_id": experiment_id,
        "rule": (
            "all physical source media is semantically sampled in contiguous "
            "five-minute-or-shorter chunks across every view; every Ark chunk "
            "either passes the strict negative gate or explicitly rejects a "
            "purposeful experimental action chain, any inconclusive chunks "
            "receive an independent visual audit, and every view receives a "
            "full production-FPS TensorRT scan with zero experiment groups"
        ),
        "source_copy_bytes": 0,
        "semantic_sweep_receipt": reference(sweep_path),
        "semantic_sweep_status": sweep["status"],
        "semantic_consensus": semantic_consensus,
        "independent_visual_audit": visual_audit_reference,
        "semantic_sweep_token_usage": sweep.get("ark_usage") or {},
        "recording_timeline_duration_seconds": sweep[
            "recording_timeline_duration_seconds"
        ],
        "media_timeline_duration_seconds": sweep["media_timeline_duration_seconds"],
        "exhaustive_cv": {
            "staging_root": str(staging_root),
            **{key: reference(path) for key, path in evidence_paths.items()},
            "predicted_experiment_count": predicted_groups,
            "unresolved_motion_candidate_ids": unresolved,
            "eligible_view_ids": sorted(eligible),
            "scanned_view_ids": sorted(scanned),
            "not_scanned_view_ids": [],
            "sample_fps": float(fine["sample_fps"]),
            "coverage": coverage,
            "tensorrt_backends": sorted(backends),
            "production_model_calls": 0,
        },
    }
    result = {
        **current,
        "verdict": "recording_or_hardware_test",
        "model_proposed_verdict": "full_timeline_semantic_plus_exhaustive_cv_consensus",
        "confidence": 1.0,
        "gate_reasons": [],
        "negative_exclusion_eligible": True,
        "review_backend": "deterministic-full-cv-without-remote-model",
        "full_timeline_adjudication": adjudication,
        "previous_receipt": {
            "path": str(previous_path),
            "sha256": previous_sha256,
        },
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    temporary = receipt_path.with_name(
        f".{receipt_path.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    return {**result, "receipt_path": str(receipt_path)}


def adjudicate_inconclusive_probe_with_cv(
    config: dict[str, Any],
    experiment_id: str,
    cv_staging_root: Path,
) -> dict[str, Any]:
    """Exclude a test only when independent Ark views and full CV all agree."""

    receipt_path = _receipt_path(config, experiment_id)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"content probe receipt is missing: {receipt_path}")
    current = json.loads(receipt_path.read_text(encoding="utf-8"))
    history_root = receipt_path.parent / "History" / experiment_id
    probe_paths = [
        *(sorted(history_root.glob("*.json")) if history_root.is_dir() else []),
        receipt_path,
    ]
    probes = [json.loads(path.read_text(encoding="utf-8")) for path in probe_paths]
    eligible_probes: list[tuple[Path, dict[str, Any], str]] = []
    for path, payload in zip(probe_paths, probes, strict=True):
        third_views = [
            str(view.get("view_id"))
            for view in payload.get("views") or []
            if view.get("role") == ViewRole.THIRD_PERSON.value
        ]
        model_result = payload.get("model_result") or {}
        if (
            payload.get("schema_version") == CONTENT_PROBE_SCHEMA_VERSION
            and payload.get("experiment_id") == experiment_id
            and payload.get("scope") == "classification_only"
            and payload.get("production_completion_eligible") is False
            and int(payload.get("source_copy_bytes", -1)) == 0
            and float(payload.get("timeline_duration_seconds") or 0.0) <= 300.0
            and abs(
                float(payload.get("decoded_timeline_seconds") or 0.0)
                - float(payload.get("timeline_duration_seconds") or 0.0)
            )
            <= 0.1
            and len(third_views) == 1
            and model_result.get("visible_experimental_action") is False
        ):
            eligible_probes.append((path, payload, third_views[0]))
    distinct_third_views = sorted({item[2] for item in eligible_probes})
    if len(distinct_third_views) < 2:
        raise RuntimeError(
            "CV adjudication requires two independent third-person probe views"
        )

    staging_root = cv_staging_root.resolve(strict=True)
    configured_staging = Path(config["storage"]["local_staging_root"]).resolve(
        strict=True
    )
    if not staging_root.is_relative_to(configured_staging):
        raise ValueError("CV adjudication root must remain below configured staging")
    json_root = staging_root / "JSON-Config-Files"
    boundary_path = json_root / "boundary_precheck.json"
    metrics_path = json_root / "run_metrics.json"
    ingest_path = json_root / "Stage-Receipts" / "original_ingest.json"
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ingest = json.loads(ingest_path.read_text(encoding="utf-8"))
    predicted_groups = int(
        (boundary.get("experiment_boundaries") or {}).get(
            "predicted_experiment_count", -1
        )
    )
    unresolved = [
        str(item)
        for item in (
            boundary.get("cross_view_cluster_completeness") or {}
        ).get("unresolved_candidate_ids")
        or []
    ]
    if predicted_groups != 0:
        raise RuntimeError("CV adjudication requires zero predicted experiment groups")
    if any(not item.startswith("MOTION-") for item in unresolved):
        raise RuntimeError("CV adjudication found a non-motion unresolved candidate")
    if metrics.get("mllm_calls"):
        raise RuntimeError("CV adjudication must precede all production model calls")
    if int(ingest.get("source_copy_bytes", -1)) != 0:
        raise RuntimeError("CV adjudication source copy invariant failed")

    previous_bytes = receipt_path.read_bytes()
    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
    adjudication_id = (
        f"cv-adjudication-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    )
    history_root.mkdir(parents=True, exist_ok=True)
    previous_path = history_root / (
        f"{adjudication_id}--previous-{previous_sha256[:12]}.json"
    )
    shutil.copy2(receipt_path, previous_path)
    total_usage = {
        key: sum(int((payload.get("ark_usage") or {}).get(key) or 0) for _, payload, _ in eligible_probes)
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")
    }
    adjudication = {
        "schema_version": "visioncortex-content-probe-cv-adjudication/1",
        "adjudication_id": adjudication_id,
        "rule": (
            "two distinct full-timeline cross-view Ark probes show no visible "
            "experimental action, and full all-view TensorRT produces zero "
            "formal experiment groups before any production model call"
        ),
        "distinct_third_person_views": distinct_third_views,
        "probe_receipts": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "third_view_id": third_view,
                "model_proposed_verdict": payload.get("model_proposed_verdict"),
                "confidence": payload.get("confidence"),
            }
            for path, payload, third_view in eligible_probes
        ],
        "cv_staging_root": str(staging_root),
        "boundary_precheck": str(boundary_path),
        "predicted_experiment_count": predicted_groups,
        "unresolved_motion_candidate_ids": unresolved,
        "production_model_calls": 0,
        "source_copy_bytes": 0,
        "ark_usage_across_probe_views": total_usage,
    }
    result = {
        **current,
        "verdict": "recording_or_hardware_test",
        "model_proposed_verdict": "multi_view_ark_plus_full_cv_consensus",
        "confidence": 1.0,
        "gate_reasons": [],
        "review_backend": "deterministic-full-cv-plus-independent-ark-views",
        "adjudication": adjudication,
        "previous_receipt": {
            "path": str(previous_path),
            "sha256": previous_sha256,
        },
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    temporary = receipt_path.with_name(
        f".{receipt_path.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    return {**result, "receipt_path": str(receipt_path)}

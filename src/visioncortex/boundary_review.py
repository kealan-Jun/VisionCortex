"""Review proposed cuts against original paired video before exporting clips.

A rejected CV continuity edge is a request for evidence, not proof that an
experiment finished. This pass may join neighboring evidence groups; it never
promotes rejected actions or discards the atomic segments used to build them.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .mllm import ArkAnalyzer
from .ordering import stable_group_uid
from .video_io import ViewFrameReader


BOUNDARY_PROMPT = """你审核湿实验视频中一个拟议切点是否错误地截断同一实验。
按时间阅读所有双视角画面，含切点前、间隙内和下一片段开始后的原视频。
片段编号、时间间隔和 CV 没检出动作均不是实验结束或新实验开始的证据。
必须区分“一个操作结束”和“整个实验结束”。打开/折整称量纸、放入天平、
取试剂、加样、读数可能是同一称量实验的连续步骤，不能按这些步骤各建一场实验。
这仅是说明粒度的例子，不能据此补写画面没有发生的步骤或默认此视频正在称量。
同一人、同一台面、同一设备静止出现也不足以证明连续。需要直接指出被操作对象、
未完成操作或准备到执行的可见承接。如果出现另一实验员、另一实验台、新样品任务，
或前一任务明确完成后另起任务，输出 different_experiment。证据不足输出 uncertain。
不要引用算法的既有划分、化学常识或预设流程作为证据。图片编号必须来自输入。
只输出 JSON：
{"relation":"same_experiment/different_experiment/uncertain",
"basis":"ongoing_operation/preparation_to_execution/sample_handoff/new_workflow/insufficient_evidence",
"same_operator_and_workstation":true,
"left_experiment_complete":false,
"before_frame_ids":["F001"],"after_frame_ids":["F010"],
"observation":"写明切点两侧真实可见的操作承接或终止依据",
"confidence":0.0,"uncertainties":[]}
"""


class BoundaryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: Literal["same_experiment", "different_experiment", "uncertain"]
    basis: Literal["ongoing_operation", "preparation_to_execution", "sample_handoff",
                   "new_workflow", "insufficient_evidence"]
    same_operator_and_workstation: bool
    left_experiment_complete: bool
    before_frame_ids: list[str]
    after_frame_ids: list[str]
    observation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str]


def review_times(left, right, *, context_seconds=12.0, maximum_pairs=20):
    """Bound the semantic review, never the duration of the experiment itself."""
    start = max(left.global_start_ms, left.global_end_ms - context_seconds * 1000)
    end = min(right.global_end_ms, right.global_start_ms + context_seconds * 1000)
    if end <= start or maximum_pairs < 6:
        raise ValueError("Boundary review needs a positive span and at least six pairs")
    # Always inspect both cut edges and their immediate neighboring frames;
    # spend the remaining budget uniformly across the previously omitted gap.
    anchors = {start, end, left.global_end_ms, right.global_start_ms,
               max(start, left.global_end_ms - 1000),
               min(end, right.global_start_ms + 1000)}
    candidates = np.linspace(start, end, maximum_pairs * 2).tolist()
    while len(anchors) < maximum_pairs and candidates:
        value = max(candidates, key=lambda x: min(abs(x - a) for a in anchors))
        anchors.add(value)
        candidates.remove(value)
    return sorted(anchors)


def join_allowed(left, right, result, frames, *, minimum_confidence=0.85):
    if result.get("status") != "completed":
        return False
    try:
        decision = BoundaryDecision.model_validate(result["decision"])
    except (KeyError, ValueError):
        return False
    if (left.first_person_view, left.third_person_view) != (
        right.first_person_view, right.third_person_view
    ):
        return False
    if not (
        decision.relation == "same_experiment"
        and decision.basis in {"ongoing_operation", "preparation_to_execution", "sample_handoff"}
        and decision.same_operator_and_workstation
        and not decision.left_experiment_complete
        and decision.confidence >= minimum_confidence
    ):
        return False
    by_id = {f["frame_id"]: f for f in frames}
    before = decision.before_frame_ids
    after = decision.after_frame_ids
    return bool(
        before and after
        and all(i in by_id and by_id[i]["global_ms"] <= left.global_end_ms for i in before)
        and all(i in by_id and by_id[i]["global_ms"] > left.global_end_ms for i in after)
        and any(by_id[i]["global_ms"] >= right.global_start_ms for i in after)
    )


def reconcile_groups(groups, segments, events, reviews):
    """Apply only validated edges and retain every original event membership."""
    by_segment = {s.segment_id: s for s in segments}
    by_event = {e.event_id: e for e in events}
    edges = {(r["left_group_uid"], r["right_group_uid"]): r for r in reviews}
    chains = []
    for group in groups:
        if chains:
            left = chains[-1][-1]
            review = edges.get((left.group_uid or left.group_id, group.group_uid or group.group_id))
            if review and join_allowed(left, group, review["result"], review["frames"],
                                       minimum_confidence=review["minimum_confidence"]):
                chains[-1].append(group)
                continue
        chains.append([group])
    result = []
    for index, chain in enumerate(chains, 1):
        group = chain[0].model_copy(deep=True)
        old_ids = {g.group_uid or g.group_id for g in chain}
        group.boundary_reviews = [r for r in reviews if r["left_group_uid"] in old_ids]
        if len(chain) > 1:
            atomic_ids = list(dict.fromkeys(sid for g in chain for sid in g.atomic_experiment_ids))
            atomic = [by_segment[sid] for sid in atomic_ids]
            group.atomic_experiment_ids = atomic_ids
            group.group_uid = stable_group_uid(atomic, by_event)
            group.global_start_ms = min(g.global_start_ms for g in chain)
            group.global_end_ms = max(g.global_end_ms for g in chain)
            group.participating_views = sorted({v for g in chain for v in g.participating_views})
            group.key_event_ids = list(dict.fromkeys(e for g in chain for e in g.key_event_ids))
            group.continuity_type = "continuous"
            group.continuity_reason = "原视频切点前后存在同一实验操作承接；原子片段与动作证据保留"
            # Every dependent name, narrative and media path must be regenerated.
            group.model_understanding = None
            group.archive_folder = None
            group.videos = {}
            group.video_json = {}
            group.experiment_name = "待模型命名实验"
            group.experiment_name_en = "Unnamed-Experiment"
        group.group_id = f"GROUP-{index:04d}"
        for sid in group.atomic_experiment_ids:
            by_segment[sid].group_id = group.group_id
        result.append(group)
    return result


def review_experiment_boundaries(layout, groups, segments, events, views, infos,
                                 transforms, config):
    """Run an auditable original-video review of neighboring same-pair cuts."""
    from .archive import (
        _read_semantic_cache, _semantic_cache_path, _semantic_cache_reads_enabled,
        _semantic_fingerprint, _write_semantic_cache, write_json,
    )

    cfg = config.get("mllm", {}).get("boundary_review") or {}
    if not cfg.get("enabled", False) or len(groups) < 2:
        return list(groups)
    maximum_pairs = max(6, min(32, int(cfg.get("maximum_pairs", 20))))
    maximum_gap_ms = max(0.0, float(cfg.get("maximum_gap_seconds", 120))) * 1000
    minimum_confidence = max(0.85, float(cfg.get("minimum_confidence", 0.85)))
    reviews = []
    by_view = {v.view_id: v for v in views}
    before_groups = [g.model_dump(mode="json") for g in groups]
    started = time.perf_counter()
    analyzer = ArkAnalyzer(config)
    try:
        for left, right in zip(groups, groups[1:], strict=False):
            gap = right.global_start_ms - left.global_end_ms
            if not 0 <= gap <= maximum_gap_ms or (
                left.first_person_view, left.third_person_view
            ) != (right.first_person_view, right.third_person_view):
                continue
            identity = {"left_group_uid": left.group_uid or left.group_id,
                        "right_group_uid": right.group_uid or right.group_id}
            edge_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            directory = layout.work / "boundary-review" / edge_id
            directory.mkdir(parents=True, exist_ok=True)
            times = review_times(left, right, maximum_pairs=maximum_pairs,
                                 context_seconds=float(cfg.get("context_seconds", 12)))
            frames, images = [], []
            with ViewFrameReader(max_open=2) as reader:
                for index, global_ms in enumerate(times, 1):
                    rendered, sources = [], []
                    for view_id in (left.first_person_view, left.third_person_view):
                        local_ms = transforms[view_id].to_local(global_ms)
                        if not 0 <= local_ms <= infos[view_id].duration_ms:
                            break
                        frame = reader.read(by_view[view_id], infos[view_id], local_ms)
                        if frame is None:
                            break
                        height = min(480, frame.shape[0])
                        rendered.append(cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height)))
                        sources.append({"view_id": view_id, "local_ms": local_ms,
                                        "decoded_pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
                    if len(rendered) != 2:
                        continue
                    height = min(f.shape[0] for f in rendered)
                    rendered = [cv2.resize(f, (round(f.shape[1] * height / f.shape[0]), height)) for f in rendered]
                    path = directory / f"F{index:03d}.jpg"
                    if not cv2.imwrite(str(path), np.hstack(rendered), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise OSError("Cannot retain boundary review frame")
                    record = {"frame_id": path.stem, "global_ms": global_ms,
                              "image": str(path.relative_to(layout.root)),
                              "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sources": sources}
                    frames.append(record)
                    images.append((f"{path.stem}; t={global_ms:.3f}ms; left=first_person; right=third_person", path))
            metadata = {"proposed_left_end_ms": left.global_end_ms,
                        "proposed_right_start_ms": right.global_start_ms,
                        "frames": [{k: f[k] for k in ("frame_id", "global_ms")} for f in frames],
                        "task": "判断拟议切点两侧是否是同一未完成实验的连续操作，不推断原片之外的完成状态"}
            fingerprint = _semantic_fingerprint("experiment-boundary", config, BOUNDARY_PROMPT, metadata, images)
            cache_path = _semantic_cache_path(config, "experiment-boundaries", fingerprint)
            result = _read_semantic_cache(cache_path, fingerprint) if _semantic_cache_reads_enabled(config) else None
            if result is None:
                if len(frames) != len(times):
                    result = {"status": "insufficient_frames", "uncertainties": ["切点原始双视角画面不完整"]}
                else:
                    raw = analyzer._call(BOUNDARY_PROMPT, metadata, images, max_images=maximum_pairs)
                    result = dict(raw)
                    if raw.get("status") == "completed":
                        try:
                            payload = {k: raw[k] for k in BoundaryDecision.model_fields}
                            result["decision"] = BoundaryDecision.model_validate(payload).model_dump(mode="json")
                        except (KeyError, ValueError):
                            result["status"] = "invalid_boundary_response"
                        else:
                            result = _write_semantic_cache(cache_path, fingerprint, result)
            reviews.append({**identity, "review_id": edge_id,
                            "left_end_ms": left.global_end_ms, "right_start_ms": right.global_start_ms,
                            "minimum_confidence": minimum_confidence, "frames": frames,
                            "input_fingerprint": fingerprint, "result": result,
                            "joined": join_allowed(left, right, result, frames, minimum_confidence=minimum_confidence)})
    finally:
        analyzer.close()
        write_json(layout.json_config / "experiment_boundary_review.json", {
            "schema_version": "visioncortex-experiment-boundary-review/1",
            "policy": "paired original-video continuity review; never physical-action confirmation",
            "original_groups": before_groups, "reviews": reviews,
            "wall_seconds": round(time.perf_counter() - started, 6),
            "joined_edge_count": sum(r["joined"] for r in reviews),
            "quality_classification": "PARTIAL_EVIDENCE",
            "formal_action_membership_changed": False,
        })
    return reconcile_groups(groups, segments, events, reviews)

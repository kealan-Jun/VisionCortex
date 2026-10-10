"""Review proposed cuts against original paired video before exporting clips.

A rejected CV continuity edge is a request for evidence, not proof that an
experiment finished. This pass may join neighboring evidence groups; it never
promotes rejected actions or discards the atomic segments used to build them.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .mllm import ArkAnalyzer
from .ordering import stable_group_uid
from .video_io import ViewFrameReader


BOUNDARY_PROMPT = """审核湿实验视频中相邻片段的关系。按时间看切点两侧和间隙内原画面。
必须区分：同一实验的不同操作（same_experiment）、有实际样品/器具/操作承接的
不同实验单元（continuous_workflow）、无承接的独立实验（different_experiment）。
例如称量→配液→携带该溶液换台→移液，可以是一个连续实验链中的多个实验单元。
这些只是粒度示例，不得套用到缺少画面证据的视频。同一人或同一台面本身不是连续依据；
换台、换摄像头、时间空隙、CV 无动作也不是实验结束的证据。优先跟踪实际被操作的
样品、容器、器具和操作状态。不能依据试剂颜色推断其成分。无法证明承接输出 uncertain。
附带的操作记录只是待核对线索，不是本次画面的事实。录像文件切片边界不是实验边界。
承接时在 object_links 记录实际被操作对象的前后状态和两侧画面；跨台时还要引用携带该对象
的 handoff_frame_ids。仅外观相似、同类检测标签或同一背景物体不能证明同一对象承接。
object_links 只列 1–3 个真正跨越候选切点的操作对象，不列右段才开始使用的背景物品。
所有 before_frame_ids 的时间必须 <= proposed_left_end_ms，after_frame_ids 的时间必须
> proposed_left_end_ms，且每个对象至少引用一张 >= proposed_right_start_ms 的后侧图片。
这里的 before/after 指候选切点两侧，不是任意单步动作的前后。不能满足就输出 uncertain。
same_experiment 需要前一个实验尚未完成；continuous_workflow 允许其中一个单元已完成，
但需可见的实物或操作承接。跨台时需引用移动或携带实物的间隙画面。
left_unit_name/right_unit_name 使用具体实验操作名称，不使用场景名或检测标签。
不能把原片段边界当作单元真正的完成时间。只输出 JSON：
{"relation":"same_experiment/continuous_workflow/different_experiment/uncertain",
"basis":"ongoing_operation/preparation_to_execution/sample_handoff/new_workflow/insufficient_evidence",
"same_operator":true,"workstation_changed":false,"left_experiment_complete":false,
"left_unit_name":"画面支持的实验单元名称","right_unit_name":"画面支持的实验单元名称",
"before_frame_ids":["F001"],"after_frame_ids":["F010"],"handoff_frame_ids":[],
"object_links":[{"object_description":"实际操作对象","before_state":"此前可见状态",
"after_state":"后续可见状态","before_frame_ids":["F001"],"after_frame_ids":["F010"],
"handoff_frame_ids":[]}],
"observation":"具体可见的承接或结束依据","confidence":0.0,"uncertainties":[]}
"""

TAIL_PROMPT = """检查一个湿实验片段末尾之后的原视频，避免将检测终点或录像终点当作实验结束。
所有图片按全局时间排列。请跟踪同一操作任务的样品、器具、操作状态，区分单步动作、
实验单元和整个连续实验链。称量结束后继续配液或携带同一容器换台移液，不是连续链结束。
同一人仍在场或背景设备不变不能独立证明承接。单凭停手、遮挡、没有检测事件也不能确认结束。
只在有明确可见任务完成依据时输出 completed；独立无关新任务开始输出 different_workflow。
尚在承接操作输出 ongoing；无法判断输出 uncertain。不得猜测录像之外发生的事情。
在 ongoing 时需引用最后一张图并说明它与之前操作的承接。完成时引用明确结束的画面，
不能引用采样终点作为完成理由。frame_ids 必须同时引用提议终点及之前的依据和后续依据。
单步完成（operation）或单个实验单元完成（experiment_unit）不能作为整个连续链完成。
completed 必须 completion_scope=workflow，且结束画面之后至少两张画面（含最后一张）
未见相关操作继续，覆盖至少 completion_followup_ms 的时间；引用 post_completion_frame_ids。
这只是对已观察范围的判断，不是保证以后不再操作。只看到最后一帧结束或缺少后续画面，输出 uncertain。
结束后又配液、携带原容器或继续移液，post_completion_state=related_continuation，不能输出 completed。
短暂停手、等候或遮挡后同一任务继续应为 ongoing；无法从画面建立承接时仍为 uncertain。
same_operator 需要画面支持，不能仅凭第一人称摄像头编号确认操作者未变。
ongoing 和 completed 均在 object_links 引用同一被操作对象跨 continuity_anchor_ms 的前后状态：
before_frame_ids <= continuity_anchor_ms，after_frame_ids > continuity_anchor_ms。
ongoing 至少一个对象引用最后一张图；completed 至少一个对象引用 completion_frame_id。
跨台时对象还须引用实际携带过程的 handoff_frame_ids。所有对象引用都须包含在 frame_ids 中。
另行判断所有后续画面是否仍在同一实验台、第三人称画面能否对应第一人称的操作对象与操作过程。
只有实际可见对应才将 third_person_continuity_observed 设为 true；同一时间或文件机位名不足以确认。
移动、遮挡、机位对应不清时将它设为 false，保留第一人称。只输出 JSON：
{"state":"ongoing/completed/different_workflow/uncertain","continuation_observed":true,
"same_operator":true,"same_workstation":true,"third_person_continuity_observed":false,
"frame_ids":["F001","F020"],"last_continuation_frame_id":"F020",
"completion_frame_id":null,"completion_scope":"none/operation/experiment_unit/workflow",
"post_completion_state":"not_observed/related_continuation/no_related_continuation/uncertain",
"post_completion_frame_ids":[],"object_links":[{"object_description":"实际操作对象",
"before_state":"此前状态","after_state":"后续状态","before_frame_ids":["F001"],
"after_frame_ids":["F020"],"handoff_frame_ids":[]}],
"observation":"具体画面依据","confidence":0.0,"uncertainties":[]}
"""


class ObjectContinuation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_description: str = Field(min_length=1, max_length=300)
    before_state: str = Field(min_length=1, max_length=600)
    after_state: str = Field(min_length=1, max_length=600)
    before_frame_ids: list[str] = Field(min_length=1)
    after_frame_ids: list[str] = Field(min_length=1)
    handoff_frame_ids: list[str]


class BoundaryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relation: Literal["same_experiment", "continuous_workflow", "different_experiment", "uncertain"]
    basis: Literal["ongoing_operation", "preparation_to_execution", "sample_handoff",
                   "new_workflow", "insufficient_evidence"]
    same_operator: bool
    workstation_changed: bool
    left_experiment_complete: bool
    left_unit_name: str = Field(min_length=1)
    right_unit_name: str = Field(min_length=1)
    before_frame_ids: list[str]
    after_frame_ids: list[str]
    handoff_frame_ids: list[str]
    # Legacy receipts remain readable; every new v3 review must supply links.
    object_links: list[ObjectContinuation] = Field(default_factory=list, max_length=8)
    observation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str]


class TailDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["ongoing", "completed", "different_workflow", "uncertain"]
    continuation_observed: bool
    same_operator: bool = False
    same_workstation: bool | None = None
    third_person_continuity_observed: bool = False
    frame_ids: list[str]
    last_continuation_frame_id: str | None
    completion_frame_id: str | None
    completion_scope: Literal["none", "operation", "experiment_unit", "workflow"] = "none"
    post_completion_state: Literal[
        "not_observed", "related_continuation", "no_related_continuation", "uncertain"
    ] = "not_observed"
    post_completion_frame_ids: list[str] = Field(default_factory=list)
    object_links: list[ObjectContinuation] = Field(default_factory=list, max_length=8)
    observation: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str]


def third_person_at(group, global_ms):
    if not group.view_timeline:
        return group.third_person_view
    for item in group.view_timeline:
        if item["start_ms"] <= global_ms < item["end_ms"]:
            return item.get("third_person_view")
    return None


def review_times(left, right, *, context_seconds=12.0, maximum_pairs=20, required_times=()):
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
    anchors.update(t for t in required_times if start <= t <= end)
    if len(anchors) > maximum_pairs:
        raise ValueError("Required recording boundaries exceed review frame budget")
    candidates = np.linspace(start, end, maximum_pairs * 2).tolist()
    while len(anchors) < maximum_pairs and candidates:
        value = max(candidates, key=lambda x: min(abs(x - a) for a in anchors))
        anchors.add(value)
        candidates.remove(value)
    return sorted(anchors)


def recording_seams(view_ids, infos, transforms, start, end):
    """Use physical segment clocks, including pauses, rather than a 15-min guess."""
    seams = []
    for view_id in view_ids:
        parts = getattr(infos.get(view_id), "segments", [])
        if view_id not in transforms:
            continue
        for index, (left, right) in enumerate(zip(parts, parts[1:], strict=False)):
            before = transforms[view_id].to_global(max(left.virtual_start_ms, left.virtual_end_ms - 250))
            after = transforms[view_id].to_global(min(right.virtual_end_ms - 1, right.virtual_start_ms + 250))
            if start <= before < after <= end:
                seams.append({"view_id": view_id, "left_segment_index": index,
                              "right_segment_index": index + 1, "before_ms": before, "after_ms": after})
    return seams


def operation_context(group, events, side):
    """Bounded, explicitly non-authoritative clues; never send CV labels as proof."""
    by_id = {e.event_id: e for e in events}
    selected = sorted((by_id[i] for i in group.key_event_ids if i in by_id),
                      key=lambda e: e.global_start_ms)
    selected = selected[-3:] if side == "end" else selected[:3]
    return {
        "first_person_route": group.first_person_view,
        "operator_identity_verified": False,
        "context_is_evidence": False,
        "completion_status": group.completion_status,
        "workflow_units": group.workflow_units[-2:] if side == "end" else group.workflow_units[:2],
        "operation_clues": [{"event_id": e.event_id, "start_ms": e.global_start_ms,
                             "end_ms": e.global_end_ms, "model_observation":
                             (e.model_understanding or {}).get("current_step", "")}
                            for e in selected],
    }


def seam_review_times(times, seams, metadata):
    anchors = {min(times), max(times)}
    anchors.update(s[k] for s in seams for k in ("before_ms", "after_ms"))
    anchors.update(metadata[k] for k in ("proposed_left_end_ms", "proposed_right_start_ms", "proposed_end_ms", "continuity_anchor_ms", "previous_review_end_ms")
                   if k in metadata and min(times) <= metadata[k] <= max(times))
    budget = min(32, max(6, len(times)))
    if len(anchors) > budget:
        raise ValueError("Required recording boundaries exceed review frame budget")
    candidates = set(times) - anchors
    while len(anchors) < budget and candidates:
        value = max(candidates, key=lambda x: (min(abs(x - a) for a in anchors), x))
        anchors.add(value)
        candidates.remove(value)
    return sorted(anchors)


def workflow_tracking(groups, infos, transforms):
    """Persist continuation and unresolved next-recording needs per wearable."""
    tracks, previous = [], {}
    for g in sorted(groups, key=lambda g: (g.global_start_ms, g.group_id)):
        prior = previous.get(g.first_person_view)
        links = [r for r in g.boundary_reviews if r.get("joined") and not r.get("superseded_by")]
        tracks.append({
            "group_uid": g.group_uid or g.group_id,
            "first_person_route": g.first_person_view,
            "operator_identity_is_camera_only": True,
            "workflow_kind": g.workflow_kind,
            "workflow_units": g.workflow_units,
            "start_ms": g.global_start_ms, "end_ms": g.global_end_ms,
            "completion_status": g.completion_status,
            "awaiting_next_recording": g.completion_status == "ongoing_at_recording_end",
            "previous_unresolved_group_uid": (prior.group_uid or prior.group_id)
                if prior and prior.completion_status != "observed_complete" else None,
            "recording_seams": recording_seams(g.participating_views, infos, transforms,
                                               g.global_start_ms, g.global_end_ms),
            "continuations": [{"review_id": r.get("review_id"),
                               "relation": r["result"]["decision"]["relation"],
                               "object_links": r["result"]["decision"].get("object_links", []),
                               "evidence_version": r["result"].get("continuity_evidence_version", 2)}
                              for r in links],
            "cross_task_stitching_applied": False,
        })
        previous[g.first_person_view] = g
    return tracks


def boundary_third_person(group, side):
    rows = group.view_timeline if side == "start" else reversed(group.view_timeline)
    return next((r["third_person_view"] for r in rows if r.get("third_person_view")), group.third_person_view)


def join_allowed(left, right, result, frames, *, minimum_confidence=0.85):
    if result.get("status") != "completed":
        return False
    try:
        decision = BoundaryDecision.model_validate(result["decision"])
    except (KeyError, ValueError):
        return False
    # A wearable is a routing constraint, never sufficient identity evidence.
    if left.first_person_view != right.first_person_view:
        return False
    if not (
        decision.relation in {"same_experiment", "continuous_workflow"}
        and decision.basis in {"ongoing_operation", "preparation_to_execution", "sample_handoff"}
        and decision.same_operator
        and (decision.relation == "continuous_workflow" or not decision.left_experiment_complete)
        and decision.confidence >= minimum_confidence
    ):
        return False
    cross_station = (decision.workstation_changed or boundary_third_person(left, "end") != boundary_third_person(right, "start"))
    if cross_station:
        valid_handoff = {f["frame_id"] for f in frames
                         if left.global_end_ms <= f["global_ms"] <= right.global_end_ms}
        if (decision.basis != "sample_handoff" or not decision.handoff_frame_ids
                or not set(decision.handoff_frame_ids) <= valid_handoff):
            return False
    by_id = {f["frame_id"]: f for f in frames}
    if result.get("continuity_evidence_version", 2) >= 3:
        if not decision.object_links:
            return False
        for link in decision.object_links:
            if (not all(i in by_id and by_id[i]["global_ms"] <= left.global_end_ms
                        for i in link.before_frame_ids)
                    or not all(i in by_id and by_id[i]["global_ms"] > left.global_end_ms
                               for i in link.after_frame_ids)):
                return False
            if not any(i in by_id and by_id[i]["global_ms"] >= right.global_start_ms
                       for i in link.after_frame_ids):
                return False
            if cross_station and (not link.handoff_frame_ids
                    or not set(link.handoff_frame_ids) <= valid_handoff):
                return False
            if not set(link.handoff_frame_ids) <= by_id.keys():
                return False
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
        preceding = next((chain for chain in reversed(chains)
                          if chain[-1].first_person_view == group.first_person_view), None)
        if preceding:
            left = preceding[-1]
            review = edges.get((left.group_uid or left.group_id, group.group_uid or group.group_id))
            if review and join_allowed(left, group, review["result"], review["frames"],
                                       minimum_confidence=review["minimum_confidence"]):
                preceding.append(group)
                continue
        chains.append([group])
    result = []
    for index, chain in enumerate(chains, 1):
        group = chain[0].model_copy(deep=True)
        group.source_archive_folders = list(dict.fromkeys(
            folder for source in chain for folder in
            [*source.source_archive_folders, source.archive_folder] if folder))
        old_ids = {g.group_uid or g.group_id for g in chain}
        combined_reviews = [r for source in chain for r in source.boundary_reviews]
        combined_reviews.extend(r for r in reviews if r["left_group_uid"] in old_ids)
        group.boundary_reviews = list({r.get("input_fingerprint") or r.get("review_id")
                                       or str((r["left_group_uid"], r["right_group_uid"])): r
                                       for r in combined_reviews}.values())
        group.completion_status = chain[-1].completion_status
        group.completion_reason = chain[-1].completion_reason
        group.boundary_extension_requires_step_review = any(
            g.boundary_extension_requires_step_review for g in chain)
        if group.completion_status == "unreviewed":
            group.completion_status = "unresolved"
            group.completion_reason = "尚未确认整个实验链的结束"
        group.view_timeline = []
        group.workflow_units = []
        for position, source in enumerate(chain):
            edge = edges.get((chain[position - 1].group_uid or chain[position - 1].group_id,
                              source.group_uid or source.group_id)) if position else None
            decision = (edge or {}).get("result", {}).get("decision", {})
            if position and chain[position - 1].global_end_ms < source.global_start_ms:
                prior_view = boundary_third_person(chain[position - 1], "end")
                next_view = boundary_third_person(source, "start")
                gap_start = chain[position - 1].global_end_ms
                if decision.get("workstation_changed") and decision.get("basis") == "sample_handoff":
                    referenced = {f["frame_id"]: f["global_ms"] for f in edge["frames"]}
                    handoff = [referenced[i] for i in decision.get("handoff_frame_ids", []) if i in referenced]
                    switch = min(source.global_start_ms, min(handoff, default=gap_start))
                    if switch > gap_start:
                        group.view_timeline.append({"start_ms": gap_start, "end_ms": switch,
                            "third_person_view": prior_view, "reason": "reviewed_workstation_until_handoff"})
                        gap_start = switch
                if gap_start < source.global_start_ms:
                    group.view_timeline.append({"start_ms": gap_start,
                    "end_ms": source.global_start_ms,
                    "third_person_view": next_view if prior_view == next_view
                        and not decision.get("workstation_changed") else None,
                    "reason": "reviewed_continuation_gap"})
            group.view_timeline.extend(source.view_timeline or [{
                "start_ms": source.global_start_ms, "end_ms": source.global_end_ms,
                "third_person_view": source.third_person_view, "reason": "source_group_view_correspondence"}])
            units = [dict(u) for u in source.workflow_units] or [{
                "name": source.experiment_name, "start_ms": source.global_start_ms,
                "end_ms": source.global_end_ms, "source_group_uids": [source.group_uid or source.group_id],
                "completion_observed": False, "boundary_precision": "candidate_span_not_exact_completion"}]
            if group.workflow_units and decision.get("relation") == "same_experiment":
                previous_unit = group.workflow_units[-1]
                previous_unit["end_ms"] = units[0]["end_ms"]
                previous_unit["source_group_uids"] = list(dict.fromkeys(
                    previous_unit["source_group_uids"] + units[0]["source_group_uids"]))
                previous_unit["completion_observed"] = units[0]["completion_observed"]
                if not source.workflow_units:
                    previous_unit["name"] = decision["right_unit_name"]
                group.workflow_units.extend(units[1:])
            else:
                if group.workflow_units and decision.get("relation") == "continuous_workflow":
                    previous_unit = group.workflow_units[-1]
                    referenced = {f["frame_id"]: f["global_ms"] for f in edge["frames"]}
                    handoff = [referenced[i] for i in decision.get("handoff_frame_ids", []) if i in referenced]
                    transition_ms = min(handoff) if handoff else source.global_start_ms
                    transition_ms = max(previous_unit["start_ms"], min(transition_ms, units[0]["end_ms"]))
                    previous_unit.update(name=decision["left_unit_name"], end_ms=transition_ms,
                                         completion_observed=decision["left_experiment_complete"])
                    if handoff:
                        previous_unit["transition_interval_ms"] = [min(handoff), max(handoff)]
                    units[0].update(name=decision["right_unit_name"], start_ms=transition_ms)
                group.workflow_units.extend(units)
        for ordinal, unit in enumerate(group.workflow_units, 1):
            unit["unit_id"] = f"UNIT-{ordinal:03d}"
        if len(group.workflow_units) > 1:
            group.workflow_kind = "continuous_workflow"
        if len(chain) > 1:
            atomic_ids = list(dict.fromkeys(sid for g in chain for sid in g.atomic_experiment_ids))
            group.atomic_experiment_ids = atomic_ids
            group.group_uid = stable_group_uid([by_segment[sid] for sid in atomic_ids], by_event)
            group.global_start_ms = min(g.global_start_ms for g in chain)
            group.global_end_ms = max(g.global_end_ms for g in chain)
            group.participating_views = sorted({v for g in chain for v in g.participating_views})
            group.key_event_ids = list(dict.fromkeys(e for g in chain for e in g.key_event_ids))
            group.continuity_type = "continuous"
            group.continuity_reason = "原视频支持操作或实物承接；实验单元、机位变化与原始动作证据分别保留"
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


def apply_semantic_units(group, understanding):
    """Name semantic units from the full storyboard without altering CV membership."""
    if understanding.get("status") != "completed":
        return False
    candidates = understanding.get("atomic_experiments") or []
    if not candidates or (group.workflow_kind == "continuous_workflow" and len(candidates) < 2):
        return False
    units = []
    source_uids = list(dict.fromkeys(uid for unit in group.workflow_units
                                    for uid in unit.get("source_group_uids", [])))
    for item in candidates:
        try:
            start, end = float(item["start_global_ms"]), float(item["end_global_ms"])
            name = str(item["name"]).strip()
        except (KeyError, ValueError, TypeError):
            return False
        # Models may round sub-millisecond source timestamps to integers.
        if abs(start - group.global_start_ms) <= 1:
            start = group.global_start_ms
        if abs(end - group.global_end_ms) <= 1:
            end = group.global_end_ms
        if (not name or not math.isfinite(start) or not math.isfinite(end)
                or start < group.global_start_ms or end > group.global_end_ms or end <= start
                or (units and start < units[-1]["end_ms"])):
            return False
        units.append({"unit_id": f"UNIT-{len(units) + 1:03d}", "name": name,
                      "start_ms": start, "end_ms": end, "source_group_uids": source_uids,
                      "completion_observed": False,
                      "boundary_precision": "model_semantic_scope_not_human_verified",
                      "purpose_observable": item.get("purpose_observable"),
                      "source": "experiment_group_understanding"})
    group.workflow_units = units
    if len(units) > 1 and understanding.get("continuity_type_confirmed") == "continuous":
        group.workflow_kind = "continuous_workflow"
        group.continuity_type = "continuous"
        group.continuity_reason = "原视频模型理解辨识出多个承接的实验单元；保留模型范围与边界复核记录"
    return True


def _review(layout, analyzer, config, view_ids, views, infos, transforms, times,
            metadata, prompt, schema, identity):
    from .archive import (
        _read_semantic_cache, _semantic_cache_path, _semantic_cache_reads_enabled,
        _semantic_fingerprint, _write_semantic_cache,
    )
    version = 4 if schema is TailDecision else 3
    if version == 4:
        identity = {**identity, "tail_evidence_version": version}
    edge_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    seams = recording_seams(view_ids, infos, transforms, min(times), max(times))
    metadata = {**metadata, "recording_seams": seams,
                "recording_cut_is_experiment_completion": False}
    try:
        times = seam_review_times(times, seams, metadata)
    except ValueError:
        return {**identity, "review_id": edge_id, "frames": [], "request_metadata": metadata,
                "result": {"status": "insufficient_boundary_coverage", "continuity_evidence_version": version}}
    directory = layout.json_config / "Boundary-Review-Frames" / edge_id
    directory.mkdir(parents=True, exist_ok=True)
    frames, images = [], []
    by_view = {v.view_id: v for v in views}
    with ViewFrameReader(max_open=len(view_ids)) as reader:
        for index, global_ms in enumerate(times, 1):
            rendered, sources = [], []
            for view_id in view_ids:
                local_ms = transforms[view_id].to_local(global_ms)
                if not 0 <= local_ms < infos[view_id].duration_ms:
                    continue
                frame = reader.read(by_view[view_id], infos[view_id], local_ms)
                if frame is None:
                    continue
                height = 720 if len(view_ids) == 1 else 360
                resized = cv2.resize(frame, (round(frame.shape[1] * height / frame.shape[0]), height))
                cv2.putText(resized, view_id, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
                rendered.append(resized)
                sources.append({"view_id": view_id, "local_ms": local_ms,
                                "decoded_pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            # A missing wearable frame prevents a continuity decision. Other
            # viewpoints may legitimately end or be absent during a transfer.
            if not sources or sources[0]["view_id"] != view_ids[0]:
                continue
            path = directory / f"F{index:03d}.jpg"
            if not cv2.imwrite(str(path), np.hstack(rendered), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError("Cannot retain boundary review frame")
            record = {"frame_id": path.stem, "global_ms": global_ms,
                      "image": str(path.relative_to(layout.root)),
                      "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sources": sources}
            frames.append(record)
            images.append((f"{path.stem}; t={global_ms:.3f}ms; columns="
                           + ",".join(source["view_id"] for source in sources), path))
    metadata = {**metadata, "frames": [
        {"frame_id": f["frame_id"], "global_ms": f["global_ms"],
         "views": [s["view_id"] for s in f["sources"]]} for f in frames]}
    fingerprint = _semantic_fingerprint(f"experiment-boundary-v{version}", config, prompt, metadata, images)
    cache_path = _semantic_cache_path(config, "experiment-boundaries", fingerprint)
    result = _read_semantic_cache(cache_path, fingerprint) if _semantic_cache_reads_enabled(config) else None
    if result is None:
        seam_coverage = all(any(f["global_ms"] == seam[key]
                               and seam["view_id"] in {s["view_id"] for s in f["sources"]}
                               for f in frames)
                            for seam in seams for key in ("before_ms", "after_ms"))
        if len(frames) != len(times) or not seam_coverage:
            result = {"status": "insufficient_frames"}
        else:
            result = dict(analyzer._call(prompt, metadata, images, max_images=len(images)))
            if result.get("status") == "completed":
                try:
                    result["decision"] = schema.model_validate(
                        {k: result[k] for k in schema.model_fields}).model_dump(mode="json")
                except (KeyError, ValueError):
                    result["status"] = "invalid_boundary_response"
                else:
                    result = _write_semantic_cache(cache_path, fingerprint, result)
    result = {**result, "continuity_evidence_version": version}
    return {**identity, "review_id": edge_id, "frames": frames, "request_metadata": metadata,
            "input_fingerprint": fingerprint, "result": result}


def refine_neighbor_review(layout, analyzer, config, left, right, record, views, infos, transforms):
    """One bounded escalation with denser wearable images, never a lower threshold."""
    decision = record["result"].get("decision") or {}
    if record.get("joined") or decision.get("relation") not in {"same_experiment", "continuous_workflow"}:
        return None
    frames = {f["frame_id"]: f["global_ms"] for f in record["frames"]}
    focus = [frames[i] for i in decision.get("handoff_frame_ids", []) if i in frames]
    if not focus:
        focus = [left.global_end_ms, right.global_start_ms]
    lo = max(left.global_start_ms, min(focus) - 8000)
    hi = min(right.global_end_ms, max(focus) + 8000)
    anchors = review_times(left, right, maximum_pairs=8)
    times = sorted(set(anchors + np.linspace(lo, hi, 24).tolist()))
    identity = {"left_group_uid": record["left_group_uid"], "right_group_uid": record["right_group_uid"],
                "kind": "neighbor", "refinement_of": record["review_id"]}
    detailed = _review(layout, analyzer, config, [left.first_person_view], views, infos, transforms, times,
                       {"proposed_left_end_ms": left.global_end_ms,
                        "proposed_right_start_ms": right.global_start_ms,
                        "left_context": record.get("request_metadata", {}).get("left_context", {}),
                        "right_context": record.get("request_metadata", {}).get("right_context", {}),
                        "scope": "高密度第一人称实物承接复核；不宣称双视角确认"},
                       BOUNDARY_PROMPT, BoundaryDecision, identity)
    detailed.update(left_end_ms=left.global_end_ms, right_start_ms=right.global_start_ms,
                    minimum_confidence=record["minimum_confidence"])
    detailed["joined"] = join_allowed(left, right, detailed["result"], detailed["frames"],
                                      minimum_confidence=record["minimum_confidence"])
    record["superseded_by"] = detailed["review_id"]
    return detailed


def _tail_object_continuity(group, review, decision, by_id):
    """Validate references linking the original task to the reviewed tail."""
    if not decision.same_operator or not decision.object_links:
        return False
    anchor = review.get("request_metadata", {}).get("continuity_anchor_ms", group.global_end_ms)
    endpoint = (decision.completion_frame_id if decision.state == "completed"
                else decision.last_continuation_frame_id)
    covered = set(decision.frame_ids)
    for link in decision.object_links:
        refs = set(link.before_frame_ids + link.after_frame_ids + link.handoff_frame_ids)
        if (not refs <= covered or not refs <= by_id.keys()
                or not all(by_id[i] <= anchor for i in link.before_frame_ids)
                or not all(by_id[i] > anchor for i in link.after_frame_ids)):
            return False
        if decision.same_workstation is False and not link.handoff_frame_ids:
            return False
        if not all(anchor < by_id[i] <= max(by_id.values()) for i in link.handoff_frame_ids):
            return False
    if not any(endpoint in link.after_frame_ids for link in decision.object_links):
        return False
    return all(group.first_person_view in {s["view_id"] for s in frame.get("sources", [])}
               for frame in review["frames"] if frame["frame_id"] in covered)


def _workflow_completion_supported(review, decision, by_id):
    """An operation/experiment unit ending does not end the continuous workflow."""
    if (review["result"].get("continuity_evidence_version", 2) < 4
            or decision.completion_scope != "workflow"
            or decision.post_completion_state != "no_related_continuation"):
        return False
    after = decision.post_completion_frame_ids
    end = by_id.get(decision.completion_frame_id)
    if (end is None or len(set(after)) < 2 or len(set(after)) != len(after)
            or not set(after) <= set(decision.frame_ids)
            or not all(i in by_id and by_id[i] > end for i in after)):
        return False
    followup_ms = max(12000, review.get("request_metadata", {}).get("completion_followup_ms", 12000))
    return (max(by_id[i] for i in after) == max(by_id.values())
            and max(by_id[i] for i in after) - end >= followup_ms)


def apply_tail_review(group, review, source_end_ms, minimum_confidence=0.85):
    """Keep an open end unless actual continuation/completion is evidenced."""
    result = review["result"]
    group.completion_status = "unresolved"
    group.completion_reason = "后续操作边界尚未确认；不能按此处认定实验结束"
    if result.get("status") != "completed":
        return False
    try:
        decision = TailDecision.model_validate(result["decision"])
    except (KeyError, ValueError):
        return False
    by_id = {f["frame_id"]: f["global_ms"] for f in review["frames"]}
    if (decision.confidence < minimum_confidence or not decision.frame_ids
            or len(by_id) != len(review["frames"])
            or not all(isinstance(t, (float, int)) and math.isfinite(t) for t in by_id.values())
            or not set(decision.frame_ids) <= by_id.keys()
            or not any(by_id[i] <= group.global_end_ms for i in decision.frame_ids)):
        return False
    if (result.get("continuity_evidence_version", 2) >= 4
            and not _tail_object_continuity(group, review, decision, by_id)):
        return False
    limit = min(source_end_ms, review.get("review_limit_ms", source_end_ms))
    if decision.state == "completed":
        end = by_id.get(decision.completion_frame_id)
        if (end is None or decision.completion_frame_id not in decision.frame_ids
                or end < group.global_end_ms or end >= limit
                or not _workflow_completion_supported(review, decision, by_id)):
            return False
        group.completion_status = "observed_complete"
        group.completion_reason = decision.observation
        group.global_end_ms = min(limit, end + 1000)
        if group.workflow_kind == "unresolved":
            group.workflow_kind = "independent_experiment"
        if group.workflow_units:
            group.workflow_units[-1]["completion_observed"] = True
    elif decision.state == "ongoing" and decision.continuation_observed:
        end = by_id.get(decision.last_continuation_frame_id)
        already_at_source_end = abs(source_end_ms - group.global_end_ms) <= 250
        if (end is None or decision.last_continuation_frame_id not in decision.frame_ids
                or end != max(by_id.values())
                or end > limit
                or (end <= group.global_end_ms and not already_at_source_end)):
            return False
        group.global_end_ms = source_end_ms if limit == source_end_ms and source_end_ms - end <= 250 else end
        group.completion_reason = decision.observation
        if group.global_end_ms >= source_end_ms:
            group.completion_status = "ongoing_at_recording_end"
    else:
        return False
    # Unknown camera correspondence remains explicitly empty in the added tail.
    prior_end = group.view_timeline[-1]["end_ms"] if group.view_timeline else review["left_end_ms"]
    if group.global_end_ms > prior_end:
        group.boundary_extension_requires_step_review = True
        group.view_timeline.append({"start_ms": prior_end, "end_ms": group.global_end_ms,
                                    "third_person_view": None, "reason": "tail_camera_correspondence_unverified"})
    if group.workflow_units:
        group.workflow_units[-1]["end_ms"] = group.global_end_ms
    if decision.same_workstation is True and decision.third_person_continuity_observed:
        third = next((row["third_person_view"] for row in reversed(group.view_timeline)
                      if row.get("third_person_view")), group.third_person_view)
        referenced_frames = [f for f in review["frames"] if f["frame_id"] in decision.frame_ids]
        if all({group.first_person_view, third} <= {s["view_id"] for s in f.get("sources", [])}
               for f in referenced_frames):
            for row in group.view_timeline:
                if (row.get("reason") == "tail_camera_correspondence_unverified"
                        and row["start_ms"] >= review["left_end_ms"] and row["end_ms"] <= group.global_end_ms):
                    row.update(third_person_view=third, reason="reviewed_tail_view_correspondence")
    return True


def review_experiment_boundaries(layout, groups, segments, events, views, infos,
                                 transforms, config):
    """Resolve per-wearable continuity, then inspect outer ends beyond CV events."""
    from .archive import write_json

    cfg = config.get("mllm", {}).get("boundary_review") or {}
    if not cfg.get("enabled", False) or not config.get("mllm", {}).get("enabled", False) or not groups:
        return list(groups)
    maximum_pairs = max(6, min(32, int(cfg.get("maximum_pairs", 20))))
    maximum_gap_ms = max(0.0, float(cfg.get("maximum_gap_seconds", 120))) * 1000
    minimum_confidence = max(0.85, float(cfg.get("minimum_confidence", 0.85)))
    groups = sorted(groups, key=lambda g: (g.global_start_ms, g.group_id))
    reviews = []
    before_groups = [g.model_dump(mode="json") for g in groups]
    started = time.perf_counter()
    analyzer = ArkAnalyzer(config)
    workers = max(1, min(8, int(cfg.get("workers", config["mllm"].get("group_workers", 2)))))
    try:
        previous = {}
        pairs = []
        for right in groups:
            left = previous.get(right.first_person_view)
            previous[right.first_person_view] = right
            if left is not None and 0 <= right.global_start_ms - left.global_end_ms <= maximum_gap_ms:
                pairs.append((left, right))

        def neighbor(pair):
            left, right = pair
            identity = {"left_group_uid": left.group_uid or left.group_id,
                        "right_group_uid": right.group_uid or right.group_id, "kind": "neighbor"}
            times = review_times(left, right, maximum_pairs=maximum_pairs,
                                 context_seconds=float(cfg.get("context_seconds", 12)))
            view_ids = list(dict.fromkeys([left.first_person_view, boundary_third_person(left, "end"), boundary_third_person(right, "start")]))
            record = _review(layout, analyzer, config, view_ids, views, infos, transforms, times,
                             {"proposed_left_end_ms": left.global_end_ms,
                              "proposed_right_start_ms": right.global_start_ms,
                              "left_context": operation_context(left, events, "end"),
                              "right_context": operation_context(right, events, "start")},
                             BOUNDARY_PROMPT, BoundaryDecision, identity)
            record.update(left_end_ms=left.global_end_ms, right_start_ms=right.global_start_ms,
                          minimum_confidence=minimum_confidence)
            record["joined"] = join_allowed(left, right, record["result"], record["frames"],
                                             minimum_confidence=minimum_confidence)
            detailed = refine_neighbor_review(layout, analyzer, config, left, right, record, views, infos, transforms)
            return [record, detailed] if detailed else [record]

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="workflow-boundary") as pool:
            for records in pool.map(neighbor, pairs):
                reviews.extend(records)
        groups = reconcile_groups(groups, segments, events, reviews)

        def tail(index_group):
            index, group = index_group
            records = []
            fp = group.first_person_view
            source_end = transforms[fp].to_global(infos[fp].duration_ms)
            next_start = min((g.global_start_ms for g in groups[index + 1:] if g.first_person_view == fp),
                             default=source_end)
            # A separate candidate group is an unresolved boundary, not a
            # source-file end. Never label it "waiting for next recording".
            limit = min(source_end, next_start)
            window_budget = max(1, int(cfg.get("maximum_tail_windows", 8)))
            uncertain_budget = max(0, min(window_budget, int(cfg.get("maximum_uncertain_tail_windows", 2))))
            uncertain_windows = 0
            scanned_end = group.global_end_ms
            for window in range(window_budget):
                end = min(limit - 100, scanned_end + float(cfg.get("tail_window_seconds", 90)) * 1000)
                at_source_end = window == 0 and abs(source_end - group.global_end_ms) <= 250
                if end <= group.global_end_ms and not at_source_end:
                    break
                start = max(group.global_start_ms, group.global_end_ms - 12000)
                anchor = max(start, min(group.global_end_ms, end - 1000)) if at_source_end else group.global_end_ms
                times = sorted(set(np.linspace(start, end, maximum_pairs).tolist()
                                   + ([anchor] if start <= anchor <= end else [])
                                   + ([scanned_end] if uncertain_windows and start <= scanned_end <= end else [])))
                identity = {"left_group_uid": group.group_uid or group.group_id,
                            "right_group_uid": "", "kind": "tail", "window": window,
                            "start_ms": start, "end_ms": end}
                last_tp = next((r["third_person_view"] for r in reversed(group.view_timeline)
                                if r.get("third_person_view")), group.third_person_view)
                record = _review(layout, analyzer, config, [fp, last_tp], views, infos, transforms, times,
                                 {"proposed_end_ms": group.global_end_ms,
                                  "continuity_anchor_ms": anchor,
                                  "recording_end_ms": source_end,
                                  "completion_followup_ms": max(12, float(cfg.get("completion_followup_seconds", 12))) * 1000,
                                  "uncertain_windows_before": uncertain_windows,
                                  "previous_review_end_ms": scanned_end,
                                  "review_does_not_extend_clip_without_continuity": True,
                                  "operation_context": operation_context(group, events, "end")},
                                 TAIL_PROMPT, TailDecision, identity)
                record.update(left_end_ms=group.global_end_ms, review_limit_ms=limit,
                              minimum_confidence=minimum_confidence, joined=False)
                progressed = apply_tail_review(group, record, source_end, minimum_confidence)
                record["tail_applied"] = progressed
                group.boundary_reviews.append(record)
                records.append(record)
                if group.completion_status in {"observed_complete", "ongoing_at_recording_end"}:
                    record["stop_reason"] = group.completion_status
                    break
                if not progressed:
                    decision = record["result"].get("decision") or {}
                    if record["result"].get("status") != "completed":
                        record["stop_reason"] = "review_failed"
                        break
                    if decision.get("state") == "different_workflow":
                        record["stop_reason"] = "different_workflow"
                        break
                    if uncertain_windows >= uncertain_budget:
                        record["stop_reason"] = "uncertainty_budget_exhausted"
                        break
                    uncertain_windows += 1
                else:
                    uncertain_windows = 0
                if end >= limit - 100:
                    record["stop_reason"] = "reviewed_limit_unresolved"
                    break
                # Advance the inspection cursor, not the accepted experiment
                # end. A later review must still bridge the original cut.
                scanned_end = end
            if records and "stop_reason" not in records[-1]:
                records[-1]["stop_reason"] = "tail_window_budget_exhausted"
            if records and group.completion_status == "unresolved":
                group.completion_reason = (
                    "已保留有依据的录像范围；后续承接或整个实验链的结束仍未确认，需要结合后续录像核对"
                )
            if group.view_timeline and group.global_end_ms != group.view_timeline[-1]["end_ms"]:
                raise ValueError("Workflow video coverage does not match reviewed extent")
            return records

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="workflow-tail") as pool:
            for records in pool.map(tail, enumerate(groups)):
                reviews.extend(records)
        # One additional pass is justified only when reviewed tail evidence
        # materially moved a rejected cut. Do not retry unchanged uncertainty.
        previous = {}
        advanced_pairs = []
        for right in groups:
            left = previous.get(right.first_person_view)
            previous[right.first_person_view] = right
            if left is None:
                continue
            progressed = any(r.get("kind") == "tail" and r.get("tail_applied") for r in left.boundary_reviews)
            if progressed and 0 <= right.global_start_ms - left.global_end_ms <= maximum_gap_ms:
                advanced_pairs.append((left, right))
        if advanced_pairs:
            extra = []
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="workflow-advanced-cut") as pool:
                for records in pool.map(neighbor, advanced_pairs):
                    extra.extend(records)
            reviews.extend(extra)
            groups = reconcile_groups(groups, segments, events, extra)
    finally:
        analyzer.close()
        write_json(layout.json_config / "experiment_boundary_review.json", {
            "schema_version": "visioncortex-experiment-boundary-review/4",
            "policy": "semantic units and workflow chains; recording end is never completion evidence",
            "original_groups": before_groups, "reviews": reviews,
            "workflow_tracks": workflow_tracking(groups, infos, transforms),
            "wall_seconds": round(time.perf_counter() - started, 6),
            "joined_edge_count": sum(r.get("joined", False) for r in reviews),
            "quality_classification": "PARTIAL_EVIDENCE", "formal_action_membership_changed": False,
        })
    return groups

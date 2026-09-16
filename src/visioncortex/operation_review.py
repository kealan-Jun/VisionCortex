"""Operation-level organization of adjudicated events, with immutable evidence.

This pass can combine evidence for an operation; it cannot create an action,
confirm experiment completion, or declare unsampled time fully understood.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .mllm import ArkAnalyzer, OBJECT_IDENTITY_RULES, OPERATION_DESCRIPTION_RULES
from .schemas import EvidenceEvent, ExperimentGroup, event_is_formal
from .step_evidence import next_operation, organization_metadata, timing_scope

GROUPS = "JSON-Config-Files/experiment_group_understanding.json"
EVENTS = "JSON-Config-Files/key_material_model_understanding.json"
CONTROL = "JSON-Config-Files/operation_review.json"
PROMPT = OBJECT_IDENTITY_RULES + OPERATION_DESCRIPTION_RULES + """
本轮只整理已审核事件的具体操作，实验名称、边界和后续判断由原记录保留，不再重写。
不要逐条照抄 CV 类别；同一具体操作、同一被操作实例的重叠证据可以合并。不确定是否同一实例时分开。
每个输入 event_id 必须恰好在一个步骤的 supporting_event_ids 出现。相隔超过 2.5 秒的事件不能合并。
current_step 清楚描述当前这一操作和可见结果，按必要信息写，不为凑字数补充背景。
当前过程和 observed_result 都不要重述此前或之后的独立动作。例如当前是折纸，就不再讲此前打开纸包、后来拿瓶；那些内容仍保存在原事件描述中。
直接写实验员的操作，不写机位比较、模型审核、推理过程或遵守规则的说明（例如“未把该接触写为……”）。必要限制直接说“读数看不清”“未看到放下”，避免枚举未接触的背景物体。
reviewed_operation 是整段审阅描述，可能包含当前动作前后的其他操作。先依据 action_type 与 action_review 找到本事件已确认的那一项操作，再写具体动词、被操作对象和直接可见结果。
例如已确认事件是天平面板操作，而整段还描述了折纸、拿瓶：本步骤只写按触面板及已知结果，不把折纸、拿瓶串进本步骤标题和过程，也不把按触写成去皮或称量完成。
following_context_not_current_evidence 仅为后续上下文，不能当当前步骤依据。未确认的完成状态、剂量或读数保持未知。
事件时间仅定位已审核证据，不是整套操作的起止时间；不得声称过程全部发生在此区间或整个操作已结束。
observed_result 只写当前操作直接可见的前后变化或结果；不能复制整段 physical_change 中属于其他操作的变化。按键身份、读数变化等不可见时直说未确认。原始整段描述仍单独保留。
只引用 reviewed_operation 中已经确认的动作，不因单帧补写吸排液、开合盖、按键或读数等功能性操作。
即使另一事件证明了某功能性动作，也不能用来升级本步骤引用事件的动作。图片仅为选定关键帧，不代表连续过程。
只返回 JSON：{"steps":[{"operation_title":"具体动词与对象","current_step":"操作过程及可见结果",
"observed_result":"仅本操作可见的变化或结果；不可见则写尚未确认",
"supporting_event_ids":["引用的输入事件ID"]}]}。不得返回其他字段或 Markdown。
"""


def coverage(group: dict, steps: list[dict]) -> dict:
    """Unrecorded intervals are review targets, not proven missed operations."""
    start = float(group.get("global_start_ms", group.get("start_ms", 0)) or 0)
    end = float(group.get("global_end_ms", group.get("end_ms", 0)) or 0)
    intervals = []
    for step in steps:
        ranges = step.get("evidence_intervals") or [step]
        for item in ranges:
            try:
                lo, hi = float(item["start_global_ms"]), float(item["end_global_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(lo) and math.isfinite(hi) and min(end, hi) > max(start, lo):
                intervals.append((max(start, lo), min(end, hi)))
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    gaps, cursor = [], start
    for lo, hi in merged + [[end, end]]:
        if lo-cursor >= 5000:
            gaps.append({"start_ms": cursor, "end_ms": lo, "seconds": round((lo-cursor)/1000, 3)})
        cursor = max(cursor, hi)
    return {"status": "PARTIAL_EVIDENCE", "all_operator_steps_proven": False,
            "recorded_intervals": merged, "review_intervals": gaps,
            "largest_review_gap_seconds": max((g["seconds"] for g in gaps), default=0),
            "completion_status": group.get("completion_status", "unreviewed"),
            "extension_requires_review": bool(group.get("boundary_extension_requires_step_review")),
            "basis": "saved step evidence intervals, not operation recall or full video coverage"}


def validate_steps(group, events, result):
    from .pipeline import validate_final_step_action_consistency

    if result.get("status") != "completed":
        raise ValueError("操作整理请求未完成")
    by_id = {e.event_id: e for e in events}
    seen, output = set(), []
    for step in result.get("steps") or []:
        ids = step.get("supporting_event_ids") or []
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= by_id.keys() or seen.intersection(ids):
            raise ValueError("步骤引用缺失、重复或不属于本次审核事件")
        referenced = sorted([by_id[i] for i in ids], key=lambda e: e.global_start_ms)
        end = referenced[0].global_end_ms
        for event in referenced[1:]:
            if event.global_start_ms-end > 2500:
                raise ValueError("步骤跨越未观察区间，不能合并")
            end = max(end, event.global_end_ms)
        start = referenced[0].global_start_ms
        lo, hi = step.get("start_global_ms"), step.get("end_global_ms")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi)) or abs(lo-start)>1 or abs(hi-end)>1:
            raise ValueError("步骤时间超出所引用事件范围")
        views = set.intersection(*(set(e.supporting_views) for e in referenced))
        if not step.get("supporting_views") or not set(step["supporting_views"]) <= views:
            raise ValueError("步骤引用了未支持该操作的机位")
        if not str(step.get("operation_title") or "").strip():
            raise ValueError("缺少具体操作名称")
        # A confirmed action elsewhere in the group cannot justify this step.
        check = group.model_copy(deep=True)
        check.key_event_ids = ids
        check.experiment_name = "操作记录"
        check.model_understanding = {"steps": [step]}
        if not validate_final_step_action_consistency([check], referenced)["passed"]:
            raise ValueError("步骤描述与其所引用的已审核动作不一致")
        row = copy.deepcopy(step)
        row.update(start_global_ms=start, end_global_ms=end,
                   evidence_intervals=[{"event_id": e.event_id, "start_global_ms": e.global_start_ms,
                                        "end_global_ms": e.global_end_ms} for e in referenced],
                   evidence_record_count=len(ids))
        row["time_scope"] = timing_scope(referenced)
        row["uncertainties"] = sorted({str(u) for e in referenced for u in
                                       [*e.uncertainty, *(e.model_understanding or {}).get("uncertainties", [])]})
        output.append(row)
        seen.update(ids)
    if seen != by_id.keys():
        raise ValueError("步骤整理遗漏了已审核事件")
    output.sort(key=lambda s: (s["start_global_ms"], s["end_global_ms"]))
    for i, step in enumerate(output, 1):
        step["step_index"] = i
    return output


def review(root: Path, groups: list, events: list, config: dict) -> list[dict]:
    from .archive import (_read_semantic_cache, _semantic_cache_path, _semantic_cache_reads_enabled,
                          _semantic_fingerprint, _write_semantic_cache, write_json)
    by_id = {e.event_id: e for e in events if event_is_formal(e) and (e.model_understanding or {}).get("status") == "completed"}
    analyzer = ArkAnalyzer(config)

    workers = max(1, min(4, int(config["mllm"].get("group_workers", 2))))
    size = max(1, min(6, int(config["mllm"].get("max_images_per_group", 12))))
    allocations, selected_by_group = [], {}
    for group in groups:
        if group.model_understanding is None:
            group.model_understanding = {}
        selected = sorted([by_id[i] for i in group.key_event_ids if i in by_id], key=lambda e:e.global_start_ms)
        selected_by_group[group.group_id] = selected
        for index, i in enumerate(range(0, len(selected), size)):
            allocations.append((group, index, selected[i:i+size]))

    def one(allocation):
        group, index, chunk = allocation
        images, image_receipts = [], []
        for event in chunk:
            path = event.key_frames.get("aligned_first_third") or event.key_frames.get(group.first_person_view)
            if path:
                p = root / path
                if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file():
                    raise ValueError("操作审核画面不可用或越出当前实验")
                image_receipts.append({"path": path, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
                images.append((f"{event.event_id}; key_global_ms={event.key_global_ms}; selected keyframe only", p))
        metadata = organization_metadata(group, chunk)
        fingerprint = _semantic_fingerprint("final-operations-v3", config, PROMPT, metadata, images)
        cache = _semantic_cache_path(config, "final-operations", fingerprint)
        retained_path = root / "JSON-Config-Files/Operation-Reviews/Allocations" / f"{fingerprint}.json"
        result = _read_semantic_cache(cache, fingerprint) if _semantic_cache_reads_enabled(config) else None
        if result is None and _semantic_cache_reads_enabled(config) and retained_path.is_file():
            # A completed request rejected by an older deterministic check can
            # be reused only after the current checks accept the exact evidence.
            try:
                retained = json.loads(retained_path.read_text())
                if (retained.get("status") == "completed"
                        and retained.get("input_fingerprint") == fingerprint
                        and retained.get("review_images") == image_receipts):
                    validate_steps(group, chunk, expand_steps(chunk, retained, selected_by_group[group.group_id]))
                    result = {**retained, "cache_reused": True,
                              "local_revalidation": "exact retained response accepted by current evidence checks"}
            except (OSError, ValueError):
                pass
        if result is None:
            result = (analyzer._call(PROMPT, metadata, images, max_images=len(images), response_kind="operations")
                      if images else {"status":"insufficient_frames", "attempts":0, "usage":{}})
        result = copy.deepcopy(result)
        accepted = []
        try:
            for image in image_receipts:
                if hashlib.sha256((root/image["path"]).read_bytes()).hexdigest() != image["sha256"]:
                    raise ValueError("审核期间画面变化，未应用结果")
            if any(not str(s.get("observed_result") or "").strip() for s in result.get("steps", [])):
                raise ValueError("缺少当前操作单独核对的结果")
            expanded = expand_steps(chunk, result, selected_by_group[group.group_id])
            accepted = validate_steps(group, chunk, expanded)
        except ValueError as exc:
            result["organization_accepted"] = False
            result["validation_error"] = str(exc)
        else:
            result["organization_accepted"] = True
            result.pop("validation_error", None)
            if not result.get("cache_reused"):
                result = _write_semantic_cache(cache, fingerprint, result)
        result.update(input_fingerprint=fingerprint, allocation_index=index, review_images=image_receipts)
        write_json(retained_path, result)
        return group.group_id, result, accepted

    records = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(one, allocations))
        for group in groups:
            selected = selected_by_group[group.group_id]
            results = [r for gid, r, _ in outputs if gid == group.group_id]
            steps = [step for gid, _, accepted in outputs if gid == group.group_id for step in accepted]
            accepted_all = bool(selected) and len(selected) == len(group.key_event_ids) and all(r.get("organization_accepted") for r in results)
            if accepted_all:
                steps.sort(key=lambda s: (s["start_global_ms"], s["end_global_ms"]))
                for i, step in enumerate(steps, 1):
                    step["step_index"] = i
                group.model_understanding["steps"] = steps
                group.model_understanding["overall_summary"] = f"已将 {len(selected)} 条审核证据整理为 {len(steps)} 个具体操作；按时间查看过程与可见结果。"
            group.model_understanding["operation_review"] = {"accepted":accepted_all, "calls":results,
                "maximum_workers":workers, "maximum_events_per_request":size,
                "policy":"all references retained; completion and event admission unchanged"}
            group.model_understanding["operation_coverage"] = coverage(group.model_dump(mode="json"), group.model_understanding.get("steps", []))
            record = {"group_id":group.group_id, "accepted":accepted_all, "calls":results}
            write_json(root / "JSON-Config-Files/Operation-Reviews" / f"{hashlib.sha256(group.group_id.encode()).hexdigest()[:24]}.json", record)
            records.append(record)
        return records
    finally:
        analyzer.close()


def expand_steps(events, result, following_events=None):
    """The model supplies prose and IDs; all other facts come from the ledger."""
    by_id = {e.event_id:e for e in events}
    expanded = copy.deepcopy(result)
    for step in expanded.get("steps", []):
        ids = step.get("supporting_event_ids") or []
        if not ids or not set(ids) <= by_id.keys():
            raise ValueError("步骤引用不属于输入事件")
        refs = sorted([by_id[i] for i in ids], key=lambda e:e.global_start_ms)
        last = max(refs, key=lambda e: (e.global_end_ms, e.global_start_ms))
        step.update(start_global_ms=min(e.global_start_ms for e in refs),
                    end_global_ms=max(e.global_end_ms for e in refs),
                    supporting_views=sorted(set.intersection(*(set(e.supporting_views) for e in refs))),
                    objects=list(dict.fromkeys(o for e in refs for o in e.objects)),
                    confidence=min(e.confidence for e in refs),
                    physical_change=step.get("observed_result") or "当前操作的结果尚未单独核对",
                    **next_operation(last, following_events if following_events is not None else events, set(ids)),
                    source_operation_records=[{"event_id":e.event_id,
                        "current_step":(e.model_understanding or {}).get("current_step"),
                        "physical_change":copy.deepcopy((e.model_understanding or {}).get("physical_change"))}
                        for e in refs])
    return expanded


def bindings(root):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else None
            for name in (GROUPS, EVENTS, "JSON-Config-Files/evidence_package.json")}


def apply(root: Path, groups: list[dict]) -> list[dict]:
    path = root / CONTROL
    if not path.is_file():
        return groups
    saved = json.loads(path.read_text())
    if saved.get("bindings") != bindings(root):
        return groups  # Never apply an old review to new evidence.
    updated = copy.deepcopy(groups)
    for group in updated:
        revised = saved.get("groups", {}).get(group["group_id"])
        if revised and all(revised.get(k) == group.get(k) for k in ("global_start_ms", "global_end_ms", "key_event_ids")):
            model = group.setdefault("model_understanding", {})
            for key in ("steps", "overall_summary", "operation_review", "operation_coverage"):
                model[key] = revised["model_understanding"][key]
    return updated


def refresh(root: Path, config: dict, revision: str) -> dict:
    from .archive import write_json
    from .speech_refresh import invalidate_reports
    original_bindings = bindings(root)
    if revision != original_bindings[GROUPS]:
        raise ValueError("步骤证据版本已改变，请刷新后重试")
    groups = [ExperimentGroup.model_validate(g) for g in json.loads((root/GROUPS).read_text())["groups"]]
    events = [EvidenceEvent.model_validate(e) for e in json.loads((root/EVENTS).read_text())["events"]]
    records = review(root, groups, events, config)
    if bindings(root) != original_bindings:
        raise ValueError("操作审核期间证据版本变化，未应用结果")
    path = root / CONTROL
    if path.is_file():
        write_json(root / f"JSON-Config-Files/Stage-Refreshes/{time.time_ns()}-previous-operations.json", json.loads(path.read_text()))
    invalidate_reports(root, reason="操作步骤理解已更新，报告等待重新生成")
    write_json(path, {"schema_version":"visioncortex-operation-revisions/1", "bindings":original_bindings,
                      "groups":{g.group_id:g.model_dump(mode="json") for g in groups}})
    return {"records":records, "calls":[c for r in records for c in r["calls"]],
            "accepted_group_count":sum(r["accepted"] for r in records), "group_count":len(groups)}

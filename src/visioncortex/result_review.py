"""Versioned completeness checks and bounded investigations of missing records.

Supplementary observations never change admitted actions or release quality.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .operation_review import apply as apply_operations, coverage
from .schemas import EvidenceEvent, ExperimentGroup

CHECK = "JSON-Config-Files/result_check.json"


def read(root, name, *, max_bytes=32 * 1024 * 1024):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("检查记录不属于当前实验")
    if not path.is_file():
        return {}
    if path.stat().st_size > max_bytes:
        raise ValueError("检查记录超过读取上限")
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def current(root):
    from .speech_refresh import apply

    groups = read(
        root, "JSON-Config-Files/evidence_package.json", max_bytes=128 * 1024 * 1024
    ).get("experiment_groups")
    if not groups:
        groups = read(
            root, "JSON-Config-Files/experiment_group_understanding.json"
        ).get("groups", [])
    groups = apply_operations(root, apply(root, groups))
    events = read(root, "JSON-Config-Files/key_material_model_understanding.json").get(
        "events", []
    )
    quality = read(root, "JSON-Config-Files/quality_acceptance.json")
    return groups, events, quality


def inspect(root: Path, *, save=False):
    from .archive import write_json
    from .pipeline import validate_final_step_action_consistency

    groups, events, quality = current(root)
    revision = digest({"groups": groups, "events": events, "quality": quality})
    saved = read(root, CHECK)
    findings, windows, summaries = [], [], []
    event_ids = {e["event_id"] for e in events}
    step_checks = []
    for g in groups:
        gid = g["group_id"]
        steps = (g.get("model_understanding") or {}).get("steps", [])
        refs = [i for s in steps for i in s.get("supporting_event_ids", [])]
        missing = sorted(set(g.get("key_event_ids", [])) - set(refs))
        unknown = sorted(set(refs) - event_ids)
        duplicate = sorted(i for i, count in Counter(refs).items() if count > 1)
        if missing or unknown or duplicate:
            findings.append(
                {
                    "group_id": gid,
                    "kind": "references",
                    "missing": missing,
                    "unknown": unknown,
                    "duplicate": duplicate,
                }
            )
        covered = coverage(g, steps)
        for gap in covered["review_intervals"]:
            findings.append({"group_id": gid, "kind": "unrecorded_interval", **gap})
            start = gap["start_ms"]
            while start < gap["end_ms"] and len(windows) < 256:
                end = min(start + 30000, gap["end_ms"])
                if end - start < 1000:
                    break
                window = {"group_id": gid, "start_ms": start, "end_ms": end}
                window["window_id"] = digest(window)[:20]
                windows.append(window)
                start = end
        state = g.get("completion_status", "unreviewed")
        if state != "observed_complete":
            findings.append(
                {
                    "group_id": gid,
                    "kind": "unfinished_boundary",
                    "completion_status": state,
                }
            )
        ordered = sorted(steps, key=lambda s: float(s.get("start_global_ms", 0)))
        overlaps = sum(
            float(a.get("end_global_ms", 0)) > float(b.get("start_global_ms", 0))
            for a, b in zip(ordered, ordered[1:], strict=False)
        )
        if overlaps:
            findings.append(
                {"group_id": gid, "kind": "overlapping_operations", "count": overlaps}
            )
        for step in steps:
            ids = step.get("supporting_event_ids") or []
            check_group = ExperimentGroup.model_validate(g)
            check_group.key_event_ids = ids
            check_group.experiment_name = "操作记录"
            check_group.model_understanding = {"steps": [step]}
            referenced = [
                EvidenceEvent.model_validate(e) for e in events if e["event_id"] in ids
            ]
            step_checks.append(
                validate_final_step_action_consistency([check_group], referenced)[
                    "passed"
                ]
            )
        summaries.append(
            {
                "group_id": gid,
                "step_count": len(steps),
                "gap_count": len(covered["review_intervals"]),
                "largest_gap_seconds": covered["largest_review_gap_seconds"],
                "completion_status": state,
            }
        )
    # Time proximity alone cannot join different operators or experiments.
    by_operator = {}
    for g in sorted(groups, key=lambda g: g["global_start_ms"]):
        previous = by_operator.get(g["first_person_view"])
        if previous and previous.get("completion_status") != "observed_complete":
            findings.append(
                {
                    "group_id": g["group_id"],
                    "previous_group_id": previous["group_id"],
                    "kind": "continuity_unverified",
                    "start_ms": previous["global_end_ms"],
                    "end_ms": g["global_start_ms"],
                }
            )
        by_operator[g["first_person_view"]] = g
    from .operation_review import bindings, CONTROL

    overlay = read(root, CONTROL)
    result = {
        "schema_version": "visioncortex-result-check/1",
        "available": True,
        "revision": revision,
        "evidence_classification": "PARTIAL_EVIDENCE",
        "formal_archive_promotion_allowed": False,
        "latest_check_current": save or saved.get("revision") == revision,
        "checked_at": datetime.now(timezone.utc).isoformat()
        if save
        else saved.get("checked_at")
        if saved.get("revision") == revision
        else None,
        "step_consistency_passed": bool(step_checks)
        and all(step_checks)
        and not any(f["kind"] == "references" for f in findings),
        "original_quality_passed": quality.get("passed"),
        "original_quality_sha256": digest(quality),
        "result_updated": bool(overlay and overlay.get("bindings") == bindings(root))
        or bool(read(root, "JSON-Config-Files/speech_group_understanding.json")),
        "full_quality_rechecked": False,
        "all_operations_found": None,
        "groups": summaries,
        "findings": findings,
        "windows": windows,
        "window_limit": 256,
        "window_seconds_limit": 30,
        "gap_observations": [],
    }
    for path in sorted((root / "JSON-Config-Files/Gap-Reviews").glob("*.json"))[-256:]:
        value = read(root, path.relative_to(root))
        if value.get("revision") == revision:
            result["gap_observations"].append(
                {
                    k: value.get(k)
                    for k in (
                        "window_id",
                        "group_id",
                        "start_ms",
                        "end_ms",
                        "observations",
                        "status",
                        "sample_count",
                        "limitation",
                    )
                }
            )
    if save:
        write_json(root / CHECK, result)
    return result


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=800)
    frame_ids: list[str] = Field(min_length=2)


class GapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation]
    limitation: str


GAP_PROMPT = """核对湿实验视频中尚无操作记录的一个短时间窗口。图片来自已保存的同步双视角视频，
左侧第一人称，右侧按既有机位时间表切换；标明未确认机位的画面不得当作第二视角证据。
按时间清晰描述可能遗漏的具体操作、对象与可见变化，不描述实验室背景。
只看见接触/倾斜不能称为开盖、去皮、吸排液；不可见剂量、成分、读数不得猜测。
每条观察引用至少两个不同时刻的 frame_ids；仅为补充观察，不是已审核动作。
图片为稀疏采样，时间为派生视频采样时间，不能声称整个窗口都没有操作或实验已经结束。
没有足够依据时 observations=[]，在 limitation 解释。只输出 JSON：
{"observations":[{"title":"具体操作","description":"按顺序描述过程与可见结果，说明尚不确定之处",
"frame_ids":["F001","F002"]}],"limitation":"采样和可见性限制"}"""


def validate_observations(payload, frames):
    decision = GapResponse.model_validate(
        {k: payload[k] for k in GapResponse.model_fields}
    )
    available = {f["frame_id"]: f for f in frames}
    for row in decision.observations:
        if not set(row.frame_ids) <= available.keys() or len(set(row.frame_ids)) < 2:
            raise ValueError("补充观察引用了不存在或重复的画面")
        times = [available[i]["global_ms"] for i in row.frame_ids]
        if max(times) <= min(times):
            raise ValueError("补充观察缺少前后时序画面")
    return decision.model_dump()


def investigate(root, config, target, revision):
    import cv2
    from .archive import (
        write_json,
        _semantic_fingerprint,
        _semantic_cache_path,
        _read_semantic_cache,
        _write_semantic_cache,
        _semantic_cache_reads_enabled,
    )
    from .mllm import ArkAnalyzer

    plan = inspect(root)
    if revision != plan["revision"]:
        raise ValueError("结果版本已经改变，请刷新后重新选择区间")
    window = next((w for w in plan["windows"] if w["window_id"] == target), None)
    if window is None:
        raise ValueError("所选区间不属于当前缺口检查")
    groups, _, _ = current(root)
    group = next(g for g in groups if g["group_id"] == window["group_id"])
    relative = (group.get("videos") or {}).get("aligned_first_third")
    if not relative:
        raise ValueError("当前片段没有已保存的同步视频")
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
    ):
        raise ValueError("同步视频不可读取或不属于本实验")
    original_stat = (path.stat().st_size, path.stat().st_mtime_ns)
    sidecar = (group.get("video_json") or {}).get("aligned_first_third")
    source = {
        "path": relative,
        "size_bytes": original_stat[0],
        "mtime_ns": original_stat[1],
        "sidecar_sha256": digest(read(root, sidecar)) if sidecar else None,
    }
    folder = (
        root
        / "JSON-Config-Files/Gap-Review-Frames"
        / digest({"revision": revision, "window": window, "source": source})[:24]
    )
    folder.mkdir(parents=True, exist_ok=True)
    frames, images = [], []
    from fractions import Fraction
    from .video_io import iter_sampled_frames, probe_video

    info = probe_video(path)
    local_start = window["start_ms"] - group["global_start_ms"]
    local_end = window["end_ms"] - group["global_start_ms"]
    samples = iter_sampled_frames(
        path,
        info,
        local_start,
        local_end,
        sample_fps=8000 / (local_end - local_start),
        max_width=min(info.width, 1280),
        hwaccel=None,
        decoder_threads=2,
    )
    try:
        for index, sample in enumerate(samples):
            if index >= 8:
                raise ValueError("选定窗口返回了过多采样画面")
            identity = getattr(sample, "source_frame", None)
            if identity is None or identity.status != "resolved":
                raise ValueError("无法验证派生视频画面的实际时间")
            actual_ms = float(identity.source_pts * Fraction(identity.time_base) * 1000)
            if not math.isfinite(actual_ms) or not local_start <= actual_ms < local_end:
                raise ValueError("派生视频采样画面不属于选定时间区间")
            image_path = folder / f"F{index + 1:03d}.jpg"
            if not cv2.imwrite(
                str(image_path), sample[2], [cv2.IMWRITE_JPEG_QUALITY, 90]
            ):
                raise OSError("无法保存补充分析画面")
            frames.append(
                {
                    "frame_id": image_path.stem,
                    "global_ms": group["global_start_ms"] + actual_ms,
                    "requested_global_ms": group["global_start_ms"] + sample[1],
                    "derived_video_frame_identity": identity.model_dump(mode="json"),
                    "image": str(image_path.relative_to(root)),
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                }
            )
            images.append(
                (
                    f"{image_path.stem}; derived-video t={frames[-1]['global_ms']:.3f}ms",
                    image_path,
                )
            )
    finally:
        samples.close()
    if len(frames) != 8 or len({f["global_ms"] for f in frames}) != 8:
        raise ValueError("选定窗口缺少八张不同时刻的有效画面")
    metadata = {
        "window": window,
        "frames": frames,
        "source": source,
        "view_timeline": group.get("view_timeline", []),
    }

    def sources_unchanged():
        return (
            (path.stat().st_size, path.stat().st_mtime_ns) == original_stat
            and (digest(read(root, sidecar)) if sidecar else None)
            == source["sidecar_sha256"]
            and all(
                hashlib.sha256((root / f["image"]).read_bytes()).hexdigest()
                == f["sha256"]
                for f in frames
            )
        )

    if not sources_unchanged():
        raise ValueError("采样期间视频或画面变化，请重新选择区间")
    fingerprint = _semantic_fingerprint(
        "gap-observation-v1", config, GAP_PROMPT, metadata, images
    )
    cache = _semantic_cache_path(config, "gap-observations", fingerprint)
    result = (
        _read_semantic_cache(cache, fingerprint)
        if _semantic_cache_reads_enabled(config)
        else None
    )
    if result is None:
        analyzer = ArkAnalyzer(config)
        try:
            result = analyzer._call(GAP_PROMPT, metadata, images, max_images=8)
        finally:
            analyzer.close()
    value = {
        **window,
        "revision": revision,
        "frames": frames,
        "source": source,
        "result": result,
        "status": "unverified",
        "observations": [],
        "sample_count": len(frames),
        "formal_action_admission": False,
        "source_video_decodes": 0,
        "derived_frame_reads": len(frames),
    }
    try:
        if result.get("status") != "completed":
            raise ValueError("补充分析请求未完成")
        decision = validate_observations(result, frames)
        if not sources_unchanged() or inspect(root)["revision"] != revision:
            raise ValueError("分析期间源视频或结果版本变化，未应用补充观察")
        value.update(status="supplementary_observations", **decision)
        if not result.get("cache_reused"):
            _write_semantic_cache(cache, fingerprint, result)
    except (ValueError, KeyError) as exc:
        value["validation_error"] = str(exc)
    write_json(root / "JSON-Config-Files/Gap-Reviews" / f"{target}.json", value)
    return value

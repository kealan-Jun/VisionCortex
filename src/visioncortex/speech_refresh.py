"""Selected group speech reinterpretation using retained, verified visual samples."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

from . import speech_worker
from .speech_semantics import SpeechContext, bind_speech_result, prompt_with_speech

CONTROL = "JSON-Config-Files/speech_group_understanding.json"
GROUPS = "JSON-Config-Files/experiment_group_understanding.json"


def input_path(root: Path, group_id: str) -> Path:
    return root / "JSON-Config-Files/Speech-Group-Inputs" / (hashlib.sha256(group_id.encode()).hexdigest()[:24] + ".json")


def retain_group_inputs(layout, group, storyboard: list, config: dict) -> None:
    if not (config.get("speech_recognition") or {}).get("enabled"):
        return
    samples = []
    for label, path in storyboard:
        record = speech_worker.file_record(path)
        target = layout.root / "Key-Materials/Experiment-Audio/Group-Samples" / f"{record['sha256']}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        if speech_worker.file_record(target) != record:
            raise ValueError("保留的片段画面完整性不匹配")
        samples.append({"label": label, "path": target.relative_to(layout.root).as_posix(), **record})
    input_path(layout.root, group.group_id).parent.mkdir(parents=True, exist_ok=True)
    speech_worker.atomic_json(input_path(layout.root, group.group_id), {
        "group_id": group.group_id, "start_global_ms": group.global_start_ms,
        "end_global_ms": group.global_end_ms, "visual_samples": samples})


def _base(root: Path) -> dict:
    return {name: speech_worker.sha256(root / name) for name in
            (GROUPS, "JSON-Config-Files/speech.json")}


def apply(root: Path, groups: list[dict]) -> list[dict]:
    path = root / CONTROL
    if not path.is_file():
        return groups
    overlay = speech_worker.read_json(path, 32*1024*1024)
    if overlay.get("bindings") != _base(root):
        raise ValueError("片段录音理解已过期，请按当前实验重新计算")
    output = copy.deepcopy(groups)
    for group in output:
        revision = overlay["groups"].get(group["group_id"])
        if revision is None:
            continue
        if revision["base_group_sha256"] != digest(group):
            raise ValueError("报告或步骤使用了不同版本的实验片段")
        model = group.setdefault("model_understanding", {})
        model.update({key: revision["result"][key] for key in ("speech_context", "speech_interpretation")})
        for step in model.get("steps", []):
            step["speech_segment_ids"] = [row["id"] for row in model["speech_context"]["segments"]
                if row["id"] in model["speech_interpretation"]["referenced_segment_ids"]
                and row["end_global_ms"] > step["start_global_ms"] and row["start_global_ms"] < step["end_global_ms"]]
        bind_speech_result(model, model["speech_context"])
    return output


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def invalidate_reports(root: Path) -> None:
    for filename in ("daily_report_manifest.json", "professional_report_manifest.json"):
        path = root / "JSON-Config-Files" / filename
        if path.is_file():
            original = speech_worker.read_json(path)
            (root / "JSON-Config-Files/Stage-Refreshes").mkdir(parents=True, exist_ok=True)
            speech_worker.atomic_json(root / f"JSON-Config-Files/Stage-Refreshes/{time.time_ns()}-previous-{filename}", original)
            # Remove navigable artifact references until a report uses the new revision.
            speech_worker.atomic_json(path, {"passed": False, "status": "stale", "reason": "录音理解已更新，报告等待重新生成"})


def refresh_group(root: Path, config: dict, target: str) -> dict:
    from .archive import (_read_semantic_cache, _semantic_cache_path, _semantic_cache_reads_enabled,
                          _semantic_fingerprint, _write_semantic_cache)
    from .mllm import ArkAnalyzer
    bindings = _base(root)
    groups = speech_worker.read_json(root / GROUPS, 32*1024*1024)["groups"]
    group = next((item for item in groups if "group:"+item["group_id"] == target), None)
    if group is None:
        raise ValueError("所选实验片段不存在")
    saved_path = input_path(root, group["group_id"])
    saved = speech_worker.read_json(saved_path)
    if (saved["group_id"], saved["start_global_ms"], saved["end_global_ms"]) != (group["group_id"], group["global_start_ms"], group["global_end_ms"]):
        raise ValueError("保留画面不属于当前片段时间窗")
    images = []
    for sample in saved["visual_samples"]:
        path = root / sample["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or speech_worker.file_record(path) != {key: sample[key] for key in ("size", "sha256")}:
            raise ValueError("片段画面完整性检查未通过")
        images.append((sample["label"], path))
    if not images:
        raise ValueError("缺少已保留的片段画面")
    context = SpeechContext(root, config).window(group["global_start_ms"], group["global_end_ms"], group["participating_views"])
    if not context or not context["segments"]:
        raise ValueError("所选片段没有可对齐的转写")
    metadata = {"group_id": group["group_id"], "speech_context": context,
                "visual_samples": saved["visual_samples"],
                "retained_visual_steps": [{key:value for key,value in step.items() if key != "speech_segment_ids"}
                                          for step in (group.get("model_understanding") or {}).get("steps", [])]}
    from .capture_quality import model_context
    metadata.update(model_context(root))
    prompt = prompt_with_speech("仅输出 JSON 对象，唯一字段 speech_interpretation。重新理解所选实验片段的录音与保留画面的关系。"
                                "已有视觉步骤和边界是固定证据；只解释录音提及了什么，不新增或修改动作。抽样没有覆盖的时刻保留不确定性。", context)
    fingerprint = _semantic_fingerprint("group-speech-refresh", config, prompt, metadata, images)
    cache = _semantic_cache_path(config, "group-speech-refresh", fingerprint)
    result = _read_semantic_cache(cache, fingerprint) if _semantic_cache_reads_enabled(config) else None
    if result is None:
        analyzer = ArkAnalyzer(config)
        try:
            result = analyzer._call(prompt, metadata, images, max_images=len(images), response_kind="speech")
        finally:
            analyzer.close()
        if result.get("status") == "completed":
            result = _write_semantic_cache(cache, fingerprint, result)
    if result.get("status") != "completed":
        speech_worker.atomic_json(root / "JSON-Config-Files/understanding_refresh_failure.json", {"target":target, "result":result})
        raise ValueError("片段理解未完成，原成果已保留")
    result = {**bind_speech_result(result, context), **{key: metadata[key] for key in ("capture_quality",) if key in metadata}}
    path = root / CONTROL
    original = speech_worker.read_json(path, 32*1024*1024) if path.is_file() else {"groups": {}, "bindings": bindings}
    if original["bindings"] != bindings or _base(root) != bindings:
        raise ValueError("片段证据版本变化，请刷新后重算")
    revision = {"base_group_sha256": digest(group), "input_sha256": speech_worker.sha256(saved_path), "result": result}
    updated = {**original, "schema_version": "visioncortex-group-speech-revisions/1",
               "groups": {**original["groups"], group["group_id"]: revision}}
    history = root / f"JSON-Config-Files/Stage-Refreshes/{time.time_ns()}-previous-group-speech.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    speech_worker.atomic_json(history, original)
    invalidate_reports(root)
    speech_worker.atomic_json(path, updated)
    apply(root, groups)  # Validate all references before downstream rendering.
    return {"parts": [result], "target": target, "affected_group_ids": [group["group_id"]],
            "updated_step_reference_count": len((group.get("model_understanding") or {}).get("steps", []))}

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from .schemas import EvidenceEvent, ExperimentGroup, ExperimentSegment


EVENT_SYSTEM_PROMPT = """你是化学湿实验视频证据审计模型。你会收到同一全局时间点、严格对齐的第一人称和第三人称关键帧，以及传统 CV 证据。
只描述画面可观察事实，不补写未出现的试剂名、剂量、读数或实验目的。第一/第三人称冲突时必须指出。
液体移动必须看到液体/液面变化、倾倒姿态、移液器与容器配合等证据之一，否则标为待确认。
输出单个 JSON 对象，字段固定为：
{
  "current_step": "当前这一步做什么，细到手、对象、状态",
  "next_step": "根据尾部状态可直接观察或谨慎推断的下一步；不能判断写未知",
  "action_type_confirmed": "hand_object_contact/object_movement/liquid_movement/container_state_change/device_panel_operation/unknown",
  "objects": ["明确可见物体"],
  "hand_object_interactions": [{"hand":"left/right/unknown", "object":"物体", "contact":"接触/抓取/释放/操作"}],
  "physical_change": {"before":"之前状态", "after":"之后状态"},
  "per_view_observations": [{"view_id":"视角ID", "observation":"该视角独立证据"}],
  "cross_view_consistency": "consistent/partial/conflict/single_view",
  "confidence": 0.0,
  "uncertainties": ["无法确认项"]
}
禁止输出 Markdown。"""


GROUP_SYSTEM_PROMPT = """你是化学湿实验有界视频的步骤级理解与命名模型。输入是一个有界实验组按时间顺序采样的第一/第三人称对齐故事板、原子边界和传统 CV 事件。
任务：
1. 判断这是单个独立实验，还是多个无中断承接的连续实验；时间接近本身不代表连续，必须看到人员、对象或物理状态承接。
2. 给出具体、保守的实验名称，例如“固体称量实验”“移液实验”“固体称量与移液连续实验”，禁止使用“实验片段一”等泛名。
3. 输出从开始到结束的细粒度步骤，说明当前做什么、下一步做什么、物体和状态变化。
4. 不得虚构试剂化学名称、质量、体积、面板读数或实验目的；看不清就写未知。
输出单个 JSON 对象，字段固定为：
{
  "experiment_name": "中文具体实验名称",
  "experiment_name_en": "ASCII-English-Experiment-Name",
  "continuity_type_confirmed": "independent/continuous/uncertain",
  "continuity_reason": "可观察的连续/独立依据",
  "atomic_experiments": [{"name":"内部原子实验名称", "start_global_ms":0, "end_global_ms":0, "purpose_observable":"可观察目标或未知"}],
  "steps": [{"step_index":1, "start_global_ms":0, "end_global_ms":0, "current_step":"当前动作", "next_step":"下一动作或未知", "objects":["物体"], "physical_change":"状态变化", "supporting_views":["视角ID"], "confidence":0.0}],
  "overall_summary": "整个有界视频的实验内容",
  "confidence": 0.0,
  "uncertainties": ["无法确认项"]
}
禁止输出 Markdown。"""


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_text(payload: dict[str, Any]) -> str:
    response_parts = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                response_parts.append(str(content.get("text") or ""))
    if response_parts:
        return "".join(response_parts)
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError(f"响应没有 choices: {payload.get('error') or payload.keys()}")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型没有返回 JSON 对象")
    return value


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_input_tokens": (
            usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        ).get("cached_tokens"),
        "server_reported": bool(usage),
    }


class ArkAnalyzer:
    def __init__(self, config: dict[str, Any]):
        self.config = config["mllm"]
        self.api_key = os.getenv(str(self.config["api_key_env"]))
        self.enabled = bool(self.config["enabled"])

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _call(
        self,
        system_prompt: str,
        metadata: dict[str, Any],
        image_paths: Sequence[tuple[str, Path]],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "uncertainties": ["多模态分析已在配置中关闭"]}
        if not self.api_key:
            return {
                "status": "skipped_missing_api_key",
                "uncertainties": [f"未设置环境变量 {self.config['api_key_env']}，未调用模型"],
            }
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": json.dumps(metadata, ensure_ascii=False)}
        ]
        selected = list(image_paths)[: int(self.config["max_images_per_event"])]
        for label, path in selected:
            content.append({"type": "input_text", "text": label})
            content.append(
                {"type": "input_image", "image_url": _image_data_url(path), "detail": "high"}
            )
        request = {
            "model": self.config["model"],
            "instructions": system_prompt,
            "input": [
                {"type": "message", "role": "user", "content": content},
            ],
            "store": False,
            "thinking": {"type": "disabled"},
            "max_output_tokens": int(self.config.get("max_output_tokens", 4096)),
        }
        url = str(self.config["base_url"]).rstrip("/") + "/responses"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(int(self.config["max_retries"])):
            try:
                with httpx.Client(timeout=float(self.config["timeout_seconds"])) as client:
                    response = client.post(url, headers=headers, json=request)
                    if response.is_error:
                        body = response.text[:1000]
                        raise httpx.HTTPStatusError(
                            f"{response.status_code} from Ark Responses API: {body}",
                            request=response.request,
                            response=response,
                        )
                    payload = response.json()
                result = _parse_json(_extract_text(payload))
                result.update(
                    {
                        "status": "completed",
                        "model": self.config["model"],
                        "usage": _usage(payload),
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "attempts": attempt + 1,
                    }
                )
                return result
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and 400 <= exc.response.status_code < 500
                    and exc.response.status_code not in {408, 409, 429}
                ):
                    break
                if attempt + 1 < int(self.config["max_retries"]):
                    time.sleep(min(8.0, 2.0**attempt))
        return {
            "status": "failed",
            "error": f"{type(last_error).__name__}: {last_error}",
            "uncertainties": ["多模态调用失败；保留 CV 与跨视角审计结果，不伪造模型理解"],
            "latency_seconds": round(time.perf_counter() - started, 6),
            "attempts": attempt + 1,
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_input_tokens": None,
                "server_reported": False,
            },
        }

    def analyze_event(
        self, event: EvidenceEvent, image_paths: Sequence[tuple[str, Path]]
    ) -> dict[str, Any]:
        return self._call(
            EVENT_SYSTEM_PROMPT,
            {
                "event_id": event.event_id,
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "key_global_ms": event.key_global_ms,
                "cv_action_candidate": event.action_type.value,
                "cv_objects": event.objects,
                "cv_confidence": event.confidence,
                "supporting_views": event.supporting_views,
                "audit_reason": event.audit_reason,
                "task": "逐视角核实当前细步骤，并描述画面支持的下一步",
            },
            image_paths,
        )

    def analyze_group(
        self,
        group: ExperimentGroup,
        segments: Sequence[ExperimentSegment],
        events: Sequence[EvidenceEvent],
        storyboards: Sequence[tuple[str, Path]],
    ) -> dict[str, Any]:
        return self._call(
            GROUP_SYSTEM_PROMPT,
            {
                "group_id": group.group_id,
                "rule_based_continuity_type": group.continuity_type,
                "rule_based_continuity_reason": group.continuity_reason,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "first_person_view": group.first_person_view,
                "third_person_view": group.third_person_view,
                "atomic_boundaries": [segment.model_dump(mode="json") for segment in segments],
                "cv_events": [
                    {
                        "event_id": event.event_id,
                        "action_type": event.action_type.value,
                        "start_global_ms": event.global_start_ms,
                        "end_global_ms": event.global_end_ms,
                        "key_global_ms": event.key_global_ms,
                        "objects": event.objects,
                        "confidence": event.confidence,
                        "supporting_views": event.supporting_views,
                    }
                    for event in events
                    if event.accepted
                ],
                "storyboard_order": [label for label, _ in storyboards],
                "task": "理解整个有界实验视频、命名实验、判断连续性并输出步骤序列",
            },
            storyboards,
        )


# Backward-compatible name used by archive.py.
ArkStepAnalyzer = ArkAnalyzer

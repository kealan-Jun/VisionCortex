from __future__ import annotations

import base64
import json
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import httpx

from .schemas import ActionType, EvidenceEvent, ExperimentGroup, ExperimentSegment


OBJECT_IDENTITY_RULES = """物体辨识规则，适用于当前步骤、下一步骤、交互列表、物理变化及摘要的全部描述：
先区分被操作的物体、外包装和包装内物。白色外观、透明袋或包装鼓起，不能单独证明里面是粉末、液体或试剂；纸张、耗材和包装反光都可能造成相似外观。
有可读标签时，可以写“标有某文字的包装”，不能把标签名称当作已验证的实际内容物。内容形态须有独立可见依据，例如散落颗粒、连续液面或被抽出的纸片；没有依据时写“包装”或“内容物不可辨认”，不得猜测物质成分。
同类对象可能有多个实例。仅描述与手或工具实际交互的实例；把背景物体与被操作物体分开，不得根据邻近、颜色相似或既有CV类别补写接触。
逐只手核对直接接触：双手出现在画面中不等于双手都在操作同一对象。只看见一只手按键时就描述该只手，不能概括成“双手按键”或给另一只手补记面板接触；左右无法辨明时使用 unknown。
"""


EVENT_SYSTEM_PROMPT = OBJECT_IDENTITY_RULES + """你是化学湿实验视频证据审计模型。你会收到第一人称和第三人称片段的时序采样，以及传统 CV 证据。
sample_scope=clip_timeline 的图片仅表示片段采样位置，clip_middle 不是动作峰值，也不代表 key_global_ms。nominal_clip_time_ms 是该片段内的近似时间，不能当全局时间。
sample_scope=selected_keyframe 的图片才是选定关键画面，requested_global_ms 是请求时刻，decoded_global_ms 是解码定位目标，帧的精确PTS与跨视角同步精度未在此验证。仅在收到这类图片时填写 selected_keyframe_observations，且只记录该张图中实际可见的交互；不得把片段较早或较晚的接触套用到关键帧。未收到时返回空列表，不得猜测。current_step 和 hand_object_interactions 描述整个片段，不能因此宣称全部对象在关键帧同时被操作。
只描述画面可观察事实，不补写未出现的试剂名、剂量、读数或实验目的。第一/第三人称冲突时必须指出。
候选动作是否存在，与两个视角是否拍到同一操作者/同一对象，是两个独立判断：只要至少一个视角清晰直接证明候选动作，就按该动作填写 action_type_confirmed，并在 candidate_action_support_by_view 对该视角标 true；另一个视角拍到并行动作时在 cross_view_consistency 标 conflict，不得仅因视角冲突把已清晰可见的动作改成 unknown。
液体移动是高风险类别，必须 fail-closed：只有看到液流/液面变化、伴随可见液体变化的倾倒，或完整的移液闭环才能确认。移液闭环必须清晰看到源容器接触、抽离/运输、目标容器接触，再加可见的排液/推杆变化；仅当 cv_observability.measurements.dual_role_transfer_sequence_verified 为 true 且故事板未反证时，才可用结构化双视角路径替代不可见的推杆细节，并把 dual_role_cv_sequence_verified 填 true。cv_observability 中的 repeated_pipette_path 只是选择密集审阅主视角的召回候选，跟踪碎片不能证明重复循环或液体移动。容器倾斜姿态、移液器靠近、仅伸入一个容器或重复轨迹候选都不能单独确认液体已移动。
pipette_transfer_operation 与 liquid_movement 是两个证据层级。只有同一个连续时序中清晰看到移液器吸头进入源容器、抽离/运输、再进入一个不同目标容器，才可确认 pipette_transfer_operation；它只声明可见的源到目标移液器操作链，不声明微量液体实际移动可见。此时 proof_type 必须为 pipette_operational_transfer_chain，source_contact_visible、withdrawal_or_transport_visible、target_contact_visible 必须都为 true；液体和柱塞相关字段仍须按画面如实填写。若液体候选只能证明该操作链，必须 relabel_suggested 为 pipette_transfer_operation，而不是放宽 liquid_movement。
必须按采样顺序比较可见状态变化；单帧中手与物体框接近不能直接证明接触，设备框出现不能直接证明面板操作，工具和容器同时出现不能直接证明液体转移。
CV 可观测性收据会明确21类检测器能直接证明什么、仍缺什么。不要把收据中的“间接候选”复述成已观察事实。
输出单个 JSON 对象，字段固定为：
{
  "current_step": "当前这一步做什么，细到手、对象、状态",
  "next_step": "根据尾部状态可直接观察或谨慎推断的下一步；不能判断写未知",
  "action_type_confirmed": "hand_object_contact/object_movement/pipette_transfer_operation/liquid_movement/container_state_change/device_panel_operation/unknown",
  "objects": ["明确可见物体"],
  "hand_object_interactions": [{"hand":"left/right/unknown", "object":"物体", "contact":"接触/抓取/释放/操作"}],
  "physical_change": {"before":"之前状态", "after":"之后状态"},
  "per_view_observations": [{"view_id":"视角ID", "observation":"该视角独立证据"}],
  "candidate_action_support_by_view": [{"view_id":"视角ID", "supports_candidate_action":true, "confidence":0.0, "reason":"该视角支持/不支持候选动作的可见依据"}],
  "confirmed_action_support_by_view": [{"view_id":"视角ID", "supports_confirmed_action":true, "confidence":0.0, "reason":"该视角对 action_type_confirmed 的直接可见依据"}],
  "action_proof": {"proof_type":"visible_liquid_flow/visible_liquid_level_change/pour_with_visible_liquid_change/pipette_closed_transfer_cycle/pipette_operational_transfer_chain/posture_only/direct_other/none", "visible_liquid_or_level_change":false, "source_contact_visible":false, "withdrawal_or_transport_visible":false, "target_contact_visible":false, "release_or_plunger_change_visible":false, "dual_role_cv_sequence_verified":false, "container_before_state_visible":false, "container_after_state_visible":false, "container_state_transition_completed":false, "reason":"可审计的直接证据"},
  "cross_view_consistency": "consistent/partial/conflict/single_view",
  "evidence_verdict": "confirmed/relabel_suggested/uncertain/rejected",
  "temporal_support": {"early":"片段较早采样事实", "middle":"片段中部采样事实，不代表动作峰值", "late":"片段较晚采样事实"},
  "selected_keyframe_observations": [{"view_id":"提供的单视角ID", "decoded_global_ms":0, "directly_interacting_objects":["当前关键帧直接交互的对象"], "observation":"只描述此关键帧中可见的事实", "uncertainties":["不可辨认或被遮挡的交互"]}],
  "confidence": 0.0,
  "uncertainties": ["无法确认项"]
}
evidence_verdict 的含义只针对候选动作本身：至少一个视角直接清晰证明则为 confirmed；候选动作未被证明但另一个动作被证明才用 relabel_suggested；证据不足用 uncertain；直接反证用 rejected。candidate_action_support_by_view 只评价输入候选；confirmed_action_support_by_view 只评价 action_type_confirmed。视角主体冲突单独写入 cross_view_consistency 和 uncertainties，不能抹去单个视角对动作本体的直接证明。
对 container_state_change，只有在同一个视角的时序中同时看到动作前容器状态、动作后容器状态，且开启/关闭/盖合/解除盖合的状态转换已完成，才能把 action_proof 中三个 container_* 字段都填 true 并确认该类。这里三个 container_* 字段表示“至少一个视角存在完整直接闭环”，不要求另一个视角重复拍到；若一个视角完整证明而另一视角模糊、遮挡或拍到并行动作，必须保留 container_state_change，把完整证明视角的 confirmed_action_support_by_view 标 true，并把另一视角的不完整性写入 cross_view_consistency/uncertainties，不得因此把三个字段改为 false 或降级为 hand_object_contact。只看到手持瓶盖对准瓶口、操作进行中、所有视角末帧仍被手遮挡，不算已完成的容器状态转换；此时才应按可见事实改为 hand_object_contact 或 object_movement。
禁止输出 Markdown。"""


GROUP_SYSTEM_PROMPT = OBJECT_IDENTITY_RULES + """你是化学湿实验有界视频的步骤级理解与命名模型。输入是一个有界实验组按时间顺序采样的第一/第三人称对齐故事板、原子边界和传统 CV 事件。
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
  "boundary_assessment": {"start_complete":true, "end_complete":true, "start_reason":"起点证据", "end_reason":"终点证据", "localized_rescan_needed":false},
  "confidence": 0.0,
  "uncertainties": ["无法确认项"]
}
禁止输出 Markdown。"""


FINAL_GROUP_SYSTEM_PROMPT = GROUP_SYSTEM_PROMPT + """

这是事件级语义审核后的最终步骤精炼，不是候选发现。输入事件均为已经确认或严格重标的最终关键事件：
1. 步骤中的动作事实必须能对应至少一个输入 final_adjudicated_event；故事板只用于描述参与对象的可见状态，不得从“手在设备附近”新增设备面板操作，不得从“工具与容器同框”新增液体移动或容器开合。
2. absent_action_classes 中列出的类别在本实验组没有被确认，步骤、摘要、实验名称和下一步都不得把这些类别写成已经发生的事实。尤其 absent_action_classes 含 device_panel_operation 时，禁止写按键、按压面板、操作按钮、读取或确认读数；含 liquid_movement 时，禁止写吸液、排液、加液、倾倒或液体转移；即使瓶体横放、翻倒或姿态发生变化，也只能写“瓶体姿态改变”，不能使用“倾倒”这个会宣称功能性液体动作的词。若 pipette_transfer_operation 已确认但 liquid_movement 未确认，可写“移液器从源容器移动并进入目标容器”或“源到目标移液器操作（液体状态不可见）”，不得升级为液体已转移。含 pipette_transfer_operation 时若该类别 absent，禁止写移液操作、源到目标移液器操作或完整移液流程。含 container_state_change 时，禁止写开盖、合盖、旋开或旋紧。
3. 可以写“手接触天平”“移动瓶盖”等已确认的较低层动作，但不得升级成未确认的功能性操作。
4. 时间范围必须来自 final_adjudicated_events，不得把相邻步骤重叠扩展到没有最终事件支持的动作。
输出结构仍严格使用上面的 JSON 合同。
"""


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


def normalize_uncalibrated_hand_identity(result: dict[str, Any]) -> dict[str, Any]:
    """Keep action evidence without claiming uncalibrated operator handedness.

    Camera-left and operator-left are not interchangeable. The current input
    contract has no calibrated hand identity. Preserve provider wording for
    audit and expose unknown identity rather than conflicting left/right tags.
    """
    if result.get("hand_identity_review") or result.get("status") != "completed":
        return result
    normalized = deepcopy(result)
    originals: dict[str, str] = {}

    def neutralize(value: Any, path: str) -> Any:
        if isinstance(value, str):
            text = re.sub(r"(?:左|右)手(?!边|侧|柄)", "手", value)
            text = re.sub(r"\b(?:left|right)[ -]hand\b(?![ -]side)", "hand", text, flags=re.IGNORECASE)
            if text != value:
                originals[path] = value
            return text
        if isinstance(value, list):
            return [neutralize(item, f"{path}/{index}") for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: neutralize(item, f"{path}/{key}") for key, item in value.items()}
        return value

    for key in ("current_step", "next_step", "objects", "physical_change",
                "per_view_observations", "selected_keyframe_observations", "candidate_action_support_by_view",
                "confirmed_action_support_by_view", "action_proof", "temporal_support",
                "uncertainties", "steps", "overall_summary", "experiment_name",
                "experiment_name_en", "atomic_experiments"):
        if key in normalized:
            normalized[key] = neutralize(normalized[key], key)
    original_interactions = deepcopy(result.get("hand_object_interactions") or [])
    interactions = []
    grouped = {}
    for interaction in original_interactions:
        if not isinstance(interaction, dict):
            interactions.append(interaction)
            continue
        item = neutralize(deepcopy(interaction), "hand_object_interactions")
        item["hand"] = "unknown"
        # Repeated cross-view assignments do not establish two different hands.
        # The narrative's explicit one-hand/two-hand count remains unchanged.
        key = str(item.get("object") or "")
        if key and key in grouped:
            prior = grouped[key]
            contacts = list(dict.fromkeys(str(prior.get("contact") or "").split("/") + str(item.get("contact") or "").split("/")))
            prior["contact"] = "/".join(value for value in contacts if value)
        else:
            interactions.append(item)
            if key:
                grouped[key] = item
    if "hand_object_interactions" in normalized:
        normalized["hand_object_interactions"] = interactions
    if originals or interactions != original_interactions:
        normalized["hand_identity_review"] = {
            "status": "uncalibrated_identity",
            "policy": "Retain visible interactions; do not assert operator left/right from screen position",
            "original_hand_object_interactions": original_interactions,
            "original_text_fields": originals,
            "human_ground_truth": False,
        }
    return normalized


class ArkAnalyzer:
    def __init__(self, config: dict[str, Any]):
        self.config = config["mllm"]
        self.api_key = os.getenv(str(self.config["api_key_env"]))
        self.enabled = bool(self.config["enabled"])
        pool_size = max(
            2,
            int(self.config.get("workers", 4)),
            int(self.config.get("group_workers", 2)),
        )
        self.client = httpx.Client(
            timeout=float(self.config["timeout_seconds"]),
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size,
                keepalive_expiry=float(self.config.get("keepalive_expiry_seconds", 60.0)),
            ),
            http2=bool(self.config.get("http2", False)),
        )

    def close(self) -> None:
        self.client.close()

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _call(
        self,
        system_prompt: str,
        metadata: dict[str, Any],
        image_paths: Sequence[tuple[str, Path]],
        *,
        max_images: int | None = None,
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
        selected = list(image_paths)[: int(max_images or self.config["max_images_per_event"])]
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
                response = self.client.post(url, headers=headers, json=request)
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
                return normalize_uncalibrated_hand_identity(result)
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and 400 <= exc.response.status_code < 500
                    and exc.response.status_code not in {408, 409, 429}
                ):
                    break
                if attempt + 1 < int(self.config["max_retries"]):
                    retry_after = None
                    if isinstance(exc, httpx.HTTPStatusError):
                        try:
                            retry_after = float(exc.response.headers.get("retry-after", ""))
                        except ValueError:
                            retry_after = None
                    delay = (
                        retry_after
                        if retry_after is not None
                        else min(
                            float(self.config.get("retry_max_seconds", 30.0)),
                            2.0**attempt,
                        )
                        + random.uniform(0.0, 0.75)
                    )
                    time.sleep(max(0.0, delay))
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
        action_limits = self.config.get("max_images_per_event_by_action") or {}
        max_images = int(
            action_limits.get(
                event.action_type.value,
                self.config["max_images_per_event"],
            )
        )
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
                "cv_observability": event.observability,
                "review_image_labels": [label for label, _ in image_paths[:max_images]],
                "task": "逐视角核实当前细步骤，并描述画面支持的下一步",
            },
            image_paths,
            max_images=max_images,
        )

    def analyze_group(
        self,
        group: ExperimentGroup,
        segments: Sequence[ExperimentSegment],
        events: Sequence[EvidenceEvent],
        storyboards: Sequence[tuple[str, Path]],
        *,
        system_prompt: str = GROUP_SYSTEM_PROMPT,
        final_adjudicated: bool = False,
    ) -> dict[str, Any]:
        return self._call(
            system_prompt,
            {
                "group_id": group.group_id,
                "rule_based_continuity_type": group.continuity_type,
                "rule_based_continuity_reason": group.continuity_reason,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "first_person_view": group.first_person_view,
                "third_person_view": group.third_person_view,
                "atomic_boundaries": [segment.model_dump(mode="json") for segment in segments],
                (
                    "final_adjudicated_events"
                    if final_adjudicated
                    else "cv_events"
                ): [
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
                **(
                    {
                        "absent_action_classes": sorted(
                            {item.value for item in ActionType}
                            - {event.action_type.value for event in events if event.accepted}
                        )
                    }
                    if final_adjudicated
                    else {}
                ),
                "storyboard_order": [label for label, _ in storyboards],
                "task": (
                    "仅用最终审核事件精炼步骤，不新增未确认动作事实"
                    if final_adjudicated
                    else "理解整个有界实验视频、命名实验、判断连续性并输出步骤序列"
                ),
            },
            storyboards,
        )


# Backward-compatible name used by archive.py.
ArkStepAnalyzer = ArkAnalyzer

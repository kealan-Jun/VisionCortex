from __future__ import annotations

import base64
import json
import random
import re
import threading
import time
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .mllm_provider import build_vision_request
from .provider_credentials import model_api_key
from .speech_semantics import (bind_speech_result, compact_speech_label,
                               compact_speech_metadata, expand_speech_references, prompt_with_speech)

from .schemas import (
    ActionType,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    event_is_formal,
)


def _http_error_message(response, api_key):
    """Capture bounded provider diagnostics for streamed HTTP errors too."""
    body = bytearray()
    try:
        for chunk in response.iter_bytes(chunk_size=1024):
            body.extend(chunk)
            if len(body) >= 8192:
                break
        payload = json.loads(bytes(body[:8192]))
        error = payload.get("error", payload)
        detail = ": ".join(str(error.get(key, ""))[:500] for key in ("code", "type", "message"))
    except (ValueError, AttributeError, httpx.HTTPError):
        detail = "provider error body unavailable"
    message = f"{response.status_code} from multimodal API: {detail}"
    return message.replace(api_key, "[REDACTED]") if api_key else message


SCENE_TIME_REFERENCE_RULES = """步骤时间引用契约：start_ms/end_ms 使用原视频的 local_ms，不能从本窗口重新计时。
每个步骤引用的每一张图片都必须满足 start_ms <= 图片.local_ms <= end_ms。
单帧步骤可以 start_ms=end_ms=该帧的精确 local_ms；引用多帧时，起止时间必须覆盖全部引用帧。
例如描述某帧中手持工具、随后另一帧中工具消失，若引用两帧，不能把步骤起止都写成第一帧时间。
不要凭空扩展动作持续时间：分别写单帧观察，或明确写两张抽样画面之间的可见差异；不把中间过程写成已全程观察。
输出前逐一核对 frame_ids 与 local_ms，包括用于前后比较的引用；没有支持的步骤应删除或写明不确定，不能虚构帧。
"""


OBJECT_IDENTITY_RULES = """物体辨识规则，适用于当前步骤、下一步骤、交互列表、物理变化及摘要的全部描述：
先区分被操作的物体、外包装和包装内物。白色外观、透明袋或包装鼓起，不能单独证明里面是粉末、液体或试剂；纸张、耗材和包装反光都可能造成相似外观。
有可读标签时，可以写“标有某文字的包装”，不能把标签名称当作已验证的实际内容物。内容形态须有独立可见依据，例如散落颗粒、连续液面或被抽出的纸片；没有依据时写“包装”或“内容物不可辨认”，不得猜测物质成分。
同类对象可能有多个实例。仅描述与手或工具实际交互的实例；把背景物体与被操作物体分开，不得根据邻近、颜色相似或既有CV类别补写接触。
逐只手核对直接接触：双手出现在画面中不等于双手都在操作同一对象。只看见一只手按键时就描述该只手，不能概括成“双手按键”或给另一只手补记面板接触；左右无法辨明时使用 unknown。
"""


OPERATION_DESCRIPTION_RULES = """操作级理解与写作规则：
分析主体是实验员的具体操作，不是实验室场景或 CV 检测类别。桌面摆设、设备出现、手进入画面本身不是实验步骤。
operation_title 用具体动词与实际操作对象命名，例如“拿起试管架”“放下样品瓶”；只有画面直接支持时才可写“旋开瓶盖”“向烧杯倒入液体”。不能用“物体移动”“手部接触”“称量场景”替代具体操作，也不能为名称好看而推断称量、移液、开盖等功能性动作。
current_step 清楚描述操作者对哪个实例做了什么、操作如何进行、可见结果怎样；如果只看到进行过程而没看到完成，就明确写未看到完成。一个字段内有多个可见动作时按发生顺序分句，保留遮挡、实例不明和否定条件。不要复制 CV 类别、候选通过/拒绝的审计结论、轨迹 ID 或解码时间校验记录；这些写入当前响应合同已有的分析依据或 uncertainties 字段，不得为此新增合同字段。
physical_change 只写操作前后的可见状态；next_step 只记录后续画面已经发生的动作或明确标注的推测，不把实验常识、典型流程或预期目的当作实际发生步骤。不知道下一步就写未知。
"""


EVENT_SYSTEM_PROMPT = OBJECT_IDENTITY_RULES + OPERATION_DESCRIPTION_RULES + """你是化学湿实验视频证据审计模型。你会收到第一人称和第三人称片段的时序采样，以及传统 CV 证据。
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
  "operation_title": "具体动词与被操作对象，不能只写检测类别",
  "current_step": "当前这一步做什么，细到手、对象、状态",
  "next_step": "根据尾部状态可直接观察或谨慎推断的下一步；不能判断写未知",
  "next_step_evidence": {"status":"observed/inferred/unknown", "reason":"下一步的画面依据或无法判断原因", "evidence_event_ids":["当前事件ID"]},
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
evidence_verdict 的含义只针对候选动作本身：至少一个视角直接清晰证明则为 confirmed；候选动作未被证明但另一个动作被证明才用 relabel_suggested；证据不足用 uncertain；直接反证用 rejected。candidate_action_support_by_view 只评价输入候选；confirmed_action_support_by_view 只评价 action_type_confirmed。视角主体冲突单独写入 cross_view_consistency 和 uncertainties，不能抹去单个视角对动作本体的直接证明。next_step_evidence.status=observed 只允许下一动作已经在时序尾部直接出现；尚未出现但可谨慎预测必须写 inferred；无依据必须写 unknown。预测不得伪装成已观察事实。
对 container_state_change，只有在同一个视角的时序中同时看到动作前容器状态、动作后容器状态，且开启/关闭/盖合/解除盖合的状态转换已完成，才能把 action_proof 中三个 container_* 字段都填 true 并确认该类。这里三个 container_* 字段表示“至少一个视角存在完整直接闭环”，不要求另一个视角重复拍到；若一个视角完整证明而另一视角模糊、遮挡或拍到并行动作，必须保留 container_state_change，把完整证明视角的 confirmed_action_support_by_view 标 true，并把另一视角的不完整性写入 cross_view_consistency/uncertainties，不得因此把三个字段改为 false 或降级为 hand_object_contact。只看到手持瓶盖对准瓶口、操作进行中、所有视角末帧仍被手遮挡，不算已完成的容器状态转换；此时才应按可见事实改为 hand_object_contact 或 object_movement。
禁止输出 Markdown。"""


GROUP_SYSTEM_PROMPT = OBJECT_IDENTITY_RULES + OPERATION_DESCRIPTION_RULES + """你是化学湿实验有界视频的步骤级理解与命名模型。输入是一个有界实验组按时间顺序采样的第一/第三人称对齐故事板、原子边界和传统 CV 事件。
任务：
1. 判断这是单个独立实验，还是多个无中断承接的连续实验；时间接近本身不代表连续，必须看到人员、对象或物理状态承接。
2. 给出具体、保守的实验名称，例如“固体称量实验”“移液实验”“固体称量与移液连续实验”，禁止使用“实验片段一”等泛名。
3. 输出从开始到结束的细粒度步骤，说明当前做什么、下一步做什么、物体和状态变化。
4. 不得虚构试剂化学名称、质量、体积、面板读数或实验目的；看不清就写未知。
5. 步骤的单位是一个可辨识的实验操作，不是一个实验室场景或一条 CV 候选。同一操作的多条证据可以共同支持一个步骤；不同操作不能只因属于同一个 CV 类别而合成一句概述。按实际观察到的先后写清具体动作、对象、时间与结果，不用预设的标准实验流程补齐未见动作。
6. 稀疏采样之间存在未观察区间时，在 uncertainties 中说明步骤覆盖缺口，不声称已完整还原每一步。保留所有支持判断的事件与视角编号；编号属于证据关联，不能代替操作描述。
7. 输入边界只是候选取材范围，不是实验已经开始或结束的事实。打开包装、整理称量纸、取工具等准备步骤结束，不代表整场实验结束。末帧仍在操作或准备后续实验时，end_complete 必须为 false，localized_rescan_needed 为 true。不能用“片段内没有后续事件”“下一段间隔若干秒”“到达片段尾部”作为完成依据；看不到后续时应明确结束状态未知。
8. atomic_experiments 表示语义上的实验单元，不能照抄 CV 片段，也不能把开纸、开瓶等单步操作各当一场实验。连续实验链中可保留称量、配液、移液等不同单元及其时间范围；换台不能自动断链。单元名称应简明，完整操作写在 steps 中。时间边界只写画面支持的范围，不假定单元结束就代表整条实验链完成。completion_status 为 ongoing_at_recording_end 或 unresolved 时不得宣称链已完成。
输出单个 JSON 对象，字段固定为：
{
  "experiment_name": "中文具体实验名称",
  "experiment_name_en": "ASCII-English-Experiment-Name",
  "continuity_type_confirmed": "independent/continuous/uncertain",
  "continuity_reason": "可观察的连续/独立依据",
  "atomic_experiments": [{"name":"内部原子实验名称", "start_global_ms":0, "end_global_ms":0, "purpose_observable":"可观察目标或未知"}],
  "steps": [{"step_index":1, "operation_title":"具体动词与操作对象", "start_global_ms":0, "end_global_ms":0, "current_step":"具体操作过程与可见结果", "next_step":"下一动作或未知", "next_step_status":"observed/inferred/unknown", "supporting_event_ids":["事件ID"], "objects":["物体"], "physical_change":"状态变化", "supporting_views":["视角ID"], "confidence":0.0}],
  "overall_summary": "整个有界视频的实验内容",
  "boundary_assessment": {"start_complete":true, "end_complete":true, "start_reason":"起点证据", "end_reason":"终点证据", "localized_rescan_needed":false},
  "confidence": 0.0,
  "uncertainties": ["无法确认项"]
}
禁止输出 Markdown。"""


class _StrictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EventPhysicalChange(_StrictResponse):
    before: str
    after: str


class _ViewObservation(_StrictResponse):
    view_id: str
    observation: str


class _CandidateViewSupport(_StrictResponse):
    view_id: str
    supports_candidate_action: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class _ConfirmedViewSupport(_StrictResponse):
    view_id: str
    supports_confirmed_action: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class _HandObjectInteraction(_StrictResponse):
    hand: str
    object: str
    contact: str


class _ActionProof(_StrictResponse):
    proof_type: Literal[
        "visible_liquid_flow",
        "visible_liquid_level_change",
        "pour_with_visible_liquid_change",
        "pipette_closed_transfer_cycle",
        "pipette_operational_transfer_chain",
        "posture_only",
        "direct_other",
        "none",
    ]
    visible_liquid_or_level_change: bool
    source_contact_visible: bool
    withdrawal_or_transport_visible: bool
    target_contact_visible: bool
    release_or_plunger_change_visible: bool
    dual_role_cv_sequence_verified: bool
    container_before_state_visible: bool
    container_after_state_visible: bool
    container_state_transition_completed: bool
    reason: str


class _TemporalSupport(_StrictResponse):
    early: str
    middle: str
    late: str


class _SelectedKeyframeObservation(_StrictResponse):
    view_id: str
    decoded_global_ms: float = Field(ge=0, allow_inf_nan=False)
    directly_interacting_objects: list[str]
    observation: str
    uncertainties: list[str]


class _NextStepEvidence(_StrictResponse):
    status: Literal["observed", "inferred", "unknown"]
    reason: str
    evidence_event_ids: list[str]


class _SpeechInterpretation(_StrictResponse):
    summary: str
    relation_to_visual: Literal["consistent", "contradiction", "unrelated", "uncertain", "no_speech"]
    referenced_segment_ids: list[str]
    uncertainties: list[str]


class _SpeechResponse(_StrictResponse):
    speech_interpretation: _SpeechInterpretation


class _EventResponse(_StrictResponse):
    speech_interpretation: _SpeechInterpretation | None = None
    # Older saved responses remain readable; new requests explicitly ask for a
    # concrete operation title separate from the CV action ontology.
    operation_title: str | None = None
    current_step: str
    next_step: str
    next_step_evidence: _NextStepEvidence
    action_type_confirmed: Literal[
        "hand_object_contact",
        "object_movement",
        "pipette_transfer_operation",
        "liquid_movement",
        "container_state_change",
        "device_panel_operation",
        "unknown",
    ]
    objects: list[str]
    hand_object_interactions: list[_HandObjectInteraction]
    physical_change: _EventPhysicalChange
    per_view_observations: list[_ViewObservation]
    candidate_action_support_by_view: list[_CandidateViewSupport]
    confirmed_action_support_by_view: list[_ConfirmedViewSupport]
    action_proof: _ActionProof
    cross_view_consistency: Literal[
        "consistent", "partial", "conflict", "single_view"
    ]
    evidence_verdict: Literal[
        "confirmed", "relabel_suggested", "uncertain", "rejected"
    ]
    temporal_support: _TemporalSupport
    selected_keyframe_observations: list[_SelectedKeyframeObservation]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: list[str]


class _AtomicExperimentResponse(_StrictResponse):
    name: str
    start_global_ms: float
    end_global_ms: float
    purpose_observable: str


class _GroupStepResponse(_StrictResponse):
    speech_segment_ids: list[str] = Field(default_factory=list)
    step_index: int = Field(ge=1)
    operation_title: str | None = None
    start_global_ms: float
    end_global_ms: float
    current_step: str
    next_step: str
    next_step_status: Literal["observed", "inferred", "unknown"]
    supporting_event_ids: list[str]
    objects: list[str]
    physical_change: str
    supporting_views: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class _BoundaryAssessment(_StrictResponse):
    start_complete: bool
    end_complete: bool
    start_reason: str
    end_reason: str
    localized_rescan_needed: bool


class _GroupResponse(_StrictResponse):
    speech_interpretation: _SpeechInterpretation | None = None
    experiment_name: str
    experiment_name_en: str
    continuity_type_confirmed: Literal["independent", "continuous", "uncertain"]
    continuity_reason: str
    atomic_experiments: list[_AtomicExperimentResponse]
    steps: list[_GroupStepResponse]
    overall_summary: str
    boundary_assessment: _BoundaryAssessment
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: list[str]


class _OperationStepResponse(_StrictResponse):
    operation_title: str = Field(min_length=1, max_length=100)
    current_step: str = Field(min_length=1, max_length=800)
    supporting_event_ids: list[str] = Field(min_length=1)


class _OperationResponse(_StrictResponse):
    steps: list[_OperationStepResponse]


def _validate_response_payload(
    payload: dict[str, Any],
    response_kind: Literal["event", "group", "speech", "operations"] | None,
) -> dict[str, Any]:
    if response_kind is None:
        return payload
    contract = {"event": _EventResponse, "group": _GroupResponse, "speech": _SpeechResponse, "operations": _OperationResponse}[response_kind]
    return contract.model_validate(payload).model_dump(mode="json")


FINAL_GROUP_SYSTEM_PROMPT = GROUP_SYSTEM_PROMPT + """

这是事件级语义审核后的最终步骤精炼，不是候选发现。输入事件均为已经确认或严格重标的最终关键事件：
1. 步骤中的动作事实必须能对应至少一个输入 final_adjudicated_event；故事板只用于描述参与对象的可见状态，不得从“手在设备附近”新增设备面板操作，不得从“工具与容器同框”新增液体移动或容器开合。
2. absent_action_classes 中列出的类别在本实验组没有被确认，步骤、摘要、实验名称和下一步都不得把这些类别写成已经发生的事实。尤其 absent_action_classes 含 device_panel_operation 时，禁止写按键、按压面板、操作按钮、读取或确认读数；含 liquid_movement 时，禁止写吸液、排液、加液、倾倒或液体转移；即使瓶体横放、翻倒或姿态发生变化，也只能写“瓶体姿态改变”，不能使用“倾倒”这个会宣称功能性液体动作的词。若 pipette_transfer_operation 已确认但 liquid_movement 未确认，可写“移液器从源容器移动并进入目标容器”或“源到目标移液器操作（液体状态不可见）”，不得升级为液体已转移。含 pipette_transfer_operation 时若该类别 absent，禁止写移液操作、源到目标移液器操作或完整移液流程。含 container_state_change 时，禁止写开盖、合盖、旋开或旋紧。
3. 可以写“手接触天平”“移动瓶盖”等已确认的较低层动作，但不得升级成未确认的功能性操作。
4. 时间范围必须来自 final_adjudicated_events，不得把相邻步骤重叠扩展到没有最终事件支持的动作。
输出结构仍严格使用上面的 JSON 合同。
"""


def _image_data_url(
    path: Path,
    *,
    max_edge: int = 0,
    jpeg_quality: int = 85,
) -> str:
    """Build a bounded wire image without changing the evidence artifact.

    Full-resolution JPEGs remain in the archive. Only the in-memory request
    representation is resized, which prevents slow links and retries from
    repeatedly uploading multi-megabyte evidence payloads.
    """

    payload = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    edge_limit = max(0, int(max_edge))
    if edge_limit > 0:
        image = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if image is not None and max(image.shape[:2]) > edge_limit:
            scale = edge_limit / float(max(image.shape[:2]))
            resized = cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            if suffix == ".png":
                encoded_ok, encoded_image = cv2.imencode(
                    ".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, 6]
                )
            else:
                encoded_ok, encoded_image = cv2.imencode(
                    ".jpg",
                    resized,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        min(100, max(1, int(jpeg_quality))),
                    ],
                )
            if encoded_ok:
                payload = encoded_image.tobytes()
    encoded = base64.b64encode(payload).decode("ascii")
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
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
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
            # Preserve the distinction between two explicitly described hands.
            # Deleting 左/右 alone turns “左、右手” into “左、手” and can make
            # “左手仍接触、右手已松开” appear internally contradictory.
            text = re.sub(r"(?:左(?:手)?[、和与及]?右手|右(?:手)?[、和与及]?左手)(?!边|侧|柄)", "双手", value)
            hands = re.findall(r"(左|右)手(?!边|侧|柄)", text)
            if len(set(hands)) == 2:
                first = hands[0]
                text = re.sub(r"(左|右)手(?!边|侧|柄)",
                              lambda match: "一只手" if match[1] == first else "另一只手", text)
            else:
                text = re.sub(r"(?:左|右)手(?!边|侧|柄)", "手", text)
            text = text.replace("手手指", "手指")
            text = re.sub(r"\b(?:left|right)[ -]hand\b(?![ -]side)", "hand", text, flags=re.IGNORECASE)
            if text != value:
                originals[path] = value
            return text
        if isinstance(value, list):
            return [neutralize(item, f"{path}/{index}") for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: neutralize(item, f"{path}/{key}") for key, item in value.items()}
        return value

    for key in ("operation_title", "current_step", "next_step", "objects", "physical_change",
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


def _read_streamed_chat(response: httpx.Response, *, deadline: float, max_bytes: int = 8 * 1024 * 1024) -> dict:
    """Collect the final answer only; incomplete streams never become evidence."""
    buffer = ""
    total_bytes = 0
    parts: list[str] = []
    result: dict[str, Any] = {"choices": []}
    finish_reason = None
    done = False
    is_sse = "text/event-stream" in response.headers.get("content-type", "").lower()
    for text in response.iter_text():
        if time.perf_counter() > deadline:
            raise httpx.ReadTimeout("视觉模型流式响应超过总等待预算", request=response.request)
        total_bytes += len(text.encode("utf-8"))
        if total_bytes > max_bytes:
            raise ValueError("视觉模型响应超过有界接收大小，未接受不完整结果。")
        buffer += text
        if not is_sse:
            continue
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            if not data:
                continue
            item = json.loads(data)
            if not isinstance(item, dict):
                raise ValueError("视觉模型流式事件必须是 JSON 对象。")
            if item.get("error"):
                raise ValueError("视觉模型流式响应报告错误，未接受部分结果。")
            if item.get("usage") is not None and not isinstance(item["usage"], dict):
                raise ValueError("视觉模型流式用量格式无效。")
            for name in ("id", "model", "usage"):
                if item.get(name) is not None:
                    result[name] = item[name]
            choices = item.get("choices") or []
            if not isinstance(choices, list):
                raise ValueError("视觉模型流式答案列表格式无效。")
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError("视觉模型流式答案格式无效。")
                if choice.get("index", 0) != 0:
                    raise ValueError("视觉模型返回多个答案，无法唯一封存。")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise ValueError("视觉模型流式增量格式无效。")
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise ValueError("视觉模型流式答案格式无效。")
                    parts.append(content)
                # Provider reasoning is intentionally not retained or displayed.
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        if done:
            break
    if not is_sse:
        return json.loads(buffer)
    if not done or not finish_reason:
        raise ValueError("视觉模型流式连接提前结束，未接受不完整结果。")
    result["choices"] = [{"message": {"content": "".join(parts)}, "finish_reason": finish_reason}]
    result["visioncortex_transport"] = {"mode": "stream", "complete": True, "received_bytes": total_bytes}
    return result


class ArkAnalyzer:
    """Historical class name; the transport now supports several vision providers."""
    def __init__(self, config: dict[str, Any]):
        self.runtime_config = config
        self.config = config["mllm"]
        self.api_key = model_api_key(self.config)
        self.enabled = bool(self.config["enabled"])
        self._failure_lock = threading.Lock()
        self._transport_failure_count = 0
        self._failure_circuit_open = False
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

    def _request_image_transport(self) -> dict[str, Any]:
        return {
            "maximum_edge_pixels": int(
                self.config.get("request_image_max_edge", 0)
            ),
            "jpeg_quality": int(
                self.config.get("request_image_jpeg_quality", 85)
            ),
            "source_artifacts_mutated": False,
        }

    def _call(
        self,
        system_prompt: str,
        metadata: dict[str, Any],
        image_paths: Sequence[tuple[str, Path]],
        *,
        max_images: int | None = None,
        response_kind: Literal["event", "group", "speech", "operations"] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "uncertainties": ["多模态分析已在配置中关闭"]}
        if not self.api_key:
            return {
                "status": "skipped_missing_api_key",
                "uncertainties": [f"未设置环境变量 {self.config['api_key_env']}，未调用模型"],
            }
        with self._failure_lock:
            circuit_open = self._failure_circuit_open
            transport_failure_count = self._transport_failure_count
        if circuit_open:
            return {
                "status": "skipped_failure_circuit_open",
                "error": "Multimodal transport failure circuit is open for this stage",
                "uncertainties": [
                    "多模态服务连续不可用；当前事件进入可重试机器隔离"
                ],
                "latency_seconds": 0.0,
                "attempts": 0,
                "transport_failure_count": transport_failure_count,
                "request_image_transport": self._request_image_transport(),
                "usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "cached_input_tokens": None,
                    "server_reported": False,
                },
            }
        scene_timing_contract = (
            metadata.get("schema_version") == "visioncortex-device-day/1"
            and "frames" in metadata and "start_ms" in metadata and "end_ms" in metadata
        )
        if scene_timing_contract:
            system_prompt = system_prompt + "\n" + SCENE_TIME_REFERENCE_RULES
        wire_metadata, speech_aliases, speech_transport = compact_speech_metadata(metadata)
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": json.dumps(wire_metadata, ensure_ascii=False, separators=(",", ":"))}
        ]
        image_limit = int(max_images or self.config["max_images_per_event"])
        selected = list(image_paths)
        if len(selected) > image_limit:
            raise ValueError(
                "MLLM evidence allocation would silently discard images: "
                f"prepared {len(selected)}, configured limit {image_limit}"
            )
        for label, path in selected:
            content.append({"type": "input_text", "text": compact_speech_label(label, {full: short for short, full in speech_aliases.items()})})
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(
                        path,
                        max_edge=int(
                            self.config.get("request_image_max_edge", 0)
                        ),
                        jpeg_quality=int(
                            self.config.get("request_image_jpeg_quality", 85)
                        ),
                    ),
                    "detail": "high",
                }
            )
        url, request = build_vision_request(self.config, system_prompt, content)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        started = time.perf_counter()
        attempt_receipts: list[dict[str, Any]] = []

        def attempt_usage() -> dict[str, Any]:
            reported = [item.get("usage") or {} for item in attempt_receipts]
            summary = {}
            for field in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
                values = [item[field] for item in reported if item.get(field) is not None]
                summary[field] = sum(values) if values else None
            summary["server_reported"] = bool(reported) and all(item.get("server_reported") for item in reported)
            summary["unknown_attempt_count"] = sum(item.get("total_tokens") is None for item in reported)
            summary["scope"] = "all_transport_attempts; unknown usage is not zero"
            return summary

        for attempt in range(int(self.config["max_retries"])):
            attempt_started = time.perf_counter()
            receipt: dict[str, Any] = {"attempt": attempt + 1, "status": "pending",
                                       "started_at": datetime.now(timezone.utc).isoformat(),
                                       "retry_wait_seconds": 0.0}
            attempt_receipts.append(receipt)
            try:
                from .provider_control import provider_request
                with provider_request(self.runtime_config, 'understanding', self.config):
                    if request.get("stream"):
                        # The read timeout bounds inactivity; the deadline also stops
                        # a provider that keeps sending data without finishing.
                        stream_budget = max(1.0, float(self.config.get("stream_total_timeout_seconds", 600)))
                        timeout = httpx.Timeout(
                            float(self.config["timeout_seconds"]), connect=15.0, write=30.0, pool=15.0,
                            read=float(self.config.get("stream_read_timeout_seconds", self.config["timeout_seconds"])))
                        with self.client.stream("POST", url, headers=headers, json=request, timeout=timeout) as response:
                            if response.is_error:
                                raise httpx.HTTPStatusError(
                                    _http_error_message(response, self.api_key),
                                    request=response.request, response=response)
                            payload = _read_streamed_chat(response, deadline=attempt_started + stream_budget)
                        receipt["transport"] = payload.get("visioncortex_transport", {"mode": "json"})
                        receipt["stream_total_timeout_seconds"] = stream_budget
                    else:
                        response = self.client.post(url, headers=headers, json=request)
                        if response.is_error:
                            body = response.text[:1000].replace(self.api_key, "[REDACTED]")
                            raise httpx.HTTPStatusError(
                                f"{response.status_code} from multimodal API: {body}",
                                request=response.request,
                                response=response,
                            )
                        payload = response.json()
                    receipt.update(request_id=payload.get("id"), response_model=payload.get("model"), usage=_usage(payload))
                    choices = payload.get("choices") or []
                    if payload.get("status") == "incomplete" or any(item.get("finish_reason") in {"length", "content_filter"} for item in choices):
                        raise ValueError("模型响应被截断或过滤，不能作为完整结构化结果使用。")
                    result = _validate_response_payload(
                        _parse_json(_extract_text(payload)), response_kind
                    )
                    if response_kind:
                        result = bind_speech_result(expand_speech_references(result, speech_aliases), metadata.get("speech_context"))
                    if speech_transport is not None:
                        result["speech_input_transport"] = speech_transport
                    receipt.update(status="completed", ended_at=datetime.now(timezone.utc).isoformat(), latency_seconds=round(time.perf_counter() - attempt_started, 6))
                    result.update(
                        {
                            "status": "completed",
                            "model": self.config["model"],
                            "provider": self.config.get("provider", "volcengine"),
                            "api_protocol": self.config.get("api_protocol", "ark_responses"),
                            "request_id": payload.get("id"),
                            "response_model": payload.get("model"),
                            "usage": attempt_usage(),
                            "attempt_receipts": attempt_receipts,
                            "latency_seconds": round(time.perf_counter() - started, 6),
                            "attempts": attempt + 1,
                            "response_contract": (
                                f"visioncortex-{response_kind}-mllm-response/1"
                                if response_kind
                                else None
                            ),
                            "request_image_transport": self._request_image_transport(),
                        }
                    )
                    if scene_timing_contract:
                        result["scene_timing_contract"] = {
                            "version": "visioncortex-scene-time-references/1",
                            "instructions": SCENE_TIME_REFERENCE_RULES,
                            "validation_relaxed": False,
                        }
                    with self._failure_lock:
                        self._transport_failure_count = 0
                    result = json.loads(json.dumps(result, ensure_ascii=False).replace(self.api_key, "[REDACTED]"))
                    return normalize_uncalibrated_hand_identity(result)
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                receipt.update(status="failed", ended_at=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__, latency_seconds=round(time.perf_counter() - attempt_started, 6))
                receipt["http_status"] = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                if isinstance(exc, (ValidationError, json.JSONDecodeError)) and response_kind:
                    # A format retry still uses every original image and the
                    # same strict validator. Report schema locations, never the
                    # provider's untrusted raw answer or credential-bearing input.
                    contract = {"event": _EventResponse, "group": _GroupResponse, "speech": _SpeechResponse, "operations": _OperationResponse}[response_kind]
                    issues = [
                        {"path": ".".join(str(part) for part in item["loc"]), "type": item["type"]}
                        for item in exc.errors(include_input=False, include_context=False)[:20]
                    ] if isinstance(exc, ValidationError) else [{"type": "invalid_json"}]
                    feedback = {
                        "instruction": "上次响应未通过格式校验。请重新核查相同画面证据，只返回符合下列 JSON Schema 的完整 JSON 对象，不要添加额外字段或 Markdown。证据不足仍应明确表达未知，不得为了通过格式校验改变事实结论。",
                        "format_errors": issues,
                        "json_schema": contract.model_json_schema(),
                    }
                    url, request = build_vision_request(
                        self.config, system_prompt,
                        content + [{"type": "input_text", "text": json.dumps(feedback, ensure_ascii=False)}],
                    )
                    receipt["format_correction_requested"] = attempt + 1 < int(self.config["max_retries"])
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
                    wait_started = time.perf_counter()
                    time.sleep(max(0.0, delay))
                    receipt["retry_wait_seconds"] = round(time.perf_counter() - wait_started, 6)
        circuit_opened = False
        if isinstance(last_error, httpx.HTTPError):
            threshold = max(
                1,
                int(
                    self.config.get(
                        "failure_circuit_breaker_threshold", 4
                    )
                ),
            )
            with self._failure_lock:
                self._transport_failure_count += 1
                if self._transport_failure_count >= threshold:
                    self._failure_circuit_open = True
                circuit_opened = self._failure_circuit_open
                transport_failure_count = self._transport_failure_count
        return {
            "status": "failed",
            "error": f"{type(last_error).__name__}: {last_error}".replace(self.api_key, "[REDACTED]"),
            "http_status": last_error.response.status_code if isinstance(last_error, httpx.HTTPStatusError) else None,
            "provider": self.config.get("provider", "volcengine"),
            "model": self.config.get("model"),
            "api_protocol": self.config.get("api_protocol", "ark_responses"),
            "uncertainties": ["多模态调用失败；保留 CV 与跨视角审计结果，不伪造模型理解"],
            "latency_seconds": round(time.perf_counter() - started, 6),
            "attempts": attempt + 1,
            "transport_failure_count": transport_failure_count,
            "failure_circuit_open": circuit_opened,
            "request_image_transport": self._request_image_transport(),
            "usage": attempt_usage(),
            "attempt_receipts": attempt_receipts,
        }

    def analyze_event(
        self, event: EvidenceEvent, image_paths: Sequence[tuple[str, Path]],
        *, speech_context: dict | None = None,
    ) -> dict[str, Any]:
        action_limits = self.config.get("max_images_per_event_by_action") or {}
        max_images = int(
            action_limits.get(
                event.action_type.value,
                self.config["max_images_per_event"],
            )
        )
        return self._call(
            prompt_with_speech(EVENT_SYSTEM_PROMPT, speech_context),
            {
                **({"speech_context": speech_context} if speech_context is not None else {}),
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
            response_kind="event",
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
        speech_context: dict | None = None,
    ) -> dict[str, Any]:
        return self._call(
            prompt_with_speech(system_prompt, speech_context),
            {
                **({"speech_context": speech_context} if speech_context is not None else {}),
                "group_id": group.group_id,
                "boundary_scope": "candidate video interval; experiment completion is not established by CV grouping",
                "workflow_kind": group.workflow_kind,
                "workflow_units": group.workflow_units,
                "completion_status": group.completion_status,
                "completion_reason": group.completion_reason,
                "view_timeline": group.view_timeline,
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
                        **({"reviewed_operation": {key: (event.model_understanding or {}).get(key) for key in (
                            "operation_title", "current_step", "physical_change", "next_step", "next_step_evidence",
                            "per_view_observations", "confidence", "uncertainties")}}
                           if final_adjudicated else {}),
                    }
                    for event in events
                    if event_is_formal(event)
                ],
                **(
                    {
                        "absent_action_classes": sorted(
                            {item.value for item in ActionType}
                            - {
                                event.action_type.value
                                for event in events
                                if event_is_formal(event)
                            }
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
            max_images=int(
                self.config.get(
                    "max_images_per_group",
                    self.config["max_images_per_event"],
                )
            ),
            response_kind="group",
        )


# Backward-compatible name used by archive.py.
ArkStepAnalyzer = ArkAnalyzer

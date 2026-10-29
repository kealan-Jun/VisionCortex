"""Framework-neutral JSON tools; every operation uses the existing application core."""

import base64
import binascii
import uuid

from fastapi import Body, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Photo(Arguments):
    camera_id: str = Field(min_length=1, max_length=100)
    image_base64: str = Field(min_length=1, max_length=16 * 1024 * 1024)


class BindingId(Arguments):
    binding_id: uuid.UUID


class JobId(Arguments):
    job_id: uuid.UUID


def install_tools(app, core):
    # Reuse original request contracts, while rejecting misspelled tool arguments.
    class Bind(core["BindingRequest"]):
        model_config = ConfigDict(extra="forbid")

    class EnterScene(core["SceneEntry"]):
        model_config = ConfigDict(extra="forbid")

    class Ocr(core["OcrRequest"]):
        model_config = ConfigDict(extra="forbid")

    def scan_photo(args):
        if not args.camera_id.strip():
            raise HTTPException(422, "camera_id must not be blank")
        try:
            raw = base64.b64decode(args.image_base64, validate=True)
        except (ValueError, binascii.Error):
            raise HTTPException(422, "image_base64 must contain plain base64 image bytes") from None
        return core["scan_image"](raw, "uploaded_photo", args.camera_id)

    registry = {
        "get_field_state": (
            Arguments,
            lambda a: core["state"](),
            "查询已登记仪器、场景、当前绑定及相机/OCR状态。",
        ),
        "capture_and_scan": (
            Arguments,
            lambda a: core["camera_capture"](),
            "从服务配置的挂脖相机取得最新照片并识别场景和仪器二维码；返回 capture_id、scan_id、匹配仪器及帧来源。无二维码也保留照片，可用于已有绑定的面板 OCR。",
        ),
        "scan_photo": (
            Photo,
            scan_photo,
            "识别调用方提供的照片中的场景和仪器二维码并保存原图证据。传图片的纯 base64 和来源 camera_id，不传路径或 URL。",
        ),
        "enter_scene": (
            EnterScene,
            lambda a: core["enter_scene"](a),
            "用实际解码的场景码确认相机进入场景；多台相机可同时进入同一场景，各自独立留证。同相机、场景及人员重复确认幂等。",
        ),
        "bind_instrument": (
            Bind,
            lambda a: core["bind"](a),
            "将实际解码的仪器绑定到照片相机和实际使用人员。同相机支持多仪器；人员变化在事务内交接所有有效使用时段并保留旧归属。提供request_id保证重试，expected_binding_id或expected_revision防并发覆盖；refresh需要request_id。同人员同内容重复扫码幂等。仪器在其他相机使用时仍需双方确认交接。",
        ),
        "read_panel": (
            Ocr,
            lambda a: core["submit_ocr"](a),
            "用户要求查看面板读数时，基于已有 capture_id 提交识别，binding_id 可省略；未绑定结果不推断仪器归属；来源照片中真实解码的二维码身份单独保留。先用常驻 PaddleOCR；本次请求达到配置的无数字等待阈值（默认 5 秒）仍无数字时使用视觉兜底。crop 为原图像素 [x,y,width,height]，省略则整图。返回 job_id，需查询结果。",
        ),
        "read_saved_panel": (
            core["SavedPhotoRequest"],
            lambda a: core["read_saved_panel"](a),
            "现有 Agent 已把语音照片保存到 NAS 后调用。传拍照结果的具体 image_path 或 photo 原图和回执，两者只能选一个。binding_id 可省略，未绑定也识别并留存；提供时严格校验绑定。目录监控独立于绑定，自动关联拍照时已有的有效绑定，无法关联时不推断仪器归属，来源照片中真实解码的二维码身份单独保留。先运行常驻 OCR；本次请求达到配置的无数字等待阈值（默认 5 秒）时最多一次视觉兜底。同一照片和选框重复提交返回原 job_id；timing 保存各阶段时间和延迟。随后查询并统一回答读数。",
        ),
        "get_panel_result": (
            JobId,
            lambda a: core["get_job"](a.job_id),
            "查询识别任务。completed 时读取 lines。并行照片模式的 running 且 result_available=true 时可先展示初读，标记复核中，并继续查询同一 job_id；result_revision 推进后原位更新同条读数，不追加第二条识别记录。confidence、polygon 可能为 null，不能补造。queued/running 时稍后查询原 job_id；failed/interrupted/cancelled 时报告未完成，不重新提交、不编造读数。后台 local_ocr/fallback 保留审计来源；parallel_review 表示模型复核状态，不等于人工确认。",
        ),
        "end_instrument_binding": (
            BindingId,
            lambda a: core["end_binding"](a.binding_id),
            "结束指定仪器使用绑定；重复结束保持同一结果。",
        ),
    }

    from field_recognition.measurements.photo_measurements import Decision, Revision, Rejection
    from field_recognition.bindings.instrument_ownership import HandoffRequest, HandoffDecision

    class MeasurementId(Arguments):
        measurement_id: uuid.UUID

    class Confirm(Decision):
        measurement_id: uuid.UUID

    class Revise(Revision):
        measurement_id: uuid.UUID

    class Reject(Rejection):
        measurement_id: uuid.UUID

    class Transfer(HandoffDecision):
        handoff_id: uuid.UUID
        action: str

    registry.update(
        {
            "get_measurement_draft": (
                MeasurementId,
                lambda a: core["measurement_actions"]["detail"](a.measurement_id),
                "读取本次测量草稿及每个字段的原图证据。",
            ),
            "revise_measurement_draft": (
                Revise,
                lambda a: core["measurement_actions"]["revise"](a.measurement_id, a),
                "由有权限的审核账号校正指定版本草稿，保留原文和修订原因。",
            ),
            "confirm_measurement_draft": (
                Confirm,
                lambda a: core["measurement_actions"]["confirm"](a.measurement_id, a),
                "审核账号确认当前版本；有冲突则拒绝，成功后读回同一条正式记录。",
            ),
            "reject_measurement_draft": (
                Reject,
                lambda a: core["measurement_actions"]["reject"](a.measurement_id, a),
                "审核账号拒绝草稿并记录原因。",
            ),
            "request_device_handoff": (
                HandoffRequest,
                lambda a: core["handoff_actions"]["request"](a),
                "接收人凭自己的新扫码证据申请交接，不会直接接管。",
            ),
            "decide_device_handoff": (
                Transfer,
                lambda a: core["handoff_actions"]["transition"](
                    a.handoff_id,
                    a.action,
                    HandoffDecision(**a.model_dump(exclude={"handoff_id", "action"})),
                ),
                "原使用人明确交出后，指定接收人确认接收；也可取消或拒绝。",
            ),
        }
    )

    @app.get("/api/tools")
    def list_tools():
        return {
            "version": "1.2",
            "transport": "http-json",
            "tools": [
                {"name": name, "description": desc, "input_schema": model.model_json_schema()}
                for name, (model, _, desc) in registry.items()
            ],
        }

    @app.post("/api/tools/{tool_name}")
    def call_tool(tool_name: str, arguments: dict = Body(...)):
        if tool_name not in registry:
            raise HTTPException(404, "Unknown tool")
        model, handler, _ = registry[tool_name]
        try:
            parsed = model.model_validate(arguments)
        except ValidationError as exc:
            # Do not echo image bytes or argument values in validation errors.
            raise HTTPException(
                422, exc.errors(include_input=False, include_context=False)
            ) from None
        result = handler(parsed)
        # Internal storage paths are not portable Agent image references.
        if isinstance(result, dict):
            result = {k: v for k, v in result.items() if k != "path"}
        return {"tool": tool_name, "result": result}

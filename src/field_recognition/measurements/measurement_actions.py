"""Shared measurement transactions for HTTP, Agent and compatibility clients."""

import copy
import json
import uuid
from fastapi import HTTPException
from field_recognition.measurements.photo_measurements import (
    Decision,
    Revision,
    Rejection,
    RULE_VERSION,
    UNITS,
    field_issues,
    blockers,
    instrument_completeness,
    summarize_candidates,
    verify_evidence,
    verify_ownership,
)


class MeasurementActions:
    def __init__(self, core):
        self.core = core

    def fetch(self, conn, mid):
        row = conn.execute(
            "SELECT document FROM photo_measurements WHERE id=?", (str(mid),)
        ).fetchone()
        if not row:
            raise HTTPException(404, "测量草稿不存在")
        from field_recognition.api.security import require_camera

        document = json.loads(row["document"])
        require_camera(document["camera_id"])
        return document

    def store(self, conn, group):
        group["revision"] += 1
        group["updated_at"] = self.core["now"]()
        group["instrument_completeness"] = instrument_completeness(group)
        group["blockers"] = blockers(group)
        conn.execute(
            "UPDATE photo_measurements SET status=?,document=? WHERE id=?",
            (group["status"], json.dumps(group), group["measurement_id"]),
        )
        return group

    def editable(self, group, body):
        if group["revision"] != body.revision:
            raise HTTPException(409, "草稿已更新，请刷新后核对最新版本")
        if group["status"] != "draft":
            raise HTTPException(409, "只有收齐照片并完成识别的草稿可修订或确认")

    def detail(self, mid: uuid.UUID):
        with self.core["db"]() as conn:
            return self.fetch(conn, mid)

    def revise(self, mid: uuid.UUID, body: Revision):
        from field_recognition.api.security import require_role, actor_name

        require_role("admin", "reviewer")
        body = body.model_copy(update={"actor": actor_name(body.actor)})
        with self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            group = self.fetch(conn, mid)
            self.editable(group, body)
            before = copy.deepcopy(group["fields"])
            fields = {f["field_id"]: f for f in group["fields"]}
            if len({f.field_id for f in body.fields}) != len(body.fields):
                raise HTTPException(422, "修订字段不可重复")
            for change in body.fields:
                field = fields.get(change.field_id)
                asset = self.core["get_instrument"](str(change.instrument_id))
                if not field or not asset or change.unit not in UNITS.get(change.name, set()):
                    raise HTTPException(422, "字段、仪器或指标单位无效")
                if field_issues(
                    change.name,
                    {"value": change.value, "unit": change.unit, "corrected": True},
                    asset,
                ):
                    raise HTTPException(422, "校正值不符合仪器登记的单位、量程或显示精度")
                field.update(
                    instrument=asset,
                    name=change.name,
                    value=change.value,
                    unit=change.unit,
                    corrected=True,
                    identity_basis="user_correction",
                    display_state=None,
                    display_value=str(change.value),
                )
            if not set(body.capture_times) <= {s["capture_id"] for s in group["sources"]}:
                raise HTTPException(422, "采集时间必须对应本次测量的照片")
            if body.experiment_context_ref:
                group["context"]["experiment_context_ref"] = body.experiment_context_ref
            group.setdefault("capture_time_corrections", {}).update(
                {cid: moment.isoformat() for cid, moment in body.capture_times.items()}
            )
            group["corrections"].append(
                {
                    "actor": body.actor,
                    "reason": body.reason,
                    "at": self.core["now"](),
                    "based_on_revision": body.revision,
                    "before_fields": before,
                    "changes": body.model_dump(mode="json"),
                }
            )
            return self.store(conn, group)

    def revalidate(self, mid: uuid.UUID, body: Decision):
        """Recheck a draft's frozen candidates without OCR or a human correction."""
        from field_recognition.api.security import require_role, actor_name

        require_role("admin", "reviewer")
        actor = actor_name(body.actor)
        with self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            group = self.fetch(conn, mid)
            self.editable(group, body)
            before = copy.deepcopy(group["fields"])
            previous_blockers = blockers(group)
            for field in group["fields"]:
                if not field.get("corrected"):
                    summarize_candidates(field)
            group.setdefault("validation_history", []).append(
                {
                    "actor": actor,
                    "at": self.core["now"](),
                    "based_on_revision": body.revision,
                    "rules": RULE_VERSION,
                    "before_fields": before,
                    "before_blockers": previous_blockers,
                    "basis": "revalidate_frozen_candidates_without_rerunning_ocr",
                }
            )
            return self.store(conn, group)

    def confirm(self, mid: uuid.UUID, body: Decision):
        from field_recognition.api.security import require_role, actor_name

        require_role("admin", "reviewer")
        body = body.model_copy(update={"actor": actor_name(body.actor)})
        with self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            group = self.fetch(conn, mid)
            if group["status"] == "confirmed":
                if group["confirmation"] != {
                    "actor": body.actor,
                    "based_on_revision": body.revision,
                }:
                    raise HTTPException(409, "本次测量已经确认")
                return json.loads(
                    conn.execute(
                        "SELECT document FROM experiment_records WHERE id=?",
                        (group.get("record_id", str(mid)),),
                    ).fetchone()[0]
                )
            self.editable(group, body)
            if group["record_mode"] != "production":
                raise HTTPException(409, "测试结果不能提交为生产实验记录")
            issues = blockers(group)
            if issues:
                raise HTTPException(409, {"message": "请先校正以下问题", "issues": issues})
            verify_evidence(self.core, group)
            verify_ownership(conn, group)
            group["status"] = "confirmed"
            group["confirmation"] = {"actor": body.actor, "based_on_revision": body.revision}
            group["confirmed_at"] = self.core["now"]()
            record_id = (
                str(mid)
                if group.get("record_revision", 1) == 1
                else str(uuid.uuid5(mid, "record-version:" + str(group["record_revision"])))
            )
            group["record_id"] = record_id
            record = copy.deepcopy(group) | {
                "record_id": record_id,
                "record_scope": "production_confirmed",
                "confirmed_fields": [
                    {k: f[k] for k in ("field_id", "instrument", "name", "value", "unit")}
                    | ({"display_state": f["display_state"]} if f.get("display_state") else {})
                    for f in group["fields"]
                ],
            }
            conn.execute(
                "INSERT INTO experiment_records(id,document) VALUES(?,?)",
                (record_id, json.dumps(record)),
            )
            self.store(conn, group)
        with self.core["db"]() as conn:
            return json.loads(
                conn.execute(
                    "SELECT document FROM experiment_records WHERE id=?", (record_id,)
                ).fetchone()[0]
            )

    def reject(self, mid: uuid.UUID, body: Rejection):
        from field_recognition.api.security import require_role, actor_name

        require_role("admin", "reviewer")
        body = body.model_copy(update={"actor": actor_name(body.actor)})
        with self.core["db"]() as conn:
            conn.execute("BEGIN IMMEDIATE")
            group = self.fetch(conn, mid)
            if group["revision"] != body.revision or group["status"] not in {"draft", "collecting"}:
                raise HTTPException(409, "草稿已更新或已经结束")
            group.update(
                status="rejected", rejection=body.model_dump() | {"at": self.core["now"]()}
            )
            return self.store(conn, group)

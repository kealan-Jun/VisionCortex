"""HTTP adapters for runs"""

from __future__ import annotations
from ..application.dependencies import ports
import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any
from fastapi import BackgroundTasks, HTTPException
from .. import ai_settings
from ..storage import (
    ARCHIVE_DIRECTORIES,
)

from ..application import (
    run_read_model,
    run_submission_service,
    runtime_host,
)

from fastapi import APIRouter

router = APIRouter()


def create_run_from_paths(
    payload: dict[str, Any], background_tasks: BackgroundTasks
) -> dict[str, Any]:
    return run_submission_service._create_run_from_paths(payload, background_tasks)


def _retain_retry_outputs(root: Path, attempt: int) -> None:
    """Rename previous derived output into history on the same share.

    Original media and input seals remain at their frozen paths. A fresh
    attempt must not mix old clip names, quarantine files or indexes with its
    own results. No media bytes are copied or removed.
    """
    json_root = root / "JSON-Config-Files"
    history = json_root / "Retry-Attempts" / str(attempt) / "Derived"
    inputs = {
        "Input-Manifests",
        "Retry-Attempts",
        "input_manifest.yaml",
        "run_manifest.json",
        "original_upload_manifest.json",
    }
    sources = [
        root / name
        for name in ARCHIVE_DIRECTORIES
        if name not in {"Original-Experiment-Videos", "JSON-Config-Files"}
    ]
    sources.extend([root / "Partial-Results", root / "run_status.json"])
    if json_root.is_dir():
        sources.extend(item for item in json_root.iterdir() if item.name not in inputs)
    moves = [
        (source, history / source.relative_to(root))
        for source in sources
        if source.exists()
    ]
    if any(destination.exists() for _, destination in moves):
        raise RuntimeError("上次复跑历史目录已存在同名产出，已停止以保留两份结果。")
    applied = []
    try:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            applied.append((source, destination))
        run_submission_service._write_json_atomic(
            history.parent / "output_relocation.json",
            {
                "schema_version": "visioncortex-retry-output-relocation/1",
                "previous_attempt": attempt,
                "source_media_moved": False,
                "copied_media_bytes": 0,
                "paths": [
                    {
                        "from": str(source.relative_to(root)),
                        "to": str(destination.relative_to(root)),
                    }
                    for source, destination in applied
                ],
            },
        )
    except OSError:
        for source, destination in reversed(applied):
            destination.replace(source)
        raise


def _refresh_in_progress(run_id: str) -> bool:
    return any(
        state.get("parent_run_id") == run_id
        and state.get("state") in {"queued", "running"}
        for state in runtime_host._runs.values()
    )


def refresh_stage(
    run_id: str, scope: str, target: str | None = None, revision: str | None = None
) -> dict[str, Any]:
    if scope not in {
        "understanding",
        "reports",
        "timeline",
        "capture_quality",
        "search",
        "operations",
        "result_check",
        "gap_review",
    }:
        raise HTTPException(422, "请选择需要更新的阶段")
    store = runtime_host._persistent_queue
    job = store.get_job(run_id) if store else None
    if not job or job["status"] not in {"failed", "completed"}:
        raise HTTPException(409, "原任务须已停止且保留持久任务记录")
    root = run_read_model._find_staging_run(runtime_host._settings(), run_id)
    if root is None or ports.read_current_release_pointer(root):
        raise HTTPException(409, "仅可刷新尚未正式发布的实验")
    settings = copy.deepcopy(job["payload"]["settings"])
    if scope in {"result_check", "gap_review"}:
        from ..result_review import inspect

        try:
            plan = inspect(root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                422, "当前保存记录不足以检查，请先核对实验产出"
            ) from exc
        if not revision or revision != plan["revision"]:
            raise HTTPException(409, "结果版本已改变，请刷新后重新检查")
        if scope == "gap_review":
            if target not in {w["window_id"] for w in plan["windows"]}:
                raise HTTPException(422, "请选择当前检查中的缺口区间")
            try:
                settings["mllm"] = ai_settings.reverified_job_mllm(settings)
            except (OSError, ValueError, RuntimeError) as exc:
                raise HTTPException(422, str(exc)) from exc
    if scope == "operations":
        from ..operation_review import bindings, GROUPS, EVENTS

        bound = bindings(root)
        if not bound.get(EVENTS) or not revision or revision != bound[GROUPS]:
            raise HTTPException(422, "缺少已审核事件，或步骤版本已经变化，请刷新后重试")
        try:
            settings["mllm"] = ai_settings.reverified_job_mllm(settings)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
    if scope == "understanding":
        groups = (
            run_read_model._read_json(
                root / "JSON-Config-Files/experiment_group_understanding.json", {}
            )
            or {}
        )
        if target is not None:
            from ..speech_refresh import input_path

            if not revision or len(revision) != 64:
                raise HTTPException(422, "片段重算需要当前理解版本")
            if target.startswith("group:"):
                group = next(
                    (
                        item
                        for item in groups.get("groups", [])
                        if "group:" + item["group_id"] == target
                    ),
                    None,
                )
                if group is None or not input_path(root, group["group_id"]).is_file():
                    raise HTTPException(
                        422, "该片段没有保留可核验的理解画面，请运行完整流程生成"
                    )
            elif (
                not target.startswith("recording:")
                or not target.split(":")[1].isdigit()
            ):
                raise HTTPException(422, "理解片段标识无效")
        elif (
            groups.get("groups")
            or not (root / "JSON-Config-Files/speech_understanding.json").is_file()
        ):
            raise HTTPException(422, "请选择一个保留了理解画面的实验片段")
        try:
            settings["mllm"] = ai_settings.reverified_job_mllm(settings)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc
    settings.setdefault("project", {})["semantic_cache_mode"] = "reuse"
    refresh_id = "refresh-" + uuid.uuid4().hex[:12]
    with runtime_host._lock:
        # Parent retry and child refresh cannot alter the same artifacts together.
        if _refresh_in_progress(run_id) or store.get_job(run_id)["status"] not in {
            "failed",
            "completed",
        }:
            raise HTTPException(409, "此实验已有阶段任务，请等待完成")
        state = {
            "state": "queued",
            "progress": 0.0,
            "parent_run_id": run_id,
            "refresh_scope": scope,
            "refresh_target": target,
            "message": "等待刷新所选阶段",
        }
        store.save_run(refresh_id, state)
        store.enqueue(
            refresh_id,
            "stage_refresh",
            {
                "settings": settings,
                "parent_run_id": run_id,
                "scope": scope,
                "target": target,
                "revision": revision,
            },
        )
        runtime_host._runs[refresh_id] = state
    runtime_host._queue_wakeup.set()
    return {
        "run_id": refresh_id,
        "parent_run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{refresh_id}",
        "scope": scope,
    }


def recovery_plan(run_id: str) -> dict[str, Any]:
    store = runtime_host._persistent_queue
    job = store.get_job(run_id) if store else None
    if job is None:
        raise HTTPException(404, "原持久任务不存在，请重新选择原输入")
    root = run_read_model._find_staging_run(runtime_host._settings(), run_id)
    if root is None or ports.read_current_release_pointer(root):
        raise HTTPException(409, "仅可恢复尚未正式发布的暂存任务")
    from ..operation_review import bindings, GROUPS, EVENTS

    bound = bindings(root)
    status = run_read_model._pipeline_status_from_root(root)
    stopped = job["status"] in {"failed", "completed"} and not _refresh_in_progress(
        run_id
    )
    retryable = stopped and (
        job["status"] == "failed"
        or (store.load_runs().get(run_id) or {}).get("state") == "partial"
    )
    model = job["payload"]["settings"].get("mllm") or {}
    model_ready = False
    model_message = "原任务未启用模型理解"
    if model.get("enabled"):
        try:
            ai_settings.reverified_job_mllm(job["payload"]["settings"])
            model_ready = True
            model_message = "已保存的连接验证可用于原任务；提交时再次检查"
        except (OSError, ValueError, RuntimeError) as exc:
            model_message = str(exc)
    from ..stage_recovery import index as recovery_index, DEPENDENCIES

    checkpoints = recovery_index(root).get("stages", {})
    identity = {
        "recovery": checkpoints,
        "run_id": run_id,
        "attempt": job["attempts"],
        "status": job["status"],
        "bindings": bound,
        "updated_at": status.get("updated_at"),
        "refresh_in_progress": _refresh_in_progress(run_id),
    }
    revision = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "run_id": run_id,
        "revision": revision,
        "group_revision": bound[GROUPS],
        "failed_stage": status.get("failed_stage") or status.get("stage"),
        "provider": model.get("provider"),
        "model": model.get("model"),
        "output_path": str(root),
        "retained_stage_count": len(run_read_model._stage_receipts_from_root(root)),
        "model_ready": model_ready,
        "model_check_message": model_message,
        "checkpoint_stages": list(checkpoints),
        "dependencies": DEPENDENCIES,
        "resume_effect": "已完成产物保留在原位置；执行时逐环节校验恢复点，重算失败、缺失、改变及受影响的后续环节。旧任务没有恢复点时使用原有缓存重新执行。",
        "actions": {
            "resume": retryable and (model_ready or not model.get("enabled")),
            "retry": retryable and (model_ready or not model.get("enabled")),
            "reports": stopped,
            "operations": stopped
            and model_ready
            and bool(bound[GROUPS] and bound[EVENTS]),
        },
        "cache_policy": "校验源文件、代码、配置和模型身份后复用；未通过校验的部分重新计算。",
        "retry_effect": "原输入不复制；已有派生产出移入 Retry-Attempts 历史目录，本轮重新生成可见成果。",
        "model_cost": "完整复跑与操作整理可能产生新的模型费用；缓存命中数与新增用量以执行回执为准，当前不作费用承诺。",
        "quality_policy": "操作整理复用已保存画面与已审核事件；不能补出未观察的动作或证明实验已结束。报告刷新不调用模型，未通过质量门时仅生成阶段报告。",
    }


def retry_run(
    run_id: str, revision: str | None = None, mode: str = "full"
) -> dict[str, Any]:
    if mode not in {"full", "resume"}:
        raise HTTPException(422, "请选择恢复未完成环节或完整复跑")
    if mode == "resume" and not revision:
        raise HTTPException(422, "恢复需要当前方案版本，请重新打开恢复方案")
    store = runtime_host._persistent_queue
    if store is None:
        raise HTTPException(503, "持久任务队列未就绪")
    job = store.get_job(run_id)
    if job is None:
        raise HTTPException(404, "原任务记录不存在，请重新选择原输入")
    if job["status"] != "failed" and not (
        job["status"] == "completed"
        and (store.load_runs().get(run_id) or {}).get("state") == "partial"
    ):
        raise HTTPException(409, "仅可复跑已停止的未完成任务")
    root = run_read_model._find_staging_run(runtime_host._settings(), run_id)
    if root is None or ports.read_current_release_pointer(root):
        raise HTTPException(409, "原任务暂存产出不可用，或已经正式发布")
    try:
        verified_mllm = ai_settings.reverified_job_mllm(job["payload"]["settings"])
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    # Preserve compact receipts before this attempt rewrites its current view.
    # No source media or derived video is copied for a retry.
    attempt_root = root / "JSON-Config-Files" / "Retry-Attempts" / str(job["attempts"])
    with runtime_host._lock:
        current = store.get_job(run_id)
        if (
            current is None
            or current["status"] != job["status"]
            or current["attempts"] != job["attempts"]
        ):
            raise HTTPException(409, "任务状态已改变，请刷新恢复方案")
        if revision is not None and recovery_plan(run_id)["revision"] != revision:
            raise HTTPException(409, "恢复方案已过期，请刷新后重试")
        if _refresh_in_progress(run_id):
            raise HTTPException(409, "此实验正在刷新阶段，请等待完成")
        try:
            retained = [
                "JSON-Config-Files/partial_delivery.json",
                "JSON-Config-Files/run_metrics.json",
                "JSON-Config-Files/pipeline_status.json",
                "JSON-Config-Files/key_material_semantic_failures.json",
                "JSON-Config-Files/experiment_group_semantic_failures.json",
                "Partial-Results/Partial-Evidence-Report.html",
            ]
            for relative in retained:
                source = root / relative
                if source.is_file():
                    destination = attempt_root / source.name
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
            if mode == "full":
                _retain_retry_outputs(root, int(job["attempts"]))
            state = store.retry_failed(
                run_id, verified_mllm=verified_mllm, resume_stages=mode == "resume"
            )
        except ValueError as exc:
            raise HTTPException(409, "任务已重新排队或尚未停止") from exc
        runtime_host._runs[run_id] = state
    runtime_host._queue_wakeup.set()
    return {
        "run_id": run_id,
        "state": "queued",
        "status_url": f"/api/runs/{run_id}",
        "source_copy_bytes": 0,
        "queue_persistence": "sqlite",
        "reuses_original_inputs": True,
        "cache_policy": "verified_reuse",
        "recovery_mode": mode,
    }


router.post("/api/runs/from-paths", status_code=202)(create_run_from_paths)


def list_runs() -> dict[str, Any]:
    return run_read_model.list_runs()


router.get("/api/runs")(list_runs)


def get_run(run_id: str) -> dict[str, Any]:
    return run_read_model.get_run(run_id=run_id)


router.get("/api/runs/{run_id}")(get_run)

router.post("/api/runs/{run_id}/refresh/{scope}", status_code=202)(refresh_stage)

router.get("/api/runs/{run_id}/recovery-plan")(recovery_plan)

router.post("/api/runs/{run_id}/retry", status_code=202)(retry_run)

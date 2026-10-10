"""Application execution host shared by Web, CLI and machine clients."""

from __future__ import annotations
from .dependencies import ports
import copy
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote
import yaml
from .contracts import ApplicationError, DeferredTasks
from .. import ai_settings
from ..identity import CONFIG_ENV
from ..device_day_service import DeviceDayService
from ..input_seal import build_input_seal, verify_input_seal, write_input_seal
from ..model_certification import audit_production_model_certification
from ..benchmark_registration import registration as benchmark_registration
from ..benchmark_registration import require_source_count
from ..nas_recordings import (
    enabled as directory_ingest_enabled,
    selection_path,
    validate_selection,
)
from ..pathing import archive_contains
from ..partial_delivery import (
    partial_result_available,
    write_partial_delivery,
)
from ..run_queue import DurableRunQueue, QueuedRunJob
from ..schemas import RunManifest
from ..storage import (
    initialize_nas_archive,
    run_staging_roots,
    safe_archive_name,
)
from ..upload_sessions import UploadSessionStore

from . import archive_read_model, run_read_model, run_submission_service
from .collection_workflow import CollectionWorkflow, CollectionPolicy


_lock = threading.Lock()


_gpu_job_lock = threading.Lock()


_upload_finalize_lock = threading.Lock()


_nas_batch_submission_lock = threading.Lock()


_web_access_audit_lock = threading.Lock()


try:
    _ARCHIVE_STREAM_LIMIT = max(
        1, int(os.getenv("VISIONCORTEX_WEB_MAX_CONCURRENT_ARCHIVE_STREAMS", "8"))
    )
except ValueError:
    _ARCHIVE_STREAM_LIMIT = 8
    _ARCHIVE_STREAM_LIMIT_ERROR = (
        "VISIONCORTEX_WEB_MAX_CONCURRENT_ARCHIVE_STREAMS must be an integer"
    )
else:
    _ARCHIVE_STREAM_LIMIT_ERROR = None


_archive_stream_slots = threading.BoundedSemaphore(_ARCHIVE_STREAM_LIMIT)


_runs: dict[str, dict[str, Any]] = {}


_persistent_queue: DurableRunQueue | None = None


_upload_sessions: UploadSessionStore | None = None


_queue_thread: threading.Thread | None = None


_queue_stop = threading.Event()


_queue_wakeup = threading.Event()


_nas_monitor_lock = threading.Lock()


_nas_monitor_stop = threading.Event()


_nas_monitor_thread: threading.Thread | None = None


_nas_monitor_snapshot: dict[str, Any] | None = None


_automation_startup_configuration: dict[str, Any] | None = None


_queue_worker_id = f"web-{os.getpid()}-{uuid.uuid4().hex[:8]}"


_QUEUE_LEASE_SECONDS = 60.0


_QUEUE_LEASE_RENEW_SECONDS = 15.0


_QUEUE_POLL_SECONDS = 1.0


_BENCHMARK_EXPERIMENT_ID = None


_BENCHMARK_ARCHIVE_NAME = None


_BENCHMARK_SUBMISSION_PROTOCOL_VERSION = 1


_PHYSICAL_ACTION_TYPES = (
    "hand_object_contact",
    "object_movement",
    "liquid_transfer",
    "container_state_change",
    "device_panel_operation",
)


_RUNTIME_HEARTBEAT_STALE_SECONDS = 90.0


_RUNTIME_HEARTBEAT_FILES = (
    "resource_telemetry_live.json",
    "source_progress.json",
    "run_metrics_live.json",
)


def _nas_monitor_receipt_path(settings: dict[str, Any]) -> Path:
    return (
        Path(settings["storage"]["local_runtime_root"])
        / "state"
        / "nas-recording-monitor.json"
    )


def _publish_nas_monitor_snapshot(
    settings: dict[str, Any], payload: dict[str, Any]
) -> None:
    global _nas_monitor_snapshot
    with _nas_monitor_lock:
        _nas_monitor_snapshot = payload
    path = _nas_monitor_receipt_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _nas_monitor_loop(settings: dict[str, Any]) -> None:
    failures = 0
    last_success: dict[str, Any] | None = None
    interval = max(
        5.0,
        float((settings.get("collection_ingest") or {}).get("poll_seconds", 30)),
    )
    camera_monitor = None
    if (settings.get("device_day") or {}).get("camera_lanes"):
        from ..device_day_monitor import CameraMonitor

        camera_monitor = CameraMonitor(
            settings, _nas_monitor_stop, _device_day_service.observe
        )
    while not _nas_monitor_stop.is_set():
        scan_started_at = ports.datetime.now().astimezone().isoformat()
        scan_started = ports.time.monotonic()
        try:
            if camera_monitor is not None:
                inventory = camera_monitor.poll()
            else:
                discovery = copy.deepcopy(settings)
                if (settings.get("device_day") or {}).get("enabled"):
                    discovery["collection_ingest"]["max_recordings_per_camera"] = 0
                    discovery["collection_ingest"]["capture_since_date"] = settings[
                        "device_day"
                    ].get("start_date")
                if (settings.get("device_day") or {}).get("enabled"):
                    # Release each uploaded slice to preprocessing as soon as it is
                    # inspected, without waiting for every camera in the NAS scan.
                    inventory = ports.scan_recordings(
                        discovery,
                        on_record=lambda record: _device_day_service.observe(
                            settings, {"recordings": [record]}
                        ),
                    )
                else:
                    inventory = ports.scan_recordings(discovery)
            if camera_monitor is None:
                _device_day_service.observe(settings, inventory)
            failures = 0
            # Freshness describes finished discovery and inventory observation. A
            # long successful scan must not publish its old start time as a
            # fresh result; a stuck next scan still leaves this result ageing.
            scan_completed_at = ports.datetime.now().astimezone().isoformat()
            last_success = inventory | {
                "monitor": {
                    "status": "watching",
                    "observed_at": scan_completed_at,
                    "scan_started_at": scan_started_at,
                    "scan_completed_at": scan_completed_at,
                    "scan_seconds": max(0.0, ports.time.monotonic() - scan_started),
                    "poll_seconds": interval,
                    "consecutive_failures": 0,
                }
            }
            _publish_nas_monitor_snapshot(settings, last_success)
        except (OSError, ValueError, TypeError) as exc:
            failures += 1
            degraded = dict(
                last_success
                or {
                    "mode": "directory_metadata",
                    "recordings": [],
                    "recording_count": 0,
                    "batches": [],
                    "errors": [],
                    "truncated": False,
                }
            )
            scan_completed_at = ports.datetime.now().astimezone().isoformat()
            degraded["monitor"] = {
                "status": "retrying",
                "observed_at": scan_completed_at,
                "scan_started_at": scan_started_at,
                "scan_completed_at": scan_completed_at,
                "scan_seconds": max(0.0, ports.time.monotonic() - scan_started),
                "poll_seconds": interval,
                "consecutive_failures": failures,
                "message": "NAS 暂时不可用，后台会继续重试。",
                "error_type": type(exc).__name__,
            }
            try:
                _publish_nas_monitor_snapshot(settings, degraded)
            except OSError:
                with _nas_monitor_lock:
                    global _nas_monitor_snapshot
                    _nas_monitor_snapshot = degraded
        _nas_monitor_stop.wait(interval)


def _start_nas_monitor(settings: dict[str, Any]) -> None:
    global _nas_monitor_snapshot, _nas_monitor_thread
    if not directory_ingest_enabled(settings):
        return
    if _nas_monitor_thread is not None and _nas_monitor_thread.is_alive():
        return
    with _nas_monitor_lock:
        _nas_monitor_snapshot = None
    _nas_monitor_stop.clear()
    _nas_monitor_thread = threading.Thread(
        target=_nas_monitor_loop,
        args=(settings,),
        name="nas-recording-monitor",
        daemon=True,
    )
    _nas_monitor_thread.start()


def _stop_nas_monitor() -> None:
    global _nas_monitor_thread
    _nas_monitor_stop.set()
    if _nas_monitor_thread is not None:
        _nas_monitor_thread.join(timeout=5)
    _nas_monitor_thread = None


def _recover_orphaned_tasks() -> None:
    """Mark only stale durable tasks interrupted after a Web service restart."""

    root = archive_read_model._archive_root()
    if not root.is_dir():
        return
    status_paths = list(root.glob("*/JSON-Config-Files/pipeline_status.json"))
    settings = _settings()
    resolved_settings = {
        **settings,
        "storage": {**settings.get("storage", {}), "archive_root": str(root)},
    }
    for staging_root in run_staging_roots(resolved_settings):
        if staging_root.is_dir():
            status_paths.extend(
                staging_root.glob("*/*/JSON-Config-Files/pipeline_status.json")
            )
    for status_path in status_paths:
        payload = run_read_model._read_json(status_path, {}) or {}
        # An unreadable NAS receipt is not an interrupted task. In particular,
        # never overwrite a read failure with a fabricated recovery receipt.
        if not isinstance(payload, dict) or not payload.get("stage"):
            continue
        stage = str(payload.get("stage") or "")
        if stage in {"completed", "partial", "failed"}:
            continue
        heartbeat = run_read_model._runtime_activity_receipt(status_path)
        recovery = (
            payload.get("recovery") if isinstance(payload.get("recovery"), dict) else {}
        )
        if heartbeat["active"]:
            previous_stage = str(recovery.get("previous_stage") or "")
            if (
                stage == "interrupted"
                and recovery.get("status") == "orphaned_after_service_restart"
                and previous_stage
                and previous_stage
                not in {"completed", "partial", "failed", "interrupted"}
            ):
                payload.update(
                    {
                        "stage": previous_stage,
                        "message": "后台流水线心跳仍活跃；Web 服务重启未中断正在运行的任务。",
                        "updated_at": ports.datetime.now().astimezone().isoformat(),
                        "recovery": {
                            "status": "active_after_service_restart",
                            "resumable": False,
                            "previous_stage": previous_stage,
                            "heartbeat": heartbeat,
                        },
                    }
                )
                run_submission_service._write_json_atomic(status_path, payload)
            continue
        if stage == "interrupted":
            continue
        payload.update(
            {
                "stage": "interrupted",
                "message": "服务重启时发现未完成任务；检测账本与模型结果缓存保留，可使用同一输入重新启动以续跑。",
                "updated_at": ports.datetime.now().astimezone().isoformat(),
                "recovery": {
                    "status": "orphaned_after_service_restart",
                    "resumable": True,
                    "previous_stage": stage,
                    "heartbeat": heartbeat,
                },
            }
        )
        run_submission_service._write_json_atomic(status_path, payload)


def _storage_maintenance() -> bool:
    return os.environ.get("VISIONCORTEX_STORAGE_MAINTENANCE", "0") == "1"


def _settings() -> dict[str, Any]:
    configured = os.getenv(CONFIG_ENV)
    settings = (
        ports.load_config(Path(configured)) if configured else ports.load_config()
    )
    return ai_settings.apply_active(settings)


def _automation_configuration_identity(settings: dict[str, Any]) -> dict[str, Any]:
    from ..automation_readiness import configuration_digest
    from ..config import _resolve_default_config

    configured = os.getenv(CONFIG_ENV)
    profile = Path(configured).expanduser().resolve() if configured else None
    return {
        "config_path": str(profile) if profile else None,
        "default_config_path": str(_resolve_default_config(profile)),
        "settings_sha256": configuration_digest(settings),
    }


def _queue_database_path(settings: dict[str, Any]) -> Path:
    return (
        Path(settings["storage"]["local_runtime_root"])
        / "state"
        / "web_run_queue.sqlite3"
    )


def _initialize_persistent_queue(settings: dict[str, Any]) -> None:
    global _persistent_queue, _upload_sessions

    store = DurableRunQueue(_queue_database_path(settings))
    restored_runs = store.load_runs()
    with _lock:
        _runs.clear()
        _runs.update(restored_runs)
    _persistent_queue = store
    _upload_sessions = UploadSessionStore(_queue_database_path(settings))


def _renew_queue_lease(
    store: DurableRunQueue,
    run_id: str,
    worker_id: str,
    stop: threading.Event,
) -> None:
    while not stop.wait(_QUEUE_LEASE_RENEW_SECONDS):
        if not store.renew_lease(
            run_id,
            worker_id,
            lease_seconds=_QUEUE_LEASE_SECONDS,
        ):
            return


def _dispatch_persisted_job(job: QueuedRunJob) -> None:
    from ..runtime_control import execution_context

    with execution_context(job_id=job.run_id, source="offline", stop=_queue_stop):
        _dispatch_job_in_context(job)


def _dispatch_job_in_context(job: QueuedRunJob) -> None:
    payload = job.payload
    # Keep queued model/input settings immutable. Admission is a live worker
    # policy so jobs submitted before the split cannot bypass shared limits.
    settings = copy.deepcopy(payload["settings"]) | {
        "runtime": _settings().get("runtime", {})
    }
    if job.reclaimed and job.kind != "stage_refresh":
        settings.setdefault("project", {})["resume_stages"] = True
    if job.kind == "stage_refresh":
        from ..stage_refresh import refresh

        root = run_read_model._find_staging_run(settings, payload["parent_run_id"])
        if root is None:
            raise ValueError("原实验暂存目录不可用")
        _update(job.run_id, state="running", progress=0.1, message="正在刷新所选阶段")
        receipt = refresh(
            root,
            payload["scope"],
            settings,
            target=payload.get("target"),
            revision=payload.get("revision"),
        )
        _update(
            job.run_id,
            state="completed",
            progress=1.0,
            message="所选阶段已刷新，原质量门保持不变",
            refresh_receipt=receipt,
        )
        return
    if settings["storage"].get("sync_to_nas"):
        audit_production_model_certification(settings)
    if job.kind == "run":
        _execute_now(
            job.run_id,
            RunManifest.model_validate(payload["manifest"]),
            settings,
            Path(payload["nas_root"]),
            payload.get("ingest"),
        )
        return
    if job.kind == "fixed_benchmark":
        _execute_fixed_benchmark_now(
            job.run_id,
            settings,
            Path(payload["nas_root"]),
            payload["timing"],
        )
        return
    if job.kind == "index_collection":
        _execute_index_collection_now(
            job.run_id,
            payload["source_experiment_id"],
            payload["archive_name"],
            settings,
            Path(payload["staging_root"]),
            Path(payload["fixed_root"]),
            Path(payload["history_root"]),
            payload["timing"],
        )
        return
    raise ValueError(f"Unsupported durable queue job kind: {job.kind}")


def _queue_worker_loop(store: DurableRunQueue) -> None:
    from ..runtime_control import ExecutionCancelled

    while not _queue_stop.is_set():
        from ..machine_worker import drain_requested

        if drain_requested():
            return
        job = store.claim_next(
            _queue_worker_id,
            lease_seconds=_QUEUE_LEASE_SECONDS,
        )
        if job is None:
            _queue_wakeup.wait(_QUEUE_POLL_SECONDS)
            _queue_wakeup.clear()
            continue

        existing_state = str((store.load_run(job.run_id) or {}).get("state") or "")
        if existing_state in {"completed", "partial", "failed"}:
            store.finish(job.run_id, _queue_worker_id, existing_state)
            _queue_wakeup.set()
            continue

        lease_stop = threading.Event()
        renewer = threading.Thread(
            target=_renew_queue_lease,
            args=(store, job.run_id, _queue_worker_id, lease_stop),
            name=f"visioncortex-queue-lease-{job.run_id}",
            daemon=True,
        )
        renewer.start()
        try:
            if job.reclaimed:
                _update(
                    job.run_id,
                    state="queued",
                    progress=0.0,
                    message="服务重启后已恢复任务，等待可用计算资源",
                    recovered_from_durable_queue=True,
                )
            _dispatch_persisted_job(job)
            latest = store.load_run(job.run_id) or {}
            final_state = str(latest.get("state") or "")
            final_error = latest.get("error")
            if (
                _queue_stop.is_set()
                and final_state == "failed"
                and str(final_error).startswith("ExecutionCancelled:")
            ):
                store.release_for_shutdown(job.run_id, _queue_worker_id)
                continue
            if final_state not in {"completed", "partial", "failed"}:
                final_state = "failed"
                final_error = (
                    "Durable queue executor returned without a terminal run state"
                )
                _update(job.run_id, state=final_state, progress=1.0, error=final_error)
            store.finish(
                job.run_id,
                _queue_worker_id,
                final_state,
                error=str(final_error) if final_error else None,
            )
        except ExecutionCancelled:
            store.release_for_shutdown(job.run_id, _queue_worker_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _update(job.run_id, state="failed", progress=1.0, error=error)
            store.finish(
                job.run_id,
                _queue_worker_id,
                "failed",
                error=error,
            )
        finally:
            lease_stop.set()
            renewer.join(timeout=1.0)
            _queue_wakeup.set()


def _start_queue_worker() -> None:
    global _queue_thread

    if _persistent_queue is None:
        raise RuntimeError("Durable run queue has not been initialized")
    if _queue_thread is not None and _queue_thread.is_alive():
        return
    _queue_stop.clear()
    _queue_wakeup.set()
    _queue_thread = threading.Thread(
        target=_queue_worker_loop,
        args=(_persistent_queue,),
        name="visioncortex-durable-run-queue",
        daemon=True,
    )
    _queue_thread.start()


def _stop_queue_worker() -> None:
    global _persistent_queue, _queue_thread

    _queue_stop.set()
    _queue_wakeup.set()
    if _queue_thread is not None:
        _queue_thread.join(timeout=5.0)
        if _queue_thread.is_alive():
            return
    _queue_thread = None
    _persistent_queue = None


def _schedule_job(
    background_tasks: DeferredTasks,
    *,
    run_id: str,
    kind: str,
    payload: dict[str, Any],
    fallback: Any,
    fallback_args: tuple[Any, ...],
) -> str:
    if _persistent_queue is None:
        # Direct function calls in focused tests do not enter the ASGI lifespan.
        # A real Web server always initializes the SQLite queue before accepting requests.
        background_tasks.add_task(fallback, *fallback_args)
        return "process_memory_fallback"
    _persistent_queue.enqueue(run_id, kind, payload)
    _queue_wakeup.set()
    return "sqlite"


def _reserve_collection_archive(
    settings: dict[str, Any], experiment_name: str, run_id: str
) -> tuple[str, Path, Path, Path]:
    """Reserve a unique formal name while writing only to run staging."""

    base_name = safe_archive_name(experiment_name)
    archive_root = archive_read_model._archive_root(settings)
    with _lock:
        archive_name = base_name
        if (archive_root / archive_name).exists() or any(
            run.get("experiment_id") == archive_name
            and run.get("state")
            not in {"completed", "partial", "failed", "interrupted"}
            for run in _runs.values()
        ):
            suffix = ports.datetime.now().strftime("%Y%m%d-%H%M%S")
            archive_name = f"{base_name}-{suffix}-{uuid.uuid4().hex[:4]}"
        fixed_root, staging_root, history_root = ports.fixed_archive_staging_paths(
            settings, archive_name, run_id
        )
        local_runtime_root = Path(settings["storage"]["local_runtime_root"]).resolve()
        settings["project"]["output_root"] = str(local_runtime_root / "runs" / run_id)
        settings["storage"]["active_archive_path"] = str(staging_root)
        # The configured archive may be local or network storage. Write derived
        # files directly into its staging tree in either case.
        settings["storage"]["run_output_mode"] = "nas_direct"
        initialize_nas_archive(settings, archive_name)
    return archive_name, fixed_root, staging_root, history_root


def _reserve_fixed_benchmark(settings: dict[str, Any], run_id: str) -> Path:
    benchmark = benchmark_registration(settings, required=True)
    archive_name = benchmark["archive_name"]
    fixed_root, nas_root, history_root = ports.fixed_archive_staging_paths(
        settings, archive_name, run_id
    )
    with _lock:
        if any(
            run.get("benchmark_archive") == archive_name
            and run.get("state") not in {"completed", "partial", "failed"}
            for existing_id, run in _runs.items()
            if existing_id != run_id
        ):
            raise ApplicationError(409, "六路三小时固定基准已经在运行")
        local_runtime_root = Path(settings["storage"]["local_runtime_root"]).resolve()
        settings["project"]["output_root"] = str(local_runtime_root / "runs" / run_id)
        settings["storage"]["active_archive_path"] = str(nas_root)
        initialize_nas_archive(settings, archive_name)
        _runs[run_id] = {
            "state": "reserving",
            "progress": 0.0,
            "experiment_id": archive_name,
            "benchmark_archive": archive_name,
            "nas_output": str(fixed_root),
            "nas_staging": str(nas_root),
            "nas_history": str(history_root),
        }
    return nas_root


def _update(run_id: str, **values: Any) -> None:
    with _lock:
        if _persistent_queue is not None:
            _runs[run_id] = _persistent_queue.patch_run(run_id, values)
        else:
            _runs.setdefault(run_id, {}).update(values)


def _write_fixed_benchmark_submission_receipt(
    nas_root: Path,
    run_id: str,
    request_received_at: str,
) -> Path:
    """Persist the asynchronous ownership contract before starting a long run."""

    receipt_path = nas_root / "JSON-Config-Files" / "run_submission.json"
    run_submission_service._write_json_atomic(
        receipt_path,
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "task": "fixed_six_view_three_hour_benchmark",
            "state": "queued",
            "submitted_at": request_received_at,
            "execution": {
                "owner": "visioncortex_web_service",
                "server_pid": os.getpid(),
                "client_process_independent": True,
                "queue_persistence": "sqlite",
                "survives_web_service_restart": True,
                "requires_web_service_alive_to_execute": True,
            },
            "monitoring": {
                "status_url": f"/api/runs/{run_id}",
                "nas_staging": str(nas_root),
                "durable_status": "JSON-Config-Files/pipeline_status.json",
            },
        },
    )
    return receipt_path


def _update_queue_recovery_state(
    nas_root: Path,
    state: str,
    *,
    error: str | None = None,
) -> None:
    receipt_path = run_submission_service._queue_recovery_receipt_path(nas_root)
    payload = run_read_model._read_json(receipt_path, {}) or {}
    if payload.get("schema_version") != "visioncortex-queue-recovery/1":
        return
    payload.update(
        {
            "state": state,
            "updated_at": ports.datetime.now().astimezone().isoformat(),
            "error": error,
        }
    )
    run_submission_service._write_json_atomic(receipt_path, payload)


def _recover_jobs_from_archive_receipts(settings: dict[str, Any]) -> None:
    """Rebuild queued browser jobs when the local SQLite queue is lost."""

    if _persistent_queue is None:
        return
    root = archive_read_model._archive_root(settings).resolve()
    if not root.is_dir():
        return
    receipt_paths = sorted(
        root.glob("*/JSON-Config-Files/Input-Manifests/queue_recovery.json")
    )
    for staging_root in run_staging_roots(settings):
        if staging_root.is_dir():
            receipt_paths.extend(
                sorted(
                    staging_root.glob(
                        "*/*/JSON-Config-Files/Input-Manifests/queue_recovery.json"
                    )
                )
            )
    for receipt_path in receipt_paths:
        payload = run_read_model._read_json(receipt_path, {}) or {}
        if payload.get("schema_version") != "visioncortex-queue-recovery/1":
            continue
        job_kind = str(payload.get("job_kind") or "")
        if job_kind not in {"run", "index_collection"} or payload.get("state") not in {
            "queued",
            "running",
        }:
            continue
        run_id = str(payload.get("run_id") or "")
        if not run_id or _persistent_queue.get_job(run_id) is not None:
            continue
        nas_root = receipt_path.parents[2].resolve()
        if not archive_contains(nas_root, root) or nas_root == root:
            continue
        status_path = nas_root / "JSON-Config-Files" / "pipeline_status.json"
        if status_path.exists():
            pipeline_status = run_read_model._read_json(status_path, {}) or {}
            if pipeline_status.get("stage") in {"completed", "partial", "failed"}:
                _update_queue_recovery_state(
                    nas_root,
                    str(pipeline_status["stage"]),
                    error=pipeline_status.get("error"),
                )
                continue
            if run_read_model._runtime_activity_receipt(status_path)["active"]:
                continue
        manifest_relative = Path(str(payload.get("manifest_relative_path") or ""))
        seal_relative_value = str(payload.get("input_seal_relative_path") or "").strip()
        seal_relative = Path(seal_relative_value) if seal_relative_value else None
        if manifest_relative.is_absolute() or (
            seal_relative is not None and seal_relative.is_absolute()
        ):
            continue
        manifest_path = (nas_root / manifest_relative).resolve()
        input_seal_path = (
            (nas_root / seal_relative).resolve() if seal_relative is not None else None
        )
        try:
            manifest_path.relative_to(nas_root)
            if input_seal_path is not None:
                input_seal_path.relative_to(nas_root)
        except ValueError:
            continue
        if not manifest_path.is_file() or (
            input_seal_path is not None and not input_seal_path.is_file()
        ):
            continue
        try:
            manifest = RunManifest.model_validate(
                yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            )
            seal = (
                json.loads(input_seal_path.read_text(encoding="utf-8"))
                if input_seal_path is not None
                else None
            )
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError):
            continue
        if seal is not None and (
            not verify_input_seal(seal)
            or seal.get("experiment_id") != manifest.experiment_id
        ):
            continue
        sources_available = True
        sources = (
            seal.get("sources") or []
            if seal is not None
            else run_submission_service._manifest_source_receipts(manifest)
        )
        for source in sources:
            source_path = Path(str(source.get("path") or ""))
            try:
                source_stat = source_path.stat()
            except OSError:
                sources_available = False
                break
            if source.get("size_bytes") is not None and int(
                source["size_bytes"]
            ) != int(source_stat.st_size):
                sources_available = False
                break
            expected_mtime = source.get("mtime_ns")
            if expected_mtime is not None and int(expected_mtime) != int(
                source_stat.st_mtime_ns
            ):
                sources_available = False
                break
        if not sources_available:
            continue
        recovered_settings = copy.deepcopy(settings)
        recovered_settings["storage"]["active_archive_path"] = str(nas_root)
        if job_kind == "run":
            context = payload.get("recovery_context") or {}
            fixed_root_value = str(context.get("formal_fixed_root") or "").strip()
            history_root_value = str(context.get("formal_history_root") or "").strip()
            if fixed_root_value and history_root_value:
                recovered_settings["storage"].update(
                    {
                        "run_output_mode": "nas_direct",
                        "formal_fixed_root": fixed_root_value,
                        "formal_history_root": history_root_value,
                        "formal_promotion_required": True,
                    }
                )
            recovered_settings["storage"]["sync_to_nas"] = (
                str((payload.get("ingest") or {}).get("retention_mode")) == "nas_only"
            )
        run_state = {
            "state": "queued",
            "progress": 0.0,
            "message": "本地队列账本丢失后，已从归档输入封条恢复任务",
            "experiment_id": manifest.experiment_id,
            "nas_output": str(
                recovered_settings["storage"].get("formal_fixed_root") or nas_root
            ),
            "nas_staging": str(nas_root),
            "recovered_from_archive_receipt": True,
        }
        _runs[run_id] = run_state
        _persistent_queue.save_run(run_id, run_state)
        try:
            if job_kind == "run":
                job_payload = {
                    "manifest": manifest.model_dump(mode="json"),
                    "settings": recovered_settings,
                    "nas_root": str(nas_root),
                    "ingest": payload.get("ingest"),
                }
            else:
                context = payload.get("recovery_context") or {}
                required = {
                    "source_experiment_id",
                    "archive_name",
                    "staging_root",
                    "fixed_root",
                    "history_root",
                    "timing",
                }
                if not required.issubset(context):
                    continue
                job_payload = {
                    "source_experiment_id": context["source_experiment_id"],
                    "archive_name": context["archive_name"],
                    "settings": recovered_settings,
                    "staging_root": context["staging_root"],
                    "fixed_root": context["fixed_root"],
                    "history_root": context["history_root"],
                    "timing": context["timing"],
                }
            _persistent_queue.enqueue(run_id, job_kind, job_payload)
        except ValueError:
            continue
        run_submission_service._write_queue_recovery_receipt(
            nas_root,
            run_id=run_id,
            state="queued",
            manifest_path=manifest_path,
            input_seal_path=input_seal_path,
            ingest=payload.get("ingest"),
            recovered_from_archive_receipt=True,
            job_kind=job_kind,
            recovery_context=payload.get("recovery_context"),
        )


def _append_web_end_to_end_metrics(
    roots: list[Path], ingest: dict[str, Any], completed: bool
) -> None:
    ended_at = ports.datetime.now().astimezone().isoformat()
    total_seconds = round(
        (
            ports.time.time() - float(ingest["request_started_epoch"])
            if ingest.get("request_started_epoch") is not None
            else ports.time.perf_counter() - float(ingest["request_started_perf"])
        ),
        6,
    )
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "delivery_metrics.json"
        metrics = run_read_model._read_json(metrics_path, {}) or {}
        metrics["web_ingest"] = {
            key: value
            for key, value in ingest.items()
            if key not in {"request_started_perf", "request_started_epoch"}
        }
        metrics["web_end_to_end"] = {
            "definition": "HTTP request arrival through analysis release-gate readiness; final publication time is authoritative in the current-release pointer",
            "request_received_at": ingest["request_received_at"],
            "completed_at": ended_at,
            "total_duration_seconds": total_seconds,
            "completed": completed,
        }
        run_submission_service._write_json_atomic(metrics_path, metrics)


def _append_fixed_benchmark_metrics(
    roots: list[Path], timing: dict[str, Any], completed: bool
) -> None:
    ended_at = ports.datetime.now().astimezone().isoformat()
    total_seconds = round(
        (
            ports.time.time() - float(timing["request_started_epoch"])
            if timing.get("request_started_epoch") is not None
            else ports.time.perf_counter() - float(timing["request_started_perf"])
        ),
        6,
    )
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "delivery_metrics.json"
        metrics = run_read_model._read_json(metrics_path, {}) or {}
        metrics["nas_index_ingest"] = {
            key: value
            for key, value in timing.items()
            if key
            not in {
                "request_started_perf",
                "request_started_epoch",
                "request_received_at",
            }
        }
        metrics["fixed_benchmark_end_to_end"] = {
            "definition": "benchmark request through analysis release-gate readiness; final publication time is authoritative in the current-release pointer",
            "request_received_at": timing["request_received_at"],
            "completed_at": ended_at,
            "total_duration_seconds": total_seconds,
            "completed": completed,
            "experiment_id": timing.get("experiment_id"),
            "archive_name": timing.get("archive_name"),
        }
        run_submission_service._write_json_atomic(metrics_path, metrics)


def _append_collection_index_metrics(
    roots: list[Path], timing: dict[str, Any], completed: bool
) -> None:
    ended_at = ports.datetime.now().astimezone().isoformat()
    total_seconds = round(
        (
            ports.time.time() - float(timing["request_started_epoch"])
            if timing.get("request_started_epoch") is not None
            else ports.time.perf_counter() - float(timing["request_started_perf"])
        ),
        6,
    )
    for root in dict.fromkeys(path.resolve() for path in roots):
        metrics_path = root / "JSON-Config-Files" / "delivery_metrics.json"
        metrics = run_read_model._read_json(metrics_path, {}) or {}
        metrics["nas_index_ingest"] = {
            key: value
            for key, value in timing.items()
            if key
            not in {
                "request_started_perf",
                "request_started_epoch",
                "request_received_at",
            }
        }
        metrics["collection_end_to_end"] = {
            "definition": "collection selection through analysis release-gate readiness; final publication time is authoritative in the current-release pointer",
            "request_received_at": timing["request_received_at"],
            "completed_at": ended_at,
            "total_duration_seconds": total_seconds,
            "completed": completed,
            "source_experiment_id": timing.get("source_experiment_id"),
            "archive_name": timing.get("archive_name"),
        }
        run_submission_service._write_json_atomic(metrics_path, metrics)


def _record_partial_completion(run_id: str, root: Path) -> bool:
    """Recognize only a fully written quality-attention result, never publish it."""
    if not partial_result_available(root):
        return False
    status = run_read_model._pipeline_status_from_root(root)
    _update_queue_recovery_state(root, "partial")
    _update(
        run_id,
        state="partial",
        progress=1.0,
        error=None,
        message=status.get("message"),
        nas_staging=str(root),
        observability_root=str(root),
        output=None,
        promotion=None,
        archive_url=None,
        result_url=f"/#/stage/{quote(run_id)}/experiments",
        evidence_classification="PARTIAL_EVIDENCE",
    )
    return True


def _execute_now(
    run_id: str,
    manifest: RunManifest,
    settings: dict[str, Any],
    nas_root: Path,
    ingest: dict[str, Any] | None = None,
) -> None:
    def progress(stage: str, value: float, message: str) -> None:
        if stage == "completed" and settings.get("storage", {}).get(
            "formal_promotion_required"
        ):
            stage = "finalizing"
            value = min(float(value), 0.999)
            message = "派生结果已通过验收，正在发布正式档案"
        display_root = str(
            settings.get("storage", {}).get("formal_fixed_root") or nas_root
        )
        _update(
            run_id,
            state=stage,
            progress=value,
            message=message,
            nas_output=display_root,
            nas_staging=str(nas_root),
        )

    try:
        _update(
            run_id,
            state="running",
            progress=0.0,
            nas_output=str(
                settings.get("storage", {}).get("formal_fixed_root") or nas_root
            ),
            nas_staging=str(nas_root),
        )
        _update_queue_recovery_state(nas_root, "running")
        output = ports.EvidencePipeline(settings, progress).run(manifest)
        if _record_partial_completion(run_id, Path(output)):
            if ingest is not None:
                _append_web_end_to_end_metrics([Path(output)], ingest, completed=False)
                write_partial_delivery(
                    Path(output), run_read_model._merged_run_metrics(Path(output))
                )
            return
        formal_fixed_value = str(
            settings.get("storage", {}).get("formal_fixed_root") or ""
        ).strip()
        formal_history_value = str(
            settings.get("storage", {}).get("formal_history_root") or ""
        ).strip()
        promotion = None
        completed_root = Path(output)
        publication_context = None
        if ingest is not None:
            _append_web_end_to_end_metrics([Path(output)], ingest, completed=True)
            publication_context = {
                "metric_key": "web_end_to_end",
                "definition": "HTTP request arrival + source retention + analysis + atomic formal archive publication",
                "request_received_at": ingest.get("request_received_at"),
                "request_started_epoch": ingest.get("request_started_epoch"),
            }
        if settings.get("storage", {}).get("formal_promotion_required"):
            _update_queue_recovery_state(nas_root, "completed")
            if not formal_fixed_value or not formal_history_value:
                raise RuntimeError(
                    "Formal Web run is missing fixed/history promotion targets"
                )
            formal_fixed_root = Path(formal_fixed_value)
            promotion = ports.promote_fixed_archive(
                Path(output),
                formal_fixed_root,
                Path(formal_history_value),
                publication_context,
            )
            completed_root = formal_fixed_root
        elif ingest is not None:
            _append_web_end_to_end_metrics([completed_root], ingest, completed=True)
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(completed_root),
            nas_output=str(completed_root),
            nas_staging=str(nas_root),
            promotion=promotion,
            archive_url=f"/?archive={quote(completed_root.name)}",
        )
        if not settings.get("storage", {}).get("formal_promotion_required"):
            _update_queue_recovery_state(completed_root, "completed")
    except Exception as exc:
        if ingest is not None:
            _append_web_end_to_end_metrics([nas_root], ingest, completed=False)
        error = f"{type(exc).__name__}: {exc}"
        _update(run_id, state="failed", progress=1.0, error=error)
        _update_queue_recovery_state(nas_root, "failed", error=error)
    finally:
        upload_session_id = (ingest or {}).get("upload_session_id")
        if upload_session_id and _upload_sessions is not None:
            _upload_sessions.release(str(upload_session_id))


def _execute(
    run_id: str,
    manifest: RunManifest,
    settings: dict[str, Any],
    nas_root: Path,
    ingest: dict[str, Any] | None = None,
) -> None:
    _update(run_id, state="queued", progress=0.0, message="等待可用计算资源")
    with _gpu_job_lock:
        _execute_now(run_id, manifest, settings, nas_root, ingest)


def _execute_fixed_benchmark_now(
    run_id: str,
    settings: dict[str, Any],
    nas_root: Path,
    timing: dict[str, Any],
) -> None:
    benchmark = benchmark_registration(settings, required=True)
    experiment_id = benchmark["experiment_id"]
    archive_name = benchmark["archive_name"]
    fixed_root, _, history_root = ports.fixed_archive_staging_paths(
        settings, archive_name, run_id
    )

    def progress(stage: str, value: float, message: str) -> None:
        if stage == "completed":
            stage = "finalizing"
            value = min(float(value), 0.999)
            message = "派生结果已通过验收，正在发布固定基准档案"
        _update(
            run_id,
            state=stage,
            progress=value,
            message=message,
            nas_output=str(fixed_root),
        )

    try:
        manifest, manifest_path, ingest = ports.prepare_from_nas_index(
            settings,
            experiment_id,
            lambda message: _update(
                run_id,
                state="nas_ingest",
                progress=0.01,
                message=message,
                nas_output=str(fixed_root),
            ),
        )
        manifest_path.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        timing.update(
            {
                "ingest_completed_at": ports.datetime.now().astimezone().isoformat(),
                "duration_seconds": round(
                    ports.time.perf_counter() - float(timing["request_started_perf"]), 6
                ),
                "manifest": str(manifest_path),
                "source_count": len(manifest.views),
                "input_reused": bool(
                    ingest.get("source_validation", {}).get("missing_count") == 0
                ),
                "ingest_details": ingest,
            }
        )
        require_source_count(benchmark, len(manifest.views))
        _update(run_id, state="running", progress=0.02, nas_output=str(fixed_root))
        output = ports.EvidencePipeline(settings, progress).run(manifest)
        if _record_partial_completion(run_id, Path(output)):
            _append_fixed_benchmark_metrics([Path(output)], timing, completed=False)
            write_partial_delivery(
                Path(output), run_read_model._merged_run_metrics(Path(output))
            )
            return
        _append_fixed_benchmark_metrics(
            [Path(output), nas_root], timing, completed=True
        )
        receipt = ports.promote_fixed_archive(
            nas_root,
            fixed_root,
            history_root,
            {
                "metric_key": "fixed_benchmark_end_to_end",
                "definition": "benchmark request + NAS index preparation + analysis + atomic fixed archive publication",
                "request_received_at": timing.get("request_received_at"),
                "request_started_epoch": timing.get("request_started_epoch"),
                "experiment_id": experiment_id,
                "archive_name": archive_name,
            },
        )
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(output),
            nas_output=str(fixed_root),
            nas_staging=str(nas_root),
            observability_root=str(fixed_root),
            promotion=receipt,
            archive_url=f"/#/archive/{quote(archive_name)}/experiments",
        )
    except Exception as exc:
        timing.setdefault(
            "duration_seconds",
            round(ports.time.perf_counter() - float(timing["request_started_perf"]), 6),
        )
        with _lock:
            failure_stage = _runs.get(run_id, {}).get("state")
        timing.setdefault("failure_stage", failure_stage)
        _append_fixed_benchmark_metrics([nas_root], timing, completed=False)
        _update(
            run_id, state="failed", progress=1.0, error=f"{type(exc).__name__}: {exc}"
        )


def _execute_fixed_benchmark(
    run_id: str,
    settings: dict[str, Any],
    nas_root: Path,
    timing: dict[str, Any],
) -> None:
    _update(run_id, state="queued", progress=0.0, message="等待可用计算资源")
    with _gpu_job_lock:
        _execute_fixed_benchmark_now(run_id, settings, nas_root, timing)


def _preflight_and_seal_collection_input(
    *,
    run_id: str,
    source_experiment_id: str,
    archive_name: str,
    manifest: RunManifest,
    manifest_path: Path,
    ingest: dict[str, Any],
    settings: dict[str, Any],
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
    timing: dict[str, Any],
) -> None:
    _update(
        run_id,
        state="input_preflight",
        progress=0.015,
        message="输入已封存，正在读取媒体头与时钟首尾，尚未占用 GPU",
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    input_preflight = ports.preflight_manifest_inputs(manifest, settings)
    preflight_path = (
        staging_root
        / "JSON-Config-Files"
        / "Input-Manifests"
        / "prequeue_input_preflight.json"
    )
    run_submission_service._write_json_atomic(preflight_path, input_preflight)
    existing_seal_path = (
        staging_root / "JSON-Config-Files" / "Input-Manifests" / "input_seal.json"
    )
    # Local zero-copy ingestion keeps its first manifest/seal in Runtime. Bind
    # the recovery receipt to an archive-owned metadata copy, never to ../ paths.
    source_seal_path = Path(
        str(
            (ingest.get("original_retention") or {}).get("input_seal")
            or existing_seal_path
        )
    )
    existing_seal = run_read_model._read_json(source_seal_path, {}) or {}
    if not verify_input_seal(existing_seal) or RunManifest.model_validate(
        existing_seal.get("manifest")
    ).model_dump(mode="json") != manifest.model_dump(mode="json"):
        raise ValueError("采集批次输入封条无效或与本次清单不一致，未启动分析。")
    archived_manifest_path = staging_root / "JSON-Config-Files" / "input_manifest.yaml"
    archived_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    archived_manifest_path.write_text(
        yaml.safe_dump(
            manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    refreshed_seal = build_input_seal(
        manifest,
        source_mode="nas_segmented_virtual_timeline",
        sources=list(existing_seal.get("sources") or []),
        role_resolution=existing_seal.get("role_resolution") or {},
        copied_source_bytes=0,
        preflight=input_preflight,
    )
    write_input_seal(existing_seal_path, refreshed_seal)
    run_submission_service._write_json_atomic(
        staging_root / "JSON-Config-Files" / "view_role_resolution.json",
        refreshed_seal["role_resolution"],
    )
    ingest["archived_input_manifest"] = str(archived_manifest_path)
    ingest.setdefault("original_retention", {}).update(
        source_input_seal=str(source_seal_path), input_seal=str(existing_seal_path)
    )
    timing["manifest"] = str(archived_manifest_path)
    ingest["prequeue_input_preflight"] = {
        "status": input_preflight["status"],
        "receipt": str(preflight_path),
    }
    ingest.setdefault("original_retention", {})["input_seal_sha256"] = refreshed_seal[
        "seal_sha256"
    ]
    run_submission_service._write_queue_recovery_receipt(
        staging_root,
        run_id=run_id,
        state="running",
        manifest_path=archived_manifest_path,
        input_seal_path=existing_seal_path,
        ingest=ingest,
        job_kind="index_collection",
        recovery_context={
            "source_experiment_id": source_experiment_id,
            "archive_name": archive_name,
            "staging_root": str(staging_root),
            "fixed_root": str(fixed_root),
            "history_root": str(history_root),
            "timing": timing,
        },
    )


def _execute_index_collection_now(
    run_id: str,
    source_experiment_id: str,
    archive_name: str,
    settings: dict[str, Any],
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
    timing: dict[str, Any],
) -> None:
    def progress(stage: str, value: float, message: str) -> None:
        if stage == "completed":
            stage = "finalizing"
            value = min(float(value), 0.999)
            message = "派生结果已通过验收，正在发布采集批次正式档案"
        _update(
            run_id,
            state=stage,
            progress=value,
            message=message,
            nas_output=str(fixed_root),
            nas_staging=str(staging_root),
        )

    try:
        if directory_ingest_enabled(settings):
            validate_selection(settings, source_experiment_id)
            settings["storage"]["index_csv"] = str(
                selection_path(settings, source_experiment_id, ".csv")
            )
        ports.record_collection_state(
            settings,
            source_experiment_id,
            archive_name=archive_name,
            run_id=run_id,
            state="processing",
            details={"staging": str(staging_root)},
        )

        def prepared(manifest, manifest_path, ingest):
            timing.update(
                {
                    "ingest_completed_at": ports.datetime.now()
                    .astimezone()
                    .isoformat(),
                    "ingest_duration_seconds": round(
                        ports.time.perf_counter()
                        - float(timing["request_started_perf"]),
                        6,
                    ),
                    "manifest": str(manifest_path),
                    "source_count": len(manifest.views),
                    "ingest_details": ingest,
                }
            )
            if isinstance(manifest, RunManifest):
                _preflight_and_seal_collection_input(
                    run_id=run_id,
                    source_experiment_id=source_experiment_id,
                    archive_name=archive_name,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    ingest=ingest,
                    settings=settings,
                    staging_root=staging_root,
                    fixed_root=fixed_root,
                    history_root=history_root,
                    timing=timing,
                )
            _update(
                run_id,
                state="running",
                progress=0.02,
                nas_output=str(fixed_root),
                nas_staging=str(staging_root),
            )

        def analyze(manifest):
            with _gpu_job_lock:
                return ports.EvidencePipeline(settings, progress).run(manifest)

        def partial(output):
            _append_collection_index_metrics([output], timing, completed=False)
            write_partial_delivery(output, run_read_model._merged_run_metrics(output))
            ports.record_collection_state(
                settings,
                source_experiment_id,
                archive_name=archive_name,
                run_id=run_id,
                state="partial",
                details={
                    "staging": str(staging_root),
                    "evidence_classification": "PARTIAL_EVIDENCE",
                },
            )

        def before_promotion(output):
            _append_collection_index_metrics(
                [output, staging_root], timing, completed=True
            )
            _update_queue_recovery_state(staging_root, "completed")

        result = CollectionWorkflow(
            prepare=ports.prepare_from_nas_index,
            analyze=analyze,
            is_partial=lambda output: _record_partial_completion(run_id, output),
            promote=ports.promote_fixed_archive,
        ).execute(
            settings=settings,
            experiment_id=source_experiment_id,
            staging_root=staging_root,
            fixed_root=fixed_root,
            history_root=history_root,
            policy=CollectionPolicy(),
            ingest_progress=lambda message: _update(
                run_id,
                state="nas_ingest",
                progress=0.01,
                message=message,
                nas_output=str(fixed_root),
                nas_staging=str(staging_root),
            ),
            on_prepared=prepared,
            on_partial=partial,
            before_promotion=before_promotion,
            promotion_metadata={
                "metric_key": "collection_end_to_end",
                "definition": "collection selection + zero-copy NAS ingest + analysis + atomic formal archive publication",
                "request_received_at": timing.get("request_received_at"),
                "request_started_epoch": timing.get("request_started_epoch"),
                "source_experiment_id": timing.get("source_experiment_id"),
                "archive_name": timing.get("archive_name"),
            },
        )
        if result.partial:
            return
        promotion = result.promotion
        ports.record_collection_state(
            settings,
            source_experiment_id,
            archive_name=archive_name,
            run_id=run_id,
            state="archived",
            details={
                "formal_archive": str(fixed_root),
                "promotion_verification": promotion.get("verification"),
            },
        )
        _update(
            run_id,
            state="completed",
            progress=1.0,
            output=str(fixed_root),
            nas_output=str(fixed_root),
            nas_staging=str(staging_root),
            promotion=promotion,
            archive_url=f"/#/archive/{quote(archive_name)}/experiments",
        )
    except Exception as exc:
        timing.setdefault(
            "ingest_duration_seconds",
            round(ports.time.perf_counter() - float(timing["request_started_perf"]), 6),
        )
        with _lock:
            timing.setdefault("failure_stage", _runs.get(run_id, {}).get("state"))
        _append_collection_index_metrics([staging_root], timing, completed=False)
        try:
            ports.record_collection_state(
                settings,
                source_experiment_id,
                archive_name=archive_name,
                run_id=run_id,
                state="failed",
                details={
                    "failure_stage": timing.get("failure_stage"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "staging": str(staging_root),
                },
            )
        except Exception as state_exc:
            timing["failure_state_record_error"] = (
                f"{type(state_exc).__name__}: {state_exc}"
            )
        error = f"{type(exc).__name__}: {exc}"
        _update_queue_recovery_state(staging_root, "failed", error=error)
        if timing.get("failure_state_record_error"):
            error += (
                "; collection_state_audit_failed="
                f"{timing['failure_state_record_error']}"
            )
        _update(
            run_id,
            state="failed",
            progress=1.0,
            error=error,
            failure_audit={
                "collection_state_recorded": not bool(
                    timing.get("failure_state_record_error")
                ),
                "collection_state_error": timing.get("failure_state_record_error"),
            },
        )


def _execute_index_collection(
    run_id: str,
    source_experiment_id: str,
    archive_name: str,
    settings: dict[str, Any],
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
    timing: dict[str, Any],
) -> None:
    _update(
        run_id,
        state="queued",
        progress=0.0,
        message="等待可用计算资源",
        nas_output=str(fixed_root),
        nas_staging=str(staging_root),
    )
    _execute_index_collection_now(
        run_id,
        source_experiment_id,
        archive_name,
        settings,
        staging_root,
        fixed_root,
        history_root,
        timing,
    )


_device_day_service = DeviceDayService(_settings, _gpu_job_lock)


class RuntimeHost:
    """Public execution boundary; owns startup, shutdown and durable queue access.

    One process has one default host and worker ownership remains enforced by
    the existing interprocess lock. HTTP servers and command adapters both use
    the same lifecycle without importing each other.
    """

    @property
    def stream_configuration_error(self) -> str | None:
        return _ARCHIVE_STREAM_LIMIT_ERROR

    @property
    def device_day_service(self) -> DeviceDayService:
        return _device_day_service

    @staticmethod
    def settings() -> dict[str, Any]:
        return _settings()

    @staticmethod
    def maintenance_active() -> bool:
        return _storage_maintenance()

    @staticmethod
    def queue_database_path(settings: dict[str, Any]) -> Path:
        return _queue_database_path(settings)

    @staticmethod
    def initialize_queue(settings: dict[str, Any]) -> None:
        _initialize_persistent_queue(settings)

    @property
    def queue(self) -> DurableRunQueue:
        if _persistent_queue is None:
            raise RuntimeError("Durable run queue has not been initialized")
        return _persistent_queue

    @property
    def worker_alive(self) -> bool:
        return _queue_thread is not None and _queue_thread.is_alive()

    @property
    def drained(self) -> bool:
        from ..machine_worker import drain_requested

        return drain_requested() and not self.worker_alive

    def status(self) -> dict[str, Any]:
        return {
            "worker_alive": self.worker_alive,
            "queue_initialized": _persistent_queue is not None,
            "maintenance": self.maintenance_active(),
        }

    def __init__(self):
        self._owner = None
        self._started = False
        self._producers_started = False

    def start(self) -> None:
        """Initialize local state and start producers only for an execution role."""
        if self._started:
            return
        if self.maintenance_active():
            return
        settings = self.settings()
        global _automation_startup_configuration
        _automation_startup_configuration = _automation_configuration_identity(settings)
        self.initialize_queue(settings)
        self._started = True
        from ..runtime_process import role, worker_owner

        if role(settings) == "web":
            return
        self._owner = worker_owner(settings)
        try:
            self._owner.__enter__()
        except BaseException:
            self._owner = None
            self._started = False
            raise
        self._producers_started = True
        try:
            from .upload_service import _expire_stale_upload_sessions

            _expire_stale_upload_sessions(settings)
            _recover_jobs_from_archive_receipts(settings)
            _recover_orphaned_tasks()
            _start_queue_worker()
            _start_nas_monitor(settings)
            _device_day_service.start()
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop producers under the decoder guard and release worker ownership."""
        try:
            if self._producers_started:
                from ..owned_subprocess import decoder_shutdown_guard

                with decoder_shutdown_guard():
                    _device_day_service.stop()
                    _stop_nas_monitor()
                    _stop_queue_worker()
                    from ..shared_inference import close_pools

                    close_pools()
        finally:
            self._producers_started = False
            self._started = False
            if self._owner is not None:
                owner, self._owner = self._owner, None
                owner.__exit__(None, None, None)

    @asynccontextmanager
    async def lifespan(self):
        self.start()
        try:
            yield
        finally:
            self.stop()

    @staticmethod
    def submit_paths(payload, background_tasks, **options):
        return run_submission_service._create_run_from_paths(
            payload, background_tasks, **options
        )

    @staticmethod
    def get_run(run_id: str) -> dict[str, Any]:
        return run_read_model.run_read_model.get(run_id)

    @staticmethod
    def archive_detail(name: str, **options) -> dict[str, Any]:
        return archive_read_model.archive_read_model.detail(name, **options)

    @staticmethod
    def search_key_events(**options) -> dict[str, Any]:
        return archive_read_model.search_key_events(**options)

    @staticmethod
    def indexed_key_event(uid: str, **options):
        return archive_read_model.indexed_key_event(uid, **options)

    @staticmethod
    def indexed_evidence(uid: str, **options):
        return archive_read_model.indexed_evidence(uid, **options)


default_host = RuntimeHost()

"""VisionCortex Web application and compatibility exports.

Execution and read models live in application components. New callers use
RuntimeHost; legacy API imports remain supported during the transition.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from .application.contracts import ApplicationError

from .build_identity import identity as build_identity
from .identity import PRODUCT_NAME
from .device_day_service import install_routes as install_device_day_routes
from .knowledge_api import install_routes as install_knowledge_routes
from .application import (
    archive_read_model,
    run_read_model,
    run_submission_service,
    runtime_host,
    upload_service,
)
from .web_api import (
    annotations as http_annotations,
    archives as http_archives,
    boundary as http_boundary,
    collections as http_collections,
    health as http_health,
    runs as http_runs,
    settings as http_settings,
    uploads as http_uploads,
)

from .application.dependencies import ports

# Explicit legacy symbol -> unique application owner.
_COMPATIBILITY_EXPORTS = {
    "ARCHIVE_DIRECTORIES": (http_runs, "ARCHIVE_DIRECTORIES"),
    "Annotated": (http_uploads, "Annotated"),
    "Any": (archive_read_model, "Any"),
    "BackgroundTasks": (http_collections, "BackgroundTasks"),
    "CONFIG_ENV": (runtime_host, "CONFIG_ENV"),
    "Counter": (http_collections, "Counter"),
    "DeviceDayService": (runtime_host, "DeviceDayService"),
    "DurableRunQueue": (runtime_host, "DurableRunQueue"),
    "EvidencePipeline": (ports, "EvidencePipeline"),
    "File": (http_uploads, "File"),
    "FileResponse": (http_annotations, "FileResponse"),
    "Form": (http_uploads, "Form"),
    "HTMLResponse": (http_boundary, "HTMLResponse"),
    "HTTPException": (http_annotations, "HTTPException"),
    "INDEX_DB_NAME": (archive_read_model, "INDEX_DB_NAME"),
    "INDEX_MANIFEST_NAME": (archive_read_model, "INDEX_MANIFEST_NAME"),
    "JSONResponse": (http_boundary, "JSONResponse"),
    "Path": (archive_read_model, "Path"),
    "Query": (http_annotations, "Query"),
    "QueuedRunJob": (runtime_host, "QueuedRunJob"),
    "Request": (http_boundary, "Request"),
    "Response": (http_boundary, "Response"),
    "RunManifest": (run_submission_service, "RunManifest"),
    "StorageReservationError": (upload_service, "StorageReservationError"),
    "UploadFile": (http_uploads, "UploadFile"),
    "UploadSessionStore": (runtime_host, "UploadSessionStore"),
    "VideoSegmentInput": (run_submission_service, "VideoSegmentInput"),
    "ViewInput": (run_submission_service, "ViewInput"),
    "_ARCHIVE_STREAM_LIMIT": (runtime_host, "_ARCHIVE_STREAM_LIMIT"),
    "_ARCHIVE_STREAM_LIMIT_ERROR": (runtime_host, "_ARCHIVE_STREAM_LIMIT_ERROR"),
    "_BENCHMARK_ARCHIVE_NAME": (runtime_host, "_BENCHMARK_ARCHIVE_NAME"),
    "_BENCHMARK_EXPERIMENT_ID": (runtime_host, "_BENCHMARK_EXPERIMENT_ID"),
    "_BENCHMARK_SUBMISSION_PROTOCOL_VERSION": (
        runtime_host,
        "_BENCHMARK_SUBMISSION_PROTOCOL_VERSION",
    ),
    "_PHYSICAL_ACTION_TYPES": (runtime_host, "_PHYSICAL_ACTION_TYPES"),
    "_QUEUE_LEASE_RENEW_SECONDS": (runtime_host, "_QUEUE_LEASE_RENEW_SECONDS"),
    "_QUEUE_LEASE_SECONDS": (runtime_host, "_QUEUE_LEASE_SECONDS"),
    "_QUEUE_POLL_SECONDS": (runtime_host, "_QUEUE_POLL_SECONDS"),
    "_RUNTIME_HEARTBEAT_FILES": (runtime_host, "_RUNTIME_HEARTBEAT_FILES"),
    "_RUNTIME_HEARTBEAT_STALE_SECONDS": (
        runtime_host,
        "_RUNTIME_HEARTBEAT_STALE_SECONDS",
    ),
    "_append_collection_index_metrics": (
        runtime_host,
        "_append_collection_index_metrics",
    ),
    "_append_fixed_benchmark_metrics": (
        runtime_host,
        "_append_fixed_benchmark_metrics",
    ),
    "_append_web_end_to_end_metrics": (runtime_host, "_append_web_end_to_end_metrics"),
    "_archive_areas_from_root": (run_read_model, "_archive_areas_from_root"),
    "_archive_catalog_database": (archive_read_model, "_archive_catalog_database"),
    "_archive_detail_from_root": (archive_read_model, "_archive_detail_from_root"),
    "_archive_links": (archive_read_model, "_archive_links"),
    "_archive_root": (archive_read_model, "_archive_root"),
    "_archive_section_payload": (archive_read_model, "_archive_section_payload"),
    "_archive_stream_slots": (runtime_host, "_archive_stream_slots"),
    "_archive_summary_payload": (archive_read_model, "_archive_summary_payload"),
    "_attach_archive_performance_display": (
        archive_read_model,
        "_attach_archive_performance_display",
    ),
    "_attach_index_urls": (archive_read_model, "_attach_index_urls"),
    "_attach_staging_index_urls": (archive_read_model, "_attach_staging_index_urls"),
    "_automation_configuration_identity": (
        runtime_host,
        "_automation_configuration_identity",
    ),
    "_automation_startup_configuration": (
        runtime_host,
        "_automation_startup_configuration",
    ),
    "_collection_card": (http_collections, "_collection_card"),
    "_create_run_from_paths": (run_submission_service, "_create_run_from_paths"),
    "_current_attempt_started_at": (run_read_model, "_current_attempt_started_at"),
    "_declared_role_resolution": (run_submission_service, "_declared_role_resolution"),
    "_decode_catalog_event_cursor": (
        archive_read_model,
        "_decode_catalog_event_cursor",
    ),
    "_decode_event_cursor": (archive_read_model, "_decode_event_cursor"),
    "_derive_archived_quality_summary": (
        archive_read_model,
        "_derive_archived_quality_summary",
    ),
    "_device_day_service": (runtime_host, "_device_day_service"),
    "_dispatch_job_in_context": (runtime_host, "_dispatch_job_in_context"),
    "_dispatch_persisted_job": (runtime_host, "_dispatch_persisted_job"),
    "_encode_catalog_event_cursor": (
        archive_read_model,
        "_encode_catalog_event_cursor",
    ),
    "_encode_event_cursor": (archive_read_model, "_encode_event_cursor"),
    "_ensure_archive_read_catalog": (
        archive_read_model,
        "_ensure_archive_read_catalog",
    ),
    "_event_has_aligned_dual_view_material": (
        archive_read_model,
        "_event_has_aligned_dual_view_material",
    ),
    "_event_has_cross_view_support": (
        archive_read_model,
        "_event_has_cross_view_support",
    ),
    "_execute": (runtime_host, "_execute"),
    "_execute_fixed_benchmark": (runtime_host, "_execute_fixed_benchmark"),
    "_execute_fixed_benchmark_now": (runtime_host, "_execute_fixed_benchmark_now"),
    "_execute_index_collection": (runtime_host, "_execute_index_collection"),
    "_execute_index_collection_now": (runtime_host, "_execute_index_collection_now"),
    "_execute_now": (runtime_host, "_execute_now"),
    "_experiment_speech_response": (http_archives, "_experiment_speech_response"),
    "_expire_stale_upload_sessions": (upload_service, "_expire_stale_upload_sessions"),
    "_file_url": (archive_read_model, "_file_url"),
    "_find_staging_run": (run_read_model, "_find_staging_run"),
    "_folder_open_command": (http_archives, "_folder_open_command"),
    "_gpu_job_lock": (runtime_host, "_gpu_job_lock"),
    "_hydrate_run_snapshot": (run_read_model, "_hydrate_run_snapshot"),
    "_initialize_persistent_queue": (runtime_host, "_initialize_persistent_queue"),
    "_latest_result_review": (run_read_model, "_latest_result_review"),
    "_load_upload_session": (upload_service, "_load_upload_session"),
    "_lock": (runtime_host, "_lock"),
    "_manifest_source_receipts": (run_submission_service, "_manifest_source_receipts"),
    "_merged_run_metrics": (run_read_model, "_merged_run_metrics"),
    "_movement_screening_payload": (archive_read_model, "_movement_screening_payload"),
    "_nas_batch_submission_lock": (runtime_host, "_nas_batch_submission_lock"),
    "_nas_monitor_lock": (runtime_host, "_nas_monitor_lock"),
    "_nas_monitor_loop": (runtime_host, "_nas_monitor_loop"),
    "_nas_monitor_receipt_path": (runtime_host, "_nas_monitor_receipt_path"),
    "_nas_monitor_snapshot": (runtime_host, "_nas_monitor_snapshot"),
    "_nas_monitor_stop": (runtime_host, "_nas_monitor_stop"),
    "_nas_monitor_thread": (runtime_host, "_nas_monitor_thread"),
    "_nas_storage_available": (http_health, "_nas_storage_available"),
    "_parse_upload_session_files": (upload_service, "_parse_upload_session_files"),
    "_persistent_queue": (runtime_host, "_persistent_queue"),
    "_pipeline_status_from_root": (run_read_model, "_pipeline_status_from_root"),
    "_preflight_and_seal_collection_input": (
        runtime_host,
        "_preflight_and_seal_collection_input",
    ),
    "_prepare_formal_run_staging": (
        run_submission_service,
        "_prepare_formal_run_staging",
    ),
    "_public_upload_session": (upload_service, "_public_upload_session"),
    "_publish_nas_monitor_snapshot": (runtime_host, "_publish_nas_monitor_snapshot"),
    "_queue_database_path": (runtime_host, "_queue_database_path"),
    "_queue_recovery_receipt_path": (
        run_submission_service,
        "_queue_recovery_receipt_path",
    ),
    "_queue_stop": (runtime_host, "_queue_stop"),
    "_queue_thread": (runtime_host, "_queue_thread"),
    "_queue_wakeup": (runtime_host, "_queue_wakeup"),
    "_queue_worker_id": (runtime_host, "_queue_worker_id"),
    "_queue_worker_loop": (runtime_host, "_queue_worker_loop"),
    "_read_json": (run_read_model, "_read_json"),
    "_record_partial_completion": (runtime_host, "_record_partial_completion"),
    "_recover_jobs_from_archive_receipts": (
        runtime_host,
        "_recover_jobs_from_archive_receipts",
    ),
    "_recover_orphaned_tasks": (runtime_host, "_recover_orphaned_tasks"),
    "_refresh_in_progress": (http_runs, "_refresh_in_progress"),
    "_renew_queue_lease": (runtime_host, "_renew_queue_lease"),
    "_require_local_ai_settings": (http_settings, "_require_local_ai_settings"),
    "_require_upload_store": (upload_service, "_require_upload_store"),
    "_reserve_archive": (run_submission_service, "_reserve_archive"),
    "_reserve_collection_archive": (runtime_host, "_reserve_collection_archive"),
    "_reserve_fixed_benchmark": (runtime_host, "_reserve_fixed_benchmark"),
    "_resolve_archive": (archive_read_model, "_resolve_archive"),
    "_resolve_staging_run": (run_read_model, "_resolve_staging_run"),
    "_retain_retry_outputs": (http_runs, "_retain_retry_outputs"),
    "_run_snapshot_from_root": (run_read_model, "_run_snapshot_from_root"),
    "_runs": (runtime_host, "_runs"),
    "_runtime_activity_receipt": (run_read_model, "_runtime_activity_receipt"),
    "_safe_file_name": (run_submission_service, "_safe_file_name"),
    "_save_upload_to_local_and_nas": (
        run_submission_service,
        "_save_upload_to_local_and_nas",
    ),
    "_schedule_job": (runtime_host, "_schedule_job"),
    "_search_archive_roots": (archive_read_model, "_search_archive_roots"),
    "_settings": (runtime_host, "_settings"),
    "_sha256_path": (upload_service, "_sha256_path"),
    "_stage_clip_group": (archive_read_model, "_stage_clip_group"),
    "_stage_receipts_from_root": (run_read_model, "_stage_receipts_from_root"),
    "_staging_file_url": (archive_read_model, "_staging_file_url"),
    "_start_nas_monitor": (runtime_host, "_start_nas_monitor"),
    "_start_queue_worker": (runtime_host, "_start_queue_worker"),
    "_stop_nas_monitor": (runtime_host, "_stop_nas_monitor"),
    "_stop_queue_worker": (runtime_host, "_stop_queue_worker"),
    "_storage_maintenance": (runtime_host, "_storage_maintenance"),
    "_summarize_key_material_verification": (
        archive_read_model,
        "_summarize_key_material_verification",
    ),
    "_unique_upload_archive_name": (upload_service, "_unique_upload_archive_name"),
    "_update": (runtime_host, "_update"),
    "_update_queue_recovery_state": (runtime_host, "_update_queue_recovery_state"),
    "_upload_finalize_lock": (runtime_host, "_upload_finalize_lock"),
    "_upload_policy": (upload_service, "_upload_policy"),
    "_upload_retention_mode": (upload_service, "_upload_retention_mode"),
    "_upload_sessions": (runtime_host, "_upload_sessions"),
    "_video_poster_response": (http_archives, "_video_poster_response"),
    "_web_access_audit_lock": (runtime_host, "_web_access_audit_lock"),
    "_write_fixed_benchmark_submission_receipt": (
        runtime_host,
        "_write_fixed_benchmark_submission_receipt",
    ),
    "_write_json_atomic": (run_submission_service, "_write_json_atomic"),
    "_write_queue_recovery_receipt": (
        run_submission_service,
        "_write_queue_recovery_receipt",
    ),
    "ai_settings": (http_health, "ai_settings"),
    "annotation_workspace": (http_annotations, "annotation_workspace"),
    "annotation_workspace_export": (http_annotations, "annotation_workspace_export"),
    "annotation_workspace_image": (http_annotations, "annotation_workspace_image"),
    "append_access_audit": (ports, "append_access_audit"),
    "archive_catalog_path": (archive_read_model, "archive_catalog_path"),
    "archive_contains": (archive_read_model, "archive_contains"),
    "archive_detail": (archive_read_model, "archive_detail"),
    "archive_file": (http_archives, "archive_file"),
    "archive_promotion_in_progress": (
        archive_read_model,
        "archive_promotion_in_progress",
    ),
    "archive_relative_posix": (archive_read_model, "archive_relative_posix"),
    "archive_speech": (http_archives, "archive_speech"),
    "archive_speech_video": (http_archives, "archive_speech_video"),
    "asynccontextmanager": (runtime_host, "asynccontextmanager"),
    "audit_production_model_certification": (
        http_boundary,
        "audit_production_model_certification",
    ),
    "authenticate_basic_authorization": (
        http_boundary,
        "authenticate_basic_authorization",
    ),
    "automation_readiness": (http_health, "automation_readiness"),
    "build_input_seal": (run_submission_service, "build_input_seal"),
    "cached_video_poster": (http_archives, "cached_video_poster"),
    "cancel_upload_session": (upload_service, "cancel_upload_session"),
    "catalog_archive_release": (archive_read_model, "catalog_archive_release"),
    "collection_detail": (http_collections, "collection_detail"),
    "component_results": (run_read_model, "component_results"),
    "connection_health": (http_health, "connection_health"),
    "copy": (http_runs, "copy"),
    "create_collection_run": (http_collections, "create_collection_run"),
    "create_fixed_benchmark_run": (http_collections, "create_fixed_benchmark_run"),
    "create_nas_batch_run": (http_collections, "create_nas_batch_run"),
    "create_run": (run_submission_service, "create_run"),
    "create_run_from_paths": (http_runs, "create_run_from_paths"),
    "create_selection": (http_collections, "create_selection"),
    "create_upload_session": (upload_service, "create_upload_session"),
    "datetime": (ports, "datetime"),
    "decode_cursor": (archive_read_model, "decode_cursor"),
    "directory_ingest_enabled": (http_collections, "directory_ingest_enabled"),
    "discover_collections": (http_collections, "discover_collections"),
    "encode_cursor": (archive_read_model, "encode_cursor"),
    "enforce_web_access": (http_boundary, "enforce_web_access"),
    "ensure_archive_catalog": (archive_read_model, "ensure_archive_catalog"),
    "evaluate_speech": (http_archives, "evaluate_speech"),
    "export_reviewed_ground_truth": (http_annotations, "export_reviewed_ground_truth"),
    "favicon": (http_health, "favicon"),
    "finalize_upload_session": (run_submission_service, "finalize_upload_session"),
    "fixed_archive_staging_paths": (ports, "fixed_archive_staging_paths"),
    "get_ai_settings": (http_settings, "get_ai_settings"),
    "get_collection": (ports, "get_collection"),
    "get_indexed_event": (archive_read_model, "get_indexed_event"),
    "get_indexed_evidence": (archive_read_model, "get_indexed_evidence"),
    "get_run": (run_read_model, "get_run"),
    "get_upload_session": (upload_service, "get_upload_session"),
    "hardware_summary": (run_read_model, "hardware_summary"),
    "hashlib": (archive_read_model, "hashlib"),
    "health": (http_health, "health"),
    "hmac": (upload_service, "hmac"),
    "index": (http_health, "index"),
    "indexed_evidence": (archive_read_model, "indexed_evidence"),
    "indexed_key_event": (archive_read_model, "indexed_key_event"),
    "indexed_physical_changes": (archive_read_model, "indexed_physical_changes"),
    "initialize_nas_archive": (run_submission_service, "initialize_nas_archive"),
    "is_allowed_lan_client": (http_boundary, "is_allowed_lan_client"),
    "json": (archive_read_model, "json"),
    "key_configured": (http_boundary, "key_configured"),
    "lightweight_release_integrity": (ports, "lightweight_release_integrity"),
    "limit_archive_streams": (http_boundary, "limit_archive_streams"),
    "list_archives": (archive_read_model, "list_archives"),
    "list_catalog_archives": (archive_read_model, "list_catalog_archives"),
    "list_collections": (http_collections, "list_collections"),
    "list_runs": (run_read_model, "list_runs"),
    "liveness": (http_health, "liveness"),
    "load_annotation_workspace": (http_annotations, "load_annotation_workspace"),
    "load_config": (ports, "load_config"),
    "load_device_registry": (upload_service, "load_device_registry"),
    "math": (upload_service, "math"),
    "model_candidates": (http_settings, "model_candidates"),
    "nas_recordings": (http_collections, "nas_recordings"),
    "open_archive_folder": (http_archives, "open_archive_folder"),
    "open_staging_folder": (http_archives, "open_staging_folder"),
    "os": (http_archives, "os"),
    "partial_result_available": (runtime_host, "partial_result_available"),
    "preflight_manifest_inputs": (ports, "preflight_manifest_inputs"),
    "prepare_from_nas_index": (ports, "prepare_from_nas_index"),
    "prioritize_retained_candidates": (
        archive_read_model,
        "prioritize_retained_candidates",
    ),
    "promote_fixed_archive": (ports, "promote_fixed_archive"),
    "quote": (archive_read_model, "quote"),
    "re": (http_boundary, "re"),
    "read_current_release_pointer": (ports, "read_current_release_pointer"),
    "record_annotation_decision": (http_annotations, "record_annotation_decision"),
    "record_collection_state": (ports, "record_collection_state"),
    "record_web_ingest_start": (http_boundary, "record_web_ingest_start"),
    "recovery_plan": (http_runs, "recovery_plan"),
    "refresh_stage": (http_runs, "refresh_stage"),
    "registered_partial_root": (run_read_model, "registered_partial_root"),
    "require_analysis_certification": (http_boundary, "require_analysis_certification"),
    "resolve_annotation_image": (http_annotations, "resolve_annotation_image"),
    "resolve_view_role": (upload_service, "resolve_view_role"),
    "retry_run": (http_runs, "retry_run"),
    "run_in_threadpool": (http_settings, "run_in_threadpool"),
    "run_staging_roots": (run_read_model, "run_staging_roots"),
    "runtime_status": (http_health, "runtime_status"),
    "safe_archive_name": (run_submission_service, "safe_archive_name"),
    "save_annotation_workspace_decision": (
        http_annotations,
        "save_annotation_workspace_decision",
    ),
    "scan_recordings": (ports, "scan_recordings"),
    "search_archive_index": (archive_read_model, "search_archive_index"),
    "search_catalog_events": (archive_read_model, "search_catalog_events"),
    "search_key_events": (archive_read_model, "search_key_events"),
    "search_physical_changes": (archive_read_model, "search_physical_changes"),
    "search_speech_library": (http_archives, "search_speech_library"),
    "search_staging_key_events": (archive_read_model, "search_staging_key_events"),
    "select_nas_recordings": (http_collections, "select_nas_recordings"),
    "selection_path": (http_collections, "selection_path"),
    "shutil": (http_archives, "shutil"),
    "speech": (http_archives, "speech"),
    "speech_worker": (http_archives, "speech_worker"),
    "sqlite3": (archive_read_model, "sqlite3"),
    "staging_archive_detail": (http_archives, "staging_archive_detail"),
    "staging_file": (http_archives, "staging_file"),
    "staging_key_event": (archive_read_model, "staging_key_event"),
    "staging_speech": (http_archives, "staging_speech"),
    "staging_speech_video": (http_archives, "staging_speech_video"),
    "subprocess": (http_archives, "subprocess"),
    "sys": (http_archives, "sys"),
    "threading": (runtime_host, "threading"),
    "time": (ports, "time"),
    "timing_summary": (run_read_model, "timing_summary"),
    "token_summary": (run_read_model, "token_summary"),
    "unicodedata": (run_submission_service, "unicodedata"),
    "upload_session_chunk": (upload_service, "upload_session_chunk"),
    "uuid": (http_collections, "uuid"),
    "validate_selection": (runtime_host, "validate_selection"),
    "validate_web_access_configuration": (ports, "validate_web_access_configuration"),
    "verify_ai_settings": (http_settings, "verify_ai_settings"),
    "verify_input_seal": (runtime_host, "verify_input_seal"),
    "web_access_mode": (http_boundary, "web_access_mode"),
    "web_https_required": (http_boundary, "web_https_required"),
    "with_operation_refreshes": (run_read_model, "with_operation_refreshes"),
    "write_input_seal": (run_submission_service, "write_input_seal"),
    "write_partial_delivery": (runtime_host, "write_partial_delivery"),
    "yaml": (run_submission_service, "yaml"),
}


@asynccontextmanager
async def _lifespan(_):
    if runtime_host.default_host.stream_configuration_error:
        raise RuntimeError(runtime_host.default_host.stream_configuration_error)
    ports.validate_web_access_configuration()
    async with runtime_host.default_host.lifespan():
        yield


app = FastAPI(
    title=PRODUCT_NAME, version=build_identity()["version"], lifespan=_lifespan
)
app.mount("/ui", StaticFiles(directory=Path(__file__).with_name("web")), name="ui")


@app.exception_handler(ApplicationError)
async def application_rejection(_, exc):
    return JSONResponse(
        {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
    )


http_boundary.install_middleware(app)
for adapter in (
    http_uploads,
    http_runs,
    http_archives,
    http_collections,
    http_settings,
    http_annotations,
    http_health,
):
    app.include_router(adapter.router)
install_device_day_routes(
    app,
    runtime_host.default_host.settings,
    runtime_host.default_host.device_day_service,
)
install_knowledge_routes(app, runtime_host.default_host.settings)


class _CompatibilityExports(ModuleType):
    """Historical imports resolve to one declared owner or dependency port."""

    def __getattr__(self, name):
        target = _COMPATIBILITY_EXPORTS.get(name)
        if target is None:
            raise AttributeError(name)
        owner, attribute = target
        return getattr(owner, attribute)

    def __setattr__(self, name, value):
        target = _COMPATIBILITY_EXPORTS.get(name)
        if target is None:
            super().__setattr__(name, value)
        else:
            owner, attribute = target
            setattr(owner, attribute, value)


sys.modules[__name__].__class__ = _CompatibilityExports

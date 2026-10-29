"""Invoke the pinned existing coarse/fine/audit chain without replacing its rules."""
from __future__ import annotations

import os
import time
from pathlib import Path

from .components import enable_components
from .rules import Rules


def screen(request):
    enable_components()
    from visioncortex_lab_legacy.device_day import retained_recording
    from visioncortex_lab_legacy.device_day_contract import DeviceDayLayout, digest
    from visioncortex_lab_legacy.device_day_models import DeviceDayModels
    from visioncortex_lab_legacy.device_day_write_guard import storage_guard
    from visioncortex_lab_legacy.nas_recordings import _inspect
    from .storage import verify_root, storage_binding

    source_root = verify_root(Path(request["source_root"]), os.environ.get("VC_LAB_NAS_MARKER_SHA256"))
    if request.get('unified_archive_required'):
        from .archive_layout import verify_archive_contract
        verify_archive_contract(source_root)
    original = request["native_record"]
    video = Path(original["video_path"])
    if video.is_symlink() or not video.resolve().is_relative_to(source_root):
        raise ValueError("Native video escaped its sealed source root")
    fresh = _inspect(source_root, video, time.time(), .5, allow_plain=False)
    if not fresh["processable"] or fresh["source_signature"] != original["source_signature"]:
        raise ValueError("Native recording changed after automatic intake")
    model = request["model"]
    # The role belongs to this explicitly configured customer camera. No site-camera defaults.
    role = model.get("camera_roles", {}).get(fresh["camera_key"])
    if role not in {"first_person", "third_person"}:
        raise ValueError("Customer camera role registration is required")
    fresh["configured_role"] = role
    runtime = Path(request["runtime_root"]) / "native"
    config = Rules().vision_configuration(model, runtime)
    config["collection_ingest"] = {"source_root": str(source_root), "settle_seconds": .5}
    config["storage"].update(archive_root=str(source_root), local_staging_root=str(runtime / "staging"))
    binding = storage_binding(source_root, os.environ.get("VC_LAB_NAS_MARKER_SHA256"))
    config["runtime"].update(binding["runtime"])
    config["device_day"].update(binding.get("device_day", {}))
    # A single native model job runs under a bounded local workflow. No old queue/database is opened.
    raw = next(p for p in video.relative_to(source_root).parts if p in {"MetaData", "MetaVideo"})
    layout = DeviceDayLayout(source_root, fresh["camera_key"], fresh["recording_start_us"], runtime / "cache", raw_directory=raw, read_only_originals=True)
    with storage_guard(config):
        layout.create()
        retained = retained_recording(layout, fresh, config)
        backend = DeviceDayModels(config)
        result = backend.vision(layout, retained, digest([request["job_id"], model["revision"], Rules().identity()]))
    return dict(result, analysis_outcome="observations", association_status="single_device_activity", normalized_fields=[],
                component="visioncortex.DeviceDayModels.vision", screening_chain="existing_coarse_fine_coverage_audit_local_recall", physical_action_confirmed=False)

"""Retain the complete recorder file set, excluding depth video containers."""
from __future__ import annotations

import time
from pathlib import Path

from .device_day_contract import artifact, atomic_json, media_sidecar_name, read_json, safe_child


def retain_capture_files(runner, record):
    from .device_day import copy_verified, exclusive
    layout = runner.layout(record)
    receipt_path = runner._receipt(layout, record, "retention")
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    if receipt.get("status") != "completed":
        return None
    root = Path(record["video_path"]).parent
    existing = {s["original_path"]: s["retained"] for s in receipt["sources"]}
    video = next(s["retained"] for s in receipt["sources"] if s["kind"] == "video")
    folder = layout.raw / "Metadata" / Path(video["path"]).stem
    path = folder / "CaptureFiles.json"
    with exclusive(runner.runtime_root / "locks" / f"{record['recording_id']}.capture-files.lock"):
        files, excluded, pending, destinations = [], [], [], set()
        for source in sorted(root.rglob("*")):
            relative = source.relative_to(root)
            if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != root and root in parent.parents):
                excluded.append({"original_name": relative.as_posix(), "reason": "symlink_not_followed"})
                continue
            if not source.is_file():
                continue
            if "depth" in source.stem.lower() and source.suffix.lower() in {".mkv", ".mp4", ".avi", ".mov", ".webm", ".bag"}:
                excluded.append({"original_name": relative.as_posix(), "reason": "depth_video_excluded_by_user"})
                continue
            reference = existing.get(str(source))
            if reference is None:
                if time.time() - source.stat().st_mtime < runner.config["collection_ingest"].get("settle_seconds", 120):
                    pending.append({"original_name": relative.as_posix(), "reason": "waiting_for_file_stability"})
                    continue
                destination = folder / "Capture"
                for part in relative.parts:
                    destination /= media_sidecar_name(Path(part))
                identity = destination.as_posix().casefold()
                if identity in destinations:
                    raise ValueError("Capture file names collide after PascalCase conversion")
                destinations.add(identity)
                copy_verified(source, destination)
                reference = artifact(layout.root, destination)
            files.append({"original_name": relative.as_posix(), "original_path": str(source), "retained": reference})
        catalog = {"recording_id": record["recording_id"], "camera_key": record["camera_key"],
                   "archive_date": layout.name[:10], "files": files, "excluded": excluded,
                   "pending": pending, "status": "waiting" if pending else "completed",
                   "capture_sources_changed": False}
        if not path.is_file() or read_json(path) != catalog:
            atomic_json(path, catalog)
        reference = artifact(layout.root, path)
        pointer = layout.receipts / record["recording_id"] / "capture-files.json"
        if not pointer.is_file() or read_json(pointer) != reference:
            atomic_json(pointer, reference)
        return reference


def publish_capture_catalog(runner, layout):
    """Add auxiliary-file references without changing visual model evidence."""
    if not layout.index.is_file():
        return
    from .device_day import exclusive
    try:
        with exclusive(runner.runtime_root / "locks" / f"{layout.name}.index.lock"):
            index = read_json(layout.index)
            changed = False
            for record in index.get("recordings", []):
                path = layout.receipts / record["recording_id"] / "capture-files.json"
                if path.is_file():
                    reference = read_json(path)
                    if safe_child(layout.root, reference["path"]).is_file() and record.get("capture_manifest") != reference:
                        record["capture_manifest"] = reference
                        changed = True
            if changed:
                atomic_json(layout.index, index)
                from .device_day_reports import render_day
                render_day(layout, index)
    except BlockingIOError:
        return


def reconcile_capture_files(runner, *, date=None):
    """Finish late auxiliary uploads independently of visual/semantic model work."""
    results, layouts = [], {}
    for path in sorted((runner.backend_root / "device-day-receipts").glob(f"{date or '*'}_*/*/retention.json")):
        retained = read_json(path)
        record = retained.get("recording")
        if retained.get("status") != "completed" or not record:
            continue
        layout = runner.layout(record)
        layouts[layout.name] = layout
        try:
            reference = retain_capture_files(runner, record)
            results.append({"recording_id": record["recording_id"], "manifest": reference})
        except (OSError, ValueError) as exc:
            results.append({"recording_id": record["recording_id"], "status": "failed", "message": str(exc)})
    for layout in layouts.values():
        publish_capture_catalog(runner, layout)
    return results

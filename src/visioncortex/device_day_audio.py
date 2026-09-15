"""Recorder audio discovery; completeness and publication are separate attributes."""
from __future__ import annotations

from pathlib import Path

from .device_day_contract import digest


def recorder_audio(video: Path, metadata: dict, ready: dict, now: float, settle: float) -> dict:
    from .nas_recordings import _child, _json
    prefix = video.name[:-len("rgb.mp4")] if video.name.endswith("rgb.mp4") else ""
    merged = metadata | ready
    names = {"audio": merged.get("task_audio_file") or merged.get("audio_file") or prefix + "audio.opus",
             "metadata": merged.get("task_audio_meta_file") or prefix + "audio_meta.json",
             "ready": merged.get("task_audio_ready_file") or prefix + "audio_ready.json",
             "timing": merged.get("task_audio_timing_file") or prefix + "audio_timing.csv"}
    paths = {kind: _child(video.parent, name) for kind, name in names.items()}
    try:
        meta = _json(paths["metadata"]) if paths["metadata"].is_file() else {}
        published = _json(paths["ready"]) if paths["ready"].is_file() else {}
    except OSError as error:
        # A locked/unavailable audio sidecar cannot invalidate a separately
        # finalized RGB recording. Fail closed for audio association only.
        issue = {"path": str(error.filename or paths["metadata"]), "errno": error.errno,
                 "reason": "audio_metadata_unreadable"}
        return {"status": "pending_publication", "files": [],
                "source_signature": digest({"paths": {k: str(p) for k, p in paths.items()}, "issue": issue}),
                "quality_status": "unknown", "capture_complete": None, "start_us": None,
                "end_us": None, "duration_us": None, "association_issues": [issue],
                "time_basis": "unknown", "transcription_status": "awaiting_audio",
                "read_errors": [issue]}
    snapshots = [(kind, str(path), path.stat().st_size, path.stat().st_mtime_ns)
                 for kind, path in paths.items() if path.is_file()]
    stable = all(now - row[3] / 1e9 >= settle for row in snapshots)
    exists = paths["audio"].is_file() and paths["audio"].stat().st_size > 0
    mismatches = [field for field in ("recording_session_id", "sender_id")
                  if len({str(d[field]) for d in (metadata, ready, meta, published) if d.get(field)}) > 1]
    state = ("association_mismatch" if mismatches else "pending_publication" if exists and
             (published.get("ready") is not True or not stable) else "provided" if exists else
             "no_input" if (meta | published).get("quality_status") == "no_input" else "not_provided")
    copyable = stable and not mismatches
    files = [{"kind": kind, "path": str(path)} for kind, path in paths.items()
             if path.is_file() and copyable and (kind != "audio" or state == "provided")]
    return {"status": state, "files": files, "source_signature": digest(snapshots),
            "quality_status": (meta | published).get("quality_status"),
            "capture_complete": (meta | published).get("audio_valid"),
            "start_us": meta.get("first_audio_global_us"), "end_us": meta.get("last_audio_global_us"),
            "duration_us": meta.get("audio_duration_us"), "association_issues": mismatches,
            "time_basis": "recorder_audio_metadata" if meta.get("first_audio_global_us") else "unknown",
            "transcription_status": "pending" if state == "provided" else "awaiting_audio"}

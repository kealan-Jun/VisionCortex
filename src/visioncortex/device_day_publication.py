"""Keep the visible clip list current; preserve superseded material in history."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from .device_day import exclusive, visual_input
from .device_day_contract import (
    artifact, atomic_json, digest, read_json, safe_child,
    validate_archive_name,
)


def _kept_clip_folders(layout, vision, record):
    """Candidate receipts and canonical exact intervals share one namespace."""
    parents = {s['segment_id']: s for s in vision.get('segments', [])}
    keep = {Path(s['json_path']).parent.as_posix() for s in parents.values()}
    if not layout.index.is_file():
        return keep
    if layout.index.stat().st_size > 8*1024*1024:
        # Optional legacy tidying is never worth hydrating a large canonical
        # day or risking a child's reference. Defer every move for this record.
        return None
    index = read_json(layout.index)
    from .device_day_content_paths import clip_name
    fields = ('recording_id', 'source_ref', 'start_ms', 'end_ms', 'start_us', 'end_us')
    for candidate in index.get('segments', []):
        parent = parents.get(candidate.get('segment_id'))
        if (not parent or candidate.get('recording_id') != record['recording_id']
                or any(candidate.get(key) != parent.get(key) for key in fields)):
            continue
        for child in candidate.get('retention_intervals') or []:
            if (child.get('candidate_segment_id') != candidate['segment_id']
                    or child.get('recording_id') != record['recording_id']
                    or child.get('source_ref') != candidate.get('source_ref')):
                continue
            # This protects paths, never grants completion or accepts bytes.
            keep.add('ProcessedClips/Clips/' + clip_name(child))
    return keep


def reconcile_outputs(runner):
    relocated = []
    if not runner.archive_root.is_dir():
        return {"relocated": relocated}
    for root in sorted(runner.archive_root.iterdir()):
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            validate_archive_name(root.name)
        except ValueError:
            continue
        receipts = safe_child(runner.backend_root, f"device-day-receipts/{root.name}")
        for receipt_dir in sorted(receipts.glob("*")):
            if not receipt_dir.is_dir() or not (receipt_dir / "retention.json").is_file():
                continue
            try:
                with ExitStack() as locks:
                    for stage in ("all", "retention", "vision", "stt", "understanding", "report"):
                        locks.enter_context(exclusive(runner.runtime_root / "locks" / f"{receipt_dir.name}.{stage}.lock"))
                    retention = read_json(receipt_dir / "retention.json")
                    record = retention.get("recording")
                    if not record:
                        continue
                    layout = runner.layout(record)
                    if layout.root != root:
                        continue
                    path = receipt_dir / "vision.json"
                    vision = read_json(path) if path.is_file() else {}
                    current = (vision.get("status") == "completed" and
                               vision.get("key") == runner._key("vision", record, visual_input(retention)))
                    if not current:
                        # A code/configuration change is not a replacement
                        # artifact. Keep the preceding visible result until
                        # its successor has successfully sealed its outputs.
                        continue
                    keep = _kept_clip_folders(layout, vision, record)
                    if keep is None:
                        continue
                    for folder in sorted((layout.processed / "Clips").glob("*")):
                        if not folder.is_dir() or folder.is_symlink() or layout.relative(folder) in keep:
                            continue
                        metadata = list(folder.glob("*.json"))
                        if not metadata:
                            continue
                        segment = read_json(metadata[0])
                        if segment.get('scope') == 'aligned_experiment':
                            continue
                        if (segment.get('record_scope') == 'retention_interval'
                                or segment.get('schema_version') == 'visioncortex-retained-interval/1'
                                or segment.get('candidate_segment_id') not in {None, segment.get('segment_id')}):
                            # A child can become durable before its canonical
                            # replay. Keep it, including older child history;
                            # reconciling CV candidates must never relocate it.
                            continue
                        if segment.get("recording_id") != record["recording_id"]:
                            continue
                        before = [artifact(root, p) for p in sorted(folder.rglob("*")) if p.is_file()]
                        destination = receipt_dir / "history" / "superseded-clips" / folder.name
                        relative_destination = destination.relative_to(runner.backend_root).as_posix()
                        safe_child(runner.backend_root, relative_destination)
                        if destination.exists():
                            raise ValueError("Historical clip destination already exists")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        old_prefix = layout.relative(folder)
                        folder.rename(destination)
                        entry = {"reason": "superseded_or_unpublished_stage_output", "from": old_prefix,
                                 "to": relative_destination, "storage_root": "local_cache_root", "artifacts_before": before,
                                 "artifacts_after": [layout.backend_artifact(p) for p in sorted(destination.rglob("*")) if p.is_file()],
                                 "source_media_changed": False, "historical_references": "resolve_original_prefix_using_from_to_mapping"}
                        atomic_json(receipt_dir / "history" / f"relocation-{digest(entry)}.json", entry)
                        relocated.append(entry)
            except BlockingIOError:
                # A live stage owns this record. Never relocate its material.
                continue
        if receipts.is_dir():
            from .device_day_capture_files import publish_capture_catalog
            for receipt_path in receipts.glob("*/retention.json"):
                retained = read_json(receipt_path)
                if retained.get("recording"):
                    publish_capture_catalog(runner, runner.layout(retained["recording"]))
                    break
    return {"relocated": relocated, "source_media_changed": False}


def write_archive_guide(runner):
    """CLI and service publish the same user-facing overview."""
    from .device_day_overview import ArchiveOverview
    return ArchiveOverview().publish(runner)

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

import yaml

from .schemas import RunManifest, VideoSegmentInput, ViewInput, ViewRole


ARCHIVE_DIRECTORIES = (
    "Experiment-Clips",
    "JSON-Config-Files",
    "Key-Materials",
    "Lab-Daily-Reports",
    "Original-Experiment-Videos",
    "Professional-PDFs",
)

DERIVED_ARCHIVE_DIRECTORIES = (
    "Experiment-Clips",
    "JSON-Config-Files",
    "Key-Materials",
    "Lab-Daily-Reports",
    "Professional-PDFs",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _directory_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": source.relative_to(root).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": _sha256_file(source),
        }
        for source in sorted(path for path in root.rglob("*") if path.is_file())
    ]


def safe_archive_name(value: str) -> str:
    """Keep readable Unicode names while excluding Windows-invalid characters."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", value).strip(" .-")
    return cleaned[:160] or f"Experiment-{uuid.uuid4().hex[:8]}"


def initialize_nas_archive(config: dict[str, Any], experiment_name: str) -> Path:
    """Create the durable six-folder archive before ingesting the first byte."""
    storage = config["storage"]
    explicit = storage.get("active_archive_path")
    root = Path(explicit) if explicit else Path(storage["archive_root"]) / safe_archive_name(experiment_name)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ARCHIVE_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def fixed_archive_staging_paths(
    config: dict[str, Any], archive_name: str, run_id: str
) -> tuple[Path, Path, Path]:
    archive_root = Path(config["storage"]["archive_root"])
    fixed_root = archive_root / safe_archive_name(archive_name)
    staging_root = archive_root / ".VisionCortex-Run-Staging" / archive_name / run_id
    history_root = archive_root / ".VisionCortex-Run-History" / archive_name / run_id
    return fixed_root, staging_root, history_root


def promote_fixed_archive(
    staging_root: Path,
    fixed_root: Path,
    history_root: Path,
) -> dict[str, Any]:
    """Promote one validated run while retaining the previous derived package."""

    evaluation_path = staging_root / "JSON-Config-Files" / "evidence_package_eval.json"
    if not evaluation_path.is_file():
        raise RuntimeError(f"Staged evidence evaluation is missing: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
    if not evaluation.get("passed"):
        raise RuntimeError("Staged evidence package did not pass evaluation; promotion refused")
    quality_path = staging_root / "JSON-Config-Files" / "quality_acceptance.json"
    if not quality_path.is_file():
        raise RuntimeError(f"Staged quality acceptance is missing: {quality_path}")
    quality = json.loads(quality_path.read_text(encoding="utf-8-sig"))
    if not quality.get("passed"):
        raise RuntimeError("Staged experiment/material quality did not pass; promotion refused")
    report_evaluations = list(
        (staging_root / "Lab-Daily-Reports").glob("*/Daily-Report-Eval.json")
    )
    if not report_evaluations:
        raise RuntimeError("Staged daily report evaluation is missing; promotion refused")
    if not all(
        json.loads(path.read_text(encoding="utf-8-sig")).get("passed")
        for path in report_evaluations
    ):
        raise RuntimeError("Staged daily report did not pass evaluation; promotion refused")
    if not any((staging_root / "Professional-PDFs").glob("Lab-Daily-Report-*.pdf")):
        raise RuntimeError("Staged daily report PDF is missing; promotion refused")

    staged_manifests = {
        directory: _directory_manifest(staging_root / directory)
        for directory in DERIVED_ARCHIVE_DIRECTORIES
        if (staging_root / directory).is_dir()
    }
    missing_manifests = [
        directory for directory in DERIVED_ARCHIVE_DIRECTORIES if directory not in staged_manifests
    ]
    if missing_manifests:
        raise RuntimeError(f"Staged directories are missing: {missing_manifests}")

    fixed_root.mkdir(parents=True, exist_ok=True)
    history_root.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path, Path, bool]] = []
    try:
        for directory in DERIVED_ARCHIVE_DIRECTORIES:
            source = staging_root / directory
            destination = fixed_root / directory
            backup = history_root / directory
            if not source.is_dir():
                raise RuntimeError(f"Staged directory is missing: {source}")
            had_previous = destination.is_dir()
            if had_previous:
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            try:
                os.replace(source, destination)
            except Exception:
                if had_previous and backup.is_dir() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if _directory_manifest(destination) != staged_manifests[directory]:
                if destination.is_dir() and not source.exists():
                    os.replace(destination, source)
                if had_previous and backup.is_dir() and not destination.exists():
                    os.replace(backup, destination)
                raise RuntimeError(f"Promoted directory checksum mismatch: {directory}")
            moved.append((source, destination, backup, had_previous))
    except Exception:
        for source, destination, backup, had_previous in reversed(moved):
            if destination.is_dir() and not source.exists():
                os.replace(destination, source)
            if had_previous and backup.is_dir() and not destination.exists():
                os.replace(backup, destination)
        raise

    receipt = {
        "schema_version": "visioncortex-fixed-archive-promotion/1",
        "fixed_root": str(fixed_root),
        "staging_root": str(staging_root),
        "history_root": str(history_root),
        "promoted_directories": list(DERIVED_ARCHIVE_DIRECTORIES),
        "previous_package_retained": any(item[3] for item in moved),
        "verification": {
            "algorithm": "sha256",
            "status": "verified",
            "directory_file_counts": {
                directory: len(manifest) for directory, manifest in staged_manifests.items()
            },
        },
    }
    receipt_path = fixed_root / "JSON-Config-Files" / "fixed_archive_promotion.json"
    temporary = receipt_path.with_name(f".{receipt_path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt


class IncrementalArchivePublisher:
    """Atomically publish completed files without exposing partial artifacts."""

    def __init__(self, local_root: Path, nas_root: Path):
        self.local_root = local_root.resolve()
        self.nas_root = nas_root

    def publish_file(self, source: Path) -> Path:
        source = source.resolve()
        relative = source.relative_to(self.local_root)
        destination = self.nas_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_size = source.stat().st_size
        if destination.is_file() and destination.stat().st_size == source_size:
            if _sha256_file(destination) == _sha256_file(source):
                return destination
        temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:8]}")
        try:
            shutil.copy2(source, temporary)
            if temporary.stat().st_size != source_size or _sha256_file(temporary) != _sha256_file(source):
                raise IOError(f"NAS published file verification failed: {destination}")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def publish_directory(self, relative: str | Path) -> int:
        source_root = self.local_root / relative
        if not source_root.is_dir():
            return 0
        count = 0
        for source in source_root.rglob("*"):
            if source.is_file() and ".work" not in source.relative_to(self.local_root).parts:
                self.publish_file(source)
                count += 1
        return count

    def publish_status(self, payload: dict[str, Any]) -> Path:
        destination = self.nas_root / "JSON-Config-Files" / "pipeline_status.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:8]}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination


KNOWN_ROLE_OVERRIDES = {
    # The NAS index says "first", but this source is obstructed in this exact
    # validation recording. The wearable lubancat stream is the usable FP view.
    "exp_20260810_144014_e918b762": {
        "lubancat-e8cc0cb3_cam01": ViewRole.FIRST_PERSON,
        "orangepi5pro-d12a4719_cam01": ViewRole.THIRD_PERSON,
    }
}


def _parts(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def _resolve_nas_path(value: str, index_csv: Path) -> Path:
    """Resolve stale Z: index entries through the index CSV's Y: share first."""
    candidate = Path(value)
    pure = PureWindowsPath(value)
    if pure.drive.upper() == "Z:":
        remapped = Path(index_csv.drive + "\\" + str(pure.relative_to(pure.anchor)))
        if remapped.exists():
            return remapped
    if candidate.exists():
        return candidate
    return candidate


def read_index_experiment(index_csv: Path, experiment_id: str) -> list[dict[str, str]]:
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("experiment_id") == experiment_id]
    if not rows:
        raise KeyError(f"experiment_id not found in {index_csv}: {experiment_id}")
    return rows


def describe_index_experiment(index_csv: Path, experiment_id: str) -> dict[str, Any]:
    rows = read_index_experiment(index_csv, experiment_id)
    cameras = []
    for row in rows:
        rgb = _parts(row.get("rgb_file"))
        frames = _parts(row.get("frames_file"))
        cameras.append(
            {
                "camera_key": row.get("camera_key"),
                "index_camera_view": row.get("camera_view"),
                "recording_start_time": row.get("recording_start_time"),
                "recording_end_time": row.get("recording_end_time"),
                "segment_count_declared": int(row.get("segment_count") or len(rgb)),
                "rgb_segments": rgb,
                "clock_segments": frames,
                "nas_segment_dirs": _parts(row.get("nas_segment_dir")),
            }
        )
    starts = [int(row["recording_start_us"]) for row in rows if row.get("recording_start_us")]
    ends = [int(row["recording_end_us"]) for row in rows if row.get("recording_end_us")]
    return {
        "experiment_id": experiment_id,
        "index_csv": str(index_csv),
        "experiment_prefix": rows[0].get("experiment_prefix"),
        "camera_count": len(cameras),
        "recording_hours": round((max(ends) - min(starts)) / 3_600_000_000, 6)
        if starts and ends
        else None,
        "cameras": cameras,
    }


def prepare_from_nas_index(
    config: dict[str, Any],
    experiment_id: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[RunManifest, Path, dict[str, Any]]:
    progress = progress or (lambda _: None)
    storage = config["storage"]
    index_csv = Path(storage["index_csv"])
    description = describe_index_experiment(index_csv, experiment_id)
    rows = read_index_experiment(index_csv, experiment_id)
    active_archive = storage.get("active_archive_path")
    if active_archive and storage.get("manifest_storage", "nas") == "nas":
        manifest_root = Path(active_archive) / "JSON-Config-Files" / "Input-Manifests"
    else:
        manifest_root = Path(storage["local_runtime_root"]) / "input-manifests" / experiment_id
    manifest_root.mkdir(parents=True, exist_ok=True)
    overrides = KNOWN_ROLE_OVERRIDES.get(experiment_id, {})

    def prepare(row: dict[str, str]) -> ViewInput:
        camera_key = str(row["camera_key"])
        progress(f"Registering NAS segments without copying: {camera_key}")
        videos = [_resolve_nas_path(item, index_csv) for item in _parts(row.get("rgb_file"))]
        clocks = [_resolve_nas_path(item, index_csv) for item in _parts(row.get("frames_file"))]
        missing = [str(item) for item in videos + clocks if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"NAS source missing for {camera_key}: {missing[:4]}")
        if clocks and len(clocks) != len(videos):
            raise ValueError(
                f"NAS segment/clock count mismatch for {camera_key}: "
                f"{len(videos)} videos, {len(clocks)} clock CSVs"
            )
        if storage.get("require_nas_source_paths"):
            expected_drive = index_csv.drive.upper()
            off_nas = [str(item) for item in videos + clocks if item.drive.upper() != expected_drive]
            if off_nas:
                raise ValueError(f"Source path escaped NAS drive {expected_drive}: {off_nas[:4]}")
        role = overrides.get(camera_key)
        if role is None:
            role = ViewRole.FIRST_PERSON if row.get("camera_view") == "first" else ViewRole.THIRD_PERSON
        return ViewInput(
            view_id=camera_key,
            role=role,
            segments=[
                VideoSegmentInput(
                    video=video,
                    timestamps_csv=clocks[index] if clocks else None,
                )
                for index, video in enumerate(videos)
            ],
        )

    views = [prepare(row) for row in rows]
    segment_counts = {view.view_id: len(view.segments) for view in views}
    if config["performance"].get("synchronized_segment_waves") and len(set(segment_counts.values())) != 1:
        raise ValueError(f"Synchronized segment waves require equal segment counts: {segment_counts}")
    manifest = RunManifest(experiment_id=experiment_id, views=views)
    manifest_path = manifest_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    ingest = {
        **description,
        "input_mode": "nas_segmented_virtual_timeline",
        "copied_source_bytes": 0,
        "continuous_source_copies_created": 0,
        "manifest_root": str(manifest_root),
        "manifest": str(manifest_path),
        "segment_counts": segment_counts,
        "role_overrides": {key: value.value for key, value in overrides.items()},
    }
    (manifest_root / "nas_ingest.json").write_text(
        json.dumps(ingest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, manifest_path, ingest


def sync_archive_to_nas(local_root: Path, config: dict[str, Any], experiment_id: str) -> Path:
    archive_root = Path(config["storage"]["archive_root"])
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / experiment_id
    if destination.exists():
        destination = archive_root / f"{experiment_id}-{uuid.uuid4().hex[:8]}"
    temporary = archive_root / f".{destination.name}.partial-{uuid.uuid4().hex[:8]}"
    shutil.copytree(local_root, temporary)
    eval_path = temporary / "JSON-Config-Files" / "evidence_package_eval.json"
    if not eval_path.is_file() or not json.loads(eval_path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError(f"Copied archive failed validation; partial copy retained at {temporary}")
    os.replace(temporary, destination)
    return destination

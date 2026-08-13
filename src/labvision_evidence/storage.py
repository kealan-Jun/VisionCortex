from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

import yaml

from .schemas import RunManifest, ViewInput, ViewRole


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
)


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
        temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:8]}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
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


def _concat_videos(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    list_path = destination.with_suffix(".concat.txt")
    lines = ["file '" + str(path.resolve()).replace("'", "'\\''") + "'" for path in paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-2000:])
    finally:
        list_path.unlink(missing_ok=True)


def _merge_clock_csvs(paths: list[Path], destination: Path) -> None:
    """Append segment CSVs while making frame indexes monotonically increasing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    fields: list[str] = []
    frame_offset = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            if not rows:
                continue
            if not fields:
                fields = list(rows[0])
            frame_field = next(
                (name for name in fields if name.lower() in {"frame", "frame_index", "rgb_frame_index"}),
                None,
            )
            if frame_field:
                values = []
                for row in rows:
                    try:
                        original = int(float(row[frame_field]))
                    except (TypeError, ValueError):
                        original = len(values)
                    values.append(original)
                    row[frame_field] = str(original + frame_offset)
                frame_offset += max(values, default=-1) + 1
            output_rows.extend(rows)
    if not fields or not output_rows:
        raise RuntimeError(f"No timestamp rows found in: {paths}")
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)


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
    input_root = Path(storage["local_input_root"]) / experiment_id
    input_root.mkdir(parents=True, exist_ok=True)
    overrides = KNOWN_ROLE_OVERRIDES.get(experiment_id, {})

    def prepare(row: dict[str, str]) -> ViewInput:
        camera_key = str(row["camera_key"])
        progress(f"Copying and concatenating {camera_key}")
        view_root = input_root / camera_key
        videos = [_resolve_nas_path(item, index_csv) for item in _parts(row.get("rgb_file"))]
        clocks = [_resolve_nas_path(item, index_csv) for item in _parts(row.get("frames_file"))]
        missing = [str(item) for item in videos + clocks if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"NAS source missing for {camera_key}: {missing[:4]}")
        video_destination = view_root / "video.mp4"
        clock_destination = view_root / "clock.csv"
        if not video_destination.is_file():
            _concat_videos(videos, video_destination)
        if not clock_destination.is_file():
            _merge_clock_csvs(clocks, clock_destination)
        role = overrides.get(camera_key)
        if role is None:
            role = ViewRole.FIRST_PERSON if row.get("camera_view") == "first" else ViewRole.THIRD_PERSON
        return ViewInput(
            view_id=camera_key,
            role=role,
            video=video_destination,
            timestamps_csv=clock_destination,
        )

    worker_count = min(int(config["performance"].get("io_workers", 4)), len(rows))
    with ThreadPoolExecutor(max_workers=max(1, worker_count), thread_name_prefix="nas-stage") as executor:
        views = list(executor.map(prepare, rows))
    manifest = RunManifest(experiment_id=experiment_id, views=views)
    manifest_path = input_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    ingest = {
        **description,
        "local_input_root": str(input_root),
        "manifest": str(manifest_path),
        "role_overrides": {key: value.value for key, value in overrides.items()},
    }
    (input_root / "nas_ingest.json").write_text(
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

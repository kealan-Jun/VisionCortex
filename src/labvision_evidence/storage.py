from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat as stat_module
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Sequence

import yaml

from .device_registry import load_device_registry, resolve_view_role
from .input_seal import build_input_seal, write_input_seal
from .schemas import RunManifest, VideoSegmentInput, ViewInput, ViewRole
from .source_path_migrations import SourcePathResolver


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

ORIGINAL_REFERENCE_NAMES = {"Original-Video-Index.json", "README.txt"}


_SOURCE_STAT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SOURCE_STAT_LOCK = threading.Lock()


def _source_cache_key(path: Path) -> str:
    """Return a stable absolute key without resolving the path through SMB."""

    return os.path.normcase(os.path.abspath(str(path)))


def _source_stat(path: Path) -> dict[str, Any]:
    try:
        value = path.stat()
    except OSError as exc:
        return {
            "is_file": False,
            "size_bytes": None,
            "mtime_ns": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    is_file = stat_module.S_ISREG(value.st_mode)
    return {
        "is_file": is_file,
        "size_bytes": int(value.st_size) if is_file else None,
        "mtime_ns": int(value.st_mtime_ns) if is_file else None,
        "error": None if is_file else "path is not a regular file",
    }


def snapshot_source_paths(
    paths: Sequence[Path],
    *,
    workers: int = 24,
    max_age_seconds: float = 120.0,
    refresh: bool = False,
) -> tuple[dict[Path, dict[str, Any]], dict[str, Any]]:
    """Stat high-latency source paths once, concurrently, and reuse briefly.

    The cache lifetime only bridges ingest, cache identity and preflight inside
    one run. It is intentionally short so a later upload or deletion is still
    observed by a long-lived web process.
    """

    started = time.perf_counter()
    ordered = list(dict.fromkeys(Path(path) for path in paths))
    now = time.monotonic()
    snapshots: dict[Path, dict[str, Any]] = {}
    pending: list[Path] = []
    cache_hits = 0
    with _SOURCE_STAT_LOCK:
        for path in ordered:
            cached = _SOURCE_STAT_CACHE.get(_source_cache_key(path))
            if (
                not refresh
                and cached is not None
                and now - cached[0] <= max(0.0, float(max_age_seconds))
            ):
                snapshots[path] = dict(cached[1])
                cache_hits += 1
            else:
                pending.append(path)

    if pending:
        with ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(pending))),
            thread_name_prefix="source-stat",
        ) as executor:
            futures = {executor.submit(_source_stat, path): path for path in pending}
            for future in as_completed(futures):
                path = futures[future]
                snapshot = future.result()
                snapshots[path] = snapshot
                with _SOURCE_STAT_LOCK:
                    _SOURCE_STAT_CACHE[_source_cache_key(path)] = (
                        time.monotonic(),
                        dict(snapshot),
                    )

    ordered_snapshots = {path: snapshots[path] for path in ordered}
    missing_count = sum(not item["is_file"] for item in ordered_snapshots.values())
    report = {
        "path_count": len(ordered),
        "verified_file_count": len(ordered) - missing_count,
        "missing_count": missing_count,
        "cache_hit_count": cache_hits,
        "fresh_stat_count": len(pending),
        "workers": max(1, min(int(workers), max(1, len(pending)))),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "cache_ttl_seconds": float(max_age_seconds),
    }
    return ordered_snapshots, report


@lru_cache(maxsize=256)
def _read_source_file_edges_cached(
    path_text: str,
    size_bytes: int,
    mtime_ns: int,
    window_bytes: int,
) -> tuple[bytes, bytes, int]:
    del mtime_ns  # part of the cache identity
    path = Path(path_text)
    with path.open("rb") as handle:
        head = handle.read(window_bytes)
        handle.seek(max(0, size_bytes - window_bytes))
        tail = handle.read()
    return head, tail, size_bytes


def read_source_file_edges(
    path: Path,
    *,
    window_bytes: int = 512 * 1024,
) -> tuple[bytes, bytes, int]:
    """Read and cache immutable CSV head/tail windows for preflight + alignment."""

    snapshots, _ = snapshot_source_paths([path], workers=1)
    snapshot = snapshots[Path(path)]
    if not snapshot["is_file"]:
        raise FileNotFoundError(f"source file does not exist: {path}")
    return _read_source_file_edges_cached(
        _source_cache_key(Path(path)),
        int(snapshot["size_bytes"]),
        int(snapshot["mtime_ns"]),
        max(4096, int(window_bytes)),
    )


def _bounded_source_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint a large NAS source with bounded edge reads and explicit strength."""

    head, tail, size_bytes = read_source_file_edges(path, window_bytes=64 * 1024)
    digest = hashlib.sha256(b"visioncortex-source-edge-fingerprint-v1\n")
    digest.update(str(size_bytes).encode("ascii"))
    digest.update(b"\n")
    digest.update(head)
    digest.update(b"\n--tail--\n")
    digest.update(tail)
    return {
        "source_fingerprint": digest.hexdigest(),
        "source_fingerprint_algorithm": "sha256-size-plus-64k-head-tail-v1",
        "identity_strength": "bounded_fingerprint_not_full_content_hash",
    }


def source_cache_diagnostics() -> dict[str, int]:
    edge_info = _read_source_file_edges_cached.cache_info()
    with _SOURCE_STAT_LOCK:
        stat_entries = len(_SOURCE_STAT_CACHE)
    return {
        "source_stat_entries": stat_entries,
        "edge_cache_hits": edge_info.hits,
        "edge_cache_misses": edge_info.misses,
        "edge_cache_entries": edge_info.currsize,
    }


def clear_source_metadata_cache() -> None:
    """Clear process-local metadata caches (primarily for isolated tests)."""

    with _SOURCE_STAT_LOCK:
        _SOURCE_STAT_CACHE.clear()
    _read_source_file_edges_cached.cache_clear()


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
    """Return a deterministic ASCII-only component for SMB/tool compatibility."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(" .-_")
    if cleaned.casefold() == "processing":
        cleaned = "Experiment-Processing"
    if cleaned:
        return cleaned[:120]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"Experiment-{digest}"


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:8]}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _original_reference_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and (path.name in ORIGINAL_REFERENCE_NAMES or path.suffix.lower() == ".ffconcat")
    )


def _publish_original_references(staging_root: Path, fixed_root: Path) -> list[str]:
    """Merge zero-copy source references without touching retained source media."""

    source_root = staging_root / "Original-Experiment-Videos"
    destination_root = fixed_root / "Original-Experiment-Videos"
    published: list[str] = []
    for source in _original_reference_files(source_root):
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / source.name
        temporary = destination.with_name(
            f".{destination.name}.partial-{uuid.uuid4().hex[:8]}"
        )
        try:
            shutil.copy2(source, temporary)
            if (
                temporary.stat().st_size != source.stat().st_size
                or _sha256_file(temporary) != _sha256_file(source)
            ):
                raise IOError(f"Original reference verification failed: {destination}")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        published.append(destination.relative_to(fixed_root).as_posix())
    return published


def initialize_nas_archive(config: dict[str, Any], experiment_name: str) -> Path:
    """Create the durable six-folder archive before ingesting the first byte."""
    storage = config["storage"]
    explicit = storage.get("active_archive_path")
    root = Path(explicit) if explicit else Path(storage["archive_root"]) / safe_archive_name(experiment_name)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ARCHIVE_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    if explicit:
        (root / "处理状态.txt").write_text(
            "处理中：各环节完成后，产出会出现在对应文件夹。\n"
            "这里的阶段产出仍可能在后续复核中更新，不代表整份实验已验收。\n"
            "任务进度请在网页的“分析进度”中查看。\n",
            encoding="utf-8",
        )
    return root


def run_staging_roots(config: dict[str, Any]) -> list[Path]:
    """Use the configured visible folder and retain access to older runs."""

    name = str(
        config["storage"].get("staging_directory_name")
        or ".VisionCortex-Run-Staging"
    )
    if name not in {"Processing", ".VisionCortex-Run-Staging"}:
        raise ValueError("Unsupported staging directory name")
    archive_root = Path(config["storage"]["archive_root"])
    return [
        archive_root / item
        for item in dict.fromkeys(
            [name, ".VisionCortex-Run-Staging", "Processing"]
        )
    ]


def fixed_archive_staging_paths(
    config: dict[str, Any], archive_name: str, run_id: str
) -> tuple[Path, Path, Path]:
    archive_root = Path(config["storage"]["archive_root"])
    fixed_root = archive_root / safe_archive_name(archive_name)
    staging_root = run_staging_roots(config)[0] / archive_name / run_id
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
    professional_pdfs = staging_root / "Professional-PDFs"
    if not (
        any(
            professional_pdfs.glob(
                "VisionCortex-Professional-Evidence-Report-*.pdf"
            )
        )
        or any(professional_pdfs.glob("Lab-Daily-Report-*.pdf"))
    ):
        raise RuntimeError("Staged professional evidence PDF is missing; promotion refused")

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

    promoted_original_references = _publish_original_references(staging_root, fixed_root)

    receipt = {
        "schema_version": "visioncortex-fixed-archive-promotion/1",
        "fixed_root": str(fixed_root),
        "staging_root": str(staging_root),
        "history_root": str(history_root),
        "promoted_directories": list(DERIVED_ARCHIVE_DIRECTORIES),
        "promoted_original_references": promoted_original_references,
        "original_media_preserved": True,
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
    if (staging_root / "处理状态.txt").is_file():
        _atomic_write_text(
            staging_root / "处理状态.txt",
            f"已归档。完整产出位置：\n{fixed_root}\n",
        )
    return receipt


class IncrementalArchivePublisher:
    """Atomically publish completed files without exposing partial artifacts."""

    def __init__(self, local_root: Path, nas_root: Path):
        self.local_root = local_root.resolve()
        self.nas_root = nas_root
        self._verified: dict[str, tuple[int, int, int, int]] = {}
        self._verified_lock = threading.Lock()

    def publish_file(self, source: Path) -> Path:
        source = source.resolve()
        relative = source.relative_to(self.local_root)
        destination = self.nas_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_stat = source.stat()
        source_size = source_stat.st_size
        ledger_key = relative.as_posix()
        if destination.is_file():
            destination_stat = destination.stat()
            with self._verified_lock:
                verified = self._verified.get(ledger_key)
            if verified == (
                source_size,
                source_stat.st_mtime_ns,
                destination_stat.st_size,
                destination_stat.st_mtime_ns,
            ):
                return destination
        if destination.is_file() and destination.stat().st_size == source_size:
            if _sha256_file(destination) == _sha256_file(source):
                destination_stat = destination.stat()
                with self._verified_lock:
                    self._verified[ledger_key] = (
                        source_size,
                        source_stat.st_mtime_ns,
                        destination_stat.st_size,
                        destination_stat.st_mtime_ns,
                    )
                return destination
        temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:8]}")
        try:
            shutil.copy2(source, temporary)
            if temporary.stat().st_size != source_size or _sha256_file(temporary) != _sha256_file(source):
                raise IOError(f"NAS published file verification failed: {destination}")
            os.replace(temporary, destination)
            destination_stat = destination.stat()
            with self._verified_lock:
                self._verified[ledger_key] = (
                    source_size,
                    source_stat.st_mtime_ns,
                    destination_stat.st_size,
                    destination_stat.st_mtime_ns,
                )
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


def _parts(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def _parse_index_integer(value: str, field_name: str) -> int:
    """Parse integral CSV numbers without losing precision to binary floats."""

    text = str(value).strip()
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"non-integral {field_name}: {value!r}")
    return int(parsed)


def _resolve_nas_path(value: str, index_csv: Path) -> Path:
    """Resolve stale Z: index entries through the index CSV's active drive.

    Existence is validated once for the complete inventory later. Performing an
    SMB metadata round trip here caused every source to be checked repeatedly.
    """

    candidate = Path(value)
    pure = PureWindowsPath(value)
    if pure.drive.upper() == "Z:":
        relative = pure.relative_to(pure.anchor)
        index_pure = PureWindowsPath(str(index_csv))
        drive_root = (
            PureWindowsPath(index_pure.drive + "\\")
            if index_pure.drive
            else None
        )
        if drive_root is not None and index_pure.parent == drive_root:
            return Path(index_pure.drive + "\\" + str(relative))
        # Linux mounts the SMB share at the directory containing the canonical
        # index CSV. Preserve the Z:-relative components without treating the
        # backslashes as literal POSIX filename characters.
        return index_csv.parent.joinpath(*relative.parts)
    if pure.drive.casefold() == r"\\realityloop\video_database".casefold():
        # Historical rows used the canonical SMB UNC share while the Ubuntu
        # workstation mounts that same share at the index CSV directory.
        # This is a deterministic same-source translation, not a substitute
        # search, and still undergoes the normal stat/decode validation.
        relative = pure.relative_to(pure.anchor)
        return index_csv.parent.joinpath(*relative.parts)
    return candidate


def _zero_frame_segment_failure_evidence(
    video: Path,
    timestamps_csv: Path | None,
) -> dict[str, Any] | None:
    """Prove that an indexed-but-absent MP4 represents a zero-frame capture."""

    suffix = "rgb.mp4"
    if not video.name.endswith(suffix):
        return None
    meta = video.with_name(video.name[: -len(suffix)] + "meta.json")
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        payload.get("closed") is not True
        or str(payload.get("rgb_file") or "") != video.name
        or int(payload.get("rgb_frames", -1)) != 0
        or float(payload.get("rgb_actual_fps") or 0.0) != 0.0
        or float(payload.get("rgb_record_fps") or 0.0) != 0.0
    ):
        return None
    if timestamps_csv is not None and not timestamps_csv.is_file():
        return None
    return {
        "video_path": str(video),
        "timestamps_csv": str(timestamps_csv) if timestamps_csv is not None else None,
        "reason": "closed_recorder_segment_contains_zero_rgb_frames",
        "meta_path": str(meta),
        "meta_sha256": _sha256_file(meta),
        "rgb_frames": 0,
        "rgb_actual_fps": 0.0,
        "rgb_record_fps": 0.0,
        "source_copy_bytes": 0,
    }


def _clock_absolute_bounds_us(path: Path) -> tuple[int, int] | None:
    """Read the first/last recorder wall-clock timestamps without scanning CSV."""

    try:
        with path.open("rb") as handle:
            header_bytes = handle.readline()
            first_bytes = handle.readline()
            while first_bytes and not first_bytes.strip():
                first_bytes = handle.readline()
            if not header_bytes or not first_bytes:
                return None
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            block_size = min(65_536, end)
            handle.seek(end - block_size)
            tail = handle.read(block_size)
            lines = [line for line in tail.splitlines() if line.strip()]
            if not lines:
                return None
            last_bytes = lines[-1]
        header = next(
            csv.reader([header_bytes.decode("utf-8-sig").rstrip("\r\n")])
        )
        first = next(csv.reader([first_bytes.decode("utf-8").rstrip("\r\n")]))
        last = next(csv.reader([last_bytes.decode("utf-8").rstrip("\r\n")]))
        indexes = {name.strip(): index for index, name in enumerate(header)}
        clock_name = next(
            (
                name
                for name in (
                    "local_time_us",
                    "packet_system_timestamp_us",
                    "rgb_system_timestamp_us",
                    "frame_system_timestamp_us",
                )
                if name in indexes
                and indexes[name] < len(first)
                and indexes[name] < len(last)
                and first[indexes[name]].strip()
                and last[indexes[name]].strip()
            ),
            None,
        )
        if clock_name is None:
            return None
        start = int(Decimal(first[indexes[clock_name]].strip()))
        finish = int(Decimal(last[indexes[clock_name]].strip()))
        return (min(start, finish), max(start, finish))
    except (
        OSError,
        UnicodeError,
        csv.Error,
        InvalidOperation,
        ValueError,
        IndexError,
    ):
        return None


def _index_recording_window_us(row: dict[str, str]) -> tuple[int, int] | None:
    try:
        start = datetime.fromisoformat(
            str(row.get("recording_start_time") or "").replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(
            str(row.get("recording_end_time") or "").replace("Z", "+00:00")
        )
    except ValueError:
        return None
    if start.tzinfo is None or end.tzinfo is None or end < start:
        return None
    return (round(start.timestamp() * 1_000_000), round(end.timestamp() * 1_000_000))


def read_index_experiment(index_csv: Path, experiment_id: str) -> list[dict[str, str]]:
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("experiment_id") == experiment_id]
    if not rows:
        raise KeyError(f"experiment_id not found in {index_csv}: {experiment_id}")
    return rows


def describe_index_experiment(
    index_csv: Path,
    experiment_id: str,
    rows: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    rows = list(rows) if rows is not None else read_index_experiment(index_csv, experiment_id)
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
    starts = [
        _parse_index_integer(row["recording_start_us"], "recording_start_us")
        for row in rows
        if row.get("recording_start_us")
    ]
    ends = [
        _parse_index_integer(row["recording_end_us"], "recording_end_us")
        for row in rows
        if row.get("recording_end_us")
    ]
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
    migration_receipt_values = storage.get("source_path_migration_receipts") or []
    if isinstance(migration_receipt_values, (str, Path)):
        migration_receipt_values = [migration_receipt_values]
    migration_resolver = SourcePathResolver(migration_receipt_values)
    rows = read_index_experiment(index_csv, experiment_id)
    description = describe_index_experiment(index_csv, experiment_id, rows)
    usable_rows = [row for row in rows if _parts(row.get("rgb_file"))]
    omitted_rows = [row for row in rows if not _parts(row.get("rgb_file"))]
    unsafe_omissions = [
        str(row.get("camera_key") or "")
        for row in omitted_rows
        if not str(row.get("sync_error") or "").strip()
    ]
    if unsafe_omissions:
        raise ValueError(
            "Indexed cameras have no video and no explicit source failure: "
            f"{unsafe_omissions}"
        )
    active_archive = storage.get("active_archive_path")
    if active_archive and storage.get("manifest_storage", "nas") == "nas":
        manifest_root = Path(active_archive) / "JSON-Config-Files" / "Input-Manifests"
    else:
        manifest_root = Path(storage["local_runtime_root"]) / "input-manifests" / experiment_id
    manifest_root.mkdir(parents=True, exist_ok=True)
    registry = load_device_registry(storage.get("device_registry_path"))
    role_receipts = [
        resolve_view_role(
            registry,
            experiment_id,
            str(row.get("camera_key") or ""),
            row.get("camera_view"),
        )
        for row in rows
    ]
    receipt_by_camera = {
        str(receipt["camera_key"]): receipt for receipt in role_receipts
    }
    usable_camera_keys = {
        str(row.get("camera_key") or "") for row in usable_rows
    }
    usable_role_receipts = [
        receipt
        for receipt in role_receipts
        if str(receipt.get("camera_key") or "") in usable_camera_keys
    ]
    role_receipt_payload = {
        "schema_version": "visioncortex-view-role-resolution-ledger/1",
        "experiment_id": experiment_id,
        "index_csv": str(index_csv),
        "registry": registry.get("provenance"),
        "status": "blocked"
        if any(receipt.get("blocking_reasons") for receipt in usable_role_receipts)
        else "degraded_resolved"
        if omitted_rows
        else "resolved",
        "resolved_first_person_views": sum(
            receipt.get("resolved_role") == ViewRole.FIRST_PERSON.value
            for receipt in usable_role_receipts
        ),
        "resolved_third_person_views": sum(
            receipt.get("resolved_role") == ViewRole.THIRD_PERSON.value
            for receipt in usable_role_receipts
        ),
        "approved_override_count": sum(
            receipt.get("status") == "approved_override" for receipt in role_receipts
        ),
        "views": [
            {
                **receipt,
                "source_included": str(receipt.get("camera_key") or "")
                in usable_camera_keys,
                "source_omission_reason": next(
                    (
                        str(row.get("sync_error") or "")
                        for row in omitted_rows
                        if str(row.get("camera_key") or "")
                        == str(receipt.get("camera_key") or "")
                    ),
                    None,
                ),
            }
            for receipt in role_receipts
        ],
    }
    role_receipt_path = (
        Path(active_archive) / "JSON-Config-Files" / "view_role_resolution.json"
        if active_archive
        else manifest_root / "view_role_resolution.json"
    )
    _atomic_write_text(
        role_receipt_path,
        json.dumps(role_receipt_payload, ensure_ascii=False, indent=2),
    )
    blocking_roles = [
        {
            "camera_key": receipt.get("camera_key"),
            "blocking_reasons": receipt.get("blocking_reasons"),
        }
        for receipt in usable_role_receipts
        if receipt.get("blocking_reasons") or not receipt.get("resolved_role")
    ]
    if blocking_roles:
        raise ValueError(f"View role resolution blocked: {blocking_roles}")

    omitted_out_of_window_segments: list[dict[str, Any]] = []

    def prepare(row: dict[str, str]) -> ViewInput:
        camera_key = str(row["camera_key"])
        progress(f"Registering NAS segments without copying: {camera_key}")
        videos = [
            migration_resolver.resolve(_resolve_nas_path(item, index_csv))
            for item in _parts(row.get("rgb_file"))
        ]
        clocks = [
            migration_resolver.resolve(_resolve_nas_path(item, index_csv))
            for item in _parts(row.get("frames_file"))
        ]
        if clocks and len(clocks) != len(videos):
            raise ValueError(
                f"NAS segment/clock count mismatch for {camera_key}: "
                f"{len(videos)} videos, {len(clocks)} clock CSVs"
            )
        recording_window = _index_recording_window_us(row)
        if len(videos) > 1 and clocks and recording_window is not None:
            clock_bounds = [_clock_absolute_bounds_us(clock) for clock in clocks]
            # Filter only with complete recorder-clock proof. If even one
            # sidecar is unreadable, retain the source-index declaration and
            # let the normal fail-closed media checks decide.
            if all(bounds is not None for bounds in clock_bounds):
                tolerance_us = round(
                    float(
                        config["performance"].get(
                            "index_segment_window_tolerance_seconds", 2.0
                        )
                    )
                    * 1_000_000
                )
                window_start_us, window_end_us = recording_window
                keep = [
                    bool(
                        bounds
                        and bounds[1] >= window_start_us - tolerance_us
                        and bounds[0] <= window_end_us + tolerance_us
                    )
                    for bounds in clock_bounds
                ]
                if not any(keep):
                    raise ValueError(
                        "No indexed segment overlaps the declared recording "
                        f"window for {camera_key}"
                    )
                if not all(keep):
                    retained_videos: list[Path] = []
                    retained_clocks: list[Path] = []
                    for index, (video, clock, bounds, retained) in enumerate(
                        zip(videos, clocks, clock_bounds, keep, strict=True)
                    ):
                        if retained:
                            retained_videos.append(video)
                            retained_clocks.append(clock)
                            continue
                        assert bounds is not None
                        omitted_out_of_window_segments.append(
                            {
                                "view_id": camera_key,
                                "segment_index": index,
                                "video_path": str(video),
                                "timestamps_csv": str(clock),
                                "clock_start_us": bounds[0],
                                "clock_end_us": bounds[1],
                                "recording_window_start_us": window_start_us,
                                "recording_window_end_us": window_end_us,
                                "tolerance_us": tolerance_us,
                                "reason": (
                                    "recorder_clock_does_not_overlap_declared_"
                                    "experiment_window"
                                ),
                                "source_copy_bytes": 0,
                            }
                        )
                    videos = retained_videos
                    clocks = retained_clocks
        if storage.get("require_nas_source_paths"):
            expected_drive = index_csv.drive.upper()
            off_nas = [str(item) for item in videos + clocks if item.drive.upper() != expected_drive]
            if off_nas:
                raise ValueError(f"Source path escaped NAS drive {expected_drive}: {off_nas[:4]}")
        role = ViewRole(receipt_by_camera[camera_key]["resolved_role"])
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

    views = [prepare(row) for row in usable_rows]
    roles = {view.role for view in views}
    if not {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}.issubset(roles):
        raise ValueError(
            "Usable indexed sources do not contain a first/third-person pair"
        )
    source_paths = [
        path
        for view in views
        for segment in view.segments
        for path in (
            [segment.video]
            + ([segment.timestamps_csv] if segment.timestamps_csv is not None else [])
        )
    ]
    progress(f"Validating {len(source_paths)} NAS source files concurrently")
    source_snapshots, source_validation = snapshot_source_paths(
        source_paths,
        workers=int(config["performance"].get("source_stat_workers", 24)),
        max_age_seconds=float(
            config["performance"].get("source_stat_cache_ttl_seconds", 120.0)
        ),
    )
    missing = [
        str(path)
        for path, snapshot in source_snapshots.items()
        if not snapshot["is_file"]
    ]
    omitted_zero_frame_segments: list[dict[str, Any]] = []
    if missing:
        filtered_views: list[ViewInput] = []
        for view in views:
            retained_segments: list[VideoSegmentInput] = []
            for segment in view.segments:
                video_snapshot = source_snapshots.get(segment.video) or {}
                if video_snapshot.get("is_file"):
                    retained_segments.append(segment)
                    continue
                evidence = _zero_frame_segment_failure_evidence(
                    segment.video, segment.timestamps_csv
                )
                if evidence is None:
                    retained_segments.append(segment)
                    continue
                omitted_zero_frame_segments.append(
                    {"view_id": view.view_id, **evidence}
                )
            if retained_segments:
                filtered_views.append(
                    view.model_copy(update={"segments": retained_segments})
                )
        views = filtered_views
        source_paths = [
            path
            for view in views
            for segment in view.segments
            for path in (
                [segment.video]
                + (
                    [segment.timestamps_csv]
                    if segment.timestamps_csv is not None
                    else []
                )
            )
        ]
        effective_snapshots, effective_validation = snapshot_source_paths(
            source_paths,
            workers=int(config["performance"].get("source_stat_workers", 24)),
            max_age_seconds=float(
                config["performance"].get(
                    "source_stat_cache_ttl_seconds", 120.0
                )
            ),
        )
        source_validation = {
            **effective_validation,
            "initial_validation": source_validation,
            "omitted_zero_frame_segment_count": len(
                omitted_zero_frame_segments
            ),
        }
        source_snapshots = effective_snapshots
        missing = [
            str(path)
            for path, snapshot in source_snapshots.items()
            if not snapshot["is_file"]
        ]
    if missing:
        raise FileNotFoundError(f"NAS source missing: {missing[:4]}")
    if omitted_zero_frame_segments:
        remaining_roles = {view.role for view in views}
        if not {ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON}.issubset(
            remaining_roles
        ):
            raise ValueError(
                "Zero-frame segment omission removed the required cross-view pair"
            )
    segment_counts = {view.view_id: len(view.segments) for view in views}
    if config["performance"].get("synchronized_segment_waves") and len(set(segment_counts.values())) != 1:
        raise ValueError(f"Synchronized segment waves require equal segment counts: {segment_counts}")
    manifest = RunManifest(experiment_id=experiment_id, views=views)
    manifest_path = manifest_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    migration_resolution_path = manifest_root / "source_path_resolution.json"
    migration_resolution = migration_resolver.resolution_receipt(
        subject=f"nas-index:{experiment_id}"
    )
    _atomic_write_text(
        migration_resolution_path,
        json.dumps(migration_resolution, ensure_ascii=False, indent=2),
    )
    ingest = {
        **description,
        "indexed_camera_count": len(rows),
        "camera_count": len(views),
        "omitted_unavailable_cameras": [
            {
                "camera_key": row.get("camera_key"),
                "sync_error": row.get("sync_error"),
                "source_copy_bytes": 0,
            }
            for row in omitted_rows
        ],
        "omitted_zero_frame_segments": omitted_zero_frame_segments,
        "omitted_out_of_window_segments": omitted_out_of_window_segments,
        "input_mode": "nas_segmented_virtual_timeline",
        "copied_source_bytes": 0,
        "continuous_source_copies_created": 0,
        "manifest_root": str(manifest_root),
        "manifest": str(manifest_path),
        "segment_counts": segment_counts,
        "source_validation": source_validation,
        "source_path_resolution": {
            "status": migration_resolution["status"],
            "resolved_path_count": migration_resolution["resolved_path_count"],
            "receipt": str(migration_resolution_path),
            "records_digest_sha256": migration_resolution[
                "records_digest_sha256"
            ],
        },
        "view_role_resolution": {
            "status": role_receipt_payload["status"],
            "receipt": str(role_receipt_path),
            "approved_override_count": role_receipt_payload[
                "approved_override_count"
            ],
        },
        "role_overrides": {
            str(receipt["camera_key"]): str(receipt["resolved_role"])
            for receipt in role_receipts
            if receipt.get("status") == "approved_override"
        },
    }
    _atomic_write_text(
        manifest_root / "nas_ingest.json",
        json.dumps(ingest, ensure_ascii=False, indent=2),
    )
    if active_archive:
        original_root = Path(active_archive) / "Original-Experiment-Videos"
    else:
        original_root = manifest_root / "Original-Video-References"
    original_root.mkdir(parents=True, exist_ok=True)
    view_records: list[dict[str, Any]] = []
    playlist_paths: list[Path] = []
    total_video_bytes = 0
    total_clock_bytes = 0
    for view in views:
        playlist_path = original_root / f"{safe_archive_name(view.view_id)}.ffconcat"
        playlist_lines = ["ffconcat version 1.0"]
        segment_records: list[dict[str, Any]] = []
        for ordinal, segment in enumerate(view.segments, 1):
            video_snapshot = source_snapshots[segment.video]
            video_bytes = int(video_snapshot.get("size_bytes") or 0)
            total_video_bytes += video_bytes
            clock_snapshot = (
                source_snapshots.get(segment.timestamps_csv)
                if segment.timestamps_csv is not None
                else None
            )
            clock_bytes = int((clock_snapshot or {}).get("size_bytes") or 0)
            total_clock_bytes += clock_bytes
            playlist_value = str(segment.video).replace("\\", "/").replace("'", "'\\''")
            playlist_lines.append(f"file '{playlist_value}'")
            segment_records.append(
                {
                    "ordinal": ordinal,
                    "video_path": str(segment.video),
                    "video_size_bytes": video_bytes,
                    "video_mtime_ns": video_snapshot.get("mtime_ns"),
                    "timestamps_csv": str(segment.timestamps_csv)
                    if segment.timestamps_csv is not None
                    else None,
                    "timestamps_size_bytes": clock_bytes,
                    "timestamps_mtime_ns": (clock_snapshot or {}).get("mtime_ns"),
                }
            )
        _atomic_write_text(playlist_path, "\n".join(playlist_lines) + "\n")
        playlist_paths.append(playlist_path)
        view_records.append(
            {
                "view_id": view.view_id,
                "role": view.role.value,
                "segment_count": len(segment_records),
                "playlist": playlist_path.name,
                "segments": segment_records,
            }
        )
    original_index_path = original_root / "Original-Video-Index.json"
    original_index = {
        "schema_version": "visioncortex-original-video-index/1",
        "experiment_id": experiment_id,
        "retention_mode": "nas_zero_copy_segment_references",
        "source_of_truth": "NAS paths recorded in this index and the experiment record index",
        "experiment_record_index": str(index_csv),
        "source_copy_bytes": 0,
        "continuous_video_copies_created": 0,
        "total_video_bytes": total_video_bytes,
        "total_clock_bytes": total_clock_bytes,
        "views": view_records,
    }
    _atomic_write_text(
        original_index_path,
        json.dumps(original_index, ensure_ascii=False, indent=2),
    )
    fingerprint_workers = max(
        1,
        min(
            int(config["performance"].get("source_stat_workers", 24)),
            len(source_paths),
        ),
    )
    with ThreadPoolExecutor(
        max_workers=fingerprint_workers,
        thread_name_prefix="input-seal",
    ) as executor:
        fingerprint_futures = {
            executor.submit(_bounded_source_fingerprint, path): path
            for path in source_paths
        }
        source_fingerprints = {
            path: future.result()
            for future, path in fingerprint_futures.items()
        }
    seal_sources: list[dict[str, Any]] = []
    for view in views:
        for ordinal, segment in enumerate(view.segments, 1):
            video_snapshot = source_snapshots[segment.video]
            seal_sources.append(
                {
                    "kind": "video",
                    "view_id": view.view_id,
                    "segment_ordinal": ordinal,
                    "path": str(segment.video),
                    "size_bytes": int(video_snapshot.get("size_bytes") or 0),
                    "mtime_ns": video_snapshot.get("mtime_ns"),
                    **source_fingerprints[segment.video],
                }
            )
            if segment.timestamps_csv is not None:
                clock_snapshot = source_snapshots[segment.timestamps_csv]
                seal_sources.append(
                    {
                        "kind": "timestamp_csv",
                        "view_id": view.view_id,
                        "segment_ordinal": ordinal,
                        "path": str(segment.timestamps_csv),
                        "size_bytes": int(clock_snapshot.get("size_bytes") or 0),
                        "mtime_ns": clock_snapshot.get("mtime_ns"),
                        **source_fingerprints[segment.timestamps_csv],
                    }
                )
    input_seal = build_input_seal(
        manifest,
        source_mode="nas_segmented_virtual_timeline",
        sources=seal_sources,
        role_resolution=role_receipt_payload,
        copied_source_bytes=0,
        preflight={
            "schema_version": "visioncortex-prequeue-input-preflight/1",
            "status": "metadata_validated_media_probe_pending",
            "source_validation": source_validation,
            "clock_window_omission_count": len(omitted_out_of_window_segments),
        },
    )
    input_seal_path = write_input_seal(
        manifest_root / "input_seal.json",
        input_seal,
    )
    readme_path = original_root / "README.txt"
    _atomic_write_text(
        readme_path,
        "VisionCortex 原视频零复制留存\n"
        "\n"
        "原始 15 分钟 MP4 分片及对应时钟 CSV 继续保存在 NAS 原路径，"
        "本目录不复制、不拼接原视频。\n"
        "Original-Video-Index.json 记录六路来源、角色、分片顺序、字节数与时钟文件。\n"
        "每个 .ffconcat 文件按原顺序引用一路视频分片，可供 FFmpeg 连续读取。\n",
    )
    ingest["original_retention"] = {
        "mode": original_index["retention_mode"],
        "root": str(original_root),
        "index": str(original_index_path),
        "playlists": [str(path) for path in playlist_paths],
        "readme": str(readme_path),
        "source_copy_bytes": 0,
        "input_seal": str(input_seal_path),
        "input_seal_sha256": input_seal["seal_sha256"],
    }
    _atomic_write_text(
        manifest_root / "nas_ingest.json",
        json.dumps(ingest, ensure_ascii=False, indent=2),
    )
    if active_archive:
        receipt_path = (
            Path(active_archive)
            / "JSON-Config-Files"
            / "Stage-Receipts"
            / "original_ingest.json"
        )
        _atomic_write_text(
            receipt_path,
            json.dumps(
                {
                    "schema_version": "visioncortex-stage-receipt/1",
                    "stage": "original_ingest",
                    "status": "completed",
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "archive_root": str(Path(active_archive)),
                    "retention_mode": original_index["retention_mode"],
                    "source_copy_bytes": 0,
                    "artifacts": [
                        str(original_index_path),
                        *(str(path) for path in playlist_paths),
                        str(readme_path),
                        str(manifest_path),
                        str(manifest_root / "nas_ingest.json"),
                        str(role_receipt_path),
                        str(input_seal_path),
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
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

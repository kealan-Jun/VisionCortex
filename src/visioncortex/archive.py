from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from openpyxl import Workbook

from .action_state_machine import build_event_state_receipt
from .alignment import iter_aligned_rows
from .action_semantics import (
    DEVICE_CLASSES,
    action_participant_visibility,
    record_semantic_review,
    semantic_action_proof_contradictions,
)
from .detection import iter_frame_evidence, nearest_frame_evidence_many
from .indexing import (
    build_archive_index,
    stable_artifact_uid,
    stable_event_uid,
    stable_evidence_uid,
)
from .key_material_verification import (
    SelectiveVerificationBudget,
    plan_selective_key_material_verification,
)
from .identity import PRODUCT_NAME
from .grouping import (
    event_stable_identities,
    event_view_actor_identities,
    event_view_stable_identities,
)
from .mllm import (
    ArkStepAnalyzer,
    EVENT_SYSTEM_PROMPT,
    FINAL_GROUP_SYSTEM_PROMPT,
    GROUP_SYSTEM_PROMPT,
    normalize_uncalibrated_hand_identity,
)
from .material_naming import (
    ACTION_CATEGORY_FOLDERS,
    key_material_action_folder,
    key_material_semantic_name as _key_material_semantic_name,
)
from .liquid_semantic import (
    analyze_liquid_semantics,
    release_liquid_semantic_model_cache,
)
from .pathing import archive_relative_posix
from .participant_visual_review import ParticipantVisualReviewer
from .schemas import (
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    ExperimentGroup,
    ExperimentSegment,
    FrameEvidence,
    PhysicalChange,
    RunManifest,
    RunSummary,
    VideoInfo,
    ViewInput,
    ViewRole,
    event_is_formal,
    set_event_admission,
)
from .temporal_segmentation import (
    audit_participant_continuity,
    release_temporal_segmentation_model_cache,
)
from .video_io import (
    ViewFrameReader,
    create_grid_video,
    extract_view_clip,
    probe_video,
    select_video_encoder,
    view_source_files,
    write_annotated_frame,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    temporary.replace(path)


class SemanticAnalysisUnavailable(RuntimeError):
    """A resumable semantic stage failed before it could adjudicate evidence."""


def _raise_for_incomplete_semantic_results(
    layout: "ArchiveLayout",
    *,
    stage: str,
    results: Sequence[tuple[str, dict[str, Any]]],
    fail_run: bool = True,
) -> list[str]:
    incomplete = [
        {
            "subject_id": subject_id,
            "status": str(result.get("status") or "unknown"),
            "error": result.get("error"),
            "attempts": result.get("attempts"),
        }
        for subject_id, result in results
        if result.get("status") != "completed"
    ]
    if not incomplete:
        return []
    path = layout.json_config / f"{stage}_semantic_failures.json"
    write_json(
        path,
        {
            "schema_version": "visioncortex-semantic-stage-failure/1",
            "stage": stage,
            "status": "resumable_failure" if fail_run else "partial_evidence",
            "evidence_classification": "NOT_PROVEN" if fail_run else "PARTIAL_EVIDENCE",
            "analysis_continuation_allowed": not fail_run,
            "formal_evidence_mutated": False,
            "completed_results_reusable": True,
            "incomplete": incomplete,
        },
    )
    if fail_run:
        raise SemanticAnalysisUnavailable(
            f"{stage} semantic analysis incomplete for {len(incomplete)} subject(s); "
            f"resume from {path}"
        )
    return [str(item["subject_id"]) for item in incomplete]


def _semantic_fingerprint(
    kind: str,
    config: dict[str, Any],
    prompt: str,
    evidence: dict[str, Any],
    images: Sequence[tuple[str, Path]],
) -> str:
    """Fingerprint exactly the evidence that can change an MLLM answer.

    Run ids, cache identities and archive folder names are deliberately absent.
    A failed run can therefore resume on the same machine without paying for an
    identical request, while any boundary, CV event, prompt, model or image
    change forces a new call.
    """

    digest = hashlib.sha256()
    header = {
        "schema": "visioncortex-semantic-cache/1",
        "kind": kind,
        "model": str(config["mllm"]["model"]),
        "base_url": str(config["mllm"].get("base_url") or ""),
        "response_language": str(config["mllm"].get("response_language") or ""),
        # Namespace separates cold benchmark campaigns without changing the
        # semantic evidence contract. Cold and hot executions of one campaign
        # deliberately use the same value.
        "cache_namespace": config.get("project", {}).get("cache_namespace"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "evidence": evidence,
        "image_order": [label for label, _ in images],
    }
    digest.update(
        json.dumps(header, ensure_ascii=False, sort_keys=True, default=_json_default).encode(
            "utf-8"
        )
    )
    for label, image_path in images:
        digest.update(b"\0label\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0image\0")
        digest.update(image_path.read_bytes())
    return digest.hexdigest()


def _semantic_cache_path(
    config: dict[str, Any], kind: str, fingerprint: str
) -> Path:
    return (
        Path(str(config["storage"]["local_cache_root"]))
        / "semantic-results-v1"
        / _safe_slug(str(config["mllm"]["model"]))
        / kind
        / f"{fingerprint}.json"
    )


def _semantic_cache_reads_enabled(config: dict[str, Any]) -> bool:
    """Return whether this execution may reuse a completed Ark response.

    A cold audit still writes its completed answer so a paired hot replay can
    prove reuse. ``semantic_cache_mode`` may independently force fresh Ark
    calls while verified CV and derived-media artifacts are reused after an
    infrastructure interruption. It only suppresses reads; no entry is deleted.
    """

    project = config.get("project", {})
    mode = project.get("semantic_cache_mode", project.get("cache_mode", "reuse"))
    return str(mode or "reuse") != "cold"


def _execution_cache_reads_enabled(config: dict[str, Any]) -> bool:
    """Return whether immutable CV/media artifacts may be reused."""

    return str(config.get("project", {}).get("cache_mode") or "reuse") != "cold"


def _read_semantic_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("status") != "completed"
        or cached.get("input_fingerprint") != fingerprint
        or cached.get("semantic_cache_schema") != "visioncortex-semantic-cache/1"
    ):
        return None
    cached["cache_reused"] = True
    return cached


def _write_semantic_cache(
    path: Path, fingerprint: str, result: dict[str, Any]
) -> dict[str, Any]:
    persisted = dict(result)
    persisted["input_fingerprint"] = fingerprint
    persisted["semantic_cache_schema"] = "visioncortex-semantic-cache/1"
    persisted["cache_reused"] = False
    write_json(path, persisted)
    return persisted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _derived_media_cache_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("performance", {}).get("derived_media_cache_enabled", False))


def _derived_media_cache_reads_enabled(config: dict[str, Any]) -> bool:
    return _derived_media_cache_enabled(config) and _execution_cache_reads_enabled(
        config
    )


def _derived_media_fingerprint(
    kind: str,
    config: dict[str, Any],
    request: dict[str, Any],
    inputs: Sequence[Path],
    *,
    content_address_inputs: bool = False,
) -> str:
    input_receipts = []
    for source in inputs:
        stat = source.stat()
        receipt: dict[str, Any] = {
            "size_bytes": int(stat.st_size),
        }
        if content_address_inputs:
            receipt["sha256"] = _sha256_file(source)
        else:
            receipt.update(
                {
                    "path": os.path.abspath(str(source)),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        input_receipts.append(receipt)
    preferred_encoder = str(
        config.get("performance", {}).get("ffmpeg_video_encoder") or "h264_nvenc"
    )
    payload = {
        "schema": "visioncortex-derived-media-cache/1",
        "kind": kind,
        "cache_namespace": config.get("project", {}).get("cache_namespace"),
        "selected_encoder": select_video_encoder(preferred_encoder),
        "request": request,
        "inputs": input_receipts,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode(
            "utf-8"
        )
    ).hexdigest()


def _link_or_copy_immutable(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.cache-{uuid.uuid4().hex[:8]}"
    )
    try:
        try:
            os.link(source, temporary)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            method = "verified_copy"
        os.replace(temporary, destination)
    except Exception:
        raise
    finally:
        # POSIX permits rename/replace to be a no-op when both paths already
        # name the same inode.  That happens during idempotent cache reuse and
        # otherwise leaves our temporary hard-link visible on NAS shares.
        temporary.unlink(missing_ok=True)
    return method


_DERIVED_MEDIA_POPULATED_THIS_PROCESS: set[str] = set()


def _materialize_derived_media(
    destination: Path,
    kind: str,
    request: dict[str, Any],
    inputs: Sequence[Path],
    config: dict[str, Any],
    generator: Callable[[], None],
    *,
    content_address_inputs: bool = False,
) -> dict[str, Any]:
    """Reuse one immutable derived video only when its complete identity matches.

    Original source video is never copied into this cache.  The request binds
    source identity, exact time bounds, selected encoder, and transformation
    parameters; the sidecar additionally binds the cached output bytes.  Any
    missing or inconsistent receipt fails closed instead of serving uncertain
    media.
    """

    if not _derived_media_cache_enabled(config):
        generator()
        return {
            "cache_enabled": False,
            "cache_reused": False,
            "cache_materialization": "generated",
        }
    fingerprint = _derived_media_fingerprint(
        kind,
        config,
        request,
        inputs,
        content_address_inputs=content_address_inputs,
    )
    suffix = destination.suffix.lower() or ".bin"
    cache_root = (
        Path(str(config["storage"]["local_cache_root"]))
        / "derived-media-v1"
        / _safe_slug(kind)
        / fingerprint[:2]
    )
    cached_media = cache_root / f"{fingerprint}{suffix}"
    cached_receipt = cache_root / f"{fingerprint}.json"
    same_process_entry = fingerprint in _DERIVED_MEDIA_POPULATED_THIS_PROCESS
    if (
        _derived_media_cache_reads_enabled(config) or same_process_entry
    ) and (
        cached_media.exists() or cached_receipt.exists()
    ):
        if not cached_media.is_file() or not cached_receipt.is_file():
            raise RuntimeError(
                f"Derived media cache entry is incomplete: {fingerprint}"
            )
        receipt = json.loads(cached_receipt.read_text(encoding="utf-8-sig"))
        expected_sha = str(receipt.get("output_sha256") or "")
        if (
            receipt.get("schema_version") != "visioncortex-derived-media-cache/1"
            or receipt.get("request_fingerprint") != fingerprint
            or int(receipt.get("output_bytes") or -1) != cached_media.stat().st_size
            or not expected_sha
            or _sha256_file(cached_media) != expected_sha
        ):
            raise RuntimeError(
                f"Derived media cache verification failed: {fingerprint}"
            )
        method = _link_or_copy_immutable(cached_media, destination)
        if _sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"Derived media cache materialization changed bytes: {fingerprint}"
            )
        return {
            "cache_enabled": True,
            "cache_reused": True,
            "cache_reuse_scope": (
                "same_process_idempotent"
                if same_process_entry
                else "persistent_verified"
            ),
            "cache_materialization": method,
            "cache_request_fingerprint": fingerprint,
            "cache_output_sha256": expected_sha,
        }

    generator()
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Derived media generator produced no file: {destination}")
    output_sha = _sha256_file(destination)
    cache_root.mkdir(parents=True, exist_ok=True)
    if cached_media.exists() or cached_receipt.exists():
        if not cached_media.is_file() or not cached_receipt.is_file():
            raise RuntimeError(
                f"Refusing incomplete derived media cache collision: {fingerprint}"
            )
        receipt = json.loads(cached_receipt.read_text(encoding="utf-8-sig"))
        cached_sha = str(receipt.get("output_sha256") or "")
        if (
            receipt.get("schema_version")
            != "visioncortex-derived-media-cache/1"
            or receipt.get("request_fingerprint") != fingerprint
            or int(receipt.get("output_bytes") or -1)
            != cached_media.stat().st_size
            or not cached_sha
            or _sha256_file(cached_media) != cached_sha
            or destination.stat().st_size != cached_media.stat().st_size
            or output_sha != cached_sha
        ):
            raise RuntimeError(
                "Refusing to overwrite a non-identical derived media cache "
                f"entry: {fingerprint}"
            )
        _DERIVED_MEDIA_POPULATED_THIS_PROCESS.add(fingerprint)
        return {
            "cache_enabled": True,
            "cache_reused": False,
            "cache_reuse_scope": "generated_verified_idempotent_collision",
            "cache_materialization": "generated",
            "cache_request_fingerprint": fingerprint,
            "cache_output_sha256": output_sha,
            "cache_idempotent_collision_verified": True,
        }
    cache_method = _link_or_copy_immutable(destination, cached_media)
    write_json(
        cached_receipt,
        {
            "schema_version": "visioncortex-derived-media-cache/1",
            "request_fingerprint": fingerprint,
            "kind": kind,
            "output_bytes": destination.stat().st_size,
            "output_sha256": output_sha,
            "cache_population": cache_method,
            "source_copy_bytes": 0,
        },
    )
    _DERIVED_MEDIA_POPULATED_THIS_PROCESS.add(fingerprint)
    return {
        "cache_enabled": True,
        "cache_reused": False,
        "cache_materialization": "generated",
        "cache_population": cache_method,
        "cache_request_fingerprint": fingerprint,
        "cache_output_sha256": output_sha,
    }


def _relative(path: Path, root: Path) -> str:
    return archive_relative_posix(path, root)


def _time_slug(ms: float) -> str:
    total = max(0.0, ms) / 1000.0
    hours = int(total // 3600)
    minutes = int(total % 3600 // 60)
    seconds = total % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_") or "unknown"


def _safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(" .-_")
    if cleaned:
        return cleaned[:120]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"Unnamed-Experiment-{digest}"


def _bounded_component(value: str, maximum_chars: int) -> str:
    cleaned = _safe_folder_name(value)
    maximum_chars = max(12, int(maximum_chars))
    if len(cleaned) <= maximum_chars:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
    prefix = cleaned[: max(1, maximum_chars - len(digest) - 1)].rstrip(" .-_")
    return f"{prefix}-{digest}"


def _component_budget(parent: Path, reserved_tail_chars: int, maximum_chars: int = 72) -> int:
    maximum_chars = max(12, int(maximum_chars))
    parent_text = str(parent)
    windows_style_path = bool(re.match(r"^[A-Za-z]:[\\/]", parent_text)) or parent_text.startswith(
        "\\\\"
    )
    if os.name != "nt" and not windows_style_path:
        return maximum_chars
    # 235 leaves headroom below classic Windows MAX_PATH for FFmpeg/OpenCV
    # temporary suffixes when LongPathsEnabled is disabled.
    return max(
        12,
        min(maximum_chars, 235 - len(str(parent)) - int(reserved_tail_chars) - 1),
    )


def _group_folder_name(
    layout: "ArchiveLayout",
    index: int,
    experiment_name: str,
    experiment_name_en: str,
) -> str:
    del experiment_name  # Chinese display name stays in JSON/report content only.
    desired = f"{index:03d}_{_safe_slug(experiment_name_en)}"
    budget = _group_folder_budget(layout)
    return _bounded_component(desired, budget)


def _group_folder_budget(layout: "ArchiveLayout") -> int:
    return min(
        _component_budget(layout.experiment_clips, 28),
        # Action category + event folder + separators + longest aligned sidecar.
        _component_budget(layout.key_frames, 104),
        _component_budget(layout.key_clips, 104),
    )


def _key_material_event_folder_name(
    layout: "ArchiveLayout",
    experiment_folder: str,
    event: EvidenceEvent,
) -> str:
    action_folder = key_material_action_folder(event.action_type)
    desired = _key_material_semantic_name(event)["file_stem"]
    budget = min(
        _component_budget(
            layout.key_frames / experiment_folder / action_folder,
            26,
            maximum_chars=40,
        ),
        _component_budget(
            layout.key_clips / experiment_folder / action_folder,
            26,
            maximum_chars=40,
        ),
    )
    return _bounded_component(desired, budget)


class ArchiveLayout:
    def __init__(self, root: Path):
        self.root = root
        self.experiment_clips = root / "Experiment-Clips"
        self.json_config = root / "JSON-Config-Files"
        self.key_materials = root / "Key-Materials"
        self.key_clips = self.key_materials / "Key-Clips"
        self.key_frames = self.key_materials / "Key-Frames"
        self.daily_reports = root / "Lab-Daily-Reports"
        self.original_videos = root / "Original-Experiment-Videos"
        self.professional_pdfs = root / "Professional-PDFs"
        self.work = root / ".work"

    def create(self) -> None:
        for path in (
            self.experiment_clips,
            self.json_config,
            self.key_clips,
            self.key_frames,
            self.daily_reports,
            self.original_videos,
            self.professional_pdfs,
            self.work,
        ):
            path.mkdir(parents=True, exist_ok=True)


def write_aligned_csv(
    path: Path,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    output_fps: float,
) -> None:
    rows = iter_aligned_rows(views, infos, transforms, output_fps)
    first = next(iter(rows), None)
    if first is None:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def materialize_experiment_clips(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    segments: list[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
    publisher: Any | None = None,
) -> None:
    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    encoder = config["performance"]["ffmpeg_video_encoder"]
    workers = max(1, min(2, int(config["performance"].get("materialization_workers", 2))))
    runtime_records: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        folder_name = _group_folder_name(
            layout,
            index,
            group.experiment_name,
            group.experiment_name_en,
        )
        group.archive_folder = folder_name
        group_root = layout.experiment_clips / folder_name
        videos_dir = group_root
        json_dir = group_root
        group_root.mkdir(parents=True, exist_ok=True)
        extraction_jobs = []

        def extract_role(role_label: str, view_id: str) -> dict[str, Any]:
            started = time.perf_counter()
            view = by_view[view_id]
            transform = transforms[view_id]
            local_start = max(0.0, transform.to_local(group.global_start_ms))
            local_end = min(
                infos[view_id].duration_ms,
                transform.to_local(group.global_end_ms),
            )
            if local_end <= local_start:
                raise ValueError(
                    f"{group.group_id}/{view_id} global boundary is outside the source video"
                )
            destination = videos_dir / f"{role_label}.mp4"
            cache = _materialize_derived_media(
                destination,
                "experiment-view-clip",
                {
                    "group_id": group.group_id,
                    "role_label": role_label,
                    "view_id": view_id,
                    "local_start_ms": local_start,
                    "local_end_ms": local_end,
                    "global_start_ms": group.global_start_ms,
                    "global_end_ms": group.global_end_ms,
                },
                view_source_files(view),
                config,
                lambda: extract_view_clip(
                    view,
                    infos[view_id],
                    destination,
                    local_start,
                    local_end - local_start,
                    encoder,
                ),
            )
            return {
                "group_id": group.group_id,
                "role_label": role_label,
                "view_id": view_id,
                "duration_seconds": round(time.perf_counter() - started, 6),
                "output_bytes": destination.stat().st_size,
                **cache,
            }

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="experiment-media",
        ) as executor:
            for role_label, view_id in (
                ("First-Person", group.first_person_view),
                ("Third-Person", group.third_person_view),
            ):
                extraction_jobs.append(executor.submit(extract_role, role_label, view_id))
            runtime_records.extend(job.result() for job in extraction_jobs)
        role_paths: dict[str, tuple[str, Path]] = {}
        for role_label, view_id in (
            ("First-Person", group.first_person_view),
            ("Third-Person", group.third_person_view),
        ):
            view = by_view[view_id]
            transform = transforms[view_id]
            local_start = max(0.0, transform.to_local(group.global_start_ms))
            local_end = min(infos[view_id].duration_ms, transform.to_local(group.global_end_ms))
            if local_end <= local_start:
                raise ValueError(f"{group.group_id}/{view_id} 全局边界映射后不在视频范围内")
            base = role_label
            destination = videos_dir / f"{base}.mp4"
            # Both role clips were exported concurrently above. Metadata and
            # publication remain ordered and atomic for a stable archive.
            relative = _relative(destination, layout.root)
            group.videos[role_label.lower()] = relative
            role_paths[role_label] = (view_id, destination)
            metadata = {
                "schema_version": "2.0.0",
                "artifact_type": "experiment_view_video",
                "group": group.model_dump(mode="json", exclude={"video_json"}),
                "view_id": view_id,
                "view_role": view.role.value,
                "video_file": relative,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "local_start_ms": local_start,
                "local_end_ms": local_end,
                "alignment": transform.model_dump(mode="json"),
                "atomic_experiments": [
                    by_segment[item].model_dump(mode="json") for item in group.atomic_experiment_ids
                ],
            }
            json_path = json_dir / f"{base}.json"
            write_json(json_path, metadata)
            if publisher is not None:
                publisher.publish_file(destination)
                publisher.publish_file(json_path)
            group.video_json[role_label.lower()] = _relative(json_path, layout.root)

        aligned = videos_dir / "Aligned_First+Third.mp4"
        aligned_started = time.perf_counter()
        aligned_inputs = [
            (f"First-Person {role_paths['First-Person'][0]}", role_paths["First-Person"][1]),
            (f"Third-Person {role_paths['Third-Person'][0]}", role_paths["Third-Person"][1]),
        ]
        aligned_cache = _materialize_derived_media(
            aligned,
            "experiment-aligned-clip",
            {
                "group_id": group.group_id,
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "layout": "first_person_left,third_person_right",
                "grid_shape": "2x1@640x360_each",
            },
            [path for _, path in aligned_inputs],
            config,
            lambda: create_grid_video(aligned_inputs, aligned, encoder),
            content_address_inputs=True,
        )
        runtime_records.append(
            {
                "group_id": group.group_id,
                "role_label": "Aligned-First-Third",
                "view_id": "aligned_first_third",
                "duration_seconds": round(time.perf_counter() - aligned_started, 6),
                "output_bytes": aligned.stat().st_size,
                **aligned_cache,
            }
        )
        group.videos["aligned_first_third"] = _relative(aligned, layout.root)
        aligned_json = json_dir / "Aligned_First+Third.json"
        write_json(
            aligned_json,
            {
                "schema_version": "2.0.0",
                "artifact_type": "aligned_first_third_experiment_video",
                "group": group.model_dump(mode="json", exclude={"video_json"}),
                "video_file": _relative(aligned, layout.root),
                "layout": "first_person_left, third_person_right",
                "global_start_ms": group.global_start_ms,
                "global_end_ms": group.global_end_ms,
                "views": [group.first_person_view, group.third_person_view],
                "alignments": {
                    view_id: transforms[view_id].model_dump(mode="json")
                    for view_id in (group.first_person_view, group.third_person_view)
                },
                "atomic_experiments": [
                    by_segment[item].model_dump(mode="json") for item in group.atomic_experiment_ids
                ],
            },
        )
        group.video_json["aligned_first_third"] = _relative(aligned_json, layout.root)
        if publisher is not None:
            publisher.publish_file(aligned)
            publisher.publish_file(aligned_json)
        for segment_id in group.atomic_experiment_ids:
            segment = by_segment[segment_id]
            segment.clips = {
                group.first_person_view: group.videos["first-person"],
                group.third_person_view: group.videos["third-person"],
            }
            segment.aligned_multiview_clip = group.videos["aligned_first_third"]

    runtime_path = layout.json_config / "experiment_clip_materialization_runtime.json"
    write_json(
        runtime_path,
        {
            "schema_version": "visioncortex-materialization-runtime/1",
            "workers": workers,
            "records": runtime_records,
        },
    )
    if publisher is not None:
        publisher.publish_file(runtime_path)


def _storyboard_times(
    group: ExperimentGroup,
    events: Sequence[EvidenceEvent],
    limit: int,
    segments: Sequence[ExperimentSegment] = (),
) -> list[float]:
    """Cover boundaries, atomic segments and action classes before uniform fill."""

    if limit < 2:
        return [(group.global_start_ms + group.global_end_ms) / 2.0]
    duration = max(1.0, group.global_end_ms - group.global_start_ms)
    uniform = [
        group.global_start_ms + duration * index / (limit - 1)
        for index in range(limit)
    ]
    accepted = [
        event
        for event in events
        if event.accepted
        and group.global_start_ms <= event.key_global_ms <= group.global_end_ms
    ]
    selected = [uniform[0], uniform[-1]]

    def add(value: float) -> None:
        if len(selected) >= limit:
            return
        bounded = min(group.global_end_ms, max(group.global_start_ms, float(value)))
        if bounded not in selected:
            selected.append(bounded)

    for segment in sorted(segments, key=lambda item: item.global_start_ms):
        segment_events = [
            event
            for event in accepted
            if event.event_id in set(segment.event_ids)
        ]
        representative = (
            max(
                segment_events,
                key=lambda event: (event.confidence, -event.key_global_ms),
            ).key_global_ms
            if segment_events
            else (segment.global_start_ms + segment.global_end_ms) / 2.0
        )
        add(representative)
    for action_type in sorted({event.action_type.value for event in accepted}):
        add(
            max(
                (
                    event
                    for event in accepted
                    if event.action_type.value == action_type
                ),
                key=lambda event: (event.confidence, -event.key_global_ms),
            ).key_global_ms
        )

    candidates = sorted(set(uniform + [event.key_global_ms for event in accepted]))
    while len(selected) < limit and candidates:
        best = max(candidates, key=lambda value: min(abs(value - item) for item in selected))
        add(best)
        candidates.remove(best)
    return sorted(set(selected))[:limit]


def analyze_experiment_groups(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    config: dict[str, Any],
    *,
    final_adjudicated: bool = False,
) -> None:
    analyzer = ArkStepAnalyzer(config)
    cache_root = layout.work / "mllm-cache" / "experiment-groups"
    by_view = {view.view_id: view for view in views}
    by_segment = {segment.segment_id: segment for segment in segments}
    base_pairs = max(2, int(config["mllm"].get("storyboard_pairs_per_group", 4)))
    maximum_pairs = max(
        base_pairs,
        int(config["mllm"].get("storyboard_pairs_per_group_max", base_pairs)),
    )
    seconds_per_pair = max(
        1.0,
        float(config["mllm"].get("storyboard_seconds_per_pair", 300.0)),
    )

    def analyze(group: ExperimentGroup) -> tuple[ExperimentGroup, dict[str, Any]]:
        storyboard: list[tuple[str, Path]] = []
        storyboard_dir = layout.work / "group-storyboards" / group.group_id
        atomic = [by_segment[item] for item in group.atomic_experiment_ids]
        event_ids = {event_id for segment in atomic for event_id in segment.event_ids}
        group_events = [event for event in events if event.event_id in event_ids]
        duration_pairs = math.ceil(
            max(1.0, group.global_end_ms - group.global_start_ms)
            / (seconds_per_pair * 1000.0)
        ) + 1
        action_pairs = len(
            {event.action_type.value for event in group_events if event.accepted}
        ) + 2
        pair_limit = min(
            maximum_pairs,
            max(base_pairs, duration_pairs, len(atomic) + 2, action_pairs),
        )
        with ViewFrameReader(max_open=2) as frame_reader:
            for index, global_ms in enumerate(
                _storyboard_times(group, group_events, pair_limit, atomic), 1
            ):
                role_paths: dict[str, Path] = {}
                for role_label, view_id in (
                    ("first_person", group.first_person_view),
                    ("third_person", group.third_person_view),
                ):
                    local_ms = transforms[view_id].to_local(global_ms)
                    if not 0.0 <= local_ms <= infos[view_id].duration_ms:
                        continue
                    frame = frame_reader.read(by_view[view_id], infos[view_id], local_ms)
                    if frame is None:
                        continue
                    path = storyboard_dir / f"{index:02d}_{role_label}_{_safe_slug(view_id)}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        continue
                    role_paths[role_label] = path
                if {"first_person", "third_person"} <= set(role_paths):
                    aligned = storyboard_dir / f"{index:02d}_aligned_first_third.jpg"
                    _write_aligned_frame(
                        role_paths["first_person"],
                        role_paths["third_person"],
                        aligned,
                        (
                            f"First-Person {group.first_person_view}",
                            f"Third-Person {group.third_person_view}",
                        ),
                    )
                    storyboard.append(
                        (
                            f"t={global_ms:.3f}ms; aligned_first_third; "
                            f"first={group.first_person_view}; third={group.third_person_view}",
                            aligned,
                        )
                    )
        if len(storyboard) < 2:
            raise RuntimeError(
                f"{group.group_id} has fewer than two complete aligned storyboard pairs"
            )
        semantic_evidence = {
            "continuity_type": group.continuity_type,
            "continuity_reason": group.continuity_reason,
            "global_start_ms": group.global_start_ms,
            "global_end_ms": group.global_end_ms,
            "first_person_view": group.first_person_view,
            "third_person_view": group.third_person_view,
            "atomic_boundaries": [
                segment.model_dump(
                    mode="json",
                    exclude={"semantic_understanding", "clips", "aligned_multiview_clip"},
                )
                for segment in atomic
            ],
            "cv_events": [
                event.model_dump(
                    mode="json",
                    exclude={"model_understanding", "key_frames", "key_clips"},
                )
                for event in group_events
                if event.accepted
            ],
            "storyboard_sampling": {
                "base_pair_count": base_pairs,
                "effective_pair_limit": pair_limit,
                "materialized_pair_count": len(storyboard),
                "maximum_pair_count": maximum_pairs,
                "seconds_per_pair": seconds_per_pair,
                "atomic_experiment_count": len(atomic),
            },
        }
        system_prompt = (
            FINAL_GROUP_SYSTEM_PROMPT if final_adjudicated else GROUP_SYSTEM_PROMPT
        )
        fingerprint = _semantic_fingerprint(
            "experiment-group",
            config,
            system_prompt,
            semantic_evidence,
            storyboard,
        )
        persistent_cache_path = _semantic_cache_path(
            config, "experiment-groups", fingerprint
        )
        run_cache_path = cache_root / f"{fingerprint}.json"
        cached = None
        if _semantic_cache_reads_enabled(config):
            cached = _read_semantic_cache(persistent_cache_path, fingerprint)
            if cached is None:
                cached = _read_semantic_cache(run_cache_path, fingerprint)
        if cached is not None:
            return group, cached
        result = analyzer.analyze_group(
            group,
            atomic,
            group_events,
            storyboard,
            system_prompt=system_prompt,
            final_adjudicated=final_adjudicated,
        )
        if result.get("status") == "completed":
            result = _write_semantic_cache(persistent_cache_path, fingerprint, result)
            write_json(run_cache_path, result)
        return group, result

    workers = max(1, int(config["mllm"].get("group_workers", 2)))
    semantic_results: list[tuple[str, dict[str, Any]]] = []
    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(groups)))) as executor:
            futures = [executor.submit(analyze, group) for group in groups]
            for future in as_completed(futures):
                group, result = future.result()
                semantic_results.append((group.group_id, result))
                group.model_understanding = result
                if result.get("status") == "completed":
                    group.experiment_name = str(
                        result.get("experiment_name") or group.experiment_name
                    )
                    group.experiment_name_en = _safe_slug(
                        str(result.get("experiment_name_en") or group.experiment_name_en)
                    )
                    confirmed = result.get("continuity_type_confirmed")
                    # Rule-based continuous chains may be split only by an explicit
                    # model conflict; rule-based independent groups never merge here.
                    if confirmed == "continuous" and group.continuity_type == "continuous":
                        group.continuity_reason += "；模型确认连续"
                    elif confirmed not in {group.continuity_type, "uncertain", None}:
                        result.setdefault("uncertainties", []).append(
                            "模型连续性判断与物理规则冲突，归档采用物理规则并保留冲突"
                        )
                index = int(group.group_id.rsplit("-", 1)[-1])
                group.archive_folder = _group_folder_name(
                    layout,
                    index,
                    group.experiment_name,
                    group.experiment_name_en,
                )
                for segment_id in group.atomic_experiment_ids:
                    segment = by_segment[segment_id]
                    segment.experiment_name = group.experiment_name
                    segment.experiment_name_en = group.experiment_name_en
                    segment.semantic_understanding = result
    finally:
        analyzer.close()
    _raise_for_incomplete_semantic_results(
        layout,
        stage="experiment_group",
        results=semantic_results,
        fail_run=bool(config.get("mllm", {}).get("fail_run_on_incomplete", False)),
    )


def _write_aligned_frame(left: Path, right: Path, destination: Path, labels: tuple[str, str]) -> None:
    images = [cv2.imread(str(left)), cv2.imread(str(right))]
    if any(image is None for image in images):
        raise RuntimeError(f"无法读取对齐关键帧: {left}, {right}")
    assert images[0] is not None and images[1] is not None
    target_height = min(images[0].shape[0], images[1].shape[0])
    rendered = []
    for image, label in zip(images, labels, strict=True):
        scale = target_height / image.shape[0]
        resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), target_height))
        cv2.rectangle(resized, (0, 0), (resized.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        rendered.append(resized)
    combined = np.hstack(rendered)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), combined):
        raise RuntimeError(f"写入对齐关键帧失败: {destination}")


def _key_material_view_pair(
    group: ExperimentGroup,
    event: EvidenceEvent,
) -> tuple[str, str]:
    selection = event.observability.get("key_material_view_selection") or {}
    return (
        str(selection.get("first_person_view") or group.first_person_view),
        str(selection.get("third_person_view") or group.third_person_view),
    )


def _select_key_material_view_pair(
    group: ExperimentGroup,
    event: EvidenceEvent,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Choose real same-role views that contain the event key timestamp.

    Short synchronized recordings can have unequal physical tails.  A group
    representative may therefore be valid for the experiment clip but no
    longer contain a late event key frame.  Select another existing view of
    the same role only when its aligned local timestamp is physically inside
    the decodable source; never clamp the timestamp or synthesize evidence.
    """

    direct = {
        str(item)
        for item in (event.semantic_review or {}).get(
            "directly_supported_view_ids", []
        )
    }
    supported = {str(item) for item in event.supporting_views}
    by_role = {
        ViewRole.FIRST_PERSON: group.first_person_view,
        ViewRole.THIRD_PERSON: group.third_person_view,
    }
    candidates_receipt: dict[str, list[dict[str, Any]]] = {}
    for role, preferred in by_role.items():
        role_candidates: list[dict[str, Any]] = []
        for view in views:
            view_id = view.view_id
            if (
                view.role != role
                or view_id not in infos
                or view_id not in transforms
            ):
                continue
            local_ms = float(transforms[view_id].to_local(event.key_global_ms))
            duration_ms = float(infos[view_id].duration_ms)
            in_bounds = 0.0 <= local_ms <= duration_ms
            margin_ms = min(local_ms, duration_ms - local_ms) if in_bounds else -1.0
            role_candidates.append(
                {
                    "view_id": view_id,
                    "role": role.value,
                    "local_key_ms": round(local_ms, 6),
                    "duration_ms": round(duration_ms, 6),
                    "in_physical_bounds": in_bounds,
                    "directly_supported": view_id in direct,
                    "candidate_supported": view_id in supported,
                    "group_preferred": view_id == preferred,
                    "physical_margin_ms": round(margin_ms, 6),
                    "stable_object_identities": sorted(
                        event_view_stable_identities(event, view_id)
                    ),
                    "stable_actor_track_ids": sorted(
                        event_view_actor_identities(event, view_id)
                    ),
                }
            )
        candidates_receipt[role.value] = role_candidates
        if not any(item["in_physical_bounds"] for item in role_candidates):
            raise ValueError(
                f"{event.event_id} has no {role.value} view containing global "
                f"key timestamp {event.key_global_ms:.3f} ms"
            )

    pair_candidates: list[dict[str, Any]] = []
    for first in candidates_receipt[ViewRole.FIRST_PERSON.value]:
        if not first["in_physical_bounds"]:
            continue
        for third in candidates_receipt[ViewRole.THIRD_PERSON.value]:
            if not third["in_physical_bounds"]:
                continue
            first_classes = {
                str(item[0]) for item in first["stable_object_identities"]
            }
            third_classes = {
                str(item[0]) for item in third["stable_object_identities"]
            }
            shared_identity_classes = sorted(first_classes & third_classes)
            both_action_supported = bool(
                first["candidate_supported"] and third["candidate_supported"]
            )
            identity_conflict = bool(
                both_action_supported
                and first_classes
                and third_classes
                and not shared_identity_classes
            )
            pair_candidates.append(
                {
                    "first_person_view": first["view_id"],
                    "third_person_view": third["view_id"],
                    "both_views_directly_supported": bool(
                        first["directly_supported"] and third["directly_supported"]
                    ),
                    "direct_support_count": int(first["directly_supported"])
                    + int(third["directly_supported"]),
                    "both_views_candidate_supported": both_action_supported,
                    "candidate_support_count": int(first["candidate_supported"])
                    + int(third["candidate_supported"]),
                    "shared_identity_classes": shared_identity_classes,
                    "stable_identity_available_in_both_views": bool(
                        first_classes and third_classes
                    ),
                    "actor_identity_available_in_both_views": bool(
                        first["stable_actor_track_ids"]
                        and third["stable_actor_track_ids"]
                    ),
                    "identity_conflict": identity_conflict,
                    "group_preferred_pair": bool(
                        first["group_preferred"] and third["group_preferred"]
                    ),
                    "minimum_physical_margin_ms": min(
                        float(first["physical_margin_ms"]),
                        float(third["physical_margin_ms"]),
                    ),
                }
            )
    eligible_pairs = [
        item for item in pair_candidates if not item["identity_conflict"]
    ]
    if not eligible_pairs:
        raise ValueError(
            f"{event.event_id} has no identity-compatible first/third-person "
            f"view pair at {event.key_global_ms:.3f} ms"
        )
    winner = max(
        eligible_pairs,
        key=lambda item: (
            bool(item["both_views_directly_supported"]),
            int(item["direct_support_count"]),
            bool(item["both_views_candidate_supported"]),
            int(item["candidate_support_count"]),
            bool(item["shared_identity_classes"]),
            bool(item["stable_identity_available_in_both_views"]),
            bool(item["actor_identity_available_in_both_views"]),
            bool(item["group_preferred_pair"]),
            float(item["minimum_physical_margin_ms"]),
            str(item["first_person_view"]),
            str(item["third_person_view"]),
        ),
    )
    pair = (str(winner["first_person_view"]), str(winner["third_person_view"]))
    receipt = {
        "schema_version": "visioncortex-key-material-view-selection/1",
        "policy": "same-role real source containing aligned key timestamp",
        "timestamp_clamped": False,
        "synthetic_cross_view_evidence": False,
        "key_global_ms": float(event.key_global_ms),
        "first_person_view": pair[0],
        "third_person_view": pair[1],
        "group_first_person_view": group.first_person_view,
        "group_third_person_view": group.third_person_view,
        "fallback_applied": pair
        != (group.first_person_view, group.third_person_view),
        "candidates": candidates_receipt,
        "pair_candidates": pair_candidates,
        "selected_pair_identity": {
            key: value
            for key, value in winner.items()
            if key
            not in {
                "first_person_view",
                "third_person_view",
                "minimum_physical_margin_ms",
                "group_preferred_pair",
            }
        },
    }
    return pair, receipt


def _select_key_material_view_pair_with_peak_fallback(
    group: ExperimentGroup,
    event: EvidenceEvent,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    accepted_peak_global_ms: float,
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Restore the accepted CV peak if a re-ranked frame lacks dual coverage."""

    selected_key_global_ms = float(event.key_global_ms)
    try:
        return _select_key_material_view_pair(
            group, event, views, infos, transforms
        )
    except ValueError as selected_error:
        accepted_peak_global_ms = float(accepted_peak_global_ms)
        if (
            accepted_peak_global_ms == selected_key_global_ms
            or not event.global_start_ms
            <= accepted_peak_global_ms
            <= event.global_end_ms
        ):
            raise
        event.key_global_ms = accepted_peak_global_ms
        try:
            pair, receipt = _select_key_material_view_pair(
                group, event, views, infos, transforms
            )
        except ValueError as fallback_error:
            event.key_global_ms = selected_key_global_ms
            raise selected_error from fallback_error
        receipt["key_timestamp_fallback"] = {
            "applied": True,
            "reason": (
                "participant-ranked frame lacked real dual-role physical "
                "coverage; restored accepted CV event peak"
            ),
            "participant_ranked_global_ms": selected_key_global_ms,
            "accepted_event_peak_global_ms": accepted_peak_global_ms,
            "timestamp_clamped": False,
        }
        return pair, receipt


def _artifact_json(
    group: ExperimentGroup,
    event: EvidenceEvent,
    artifact_type: str,
    artifact_file: str,
    view_id: str | None,
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
) -> dict[str, Any]:
    first_material_view, third_material_view = _key_material_view_pair(group, event)
    understanding = event.model_understanding or {}
    physical_change = understanding.get("physical_change") or {}
    before_state = str(physical_change.get("before") or "unknown")
    after_state = str(physical_change.get("after") or "unknown")
    semantic_review = event.semantic_review or {}
    final_participants = [
        str(item)
        for item in semantic_review.get("final_participant_objects") or []
        if str(item).strip()
    ]
    # The top-level normalized contract describes the curated event only.
    # Background/model-context objects and the rejected CV hypothesis remain
    # available in provenance, but must never leak back into the final object
    # slots or participant-only presentation.
    object_names = list(dict.fromkeys(final_participants or event.objects))
    material_name = _key_material_semantic_name(event)

    def contains(name: str, keywords: tuple[str, ...]) -> bool:
        lowered = name.lower()
        return any(keyword in lowered for keyword in keywords)

    def first_object(keywords: tuple[str, ...], excluded: set[str] | None = None) -> str | None:
        excluded = excluded or set()
        return next(
            (
                name
                for name in object_names
                if name not in excluded and contains(name, keywords)
            ),
            None,
        )

    actor = first_object(("gloved_hand", "hand", "手套", "手"))
    closure = first_object(("bottle_cap", "tube_cap", "瓶盖", "管盖"))
    tool = first_object(
        (
            "pipette",
            "spearhead",
            "spatula",
            "dropper",
            "移液",
            "吸头",
            "药勺",
            "滴管",
        )
    )
    source = first_object(
        ("reagent_bottle", "bottle", "试剂瓶", "试剂容器", "瓶"),
        {item for item in (tool, closure) if item},
    )
    target = first_object(
        ("tube", "paper", "balance", "beaker", "flask", "离心管", "称量纸", "天平", "烧杯", "锥形瓶"),
        {item for item in (tool, source) if item},
    )

    def tracked_id(name: str | None, fallback: str) -> str:
        if not name:
            return "unknown"
        slug = _safe_slug(name).lower()
        if slug == "unknown":
            known = {
                "移液器": "pipette",
                "移液枪": "pipette",
                "枪头": "pipette-tip",
                "吸头": "pipette-tip",
                "药勺": "spatula",
                "称量纸": "weighing-paper",
                "试剂瓶": "reagent-bottle",
                "离心管": "centrifuge-tube",
                "分析天平": "analytical-balance",
            }
            slug = next((value for key, value in known.items() if key in name), fallback)
        track_id = None
        for candidate in event.candidates:
            if name not in candidate.objects:
                continue
            for evidence in candidate.evidence:
                for key in ("object_track_id", "track_id", "tool_track_id", "container_track_id"):
                    if evidence.get(key) is not None:
                        track_id = evidence[key]
                        break
                if track_id is not None:
                    break
            if track_id is not None:
                break
        return f"{slug}-{int(track_id):03d}" if track_id is not None else f"{slug}-unresolved"

    normalized_action = {
        "liquid_movement": "liquid_transfer",
    }.get(event.action_type.value, event.action_type.value)
    raw_state_receipt = event.state_machine or {}
    raw_state_action = str(raw_state_receipt.get("action_type") or "")
    state_receipt = (
        raw_state_receipt if raw_state_action == normalized_action else {}
    )
    combined = " ".join(object_names).lower()
    if normalized_action in {"liquid_transfer", "pipette_transfer_operation"}:
        action_subtype = "pipette_transfer" if any(
            keyword in combined for keyword in ("pipette", "spearhead", "移液", "吸头")
        ) else "container_pour"
        phases = [
            "source_approach",
            "source_contact",
            "source_withdraw",
            "transport",
            "target_contact",
            "target_release",
        ]
    elif normalized_action == "container_state_change":
        action_subtype = "container_open_close"
        phases = ["container_approach", "container_contact", "state_transition", "release"]
    elif normalized_action == "device_panel_operation":
        action_subtype = "device_control_operation"
        phases = ["panel_approach", "panel_contact", "control_activation", "withdraw"]
    elif normalized_action == "object_movement":
        action_subtype = "tool_movement" if tool else "object_relocation"
        phases = ["object_approach", "grasp", "lift", "transport", "place", "release"]
    else:
        action_subtype = "tool_contact" if tool else "generic_hand_object_contact"
        phases = ["object_approach", "contact", "manipulation", "release"]
    action_subtype = str(state_receipt.get("action_subtype") or action_subtype)
    phases = list(state_receipt.get("phases") or phases)

    observations = []
    for index, item in enumerate(understanding.get("per_view_observations") or [], 1):
        observation_view = str(item.get("view_id") or "unknown")
        observations.append(
            {
                "observation_id": f"{event.event_id}-obs-{index:02d}",
                "view_id": observation_view,
                "view_role": (
                    "first_person"
                    if observation_view == first_material_view
                    else "third_person"
                    if observation_view == third_material_view
                    else "unknown"
                ),
                "timestamp_us": round(event.key_global_ms * 1000.0),
                "observed_fact": str(item.get("observation") or ""),
            }
        )
    if not observations:
        observations = [
            {
                "observation_id": f"{event.event_id}-obs-{index:02d}",
                "view_id": supported_view,
                "view_role": (
                    "first_person" if supported_view == first_material_view else "third_person"
                ),
                "timestamp_us": round(event.key_global_ms * 1000.0),
                "observed_fact": f"CV accepted {event.action_type.value}: {', '.join(event.objects)}",
            }
            for index, supported_view in enumerate(event.supporting_views, 1)
        ]

    alignment_uncertainty_us = max(
        80_000,
        round(
            max(
                (
                    max(
                        float(transforms[item].uncertainty_ms),
                        float(transforms[item].csv_rmse_ms or 0.0),
                    )
                    for item in (first_material_view, third_material_view)
                    if item in transforms
                ),
                default=80.0,
            )
            * 1000.0
        ),
    )
    cross_view_associations = [
        {
            "association_id": f"{event.event_id}-first-third",
            "source_view_id": first_material_view,
            "target_view_id": third_material_view,
            "global_timestamp_us": round(event.key_global_ms * 1000.0),
            "source_local_timestamp_us": round(
                transforms[first_material_view].to_local(event.key_global_ms) * 1000.0
            ),
            "target_local_timestamp_us": round(
                transforms[third_material_view].to_local(event.key_global_ms) * 1000.0
            ),
            "time_uncertainty_us": alignment_uncertainty_us,
            "consistency": understanding.get("cross_view_consistency", "unreviewed"),
            "both_views_support_action": all(
                item in event.supporting_views
                for item in (first_material_view, third_material_view)
            ),
        }
    ]

    current_step = str(understanding.get("current_step") or "")
    next_step = str(understanding.get("next_step") or "")
    next_step_evidence = dict(understanding.get("next_step_evidence") or {})
    next_step_status = str(next_step_evidence.get("status") or "unknown")
    if next_step_status not in {"observed", "inferred", "unknown"}:
        next_step_status = "unknown"
    uncertainties = list(
        dict.fromkeys(
            [
                *[str(item) for item in event.uncertainty],
                *[
                        f"CV未直接观察: {item}"
                        for item in (
                            event.observability.get("unmet_visual_requirements") or []
                            if event.observability
                            else []
                        )
                ],
                *[str(item) for item in understanding.get("uncertainties") or []],
            ]
        )
    )
    consistency = str(understanding.get("cross_view_consistency") or "unreviewed")
    model_confidence = float(understanding.get("confidence") or 0.0)
    confirmed_action = str(understanding.get("action_type_confirmed") or "unknown")
    status = "confirmed"
    if understanding.get("status") != "completed":
        status = "provisional_cv_only"
    elif confirmed_action == "unknown" or consistency == "conflict" or model_confidence < 0.55:
        status = "uncertain"
    if state_receipt.get("lifecycle_state") == "incomplete_end":
        status = "incomplete_observation"

    role_for = (
        "aligned_first_third"
        if view_id is None
        else "first_person"
        if view_id == first_material_view
        else "third_person"
    )
    event_uid = (
        stable_event_uid(
            archive_id,
            group.group_uid or group.group_id,
            event.event_id,
        )
        if archive_id
        else None
    )
    key_frames = [
        {
            "view_id": item,
            "view_role": (
                "aligned_first_third"
                if item == "aligned_first_third"
                else "first_person"
                if item == first_material_view
                else "third_person"
            ),
            "timestamp_us": round(event.key_global_ms * 1000.0),
            "path": path,
            **(
                {
                    "artifact_uid": stable_artifact_uid(event_uid, "key_frame", item),
                    "sidecar_path": str(Path(path).with_suffix(".json")).replace("\\", "/"),
                }
                if event_uid
                else {}
            ),
        }
        for item, path in event.key_frames.items()
    ]
    key_clips = [
        {
            "view_id": item,
            "view_role": (
                "aligned_first_third"
                if item == "aligned_first_third"
                else "first_person"
                if item == first_material_view
                else "third_person"
            ),
            "start_us": round(max(0.0, event.global_start_ms - 2000.0) * 1000.0),
            "end_us": round((event.global_end_ms + 3000.0) * 1000.0),
            "path": path,
            **(
                {
                    "artifact_uid": stable_artifact_uid(event_uid, "key_clip", item),
                    "sidecar_path": str(Path(path).with_suffix(".json")).replace("\\", "/"),
                }
                if event_uid
                else {}
            ),
        }
        for item, path in event.key_clips.items()
    ]
    candidate_ids = [candidate.candidate_id for candidate in event.candidates]
    frame_evidence_ids = [
        f"{candidate.view_id}:frame-{item['frame_index']}"
        for candidate in event.candidates
        for item in candidate.evidence
        if item.get("frame_index") is not None
    ]
    evidence_ids = list(dict.fromkeys([*candidate_ids, *frame_evidence_ids]))
    observed_facts = [item for item in [current_step] if item]
    observed_facts.extend(
        item["observed_fact"] for item in observations if item.get("observed_fact")
    )
    supported_inferences = []
    if next_step and next_step not in {"未知", "unknown", "不确定"}:
        if next_step_status == "observed":
            observed_facts.append(f"已观察后续动作：{next_step}")
        elif next_step_status == "inferred":
            supported_inferences.append(f"预测下一步：{next_step}")
    contradictions = []
    if consistency == "conflict":
        contradictions.append("第一人称与第三人称观察发生冲突，详见 observations")

    cross_view_score = {
        "consistent": 0.95,
        "partial": 0.65,
        "conflict": 0.20,
        "single_view": 0.40,
    }.get(consistency, 0.35)
    change_observed = before_state != "unknown" and after_state != "unknown"
    action_agrees = confirmed_action in {event.action_type.value, normalized_action}
    if normalized_action == ActionType.CONTAINER_STATE_CHANGE.value:
        normalized_objects = {
            "actor": tracked_id(actor, "actor"),
            "container": tracked_id(source, "container"),
            "closure": tracked_id(closure, "closure"),
        }
        receipt_before = str(
            (state_receipt.get("state_before") or {}).get("container")
            or "unknown"
        )
        receipt_after = str(
            (state_receipt.get("state_after") or {}).get("container")
            or "unknown"
        )
        normalized_state_before = {
            "container": before_state if before_state != "unknown" else receipt_before
        }
        normalized_state_after = {
            "container": after_state if after_state != "unknown" else receipt_after
        }
    elif normalized_action in {
        ActionType.HAND_OBJECT_CONTACT.value,
        ActionType.DEVICE_PANEL_OPERATION.value,
    }:
        normalized_objects = {
            "actor": tracked_id(actor, "actor"),
            "target": tracked_id(target, "target"),
        }
        normalized_state_before = {
            "target": before_state,
            **dict(state_receipt.get("state_before") or {}),
        }
        normalized_state_after = {
            "target": after_state,
            **dict(state_receipt.get("state_after") or {}),
        }
    else:
        normalized_objects = {
            "tool": tracked_id(tool, "tool"),
            "source": tracked_id(source, "source"),
            "target": tracked_id(target, "target"),
        }
        normalized_state_before = {
            "tool": before_state,
            "source": "unknown",
            "target": "unknown",
            **dict(state_receipt.get("state_before") or {}),
        }
        normalized_state_after = {
            "tool": after_state,
            "source": "unknown",
            "target": "unknown",
            **dict(state_receipt.get("state_after") or {}),
        }

    pre_curation_action = str(
        semantic_review.get("pre_curation_action_type") or event.action_type.value
    )
    pre_curation_objects = list(
        semantic_review.get("pre_curation_objects") or event.objects
    )
    pre_curation_state_machine = (
        semantic_review.get("pre_curation_state_machine") or raw_state_receipt
    )

    return {
        "event_id": event.event_id,
        "parent_event_id": group.group_id,
        "parent_event_uid": group.group_uid or group.group_id,
        "actor_id": "operator-01",
        "workstation_id": (
            "weighing-station-01"
            if any(keyword in combined for keyword in ("balance", "天平"))
            else "pipetting-station-01"
            if any(keyword in combined for keyword in ("pipette", "spearhead", "tube", "移液", "吸头", "离心管"))
            else "wet-lab-bench-01"
        ),
        "action_type": normalized_action,
        "action_subtype": action_subtype,
        "start_us": round(event.global_start_ms * 1000.0),
        "end_us": round(event.global_end_ms * 1000.0),
        "peak_timestamp_us": round(event.key_global_ms * 1000.0),
        "time_uncertainty_us": alignment_uncertainty_us,
        "phases": phases,
        "objects": normalized_objects,
        "state_before": normalized_state_before,
        "state_after": normalized_state_after,
        "observations": observations,
        "cross_view_associations": cross_view_associations,
        "decision": {
            "status": status,
            "observed_facts": observed_facts,
            "supported_inferences": supported_inferences,
            "uncertain_claims": uncertainties,
            "contradictions": contradictions,
        },
        "scores": {
            "temporal_continuity": round(min(1.0, 0.5 + float(event.confidence) * 0.5), 4),
            "object_identity": round(min(1.0, 0.35 + min(len(event.objects), 4) * 0.08 + model_confidence * 0.3), 4),
            "phase_completeness": round((0.45 + model_confidence * 0.5) if change_observed else (0.25 + model_confidence * 0.35), 4),
            "cross_view_support": cross_view_score,
            "state_change_support": round(model_confidence if change_observed else 0.25, 4),
            "model_agreement": round((float(event.confidence) + model_confidence) / 2.0 if action_agrees else model_confidence * 0.5, 4),
            "contradiction_penalty": 0.8 if consistency == "conflict" else 0.0,
            **dict(state_receipt.get("scores") or {}),
        },
        "key_frames": key_frames,
        "key_clips": key_clips,
        "evidence_ids": evidence_ids,
        "provenance": {
            "schema_version": "key-material-event-v1.0.0",
            "time_base": "aligned_global_timeline_microseconds",
            "experiment_group_id": group.group_id,
            "experiment_name": group.experiment_name,
            "continuity_type": group.continuity_type,
            "atomic_experiment_ids": group.atomic_experiment_ids,
            "archive_classification": {
                "hierarchy_version": "2.0.0",
                "experiment_folder": (
                    group.archive_folder or _safe_folder_name(group.group_id)
                ),
                "action_type": event.action_type.value,
                "action_category_folder": key_material_action_folder(event.action_type),
                "semantic_file_stem": material_name["file_stem"],
                "primary_object": material_name["primary_object"],
                "object_labels": material_name["object_labels"],
            },
            "sidecar_for": {
                "artifact_type": artifact_type,
                "artifact_file": artifact_file,
                "view_id": view_id,
                "view_role": role_for,
            },
            "cv": {
                "action_type": pre_curation_action,
                "objects": pre_curation_objects,
                "confidence": event.confidence,
                "audit_reason": event.audit_reason,
                "supporting_views": list(
                    semantic_review.get("pre_curation_supporting_views")
                    or event.supporting_views
                ),
                "observability": event.observability,
                "semantic_review": event.semantic_review,
                "continuous_state_machine": pre_curation_state_machine,
            },
            "semantic_final": {
                "action_type": event.action_type.value,
                "participant_objects": object_names,
                "supporting_views": event.supporting_views,
                "continuous_state_machine": state_receipt,
            },
            "mllm": {
                "model": understanding.get("model"),
                "status": understanding.get("status"),
                "usage": understanding.get("usage") or {},
                "latency_seconds": understanding.get("latency_seconds"),
                "attempts": understanding.get("attempts"),
                "current_step": current_step or None,
                "next_step": next_step or None,
                "next_step_status": next_step_status,
                "next_step_evidence": next_step_evidence,
            },
            "alignment": {
                item: transforms[item].model_dump(mode="json")
                for item in (first_material_view, third_material_view)
            },
            "score_method": "deterministic_cv_mllm_evidence_mapping_v1",
            **(
                {
                    "index": {
                        "event_uid": event_uid,
                        "database": "JSON-Config-Files/evidence_index.sqlite",
                        "artifact_registry": "JSON-Config-Files/artifact_registry.jsonl",
                        "evidence_registry": "JSON-Config-Files/evidence_registry.jsonl",
                        "decision_receipt_registry": "JSON-Config-Files/decision_receipt_registry.jsonl",
                        "evidence_refs": [
                            {
                                "evidence_id": evidence_id,
                                "evidence_uid": stable_evidence_uid(
                                    event_uid, evidence_id
                                ),
                            }
                            for evidence_id in evidence_ids
                        ],
                    }
                }
                if event_uid
                else {}
            ),
        },
    }


def prepare_key_material_category_layout(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent] | None = None,
    *,
    include_empty_categories: bool = True,
) -> None:
    """Create the stable experiment/action hierarchy before materialization."""

    event_by_id = {
        event.event_id: event
        for event in (events or [])
        if event.accepted
    }
    for group in groups:
        experiment_folder = group.archive_folder or _safe_folder_name(group.group_id)
        observed_actions = {
            event_by_id[event_id].action_type.value
            for event_id in group.key_event_ids
            if event_id in event_by_id
        }
        for action_type, action_folder in ACTION_CATEGORY_FOLDERS.items():
            if (
                events is not None
                and not include_empty_categories
                and action_type not in observed_actions
            ):
                continue
            (layout.key_frames / experiment_folder / action_folder).mkdir(
                parents=True, exist_ok=True
            )
            (layout.key_clips / experiment_folder / action_folder).mkdir(
                parents=True, exist_ok=True
            )


def write_key_material_category_index(
    layout: ArchiveLayout,
    groups: Sequence[ExperimentGroup],
    events: Sequence[EvidenceEvent],
    publisher: Any | None = None,
    *,
    include_empty_categories: bool = True,
) -> Path:
    """Write a human-browsable and machine-indexable six-category manifest."""

    event_by_id = {
        event.event_id: event for event in events if event.accepted
    }
    experiments: list[dict[str, Any]] = []
    for group in groups:
        experiment_folder = group.archive_folder or _safe_folder_name(group.group_id)
        group_events = [
            event_by_id[event_id]
            for event_id in group.key_event_ids
            if event_id in event_by_id
        ]
        categories: list[dict[str, Any]] = []
        for action_type, action_folder in ACTION_CATEGORY_FOLDERS.items():
            category_events = [
                event
                for event in group_events
                if event.action_type.value == action_type
            ]
            frame_category_folder = (
                layout.key_frames / experiment_folder / action_folder
            )
            clip_category_folder = (
                layout.key_clips / experiment_folder / action_folder
            )
            category_materialized = bool(
                category_events or include_empty_categories
            )
            category_summary = {
                "schema_version": "visioncortex-key-material-category/1.0.0",
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "experiment_folder": experiment_folder,
                "action_type": action_type,
                "action_category_folder": action_folder,
                "event_count": len(category_events),
                "coverage_status": (
                    "observed" if category_events else "not_observed"
                ),
                "absence_reason": (
                    None
                    if category_events
                    else (
                        "No accepted evidence event of this action type was "
                        "observed in the bounded experiment."
                    )
                ),
                "event_ids": [event.event_id for event in category_events],
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_folder": _key_material_event_folder_name(
                            layout, experiment_folder, event
                        ),
                        "material_name": _key_material_semantic_name(event),
                    }
                    for event in category_events
                ],
            }
            frame_summary_path = frame_category_folder / "Category.json"
            clip_summary_path = clip_category_folder / "Category.json"
            if category_materialized:
                frame_category_folder.mkdir(parents=True, exist_ok=True)
                clip_category_folder.mkdir(parents=True, exist_ok=True)
                write_json(
                    frame_summary_path,
                    {**category_summary, "media_kind": "key_frame"},
                )
                write_json(
                    clip_summary_path,
                    {**category_summary, "media_kind": "key_clip"},
                )
                if publisher is not None:
                    publisher.publish_file(frame_summary_path)
                    publisher.publish_file(clip_summary_path)
            else:
                # A later semantic or visual-quality pass can quarantine every
                # event that initially populated a category.  Remove the stale
                # summary and its now-empty directory so the user-facing NAS
                # tree never claims that a JSON-only category contains media.
                for summary_path, category_folder in (
                    (frame_summary_path, frame_category_folder),
                    (clip_summary_path, clip_category_folder),
                ):
                    summary_path.unlink(missing_ok=True)
                    try:
                        category_folder.rmdir()
                    except OSError:
                        # Preserve any unexpected material for audit instead of
                        # deleting a non-empty directory here.
                        pass
            categories.append(
                {
                    "action_type": action_type,
                    "folder": action_folder if category_materialized else None,
                    "event_count": len(category_events),
                    "coverage_status": category_summary["coverage_status"],
                    "absence_reason": category_summary["absence_reason"],
                    "materialized": category_materialized,
                    "key_frames_folder": (
                        Path("Key-Materials")
                        / "Key-Frames"
                        / experiment_folder
                        / action_folder
                    ).as_posix()
                    if category_materialized
                    else None,
                    "key_clips_folder": (
                        Path("Key-Materials")
                        / "Key-Clips"
                        / experiment_folder
                        / action_folder
                    ).as_posix()
                    if category_materialized
                    else None,
                    "key_frames_category_summary": _relative(
                        frame_summary_path, layout.root
                    )
                    if category_materialized
                    else None,
                    "key_clips_category_summary": _relative(
                        clip_summary_path, layout.root
                    )
                    if category_materialized
                    else None,
                    "events": [
                        {
                            "event_id": event.event_id,
                            "event_folder": _key_material_event_folder_name(
                                layout, experiment_folder, event
                            ),
                            "material_name": _key_material_semantic_name(event),
                            "peak_timestamp_us": round(
                                event.key_global_ms * 1000.0
                            ),
                            "key_frames": dict(event.key_frames),
                            "key_clips": dict(event.key_clips),
                        }
                        for event in category_events
                    ],
                }
            )
        experiments.append(
            {
                "group_id": group.group_id,
                "experiment_name": group.experiment_name,
                "experiment_name_en": group.experiment_name_en,
                "experiment_folder": experiment_folder,
                "key_event_count": len(group_events),
                "action_categories": categories,
            }
        )
    path = layout.key_materials / "Key-Material-Category-Index.json"
    write_json(
        path,
        {
            "schema_version": "visioncortex-key-material-category-index/1.0.0",
            "hierarchy": (
                "Key-Materials/{Key-Frames|Key-Clips}/"
                "{Experiment}/{Action-Category}/{Event}"
            ),
            "category_count": len(ACTION_CATEGORY_FOLDERS),
            "categories": [
                {
                    "order": index,
                    "action_type": action_type,
                    "folder": action_folder,
                }
                for index, (action_type, action_folder) in enumerate(
                    ACTION_CATEGORY_FOLDERS.items(), 1
                )
            ],
            "experiments": experiments,
        },
    )
    if publisher is not None:
        publisher.publish_file(path)
    return path


def _event_participant_boxes(
    event: EvidenceEvent,
    detections: Sequence[dict[str, Any]],
    view_id: str | None = None,
    maximum_interaction_gap_norm: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter a delivery frame to the objects participating in one event.

    Full detector output remains in the frame-evidence ledger. Key-material
    images are explanatory evidence, so background detections must not appear.
    """

    def normalize(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    participant_classes = {
        normalize(item)
        for item in event.objects
        if str(item).strip()
    }
    actor_classes = {"hand", "gloved_hand"}
    tool_classes = {"pipette", "spearhead"}
    vessel_classes = {
        "container",
        "beaker",
        "tube",
        "sample_bottle",
        "sample_bottle_blue",
        "reagent_bottle",
    }
    detected_classes = {normalize(box.get("class_name")) for box in detections}
    groups: list[tuple[str, set[str]]] = []
    consumed: set[str] = set()
    if participant_classes & actor_classes:
        groups.append(("actor", actor_classes))
        consumed.update(actor_classes)
    if event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        # A manipulation frame is only explanatory when it shows the operated
        # tool and its nearby vessel.  The actor is optional in the ontology but
        # is a useful visible participant whenever the detector sees it.
        if detected_classes & actor_classes:
            groups.append(("actor", actor_classes))
            consumed.update(actor_classes)
        participant_tool_classes = participant_classes & tool_classes
        participant_vessel_classes = participant_classes & vessel_classes
        if participant_tool_classes:
            groups.append(("tool", participant_tool_classes))
            consumed.update(participant_tool_classes)
        if participant_vessel_classes:
            # ``container`` is a semantic slot rather than a trained visual
            # class, and the two sample-bottle labels are detector aliases.
            # Broaden only those explicitly requested slots; never add every
            # bench vessel merely because the action is a pipette operation.
            vessel_slot_classes = set(participant_vessel_classes)
            if "container" in participant_vessel_classes:
                vessel_slot_classes.update(vessel_classes)
            if participant_vessel_classes & {
                "sample_bottle",
                "sample_bottle_blue",
            }:
                vessel_slot_classes.update(
                    {"sample_bottle", "sample_bottle_blue"}
                )
            groups.append(("vessel", vessel_slot_classes))
            consumed.update(vessel_slot_classes)
    elif event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        # Semantic review can correctly identify the physical role (the
        # bottle being opened) even when the detector uses a neighbouring
        # bottle class.  Treat those detector labels as one *container slot*,
        # then choose the single instance closest to the manipulating hand.
        # This preserves participant-only rendering without boxing every
        # bottle on a crowded bench.
        closure_classes: set[str] = set()
        container_classes: set[str] = set()
        if "bottle_cap" in participant_classes:
            closure_classes.add("bottle_cap")
            container_classes.update(
                {
                    "container",
                    "sample_bottle",
                    "sample_bottle_blue",
                    "reagent_bottle",
                }
            )
        if "tube_cap" in participant_classes:
            closure_classes.add("tube_cap")
            container_classes.update({"container", "tube"})
        if participant_classes & vessel_classes and not container_classes:
            container_classes.update(participant_classes & vessel_classes)
        if closure_classes:
            groups.append(("closure", closure_classes))
        if container_classes:
            groups.append(("container", container_classes))
        consumed.update(closure_classes | container_classes)
    elif participant_classes & tool_classes:
        groups.append(("tool", participant_classes & tool_classes))
        consumed.update(tool_classes)
    for class_name in sorted(participant_classes - consumed):
        groups.append((class_name, {class_name}))

    # Avoid duplicate semantic slots when the action-specific branch already
    # inserted an actor group.
    unique_groups: list[tuple[str, set[str]]] = []
    seen_group_names: set[str] = set()
    for name, classes in groups:
        if name in seen_group_names:
            continue
        seen_group_names.add(name)
        unique_groups.append((name, classes))
    groups = unique_groups
    boxes_by_group = [
        [
            dict(box)
            for box in detections
            if normalize(box.get("class_name")) in classes
        ]
        for _, classes in groups
    ]
    relation_rejected_count = 0
    relation_gate_applied = False
    boxes_by_name = {
        group[0]: boxes
        for group, boxes in zip(groups, boxes_by_group, strict=True)
    }
    actor_box_pool = boxes_by_name.get("actor") or []
    actor_required_actions = {
        ActionType.HAND_OBJECT_CONTACT,
        ActionType.CONTAINER_STATE_CHANGE,
        ActionType.DEVICE_PANEL_OPERATION,
        ActionType.PIPETTE_TRANSFER_OPERATION,
    }

    def adjacent_to_any(
        box: dict[str, Any], references: Sequence[dict[str, Any]]
    ) -> bool:
        return bool(references) and min(
            _box_edge_gap_norm(box, reference) for reference in references
        ) <= float(maximum_interaction_gap_norm)

    if event.action_type in actor_required_actions:
        relation_gate_applied = True
        for group_name, boxes in list(boxes_by_name.items()):
            if group_name == "actor":
                continue
            references = actor_box_pool
            if (
                event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION
                and group_name == "vessel"
            ):
                references = boxes_by_name.get("tool") or []
            eligible = [box for box in boxes if adjacent_to_any(box, references)]
            relation_rejected_count += len(boxes) - len(eligible)
            boxes_by_name[group_name] = eligible
    elif actor_box_pool:
        # For movement classes an actor may be occluded, but when it is visible
        # a distant static instance cannot be presented as the manipulated one.
        relation_gate_applied = True
        for group_name, boxes in list(boxes_by_name.items()):
            if group_name == "actor":
                continue
            eligible = [box for box in boxes if adjacent_to_any(box, actor_box_pool)]
            relation_rejected_count += len(boxes) - len(eligible)
            boxes_by_name[group_name] = eligible
    boxes_by_group = [boxes_by_name[group[0]] for group in groups]
    available = [
        (group, boxes)
        for group, boxes in zip(groups, boxes_by_group, strict=True)
        if boxes
    ]

    candidate_track_ids: set[int] = set()
    if view_id is not None:
        for candidate in event.candidates:
            if candidate.view_id != view_id:
                continue
            for evidence in candidate.evidence:
                for key, value in evidence.items():
                    if not key.endswith("track_id") or value is None:
                        continue
                    try:
                        candidate_track_ids.add(int(value))
                    except (TypeError, ValueError):
                        continue

    def center(box: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(item) for item in box.get("xyxy_norm") or (0, 0, 0, 0))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
        lx, ly = center(left)
        rx, ry = center(right)
        return math.hypot(lx - rx, ly - ry)

    best_choice: tuple[dict[str, Any], ...] = ()
    best_score = float("-inf")
    if available:
        for choice in product(*(boxes for _, boxes in available)):
            labels = [group[0] for group, _ in available]
            by_label = dict(zip(labels, choice, strict=True))
            score = sum(float(box.get("confidence") or 0.0) for box in choice)
            score += sum(min(float(box.get("roi_motion") or 0.0), 100.0) / 100.0 for box in choice)
            score += 1.5 * sum(
                int(box.get("track_id")) in candidate_track_ids
                for box in choice
                if box.get("track_id") is not None
            )
            specific_proximity_applied = False
            if "actor" in by_label and "tool" in by_label:
                score -= 4.0 * distance(by_label["actor"], by_label["tool"])
                specific_proximity_applied = True
            if "tool" in by_label and "vessel" in by_label:
                score -= 4.0 * distance(by_label["tool"], by_label["vessel"])
                specific_proximity_applied = True
            if "actor" in by_label and "container" in by_label:
                score -= 5.0 * distance(
                    by_label["actor"], by_label["container"]
                )
                specific_proximity_applied = True
            if "actor" in by_label and "closure" in by_label:
                score -= 5.0 * distance(
                    by_label["actor"], by_label["closure"]
                )
                specific_proximity_applied = True
            if "closure" in by_label and "container" in by_label:
                score -= 3.0 * distance(
                    by_label["closure"], by_label["container"]
                )
                specific_proximity_applied = True
            if not specific_proximity_applied and "actor" in by_label and len(choice) > 1:
                score -= 4.0 * min(
                    distance(by_label["actor"], box)
                    for label, box in by_label.items()
                    if label != "actor"
                )
            elif not specific_proximity_applied and len(choice) > 1:
                score -= 2.0 * min(
                    distance(left, right)
                    for index, left in enumerate(choice)
                    for right in choice[index + 1 :]
                )
            if score > best_score:
                best_score = score
                best_choice = tuple(dict(box) for box in choice)
    rendered = list(best_choice)
    effective_participant_classes = participant_classes | {
        normalize(box.get("class_name")) for box in rendered
    }
    rendered_classes = sorted(
        {
            str(box.get("class_name") or "")
            for box in rendered
            if str(box.get("class_name") or "").strip()
        }
    )
    suppressed_classes = sorted(
        {
            str(box.get("class_name") or "")
            for box in detections
            if box not in rendered and str(box.get("class_name") or "").strip()
        }
    )
    same_class_suppressed = sum(
        normalize(box.get("class_name")) in effective_participant_classes
        and box not in rendered
        for box in detections
    )
    return rendered, {
        "mode": "event_participants_only",
        "instance_policy": "single_interacting_instance_per_semantic_slot",
        "participant_classes": sorted(effective_participant_classes),
        "participant_class_groups": [
            {"name": name, "classes": sorted(classes)} for name, classes in groups
        ],
        "detected_box_count": len(detections),
        "rendered_box_count": len(rendered),
        "suppressed_background_box_count": len(detections) - len(rendered),
        "suppressed_same_class_instance_count": same_class_suppressed,
        "rendered_classes": rendered_classes,
        "rendered_detections": [
            {
                "class_name": str(box.get("class_name") or ""),
                "confidence": round(float(box.get("confidence") or 0.0), 6),
                "detector_source": str(
                    box.get("detector_source") or "closed_set_yolo_tensorrt"
                ),
                "track_id": box.get("track_id"),
            }
            for box in rendered
        ],
        "minimum_rendered_confidence": (
            round(
                min(float(box.get("confidence") or 0.0) for box in rendered),
                6,
            )
            if rendered
            else None
        ),
        "rendered_track_ids": [
            box.get("track_id") for box in rendered if box.get("track_id") is not None
        ],
        "suppressed_background_classes": suppressed_classes,
        "extraneous_rendered_classes": sorted(
            set(rendered_classes) - effective_participant_classes
        ),
        "instance_selection_score": (
            round(best_score, 6) if math.isfinite(best_score) else None
        ),
        "interaction_relation_gate": {
            "applied": relation_gate_applied,
            "maximum_edge_gap_norm": float(maximum_interaction_gap_norm),
            "rejected_box_count": relation_rejected_count,
            "rule": (
                "actor_to_object; pipette_tool_to_vessel"
                if event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION
                else "actor_to_manipulated_object"
            ),
        },
    }


def _event_key_frame_score(
    event: EvidenceEvent,
    frame: FrameEvidence,
) -> tuple[float, dict[str, Any]]:
    detections = [box.model_dump() for box in frame.detections]
    rendered, receipt = _event_participant_boxes(
        event, detections, view_id=frame.view_id
    )
    group_names = {
        item["name"]
        for item in receipt.get("participant_class_groups") or []
        if any(
            str(box.get("class_name") or "") in set(item.get("classes") or [])
            for box in rendered
        )
    }
    score = 100.0 * len(group_names)
    if event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        score += 100.0 * int({"tool", "vessel"}.issubset(group_names))
        score += 40.0 * int("actor" in group_names)
    key_frame_phase_bias = 0.0
    if event.action_type == ActionType.CONTAINER_STATE_CHANGE:
        # Prefer an equally participant-rich frame that shows the resulting
        # open/closed state while the operator or closure is still nearby.
        # Participant coverage remains dominant; this bounded term is only a
        # temporal tie-breaker against pre-contact detector confidence.
        span_ms = max(1.0, event.global_end_ms - event.global_start_ms)
        progress = min(
            1.0,
            max(0.0, (frame.global_ms - event.global_start_ms) / span_ms),
        )
        key_frame_phase_bias = 4.0 * (1.0 - abs(progress - 0.80))
        score += key_frame_phase_bias
    score += float(receipt.get("instance_selection_score") or 0.0)
    score += min(float(frame.motion_score or 0.0), 100.0) / 100.0
    receipt["key_frame_phase_bias"] = round(key_frame_phase_bias, 6)
    return score, receipt


def _best_event_frames_many(
    path: Path,
    events: Sequence[EvidenceEvent],
) -> dict[str, tuple[FrameEvidence, float, dict[str, Any]] | None]:
    """Select participant-rich frames for all events in one ledger pass."""

    accepted = [event for event in events if event.accepted]
    results: dict[str, tuple[FrameEvidence, float, dict[str, Any]] | None] = {
        event.event_id: None for event in accepted
    }
    # Some callers intentionally materialize media without a persisted
    # detection ledger (for example, compatibility tests and manually supplied
    # events).  In that case retain each event's existing key timestamp; the
    # normal nearest-frame lookup below remains the single fallback source of
    # annotation data.  Production runs always provide immutable ledgers.
    if not path.is_file():
        return results
    for frame in iter_frame_evidence(path):
        if frame.global_ms is None:
            continue
        timestamp = float(frame.global_ms)
        for event in accepted:
            if not event.global_start_ms <= timestamp <= event.global_end_ms:
                continue
            score, receipt = _event_key_frame_score(event, frame)
            previous = results[event.event_id]
            if previous is None or score > previous[1]:
                results[event.event_id] = (frame, score, receipt)
    return results


def _bounded_grounding_dino_temporal_rescue(
    event: EvidenceEvent,
    group: ExperimentGroup,
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Find an explanatory participant frame without repeating a video scan.

    The immutable fine ledgers provide candidate timestamps and actor boxes.
    At most a configured number of real source frames per eligible view are
    decoded in place.  Grounding DINO is then asked only for the missing
    semantic participant classes.  No Ark request or source copy is possible
    in this helper.
    """

    settings = config.get("models", {}).get("open_vocabulary_key_frame") or {}
    fallback = dict(settings.get("grounding_dino_fallback") or {})
    # Temporal rescue runs separately from the final annotation model stack.
    # A small GPU can accelerate this phase without keeping DINO resident
    # alongside YOLO-World and SAM2 during final annotation.
    if fallback.get("temporal_rescue_device"):
        fallback["device"] = fallback["temporal_rescue_device"]
        settings = {**settings, "grounding_dino_fallback": fallback}
    if not (
        settings.get("enabled")
        and fallback.get("enabled")
        and fallback.get("temporal_rescue_enabled")
    ):
        return None, {"status": "disabled"}
    if str((event.model_understanding or {}).get("status") or "") != "completed":
        return None, {"status": "deferred_until_semantic_curation"}

    def normalize(value: Any) -> str:
        return (
            str(value or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
    actor_classes = {"hand", "gloved_hand"}
    prompt_classes = {
        normalize(canonical)
        for canonical in dict(fallback.get("prompt_map") or {}).values()
    }
    participant_classes = {
        normalize(item) for item in event.objects if str(item).strip()
    }
    active_actor_classes = participant_classes & actor_classes or actor_classes
    target_classes = (participant_classes - actor_classes) & prompt_classes
    if not target_classes:
        return None, {
            "status": "not_applicable",
            "participant_classes": sorted(participant_classes),
        }

    by_view = {view.view_id: view for view in views}
    direct_views = {
        str(item)
        for item in (event.semantic_review or {}).get(
            "directly_supported_view_ids", []
        )
    }
    preferred_views = [
        view_id
        for view_id in (group.first_person_view, group.third_person_view)
        if view_id in detection_paths
        and view_id in by_view
        and view_id in infos
        and view_id in transforms
    ]
    direct_preferred = [
        view_id for view_id in preferred_views if view_id in direct_views
    ]
    # Model-declared direct support is recorded, not used as an exclusion.
    # A conservative semantic review can mark a visually clearer role
    # uncertain; bounded geometry must still inspect both real role views.
    eligible_views = preferred_views
    maximum_frames = max(
        1, int(fallback.get("temporal_rescue_max_frames_per_view", 12))
    )
    interval_ms = max(
        1.0, float(fallback.get("temporal_rescue_sample_interval_ms", 500.0))
    )
    maximum_gap = float(
        fallback.get("temporal_rescue_max_actor_gap_norm", 0.02)
    )
    pipette_aspect = float(settings.get("pipette_minimum_aspect_ratio", 1.7))
    dino_pipette_aspect = float(
        fallback.get("pipette_minimum_aspect_ratio", 1.4)
    )
    minimum_class_confidence = {
        normalize(class_name): float(limit)
        for class_name, limit in dict(
            settings.get("minimum_manipulated_object_confidence") or {}
        ).items()
    }
    pipette_center_must_overlap_actor = bool(
        settings.get("pipette_center_must_overlap_actor", False)
    )
    pipette_maximum_actor_iou = float(
        settings.get("pipette_maximum_actor_iou", 0.65)
    )
    closure_maximum_gap = float(
        settings.get("closure_max_actor_gap_norm", 0.01)
    )
    closure_minimum_confidence = float(
        settings.get("closure_minimum_confidence", 0.12)
    )
    grounded_actor_maximum_area = float(
        settings.get("grounded_actor_maximum_box_area_norm", 0.15)
    )
    grounded_actor_confidence_ratio = float(
        settings.get("grounded_actor_confidence_ratio", 0.75)
    )
    preferred_grounding_terms = _preferred_grounding_terms_by_class(event)
    inspected: list[dict[str, Any]] = []
    winners: list[tuple[tuple[float, ...], float, str, dict[str, Any]]] = []
    model_load_seconds = 0.0
    inference_seconds = 0.0

    for view_id in eligible_views:
        candidate_frames = [
            frame
            for frame in iter_frame_evidence(detection_paths[view_id])
            if frame.global_ms is not None
            and event.global_start_ms
            <= float(frame.global_ms)
            <= event.global_end_ms
        ]
        if not candidate_frames:
            inspected.append(
                {"view_id": view_id, "status": "no_fine_ledger_frames"}
            )
            continue
        desired_count = min(
            maximum_frames,
            max(
                2,
                int(
                    math.ceil(
                        max(0.0, event.global_end_ms - event.global_start_ms)
                        / interval_ms
                    )
                )
                + 1,
            ),
            len(candidate_frames),
        )
        if desired_count == 1:
            sampled_frames = [candidate_frames[0]]
        else:
            indices = {
                round(index * (len(candidate_frames) - 1) / (desired_count - 1))
                for index in range(desired_count)
            }
            sampled_frames = [
                candidate_frames[index] for index in sorted(indices)
            ]
        current_frame = min(
            candidate_frames,
            key=lambda item: abs(
                float(item.global_ms) - float(event.key_global_ms)
            ),
        )
        if current_frame not in sampled_frames:
            sampled_frames.append(current_frame)
            sampled_frames.sort(key=lambda item: float(item.global_ms))

        reader = ViewFrameReader(max_open=1)
        try:
            for candidate in sampled_frames:
                global_ms = float(candidate.global_ms)
                dual_role_coverage = all(
                    0.0
                    <= float(transforms[pair_view].to_local(global_ms))
                    <= float(infos[pair_view].duration_ms)
                    for pair_view in preferred_views
                )
                if not dual_role_coverage:
                    inspected.append(
                        {
                            "view_id": view_id,
                            "global_ms": global_ms,
                            "status": "rejected_no_real_dual_role_coverage",
                        }
                    )
                    continue
                local_ms = float(transforms[view_id].to_local(global_ms))
                if not 0.0 <= local_ms <= float(infos[view_id].duration_ms):
                    continue
                frame = reader.read(by_view[view_id], infos[view_id], local_ms)
                if frame is None:
                    inspected.append(
                        {
                            "view_id": view_id,
                            "global_ms": global_ms,
                            "status": "decode_failed",
                        }
                    )
                    continue
                closed_actor_boxes = [
                    box.model_dump()
                    for box in candidate.detections
                    if normalize(box.class_name) in actor_classes
                ]
                requested_classes = set(target_classes)
                if not closed_actor_boxes:
                    requested_classes.update(active_actor_classes)
                grounded, grounding_receipt = (
                    _grounding_dino_key_frame_detections(
                        frame, requested_classes, settings
                    )
                )
                model_load_seconds += float(
                    grounding_receipt.get("model_load_seconds") or 0.0
                )
                inference_seconds += float(
                    grounding_receipt.get("inference_seconds") or 0.0
                )
                raw_grounded_actor_boxes = [
                    box
                    for box in grounded
                    if normalize(box.get("class_name"))
                    in active_actor_classes
                ]
                actor_boxes = closed_actor_boxes or _filter_grounded_actor_boxes(
                    raw_grounded_actor_boxes,
                    maximum_area_norm=grounded_actor_maximum_area,
                    confidence_ratio=grounded_actor_confidence_ratio,
                )
                admitted_objects: list[dict[str, Any]] = []
                selection_receipts: dict[str, Any] = {}
                for canonical_class in sorted(target_classes):
                    class_maximum_gap = (
                        closure_maximum_gap
                        if canonical_class in {"bottle_cap", "tube_cap"}
                        else maximum_gap
                    )
                    class_minimum_confidence = max(
                        float(
                            minimum_class_confidence.get(
                                canonical_class, 0.0
                            )
                        ),
                        (
                            closure_minimum_confidence
                            if canonical_class in {"bottle_cap", "tube_cap"}
                            else 0.0
                        ),
                    )
                    selected, selection_receipt = (
                        _select_manipulated_object_candidate(
                            [
                                box
                                for box in grounded
                                if normalize(box.get("class_name"))
                                == canonical_class
                            ],
                            actor_boxes,
                            canonical_class=canonical_class,
                            maximum_actor_gap=class_maximum_gap,
                            pipette_minimum_aspect_ratio=pipette_aspect,
                            grounding_dino_pipette_minimum_aspect_ratio=(
                                dino_pipette_aspect
                            ),
                            minimum_confidence=class_minimum_confidence,
                            pipette_center_must_overlap_actor=(
                                pipette_center_must_overlap_actor
                            ),
                            pipette_maximum_actor_iou=(
                                pipette_maximum_actor_iou
                            ),
                            preferred_grounding_terms=(
                                preferred_grounding_terms.get(
                                    canonical_class, ()
                                )
                            ),
                        )
                    )
                    selection_receipts[canonical_class] = selection_receipt
                    if selected is not None:
                        admitted_objects.append(selected)
                rendered, render_receipt = _event_participant_boxes(
                    event,
                    [*actor_boxes, *admitted_objects],
                    view_id=view_id,
                    maximum_interaction_gap_norm=maximum_gap,
                )
                rendered_actor_boxes = [
                    box
                    for box in rendered
                    if normalize(box.get("class_name")) in actor_classes
                ]
                rendered_object_boxes = [
                    box
                    for box in rendered
                    if normalize(box.get("class_name")) not in actor_classes
                ]
                selected_object_classes = {
                    normalize(box.get("class_name"))
                    for box in rendered_object_boxes
                }
                pair_visible = bool(
                    rendered_actor_boxes
                    and target_classes.issubset(selected_object_classes)
                )
                best_gap = (
                    min(
                        _box_edge_gap_norm(obj, actor)
                        for obj in rendered_object_boxes
                        for actor in rendered_actor_boxes
                    )
                    if pair_visible
                    else None
                )
                best_confidence = max(
                    (
                        float(box.get("confidence") or 0.0)
                        for box in rendered_object_boxes
                    ),
                    default=0.0,
                )
                record = {
                    "view_id": view_id,
                    "global_ms": global_ms,
                    "status": "eligible" if pair_visible else "rejected",
                    "rendered_classes": render_receipt.get("rendered_classes"),
                    "selected_object_classes": sorted(selected_object_classes),
                    "best_object_confidence": round(best_confidence, 6),
                    "best_actor_gap_norm": (
                        round(float(best_gap), 6)
                        if best_gap is not None
                        else None
                    ),
                    "grounding": {
                        key: value
                        for key, value in grounding_receipt.items()
                        if key
                        in {
                            "status",
                            "raw_detection_count",
                            "admitted_area_bounded_count",
                            "rejected_box_area_count",
                        }
                    },
                    "grounded_actor_filter": {
                        "raw_candidate_count": len(raw_grounded_actor_boxes),
                        "admitted_candidate_count": (
                            len(actor_boxes)
                            if not closed_actor_boxes
                            else 0
                        ),
                        "closed_set_actor_used": bool(closed_actor_boxes),
                        "maximum_box_area_norm": grounded_actor_maximum_area,
                        "confidence_ratio": grounded_actor_confidence_ratio,
                    },
                    "selection": selection_receipts,
                }
                inspected.append(record)
                if pair_visible:
                    temporal_phase_score = 0.0
                    if event.action_type in {
                        ActionType.CONTAINER_STATE_CHANGE,
                        ActionType.HAND_OBJECT_CONTACT,
                    }:
                        span_ms = max(
                            1.0,
                            event.global_end_ms - event.global_start_ms,
                        )
                        progress = min(
                            1.0,
                            max(
                                0.0,
                                (global_ms - event.global_start_ms) / span_ms,
                            ),
                        )
                        target_progress = (
                            0.80
                            if event.action_type
                            == ActionType.CONTAINER_STATE_CHANGE
                            else 0.50
                        )
                        temporal_phase_score = 1.0 - abs(
                            progress - target_progress
                        )
                        record["temporal_phase_score"] = round(
                            temporal_phase_score, 6
                        )
                    score = (
                        float(len(record["selected_object_classes"])),
                        temporal_phase_score,
                        -float(best_gap or 0.0),
                        best_confidence,
                        float(candidate.motion_score or 0.0),
                        -abs(global_ms - float(event.key_global_ms)),
                    )
                    winners.append((score, global_ms, view_id, record))
        finally:
            reader.close()

    winner = max(winners, key=lambda item: item[0], default=None)
    return (
        float(winner[1]) if winner is not None else None,
        {
            "schema_version": (
                "visioncortex-grounding-dino-temporal-rescue/1"
            ),
            "status": "selected" if winner is not None else "no_eligible_frame",
            "scope": "bounded immutable-ledger candidate frames",
            "full_scan_repeated": False,
            "source_copy_bytes": 0,
            "ark_calls": 0,
            "token_usage": 0,
            "target_classes": sorted(target_classes),
            "eligible_views": eligible_views,
            "direct_support_views": direct_preferred,
            "direct_support_used_as_exclusion": False,
            "sample_interval_ms": interval_ms,
            "maximum_frames_per_view": maximum_frames,
            "inspected_frame_count": len(inspected),
            "eligible_frame_count": len(winners),
            "model_load_seconds": round(model_load_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "selected_view_id": winner[2] if winner is not None else None,
            "selected_global_ms": winner[1] if winner is not None else None,
            "selected_receipt": winner[3] if winner is not None else None,
            "inspected": inspected,
        },
    )


RELABEL_OBJECT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tube_cap", (r"管盖", r"离心管盖", r"tube[_ -]?cap")),
    (
        "bottle_cap",
        (
            r"瓶盖",
            r"红盖",
            r"红色小盖(?:容器)?",
            r"bottle[_ -]?cap",
            r"(?:blue|red)[_ -]?cap",
        ),
    ),
    (
        "paper",
        (
            r"称量纸",
            r"白色纸片",
            r"纸片",
            r"纸张",
            r"持纸",
            r"^纸$",
            r"weighing[_ -]?paper",
        ),
    ),
    ("spatula", (r"药匙", r"药勺", r"勺状(?:金属)?工具", r"spatula")),
    ("pipette", (r"移液器", r"移液枪", r"pipette")),
    ("spearhead", (r"枪头", r"吸头", r"pipette[_ -]?tip", r"spearhead")),
    (
        "reagent_bottle",
        (
            r"试剂瓶",
            r"棕色瓶",
            r"透明(?:玻璃)?(?:试剂)?瓶",
            r"reagent[_ -]?bottle",
            r"(?:brown|amber)[_ -]?(?:reagent[_ -]?)?bottle",
            r"(?:clear|transparent|glass)[_ -]?(?:reagent[_ -]?)?bottle",
        ),
    ),
    ("sample_bottle", (r"样品瓶", r"sample[_ -]?bottle")),
    ("tube_rack", (r"离心管架", r"试管架", r"tube[_ -]?rack")),
    (
        "tube",
        (
            r"离心管",
            r"试管",
            r"\btube\b",
            r"(?:centrifuge[_ -]?)?tube(?:[_ /-]|$)",
        ),
    ),
    ("balance", (r"天平", r"balance")),
    ("beaker", (r"烧杯", r"beaker")),
    ("magnetic_stirrer", (r"磁力搅拌", r"magnetic[_ -]?stirrer")),
    ("container", (r"容器", r"container")),
)


def _semantic_interaction_is_direct(item: dict[str, Any]) -> bool:
    """Accept only structured interactions that assert physical contact."""

    contact = str(item.get("contact") or "").strip().lower()
    if not contact:
        return False
    negative_terms = (
        "未",
        "接近",
        "靠近",
        "不清晰",
        "无法确认",
        "no contact",
        "not clear",
        "unclear",
        "near",
        "approach",
    )
    if any(term in contact for term in negative_terms):
        return False
    positive_terms = (
        "接触",
        "抓取",
        "握持",
        "按压",
        "触碰",
        "拿取",
        "操作",
        "contact",
        "touch",
        "grasp",
        "grip",
        "hold",
        "press",
    )
    return any(term in contact for term in positive_terms)


def _participant_class_mentions(text: str) -> list[tuple[str, int, int]]:
    """Recognize canonical structured labels as well as natural-language aliases."""

    mentions: list[tuple[str, int, int]] = []
    for class_name, patterns in RELABEL_OBJECT_PATTERNS:
        canonical_pattern = rf"(?<!\w){re.escape(class_name)}(?!\w)"
        for pattern in (canonical_pattern, *patterns):
            mentions.extend(
                (class_name, match.start(), match.end())
                for match in re.finditer(pattern, text.lower())
            )
    return mentions


def _semantic_participant_conflicts(event: EvidenceEvent) -> list[dict[str, str]]:
    """Record explicit background-only claims contradicting a contact record.

    Use the model's aggregate action proof, not one occluded view or an absent
    device state change. The result marks an internal contradiction; it does
    not establish that physical contact never occurred in the source video.
    """

    understanding = event.model_understanding or {}
    direct_classes = {
        class_name
        for item in understanding.get("hand_object_interactions") or []
        if isinstance(item, dict) and _semantic_interaction_is_direct(item)
        for class_name, _start, _end in _participant_class_mentions(
            str(item.get("object") or "")
        )
    }
    reason = str((understanding.get("action_proof") or {}).get("reason") or "")
    conflicts: list[dict[str, str]] = []
    for clause in re.split(r"[，,。；;.!?\n]", reason):
        background = re.search(
            r"(?:仅|只)(?:作为|是|在|出现在)?[^，,。；;\n]{0,6}背景"
            r"|\bonly\s+(?:(?:as|in)\s+(?:a\s+|the\s+)?)?background\b",
            clause.lower(),
        )
        if background is None:
            continue
        prefix = clause[:background.start()].lower()
        if re.search(r"(?:并非|不是|不仅|不只是|不|not)\s*$", prefix):
            continue
        mentions = [
            (class_name, end)
            for class_name, _start, end in _participant_class_mentions(prefix)
            if len(prefix) - end <= 24
        ]
        if not mentions:
            continue
        nearest_end = max(end for _class_name, end in mentions)
        for class_name in sorted({name for name, end in mentions if end == nearest_end}):
            if class_name in direct_classes:
                conflicts.append({
                    "class_name": class_name,
                    "source": "action_proof.reason",
                    "statement": clause.strip(),
                    "reason": "direct_contact_and_background_only_claims_conflict",
                })
    return conflicts


def _interaction_participant_classes(text: str) -> list[str]:
    classes = list(dict.fromkeys(
        name for name, _start, _end in _participant_class_mentions(text.lower())
    ))
    # A cap used to describe a bottle is not a separately manipulated cap.
    # Match a whole, single noun phrase; explicit lists retain both objects.
    capped_bottle = bool(re.fullmatch(
        r"(?:带(?:有)?|装有|配有|盖有)?[^/，,。；;和与及、]{0,12}"
        r"(?:瓶盖|盖子|盖)的(?:棕色|玻璃|塑料|透明)?"
        r"(?:试剂瓶|样品瓶|棕色瓶子|瓶子|瓶)", text.strip()
    ))
    if capped_bottle and set(classes) & {"reagent_bottle", "sample_bottle", "sample_bottle_blue", "container"}:
        classes = [name for name in classes if name != "bottle_cap"]
    return classes


def _relabel_participant_objects(event: EvidenceEvent) -> list[str]:
    """Map model-described interaction participants back to detector classes."""

    understanding = event.model_understanding or {}
    structured_interactions = [
        item
        for item in (understanding.get("hand_object_interactions") or [])
        if isinstance(item, dict) and str(item.get("object") or "").strip()
    ]
    direct_interactions = [
        item for item in structured_interactions if _semantic_interaction_is_direct(item)
    ]
    interaction_objects = [
        str(item.get("object") or "") for item in direct_interactions
    ]
    primary_text = " ".join(interaction_objects).lower()
    fallback_text = " ".join(
        str(understanding.get(key) or "")
        for key in ("current_step",)
    ).lower()
    interaction_matches = list(dict.fromkeys(
        name for text in interaction_objects for name in _interaction_participant_classes(text)
    ))
    selected_observations = [
        item
        for item in understanding.get("selected_keyframe_observations") or []
        if isinstance(item, dict)
    ]
    selected_interaction_labels = [
        str(label)
        for item in selected_observations
        for label in item.get("directly_interacting_objects") or []
        if str(label).strip()
    ]
    selected_interaction_matches = list(dict.fromkeys(
        name
        for text in selected_interaction_labels
        for name in _interaction_participant_classes(text)
        if name not in {"hand", "gloved_hand"}
    ))
    if event.action_type == ActionType.DEVICE_PANEL_OPERATION:
        selected_interaction_matches = [
            item for item in selected_interaction_matches if item in DEVICE_CLASSES
        ]
    elif event.action_type == ActionType.PIPETTE_TRANSFER_OPERATION:
        selected_interaction_matches = [
            item
            for item in selected_interaction_matches
            if item in {"pipette", "spearhead"}
        ]
    step_matches = list(dict.fromkeys(
        name for name, _start, _end in _participant_class_mentions(fallback_text)
    ))
    # Structured hand-object interactions are already participant-scoped.
    # Do not drop one explicit participant merely because the free-text step
    # repeats another participant but omits this one's class name.
    matched = interaction_matches if structured_interactions else step_matches
    if selected_observations:
        # Final annotations explain the selected key frame, so the explicit
        # selected-frame interaction list outranks objects touched elsewhere
        # in a long event window.  An unmapped selected-frame object remains
        # empty and is quarantined downstream; a convenient mapped background
        # object must never substitute for it.
        matched = selected_interaction_matches
    # Ark may use a deliberately generic structured label such as
    # ``red small container`` while the participant-scoped current-step text
    # identifies the same object as a cap.  Refine only a generic container
    # through specific container/closure aliases from that current-step text;
    # never pull tools or unrelated background inventory into the participant
    # list.  This keeps the final detector from boxing a nearby bottle or rack
    # merely because the semantic participant was left as ``container``.
    if structured_interactions and "container" in matched:
        container_refinement_classes = {
            "tube_cap",
            "bottle_cap",
            "reagent_bottle",
            "sample_bottle",
            "tube",
            "beaker",
        }
        refinements = [
            item for item in step_matches if item in container_refinement_classes
        ]
        if refinements:
            matched = [item for item in matched if item != "container"]
            matched = list(dict.fromkeys([*matched, *refinements]))
    proof = understanding.get("action_proof") or {}
    if (
        event.action_type == ActionType.OBJECT_MOVEMENT
        and proof.get("source_contact_visible")
        and proof.get("withdrawal_or_transport_visible")
        and proof.get("target_contact_visible")
    ):
        # Long solid-transfer windows can mention incidental cap or package
        # touches. Keep only classes named by the proof of the confirmed
        # source-to-target movement. If the proof cannot be mapped, retain the
        # structured interactions and fail closed downstream.
        proof_classes = _interaction_participant_classes(
            str(proof.get("reason") or "")
        )
        if proof_classes:
            matched = [item for item in matched if item in proof_classes]
        paper_interactions = [
            text
            for text in interaction_objects
            if "paper" in _interaction_participant_classes(text)
        ]
        package_pattern = r"(?:package|packaging|packet|wrapper|包装|纸包)"
        if (
            "paper" in matched
            and paper_interactions
            and all(re.search(package_pattern, text.lower()) for text in paper_interactions)
            and not re.search(package_pattern, str(proof.get("reason") or "").lower())
        ):
            # A transfer window may contain a direct touch of the weighing-
            # paper package while the confirmed movement ends on a separate
            # sheet. Both phrases map to ``paper`` but they are different
            # physical instances. The receiving sheet is not a hand-object
            # participant unless a direct interaction names the sheet itself.
            matched = [item for item in matched if item != "paper"]
    conflicting_classes = {
        item["class_name"] for item in _semantic_participant_conflicts(event)
    }
    matched = [item for item in matched if item not in conflicting_classes]
    actor_classes = [
        item
        for item in event.objects
        if str(item).strip().lower().replace("-", "_") in {"hand", "gloved_hand"}
    ]
    # ``hand`` is the generic superclass of ``gloved_hand``. Keeping both as
    # separate semantic slots makes open-vocabulary grounding request a bare
    # hand even when Ark explicitly observed blue gloves; a weak background
    # false positive can then suppress the real hand-held tool. Prefer the
    # more specific gloved actor whenever it is present.
    if "gloved_hand" in actor_classes:
        actor_classes = ["gloved_hand"]
    if not actor_classes and interaction_objects:
        actor_text = " ".join(
            [primary_text, fallback_text, " ".join(map(str, understanding.get("objects") or []))]
        ).lower()
        actor_classes = [
            "gloved_hand"
            if re.search(r"手套|glov(?:e|ed)", actor_text)
            else "hand"
        ]
    return list(dict.fromkeys([*actor_classes, *matched]))


def _view_specific_participant_objects(
    event: EvidenceEvent, view_id: str
) -> list[str]:
    """Constrain final boxes to objects Ark actually described in one view."""

    def normalize(value: Any) -> str:
        return (
            str(value or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
    actor_objects = [
        item
        for item in event.objects
        if normalize(item) in {"hand", "gloved_hand"}
    ]
    non_actor_objects = [
        item
        for item in event.objects
        if normalize(item) not in {"hand", "gloved_hand"}
    ]
    understanding = event.model_understanding or {}
    observation = next(
        (
            str(item.get("observation") or "")
            for item in understanding.get("per_view_observations") or []
            if isinstance(item, dict)
            and str(item.get("view_id") or "") == str(view_id)
        ),
        "",
    )
    support = next(
        (
            item
            for item in understanding.get("confirmed_action_support_by_view")
            or []
            if isinstance(item, dict)
            and str(item.get("view_id") or "") == str(view_id)
        ),
        None,
    )
    confirmed_support = [
        item
        for item in understanding.get("confirmed_action_support_by_view")
        or []
        if isinstance(item, dict)
    ]
    if support is not None and support.get("supports_confirmed_action") is False:
        return list(dict.fromkeys(actor_objects))
    if support is None and confirmed_support:
        # A populated per-view support ledger is authoritative.  If this view
        # has no entry, do not propagate objects proved only by another view.
        return list(dict.fromkeys(actor_objects))
    support_reason = str((support or {}).get("reason") or "")
    # Prefer the participant-scoped support reason.  Full observations often
    # contain explicit negative statements (for example "未看到接触移液器");
    # naïve keyword matching would turn that absence into a rendered object.
    scoped_text = (support_reason or observation).lower()
    pattern_map = dict(RELABEL_OBJECT_PATTERNS)
    canonical_alias = {
        "weighing_paper": "paper",
        "sample_bottle_blue": "sample_bottle",
    }
    matched_non_actors: list[str] = []
    for item in non_actor_objects:
        canonical = canonical_alias.get(normalize(item), normalize(item))
        patterns = pattern_map.get(canonical, ())
        if (
            re.search(rf"(?<![a-z0-9_]){re.escape(canonical)}(?![a-z0-9_])", scoped_text)
            or any(re.search(pattern, scoped_text) for pattern in patterns)
        ):
            matched_non_actors.append(item)
    if matched_non_actors:
        return list(dict.fromkeys([*actor_objects, *matched_non_actors]))
    if (
        observation
        and str(
            (event.semantic_review or {}).get("cross_view_consistency")
            or understanding.get("cross_view_consistency")
            or ""
        )
        == "conflict"
    ):
        # The view explicitly depicts another concurrent object. Do not carry
        # the primary view's class into it merely to fill an annotation slot.
        return list(dict.fromkeys(actor_objects))
    return list(event.objects)


def _rerender_curated_participant_annotations(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render final boxes after semantic relabeling corrected participants."""

    verification_started = time.perf_counter()
    verification_settings = dict(
        ((config or {}).get("key_materials") or {}).get(
            "selective_verification"
        )
        or {}
    )
    verification_budget = SelectiveVerificationBudget.from_settings(
        verification_settings
    )
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    records: list[dict[str, Any]] = []
    segmentation_records: list[dict[str, Any]] = []
    liquid_semantic_records: list[dict[str, Any]] = []
    visual_reviewer = (
        ParticipantVisualReviewer(
            config, layout.work, layout.json_config,
            _grounding_dino_key_frame_detections,
        )
        if config is not None and config.get("key_materials", {}).get("participant_visual_review", {}).get("enabled")
        else None
    )
    for event in events:
        input_root = layout.work / "key-material-annotation-inputs" / event.event_id
        if not input_root.is_dir():
            continue
        group = group_by_event[event.event_id]
        first_material_view, third_material_view = _key_material_view_pair(
            group, event
        )
        visual_plan: dict[str, dict[str, Any]] = {}
        visual_receipts: list[dict[str, Any]] = []
        review_classes = visual_reviewer.eligible_classes(event) if visual_reviewer is not None else []
        if review_classes:
            if config.get("performance", {}).get("release_auxiliary_models_between_stages"):
                _release_auxiliary_model_caches()
            for participant_class in review_classes:
                visual_views = []
                for role_label, view_id in (("First-Person", first_material_view), ("Third-Person", third_material_view)):
                    if participant_class not in _view_specific_participant_objects(event, view_id):
                        continue
                    visual_views.append({
                        "view_id": view_id, "role_label": role_label,
                        "raw_path": input_root / f"{role_label}.jpg",
                        "detections": json.loads((input_root / f"{role_label}.json").read_text())["detections"],
                    })
                if not visual_views:
                    continue
                try:
                    visual_receipt, class_plan = visual_reviewer.review(
                        event, visual_views, participant_class=participant_class
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    if not message.startswith((
                        "Participant visual review ",
                        "Invalid cached participant review",
                    )):
                        raise
                    review_records = [
                        row
                        for row in (
                            event.observability.get("participant_visual_review")
                            or {}
                        ).values()
                        if isinstance(row, dict)
                        and row.get("participant_class") == participant_class
                    ]
                    if not review_records:
                        raise
                    visual_receipt = review_records[-1]
                    fingerprint = str(
                        visual_receipt.get("input_fingerprint") or "unrecorded"
                    )
                    class_plan = {}
                    for view in visual_views:
                        actors = [
                            dict(box)
                            for box in view["detections"]
                            if str(box.get("class_name") or "")
                            .strip()
                            .lower()
                            .replace("-", "_")
                            .replace(" ", "_")
                            in {"hand", "gloved_hand"}
                        ]
                        class_plan[view["view_id"]] = {
                            "boxes": [],
                            "actors": actors,
                            "selected_candidate_ids": [],
                            "target_visible": False,
                            "reason": (
                                "参与对象视觉复核未产生可验证选择；该对象从已确认"
                                "画面中移除并交由待复核素材处理。"
                            ),
                            "input_fingerprint": fingerprint,
                            "review_status": visual_receipt.get("status"),
                            "failure_reason": message,
                        }
                    visual_receipt = {
                        **visual_receipt,
                        "input_fingerprint": fingerprint,
                        "status": "review_failed_quarantined",
                        "source_status": visual_receipt.get("status"),
                        "failure_reason": message,
                    }
                    note = (
                        f"{participant_class} 参与对象视觉复核未完成；"
                        "相关候选已从已确认素材中移除并保留待复核记录。"
                    )
                    if note not in event.uncertainty:
                        event.uncertainty.append(note)
                visual_receipts.append({
                    "input_fingerprint": visual_receipt["input_fingerprint"],
                    "status": visual_receipt["status"],
                    "participant_class": participant_class,
                    "localized_view_count": sum(bool(item["boxes"]) for item in class_plan.values()),
                    **(
                        {
                            "source_status": visual_receipt.get("source_status"),
                            "failure_reason": visual_receipt.get("failure_reason"),
                        }
                        if visual_receipt["status"]
                        == "review_failed_quarantined"
                        else {}
                    ),
                })
                for view_id, selected in class_plan.items():
                    plan = visual_plan.setdefault(view_id, {
                        "boxes": [], "actors": selected["actors"],
                        "input_fingerprint": selected["input_fingerprint"],
                        "input_fingerprints": [], "reviewed_classes": [], "reviews": [],
                    })
                    plan["boxes"].extend(selected["boxes"])
                    plan["input_fingerprints"].append(selected["input_fingerprint"])
                    plan["reviewed_classes"].append(participant_class)
                    plan["reviews"].append({
                        "participant_class": participant_class,
                        **{key: value for key, value in selected.items() if key not in {"boxes", "actors"}},
                    })
        rendered_frames: dict[str, Path] = {}
        annotation = {
            "schema_version": "visioncortex-key-material-annotation/1",
            "mode": "event_participants_only",
            "render_pass": "post_semantic_curation",
            "views": {},
        }
        if visual_receipts:
            failed_review_count = sum(
                item["status"] == "review_failed_quarantined"
                for item in visual_receipts
            )
            annotation["participant_visual_review"] = {
                "input_fingerprint": visual_receipts[0]["input_fingerprint"],
                "status": (
                    "completed_with_quarantined_review_failures"
                    if failed_review_count
                    else "completed"
                ),
                "participant_class": visual_receipts[0]["participant_class"] if len(visual_receipts) == 1 else "multiple",
                "reviews": visual_receipts,
                "localized_view_count": sum(bool(item["boxes"]) for item in visual_plan.values()),
                "quarantined_review_failure_count": failed_review_count,
            }
        for role_label, view_id in (
            ("First-Person", first_material_view),
            ("Third-Person", third_material_view),
        ):
            raw_path = input_root / f"{role_label}.jpg"
            detection_path = input_root / f"{role_label}.json"
            if not raw_path.is_file() or not detection_path.is_file():
                raise RuntimeError(
                    f"Final annotation input is incomplete: {event.event_id}/{role_label}"
                )
            raw_frame = cv2.imread(str(raw_path))
            if raw_frame is None:
                raise RuntimeError(f"Final annotation raw frame is unreadable: {raw_path}")
            detected_boxes = json.loads(detection_path.read_text(encoding="utf-8"))[
                "detections"
            ]
            view_event = event.model_copy(deep=True)
            view_event.objects = _view_specific_participant_objects(
                event, view_id
            )
            maximum_interaction_gap_norm = float(
                (
                    (config or {})
                    .get("models", {})
                    .get("open_vocabulary_key_frame", {})
                    .get("manipulated_object_max_actor_gap_norm", 0.08)
                )
            )
            _, closed_set_participant_receipt = _event_participant_boxes(
                view_event,
                detected_boxes,
                view_id=view_id,
                maximum_interaction_gap_norm=maximum_interaction_gap_norm,
            )
            verification_decision = plan_selective_key_material_verification(
                view_event,
                view_id,
                detected_boxes,
                closed_set_participant_receipt,
                verification_settings,
                verification_budget,
            )
            supplement_receipt: dict[str, Any] | None = None
            phase_isolation = bool(
                (config or {}).get("performance", {}).get("release_auxiliary_models_between_stages")
            )
            if phase_isolation:
                _release_auxiliary_model_caches()
            if view_id in visual_plan and set(view_event.objects) <= {"hand", "gloved_hand", *visual_plan[view_id]["reviewed_classes"]}:
                supplement_receipt = {
                    "status": "replaced_by_visual_candidate_review",
                    "input_fingerprint": visual_plan[view_id]["input_fingerprint"],
                    "input_fingerprints": visual_plan[view_id]["input_fingerprints"],
                    "purpose": "Use the reviewed DINO proposal without redundant YOLO-World inference",
                }
            elif config is not None and verification_decision["should_run"]:
                supplement_boxes, supplement_receipt = (
                    _open_vocabulary_key_frame_supplement(
                        raw_frame,
                        view_event,
                        detected_boxes,
                        config,
                        view_role=role_label,
                    )
                )
                if supplement_receipt.get("status") == "executed":
                    # The supplement is a fail-closed replacement for ambiguous
                    # closed-set instances, not an additive source. Remove the
                    # named classes even when grounding finds no eligible
                    # hand-adjacent object; the visibility gate must then fail
                    # instead of silently drawing a background rack instance.
                    replaced_classes = set(
                        supplement_receipt.get("closed_set_replaced_classes")
                        or []
                    )
                    detected_boxes = [
                        box
                        for box in detected_boxes
                        if str(box.get("class_name") or "")
                        .strip()
                        .lower()
                        .replace("-", "_")
                        .replace(" ", "_")
                        not in replaced_classes
                    ]
                    detected_boxes = [*detected_boxes, *supplement_boxes]
            elif config is not None:
                supplement_receipt = {
                    "schema_version": "visioncortex-open-vocabulary-key-frame/1",
                    "status": "skipped_by_selective_verification",
                    "scope": "final accepted key frames only",
                    "full_timeline_inference": False,
                    "source_copy_bytes": 0,
                    "token_usage": 0,
                    "ark_calls": 0,
                }
            if view_id in visual_plan:
                selected = visual_plan[view_id]
                # A validated empty selection also replaces the old target box;
                # SAM2 must never propagate a rejected background instance.
                detected_boxes = [box for box in detected_boxes if box.get("class_name") not in selected["reviewed_classes"]]
                detected_boxes.extend(selected["boxes"])
                if "bottle_cap" in selected["reviewed_classes"] or not any(box.get("class_name") in {"hand", "gloved_hand"} for box in detected_boxes):
                    detected_boxes.extend(actor for actor in selected["actors"] if actor not in detected_boxes)
            boxes, receipt = _event_participant_boxes(
                view_event,
                detected_boxes,
                view_id=view_id,
                maximum_interaction_gap_norm=maximum_interaction_gap_norm,
            )
            receipt["selective_verification"] = verification_decision
            if view_id in visual_plan:
                receipt["participant_visual_review"] = {
                    key: value for key, value in visual_plan[view_id].items()
                    if key not in {"boxes", "actors"}
                }
            if supplement_receipt is not None:
                receipt["open_vocabulary_supplement"] = supplement_receipt
            if verification_decision["status"] == "deferred_budget_exhausted":
                deferred_note = (
                    "关键素材本地二次复核预算已用尽；保留闭集检测证据，"
                    f"未执行开放词汇补全（{verification_decision['reason']}）"
                )
                if deferred_note not in event.uncertainty:
                    event.uncertainty.append(deferred_note)
            if phase_isolation:
                _release_auxiliary_model_caches()
            segmentation_receipt: dict[str, Any] | None = None
            if config is not None:
                segmentation_settings = (
                    (config.get("models") or {}).get(
                        "temporal_participant_segmentation"
                    )
                    or {}
                )
                if segmentation_settings.get("enabled"):
                    relative_clip = event.key_clips.get(view_id)
                    if not relative_clip:
                        if segmentation_settings.get(
                            "required_for_final_key_material", False
                        ):
                            raise RuntimeError(
                                "SAM2 final participant audit requires a bounded "
                                f"key clip: {event.event_id}/{view_id}"
                            )
                        segmentation_receipt = {
                            "status": "not_available_no_key_clip"
                        }
                    else:
                        clip_pre_ms = float(
                            config["segmentation"]["key_clip_pre_seconds"]
                        ) * 1000.0
                        clip_post_ms = float(
                            config["segmentation"]["key_clip_post_seconds"]
                        ) * 1000.0
                        clip_start_ms = max(
                            0.0, event.global_start_ms - clip_pre_ms
                        )
                        clip_end_ms = event.global_end_ms + clip_post_ms
                        seed_fraction = (
                            (event.key_global_ms - clip_start_ms)
                            / max(1.0, clip_end_ms - clip_start_ms)
                        )
                        boxes, segmentation_receipt = (
                            audit_participant_continuity(
                                layout.root / relative_clip,
                                raw_frame,
                                boxes,
                                layout.work
                                / "sam2-participant-continuity",
                                config,
                                event_id=event.event_id,
                                view_id=view_id,
                                action_type=event.action_type.value,
                                seed_fraction=seed_fraction,
                            )
                        )
                    receipt["temporal_participant_segmentation"] = (
                        segmentation_receipt
                    )
                    segmentation_records.append(
                        {
                            "event_id": event.event_id,
                            "view_id": view_id,
                            "role_label": role_label,
                            **dict(segmentation_receipt or {}),
                        }
                    )
                liquid_settings = (
                    (config.get("models") or {}).get("liquid_semantic_sidecar")
                    or {}
                )
                enabled_actions = {
                    str(item) for item in liquid_settings.get("enabled_actions") or []
                }
                if liquid_settings.get("enabled") and (
                    not enabled_actions or event.action_type.value in enabled_actions
                ):
                    liquid_output = (
                        layout.key_materials
                        / "Liquid-State-Observations"
                        / str(group.archive_folder or group.group_id)
                        / key_material_action_folder(event.action_type)
                        / event.event_id
                    )
                    liquid_receipt = analyze_liquid_semantics(
                        raw_frame,
                        config,
                        liquid_output,
                        artifact_stem=view_id,
                    )
                    for path_key in ("overlay_path", "mask_path"):
                        liquid_receipt[path_key] = archive_relative_posix(
                            Path(str(liquid_receipt[path_key])), layout.root
                        )
                    receipt["liquid_semantic_sidecar"] = liquid_receipt
                    event.observability.setdefault(
                        "liquid_semantic_sidecar", {}
                    )[view_id] = liquid_receipt
                    liquid_semantic_records.append(
                        {
                            "event_id": event.event_id,
                            "view_id": view_id,
                            "role_label": role_label,
                            **liquid_receipt,
                        }
                    )
            destination = layout.root / event.key_frames[view_id]
            write_annotated_frame(raw_frame, boxes, destination)
            receipt["rendered_classes"] = sorted({box["class_name"] for box in boxes})
            receipt["rendered_box_count"] = len(boxes)
            rendered_frames[role_label] = destination
            annotation["views"][view_id] = receipt
            records.append(
                {
                    "event_id": event.event_id,
                    "view_id": view_id,
                    "role_label": role_label,
                    "participant_objects": list(view_event.objects),
                    "event_participant_objects": list(event.objects),
                    **receipt,
                }
            )
        aligned = layout.root / event.key_frames["aligned_first_third"]
        _write_aligned_frame(
            rendered_frames["First-Person"],
            rendered_frames["Third-Person"],
            aligned,
            (first_material_view, third_material_view),
        )
        if visual_receipts:
            for review in visual_receipts:
                review["reviewed_localized_view_count"] = review["localized_view_count"]
                review["localized_view_count"] = sum(
                    review["participant_class"] in item.get("rendered_classes", [])
                    for item in annotation["views"].values()
                )
            annotation["participant_visual_review"]["localized_view_count"] = sum(
                bool(set(item.get("rendered_classes", [])) & set(review_classes))
                for item in annotation["views"].values()
            )
        event.observability["key_material_annotation"] = annotation
        if bool(
            ((config or {}).get("performance") or {}).get(
                "release_auxiliary_models_after_event", False
            )
        ):
            _release_auxiliary_model_caches()
    decisions = [
        record["selective_verification"]
        for record in records
        if isinstance(record.get("selective_verification"), dict)
    ]
    decision_status_counts: dict[str, int] = {}
    for decision in decisions:
        status = str(decision.get("status") or "unknown")
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1
    supplements = [
        record["open_vocabulary_supplement"]
        for record in records
        if isinstance(record.get("open_vocabulary_supplement"), dict)
    ]
    grounding_receipts = [
        supplement["grounding_dino_fallback"]
        for supplement in supplements
        if isinstance(supplement.get("grounding_dino_fallback"), dict)
    ]
    verification_summary = {
        "schema_version": "visioncortex-selective-key-material-verification-index/1",
        "enabled": bool(verification_settings.get("enabled", False)),
        "mode": str(
            verification_settings.get("mode") or "ambiguous_or_high_risk"
        ),
        "policy": (
            "closed-set evidence for every accepted event; expensive local models "
            "only for high-risk or ambiguous final role frames"
        ),
        "decision_count": len(decisions),
        "decision_status_counts": dict(sorted(decision_status_counts.items())),
        "open_vocabulary_executed_count": sum(
            item.get("status") == "executed" for item in supplements
        ),
        "grounding_dino_executed_count": sum(
            item.get("status") == "executed" for item in grounding_receipts
        ),
        "open_vocabulary_model_load_seconds": round(
            sum(float(item.get("model_load_seconds") or 0.0) for item in supplements),
            6,
        ),
        "open_vocabulary_inference_seconds": round(
            sum(float(item.get("inference_seconds") or 0.0) for item in supplements),
            6,
        ),
        "grounding_dino_model_load_seconds": round(
            sum(
                float(item.get("model_load_seconds") or 0.0)
                for item in grounding_receipts
            ),
            6,
        ),
        "grounding_dino_inference_seconds": round(
            sum(
                float(item.get("inference_seconds") or 0.0)
                for item in grounding_receipts
            ),
            6,
        ),
        "wall_seconds": round(time.perf_counter() - verification_started, 6),
        "budget": verification_budget.summary(),
        "full_timeline_inference": False,
        "source_copy_bytes": 0,
        "ark_calls": 0,
        "token_usage": 0,
    }
    report = {
        "schema_version": "visioncortex-final-key-material-annotation/1",
        "mode": "event_participants_only",
        "render_pass": "post_semantic_curation",
        "event_count": len(events),
        "rendered_view_count": len(records),
        "selective_verification": verification_summary,
        "participant_visual_review": {
            "enabled": bool(visual_reviewer is not None and visual_reviewer.enabled),
            "requests": visual_reviewer.records if visual_reviewer is not None else {},
            "purpose": "participant localization only; not action or quality certification",
        },
        "records": records,
    }
    write_json(layout.json_config / "final_key_material_annotation.json", report)
    write_json(
        layout.json_config / "temporal_participant_segmentation.json",
        {
            "schema_version": "visioncortex-sam2-participant-continuity-index/1",
            "policy": (
                "bounded final key clips only; continuity and box refinement, "
                "never action confirmation"
            ),
            "record_count": len(segmentation_records),
            "passed_count": sum(
                item.get("status") == "completed" and item.get("passed") is True
                for item in segmentation_records
            ),
            "failed_count": sum(
                item.get("status") == "completed" and item.get("passed") is False
                for item in segmentation_records
            ),
            "records": segmentation_records,
        },
    )
    write_json(
        layout.json_config / "liquid_semantic_observations.json",
        {
            "schema_version": "visioncortex-labpics-liquid-semantic-index/1",
            "policy": "bounded final key frames; observation only; never action confirmation",
            "record_count": len(liquid_semantic_records),
            "records": liquid_semantic_records,
        },
    )
    return report


_OPEN_VOCABULARY_MODEL_CACHE: dict[str, Any] = {}
_OPEN_VOCABULARY_ASSET_VALIDATION: set[tuple[str, str, str, str]] = set()
_GROUNDING_DINO_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_GROUNDING_DINO_ASSET_VALIDATION: set[tuple[str, str]] = set()


def _release_auxiliary_model_caches() -> dict[str, int]:
    """Bound peak RAM/VRAM by dropping event-scoped auxiliary model caches."""

    open_vocabulary = len(_OPEN_VOCABULARY_MODEL_CACHE)
    grounding_dino = len(_GROUNDING_DINO_MODEL_CACHE)
    for cached in [
        *_OPEN_VOCABULARY_MODEL_CACHE.values(),
        *_GROUNDING_DINO_MODEL_CACHE.values(),
    ]:
        model = cached.get("model") if isinstance(cached, dict) else None
        if model is not None and hasattr(model, "to"):
            try:
                model.to("cpu")
            except (RuntimeError, TypeError, ValueError):
                pass
    _OPEN_VOCABULARY_MODEL_CACHE.clear()
    _GROUNDING_DINO_MODEL_CACHE.clear()
    temporal = release_temporal_segmentation_model_cache()
    liquid = release_liquid_semantic_model_cache()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {
        "open_vocabulary": open_vocabulary,
        "grounding_dino": grounding_dino,
        "temporal_segmentation": temporal,
        "liquid_semantic": liquid,
    }


def _box_edge_gap_norm(
    left: dict[str, Any], right: dict[str, Any]
) -> float:
    """Return normalized rectangle edge distance (zero for touch/overlap)."""

    lx1, ly1, lx2, ly2 = (
        float(item) for item in left.get("xyxy_norm") or (0, 0, 0, 0)
    )
    rx1, ry1, rx2, ry2 = (
        float(item) for item in right.get("xyxy_norm") or (0, 0, 0, 0)
    )
    dx = max(lx1 - rx2, rx1 - lx2, 0.0)
    dy = max(ly1 - ry2, ry1 - ly2, 0.0)
    return math.hypot(dx, dy)


def _box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx1, ly1, lx2, ly2 = (
        float(item) for item in left.get("xyxy_norm") or (0, 0, 0, 0)
    )
    rx1, ry1, rx2, ry2 = (
        float(item) for item in right.get("xyxy_norm") or (0, 0, 0, 0)
    )
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box_aspect_ratio(box: dict[str, Any]) -> float:
    """Return the orientation-independent aspect ratio of one box."""

    x1, y1, x2, y2 = (
        float(item) for item in box.get("xyxy_norm") or (0, 0, 0, 0)
    )
    # Normalized x/y coordinates are not metrically comparable on a 16:9
    # image.  Grounded boxes retain the source aspect ratio so a broad balance
    # cannot pass as an elongated pipette merely because normalized height is
    # measured against fewer pixels.
    image_aspect_ratio = float(box.get("image_aspect_ratio") or 1.0)
    width = max(0.0, x2 - x1) * image_aspect_ratio
    height = max(0.0, y2 - y1)
    shorter = min(width, height)
    return max(width, height) / shorter if shorter > 0.0 else 0.0


def _filter_grounded_actor_boxes(
    boxes: Sequence[dict[str, Any]],
    *,
    maximum_area_norm: float,
    confidence_ratio: float,
) -> list[dict[str, Any]]:
    """Reject oversized/weak open-vocabulary hand unions fail-closed."""

    area_eligible = []
    for box in boxes:
        x1, y1, x2, y2 = (
            float(item) for item in box.get("xyxy_norm") or (0, 0, 0, 0)
        )
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= maximum_area_norm:
            area_eligible.append(box)
    if not area_eligible:
        return []
    best_confidence = max(
        float(box.get("confidence") or 0.0) for box in area_eligible
    )
    threshold = best_confidence * confidence_ratio
    return [
        box
        for box in area_eligible
        if float(box.get("confidence") or 0.0) >= threshold
    ]


def _preferred_grounding_terms_by_class(
    event: EvidenceEvent,
) -> dict[str, tuple[str, ...]]:
    """Extract only explicit participant appearance constraints from Ark text."""

    understanding = event.model_understanding or {}
    physical_change = understanding.get("physical_change") or {}
    semantic_text = " ".join(
        [
            str(understanding.get("current_step") or ""),
            " ".join(map(str, understanding.get("objects") or [])),
            " ".join(
                str(item.get("object") or "")
                for item in understanding.get("hand_object_interactions") or []
                if isinstance(item, dict)
            ),
            str(physical_change.get("before") or ""),
            str(physical_change.get("after") or ""),
        ]
    ).lower()
    preferred: dict[str, tuple[str, ...]] = {}
    if any(term in semantic_text for term in ("brown", "amber", "棕", "琥珀")):
        preferred["reagent_bottle"] = ("brown", "amber")
    # Closure colour is used only for the closure class. This avoids treating
    # the routinely blue glove as evidence for a blue bottle or cap.
    closure_colour = (
        "red"
        if "red" in semantic_text or "红" in semantic_text
        else "orange"
        if "orange" in semantic_text or "橙" in semantic_text
        else None
    )
    if closure_colour is not None:
        preferred["bottle_cap"] = (closure_colour,)
        preferred["tube_cap"] = (closure_colour,)
    return preferred


def _select_manipulated_object_candidate(
    candidates: Sequence[dict[str, Any]],
    actor_boxes: Sequence[dict[str, Any]],
    *,
    canonical_class: str,
    maximum_actor_gap: float,
    pipette_minimum_aspect_ratio: float,
    grounding_dino_pipette_minimum_aspect_ratio: float | None = None,
    minimum_confidence: float = 0.0,
    pipette_center_must_overlap_actor: bool = False,
    pipette_maximum_actor_iou: float | None = None,
    preferred_grounding_terms: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Choose one active object instance using contact before confidence.

    Crowded laboratory benches can contain many high-confidence static tools.
    A final participant annotation is allowed to show only the instance that
    touches or nearly touches the operator. Pipettes and tips additionally
    need an elongated box so a hand-shaped open-vocabulary false positive
    cannot satisfy the interaction gate.
    """

    shape_rejected = 0
    contact_rejected = 0
    confidence_rejected = 0
    centre_rejected = 0
    actor_overlap_rejected = 0
    eligible: list[tuple[float, dict[str, Any]]] = []
    elongated_classes = {"pipette", "spearhead"}
    for candidate in candidates:
        if float(candidate.get("confidence") or 0.0) < float(
            minimum_confidence
        ):
            confidence_rejected += 1
            continue
        candidate_minimum_aspect_ratio = pipette_minimum_aspect_ratio
        if (
            str(candidate.get("detector_source") or "").startswith(
                "grounding_dino"
            )
            and grounding_dino_pipette_minimum_aspect_ratio is not None
        ):
            candidate_minimum_aspect_ratio = float(
                grounding_dino_pipette_minimum_aspect_ratio
            )
        if (
            canonical_class in elongated_classes
            and _box_aspect_ratio(candidate) < candidate_minimum_aspect_ratio
        ):
            shape_rejected += 1
            continue
        relaxed_dino_shape_requires_centre = bool(
            canonical_class in elongated_classes
            and str(candidate.get("detector_source") or "").startswith(
                "grounding_dino"
            )
            and _box_aspect_ratio(candidate) < pipette_minimum_aspect_ratio
        )
        if canonical_class in elongated_classes and (
            pipette_center_must_overlap_actor
            or relaxed_dino_shape_requires_centre
        ):
            x1, y1, x2, y2 = (
                float(item)
                for item in candidate.get("xyxy_norm") or (0, 0, 0, 0)
            )
            centre_x = (x1 + x2) / 2.0
            centre_y = (y1 + y2) / 2.0
            if not any(
                float((actor.get("xyxy_norm") or (0, 0, 0, 0))[0])
                <= centre_x
                <= float((actor.get("xyxy_norm") or (0, 0, 0, 0))[2])
                and float((actor.get("xyxy_norm") or (0, 0, 0, 0))[1])
                <= centre_y
                <= float((actor.get("xyxy_norm") or (0, 0, 0, 0))[3])
                for actor in actor_boxes
            ):
                centre_rejected += 1
                continue
        if (
            canonical_class in elongated_classes
            and pipette_maximum_actor_iou is not None
            and actor_boxes
            and max(_box_iou(candidate, actor) for actor in actor_boxes)
            > float(pipette_maximum_actor_iou)
        ):
            actor_overlap_rejected += 1
            continue
        actor_gap = (
            min(_box_edge_gap_norm(candidate, actor) for actor in actor_boxes)
            if actor_boxes
            else float("inf")
        )
        if actor_gap > maximum_actor_gap:
            contact_rejected += 1
            continue
        eligible.append((actor_gap, candidate))

    preferred_terms = tuple(
        str(item).strip().lower()
        for item in preferred_grounding_terms
        if str(item).strip()
    )
    preferred_eligible = [
        item
        for item in eligible
        if any(
            term in str(item[1].get("grounding_prompt") or "").lower()
            for term in preferred_terms
        )
    ]
    ranked_eligible = preferred_eligible or eligible
    selected_pair = min(
        ranked_eligible,
        key=lambda item: (
            item[0],
            -float(item[1].get("confidence") or 0.0),
        ),
        default=None,
    )
    selected = selected_pair[1] if selected_pair is not None else None
    return selected, {
        "rule": (
            "explicit_semantic_appearance_then_minimum_actor_edge_gap_then_confidence"
            if preferred_eligible
            else "minimum_actor_edge_gap_then_confidence"
        ),
        "canonical_class": canonical_class,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "shape_rejected_count": shape_rejected,
        "contact_rejected_count": contact_rejected,
        "confidence_rejected_count": confidence_rejected,
        "centre_rejected_count": centre_rejected,
        "actor_overlap_rejected_count": actor_overlap_rejected,
        "minimum_confidence": float(minimum_confidence),
        "preferred_grounding_terms": list(preferred_terms),
        "preferred_eligible_candidate_count": len(preferred_eligible),
        "selected_grounding_prompt": (
            str(selected.get("grounding_prompt") or "")
            if selected is not None
            else None
        ),
        "pipette_center_must_overlap_actor": bool(
            pipette_center_must_overlap_actor
            if canonical_class in elongated_classes
            else False
        ),
        "relaxed_grounding_dino_shape_requires_actor_center_overlap": True,
        "pipette_maximum_actor_iou": (
            float(pipette_maximum_actor_iou)
            if canonical_class in elongated_classes
            and pipette_maximum_actor_iou is not None
            else None
        ),
        "maximum_actor_gap_norm": maximum_actor_gap,
        "minimum_aspect_ratio": (
            pipette_minimum_aspect_ratio
            if canonical_class in elongated_classes
            else None
        ),
        "grounding_dino_minimum_aspect_ratio": (
            grounding_dino_pipette_minimum_aspect_ratio
            if canonical_class in elongated_classes
            else None
        ),
        "selected_actor_gap_norm": (
            round(float(selected_pair[0]), 6)
            if selected_pair is not None
            else None
        ),
        "fail_closed": selected is None,
    }


def _select_state_container_candidate(
    candidates: Sequence[dict[str, Any]],
    actor_boxes: Sequence[dict[str, Any]],
    *,
    view_role: str,
    state_direction: str,
    maximum_actor_gap: float,
    first_person_minimum_top_offset: float = 0.0,
    closure_boxes: Sequence[dict[str, Any]] = (),
    maximum_closure_gap: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select a state-bearing container without preferring background vessels.

    In a first-person post-opening frame the manipulated vessel starts at or
    below the operating hand's vertical centre.  Crowded benches often contain
    higher, unrelated open vessels that score much better in open-vocabulary
    detection.  Treat the spatial relation as a fail-closed eligibility rule,
    then rank only eligible candidates by confidence.
    """

    eligible: list[dict[str, Any]] = []
    spatial_rule = (
        view_role == "First-Person"
        and state_direction == "opening"
        and bool(actor_boxes)
    )
    # After an opening transition the closure is normally already in the
    # operator's hand and can be spatially separated from the bottle mouth.
    # Requiring closure-to-container proximity at that phase silently removes
    # the exact participant that proves the opened state.  Keep that relation
    # strict for closing/unknown transitions, while opening requires the actor
    # to remain related independently to both the closure and the container.
    closure_proximity_required = bool(
        closure_boxes and state_direction != "opening"
    )
    for candidate in candidates:
        coordinates = candidate.get("xyxy_norm") or (0, 0, 0, 0)
        candidate_top = float(coordinates[1])
        for actor in actor_boxes:
            actor_coordinates = actor.get("xyxy_norm") or (0, 0, 0, 0)
            actor_centre_y = (
                float(actor_coordinates[1]) + float(actor_coordinates[3])
            ) / 2.0
            if (
                _box_edge_gap_norm(candidate, actor) <= maximum_actor_gap
                and (
                    not closure_proximity_required
                    or maximum_closure_gap is None
                    or min(
                        _box_edge_gap_norm(candidate, closure)
                        for closure in closure_boxes
                    )
                    <= maximum_closure_gap
                )
                and (
                    not spatial_rule
                    or candidate_top
                    >= actor_centre_y + first_person_minimum_top_offset
                )
            ):
                eligible.append(candidate)
                break
    relation_selected = min(
        eligible,
        key=lambda box: (
            min(
                (_box_edge_gap_norm(box, closure) for closure in closure_boxes),
                default=0.0,
            ),
            -float(box.get("confidence") or 0.0),
        ),
        default=None,
    )
    extent_candidates: list[dict[str, Any]] = []
    if relation_selected is not None:
        selected_confidence = float(
            relation_selected.get("confidence") or 0.0
        )
        sx1, sy1, sx2, sy2 = (
            float(item)
            for item in relation_selected.get("xyxy_norm")
            or (0, 0, 0, 0)
        )
        selected_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        for candidate in eligible:
            cx1, cy1, cx2, cy2 = (
                float(item)
                for item in candidate.get("xyxy_norm") or (0, 0, 0, 0)
            )
            intersection_width = max(0.0, min(sx2, cx2) - max(sx1, cx1))
            intersection_height = max(0.0, min(sy2, cy2) - max(sy1, cy1))
            intersection = intersection_width * intersection_height
            candidate_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            union = selected_area + candidate_area - intersection
            overlap = intersection / union if union > 0.0 else 0.0
            if (
                float(candidate.get("confidence") or 0.0)
                >= selected_confidence * 0.80
                and overlap >= 0.30
            ):
                extent_candidates.append(candidate)
        selected = max(
            extent_candidates,
            key=lambda box: (
                max(
                    0.0,
                    float((box.get("xyxy_norm") or (0, 0, 0, 0))[2])
                    - float((box.get("xyxy_norm") or (0, 0, 0, 0))[0]),
                )
                * max(
                    0.0,
                    float((box.get("xyxy_norm") or (0, 0, 0, 0))[3])
                    - float((box.get("xyxy_norm") or (0, 0, 0, 0))[1]),
                ),
                float(box.get("confidence") or 0.0),
            ),
            default=relation_selected,
        )
    else:
        selected = None
    return selected, {
        "rule": (
            "closure_proximity_then_first_person_post_opening_geometry"
            if closure_proximity_required and spatial_rule
            else "closure_proximity_then_actor_contact"
            if closure_proximity_required
            else "actor_contact_with_independently_held_closure"
            if closure_boxes and state_direction == "opening"
            else "first_person_post_opening_below_actor_centre"
            if spatial_rule
            else "actor_contact_gate_then_highest_confidence"
        ),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "extent_expansion_candidate_count": len(extent_candidates),
        "extent_confidence_ratio": 0.80,
        "extent_minimum_iou": 0.30,
        "first_person_minimum_top_offset_norm": (
            first_person_minimum_top_offset if spatial_rule else None
        ),
        "closure_candidate_count": len(closure_boxes),
        "maximum_closure_gap_norm": maximum_closure_gap,
        "selected_closure_gap_norm": (
            round(
                min(
                    _box_edge_gap_norm(selected, closure)
                    for closure in closure_boxes
                ),
                6,
            )
            if selected is not None and closure_boxes
            else None
        ),
        "fail_closed": not bool(eligible),
    }


def _canonical_grounding_label(
    label: str, prompt_map: dict[str, str]
) -> str | None:
    """Resolve complete prompt phrases, including same-class merged labels.

    Grounding DINO can return several activated phrases for one box, such as
    ``brown reagent bottle brown bottle``. Keep that box only when every word
    is covered by configured prompts and all possible matches name one class.
    Partial phrases and combinations of different classes remain unresolved.
    """

    tokens = tuple(re.findall(r"\w+", str(label).lower()))
    if not tokens:
        return None
    phrases = [
        (tuple(re.findall(r"\w+", str(prompt).lower())), canonical)
        for prompt, canonical in prompt_map.items()
        if str(prompt).strip()
    ]
    exact_classes = {canonical for phrase, canonical in phrases if phrase == tokens}
    if exact_classes:
        return next(iter(exact_classes)) if len(exact_classes) == 1 else None
    resolved: dict[int, set[str]] = {0: set()}
    for start in range(len(tokens)):
        if start not in resolved:
            continue
        for phrase, canonical in phrases:
            end = start + len(phrase)
            if phrase and tokens[start:end] == phrase:
                resolved.setdefault(end, set()).update(
                    resolved[start] | {canonical}
                )
    classes = resolved.get(len(tokens), set())
    return next(iter(classes)) if len(classes) == 1 else None


def _grounding_dino_key_frame_detections(
    frame: np.ndarray,
    canonical_classes: set[str],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ground only missing participant classes with the pinned local model."""

    fallback = dict(settings.get("grounding_dino_fallback") or {})
    if not fallback.get("enabled"):
        return [], {"status": "disabled"}
    def normalize(value: Any) -> str:
        return (
            str(value or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
    requested_classes = {normalize(item) for item in canonical_classes}
    prompt_map = {
        str(prompt).strip(): normalize(canonical)
        for prompt, canonical in dict(fallback.get("prompt_map") or {}).items()
        if str(prompt).strip() and normalize(canonical) in requested_classes
    }
    if not prompt_map:
        return [], {
            "status": "not_applicable",
            "requested_classes": sorted(requested_classes),
        }
    model_path = Path(str(fallback.get("model_path") or "")).resolve()
    weights_path = model_path / "model.safetensors"
    if not model_path.is_dir() or not weights_path.is_file():
        raise RuntimeError(
            f"Pinned Grounding DINO model is missing: {model_path}"
        )
    model_sha256 = str(fallback.get("model_sha256") or "")
    validation_key = (str(weights_path), model_sha256)
    if validation_key not in _GROUNDING_DINO_ASSET_VALIDATION:
        if model_sha256 and _sha256_file(weights_path) != model_sha256:
            raise RuntimeError(
                f"Grounding DINO model hash mismatch: {weights_path}"
            )
        _GROUNDING_DINO_ASSET_VALIDATION.add(validation_key)

    from PIL import Image
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    device = str(fallback.get("device") or "cuda")
    cache_key = (str(model_path), model_sha256, device)
    cached = _GROUNDING_DINO_MODEL_CACHE.get(cache_key)
    model_load_seconds = 0.0
    if cached is None:
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
        )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Grounding DINO requires CUDA but CUDA is unavailable")
        model = model.to(device).eval()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        cached = {"processor": processor, "model": model, "device": device}
        _GROUNDING_DINO_MODEL_CACHE[cache_key] = cached
    processor = cached["processor"]
    model = cached["model"]
    device = str(cached["device"])

    prompts = list(prompt_map)
    prompt_text = ". ".join(prompts) + "."
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inference_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=float(fallback.get("box_threshold", 0.20)),
        text_threshold=float(fallback.get("text_threshold", 0.20)),
        target_sizes=[image.size[::-1]],
    )[0]
    labels = result.get("text_labels") or result.get("labels") or []
    height, width = frame.shape[:2]
    maximum_area = float(fallback.get("maximum_box_area_norm", 0.50))
    maximum_class_areas = {
        normalize(class_name): float(limit)
        for class_name, limit in dict(
            fallback.get("maximum_class_box_area_norm") or {}
        ).items()
    }
    admitted: list[dict[str, Any]] = []
    rejected_area = 0
    rejected_label = 0
    recovered_composite_label_count = 0
    for confidence, label, coordinates in zip(
        result["scores"], labels, result["boxes"], strict=True
    ):
        grounded_prompt = str(label).strip().strip(".")
        canonical = prompt_map.get(grounded_prompt)
        if canonical is None:
            canonical = prompt_map.get(grounded_prompt.lower())
        if canonical is None:
            canonical = _canonical_grounding_label(grounded_prompt, prompt_map)
            if canonical is not None:
                recovered_composite_label_count += 1
        if canonical is None:
            rejected_label += 1
            continue
        x1, y1, x2, y2 = (float(item) for item in coordinates)
        normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
        area = max(0.0, normalized[2] - normalized[0]) * max(
            0.0, normalized[3] - normalized[1]
        )
        class_maximum_area = min(
            maximum_area,
            float(maximum_class_areas.get(canonical, maximum_area)),
        )
        if area <= 0.0 or area > class_maximum_area:
            rejected_area += 1
            continue
        admitted.append(
            {
                "class_name": canonical,
                "confidence": float(confidence),
                "xyxy_norm": normalized,
                "track_id": None,
                "roi_motion": 0.0,
                "detector_source": "grounding_dino_base_key_frame_fallback",
                "grounding_prompt": grounded_prompt,
                "image_aspect_ratio": width / height,
            }
        )
    return admitted, {
        "schema_version": "visioncortex-grounding-dino-key-frame/1",
        "status": "executed",
        "scope": "missing final participant classes only",
        "full_timeline_inference": False,
        "model": str(model_path),
        "model_revision": str(fallback.get("model_revision") or ""),
        "model_sha256": model_sha256,
        "requested_classes": sorted(requested_classes),
        "prompts": prompts,
        "device": device,
        "precision": "float32",
        "raw_detection_count": len(result["scores"]),
        "admitted_area_bounded_count": len(admitted),
        "rejected_box_area_count": rejected_area,
        "rejected_label_count": rejected_label,
        "recovered_composite_label_count": recovered_composite_label_count,
        "maximum_box_area_norm": maximum_area,
        "model_load_seconds": round(model_load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "token_usage": 0,
        "ark_calls": 0,
    }


def _missing_state_transition_fallback_classes(
    grounded: Sequence[dict[str, Any]],
    *,
    active_object_classes: set[str],
    state_prompts: set[str],
    actor_boxes: Sequence[dict[str, Any]],
    maximum_actor_gap: float,
    closure_maximum_actor_gap: float,
    closure_minimum_confidence: float,
) -> set[str]:
    """Return only missing closure/container slots for one final state frame."""

    missing = {
        canonical_class
        for canonical_class in active_object_classes & {"bottle_cap", "tube_cap"}
        if not any(
            str(box.get("class_name") or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            == canonical_class
            and float(box.get("confidence") or 0.0)
            >= closure_minimum_confidence
            and actor_boxes
            and min(_box_edge_gap_norm(box, actor) for actor in actor_boxes)
            <= closure_maximum_actor_gap
            for box in grounded
        )
    }
    container_aliases = {
        "container",
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
        "tube",
    }
    requested_container_aliases = active_object_classes & container_aliases
    state_container_present = any(
        str(box.get("class_name") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in ({"container"} | requested_container_aliases)
        and actor_boxes
        and min(_box_edge_gap_norm(box, actor) for actor in actor_boxes)
        <= maximum_actor_gap
        for box in grounded
    )
    if state_prompts and not state_container_present:
        # Request both state-specific generic prompts and the event's explicit
        # semantic bottle/tube alias.  The contact gate chooses only the
        # manipulated physical instance on a crowded bench.
        missing.update({"container", *requested_container_aliases})
    return missing


def _open_vocabulary_key_frame_supplement(
    frame: np.ndarray,
    event: EvidenceEvent,
    closed_set_detections: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    view_role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ground missing manipulated objects only on final accepted key frames.

    This is deliberately not a video-wide detector.  It runs at most on the
    two final role frames for an accepted event, and admits a grounded object
    only when its box touches or nearly touches an actor box.  Thus an open-set
    model cannot reintroduce the crowded-background boxes that the final
    participant-only contract is designed to suppress.
    """

    settings = config.get("models", {}).get("open_vocabulary_key_frame") or {}
    if not settings.get("enabled"):
        return [], {"status": "disabled"}
    def normalize(value: Any) -> str:
        return (
            str(value or "")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
    participant_classes = {
        normalize(item) for item in event.objects if str(item).strip()
    }
    active_actor_classes = participant_classes & {"hand", "gloved_hand"} or {
        "hand",
        "gloved_hand",
    }
    state_transition = event.action_type == ActionType.CONTAINER_STATE_CHANGE
    supplementable_classes = {
        "pipette",
        "spearhead",
        "paper",
        "tube",
        "tube_rack",
        "balance",
        "beaker",
        "container",
        "reagent_bottle",
        "sample_bottle",
        "sample_bottle_blue",
        "bottle_cap",
        "tube_cap",
    }
    active_object_classes = participant_classes & supplementable_classes
    if not state_transition and not active_object_classes:
        return [], {"status": "not_applicable"}
    model_path = Path(str(settings.get("model_path") or "")).resolve()
    clip_path = Path(str(settings.get("clip_model_path") or "")).resolve()
    if not model_path.is_file() or not clip_path.is_file():
        raise RuntimeError(
            "Open-vocabulary key-frame assets are missing: "
            f"model={model_path} clip={clip_path}"
        )
    model_sha256 = str(settings.get("model_sha256") or "")
    clip_sha256 = str(settings.get("clip_model_sha256") or "")
    validation_key = (
        str(model_path),
        model_sha256,
        str(clip_path),
        clip_sha256,
    )
    if validation_key not in _OPEN_VOCABULARY_ASSET_VALIDATION:
        if model_sha256 and _sha256_file(model_path) != model_sha256:
            raise RuntimeError(
                f"Open-vocabulary model hash mismatch: {model_path}"
            )
        if clip_sha256 and _sha256_file(clip_path) != clip_sha256:
            raise RuntimeError(f"CLIP model hash mismatch: {clip_path}")
        _OPEN_VOCABULARY_ASSET_VALIDATION.add(validation_key)
    prompt_map = dict(settings.get("prompt_map") or {})
    physical_change = (event.model_understanding or {}).get(
        "physical_change"
    ) or {}
    after_state_text = " ".join(
        str(item or "")
        for item in (
            (
                physical_change.get("after")
                if isinstance(physical_change, dict)
                else physical_change
            ),
            (event.model_understanding or {}).get("current_step"),
        )
    ).lower()
    opening_terms = (
        "open",
        "uncapped",
        "打开",
        "开启",
        "开放",
        "敞口",
        "瓶口露出",
        "瓶口暴露",
        "瓶盖与瓶口分离",
        "瓶口敞开",
        "瓶盖已被取下",
        "瓶盖被取下",
        "瓶盖离开瓶口",
        "瓶盖从瓶口取下",
    )
    closing_terms = (
        "closed",
        "capped",
        "盖合",
        "盖住",
        "封闭",
        "密封",
        "拧紧",
    )
    state_direction = (
        "opening"
        if any(term in after_state_text for term in opening_terms)
        else (
            "closing"
            if any(term in after_state_text for term in closing_terms)
            else "unknown"
        )
    )
    opening_prompts = {
        str(item)
        for item in settings.get("opening_container_prompts") or []
    }
    closing_prompts = {
        str(item)
        for item in settings.get("closing_container_prompts") or []
    }
    state_prompts = (
        opening_prompts
        if state_direction == "opening"
        else closing_prompts if state_direction == "closing" else set()
    )
    actor_prompts = {
        prompt
        for prompt, canonical in prompt_map.items()
        if normalize(canonical) in active_actor_classes
    }
    closure_prompts = {
        prompt
        for prompt, canonical in prompt_map.items()
        if str(canonical) in {"bottle_cap", "tube_cap"}
    }
    participant_prompts = {
        prompt
        for prompt, canonical in prompt_map.items()
        if normalize(canonical) in active_object_classes
    }
    active_prompts = actor_prompts | participant_prompts
    if state_transition:
        active_prompts |= state_prompts | closure_prompts
    prompts = [
        str(item)
        for item in prompt_map
        if str(item).strip() and item in active_prompts
    ]
    if not prompts:
        raise RuntimeError("Open-vocabulary prompt_map is empty")
    from ultralytics import YOLOWorld

    cache_key = str(model_path)
    cached = _OPEN_VOCABULARY_MODEL_CACHE.get(cache_key)
    model_cache_hit = cached is not None
    model_load_seconds = 0.0
    if cached is None:
        model_load_started = time.perf_counter()
        cached = {
            "model": YOLOWorld(str(model_path)),
            "prompts": None,
        }
        model_load_seconds = time.perf_counter() - model_load_started
        _OPEN_VOCABULARY_MODEL_CACHE[cache_key] = cached
    model = cached["model"]
    if cached.get("prompts") != prompts:
        # Ultralytics leaves the world model on the prediction device. Its
        # CLIP tokenizer creates CPU token tensors, so changing classes after a
        # GPU prediction otherwise mixes CPU indices with CUDA embeddings.
        # Final-key-frame grounding is bounded; moving back to CPU before the
        # rare prompt change is deterministic and avoids a second model copy.
        model.to("cpu")
        model.set_classes(prompts)
        cached["prompts"] = list(prompts)
    inference_started = time.perf_counter()
    result = model.predict(
        frame,
        device=int(settings.get("device", 0)),
        imgsz=int(settings.get("image_size", 1280)),
        conf=float(settings.get("confidence", 0.03)),
        iou=float(settings.get("iou", 0.50)),
        verbose=False,
    )[0]
    inference_seconds = time.perf_counter() - inference_started
    height, width = frame.shape[:2]
    grounded: list[dict[str, Any]] = []
    for class_index, confidence, coordinates in zip(
        result.boxes.cls,
        result.boxes.conf,
        result.boxes.xyxy,
        strict=True,
    ):
        prompt = str(result.names[int(class_index)])
        canonical = str(prompt_map.get(prompt) or "").strip()
        if not canonical:
            continue
        x1, y1, x2, y2 = (float(item) for item in coordinates)
        grounded.append(
            {
                "class_name": canonical,
                "confidence": float(confidence),
                "xyxy_norm": [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ],
                "track_id": None,
                "roi_motion": 0.0,
                "detector_source": "yolo_world_v2_key_frame_supplement",
                "grounding_prompt": prompt,
            }
        )
    actor_classes = {"hand", "gloved_hand"}
    closed_set_actor_boxes = [
        dict(box)
        for box in closed_set_detections
        if str(box.get("class_name") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in actor_classes
    ]
    raw_grounded_actor_boxes = [
        dict(box)
        for box in grounded
        if str(box.get("class_name") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in actor_classes
    ]
    grounded_actor_maximum_area = float(
        settings.get("grounded_actor_maximum_box_area_norm", 0.15)
    )
    grounded_actor_confidence_ratio = float(
        settings.get("grounded_actor_confidence_ratio", 0.75)
    )
    grounded_actor_boxes = _filter_grounded_actor_boxes(
        raw_grounded_actor_boxes,
        maximum_area_norm=grounded_actor_maximum_area,
        confidence_ratio=grounded_actor_confidence_ratio,
    )
    # Prefer the task-trained closed-set actor boxes whenever they exist. Open
    # vocabulary hand prompts can expand into an oversized arm/torso region and
    # incorrectly make a static rack tool appear adjacent. Grounded actors are
    # a fallback only for views where the closed-set detector saw no hand.
    actor_boxes = closed_set_actor_boxes or grounded_actor_boxes
    grounding_dino_receipt: dict[str, Any] = {"status": "not_needed"}
    grounding_dino_minimum_aspect_ratio = float(
        (
            settings.get("grounding_dino_fallback") or {}
        ).get("pipette_minimum_aspect_ratio", 1.40)
    )
    minimum_class_confidence = {
        normalize(class_name): float(limit)
        for class_name, limit in dict(
            settings.get("minimum_manipulated_object_confidence") or {}
        ).items()
    }
    pipette_center_must_overlap_actor = bool(
        settings.get("pipette_center_must_overlap_actor", False)
    )
    pipette_maximum_actor_iou = float(
        settings.get("pipette_maximum_actor_iou", 0.65)
    )
    state_maximum_gap = float(
        settings.get("state_container_max_actor_gap_norm", 0.08)
    )
    state_maximum_closure_gap = float(
        settings.get("state_container_max_closure_gap_norm", 0.03)
    )
    closure_maximum_gap = float(
        settings.get("closure_max_actor_gap_norm", 0.01)
    )
    closure_minimum_confidence = float(
        settings.get("closure_minimum_confidence", 0.12)
    )
    preferred_grounding_terms = _preferred_grounding_terms_by_class(event)
    if state_transition:
        # YOLO-World can miss either the small closure or the state-bearing
        # bottle/tube as a whole.  A state-transition final frame is complete
        # only when both slots accompany the actor, so request every missing
        # slot in one bounded local Grounding DINO call.  This remains
        # one-frame inference: no Ark call and no full-timeline rescan.
        missing_state_classes = _missing_state_transition_fallback_classes(
            grounded,
            active_object_classes=active_object_classes,
            state_prompts=state_prompts,
            actor_boxes=actor_boxes,
            maximum_actor_gap=state_maximum_gap,
            closure_maximum_actor_gap=closure_maximum_gap,
            closure_minimum_confidence=closure_minimum_confidence,
        )
        if missing_state_classes:
            requested_classes = set(missing_state_classes)
            if not actor_boxes:
                requested_classes.update(active_actor_classes)
            fallback_boxes, grounding_dino_receipt = (
                _grounding_dino_key_frame_detections(
                    frame, requested_classes, settings
                )
            )
            grounded.extend(fallback_boxes)
            if not actor_boxes:
                raw_grounded_actor_boxes = [
                    dict(box)
                    for box in grounded
                    if normalize(box.get("class_name")) in actor_classes
                ]
                grounded_actor_boxes = _filter_grounded_actor_boxes(
                    raw_grounded_actor_boxes,
                    maximum_area_norm=grounded_actor_maximum_area,
                    confidence_ratio=grounded_actor_confidence_ratio,
                )
                actor_boxes = grounded_actor_boxes
    else:
        provisional_maximum_gap = float(
            settings.get("manipulated_object_max_actor_gap_norm", 0.08)
        )
        provisional_pipette_aspect = float(
            settings.get("pipette_minimum_aspect_ratio", 1.7)
        )
        missing_classes: set[str] = set()
        for canonical_class in sorted(active_object_classes):
            class_maximum_gap = (
                closure_maximum_gap
                if canonical_class in {"bottle_cap", "tube_cap"}
                else provisional_maximum_gap
            )
            class_minimum_confidence = max(
                float(minimum_class_confidence.get(canonical_class, 0.0)),
                (
                    closure_minimum_confidence
                    if canonical_class in {"bottle_cap", "tube_cap"}
                    else 0.0
                ),
            )
            candidates = [
                box
                for box in grounded
                if normalize(box.get("class_name")) == canonical_class
            ]
            selected, _ = _select_manipulated_object_candidate(
                candidates,
                actor_boxes,
                canonical_class=canonical_class,
                maximum_actor_gap=class_maximum_gap,
                pipette_minimum_aspect_ratio=provisional_pipette_aspect,
                grounding_dino_pipette_minimum_aspect_ratio=(
                    grounding_dino_minimum_aspect_ratio
                ),
                minimum_confidence=class_minimum_confidence,
                pipette_center_must_overlap_actor=(
                    pipette_center_must_overlap_actor
                ),
                pipette_maximum_actor_iou=pipette_maximum_actor_iou,
                preferred_grounding_terms=preferred_grounding_terms.get(
                    canonical_class, ()
                ),
            )
            if selected is None:
                missing_classes.add(canonical_class)
        if missing_classes:
            requested_classes = set(missing_classes)
            if not actor_boxes:
                requested_classes.update(active_actor_classes)
            fallback_boxes, grounding_dino_receipt = (
                _grounding_dino_key_frame_detections(
                    frame, requested_classes, settings
                )
            )
            grounded.extend(fallback_boxes)
            if not actor_boxes:
                raw_grounded_actor_boxes = [
                    dict(box)
                    for box in grounded
                    if normalize(box.get("class_name")) in actor_classes
                ]
                grounded_actor_boxes = _filter_grounded_actor_boxes(
                    raw_grounded_actor_boxes,
                    maximum_area_norm=grounded_actor_maximum_area,
                    confidence_ratio=grounded_actor_confidence_ratio,
                )
                actor_boxes = grounded_actor_boxes
    maximum_gap = float(settings.get("max_actor_object_gap_norm", 0.02))
    first_person_minimum_top_offset = float(
        settings.get("first_person_minimum_container_top_offset_norm", 0.0)
    )
    eligible_closure_boxes = [
        box
        for box in grounded
        if normalize(box.get("class_name")) in {"bottle_cap", "tube_cap"}
        and float(box.get("confidence") or 0.0)
        >= closure_minimum_confidence
        and actor_boxes
        and min(_box_edge_gap_norm(box, actor) for actor in actor_boxes)
        <= closure_maximum_gap
    ]
    state_semantic_text = " ".join(
        str(item or "")
        for item in (
            (event.model_understanding or {}).get("current_step"),
            physical_change.get("before"),
            physical_change.get("after"),
        )
    ).lower()
    preferred_closure_color = (
        "red"
        if "red" in state_semantic_text or "红" in state_semantic_text
        else "orange"
        if "orange" in state_semantic_text or "橙" in state_semantic_text
        else "blue"
        if "blue" in state_semantic_text or "蓝" in state_semantic_text
        else None
    )
    if preferred_closure_color:
        color_specific_closures = [
            box
            for box in eligible_closure_boxes
            if preferred_closure_color
            in str(box.get("grounding_prompt") or "").lower()
        ]
        if color_specific_closures:
            eligible_closure_boxes = color_specific_closures
    best_state_closure: dict[str, Any] | None = None
    if state_prompts:
        state_container_classes = {
            "container",
            *(
                active_object_classes
                & {
                    "reagent_bottle",
                    "sample_bottle",
                    "sample_bottle_blue",
                    "tube",
                }
            ),
        }
        state_grounded = [
            box
            for box in grounded
            if normalize(box.get("class_name")) in state_container_classes
        ]
        best_state_container, state_selection = (
            _select_state_container_candidate(
                state_grounded,
                actor_boxes,
                view_role=view_role,
                state_direction=state_direction,
                maximum_actor_gap=state_maximum_gap,
                first_person_minimum_top_offset=(
                    first_person_minimum_top_offset
                ),
                closure_boxes=eligible_closure_boxes,
                maximum_closure_gap=state_maximum_closure_gap,
            )
        )
        if best_state_container is not None and eligible_closure_boxes:
            best_state_closure = min(
                eligible_closure_boxes,
                key=lambda box: (
                    _box_edge_gap_norm(box, best_state_container),
                    min(
                        _box_edge_gap_norm(box, actor)
                        for actor in actor_boxes
                    ),
                    -float(box.get("confidence") or 0.0),
                ),
            )
        state_selection["preferred_closure_color"] = preferred_closure_color
        state_selection["selected_closure_prompt"] = (
            str(best_state_closure.get("grounding_prompt") or "")
            if best_state_closure is not None
            else None
        )
    else:
        best_state_container = None
        state_selection = {
            "rule": "not_applicable",
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "first_person_minimum_top_offset_norm": None,
            "fail_closed": False,
        }
    manipulated_object_maximum_gap = float(
        settings.get("manipulated_object_max_actor_gap_norm", 0.08)
    )
    pipette_minimum_aspect_ratio = float(
        settings.get("pipette_minimum_aspect_ratio", 1.7)
    )
    selected_manipulated_objects: dict[str, dict[str, Any]] = {}
    manipulated_object_selection: dict[str, dict[str, Any]] = {}
    if not state_transition:
        for canonical_class in sorted(active_object_classes):
            class_maximum_gap = (
                closure_maximum_gap
                if canonical_class in {"bottle_cap", "tube_cap"}
                else manipulated_object_maximum_gap
            )
            class_minimum_confidence = max(
                float(minimum_class_confidence.get(canonical_class, 0.0)),
                (
                    closure_minimum_confidence
                    if canonical_class in {"bottle_cap", "tube_cap"}
                    else 0.0
                ),
            )
            canonical_candidates = [
                box
                for box in grounded
                if normalize(box.get("class_name")) == canonical_class
            ]
            selected, selection_receipt = _select_manipulated_object_candidate(
                canonical_candidates,
                actor_boxes,
                canonical_class=canonical_class,
                maximum_actor_gap=class_maximum_gap,
                pipette_minimum_aspect_ratio=pipette_minimum_aspect_ratio,
                grounding_dino_pipette_minimum_aspect_ratio=(
                    grounding_dino_minimum_aspect_ratio
                ),
                minimum_confidence=class_minimum_confidence,
                pipette_center_must_overlap_actor=(
                    pipette_center_must_overlap_actor
                ),
                pipette_maximum_actor_iou=pipette_maximum_actor_iou,
                preferred_grounding_terms=preferred_grounding_terms.get(
                    canonical_class, ()
                ),
            )
            manipulated_object_selection[canonical_class] = selection_receipt
            if selected is not None:
                selected_manipulated_objects[canonical_class] = selected
    admitted: list[dict[str, Any]] = []
    rejected_noncontact = 0
    for box in grounded:
        canonical = str(box.get("class_name") or "")
        if canonical in actor_classes:
            if not closed_set_actor_boxes and box in actor_boxes:
                admitted.append(box)
            continue
        normalized_canonical = normalize(canonical)
        if not state_transition and normalized_canonical in active_object_classes:
            if box is selected_manipulated_objects.get(normalized_canonical):
                admitted.append(box)
            else:
                rejected_noncontact += 1
            continue
        actor_gap = (
            min(_box_edge_gap_norm(box, actor) for actor in actor_boxes)
            if actor_boxes
            else float("inf")
        )
        if canonical == "container" and state_prompts:
            if box is best_state_container and actor_gap <= state_maximum_gap:
                admitted.append(box)
            else:
                rejected_noncontact += 1
            continue
        if canonical in {"bottle_cap", "tube_cap"}:
            if state_transition and box is best_state_closure:
                admitted.append(box)
            elif not state_transition and (
                float(box.get("confidence") or 0.0)
                >= closure_minimum_confidence
                and actor_gap <= closure_maximum_gap
            ):
                admitted.append(box)
            else:
                rejected_noncontact += 1
            continue
        if actor_gap <= maximum_gap:
            admitted.append(box)
        else:
            rejected_noncontact += 1
    if state_transition:
        closed_set_replaced_classes = sorted(
            {
                "container",
                "beaker",
                "tube",
                "sample_bottle",
                "sample_bottle_blue",
                "reagent_bottle",
            }
        )
    else:
        closed_set_replaced_classes = sorted(active_object_classes)
    return admitted, {
        "schema_version": "visioncortex-open-vocabulary-key-frame/1",
        "status": "executed",
        "scope": "final accepted key frames only",
        "full_timeline_inference": False,
        "model": str(model_path),
        "model_sha256": model_sha256,
        "model_cache_hit": model_cache_hit,
        "model_load_seconds": round(model_load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "clip_model": str(clip_path),
        "clip_model_sha256": clip_sha256,
        "prompts": prompts,
        "raw_detection_count": len(grounded),
        "admitted_contact_detection_count": len(admitted),
        "rejected_noncontact_detection_count": rejected_noncontact,
        "max_actor_object_gap_norm": maximum_gap,
        "actor_box_source": (
            "closed_set"
            if closed_set_actor_boxes
            else "open_vocabulary_fallback"
            if grounded_actor_boxes
            else "missing"
        ),
        "actor_box_count": len(actor_boxes),
        "grounded_actor_filter": {
            "raw_candidate_count": len(raw_grounded_actor_boxes),
            "admitted_candidate_count": len(grounded_actor_boxes),
            "maximum_box_area_norm": grounded_actor_maximum_area,
            "confidence_ratio": grounded_actor_confidence_ratio,
        },
        "manipulated_object_max_actor_gap_norm": (
            manipulated_object_maximum_gap if not state_transition else None
        ),
        "pipette_minimum_aspect_ratio": (
            pipette_minimum_aspect_ratio if not state_transition else None
        ),
        "manipulated_object_selection": manipulated_object_selection,
        "grounding_dino_fallback": grounding_dino_receipt,
        "closed_set_replaced_classes": closed_set_replaced_classes,
        "state_direction": state_direction,
        "state_container_prompts": sorted(state_prompts),
        "state_container_max_actor_gap_norm": state_maximum_gap,
        "state_container_selection": state_selection,
        "view_role": view_role,
        "closure_max_actor_gap_norm": closure_maximum_gap,
        "closure_minimum_confidence": closure_minimum_confidence,
        "admitted": [
            {
                "class_name": box["class_name"],
                "confidence": round(float(box["confidence"]), 6),
                "grounding_prompt": box["grounding_prompt"],
                "xyxy_norm": [
                    round(float(item), 6) for item in box["xyxy_norm"]
                ],
            }
            for box in admitted
        ],
    }


def _review_bounded_cap_keyframe(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    pair: tuple[str, str],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Validate the cap instance on a few neighboring event frames.

    A missing visible cap does not justify deleting a real sequence-level
    action. Keep the current frame first, then inspect bounded alternatives;
    only the reviewer may admit an existing cap proposal. All requests share
    the final-annotation budget and content-addressed cache.
    """
    settings = config.get("key_materials", {}).get("participant_visual_review") or {}
    temporal = settings.get("temporal_keyframe_review") or {}
    if not (
        settings.get("enabled") and temporal.get("enabled")
        and "bottle_cap" in settings.get("classes", ["paper"])
        and "bottle_cap" in event.objects
        and config.get("mllm", {}).get("enabled")
        and (event.model_understanding or {}).get("status") == "completed"
    ):
        return None, {"status": "not_applicable"}
    maximum = int(temporal.get("max_frames_per_event", 5))
    if not 1 <= maximum <= 5:
        raise ValueError("Cap keyframe review accepts one to five candidate times")
    span = max(0.0, event.global_end_ms - event.global_start_ms)
    candidates = []
    values = (
        event.key_global_ms,
        event.key_global_ms + 0.25 * span,
        event.key_global_ms - 0.25 * span,
        event.global_end_ms,
        event.global_start_ms,
    )
    for value in values:
        timestamp = round(min(event.global_end_ms, max(event.global_start_ms, value)), 3)
        if timestamp not in candidates:
            candidates.append(timestamp)
    candidates = candidates[:maximum]
    nearest = {
        view_id: nearest_frame_evidence_many(detection_paths[view_id], candidates)
        for view_id in pair
    }
    by_view = {view.view_id: view for view in views}
    reviewer = ParticipantVisualReviewer(
        config, layout.work, layout.json_config, _grounding_dino_key_frame_detections
    )
    receipt: dict[str, Any] = {
        "status": "no_verified_candidate", "participant_class": "bottle_cap",
        "original_key_global_ms": float(event.key_global_ms),
        "maximum_candidate_times": maximum, "candidates": [],
        "full_scan_repeated": False, "source_copy_bytes": 0,
        "purpose": "instance-localized representative frame; not action certification",
    }
    reader = ViewFrameReader(max_open=2)
    try:
        for timestamp in candidates:
            prepared = []
            root = layout.work / "participant-keyframe-candidates" / event.event_id / f"{timestamp:.3f}"
            root.mkdir(parents=True, exist_ok=True)
            for role, view_id in zip(("First-Person", "Third-Person"), pair, strict=True):
                local_ms = transforms[view_id].to_local(timestamp)
                if not 0 <= local_ms <= infos[view_id].duration_ms:
                    break
                frame = reader.read(by_view[view_id], infos[view_id], local_ms)
                if frame is None:
                    break
                path = root / f"{role}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError("Could not retain cap keyframe candidate")
                evidence = nearest[view_id].get(timestamp)
                prepared.append({
                    "view_id": view_id, "role_label": role, "raw_path": path,
                    "detections": [box.model_dump() for box in evidence.detections] if evidence else [],
                })
            if len(prepared) != 2:
                receipt["candidates"].append({"global_ms": timestamp, "status": "missing_view"})
                continue
            record, plan = reviewer.review(
                event, prepared, participant_class="bottle_cap"
            )
            visible = [view for view, item in plan.items() if item["boxes"]]
            receipt["candidates"].append({
                "global_ms": timestamp, "status": record["status"],
                "input_fingerprint": record["input_fingerprint"],
                "localized_view_ids": visible,
                "images": {view["view_id"]: str(view["raw_path"]) for view in prepared},
            })
            if visible:
                receipt.update(status="selected", selected_global_ms=timestamp)
                return timestamp, receipt
    finally:
        reader.close()
    return None, receipt


def _select_key_material_media_source(
    view: ViewInput,
    info: VideoInfo,
    transform: AlignmentTransform,
    event: EvidenceEvent,
    group: ExperimentGroup,
    local_experiment_source: tuple[ViewInput, VideoInfo] | None,
    *,
    before_ms: float,
    after_ms: float,
) -> tuple[ViewInput, VideoInfo, float, float, float, dict[str, Any]]:
    """Use a derived experiment clip only when it honestly covers the event.

    Encoded experiment clips can end a fraction of a frame before the requested
    global group boundary. An event close to that tail must fall back to the
    original source instead of aborting the archive or clamping its timestamp.
    """

    clip_start_global = max(event.global_start_ms - before_ms, 0.0)
    clip_end_global = event.global_end_ms + after_ms
    original_key_ms = transform.to_local(event.key_global_ms)
    original_start_ms = max(0.0, transform.to_local(clip_start_global))
    original_end_ms = min(
        info.duration_ms,
        transform.to_local(clip_end_global),
    )
    selection: dict[str, Any] = {
        "schema_version": "visioncortex-key-material-source-selection/1",
        "selected_source": "original_source",
        "fallback_reason": "experiment_clip_unavailable",
        "requested_key_global_ms": float(event.key_global_ms),
    }
    if local_experiment_source is None:
        return (
            view,
            info,
            original_key_ms,
            original_start_ms,
            original_end_ms,
            selection,
        )

    candidate_view, candidate_info = local_experiment_source
    candidate_key_ms = event.key_global_ms - group.global_start_ms
    candidate_event_start_ms = event.global_start_ms - group.global_start_ms
    candidate_event_end_ms = event.global_end_ms - group.global_start_ms
    requested_start_ms = clip_start_global - group.global_start_ms
    requested_end_ms = clip_end_global - group.global_start_ms
    candidate_start_ms = max(
        0.0,
        requested_start_ms,
    )
    candidate_end_ms = min(
        candidate_info.duration_ms,
        requested_end_ms,
    )
    key_covered = 0.0 <= candidate_key_ms < candidate_info.duration_ms
    interval_covered = bool(
        0.0 <= candidate_event_start_ms < candidate_event_end_ms
        and candidate_event_end_ms <= candidate_info.duration_ms
    )
    selection.update(
        {
            "experiment_clip_path": str(candidate_view.video),
            "experiment_clip_duration_ms": float(candidate_info.duration_ms),
            "experiment_relative_key_ms": float(candidate_key_ms),
            "experiment_relative_event_start_ms": float(
                candidate_event_start_ms
            ),
            "experiment_relative_event_end_ms": float(candidate_event_end_ms),
            "experiment_requested_clip_start_ms": float(requested_start_ms),
            "experiment_requested_clip_end_ms": float(requested_end_ms),
            "experiment_relative_clip_start_ms": float(candidate_start_ms),
            "experiment_relative_clip_end_ms": float(candidate_end_ms),
            "experiment_clip_key_covered": key_covered,
            "experiment_clip_interval_covered": interval_covered,
        }
    )
    if key_covered and interval_covered:
        selection.update(
            {
                "selected_source": "verified_local_experiment_clip",
                "fallback_reason": None,
            }
        )
        return (
            candidate_view,
            candidate_info,
            candidate_key_ms,
            candidate_start_ms,
            candidate_end_ms,
            selection,
        )

    selection["fallback_reason"] = (
        "experiment_clip_key_timestamp_outside_media"
        if not key_covered
        else "experiment_clip_event_interval_outside_media"
    )
    return (
        view,
        info,
        original_key_ms,
        original_start_ms,
        original_end_ms,
        selection,
    )


def materialize_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    views: Sequence[ViewInput],
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    detection_paths: dict[str, Path],
    config: dict[str, Any],
    publisher: Any | None = None,
    archive_id: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    materialize_event_ids: set[str] | None = None,
) -> None:
    by_view = {view.view_id: view for view in views}
    before = float(config["segmentation"]["key_clip_pre_seconds"]) * 1000.0
    after = float(config["segmentation"]["key_clip_post_seconds"]) * 1000.0
    encoder = config["performance"]["ffmpeg_video_encoder"]
    workers = max(1, min(2, int(config["performance"].get("materialization_workers", 2))))
    group_by_event = {event_id: group for group in groups for event_id in group.key_event_ids}
    runtime_records: list[dict[str, Any]] = []
    stage_started = time.perf_counter()
    all_accepted_events = [
        event
        for event in events
        if event.accepted and event.event_id in group_by_event
    ]
    accepted_events = [
        event
        for event in all_accepted_events
        if materialize_event_ids is None or event.event_id in materialize_event_ids
    ]
    include_empty_categories = bool(
        config.get("archive", {}).get(
            "include_empty_action_categories", True
        )
    )
    prepare_key_material_category_layout(
        layout,
        groups,
        all_accepted_events,
        include_empty_categories=include_empty_categories,
    )
    best_frames_by_view = {
        view_id: _best_event_frames_many(path, accepted_events)
        for view_id, path in detection_paths.items()
    }
    material_view_pairs: dict[str, tuple[str, str]] = {}
    key_frame_selection_records: list[dict[str, Any]] = []
    for event_index, event in enumerate(accepted_events):
        if progress_callback is not None:
            progress_callback(event_index, len(accepted_events))
        group = group_by_event[event.event_id]
        ranked: list[tuple[float, int, str, FrameEvidence, dict[str, Any]]] = []
        for role_rank, view_id in enumerate(
            (group.first_person_view, group.third_person_view)
        ):
            selected = (best_frames_by_view.get(view_id) or {}).get(event.event_id)
            if selected is None:
                continue
            frame, score, receipt = selected
            ranked.append((score, -role_rank, view_id, frame, receipt))
        previous_key_global_ms = float(event.key_global_ms)
        if ranked:
            directly_supported_view_ids = {
                str(item)
                for item in (event.semantic_review or {}).get(
                    "directly_supported_view_ids", []
                )
            }
            directly_supported_ranked = [
                item for item in ranked if item[2] in directly_supported_view_ids
            ]
            eligible_ranked = directly_supported_ranked or ranked
            score, _, source_view_id, frame, receipt = max(
                eligible_ranked, key=lambda item: (item[0], item[1])
            )
            event.key_global_ms = float(frame.global_ms)
            event.observability["key_frame_selection"] = {
                "schema_version": "visioncortex-participant-key-frame-selection/1",
                "policy": "highest participant-instance coverage within accepted event bounds",
                "source_view_id": source_view_id,
                "directly_supported_view_ids": sorted(
                    directly_supported_view_ids
                ),
                "direct_support_priority_applied": bool(
                    directly_supported_ranked
                ),
                "previous_key_global_ms": previous_key_global_ms,
                "selected_key_global_ms": event.key_global_ms,
                "selection_offset_ms": round(
                    event.key_global_ms - previous_key_global_ms, 3
                ),
                "selection_score": round(score, 6),
                "selection_receipt": receipt,
            }
        actor_required_actions = {
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.DEVICE_PANEL_OPERATION,
            ActionType.PIPETTE_TRANSFER_OPERATION,
        }
        if event.action_type in actor_required_actions:
            rescue_global_ms, rescue_receipt = (
                _bounded_grounding_dino_temporal_rescue(
                    event,
                    group,
                    views,
                    infos,
                    transforms,
                    detection_paths,
                    config,
                )
            )
            event.observability.setdefault("key_frame_selection", {})[
                "grounding_dino_temporal_rescue"
            ] = rescue_receipt
            if rescue_global_ms is not None:
                event.key_global_ms = float(rescue_global_ms)
                event.observability["key_frame_selection"].update(
                    {
                        "policy": (
                            "highest bounded grounded participant interaction "
                            "within accepted event bounds"
                        ),
                        "source_view_id": rescue_receipt.get(
                            "selected_view_id"
                        ),
                        "selected_key_global_ms": event.key_global_ms,
                        "selection_offset_ms": round(
                            event.key_global_ms - previous_key_global_ms, 3
                        ),
                        "grounding_dino_temporal_rescue_applied": True,
                    }
                )
        try:
            pair, pair_receipt = _select_key_material_view_pair_with_peak_fallback(
                group,
                event,
                views,
                infos,
                transforms,
                previous_key_global_ms,
            )
        except ValueError as error:
            # One CV candidate can outlive a shorter physical camera tail after
            # alignment or participant-keyframe reranking.  That candidate has
            # no honest dual-role key material, but it must not abort unrelated
            # events or the experiment package.  Reject it fail-closed and let
            # semantic curation write the normal machine-quarantine receipt.
            event.key_global_ms = previous_key_global_ms
            set_event_admission(event, "rejected")
            event.uncertainty = list(
                dict.fromkeys(
                    [*event.uncertainty, "key_material_dual_role_coverage_unavailable"]
                )
            )
            event.audit_reason = (
                f"{event.audit_reason}; " if event.audit_reason else ""
            ) + "machine-quarantined: no honest dual-role key material"
            failure_receipt = {
                "schema_version": "visioncortex-key-material-view-selection/1",
                "status": "machine_quarantined_missing_dual_role_key_material",
                "evidence_classification": "PARTIAL_EVIDENCE",
                "reason": str(error),
                "key_global_ms": previous_key_global_ms,
                "timestamp_clamped": False,
                "synthetic_cross_view_evidence": False,
                "manual_fallback_required": False,
                "analysis_continuation_allowed": True,
            }
            event.observability["key_material_view_selection"] = failure_receipt
            event.semantic_review = {
                **dict(event.semantic_review or {}),
                "model_status": "not_run",
                "error_class": "key_material_dual_role_coverage_unavailable",
                "error": str(error),
                "evidence_classification": "PARTIAL_EVIDENCE",
                "retryable": False,
            }
            key_frame_selection_records.append(
                {
                    "event_id": event.event_id,
                    **dict(event.observability.get("key_frame_selection") or {}),
                    "key_material_view_selection": failure_receipt,
                }
            )
            continue
        if pair_receipt.get("key_timestamp_fallback"):
            event.observability.setdefault("key_frame_selection", {}).update(
                {
                    "selected_key_global_ms": float(event.key_global_ms),
                    "selection_offset_ms": round(
                        event.key_global_ms - previous_key_global_ms, 3
                    ),
                    "cross_view_coverage_fallback": dict(
                        pair_receipt["key_timestamp_fallback"]
                    ),
                }
            )
        material_view_pairs[event.event_id] = pair
        event.observability["key_material_view_selection"] = pair_receipt
        reviewed_key_ms, cap_review = _review_bounded_cap_keyframe(
            layout, event, pair, views, infos, transforms, detection_paths, config
        )
        if cap_review["status"] != "not_applicable":
            event.observability.setdefault("key_frame_selection", {})[
                "cap_visual_candidate_review"
            ] = cap_review
        if reviewed_key_ms is not None:
            event.key_global_ms = reviewed_key_ms
            selection = event.observability["key_frame_selection"]
            selection["prior_cv_selection"] = {
                key: selection[key] for key in ("source_view_id", "selection_score", "selection_receipt")
                if key in selection
            }
            selection.pop("selection_score", None)
            event.observability["key_frame_selection"].update({
                "policy": "visually reviewed bottle-cap instance on bounded event frames",
                "source_view_id": cap_review["candidates"][-1]["localized_view_ids"][0],
                "selection_receipt": cap_review["candidates"][-1],
                "selected_key_global_ms": reviewed_key_ms,
                "selection_offset_ms": round(reviewed_key_ms - previous_key_global_ms, 3),
            })
        key_frame_selection_records.append(
            {
                "event_id": event.event_id,
                **dict(event.observability.get("key_frame_selection") or {}),
                "key_material_view_selection": pair_receipt,
            }
        )
    accepted_events = [event for event in accepted_events if event.accepted]
    if progress_callback is not None:
        progress_callback(len(accepted_events), len(accepted_events))
    if bool(config.get("performance", {}).get("release_auxiliary_models_after_event")):
        _release_auxiliary_model_caches()
    write_json(
        layout.json_config / "key_frame_selection.json",
        {
            "schema_version": "visioncortex-participant-key-frame-selection/1",
            "event_count": len(key_frame_selection_records),
            "records": key_frame_selection_records,
        },
    )
    accepted_timestamps = [event.key_global_ms for event in accepted_events]
    lookup_started = time.perf_counter()
    nearest_by_view = {
        view_id: nearest_frame_evidence_many(path, accepted_timestamps)
        for view_id, path in detection_paths.items()
    }
    lookup_seconds = time.perf_counter() - lookup_started
    local_experiment_sources: dict[
        tuple[str, str], tuple[ViewInput, VideoInfo]
    ] = {}
    if bool(
        config.get("performance", {}).get(
            "reuse_experiment_clips_for_key_materials", True
        )
    ):
        for group in groups:
            for role_label, view_id in (
                ("First-Person", group.first_person_view),
                ("Third-Person", group.third_person_view),
            ):
                relative = group.videos.get(role_label.lower())
                if not relative:
                    continue
                path = layout.root / relative
                if not path.is_file():
                    continue
                source_view = ViewInput(
                    view_id=view_id,
                    role=by_view[view_id].role,
                    video=path,
                )
                local_experiment_sources[(group.group_id, view_id)] = (
                    source_view,
                    probe_video(path),
                )
    overlap_aligned = bool(
        publisher is None
        and config.get("performance", {}).get(
            "overlap_aligned_key_materials", False
        )
    )
    aligned_executor = (
        ThreadPoolExecutor(
            max_workers=max(
                1,
                int(
                    config.get("performance", {}).get(
                        "aligned_key_material_workers", 1
                    )
                ),
            ),
            thread_name_prefix="key-material-aligned",
        )
        if overlap_aligned
        else None
    )
    aligned_jobs: list[Any] = []

    def materialize_aligned(
        event: EvidenceEvent,
        group: ExperimentGroup,
        action_folder: str,
        frame_dir: Path,
        clip_dir: Path,
        first_material_view: str,
        third_material_view: str,
        frame_paths: dict[str, Path],
        clip_paths: dict[str, Path],
        event_started: float,
    ) -> dict[str, Any]:
        aligned_started = time.perf_counter()
        aligned_frame = frame_dir / "Aligned_First+Third.jpg"
        _write_aligned_frame(
            frame_paths["First-Person"],
            frame_paths["Third-Person"],
            aligned_frame,
            (first_material_view, third_material_view),
        )
        aligned_frame_relative = _relative(aligned_frame, layout.root)
        event.key_frames["aligned_first_third"] = aligned_frame_relative
        aligned_frame_json = frame_dir / "Aligned_First+Third.json"
        write_json(
            aligned_frame_json,
            _artifact_json(
                group,
                event,
                "aligned_first_third_key_frame",
                aligned_frame_relative,
                None,
                transforms,
                archive_id,
            ),
        )
        aligned_clip = clip_dir / "Aligned_First+Third.mp4"
        aligned_inputs = [
            (first_material_view, clip_paths["First-Person"]),
            (third_material_view, clip_paths["Third-Person"]),
        ]
        aligned_cache = _materialize_derived_media(
            aligned_clip,
            "key-aligned-clip",
            {
                "event_id": event.event_id,
                "global_start_ms": max(event.global_start_ms - before, 0.0),
                "global_end_ms": event.global_end_ms + after,
                "layout": "first_person_left,third_person_right",
                "grid_shape": "2x1@640x360_each",
            },
            [path for _, path in aligned_inputs],
            config,
            lambda: create_grid_video(aligned_inputs, aligned_clip, encoder),
            content_address_inputs=True,
        )
        aligned_clip_relative = _relative(aligned_clip, layout.root)
        event.key_clips["aligned_first_third"] = aligned_clip_relative
        aligned_clip_json = clip_dir / "Aligned_First+Third.json"
        write_json(
            aligned_clip_json,
            _artifact_json(
                group,
                event,
                "aligned_first_third_key_clip",
                aligned_clip_relative,
                None,
                transforms,
                archive_id,
            ),
        )
        if publisher is not None:
            for artifact in (
                aligned_frame,
                aligned_frame_json,
                aligned_clip,
                aligned_clip_json,
            ):
                publisher.publish_file(artifact)
        return {
            "event_id": event.event_id,
            "experiment_group_id": group.group_id,
            "action_type": event.action_type.value,
            "action_category_folder": action_folder,
            "role_label": "Aligned-First-Third",
            "view_id": "aligned_first_third",
            "duration_seconds": round(time.perf_counter() - aligned_started, 6),
            "frame_output_bytes": aligned_frame.stat().st_size,
            "clip_output_bytes": aligned_clip.stat().st_size,
            "event_wall_duration_seconds": round(
                time.perf_counter() - event_started, 6
            ),
            "overlapped_with_next_event": overlap_aligned,
            **aligned_cache,
        }

    selected_event_ids = {event.event_id for event in accepted_events}
    for event in events:
        if event.event_id not in selected_event_ids:
            continue
        event_started = time.perf_counter()
        group = group_by_event[event.event_id]
        first_material_view, third_material_view = material_view_pairs[
            event.event_id
        ]
        folder = group.archive_folder or _safe_folder_name(group.group_id)
        action_folder = key_material_action_folder(event.action_type)
        event_folder = _key_material_event_folder_name(layout, folder, event)
        frame_dir = layout.key_frames / folder / action_folder / event_folder
        clip_dir = layout.key_clips / folder / action_folder / event_folder
        frame_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: dict[str, Path] = {}
        clip_paths: dict[str, Path] = {}

        def extract_role(role_label: str, view_id: str) -> dict[str, Any]:
            role_started = time.perf_counter()
            view = by_view[view_id]
            transform = transforms[view_id]
            clip_start_global = max(event.global_start_ms - before, 0.0)
            clip_end_global = event.global_end_ms + after
            local_experiment_source = local_experiment_sources.get(
                (group.group_id, view_id)
            )
            (
                material_view,
                material_info,
                local_key_ms,
                local_start,
                local_end,
                material_source_selection,
            ) = _select_key_material_media_source(
                view,
                infos[view_id],
                transform,
                event,
                group,
                local_experiment_source,
                before_ms=before,
                after_ms=after,
            )
            material_source = str(
                material_source_selection["selected_source"]
            )
            material_source_path = (
                str(material_view.video)
                if material_source == "verified_local_experiment_clip"
                else None
            )
            if not 0.0 <= local_key_ms < material_info.duration_ms:
                raise ValueError(
                    f"{event.event_id}/{view_id} key timestamp is outside the material source"
                )
            frame_started = time.perf_counter()
            frame = None
            used_offset_ms = 0.0
            frame_reader = ViewFrameReader(max_open=1)
            try:
                for offset_ms in (0.0, -100.0, 100.0, -250.0, 250.0):
                    candidate_ms = local_key_ms + offset_ms
                    if not 0.0 <= candidate_ms <= material_info.duration_ms:
                        continue
                    frame = frame_reader.read(
                        material_view, material_info, candidate_ms
                    )
                    if frame is not None:
                        used_offset_ms = offset_ms
                        break
            finally:
                frame_reader.close()
            if frame is None:
                raise RuntimeError(f"{event.event_id}/{view_id} key frame decode failed")
            frame_seconds = time.perf_counter() - frame_started
            nearest = nearest_by_view[view_id].get(float(event.key_global_ms))
            detected_boxes = (
                [box.model_dump() for box in nearest.detections] if nearest else []
            )
            annotation_input_root = (
                layout.work / "key-material-annotation-inputs" / event.event_id
            )
            annotation_input_root.mkdir(parents=True, exist_ok=True)
            raw_annotation_frame = annotation_input_root / f"{role_label}.jpg"
            if not cv2.imwrite(
                str(raw_annotation_frame),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(
                    f"Unable to retain raw key-material annotation frame: {raw_annotation_frame}"
                )
            write_json(
                annotation_input_root / f"{role_label}.json",
                {
                    "schema_version": "visioncortex-key-material-annotation-input/1",
                    "event_id": event.event_id,
                    "view_id": view_id,
                    "role_label": role_label,
                    "requested_key_global_ms": float(event.key_global_ms),
                    "decoded_key_global_ms": (
                        group.global_start_ms + local_key_ms + used_offset_ms
                        if material_source == "verified_local_experiment_clip"
                        else transform.to_global(local_key_ms + used_offset_ms)
                    ),
                    "key_frame_time_basis": (
                        "verified_experiment_clip_relative_time; source_frame_pts_not_recorded"
                        if material_source == "verified_local_experiment_clip"
                        else "decoder_seek_target; source_frame_pts_not_recorded"
                    ),
                    "material_source": material_source,
                    "material_source_path": material_source_path,
                    "material_source_selection": material_source_selection,
                    "upstream_source_files": [
                        str(path) for path in view_source_files(view)
                    ],
                    "frame_decode_offset_ms": used_offset_ms,
                    "detections": detected_boxes,
                    "retention": "existing_persistent_run_cache",
                    "formally_published": False,
                },
            )
            boxes, annotation_filter = _event_participant_boxes(
                event, detected_boxes, view_id=view_id
            )
            base = role_label
            frame_path = frame_dir / f"{base}.jpg"
            write_annotated_frame(frame, boxes, frame_path)
            if local_end <= local_start:
                raise ValueError(
                    f"{event.event_id}/{view_id} key clip boundary is outside the material source"
                )
            clip_path = clip_dir / f"{base}.mp4"
            clip_started = time.perf_counter()
            clip_cache = _materialize_derived_media(
                clip_path,
                "key-view-clip",
                {
                    "event_id": event.event_id,
                    "role_label": role_label,
                    "view_id": view_id,
                    "local_start_ms": local_start,
                    "local_end_ms": local_end,
                    "global_start_ms": clip_start_global,
                    "global_end_ms": clip_end_global,
                    "material_source": material_source,
                },
                view_source_files(material_view),
                config,
                lambda: extract_view_clip(
                    material_view,
                    material_info,
                    clip_path,
                    local_start,
                    local_end - local_start,
                    encoder,
                ),
            )
            clip_seconds = time.perf_counter() - clip_started
            return {
                "event_id": event.event_id,
                "experiment_group_id": group.group_id,
                "action_type": event.action_type.value,
                "action_category_folder": action_folder,
                "role_label": role_label,
                "view_id": view_id,
                "frame_path": frame_path,
                "clip_path": clip_path,
                "frame_decode_offset_ms": used_offset_ms,
                "material_source": material_source,
                "material_source_path": material_source_path,
                "material_source_selection": material_source_selection,
                "upstream_source_files": [
                    str(path) for path in view_source_files(view)
                ],
                "frame_duration_seconds": round(frame_seconds, 6),
                "clip_duration_seconds": round(clip_seconds, 6),
                "duration_seconds": round(time.perf_counter() - role_started, 6),
                "clip_source_duration_seconds": round((local_end - local_start) / 1000.0, 6),
                "frame_output_bytes": frame_path.stat().st_size,
                "clip_output_bytes": clip_path.stat().st_size,
                "annotation_filter": annotation_filter,
                **clip_cache,
            }

        role_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="key-material-media",
        ) as executor:
            futures = {
                role_label: executor.submit(extract_role, role_label, view_id)
                for role_label, view_id in (
                    ("First-Person", first_material_view),
                    ("Third-Person", third_material_view),
                )
            }
            for role_label, future in futures.items():
                role_results[role_label] = future.result()

        # Keep archive mutation and incremental publication ordered. Readers
        # never see a sidecar before the corresponding media is complete.
        for role_label, view_id in (
            ("First-Person", first_material_view),
            ("Third-Person", third_material_view),
        ):
            result = role_results[role_label]
            frame_path = Path(result["frame_path"])
            clip_path = Path(result["clip_path"])
            relative_frame = _relative(frame_path, layout.root)
            relative_clip = _relative(clip_path, layout.root)
            event.key_frames[view_id] = relative_frame
            event.key_clips[view_id] = relative_clip
            frame_paths[role_label] = frame_path
            clip_paths[role_label] = clip_path
            if result["frame_decode_offset_ms"]:
                event.uncertainty.append(
                    f"{view_id} key frame decode offset {result['frame_decode_offset_ms']:+.0f} ms"
                )
            frame_json = frame_dir / f"{role_label}.json"
            clip_json = clip_dir / f"{role_label}.json"
            write_json(
                frame_json,
                _artifact_json(
                    group,
                    event,
                    "key_frame",
                    relative_frame,
                    view_id,
                    transforms,
                    archive_id,
                ),
            )
            write_json(
                clip_json,
                _artifact_json(
                    group,
                    event,
                    "key_clip",
                    relative_clip,
                    view_id,
                    transforms,
                    archive_id,
                ),
            )
            if publisher is not None:
                for artifact in (frame_path, frame_json, clip_path, clip_json):
                    publisher.publish_file(artifact)
            runtime_records.append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"frame_path", "clip_path"}
                }
            )
            event.observability.setdefault(
                "key_material_annotation", {
                    "schema_version": "visioncortex-key-material-annotation/1",
                    "mode": "event_participants_only",
                    "views": {},
                }
            )["views"][view_id] = dict(result["annotation_filter"])

        aligned_arguments = (
            event,
            group,
            action_folder,
            frame_dir,
            clip_dir,
            first_material_view,
            third_material_view,
            dict(frame_paths),
            dict(clip_paths),
            event_started,
        )
        if aligned_executor is None:
            runtime_records.append(materialize_aligned(*aligned_arguments))
        else:
            aligned_jobs.append(
                aligned_executor.submit(materialize_aligned, *aligned_arguments)
            )

    if aligned_executor is not None:
        try:
            runtime_records.extend(job.result() for job in aligned_jobs)
        finally:
            aligned_executor.shutdown(wait=True, cancel_futures=True)
        event_order = {
            event.event_id: index for index, event in enumerate(accepted_events)
        }
        role_order = {
            "First-Person": 0,
            "Third-Person": 1,
            "Aligned-First-Third": 2,
        }
        runtime_records.sort(
            key=lambda item: (
                event_order.get(str(item.get("event_id")), len(event_order)),
                role_order.get(str(item.get("role_label")), len(role_order)),
            )
        )

    category_index_path = write_key_material_category_index(
        layout,
        groups,
        all_accepted_events,
        publisher=publisher,
        include_empty_categories=include_empty_categories,
    )
    runtime_path = layout.json_config / "key_material_materialization_runtime.json"
    previous_runtime: dict[str, Any] = {}
    if materialize_event_ids is not None and runtime_path.exists():
        try:
            previous_runtime = json.loads(
                runtime_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, ValueError):
            previous_runtime = {}
    previous_records = [
        item
        for item in previous_runtime.get("records") or []
        if str(item.get("event_id") or "") not in selected_event_ids
    ]
    previous_passes = list(previous_runtime.get("materialization_passes") or [])
    if not previous_passes and previous_runtime:
        previous_passes = [
            {
                "scope": "initial",
                "event_ids": sorted(
                    {
                        str(item.get("event_id"))
                        for item in previous_runtime.get("records") or []
                        if item.get("event_id")
                    }
                ),
                "duration_seconds": previous_runtime.get(
                    "total_duration_seconds", 0.0
                ),
            }
        ]
    current_duration = round(time.perf_counter() - stage_started, 6)
    materialization_passes = previous_passes + [
        {
            "scope": (
                "initial"
                if materialize_event_ids is None
                else "post_semantic_selective_refresh"
            ),
            "event_ids": sorted(selected_event_ids),
            "duration_seconds": current_duration,
        }
    ]
    write_json(
        runtime_path,
        {
            "schema_version": "visioncortex-key-materialization-runtime/1",
            "workers": workers,
            "total_duration_seconds": round(
                sum(
                    float(item.get("duration_seconds") or 0.0)
                    for item in materialization_passes
                ),
                6,
            ),
            "detection_ledger_lookup_seconds": round(
                float(previous_runtime.get("detection_ledger_lookup_seconds") or 0.0)
                + lookup_seconds,
                6,
            ),
            "detection_ledger_passes": int(
                previous_runtime.get("detection_ledger_passes") or 0
            )
            + len(nearest_by_view),
            "archive_hierarchy_version": "2.0.0",
            "category_index": _relative(category_index_path, layout.root),
            "candidate_event_count": len(all_accepted_events),
            "accepted_event_count": sum(
                event.accepted for event in all_accepted_events
            ),
            "machine_quarantined_event_count": sum(
                not event.accepted for event in all_accepted_events
            ),
            "last_pass_materialized_event_count": len(accepted_events),
            "materialization_passes": materialization_passes,
            "records": previous_records + runtime_records,
        },
    )
    if publisher is not None:
        publisher.publish_file(runtime_path)


def extract_temporal_review_frames(
    clip_path: Path,
    output_dir: Path,
    view_id: str,
    samples_per_view: int = 3,
) -> list[tuple[str, Path]]:
    """Sample a bounded timeline from an already-made key clip.

    This avoids another seek through the original NAS video and gives the MLLM
    temporal evidence instead of asking it to infer an action from one still.
    """

    # High-risk pipette review needs several frames inside each source,
    # transport and target dwell, not just one uniform snapshot per phase.
    # Fifteen remains bounded and is read from the already-materialized clip.
    count = max(1, min(15, int(samples_per_view)))
    capture = cv2.VideoCapture(str(clip_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return []
        if count == 1:
            fractions = [0.5]
            phases = ["clip_middle"]
        elif count == 2:
            fractions = [0.2, 0.8]
            phases = ["clip_early", "clip_late"]
        elif count == 3:
            fractions = [0.15, 0.5, 0.85]
            phases = ["clip_early", "clip_middle", "clip_late"]
        elif count in {4, 5}:
            fractions = [0.1 + 0.8 * index / (count - 1) for index in range(count)]
            # These are positions in a clip, not observed action phases.
            # Its midpoint need not coincide with the selected key frame.
            if count == 4:
                phases = [
                    "clip_early", "clip_mid_early", "clip_mid_late", "clip_late"
                ]
            else:
                phases = [
                    "clip_early", "clip_mid_early", "clip_middle",
                    "clip_mid_late", "clip_late",
                ]
        else:
            fractions = [0.1 + 0.8 * index / (count - 1) for index in range(count)]
            phases = [f"timeline_{index:02d}" for index in range(1, count + 1)]
        samples: list[tuple[str, Path]] = []
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_view = _safe_slug(view_id)
        for phase, fraction in zip(phases, fractions, strict=True):
            frame_index = min(frame_count - 1, max(0, round((frame_count - 1) * fraction)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            path = output_dir / f"{safe_view}_{phase}.jpg"
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                continue
            nominal_ms = (
                round(frame_index * 1000 / fps, 3)
                if math.isfinite(fps) and fps > 0
                else "unknown"
            )
            samples.append(
                (
                    f"view_id={view_id}; sample_scope=clip_timeline; "
                    f"temporal_phase={phase}; key_clip_frame={frame_index}; "
                    f"nominal_clip_time_ms={nominal_ms}",
                    path,
                )
            )
        return samples
    finally:
        capture.release()


def _with_selected_keyframe_review_images(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    config: dict[str, Any],
    timeline_images: list[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    """Add the actual retained key frames without truncating temporal evidence.

    Raw-frame sidecars bind the image to its requested and decoder-seek time. Old
    caches without that binding remain usable as timeline evidence only.
    Never silently append one view or evict liquid-cycle context to fit a limit.
    """
    settings = config["mllm"]
    limit = int((settings.get("max_images_per_event_by_action") or {}).get(
        event.action_type.value, settings.get("max_images_per_event", 8)
    ))
    views = [view for view in event.key_frames if view != "aligned_first_third"]
    if not views or len(timeline_images) + len(views) > limit:
        return timeline_images
    root = layout.work / "key-material-annotation-inputs" / event.event_id
    selected = {}
    for sidecar in sorted(root.glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            requested = float(meta["requested_key_global_ms"])
            decoded = float(meta["decoded_key_global_ms"])
        except (OSError, ValueError, TypeError, KeyError):
            continue
        view_id = meta.get("view_id")
        image_path = sidecar.with_suffix(".jpg")
        if (
            meta.get("event_id") != event.event_id
            or view_id not in views
            or not math.isfinite(requested)
            or not math.isfinite(decoded)
            or abs(requested - event.key_global_ms) > 0.001
            or not image_path.is_file()
        ):
            continue
        if view_id in selected:
            # Multiple candidate files cannot establish which raw image is current.
            return timeline_images
        selected[view_id] = (
            f"view_id={view_id}; sample_scope=selected_keyframe; "
            f"requested_global_ms={requested:.3f}; decoded_global_ms={decoded:.3f}; "
            "time_basis=decoder_seek_target; exact_frame_pts=unverified",
            image_path,
        )
    if set(selected) != set(views):
        return timeline_images
    return [*timeline_images, *(selected[view] for view in views)]


def key_material_review_images(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    config: dict[str, Any],
) -> list[tuple[str, Path]]:
    """Select bounded temporal evidence without discarding dual-view context.

    Timeline positions never claim an action peak. When image budget permits,
    separately include both retained key frames with their recorded times.
    """

    fallback_images = [
        (view_id, layout.root / relative)
        for view_id, relative in event.key_frames.items()
        if view_id != "aligned_first_third"
    ]
    if not bool(config["mllm"].get("use_key_clip_temporal_samples", True)):
        return fallback_images
    samples_by_action = config["mllm"].get(
        "temporal_samples_per_view_by_action", {}
    )
    samples_per_view = int(
        samples_by_action.get(
            event.action_type.value,
            config["mllm"].get("temporal_samples_per_view", 3),
        )
    )
    compact_actions = {
        str(value)
        for value in config["mllm"].get("compact_aligned_temporal_actions", [])
    }
    measurements = (event.observability or {}).get("measurements") or {}
    repeated_path = measurements.get("repeated_pipette_path") or {}
    if event.action_type == ActionType.LIQUID_MOVEMENT:
        primary_view = str(repeated_path.get("view_id") or "")
        if not primary_view:
            primary_view = next(
                (
                    view_id
                    for view_id, relative in event.key_clips.items()
                    if view_id != "aligned_first_third" and relative
                ),
                "",
            )
        primary_relative = event.key_clips.get(primary_view)
        if primary_relative:
            primary = extract_temporal_review_frames(
                layout.root / primary_relative,
                layout.work
                / "mllm-temporal-samples-v2"
                / event.event_id
                / "liquid-dense-primary",
                primary_view,
                samples_per_view=int(
                    config["mllm"].get("liquid_primary_samples", 9)
                ),
            )
            required_primary_count = int(
                config["mllm"].get("liquid_primary_samples", 15)
            )
            if len(primary) == required_primary_count:
                context: list[tuple[str, Path]] = []
                for view_id, relative in event.key_clips.items():
                    if view_id in {primary_view, "aligned_first_third"}:
                        continue
                    context.extend(
                        extract_temporal_review_frames(
                            layout.root / relative,
                            layout.work
                            / "mllm-temporal-samples-v2"
                            / event.event_id
                            / "liquid-context",
                            view_id,
                            samples_per_view=int(
                                config["mllm"].get("liquid_context_samples", 3)
                            ),
                        )
                    )
                    break
                return _with_selected_keyframe_review_images(
                    layout, event, config, primary + context
                )

    if event.action_type.value in compact_actions:
        aligned_relative = event.key_clips.get("aligned_first_third")
        if aligned_relative:
            aligned = extract_temporal_review_frames(
                layout.root / aligned_relative,
                layout.work / "mllm-temporal-samples-v2" / event.event_id / "aligned",
                "aligned_first_third",
                samples_per_view=samples_per_view,
            )
            if aligned:
                return _with_selected_keyframe_review_images(
                    layout, event, config, aligned
                )

    sample_dir = layout.work / "mllm-temporal-samples-v2" / event.event_id
    images: list[tuple[str, Path]] = []
    for view_id, relative in event.key_clips.items():
        if view_id == "aligned_first_third":
            continue
        for label, path in extract_temporal_review_frames(
            layout.root / relative,
            sample_dir,
            view_id,
            samples_per_view=samples_per_view,
        ):
            images.append((label, path))

    def sample_order(item: tuple[str, Path]) -> tuple[int, str]:
        label = item[0]
        match = re.search(r"temporal_phase=([^;]+)", label)
        phase = match.group(1) if match else ""
        phases = [
            "clip_early", "clip_mid_early", "clip_middle", "clip_mid_late", "clip_late"
        ]
        if phase in phases:
            return phases.index(phase), label
        if phase.startswith("timeline_") and phase[9:].isdigit():
            return int(phase[9:]), label
        return 999, label

    images.sort(key=sample_order)
    return _with_selected_keyframe_review_images(
        layout, event, config, images
    ) if images else fallback_images


def _run_bounded_semantic_waves(
    items: Sequence[Any],
    analyze: Callable[[Any], Any],
    *,
    workers: int,
    failure_threshold: int,
) -> list[Any]:
    """Run semantic calls in waves no larger than the transport failure gate."""

    wave_size = max(1, min(int(workers), int(failure_threshold)))
    completed: list[Any] = []
    for wave_start in range(0, len(items), wave_size):
        wave = items[wave_start : wave_start + wave_size]
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [executor.submit(analyze, item) for item in wave]
            completed.extend(future.result() for future in as_completed(futures))
    return completed


def analyze_key_materials(layout: ArchiveLayout, events: Sequence[EvidenceEvent], config: dict[str, Any]) -> None:
    analyzer = ArkStepAnalyzer(config)
    accepted = [event for event in events if event.accepted]
    cache_root = layout.work / "mllm-cache" / "key-materials"

    def analyze(event: EvidenceEvent) -> tuple[EvidenceEvent, dict[str, Any]]:
        images = key_material_review_images(layout, event, config)
        semantic_evidence = event.model_dump(
            mode="json",
            exclude={"model_understanding", "key_frames", "key_clips"},
        )
        fingerprint = _semantic_fingerprint(
            "key-material",
            config,
            EVENT_SYSTEM_PROMPT,
            semantic_evidence,
            images,
        )
        persistent_cache_path = _semantic_cache_path(
            config, "key-materials", fingerprint
        )
        run_cache_path = cache_root / f"{fingerprint}.json"
        cached = None
        if _semantic_cache_reads_enabled(config):
            cached = _read_semantic_cache(persistent_cache_path, fingerprint)
            if cached is None:
                cached = _read_semantic_cache(run_cache_path, fingerprint)
        if cached is not None:
            return event, cached
        result = analyzer.analyze_event(event, images)
        if result.get("status") == "completed":
            result = _write_semantic_cache(persistent_cache_path, fingerprint, result)
            write_json(run_cache_path, result)
        return event, result

    workers = max(1, int(config["mllm"].get("workers", 4)))
    failure_threshold = max(
        1, int(config["mllm"].get("failure_circuit_breaker_threshold", 4))
    )
    semantic_results: list[tuple[str, dict[str, Any]]] = []
    try:
        # Submit one failure-bounded wave at a time.  Submitting the entire queue
        # lets every worker enter the transport before the shared circuit can
        # observe a complete failed wave, defeating the circuit breaker during
        # a provider outage.
        completed = _run_bounded_semantic_waves(
            accepted,
            analyze,
            workers=workers,
            failure_threshold=failure_threshold,
        )
        for event, result in completed:
            result = normalize_uncalibrated_hand_identity(result)
            semantic_results.append((event.event_id, result))
            event.model_understanding = result
            record_semantic_review(event, result)
    finally:
        analyzer.close()
    _raise_for_incomplete_semantic_results(
        layout,
        stage="key_material",
        results=semantic_results,
        fail_run=bool(config.get("mllm", {}).get("fail_run_on_incomplete", False)),
    )


def _move_curated_event_media(
    layout: ArchiveLayout,
    event: EvidenceEvent,
    group: ExperimentGroup,
    quarantine_root: Path,
    *,
    destination_action: ActionType | None,
) -> dict[str, Any]:
    """Move one event atomically between formal and review material trees."""

    moved: list[dict[str, str]] = []
    retained: list[dict[str, str]] = []
    experiment_folder = group.archive_folder or _safe_folder_name(group.group_id)
    for attribute, media_root, media_kind in (
        ("key_frames", layout.key_frames, "Key-Frames"),
        ("key_clips", layout.key_clips, "Key-Clips"),
    ):
        collection = dict(getattr(event, attribute))
        if not collection:
            continue
        source_files = [layout.root / relative for relative in collection.values()]
        source_directories = {path.parent.resolve(strict=True) for path in source_files}
        if len(source_directories) != 1:
            raise RuntimeError(
                f"{event.event_id} {attribute} spans multiple event directories"
            )
        source_directory = source_directories.pop()
        if not source_directory.is_relative_to(media_root.resolve(strict=True)):
            raise RuntimeError(
                f"{event.event_id} media escaped the formal key-material root"
            )
        if destination_action is None:
            destination_directory = quarantine_root / event.event_id / media_kind
        else:
            destination_directory = (
                media_root
                / experiment_folder
                / key_material_action_folder(destination_action)
                / _key_material_event_folder_name(
                    layout, experiment_folder, event
                )
            )
        if destination_directory.exists():
            resolved_destination = destination_directory.resolve(strict=True)
            if resolved_destination == source_directory:
                retained.append(
                    {
                        "media_kind": media_kind,
                        "path": _relative(source_directory, layout.root),
                        "operation": "retained_in_place",
                    }
                )
                continue
            raise RuntimeError(
                "Semantic curation destination already exists: "
                f"{destination_directory}"
            )
        destination_directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_directory), str(destination_directory))
        if destination_action is None:
            setattr(event, attribute, {})
        else:
            setattr(
                event,
                attribute,
                {
                    view_id: _relative(
                        destination_directory / Path(relative).name,
                        layout.root,
                    )
                    for view_id, relative in collection.items()
                },
            )
        moved.append(
            {
                "media_kind": media_kind,
                "from": _relative(source_directory, layout.root),
                "to": _relative(destination_directory, layout.root),
            }
        )
    return {"moved_media": moved, "retained_media": retained}


def curate_semantically_reviewed_key_materials(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    config: dict[str, Any],
    publisher: Any | None = None,
) -> tuple[list[EvidenceEvent], dict[str, Any]]:
    """Keep only semantically confirmed actions in final Key-Materials.

    CV acceptance remains in each semantic-review receipt. Rejected or
    uncertain generated media moves into a machine-quarantine tree, so recall
    disputes remain visually auditable without becoming a manual completion
    gate or being presented as confirmed key material.
    """

    if publisher is not None:
        raise RuntimeError(
            "Semantic key-material curation requires direct staging output; "
            "incremental publication cannot retract rejected candidates safely"
        )
    relabel_min_confidence = float(
        config.get("key_materials", {}).get(
            "semantic_relabel_min_confidence", 0.70
        )
    )
    known_actions = {item.value: item for item in ActionType}
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    quarantine_root = layout.key_materials / "Machine-Quarantine"
    records: list[dict[str, Any]] = []
    curated: list[EvidenceEvent] = []

    def move_event_media(
        event: EvidenceEvent,
        *,
        destination_action: ActionType | None,
    ) -> dict[str, Any]:
        return _move_curated_event_media(
            layout,
            event,
            group_by_event[event.event_id],
            quarantine_root,
            destination_action=destination_action,
        )

    for event in events:
        review = event.semantic_review or {}
        original_action = event.action_type
        original_objects = list(event.objects)
        original_supporting_views = list(event.supporting_views)
        original_supporting_roles = [
            role.value if isinstance(role, ViewRole) else str(role)
            for role in event.supporting_roles
        ]
        original_state_machine = dict(event.state_machine or {})
        original_paths = {
            "key_frames": dict(event.key_frames),
            "key_clips": dict(event.key_clips),
        }
        # A deterministic presentation repair may re-run this curation over an
        # already curated package.  Preserve the earliest CV hypothesis and
        # media/state receipts instead of replacing provenance with the final
        # semantic action on the second pass.
        provenance_action_type = str(
            review.get("pre_curation_action_type") or original_action.value
        )
        provenance_objects = list(
            review.get("pre_curation_objects") or original_objects
        )
        provenance_supporting_views = list(
            review.get("pre_curation_supporting_views")
            or original_supporting_views
        )
        provenance_supporting_roles = list(
            review.get("pre_curation_supporting_roles")
            or original_supporting_roles
        )
        provenance_state_machine = dict(
            review.get("pre_curation_state_machine") or original_state_machine
        )
        provenance_media = dict(
            review.get("pre_curation_media") or original_paths
        )
        if str(review.get("model_status") or "") != "completed":
            movement = move_event_media(event, destination_action=None)
            set_event_admission(event, "rejected")
            disposition = "machine_quarantined_semantic_unavailable"
            review.update(
                {
                    "pre_curation_verdict": (
                        review.get("pre_curation_verdict")
                        or review.get("verdict")
                        or None
                    ),
                    "pre_curation_action_type": provenance_action_type,
                    "pre_curation_objects": provenance_objects,
                    "pre_curation_supporting_views": provenance_supporting_views,
                    "pre_curation_supporting_roles": provenance_supporting_roles,
                    "pre_curation_state_machine": provenance_state_machine,
                    "pre_curation_media": provenance_media,
                    "verdict": "semantic_unavailable",
                    "final_delivery_accepted": False,
                    "final_action_type": None,
                    "final_participant_objects": [],
                    "curation_disposition": disposition,
                    "curation_policy": "semantic-final-key-material-v1",
                    "evidence_classification": "PARTIAL_EVIDENCE",
                    "retryable": True,
                    **movement,
                }
            )
            event.semantic_review = review
            records.append(
                {
                    "event_id": event.event_id,
                    "cv_action_type": provenance_action_type,
                    "cv_objects": provenance_objects,
                    "model_action_type": None,
                    "pre_curation_verdict": review.get("pre_curation_verdict"),
                    "model_confidence": 0.0,
                    "disposition": disposition,
                    "final_action_type": None,
                    "final_participant_objects": [],
                    "semantic_participant_refined": False,
                    "semantic_participant_refinement_changed": False,
                    "final_delivery_accepted": False,
                    "semantic_state_machine_rebuilt": False,
                    "evidence_classification": "PARTIAL_EVIDENCE",
                    "retryable": True,
                    **movement,
                }
            )
            continue
        model_action_value = str(review.get("model_action_type") or "")
        model_confidence = float(review.get("model_confidence") or 0.0)
        review_verdict = str(review.get("verdict") or "")
        raw_model_verdict = str(review.get("model_evidence_verdict") or "")
        participant_conflicts = _semantic_participant_conflicts(event)
        relabel_objects = _relabel_participant_objects(event)
        relabel_non_actor_objects = {
            item for item in relabel_objects if item not in {"hand", "gloved_hand"}
        }
        original_non_actor_objects = {
            item for item in original_objects if item not in {"hand", "gloved_hand"}
        }
        shared_relabel_objects = (
            original_non_actor_objects & relabel_non_actor_objects
        )
        observability = event.observability or {}
        measurements = observability.get("measurements") or {}
        both_roles = {role.value for role in event.supporting_roles} >= {
            "first_person",
            "third_person",
        }
        source_direct_cv = bool(
            observability.get("can_cv_directly_prove_action") and both_roles
        )
        minimum_box_distance = measurements.get("minimum_box_distance_norm")
        strong_dual_role_contact = bool(
            original_action == ActionType.HAND_OBJECT_CONTACT
            and both_roles
            and event.confidence >= 0.60
            and int(measurements.get("candidate_count") or 0) >= 2
            and int(measurements.get("evidence_observation_count") or 0) >= 12
            and int(measurements.get("stable_track_token_count") or 0) >= 4
            and minimum_box_distance is not None
            and float(minimum_box_distance) <= 0.01
        )
        objective_cv_preserve = bool(
            source_direct_cv or strong_dual_role_contact
        )
        direct_action_values = {
            ActionType.HAND_OBJECT_CONTACT.value,
            ActionType.OBJECT_MOVEMENT.value,
            # Direct here refers to a visually complete tool-operation chain,
            # not to visible fluid. Its independent proof gate below requires
            # source contact, transport, distinct target contact and one
            # directly supporting view.
            ActionType.PIPETTE_TRANSFER_OPERATION.value,
        }
        confirmed_support = [
            item
            for item in review.get("confirmed_action_support_by_view") or []
            if isinstance(item, dict)
            and item.get("supports_confirmed_action") is True
            and str(item.get("view_id") or "").strip()
        ]
        reviewed_view_ids = {
            str(item.get("view_id") or "").strip()
            for item in [
                *(review.get("candidate_action_support_by_view") or []),
                *(review.get("confirmed_action_support_by_view") or []),
            ]
            if isinstance(item, dict) and str(item.get("view_id") or "").strip()
        }
        event_group = group_by_event[event.event_id]
        semantic_context_both_roles = bool(
            event_group.first_person_view in reviewed_view_ids
            and event_group.third_person_view in reviewed_view_ids
        )
        strong_direct_relabel_views = sorted(
            {
                str(item["view_id"])
                for item in confirmed_support
                if float(item.get("confidence") or 0.0) >= 0.80
            }
        )
        strong_direct_relabel_consensus = bool(
            model_action_value in direct_action_values
            and len(strong_direct_relabel_views) >= 2
        )
        low_risk_direct_relabel_views = sorted(
            {
                str(item["view_id"])
                for item in confirmed_support
                if float(item.get("confidence") or 0.0) >= 0.70
            }
        )
        low_risk_direct_relabel_consensus = bool(
            model_action_value == ActionType.HAND_OBJECT_CONTACT.value
            and len(low_risk_direct_relabel_views) >= 2
            and any(
                float(item.get("confidence") or 0.0) >= 0.80
                for item in confirmed_support
            )
        )
        direct_relabel_support_views = (
            low_risk_direct_relabel_views
            if low_risk_direct_relabel_consensus
            else strong_direct_relabel_views
        )
        strongest_direct_relabel_confidence = max(
            (
                float(item.get("confidence") or 0.0)
                for item in confirmed_support
                if str(item.get("view_id") or "")
                in strong_direct_relabel_views
            ),
            default=0.0,
        )
        action_proof = (event.model_understanding or {}).get("action_proof") or {}
        state_relabel_object_complete = bool(
            set(relabel_objects) & {"bottle_cap", "tube_cap"}
            and set(relabel_objects)
            & {
                "container",
                "reagent_bottle",
                "sample_bottle",
                "tube",
            }
        )
        strict_single_view_state_relabel = bool(
            model_action_value == ActionType.CONTAINER_STATE_CHANGE.value
            and semantic_context_both_roles
            # The aggregate is intentionally allowed to reflect a weak or
            # occluded context view.  Direct state proof keeps the stricter
            # threshold below, so cross-view image quality cannot turn a
            # clearly completed single-view transition into a false negative.
            and model_confidence >= relabel_min_confidence
            and len(strong_direct_relabel_views) >= 1
            and strongest_direct_relabel_confidence >= 0.80
            and action_proof.get("container_before_state_visible") is True
            and action_proof.get("container_after_state_visible") is True
            and action_proof.get("container_state_transition_completed") is True
            # Cross-view identity conflict is retained as uncertainty, but it
            # cannot erase a completed transition directly visible in one
            # view.  This mirrors the event semantic contract.
            and state_relabel_object_complete
        )
        relabel_confidence_gate = bool(
            model_confidence >= relabel_min_confidence
            or strong_direct_relabel_consensus
            or low_risk_direct_relabel_consensus
        )
        semantic_relabel_requested = bool(
            review_verdict == "relabel_suggested"
            or (
                str(review.get("pre_curation_verdict") or "")
                == "relabel_suggested"
                and model_action_value != original_action.value
            )
            or (
                review_verdict == "confirmed"
                and raw_model_verdict in {"uncertain", "rejected"}
                and model_action_value != original_action.value
                and (
                    strong_direct_relabel_consensus
                    or low_risk_direct_relabel_consensus
                )
            )
        )
        semantic_relabel_overrides_objective_cv = bool(
            raw_model_verdict in {"uncertain", "rejected"}
            and (
                strong_direct_relabel_consensus
                or low_risk_direct_relabel_consensus
            )
        )
        relabel_base_conditions = all(
            (
                semantic_relabel_requested,
                # ``record_semantic_review`` independently derives the
                # effective verdict from the per-view support contract.  A
                # model can conservatively mark the *input candidate* as
                # uncertain while still naming a different directly visible
                # action and supporting it strongly in two views.  Accept
                # that narrow case; never override a raw rejection and never
                # apply it to state/liquid/device classes.
                raw_model_verdict == "relabel_suggested"
                or (
                    raw_model_verdict in {"uncertain", "rejected"}
                    and (
                        strong_direct_relabel_consensus
                        or low_risk_direct_relabel_consensus
                        or strict_single_view_state_relabel
                    )
                ),
                model_action_value in known_actions,
                model_action_value != original_action.value,
                relabel_confidence_gate,
                str(review.get("model_status") or "") == "completed",
                bool(relabel_non_actor_objects),
            )
        )
        # A model may see only three storyboard instants while deterministic CV
        # measures the complete interval.  Do not let a conflicting storyboard
        # overwrite a direct, dual-role tracked action.  For high-risk semantic
        # classes, also require object continuity and no cross-view conflict;
        # this prevents an unrelated nearby device from becoming a panel event.
        high_risk_relabel_safe = bool(
            model_action_value in direct_action_values
            or strict_single_view_state_relabel
            or (
                shared_relabel_objects
                and str(review.get("cross_view_consistency") or "") != "conflict"
            )
        )
        proposed_action = (
            known_actions.get(model_action_value, original_action)
            if semantic_relabel_requested
            else original_action
        )
        proof_contradictions = semantic_action_proof_contradictions(
            proposed_action,
            event.model_understanding,
            event.observability,
        )
        state_contact_support = [
            item
            for item in review.get("confirmed_action_support_by_view") or []
            if isinstance(item, dict)
            and item.get("supports_confirmed_action") is True
            and float(item.get("confidence") or 0.0) >= 0.70
            and str(item.get("view_id") or "").strip()
        ]
        state_contact_interactions = [
            item
            for item in (event.model_understanding or {}).get(
                "hand_object_interactions"
            )
            or []
            if isinstance(item, dict)
            and str(item.get("hand") or "").strip()
            and str(item.get("object") or "").strip()
            and str(item.get("contact") or "").strip()
        ]
        safe_state_to_contact_fallback = bool(
            proof_contradictions
            and original_action == ActionType.CONTAINER_STATE_CHANGE
            and model_action_value == ActionType.CONTAINER_STATE_CHANGE.value
            and str(review.get("model_status") or "") == "completed"
            and model_confidence >= 0.70
            and str(review.get("cross_view_consistency") or "") != "conflict"
            and len(
                {
                    str(item["view_id"])
                    for item in state_contact_support
                }
            )
            >= 2
            and bool(set(original_objects) & {"hand", "gloved_hand"})
            and bool(original_non_actor_objects)
            and bool(state_contact_interactions)
        )
        candidate_support_receipts = [
            item
            for item in review.get("candidate_action_support_by_view") or []
            if isinstance(item, dict)
            and str(item.get("view_id") or "").strip()
        ]
        objective_movement_unanimously_refuted = bool(
            original_action == ActionType.OBJECT_MOVEMENT
            and source_direct_cv
            and semantic_context_both_roles
            and raw_model_verdict in {"uncertain", "rejected"}
            and model_action_value not in known_actions
            and len(candidate_support_receipts) >= 2
            and all(
                item.get("supports_candidate_action") is False
                for item in candidate_support_receipts
            )
            and str(review.get("cross_view_consistency") or "") == "conflict"
        )
        movement_semantic_object_mismatch = bool(
            original_action == ActionType.OBJECT_MOVEMENT
            and source_direct_cv
            and semantic_context_both_roles
            and model_action_value == ActionType.OBJECT_MOVEMENT.value
            and str(review.get("cross_view_consistency") or "") == "conflict"
            and not shared_relabel_objects
            and not relabel_non_actor_objects
            and len(confirmed_support) == 1
            and any(
                item.get("supports_candidate_action") is False
                for item in candidate_support_receipts
            )
        )
        final_action: ActionType | None = None
        disposition = "excluded_semantically_unconfirmed"
        unmapped_interaction_participants = bool(
            proposed_action == ActionType.HAND_OBJECT_CONTACT
            and model_action_value == ActionType.HAND_OBJECT_CONTACT.value
            and str(review.get("model_status") or "") == "completed"
            and model_confidence >= relabel_min_confidence
            and original_non_actor_objects
            and not relabel_non_actor_objects
            and any(
                isinstance(item, dict)
                and str(item.get("object") or "").strip()
                and _semantic_interaction_is_direct(item)
                for item in (event.model_understanding or {}).get(
                    "hand_object_interactions", []
                ) or []
            )
        )
        if unmapped_interaction_participants:
            # A known action type does not confirm the CV target's identity.
            # For example, handling gloves cannot validate a nearby paper box.
            # Preserve the media for review instead of retaining unrelated CV
            # targets when the explicit semantic participants are unmapped.
            disposition = (
                "review_candidate_conflicting_interaction_participants"
                if participant_conflicts
                else "review_candidate_unmapped_interaction_participants"
            )
        elif safe_state_to_contact_fallback:
            # A hand visibly manipulating a cap can safely establish contact
            # even when the before/after evidence is too weak to publish the
            # higher-risk open/close state transition.  Keep only the CV
            # participant objects so background objects named by the model do
            # not leak into the final annotations.
            final_action = ActionType.HAND_OBJECT_CONTACT
            disposition = "accepted_state_safety_downclass_to_contact"
        elif proof_contradictions:
            disposition = "excluded_semantic_proof_contradiction"
        elif objective_movement_unanimously_refuted:
            disposition = "excluded_unanimously_refuted_object_movement"
        elif movement_semantic_object_mismatch:
            disposition = (
                "review_candidate_semantic_object_identity_mismatch"
            )
        elif (
            relabel_base_conditions
            and high_risk_relabel_safe
            and semantic_relabel_overrides_objective_cv
        ):
            final_action = known_actions[model_action_value]
            disposition = "accepted_strict_relabel"
        elif review_verdict == "confirmed":
            final_action = original_action
            disposition = "accepted_confirmed"
        elif objective_cv_preserve and (
            review_verdict in {"uncertain", "relabel_suggested"}
            or (source_direct_cv and review_verdict == "rejected")
        ):
            final_action = original_action
            disposition = "accepted_objective_cv_preserved_over_sparse_semantics"
        elif relabel_base_conditions and high_risk_relabel_safe:
            final_action = known_actions[model_action_value]
            disposition = "accepted_strict_relabel"

        movement: dict[str, Any] = {
            "moved_media": [],
            "retained_media": [],
        }
        if final_action is None:
            movement = move_event_media(event, destination_action=None)
            set_event_admission(event, "rejected")
        elif final_action != original_action:
            event.action_type = final_action
            event.objects = (
                original_objects
                if disposition == "accepted_state_safety_downclass_to_contact"
                else relabel_objects
            )
            movement = move_event_media(event, destination_action=final_action)
            set_event_admission(event, "formal")
        else:
            set_event_admission(event, "formal")

        semantic_participant_refined = bool(
            review.get("semantic_participant_refined")
        )
        semantic_participant_refinement_changed = False
        participant_refinement_actions = {
            ActionType.HAND_OBJECT_CONTACT,
            ActionType.OBJECT_MOVEMENT,
            ActionType.CONTAINER_STATE_CHANGE,
            ActionType.DEVICE_PANEL_OPERATION,
            ActionType.PIPETTE_TRANSFER_OPERATION,
        }
        structured_interactions = [
            item
            for item in (event.model_understanding or {}).get(
                "hand_object_interactions"
            )
            or []
            if isinstance(item, dict)
            and str(item.get("object") or "").strip()
            and str(item.get("contact") or "").strip()
        ]
        semantic_participant_refinement_safe = bool(
            final_action is not None
            and final_action in participant_refinement_actions
            and model_action_value == final_action.value
            and str(review.get("model_status") or "") == "completed"
            and model_confidence >= relabel_min_confidence
            and structured_interactions
            and relabel_non_actor_objects
        )
        if semantic_participant_refinement_safe:
            refined_objects = list(relabel_objects)
            if refined_objects != list(event.objects):
                event.objects = refined_objects
                semantic_participant_refined = True
                semantic_participant_refinement_changed = True
                if final_action == original_action:
                    movement = move_event_media(
                        event, destination_action=final_action
                    )

        semantic_relabel_direct_support = bool(
            final_action is not None
            and final_action != original_action
            and direct_relabel_support_views
            and disposition == "accepted_strict_relabel"
        )
        if semantic_relabel_direct_support:
            # The per-view semantic contract can directly prove a relabelled
            # action in views that were only context for the original CV
            # candidate. Replace the original-candidate topology with those
            # *semantic* supports so the final action never inherits direct
            # support claims from a rejected class. Aligned opposite-role
            # media remains available as explicitly labelled context.
            role_by_view: dict[str, ViewRole] = {
                candidate.view_id: candidate.role for candidate in event.candidates
            }
            group = group_by_event[event.event_id]
            if group.first_person_view:
                role_by_view.setdefault(
                    group.first_person_view, ViewRole.FIRST_PERSON
                )
            if group.third_person_view:
                role_by_view.setdefault(
                    group.third_person_view, ViewRole.THIRD_PERSON
                )
            if len(event.supporting_views) == len(event.supporting_roles) == 1:
                role_by_view.setdefault(
                    event.supporting_views[0], event.supporting_roles[0]
                )
            recall_admission = (event.observability or {}).get(
                "semantic_recall_admission"
            ) or {}
            context_view_id = str(
                recall_admission.get("context_view_id") or ""
            ).strip()
            context_role_value = str(
                recall_admission.get("context_role") or ""
            ).strip()
            if context_view_id and context_role_value in {
                item.value for item in ViewRole
            }:
                role_by_view[context_view_id] = ViewRole(context_role_value)
            supported_role_pairs = [
                (view_id, role_by_view[view_id])
                for view_id in direct_relabel_support_views
                if view_id in role_by_view
            ]
            event.supporting_views = sorted(
                view_id for view_id, _role in supported_role_pairs
            )
            event.supporting_roles = sorted(
                {
                    *(role for _view_id, role in supported_role_pairs),
                },
                key=lambda role: role.value,
            )
            observability_update = dict(event.observability or {})
            observability_update["semantic_direct_support"] = {
                "schema_version": "visioncortex-semantic-direct-support/1",
                "action_type": final_action.value,
                "view_ids": [
                    view_id for view_id, _role in supported_role_pairs
                ],
                "roles": [
                    role.value for _view_id, role in supported_role_pairs
                ],
                "source": "confirmed_action_support_by_view",
                "cv_direct_support_unchanged": True,
            }
            event.observability = observability_update

        semantic_state_machine_rebuilt = False
        if final_action is not None and (
            final_action != original_action
            or semantic_participant_refinement_changed
        ):
            # A deterministic CV state machine is valid only for the action it
            # was built from.  Once semantic adjudication relabels the event,
            # retain that receipt as pre-curation provenance and derive a new
            # final state contract from the curated action, participants and
            # direct-support topology.  Candidate track ids are deliberately
            # cleared because they belong to the rejected CV hypothesis.
            rebuilt_state = build_event_state_receipt(event, config)
            rebuilt_state["derivation"] = {
                "source": (
                    "semantic_relabel_confirmed_state_proof"
                    if final_action != original_action
                    else "semantic_participant_refinement"
                ),
                "pre_curation_action_type": original_action.value,
                "final_action_type": final_action.value,
                "model_confidence": model_confidence,
                "direct_supporting_views": list(event.supporting_views),
            }
            for transition in rebuilt_state.get("transition_trace") or []:
                transition["source"] = rebuilt_state["derivation"]["source"]
            identity = rebuilt_state.get("object_identity") or {}
            identity["track_tokens"] = []
            identity["identity_status"] = "semantic_participant_classes"
            rebuilt_state["object_identity"] = identity
            event.state_machine = rebuilt_state
            semantic_state_machine_rebuilt = True

        review.update(
            {
                "pre_curation_verdict": (
                    review.get("pre_curation_verdict")
                    or review_verdict
                    or None
                ),
                "pre_curation_action_type": provenance_action_type,
                "pre_curation_objects": provenance_objects,
                "pre_curation_supporting_views": provenance_supporting_views,
                "pre_curation_supporting_roles": provenance_supporting_roles,
                "pre_curation_state_machine": provenance_state_machine,
                "final_delivery_accepted": final_action is not None,
                "final_action_type": final_action.value if final_action else None,
                "final_participant_objects": (
                    list(event.objects) if final_action else []
                ),
                "semantic_participant_refined": semantic_participant_refined,
                "semantic_participant_refinement_changed": (
                    semantic_participant_refinement_changed
                ),
                "semantic_participant_refinement_safe": (
                    semantic_participant_refinement_safe
                ),
                "curation_disposition": disposition,
                "curation_policy": "semantic-final-key-material-v1",
                "curation_relabel_min_confidence": relabel_min_confidence,
                "semantic_state_machine_rebuilt": semantic_state_machine_rebuilt,
                "semantic_proof_contradictions": proof_contradictions,
                "semantic_participant_conflicts": participant_conflicts,
                "relabel_safety_gate": {
                    "source_direct_dual_role_cv": source_direct_cv,
                    "strong_dual_role_contact": strong_dual_role_contact,
                    "objective_cv_preserve": objective_cv_preserve,
                    "target_is_direct_action": model_action_value
                    in direct_action_values,
                    "strong_direct_relabel_view_ids": (
                        strong_direct_relabel_views
                    ),
                    "strong_direct_relabel_consensus": (
                        strong_direct_relabel_consensus
                    ),
                    "low_risk_direct_relabel_view_ids": (
                        low_risk_direct_relabel_views
                    ),
                    "low_risk_direct_relabel_consensus": (
                        low_risk_direct_relabel_consensus
                    ),
                    "strict_single_view_state_relabel": (
                        strict_single_view_state_relabel
                    ),
                    "semantic_context_both_roles": semantic_context_both_roles,
                    "semantic_reviewed_view_ids": sorted(reviewed_view_ids),
                    "state_relabel_object_complete": (
                        state_relabel_object_complete
                    ),
                    "state_relabel_model_confidence_min": (
                        relabel_min_confidence
                    ),
                    "state_relabel_direct_confidence_min": 0.80,
                    "strongest_direct_relabel_confidence": (
                        strongest_direct_relabel_confidence
                    ),
                    "relabel_confidence_gate": relabel_confidence_gate,
                    "semantic_relabel_requested": semantic_relabel_requested,
                    "semantic_relabel_overrides_objective_cv": (
                        semantic_relabel_overrides_objective_cv
                    ),
                    "shared_non_actor_objects": sorted(shared_relabel_objects),
                    "cross_view_consistency": review.get(
                        "cross_view_consistency"
                    ),
                    "high_risk_relabel_safe": high_risk_relabel_safe,
                    "objective_movement_unanimously_refuted": (
                        objective_movement_unanimously_refuted
                    ),
                    "movement_semantic_object_mismatch": (
                        movement_semantic_object_mismatch
                    ),
                },
                "pre_curation_media": provenance_media,
                **movement,
            }
        )
        if final_action is not None:
            review["verdict"] = "confirmed"
            curated.append(event)
        event.semantic_review = review
        records.append(
            {
                "event_id": event.event_id,
                "cv_action_type": provenance_action_type,
                "cv_objects": provenance_objects,
                "model_action_type": model_action_value or None,
                "pre_curation_verdict": review_verdict or None,
                "model_confidence": model_confidence,
                "disposition": disposition,
                "final_action_type": final_action.value if final_action else None,
                "final_participant_objects": (
                    list(event.objects) if final_action else []
                ),
                "semantic_participant_refined": semantic_participant_refined,
                "semantic_participant_refinement_changed": (
                    semantic_participant_refinement_changed
                ),
                "final_delivery_accepted": final_action is not None,
                "semantic_state_machine_rebuilt": semantic_state_machine_rebuilt,
                **movement,
            }
        )

    # A strict relabel can collide with an event that was already selected in
    # the corrected class. Run a second physical-action deduplication after
    # relabeling so the final user tree never contains semantic duplicates.
    actor_objects = {"hand", "gloved_hand", "lab_coat"}
    record_by_event = {item["event_id"]: item for item in records}
    duplicate_event_ids: set[str] = set()
    deduplication_records: list[dict[str, Any]] = []
    subsumed_event_ids: set[str] = set()
    state_subsumption_records: list[dict[str, Any]] = []
    separation_ms = float(
        config.get("key_materials", {}).get("minimum_separation_seconds", 1.5)
    ) * 1000.0

    def retention_rank(event: EvidenceEvent) -> tuple[int, float, float, str]:
        record = record_by_event[event.event_id]
        return (
            int(record["disposition"] == "accepted_confirmed"),
            float(record.get("model_confidence") or 0.0),
            float(event.confidence),
            event.event_id,
        )

    def state_transition_signature(event: EvidenceEvent) -> str | None:
        if event.action_type != ActionType.CONTAINER_STATE_CHANGE:
            return None
        physical_change = (event.model_understanding or {}).get(
            "physical_change"
        ) or {}
        before = str(physical_change.get("before") or "").lower()
        after = str(physical_change.get("after") or "").lower()
        # Ark commonly describes the same transition with nearby Chinese
        # variants (for example ``瓶盖盖住`` -> ``瓶口露出`` or
        # ``瓶口封闭`` -> ``瓶口开放``).  Directional semantic
        # deduplication must recognise those variants; otherwise two heavily
        # overlapping CV candidates can both be published as the same opening.
        closed_terms = (
            "closed",
            "capped",
            "盖合",
            "盖住",
            "盖着",
            "封闭",
            "密封",
            "瓶盖在瓶口",
            "瓶盖覆盖",
        )
        open_terms = (
            "open",
            "uncapped",
            "打开",
            "开启",
            "开放",
            "敞口",
            "瓶口开放",
            "瓶口露出",
            "瓶口暴露",
            "瓶盖与瓶口分离",
            "瓶口敞开",
            "瓶盖已被取下",
            "瓶盖被取下",
            "瓶盖离开瓶口",
        )
        before_closed = any(term in before for term in closed_terms)
        before_open = any(term in before for term in open_terms)
        after_closed = any(term in after for term in closed_terms)
        after_open = any(term in after for term in open_terms)
        if before_closed and after_open:
            return "opening"
        if before_open and after_closed:
            return "closing"
        return None

    for group in groups:
        bucket: list[EvidenceEvent] = []
        group_events = sorted(
            (
                event
                for event in curated
                if event.event_id in group.key_event_ids
            ),
            key=lambda event: (event.key_global_ms, event.event_id),
        )
        for event in group_events:
            duplicate_of: EvidenceEvent | None = None
            for existing in bucket:
                if existing.action_type != event.action_type:
                    continue
                shared_objects = (
                    set(existing.objects) & set(event.objects)
                ) - actor_objects
                existing_identities = event_stable_identities(existing)
                event_identities = event_stable_identities(event)
                stable_identity_available = bool(
                    existing_identities and event_identities
                )
                shared_stable_identities = (
                    existing_identities & event_identities
                )
                identity_compatible = bool(
                    shared_stable_identities
                    if stable_identity_available
                    else shared_objects
                )
                interval_overlap_ms = max(
                    0.0,
                    min(existing.global_end_ms, event.global_end_ms)
                    - max(existing.global_start_ms, event.global_start_ms),
                )
                peak_distance_ms = abs(
                    existing.key_global_ms - event.key_global_ms
                )
                minimum_duration_ms = max(
                    1.0,
                    min(
                        existing.global_end_ms - existing.global_start_ms,
                        event.global_end_ms - event.global_start_ms,
                    ),
                )
                overlap_ratio = interval_overlap_ms / minimum_duration_ms
                existing_transition = state_transition_signature(existing)
                event_transition = state_transition_signature(event)
                same_completed_state_transition = bool(
                    existing_transition
                    and existing_transition == event_transition
                    and overlap_ratio >= 0.50
                )
                if (
                    identity_compatible
                    and interval_overlap_ms > 0.0
                    and (
                        peak_distance_ms < separation_ms
                        or same_completed_state_transition
                    )
                ):
                    duplicate_of = existing
                    break
            if duplicate_of is None:
                bucket.append(event)
                continue
            retained, dropped = sorted(
                (duplicate_of, event),
                key=retention_rank,
                reverse=True,
            )
            if retained is event:
                bucket[bucket.index(duplicate_of)] = event
            duplicate_event_ids.add(dropped.event_id)
            movement = move_event_media(dropped, destination_action=None)
            set_event_admission(dropped, "rejected")
            dropped_review = dropped.semantic_review or {}
            dropped_review.update(
                {
                    "verdict": "duplicate",
                    "final_delivery_accepted": False,
                    "final_action_type": None,
                    "curation_disposition": "excluded_post_relabel_duplicate",
                    "semantic_duplicate_of": retained.event_id,
                    "moved_media": [
                        *(dropped_review.get("moved_media") or []),
                        *movement["moved_media"],
                    ],
                }
            )
            dropped.semantic_review = dropped_review
            record = record_by_event[dropped.event_id]
            record.update(
                {
                    "disposition": "excluded_post_relabel_duplicate",
                    "final_action_type": None,
                    "final_delivery_accepted": False,
                    "semantic_duplicate_of": retained.event_id,
                    "moved_media": [
                        *(record.get("moved_media") or []),
                        *movement["moved_media"],
                    ],
                }
            )
            deduplication_records.append(
                {
                    "dropped_event_id": dropped.event_id,
                    "retained_event_id": retained.event_id,
                    "final_action_type": retained.action_type.value,
                    "shared_objects": sorted(
                        (set(retained.objects) & set(dropped.objects))
                        - actor_objects
                    ),
                    "shared_stable_identities": sorted(
                        event_stable_identities(retained)
                        & event_stable_identities(dropped)
                    ),
                    "stable_identity_required_when_available": True,
                    "interval_overlap_ms": round(
                        max(
                            0.0,
                            min(retained.global_end_ms, dropped.global_end_ms)
                            - max(retained.global_start_ms, dropped.global_start_ms),
                        ),
                        3,
                    ),
                    "peak_distance_ms": round(
                        abs(retained.key_global_ms - dropped.key_global_ms), 3
                    ),
                    "interval_overlap_ratio": round(
                        max(
                            0.0,
                            min(retained.global_end_ms, dropped.global_end_ms)
                            - max(retained.global_start_ms, dropped.global_start_ms),
                        )
                        / max(
                            1.0,
                            min(
                                retained.global_end_ms - retained.global_start_ms,
                                dropped.global_end_ms - dropped.global_start_ms,
                            ),
                        ),
                        4,
                    ),
                    "state_transition_signature": state_transition_signature(
                        retained
                    ),
                    "retention_policy": (
                        "prefer_original_confirmed_then_model_confidence_then_cv_confidence"
                    ),
                }
            )

    # A confirmed, completed container transition already contains the
    # operator-to-cap/bottle contact that causes it. Publishing nested generic
    # hand-contact candidates as separate key materials adds duplicate frames
    # and invites background-instance errors without adding a distinct lab
    # fact. Keep the richer state transition and retain the lower-level
    # candidates only in the persistent audit cache.
    for group in groups:
        live_group_events = [
            event
            for event in curated
            if event.event_id in group.key_event_ids
            and event.event_id not in duplicate_event_ids
        ]
        completed_states = [
            event
            for event in live_group_events
            if event.action_type == ActionType.CONTAINER_STATE_CHANGE
            and bool(
                (
                    (event.model_understanding or {}).get("action_proof")
                    or {}
                ).get("container_state_transition_completed")
            )
        ]
        for contact in live_group_events:
            if contact.action_type != ActionType.HAND_OBJECT_CONTACT:
                continue
            contact_duration_ms = max(
                1.0, contact.global_end_ms - contact.global_start_ms
            )
            for state_event in completed_states:
                shared_objects = (
                    set(contact.objects) & set(state_event.objects)
                ) - actor_objects
                contact_identities = event_stable_identities(contact)
                state_identities = event_stable_identities(state_event)
                stable_identity_available = bool(
                    contact_identities and state_identities
                )
                shared_stable_identities = (
                    contact_identities & state_identities
                )
                identity_compatible = bool(
                    shared_stable_identities
                    if stable_identity_available
                    else shared_objects
                )
                interval_overlap_ms = max(
                    0.0,
                    min(contact.global_end_ms, state_event.global_end_ms)
                    - max(contact.global_start_ms, state_event.global_start_ms),
                )
                overlap_ratio = interval_overlap_ms / contact_duration_ms
                if not (
                    identity_compatible
                    and overlap_ratio >= 0.80
                    and state_event.global_start_ms
                    <= contact.key_global_ms
                    <= state_event.global_end_ms
                ):
                    continue
                subsumed_event_ids.add(contact.event_id)
                movement = move_event_media(contact, destination_action=None)
                set_event_admission(contact, "rejected")
                contact_review = contact.semantic_review or {}
                contact_review.update(
                    {
                        "verdict": "subsumed",
                        "final_delivery_accepted": False,
                        "final_action_type": None,
                        "curation_disposition": (
                            "excluded_subsumed_by_container_state_transition"
                        ),
                        "semantic_subsumed_by": state_event.event_id,
                        "moved_media": [
                            *(contact_review.get("moved_media") or []),
                            *movement["moved_media"],
                        ],
                    }
                )
                contact.semantic_review = contact_review
                record = record_by_event[contact.event_id]
                record.update(
                    {
                        "disposition": (
                            "excluded_subsumed_by_container_state_transition"
                        ),
                        "final_action_type": None,
                        "final_delivery_accepted": False,
                        "semantic_subsumed_by": state_event.event_id,
                        "moved_media": [
                            *(record.get("moved_media") or []),
                            *movement["moved_media"],
                        ],
                    }
                )
                state_subsumption_records.append(
                    {
                        "dropped_event_id": contact.event_id,
                        "retained_event_id": state_event.event_id,
                        "dropped_action_type": contact.action_type.value,
                        "retained_action_type": state_event.action_type.value,
                        "shared_objects": sorted(shared_objects),
                        "shared_stable_identities": sorted(
                            shared_stable_identities
                        ),
                        "stable_identity_required_when_available": True,
                        "interval_overlap_ms": round(interval_overlap_ms, 3),
                        "contact_interval_overlap_ratio": round(
                            overlap_ratio, 4
                        ),
                        "policy": (
                            "generic_contact_subsumed_by_completed_container_state_transition"
                        ),
                    }
                )
                break

    curated = [
        event
        for event in curated
        if event.event_id not in duplicate_event_ids | subsumed_event_ids
    ]

    curated_ids = {event.event_id for event in curated}
    for group in groups:
        group.key_event_ids = [
            event_id for event_id in group.key_event_ids if event_id in curated_ids
        ]
    final_annotation = _rerender_curated_participant_annotations(
        layout, curated, groups, config
    )
    write_key_material_category_index(
        layout,
        groups,
        curated,
        include_empty_categories=bool(
            config.get("archive", {}).get(
                "include_empty_action_categories", True
            )
        ),
    )
    event_by_id = {event.event_id: event for event in events}
    review_candidates: list[dict[str, Any]] = []
    for record in records:
        if record.get("final_delivery_accepted") is True:
            continue
        event_id = str(record["event_id"])
        event = event_by_id[event_id]
        media_root = quarantine_root / event_id
        review_candidates.append(
            {
                "event_id": event_id,
                "status": "machine_quarantined_not_confirmed_key_material",
                "disposition": record.get("disposition"),
                "cv_action_type": record.get("cv_action_type"),
                "cv_objects": record.get("cv_objects") or [],
                "model_action_type": record.get("model_action_type"),
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "key_global_ms": event.key_global_ms,
                "source_views": list(
                    (event.semantic_review or {}).get(
                        "pre_curation_supporting_views"
                    )
                    or event.supporting_views
                ),
                "media": sorted(
                    _relative(path, layout.root)
                    for path in media_root.rglob("*")
                    if path.is_file()
                ),
                "semantic_receipt": (
                    "JSON-Config-Files/semantic_key_material_curation.json"
                ),
                "source_reference": (
                    "Original-Experiment-Videos/Original-Video-Index.json"
                ),
            }
        )
    review_candidate_index_path = quarantine_root / "Machine-Quarantine-Index.json"
    write_json(
        review_candidate_index_path,
        {
            "schema_version": "visioncortex-machine-quarantine-index/1",
            "policy": (
                "unconfirmed candidates are automatically quarantined and indexed; "
                "they never require manual fallback and are never counted as "
                "confirmed key material"
            ),
            "candidate_count": len(review_candidates),
            "candidates": review_candidates,
        },
    )
    report = {
        "schema_version": "visioncortex-semantic-key-material-curation/1",
        "policy": "semantic-final-key-material-v1",
        "candidate_count": len(events),
        "accepted_count": len(curated),
        "excluded_count": len(events) - len(curated),
        "confirmed_count": sum(
            item["disposition"] == "accepted_confirmed"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "strict_relabel_count": sum(
            item["disposition"] == "accepted_strict_relabel"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "objective_cv_preserved_count": sum(
            item["disposition"]
            == "accepted_objective_cv_preserved_over_sparse_semantics"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "state_safety_downclass_count": sum(
            item["disposition"]
            == "accepted_state_safety_downclass_to_contact"
            and item["final_delivery_accepted"]
            for item in records
        ),
        "unsafe_relabel_excluded_count": sum(
            item["pre_curation_verdict"] == "relabel_suggested"
            and item["disposition"] == "excluded_semantically_unconfirmed"
            for item in records
        ),
        "semantic_proof_contradiction_excluded_count": sum(
            item["disposition"] == "excluded_semantic_proof_contradiction"
            for item in records
        ),
        "post_relabel_duplicate_count": len(duplicate_event_ids),
        "post_relabel_deduplication": deduplication_records,
        "state_subsumed_contact_count": len(subsumed_event_ids),
        "state_contact_subsumption": state_subsumption_records,
        "semantic_relabel_min_confidence": relabel_min_confidence,
        "unconfirmed_media_retention": "automatic_machine_quarantine_with_source_reference",
        "unconfirmed_media_formally_published": True,
        "unconfirmed_media_counted_as_confirmed": False,
        "manual_fallback_required": False,
        "review_candidate_count": len(review_candidates),
        "review_candidate_index": _relative(
            review_candidate_index_path, layout.root
        ),
        "final_annotation": {
            key: value
            for key, value in final_annotation.items()
            if key != "records"
        },
        "records": records,
    }
    write_json(layout.json_config / "semantic_key_material_curation.json", report)
    return curated, report


def _annotation_supports_curated_action(
    event: EvidenceEvent,
    annotation: dict[str, Any],
) -> bool:
    """Apply the final same-view participant contract before publication."""

    rendered_by_view = {
        str(view_id): {
            str(item).strip().lower().replace("-", "_").replace(" ", "_")
            for item in (receipt.get("rendered_classes") or [])
        }
        for view_id, receipt in (annotation.get("views") or {}).items()
        if isinstance(receipt, dict)
    }
    return bool(
        action_participant_visibility(
            event.action_type, event.objects, rendered_by_view
        )["passed"]
    )


def reconcile_visually_reviewed_participants(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    semantic_curation: dict[str, Any],
) -> tuple[list[EvidenceEvent], dict[str, Any]]:
    """Prune unsupported optional participants and quarantine invalid events.

    Semantic understanding can name several objects observed across a long
    event.  Final participant review is tied to the selected key frame.  A
    paper or cap that cannot be rendered there must not remain a claimed final
    participant.  If other rendered objects still satisfy the action contract,
    retain the event with the unsupported class removed.  Otherwise move the
    event to the formal review-candidate tree instead of failing the complete
    archive or publishing an unverifiable key material.
    """

    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    record_by_event = {
        str(record.get("event_id")): record
        for record in semantic_curation.get("records") or []
        if isinstance(record, dict) and record.get("event_id")
    }
    quarantine_root = layout.key_materials / "Review-Candidates"
    retained: list[EvidenceEvent] = []
    pruned_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    for event in events:
        annotation = (event.observability or {}).get(
            "key_material_annotation"
        ) or {}
        visual_review = annotation.get("participant_visual_review") or {}
        unsupported_classes = sorted(
            {
                str(review.get("participant_class") or "")
                for review in visual_review.get("reviews") or []
                if isinstance(review, dict)
                and (
                    (
                        review.get("status") == "completed"
                        and int(review.get("localized_view_count") or 0) == 0
                        and str(review.get("participant_class") or "")
                        in {"paper", "bottle_cap"}
                    )
                    or (
                        review.get("status") == "review_failed_quarantined"
                        and str(review.get("participant_class") or "")
                        in {"paper", "bottle_cap", "balance"}
                    )
                )
            }
        )
        previous_objects = list(event.objects)
        event.objects = [
            item for item in event.objects if item not in unsupported_classes
        ]
        supported = _annotation_supports_curated_action(event, annotation)
        if not unsupported_classes and supported:
            retained.append(event)
            continue
        group = group_by_event[event.event_id]
        destination_action = event.action_type if supported else None
        movement = _move_curated_event_media(
            layout,
            event,
            group,
            quarantine_root,
            destination_action=destination_action,
        )
        receipt = {
            "schema_version": "visioncortex-visual-participant-reconciliation/1",
            "event_id": event.event_id,
            "unsupported_classes": unsupported_classes,
            "previous_participant_objects": previous_objects,
            "final_participant_objects": list(event.objects) if supported else [],
            "action_participant_visibility_after_pruning": supported,
            "disposition": (
                "accepted_after_visual_participant_pruning"
                if supported
                else "review_candidate_missing_required_visual_participant"
            ),
            "policy": (
                "remove only visually unsupported reviewed classes; retain the "
                "event only when remaining rendered objects satisfy its action contract"
            ),
            **movement,
        }
        review = event.semantic_review or {}
        review["visual_participant_reconciliation"] = receipt
        review["final_participant_objects"] = receipt[
            "final_participant_objects"
        ]
        record = record_by_event.get(event.event_id)
        if record is not None:
            record["visual_participant_reconciliation"] = receipt
            record["final_participant_objects"] = receipt[
                "final_participant_objects"
            ]
            record["post_annotation_disposition"] = receipt["disposition"]
            record["moved_media"] = [
                *(record.get("moved_media") or []),
                *(movement.get("moved_media") or []),
            ]
            record["retained_media"] = [
                *(record.get("retained_media") or []),
                *(movement.get("retained_media") or []),
            ]
        if supported:
            event.accepted = True
            review["final_delivery_accepted"] = True
            review["post_annotation_disposition"] = receipt["disposition"]
            if record is not None:
                record["final_delivery_accepted"] = True
            pruned_records.append(receipt)
            retained.append(event)
        else:
            event.accepted = False
            review.update(
                {
                    "verdict": "visually_unconfirmed",
                    "final_delivery_accepted": False,
                    "final_action_type": None,
                    "curation_disposition": receipt["disposition"],
                }
            )
            if record is not None:
                record.update(
                    {
                        "disposition": receipt["disposition"],
                        "final_delivery_accepted": False,
                        "final_action_type": None,
                    }
                )
            excluded_records.append(receipt)
        event.semantic_review = review

    retained_ids = {event.event_id for event in retained}
    for group in groups:
        group.key_event_ids = [
            event_id for event_id in group.key_event_ids if event_id in retained_ids
        ]

    if excluded_records:
        index_path = quarantine_root / "Candidate-Index.json"
        index = (
            json.loads(index_path.read_text(encoding="utf-8-sig"))
            if index_path.is_file()
            else {
                "schema_version": "visioncortex-review-candidate-index/1",
                "policy": (
                    "unconfirmed candidates remain visible and indexed; they are "
                    "never counted as confirmed key material"
                ),
                "candidates": [],
            }
        )
        by_id = {
            str(item.get("event_id")): item
            for item in index.get("candidates") or []
            if isinstance(item, dict) and item.get("event_id")
        }
        event_by_id = {event.event_id: event for event in events}
        for receipt in excluded_records:
            event = event_by_id[receipt["event_id"]]
            media_root = quarantine_root / event.event_id
            by_id[event.event_id] = {
                "event_id": event.event_id,
                "status": "review_candidate_not_confirmed_key_material",
                "disposition": receipt["disposition"],
                "cv_action_type": (
                    (event.semantic_review or {}).get("pre_curation_action_type")
                    or event.action_type.value
                ),
                "cv_objects": (
                    (event.semantic_review or {}).get("pre_curation_objects")
                    or receipt["previous_participant_objects"]
                ),
                "model_action_type": (
                    (event.semantic_review or {}).get("model_action_type")
                ),
                "global_start_ms": event.global_start_ms,
                "global_end_ms": event.global_end_ms,
                "key_global_ms": event.key_global_ms,
                "source_views": list(event.supporting_views),
                "media": sorted(
                    _relative(path, layout.root)
                    for path in media_root.rglob("*")
                    if path.is_file()
                ),
                "semantic_receipt": (
                    "JSON-Config-Files/semantic_key_material_curation.json"
                ),
                "source_reference": (
                    "Original-Experiment-Videos/Original-Video-Index.json"
                ),
            }
        index["candidates"] = list(by_id.values())
        index["candidate_count"] = len(index["candidates"])
        write_json(index_path, index)
        semantic_curation["review_candidate_count"] = index["candidate_count"]

    records = semantic_curation.get("records") or []
    accepted_count = sum(
        bool(record.get("final_delivery_accepted"))
        for record in records
        if isinstance(record, dict)
    )
    semantic_curation.update(
        {
            "accepted_count": accepted_count,
            "excluded_count": int(semantic_curation.get("candidate_count") or 0)
            - accepted_count,
            "confirmed_count": sum(
                record.get("disposition") == "accepted_confirmed"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "strict_relabel_count": sum(
                record.get("disposition") == "accepted_strict_relabel"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "objective_cv_preserved_count": sum(
                record.get("disposition")
                == "accepted_objective_cv_preserved_over_sparse_semantics"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "state_safety_downclass_count": sum(
                record.get("disposition")
                == "accepted_state_safety_downclass_to_contact"
                and bool(record.get("final_delivery_accepted"))
                for record in records
                if isinstance(record, dict)
            ),
            "visual_participant_reconciliation": {
                "schema_version": (
                    "visioncortex-visual-participant-reconciliation-index/1"
                ),
                "policy": (
                    "prune unsupported reviewed participants and quarantine an "
                    "event if its remaining final frame cannot satisfy the action contract"
                ),
                "pruned_event_count": len(pruned_records),
                "excluded_event_count": len(excluded_records),
                "pruned_events": pruned_records,
                "excluded_events": excluded_records,
            },
        }
    )
    return retained, semantic_curation


def refresh_key_material_metadata(
    layout: ArchiveLayout,
    events: Sequence[EvidenceEvent],
    groups: Sequence[ExperimentGroup],
    transforms: dict[str, AlignmentTransform],
    archive_id: str | None = None,
) -> None:
    """Rewrite sidecars after MLLM so JSON and media never disagree."""
    group_by_event = {event_id: group for group in groups for event_id in group.key_event_ids}
    for event in events:
        group = group_by_event[event.event_id]
        for collection, artifact in (
            (event.key_frames, "key_frame"),
            (event.key_clips, "key_clip"),
        ):
            for view_id, relative in collection.items():
                media = layout.root / relative
                json_path = media.with_suffix(".json")
                aligned = view_id == "aligned_first_third"
                write_json(
                    json_path,
                    _artifact_json(
                        group,
                        event,
                        f"aligned_first_third_{artifact}" if aligned else artifact,
                        relative,
                        None if aligned else view_id,
                        transforms,
                        archive_id,
                    ),
                )


def _timestamp_rows(events: Sequence[EvidenceEvent]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if not event_is_formal(event):
            continue
        rows.append(
            {
                "event_id": event.event_id,
                "action_type": event.action_type.value,
                "global_start_ms": round(event.global_start_ms, 3),
                "global_end_ms": round(event.global_end_ms, 3),
                "key_global_ms": round(event.key_global_ms, 3),
                "objects": ", ".join(event.objects),
                "supporting_views": ", ".join(event.supporting_views),
                "supporting_roles": ", ".join(role.value for role in event.supporting_roles),
                "confidence": round(event.confidence, 5),
                "cross_view": len(event.supporting_views) > 1,
                "uncertainty": "；".join(event.uncertainty),
            }
        )
    return rows


def write_timestamp_tables(layout: ArchiveLayout, events: Sequence[EvidenceEvent], create_xlsx: bool) -> None:
    rows = _timestamp_rows(events)
    csv_path = layout.key_materials / "Key-Material-Timestamps.csv"
    fields = list(rows[0]) if rows else ["event_id", "action_type", "global_start_ms", "global_end_ms"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if create_xlsx:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Key Materials"
        sheet.append(fields)
        for row in rows:
            sheet.append([row.get(field) for field in fields])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            width = min(60, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
            sheet.column_dimensions[column[0].column_letter].width = width
        workbook.save(layout.key_materials / "Key-Material-Timestamps.xlsx")


def write_screening_notes(
    layout: ArchiveLayout,
    segments: Sequence[ExperimentSegment],
    events: Sequence[EvidenceEvent],
    rejected_candidates: Sequence[dict[str, Any]],
    all_views: Sequence[ViewInput],
) -> None:
    lines = [
        f"{PRODUCT_NAME} 有界实验片段筛选记录",
        "= 输入路数不等于输出路数；仅通过物理动作持续性与边界审计的视角会生成 MP4。",
        f"输入视角: {len(all_views)}",
        f"正式事件: {sum(event_is_formal(event) for event in events)}",
        f"待复核事件: {sum(event.formal_admission_status == 'provisional' for event in events)}",
        f"拒绝事件: {sum(not event.accepted for event in events)}",
        f"有效实验段: {len(segments)}",
        "",
    ]
    for segment in segments:
        lines.append(
            f"[{segment.segment_id}] {_time_slug(segment.global_start_ms)} - {_time_slug(segment.global_end_ms)}; "
            f"有效视角={','.join(segment.participating_views)}"
        )
        for view_id, reason in segment.rejected_views.items():
            lines.append(f"  REJECT_VIEW {view_id}: {reason}")
    for item in rejected_candidates:
        lines.append(f"REJECT_EVENT {item['event_id']}: {item['reason']} confidence={item['confidence']:.3f}")
    (layout.key_materials / "Screening-Notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_package_eval(
    root: Path,
    groups: Sequence[ExperimentGroup],
    segments: Sequence[ExperimentSegment],
    key_events: Sequence[EvidenceEvent],
    transforms: dict[str, AlignmentTransform],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    group_by_event = {
        event_id: group for group in groups for event_id in group.key_event_ids
    }
    for group in groups:
        checks.append(
            {
                "check": "experiment_group_has_atomic_boundaries",
                "id": group.group_id,
                "passed": bool(group.atomic_experiment_ids),
            }
        )
        for artifact in ("first-person", "third-person", "aligned_first_third"):
            relative = group.videos.get(artifact)
            ok = bool(relative and (root / relative).is_file() and (root / relative).stat().st_size > 0)
            checks.append(
                {"check": "experiment_video_exists", "id": f"{group.group_id}:{artifact}", "passed": ok}
            )
            metadata = group.video_json.get(artifact)
            metadata_ok = bool(metadata and (root / metadata).is_file())
            checks.append(
                {"check": "experiment_video_json_exists", "id": f"{group.group_id}:{artifact}", "passed": metadata_ok}
            )
        for view_id in (group.first_person_view, group.third_person_view):
            checks.append(
                {
                    "check": "alignment_not_failed",
                    "id": f"{group.group_id}:{view_id}",
                    "passed": transforms[view_id].state != "failed",
                }
            )
    for event in key_events:
        group = group_by_event.get(event.event_id)
        expected_keys = (
            {*_key_material_view_pair(group, event), "aligned_first_third"}
            if group is not None
            else set()
        )
        frame_keys = set(event.key_frames)
        clip_keys = set(event.key_clips)
        frame_media_complete = bool(expected_keys) and frame_keys == expected_keys and all(
            (root / event.key_frames[key]).is_file()
            and (root / event.key_frames[key]).stat().st_size > 0
            for key in expected_keys
        )
        clip_media_complete = bool(expected_keys) and clip_keys == expected_keys and all(
            (root / event.key_clips[key]).is_file()
            and (root / event.key_clips[key]).stat().st_size > 0
            for key in expected_keys
        )
        checks.append(
            {
                "check": "aligned_dual_view_key_frames_exist",
                "id": event.event_id,
                "passed": frame_media_complete,
            }
        )
        checks.append(
            {
                "check": "aligned_dual_view_key_clips_exist",
                "id": event.event_id,
                "passed": clip_media_complete,
            }
        )
        checks.append(
            {
                "check": "cross_view_or_explicit_uncertainty",
                "id": event.event_id,
                "passed": len(event.supporting_views) > 1 or bool(event.uncertainty),
            }
        )
    failures = [check for check in checks if not check["passed"]]
    empty_observation = not segments
    return {
        # Package integrity and observational sufficiency are separate claims.
        # A fully scanned input with no accepted segment is still a valid,
        # reportable package; it is not proof that no physical action occurred.
        "passed": not failures,
        "package_integrity_status": (
            "passed_empty_observation"
            if not failures and empty_observation
            else "passed"
            if not failures
            else "failed"
        ),
        "observation_status": (
            "no_accepted_observation" if empty_observation else "observations_present"
        ),
        "observation_evidence_classification": (
            "PARTIAL_EVIDENCE" if empty_observation else "PROVEN"
        ),
        "negative_action_claim_supported": False if empty_observation else None,
        "checks": checks,
        "failures": failures,
        "segment_count": len(segments),
        "experiment_group_count": len(groups),
        "key_event_count": len(key_events),
        "cross_view_event_count": sum(len(event.supporting_views) > 1 for event in key_events),
        "dual_view_material_count": sum(
            all(
                check["passed"]
                for check in checks
                if check["id"] == event.event_id
                and check["check"]
                in {"aligned_dual_view_key_frames_exist", "aligned_dual_view_key_clips_exist"}
            )
            for event in key_events
        ),
    }


def finalize_archive(
    layout: ArchiveLayout,
    manifest: RunManifest,
    infos: dict[str, VideoInfo],
    transforms: dict[str, AlignmentTransform],
    events: list[EvidenceEvent],
    segments: list[ExperimentSegment],
    groups: list[ExperimentGroup],
    key_events: list[EvidenceEvent],
    physical_changes: list[PhysicalChange],
    rejected_candidates: Sequence[dict[str, Any]],
    config: dict[str, Any],
    disk_report: dict[str, int],
    run_metrics: dict[str, Any] | None = None,
) -> RunSummary:
    write_timestamp_tables(layout, key_events, bool(config["archive"]["create_xlsx"]))
    write_screening_notes(layout, segments, events, rejected_candidates, manifest.views)
    summary = RunSummary(
        experiment_id=manifest.experiment_id,
        views=manifest.views,
        alignments=list(transforms.values()),
        events=events,
        segments=segments,
        experiment_groups=groups,
        physical_change_log=physical_changes,
        stats={
            "input_view_count": len(manifest.views),
            "output_clip_view_count": len({view for segment in segments for view in segment.participating_views}),
            "accepted_event_count": sum(event_is_formal(event) for event in events),
            "provisional_event_count": sum(
                event.formal_admission_status == "provisional" for event in events
            ),
            "rejected_event_count": sum(not event.accepted for event in events),
            "experiment_group_count": len(groups),
            "key_event_count": len(key_events),
            "disk_preflight": disk_report,
            "run_metrics": run_metrics or {},
        },
    )
    write_json(layout.json_config / "run_manifest.json", manifest.model_dump(mode="json"))
    write_json(layout.json_config / "time_alignment.json", [item.model_dump(mode="json") for item in transforms.values()])
    write_json(layout.json_config / "physical_change_log.json", [item.model_dump(mode="json") for item in physical_changes])
    write_json(layout.json_config / "evidence_package.json", summary.model_dump(mode="json"))
    if run_metrics is not None:
        write_json(layout.json_config / "run_metrics.json", run_metrics)
    normalized_events = [
        _artifact_json(
            next(group for group in groups if event.event_id in group.key_event_ids),
            event,
            "key_material_event_index",
            "",
            None,
            transforms,
            manifest.experiment_id,
        )
        for event in key_events
    ]
    write_json(
        layout.key_materials / "Key-Materials-Model-Understanding.json",
        normalized_events,
    )
    clip_analysis = {
        "experiment_id": manifest.experiment_id,
        "global_timeline": True,
        "segments": [segment.model_dump(mode="json") for segment in segments],
        "experiment_groups": [group.model_dump(mode="json") for group in groups],
        "steps": [
            {
                "event_id": event.event_id,
                "action_type": event.action_type.value,
                "start_global_ms": event.global_start_ms,
                "end_global_ms": event.global_end_ms,
                "supporting_views": event.supporting_views,
                "model_understanding": event.model_understanding,
            }
            for event in key_events
        ],
    }
    write_json(layout.json_config / "Experiment-Groups-Step-Level-Analysis.json", clip_analysis)
    evaluation = evidence_package_eval(layout.root, groups, segments, key_events, transforms)
    write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    index_manifest = build_archive_index(
        layout.root,
        manifest.experiment_id,
        normalized_events,
        events,
        groups,
        infos,
        hash_workers=int(config.get("performance", {}).get("io_workers", 4)),
    )
    summary.stats["evidence_index"] = {
        "schema_version": index_manifest["schema_version"],
        "manifest": "JSON-Config-Files/evidence_index_manifest.json",
        "counts": index_manifest["counts"],
        "fts5_enabled": index_manifest["fts5_enabled"],
    }
    index_validation = index_manifest.get("validation") or {}
    index_check = {
        "id": manifest.experiment_id,
        "check": "evidence_index_and_one_hop_registries",
        "passed": index_validation.get("passed") is True
        and index_validation.get("all_artifacts_have_integrity_receipts") is True,
        "details": index_validation,
    }
    evaluation["checks"].append(index_check)
    if not index_check["passed"]:
        evaluation["failures"].append(
            "Evidence index or one-hop artifact/evidence registry validation failed"
        )
        evaluation["passed"] = False
    write_json(layout.json_config / "evidence_package_eval.json", evaluation)
    write_json(layout.json_config / "evidence_package.json", summary.model_dump(mode="json"))
    return summary

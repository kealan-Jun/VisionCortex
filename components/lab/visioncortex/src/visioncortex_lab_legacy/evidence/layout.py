"""Lightweight historical experiment layout and path naming; no model/media imports."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from ..material_naming import (
    key_material_action_folder,
)
from ..material_naming import (
    key_material_semantic_name as _key_material_semantic_name,
)
from ..pathing import archive_relative_posix
from ..schemas import EvidenceEvent


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


def _component_budget(
    parent: Path, reserved_tail_chars: int, maximum_chars: int = 72
) -> int:
    maximum_chars = max(12, int(maximum_chars))
    parent_text = str(parent)
    windows_style_path = bool(
        re.match(r"^[A-Za-z]:[\\/]", parent_text)
    ) or parent_text.startswith("\\\\")
    if os.name != "nt" and not windows_style_path:
        return maximum_chars
    # 235 leaves headroom below classic Windows MAX_PATH for FFmpeg/OpenCV
    # temporary suffixes when LongPathsEnabled is disabled.
    return max(
        12,
        min(maximum_chars, 235 - len(str(parent)) - int(reserved_tail_chars) - 1),
    )


def _group_folder_name(
    layout: "ArchiveLayout", index: int, experiment_name: str, experiment_name_en: str
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
    layout: "ArchiveLayout", experiment_folder: str, event: EvidenceEvent
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

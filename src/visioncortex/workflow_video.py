"""Render a workflow's changing third-person camera without hiding coverage gaps."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

from .video_io import (
    _encoder_quality_arguments, _run, extract_view_clip, select_video_encoder,
)


def routed_intervals(group):
    rows = group.view_timeline or [{"start_ms": group.global_start_ms,
                                   "end_ms": group.global_end_ms,
                                   "third_person_view": group.third_person_view}]
    end = group.global_start_ms
    result = []
    for row in rows:
        if (not math.isfinite(row["start_ms"]) or not math.isfinite(row["end_ms"])
                or abs(row["start_ms"] - end) > 0.01 or row["end_ms"] <= row["start_ms"]):
            raise ValueError("Workflow view intervals must cover the clip without overlaps or gaps")
        item = dict(row)
        if result and result[-1].get("third_person_view") == item.get("third_person_view"):
            result[-1]["end_ms"] = item["end_ms"]
        else:
            result.append(item)
        end = item["end_ms"]
    if abs(end - group.global_end_ms) > 0.01:
        raise ValueError("Workflow view intervals do not cover the complete clip")
    return result


def covered_intervals(group, infos, transforms):
    """Split a declared route at actual source limits, retaining explicit gaps."""
    from .alignment import _available_global_intervals

    result = []
    for row in routed_intervals(group):
        view_id = row.get("third_person_view")
        if not view_id:
            result.append(row)
            continue
        transform, info = transforms[view_id], infos[view_id]
        # Segment clock ranges may extend beyond the physical MP4 after retiming.
        physical_start = transform.to_global(0)
        physical_end = transform.to_global(info.duration_ms)
        cursor = row["start_ms"]
        for left, right in _available_global_intervals(transform, info):
            left = max(cursor, left, physical_start)
            right = min(row["end_ms"], right, physical_end)
            if right <= left:
                continue
            if left > cursor:
                result.append({"start_ms": cursor, "end_ms": left, "third_person_view": None,
                               "reason": "source_coverage_gap", "requested_view": view_id})
            result.append({**row, "start_ms": left, "end_ms": right})
            cursor = right
        if cursor < row["end_ms"]:
            result.append({"start_ms": cursor, "end_ms": row["end_ms"], "third_person_view": None,
                           "reason": "source_coverage_gap", "requested_view": view_id})
    return result


def materialize_third_person_timeline(group, by_view, infos, transforms, destination, encoder):
    """Normalize each routed interval before concatenation; blanks are explicit."""
    intervals = covered_intervals(group, infos, transforms)
    selected = select_video_encoder(encoder)
    with tempfile.TemporaryDirectory(prefix="workflow-", dir=destination.parent) as temporary:
        root = Path(temporary)
        parts = []
        for index, item in enumerate(intervals):
            view_id = item.get("third_person_view")
            duration = (item["end_ms"] - item["start_ms"]) / 1000
            if view_id:
                start = transforms[view_id].to_local(item["start_ms"])
                end = transforms[view_id].to_local(item["end_ms"])
                if start < -0.001 or end > infos[view_id].duration_ms + 1:
                    raise ValueError("Routed view does not cover its assigned time interval")
                start = max(0.0, start)
                end = min(infos[view_id].duration_ms, end)
                source = root / f"source-{index:04d}.mp4"
                extract_view_clip(by_view[view_id], infos[view_id], source, start, end - start, encoder)
                inputs = ["-i", str(source)]
                label = view_id
            else:
                inputs = ["-f", "lavfi", "-i", "color=c=0x18353b:s=640x360:r=30"]
                label = "Source view unavailable in this interval"
            # A textfile keeps camera IDs out of FFmpeg's expression language.
            textfile = root / f"label-{index:04d}.txt"
            textfile.write_text(label, encoding="utf-8")
            filter_path = str(textfile).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            filters = ("fps=30,scale=640:360:force_original_aspect_ratio=decrease,"
                       "pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                       f"drawtext=textfile='{filter_path}':expansion=none:fontcolor=white:fontsize=18:x=12:y=18,"
                       "tpad=stop_mode=clone:stop_duration=1")
            part = root / f"part-{index:04d}.mp4"
            command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
                       "-vf", filters, "-t", f"{duration:.6f}", "-an", "-c:v", selected,
                       *_encoder_quality_arguments(selected), "-pix_fmt", "yuv420p", str(part)]
            response = _run(command, timeout=max(60, duration * 4))
            if response.returncode:
                raise RuntimeError(response.stderr.decode(errors="replace")[-2000:])
            parts.append(part)
        listing = root / "parts.txt"
        listing.write_text("\n".join("file '" + str(p).replace("'", "'\\''") + "'" for p in parts) + "\n")
        response = _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                         "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
                         str(destination)], timeout=120)
        if response.returncode:
            raise RuntimeError(response.stderr.decode(errors="replace")[-2000:])

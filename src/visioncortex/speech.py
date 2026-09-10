"""Recorder audio discovery and local, provenance-bound transcription jobs."""

from __future__ import annotations

from datetime import date
from bisect import bisect_left
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import shutil
from typing import Any


from . import speech_worker


def audio_files(view: Any) -> list[Path]:
    return [item.audio for item in (view.segments or [view]) if item.audio is not None]


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_LIMIT = 2000
_AUDIO_KEYS = (
    "task_audio_file",
    "task_audio_meta_file",
    "task_audio_ready_file",
    "task_audio_timing_file",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def enabled(settings: dict[str, Any]) -> bool:
    return bool((settings.get("speech_recognition") or {}).get("enabled"))


def _source_root(settings: dict[str, Any], *, for_archive: bool = False) -> Path:
    ingest = settings.get("collection_ingest") or {}
    # A local/no-NAS profile must return before touching an inherited path.
    if (
        (not enabled(settings) and not for_archive)
        or not ingest.get("enabled")
        or ingest.get("mode") != "directory_metadata"
    ):
        raise ValueError("当前配置未启用 NAS 录音转写")
    root = Path(ingest["source_root"])
    if not root.is_dir():
        raise OSError("NAS 采集目录不可用")
    return root.resolve()


def _directory(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or any(p in {".", ".."} for p in relative.split("/"))
    ):
        raise ValueError("无效的录音目录")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("录音目录不能包含符号链接")
    if not current.resolve().is_relative_to(root) or not current.is_dir():
        raise ValueError("录音目录不在配置的采集根目录内")
    return current


def _child(folder: Path, name: Any) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("录音说明包含无效文件名")
    path = folder / name
    if path.is_symlink() or path.resolve().parent != folder.resolve():
        raise ValueError("录音说明引用了目录外的文件")
    return path


def inspect_source(settings: dict[str, Any], relative: str, *, for_archive: bool = False) -> dict[str, Any]:
    root = _source_root(settings, for_archive=True) if for_archive else _source_root(settings)
    parts = PurePosixPath(relative).parts
    if len(parts) != 3 or not fnmatch.fnmatchcase(
        parts[0],
        (settings["collection_ingest"].get("camera_directory_glob") or "*_cam*"),
    ):
        raise ValueError("请选择设备、日期和录音分段")
    date.fromisoformat(parts[1])
    folder = _directory(root, relative)
    base = {
        "source": relative,
        "camera": parts[0],
        "date": parts[1],
        "name": parts[2],
        "status": "missing",
        "available": False,
        "duration_seconds": None,
    }
    paths = [folder / "meta.json", folder / "recording_ready.json"]
    if not all(path.is_file() for path in paths):
        return base | {"message": "缺少视频采集说明"}
    meta, recording = [speech_worker.read_json(path) for path in paths]
    if not any(meta.get(key) for key in _AUDIO_KEYS):
        return base | {"message": "该视频没有关联独立录音"}
    files = {key: _child(folder, meta.get(key)) for key in _AUDIO_KEYS if meta.get(key)}
    if any(recording.get(key) != meta.get(key) for key in _AUDIO_KEYS):
        raise ValueError("视频采集说明的音频引用不一致")
    audio_meta = speech_worker.read_json(files["task_audio_meta_file"])
    ready = speech_worker.read_json(files["task_audio_ready_file"])
    identities = [meta, recording, audio_meta, ready]
    for field in ("recording_session_id", "sender_id"):
        values = [item.get(field) for item in identities]
        if any(value in (None, "") for value in values) or any(
            value != values[0] for value in values[1:]
        ):
            raise ValueError("音频与视频的采集会话或设备不一致")
    start, end = (
        audio_meta.get("segment_window_start_global_us"),
        audio_meta.get("segment_window_end_global_us"),
    )
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start <= 0
        or end <= start
    ):
        raise ValueError("录音缺少有效全局时间范围")
    if any(
        item.get("segment_window_start_global_us") != start
        or item.get("segment_window_end_global_us") != end
        for item in (meta, recording)
    ):
        raise ValueError("音频与视频的采集时间窗不一致")
    status = str(ready.get("quality_status") or "partial")
    base.update(
        status=status,
        duration_seconds=(end - start) / 1000000,
        start_global_us=start,
        end_global_us=end,
        recording_session_id=meta["recording_session_id"],
        sender_id=meta["sender_id"],
        message="",
    )
    if (
        status != "complete"
        or ready.get("audio_valid") is not True
        or audio_meta.get("audio_valid") is not True
    ):
        return base | {
            "message": "未收到音频输入" if status == "no_input" else "录音采集不完整"
        }
    if (
        ready.get("ready") is not True
        or meta.get("closed") is not True
        or recording.get("ready") is not True
    ):
        return base | {"status": "pending", "message": "等待采集完成"}
    expected = ready.get("files") or {}
    for key in ("task_audio_file", "task_audio_meta_file", "task_audio_timing_file"):
        if key not in files:
            raise ValueError("完整录音缺少文件引用")
        path = files[key]
        spec = expected.get(path.name) or {}
        if not re.fullmatch(
            r"[a-f0-9]{64}", str(spec.get("sha256") or "")
        ) or not isinstance(spec.get("size"), int):
            raise ValueError("音频回执缺少文件哈希")
        if not path.is_file() or path.stat().st_size != spec["size"]:
            raise ValueError("音频文件与采集回执大小不一致")
    # Discovery hashes small sidecars only. Full audio/model hashing is a queued runtime action.
    sealed = {
        path.name: speech_worker.file_record(path)
        for path in [*paths, files["task_audio_ready_file"]]
    }
    for key in ("task_audio_meta_file", "task_audio_timing_file"):
        if key not in files:
            raise ValueError("完整录音缺少文件引用")
        path = files[key]
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("音频时钟或说明文件过大")
        if speech_worker.file_record(path) != expected[path.name]:
            raise ValueError("音频采集说明哈希不匹配")
    sealed.update(
        {
            files[key].name: expected[files[key].name]
            for key in (
                "task_audio_file",
                "task_audio_meta_file",
                "task_audio_timing_file",
            )
        }
    )
    audio_path = files["task_audio_file"]
    if audio_path.stat().st_size > int(
        (settings.get("speech_recognition") or {}).get(
            "max_source_bytes", 128 * 1024 * 1024
        )
    ):
        raise ValueError("该录音超过单段读取预算")
    playback = [
        {
            "event_type": item.get("event_type"),
            "source": str(item.get("source") or "")[:80],
            "global_timestamp_us": item["global_timestamp_us"],
        }
        for item in audio_meta.get("playback_events", [])
        if isinstance(item, dict) and isinstance(item.get("global_timestamp_us"), int)
    ]
    private = {
        "folder": str(folder),
        "resolved_folder": str(folder.resolve()),
        "files": sealed,
        "audio_file": audio_path.name,
        "video_file": _child(folder, meta.get("rgb_file")).name,
        "audio_stat": {
            "size": audio_path.stat().st_size,
            "mtime_ns": audio_path.stat().st_mtime_ns,
        },
        "playback_events": playback,
        "camera": parts[0],
        "recording_session_id": meta["recording_session_id"],
        "sender_id": meta["sender_id"],
        "start_global_us": audio_meta.get("first_audio_global_us", start),
        "end_global_us": end,
        "relative_source": relative,
        "global_timing": "PARTIAL_EVIDENCE",
    }
    return base | {"available": True, "signature": _digest(private), "_sealed": private}


def probe_audio(path: Path) -> dict[str, Any] | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = next(
        (item for item in data.get("streams", []) if item.get("codec_type") == "audio"),
        None,
    )
    if stream is None:
        return None
    duration = float(
        stream.get("duration") or data.get("format", {}).get("duration") or 0
    )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("录音缺少有效时长")
    return {
        "duration_seconds": duration,
        "codec": stream.get("codec_name"),
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
    }


def discover(config: dict[str, Any], manifest: Any, *, include_untranscribed: bool = False) -> list[dict[str, Any]]:
    """Discover only sources belonging to this submitted experiment, never scan NAS."""
    if not enabled(config) and not include_untranscribed:
        return []
    items = []
    ingest = config.get("collection_ingest") or {}
    nas_enabled = ingest.get("enabled") and ingest.get("mode") == "directory_metadata"
    for view in manifest.views:
        for ordinal, part in enumerate(view.segments or [view]):
            video = part.video
            item = {
                "id": f"{view.view_id}-{ordinal + 1:04d}",
                "view_id": view.view_id,
                "segment_ordinal": ordinal,
                "video": str(video),
                "status": "no_audio",
                "available": False,
                "alignment": "NOT_PROVEN",
                "audio_offset_ms": None,
            }
            if part.audio is None and nas_enabled:
                # Use the frozen video path's own explicit references. Do not
                # infer a nearby recording merely from a timestamp or filename.
                root = _source_root(config, for_archive=True) if include_untranscribed else _source_root(config)
                if video.parent.resolve().is_relative_to(root):
                    relative = video.parent.resolve().relative_to(root).as_posix()
                    # Upload archives may live below the same NAS root. Only
                    # recorder-native camera/date/segment folders own automatic
                    # audio references; other videos use their explicit audio
                    # attachment or embedded track below.
                    parts = PurePosixPath(relative).parts
                    recorder_folder = len(parts) == 3 and fnmatch.fnmatchcase(
                        parts[0], ingest.get("camera_directory_glob") or "*_cam*"
                    )
                    if recorder_folder:
                        try:
                            date.fromisoformat(parts[1])
                        except ValueError:
                            recorder_folder = False
                    paired = (
                        inspect_source(config, relative, for_archive=True) if include_untranscribed and recorder_folder else
                        inspect_source(config, relative)
                        if recorder_folder else {"status": "missing"}
                    )
                    if paired.get("status") != "missing":
                        item.update(paired)
                        if paired["available"]:
                            from .alignment import read_timestamp_csv_endpoints

                            sealed = item["_sealed"]
                            if sealed["video_file"] != video.name:
                                raise ValueError("音频元数据未引用本次提交的视频")
                            if part.timestamps_csv is not None:
                                first = read_timestamp_csv_endpoints(
                                    part.timestamps_csv, 1.0
                                )[0]
                                if first.source_ms is not None:
                                    item["audio_offset_ms"] = (
                                        sealed["start_global_us"] / 1000
                                        - first.source_ms
                                    )
                                    item["alignment"] = "PARTIAL_EVIDENCE"
                                    item["alignment_basis"] = "recorder_shared_clock"
                        items.append(item)
                        continue
            if include_untranscribed and not enabled(config) and part.audio is None:
                item.update(status="not_transcribed", message="未启用转写，原视频音轨随视频保留")
                items.append(item)
                continue
            path = part.audio or video
            if path.is_symlink() or not path.is_file():
                raise ValueError("实验录音源文件不可用")
            before = (path.stat().st_size, path.stat().st_mtime_ns)
            info = probe_audio(path)
            if info is None:
                if part.audio is not None:
                    raise ValueError("附带录音文件没有音轨")
                items.append(item)
                continue
            # This is part of the queued real run; explicit input hashes are
            # recorded even when the containing MP4 is large.
            source = {
                "folder": str(path.parent),
                "resolved_folder": str(path.parent.resolve()),
                "audio_file": path.name,
                "files": {path.name: speech_worker.file_record(path)},
                "start_global_us": None,
                "playback_events": [],
            }
            if before != (path.stat().st_size, path.stat().st_mtime_ns):
                raise ValueError("读取期间录音输入已变化")
            offset = part.audio_offset_ms if part.audio is not None else 0.0
            item.update(
                info,
                available=True,
                status="complete",
                _sealed=source,
                signature=_digest(source),
                audio_offset_ms=offset,
                alignment="PARTIAL_EVIDENCE" if offset is not None else "NOT_PROVEN",
                alignment_basis="user_declared_offset"
                if part.audio is not None
                else "embedded_media_timeline",
            )
            items.append(item)
    return items


def runtime_request(config: dict[str, Any]) -> dict[str, Any]:
    options = config.get("speech_recognition") or {}
    model_directory = Path(options.get("model_directory") or "")
    python = Path(options.get("python_executable") or sys.executable)
    registry_path = Path(
        options.get("model_registry") or "configs/models/speech-recognition.json"
    )
    if not registry_path.is_absolute():
        registry_path = REPOSITORY / registry_path
    registry = speech_worker.read_json(registry_path)
    if (
        registry.get("schema_version") != "visioncortex-speech-model/1"
        or not options.get("model_directory")
        or not model_directory.is_dir()
        or not python.is_file()
    ):
        raise ValueError("请配置本机语音模型与 Python 环境")
    if not {"model.bin", "config.json", "tokenizer.json", "vocabulary.txt"}.issubset(
        registry.get("files", {})
    ):
        raise ValueError("语音模型注册表不完整")
    threads = int(options.get("cpu_threads", 4))
    maximum = float(options.get("max_audio_seconds", 1800))
    if not 1 <= threads <= 16 or not 1 <= maximum <= 1800:
        raise ValueError("语音运行预算无效")
    return {
        "schema_version": "visioncortex-speech-request/1",
        "model": registry,
        "model_directory": str(model_directory.resolve()),
        "python_executable": str(python.absolute()),
        "worker_sha256": speech_worker.sha256(Path(speech_worker.__file__)),
        "repository": {
            "implementation_identity": "worker_sha256",
            "release_certified": False,
        },
        "max_audio_seconds": maximum,
        "cpu_threads": threads,
        "language": str(options.get("language") or "zh"),
    }


def invoke(directory: Path, request: dict[str, Any], *, reuse: bool = True) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    speech_worker.atomic_json(request_path, request)
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        OMP_NUM_THREADS=str(request["cpu_threads"]),
    )
    result = subprocess.run(
        [
            request["python_executable"],
            str(Path(speech_worker.__file__)),
            str(request_path),
            *([] if reuse else ["--cold"]),
        ],
        capture_output=True,
        check=False,
        timeout=max(120, 30 * (request["end_seconds"] - request["start_seconds"])),
        env=environment,
    )
    if result.returncode:
        raise ValueError("实验录音转写失败，请检查源文件、模型哈希和语音环境")
    receipt = speech_worker.read_json(directory / "receipt.json")
    if receipt.get("status") != "completed" or receipt.get(
        "request_sha256"
    ) != speech_worker.sha256(request_path):
        raise ValueError("录音转写回执无效")
    speech_worker.verify_files(directory, receipt["artifacts"])
    execution = speech_worker.read_json(directory / "execution.json")
    if execution.get("request_sha256") != receipt["request_sha256"] or execution.get("receipt_sha256") != speech_worker.sha256(directory / "receipt.json"):
        raise ValueError("录音执行回执不匹配")
    return {**receipt, "execution": execution}


def recorder_time_mapper(path: Path, fps: float):
    """Map shared recorder time onto stored RGB samples, including retimed MP4."""
    from .alignment import read_timestamp_csv

    points = read_timestamp_csv(path, fps)
    points = [
        point
        for point in points
        if point.source_ms is not None and point.clock_sync_valid is not False
    ]
    if len(points) < 2 or any(
        b.source_ms <= a.source_ms or b.local_ms <= a.local_ms
        for a, b in zip(points, points[1:], strict=False)
    ):
        raise ValueError("录音关联的视频时钟缺失或倒退")
    times = [point.source_ms for point in points]

    def to_local(epoch_ms: float) -> float | None:
        index = bisect_left(times, epoch_ms)
        if index < len(times) and times[index] == epoch_ms:
            return points[index].local_ms
        if index == 0 or index == len(times):
            return None
        left, right = points[index - 1], points[index]
        if right.source_ms - left.source_ms > 2000:
            return None
        ratio = (epoch_ms - left.source_ms) / (right.source_ms - left.source_ms)
        return left.local_ms + ratio * (right.local_ms - left.local_ms)

    return to_local


def run_stage(
    config: dict[str, Any],
    manifest: Any,
    layout: Any,
    infos: Any,
    transforms: Any,
    progress: Any = None,
    publisher: Any = None,
) -> dict[str, Any]:
    """Produce derived audio and ASR in the same experiment's archive transaction."""
    index = {
        "schema_version": "visioncortex-experiment-speech/1",
        "experiment_id": manifest.experiment_id,
        "status": "running" if enabled(config) else "disabled",
        "sources": [],
        "physical_action_confirmation": False,
        "accuracy": "NOT_PROVEN",
        "integration_sha256": speech_worker.sha256(Path(__file__)),
        "alignment_file": "JSON-Config-Files/time_alignment.json",
        "alignment_file_sha256": speech_worker.sha256(
            layout.json_config / "time_alignment.json"
        )
        if (layout.json_config / "time_alignment.json").is_file()
        else None,
    }
    executions = []
    index_path = layout.json_config / "speech.json"
    # A full run supersedes derived search/interpretation revisions. Retain their
    # exact bytes for audit before the new transcript identity becomes visible.
    for name in ("speech_search.json", "speech_search_receipt.json", "speech_group_understanding.json", "speech_timeline.json"):
        previous = layout.json_config / name
        if previous.is_file():
            import time
            retained = layout.json_config / "Stage-Refreshes" / f"{time.time_ns()}-superseded-{name}"
            retained.parent.mkdir(parents=True, exist_ok=True)
            previous.replace(retained)
    speech_worker.atomic_json(index_path, index)
    try:
        sources = discover(config, manifest) if enabled(config) else discover(config, manifest, include_untranscribed=True)
        from .speech_archive import preserve
        for item in sources:
            public = {
                key: value for key, value in item.items() if not key.startswith("_")
            }
            public["chunks"] = []
            index["sources"].append(public)
            if item["available"]:
                public["original"] = preserve(layout.root, item)
            speech_worker.atomic_json(index_path, index)
            if publisher is not None:
                if (layout.key_materials / "Experiment-Audio").is_dir():
                    publisher.publish_directory("Key-Materials/Experiment-Audio")
                publisher.publish_file(index_path)
        if not enabled(config):
            index["archive_status"] = "saved" if any(s.get("original", {}).get("status") == "saved" for s in index["sources"]) else "no_separate_recording"
            speech_worker.atomic_json(index_path, index)
            if publisher is not None:
                publisher.publish_file(index_path)
            return index
        # All original recordings survive even if ASR initialization fails.
        runtime = runtime_request(config) if any(item["available"] for item in sources) else None
        for item, public in zip(sources, index["sources"], strict=True):
            if not item["available"]:
                continue
            clock_mapper = None
            if item.get("alignment_basis") == "recorder_shared_clock":
                view = next(
                    view for view in manifest.views if view.view_id == item["view_id"]
                )
                part = (view.segments or [view])[item["segment_ordinal"]]
                info = infos[item["view_id"]]
                physical_info = (
                    info.segments[item["segment_ordinal"]] if info.segments else info
                )
                clock_record = speech_worker.file_record(part.timestamps_csv)
                clock_mapper = recorder_time_mapper(
                    part.timestamps_csv, physical_info.fps
                )
                if part.timestamps_csv.parent.resolve() != Path(
                    item["_sealed"]["resolved_folder"]
                ):
                    raise ValueError("录音时钟必须来自同一冻结采集目录")
                item["_sealed"]["files"][part.timestamps_csv.name] = clock_record
                public["video_clock_sha256"] = clock_record["sha256"]
            duration = item["duration_seconds"]
            count = math.ceil(duration / runtime["max_audio_seconds"])
            for chunk_index in range(count):
                start = chunk_index * runtime["max_audio_seconds"]
                end = min(duration, start + runtime["max_audio_seconds"])
                if progress:
                    progress(
                        f"正在转写 {item['view_id']} 第 {item['segment_ordinal'] + 1} 段录音（{chunk_index + 1}/{count}）"
                    )
                request = {
                    **runtime,
                    "source": item["_sealed"],
                    "start_seconds": start,
                    "end_seconds": end,
                }
                cache_root = Path((config.get("storage") or {}).get("local_cache_root") or layout.work)
                namespace = _digest({"namespace": config.get("project", {}).get("cache_namespace")})
                work = cache_root / "speech-v1" / namespace / _digest(request)
                receipt = invoke(work, request, reuse=config.get("project", {}).get("cache_mode") != "cold")
                executions.append({"source_id": item["id"], "chunk_index": chunk_index,
                                   **receipt.get("execution", {})})
                speech_worker.atomic_json(layout.json_config / "speech_execution.json", {"chunks": executions})
                transcript = speech_worker.read_json(
                    work / "transcript.json", 32 * 1024 * 1024
                )
                info = infos[item["view_id"]]
                virtual_start = (
                    info.segments[item["segment_ordinal"]].virtual_start_ms
                    if info.segments
                    else 0
                )
                transform = transforms[item["view_id"]]
                for row in transcript["segments"]:
                    row["speaker_identity"] = {"status": "unknown", "person_id": None, "name": None}
                    row["source_id"] = item["id"]
                    epoch = item["_sealed"].get("start_global_us")
                    row["recorded_start_global_us"] = round(epoch + row["start_seconds"] * 1000000) if epoch is not None else None
                    row["recorded_end_global_us"] = round(epoch + row["end_seconds"] * 1000000) if epoch is not None else None
                    row["playback_start_seconds"] = row["start_seconds"] - start
                    row["playback_end_seconds"] = row["end_seconds"] - start
                    # Recorder clocks establish pairing, but variable video
                    # pacing and device drift still require real sync review.
                    for edge in ("start", "end"):
                        seconds = row[f"{edge}_seconds"]
                        local = (
                            clock_mapper(
                                item["_sealed"]["start_global_us"] / 1000
                                + seconds * 1000
                            )
                            if clock_mapper is not None
                            else item["audio_offset_ms"] + seconds * 1000
                            if item["audio_offset_ms"] is not None
                            else None
                        )
                        row[f"aligned_{edge}_ms"] = (
                            transform.to_global(virtual_start + local)
                            if local is not None
                            else None
                        )
                transcript.update(
                    alignment=item["alignment"],
                    alignment_basis=item.get("alignment_basis"),
                )
                relative = (
                    Path("Key-Materials")
                    / "Experiment-Audio"
                    / item["id"]
                    / f"{chunk_index + 1:04d}"
                )
                target = layout.root / relative
                target.mkdir(parents=True, exist_ok=True)
                for name in (
                    "audio.m4a",
                    "transcript.srt",
                    "transcript.vtt",
                    "transcript.txt",
                    "transcript.json",
                    "receipt.json",
                    "request.json",
                ):
                    temporary = target / (name + ".partial")
                    shutil.copyfile(work / name, temporary)
                    os.replace(temporary, target / name)
                speech_worker.atomic_json(
                    target / "aligned-transcript.json", transcript
                )
                artifacts = {
                    name: {
                        **speech_worker.file_record(target / name),
                        "path": (relative / name).as_posix(),
                    }
                    for name in (
                        "audio.m4a",
                        "transcript.json",
                        "transcript.srt",
                        "transcript.vtt",
                        "transcript.txt",
                        "aligned-transcript.json",
                        "receipt.json",
                        "request.json",
                    )
                }
                public["chunks"].append(
                    {
                        "id": f"{item['id']}-{chunk_index + 1:04d}",
                        "start_seconds": start,
                        "end_seconds": end,
                        "outcome": receipt["outcome"],
                        "segment_count": len(transcript["segments"]),
                        "files": artifacts,
                    }
                )
                speech_worker.atomic_json(index_path, index)
                if publisher is not None:
                    publisher.publish_directory(relative)
                    publisher.publish_file(index_path)
        index["status"] = "completed"
    except Exception:
        index["status"] = "failed"
        speech_worker.atomic_json(index_path, index)
        raise
    speech_worker.atomic_json(index_path, index)
    if index.get("status") == "completed":
        from .speech_search import build
        build(layout.root)
    return index


def archive_result(
    root: Path,
    query: str = "",
    offset: int = 0,
    limit: int = 100,
    chunk_id: str | None = None,
    *, fold: bool = False, phrase: str | None = None, aliases: bool = True,
    hint: str | None = None,
) -> dict[str, Any]:
    path = root / "JSON-Config-Files" / "speech.json"
    if not path.exists():
        return {
            "status": "not_available",
            "sources": [],
            "segments": [],
            "total": 0,
            "next_offset": None,
        }
    identity = speech_worker.sha256(path)
    payload = speech_worker.read_json(path, 16 * 1024 * 1024)
    understanding_path = root / "JSON-Config-Files/speech_understanding.json"
    if understanding_path.is_file():
        understanding = speech_worker.read_json(understanding_path, 16 * 1024 * 1024)
        if understanding.get("index_sha256") != speech_worker.sha256(path):
            raise ValueError("录音理解引用了其他转写版本")
        payload["model_understanding"] = understanding
    from .speech_search import search
    capture_path = root / "JSON-Config-Files/capture_quality.json"
    if capture_path.is_file():
        payload["capture_quality"] = speech_worker.read_json(capture_path)
    matches = search(root, payload, query, offset, limit, chunk_id,
                     fold=fold, phrase=phrase, aliases=aliases, hint=hint)
    if speech_worker.sha256(path) != identity or matches["search"]["speech_sha256"] != identity:
        raise ValueError("查询期间转写版本变化")
    return {**payload, **matches}

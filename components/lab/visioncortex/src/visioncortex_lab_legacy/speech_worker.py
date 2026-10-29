"""Offline ASR worker. May run in an isolated interpreter without the Web stack.

Only explicit, sealed recorder audio is accepted. This module never downloads
models, contacts a provider, copies source media, or confirms physical actions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def read_json(path: Path, limit: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("录音说明不能是符号链接")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("录音说明超过大小限制")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("录音说明格式错误")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": sha256(path)}


def resolve_source(source):
    """Resolve a mutable location only against an immutable content binding.

    The request and all utterance/chunk checkpoints keep the same hash when
    capture media is later read from its independently verified MetaVideo copy.
    The normal verify_files gate still verifies actual bytes at this location.
    """
    if 'content_locator' not in source:
        return source
    located = read_json(Path(source['content_locator']))
    if (located.get('binding') != source['binding'] or len(located.get('files', {})) != 1
            or located.get('audio_file') not in located['files']
            or located['files'][located['audio_file']] != source['content']):
        raise ValueError('Audio location does not match sealed content binding')
    return located | {'start_global_us': source['start_global_us'], 'playback_events': []}


def verify_files(root: Path, files: dict[str, Any]) -> None:
    for name, expected in files.items():
        if Path(name).name != name or name in {"", ".", ".."} or "\\" in name:
            raise ValueError("文件清单包含无效路径")
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != root.resolve()
        ):
            raise ValueError("录音或模型文件不可用")
        if (
            path.stat().st_size != expected["size"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError("录音或模型文件与冻结清单不一致")


def timestamp(seconds: float, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    whole, fraction = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{separator}{fraction:03d}"


def write_transcript(
    output: Path, request: dict[str, Any], segments: list[dict[str, Any]]
) -> None:
    previous = -1.0
    for segment in segments:
        start, end = segment["start_seconds"], segment["end_seconds"]
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and request["start_seconds"] <= start < end <= request["end_seconds"] + 0.05
            and start >= previous
        ):
            raise ValueError("转写时间戳无效")
        previous = start
    payload = {
        "schema_version": "visioncortex-speech-transcript/1",
        "source": request["source"],
        "model": request["model"],
        "range": {
            "start_seconds": request["start_seconds"],
            "end_seconds": request["end_seconds"],
        },
        "segments": segments,
        "human_reviewed": False,
        "evidence_kind": "machine_transcribed_speech",
        "physical_action_confirmation": False,
        "accuracy": "NOT_PROVEN",
        "global_timing": "PARTIAL_EVIDENCE",
        "subtitle_timeline": "derived_audio_chunk_seconds",
    }
    atomic_json(output / "transcript.json", payload)
    plain, srt, vtt = [], [], ["WEBVTT", ""]
    for index, segment in enumerate(segments, 1):
        text = segment["text"].strip()
        plain.append(
            f"[{timestamp(segment['start_seconds'], '.')}–{timestamp(segment['end_seconds'], '.')}] {text}"
        )
        # Cue payloads are plain text, never HTML or new cue declarations.
        cue = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\r", " ")
            .replace("\n", " ")
        )
        for destination, separator in ((srt, ","), (vtt, ".")):
            destination.extend(
                [
                    str(index),
                    f"{timestamp(segment['start_seconds'] - request['start_seconds'], separator)} --> {timestamp(segment['end_seconds'] - request['start_seconds'], separator)}",
                    cue,
                    "",
                ]
            )
    for name, lines in (
        ("transcript.txt", plain),
        ("transcript.srt", srt),
        ("transcript.vtt", vtt),
    ):
        path = output / name
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def execute(request_path: Path, *, reuse: bool = True) -> None:
    # Set these before importing inference libraries: the worker is offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    request = read_json(request_path)
    output = request_path.parent
    if str(Path(sys.executable).absolute()) != request["python_executable"]:
        raise ValueError("转写解释器与冻结环境不一致")
    started = time.perf_counter()
    request_hash = sha256(request_path)
    if sha256(Path(__file__)) != request["worker_sha256"]:
        raise ValueError("转写执行代码已变化，请重新提交")
    source = resolve_source(request["source"])
    folder = Path(source["folder"])
    # Check resolved location again after queueing, including parent symlinks.
    if folder.resolve() != Path(source["resolved_folder"]):
        raise ValueError("录音目录已变化")
    verify_files(folder, source["files"])
    model_root = Path(request["model_directory"])
    verify_files(model_root, request["model"]["files"])
    duration = request["end_seconds"] - request["start_seconds"]
    if not 0 < duration <= request["max_audio_seconds"]:
        raise ValueError("转写范围超过单段运行预算")
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    import faster_whisper.vad as vad_module
    import numpy as np

    runtime = {
        name: importlib.metadata.version(name)
        for name in ("faster-whisper", "ctranslate2", "av", "numpy", "onnxruntime")
    }
    if runtime["faster-whisper"] != "1.2.1":
        raise ValueError("转写环境需要 faster-whisper 1.2.1")
    invocation = {
        "schema_version": "visioncortex-speech-receipt/1",
        "status": "running",
        "request_sha256": request_hash,
        "repository": request["repository"],
        "worker_sha256": request["worker_sha256"],
        "runtime": runtime,
        "python": sys.executable,
        "python_version": sys.version,
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": request["cpu_threads"],
        "model": request["model"],
        "source_hashes_verified": True,
        "audio_uploaded": False,
        "source_copy_bytes": 0,
        "vad_assets": {
            p.name: file_record(p)
            for p in (Path(vad_module.__file__).parent / "assets").glob("*.onnx")
        },
        "model_invocations": 0,
        "reused_utterances": 0,
    }
    checkpoint_runtime_sha256 = hashlib.sha256(
        json.dumps(
            {
                "runtime": runtime,
                "vad_assets": invocation["vad_assets"],
                "python_version": sys.version,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    invocation["runtime_sha256"] = checkpoint_runtime_sha256
    previous = completed_cache(output, request_hash, checkpoint_runtime_sha256) if reuse else None
    if previous is not None:
        # The original receipt is immutable, so verified reuse does not change
        # downstream semantic identity. Current execution has its own receipt.
        atomic_json(output / "execution.json", {
            "request_sha256": request_hash, "receipt_sha256": sha256(output / "receipt.json"),
            "cache_reused": True, "model_invocations": 0, "actual_asr_invocation": "NOT_PROVEN",
            "decoded_audio": False, "wall_seconds": round(time.perf_counter() - started, 3)})
        return
    invocation.update(cache_reused=False, decoded_audio=True)
    atomic_json(output / "receipt.json", invocation)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(request["start_seconds"]),
        "-i",
        str(folder / source["audio_file"]),
        "-t",
        str(duration),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-threads",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    decoded = subprocess.run(
        command, capture_output=True, timeout=max(30, duration), check=False
    )
    if decoded.returncode or not decoded.stdout:
        raise ValueError("音频无法解码")
    audio = np.frombuffer(decoded.stdout, dtype="<f4")
    if abs(len(audio) / 16000 - duration) > 0.1 or not np.all(np.isfinite(audio)):
        raise ValueError("音频实际时长与转写范围不一致")
    # Archive a derived, normalized listening copy; preserve the source hash.
    preview = output / "audio.partial.m4a"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "48k",
            "-movflags",
            "+faststart",
            str(preview),
        ],
        input=decoded.stdout,
        capture_output=True,
        timeout=max(30, duration),
        check=False,
    )
    if encoded.returncode:
        raise ValueError("录音试听文件生成失败")
    os.replace(preview, output / "audio.m4a")
    options = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        max_speech_duration_s=20,
    )
    regions = get_speech_timestamps(audio, options)
    model = None
    segments: list[dict[str, Any]] = []
    checkpoint_root = output / "utterances"
    checkpoint_root.mkdir(exist_ok=True)
    for index, region in enumerate(regions):
        begin = max(0, region["start"] - 3200)
        end = min(len(audio), region["end"] + 3200)
        checkpoint = checkpoint_root / f"{index:06d}.json"
        cached = read_json(checkpoint) if checkpoint.is_file() else {}
        cached_digest = hashlib.sha256(
            json.dumps(
                cached.get("segments"), ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest()
        if (
            reuse and cached.get("request_sha256") == request_hash
            and cached.get("samples") == [begin, end]
            and cached.get("segments_sha256") == cached_digest
            and cached.get("runtime_sha256") == checkpoint_runtime_sha256
        ):
            rows = cached["segments"]
            invocation["reused_utterances"] += 1
        else:
            if model is None:
                model = WhisperModel(
                    str(model_root),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=request["cpu_threads"],
                    num_workers=1,
                    local_files_only=True,
                )
            iterator, _ = model.transcribe(
                audio[begin:end],
                language=request["language"],
                beam_size=5,
                temperature=0.0,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            rows = []
            for item in iterator:
                offset = request["start_seconds"] + begin / 16000
                start = offset + item.start
                finish = min(request["start_seconds"] + end / 16000, offset + item.end)
                if finish <= start or not item.text.strip():
                    continue
                global_start = (
                    source["start_global_us"] + round(start * 1000000)
                    if source.get("start_global_us") is not None
                    else None
                )
                global_end = (
                    source["start_global_us"] + round(finish * 1000000)
                    if global_start is not None
                    else None
                )
                playback = [
                    event
                    for event in source.get("playback_events", [])
                    if global_start is not None
                    and global_start - 1000000
                    <= event["global_timestamp_us"]
                    <= global_end + 1000000
                ]
                rows.append(
                    {
                        "start_seconds": start,
                        "end_seconds": finish,
                        "text": item.text,
                        "estimated_global_start_us": global_start,
                        "estimated_global_end_us": global_end,
                        "avg_logprob": item.avg_logprob,
                        "no_speech_prob": item.no_speech_prob,
                        "words": [
                            {
                                "start_seconds": offset + word.start,
                                "end_seconds": offset + word.end,
                                "text": word.word,
                                "probability": word.probability,
                            }
                            for word in (item.words or [])
                        ],
                        "nearby_device_playback": playback,
                        "human_reviewed": False,
                    }
                )
            invocation["model_invocations"] += 1
            atomic_json(
                checkpoint,
                {
                    "request_sha256": request_hash,
                    "samples": [begin, end],
                    "segments": rows,
                    "runtime_sha256": checkpoint_runtime_sha256,
                    "segments_sha256": hashlib.sha256(
                        json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest(),
                },
            )
        segments.extend(rows)
        atomic_json(
            output / "progress.json",
            {
                "state": "running",
                "completed_utterances": index + 1,
                "total_utterances": len(regions),
                "progress": (index + 1) / max(1, len(regions)),
            },
        )
    segments.sort(key=lambda row: (row["start_seconds"], row["end_seconds"]))
    for index, segment in enumerate(segments):
        segment["id"] = index + 1
    # Recheck source identity at completion; an in-flight change cannot publish.
    verify_files(folder, source["files"])
    write_transcript(output, request, segments)
    invocation.update(
        status="completed",
        outcome="transcribed"
        if segments
        else "no_transcript"
        if regions
        else "no_speech",
        segment_count=len(segments),
        vad_region_count=len(regions),
        speech_seconds=sum(item["end"] - item["start"] for item in regions) / 16000,
        wall_seconds=round(time.perf_counter() - started, 3),
        actual_asr_invocation="PROVEN"
        if invocation["model_invocations"]
        else "NOT_PROVEN",
        accuracy="NOT_PROVEN",
        physical_action_confirmation=False,
        artifacts={
            name: file_record(output / name)
            for name in (
                "transcript.json",
                "transcript.txt",
                "transcript.srt",
                "transcript.vtt",
                "audio.m4a",
            )
        },
    )
    atomic_json(output / "receipt.json", invocation)
    atomic_json(output / "execution.json", {
        "request_sha256": request_hash, "receipt_sha256": sha256(output / "receipt.json"),
        **{key: invocation[key] for key in ("cache_reused", "model_invocations", "actual_asr_invocation", "decoded_audio", "wall_seconds")}})


def completed_cache(output: Path, request_hash: str, runtime_hash: str) -> dict | None:
    try:
        previous = read_json(output / "receipt.json")
        if (previous.get("status") != "completed" or previous.get("request_sha256") != request_hash
                or previous.get("runtime_sha256") != runtime_hash
                or set(previous.get("artifacts", {})) != {
                    "transcript.json", "transcript.txt", "transcript.srt", "transcript.vtt", "audio.m4a"}):
            return None
        verify_files(output, previous["artifacts"])
        return previous
    except (OSError, ValueError, KeyError, TypeError):
        return None


if __name__ == "__main__":
    request_file = Path(sys.argv[1])
    try:
        from filelock import FileLock

        # A restarted parent must not race an orphaned, still-running worker.
        with FileLock(str(request_file.parent / "worker.lock")):
            execute(request_file, reuse="--cold" not in sys.argv[2:])
    except Exception as exc:
        # Never return a subprocess traceback containing runtime paths or context.
        atomic_json(
            request_file.parent / "failure.json",
            {"status": "failed", "error_type": type(exc).__name__},
        )
        sys.exit(1)

"""Preserve submitted recordings independently of transcription and publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import speech_worker


def _copy_verified(source: Path, target: Path, expected: dict) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("原始录音文件不可用")
    before = source.stat()
    if before.st_size != expected["size"]:
        raise ValueError("原始录音与采集记录不一致")
    if target.exists() or target.is_symlink():
        if not target.is_symlink() and target.is_file() and speech_worker.file_record(target) == expected:
            return
        raise ValueError("已保存的原始录音与本次来源不一致")
    temporary = target.with_suffix(target.suffix + ".partial")
    digest, size = hashlib.sha256(), 0
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                writer.write(block)
        after = source.stat()
        if (size != expected["size"] or digest.hexdigest() != expected["sha256"]
                or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError("原始录音在保存期间变化或与采集记录不一致")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def preserve(root: Path, source: dict) -> dict:
    sealed = source["_sealed"]
    folder = Path(sealed["folder"])
    if folder.resolve() != Path(sealed["resolved_folder"]):
        raise ValueError("原始录音目录发生变化")
    name = sealed["audio_file"]
    expected = sealed["files"][name]
    reference = {"source_path": str(folder / name), "source_identity": expected,
                 "recording_session_id": sealed.get("recording_session_id"),
                 "recording_device_id": sealed.get("sender_id"),
                 "recorded_start_global_us": sealed.get("start_global_us"),
                 "clock_basis": source.get("alignment_basis"),
                 "speaker_identity": {"status": "unknown", "person_id": None, "name": None}}
    if source.get("alignment_basis") == "embedded_media_timeline":
        # The submitted MP4 is already retained by the input archive contract.
        # Never duplicate a full video merely to archive its embedded track.
        return {"status": "source_referenced", "kind": "embedded_audio", **reference}
    # Short, identity-bound names avoid reintroducing Windows path growth.
    identity = hashlib.sha256(json.dumps(sealed["files"], sort_keys=True).encode()).hexdigest()
    relative = Path("Key-Materials/Experiment-Audio/Sources") / identity[:24]
    destination = root / relative
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("录音保存路径无效")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for ordinal, (filename, spec) in enumerate(sorted(sealed["files"].items())):
        if Path(filename).name != filename or filename in {"", ".", ".."} or "\\" in filename:
            raise ValueError("录音清单文件名无效")
        suffix = Path(filename).suffix.lower()
        if len(suffix) > 10 or not suffix[1:].isalnum():
            suffix = ".bin"
        target_name = ("audio" if filename == name else f"source-{ordinal:02d}") + suffix
        _copy_verified(folder / filename, destination / target_name, spec)
        artifacts[filename] = {**spec, "path": (relative / target_name).as_posix()}
    result = {"status": "saved", "kind": "original_recording", **reference,
              "file": artifacts[name], "source_files": artifacts}
    speech_worker.atomic_json(destination / "source.json", result)
    return result

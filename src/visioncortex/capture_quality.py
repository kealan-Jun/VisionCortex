"""Bounded capture diagnostics before inference; samples are never whole-run proof."""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from . import speech, speech_worker


def audio_metrics(pcm: bytes, sample_rate: int = 16000) -> dict:
    samples = np.frombuffer(pcm, dtype="<f4")
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("没有有效音频样本")
    rms = float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
    frames = [samples[i:i+sample_rate//10] for i in range(0, len(samples), sample_rate//10)]
    silent = [float(np.sqrt(np.mean(frame.astype(np.float64)**2))) < 10**(-50/20) for frame in frames]
    longest = current = 0
    for value in silent:
        current = current+1 if value else 0
        longest = max(longest, current)
    return {"seconds": len(samples)/sample_rate, "rms_dbfs": round(20*math.log10(max(rms, 1e-12)), 2),
            "silent_fraction": sum(silent)/len(silent), "longest_sampled_silence_seconds": min(len(samples)/sample_rate, longest*.1),
            "clipped_sample_fraction": float(np.mean(np.abs(samples) >= .995))}


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns}


def _decode(path: Path, start: float, seconds: float, audio: bool) -> bytes:
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-ss", str(start), "-i", str(path)]
    command += (["-map", "0:a:0", "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le"] if audio else
                ["-map", "0:v:0", "-frames:v", "1", "-vf", "scale=160:90,format=gray", "-threads", "1", "-f", "rawvideo"])
    return subprocess.run([*command, "pipe:1"], capture_output=True, timeout=20, check=True).stdout


def inspect(manifest, config: dict, infos: dict | None = None) -> dict:
    options = config.get("capture_quality") or {}
    if not options.get("enabled"):
        return {"status": "disabled", "accuracy_evidence": "NOT_PROVEN"}
    started = time.perf_counter()
    count = min(8, max(2, int(options.get("sample_windows", 4))))
    seconds = min(5.0, max(.5, float(options.get("sample_seconds", 3))))
    sources = speech.discover(config, manifest) if speech.enabled(config) else []
    by_part = {(item["view_id"], item["segment_ordinal"]): item for item in sources}
    parts = [(view, ordinal, part) for view in manifest.views for ordinal, part in enumerate(view.segments or [view])]
    records = []
    # Work is bounded even for a large uploaded collection; expose any omissions.
    for view, ordinal, part in parts[:24]:
        item = {"view_id": view.view_id, "segment_ordinal": ordinal, "video_samples": [], "audio_samples": [], "warnings": []}
        records.append(item)
        try:
            video_before = _identity(part.video)
            if infos:
                info = infos[view.view_id]
                duration = (info.segments[ordinal] if info.segments else info).duration_ms/1000
            else:
                probe = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(part.video)], capture_output=True, text=True, check=True, timeout=20)
                duration = float(json.loads(probe.stdout)["format"]["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("无有效视频时长")
            item["video_duration_seconds"] = duration
            for fraction in np.linspace(.05, .95, count):
                at = float(fraction * max(0, duration-.1))
                frame = np.frombuffer(_decode(part.video, at, 0, False), dtype=np.uint8)
                if len(frame) != 160*90:
                    raise ValueError("视频抽样解码不完整")
                item["video_samples"].append({"at_seconds": round(at, 3), "mean_luma": round(float(frame.mean()), 2), "dark": bool(np.mean(frame < 20) >= .95)})
            if video_before != _identity(part.video):
                raise ValueError("抽样期间视频改变")
            item["video_identity"] = video_before
            item["dark_sample_fraction"] = sum(row["dark"] for row in item["video_samples"])/count
            item["visible_sample_fraction"] = 1-item["dark_sample_fraction"]
            if item["dark_sample_fraction"] >= .75:
                item["warnings"].append("抽样画面持续偏暗；请检查遮挡、镜头与照明，完整时段尚未逐帧确认")
            source = by_part.get((view.view_id, ordinal))
            if source and not source.get("available"):
                item["warnings"].append("此路没有可用录音")
                item["audio_status"] = source.get("status", "no_audio")
                continue
            sealed = (source or {}).get("_sealed") or {}
            audio = Path(sealed["folder"]) / sealed["audio_file"] if sealed else part.audio or part.video
            audio_before = _identity(audio)
            probe = speech.probe_audio(audio)
            if probe is None:
                item["audio_status"] = "no_audio"
                item["warnings"].append("此路没有音轨")
                continue
            audio_duration = probe["duration_seconds"]
            # Nonoverlapping windows, including short clips.
            span = min(seconds, audio_duration/count)
            for at in np.linspace(0, max(0, audio_duration-span), count):
                item["audio_samples"].append({"at_seconds": round(float(at), 3), **audio_metrics(_decode(audio, float(at), span, True))})
            if audio_before != _identity(audio):
                raise ValueError("抽样期间录音改变")
            item.update(audio_status="sampled", audio_identity=audio_before, audio_duration_seconds=audio_duration,
                        sampled_audio_seconds=sum(row["seconds"] for row in item["audio_samples"]))
            item["audio_sample_coverage"] = min(1.0, item["sampled_audio_seconds"]/audio_duration)
            item["silent_sample_fraction"] = sum(row["silent_fraction"] for row in item["audio_samples"])/count
            item["usable_audio_sample_fraction"] = sum(row["silent_fraction"] < .9 and row["clipped_sample_fraction"] < .01 for row in item["audio_samples"])/count
            if item["silent_sample_fraction"] >= .9:
                item["warnings"].append("抽样录音大部分为静音，请检查麦克风")
            if any(row["clipped_sample_fraction"] >= .01 for row in item["audio_samples"]):
                item["warnings"].append("抽样录音存在削波，可能过载失真")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            item["status"] = "NOT_PROVEN"
            item["warnings"].append("部分采集样本无法核验；不据此认定输入质量合格")
    return {"schema_version": "visioncortex-capture-quality/1", "status": "warning" if any(row["warnings"] for row in records) else "sampled",
            "evidence_classification": "PARTIAL_EVIDENCE", "accuracy_evidence": "NOT_PROVEN",
            "accuracy_missing_gate": "没有人工校对基线，不能报告识别准确率",
            "method": {"video": "4–8 个分布时间点的 160x90 灰度帧", "audio": "2–8 个非重叠短窗；静音 < -50 dBFS，削波幅度 ≥ 0.995", "sample_windows": count, "sample_seconds": seconds},
            "records": records, "total_sources": len(parts), "omitted_sources": max(0, len(parts)-24),
            "full_media_quality_proven": False, "wall_seconds": round(time.perf_counter()-started, 3)}


def inspect_archive(root: Path, config: dict) -> dict:
    from .input_seal import verify_input_seal
    from .schemas import RunManifest
    from .storage import _bounded_source_fingerprint
    seal_path = root / "JSON-Config-Files/Input-Manifests/input_seal.json"
    seal = speech_worker.read_json(seal_path)
    if not verify_input_seal(seal):
        raise ValueError("输入封存记录无效")
    manifest = RunManifest.model_validate(seal["manifest"])
    for view in manifest.views:
        for part in view.segments or [view]:
            record = next((row for row in seal["sources"] if row.get("kind") == "video" and row.get("view_id") == view.view_id and row.get("path") == str(part.video)), None)
            if not record or part.video.stat().st_size != record.get("size_bytes"):
                raise ValueError("视频与封存输入不匹配")
            if record.get("source_fingerprint"):
                valid = _bounded_source_fingerprint(part.video)["source_fingerprint"] == record["source_fingerprint"] and part.video.stat().st_mtime_ns == record.get("mtime_ns")
            elif record.get("content_hash_algorithm") == "sha256":
                valid = speech_worker.sha256(part.video) == record.get("content_hash")
            else:
                valid = False
            if not valid:
                raise ValueError("原视频身份已变化或缺少可验证身份")
    result = inspect(manifest, config)
    result["input_seal_sha256"] = speech_worker.sha256(seal_path)
    speech_worker.atomic_json(root / "JSON-Config-Files/capture_quality.json", result)
    return result


def model_context(root: Path) -> dict:
    path = root / "JSON-Config-Files/capture_quality.json"
    if not path.is_file():
        return {}
    receipt = speech_worker.read_json(path)
    return {"capture_quality": {"receipt_sha256": speech_worker.sha256(path),
        "scope": "bounded_samples_only; sample timestamps are physical source seconds, not global experiment time",
        "full_media_quality_proven": False, "transcript_accuracy": "NOT_PROVEN",
        "records": [{key: row[key] for key in ("view_id", "segment_ordinal", "video_samples", "audio_samples", "warnings", "audio_sample_coverage") if key in row}
                    for row in receipt.get("records", [])]}}

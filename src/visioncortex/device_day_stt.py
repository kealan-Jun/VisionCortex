"""Configured STT from retained audio into traceable machine comments."""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from .device_day_contract import artifact, atomic_bytes, atomic_json, digest, media_interval_name, read_json, safe_child


def transcript_text(recording, source, comments, outcome):
    def clock(value):
        return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')
    status = {'transcribed': '已生成机器转写，未经人工复核。',
              'no_speech': '语音活动检测未检测到语音，未生成转写；请回听原录音确认。',
              'no_transcript': '识别模型未返回文字；这不证明原录音没有语音。'}[outcome]
    lines = [f"录音：{recording['recording_id']}", f"原录音：{source['path']}",
             f'识别状态：{outcome}；{status}', '时间：Asia/Shanghai，依据采集音频时钟；精确对齐未经验证。', '']
    lines.extend(f"[{clock(row['start_us'])} – {clock(row['end_us'])}] {row['text'].strip()}" for row in comments)
    return '\n'.join(lines) + '\n'


def audio_windows(duration, maximum):
    count = math.ceil(duration / maximum)
    edges = [min(duration, index * maximum) for index in range(count + 1)]
    # Container padding can leave a millisecond tail after a nominal 15-minute
    # recording. Split the final two windows evenly, preserving every sample.
    if count > 1 and edges[-1] - edges[-2] < 1:
        edges[-2] = (edges[-3] + edges[-1]) / 2
    return list(zip(edges, edges[1:], strict=False))


def transcribe(config, layout, retention, key):
    from . import speech
    recording = retention["recording"]
    audio = retention.get("audio") or {}
    if audio.get("status") in {"pending_publication", "association_mismatch"}:
        raise ValueError("Audio awaits publication or has a mismatched recorder identity")
    audio_sources = [s for s in retention["sources"] if s["kind"] == "audio_audio"]
    embedded = False
    if not audio_sources:
        # An embedded audio track is still a source, even without audio.opus.
        source = next(s for s in retention["sources"] if s["kind"] == "video")
        path = safe_child(layout.root, source["retained"]["path"])
        info = speech.probe_audio(path)
        if info is None:
            return {"outcome": "no_audio", "comments": [], "artifacts": [],
                    "source_status": audio.get("status", "not_provided"),
                    "model_invocation": "NOT_PROVEN", "accuracy": "NOT_PROVEN"}
        embedded = True
    else:
        source = audio_sources[0]
        path = safe_child(layout.root, source["retained"]["path"])
        info = speech.probe_audio(path)
        if info is None:
            raise ValueError("Published recorder audio contains no decodable audio track")
    if not config.get("speech_recognition", {}).get("enabled"):
        raise ValueError("Audio is present but STT is disabled")
    runtime = speech.runtime_request(config)
    origin = recording["recording_start_us"] if embedded else audio.get("start_us")
    time_basis = "embedded_media_start" if embedded else "recorder_audio_clock" if origin else "recording_start_estimate"
    sealed = {"folder": str(path.parent), "resolved_folder": str(path.parent.resolve()),
              "files": {path.name: {"size": source["retained"]["size_bytes"], "sha256": source["retained"]["sha256"]}},
              "audio_file": path.name, "start_global_us": origin, "playback_events": []}
    from .device_day_content import time_folder
    root = layout.comments / time_folder(recording['recording_start_us'], recording['recording_end_us']) / 'Recognition' / key
    duration, maximum = info["duration_seconds"], runtime["max_audio_seconds"]
    comments, artifacts, chunks, audit_artifacts = [], [], [], []
    for ordinal, (start, end) in enumerate(audio_windows(duration, maximum)):
        folder = root / media_interval_name(start * 1000, end * 1000)
        worker_folder = layout.receipts / recording["recording_id"] / "speech" / key / f"{ordinal:04d}"
        request = {**runtime, "source": sealed, "start_seconds": start, "end_seconds": end}
        from .device_day import copy_verified
        # Stage/publication revisions must not trigger identical billable ASR
        # calls. Reuse only an identical request and a fully verified receipt;
        # copy its sealed artifacts so historical execution files stay intact.
        if not (worker_folder / "receipt.json").is_file():
            from . import speech_worker
            for previous in worker_folder.parent.parent.glob(f"*/{ordinal:04d}"):
                if previous == worker_folder or not (previous / "receipt.json").is_file():
                    continue
                old = read_json(previous / "receipt.json")
                if (old.get("status") != "completed" or read_json(previous / "request.json") != request
                        or old.get("request_sha256") != speech_worker.sha256(previous / "request.json")):
                    continue
                speech_worker.verify_files(previous, old["artifacts"])
                names = [*old["artifacts"], "request.json"]
                if old.get("submitted_audio", {}).get("path") == "audio.wav":
                    speech_worker.verify_files(previous, {"audio.wav": old["submitted_audio"]})
                    names.append("audio.wav")
                names.append("receipt.json")
                for name in names:
                    copy_verified(previous / name, worker_folder / name)
                break
        receipt = speech.invoke(worker_folder, request, config=config)
        transcript = read_json(worker_folder / "transcript.json")
        public_files = []
        for name in receipt["artifacts"]:
            original = worker_folder / name
            target = folder / name.capitalize()
            copy_verified(original, target)
            public_files.append(artifact(layout.root, target))
        execution = [layout.backend_artifact(worker_folder / name)
                     for name in ("request.json", "receipt.json", "execution.json", *receipt["artifacts"])]
        audit_artifacts.extend(execution)
        result_path = folder / "Result.json"
        atomic_json(result_path, {"source_ref": source["retained"], "files": public_files,
                                 "execution_artifacts": execution, "actual_asr_invocation": receipt["actual_asr_invocation"],
                                 "model": receipt.get("model"), "request_ids": receipt.get("request_ids", []),
                                 "current_execution": receipt.get("execution", {}),
                                 "wall_seconds": receipt["wall_seconds"], "outcome": receipt["outcome"]})
        artifacts.extend([*public_files, artifact(layout.root, result_path)])
        for row in transcript["segments"]:
            left = (origin or recording["recording_start_us"]) + round(row["start_seconds"] * 1e6)
            right = (origin or recording["recording_start_us"]) + round(row["end_seconds"] * 1e6)
            comments.append({"comment_id": "stt-" + digest([key, ordinal, row["id"]])[:24],
                             "recording_id": recording["recording_id"], "start_us": left, "end_us": right,
                             "text": row["text"], "source": "machine_transcribed_speech", "author": "STT",
                             "human_reviewed": False, "accuracy": "NOT_PROVEN", "physical_action_confirmed": False,
                             "model": receipt.get("model"),
                             "time_basis": time_basis, "time_alignment": "PARTIAL_EVIDENCE" if origin else "NOT_PROVEN",
                             "audio_start_seconds": row["start_seconds"], "audio_end_seconds": row["end_seconds"],
                             "audio_ref": source["retained"], "transcript_path": layout.relative(folder / "Transcript.json"),
                             "words": row.get("words", [])})
        chunks.append({"start_seconds": start, "end_seconds": end, "receipt": layout.relative(result_path),
                       "model_invocation": receipt["actual_asr_invocation"], "outcome": receipt["outcome"],
                       "model": receipt.get("model"), "current_execution": receipt.get("execution", {}),
                       "wall_seconds": receipt["wall_seconds"], "segment_count": receipt["segment_count"]})
    comment_path = root / "Comments.json"
    outcome = "transcribed" if comments else "no_transcript" if any(c["outcome"] == "no_transcript" for c in chunks) else "no_speech"
    text_path = root / "Transcript.txt"
    atomic_bytes(text_path, transcript_text(recording, source['retained'], comments, outcome).encode('utf-8'))
    text_ref = artifact(layout.root, text_path)
    atomic_json(comment_path, {"source": "machine_transcribed_speech", "comments": comments,
                               "audio_ref": source["retained"], "chunks": chunks,
                               "outcome": outcome, "transcript_file": text_ref})
    artifacts.extend([artifact(layout.root, comment_path), text_ref])
    return {"outcome": outcome, "comments": comments, "artifacts": artifacts,
            "transcript_file": text_ref,
            "audit_artifacts": audit_artifacts,
            "audio_ref": source["retained"], "chunks": chunks, "accuracy": "NOT_PROVEN",
            "model_invocation": "PROVEN" if any(c["model_invocation"] == "PROVEN" for c in chunks) else "NOT_PROVEN"}

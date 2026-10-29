"""Configured STT from retained audio into traceable machine comments."""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from .device_day_contract import atomic_bytes, atomic_json, digest, media_interval_name, read_json


def transcript_text(recording, source, comments, outcome, *, audio=None):
    def clock(value):
        return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')
    status = {'transcribed': '已自动生成语音转写。',
              'no_usable_audio': '源录音仅含声明的补静音或中断区间，没有向识别模型提交音频；不能据此判断现场无人说话。',
              'no_speech': '语音活动检测未检测到语音，未生成转写；原录音仍保留。',
              'no_transcript': '识别模型未返回文字；这不证明原录音没有语音。'}[outcome]
    basis = ('依据回放样本时间网格；原始录音时钟未经验证，缺失与中断标记见JSON。'
             if (audio or {}).get('artifact_kind') == 'derived_replay' else '依据采集音频时钟；精确对齐未经验证。')
    lines = [f"录音：{recording['recording_id']}", f"原录音：{source['path']}",
             f'识别状态：{outcome}；{status}', '时间：Asia/Shanghai，' + basis, '']
    lines.extend(f"[{clock(row['start_us'])} – {clock(row['end_us'])}] {row['text'].strip()}" for row in comments)
    return '\n'.join(lines) + '\n'


def audio_windows(duration, maximum, *, offset=0.0):
    count = math.ceil(duration / maximum)
    edges = [min(duration, index * maximum) for index in range(count + 1)]
    # Container padding can leave a millisecond tail after a nominal 15-minute
    # recording. Split the final two windows evenly, preserving every sample.
    if count > 1 and edges[-1] - edges[-2] < 1:
        edges[-2] = (edges[-3] + edges[-1]) / 2
    windows = []
    for start, end in zip(edges, edges[1:], strict=False):
        left, right = (offset + start, offset + end) if offset else (start, end)
        # Adding a fractional replay origin can make an exactly bounded
        # window subtract to maximum + one ULP. Keep the conservative provider
        # gate unchanged; move only that representation's upper edge inward.
        # This does not change its microsecond or decoded audio-sample bound.
        if right - left > maximum:
            right = math.nextafter(right, left)
        if not 0 < right - left <= maximum:
            raise ValueError('Audio window cannot fit its configured budget')
        windows.append((left, right))
    return windows


def usable_audio_windows(audio, duration, maximum):
    """Skip placeholder spans; keep original partial audio in coherent chunks.

    Small original outages remain quality ranges. Cutting at every dropped
    packet would fragment spoken sentences and create tiny provider requests.
    """
    if audio.get('artifact_kind') != 'derived_replay':
        return audio_windows(duration, maximum)
    origin = audio.get('start_us')
    if type(origin) is not int or not origin or type(audio.get('duration_us')) is not int:
        raise ValueError('Derived replay audio has no verified sample origin')
    duration = min(duration, audio['duration_us']/1000000)
    gaps = []
    for row in audio.get('quality_evidence', []):
        if row.get('kind') != 'synthetic_silence':
            continue
        left, right = row.get('start_us'), row.get('end_us')
        if type(left) is not int or type(right) is not int or left >= right:
            raise ValueError('Derived replay audio has invalid quality ranges')
        gaps.append((max(0, (left-origin)/1000000), min(duration, (right-origin)/1000000)))
    windows, cursor = [], 0
    for left, right in sorted(gaps):
        if right <= cursor or left >= duration:
            continue
        if left > cursor:
            windows.extend(audio_windows(left-cursor, maximum, offset=cursor))
        cursor = max(cursor, right)
    if cursor < duration:
        windows.extend(audio_windows(duration-cursor, maximum, offset=cursor))
    return windows


def audio_quality_ranges(audio, left, right):
    return [row for row in audio.get('quality_evidence', [])
            if row.get('kind') in {'synthetic_silence', 'source_outage'}
            and row['start_us'] < right and row['end_us'] > left]


def transcribe(config, layout, retention, key):
    from .device_day_inputs import source_path
    from .runtime_control import check_yield, sync_files
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
        path = source_path(layout, source)
        info = speech.probe_audio(path)
        if info is None:
            return {"outcome": "no_audio", "comments": [], "artifacts": [],
                    "source_status": audio.get("status", "not_provided"),
                    "model_invocation": "NOT_PROVEN", "accuracy": "NOT_PROVEN"}
        embedded = True
    else:
        source = audio_sources[0]
        path = source_path(layout, source)
        info = speech.probe_audio(path)
        if info is None:
            raise ValueError("Published recorder audio contains no decodable audio track")
    if not config.get("speech_recognition", {}).get("enabled"):
        raise ValueError("Audio is present but STT is disabled")
    runtime = speech.runtime_request(config)
    origin = recording["recording_start_us"] if embedded else audio.get("start_us")
    time_basis = ('derived_replay_sample_clock' if audio.get('artifact_kind') == 'derived_replay' else
                  "embedded_media_start" if embedded else "recorder_audio_clock" if origin else "recording_start_estimate")
    sealed = {"folder": str(path.parent), "resolved_folder": str(path.parent.resolve()),
              "files": {path.name: {"size": source["retained"]["size_bytes"], "sha256": source["retained"]["sha256"]}},
              "audio_file": path.name, "start_global_us": origin, "playback_events": []}
    if retention.get('input_binding_version') == 1:
        from .device_day_inputs import durable_json
        locator = layout.receipts / recording['recording_id'] / 'InputLocations' / (retention['input_binding'] + '.json')
        durable_json(locator, sealed | {'binding': retention['input_binding']})
        sealed = {'content_locator': str(locator), 'binding': retention['input_binding'],
                  'content': {'size': source['retained']['size_bytes'], 'sha256': source['retained']['sha256']},
                  'start_global_us': origin, 'playback_events': []}
    from .device_day_content_paths import private_folder
    root = private_folder(layout, 'Comment', 'Recognition', recording['recording_id'], key)
    duration, maximum = info["duration_seconds"], runtime["max_audio_seconds"]
    comments, artifacts, chunks, audit_artifacts, excluded = [], [], [], [], []
    windows = usable_audio_windows(audio, duration, maximum)
    for ordinal, (start, end) in enumerate(windows):
        check_yield(checkpoint=False)
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
            public_files.append(layout.artifact(target))
        execution = [layout.backend_artifact(worker_folder / name)
                     for name in ("request.json", "receipt.json", "execution.json", *receipt["artifacts"])]
        audit_artifacts.extend(execution)
        result_path = folder / "Result.json"
        atomic_json(result_path, {"source_ref": source["retained"], "files": public_files,
                                 "execution_artifacts": execution, "actual_asr_invocation": receipt["actual_asr_invocation"],
                                 "model": receipt.get("model"), "request_ids": receipt.get("request_ids", []),
                                 "current_execution": receipt.get("execution", {}),
                                 "wall_seconds": receipt["wall_seconds"], "outcome": receipt["outcome"]})
        artifacts.extend([*public_files, layout.artifact(result_path)])
        for row in transcript["segments"]:
            left = (origin or recording["recording_start_us"]) + round(row["start_seconds"] * 1e6)
            right = (origin or recording["recording_start_us"]) + round(row["end_seconds"] * 1e6)
            quality_ranges = audio_quality_ranges(audio, left, right)
            if any(gap['start_us'] <= left and gap['end_us'] >= right for gap in quality_ranges):
                excluded.append({'start_us': left, 'end_us': right, 'text': row['text'],
                    'reason': 'provider_text_inside_declared_audio_gap', 'source_quality_ranges': quality_ranges,
                    'transcript_path': layout.relative(folder/'Transcript.json')})
                continue
            comments.append({"comment_id": "stt-" + digest([key, ordinal, row["id"]])[:24],
                             "recording_id": recording["recording_id"], "start_us": left, "end_us": right,
                             "text": row["text"], "source": "machine_transcribed_speech", "author": "STT",
                             "human_reviewed": False, "accuracy": "NOT_PROVEN", "physical_action_confirmed": False,
                             "model": receipt.get("model"),
                             "time_basis": time_basis, "time_alignment": "PARTIAL_EVIDENCE" if origin else "NOT_PROVEN",
                             **({'capture_quality': audio.get('quality_status'),
                                 'source_quality_flags': audio.get('quality_flags', []),
                                 'source_quality_ranges': quality_ranges,
                                 'source_clock_verified': False} if audio.get('artifact_kind') == 'derived_replay' else {}),
                             "audio_start_seconds": row["start_seconds"], "audio_end_seconds": row["end_seconds"],
                             "audio_ref": source["retained"], "transcript_path": layout.relative(folder / "Transcript.json"),
                             "words": row.get("words", [])})
        chunks.append({"start_seconds": start, "end_seconds": end, "receipt": layout.relative(result_path),
                       "model_invocation": receipt["actual_asr_invocation"], "outcome": receipt["outcome"],
                       "model": receipt.get("model"), "current_execution": receipt.get("execution", {}),
                       "wall_seconds": receipt["wall_seconds"], "segment_count": receipt["segment_count"]})
        check_yield(checkpoint=bool(receipt.get('execution', {}).get('model_invocations', 1)),
                    persist=lambda: sync_files(result_path, worker_folder/'receipt.json',
                    worker_folder/'execution.json', *[folder / name.capitalize() for name in receipt['artifacts']]))
    comment_path = root / "Comments.json"
    outcome = ("no_usable_audio" if not windows else "transcribed" if comments else "no_transcript"
               if excluded or any(c["outcome"] == "no_transcript" for c in chunks) else "no_speech")
    text_path = root / "Transcript.txt"
    atomic_bytes(text_path, transcript_text(recording, source['retained'], comments, outcome, audio=audio).encode('utf-8'))
    text_ref = layout.artifact(text_path)
    atomic_json(comment_path, {"source": "machine_transcribed_speech", "comments": comments,
                               "audio_ref": source["retained"], "chunks": chunks,
                               "outcome": outcome, "transcript_file": text_ref})
    artifacts.extend([layout.artifact(comment_path), text_ref])
    return {"outcome": outcome, "comments": comments, "artifacts": artifacts,
            **({'source_quality': {k: audio.get(k) for k in ('artifact_kind', 'quality_status', 'quality_flags',
                'quality_evidence', 'capture_clock_verified')}, 'usable_audio_windows': windows,
                'excluded_gap_transcripts': excluded}
               if audio.get('artifact_kind') == 'derived_replay' else {}),
            "transcript_file": text_ref,
            "audit_artifacts": audit_artifacts,
            "audio_ref": source["retained"], "chunks": chunks, "accuracy": "NOT_PROVEN",
            "model_invocation": "PROVEN" if any(c["model_invocation"] == "PROVEN" for c in chunks) else "NOT_PROVEN"}

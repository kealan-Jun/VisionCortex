"""Existing production ASR provider/segmentation/receipt path, explicitly gated."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .components import pipeline_module
from .rules import Rules


def transcribe(request):
    speech = pipeline_module("speech")
    cloud = pipeline_module("speech_qwen")
    windows = pipeline_module("device_day_stt").audio_windows
    profile = Rules().profile("asr")["settings"]["speech_recognition"]
    model = request["model"]
    if profile["provider"] != "aliyun_qwen" or model.get("engine") != "existing_aliyun_qwen" or request.get("external_calls_authorized") is not True:
        raise ValueError("Existing ASR provider contract and explicit customer external-call configuration are required")
    config = {"speech_recognition": dict(profile, enabled=True, adapter_sha256=hashlib.sha256(Path(cloud.__file__).read_bytes()).hexdigest(), connection=model["connection"]), "runtime": {"local_only": False}}
    runtime = speech.runtime_request(config)
    path = Path(request["media_path"])
    if request["end_seconds"] is None:
        raise ValueError("ASR requires a sealed audio interval")
    source = dict(folder=str(path.parent), resolved_folder=str(path.parent.resolve()), files={path.name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}, audio_file=path.name, start_global_us=0)
    key = hashlib.sha256(json.dumps(dict(source=source, model_revision=model["revision"], rules=Rules().identity()), sort_keys=True).encode()).hexdigest()
    # Persistent exact-request receipts survive a crash and prevent routine repeat billing.
    output = Path(request["runtime_root"]) / "asr-receipts" / key
    segments, receipts = [], []
    start, end = float(request["start_seconds"]), float(request["end_seconds"])
    for n, (left, right) in enumerate(windows(end - start, runtime["max_audio_seconds"], offset=start)):
        directory = output / f"{n:04d}"
        receipt = speech.invoke(directory, dict(runtime, source=source, start_seconds=left, end_seconds=right), reuse=True, config=config)
        transcript = json.loads((directory / "transcript.json").read_text())
        segments.extend({k: v for k, v in s.items() if "global" not in k} for s in transcript.get("segments", []))
        receipts.append(dict(path=str(directory), request_sha256=receipt["request_sha256"], actual_asr_invocation=receipt["actual_asr_invocation"], execution=receipt.get("execution")))
    return dict(analysis_outcome="observations" if segments else "no_transcript", normalized_fields=[], segments=segments, component="visioncortex.pipeline.speech_qwen", provider=profile["provider"], provider_model=profile["model"], execution_receipts=receipts, accuracy="NOT_PROVEN")

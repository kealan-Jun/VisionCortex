"""Qwen dedicated ASR with sealed audio, provider receipts and timed subtitles."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from . import speech_worker as files

MODEL = "qwen-audio-3.0-asr-flash"
MODELS = {MODEL, "fun-asr-flash-2026-06-15"}
NO_WORDS = "ASR_RESPONSE_HAVE_NO_WORDS"


def runtime_request(config):
    from .mllm_provider import normalize_connection
    from .provider_credentials import model_api_key

    options = config["speech_recognition"]
    binding = options.get("connection") or config.get("mllm") or {}
    connection = normalize_connection(binding)
    host = urlsplit(connection["base_url"]).hostname
    if connection["provider"] != "aliyun" or host not in {
        "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
    }:
        raise ValueError("Qwen ASR requires the existing verified Aliyun connection")
    model = options.get("model")
    if model not in MODELS:
        raise ValueError("Unsupported Qwen ASR model; do not silently fall back")
    identity = files.sha256(Path(__file__))
    if options.get("adapter_sha256") != identity:
        raise ValueError("Qwen ASR adapter differs from the configured implementation")
    connection.update({k: binding.get(k) for k in ("credential_ref", "api_key_env")})
    if not model_api_key(connection):
        raise ValueError("The bound Aliyun API credential is unavailable")
    maximum = float(options.get("max_audio_seconds", 180))
    if not 1 <= maximum <= 180:
        raise ValueError("Qwen inline audio must be split into at most 180 seconds")
    return {
        "schema_version": "visioncortex-speech-request/1", "provider": "aliyun_qwen",
        "model": {"provider": "aliyun", "model": model, "revision": "provider_managed_alias" if model == MODEL else model,
                  "weights_identity": "NOT_PROVEN"},
        "endpoint": f"https://{host}/api/v1/services/aigc/multimodal-generation/generation",
        "credential_connection": connection, "worker_sha256": identity,
        "repository": {"implementation_identity": "worker_sha256", "release_certified": False},
        "max_audio_seconds": maximum, "language_hints": ["zh", "en"] if model == MODEL else ["zh"],
    }


def recognize(request, audio):
    """Only final SSE sentences are accepted. Never persist headers or Base64."""
    from .provider_credentials import model_api_key

    encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
    if len(encoded) > 10_000_000:
        raise ValueError("Qwen encoded audio exceeds 10 MB")
    payload = {"model": request["model"]["model"], "input": {"messages": [
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {
            "data": "data:audio/wav;base64," + encoded}}]}]},
        "parameters": {"format": "wav", "sample_rate": "16000",
                       "language_hints": request["language_hints"]}}
    connection = request["credential_connection"]
    host = urlsplit(connection["base_url"]).hostname
    expected = f"https://{host}/api/v1/services/aigc/multimodal-generation/generation"
    if host not in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"} or request["endpoint"] != expected:
        raise ValueError("Qwen ASR endpoint does not match the bound provider")
    key = model_api_key(connection)
    if not key:
        raise ValueError("The bound Aliyun API credential is unavailable")
    call = Request(request["endpoint"], data=json.dumps(payload).encode(), headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
        "X-DashScope-SSE": "enable"})
    started, events = time.monotonic(), []
    try:
        with urlopen(call, timeout=60) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                events.append(json.load(response))
            else:
                data = []
                for raw in response:
                    if time.monotonic() - started > 480:
                        raise TimeoutError("Qwen ASR total response time exceeded 480 seconds")
                    line = raw.decode("utf-8").strip()
                    if line.startswith("data:"):
                        data.append(line[5:].strip())
                    elif not line and data:
                        joined = "\n".join(data)
                        if joined != "[DONE]":
                            events.append(json.loads(joined))
                        data = []
                if data and "\n".join(data) != "[DONE]":
                    events.append(json.loads("\n".join(data)))
        return {"http_status": 200, "events": events,
                "wall_seconds": round(time.monotonic() - started, 3)}
    except HTTPError as error:
        # Persist only the API's structured result, never the request headers.
        value = json.loads(error.read(1_000_000))
        return {"http_status": error.code, "events": [value],
                "wall_seconds": round(time.monotonic() - started, 3)}


def parse_segments(response, request):
    start, end = request["start_seconds"], request["end_seconds"]
    final, unfinished = {}, set()
    events = response["events"]
    if not events:
        raise ValueError("Qwen returned no completion receipt")
    empty = response["http_status"] in {200, 400} and len(events) == 1 and events[0].get("message") == NO_WORDS
    if empty:
        return []
    if response["http_status"] != 200:
        raise ValueError(f"Qwen ASR HTTP {response['http_status']}: {events[0].get('code', 'error')}")
    for event in events:
        if event.get("code") or "output" not in event:
            raise ValueError("Qwen ASR response contains a provider error")
        sentence = event["output"].get("sentence")
        if not sentence:
            if event["output"].get("text"):
                raise ValueError("Qwen transcript is missing source timestamps")
            continue
        identity = (sentence.get("channel_id", 0), sentence["sentence_id"])
        if not sentence.get("sentence_end"):
            unfinished.add(identity)
            continue
        unfinished.discard(identity)
        if not sentence.get("text", "").strip():
            continue
        left, right = start + sentence["begin_time"] / 1000, start + sentence["end_time"] / 1000
        if not (math.isfinite(left) and math.isfinite(right) and start <= left < right <= end + .05):
            raise ValueError("Qwen sentence timestamps exceed the submitted audio")
        words = []
        for word in sentence.get("words", []):
            a, b = start + word["begin_time"] / 1000, start + word["end_time"] / 1000
            if not (math.isfinite(a) and math.isfinite(b) and left <= a <= b <= right + .05):
                raise ValueError("Qwen word timestamps exceed the sentence")
            words.append({"word": word["text"] + word.get("punctuation", ""),
                          "start_seconds": a, "end_seconds": b})
        # Some provider deployments emit cumulative sentence/word snapshots
        # under increasing sentence IDs. The later snapshot supersedes its
        # contained prefix; retaining both would duplicate recognized speech.
        for previous_id, previous in list(final.items()):
            if (previous_id[0] == identity[0] and previous["start_seconds"] == left
                    and previous["end_seconds"] <= right
                    and sentence["text"].startswith(previous["text"])):
                del final[previous_id]
        final[identity] = {"start_seconds": left, "end_seconds": right, "text": sentence["text"],
                           "words": words, "channel_id": identity[0], "speaker_identity": "NOT_PROVEN"}
    if unfinished:
        raise ValueError("Qwen stream ended with unfinished sentences")
    segments = sorted(final.values(), key=lambda x: x["start_seconds"])
    # The cumulative text must be accounted for by the preserved final sentences.
    cumulative = events[-1].get("output", {}).get("text", "")
    def normalize(text):
        return "".join(text.split())
    if cumulative and normalize(cumulative) != normalize("".join(s["text"] for s in segments)):
        raise ValueError("Qwen final sentences do not account for the complete transcript")
    timed = []
    for segment in segments:
        words = segment["words"]
        if not words or normalize("".join(w["word"] for w in words)) != normalize(segment["text"]):
            timed.append(segment)
            continue
        group = []
        for index, word in enumerate(words):
            group.append(word)
            last = index == len(words) - 1
            pause = not last and words[index + 1]["start_seconds"] - word["end_seconds"] >= 1.2
            if last or pause or word["word"].endswith(("。", "！", "？", "!", "?")):
                timed.append(dict(segment, start_seconds=group[0]["start_seconds"],
                                  end_seconds=group[-1]["end_seconds"], words=group,
                                  text="".join(w["word"] for w in group)))
                group = []
    return [dict(row, id=index) for index, row in enumerate(timed, 1)]


def invoke(directory, request, *, reuse=True):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if request["worker_sha256"] != files.sha256(Path(__file__)):
        raise ValueError("Qwen ASR implementation changed after queueing")
    source = request["source"]
    folder = Path(source["folder"])
    if folder.resolve() != Path(source["resolved_folder"]):
        raise ValueError("Audio source directory changed after queueing")
    files.verify_files(folder, source["files"])
    if source["audio_file"] not in source["files"]:
        raise ValueError("Audio source is not in the sealed file manifest")
    duration = request["end_seconds"] - request["start_seconds"]
    if not 0 < duration <= min(request["max_audio_seconds"], 180):
        raise ValueError("Qwen ASR audio range exceeds its budget")
    request_path = directory / "request.json"
    files.atomic_json(request_path, request)
    request_hash = files.sha256(request_path)
    receipt_path = directory / "receipt.json"
    if reuse and receipt_path.is_file():
        previous = files.read_json(receipt_path)
        if previous.get("status") == "completed" and previous.get("request_sha256") == request_hash:
            files.verify_files(directory, previous["artifacts"])
            execution = {"request_sha256": request_hash, "receipt_sha256": files.sha256(receipt_path),
                         "cache_reused": True, "model_invocations": 0, "actual_asr_invocation": "NOT_PROVEN",
                         "wall_seconds": round(time.monotonic() - started, 3)}
            files.atomic_json(directory / "execution.json", execution)
            return dict(previous, execution=execution)
    audio = directory / "audio.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(request["start_seconds"]),
                    "-i", str(folder / source["audio_file"]), "-t", str(duration), "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(audio)],
                   check=True, capture_output=True, timeout=max(60, duration * 2))
    response = recognize(request, audio)
    files.atomic_json(directory / "response.json", response)
    segments = parse_segments(response, request)
    files.verify_files(folder, source["files"])
    files.write_transcript(directory, request, segments)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio),
                    "-c:a", "aac", "-b:a", "64k", str(directory / "audio.m4a")],
                   check=True, capture_output=True, timeout=max(60, duration * 2))
    receipt = {
        "schema_version": "visioncortex-speech-receipt/1", "status": "completed",
        "request_sha256": request_hash, "worker_sha256": request["worker_sha256"],
        "model": request["model"], "endpoint": request["endpoint"], "source_hashes_verified": True,
        "audio_uploaded": True, "cache_reused": False, "model_invocations": 1,
        "actual_asr_invocation": "PROVEN", "accuracy": "NOT_PROVEN", "physical_action_confirmation": False,
        "request_ids": sorted({e["request_id"] for e in response["events"] if e.get("request_id")}),
        "usage": [e["usage"] for e in response["events"] if e.get("usage")],
        "outcome": "transcribed" if segments else "no_transcript", "segment_count": len(segments),
        "wall_seconds": round(time.monotonic() - started, 3), "provider_wall_seconds": response["wall_seconds"],
        "audio_seconds": duration, "empty_result_is_not_proof_of_silence": not bool(segments),
        "submitted_audio": {"path": "audio.wav", **files.file_record(audio)},
        "artifacts": {name: files.file_record(directory / name) for name in
                      ("transcript.json", "transcript.txt", "transcript.srt", "transcript.vtt", "audio.m4a", "response.json")},
    }
    files.atomic_json(receipt_path, receipt)
    execution = {"request_sha256": request_hash, "receipt_sha256": files.sha256(receipt_path),
                 "cache_reused": False, "model_invocations": 1, "actual_asr_invocation": "PROVEN",
                 "wall_seconds": receipt["wall_seconds"]}
    files.atomic_json(directory / "execution.json", execution)
    return dict(receipt, execution=execution)

"""Opt-in native Transformers ASR using an already verified local checkpoint.

Audio and generated text are evidence data. This backend neither interprets
experiment commands nor claims word alignment or speaker identity.
"""

import hashlib
import importlib.metadata
from pathlib import Path
import time


def settings(configuration):
    device = configuration.get("device")
    dtype = configuration.get("dtype")
    threads = configuration.get("cpu_threads")
    tokens = configuration.get("max_new_tokens", 1024)
    seconds = configuration.get("max_generation_seconds", 120)
    audio_seconds = configuration.get("max_audio_seconds", 180)
    language = configuration.get("language", "Chinese")
    if (
        device not in {"cpu", "cuda:0"}
        or dtype not in {"float32", "float16", "bfloat16"}
        or (device == "cpu" and dtype != "float32")
        or type(threads) is not int
        or not 1 <= threads <= 8
        or type(tokens) is not int
        or not 1 <= tokens <= 1024
        or type(seconds) is not int
        or not 1 <= seconds <= 180
        or type(audio_seconds) is not int
        or not 1 <= audio_seconds <= 180
        or (
            language is not None
            and (not isinstance(language, str) or not 1 <= len(language) <= 32)
        )
    ):
        raise ValueError(
            "qwen3-asr-hf requires explicit bounded device, dtype and cpu_threads"
        )
    memory = configuration.get("max_cuda_memory_bytes")
    if device == "cuda:0" and (
        type(memory) is not int or not 512 * 1024**2 <= memory <= 8 * 1024**3
    ):
        raise ValueError(
            "qwen3-asr-hf CUDA requires an explicit bounded memory allowance"
        )
    if device == "cpu" and memory is not None:
        raise ValueError("CPU ASR cannot declare a CUDA memory allowance")
    return dict(
        engine="qwen3-asr-hf",
        device=device,
        dtype=dtype,
        cpu_threads=threads,
        max_new_tokens=tokens,
        max_generation_seconds=seconds,
        max_audio_seconds=audio_seconds,
        max_cuda_memory_bytes=memory,
        language=language,
        attention="sdpa",
        do_sample=False,
        local_files_only=True,
        trust_remote_code=False,
        word_alignment="not_loaded",
        batch_size=1,
    )


def runtime_identity(configuration):
    identity = settings(configuration)
    identity["helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "tokenizers", "numpy", "av")
    }
    return identity


def make_backend(model_root, configuration, decode_audio):
    """Load only the supplied checkpoint; never download, align or fall back."""
    selected = settings(configuration)
    import numpy as np
    import torch
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.13.0"):
        raise ValueError("qwen3-asr-hf requires Transformers >=5.13.0")
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.set_num_threads(selected["cpu_threads"])
    cuda = selected["device"] == "cuda:0"
    if cuda:
        if not torch.cuda.is_available():
            raise ValueError("Explicit CUDA ASR device is unavailable; no CPU fallback")
        free, total = torch.cuda.mem_get_info(0)
        if free < selected["max_cuda_memory_bytes"] + 2 * 1024**3:
            raise ValueError(
                "Insufficient free CUDA memory with shared-service reserve"
            )
        torch.cuda.set_per_process_memory_fraction(
            selected["max_cuda_memory_bytes"] / total, 0
        )
    dtype = getattr(torch, selected["dtype"])
    processor = AutoProcessor.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        AutoModelForMultimodalLM.from_pretrained(
            str(model_root),
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        )
        .to(selected["device"])
        .eval()
    )

    def infer(raw):
        started = time.monotonic()
        samples, rate = decode_audio(raw, max_seconds=selected["max_audio_seconds"])
        samples = np.asarray(samples, dtype=np.float32)
        if (
            rate != 16000
            or samples.ndim != 1
            or not np.isfinite(samples).all()
            or not 0 < len(samples) <= rate * selected["max_audio_seconds"]
        ):
            raise ValueError("ASR requires finite bounded mono 16kHz decoded samples")
        duration = len(samples) / rate
        inputs = processor.apply_transcription_request(
            audio=samples,
            language=selected["language"],
            processor_kwargs={
                "audio_kwargs": {
                    "sampling_rate": rate,
                    "truncation": False,
                    "padding": "longest",
                }
            },
        )
        token_ids = inputs["input_ids"]
        if token_ids.shape[0] != 1:
            raise ValueError("ASR backend accepts exactly one audio recording")
        expanded = processor.decode(token_ids, skip_special_tokens=False)[0]
        if not isinstance(expanded, str) or len(expanded.encode()) > 256 * 1024:
            raise ValueError("ASR expanded input template exceeds evidence bound")
        input_sha = hashlib.sha256(
            token_ids.detach().cpu().numpy().astype("<i8").tobytes()
        ).hexdigest()
        inputs = inputs.to(model.device, model.dtype)
        prepared_seconds = time.monotonic() - started
        if cuda:
            torch.cuda.reset_peak_memory_stats(0)
        generated_at = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=selected["max_new_tokens"],
                max_time=selected["max_generation_seconds"],
            )
        if cuda:
            torch.cuda.synchronize(0)
        generation_seconds = time.monotonic() - generated_at
        new_ids = output[:, token_ids.shape[1] :]
        count = int(new_ids.shape[1])
        if not 0 < count <= selected["max_new_tokens"]:
            raise ValueError("ASR returned an invalid token sequence")
        parsed = processor.decode(new_ids, return_format="parsed")[0]
        original = processor.decode(new_ids, return_format="raw")[0]
        if (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("transcription"), str)
            or not isinstance(original, str)
            or len(original.encode()) > 256 * 1024
            or len(parsed["transcription"].encode()) > 256 * 1024
        ):
            raise ValueError("ASR returned invalid or oversized text")
        language = parsed.get("language")
        if language is not None and (
            not isinstance(language, str) or len(language) > 64
        ):
            raise ValueError("ASR returned invalid language metadata")
        eos_ids = model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        eos = int(new_ids[0, -1]) in eos_ids
        finish = (
            "eos"
            if eos
            else "output_limit"
            if count == selected["max_new_tokens"]
            else "time_limit_or_other_stop"
        )
        reasons = [
            "word_alignment_not_loaded",
            "speaker_identity_unknown",
            "accuracy_not_proven",
        ]
        if not eos:
            reasons.append("generation_completion_not_verified")
        return dict(
            text=parsed["transcription"],
            raw_text=original,
            language=language,
            duration_seconds=duration,
            words=[],
            segments=[],
            time_basis="input_audio_support_only_not_speech_alignment",
            input_audio_span=dict(start_seconds=0.0, end_seconds=duration),
            alignment_revision=None,
            alignment_available=False,
            speaker_identity=None,
            asr_engine="qwen3-asr-hf",
            accuracy="NOT_PROVEN",
            needs_attention=reasons,
            finish_reason=finish,
            generated_tokens=count,
            peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
            execution=dict(
                prepare_seconds=prepared_seconds,
                generate_seconds=generation_seconds,
                sample_rate=rate,
                sample_count=len(samples),
                decoded_float32_sha256=hashlib.sha256(
                    samples.astype("<f4").tobytes()
                ).hexdigest(),
                input_token_count=int(token_ids.shape[1]),
                input_token_sha256=input_sha,
                expanded_template=expanded,
                expanded_template_sha256=hashlib.sha256(expanded.encode()).hexdigest(),
                effective_settings=selected,
            ),
        )

    return infer

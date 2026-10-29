"""Bounded native-HF contract fixtures; no model or optional GPU dependency."""

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from .test_model_services import load, setup


def configured(tmp_path, monkeypatch):
    service = load("lab_model_service")
    backend = load("qwen3_asr")
    monkeypatch.setitem(sys.modules, "qwen3_asr", backend)
    root, weights, _, _, configuration, _ = setup(tmp_path)
    configuration["services"]["asr"] = dict(
        engine="qwen3-asr-hf",
        checkpoint="fixture-vlm",
        device="cpu",
        dtype="float32",
        cpu_threads=1,
        max_new_tokens=4,
    )
    return service, backend, root, weights, configuration


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape

    def __getitem__(self, item):
        result = self.value[item]
        return Tensor(result) if result.ndim else int(result)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def fake_runtime(monkeypatch, output=None):
    calls = []

    class Inputs(dict):
        def to(self, device, dtype):
            calls.append(("inputs_to", device, dtype))
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor_load", path, kwargs))
            return cls()

        def apply_transcription_request(self, **kwargs):
            calls.append(("prepare", kwargs))
            return Inputs(input_ids=Tensor([[1, 2, 3]]))

        def decode(self, value, **kwargs):
            if kwargs.get("return_format") == "parsed":
                return [dict(language="Chinese", transcription="候选原文")]
            if kwargs.get("return_format") == "raw":
                return ["language Chinese<asr_text>候选原文<eos>"]
            return ["fixture expanded audio template"]

    class Model:
        device = "fixture-device"
        dtype = "fixture-dtype"
        generation_config = SimpleNamespace(eos_token_id=[99])

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model_load", path, kwargs))
            return cls()

        def to(self, device):
            calls.append(("model_to", device))
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return Tensor([[1, 2, 3] + (output if output is not None else [7, 99])])

    torch = SimpleNamespace(
        set_num_threads=lambda value: calls.append(("threads", value)),
        float32="fixture-f32",
        bfloat16="fixture-bf16",
        float16="fixture-f16",
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    transformers = SimpleNamespace(
        __version__="5.13.1",
        AutoProcessor=Processor,
        AutoModelForMultimodalLM=Model,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls, torch, transformers


def test_verified_single_checkpoint_native_entry_preserves_unaligned_candidate(
    tmp_path, monkeypatch
):
    service, _, root, weights, configuration = configured(tmp_path, monkeypatch)
    calls, _, _ = fake_runtime(monkeypatch)
    samples = np.zeros(32000, np.float32)
    monkeypatch.setattr(
        service, "decode_audio_bounded", lambda raw, **kwargs: (samples, 16000)
    )
    verified = {}
    infer, revision = service.make_asr(root, configuration, verified)
    result = infer(b"generated-audio-fixture-not-real-model")
    assert revision == "fixture/1" and set(verified) == {"fixture-vlm"}
    loads = [item for item in calls if item[0].endswith("load")]
    assert all(item[1] == str(weights) for item in loads)
    assert all(
        item[2]["local_files_only"] and not item[2]["trust_remote_code"]
        for item in loads
    )
    prepared = next(item[1] for item in calls if item[0] == "prepare")
    assert prepared["audio"] is samples
    assert prepared["processor_kwargs"]["audio_kwargs"] == dict(
        sampling_rate=16000,
        truncation=False,
        padding="longest",
    )
    assert result["text"] == "候选原文" and "<asr_text>" in result["raw_text"]
    assert result["words"] == result["segments"] == []
    assert result["alignment_available"] is False and result["speaker_identity"] is None
    assert result["input_audio_span"] == dict(start_seconds=0.0, end_seconds=2.0)
    assert result["time_basis"] == "input_audio_support_only_not_speech_alignment"
    assert result["finish_reason"] == "eos" and result["accuracy"] == "NOT_PROVEN"
    assert (
        result["execution"]["decoded_float32_sha256"]
        == hashlib.sha256(samples.tobytes()).hexdigest()
    )
    assert (
        result["execution"]["expanded_template_sha256"]
        == hashlib.sha256(
            result["execution"]["expanded_template"].encode(),
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", None),
        ("dtype", "auto"),
        ("cpu_threads", True),
        ("max_new_tokens", 1025),
        ("max_generation_seconds", 181),
        ("max_audio_seconds", 181),
        ("language", ["Chinese"]),
    ],
)
def test_explicit_bounded_parameters_required(tmp_path, monkeypatch, field, value):
    service, _, root, _, configuration = configured(tmp_path, monkeypatch)
    configuration["services"]["asr"][field] = value
    with pytest.raises(ValueError, match="explicit bounded"):
        service.make_asr(root, configuration)


def test_gpu_unavailable_does_not_fall_back_or_load_model(tmp_path, monkeypatch):
    service, _, root, _, configuration = configured(tmp_path, monkeypatch)
    calls, _, _ = fake_runtime(monkeypatch)
    configuration["services"]["asr"].update(
        device="cuda:0",
        dtype="bfloat16",
        max_cuda_memory_bytes=4 * 1024**3,
    )
    with pytest.raises(ValueError, match="no CPU fallback"):
        service.make_asr(root, configuration)
    assert not [item for item in calls if item[0].endswith("load")]


def test_gpu_budget_preserves_other_service_reserve_before_load(tmp_path, monkeypatch):
    service, _, root, _, configuration = configured(tmp_path, monkeypatch)
    calls, torch, _ = fake_runtime(monkeypatch)
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda device: (5 * 1024**3, 24 * 1024**3),
    )
    configuration["services"]["asr"].update(
        device="cuda:0",
        dtype="bfloat16",
        max_cuda_memory_bytes=4 * 1024**3,
    )
    with pytest.raises(ValueError, match="shared-service reserve"):
        service.make_asr(root, configuration)
    assert not [item for item in calls if item[0].endswith("load")]


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.zeros(16000 * 180 + 1, np.float32), 16000),
        (np.array([float("nan")], np.float32), 16000),
        (np.zeros((1, 100), np.float32), 16000),
        (np.zeros(10, np.float32), 8000),
    ],
)
def test_invalid_decoded_audio_never_generates(tmp_path, monkeypatch, samples, rate):
    service, _, root, _, configuration = configured(tmp_path, monkeypatch)
    calls, _, _ = fake_runtime(monkeypatch)
    monkeypatch.setattr(
        service, "decode_audio_bounded", lambda raw, **kwargs: (samples, rate)
    )
    infer, _ = service.make_asr(root, configuration)
    with pytest.raises(ValueError, match="finite bounded mono"):
        infer(b"fixture")
    assert not [item for item in calls if item[0] == "generate"]


def test_output_cap_is_attention_and_not_complete_transcription(tmp_path, monkeypatch):
    service, _, root, _, configuration = configured(tmp_path, monkeypatch)
    fake_runtime(monkeypatch, output=[6, 7, 8, 9])
    monkeypatch.setattr(
        service,
        "decode_audio_bounded",
        lambda raw, **kwargs: (np.zeros(16000, np.float32), 16000),
    )
    infer, _ = service.make_asr(root, configuration)
    result = infer(b"fixture")
    assert result["finish_reason"] == "output_limit"
    assert "generation_completion_not_verified" in result["needs_attention"]
    assert result["alignment_available"] is False


def test_execution_identity_pins_backend_source_packages_parameters_and_checkpoint(
    tmp_path, monkeypatch
):
    service, backend, root, _, configuration = configured(tmp_path, monkeypatch)
    monkeypatch.setattr(
        backend.importlib.metadata, "version", lambda name: "fixture-" + name
    )
    verified = {}
    service.checkpoint(root, "fixture-vlm", configuration, verified)
    initial = service.inference_identity("asr", configuration, verified)
    for field, value in [
        ("language", "English"),
        ("max_new_tokens", 2),
        ("cpu_threads", 2),
    ]:
        changed = deepcopy(configuration)
        changed["services"]["asr"][field] = value
        assert service.inference_identity("asr", changed, verified) != initial
    monkeypatch.setattr(
        backend.importlib.metadata, "version", lambda name: "changed-" + name
    )
    assert service.inference_identity("asr", configuration, verified) != initial


def test_unsupported_transformers_version_is_actionable_without_loading(
    tmp_path, monkeypatch
):
    service, _, root, _, configuration = configured(tmp_path, monkeypatch)
    calls, _, transformers = fake_runtime(monkeypatch)
    transformers.__version__ = "5.6.0"
    with pytest.raises(ValueError, match="Transformers >=5.13.0"):
        service.make_asr(root, configuration)
    assert not calls

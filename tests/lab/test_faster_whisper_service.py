"""Fake optional backend only: no real model, GPU, audio source or service starts."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from .test_model_services import load, setup


def configured(tmp_path):
    root, weights, _, _, config, _ = setup(tmp_path)
    config['services']['asr'] = dict(engine='faster-whisper', checkpoint='fixture-vlm',
                                    device='cpu', cpu_threads=2, compute_type='int8')
    return root, weights, config


def backend(monkeypatch, segments=None):
    calls = []
    if segments is None:
        segments = [SimpleNamespace(text='混合。', start=.1, end=1.5,
                      words=[SimpleNamespace(word='混合。', start=.1, end=1.5)])]
    class FakeWhisper:
        def __init__(self, path, **kwargs):
            calls.append(('construct', path, kwargs))
        def transcribe(self, samples, **kwargs):
            calls.append(('transcribe', len(samples), kwargs))
            return iter(segments), SimpleNamespace(language='zh')
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeWhisper))
    return calls


def test_explicit_faster_backend_uses_local_checkpoint_and_source_relative_words(tmp_path, monkeypatch):
    source = load('lab_model_service')
    root, weights, config = configured(tmp_path)
    calls = backend(monkeypatch)
    monkeypatch.setattr(source, 'decode_audio_bounded', lambda raw, **kwargs: (np.zeros(32000, np.float32), 16000))
    verified = {}
    infer, revision = source.make_asr(root, config, verified)
    assert calls[0] == ('construct', str(weights), dict(device='cpu', compute_type='int8',
                                  cpu_threads=2, local_files_only=True, num_workers=1))
    assert set(verified) == {'fixture-vlm'} and revision == 'fixture/1'
    result = infer(b'synthetic-audio-decoder-fixture')
    assert calls[1][2]['word_timestamps'] is True
    assert result['words'] == [dict(text='混合。', start_seconds=.1, end_seconds=1.5)]
    assert result['segments'] == [dict(text='混合。', start_seconds=.1, end_seconds=1.5)]
    assert result['duration_seconds'] == 2 and result['text'] == '混合。'
    assert result['time_basis'] == 'local_model_timestamps_source_relative'
    assert result['alignment_revision'] is result['peak_torch_gpu_bytes'] is None
    assert result['accuracy'] == 'NOT_PROVEN'


@pytest.mark.parametrize('field', ['device', 'cpu_threads', 'compute_type'])
def test_faster_backend_requires_explicit_execution_parameters(tmp_path, field):
    source = load('lab_model_service')
    root, _, config = configured(tmp_path)
    del config['services']['asr'][field]
    with pytest.raises(ValueError, match='explicit'):
        source.make_asr(root, config)


def test_decoder_enforces_audio_sample_bound_before_buffering_more(monkeypatch):
    source = load('lab_model_service')
    class Frame:
        def to_ndarray(self):
            return np.zeros((1, 16000), np.float32)
    class Container:
        streams = SimpleNamespace(audio=[True])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def decode(self, **kwargs):
            return iter([Frame(), Frame()])
    class Resampler:
        def __init__(self, **kwargs):
            assert kwargs == dict(format='flt', layout='mono', rate=16000)
        def resample(self, frame):
            return [frame] if frame is not None else []
    class DecodeError(Exception):
        pass
    av = SimpleNamespace(open=lambda *args, **kwargs: Container(), AudioResampler=Resampler,
                         error=SimpleNamespace(FFmpegError=DecodeError))
    monkeypatch.setitem(sys.modules, 'av', av)
    with pytest.raises(ValueError, match='at most 180 seconds'):
        source.decode_audio_bounded(b'fixture', max_seconds=1)


def test_default_qwen_path_still_selects_qwen_and_forced_aligner(tmp_path, monkeypatch):
    source = load('lab_model_service')
    root, weights, _, _, config, _ = setup(tmp_path)
    manifest = config['checkpoints']['fixture-vlm']
    config['checkpoints'].update({'Qwen3-ASR-1.7B': manifest, 'Qwen3-ForcedAligner-0.6B': manifest})
    calls = []
    class Qwen:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='fixture-bfloat16'))
    monkeypatch.setitem(sys.modules, 'qwen_asr', SimpleNamespace(Qwen3ASRModel=Qwen))
    verified = {}
    infer, revision = source.make_asr(root, config, verified)
    assert callable(infer) and revision == 'fixture/1'
    assert calls[0][0] == str(weights)
    assert calls[0][1]['forced_aligner'] == str(weights)
    assert set(verified) == {'Qwen3-ASR-1.7B', 'Qwen3-ForcedAligner-0.6B'}


def test_faster_identity_pins_engine_and_effective_execution_parameters(tmp_path):
    source = load('lab_model_service')
    root, _, config = configured(tmp_path)
    verified = {}
    source.checkpoint(root, 'fixture-vlm', config, verified)
    original = source.inference_identity('asr', config, verified)
    for field, value in [('device', 'cuda'), ('cpu_threads', 4), ('compute_type', 'float32')]:
        changed = deepcopy(config)
        changed['services']['asr'][field] = value
        assert source.inference_identity('asr', changed, verified) != original
    changed = deepcopy(verified)
    changed['fixture-vlm']['manifest_sha256'] = 'a' * 64
    assert source.inference_identity('asr', config, changed) != original
    qwen = deepcopy(config)
    qwen['services']['asr'] = {}
    qwen_verified = dict(verified, **{'Qwen3-ASR-1.7B': verified['fixture-vlm'],
                                   'Qwen3-ForcedAligner-0.6B': verified['fixture-vlm']})
    assert source.inference_identity('asr', qwen, qwen_verified) != original


def test_faster_rejects_long_audio_and_timestamp_escape(tmp_path, monkeypatch):
    source = load('lab_model_service')
    root, _, config = configured(tmp_path)
    calls = backend(monkeypatch, [SimpleNamespace(text='invalid', start=0, end=12, words=[])])
    infer, _ = source.make_asr(root, config)
    class TooLong:
        def __len__(self):
            return 16000 * 180 + 1
    monkeypatch.setattr(source, 'decode_audio_bounded', lambda raw, **kwargs: (TooLong(), 16000))
    with pytest.raises(ValueError, match='at most 180 seconds'):
        infer(b'fixture')
    assert len(calls) == 1  # No transcription request is made for oversized input.
    monkeypatch.setattr(source, 'decode_audio_bounded', lambda raw, **kwargs: (np.zeros(16000, np.float32), 16000))
    with pytest.raises(ValueError, match='escaped'):
        infer(b'fixture')

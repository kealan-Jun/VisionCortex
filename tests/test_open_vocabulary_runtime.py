from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from visioncortex.open_vocabulary_runtime import load_yolo_world_with_local_clip


def test_world_text_encoder_loads_configured_path_without_named_download(tmp_path, monkeypatch):
    world = tmp_path / "world.pt"
    clip = tmp_path / "custom-name.pt"
    world.write_bytes(b"test")
    clip.write_bytes(b"test")
    calls = []
    model = SimpleNamespace(model=SimpleNamespace(clip_model=None))
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLOWorld = lambda path: model
    encoder = ModuleType("ultralytics.nn.text_model")

    def local_encoder(path, device):
        calls.append((path, device))
        return "local-encoder"

    encoder.CLIP = local_encoder
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.text_model", encoder)
    result = load_yolo_world_with_local_clip({"model_path": str(world), "clip_model_path": str(clip)})
    assert calls == [(str(clip), "cpu")]
    assert result.model.clip_model == "local-encoder"


def test_missing_local_clip_fails_before_import_or_download(tmp_path):
    world = tmp_path / "world.pt"
    world.write_bytes(b"test")
    with pytest.raises(FileNotFoundError, match="Configured open-vocabulary asset"):
        load_yolo_world_with_local_clip({"model_path": str(world), "clip_model_path": str(tmp_path / "missing.pt")})


def test_shared_mutable_model_prompt_and_inference_are_atomic():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex.open_vocabulary_runtime import serialized_open_vocabulary
    state = {}
    start = threading.Barrier(4)

    @serialized_open_vocabulary
    def predict(prompt):
        state['prompt'] = prompt
        time.sleep(.005)
        return state['prompt']

    def worker(prompt):
        start.wait()
        return predict(prompt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(worker, ['a', 'b', 'c', 'd'])) == ['a', 'b', 'c', 'd']


def test_serialized_receipt_separates_wait_and_service_without_mutating_result(monkeypatch):
    from visioncortex import open_vocabulary_runtime as runtime
    clock = [10.0]
    released = []

    class Lock:
        def __enter__(self):
            clock[0] += 2

        def __exit__(self, *_args):
            released.append(True)

    monkeypatch.setattr(runtime, '_OPEN_VOCABULARY_LOCK', Lock())
    monkeypatch.setattr(runtime.time, 'perf_counter', lambda: clock[0])
    receipt = {'status': 'executed'}
    boxes = []

    @runtime.serialized_open_vocabulary
    def predict(fail=False):
        clock[0] += 3
        if fail:
            raise ValueError('model failure')
        return boxes, receipt

    detections, measured = predict()
    assert detections is boxes
    assert receipt == {'status': 'executed'}
    assert measured['serialization_wait_seconds'] == 2
    assert measured['serialized_service_seconds'] == 3
    assert measured['serialized_timing_scope'].endswith('not_gpu_only')
    with pytest.raises(ValueError, match='model failure'):
        predict(fail=True)
    assert released == [True, True]



def test_coarse_lanes_preserve_each_prompt_and_share_local_loader(tmp_path, monkeypatch):
    import threading
    import time
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from visioncortex import archive, coarse_recall
    model_path = tmp_path / 'world.pt'
    model_path.write_bytes(b'fixture')
    loads = []
    barrier = threading.Barrier(4)

    class Model:
        def to(self, device):
            return self

        def set_classes(self, prompts):
            self.prompts = prompts

        def predict(self, frame, **kwargs):
            expected = [str(int(frame[0, 0, 0]))]
            time.sleep(.01)
            assert self.prompts == expected
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setattr(archive, '_OPEN_VOCABULARY_MODEL_CACHE', {})
    monkeypatch.setattr(coarse_recall, 'load_yolo_world_with_local_clip',
                        lambda settings: (loads.append(settings), Model())[1])
    def invoke(number):
        barrier.wait()
        return coarse_recall._yolo_world_detections(
            np.full((4, 4, 3), number, dtype=np.uint8),
            {'model_path': str(model_path), 'prompt_map': {str(number): 'tube'}})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(4)))
    assert len(loads) == 1
    assert all(report['status'] == 'executed' for _, report in results)


@pytest.mark.parametrize('device', [0, 'cuda:0'])
def test_typed_cuda_oom_retries_same_input_on_cpu_after_releasing_traceback(monkeypatch, device):
    import weakref
    from visioncortex.open_vocabulary_runtime import run_with_cuda_oom_cpu_fallback

    class OOM(RuntimeError):
        pass

    class Allocation:
        pass

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=OOM, is_available=lambda: False)))
    refs, calls, cleanup_calls = [], [], []
    source = object()
    settings = {'confidence': .03, 'image_size': 1280, 'prompts': ['hand', 'pipette']}

    def infer_once(actual_device):
        calls.append((source, actual_device, dict(settings)))
        if actual_device != 'cpu':
            allocation = Allocation()
            refs.append(weakref.ref(allocation))
            raise OOM('CUDA allocation failed')
        return ['detection'], {'status': 'executed', 'actual_device': 'cpu'}

    def cleanup():
        assert refs[0]() is None
        cleanup_calls.append(True)

    boxes, receipt = run_with_cuda_oom_cpu_fallback(
        infer_once, device=device, enabled=True, cleanup=cleanup)
    assert boxes == ['detection']
    assert calls == [(source, device, settings), (source, 'cpu', settings)]
    assert cleanup_calls == [True]
    assert receipt['actual_device'] == 'cpu'
    assert receipt['requested_device'] == str(device)
    assert receipt['precision'] == 'float32'
    assert receipt['device_fallback'] == 'cuda_out_of_memory_to_cpu'


@pytest.mark.parametrize('error_kind', ['configuration', 'generic_oom', 'cancel', 'typed_oom_disabled'])
def test_cpu_recovery_does_not_retry_configuration_cancellation_or_untyped_errors(monkeypatch, error_kind):
    from visioncortex.open_vocabulary_runtime import run_with_cuda_oom_cpu_fallback
    from visioncortex.runtime_control import ExecutionCancelled

    class OOM(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM)))
    error = {'configuration': ValueError('missing pinned asset'),
             'generic_oom': RuntimeError('out of memory'),
             'cancel': ExecutionCancelled('cancelled'),
             'typed_oom_disabled': OOM('CUDA allocation')}[error_kind]
    calls = []

    def infer_once(device):
        calls.append(device)
        raise error

    with pytest.raises(type(error)) as observed:
        run_with_cuda_oom_cpu_fallback(infer_once, device=0,
            enabled=error_kind != 'typed_oom_disabled', cleanup=lambda: pytest.fail('unexpected cleanup'))
    assert observed.value is error
    assert calls == [0]


def test_failed_cpu_attempt_is_marked_to_prevent_outer_gpu_retry(monkeypatch):
    from visioncortex.open_vocabulary_runtime import run_with_cuda_oom_cpu_fallback

    class OOM(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=OOM, is_available=lambda: False)))
    calls = []

    def infer_once(device):
        calls.append(device)
        raise OOM('CUDA allocation') if device == 0 else RuntimeError('CPU failure')

    with pytest.raises(RuntimeError, match='CPU failure') as error:
        run_with_cuda_oom_cpu_fallback(infer_once, device=0, enabled=True, cleanup=lambda: None)
    assert calls == [0, 'cpu']
    assert error.value.open_vocabulary_recovery_attempted is True


def test_cpu_parking_resets_predictor_under_shared_lock(monkeypatch):
    from visioncortex import open_vocabulary_runtime as runtime

    calls = []

    class Guard:
        def __enter__(self):
            calls.append('locked')

        def __exit__(self, *_args):
            calls.append('unlocked')

    class Model:
        predictor = object()

        def to(self, device):
            assert calls == ['locked']
            calls.append(device)

    monkeypatch.setattr(runtime, '_OPEN_VOCABULARY_LOCK', Guard())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    model = Model()
    runtime.park_open_vocabulary_model(model)
    assert model.predictor is None
    assert calls == ['locked', 'cpu', 'unlocked']


@pytest.mark.parametrize('actual,expected', [('cpu', 'cpu'), ('cuda:1', 'cuda:1'), (None, 'unknown')])
def test_execution_receipt_never_substitutes_requested_device(monkeypatch, actual, expected):
    from visioncortex.open_vocabulary_runtime import run_with_cuda_oom_cpu_fallback

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError)))
    _, receipt = run_with_cuda_oom_cpu_fallback(
        lambda _: ([], {'actual_device': actual}), device=0, enabled=True, cleanup=lambda: None)
    assert receipt['actual_device'] == expected
    assert receipt['requested_device'] == '0'


def test_recovery_cannot_report_cpu_success_when_predictor_still_uses_cuda(monkeypatch):
    from visioncortex.open_vocabulary_runtime import run_with_cuda_oom_cpu_fallback

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=MemoryError, is_available=lambda: False)))
    def infer_once(device):
        if device == 0:
            raise MemoryError('allocation')
        return [], {'actual_device': 'cuda:0'}
    with pytest.raises(RuntimeError, match='did not confirm CPU') as error:
        run_with_cuda_oom_cpu_fallback(infer_once, device=0, enabled=True, cleanup=lambda: None)
    assert error.value.open_vocabulary_recovery_attempted


def test_cpu_retry_cancellation_propagates_without_error_conversion(monkeypatch):
    from visioncortex.open_vocabulary_runtime import run_with_cuda_oom_cpu_fallback
    from visioncortex.runtime_control import ExecutionCancelled

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=MemoryError, is_available=lambda: False)))
    cancellation = ExecutionCancelled('cancelled during recovery')
    def infer_once(device):
        if device == 0:
            raise MemoryError('allocation')
        raise cancellation
    with pytest.raises(ExecutionCancelled) as observed:
        run_with_cuda_oom_cpu_fallback(infer_once, device=0, enabled=True, cleanup=lambda: None)
    assert observed.value is cancellation
    assert not hasattr(cancellation, 'open_vocabulary_recovery_attempted')


def test_device_receipt_reads_active_predictor_before_wrapper_device():
    from visioncortex.open_vocabulary_runtime import yolo_world_prediction_device

    model = SimpleNamespace(device='cuda:0', predictor=SimpleNamespace(device='cpu'))
    assert yolo_world_prediction_device(model) == 'cpu'
    model.predictor = None
    assert yolo_world_prediction_device(model) == 'cuda:0'
    assert yolo_world_prediction_device(object()) == 'unknown'


def test_yolo_warm_cache_restores_predictor_after_cpu_parking(tmp_path, monkeypatch):
    import numpy as np
    from visioncortex import archive, coarse_recall

    model_path = tmp_path / 'world.pt'
    model_path.write_bytes(b'fake model')
    class Model:
        device = 'cuda:0'
        predictor = SimpleNamespace(device='cuda:0')
        setups = 0

        def to(self, device):
            self.device = device
            return self

        def predict(self, _frame, **kwargs):
            if self.predictor is None:
                self.device = 'cuda:0' if kwargs['device'] == 0 else kwargs['device']
                self.predictor = SimpleNamespace(device=self.device)
                self.setups += 1
            return [SimpleNamespace(boxes=None)]

    model = Model()
    monkeypatch.setattr(archive, '_OPEN_VOCABULARY_MODEL_CACHE', {
        str(model_path): {'model': model, 'prompts': ['paper']}})
    monkeypatch.setattr(archive, '_GROUNDING_DINO_MODEL_CACHE', {})
    monkeypatch.setattr(archive, 'release_temporal_segmentation_model_cache', lambda **_: 0)
    monkeypatch.setattr(archive, 'release_liquid_semantic_model_cache', lambda **_: 0)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=MemoryError, is_available=lambda: False)))
    archive._release_auxiliary_model_caches(retain_on_cpu=True)
    assert model.predictor is None
    assert model.device == 'cpu'
    _, receipt = coarse_recall._yolo_world_detections(np.zeros((8, 8, 3), dtype=np.uint8), {
        'model_path': str(model_path), 'prompt_map': {'paper': 'paper'}, 'device': 0,
    })
    assert model.setups == 1
    assert receipt['actual_device'] == 'cuda:0'

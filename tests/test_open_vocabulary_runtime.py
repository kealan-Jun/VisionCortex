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

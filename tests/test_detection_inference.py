import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from threading import get_ident

import numpy as np
import pytest

from visioncortex.detection import FramePacket, RoleScanner, _read_checkpoint, _write_checkpoint
from visioncortex.detection_inference import prediction_branches, prediction_contract
from visioncortex.schemas import ViewInput, ViewRole


@pytest.mark.parametrize('timeouts', [0, 1, 2, 3])
def test_nms_timeouts_retry_whole_batch_or_refuse_partial_evidence(timeouts):
    import logging
    from visioncortex.detection_inference import predict_complete_batch
    logger = logging.getLogger('ultralytics')
    before = list(logger.handlers)
    calls = []
    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            if len(calls) <= timeouts:
                logger.warning('NMS time limit 2.800s exceeded')
                return ['partial']
            return ['complete', 'complete']
    if timeouts == 3:
        with pytest.raises(RuntimeError, match='refusing incomplete'):
            predict_complete_batch(Model(), source=['a', 'b'], options={'conf': .25})
    else:
        results, retries = predict_complete_batch(Model(), source=['a', 'b'], options={'conf': .25})
        assert results == ['complete', 'complete']
        assert retries == timeouts
    assert len(calls) == min(timeouts + 1, 3)
    assert all(c == {'source': ['a', 'b'], 'conf': .25} for c in calls)
    assert logger.handlers == before


def test_nms_timeout_in_other_inference_thread_does_not_discard_this_batch():
    import logging
    from threading import Thread
    from visioncortex.detection_inference import predict_complete_batch
    class Model:
        def predict(self, **kwargs):
            other = Thread(target=lambda: logging.getLogger('ultralytics').warning('NMS time limit exceeded'))
            other.start()
            other.join()
            return ['complete']
    assert predict_complete_batch(Model(), source=['a'], options={}) == (['complete'], 0)


@pytest.mark.parametrize("value", [None, [], True, {"cam01": "one2many"},
                                  {"first_person": False}, {"third_person": "automatic"}])
def test_bad_branch_configuration_does_not_silently_fall_back(value):
    with pytest.raises(ValueError):
        prediction_branches({"models": {"prediction_branch_by_role": value}})


def test_branch_override_is_opt_in_and_role_specific():
    assert prediction_branches({}) == {}
    assert prediction_branches({"models": {"prediction_branch_by_role": {
        "third_person": "one2many",
    }}}) == {ViewRole.THIRD_PERSON: "one2many"}


def test_changed_branch_rejects_even_unfinished_checkpoint_without_touching_output(tmp_path):
    ledger, checkpoint = tmp_path / "partial.jsonl", tmp_path / "checkpoint.json"
    ledger.write_bytes(b"partially emitted evidence\n")
    _write_checkpoint(checkpoint, set(), ledger, prediction_policy={"branch": "one2one"})
    with pytest.raises(RuntimeError, match="prediction policy changed"):
        _read_checkpoint(checkpoint, ledger, prediction_policy={"branch": "one2many"})
    assert ledger.read_bytes() == b"partially emitted evidence\n"


def local_config(tmp_path, default_config):
    config = copy.deepcopy(default_config)
    for role in ViewRole:
        model = tmp_path / f"{role.value}.pt"
        model.write_bytes(b"fake model for CPU contract test")
        config["models"][role.value] = str(model)
    config["models"]["expected_class_count"] = 1
    config["performance"]["tensor_rt"] = "false"
    return config


@pytest.mark.parametrize("change", ["branch", "weight", "confidence", "image_size", "disabled"])
def test_branch_contract_drift_rejects_resume_before_truncating_tail(tmp_path, default_config, change):
    config = local_config(tmp_path, default_config)
    model = Path(config["models"]["first_person"])
    policy = prediction_contract("one2many", model, config, 960)
    ledger, checkpoint = tmp_path / "evidence.jsonl", tmp_path / "checkpoint.json"
    ledger.write_bytes(b"retained frame\n")
    _write_checkpoint(checkpoint, {0}, ledger, prediction_policy=policy)
    saved_checkpoint = checkpoint.read_bytes()
    ledger.write_bytes(b"retained frame\nuncommitted tail\n")
    if change == "weight":
        model.write_bytes(b"different model")
    if change == "confidence":
        config["models"]["confidence"] = .123
    replacement = None if change == "disabled" else prediction_contract(
        "one2one" if change == "branch" else "one2many", model, config,
        640 if change == "image_size" else 960,
    )
    with pytest.raises(RuntimeError, match="prediction policy changed"):
        _read_checkpoint(checkpoint, ledger, prediction_policy=replacement)
    assert ledger.read_bytes() == b"retained frame\nuncommitted tail\n"
    assert checkpoint.read_bytes() == saved_checkpoint
    assert _read_checkpoint(checkpoint, ledger, prediction_policy=policy) == {0}
    assert ledger.read_bytes() == b"retained frame\n"


def fake_yolo(monkeypatch, *, ignore_requested_branch=False, supported=True,
              setup_error=None, warmup_error=None, device_type="cpu",
              exported_size=None, dynamic=False):
    created = []

    class Detector:
        def __init__(self, path):
            head = SimpleNamespace(cv2=object(), cv3=object())
            if supported:
                head.one2one_cv2, head.one2one_cv3 = object(), object()
            self.model = SimpleNamespace(model=[head], end2end=True)
            self.calls = []
            self.overrides = {}
            self.callbacks = {}
            self.predictor = None
            self.predictors = []
            self.path = path
            created.append(self)

        @property
        def names(self):
            raise AssertionError("Model.names would create a disposable engine context")

        def _smart_load(self, key):
            assert key == "predictor"
            detector = self

            class Predictor:
                def __init__(self, *, overrides, _callbacks):
                    self.args = SimpleNamespace(**overrides)
                    self.done_warmup = False
                    self.model = None
                    self.setup_calls = 0
                    self.warmup_shapes = []
                    self.owner = get_ident()
                    detector.predictors.append(self)

                def setup_model(self, *, model, verbose):
                    self.setup_calls += 1
                    if setup_error is not None:
                        raise setup_error
                    self.model = SimpleNamespace(
                        names={0: "pipette"},
                        end2end=True if ignore_requested_branch else getattr(self.args, "end2end", True),
                        stride=32, channels=3,
                        format="engine" if detector.path.endswith(".engine") else "pt",
                        warmup=self.warmup,
                    )
                    self.device = SimpleNamespace(type=device_type)
                    if exported_size is not None:
                        self.model.imgsz = exported_size
                        self.model.dynamic = dynamic
                        if not dynamic:
                            self.args.imgsz = exported_size

                def warmup(self, *, imgsz):
                    import torch

                    assert self.owner == get_ident()
                    assert torch.is_inference_mode_enabled()
                    self.warmup_shapes.append(imgsz)
                    if warmup_error is not None:
                        raise warmup_error

            return Predictor

        def predict(self, **kwargs):
            assert self.predictor is not None and self.predictor.done_warmup
            assert self.predictor.args.device == kwargs["device"]
            assert self.predictor.owner == get_ident()
            self.calls.append(kwargs)
            observed = True if ignore_requested_branch else kwargs.get("end2end", True)
            self.predictor.model.end2end = observed
            return [SimpleNamespace(boxes=None) for _ in kwargs["source"]]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.checks", SimpleNamespace(
        check_imgsz=lambda size, **kwargs: [size, size],
    ))
    return created


@pytest.mark.parametrize('requested,exported,dynamic,accepted', [
    (513, [544, 544], False, True),
    (640, [544, 544], False, False),
    (640, [544, 544], True, True),
    (544, [544, 576], False, False),
])
def test_engine_prepare_validates_requested_shape_before_ready(
    tmp_path, default_config, monkeypatch, requested, exported, dynamic, accepted,
):
    import json
    import math

    config = local_config(tmp_path, default_config)
    engine = tmp_path / 'model.engine'
    metadata = json.dumps({'batch': 4, 'dynamic': dynamic}).encode()
    engine.write_bytes(len(metadata).to_bytes(4, 'little') + metadata + b'fake-plan')
    config['models']['first_person_engine'] = str(engine)
    config['performance']['tensor_rt'] = 'required'
    created = fake_yolo(monkeypatch, exported_size=exported, dynamic=dynamic)
    monkeypatch.setattr(sys.modules['ultralytics.utils.checks'], 'check_imgsz',
                        lambda size, stride, **_: [math.ceil(size / stride) * stride] * 2)
    scanner = RoleScanner(ViewRole.FIRST_PERSON, config, image_size=requested, batch_size=2)
    try:
        if accepted:
            scanner.prepare()
            assert scanner._prepared
            size = math.ceil(requested / 32) * 32
            batch = 2 if dynamic else 4
            assert created[0].predictor.warmup_shapes == [(batch, 3, size, size)]
            assert scanner._prediction_options()['imgsz'] == requested
        else:
            with pytest.raises(ValueError, match='does not match static engine export'):
                scanner.prepare()
            assert scanner.initialization_failure_phase == 'image_size'
            assert not scanner._prepared
            assert created[0].predictors[0].warmup_shapes == []
        assert created[0].calls == []
    finally:
        scanner.close()


@pytest.mark.parametrize("branch,expected", [(None, None), ("one2one", True), ("one2many", False)])
def test_scanner_requests_and_observes_selected_branch(tmp_path, default_config, monkeypatch, branch, expected):
    config = local_config(tmp_path, default_config)
    if branch is not None:
        config["models"]["prediction_branch_by_role"] = {"first_person": branch}
    fake_yolo(monkeypatch)
    scanner = RoleScanner(ViewRole.FIRST_PERSON, config, image_size=960, batch_size=2)
    view = ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=tmp_path / "local.mp4")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    packet = FramePacket(view=view, frame_index=0, local_ms=0, frame=frame,
                         gray=frame[:, :, 0], previous_gray=None, motion_score=0)
    assert scanner.infer([packet]) == [[]]
    if branch is None:
        assert "end2end" not in scanner.model.calls[0]
    else:
        assert scanner.model.calls[0]["end2end"] is expected
        assert scanner.prediction_policy["branch"] == branch
    assert scanner.last_prediction_end2end is expected


def test_unsupported_engine_or_head_is_rejected(tmp_path, default_config, monkeypatch):
    config = local_config(tmp_path, default_config)
    config["models"]["prediction_branch_by_role"] = {"first_person": "one2many"}
    fake_yolo(monkeypatch, supported=False)
    with pytest.raises(ValueError, match="both candidate prediction branches"):
        RoleScanner(ViewRole.FIRST_PERSON, config)
    config["models"]["first_person"] = str(tmp_path / "frozen.engine")
    with pytest.raises(ValueError, match="PyTorch"):
        RoleScanner(ViewRole.FIRST_PERSON, config)


def test_backend_cannot_ignore_branch_request(tmp_path, default_config, monkeypatch):
    config = local_config(tmp_path, default_config)
    config["models"]["prediction_branch_by_role"] = {"first_person": "one2many"}
    fake_yolo(monkeypatch, ignore_requested_branch=True)
    scanner = RoleScanner(ViewRole.FIRST_PERSON, config)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    packet = SimpleNamespace(frame=frame)
    with pytest.raises(RuntimeError, match="did not honor"):
        scanner.infer([packet])


def test_broker_ready_has_one_persistent_warmed_predictor(tmp_path, default_config, monkeypatch):
    from visioncortex.shared_inference import InferenceBroker

    config = local_config(tmp_path, default_config)
    created = fake_yolo(monkeypatch)
    broker = InferenceBroker(lambda: RoleScanner(ViewRole.FIRST_PERSON, config, image_size=960, batch_size=2))
    try:
        detector = created[0]
        predictor = detector.predictor
        assert predictor.owner == broker.thread.ident
        assert predictor.setup_calls == 1
        assert predictor.warmup_shapes == [(1, 3, 960, 960)]
        assert detector.calls == []
        assert broker.stats()['initialization_phase'] == 'ready'
        assert broker.stats()['model_calls'] == 0
        assert broker.stats()['frames'] == 0
        assert broker.stats()['engine_batch_size_counts'] == {}
        packets = [SimpleNamespace(frame=np.zeros((8, 8, 3), dtype=np.uint8))]
        assert broker.submit(packets) == [[]]
        assert broker.submit(packets) == [[]]
        assert detector.predictor is predictor
        assert len(detector.predictors) == 1
        assert len(predictor.warmup_shapes) == 1
        assert broker.stats()['frames'] == 2
    finally:
        assert broker.close()
    assert detector.predictor is None
    assert predictor.model is None


@pytest.mark.parametrize('exact,expected_batch', [(False, 2), (True, 4)])
def test_engine_preparation_uses_execution_batch_without_evidence(
    tmp_path, default_config, monkeypatch, exact, expected_batch,
):
    import json

    config = local_config(tmp_path, default_config)
    engine = tmp_path / 'model.engine'
    metadata = json.dumps({'batch': 4, 'dynamic': not exact}).encode()
    engine.write_bytes(len(metadata).to_bytes(4, 'little') + metadata + b'fake-plan')
    config['models']['first_person_engine'] = str(engine)
    config['performance']['tensor_rt'] = 'required'
    created = fake_yolo(monkeypatch)
    scanner = RoleScanner(ViewRole.FIRST_PERSON, config, batch_size=2)
    try:
        scanner.prepare()
        scanner.prepare()
        assert created[0].predictor.warmup_shapes == [(expected_batch, 3, scanner.image_size, scanner.image_size)]
        assert scanner.last_engine_batch_sizes == []
        assert scanner.last_inference_batch_sizes == []
        assert scanner.exact_batch_padding_frames == 0
        assert scanner.batch_contractions == []
    finally:
        scanner.close()


@pytest.mark.parametrize('failure_phase', ['predictor_setup', 'class_names', 'warmup'])
def test_failed_preparation_discards_predictor_and_cannot_be_reused(
    tmp_path, default_config, monkeypatch, failure_phase,
):
    config = local_config(tmp_path, default_config)
    if failure_phase == 'class_names':
        config['models']['expected_class_count'] = 2
    created = fake_yolo(
        monkeypatch,
        setup_error=AttributeError('context unavailable') if failure_phase == 'predictor_setup' else None,
        warmup_error=RuntimeError('CUDA out of memory') if failure_phase == 'warmup' else None,
    )
    scanner = RoleScanner(ViewRole.FIRST_PERSON, config)
    with pytest.raises((AttributeError, ValueError, RuntimeError)):
        scanner.prepare()
    assert scanner.initialization_failure_phase == failure_phase
    assert scanner.model is None
    assert not scanner._prepared
    assert created[0].predictor is None
    assert created[0].predictors[0].model is None
    assert created[0].calls == []
    with pytest.raises(RuntimeError, match='already closed'):
        scanner.prepare()
    scanner.close()


@pytest.mark.parametrize('fails', [False, True])
def test_cuda_completion_precedes_prepared_state_without_real_cuda(
    tmp_path, default_config, monkeypatch, fails,
):
    import torch

    created = fake_yolo(monkeypatch, device_type='cuda')
    scanner = RoleScanner(ViewRole.FIRST_PERSON, local_config(tmp_path, default_config))
    synchronized = []

    def synchronize(device):
        assert device.type == 'cuda'
        assert not scanner._prepared
        assert not created[0].predictor.done_warmup
        synchronized.append(True)
        if fails:
            raise RuntimeError('warmup execution failed')

    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    try:
        if fails:
            with pytest.raises(RuntimeError, match='warmup execution failed'):
                scanner.prepare()
            assert scanner.model is None
            assert scanner.initialization_failure_phase == 'warmup'
        else:
            scanner.prepare()
            assert scanner._prepared
            assert created[0].predictor.done_warmup
        assert synchronized == [True]
        assert created[0].calls == []
    finally:
        scanner.close()

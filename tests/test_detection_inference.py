import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex.detection import FramePacket, RoleScanner, _read_checkpoint, _write_checkpoint
from visioncortex.detection_inference import prediction_branches, prediction_contract
from visioncortex.schemas import ViewInput, ViewRole


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


def fake_yolo(monkeypatch, *, ignore_requested_branch=False, supported=True):
    class Detector:
        def __init__(self, path):
            self.names = {0: "pipette"}
            head = SimpleNamespace(cv2=object(), cv3=object())
            if supported:
                head.one2one_cv2, head.one2one_cv3 = object(), object()
            self.model = SimpleNamespace(model=[head], end2end=True)
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            observed = True if ignore_requested_branch else kwargs.get("end2end", True)
            self.predictor = SimpleNamespace(model=SimpleNamespace(end2end=observed))
            return [SimpleNamespace(boxes=None) for _ in kwargs["source"]]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))


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

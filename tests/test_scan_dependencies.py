from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from visioncortex.device_day_contract import digest
from visioncortex.device_day_models import DeviceDayModels
from visioncortex.scan_dependencies import phase_config
from visioncortex.schemas import ViewRole


@pytest.fixture
def setup(default_config, tmp_path):
    config = deepcopy(default_config)
    config["storage"].update(local_cache_root=str(tmp_path / "cache"),
                             local_runtime_root=str(tmp_path / "runtime"))
    config["performance"].update(tensor_rt="required", coarse_inference_batch_wait_ms=17)
    for role in ViewRole:
        for suffix in ("", "_engine", "_coarse_engine"):
            key = role.value + suffix
            path = tmp_path / (key + (".engine" if suffix else ".pt"))
            path.write_bytes(key.encode())
            config["models"][key] = str(path)
    info = SimpleNamespace(duration_ms=1000, fps=30, frame_count=30)
    retention = {"recording": {"camera_key": "arbitrary-sensor", "configured_role": "third_person"},
                 "sources": [{"kind": kind, "retained": {"sha256": kind}} for kind in ("video", "clock")]}
    return config, retention, info


def identity(setup, role="third_person", phase="fine"):
    config, retention, info = setup
    retention = deepcopy(retention)
    retention["recording"]["configured_role"] = role
    return DeviceDayModels(config).scan_identity(retention, info, phase, [(0, 1000)], 2, 640)


def test_third_fine_upgrade_invalidates_only_its_selected_scan(setup, tmp_path):
    config, _, _ = setup
    before = {(r.value, phase): identity(setup, r.value, phase)
              for r in ViewRole for phase in ("fine", "coarse")}
    for key, suffix in [("third_person", ".pt"), ("third_person_engine", ".engine")]:
        path = tmp_path / ("replacement" + suffix)
        path.write_bytes(b"replacement")
        config["models"][key] = str(path)
    after = {(r, phase): identity(setup, r, phase) for r, phase in before}
    assert [key for key in before if digest(before[key]) != digest(after[key])] == [("third_person", "fine")]
    assert after["third_person", "fine"]["selected_detector"]["backend"] == "TensorRT"


def test_coarse_without_override_depends_on_fine_engine(setup):
    config, _, _ = setup
    del config["models"]["third_person_coarse_engine"]
    before = identity(setup, phase="coarse")
    Path(config["models"]["third_person_engine"]).write_bytes(b"new-engine")
    after = identity(setup, phase="coarse")
    assert digest(before) != digest(after)
    assert before["selected_detector"]["path"] == config["models"]["third_person_engine"]


def test_fine_role_calibration_invalidates_only_affected_scan(setup):
    config, _, _ = setup
    before = {(r.value, phase): identity(setup, r.value, phase)
              for r in ViewRole for phase in ("fine", "coarse")}
    config['models']['confidence_by_role'] = {'first_person': .225}
    changed = [key for key, value in before.items() if value != identity(setup, *key)]
    assert changed == [('first_person', 'fine')]
    before = {(r.value, phase): identity(setup, r.value, phase)
              for r in ViewRole for phase in ("fine", "coarse")}
    config['models']['coarse_confidence_by_role'] = {'third_person': .32}
    assert [key for key, value in before.items() if value != identity(setup, *key)] == [('third_person', 'coarse')]


def test_auto_fallback_and_pytorch_mode_track_selected_bytes(setup):
    config, _, _ = setup
    config["performance"]["tensor_rt"] = "auto"
    engine = Path(config["models"]["third_person_engine"])
    before = identity(setup)
    engine.unlink()
    fallback = identity(setup)
    assert fallback["selected_detector"]["backend"] == "PyTorch"
    assert digest(fallback) != digest(before)
    config["performance"]["tensor_rt"] = "false"
    selected = identity(setup)
    engine.write_bytes(b"irrelevant-engine")
    assert identity(setup) == selected
    Path(config["models"]["third_person"]).write_bytes(b"new-checkpoint")
    assert identity(setup) != selected


def test_missing_required_engine_cannot_reuse_cache(setup):
    config, _, _ = setup
    Path(config["models"]["third_person_engine"]).unlink()
    with pytest.raises(FileNotFoundError):
        identity(setup)


@pytest.mark.parametrize("change", ["source", "clock", "threshold", "window", "fps", "code", "unknown"])
def test_relevant_or_unknown_changes_invalidate(setup, change, monkeypatch):
    config, retention, info = setup
    model = DeviceDayModels(config)
    before = model.scan_identity(retention, info, "fine", [(0, 1000)], 2, 640)
    windows, fps = [(0, 1000)], 2
    if change in {"source", "clock"}:
        retention["sources"][0 if change == "source" else 1]["retained"]["sha256"] = "changed"
    elif change == "threshold":
        model.config["models"]["confidence"] = .731
    elif change == "window":
        windows = [(0, 750)]
    elif change == "fps":
        fps = 3
    elif change == "unknown":
        model.config["models"]["future_model_setting"] = {"mode": "new"}
    else:
        old_hash = model._identity_hash
        monkeypatch.setattr(model, "_identity_hash", lambda p: "new-code" if p.name == "scan_dependencies.py" else old_hash(p))
    after = model.scan_identity(retention, info, "fine", windows, fps, 640)
    assert digest(before) != digest(after)


def test_phase_config_shared_by_scanner_without_mutating_input(setup, monkeypatch, tmp_path):
    from visioncortex import scan_scheduler
    config, _, _ = setup
    saved = deepcopy(config)
    config["performance"]["coarse_cuda_max_concurrent_sources"] = 0
    received = []
    monkeypatch.setattr(scan_scheduler, "_scan_views_concurrently", lambda cfg, *a, **k: received.append(cfg))
    scan_scheduler._admitted_scan(config, [], {}, {}, tmp_path, phase="coarse")
    assert received[0] == phase_config(config, "coarse")
    assert received[0]["performance"]["inference_batch_wait_ms"] == 17
    assert config["models"] == saved["models"]
    assert config["performance"]["inference_batch_wait_ms"] == saved["performance"]["inference_batch_wait_ms"]


def test_real_cache_boundary_reuses_unrelated_change_but_rejects_tampering(setup, monkeypatch):
    from visioncortex import scan_scheduler
    config, retention, info = setup
    view = SimpleNamespace(view_id="arbitrary-sensor", role=ViewRole.THIRD_PERSON)
    calls = []
    def scan(cfg, views, infos, transforms, directory, **kwargs):
        calls.append(1)
        directory.mkdir(parents=True)
        ledger = directory / "fixture.jsonl"
        ledger.write_text("deterministic-fixture-not-real-inference")
        return {view.view_id: ledger}
    monkeypatch.setattr(scan_scheduler, "scan_views_concurrently", scan)
    model = DeviceDayModels(config)
    args = (view, info, None, retention, "fine", [(0, 1000)], 2, 640)
    _, directory, first = model._scan(*args)
    Path(config["models"]["first_person_engine"]).write_bytes(b"unrelated-first-upgrade")
    _, same, second = model._scan(*args)
    assert len(calls) == 1 and not first["reused"] and second["reused"] and same == directory
    Path(config["models"]["third_person_engine"]).write_bytes(b"selected-third-upgrade")
    _, changed, third = model._scan(*args)
    assert len(calls) == 2 and not third["reused"] and changed != directory
    (changed / "fixture.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="cache was modified"):
        model._scan(*args)

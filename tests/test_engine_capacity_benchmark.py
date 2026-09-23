from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "engine_capacity_benchmark", ROOT / "tools" / "benchmark_engine_capacity.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def config():
    return {
        "performance": {
            "batch_size": 4, "fine_batch_size": 4, "coarse_batch_size": 16,
            "image_size": 640, "detection_fps": 20, "inference_batch_wait_ms": 25,
        },
        "models": {
            "first_person": "/models/first.pt", "first_person_engine": "/engines/first.engine",
            "confidence": 0.25,
        },
        "storage": {
            "local_cache_root": "/baseline/Cache", "local_runtime_root": "/baseline/Runtime",
            "archive_root": "/archive",
        },
        "actions": {"minimum_contact_frames": 3},
    }


@pytest.fixture
def clean_git(monkeypatch):
    def run(command, **kwargs):
        return SimpleNamespace(stdout="a" * 40 + "\n" if "rev-parse" in command else "")

    monkeypatch.setattr(benchmark.subprocess, "run", run)


def test_scan_rejects_partial_output_before_importing_runtime(tmp_path):
    output = tmp_path / "partial"
    output.mkdir()
    checkpoint = output / "partial.checkpoint.json"
    checkpoint.write_text("{}")
    with pytest.raises(FileExistsError, match="must be empty"):
        benchmark.scan(SimpleNamespace(output=output))
    assert checkpoint.read_text() == "{}"


def test_build_rejects_existing_graph_before_loading_tensorrt(tmp_path, monkeypatch):
    output = tmp_path / "old-build"
    output.mkdir()
    graph = output / "source.onnx"
    graph.write_bytes(b"unverified graph")
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(FileExistsError, match="must be empty"):
        benchmark.build(SimpleNamespace(output=output))
    assert graph.read_bytes() == b"unverified graph"


def test_fresh_output_allows_only_absent_or_empty_directories(tmp_path):
    output = tmp_path / "new"
    assert benchmark.fresh_output_directory(output) == output
    assert benchmark.fresh_output_directory(output) == output
    (output / "Cache").mkdir()
    with pytest.raises(FileExistsError):
        benchmark.fresh_output_directory(output)


def test_identity_allows_only_engine_batch_and_owned_output_changes(config, clean_git, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"views": []}')
    original = deepcopy(config)
    baseline = benchmark.comparison_identity(config, manifest)
    candidate = deepcopy(config)
    candidate["performance"].update(batch_size=16, fine_batch_size=16, coarse_batch_size=32)
    candidate["models"]["first_person_engine"] = "/candidate/first.engine"
    candidate["models"]["third_person_coarse_engine"] = "/candidate/coarse.engine"
    candidate["storage"].update(local_cache_root="/candidate/Cache", local_runtime_root="/candidate/Runtime")
    assert benchmark.comparison_identity(candidate, manifest) == baseline
    assert config == original
    assert baseline["code_sha"] == "a" * 40
    assert baseline["git_dirty"] is False
    assert baseline["manifest_sha256"] == benchmark.sha256(manifest)


@pytest.mark.parametrize(("section", "key", "value"), [
    ("models", "confidence", 0.10),
    ("models", "first_person", "/other/weights.pt"),
    ("performance", "image_size", 512),
    ("performance", "detection_fps", 10),
    ("performance", "inference_batch_wait_ms", 500),
    ("storage", "archive_root", "/other/archive"),
    ("actions", "minimum_contact_frames", 2),
])
def test_identity_preserves_all_semantic_settings(config, clean_git, tmp_path, section, key, value):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    baseline = benchmark.comparison_identity(config, manifest)
    config[section][key] = value
    candidate = benchmark.comparison_identity(config, manifest)
    assert candidate["config_sha256"] != baseline["config_sha256"]


def test_identity_records_dirty_source_and_manifest_change(config, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(benchmark.subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        stdout="b" * 40 if "rev-parse" in command else " M tools/benchmark_engine_capacity.py\n"
    ))
    baseline = benchmark.comparison_identity(config, manifest)
    manifest.write_text('{"changed": true}')
    candidate = benchmark.comparison_identity(config, manifest)
    assert baseline["git_dirty"] is True
    assert candidate["manifest_sha256"] != baseline["manifest_sha256"]


@pytest.mark.parametrize(("capacity", "effective"), [(None, 16), (4, 4), (32, 8)])
def test_capacity_guard_rejects_silent_clipping_or_warmup_oom(capacity, effective):
    scanner = SimpleNamespace(engine_build_batch=capacity, batch_size=effective)
    with pytest.raises(ValueError, match="NOT_PROVEN"):
        benchmark.require_requested_batch(scanner, 16)


def test_engine_histogram_counts_actual_calls_including_internal_chunks():
    histogram = Counter()
    scanner = SimpleNamespace(engine_build_batch=32, batch_size=16, last_engine_batch_sizes=[16])
    benchmark.require_requested_batch(scanner, 16)
    benchmark.record_engine_batches(histogram, scanner)
    scanner.last_engine_batch_sizes = [8, 8]
    benchmark.record_engine_batches(histogram, scanner)
    assert histogram == {16: 1, 8: 2}
    scanner.last_engine_batch_sizes = []
    with pytest.raises(ValueError, match="missing"):
        benchmark.record_engine_batches(histogram, scanner)


@pytest.fixture
def model_files(tmp_path):
    paths = {}
    for key in ("first_person", "third_person", "first_person_engine", "third_person_engine"):
        path = tmp_path / (key + (".engine" if key.endswith("_engine") else ".pt"))
        path.write_bytes(key.encode())
        paths[key] = str(path)
    return {"models": paths}


def test_model_snapshot_embeds_existing_build_identity_without_inventing_missing_receipts(model_files):
    engine = Path(model_files["models"]["first_person_engine"])
    sidecar = engine.with_suffix(".build.json")
    build = {"engine_sha256": benchmark.sha256(engine),
             "weights_sha256": benchmark.sha256(Path(model_files["models"]["first_person"]))}
    sidecar.write_text(json.dumps(build))
    identities = benchmark.model_identities(model_files)
    candidate = identities["first_person_engine"]
    assert candidate["build_receipt"] == build
    assert candidate["build_receipt_sha256"] == benchmark.sha256(sidecar)
    assert candidate["sha256"] == build["engine_sha256"]
    assert identities["first_person"]["sha256"] == build["weights_sha256"]
    assert "build_receipt" not in identities["third_person_engine"]
    assert "build_receipt_sha256" not in identities["third_person_engine"]
    benchmark.verify_model_identities(model_files, identities)


@pytest.mark.parametrize("changed", ["weights", "engine", "sidecar"])
def test_model_snapshot_rejects_midrun_identity_changes(model_files, changed):
    identities = benchmark.model_identities(model_files)
    if changed == "sidecar":
        path = Path(model_files["models"]["first_person_engine"]).with_suffix(".build.json")
        path.write_text('{"unexpected": "new receipt"}')
    else:
        path = Path(model_files["models"]["first_person" if changed == "weights" else "first_person_engine"])
        path.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="changed during the scan"):
        benchmark.verify_model_identities(model_files, identities)


def test_model_snapshot_rejects_file_changed_while_hashing(model_files, monkeypatch):
    original = benchmark.sha256

    def mutate(path):
        digest = original(path)
        path.write_bytes(path.read_bytes() + b"changed")
        return digest

    monkeypatch.setattr(benchmark, "sha256", mutate)
    with pytest.raises(RuntimeError, match="changed while reading"):
        benchmark.model_identities(model_files)


@pytest.mark.parametrize("status", ["passed", "rejected", "insufficient_evidence"])
def test_compare_cli_writes_receipt_and_rejects_failed_gates(monkeypatch, tmp_path, capsys, status):
    module = ModuleType("visioncortex.fine_batch_acceptance")
    calls = []

    def compare(baseline, candidate, target_batch):
        calls.append((baseline, candidate, target_batch))
        return {"comparison_status": status, "promotion_ready": False}

    module.compare_fine_scans = compare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "comparison.json"
    baseline, candidate = tmp_path / "Baseline", tmp_path / "Candidate"
    monkeypatch.setattr(sys, "argv", ["benchmark", "compare", "--baseline", str(baseline),
                                    "--candidate", str(candidate), "--output", str(output)])
    if status == "passed":
        benchmark.main()
    else:
        with pytest.raises(SystemExit) as exc:
            benchmark.main()
        assert exc.value.code == 1
    assert calls == [(baseline, candidate, 16)]
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)
    with pytest.raises(FileExistsError):
        benchmark.main()
    assert len(calls) == 1


@pytest.mark.parametrize("contract_during_timing", [False, True])
def test_infer_records_only_timed_engine_batches(monkeypatch, tmp_path, contract_during_timing):
    from visioncortex.schemas import ViewRole

    source = tmp_path / "source.mp4"
    engine = tmp_path / "candidate.engine"
    engine.write_bytes(b"test engine identity only")
    role = ViewRole.FIRST_PERSON
    frame = SimpleNamespace(tobytes=lambda: b"deterministic test frame")
    capture = SimpleNamespace(isOpened=lambda: True, get=lambda key: 10, set=lambda *args: None,
                              read=lambda: (True, frame), release=lambda: None)
    modules = {
        "cv2": {"VideoCapture": lambda _: capture, "cvtColor": lambda frame, _: frame,
                "COLOR_BGR2GRAY": 1, "CAP_PROP_FPS": 2, "CAP_PROP_FRAME_COUNT": 3,
                "CAP_PROP_POS_MSEC": 4},
        "torch": {"cuda": SimpleNamespace(synchronize=lambda: None)},
        "visioncortex.config": {
            "load_config": lambda _: {"models": {}, "performance": {}},
            "load_manifest": lambda _: SimpleNamespace(views=[SimpleNamespace(
                role=role, video=source, view_id="first-camera")]),
        },
    }

    class Scanner:
        def __init__(self, *args, **kwargs):
            self.engine_build_batch = 32
            self.batch_size = 16
            self.batch_contractions = []
            self.calls = 0

        def infer(self, packets):
            self.calls += 1
            if contract_during_timing and self.calls == 7:
                self.batch_size = 8
                self.batch_contractions.append({"from_batch_size": 16, "to_batch_size": 8})
            self.last_engine_batch_sizes = [8, 8] if self.calls == 7 and contract_during_timing else [len(packets)]
            return [[] for _ in packets]

        def close(self):
            pass

    modules["visioncortex.detection"] = {
        "RoleScanner": Scanner, "FramePacket": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    modules["visioncortex.telemetry"] = {"ResourceMonitor": lambda *args: SimpleNamespace(
        start=lambda: None, stop=lambda: None)}
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    clock = iter([0, .1, .2, .3, .4, .5, .6, 1.1, 1.2])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    output = tmp_path / "run"
    args = SimpleNamespace(output=output, config=tmp_path / "config.yaml", role=role.value,
                           engine=engine, batch=16, manifest=tmp_path / "manifest.json", samples=2,
                           image_size=640, seconds=1)
    if contract_during_timing:
        with pytest.raises(RuntimeError, match="did not sustain"):
            benchmark.infer(args)
    else:
        benchmark.infer(args)
    receipt = json.loads((output / "result.json").read_text())
    assert receipt["timed_engine_batch_histogram"] == ({"8": 2, "16": 1} if contract_during_timing else {"16": 2})
    assert receipt["requested_batch_reached"] is not contract_during_timing
    assert receipt["capacity_evidence"] == ("NOT_PROVEN" if contract_during_timing else "PROVEN")
    assert receipt["frames"] == 32

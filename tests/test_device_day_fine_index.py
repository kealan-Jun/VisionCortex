"""CPU-only evidence for fine-index I/O and unchanged coverage gates."""

from copy import deepcopy
from pathlib import Path

import pytest

from visioncortex import candidate_index
from visioncortex.device_day_contract import DeviceDayLayout, read_json, verify_artifact
from visioncortex.device_day_models import DeviceDayModels
from visioncortex.schemas import BoxEvidence, FrameEvidence, VideoInfo, ViewInput, ViewRole


@pytest.fixture
def fine_case(default_config, tmp_path):
    config = deepcopy(default_config)
    config["storage"].update(
        local_runtime_root=str(tmp_path / "runtime"),
        local_cache_root=str(tmp_path / "backend"),
        archive_root=str(tmp_path / "archive"),
    )
    config["performance"].update(
        detection_fps=2, fine_frame_index_enabled=True,
        fine_coverage_gate_enabled=True, fine_track_stitching_enabled=True,
    )
    layout = DeviceDayLayout(tmp_path / "archive", "camera", 1789005600000000, tmp_path / "backend")
    view = ViewInput(view_id="camera", role=ViewRole.FIRST_PERSON, video=tmp_path / "fixture.mp4")
    info = VideoInfo(path=view.video, duration_ms=1000, fps=30, frame_count=30, width=640, height=480)
    source = tmp_path / "fine.jsonl"

    def run(times=(0, 500)):
        source.write_text("\n".join(FrameEvidence(
            view_id=view.view_id, role=view.role, frame_index=i, local_ms=moment, global_ms=moment,
            width=640, height=480, detections=[BoxEvidence(
                class_id=0, class_name="tube", confidence=.9, xyxy_norm=(.1, .1, .2, .2), track_id=10,
            )],
        ).model_dump_json() for i, moment in enumerate(times)) + "\n")
        return DeviceDayModels(config)._index_fine(
            layout, {"recording_id": "fixture"}, "key", 0, view, info, {view.view_id: source}, [(0, 1000)],
        )

    return config, layout, source, run


def test_fine_source_parsed_once_and_local_audit_bytes_published_verified(fine_case, monkeypatch):
    config, layout, source, run = fine_case
    calls = []
    original = candidate_index.iter_frame_evidence

    def frames(path):
        calls.append(path)
        yield from original(path)

    monkeypatch.setattr(candidate_index, "iter_frame_evidence", frames)
    # Published large files use copy_verified's independent read-back digest.
    # Only the small final manifest still needs backend_artifact hashing.
    hashed = []
    original_artifact = layout.backend_artifact

    def artifact(path):
        hashed.append(path.name)
        return original_artifact(path)

    monkeypatch.setattr(layout, "backend_artifact", artifact)
    ledgers, report, artifacts = run((500, 0, 500))
    assert calls == [source]
    assert hashed == ["Manifest.json"]
    assert ledgers["camera"].is_relative_to(Path(config["storage"]["local_runtime_root"]))
    frames = list(original(ledgers["camera"]))
    assert [frame.local_ms for frame in frames] == [0, 500]
    assert [frame.frame_index for frame in frames] == [1, 2]  # Last overlapping observation wins.
    assert {box.track_id for frame in frames for box in frame.detections} == {1}
    assert all(verify_artifact(layout.backend_root, ref) for ref in artifacts)
    published = next(ref for ref in artifacts if ref["path"].endswith(".jsonl"))
    assert (layout.backend_root / published["path"]).read_bytes() == ledgers["camera"].read_bytes()
    manifest = next(ref for ref in artifacts if ref["path"].endswith("Manifest.json"))
    assert read_json(layout.backend_root / manifest["path"]) == report
    assert report["formal_evidence_ready"] is True
    assert report["window_coverage"][0]["sample_count"] == 2
    assert report["ledger_parse_passes"] == 1
    for phase in ("index_ingest", "coverage_query", "ledger_materialization", "index_publication", "ledger_publication"):
        assert report["component_timings"][phase + "_seconds"] >= 0
        assert report["component_timings"][phase + "_thread_cpu_seconds"] >= 0
        assert report["component_timings"][phase + "_non_cpu_seconds"] >= 0


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("times", [(500, 1000), (), (0, 0, 0)])
def test_half_open_window_coverage_still_fails_without_aggregate_gate(fine_case, indexed, times):
    config, layout, _, run = fine_case
    config["performance"].update(fine_frame_index_enabled=indexed, fine_coverage_gate_enabled=False)
    with pytest.raises(ValueError, match="sampling incomplete"):
        run(times)
    assert not list(layout.backend_root.rglob("*.jsonl"))


def test_fine_publication_refuses_existing_modified_snapshot(fine_case):
    _, layout, _, run = fine_case
    _, _, artifacts = run()
    snapshot = layout.backend_root / next(ref["path"] for ref in artifacts if ref["path"].endswith(".sqlite3"))
    snapshot.write_bytes(b"modified existing evidence")
    with pytest.raises(ValueError, match="collision"):
        run()
    assert snapshot.read_bytes() == b"modified existing evidence"


@pytest.mark.parametrize("fail", [False, True])
def test_local_audit_copy_released_but_published_evidence_preserved(fine_case, monkeypatch, fail):
    config, layout, source, run = fine_case
    ledgers, report, artifacts = run()
    model = DeviceDayModels(config)
    monkeypatch.setattr("visioncortex.actions.generate_candidates", lambda *args: [])

    def audit(*args):
        assert ledgers["camera"].is_file()
        if fail:
            raise ValueError("audit gate failed")
        return [], {}

    monkeypatch.setattr(model, "_audit_activity", audit)
    monkeypatch.setattr(model, "_key_frame_choices", lambda *args: {})
    if fail:
        with pytest.raises(ValueError, match="audit gate failed"):
            model._audit_fine(None, None, ledgers, report, [], [])
    else:
        assert model._audit_fine(None, None, ledgers, report, [], []) == ([], [], {}, {})
    assert not ledgers["camera"].exists()
    assert source.is_file()
    assert all(verify_artifact(layout.backend_root, ref) for ref in artifacts)
    # The index-disabled route still consumes its original authoritative file.
    monkeypatch.setattr(model, "_audit_activity", lambda *args: ([], {}))
    model._audit_fine(None, None, {"camera": source}, {"enabled": False}, [], [])
    assert source.is_file()


def test_optional_thread_cpu_timing_does_not_claim_io_is_nas(monkeypatch):
    from visioncortex.performance_stages import StageTimings
    wall, cpu = [100.0], [10.0]
    monkeypatch.setattr("visioncortex.performance_stages.time.perf_counter", lambda: wall[0])
    monkeypatch.setattr("visioncortex.performance_stages.time.thread_time", lambda: cpu[0])
    timings = StageTimings()
    with pytest.raises(ValueError), timings.measure("publication_seconds", cpu=True):
        wall[0] += 5
        cpu[0] += 2
        raise ValueError("fixture")
    assert timings.snapshot() == {
        "publication_seconds": 5, "publication_thread_cpu_seconds": 2, "publication_non_cpu_seconds": 3,
    }

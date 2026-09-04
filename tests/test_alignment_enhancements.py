from pathlib import Path

import numpy as np
import pytest

from visioncortex import alignment
from visioncortex.actions import audit_candidates
from visioncortex.alignment import (
    alignment_quality_report,
    audit_timestamp_series,
    build_alignments,
    iter_aligned_rows,
    read_timestamp_csv_bounded,
)
from visioncortex.config import load_config
from visioncortex.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentSegmentTransform,
    AlignmentTransform,
    VideoInfo,
    VideoSegmentInfo,
    VideoSegmentInput,
    ViewInput,
    ViewRole,
)


def _info(path: str, duration_ms: float) -> VideoInfo:
    return VideoInfo(
        path=Path(path),
        duration_ms=duration_ms,
        fps=10.0,
        width=64,
        height=48,
        frame_count=round(duration_ms / 100.0),
    )


def _view(view_id: str, role: ViewRole, clock: Path | None = None) -> ViewInput:
    return ViewInput(
        view_id=view_id,
        role=role,
        video=Path(f"{view_id}.mp4"),
        timestamps_csv=clock,
    )


def _write_clock(path: Path, source_start_ms: float, source_step_ms: float = 1000.0) -> None:
    rows = [
        f"{index * 10},{index * 1000},{source_start_ms + index * source_step_ms},1"
        for index in range(5)
    ]
    path.write_text(
        "frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def test_bounded_clock_samples_detect_mid_file_jump(tmp_path):
    clock = tmp_path / "clock.csv"
    clock.write_text(
        "frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n"
        "0,0,1000000,1\n"
        "10,1000,1001000,1\n"
        "20,2000,1005000,1\n"
        "30,3000,1006000,1\n"
        "40,4000,1007000,1\n",
        encoding="utf-8",
    )

    points = read_timestamp_csv_bounded(clock, 10.0, sample_count=5)
    receipt = audit_timestamp_series(
        points,
        jump_threshold_ms=500.0,
        max_rate_error_ppm=10_000.0,
    )

    assert len(points) == 5
    assert any(
        str(item).startswith("clock_step_ms:") for item in receipt["anomalies"]
    )


def test_large_clock_reader_uses_distributed_bounded_samples(tmp_path):
    clock = tmp_path / "clock.csv"
    rows = [
        f"{index},{index * 100},{1_000_000 + index * 100},1"
        for index in range(100)
    ]
    clock.write_text(
        "frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )

    points = read_timestamp_csv_bounded(
        clock,
        10.0,
        sample_count=5,
        full_read_limit_bytes=1,
    )

    assert len(points) == 5
    assert points[0].frame_index == 0
    assert points[-1].frame_index == 99
    assert any(0 < point.frame_index < 99 for point in points)


def test_invalid_clock_sync_samples_are_not_used_as_absolute_time(tmp_path):
    clock = tmp_path / "clock.csv"
    clock.write_text(
        "frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n"
        "0,0,1000000,1\n"
        "10,1000,1001000,0\n"
        "20,2000,1002000,1\n",
        encoding="utf-8",
    )

    points = read_timestamp_csv_bounded(clock, 10.0, sample_count=3)
    receipt = audit_timestamp_series(
        points,
        jump_threshold_ms=500.0,
        max_rate_error_ppm=10_000.0,
    )

    assert points[1].clock_sync_valid is False
    assert points[1].source_ms is None
    assert receipt["valid_source_count"] == 2
    assert "clock_sync_invalid_samples:1" in receipt["anomalies"]


def test_raw_system_clock_requires_explicit_sync_validity(tmp_path):
    clock = tmp_path / "clock.csv"
    clock.write_text(
        "frame_index,local_timestamp_ms,frame_system_timestamp_us\n"
        "0,0,1000000000\n"
        "10,1000,1001000000\n",
        encoding="utf-8",
    )

    points = read_timestamp_csv_bounded(clock, 10.0, sample_count=2)

    assert all(point.source_column == "frame_system_timestamp_us" for point in points)
    assert all(point.source_ms is None for point in points)


def test_two_clock_endpoints_cannot_claim_high_confidence(tmp_path, monkeypatch):
    fp_clock = tmp_path / "fp.csv"
    tp_clock = tmp_path / "tp.csv"
    header = "frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n"
    fp_clock.write_text(
        header + "0,0,1000000,1\n100,10000,1010000,1\n",
        encoding="utf-8",
    )
    tp_clock.write_text(
        header + "0,0,1000200,1\n100,10000,1010200,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        alignment,
        "visual_anchor_calibration",
        lambda *_args, **_kwargs: (0.0, 0.0, []),
    )
    config = load_config()
    views = [
        _view("fp", ViewRole.FIRST_PERSON, fp_clock),
        _view("tp", ViewRole.THIRD_PERSON, tp_clock),
    ]
    infos = {"fp": _info("fp.mp4", 10_000.0), "tp": _info("tp.mp4", 10_000.0)}

    transforms, _ = build_alignments(views, infos, config)

    assert transforms["tp"].confidence < config["alignment"][
        "visual_audit_csv_confidence_threshold"
    ]
    assert transforms["tp"].segment_transforms[0].state == "uncertain"


def test_best_clock_quality_replaces_invalid_preferred_reference(tmp_path, monkeypatch):
    fp_clock = tmp_path / "fp.csv"
    tp_clock = tmp_path / "tp.csv"
    fp_clock.write_text(
        "frame_index,local_timestamp_ms,global_timestamp_ms,clock_sync_valid\n"
        "0,0,1000000,0\n10,1000,1001000,0\n20,2000,1002000,0\n",
        encoding="utf-8",
    )
    _write_clock(tp_clock, 1_000_000.0)
    monkeypatch.setattr(
        alignment,
        "visual_anchor_calibration",
        lambda *_args, **_kwargs: (0.0, 0.0, []),
    )
    views = [
        _view("fp", ViewRole.FIRST_PERSON, fp_clock),
        _view("tp", ViewRole.THIRD_PERSON, tp_clock),
    ]
    infos = {"fp": _info("fp.mp4", 5000.0), "tp": _info("tp.mp4", 5000.0)}

    transforms, _ = build_alignments(views, infos, load_config())

    assert {item.reference_view_id for item in transforms.values()} == {"tp"}


def test_high_confidence_clock_still_runs_short_visual_audit(tmp_path, monkeypatch):
    fp_clock = tmp_path / "fp.csv"
    tp_clock = tmp_path / "tp.csv"
    _write_clock(fp_clock, 1_000_000.0)
    _write_clock(tp_clock, 1_000_200.0)
    calls: list[tuple[float, int, float, float]] = []

    def visual(*args, **_kwargs):
        calls.append(tuple(args[-4:]))
        return 0.0, 0.9, [
            {"reliable": True},
            {"reliable": True},
            {"reliable": True},
        ]

    monkeypatch.setattr(alignment, "visual_anchor_calibration", visual)
    config = load_config()
    views = [
        _view("fp", ViewRole.FIRST_PERSON, fp_clock),
        _view("tp", ViewRole.THIRD_PERSON, tp_clock),
    ]
    infos = {"fp": _info("fp.mp4", 5000.0), "tp": _info("tp.mp4", 5000.0)}

    transforms, _ = build_alignments(views, infos, config)

    assert calls == [
        (
            config["alignment"]["visual_audit_search_seconds"],
            config["alignment"]["visual_audit_anchor_count"],
            config["alignment"]["visual_audit_duration_seconds"],
            config["alignment"]["visual_audit_fps"],
        )
    ]
    assert transforms["tp"].alignment_basis.endswith("+visual_anchor")


def test_visual_audit_reuses_reference_signatures_across_views(monkeypatch):
    calls = {"fp": 0, "tp-1": 0, "tp-2": 0}

    def signature(view, _info, times):
        calls[view.view_id] += 1
        return np.arange(len(times), dtype=np.float64)

    monkeypatch.setattr(alignment, "view_motion_signature", signature)
    reference = _view("fp", ViewRole.FIRST_PERSON)
    targets = [
        _view("tp-1", ViewRole.THIRD_PERSON),
        _view("tp-2", ViewRole.THIRD_PERSON),
    ]
    infos = {view.view_id: _info(f"{view.view_id}.mp4", 60_000.0) for view in [reference, *targets]}
    ref_transform = AlignmentTransform(
        view_id="fp", reference_view_id="fp", state="aligned"
    )
    cache: dict[tuple[str, tuple[float, ...]], np.ndarray] = {}
    for target in targets:
        alignment.visual_anchor_calibration(
            reference,
            target,
            ref_transform,
            AlignmentTransform(
                view_id=target.view_id,
                reference_view_id="fp",
                state="aligned",
            ),
            infos["fp"],
            infos[target.view_id],
            1.0,
            3,
            2.0,
            2.0,
            signature_cache=cache,
        )

    assert calls["fp"] == 3
    assert calls["tp-1"] == 3
    assert calls["tp-2"] == 3


def test_build_alignments_keeps_per_segment_drift(tmp_path, monkeypatch):
    fp_clocks = [tmp_path / "fp-1.csv", tmp_path / "fp-2.csv"]
    tp_clocks = [tmp_path / "tp-1.csv", tmp_path / "tp-2.csv"]
    _write_clock(fp_clocks[0], 1_000_000.0)
    _write_clock(fp_clocks[1], 1_005_000.0)
    _write_clock(tp_clocks[0], 1_000_000.0)
    _write_clock(tp_clocks[1], 1_005_000.0, source_step_ms=1001.0)
    monkeypatch.setattr(
        alignment,
        "visual_anchor_calibration",
        lambda *_args, **_kwargs: (0.0, 0.0, []),
    )

    def segmented_view(view_id: str, role: ViewRole, clocks: list[Path]) -> ViewInput:
        return ViewInput(
            view_id=view_id,
            role=role,
            segments=[
                VideoSegmentInput(video=Path(f"{view_id}-{index}.mp4"), timestamps_csv=clock)
                for index, clock in enumerate(clocks)
            ],
        )

    def segmented_info(view_id: str, clocks: list[Path]) -> VideoInfo:
        segments = [
            VideoSegmentInfo(
                path=Path(f"{view_id}-{index}.mp4"),
                timestamps_csv=clock,
                virtual_start_ms=index * 5000.0,
                virtual_end_ms=(index + 1) * 5000.0,
                frame_start_index=index * 50,
                duration_ms=5000.0,
                fps=10.0,
                width=64,
                height=48,
                frame_count=50,
            )
            for index, clock in enumerate(clocks)
        ]
        return VideoInfo(
            path=segments[0].path,
            duration_ms=10_000.0,
            fps=10.0,
            width=64,
            height=48,
            frame_count=100,
            segments=segments,
        )

    views = [
        segmented_view("fp", ViewRole.FIRST_PERSON, fp_clocks),
        segmented_view("tp", ViewRole.THIRD_PERSON, tp_clocks),
    ]
    infos = {
        "fp": segmented_info("fp", fp_clocks),
        "tp": segmented_info("tp", tp_clocks),
    }

    transforms, _ = build_alignments(views, infos, load_config())

    segment_transforms = transforms["tp"].segment_transforms
    assert len(segment_transforms) == 2
    assert segment_transforms[0].scale == pytest.approx(1.0, abs=1e-6)
    assert segment_transforms[1].scale == pytest.approx(1.001, abs=1e-6)


def test_segment_transform_mapping_and_gap_availability():
    transform = AlignmentTransform(
        view_id="tp",
        reference_view_id="fp",
        state="aligned",
        segment_transforms=[
            AlignmentSegmentTransform(
                segment_index=0,
                local_start_ms=0.0,
                local_end_ms=1000.0,
                offset_ms=0.0,
                state="aligned",
            ),
            AlignmentSegmentTransform(
                segment_index=1,
                local_start_ms=2000.0,
                local_end_ms=3000.0,
                offset_ms=500.0,
                state="aligned",
            ),
        ],
    )

    assert transform.to_global(2500.0) == pytest.approx(3000.0)
    assert transform.to_local(3000.0) == pytest.approx(2500.0)
    assert transform.is_available_at_global(500.0) is True
    assert transform.is_available_at_global(1500.0) is False


def test_master_timeline_is_not_truncated_by_short_view():
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp", ViewRole.THIRD_PERSON),
    ]
    infos = {"fp": _info("fp.mp4", 8000.0), "tp": _info("tp.mp4", 2000.0)}
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp",
            reference_view_id="fp",
            state="aligned",
            local_coverage_start_ms=0.0,
            local_coverage_end_ms=8000.0,
        ),
        "tp": AlignmentTransform(
            view_id="tp",
            reference_view_id="fp",
            state="aligned",
            local_coverage_start_ms=0.0,
            local_coverage_end_ms=2000.0,
        ),
    }

    rows = list(iter_aligned_rows(views, infos, transforms, output_fps=1.0))

    assert rows[-1]["global_ms"] == pytest.approx(8000.0)
    assert rows[-1]["fp_available"] is True
    assert rows[-1]["tp_available"] is False


def test_alignment_gate_quarantines_one_failed_auxiliary_view():
    config = load_config()
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp-good", ViewRole.THIRD_PERSON),
        _view("tp-bad", ViewRole.THIRD_PERSON),
    ]
    infos = {view.view_id: _info(f"{view.view_id}.mp4", 5000.0) for view in views}
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp", reference_view_id="fp", state="aligned"
        ),
        "tp-good": AlignmentTransform(
            view_id="tp-good", reference_view_id="fp", state="aligned"
        ),
        "tp-bad": AlignmentTransform(
            view_id="tp-bad", reference_view_id="fp", state="failed"
        ),
    }

    receipt = alignment_quality_report(views, infos, transforms, config)

    assert receipt["formal_evidence_ready"] is True
    assert receipt["status"] == "passed_degraded"


def test_alignment_gate_does_not_double_count_overlapping_views():
    config = load_config()
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp-1", ViewRole.THIRD_PERSON),
        _view("tp-2", ViewRole.THIRD_PERSON),
    ]
    infos = {view.view_id: _info(f"{view.view_id}.mp4", 5000.0) for view in views}
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
        )
        for view in views
    }

    receipt = alignment_quality_report(views, infos, transforms, config)

    assert receipt["common_first_third_person_overlap_ms"] == pytest.approx(5000.0)


def test_alignment_gate_fails_without_aligned_third_person():
    config = load_config()
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp", ViewRole.THIRD_PERSON),
    ]
    infos = {view.view_id: _info(f"{view.view_id}.mp4", 5000.0) for view in views}
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp", reference_view_id="fp", state="aligned"
        ),
        "tp": AlignmentTransform(
            view_id="tp", reference_view_id="fp", state="failed"
        ),
    }

    receipt = alignment_quality_report(views, infos, transforms, config)

    assert receipt["formal_evidence_ready"] is False
    assert "no_aligned_third_person_coverage" in receipt["errors"]


def test_production_gate_rejects_unverified_local_timeline_assumption():
    config = load_config(Path("configs/rtx3090ti-ubuntu-production.yaml"))
    views = [
        _view("fp", ViewRole.FIRST_PERSON),
        _view("tp", ViewRole.THIRD_PERSON),
    ]
    infos = {view.view_id: _info(f"{view.view_id}.mp4", 5000.0) for view in views}
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp",
            reference_view_id="fp",
            state="aligned",
            alignment_basis="reference_local_timeline",
        ),
        "tp": AlignmentTransform(
            view_id="tp",
            reference_view_id="fp",
            state="aligned",
            alignment_basis="local_timeline_assumption",
        ),
    }

    receipt = alignment_quality_report(views, infos, transforms, config)

    assert receipt["formal_evidence_ready"] is False
    assert "no_aligned_third_person_coverage" in receipt["errors"]


def test_cross_view_association_uses_bounded_alignment_uncertainty():
    config = load_config()
    transforms = {
        "fp": AlignmentTransform(
            view_id="fp",
            reference_view_id="fp",
            state="aligned",
            uncertainty_ms=600.0,
        ),
        "tp": AlignmentTransform(
            view_id="tp",
            reference_view_id="fp",
            state="aligned",
            uncertainty_ms=600.0,
        ),
    }
    candidates = [
        ActionCandidate(
            candidate_id="fp-1",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id="fp",
            role=ViewRole.FIRST_PERSON,
            local_start_ms=0.0,
            local_end_ms=1000.0,
            global_start_ms=0.0,
            global_end_ms=1000.0,
            key_global_ms=500.0,
            objects=["tube"],
            confidence=0.9,
        ),
        ActionCandidate(
            candidate_id="tp-1",
            action_type=ActionType.HAND_OBJECT_CONTACT,
            view_id="tp",
            role=ViewRole.THIRD_PERSON,
            local_start_ms=2200.0,
            local_end_ms=3000.0,
            global_start_ms=2200.0,
            global_end_ms=3000.0,
            key_global_ms=2600.0,
            objects=["tube"],
            confidence=0.9,
        ),
    ]

    events, _ = audit_candidates(candidates, transforms, config)

    assert len(events) == 1
    receipt = events[0].observability["alignment_association"]
    assert receipt["effective_cluster_tolerance_ms"] > config["alignment"][
        "cross_view_event_tolerance_ms"
    ]
    assert receipt["effective_cluster_tolerance_ms"] <= config["alignment"][
        "cross_view_event_max_tolerance_ms"
    ]

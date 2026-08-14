from pathlib import Path

from labvision_evidence.actions import (
    refine_motion_candidates_with_coarse,
    select_fine_scan_views,
)
from labvision_evidence.pipeline import EvidencePipeline
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    RunManifest,
    ViewInput,
    ViewRole,
)


def _candidate(
    candidate_id: str,
    start_ms: float,
    end_ms: float,
    *,
    view_id: str = "fp",
    role: ViewRole = ViewRole.FIRST_PERSON,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=ActionType.OBJECT_MOVEMENT,
        view_id=view_id,
        role=role,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=["beaker"],
        confidence=0.8,
    )


def test_coarse_refinement_tightens_supported_windows_without_erasing_recall(default_config):
    default_config["performance"].update(
        {
            "coarse_refinement_association_margin_seconds": 0.0,
            "coarse_refinement_min_candidates": 2,
            "coarse_refinement_min_span_seconds": 30.0,
        }
    )
    motion = [
        _candidate("motion-1", 100_000, 300_000),
        _candidate("motion-2", 500_000, 600_000),
    ]
    coarse = [
        _candidate("coarse-1", 120_000, 150_000),
        _candidate("coarse-2", 210_000, 240_000),
        _candidate("coarse-unmatched", 800_000, 820_000),
    ]

    refined, report = refine_motion_candidates_with_coarse(
        motion, coarse, default_config
    )

    assert [(item.candidate_id, item.global_start_ms, item.global_end_ms) for item in refined] == [
        ("REFINED-motion-1", 120_000, 240_000),
        ("motion-2", 500_000, 600_000),
        ("coarse-unmatched", 800_000, 820_000),
    ]
    assert report["refined_motion_count"] == 1
    assert report["retained_motion_count"] == 1
    assert report["unmatched_coarse_candidate_count"] == 1


def test_sentinel_coarse_scan_keeps_all_views_for_fine_quality_fallback(default_config):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp1", role=ViewRole.THIRD_PERSON, video=Path("tp1.mp4")),
        ViewInput(view_id="tp2", role=ViewRole.THIRD_PERSON, video=Path("tp2.mp4")),
    ]
    manifest = RunManifest(experiment_id="test", views=views)
    default_config["performance"]["coarse_first_person_views"] = 1
    default_config["performance"]["coarse_third_person_views"] = 1
    default_config["performance"]["fine_min_third_person_views"] = 2

    pipeline = EvidencePipeline(default_config)
    coarse_views = pipeline._coarse_scan_views(manifest)
    selected, report = select_fine_scan_views(
        views,
        detection_paths={},
        coarse_candidates=[_candidate("motion", 0, 60_000)],
        config=default_config,
    )

    assert [view.view_id for view in coarse_views] == ["fp", "tp1"]
    assert {view.view_id for view in selected} == {"fp", "tp1", "tp2"}
    assert report["tp1"]["selected"] is True
    assert report["tp2"]["selected"] is True


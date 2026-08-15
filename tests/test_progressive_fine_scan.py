from pathlib import Path

from labvision_evidence.config import load_config
from labvision_evidence.pipeline import EvidencePipeline
from labvision_evidence.schemas import (
    ActionCandidate,
    ActionType,
    AlignmentTransform,
    EvidenceEvent,
    RunManifest,
    VideoInfo,
    ViewInput,
    ViewRole,
)


def _candidate(
    candidate_id: str,
    *,
    view_id: str = "fp",
    role: ViewRole = ViewRole.FIRST_PERSON,
    start_ms: float = 10_000.0,
    end_ms: float = 20_000.0,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        action_type=ActionType.HAND_OBJECT_CONTACT,
        view_id=view_id,
        role=role,
        local_start_ms=start_ms,
        local_end_ms=end_ms,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=["gloved_hand", "pipette"],
        confidence=0.9,
    )


def _event(
    event_id: str,
    roles: list[ViewRole],
    *,
    action_type: ActionType = ActionType.HAND_OBJECT_CONTACT,
    objects: list[str] | None = None,
    accepted: bool = True,
    start_ms: float = 12_000.0,
    end_ms: float = 14_000.0,
) -> EvidenceEvent:
    views = ["fp" if role == ViewRole.FIRST_PERSON else "tp" for role in roles]
    return EvidenceEvent(
        event_id=event_id,
        action_type=action_type,
        global_start_ms=start_ms,
        global_end_ms=end_ms,
        key_global_ms=(start_ms + end_ms) / 2.0,
        objects=objects or ["gloved_hand", "pipette"],
        confidence=0.9,
        accepted=accepted,
        audit_reason="test",
        supporting_views=views,
        supporting_roles=roles,
        candidates=[],
    )


def test_progressive_order_uses_coarse_anchor_quality_then_manifest_order(default_config):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp0", role=ViewRole.THIRD_PERSON, video=Path("tp0.mp4")),
        ViewInput(view_id="tp1", role=ViewRole.THIRD_PERSON, video=Path("tp1.mp4")),
        ViewInput(view_id="tp2", role=ViewRole.THIRD_PERSON, video=Path("tp2.mp4")),
    ]
    report = {
        "fp": {},
        "tp0": {"active_anchor_frames": 0, "anchor_frames": 1},
        "tp1": {"active_anchor_frames": 4, "anchor_frames": 5},
        "tp2": {"active_anchor_frames": 2, "anchor_frames": 8},
    }

    initial, supplemental = EvidencePipeline._progressive_fine_view_order(
        views, report, 1
    )

    assert [view.view_id for view in initial] == ["fp", "tp1"]
    assert [view.view_id for view in supplemental] == ["tp2", "tp0"]

    preferred, fallback = EvidencePipeline._progressive_fine_view_order(
        views, report, 1, ["tp0", "tp2"]
    )
    assert [view.view_id for view in preferred] == ["fp", "tp0"]
    assert [view.view_id for view in fallback] == ["tp2", "tp1"]


def test_rtx4060_profile_does_not_hardcode_validation_camera_ids():
    config = load_config(Path("configs/rtx4060-laptop-production.yaml"))

    assert config["performance"]["fine_progressive_cross_view"] is True
    assert config["performance"]["fine_dynamic_cross_view_scout"] is False
    assert config["performance"]["fine_initial_third_person_views"] == 2
    assert config["performance"]["fine_supplemental_view_batch_size"] == 2
    assert config["performance"]["fine_scout_fps"] == 1.0
    assert config["performance"]["fine_scout_anchor_radius_seconds"] == 3.0
    assert config["performance"]["fine_scout_peak_cluster_gap_seconds"] == 60.0
    assert config["performance"]["fine_scout_representatives_per_cluster"] == 1
    assert config["performance"]["fine_progressive_anchor_padding_seconds"] == 10.0
    assert config["performance"]["fine_preferred_third_person_views"] == []
    assert config["performance"]["motion_probe_sequential_segment_workers"] == 4
    assert config["performance"]["fine_first_person_decode_workers"] == 4
    assert config["performance"]["fine_third_person_decode_workers"] == 4
    assert config["performance"]["fine_decode_prefetch_frames"] == 32


def test_progressive_gap_ignores_incomplete_liquid_start_anchor(default_config):
    default_config["performance"]["fine_window_padding_seconds"] = 0.0
    pipeline = EvidencePipeline(default_config)
    boundary = [_candidate("WINDOW")]
    first_only = _event("FIRST", [ViewRole.FIRST_PERSON], accepted=False)
    incomplete_liquid = _event(
        "LIQUID",
        [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
        action_type=ActionType.LIQUID_MOVEMENT,
        objects=["sample_bottle", "spearhead", "tube"],
    )

    status = pipeline._progressive_target_status(
        boundary, [first_only, incomplete_liquid]
    )

    assert status[0]["status"] == "needs_third_person_supplement"
    assert status[0]["first_person_anchor_event_ids"] == ["FIRST"]
    assert status[0]["cross_view_anchor_event_ids"] == []

    covered = pipeline._progressive_target_status(
        boundary,
        [first_only, _event("BOTH", [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON])],
    )
    assert covered[0]["status"] == "cross_view_covered"
    assert covered[0]["cross_view_anchor_event_ids"] == ["BOTH"]


def test_progressive_status_does_not_share_one_anchor_across_padded_windows(
    default_config,
):
    default_config["performance"]["fine_window_padding_seconds"] = 75.0
    default_config["performance"]["fine_progressive_audit_margin_seconds"] = 5.0
    pipeline = EvidencePipeline(default_config)
    boundaries = [
        _candidate("WINDOW-1", start_ms=10_000.0, end_ms=20_000.0),
        _candidate("WINDOW-2", start_ms=80_000.0, end_ms=90_000.0),
    ]
    events = [
        _event("BOTH-1", [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON]),
        _event(
            "FIRST-2",
            [ViewRole.FIRST_PERSON],
            accepted=False,
            start_ms=82_000.0,
            end_ms=84_000.0,
        ),
    ]

    status = pipeline._progressive_target_status(boundaries, events)

    assert status[0]["status"] == "cross_view_covered"
    assert status[1]["status"] == "needs_third_person_supplement"
    assert status[1]["cross_view_anchor_event_ids"] == []
    assert status[1]["first_person_anchor_event_ids"] == ["FIRST-2"]


def test_progressive_scan_stops_after_first_successful_supplement(
    monkeypatch, tmp_path, default_config
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(
            view_id="tp_best", role=ViewRole.THIRD_PERSON, video=Path("tp_best.mp4")
        ),
        ViewInput(
            view_id="tp_unused", role=ViewRole.THIRD_PERSON, video=Path("tp_unused.mp4")
        ),
        ViewInput(
            view_id="tp_fallback",
            role=ViewRole.THIRD_PERSON,
            video=Path("tp_fallback.mp4"),
        ),
    ]
    manifest = RunManifest(experiment_id="progressive", views=views)
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=100_000.0,
            fps=30.0,
            width=16,
            height=16,
            frame_count=3_000,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view in views
    }
    boundary = [_candidate("WINDOW")]
    default_config["performance"].update(
        {
            "fine_window_padding_seconds": 1.0,
            "fine_initial_third_person_views": 1,
            "fine_decode_lanes": ["cuda", "cuda", "cuda", "cpu"],
            "detection_fps": 10.0,
        }
    )
    fine_report = {
        "fp": {},
        "tp_best": {"active_anchor_frames": 5, "anchor_frames": 6},
        "tp_fallback": {"active_anchor_frames": 2, "anchor_frames": 3},
        "tp_unused": {"active_anchor_frames": 0, "anchor_frames": 0},
    }
    pipeline = EvidencePipeline(default_config)
    fine_windows = pipeline._fine_windows(boundary, infos, transforms)
    scan_calls = []
    refinement_calls = []

    def fake_scan(
        pass_manifest,
        _infos,
        _transforms,
        pass_dir,
        **kwargs,
    ):
        scan_calls.append(
            {
                "views": [view.view_id for view in pass_manifest.views],
                "windows": kwargs["windows"],
                "decode_backends": kwargs["decode_backends"],
            }
        )
        return {
            view.view_id: pass_dir / f"{view.view_id}.jsonl"
            for view in pass_manifest.views
        }

    def fake_generate(scanned_views, _paths, _config):
        return [
            _candidate(
                f"CAND-{view.view_id}", view_id=view.view_id, role=view.role
            )
            for view in scanned_views
        ]

    def fake_audit(candidates, _transforms, _config):
        ids = {candidate.view_id for candidate in candidates}
        if "tp_fallback" in ids:
            return [
                _event("BOTH", [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON])
            ], []
        # Before hand-context refinement this looks cross-view complete. The
        # third-person contribution is deliberately removed below so the
        # scheduler must not stop before trying the aligned fallback view.
        return [
            _event(
                "LIQUID",
                [ViewRole.FIRST_PERSON, ViewRole.THIRD_PERSON],
                action_type=ActionType.LIQUID_MOVEMENT,
                objects=["pipette", "tube"],
            )
        ], []

    def fake_refine(events, paths):
        refinement_calls.append(set(paths))
        for event in events:
            if event.action_type == ActionType.LIQUID_MOVEMENT:
                event.supporting_views = ["fp"]
                event.supporting_roles = [ViewRole.FIRST_PERSON]
        return []

    monkeypatch.setattr(pipeline, "_scan_all_views_concurrently", fake_scan)
    monkeypatch.setattr("labvision_evidence.pipeline.generate_candidates", fake_generate)
    monkeypatch.setattr("labvision_evidence.pipeline.audit_candidates", fake_audit)
    monkeypatch.setattr(
        "labvision_evidence.pipeline.refine_liquid_events_with_context", fake_refine
    )

    paths, scanned, _, report = pipeline._run_progressive_fine_scan(
        manifest,
        views,
        fine_report,
        boundary,
        fine_windows,
        infos,
        transforms,
        tmp_path,
    )

    assert [call["views"] for call in scan_calls] == [
        ["fp", "tp_best"],
        ["tp_fallback"],
    ]
    assert refinement_calls == [
        {"fp", "tp_best"},
        {"fp", "tp_best", "tp_fallback"},
    ]
    # The supplemental subset keeps its original six-view lane identity; it is
    # not incorrectly reindexed onto lane zero.
    assert scan_calls[1]["decode_backends"] == {"tp_fallback": "cpu"}
    assert [view.view_id for view in scanned] == ["fp", "tp_best", "tp_fallback"]
    assert set(paths) == {"fp", "tp_best", "tp_fallback"}
    assert report["not_scanned_view_ids"] == ["tp_unused"]
    assert report["stopping_reason"] == "all_demanded_windows_have_dual_role_anchor"
    assert report["unresolved_candidate_ids"] == []
    assert report["avoided_selected_seconds"] > 0
    assert report["passes"][1]["target_candidate_ids"] == ["WINDOW"]


def test_progressive_scan_exhausts_all_views_when_gap_remains(
    monkeypatch, tmp_path, default_config
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp0", role=ViewRole.THIRD_PERSON, video=Path("tp0.mp4")),
        ViewInput(view_id="tp1", role=ViewRole.THIRD_PERSON, video=Path("tp1.mp4")),
        ViewInput(view_id="tp2", role=ViewRole.THIRD_PERSON, video=Path("tp2.mp4")),
    ]
    manifest = RunManifest(experiment_id="progressive-exhaust", views=views)
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=100_000.0,
            fps=30.0,
            width=16,
            height=16,
            frame_count=3_000,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view in views
    }
    boundary = [_candidate("WINDOW")]
    default_config["performance"]["fine_window_padding_seconds"] = 1.0
    default_config["performance"]["fine_supplemental_view_batch_size"] = 2
    pipeline = EvidencePipeline(default_config)
    calls = []

    def fake_scan(pass_manifest, _infos, _transforms, pass_dir, **_kwargs):
        calls.append([view.view_id for view in pass_manifest.views])
        return {
            view.view_id: pass_dir / f"{view.view_id}.jsonl"
            for view in pass_manifest.views
        }

    monkeypatch.setattr(pipeline, "_scan_all_views_concurrently", fake_scan)
    monkeypatch.setattr(
        "labvision_evidence.pipeline.generate_candidates", lambda *_args: []
    )
    monkeypatch.setattr(
        "labvision_evidence.pipeline.audit_candidates",
        lambda *_args: ([_event("FIRST", [ViewRole.FIRST_PERSON])], []),
    )

    _, scanned, _, report = pipeline._run_progressive_fine_scan(
        manifest,
        views,
        {view.view_id: {} for view in views},
        boundary,
        pipeline._fine_windows(boundary, infos, transforms),
        infos,
        transforms,
        tmp_path,
    )

    assert calls == [["fp", "tp0"], ["tp1", "tp2"]]
    assert [view.view_id for view in scanned] == ["fp", "tp0", "tp1", "tp2"]
    assert report["supplemental_view_batch_size"] == 2
    assert report["supplemental_waves"] == [["tp1", "tp2"]]
    assert report["stopping_reason"] == "all_eligible_third_person_views_exhausted"
    assert report["unresolved_candidate_ids"] == ["WINDOW"]


def test_dynamic_scout_ranks_all_third_person_views_then_scans_narrow_anchor_window(
    monkeypatch, tmp_path, default_config
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp0", role=ViewRole.THIRD_PERSON, video=Path("tp0.mp4")),
        ViewInput(view_id="tp1", role=ViewRole.THIRD_PERSON, video=Path("tp1.mp4")),
        ViewInput(view_id="tp2", role=ViewRole.THIRD_PERSON, video=Path("tp2.mp4")),
    ]
    manifest = RunManifest(experiment_id="dynamic-scout", views=views)
    infos = {
        view.view_id: VideoInfo(
            path=view.video,
            duration_ms=100_000.0,
            fps=30.0,
            width=16,
            height=16,
            frame_count=3_000,
        )
        for view in views
    }
    transforms = {
        view.view_id: AlignmentTransform(
            view_id=view.view_id,
            reference_view_id="fp",
            state="aligned",
            confidence=1.0,
        )
        for view in views
    }
    boundary = [_candidate("WINDOW")]
    default_config["performance"].update(
        {
            "fine_window_padding_seconds": 75.0,
            "fine_dynamic_cross_view_scout": True,
            "fine_scout_fps": 1.0,
            "fine_scout_image_size": 416,
            "fine_scout_anchor_radius_seconds": 1.0,
            "fine_progressive_anchor_padding_seconds": 1.0,
            "fine_decode_lanes": ["cuda", "cuda", "cuda", "cpu"],
            "detection_fps": 10.0,
        }
    )
    pipeline = EvidencePipeline(default_config)
    scan_calls = []

    def fake_scan(pass_manifest, _infos, _transforms, pass_dir, **kwargs):
        scan_calls.append(
            {
                "phase": kwargs["phase"],
                "views": [view.view_id for view in pass_manifest.views],
                "windows": kwargs["windows"],
                "decode_backends": kwargs["decode_backends"],
                "sample_fps": kwargs["sample_fps"],
            }
        )
        return {
            view.view_id: pass_dir / f"{view.view_id}.jsonl"
            for view in pass_manifest.views
        }

    def fake_generate(scanned_views, _paths, _config):
        return [
            _candidate(f"CAND-{view.view_id}", view_id=view.view_id, role=view.role)
            for view in scanned_views
        ]

    def fake_audit(candidates, _transforms, _config):
        view_ids = {candidate.view_id for candidate in candidates}
        roles = [ViewRole.FIRST_PERSON]
        if "tp2" in view_ids:
            roles.append(ViewRole.THIRD_PERSON)
        return [_event("ANCHOR", roles)], []

    def fake_select(_views, _paths, _candidates, _config):
        report = {
            "tp0": {"active_anchor_frames": 0, "anchor_frames": 1},
            "tp1": {"active_anchor_frames": 2, "anchor_frames": 3},
            "tp2": {"active_anchor_frames": 7, "anchor_frames": 8},
        }
        return list(_views), report

    monkeypatch.setattr(pipeline, "_scan_all_views_concurrently", fake_scan)
    monkeypatch.setattr("labvision_evidence.pipeline.generate_candidates", fake_generate)
    monkeypatch.setattr("labvision_evidence.pipeline.audit_candidates", fake_audit)
    monkeypatch.setattr("labvision_evidence.pipeline.select_fine_scan_views", fake_select)

    paths, scanned, _, report = pipeline._run_progressive_fine_scan(
        manifest,
        views,
        {view.view_id: {} for view in views},
        boundary,
        pipeline._fine_windows(boundary, infos, transforms),
        infos,
        transforms,
        tmp_path,
    )

    assert [(call["phase"], call["views"]) for call in scan_calls] == [
        ("fine", ["fp"]),
        ("fine_scout", ["tp0", "tp1", "tp2"]),
        ("fine", ["tp2"]),
    ]
    assert scan_calls[1]["sample_fps"] == 1.0
    assert scan_calls[1]["windows"]["tp0"] == [(12_000.0, 14_000.0)]
    assert scan_calls[2]["decode_backends"] == {"tp2": "cpu"}
    # The formal third-person pass is aligned to the 12-14 second FP event
    # plus one second, rather than inheriting the 75-second coarse padding.
    assert scan_calls[2]["windows"]["tp2"] == [(11_000.0, 15_000.0)]
    assert [view.view_id for view in scanned] == ["fp", "tp2"]
    assert set(paths) == {"fp", "tp2"}
    assert report["dynamic_cross_view_scout"]["view_ids"] == ["tp0", "tp1", "tp2"]
    assert report["dynamic_cross_view_scout"]["anchor_radius_seconds"] == 1.0
    assert (
        report["dynamic_cross_view_scout"]["window_strategy"]
        == "aligned_first_person_peak_windows"
    )
    assert report["dynamic_cross_view_scout"]["pre_merge_window_count_per_view"] == 1
    assert report["dynamic_cross_view_scout"]["post_merge_window_count_by_view"] == {
        "tp0": 1,
        "tp1": 1,
        "tp2": 1,
    }
    assert report["dynamic_cross_view_scout"]["ranking"][0]["view_id"] == "tp2"
    assert report["supplemental_priority"] == ["tp2", "tp1", "tp0"]
    assert report["not_scanned_view_ids"] == ["tp0", "tp1"]
    assert report["stopping_reason"] == "all_demanded_windows_have_dual_role_anchor"
    assert report["scout_estimated_frames"] > 0


def test_scout_anchor_selection_deduplicates_and_clusters_across_candidates():
    statuses = [
        {
            "candidate_id": "C1",
            "first_person_anchor_windows": [
                {
                    "event_id": "E1",
                    "key_global_ms": 10_000.0,
                    "global_start_ms": 9_000.0,
                    "global_end_ms": 11_000.0,
                    "confidence": 0.8,
                },
                {
                    "event_id": "E2",
                    "key_global_ms": 20_000.0,
                    "global_start_ms": 19_000.0,
                    "global_end_ms": 21_000.0,
                    "confidence": 0.9,
                },
            ],
        },
        {
            "candidate_id": "C2",
            "first_person_anchor_windows": [
                {
                    "event_id": "E1",
                    "key_global_ms": 10_000.0,
                    "global_start_ms": 9_000.0,
                    "global_end_ms": 11_000.0,
                    "confidence": 0.8,
                },
                {
                    "event_id": "E3",
                    "key_global_ms": 20_100.0,
                    "global_start_ms": 19_100.0,
                    "global_end_ms": 21_100.0,
                    "confidence": 0.7,
                },
            ],
        },
        {
            "candidate_id": "C3",
            "first_person_anchor_windows": [
                {
                    "event_id": "E4",
                    "key_global_ms": 100_000.0,
                    "global_start_ms": 99_000.0,
                    "global_end_ms": 101_000.0,
                    "confidence": 0.95,
                }
            ],
        },
    ]

    selected, diagnostics = (
        EvidencePipeline._progressive_scout_anchor_representatives(
            statuses,
            {"C1", "C2", "C3"},
            dedup_tolerance_seconds=0.25,
            cluster_gap_seconds=60.0,
            representatives_per_cluster=1,
        )
    )

    assert diagnostics["raw_anchor_occurrence_count"] == 5
    assert diagnostics["unique_event_count"] == 4
    assert diagnostics["unique_peak_count"] == 3
    assert diagnostics["cluster_count"] == 2
    assert diagnostics["representative_count"] == 2
    assert [item["event_id"] for item in selected] == ["E2", "E4"]
    assert diagnostics["clusters"][0]["candidate_ids"] == ["C1", "C2"]

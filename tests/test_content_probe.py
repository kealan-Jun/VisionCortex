from pathlib import Path

from labvision_evidence.schemas import RunManifest, VideoInfo, ViewInput, ViewRole
from labvision_evidence.content_probe import (
    ProbeFrame,
    _select_review_frames,
    distributed_probe_intervals,
    gate_probe_verdict,
    motion_followup_interval,
    run_full_timeline_content_sweep,
)
from labvision_evidence.collection_curation import (
    evaluate_full_timeline_semantic_consensus,
)


def test_full_sweep_can_prove_no_action_chain_without_relabeling_object_motion():
    sweep = {
        "status": "inconclusive",
        "all_ark_completed": True,
        "all_chunks_negative": False,
        "all_source_views_sampled": True,
        "chunks": [
            {
                "chunk_index": 1,
                "ark_status": "completed",
                "gated_verdict": "inconclusive",
                "model_result": {
                    "status": "completed",
                    "verdict": "inconclusive",
                    "visible_experimental_action": True,
                    "purposeful_experimental_action_chain": False,
                    "action_chain_steps": [],
                },
            },
            {
                "chunk_index": 2,
                "ark_status": "completed",
                "gated_verdict": "recording_or_hardware_test",
                "model_result": {
                    "status": "completed",
                    "verdict": "recording_or_hardware_test",
                    "purposeful_experimental_action_chain": False,
                    "action_chain_steps": [],
                },
            },
        ],
    }

    result = evaluate_full_timeline_semantic_consensus(sweep)

    assert result["eligible"] is True
    assert result["mode"] == "purposeful_action_chain_absent"
    assert result["inconclusive_chunk_indices"] == [1]


def test_full_sweep_action_chain_presence_fails_closed():
    sweep = {
        "status": "inconclusive",
        "all_ark_completed": True,
        "all_chunks_negative": False,
        "all_source_views_sampled": True,
        "chunks": [
            {
                "chunk_index": 1,
                "ark_status": "completed",
                "gated_verdict": "inconclusive",
                "model_result": {
                    "status": "completed",
                    "verdict": "real_experiment",
                    "purposeful_experimental_action_chain": True,
                    "action_chain_steps": [{"observable_step": "将样品加入容器"}],
                },
            }
        ],
    }

    result = evaluate_full_timeline_semantic_consensus(sweep)

    assert result["eligible"] is False
    assert result["reason"] == "purposeful_action_chain_not_unanimously_absent"


def test_long_timeline_negative_probe_is_forced_inconclusive(monkeypatch, tmp_path):
    """A sparse long-video sample may prove presence, never prove absence."""

    # The end-to-end receipt rule is exercised more cheaply through the
    # explicit gate invariant in collection curation; keep the model-only gate
    # test below focused on direct evidence requirements.
    from labvision_evidence.collection_curation import _read_probe

    config = {
        "storage": {"local_cache_root": str(tmp_path)},
        "collection_curation": {
            "content_probe_max_seconds": 300,
            "content_probe_max_views": 2,
            "content_probe_receipt_dir": str(tmp_path / "receipts"),
        },
    }
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    (receipt_dir / "long-negative.json").write_text(
        __import__("json").dumps(
            {
                "schema_version": "visioncortex-content-probe/2",
                "experiment_id": "long-negative",
                "scope": "classification_only",
                "production_completion_eligible": False,
                "source_copy_bytes": 0,
                "decoded_timeline_seconds": 250,
                "timeline_duration_seconds": 3600,
                "distribution_strategy": (
                    "full_timeline_stratified_anchors_plus_motion_peaks"
                ),
                "sampled_intervals": [
                    {
                        "start_seconds": max(0, 3600 * fraction - 20),
                        "end_seconds": min(3600, 3600 * fraction + 20),
                        "selection_reason": "uniform_anchor",
                    }
                    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
                ]
                + [
                    {
                        "start_seconds": 1775,
                        "end_seconds": 1825,
                        "selection_reason": "motion_peak",
                    }
                ],
                "views": [
                    {"view_id": "fp", "role": "first_person"},
                    {"view_id": "tp", "role": "third_person"},
                ],
                "verdict": "recording_or_hardware_test",
            }
        ),
        encoding="utf-8",
    )

    result = _read_probe(config, "long-negative", 300, 2, 3600)

    assert result["valid"] is False
    assert "negative_long_timeline_probe_cannot_exclude_experiment" in result["errors"]


def test_review_storyboard_balances_cross_view_images():
    frames = [
        ProbeFrame(
            view_id=view_id,
            role=role,
            interval_index=1,
            selection_reason="uniform_anchor",
            time_seconds=float(index),
            motion_score=float(100 - index if view_id == "fp" else 10 - index),
            jpeg=b"jpeg",
        )
        for view_id, role in (("fp", "first_person"), ("tp", "third_person"))
        for index in range(20)
    ]

    selected = _select_review_frames(frames, maximum_images=24)

    assert len(selected) == 24
    assert sum(item.view_id == "fp" for item in selected) == 12
    assert sum(item.view_id == "tp" for item in selected) == 12


def test_short_probe_covers_entire_timeline():
    intervals = distributed_probe_intervals(55.258, 300.0)

    assert intervals == [
        {
            "start_seconds": 0.0,
            "end_seconds": 55.258,
            "selection_reason": "uniform_anchor",
            "anchor_fraction": None,
        }
    ]


def test_long_probe_spans_anchors_and_adds_nonoverlapping_motion_followup():
    anchors = distributed_probe_intervals(1105.586, 300.0)
    used = sum(item["end_seconds"] - item["start_seconds"] for item in anchors)
    motion = motion_followup_interval(1105.586, anchors, 560.0, 300.0 - used)
    all_intervals = sorted([*anchors, motion], key=lambda item: item["start_seconds"])

    assert len(anchors) == 5
    assert motion["selection_reason"] == "motion_peak"
    assert sum(
        item["end_seconds"] - item["start_seconds"] for item in all_intervals
    ) <= 300.0
    assert all(
        current["start_seconds"] >= previous["end_seconds"]
        for previous, current in zip(all_intervals, all_intervals[1:])
    )


def test_real_probe_requires_direct_labeled_action_evidence():
    verdict, reasons = gate_probe_verdict(
        {
            "verdict": "real_experiment",
            "confidence": 0.9,
            "visible_experimental_action": True,
            "purposeful_experimental_action_chain": True,
            "action_chain_steps": [
                {
                    "image_label": "image=01",
                    "observable_step": "打开试剂瓶",
                    "experimental_change": "瓶盖由闭合变为分离",
                },
                {
                    "image_label": "image=02",
                    "observable_step": "移液器从瓶内吸液后进入试管",
                    "experimental_change": "液体由源瓶转移到试管",
                },
            ],
            "action_evidence": [
                {"image_label": "image=01", "observable_action": "手持移液器进入管口"}
            ],
        }
    )
    assert verdict == "real_experiment"
    assert reasons == []

    verdict, reasons = gate_probe_verdict(
        {
            "verdict": "real_experiment",
            "confidence": 0.95,
            "visible_experimental_action": True,
            "purposeful_experimental_action_chain": True,
            "action_chain_steps": [
                {
                    "image_label": "image=01",
                    "observable_step": "打开试剂瓶",
                    "experimental_change": "瓶盖分离",
                },
                {
                    "image_label": "image=02",
                    "observable_step": "吸液",
                    "experimental_change": "液体进入枪头",
                },
            ],
            "action_evidence": [],
        }
    )
    assert verdict == "inconclusive"
    assert reasons == ["real_experiment_purposeful_action_chain_gate_not_met"]


def test_real_probe_rejects_object_repositioning_without_experimental_change():
    verdict, reasons = gate_probe_verdict(
        {
            "verdict": "real_experiment",
            "confidence": 0.92,
            "visible_experimental_action": True,
            "purposeful_experimental_action_chain": True,
            "action_chain_steps": [
                {
                    "image_label": "image=01",
                    "observable_step": "拿起未开合试剂瓶",
                    "experimental_change": "",
                },
                {
                    "image_label": "image=02",
                    "observable_step": "把试剂瓶放到另一位置",
                    "experimental_change": "",
                },
            ],
            "action_evidence": [
                {"image_label": "image=01", "observable_action": "移动试剂瓶"}
            ],
        }
    )

    assert verdict == "inconclusive"
    assert reasons == ["real_experiment_purposeful_action_chain_gate_not_met"]


def test_test_verdict_requires_all_interval_denial():
    verdict, reasons = gate_probe_verdict(
        {
            "verdict": "recording_or_hardware_test",
            "confidence": 0.91,
            "visible_experimental_action": False,
            "no_operation_across_all_intervals": True,
        }
    )
    assert verdict == "recording_or_hardware_test"
    assert reasons == []

    verdict, reasons = gate_probe_verdict(
        {
            "verdict": "recording_or_hardware_test",
            "confidence": 0.91,
            "visible_experimental_action": False,
            "no_operation_across_all_intervals": False,
        }
    )
    assert verdict == "inconclusive"
    assert reasons == ["recording_test_all_intervals_gate_not_met"]


def test_full_timeline_sweep_covers_every_media_band_and_source_view(
    monkeypatch, tmp_path
):
    views = [
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=Path("fp.mp4")),
        ViewInput(view_id="tp-a", role=ViewRole.THIRD_PERSON, video=Path("a.mp4")),
        ViewInput(view_id="tp-b", role=ViewRole.THIRD_PERSON, video=Path("b.mp4")),
    ]
    manifest = RunManifest(experiment_id="long", views=views)
    durations = {"fp": 601.0, "tp-a": 600.0, "tp-b": 500.0}
    monkeypatch.setattr(
        "labvision_evidence.content_probe.probe_views",
        lambda _views: {
            view.view_id: VideoInfo(
                path=view.video,
                duration_ms=durations[view.view_id] * 1000,
                fps=30,
                width=640,
                height=360,
                frame_count=int(durations[view.view_id] * 30),
                size_bytes=1,
            )
            for view in views
        },
    )

    def frames(selected, _infos, intervals, sampling_fps):
        assert sampling_fps == 1.0
        timestamp = float(intervals[0]["start_seconds"]) + 0.1
        return [
            ProbeFrame(
                view_id=view.view_id,
                role=view.role.value,
                interval_index=1,
                selection_reason="uniform_anchor",
                time_seconds=timestamp,
                motion_score=1.0,
                jpeg=b"jpeg",
            )
            for view in selected
        ]

    monkeypatch.setattr("labvision_evidence.content_probe._decode_probe_frames", frames)

    class Analyzer:
        def __init__(self, _config):
            pass

        def _call(self, *_args, **_kwargs):
            return {
                "status": "completed",
                "model": "ark-test",
                "verdict": "recording_or_hardware_test",
                "confidence": 0.9,
                "visible_experimental_action": False,
                "no_operation_across_all_intervals": True,
                "usage": {"total_tokens": 10},
            }

        def close(self):
            pass

    monkeypatch.setattr("labvision_evidence.content_probe.ArkStepAnalyzer", Analyzer)
    result = run_full_timeline_content_sweep(
        {
            "storage": {"local_cache_root": str(tmp_path)},
            "collection_curation": {"content_probe_max_seconds": 300},
        },
        manifest,
        display_name="test_long",
        expected_timeline_seconds=660.0,
    )

    assert result["media_timeline_duration_seconds"] == 601.0
    assert result["indexed_capture_gap_seconds"] == 59.0
    assert [(item["start_seconds"], item["end_seconds"]) for item in result["chunks"]] == [
        (0.0, 300.0),
        (300.0, 600.0),
        (600.0, 601.0),
    ]
    assert result["all_source_views_sampled"] is True
    assert result["all_chunks_negative"] is True
    assert result["status"] == "no_visible_experimental_action_across_full_timeline"

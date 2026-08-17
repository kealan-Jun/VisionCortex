import json

import pytest

from labvision_evidence.liquid_state import (
    DisabledLiquidStateExpert,
    LiquidStateBackendUnavailable,
    build_liquid_benchmark_manifest,
    build_liquid_state_expert,
    evaluate_liquid_state_predictions,
    register_liquid_state_backend,
    write_liquid_benchmark_manifest,
)
from labvision_evidence.schemas import (
    LiquidContainerState,
    LiquidExpertSample,
    LiquidFlowDirection,
    LiquidStateEvidence,
    LiquidStateStatus,
    LiquidViewObservation,
    NormalizedPoint,
    ViewRole,
)


def _evidence(*, fill_offset: float = 0.0) -> LiquidStateEvidence:
    return LiquidStateEvidence(
        status=LiquidStateStatus.OBSERVED,
        backend="test-backend",
        source_before=LiquidContainerState(
            container_id="bottle-01", liquid_present=True, fill_ratio=0.8
        ),
        source_after=LiquidContainerState(
            container_id="bottle-01",
            liquid_present=True,
            fill_ratio=0.6 + fill_offset,
        ),
        target_before=LiquidContainerState(
            container_id="tube-01", liquid_present=False, fill_ratio=0.0
        ),
        target_after=LiquidContainerState(
            container_id="tube-01",
            liquid_present=True,
            fill_ratio=0.2 + fill_offset,
        ),
        per_view_observations=[
            LiquidViewObservation(
                view_id="fp",
                view_role=ViewRole.FIRST_PERSON,
                timestamp_us=16_700_000,
                container_id="tube-01",
                liquid_present=True,
                fill_ratio=0.2 + fill_offset,
                visible_flow=True,
                flow_direction=LiquidFlowDirection.SOURCE_TO_TARGET,
                visibility=0.95,
                confidence=0.9,
                meniscus_polyline=[
                    NormalizedPoint(x=0.2, y=0.6),
                    NormalizedPoint(x=0.8, y=0.6 + fill_offset),
                ],
                observed_facts=["Liquid surface is visible inside tube-01."],
            )
        ],
        visible_flow=True,
        flow_direction=LiquidFlowDirection.SOURCE_TO_TARGET,
        state_change_confirmed=True,
        confidence=0.9,
        observed_facts=["Source level decreased and target level increased."],
    )


def test_liquid_schema_rejects_invalid_normalized_roi():
    with pytest.raises(ValueError, match="normalized non-empty box"):
        LiquidExpertSample(
            sample_id="sample-1",
            event_id="event-1",
            group_id="group-1",
            view_id="fp",
            view_role=ViewRole.FIRST_PERSON,
            timestamp_us=10,
            media_ref="Key-Materials/frame.jpg",
            roi_xyxy_norm=(0.8, 0.2, 0.1, 0.9),
        )


def test_disabled_expert_is_zero_work_and_zero_token():
    expert = build_liquid_state_expert({"enabled": False, "backend": "disabled"})
    assert isinstance(expert, DisabledLiquidStateExpert)
    results = expert.analyze(
        [
            LiquidExpertSample(
                sample_id="sample-1",
                event_id="event-1",
                group_id="group-1",
                view_id="fp",
                view_role=ViewRole.FIRST_PERSON,
                timestamp_us=10,
                media_ref="frame.jpg",
            )
        ]
    )
    assert results["event-1"].status == LiquidStateStatus.NOT_EVALUATED
    assert results["event-1"].model_receipt["total_tokens"] == 0
    assert expert.receipt()["invocations"] == 0


def test_enabled_unregistered_backend_fails_closed():
    with pytest.raises(LiquidStateBackendUnavailable, match="not registered"):
        build_liquid_state_expert({"enabled": True, "backend": "sam2"})


def test_registered_adapter_can_be_injected_without_core_dependency():
    class DummyExpert:
        name = "unit-test-dummy"

        def analyze(self, samples):
            return {sample.event_id: _evidence() for sample in samples}

        def receipt(self):
            return {"backend": self.name, "invocations": 1, "total_tokens": 0}

    register_liquid_state_backend(
        "unit-test-dummy", lambda settings: DummyExpert(), replace=True
    )
    expert = build_liquid_state_expert(
        {"enabled": True, "backend": "unit-test-dummy"}
    )
    assert expert.name == "unit-test-dummy"


def test_manifest_is_reference_only_and_splits_by_experiment_group(tmp_path):
    events = [
        {
            "event_id": "event-001",
            "parent_event_id": "group-01",
            "action_type": "liquid_transfer",
            "action_subtype": "pipette_transfer",
            "peak_timestamp_us": 16_700_000,
            "decision": {"status": "confirmed"},
            "key_frames": [{"path": "Key-Materials/Key-Frames/event-001/fp.jpg"}],
            "key_clips": [{"path": "Key-Materials/Key-Clips/event-001/fp.mp4"}],
        },
        {
            "event_id": "event-002",
            "parent_event_id": "group-01",
            "action_type": "container_state_change",
            "peak_timestamp_us": 17_000_000,
            "decision": {"status": "confirmed"},
            "key_frames": [{"path": "Key-Materials/Key-Frames/event-002/fp.jpg"}],
            "key_clips": [],
        },
    ]
    manifest = build_liquid_benchmark_manifest(
        events,
        benchmark_id="golden-v1",
        source_archive="Archive-01",
        annotations={"event-001": _evidence()},
    )
    assert manifest.media_policy == "references_only"
    assert len(manifest.samples) == 2
    assert manifest.samples[0].split == manifest.samples[1].split
    assert manifest.samples[0].ground_truth is not None
    destination = tmp_path / "golden-manifest.json"
    write_liquid_benchmark_manifest(destination, manifest)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["samples"][0]["image_refs"][0].endswith("fp.jpg")
    assert not (tmp_path / "Key-Materials").exists()


def test_evaluator_reports_state_flow_fill_and_meniscus_metrics():
    truth = {"event-001": _evidence()}
    prediction = {"event-001": _evidence(fill_offset=0.05)}
    metrics = evaluate_liquid_state_predictions(truth, prediction)
    assert metrics["events"]["covered"] == 1
    assert metrics["status_accuracy"] == 1.0
    assert metrics["liquid_presence"]["recall"] == 1.0
    assert metrics["visible_flow"]["recall"] == 1.0
    assert metrics["flow_direction"]["accuracy"] == 1.0
    assert metrics["state_change"]["accuracy"] == 1.0
    assert metrics["fill_ratio"]["paired_observations"] == 5
    assert metrics["fill_ratio"]["mae"] == pytest.approx(0.03)
    assert metrics["meniscus"]["mean_normalized_distance"] > 0
    assert metrics["token_usage"]["total_tokens"] == 0

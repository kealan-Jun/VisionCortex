from visioncortex import archive
from visioncortex.schemas import ActionType, EvidenceEvent


def _event(action: ActionType) -> EvidenceEvent:
    return EvidenceEvent(
        event_id="EVT-000001",
        action_type=action,
        global_start_ms=1000,
        global_end_ms=2000,
        key_global_ms=1500,
        objects=["tube"],
        confidence=0.9,
        accepted=True,
        audit_reason="test",
        supporting_views=[],
        supporting_roles=[],
        candidates=[],
        key_frames={"fp": "fp.jpg", "tp": "tp.jpg"},
        key_clips={
            "fp": "fp.mp4",
            "tp": "tp.mp4",
            "aligned_first_third": "aligned.mp4",
        },
    )


def test_common_action_uses_three_aligned_temporal_frames(monkeypatch, tmp_path):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.work = tmp_path / "work"
    calls = []

    def fake_extract(clip, output, view_id, samples_per_view=3):
        calls.append((clip, view_id, samples_per_view))
        return [
            (f"view_id={view_id}; temporal_phase={phase}", output / f"{phase}.jpg")
            for phase in ("before", "peak", "after")
        ]

    monkeypatch.setattr(archive, "extract_temporal_review_frames", fake_extract)
    images = archive.key_material_review_images(
        layout,
        _event(ActionType.HAND_OBJECT_CONTACT),
        {
            "mllm": {
                "use_key_clip_temporal_samples": True,
                "temporal_samples_per_view": 3,
                "compact_aligned_temporal_actions": ["hand_object_contact"],
            }
        },
    )
    assert len(images) == 3
    assert calls == [(layout.root / "aligned.mp4", "aligned_first_third", 3)]


def test_liquid_action_uses_action_specific_temporal_frame_count(monkeypatch, tmp_path):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.work = tmp_path / "work"
    calls = []

    def fake_extract(clip, output, view_id, samples_per_view=3):
        calls.append((view_id, samples_per_view))
        return [
            (
                f"view_id={view_id}; temporal_phase=timeline_{index:02d}",
                output / f"{view_id}-{index:02d}.jpg",
            )
            for index in range(1, samples_per_view + 1)
        ]

    monkeypatch.setattr(archive, "extract_temporal_review_frames", fake_extract)
    images = archive.key_material_review_images(
        layout,
        _event(ActionType.LIQUID_MOVEMENT),
        {
            "mllm": {
                "use_key_clip_temporal_samples": True,
                "temporal_samples_per_view": 3,
                "temporal_samples_per_view_by_action": {
                    "liquid_movement": 15,
                },
                "liquid_primary_samples": 15,
                "liquid_context_samples": 3,
                "compact_aligned_temporal_actions": ["hand_object_contact"],
            }
        },
    )
    assert len(images) == 18
    assert calls == [("fp", 15), ("tp", 3)]


def test_liquid_review_candidate_selects_dense_primary_view_without_becoming_proof(
    monkeypatch, tmp_path
):
    layout = archive.ArchiveLayout(tmp_path / "archive")
    layout.work = tmp_path / "work"
    event = _event(ActionType.LIQUID_MOVEMENT)
    event.observability = {
        "measurements": {
            "repeated_pipette_transfer_path_candidate": True,
            "repeated_pipette_transfer_path_verified": False,
            "repeated_pipette_path": {
                "candidate": True,
                "view_id": "tp",
            },
        }
    }
    calls = []

    def fake_context(clip, output, view_id, samples_per_view=3):
        calls.append((clip, view_id, samples_per_view))
        return [
            (
                f"view_id={view_id}; temporal_phase=timeline_{index:02d}",
                output / f"{index:02d}.jpg",
            )
            for index in range(1, samples_per_view + 1)
        ]

    monkeypatch.setattr(archive, "extract_temporal_review_frames", fake_context)

    images = archive.key_material_review_images(
        layout,
        event,
        {
            "mllm": {
                "use_key_clip_temporal_samples": True,
                "temporal_samples_per_view": 3,
                "liquid_primary_samples": 15,
                "liquid_context_samples": 3,
            },
        },
    )

    assert len(images) == 18
    assert calls == [
        (layout.root / "tp.mp4", "tp", 15),
        (layout.root / "fp.mp4", "fp", 3),
    ]

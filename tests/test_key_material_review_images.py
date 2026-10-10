import hashlib
import json

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
            for phase in ("clip_early", "clip_middle", "clip_late")
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


def _retained_frames(layout, event):
    root = layout.work / "key-material-annotation-inputs" / event.event_id
    root.mkdir(parents=True)
    for view in ("fp", "tp"):
        (root / f"{view}.jpg").write_bytes(b"retained frame")
        (root / f"{view}.json").write_text(json.dumps({
            "event_id": event.event_id, "view_id": view,
            "requested_key_global_ms": event.key_global_ms,
            "decoded_key_global_ms": event.key_global_ms + 100,
        }))
    return root


def test_selected_frames_have_own_times_and_preserve_sequence(tmp_path):
    layout = archive.ArchiveLayout(tmp_path)
    event = _event(ActionType.HAND_OBJECT_CONTACT)
    _retained_frames(layout, event)
    temporal = [("clip_timeline", tmp_path / f"frame-{index}.jpg") for index in range(3)]
    result = archive._with_selected_keyframe_review_images(layout, event, {"mllm": {}}, temporal)
    assert result[:3] == temporal and len(result) == 5
    assert all("sample_scope=selected_keyframe" in label and "requested_global_ms=1500.000; decoded_global_ms=1600.000" in label for label, _ in result[3:])


def test_native_selected_frames_require_image_binding_and_preserve_uncertainty(tmp_path):
    layout = archive.ArchiveLayout(tmp_path)
    event = _event(ActionType.HAND_OBJECT_CONTACT)
    root = _retained_frames(layout, event)
    for view in ("fp", "tp"):
        path = root / f"{view}.json"
        meta = json.loads(path.read_text())
        meta.update(
            schema_version="visioncortex-key-material-annotation-input/2",
            image_sha256=hashlib.sha256(path.with_suffix(".jpg").read_bytes()).hexdigest(),
            source_frame_verification={
                "status": "verified", "detections_bound_to_pixels": True,
                "view_id": view, "decoded_global_ms": meta["decoded_key_global_ms"],
            },
        )
        path.write_text(json.dumps(meta))
    temporal = [("clip_timeline", tmp_path / "frame.jpg")]
    result = archive._with_selected_keyframe_review_images(layout, event, {"mllm": {}}, temporal)
    assert len(result) == 3
    assert all("exact_frame_pts=verified; physical_cross_view_sync=unverified" in label for label, _ in result[1:])
    (root / "tp.jpg").write_bytes(b"replaced image")
    assert archive._with_selected_keyframe_review_images(layout, event, {"mllm": {}}, temporal) == temporal


def test_stale_or_legacy_raw_frames_are_not_claimed_as_selected(tmp_path):
    layout = archive.ArchiveLayout(tmp_path)
    event = _event(ActionType.HAND_OBJECT_CONTACT)
    root = _retained_frames(layout, event)
    temporal = [("clip_timeline", tmp_path / "frame.jpg")]
    event.key_global_ms += 500
    assert archive._with_selected_keyframe_review_images(layout, event, {"mllm": {}}, temporal) == temporal
    event.key_global_ms -= 500
    (root / "tp.json").write_text(json.dumps({"event_id": event.event_id, "view_id": "tp"}))
    assert archive._with_selected_keyframe_review_images(layout, event, {"mllm": {}}, temporal) == temporal


def test_selected_frames_do_not_evict_dense_liquid_context_or_one_view(tmp_path):
    layout = archive.ArchiveLayout(tmp_path)
    event = _event(ActionType.LIQUID_MOVEMENT)
    _retained_frames(layout, event)
    temporal = [("clip_timeline", tmp_path / f"frame-{index}.jpg") for index in range(18)]
    config = {"mllm": {"max_images_per_event": 8, "max_images_per_event_by_action": {"liquid_movement": 18}}}
    assert archive._with_selected_keyframe_review_images(layout, event, config, temporal) == temporal


def test_five_sample_dual_view_sequence_stays_in_order(monkeypatch, tmp_path):
    layout = archive.ArchiveLayout(tmp_path)
    phases = ["clip_early", "clip_mid_early", "clip_middle", "clip_mid_late", "clip_late"]

    def fake_extract(clip, output, view_id, samples_per_view=3):
        return [(f"view_id={view_id}; temporal_phase={phase}; key_clip_frame={index}", output / f"{phase}.jpg") for index, phase in enumerate(phases)]

    monkeypatch.setattr(archive, "extract_temporal_review_frames", fake_extract)
    images = archive.key_material_review_images(layout, _event(ActionType.DEVICE_PANEL_OPERATION), {"mllm": {"temporal_samples_per_view": 5}})
    assert [label.split("temporal_phase=", 1)[1].split(";", 1)[0] for label, _ in images] == [phase for phase in phases for _ in range(2)]

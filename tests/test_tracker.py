from visioncortex.detection import ByteSortTracker
from visioncortex.schemas import BoxEvidence


def _box(x1: float, confidence: float = 0.9) -> BoxEvidence:
    return BoxEvidence(
        class_id=12,
        class_name="pipette",
        confidence=confidence,
        xyxy_norm=(x1, 0.1, x1 + 0.2, 0.4),
    )


def test_bytesort_retains_track_through_low_confidence_detection():
    tracker = ByteSortTracker()
    first = tracker.update([_box(0.1)], 0.0)[0]
    second = tracker.update([_box(0.11, confidence=0.25)], 125.0)[0]
    assert first.track_id == second.track_id


def test_bytesort_motion_prediction_recovers_a_fast_non_overlapping_box():
    tracker = ByteSortTracker(
        motion_prediction_enabled=True,
        maximum_center_distance=0.30,
    )
    first = tracker.update([_box(0.10)], 0.0)[0]
    second = tracker.update([_box(0.18)], 100.0)[0]
    third = tracker.update([_box(0.38)], 200.0)[0]

    assert first.track_id == second.track_id == third.track_id


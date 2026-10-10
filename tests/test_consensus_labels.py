from visioncortex.consensus_labels import build_consensus_box_labels


def test_consensus_accepts_independent_families_and_retains_rejections():
    result = build_consensus_box_labels(
        [
            {
                "image_id": "image-1",
                "model_family": "closed_yolo",
                "class_name": "pipette",
                "confidence": 0.9,
                "xyxy_norm": [0.1, 0.1, 0.5, 0.5],
            },
            {
                "image_id": "image-1",
                "model_family": "grounding_dino",
                "class_name": "pipette",
                "confidence": 0.8,
                "xyxy_norm": [0.11, 0.1, 0.51, 0.5],
            },
            {
                "image_id": "image-1",
                "model_family": "sam2",
                "class_name": "paper",
                "confidence": 0.7,
                "xyxy_norm": [0.6, 0.6, 0.8, 0.8],
            },
        ]
    )

    assert result["truth_status"] == "pseudo_labels_not_ground_truth"
    assert result["accepted_count"] == 1
    assert result["accepted"][0]["model_families"] == [
        "closed_yolo",
        "grounding_dino",
    ]
    assert result["rejected_count"] == 1
    assert result["rejected"][0]["reason"] == "insufficient_independent_models"


def test_consensus_never_merges_observations_from_different_frames():
    result = build_consensus_box_labels(
        [
            {
                "frame_id": "frame-1",
                "model_family": "closed_yolo",
                "class_name": "pipette",
                "confidence": 0.9,
                "xyxy_norm": [0.1, 0.1, 0.5, 0.5],
            },
            {
                "frame_id": "frame-2",
                "model_family": "grounding_dino",
                "class_name": "pipette",
                "confidence": 0.9,
                "xyxy_norm": [0.1, 0.1, 0.5, 0.5],
            },
        ]
    )

    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 2


def test_consensus_rejects_missing_context_instead_of_cross_image_guessing():
    observation = {
        "model_family": "closed_yolo",
        "class_name": "pipette",
        "confidence": 0.9,
        "xyxy_norm": [0.1, 0.1, 0.5, 0.5],
    }

    result = build_consensus_box_labels([observation])

    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 1
    assert result["invalid_observations"][0]["reason"] == (
        "missing_or_invalid_consensus_identity"
    )

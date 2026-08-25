from labvision_evidence.model_certification import (
    _artifact_paths,
    summarize_certification_metrics,
)


def _event_report(tp=230, fp=10, fn=5):
    return {
        "threshold_results": [
            {
                "temporal_iou_threshold": 0.5,
                "true_positives": tp,
                "false_posititives": fp,
                "false_positives": fp,
                "false_negatives": fn,
                "per_class": [
                    {
                        "action_type": "hand_object_contact",
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                    }
                ],
            }
        ]
    }


def _box_report(tp=960, fp=20, fn=40):
    return {
        "status": "completed",
        "image_count": 300,
        "micro": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
    }


def _targets(required_actions):
    return {
        "minimum_event_dataset_count": 1,
        "minimum_event_ground_truth_count": 200,
        "minimum_event_precision": 0.92,
        "minimum_event_recall": 0.95,
        "minimum_event_precision_ci_lower": 0.85,
        "minimum_event_recall_ci_lower": 0.85,
        "minimum_ground_truth_per_action": 20,
        "minimum_per_action_precision": 0.90,
        "minimum_per_action_recall": 0.90,
        "required_action_types": required_actions,
        "minimum_box_dataset_count": 1,
        "minimum_box_ground_truth_instances": 500,
        "minimum_box_precision": 0.92,
        "minimum_box_recall": 0.95,
    }


def test_certification_fails_when_required_action_coverage_is_missing():
    result = summarize_certification_metrics(
        [_event_report()],
        [_box_report()],
        _targets(["hand_object_contact", "liquid_movement"]),
    )

    assert result["passed"] is False
    assert "insufficient_action_ground_truth:liquid_movement" in result["failures"]


def test_certification_passes_only_when_event_and_box_targets_pass():
    result = summarize_certification_metrics(
        [_event_report()],
        [_box_report()],
        _targets(["hand_object_contact"]),
    )

    assert result["passed"] is True
    assert result["events"]["recall"] > 0.95
    assert result["participant_boxes_iou_0_5"]["precision"] > 0.95


def test_certification_covers_text_encoder_and_video_segmentation(tmp_path):
    settings = {
        "models": {
            "first_person_engine": tmp_path / "fp.engine",
            "third_person_engine": tmp_path / "tp.engine",
            "open_vocabulary_key_frame": {
                "model_path": tmp_path / "world.pt",
                "clip_model_path": tmp_path / "clip.pt",
                "grounding_dino_fallback": {
                    "model_path": tmp_path / "dino"
                },
            },
            "temporal_participant_segmentation": {
                "checkpoint_path": tmp_path / "sam2.pt"
            },
        }
    }

    names = {name for name, _path in _artifact_paths(settings)}

    assert names == {
        "first_person_tensor_rt",
        "third_person_tensor_rt",
        "yolo_world",
        "clip_text_encoder",
        "grounding_dino",
        "sam2_video_segmentation",
    }

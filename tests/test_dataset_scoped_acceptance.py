from labvision_evidence.pipeline import EvidencePipeline


def test_six_view_baseline_only_applies_to_declared_experiment_id():
    pipeline = EvidencePipeline(
        {
            "validation": {
                "acceptance_baseline": (
                    "./configs/acceptance/"
                    "six-view-three-hour-reviewed-baseline.json"
                )
            }
        }
    )

    pipeline._current_experiment_id = "CustomFlow_standard_correct_12_A_0004"
    assert pipeline._acceptance_baseline() is None
    assert pipeline._acceptance_baseline_selection["reason"] == (
        "experiment_id_not_applicable"
    )

    pipeline._current_experiment_id = "exp_20260810_144014_e918b762"
    baseline = pipeline._acceptance_baseline()
    assert baseline is not None
    assert baseline["minimum_selected_key_events"] == 80
    assert pipeline._acceptance_baseline_selection["applied"] is True

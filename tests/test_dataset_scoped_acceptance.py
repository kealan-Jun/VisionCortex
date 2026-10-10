import json

from visioncortex.pipeline import EvidencePipeline

def fixture_baseline(tmp_path):
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps({"baseline_id": "fixture-baseline", "applicability": {"experiment_ids": ["fixture-experiment"]}, "minimum_selected_key_events": 80}))
    return str(path)


def test_six_view_baseline_only_applies_to_declared_experiment_id(tmp_path):
    pipeline = EvidencePipeline(
        {
            "validation": {
                "acceptance_baseline": (
                    fixture_baseline(tmp_path)
                )
            }
        }
    )

    pipeline._current_experiment_id = "fixture-unrelated-experiment"
    assert pipeline._acceptance_baseline() is None
    assert pipeline._acceptance_baseline_selection["reason"] == (
        "experiment_id_not_applicable"
    )

    pipeline._current_experiment_id = "fixture-experiment"
    baseline = pipeline._acceptance_baseline()
    assert baseline is not None
    assert baseline["minimum_selected_key_events"] == 80
    assert pipeline._acceptance_baseline_selection["applied"] is True


def test_dataset_map_selects_only_the_current_experiment_artifact(tmp_path):
    pipeline = EvidencePipeline(
        {
            "validation": {
                "acceptance_baseline": {
                    "fixture-experiment": (
                        fixture_baseline(tmp_path)
                    )
                }
            }
        }
    )

    pipeline._current_experiment_id = "exp-not-configured"
    assert pipeline._acceptance_baseline() is None
    assert pipeline._acceptance_baseline_selection["reason"] == (
        "no_artifact_configured_for_experiment_id"
    )

    pipeline._current_experiment_id = "fixture-experiment"
    baseline = pipeline._acceptance_baseline()
    assert baseline is not None
    assert pipeline._acceptance_baseline_selection[
        "configured_via_dataset_map"
    ] is True

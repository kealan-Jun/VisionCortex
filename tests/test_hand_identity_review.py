from copy import deepcopy

from visioncortex.mllm import normalize_uncalibrated_hand_identity


def test_cross_view_hand_assignments_remain_unknown_without_calibration():
    raw = {
        "status": "completed",
        "current_step": "一只手按键，随后双手拿起包装",
        "hand_object_interactions": [
            {"hand": "left", "object": "天平", "contact": "操作"},
            {"hand": "right", "object": "天平", "contact": "接触"},
        ],
        "per_view_observations": [
            {"view_id": "fp", "observation": "右手指尖按压按键"},
            {"view_id": "tp", "observation": "左手指尖按压按键"},
        ],
        "usage": {"total_tokens": 125},
    }
    before = deepcopy(raw)
    result = normalize_uncalibrated_hand_identity(raw)
    assert raw == before
    assert result["current_step"] == raw["current_step"]
    assert result["usage"] == raw["usage"]
    assert result["hand_object_interactions"] == [{"hand": "unknown", "object": "天平", "contact": "操作/接触"}]
    assert all(item["observation"] == "手指尖按压按键" for item in result["per_view_observations"])
    assert result["hand_identity_review"]["original_hand_object_interactions"] == raw["hand_object_interactions"]
    assert len(result["hand_identity_review"]["original_text_fields"]) == 2
    assert normalize_uncalibrated_hand_identity(result) == result


def test_hand_identity_review_preserves_spatial_direction_and_two_hand_narrative():
    raw = {"status": "completed", "current_step": "双手拿起右手边的包装", "next_step": "Move the package to the left-hand side", "steps": [{"current_step": "The right hand holds a bottle"}]}
    result = normalize_uncalibrated_hand_identity(raw)
    assert result["current_step"] == raw["current_step"]
    assert result["next_step"] == raw["next_step"]
    assert result["steps"][0]["current_step"] == "The hand holds a bottle"


def test_failed_or_identity_free_responses_are_not_reinterpreted():
    failed = {"status": "failed", "error": "left hand request failed"}
    assert normalize_uncalibrated_hand_identity(failed) is failed
    candidate_review = {"status": "completed", "views": [{"view_id": "fp", "selected_candidate_ids": ["V1-01"], "reason": "hand holds paper"}]}
    assert normalize_uncalibrated_hand_identity(candidate_review) == candidate_review

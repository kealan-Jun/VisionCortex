from visioncortex.archive import evidence_package_eval


def test_empty_observation_is_reportable_without_becoming_negative_proof(tmp_path):
    evaluation = evidence_package_eval(tmp_path, [], [], [], {})

    assert evaluation["passed"] is True
    assert evaluation["package_integrity_status"] == "passed_empty_observation"
    assert evaluation["observation_status"] == "no_accepted_observation"
    assert evaluation["observation_evidence_classification"] == "PARTIAL_EVIDENCE"
    assert evaluation["negative_action_claim_supported"] is False
    assert evaluation["segment_count"] == 0

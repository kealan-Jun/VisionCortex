import json

import pytest

from visioncortex.engine_export import export_branch_options, verify_export_branch


def test_native_default_and_explicit_trained_head():
    assert export_branch_options({}, "first_person") == {}
    assert export_branch_options({"models": {"engine_branch_by_role": {"first_person": "one2many"}}}, "first_person") == {"end2end": False}
    with pytest.raises(ValueError):
        export_branch_options({"models": {"engine_branch_by_role": {"wrong_role": "one2many"}}}, "first_person")


@pytest.mark.parametrize("observed", [True, None, "false"])
def test_wrong_or_unproven_branch_is_rejected(tmp_path, observed):
    payload = json.dumps({"end2end": observed}).encode()
    p = tmp_path / "model.engine"
    p.write_bytes(len(payload).to_bytes(4, "little") + payload + b"plan")
    with pytest.raises(ValueError, match="trained branch"):
        verify_export_branch(p, {"end2end": False})


def test_verified_one2many_engine(tmp_path):
    payload = json.dumps({"end2end": False}).encode()
    p = tmp_path / "model.engine"
    p.write_bytes(len(payload).to_bytes(4, "little") + payload + b"plan")
    assert verify_export_branch(p, {"end2end": False})["end2end"] is False

from __future__ import annotations

from repo_paths import ROOT

import pytest

from tools.diagnostics.config_profiles import inspect_profiles, profile_chain
from visioncortex.config import load_config


def test_local_profiles_have_no_production_ancestor_or_implicit_speech_provider():
    report = inspect_profiles(ROOT / "configs")
    for row in report["profiles"]:
        if row["kind"] == "local":
            assert not any("production" in name for name in row["chain"])
            assert row["collection_enabled"] is False
            assert row["nas_sync_enabled"] is False
            assert row["speech_enabled"] is False
    assert report["storage_or_model_access"] is False


def test_profile_graph_rejects_escape_and_cycle(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text('extends: ../outside.yaml\n')
    with pytest.raises(ValueError, match="stay"):
        profile_chain(path)
    path.write_text('extends: ./profile.yaml\n')
    with pytest.raises(ValueError, match="cycle"):
        profile_chain(path)


@pytest.mark.parametrize("hardware", ["rtx3050-6gb-ubuntu20", "rtx3090ti-ubuntu"])
def test_local_and_production_share_hardware_analysis_without_storage(hardware):
    local = load_config(ROOT / "configs" / f"{hardware}-local.yaml")
    production = load_config(ROOT / "configs" / f"{hardware}-production.yaml")
    assert local["performance"] == production["performance"]
    assert local["models"] == production["models"]
    assert local["storage"]["sync_to_nas"] is False
    assert production["storage"]["sync_to_nas"] is False
    assert production["project"]["site_configuration_required"] is True


# Public production names are unconfigured templates. Frozen private deployments
# retain their own immutable configuration, rather than a public host-specific hash.
@pytest.mark.parametrize("profile", [
    "rtx4060-laptop-production.yaml", "rtx3090ti-ubuntu-production.yaml",
    "rtx3050-6gb-ubuntu20-production.yaml", "rtx4090-production.yaml",
])
def test_public_deployment_templates_require_explicit_site_configuration(profile):
    value = load_config(ROOT / "configs" / profile)
    assert value["project"]["site_configuration_required"] is True
    assert value["project"]["run_purpose"] != "production"
    assert value["collection_ingest"]["enabled"] is False
    assert value["device_day"]["enabled"] is False
    assert value["mllm"]["enabled"] is False
    assert value["speech_recognition"]["enabled"] is False
    assert value["storage"]["sync_to_nas"] is False

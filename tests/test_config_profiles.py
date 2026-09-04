from __future__ import annotations

from pathlib import Path

import pytest

from visioncortex.config import load_config


def test_rtx4090_profile_inherits_quality_rules_and_keeps_view_count_dynamic():
    profile = Path(__file__).resolve().parents[1] / "configs" / "rtx4090-production.yaml"

    config = load_config(profile)

    performance = config["performance"]
    assert performance["profile"] == "rtx4090-24gb-dynamic-multiview"
    assert performance["detection_fps"] == 10.0
    assert performance["fine_group_local_recall_enabled"] is True
    assert performance["fine_progressive_cross_view"] is True
    assert performance["coarse_decode_lanes"] == ["cuda"]
    assert performance["fine_decode_lanes"] == ["cuda"]
    assert performance["fine_min_third_person_views"] is None
    assert performance["synchronized_segment_waves"] is False
    assert performance["batch_size"] == 32
    assert config["models"]["expected_class_count"] == 21


def test_rtx3090ti_ubuntu_profile_matches_host_and_keeps_view_count_dynamic():
    profile = Path(__file__).resolve().parents[1] / "configs" / "rtx3090ti-ubuntu-production.yaml"

    config = load_config(profile)

    assert config["collection_ingest"] == {
        "enabled": True,
        "poll_seconds": 30,
        "settle_seconds": 120,
        "max_results": 200,
        "persist_snapshot": True,
        "snapshot_path": None,
        "mode": "directory_metadata",
        "source_root": "/home/x1/桌面/nas",
        "camera_directories": [],
        "camera_directory_glob": "*_cam*",
        "camera_role_map": {
            "lubancat-4df661d7_cam01": "first_person",
            "lubancat-e8cc0cb3_cam01": "first_person",
            "orangepi5pro-ab748372_cam01": "third_person",
            "orangepi5pro-b439137c_cam02": "third_person",
            "orangepi5pro-d12a4719_cam01": "first_person",
            "orangepi5pro-f022c4_cam01": "third_person",
            "rk3588-ubuntu_cam01": "third_person",
        },
        "discover_plain_video_csv": False,
        "max_scan_directories": 20000,
        "max_recordings": 5000,
        "max_recordings_per_camera": 32,
    }

    performance = config["performance"]
    assert performance["profile"] == "rtx3090ti-24gb-ubuntu-dynamic-multiview"
    assert performance["tensor_rt"] == "required"
    assert performance["batch_size"] == 16
    assert performance["engine_batch_size"] == 4
    assert performance["coarse_decode_lanes"] == ["cuda"]
    assert performance["fine_decode_lanes"] == ["cuda"]
    assert performance["fine_min_third_person_views"] is None
    assert performance["synchronized_segment_waves"] is False
    assert config["models"]["expected_class_count"] == 21
    assert config["storage"]["index_csv"] == "/home/x1/桌面/nas/experiment_record_index.csv"
    assert config["storage"]["archive_root"] == "/home/x1/桌面/nas/VisionCortexExperimentArchive"
    assert config["storage"]["local_cache_root"] == "/home/x1/桌面/nas/VisionCortexExperimentCache"
    assert config["storage"]["local_staging_root"].endswith("/.VisionCortex-Run-Staging")
    assert config["storage"]["web_upload_retention_mode"] == "nas_only"
    assert config["web_upload"]["chunk_size_mib"] == 16
    assert config["web_upload"]["session_ttl_hours"] == 168
    assert config["archive"]["include_empty_action_categories"] is False
    assert config["models"]["first_person_engine"].startswith(
        "/srv/sentinel-data/VisionCortex3090Ti/Engines/"
    )
    segmentation = config["models"]["temporal_participant_segmentation"]
    assert segmentation["enabled"] is True
    assert segmentation["required_for_final_key_material"] is True
    assert segmentation["model"] == "sam2.1_hiera_base_plus"
    assert segmentation["maximum_frames_per_clip"] == 9
    assert segmentation["checkpoint_path"].startswith(
        "/srv/sentinel-data/VisionCortex3090Ti/Engines/"
    )
    liquid = config["models"]["liquid_semantic_sidecar"]
    assert liquid["enabled"] is True
    assert liquid["required_for_selected_actions"] is True
    assert len(liquid["checkpoint_sha256"]) == 64
    assert liquid["license"] == "CC-BY-4.0"
    verification = config["key_materials"]["selective_verification"]
    assert verification["enabled"] is True
    assert verification["mode"] == "ambiguous_or_high_risk"
    assert verification["max_events_per_run"] == 120
    assert verification["max_views_per_event"] == 2


def test_rtx3090ti_local_profile_has_no_nas_storage_paths():
    profile = Path(__file__).resolve().parents[1] / "configs" / "rtx3090ti-ubuntu-local.yaml"

    config = load_config(profile)

    assert config["mllm"]["enabled"] is False
    assert config["storage"]["sync_to_nas"] is False
    assert config["key_materials"]["selective_verification"]["enabled"] is True
    for key in (
        "index_csv",
        "device_registry_path",
        "archive_root",
        "local_input_root",
        "local_runtime_root",
        "local_cache_root",
        "local_staging_root",
    ):
        assert str(config["storage"][key]).startswith(
            "/srv/sentinel-data/VisionCortex3090Ti/Runtime/NoNasWeb/"
        )


def test_ubuntu_runtime_paths_and_engines_can_be_overridden(monkeypatch):
    profile = Path(__file__).resolve().parents[1] / "configs" / "rtx3090ti-ubuntu-production.yaml"
    monkeypatch.setenv("VISIONCORTEX_OUTPUT_ROOT", "/runtime/outputs")
    monkeypatch.setenv("VISIONCORTEX_LOCAL_STAGING_ROOT", "/runtime/staging")
    monkeypatch.setenv("VISIONCORTEX_FIRST_PERSON_ENGINE", "/cache/fp.engine")
    monkeypatch.setenv("VISIONCORTEX_THIRD_PERSON_ENGINE", "/cache/tp.engine")
    monkeypatch.setenv(
        "VISIONCORTEX_SELECTIVE_KEY_MATERIAL_VERIFICATION", "false"
    )

    config = load_config(profile)

    assert config["project"]["output_root"] == "/runtime/outputs"
    assert config["storage"]["local_staging_root"] == "/runtime/staging"
    assert config["models"]["first_person_engine"] == "/cache/fp.engine"
    assert config["models"]["third_person_engine"] == "/cache/tp.engine"
    assert config["key_materials"]["selective_verification"]["enabled"] is False


def test_invalid_selective_verification_environment_override_fails(monkeypatch):
    monkeypatch.setenv(
        "VISIONCORTEX_SELECTIVE_KEY_MATERIAL_VERIFICATION", "sometimes"
    )

    with pytest.raises(ValueError, match="must be true or false"):
        load_config()


def test_profile_inheritance_rejects_parent_directory_escape(tmp_path: Path):
    outside = tmp_path / "outside.yaml"
    outside.write_text("performance: {}\n", encoding="utf-8")
    child_root = tmp_path / "profiles"
    child_root.mkdir()
    child = child_root / "child.yaml"
    child.write_text("extends: ../outside.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must stay"):
        load_config(child)


def test_profile_inheritance_rejects_cycles(tmp_path: Path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: ./second.yaml\n", encoding="utf-8")
    second.write_text("extends: ./first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        load_config(first)


def test_mllm_config_rejects_evidence_budget_that_would_drop_a_view(
    tmp_path: Path,
):
    profile = tmp_path / "bad-mllm-budget.yaml"
    profile.write_text(
        "mllm:\n  max_images_per_event: 5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preserve both temporal views"):
        load_config(profile)

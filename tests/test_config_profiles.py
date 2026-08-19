from __future__ import annotations

from pathlib import Path

import pytest

from labvision_evidence.config import load_config


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

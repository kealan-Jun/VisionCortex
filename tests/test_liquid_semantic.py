from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from labvision_evidence.liquid_semantic import validate_liquid_semantic_runtime


def test_liquid_semantic_runtime_can_be_disabled():
    assert validate_liquid_semantic_runtime({"models": {}}) == {
        "enabled": False,
        "status": "disabled",
    }


def test_liquid_semantic_runtime_validates_pinned_cpu_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "liquid.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = {
        "models": {
            "liquid_semantic_sidecar": {
                "enabled": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": digest,
                "device": "cpu",
                "source_record": "https://zenodo.org/records/3697767",
                "license": "CC-BY-4.0",
            }
        }
    }

    result = validate_liquid_semantic_runtime(config)

    assert result["status"] == "validated"
    assert result["policy"] == "observation_only_never_action_confirmation"
    assert result["checkpoint_sha256"] == digest


def test_liquid_semantic_runtime_fails_closed_on_hash_mismatch(tmp_path: Path):
    checkpoint = tmp_path / "liquid.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "models": {
            "liquid_semantic_sidecar": {
                "enabled": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": "0" * 64,
                "device": "cpu",
            }
        }
    }

    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_liquid_semantic_runtime(config)


def test_liquid_semantic_runtime_revalidates_a_changed_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "liquid.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = {
        "models": {
            "liquid_semantic_sidecar": {
                "enabled": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": digest,
                "device": "cpu",
            }
        }
    }
    assert validate_liquid_semantic_runtime(config)["status"] == "validated"

    checkpoint.write_bytes(b"checkpoint-tampered")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_liquid_semantic_runtime(config)

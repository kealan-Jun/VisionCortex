from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from labvision_evidence.public_model_assets import prepare_public_model_assets


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> dict:
    world = tmp_path / "world.pt"
    clip = tmp_path / "clip.pt"
    sam2 = tmp_path / "sam2.pt"
    dino = tmp_path / "dino"
    dino.mkdir()
    weights = dino / "model.safetensors"
    for path, payload in (
        (world, b"world"),
        (clip, b"clip"),
        (sam2, b"sam2"),
        (weights, b"dino"),
        (dino / "config.json", b"{}"),
        (dino / "preprocessor_config.json", b"{}"),
    ):
        path.write_bytes(payload)
    return {
        "models": {
            "open_vocabulary_key_frame": {
                "enabled": True,
                "model_path": str(world),
                "model_sha256": _sha(world),
                "model_download_url": "https://example.invalid/world.pt",
                "clip_model_path": str(clip),
                "clip_model_sha256": _sha(clip),
                "clip_model_download_url": "https://example.invalid/clip.pt",
                "grounding_dino_fallback": {
                    "enabled": True,
                    "model_path": str(dino),
                    "model_sha256": _sha(weights),
                    "model_repository": "test/dino",
                    "model_revision": "frozen",
                },
            },
            "temporal_participant_segmentation": {
                "enabled": True,
                "checkpoint_path": str(sam2),
                "checkpoint_sha256": _sha(sam2),
                "checkpoint_download_url": "https://example.invalid/sam2.pt",
            },
        }
    }


def test_existing_public_assets_are_hash_checked_without_network(tmp_path: Path):
    result = prepare_public_model_assets(_config(tmp_path))

    assert result["status"] == "completed"
    assert result["network_used"] is False
    assert result["asset_count"] == 4
    assert {item["status"] for item in result["assets"]} == {
        "reused_verified"
    }


def test_existing_public_asset_hash_mismatch_fails_closed(tmp_path: Path):
    config = _config(tmp_path)
    config["models"]["open_vocabulary_key_frame"]["model_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="unexpected hash"):
        prepare_public_model_assets(config)

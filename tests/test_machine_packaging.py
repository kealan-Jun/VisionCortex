from pathlib import Path

from visioncortex.config import load_config


def test_container_profile_does_not_inherit_host_paths_or_background_producers(
    monkeypatch,
):
    monkeypatch.delenv("VISIONCORTEX_CONFIG", raising=False)
    root = Path(__file__).parents[1]
    config = load_config(root / "deployment/inference/pipeline.example.yaml")
    assert config["runtime"]["role"] == "web"
    assert config["runtime"]["local_only"] is True
    assert config["device_day"]["enabled"] is False
    assert config["collection_ingest"]["enabled"] is False
    assert config["mllm"]["enabled"] is False
    runtime_root = Path(config["storage"]["local_runtime_root"])
    assert Path(config["project"]["output_root"]).is_relative_to(runtime_root)
    for field in (
        "archive_root",
        "local_runtime_root",
        "local_input_root",
        "local_cache_root",
        "local_staging_root",
    ):
        assert Path(config["storage"][field]).is_relative_to(runtime_root)
    for field in (
        "first_person",
        "third_person",
        "first_person_engine",
        "third_person_engine",
    ):
        assert config["models"][field].startswith("/models/")

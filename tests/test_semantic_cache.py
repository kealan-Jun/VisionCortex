from pathlib import Path

from labvision_evidence.archive import (
    _read_semantic_cache,
    _semantic_cache_reads_enabled,
    _semantic_fingerprint,
    _write_semantic_cache,
)


def test_semantic_cache_is_content_addressed_and_rejects_changed_boundaries(
    tmp_path, default_config
):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"same-visible-evidence")
    default_config["storage"]["local_cache_root"] = str(tmp_path / "cache")
    evidence = {"global_start_ms": 1000.0, "global_end_ms": 2000.0}
    first = _semantic_fingerprint(
        "experiment-group",
        default_config,
        "prompt-v1",
        evidence,
        [("first_person", image)],
    )
    same = _semantic_fingerprint(
        "experiment-group",
        default_config,
        "prompt-v1",
        dict(evidence),
        [("first_person", image)],
    )
    changed = _semantic_fingerprint(
        "experiment-group",
        default_config,
        "prompt-v1",
        {**evidence, "global_end_ms": 2500.0},
        [("first_person", image)],
    )
    assert first == same
    assert first != changed

    cache_path = Path(default_config["storage"]["local_cache_root"]) / "result.json"
    persisted = _write_semantic_cache(
        cache_path,
        first,
        {"status": "completed", "usage": {"total_tokens": 123}},
    )
    assert persisted["cache_reused"] is False
    reused = _read_semantic_cache(cache_path, first)
    assert reused is not None
    assert reused["cache_reused"] is True
    assert _read_semantic_cache(cache_path, changed) is None


def test_semantic_cache_namespace_and_cold_read_policy(tmp_path, default_config):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"visible-evidence")
    evidence = {"global_start_ms": 1000.0, "global_end_ms": 2000.0}

    default_config["project"]["cache_namespace"] = "audit-a"
    default_config["project"]["cache_mode"] = "cold"
    first = _semantic_fingerprint(
        "key-material", default_config, "prompt", evidence, [("fp", image)]
    )
    assert _semantic_cache_reads_enabled(default_config) is False

    default_config["project"]["cache_mode"] = "reuse"
    hot = _semantic_fingerprint(
        "key-material", default_config, "prompt", evidence, [("fp", image)]
    )
    assert first == hot
    assert _semantic_cache_reads_enabled(default_config) is True

    default_config["project"]["cache_namespace"] = "audit-b"
    isolated = _semantic_fingerprint(
        "key-material", default_config, "prompt", evidence, [("fp", image)]
    )
    assert isolated != hot

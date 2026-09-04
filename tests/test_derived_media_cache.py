from pathlib import Path

from visioncortex.archive import (
    _link_or_copy_immutable,
    _materialize_derived_media,
)


def test_immutable_materialization_cleans_noop_replace_hardlink(tmp_path):
    source = tmp_path / "cache.mp4"
    destination = tmp_path / "result.mp4"
    source.write_bytes(b"same-derived-media")
    destination.hardlink_to(source)

    assert _link_or_copy_immutable(source, destination) == "hardlink"
    assert destination.read_bytes() == b"same-derived-media"
    assert list(tmp_path.glob(".*.cache-*")) == []


def _config(tmp_path: Path, mode: str) -> dict:
    return {
        "project": {"cache_mode": mode, "cache_namespace": "bounded-audit"},
        "performance": {
            "derived_media_cache_enabled": True,
            "ffmpeg_video_encoder": "libx264",
        },
        "storage": {"local_cache_root": str(tmp_path / "cache")},
    }


def test_derived_media_cache_cold_populates_and_reuse_verifies(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable-source-identity")
    cold_output = tmp_path / "cold.mp4"
    calls = []

    def generate_cold():
        calls.append("cold")
        cold_output.write_bytes(b"exact-derived-media")

    monkeypatch.setattr(
        "visioncortex.archive.select_video_encoder", lambda preferred: preferred
    )
    cold = _materialize_derived_media(
        cold_output,
        "key-view-clip",
        {"start_ms": 1000.0, "duration_ms": 3000.0},
        [source],
        _config(tmp_path, "cold"),
        generate_cold,
    )
    assert calls == ["cold"]
    assert cold["cache_enabled"] is True
    assert cold["cache_reused"] is False

    hot_output = tmp_path / "hot.mp4"

    def must_not_generate():
        raise AssertionError("verified cache hit must not re-encode")

    hot = _materialize_derived_media(
        hot_output,
        "key-view-clip",
        {"start_ms": 1000.0, "duration_ms": 3000.0},
        [source],
        _config(tmp_path, "reuse"),
        must_not_generate,
    )
    assert hot["cache_reused"] is True
    assert hot_output.read_bytes() == cold_output.read_bytes()
    assert hot["cache_output_sha256"] == cold["cache_output_sha256"]


def test_cold_run_reuses_same_process_idempotent_media(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable-source-identity")
    first_output = tmp_path / "first.mp4"
    second_output = tmp_path / "second.mp4"
    monkeypatch.setattr(
        "visioncortex.archive.select_video_encoder", lambda preferred: preferred
    )

    _materialize_derived_media(
        first_output,
        "key-view-clip",
        {"event_id": "EVT-1", "start_ms": 1000.0, "duration_ms": 3000.0},
        [source],
        _config(tmp_path, "cold"),
        lambda: first_output.write_bytes(b"exact-derived-media"),
    )

    def must_not_regenerate():
        raise AssertionError("same cold run must reuse its verified immutable entry")

    result = _materialize_derived_media(
        second_output,
        "key-view-clip",
        {"event_id": "EVT-1", "start_ms": 1000.0, "duration_ms": 3000.0},
        [source],
        _config(tmp_path, "cold"),
        must_not_regenerate,
    )

    assert result["cache_reused"] is True
    assert result["cache_reuse_scope"] == "same_process_idempotent"
    assert second_output.read_bytes() == first_output.read_bytes()


def test_derived_media_cache_changed_bounds_force_new_generation(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable-source-identity")
    monkeypatch.setattr(
        "visioncortex.archive.select_video_encoder", lambda preferred: preferred
    )
    first = tmp_path / "first.mp4"
    _materialize_derived_media(
        first,
        "experiment-view-clip",
        {"start_ms": 0.0, "duration_ms": 1000.0},
        [source],
        _config(tmp_path, "cold"),
        lambda: first.write_bytes(b"first-window"),
    )
    changed = tmp_path / "changed.mp4"
    calls = []

    def generate_changed():
        calls.append("changed")
        changed.write_bytes(b"second-window")

    result = _materialize_derived_media(
        changed,
        "experiment-view-clip",
        {"start_ms": 0.0, "duration_ms": 2000.0},
        [source],
        _config(tmp_path, "reuse"),
        generate_changed,
    )
    assert calls == ["changed"]
    assert result["cache_reused"] is False

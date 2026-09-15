from pathlib import Path

from visioncortex import api


class _Manifest:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.views = [object(), object()]

    def model_dump(self, mode="python"):
        return {"experiment_id": self.experiment_id, "views": []}


def _pipeline_spy(observed: dict[str, str], output: Path):
    class Pipeline:
        def __init__(self, _settings, _progress):
            pass

        def run(self, manifest):
            observed["experiment_id"] = manifest.experiment_id
            return output

    return Pipeline


def test_web_fixed_benchmark_preserves_index_experiment_id(monkeypatch, tmp_path):
    source_id = api._BENCHMARK_EXPERIMENT_ID
    manifest = _Manifest(source_id)
    manifest_path = tmp_path / "manifest.yaml"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    history = tmp_path / "history"
    observed = {}

    monkeypatch.setattr(
        api,
        "fixed_archive_staging_paths",
        lambda *_args: (fixed, staging, history),
    )
    monkeypatch.setattr(
        api,
        "prepare_from_nas_index",
        lambda *_args: (
            manifest,
            manifest_path,
            {"source_validation": {"missing_count": 0}},
        ),
    )
    monkeypatch.setattr(api, "EvidencePipeline", _pipeline_spy(observed, staging))
    monkeypatch.setattr(api, "_update", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api, "_append_fixed_benchmark_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(api, "promote_fixed_archive", lambda *_args: {})

    api._execute_fixed_benchmark(
        "run-fixed",
        {"storage": {}},
        staging,
        {"request_started_perf": 0.0},
    )

    assert observed["experiment_id"] == source_id


def test_web_index_collection_preserves_source_experiment_id(monkeypatch, tmp_path):
    source_id = "exp-source"
    manifest = _Manifest(source_id)
    manifest_path = tmp_path / "manifest.yaml"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    history = tmp_path / "history"
    observed = {}

    monkeypatch.setattr(
        api,
        "prepare_from_nas_index",
        lambda *_args: (manifest, manifest_path, {}),
    )
    monkeypatch.setattr(api, "EvidencePipeline", _pipeline_spy(observed, staging))
    monkeypatch.setattr(api, "record_collection_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api, "_update", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        api, "_append_collection_index_metrics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(api, "promote_fixed_archive", lambda *_args: {})

    api._execute_index_collection(
        "run-index",
        source_id,
        "Collection-01",
        {"storage": {}},
        staging,
        fixed,
        history,
        {"request_started_perf": 0.0},
    )

    assert observed["experiment_id"] == source_id


def test_local_collection_recovery_seals_metadata_inside_archive_without_copying_video(monkeypatch, tmp_path):
    import json
    import yaml
    from visioncortex.input_seal import build_input_seal, verify_input_seal, write_input_seal
    from visioncortex.schemas import RunManifest, ViewInput, ViewRole

    source = tmp_path / "original.mp4"
    source.write_bytes(b"synthetic source; must remain at original path")
    manifest = RunManifest(experiment_id="local-source", views=[
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=source),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=source),
    ])
    manifest_path = tmp_path / "Runtime/input-manifests/manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(yaml.safe_dump(manifest.model_dump(mode="json")))
    role_resolution = {"status": "resolved", "views": [{"view_id": "fp", "role": "first_person"}]}
    sources = [{"path": str(source), "size_bytes": source.stat().st_size}]
    initial = build_input_seal(manifest, source_mode="nas_segmented_virtual_timeline",
                               sources=sources, role_resolution=role_resolution, copied_source_bytes=0)
    seal_path = write_input_seal(manifest_path.with_name("input_seal.json"), initial)
    before = seal_path.read_bytes()
    staging = tmp_path / "Archives/.staging/local-run"
    ingest = {"original_retention": {"input_seal": str(seal_path)}}
    monkeypatch.setattr(api, "_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "preflight_manifest_inputs", lambda *args: {"status": "passed"})
    api._preflight_and_seal_collection_input(
        run_id="run-local", source_experiment_id="local-source", archive_name="Local-Archive",
        manifest=manifest, manifest_path=manifest_path, ingest=ingest, settings={},
        staging_root=staging, fixed_root=tmp_path / "Archives/Local-Archive", history_root=tmp_path / "History", timing={},
    )
    recovery = json.loads((staging / "JSON-Config-Files/Input-Manifests/queue_recovery.json").read_text())
    assert recovery["manifest_relative_path"] == "JSON-Config-Files/input_manifest.yaml"
    archived = RunManifest.model_validate(yaml.safe_load((staging / recovery["manifest_relative_path"]).read_text()))
    assert archived == manifest
    refreshed = json.loads((staging / recovery["input_seal_relative_path"]).read_text())
    assert verify_input_seal(refreshed)
    assert refreshed["sources"] == sources and refreshed["role_resolution"] == role_resolution
    assert refreshed["copied_source_bytes"] == 0
    assert seal_path.read_bytes() == before
    assert list(tmp_path.rglob("*.mp4")) == [source]


def test_mismatched_collection_input_seal_cannot_create_a_recovery_receipt(monkeypatch, tmp_path):
    import pytest
    from visioncortex.schemas import RunManifest, ViewInput, ViewRole

    manifest = RunManifest(experiment_id="local-source", views=[
        ViewInput(view_id="fp", role=ViewRole.FIRST_PERSON, video=tmp_path / "fp.mp4"),
        ViewInput(view_id="tp", role=ViewRole.THIRD_PERSON, video=tmp_path / "tp.mp4"),
    ])
    monkeypatch.setattr(api, "_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "preflight_manifest_inputs", lambda *args: {"status": "passed"})
    staging = tmp_path / "staging"
    with pytest.raises(ValueError, match="封条无效"):
        api._preflight_and_seal_collection_input(
            run_id="run-bad", source_experiment_id="local-source", archive_name="Local-Archive",
            manifest=manifest, manifest_path=tmp_path / "manifest.yaml", ingest={}, settings={},
            staging_root=staging, fixed_root=tmp_path / "fixed", history_root=tmp_path / "history", timing={},
        )
    assert not (staging / "JSON-Config-Files/Input-Manifests/queue_recovery.json").exists()

from pathlib import Path

from labvision_evidence import api


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

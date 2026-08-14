import json

import pytest

from labvision_evidence import pipeline as pipeline_module
from labvision_evidence.config import load_config
from labvision_evidence.pipeline import EvidencePipeline
from labvision_evidence.schemas import RunManifest, ViewInput, ViewRole


class _ReachedVideoProbe(RuntimeError):
    pass


class _NoopResourceMonitor:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        pass

    def set_stage(self, _stage):
        pass

    def stop(self):
        return {}


def test_real_pipeline_preflight_calls_imported_video_probe(tmp_path, monkeypatch):
    """Regression for a local variable shadowing the imported probe_views function."""

    first_video = tmp_path / "first.mp4"
    third_video = tmp_path / "third.mp4"
    first_video.write_bytes(b"first")
    third_video.write_bytes(b"third")
    manifest = RunManifest(
        experiment_id="preflight-shadow-regression",
        views=[
            ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=first_video),
            ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=third_video),
        ],
    )
    config = load_config()
    config["project"]["output_root"] = str(tmp_path / "output")
    config["storage"]["local_cache_root"] = str(tmp_path / "cache")
    config["storage"]["sync_to_nas"] = False
    config["storage"]["run_output_mode"] = "local"

    calls = []

    def stop_at_video_probe(views, *, workers, prefer_clock_metadata):
        calls.append((views, workers, prefer_clock_metadata))
        raise _ReachedVideoProbe("preflight reached video probe")

    monkeypatch.setattr(pipeline_module, "ResourceMonitor", _NoopResourceMonitor)
    monkeypatch.setattr(pipeline_module, "validate_models", lambda _config: {})
    monkeypatch.setattr(
        pipeline_module,
        "video_encoder_preflight",
        lambda _encoder: {"selected_encoder": "libx264"},
    )
    monkeypatch.setattr(pipeline_module, "probe_views", stop_at_video_probe)

    with pytest.raises(_ReachedVideoProbe, match="preflight reached video probe"):
        EvidencePipeline(config).run(manifest)

    assert len(calls) == 1
    assert [view.view_id for view in calls[0][0]] == ["fp01", "tp01"]
    assert "probe_views" not in EvidencePipeline.run.__code__.co_varnames
    status_paths = list((tmp_path / "output").rglob("run_status.json"))
    assert len(status_paths) == 1
    status = json.loads(status_paths[0].read_text(encoding="utf-8"))
    assert status["stage"] == "failed"
    assert status["failed_stage"] == "preflight"
    assert "preflight" not in status["completed_stages"]

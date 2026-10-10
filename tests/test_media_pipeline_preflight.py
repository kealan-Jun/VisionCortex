from pathlib import Path

from visioncortex import pipeline as pipeline_module
from visioncortex.pipeline import run_media_pipeline_preflight
from visioncortex.schemas import RunManifest, VideoInfo, ViewInput, ViewRole


def _info(path: Path) -> VideoInfo:
    return VideoInfo(
        path=path,
        duration_ms=10_000,
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=300,
        size_bytes=100,
    )


def test_media_pipeline_preflight_exercises_both_views_and_grid(tmp_path, monkeypatch):
    first_path = tmp_path / "first.mp4"
    third_path = tmp_path / "third.mp4"
    manifest = RunManifest(
        experiment_id="media-smoke",
        views=[
            ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=first_path),
            ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=third_path),
        ],
    )
    infos = {"fp01": _info(first_path), "tp01": _info(third_path)}
    extracted = []

    def fake_extract(view, _info_value, destination, start_ms, duration_ms, encoder):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"clip")
        extracted.append((view.view_id, start_ms, duration_ms, encoder))

    def fake_grid(clips, destination, encoder):
        assert [label for label, _path in clips] == ["First-Person", "Third-Person"]
        assert encoder == "h264_nvenc"
        destination.write_bytes(b"grid")

    monkeypatch.setattr(pipeline_module, "extract_view_clip", fake_extract)
    monkeypatch.setattr(pipeline_module, "create_grid_video", fake_grid)
    monkeypatch.setattr(
        pipeline_module,
        "probe_video",
        lambda path: VideoInfo(
            path=path,
            duration_ms=1000,
            fps=30.0,
            width=640,
            height=360,
            frame_count=30,
            size_bytes=path.stat().st_size,
        ),
    )

    report = run_media_pipeline_preflight(
        manifest,
        infos,
        tmp_path / "work",
        "h264_nvenc",
        1.0,
    )

    assert report["status"] == "passed"
    assert [item[0] for item in extracted] == ["fp01", "tp01"]
    assert report["aligned_output"]["width"] == 640
    assert report["temporary_artifacts_retained"] is False
    assert not (tmp_path / "work" / "media-pipeline-preflight").exists()


def test_media_pipeline_preflight_retains_smoke_evidence_on_failure(tmp_path, monkeypatch):
    first_path = tmp_path / "first.mp4"
    third_path = tmp_path / "third.mp4"
    manifest = RunManifest(
        experiment_id="media-smoke-failure",
        views=[
            ViewInput(view_id="fp01", role=ViewRole.FIRST_PERSON, video=first_path),
            ViewInput(view_id="tp01", role=ViewRole.THIRD_PERSON, video=third_path),
        ],
    )
    infos = {"fp01": _info(first_path), "tp01": _info(third_path)}

    def fail_extract(_view, _info_value, destination, *_args):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"partial")
        raise RuntimeError("decode failed")

    monkeypatch.setattr(pipeline_module, "extract_view_clip", fail_extract)

    try:
        run_media_pipeline_preflight(
            manifest,
            infos,
            tmp_path / "work",
            "libx264",
            1.0,
        )
    except RuntimeError as exc:
        assert "decode failed" in str(exc)
    else:
        raise AssertionError("preflight should fail")

    assert (tmp_path / "work" / "media-pipeline-preflight").is_dir()

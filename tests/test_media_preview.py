import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from visioncortex import api, media_preview


def test_real_video_poster_cache_invalidation_and_archive_boundaries(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for real decoder verification")
    root, runtime = tmp_path / "archive", tmp_path / "runtime"
    source = root / "Experiment-Clips/test.mp4"
    source.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10",
                    "-t", "0.4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
    original = source.read_bytes()
    poster = media_preview.cached_video_poster(source, runtime)
    assert poster.read_bytes().startswith(b"\xff\xd8")
    assert poster.is_relative_to(runtime)
    stat = poster.stat()
    assert media_preview.cached_video_poster(source, runtime).stat().st_mtime_ns == stat.st_mtime_ns
    source.touch()
    assert media_preview.cached_video_poster(source, runtime) != poster
    assert source.read_bytes() == original
    assert list(source.parent.iterdir()) == [source]

    monkeypatch.setattr(api, "_settings", lambda: {"storage": {"local_runtime_root": str(runtime)}})
    monkeypatch.setattr(api, "_resolve_archive", lambda _: root)
    monkeypatch.setattr(api, "_resolve_staging_run", lambda _: root)
    client = TestClient(api.app)
    for endpoint, params in [("archive-file", {"archive": "test"}), ("staging-file", {"run_id": "test"})]:
        response = client.get(f"/api/{endpoint}", params={**params, "path": "Experiment-Clips/test.mp4", "poster": "true"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["x-visioncortex-preview"] == "first-decoded-frame"
        assert client.get(f"/api/{endpoint}", params={**params, "path": "Experiment-Clips/test.mp4"}, headers={"Range": "bytes=0-19"}).content == original[:20]
        assert client.get(f"/api/{endpoint}", params={**params, "path": "../outside.mp4", "poster": "true"}).status_code == 404
    assert client.get("/api/archive-file", params={"archive": "test", "path": "Experiment-Clips/test.mp4", "poster": "true", "release": "stale"}).status_code == 409
    source.write_bytes(b"invalid media")
    assert client.get("/api/staging-file", params={"run_id": "test", "path": "Experiment-Clips/test.mp4", "poster": "true"}).status_code == 503


def test_poster_cache_cannot_escape_local_runtime(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"unused")
    runtime, outside = tmp_path / "runtime", tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    try:
        (runtime / "cache").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError, match="local runtime"):
        media_preview.cached_video_poster(source, runtime)
    assert list(outside.iterdir()) == []

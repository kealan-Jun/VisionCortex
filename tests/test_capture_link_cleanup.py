from pathlib import Path
import pytest
from visioncortex.capture_link_cleanup import replace_capture_video_with_link
from visioncortex.device_day_contract import artifact, digest


def fixture(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    source = root / "rgb.mp4"
    source.write_bytes(b"owned-test-source")
    archive = root / "archive"
    meta = archive / "MetaVideo"
    meta.mkdir(parents=True)
    target = meta / "Video.mp4"
    target.write_bytes(source.read_bytes())
    ref = artifact(archive, target)
    stat = source.stat()
    retention = {
        "status": "completed",
        "capture_root": str(root),
        "recording": {"recording_id": "test", "video_path": str(source)},
        "sources": [
            {
                "kind": "video",
                "original_path": str(source),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": ref["sha256"],
                "retained": ref,
            }
        ],
        "artifacts": [ref],
    }
    vision = {
        "status": "completed",
        "retention_digest": digest(retention),
        "artifacts": [ref],
    }
    return source, archive, target, retention, vision


def test_atomic_link_and_idempotence(tmp_path):
    source, archive, target, r, v = fixture(tmp_path)
    result = replace_capture_video_with_link(archive, r, v, enabled=True)
    assert result["replaced"] and source.is_symlink()
    assert source.read_bytes() == target.read_bytes()
    assert not Path(result["target"]).is_absolute()
    assert (
        replace_capture_video_with_link(archive, r, v, enabled=True)["status"]
        == "already_linked"
    )


def test_unsupported_link_never_deletes_source(tmp_path, monkeypatch):
    source, archive, target, r, v = fixture(tmp_path)

    def denied(*args):
        raise OSError("native links unavailable")

    monkeypatch.setattr("visioncortex.capture_link_cleanup.os.symlink", denied)
    with pytest.raises(OSError):
        replace_capture_video_with_link(archive, r, v, enabled=True)
    assert (
        source.is_file()
        and not source.is_symlink()
        and source.read_bytes() == target.read_bytes()
    )


def test_changed_source_and_unfinished_processing_fail_closed(tmp_path):
    source, archive, target, r, v = fixture(tmp_path)
    with pytest.raises(ValueError):
        replace_capture_video_with_link(
            archive, r, v | {"status": "running"}, enabled=True
        )
    source.write_bytes(b"changed")
    with pytest.raises(ValueError):
        replace_capture_video_with_link(archive, r, v, enabled=True)
    assert source.read_bytes() == b"changed"
    assert not replace_capture_video_with_link(archive, r, v)["replaced"]

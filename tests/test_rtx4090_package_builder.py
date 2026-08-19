from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "tools" / "build_rtx4090_complete_package.py"
    spec = importlib.util.spec_from_file_location("rtx4090_package_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_round_trip_and_package_escape_rejection(tmp_path: Path):
    builder = _load_builder()
    package = tmp_path / "package"
    (package / "models" / "first_person").mkdir(parents=True)
    (package / "models" / "third_person").mkdir(parents=True)
    (package / "vendor" / "wheelhouse").mkdir(parents=True)
    (package / "models" / "first_person" / "best.pt").write_bytes(b"fp")
    (package / "models" / "third_person" / "best.pt").write_bytes(b"tp")
    (package / "vendor" / "wheelhouse" / "example.whl").write_bytes(b"wheel")

    builder.MODEL_HASHES = {
        "models/first_person/best.pt": builder._sha256(package / "models" / "first_person" / "best.pt"),
        "models/third_person/best.pt": builder._sha256(package / "models" / "third_person" / "best.pt"),
    }
    builder._write_manifest(package, "a" * 40)
    manifest = builder._verify_directory(package)

    assert manifest["source_commit"] == "a" * 40
    assert manifest["file_count"] >= 4
    payload = json.loads((package / "SHA256SUMS.json").read_text(encoding="utf-8"))
    payload["files"][0]["path"] = "../escape"
    (package / "SHA256SUMS.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        builder._verify_directory(package)
    except RuntimeError as error:
        assert "escapes" in str(error)
    else:
        raise AssertionError("unsafe manifest path was accepted")


def test_tensorrt_repack_requires_one_metadata_directory(tmp_path: Path):
    builder = _load_builder()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    try:
        builder._pack_installed_tensorrt(site_packages, wheelhouse, Path("python"))
    except RuntimeError as error:
        assert "Expected one installed tensorrt_cu12 metadata directory" in str(error)
    else:
        raise AssertionError("missing TensorRT installation was accepted")

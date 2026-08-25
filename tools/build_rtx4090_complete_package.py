from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MODEL_HASHES = {
    "models/first_person/best.pt": "a541c59ef8b09158b9b22851dcada6231dbab0f2f1478ae824bcf609851c58ea",
    "models/third_person/best.pt": "ef5a867abf21a8d790eaba054e92d114ae4567c1f867c041ed079cde0a01a36b",
}


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _git_archive(repo: Path, commit: str, destination: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="visioncortex-git-archive-") as temporary:
        archive = Path(temporary) / "source.tar"
        with archive.open("wb") as handle:
            result = subprocess.run(
                ["git", "archive", "--format=tar", commit],
                cwd=repo,
                stdout=handle,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        with tarfile.open(archive, "r") as bundle:
            root = destination.resolve()
            for member in bundle.getmembers():
                resolved = (destination / member.name).resolve()
                if root not in resolved.parents and resolved != root:
                    raise RuntimeError(f"Unsafe git archive entry: {member.name}")
            bundle.extractall(destination, filter="data")


def _build_wheelhouse(repo: Path, package: Path, python: Path) -> None:
    wheelhouse = package / "vendor" / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    lock_file = repo / "deployment" / "rtx4090" / "requirements-lock.txt"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheelhouse),
            "--only-binary=:all:",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu124",
            "--requirement",
            str(lock_file),
        ],
        cwd=repo,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "wheel",
            "--wheel-dir",
            str(wheelhouse),
            "--no-deps",
            str(repo),
        ],
        cwd=repo,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheelhouse),
            "--only-binary=:all:",
            "pip",
            "setuptools>=69",
            "wheel",
        ],
        cwd=repo,
    )


def _pack_installed_tensorrt(site_packages: Path, wheelhouse: Path, python: Path) -> None:
    distributions = {
        "tensorrt_cu12": "tensorrt",
        "tensorrt_cu12_bindings": "tensorrt_bindings",
        "tensorrt_cu12_libs": "tensorrt_libs",
    }
    with tempfile.TemporaryDirectory(prefix="visioncortex-tensorrt-wheels-") as temporary:
        temporary_root = Path(temporary)
        for distribution, package_name in distributions.items():
            matches = sorted(site_packages.glob(f"{distribution}-*.dist-info"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one installed {distribution} metadata directory, found {len(matches)}"
                )
            source_root = temporary_root / distribution
            shutil.copytree(
                site_packages / package_name,
                source_root / package_name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            shutil.copytree(matches[0], source_root / matches[0].name)
            _run(
                [str(python), "-m", "wheel", "pack", str(source_root), "--dest-dir", str(wheelhouse)]
            )


def _write_manifest(package: Path, commit: str) -> Path:
    metadata = {
        "schema_version": 1,
        "product": "VisionCortex",
        "target": "Windows RTX 4090",
        "source_commit": commit,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "portable_assets": {
            "python": "3.12.10",
            "pytorch": "2.6.0+cu124",
            "yolo_class_count": 21,
            "tensorrt_engine_policy": "build_on_target_gpu",
            "input_view_policy": "dynamic_from_nas_index",
        },
    }
    metadata_path = package / "BUNDLE-METADATA.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = package / "SHA256SUMS.json"
    files = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        files.append(
            {
                "path": path.relative_to(package).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _verify_directory(package: Path) -> dict[str, object]:
    manifest = json.loads((package / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (package / entry["path"]).resolve()
        if package.resolve() not in path.parents:
            raise RuntimeError(f"Manifest path escapes package: {entry['path']}")
        if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Manifest verification failed: {entry['path']}")
    for relative, expected in MODEL_HASHES.items():
        if _sha256(package / relative) != expected:
            raise RuntimeError(f"Model identity mismatch: {relative}")
    wheels = list((package / "vendor" / "wheelhouse").glob("*.whl"))
    if not wheels:
        raise RuntimeError("Offline wheelhouse is empty")
    return manifest


def _zip_and_verify(package: Path) -> Path:
    archive = package.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            bundle.write(path, (Path(package.name) / path.relative_to(package)).as_posix())
    with zipfile.ZipFile(archive, "r", allowZip64=True) as bundle:
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC verification failed: {corrupt}")
        manifest_name = f"{package.name}/SHA256SUMS.json"
        if manifest_name not in bundle.namelist():
            raise RuntimeError("ZIP manifest is missing")
    checksum = _sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the complete offline RTX 4090 package.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("D:/"))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--python-installer", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", type=Path, required=True)
    parser.add_argument("--first-onnx", type=Path, required=True)
    parser.add_argument("--third-onnx", type=Path, required=True)
    parser.add_argument("--tensorrt-site-packages", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if _run(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("Repository must be clean so the artifact maps to one immutable commit")
    package = args.output_root.resolve() / f"VisionCortex-RTX4090-Complete-{commit[:12]}"
    if package.exists() or package.with_suffix(".zip").exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {package}")
    package.mkdir(parents=True)

    try:
        _git_archive(repo, commit, package)
        _safe_copy(
            args.python_installer,
            package / "vendor" / "python" / "python-3.12.10-amd64.exe",
        )
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            _safe_copy(args.ffmpeg_bin / name, package / "vendor" / "ffmpeg" / "bin" / name)
        _safe_copy(args.first_onnx, package / "models" / "first_person" / "best.onnx")
        _safe_copy(args.third_onnx, package / "models" / "third_person" / "best.onnx")
        _build_wheelhouse(repo, package, args.python.resolve())
        _pack_installed_tensorrt(
            args.tensorrt_site_packages.resolve(),
            package / "vendor" / "wheelhouse",
            args.python.resolve(),
        )
        _write_manifest(package, commit)
        manifest = _verify_directory(package)
        archive = _zip_and_verify(package)
    except Exception:
        failure = package / "PACKAGE-BUILD-FAILED.txt"
        failure.write_text("Build failed. This directory is not a release artifact.\n", encoding="utf-8")
        raise

    print(
        json.dumps(
            {
                "package": str(package),
                "zip": str(archive),
                "zip_sha256": _sha256(archive),
                "source_commit": commit,
                "files": manifest["file_count"],
                "bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

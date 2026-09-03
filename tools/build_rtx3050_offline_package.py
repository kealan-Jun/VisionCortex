from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = Path("/srv/sentinel-data/VisionCortex3090Ti")
DEFAULT_PYTHON = Path("/home/x1/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu")
DEFAULT_UV = Path("/home/x1/.local/bin/uv")
DEFAULT_FFMPEG_ARCHIVE = Path(
    "/srv/sentinel-data/VisionCortexRTX3050-PackageBuild-20260903/ffmpeg-bundle/"
    "ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
)
FFMPEG_ARCHIVE_SHA256 = (
    "da9012f5a0e8c961f4b5853e4cdb268d4c1cee6c1c43e113a49a1db7bf7be8f0"
)
MODEL_HASHES = {
    "models/ClosedSetYOLO/first_person/best.pt": (
        "a541c59ef8b09158b9b22851dcada6231dbab0f2f1478ae824bcf609851c58ea"
    ),
    "models/ClosedSetYOLO/third_person/best.pt": (
        "ef5a867abf21a8d790eaba054e92d114ae4567c1f867c041ed079cde0a01a36b"
    ),
    "models/Public/yolov8s-worldv2.pt": (
        "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"
    ),
    "models/Public/ViT-B-32.pt": (
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    ),
    "models/Public/grounding-dino-base/model.safetensors": (
        "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
    ),
    "models/Public/sam2.1_hiera_base_plus.pt": (
        "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
    ),
    "models/Public/labpics-semantic-materials.torch": (
        "767bf7838ab89c4134628f189649c092a3af6dea00bfd19c69b93ab0c307a83f"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(
    source: Path, destination: Path, expected: str | None = None
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if expected is not None:
        actual = _sha256(source)
        if actual != expected:
            raise RuntimeError(
                f"Model checksum mismatch: {source}; expected={expected} actual={actual}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _export_app(destination: Path, commit: str) -> None:
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="visioncortex-rtx3050-archive-") as raw:
        archive = Path(raw) / "app.tar"
        with archive.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", commit],
                cwd=ROOT,
                check=True,
                stdout=handle,
            )
        with tarfile.open(archive, "r") as bundle:
            bundle.extractall(destination, filter="data")


def _copy_runtime_models(package: Path, runtime: Path) -> None:
    sources = {
        "models/ClosedSetYOLO/first_person/best.pt": runtime
        / "Models/ClosedSetYOLO/first_person/best.pt",
        "models/ClosedSetYOLO/third_person/best.pt": runtime
        / "Models/ClosedSetYOLO/third_person/best.pt",
        "models/Public/yolov8s-worldv2.pt": runtime / "Engines/yolov8s-worldv2.pt",
        "models/Public/ViT-B-32.pt": Path("/home/x1/.cache/clip/ViT-B-32.pt"),
        "models/Public/grounding-dino-base/model.safetensors": runtime
        / "Engines/grounding-dino-base/model.safetensors",
        "models/Public/sam2.1_hiera_base_plus.pt": runtime
        / "Engines/sam2.1_hiera_base_plus.pt",
        "models/Public/labpics-semantic-materials.torch": runtime
        / "Engines/labpics-semantic-materials.torch",
    }
    for relative, source in sources.items():
        _copy_verified(source, package / relative, MODEL_HASHES[relative])
    grounding = runtime / "Engines/grounding-dino-base"
    for name in (
        "config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        _copy_verified(
            grounding / name,
            package / "models/Public/grounding-dino-base" / name,
        )


def _copy_research_candidates(package: Path, runtime: Path) -> list[dict[str, Any]]:
    registry_path = ROOT / "configs/models/public-apparatus-candidates.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    destination_root = package / "research-candidates"
    for candidate_id, record in sorted(registry.get("candidates", {}).items()):
        artifact = dict(record.get("artifact") or {})
        source = Path(str(artifact.get("path") or ""))
        expected = str(artifact.get("sha256") or "")
        candidate_root = destination_root / candidate_id
        _copy_verified(source, candidate_root / "weights/best.pt", expected)
        (candidate_root / "candidate-record.json").parent.mkdir(
            parents=True, exist_ok=True
        )
        (candidate_root / "candidate-record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        copied_receipts = []
        for key, value in sorted(artifact.items()):
            if key in {"path", "sha256", "size_bytes"} or not isinstance(value, str):
                continue
            receipt = Path(value)
            if not receipt.is_file():
                continue
            safe_name = f"{key}--{receipt.name}"
            _copy_verified(receipt, candidate_root / "receipts" / safe_name)
            copied_receipts.append(safe_name)
        records.append(
            {
                "candidate_id": candidate_id,
                "status": record.get("status"),
                "production_enabled": bool(
                    (record.get("policy") or {}).get("production_enabled", False)
                ),
                "sha256": expected,
                "copied_receipts": copied_receipts,
            }
        )
    return records


def _copy_wheelhouse(source: Path, destination: Path) -> int:
    wheels = sorted(source.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"Offline wheelhouse is empty: {source}")
    destination.mkdir(parents=True)
    for wheel in wheels:
        shutil.copy2(wheel, destination / wheel.name)
    names = [wheel.name.lower().replace("-", "_") for wheel in wheels]
    for required in ("torch_", "tensorrt_cu12_", "clip_", "sam_2_"):
        if not any(name.startswith(required) for name in names):
            raise RuntimeError(f"Offline wheelhouse is missing {required.rstrip('_')}")
    return len(wheels)


def _validate_wheelhouse_resolution(
    wheelhouse: Path,
    uv_binary: Path,
    python_runtime: Path,
) -> int:
    requirements = ROOT / "deployment/rtx3050-ubuntu20/requirements-offline.txt"
    with tempfile.TemporaryDirectory(prefix="visioncortex-wheel-resolution-") as raw:
        venv = Path(raw) / ".venv"
        subprocess.run(
            [
                str(uv_binary),
                "venv",
                "--python",
                str(python_runtime / "bin/python3.12"),
                "--no-python-downloads",
                str(venv),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [
                str(uv_binary),
                "pip",
                "install",
                "--dry-run",
                "--python",
                str(venv / "bin/python"),
                "--offline",
                "--no-index",
                "--no-cache",
                "--find-links",
                str(wheelhouse),
                "--requirement",
                str(requirements),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    match = re.search(r"Resolved (\d+) packages", result.stderr + result.stdout)
    if match is None:
        raise RuntimeError(
            "Offline wheelhouse resolved without an auditable package count"
        )
    return int(match.group(1))


def _copy_ffmpeg_runtime(archive: Path, destination: Path) -> dict[str, Any]:
    if _sha256(archive) != FFMPEG_ARCHIVE_SHA256:
        raise RuntimeError(f"FFmpeg archive checksum mismatch: {archive}")
    with tempfile.TemporaryDirectory(prefix="visioncortex-ffmpeg-") as raw:
        extracted = Path(raw) / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:xz") as bundle:
            bundle.extractall(extracted, filter="data")
        candidates = sorted(extracted.glob("*/bin/ffmpeg"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one FFmpeg runtime root; found {len(candidates)}"
            )
        runtime = candidates[0].parent.parent
        if not (runtime / "bin/ffprobe").is_file():
            raise RuntimeError("Portable FFmpeg archive is missing ffprobe")
        shutil.copytree(runtime, destination)
    for name in ("ffmpeg", "ffprobe"):
        executable = destination / "bin" / name
        executable.chmod(executable.stat().st_mode | 0o111)
    return {
        "archive": archive.name,
        "archive_sha256": FFMPEG_ARCHIVE_SHA256,
        "distribution": "BtbN FFmpeg-Builds Linux x86_64 GPL static 8.1",
    }


def _safe_manifest_path(package: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"Manifest path escapes package: {relative}")
    candidate = (package / Path(*pure.parts)).resolve()
    candidate.relative_to(package.resolve())
    return candidate


def _write_manifests(
    package: Path,
    commit: str,
    wheel_count: int,
    candidates: list[dict[str, Any]],
    ffmpeg: dict[str, Any],
    resolved_distribution_count: int,
) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        relative = path.relative_to(package).as_posix()
        if relative in {"SHA256SUMS", "PACKAGE-MANIFEST.json"}:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": "visioncortex-rtx3050-offline-package/1",
        "source_commit": commit,
        "target": {
            "os": "Ubuntu 20.04 x86_64",
            "kernel_minimum": "5.15.0",
            "gpu": "NVIDIA GeForce RTX 3050",
            "vram_minimum_mib": 5900,
            "driver_minimum": "570.0",
            "system_memory_minimum_gib": 14,
            "swap_minimum_gib": 1.5,
            "install_free_space_minimum_gib": 18,
            "post_install_runtime_free_space_minimum_gib": 8,
        },
        "installation": {
            "root": "/opt/visioncortex-rtx3050",
            "usb_filesystem": "exFAT_or_ext4_required; FAT32_unsupported",
            "wheelhouse_remains_on_usb": True,
            "target_builds_own_tensorrt_engines": True,
            "engine_batch_candidates": [16, 8, 4, 2, 1],
            "engine_selection": (
                "first descending candidate passing repeated TensorRT execution "
                "and the reserved-memory contract"
            ),
        },
        "model_policy": {
            "production_closed_set_weights": "included_verified",
            "public_sidecars": "included_verified",
            "research_candidates": "included_on_usb_not_production_enabled",
            "foreign_tensorrt_engines": "excluded",
        },
        "wheel_count": wheel_count,
        "resolved_distribution_count": resolved_distribution_count,
        "research_candidates": candidates,
        "ffmpeg": ffmpeg,
        "payload_file_count": len(files),
        "payload_bytes": sum(item["bytes"] for item in files),
    }
    manifest_path = package / "PACKAGE-MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files.append(
        {
            "path": "PACKAGE-MANIFEST.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        }
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files),
        encoding="utf-8",
    )
    return manifest


def _verify(package: Path) -> None:
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = _safe_manifest_path(package, relative)
        if not path.is_file():
            raise RuntimeError(f"Package file is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Package checksum mismatch: {relative}; expected={expected} actual={actual}"
            )


def build_package(
    output: Path,
    wheelhouse: Path,
    python_runtime: Path,
    uv_binary: Path,
    runtime: Path,
    ffmpeg_archive: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Package output already exists: {output}")
    commit = _git_commit()
    resolved_distribution_count = _validate_wheelhouse_resolution(
        wheelhouse, uv_binary, python_runtime.resolve()
    )
    output.mkdir(parents=True)
    _export_app(output / "app", commit)
    shutil.copytree(python_runtime.resolve(), output / "vendor/python")
    _copy_verified(uv_binary, output / "vendor/uv")
    wheel_count = _copy_wheelhouse(wheelhouse, output / "vendor/wheelhouse")
    ffmpeg = _copy_ffmpeg_runtime(ffmpeg_archive, output / "vendor/ffmpeg")
    _copy_runtime_models(output, runtime)
    candidates = _copy_research_candidates(output, runtime)
    # Copy every top-level handoff file from the immutable git archive, not
    # from the live worktree.  This keeps source_commit truthful even when the
    # operator has unrelated local edits while assembling a package.
    for source_relative, destination_name, executable in (
        (
            "deployment/rtx3050-ubuntu20/Install-VisionCortex.sh",
            "Install-VisionCortex.sh",
            True,
        ),
        (
            "deployment/rtx3050-ubuntu20/Verify-Package.sh",
            "Verify-Package.sh",
            True,
        ),
        (
            "docs/VisionCortex-RTX3050-离线部署与使用交付手册.md",
            "VisionCortex-RTX3050-交付手册.md",
            False,
        ),
    ):
        _copy_verified(
            output / "app" / source_relative,
            output / destination_name,
        )
        if executable:
            (output / destination_name).chmod(0o755)
    manifest = _write_manifests(
        output,
        commit,
        wheel_count,
        candidates,
        ffmpeg,
        resolved_distribution_count,
    )
    _verify(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the verified RTX 3050 Ubuntu 20.04 USB package."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--python-runtime", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--uv", type=Path, default=DEFAULT_UV)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--ffmpeg-archive", type=Path, default=DEFAULT_FFMPEG_ARCHIVE)
    args = parser.parse_args()
    manifest = build_package(
        args.output.resolve(),
        args.wheelhouse.resolve(),
        args.python_runtime,
        args.uv,
        args.runtime.resolve(),
        args.ffmpeg_archive.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output.resolve()),
                "source_commit": manifest["source_commit"],
                "payload_file_count": manifest["payload_file_count"],
                "payload_bytes": manifest["payload_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

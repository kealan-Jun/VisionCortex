from __future__ import annotations

import hashlib
import uuid
import zipfile
from pathlib import Path
from typing import Any

import httpx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_file(
    name: str,
    path: Path,
    expected_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{name} asset exists with an unexpected hash: {path}; "
            f"expected={expected_sha256} actual={actual}"
        )
    return {
        "name": name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "status": "reused_verified",
    }


def _download_pinned_file(
    name: str,
    path: Path,
    url: str,
    expected_sha256: str,
) -> dict[str, Any]:
    reused = _validated_file(name, path, expected_sha256)
    if reused is not None:
        return reused
    if path.exists():
        raise RuntimeError(f"{name} asset path is not a regular file: {path}")
    if not url.startswith("https://"):
        raise RuntimeError(f"{name} download URL must use HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    total = 0
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=httpx.Timeout(600.0)
    ) as response:
        response.raise_for_status()
        with partial.open("xb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{name} downloaded hash mismatch; retained for audit at {partial}; "
            f"expected={expected_sha256} actual={actual}"
        )
    if path.exists():
        raise RuntimeError(
            f"{name} target appeared during download; refusing to overwrite: {path}"
        )
    partial.replace(path)
    return {
        "name": name,
        "path": str(path),
        "bytes": total,
        "sha256": actual,
        "status": "downloaded_verified",
        "url": url,
    }


def _prepare_grounding_dino(settings: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(settings.get("model_path") or "")).resolve()
    expected = str(settings.get("model_sha256") or "").strip().lower()
    weights = root / "model.safetensors"
    reused = _validated_file("grounding_dino", weights, expected)
    required = [root / "config.json", root / "preprocessor_config.json"]
    if reused is not None and all(path.is_file() for path in required):
        return {
            **reused,
            "model_revision": str(settings.get("model_revision") or ""),
            "repository": str(settings.get("model_repository") or ""),
        }
    if root.exists() and not root.is_dir():
        raise RuntimeError(
            f"Grounding DINO asset path is not a directory: {root}"
        )
    repository = str(settings.get("model_repository") or "").strip()
    revision = str(settings.get("model_revision") or "").strip()
    if not repository or not revision:
        raise RuntimeError(
            "Grounding DINO model_repository and model_revision are required"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=root,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ],
    )
    validated = _validated_file("grounding_dino", weights, expected)
    if validated is None or not all(path.is_file() for path in required):
        raise RuntimeError(
            f"Grounding DINO snapshot is incomplete after download: {root}"
        )
    return {
        **validated,
        "status": "downloaded_verified",
        "model_revision": revision,
        "repository": repository,
    }


def _prepare_liquid_semantic(settings: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(str(settings.get("checkpoint_path") or "")).resolve()
    checkpoint_sha = str(settings.get("checkpoint_sha256") or "").strip().lower()
    reused = _validated_file("labpics_liquid_semantic", checkpoint, checkpoint_sha)
    if reused is not None:
        return {**reused, "source_record": str(settings.get("source_record") or "")}
    archive = Path(str(settings.get("source_archive_path") or "")).resolve()
    archive_record = _download_pinned_file(
        "labpics_liquid_semantic_archive",
        archive,
        str(settings.get("source_archive_url") or ""),
        str(settings.get("source_archive_sha256") or "").strip().lower(),
    )
    member = str(settings.get("source_archive_member") or "").strip()
    if not member or member.startswith("/") or ".." in Path(member).parts:
        raise RuntimeError("LabPics liquid semantic archive member is unsafe")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_name(f".{checkpoint.name}.partial-{uuid.uuid4().hex}")
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(member)
        except KeyError as exc:
            raise RuntimeError(
                f"LabPics liquid semantic member is missing: {member}"
            ) from exc
        with bundle.open(info) as source, partial.open("xb") as destination:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                destination.write(chunk)
                digest.update(chunk)
    actual = digest.hexdigest()
    if actual != checkpoint_sha:
        raise RuntimeError(
            "LabPics liquid semantic extracted checkpoint hash mismatch; "
            f"retained for audit at {partial}; expected={checkpoint_sha} actual={actual}"
        )
    if checkpoint.exists():
        raise RuntimeError(
            "LabPics liquid semantic target appeared during extraction: "
            f"{checkpoint}"
        )
    partial.replace(checkpoint)
    return {
        "name": "labpics_liquid_semantic",
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": actual,
        "status": "extracted_verified",
        "archive": archive_record,
        "source_record": str(settings.get("source_record") or ""),
    }


def prepare_public_model_assets(config: dict[str, Any]) -> dict[str, Any]:
    """Download only pinned public model files required by the active profile."""

    models = config.get("models") or {}
    records: list[dict[str, Any]] = []
    open_vocabulary = models.get("open_vocabulary_key_frame") or {}
    if open_vocabulary.get("enabled"):
        records.append(
            _download_pinned_file(
                "yolo_world",
                Path(str(open_vocabulary.get("model_path") or "")).resolve(),
                str(open_vocabulary.get("model_download_url") or ""),
                str(open_vocabulary.get("model_sha256") or "").strip().lower(),
            )
        )
        records.append(
            _download_pinned_file(
                "clip_text_encoder",
                Path(
                    str(open_vocabulary.get("clip_model_path") or "")
                ).resolve(),
                str(open_vocabulary.get("clip_model_download_url") or ""),
                str(open_vocabulary.get("clip_model_sha256") or "")
                .strip()
                .lower(),
            )
        )
        fallback = dict(open_vocabulary.get("grounding_dino_fallback") or {})
        if fallback.get("enabled"):
            records.append(_prepare_grounding_dino(fallback))

    segmentation = models.get("temporal_participant_segmentation") or {}
    if segmentation.get("enabled"):
        records.append(
            _download_pinned_file(
                "sam2_video_segmentation",
                Path(str(segmentation.get("checkpoint_path") or "")).resolve(),
                str(segmentation.get("checkpoint_download_url") or ""),
                str(segmentation.get("checkpoint_sha256") or "")
                .strip()
                .lower(),
            )
        )
    liquid_semantic = models.get("liquid_semantic_sidecar") or {}
    if liquid_semantic.get("enabled"):
        records.append(_prepare_liquid_semantic(liquid_semantic))
    return {
        "schema_version": "visioncortex-public-model-assets/1",
        "status": "completed",
        "network_used": any(
            item.get("status") == "downloaded_verified" for item in records
        ),
        "asset_count": len(records),
        "assets": records,
    }

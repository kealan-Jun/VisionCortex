"""Content-addressed semantic/media cache and immutable materialization policy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ArchiveCacheServices:
    """Named execution ports; supplied explicitly at the composition boundary."""

    _DERIVED_MEDIA_POPULATED_THIS_PROCESS: Any
    _derived_media_cache_enabled: Callable[..., Any]
    _derived_media_cache_reads_enabled: Callable[..., Any]
    _derived_media_fingerprint: Callable[..., Any]
    _execution_cache_reads_enabled: Callable[..., Any]
    _generate_detached_media: Callable[..., Any]
    _json_default: Callable[..., Any]
    _link_or_copy_immutable: Callable[..., Any]
    _safe_slug: Callable[..., Any]
    _sha256_file: Callable[..., Any]
    select_video_encoder: Callable[..., Any]
    vision_request_identity: Callable[..., Any]
    write_json: Callable[..., Any]


_DERIVED_MEDIA_POPULATED_THIS_PROCESS: set[str] = set()


def _semantic_fingerprint(
    kind: str,
    config: dict[str, Any],
    prompt: str,
    evidence: dict[str, Any],
    images: Sequence[tuple[str, Path]],
    *,
    services: ArchiveCacheServices,
) -> str:
    """Fingerprint exactly the evidence that can change an MLLM answer.

    Run ids, cache identities and archive folder names are deliberately absent.
    A failed run can therefore resume on the same machine without paying for an
    identical request, while any boundary, CV event, prompt, model or image
    change forces a new call.
    """

    digest = hashlib.sha256()
    header = {
        "schema": "visioncortex-semantic-cache/1",
        "request_policy_sha256": services.vision_request_identity(config["mllm"]),
        "kind": kind,
        "model": str(config["mllm"]["model"]),
        "base_url": str(config["mllm"].get("base_url") or ""),
        "response_language": str(config["mllm"].get("response_language") or ""),
        # Namespace separates cold benchmark campaigns without changing the
        # semantic evidence contract. Cold and hot executions of one campaign
        # deliberately use the same value.
        "cache_namespace": config.get("project", {}).get("cache_namespace"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "evidence": evidence,
        "image_order": [label for label, _ in images],
    }
    digest.update(
        json.dumps(
            header, ensure_ascii=False, sort_keys=True, default=services._json_default
        ).encode("utf-8")
    )
    for label, image_path in images:
        digest.update(b"\0label\0")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0image\0")
        digest.update(image_path.read_bytes())
    return digest.hexdigest()


def _semantic_cache_path(
    config: dict[str, Any],
    kind: str,
    fingerprint: str,
    *,
    services: ArchiveCacheServices,
) -> Path:
    return (
        Path(str(config["storage"]["local_cache_root"]))
        / "semantic-results-v1"
        / services._safe_slug(str(config["mllm"]["model"]))
        / kind
        / f"{fingerprint}.json"
    )


def _semantic_cache_reads_enabled(config: dict[str, Any]) -> bool:
    """Return whether this execution may reuse a completed Ark response.

    A cold audit still writes its completed answer so a paired hot replay can
    prove reuse. ``semantic_cache_mode`` may independently force fresh Ark
    calls while verified CV and derived-media artifacts are reused after an
    infrastructure interruption. It only suppresses reads; no entry is deleted.
    """

    project = config.get("project", {})
    mode = project.get("semantic_cache_mode", project.get("cache_mode", "reuse"))
    return str(mode or "reuse") != "cold"


def _execution_cache_reads_enabled(config: dict[str, Any]) -> bool:
    """Return whether immutable CV/media artifacts may be reused."""

    return str(config.get("project", {}).get("cache_mode") or "reuse") != "cold"


def _read_semantic_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("status") != "completed"
        or cached.get("input_fingerprint") != fingerprint
        or cached.get("semantic_cache_schema") != "visioncortex-semantic-cache/1"
    ):
        return None
    cached["cache_reused"] = True
    return cached


def _write_semantic_cache(
    path: Path,
    fingerprint: str,
    result: dict[str, Any],
    *,
    services: ArchiveCacheServices,
) -> dict[str, Any]:
    persisted = dict(result)
    persisted["input_fingerprint"] = fingerprint
    persisted["semantic_cache_schema"] = "visioncortex-semantic-cache/1"
    persisted["cache_reused"] = False
    services.write_json(path, persisted)
    return persisted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _derived_media_cache_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("performance", {}).get("derived_media_cache_enabled", False))


def _derived_media_cache_reads_enabled(
    config: dict[str, Any], *, services: ArchiveCacheServices
) -> bool:
    return services._derived_media_cache_enabled(
        config
    ) and services._execution_cache_reads_enabled(config)


def _derived_media_fingerprint(
    kind: str,
    config: dict[str, Any],
    request: dict[str, Any],
    inputs: Sequence[Path],
    *,
    content_address_inputs: bool = False,
    services: ArchiveCacheServices,
) -> str:
    input_receipts = []
    for source in inputs:
        stat = source.stat()
        receipt: dict[str, Any] = {
            "size_bytes": int(stat.st_size),
        }
        if content_address_inputs:
            receipt["sha256"] = services._sha256_file(source)
        else:
            receipt.update(
                {
                    "path": os.path.abspath(str(source)),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        input_receipts.append(receipt)
    preferred_encoder = str(
        config.get("performance", {}).get("ffmpeg_video_encoder") or "h264_nvenc"
    )
    payload = {
        "schema": "visioncortex-derived-media-cache/1",
        "kind": kind,
        "cache_namespace": config.get("project", {}).get("cache_namespace"),
        "selected_encoder": services.select_video_encoder(preferred_encoder),
        "request": request,
        "inputs": input_receipts,
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=services._json_default
        ).encode("utf-8")
    ).hexdigest()


def _link_or_copy_immutable(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.cache-{uuid.uuid4().hex[:8]}"
    )
    try:
        try:
            os.link(source, temporary)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            method = "verified_copy"
        os.replace(temporary, destination)
    except Exception:
        raise
    finally:
        # POSIX permits rename/replace to be a no-op when both paths already
        # name the same inode.  That happens during idempotent cache reuse and
        # otherwise leaves our temporary hard-link visible on NAS shares.
        temporary.unlink(missing_ok=True)
    return method


def _generate_detached_media(destination: Path, generator: Callable[[], None]) -> None:
    """Keep earlier cached/historical hardlinks immutable during regeneration."""
    previous = destination.with_name(
        f".{destination.name}.previous-{uuid.uuid4().hex[:8]}"
    )
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, previous)
    try:
        generator()
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError(
                f"Derived media generator produced no file: {destination}"
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        if had_previous:
            os.replace(previous, destination)
        raise
    else:
        previous.unlink(missing_ok=True)


def _materialize_derived_media(
    destination: Path,
    kind: str,
    request: dict[str, Any],
    inputs: Sequence[Path],
    config: dict[str, Any],
    generator: Callable[[], None],
    *,
    content_address_inputs: bool = False,
    services: ArchiveCacheServices,
) -> dict[str, Any]:
    """Reuse one immutable derived video only when its complete identity matches.

    Original source video is never copied into this cache.  The request binds
    source identity, exact time bounds, selected encoder, and transformation
    parameters; the sidecar additionally binds the cached output bytes.  Any
    missing or inconsistent receipt fails closed instead of serving uncertain
    media.
    """

    if not services._derived_media_cache_enabled(config):
        services._generate_detached_media(destination, generator)
        return {
            "cache_enabled": False,
            "cache_reused": False,
            "cache_materialization": "generated",
        }
    fingerprint = services._derived_media_fingerprint(
        kind,
        config,
        request,
        inputs,
        content_address_inputs=content_address_inputs,
    )
    suffix = destination.suffix.lower() or ".bin"
    cache_root = (
        Path(str(config["storage"]["local_cache_root"]))
        / "derived-media-v1"
        / services._safe_slug(kind)
        / fingerprint[:2]
    )
    cached_media = cache_root / f"{fingerprint}{suffix}"
    cached_receipt = cache_root / f"{fingerprint}.json"
    same_process_entry = fingerprint in services._DERIVED_MEDIA_POPULATED_THIS_PROCESS
    if (services._derived_media_cache_reads_enabled(config) or same_process_entry) and (
        cached_media.exists() or cached_receipt.exists()
    ):
        if not cached_media.is_file() or not cached_receipt.is_file():
            raise RuntimeError(
                f"Derived media cache entry is incomplete: {fingerprint}"
            )
        receipt = json.loads(cached_receipt.read_text(encoding="utf-8-sig"))
        expected_sha = str(receipt.get("output_sha256") or "")
        if (
            receipt.get("schema_version") != "visioncortex-derived-media-cache/1"
            or receipt.get("request_fingerprint") != fingerprint
            or int(receipt.get("output_bytes") or -1) != cached_media.stat().st_size
            or not expected_sha
            or services._sha256_file(cached_media) != expected_sha
        ):
            raise RuntimeError(
                f"Derived media cache verification failed: {fingerprint}"
            )
        method = services._link_or_copy_immutable(cached_media, destination)
        if services._sha256_file(destination) != expected_sha:
            raise RuntimeError(
                f"Derived media cache materialization changed bytes: {fingerprint}"
            )
        return {
            "cache_enabled": True,
            "cache_reused": True,
            "cache_reuse_scope": (
                "same_process_idempotent"
                if same_process_entry
                else "persistent_verified"
            ),
            "cache_materialization": method,
            "cache_request_fingerprint": fingerprint,
            "cache_output_sha256": expected_sha,
        }

    services._generate_detached_media(destination, generator)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Derived media generator produced no file: {destination}")
    output_sha = services._sha256_file(destination)
    cache_root.mkdir(parents=True, exist_ok=True)
    if cached_media.exists() or cached_receipt.exists():
        if not cached_media.is_file() or not cached_receipt.is_file():
            raise RuntimeError(
                f"Refusing incomplete derived media cache collision: {fingerprint}"
            )
        receipt = json.loads(cached_receipt.read_text(encoding="utf-8-sig"))
        cached_sha = str(receipt.get("output_sha256") or "")
        if (
            receipt.get("schema_version") != "visioncortex-derived-media-cache/1"
            or receipt.get("request_fingerprint") != fingerprint
            or int(receipt.get("output_bytes") or -1) != cached_media.stat().st_size
            or not cached_sha
            or services._sha256_file(cached_media) != cached_sha
            or destination.stat().st_size != cached_media.stat().st_size
            or output_sha != cached_sha
        ):
            raise RuntimeError(
                "Refusing to overwrite a non-identical derived media cache "
                f"entry: {fingerprint}"
            )
        services._DERIVED_MEDIA_POPULATED_THIS_PROCESS.add(fingerprint)
        return {
            "cache_enabled": True,
            "cache_reused": False,
            "cache_reuse_scope": "generated_verified_idempotent_collision",
            "cache_materialization": "generated",
            "cache_request_fingerprint": fingerprint,
            "cache_output_sha256": output_sha,
            "cache_idempotent_collision_verified": True,
        }
    cache_method = services._link_or_copy_immutable(destination, cached_media)
    services.write_json(
        cached_receipt,
        {
            "schema_version": "visioncortex-derived-media-cache/1",
            "request_fingerprint": fingerprint,
            "kind": kind,
            "output_bytes": destination.stat().st_size,
            "output_sha256": output_sha,
            "cache_population": cache_method,
            "source_copy_bytes": 0,
        },
    )
    services._DERIVED_MEDIA_POPULATED_THIS_PROCESS.add(fingerprint)
    return {
        "cache_enabled": True,
        "cache_reused": False,
        "cache_materialization": "generated",
        "cache_population": cache_method,
        "cache_request_fingerprint": fingerprint,
        "cache_output_sha256": output_sha,
    }

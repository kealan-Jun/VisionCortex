from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
import zipfile
from pathlib import Path
from typing import Any

import httpx


DATASET_RECEIPT_SCHEMA = "visioncortex-public-dataset-receipt/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_local_destination(path: Path) -> Path:
    resolved = path.resolve()
    normalized = resolved.as_posix().casefold()
    forbidden = (
        "/home/x1/桌面/nas",
        "/visioncortexexperimentarchive",
        "/visioncortexexperimentcache",
    )
    if any(marker in normalized for marker in forbidden):
        raise RuntimeError(f"Public datasets require a local non-NAS path: {resolved}")
    return resolved


def load_public_dataset_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "visioncortex-public-dataset-registry/1":
        raise ValueError("Unsupported public dataset registry schema")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Public dataset registry is empty")
    return payload


def _download(path: Path, url: str, expected_sha256: str) -> dict[str, Any]:
    if path.is_file():
        actual = _sha256(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Public dataset hash mismatch: {path}; "
                f"expected={expected_sha256} actual={actual}"
            )
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "status": "reused_verified",
        }
    if path.exists():
        raise RuntimeError(f"Public dataset target is not a regular file: {path}")
    if not url.startswith("https://"):
        raise RuntimeError("Public dataset URL must use HTTPS")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    digest = hashlib.sha256()
    total = 0
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=httpx.Timeout(3600.0)
    ) as response:
        response.raise_for_status()
        with partial.open("xb") as handle:
            for chunk in response.iter_bytes(4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                total += len(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            "Downloaded public dataset hash mismatch; retained at "
            f"{partial}; expected={expected_sha256} actual={actual}"
        )
    if path.exists():
        raise RuntimeError(f"Public dataset target appeared during download: {path}")
    os.replace(partial, path)
    return {
        "path": str(path),
        "bytes": total,
        "sha256": actual,
        "status": "downloaded_verified",
    }


def _safe_extract(
    archive: Path,
    destination: Path,
    *,
    expected_archive_sha256: str,
    maximum_uncompressed_bytes: int,
) -> dict[str, Any]:
    receipt_path = destination / ".visioncortex-extraction-receipt.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("archive_sha256") == expected_archive_sha256
            and payload.get("status") == "completed"
        ):
            return {
                **payload,
                "status": "reused_receipt",
                "content_revalidated": False,
            }
        raise RuntimeError(f"Stale public dataset extraction receipt: {receipt_path}")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(
            f"Public dataset extraction destination is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    file_count = 0
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for info in members:
            member = Path(info.filename)
            unix_mode = info.external_attr >> 16
            if (
                member.is_absolute()
                or ".." in member.parts
                or stat.S_ISLNK(unix_mode)
            ):
                raise RuntimeError(
                    f"Unsafe public dataset ZIP member: {info.filename}"
                )
            total += int(info.file_size)
            if total > maximum_uncompressed_bytes:
                raise RuntimeError("Public dataset exceeds extraction size limit")
        for info in members:
            target = (destination / info.filename).resolve()
            if destination.resolve() not in (target, *target.parents):
                raise RuntimeError(f"ZIP member escapes destination: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(chunk)
            file_count += 1
    payload = {
        "schema_version": "visioncortex-public-dataset-extraction/1",
        "status": "completed",
        "archive": str(archive),
        "archive_sha256": expected_archive_sha256,
        "destination": str(destination),
        "file_count": file_count,
        "uncompressed_bytes": total,
    }
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, receipt_path)
    return payload


def prepare_public_dataset(
    registry_path: Path,
    dataset_id: str,
    destination_root: Path,
    *,
    extract: bool = True,
) -> dict[str, Any]:
    registry = load_public_dataset_registry(registry_path)
    try:
        dataset = dict(registry["datasets"][dataset_id])
    except KeyError as exc:
        raise ValueError(f"Unknown public dataset id: {dataset_id}") from exc
    if dataset.get("automated_acquisition_allowed") is not True:
        raise RuntimeError(
            f"Dataset requires manual terms/license handling: {dataset_id}"
        )
    license_id = str(dataset.get("license") or "").strip()
    source = str(dataset.get("source_record") or "").strip()
    if not license_id or not source.startswith("https://"):
        raise RuntimeError(f"Dataset source or license is incomplete: {dataset_id}")
    root = _require_local_destination(destination_root) / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    artifact = dict(dataset.get("artifact") or {})
    expected = str(artifact.get("sha256") or "").strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RuntimeError(f"Dataset SHA-256 is not pinned: {dataset_id}")
    archive = root / str(artifact.get("filename") or "dataset.zip")
    acquisition = _download(archive, str(artifact.get("url") or ""), expected)
    extraction = None
    if extract:
        extraction = _safe_extract(
            archive,
            root / "extracted",
            expected_archive_sha256=expected,
            maximum_uncompressed_bytes=int(
                artifact.get("maximum_uncompressed_bytes") or 30_000_000_000
            ),
        )
    receipt = {
        "schema_version": DATASET_RECEIPT_SCHEMA,
        "status": "completed",
        "dataset_id": dataset_id,
        "title": dataset.get("title"),
        "license": license_id,
        "source_record": source,
        "scope": dataset.get("scope"),
        "acquisition": acquisition,
        "extraction": extraction,
        "nas_accessed": False,
    }
    receipt_path = root / "dataset-receipt.json"
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt

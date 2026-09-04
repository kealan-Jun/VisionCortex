from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import uuid
import zipfile
from pathlib import Path, PurePosixPath
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
    lexical = path.expanduser().absolute()
    resolved = path.resolve()
    normalized_candidates = (
        lexical.as_posix().casefold(),
        resolved.as_posix().casefold(),
    )
    forbidden = (
        "/home/x1/桌面/nas",
        "/mnt/realityloop-nas",
        "/visioncortexexperimentarchive",
        "/visioncortexexperimentcache",
    )
    if any(
        marker in candidate
        for candidate in normalized_candidates
        for marker in forbidden
    ):
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


def _safe_extract_zip(
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
    if destination.exists():
        raise RuntimeError(
            f"Public dataset extraction destination already exists: {destination}"
        )
    temporary = destination.with_name(
        f".{destination.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    file_count = 0
    total = 0
    normalized: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for info in members:
            member = PurePosixPath(info.filename)
            normalized_name = member.as_posix().rstrip("/")
            unix_mode = info.external_attr >> 16
            if (
                not normalized_name
                or member.is_absolute()
                or ".." in member.parts
                or (len(member.parts[0]) == 2 and member.parts[0][1] == ":")
                or "\\" in info.filename
                or normalized_name in normalized
                or stat.S_ISLNK(unix_mode)
            ):
                raise RuntimeError(
                    f"Unsafe public dataset ZIP member: {info.filename}"
                )
            normalized.add(normalized_name)
            file_type = stat.S_IFMT(unix_mode)
            if file_type and not (
                stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode)
            ):
                raise RuntimeError(
                    f"Unsupported public dataset ZIP entry: {info.filename}"
                )
            total += int(info.file_size)
            if total > maximum_uncompressed_bytes:
                raise RuntimeError("Public dataset exceeds extraction size limit")
        temporary.mkdir(parents=True)
        for info in members:
            target = (temporary / info.filename).resolve()
            if temporary.resolve() not in (target, *target.parents):
                raise RuntimeError(f"ZIP member escapes destination: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(chunk)
            file_count += 1
    extracted_files = 0
    extracted_total = 0
    for item in temporary.rglob("*"):
        mode = item.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError(f"Unsafe extracted ZIP entry type: {item}")
        if stat.S_ISREG(mode):
            extracted_files += 1
            extracted_total += item.stat().st_size
            if extracted_total > maximum_uncompressed_bytes:
                raise RuntimeError("Extracted public dataset exceeds size limit")
    if extracted_files != file_count or extracted_total != total:
        raise RuntimeError("ZIP extracted contents do not match validated listing")
    payload = {
        "schema_version": "visioncortex-public-dataset-extraction/1",
        "status": "completed",
        "archive": str(archive),
        "archive_sha256": expected_archive_sha256,
        "destination": str(destination),
        "file_count": file_count,
        "uncompressed_bytes": total,
    }
    (temporary / receipt_path.name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, destination)
    return payload


def _run_archive_tool(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Public dataset archive tool failed: {command[0]}") from exc


def _validate_rar_listing(
    archive: Path,
    tool: str,
    maximum_uncompressed_bytes: int,
) -> tuple[list[str], int, int]:
    names = _run_archive_tool([tool, "-tf", str(archive)]).stdout.splitlines()
    verbose = _run_archive_tool([tool, "-tvf", str(archive)]).stdout.splitlines()
    if not names or len(names) != len(verbose):
        raise RuntimeError("RAR listing is empty or internally inconsistent")
    normalized: set[str] = set()
    total = 0
    file_count = 0
    for name, row in zip(names, verbose, strict=True):
        member = PurePosixPath(name)
        normalized_name = member.as_posix().rstrip("/")
        if (
            not normalized_name
            or member.is_absolute()
            or ".." in member.parts
            or (len(member.parts[0]) == 2 and member.parts[0][1] == ":")
            or "\\" in name
            or normalized_name in normalized
        ):
            raise RuntimeError(f"Unsafe public dataset RAR member: {name}")
        normalized.add(normalized_name)
        fields = row.split(maxsplit=8)
        if len(fields) < 8 or fields[0][0] not in {"-", "d"}:
            raise RuntimeError(f"Unsupported public dataset RAR entry: {name}")
        try:
            size = int(fields[4])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Cannot validate RAR entry size: {name}") from exc
        if size < 0:
            raise RuntimeError(f"Invalid public dataset RAR entry size: {name}")
        if fields[0][0] == "-":
            file_count += 1
            total += size
            if total > maximum_uncompressed_bytes:
                raise RuntimeError("Public dataset exceeds extraction size limit")
    return names, total, file_count


def _safe_extract_rar(
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
            and payload.get("archive_format") == "rar"
        ):
            return {
                **payload,
                "status": "reused_receipt",
                "content_revalidated": False,
            }
        raise RuntimeError(f"Stale public dataset extraction receipt: {receipt_path}")
    if destination.exists():
        raise RuntimeError(
            f"Public dataset RAR extraction destination already exists: {destination}"
        )
    tool = shutil.which("bsdtar")
    if not tool:
        raise RuntimeError("Safe RAR extraction requires bsdtar/libarchive")
    names, listed_total, listed_file_count = _validate_rar_listing(
        archive, tool, maximum_uncompressed_bytes
    )
    temporary = destination.with_name(
        f".{destination.name}.partial-{uuid.uuid4().hex[:8]}"
    )
    temporary.mkdir(parents=True)
    _run_archive_tool([tool, "-xmf", str(archive), "-C", str(temporary)])
    extracted_files = 0
    extracted_total = 0
    for item in temporary.rglob("*"):
        mode = item.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError(f"Unsafe extracted RAR entry type: {item}")
        if stat.S_ISREG(mode):
            extracted_files += 1
            extracted_total += item.stat().st_size
            if extracted_total > maximum_uncompressed_bytes:
                raise RuntimeError("Extracted public dataset exceeds size limit")
    if extracted_files != listed_file_count:
        raise RuntimeError("RAR extracted file count does not match validated listing")
    if extracted_total != listed_total:
        raise RuntimeError("RAR extracted byte count does not match validated listing")
    payload = {
        "schema_version": "visioncortex-public-dataset-extraction/1",
        "status": "completed",
        "archive_format": "rar",
        "archive": str(archive),
        "archive_sha256": expected_archive_sha256,
        "destination": str(destination),
        "file_count": extracted_files,
        "uncompressed_bytes": extracted_total,
        "extractor": tool,
    }
    (temporary / receipt_path.name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, destination)
    return payload


def _safe_extract(
    archive: Path,
    destination: Path,
    *,
    expected_archive_sha256: str,
    maximum_uncompressed_bytes: int,
    archive_format: str,
) -> dict[str, Any]:
    normalized = archive_format.strip().casefold().lstrip(".")
    if normalized == "zip":
        return _safe_extract_zip(
            archive,
            destination,
            expected_archive_sha256=expected_archive_sha256,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
        )
    if normalized == "rar":
        return _safe_extract_rar(
            archive,
            destination,
            expected_archive_sha256=expected_archive_sha256,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
        )
    raise RuntimeError(f"Unsupported public dataset archive format: {archive_format}")


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
            archive_format=str(artifact.get("format") or archive.suffix),
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

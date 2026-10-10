from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml


MIGRATION_RECEIPT_SCHEMA = "visioncortex-source-path-migration/1"
RESOLUTION_RECEIPT_SCHEMA = "visioncortex-source-path-resolution/1"

_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_HASH_CACHE_LOCK = threading.Lock()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file_uncached(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (os.path.abspath(str(path)), int(stat.st_size), int(stat.st_mtime_ns))
    with _HASH_CACHE_LOCK:
        cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    result = _sha256_file_uncached(path)
    with _HASH_CACHE_LOCK:
        _HASH_CACHE[key] = result
    return result


def mapping_digest(entries: Sequence[dict[str, Any]]) -> str:
    canonical = [
        {
            "old_path": str(item["old_path"]),
            "new_path": str(item["new_path"]),
            "relative_path": str(item["relative_path"]),
            "size_bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for item in sorted(entries, key=lambda item: str(item["old_path"]))
    ]
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} is not a SHA-256 hex digest")
    return normalized


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def validate_migration_receipt(payload: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    if payload.get("schema_version") != MIGRATION_RECEIPT_SCHEMA:
        raise ValueError(f"Unsupported source path migration receipt: {receipt_path}")
    if payload.get("status") != "verified":
        raise ValueError(f"Source path migration receipt is not verified: {receipt_path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Source path migration receipt has no entries: {receipt_path}")
    old_root = Path(str(payload.get("source_root_before") or ""))
    new_root = Path(str(payload.get("source_root_after") or ""))
    if not old_root.is_absolute() or not new_root.is_absolute():
        raise ValueError(f"Migration roots must be absolute: {receipt_path}")
    seen: set[str] = set()
    normalized_entries: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid migration entry {index}: {receipt_path}")
        old_path = Path(str(raw.get("old_path") or ""))
        new_path = Path(str(raw.get("new_path") or ""))
        relative = Path(str(raw.get("relative_path") or ""))
        if not old_path.is_absolute() or not new_path.is_absolute() or relative.is_absolute():
            raise ValueError(f"Migration entry paths are not canonical at {index}: {receipt_path}")
        if not _within(old_path, old_root) or not _within(new_path, new_root):
            raise ValueError(f"Migration entry escapes a declared root at {index}: {receipt_path}")
        if old_path != old_root / relative or new_path != new_root / relative:
            raise ValueError(f"Migration relative path mismatch at {index}: {receipt_path}")
        old_key = os.path.normcase(os.path.abspath(str(old_path)))
        if old_key in seen:
            raise ValueError(f"Duplicate source path mapping: {old_path}")
        seen.add(old_key)
        normalized_entries.append(
            {
                "old_path": str(old_path),
                "new_path": str(new_path),
                "relative_path": relative.as_posix(),
                "size_bytes": int(raw["size_bytes"]),
                "sha256": _require_sha256(raw.get("sha256"), f"entry {index} sha256"),
            }
        )
    expected_digest = _require_sha256(
        payload.get("mapping_digest_sha256"), "mapping_digest_sha256"
    )
    actual_digest = mapping_digest(normalized_entries)
    if actual_digest != expected_digest:
        raise ValueError(
            f"Source path migration mapping digest mismatch: {receipt_path}"
        )
    declared_count = int(payload.get("file_count", len(normalized_entries)))
    declared_bytes = int(
        payload.get("total_bytes", sum(item["size_bytes"] for item in normalized_entries))
    )
    if declared_count != len(normalized_entries):
        raise ValueError(f"Source path migration file count mismatch: {receipt_path}")
    if declared_bytes != sum(item["size_bytes"] for item in normalized_entries):
        raise ValueError(f"Source path migration byte count mismatch: {receipt_path}")
    return {
        **payload,
        "entries": normalized_entries,
        "mapping_digest_sha256": actual_digest,
        "file_count": declared_count,
        "total_bytes": declared_bytes,
    }


def load_migration_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read source path migration receipt: {receipt_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Source path migration receipt must contain an object: {receipt_path}")
    validated = validate_migration_receipt(payload, receipt_path)
    validated["receipt_path"] = str(receipt_path)
    validated["receipt_sha256"] = sha256_file(receipt_path)
    return validated


class SourcePathResolver:
    """Resolve only exact, integrity-verified historical path migrations."""

    def __init__(self, receipt_paths: Iterable[str | Path] = ()) -> None:
        self.receipts = [load_migration_receipt(Path(path)) for path in receipt_paths]
        self._entries: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.resolutions: list[dict[str, Any]] = []
        for receipt in self.receipts:
            for entry in receipt["entries"]:
                key = os.path.normcase(os.path.abspath(str(entry["old_path"])))
                previous = self._entries.get(key)
                if previous is not None and previous[0] != entry:
                    raise ValueError(f"Conflicting source path migrations: {entry['old_path']}")
                self._entries[key] = (entry, receipt)

    def resolve(self, path: Path) -> Path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        key = os.path.normcase(os.path.abspath(str(candidate)))
        match = self._entries.get(key)
        if match is None:
            return candidate
        entry, receipt = match
        target = Path(entry["new_path"])
        if not target.is_file():
            raise FileNotFoundError(
                f"Mapped source file is missing: {candidate} -> {target}"
            )
        stat = target.stat()
        expected_size = int(entry["size_bytes"])
        if stat.st_size != expected_size:
            raise ValueError(
                f"Mapped source size mismatch: {candidate} -> {target} "
                f"({stat.st_size} != {expected_size})"
            )
        expected_hash = str(entry["sha256"])
        # Re-read the complete mapped file on every resolution. A stat cache is
        # insufficient for an identity gate because size and mtime can be
        # preserved while bytes change.
        actual_hash = _sha256_file_uncached(target)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Mapped source SHA-256 mismatch: {candidate} -> {target}"
            )
        self.resolutions.append(
            {
                "requested_path": str(candidate),
                "resolved_path": str(target),
                "size_bytes": expected_size,
                "sha256": actual_hash,
                "migration_receipt": receipt["receipt_path"],
                "migration_receipt_sha256": receipt["receipt_sha256"],
                "mapping_digest_sha256": receipt["mapping_digest_sha256"],
                "status": "verified",
            }
        )
        return target

    def resolution_receipt(self, *, subject: str | None = None) -> dict[str, Any]:
        records = sorted(
            {json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in self.resolutions}.values(),
            key=lambda item: (item["requested_path"], item["resolved_path"]),
        )
        digest = hashlib.sha256(_canonical_json(records)).hexdigest()
        return {
            "schema_version": RESOLUTION_RECEIPT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": subject,
            "status": "verified" if records else "not_needed",
            "resolved_path_count": len(records),
            "records_digest_sha256": digest,
            "records": records,
            "policy": {
                "original_receipts_modified": False,
                "resolution": "exact_old_path_match_only",
                "target_validation": "regular_file_size_and_sha256",
            },
        }


def create_migration_receipt(
    source_root_before: Path,
    source_root_after: Path,
    *,
    workers: int = 2,
    progress: Any | None = None,
) -> dict[str, Any]:
    old_root = Path(os.path.abspath(str(source_root_before)))
    new_root = Path(os.path.abspath(str(source_root_after)))
    if not new_root.is_dir():
        raise FileNotFoundError(f"Migrated source root does not exist: {new_root}")
    files = sorted(path for path in new_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Migrated source root contains no files: {new_root}")
    callback = progress or (lambda _done, _total, _path: None)
    hashed: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(files)))) as executor:
        futures = {executor.submit(sha256_file, path): path for path in files}
        done = 0
        for future in as_completed(futures):
            path = futures[future]
            hashed[path] = future.result()
            done += 1
            callback(done, len(files), path)
    entries = []
    for path in files:
        relative = path.relative_to(new_root)
        entries.append(
            {
                "old_path": str(old_root / relative),
                "new_path": str(new_root / relative),
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashed[path],
            }
        )
    return {
        "schema_version": MIGRATION_RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "source_root_before": str(old_root),
        "source_root_after": str(new_root),
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "mapping_digest_sha256": mapping_digest(entries),
        "entries": entries,
        "policy": {
            "source_content_modified": False,
            "historical_receipts_modified": False,
            "mapping": "same_relative_path",
            "file_identity": "size_bytes_and_sha256",
        },
    }


def resolve_manifest_copy(
    manifest_path: Path,
    output_dir: Path,
    receipt_paths: Sequence[Path],
) -> tuple[Path, Path, dict[str, Any]]:
    source = manifest_path.expanduser().resolve()
    original_bytes = source.read_bytes()
    payload = yaml.safe_load(original_bytes) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("views"), list):
        raise ValueError(f"Manifest does not contain views: {source}")
    resolver = SourcePathResolver(receipt_paths)
    for view in payload["views"]:
        if not isinstance(view, dict):
            continue
        for key in ("video", "timestamps_csv"):
            if view.get(key):
                view[key] = str(resolver.resolve(Path(str(view[key]))))
        for segment in view.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            for key in ("video", "timestamps_csv"):
                if segment.get(key):
                    segment[key] = str(resolver.resolve(Path(str(segment[key]))))
    destination_root = output_dir.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    resolved_manifest = destination_root / "manifest.resolved.yaml"
    resolved_manifest.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    receipt = resolver.resolution_receipt(subject=str(source))
    receipt.update(
        {
            "original_manifest": str(source),
            "original_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "resolved_manifest": str(resolved_manifest),
            "resolved_manifest_sha256": sha256_file(resolved_manifest),
        }
    )
    receipt_path = destination_root / "source_path_resolution.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["original_manifest_sha256"]:
        raise RuntimeError(f"Frozen source manifest changed during replay preparation: {source}")
    return resolved_manifest, receipt_path, receipt

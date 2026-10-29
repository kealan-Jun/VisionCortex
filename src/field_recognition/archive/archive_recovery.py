"""Recover missing acknowledged artifacts from byte-verified local evidence only."""

import hashlib
import uuid
from pathlib import Path


def recover_missing(paths, receipts, data_root):
    data_root = Path(data_root).resolve()
    candidates, artifacts = {}, {}

    def source(doc):
        for digest, relative in [
            (doc.get("image_sha256"), "Images/" + str(doc.get("capture_id")) + ".png"),
            (doc.get("source_sha256"), doc.get("original_blob")),
            (doc.get("crop_image_sha256"), "Images/" + str(doc.get("job_id")) + ".png"),
        ]:
            if digest and relative:
                path = (data_root / relative).resolve()
                path.relative_to(data_root)
                candidates.setdefault(digest, set()).add(path)
        for region in doc.get("panel_regions", []):
            if region.get("image_sha256") and region.get("evidence_id"):
                ident = str(uuid.UUID(region["evidence_id"]))
                candidates.setdefault(region["image_sha256"], set()).add(
                    data_root / "Images" / (ident + ".png")
                )
        for item in doc.get("sources", []):
            source(item)

    for receipt in receipts:
        source(receipt["document"])
        for artifact in receipt.get("artifacts", {}).values():
            if isinstance(artifact, dict):
                artifacts.setdefault(artifact["sha256"], artifact)
    restored = []
    for digest, artifact in artifacts.items():
        stored = paths.data["assets"].get(digest, artifact)
        if paths._resolve(stored["path"])[2]:
            continue
        for local in sorted(candidates.get(digest, ())):
            if not local.is_file():
                continue
            before = local.stat()
            if not 0 < before.st_size <= 64 * 1024 * 1024:
                continue
            raw = local.read_bytes()
            after = local.stat()
            if (
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or len(raw) != artifact["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != digest
            ):
                continue
            recovered = paths.asset(artifact["path"], raw, save=False)
            restored.append(
                {"artifact": artifact["path"], "sha256": digest, "restored": recovered["path"]}
            )
            break
    if restored:
        from field_recognition.archive.archive_readable import raw_json
        from field_recognition.archive.archive_store import immutable_write

        paths.save()
        immutable_write(
            paths.root / ".System/Recovery" / (uuid.uuid4().hex + ".json"),
            raw_json(
                {
                    "schema": "field-artifact-recovery/1",
                    "basis": "exact_sha256_and_stable_local_source",
                    "immutable_receipts_rewritten": False,
                    "artifacts": restored,
                }
            ),
        )
    return restored

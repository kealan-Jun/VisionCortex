from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from visioncortex.source_path_migrations import (
    create_migration_receipt,
    load_migration_receipt,
    resolve_manifest_copy,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and apply fail-closed VisionCortex source path migration receipts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--old-root", type=Path, required=True)
    create.add_argument("--new-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--workers", type=int, default=2)
    resolve = commands.add_parser("resolve-manifest")
    resolve.add_argument("--manifest", type=Path, required=True)
    resolve.add_argument("--receipt", type=Path, action="append", required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create":
        payload = create_migration_receipt(
            args.old_root,
            args.new_root,
            workers=args.workers,
        )
        _atomic_json(args.output, payload)
        verified = load_migration_receipt(args.output)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "receipt": str(args.output.resolve()),
                    "receipt_sha256": verified["receipt_sha256"],
                    "mapping_digest_sha256": verified["mapping_digest_sha256"],
                    "file_count": verified["file_count"],
                    "total_bytes": verified["total_bytes"],
                },
                ensure_ascii=False,
            )
        )
        return 0

    manifest, receipt, payload = resolve_manifest_copy(
        args.manifest, args.output_dir, args.receipt
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "resolved_path_count": payload["resolved_path_count"],
                "manifest": str(manifest),
                "resolution_receipt": str(receipt),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())

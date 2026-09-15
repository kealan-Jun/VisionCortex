"""Resumable relocation from the initial device/day draft to its fixed layout.

Only archive-owned outputs are renamed. Capture sources are never touched.
Old recorded JSON stays byte-identical; the journal resolves its old references.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from .device_day import exclusive
from .device_day_contract import STAGES, atomic_json, digest, read_json, safe_child, verify_artifact


def _finish_case_move(temporary, target):
    # Some CIFS clients preserve the old readdir spelling when the same inode is
    # renamed back. A new directory entry avoids that alias, without media copies.
    if temporary.is_dir():
        target.mkdir(exist_ok=True)
        for child in temporary.iterdir():
            destination = target / child.name
            if destination.exists():
                raise ValueError("Case migration child collision")
            child.rename(destination)
        temporary.rmdir()
    else:
        from .device_day_contract import file_hash
        import shutil
        if not target.exists():
            shutil.copy2(temporary, target)
        if file_hash(temporary) != file_hash(target):
            raise ValueError("Case migration file checksum mismatch")
        temporary.unlink()


def _canonical_case(root, relative, journal, journal_path):
    """CIFS can resolve different casing to the same file; rename via a sibling."""
    parent = root
    for name in Path(relative).parts:
        target = parent / name
        candidates = [p for p in parent.iterdir() if p.name.casefold() == name.casefold()]
        if len(candidates) > 1:
            raise ValueError("Case-only migration has distinct colliding entries")
        exact = next((p for p in candidates if p.name == name), None)
        if exact is None:
            if len(candidates) != 1:
                raise ValueError("Case-only migration is ambiguous")
            origin = candidates[0]
            temporary = parent / (".Case-" + digest([str(origin), str(target)])[:16])
            entry = {"from": str(origin.relative_to(root)), "via": str(temporary.relative_to(root)),
                     "to": str(target.relative_to(root)), "completed": False}
            journal.setdefault("case_moves", []).append(entry)
            atomic_json(journal_path, journal)
            origin.rename(temporary)
            _finish_case_move(temporary, target)
            entry["completed"] = True
            atomic_json(journal_path, journal)
        parent = target


def migrate_names(runner, recordings, *, pascal_case=False):
    """Apply the user's final spelling without copying source media again.

    Derived documents are preserved in backend history and rebuilt from verified
    scan/semantic caches. Their original recorded paths resolve via this journal.
    """
    if not recordings:
        return {"status": "not_needed"}
    layout = runner.layout(recordings[0])
    if any(runner.layout(r).name != layout.name for r in recordings):
        raise ValueError("Naming migration must be isolated to one device/day")
    journal_path = layout.receipts / ("pascal-case-migration.json" if pascal_case else "naming-migration.json")
    history = layout.receipts / "history" / ("before-pascal-case" if pascal_case else "before-final-naming")
    old_names = (("Metavideo", "Processedclips", "Multimodalunderstanding", "Laboratorydailyreport", "Comment")
                 if pascal_case else ("1. meta video", "2. processed clips", "3. multimodal understanding", "4. laboratory daily report", "5. comment"))
    with ExitStack() as locks:
        for record in sorted(recordings, key=lambda r: r["recording_id"]):
            for stage in ("all", *STAGES):
                locks.enter_context(exclusive(runner.runtime_root / "locks" / f"{record['recording_id']}.{stage}.lock"))
        if journal_path.is_file():
            journal = read_json(journal_path)
            for entry in journal.get("case_moves", []):
                temporary = safe_child(layout.root, entry["via"])
                if not entry["completed"] and temporary.exists():
                    _finish_case_move(temporary, safe_child(layout.root, entry["to"]))
                    entry["completed"] = True
                    atomic_json(journal_path, journal)
        else:
            moves = []
            for record in recordings:
                receipt = runner._receipt(layout, record, "retention")
                if not receipt.is_file():
                    continue
                for source in read_json(receipt).get("sources", []):
                    target = layout.source_path(record, source["kind"], Path(source["original_path"]))
                    if source["retained"]["path"] != layout.relative(target):
                        moves.append({"from": source["retained"]["path"], "to": layout.relative(target),
                                      "destination_root": "archive", "artifact": source["retained"], "completed": False})
            for old in old_names[1:4]:
                if (layout.root / old).is_dir():
                    moves.append({"from": old, "to": (history / old).relative_to(runner.backend_root).as_posix(),
                                  "destination_root": "local_cache_root", "completed": False})
            old_comment = layout.root / old_names[4]
            if old_comment.is_dir() and old_comment != layout.comments:
                for source in sorted(old_comment.iterdir()):
                    if source.name in {"comment.jsonl", "protocol.json"}:
                        target = layout.comments / source.name.capitalize()
                        moves.append({"from": layout.relative(source), "to": layout.relative(target),
                                      "destination_root": "archive", "completed": False})
                    else:
                        moves.append({"from": layout.relative(source), "to": (history / "5. comment" / source.name).relative_to(runner.backend_root).as_posix(),
                                      "destination_root": "local_cache_root", "completed": False})
            journal = {"archive": layout.name, "status": "running", "moves": moves,
                       "capture_sources_changed": False, "new_model_invocation": False,
                       "reference_resolution": "historical from prefixes resolve through the preserved to mapping"}
            atomic_json(journal_path, journal)
        for move in journal["moves"]:
            if move["completed"]:
                continue
            origin = safe_child(layout.root, move["from"])
            destination_root = layout.root if move["destination_root"] == "archive" else runner.backend_root
            target = safe_child(destination_root, move["to"])
            if not origin.exists() and not target.exists():
                # Earlier flat-media migration deliberately kept receipt bytes
                # unchanged. Resolve that recorded relocation before this one.
                legacy_paths = list((layout.receipts / "layout-migrations").glob("*.json"))
                previous_naming = layout.receipts / "naming-migration.json"
                if previous_naming != journal_path and previous_naming.is_file():
                    legacy_paths.append(previous_naming)
                redirects = {}
                for legacy_path in legacy_paths:
                    for previous in read_json(legacy_path).get("moves", []):
                        if previous.get("destination_root") == "archive" and previous.get("completed"):
                            redirects[previous["from"]] = previous["to"]
                resolved_name, visited = move["from"], set()
                while resolved_name in redirects and resolved_name not in visited:
                    visited.add(resolved_name)
                    resolved_name = redirects[resolved_name]
                resolved = safe_child(layout.root, resolved_name)
                if resolved.exists():
                    move["legacy_from"] = move["from"]
                    move["from"] = resolved_name
                    origin = resolved
                    atomic_json(journal_path, journal)
            if origin.exists():
                if target.exists():
                    if destination_root == layout.root and move["from"].casefold() == move["to"].casefold():
                        _canonical_case(layout.root, move["to"], journal, journal_path)
                    else:
                        raise ValueError("Naming migration collision; refusing overwrite")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    origin.rename(target)
            if not target.exists():
                raise ValueError("Naming migration source and destination are missing")
            if "artifact" in move and not verify_artifact(destination_root, move["artifact"] | {"path": move["to"]}):
                raise ValueError("Naming migration media checksum mismatch")
            move["completed"] = True
            atomic_json(journal_path, journal)
        for name in (old_names[0], old_names[4]):
            old = layout.root / name
            if old.is_dir():
                for child in sorted(old.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if child.is_dir() and not any(child.iterdir()):
                        child.rmdir()
                if not any(old.iterdir()):
                    old.rmdir()
        layout.create()
        journal["status"] = "completed"
        atomic_json(journal_path, journal)
        return journal


def migrate_layout(runner, recording):
    layout = runner.layout(recording)
    identifier = recording["recording_id"]
    old_receipts = layout.processed / "receipts" / identifier
    journal_path = layout.receipts / "layout-migrations" / f"{identifier}.json"
    with ExitStack() as locks:
        for stage in ("all", *STAGES):
            locks.enter_context(exclusive(runner.runtime_root / "locks" / f"{identifier}.{stage}.lock"))
        if journal_path.is_file():
            journal = read_json(journal_path)
            if journal["status"] == "completed":
                return journal
        else:
            if not (old_receipts / "retention.json").is_file():
                return {"status": "not_needed", "recording_id": identifier}
            retained = read_json(old_receipts / "retention.json")
            if retained.get("status") != "completed" or retained["recording"]["recording_id"] != identifier:
                raise ValueError("Cannot migrate an unsealed retained source")
            moves = []
            for source in retained["sources"]:
                old = safe_child(layout.root, source["retained"]["path"])
                target = layout.source_path(recording, source["kind"], Path(source["original_path"]))
                moves.append({"from": layout.relative(old), "to": layout.relative(target),
                              "destination_root": "archive", "artifact": source["retained"], "completed": False})
            old_raw = layout.raw / identifier
            if (old_raw / "原片留存.json").is_file():
                moves.append({"from": layout.relative(old_raw / "原片留存.json"),
                              "to": f"device-day-receipts/{layout.name}/{identifier}/history/original-retention.json",
                              "destination_root": "local_cache_root", "completed": False})
            # Move the receipt directory first. Old hashes and references are kept
            # unchanged, and every old prefix can be resolved with this journal.
            moves.insert(0, {"from": layout.relative(old_receipts),
                             "to": f"device-day-receipts/{layout.name}/{identifier}",
                             "destination_root": "local_cache_root", "completed": False})
            journal = {"recording_id": identifier, "archive": layout.name, "status": "running",
                       "capture_sources_changed": False, "moves": moves,
                       "reference_resolution": "historical archive paths use the recorded from/to mapping"}
            atomic_json(journal_path, journal)
        for move in journal["moves"]:
            origin = safe_child(layout.root, move["from"])
            destination_root = layout.root if move["destination_root"] == "archive" else runner.backend_root
            destination = safe_child(destination_root, move["to"])
            if move["completed"]:
                continue
            if origin.exists():
                if destination.exists():
                    raise ValueError("Migration destination collision; no archive artifact was overwritten")
                destination.parent.mkdir(parents=True, exist_ok=True)
                origin.rename(destination)
            if not destination.exists():
                raise ValueError("Migration source and destination are both missing")
            if "artifact" in move and not verify_artifact(destination_root, move["artifact"] | {"path": move["to"]}):
                raise ValueError("Relocated source checksum verification failed")
            move["completed"] = True
            atomic_json(journal_path, journal)
        old_raw = layout.raw / identifier
        if old_raw.is_dir():
            # Empty archive-owned containers only. Unknown files are retained.
            for folder in sorted((p for p in old_raw.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if not any(folder.iterdir()):
                    folder.rmdir()
            if not any(old_raw.iterdir()):
                old_raw.rmdir()
        old_parent = layout.processed / "receipts"
        if old_parent.is_dir() and not any(old_parent.iterdir()):
            old_parent.rmdir()
        journal["status"] = "completed"
        atomic_json(journal_path, journal)
        return journal

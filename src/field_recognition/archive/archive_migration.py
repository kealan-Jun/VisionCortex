"""Single-writer archive upgrade; immutable bytes move, derived views regenerate."""

import hashlib
import json
import sqlite3
import time

from field_recognition.archive.archive_events import event_rows, make_views, prepare
from field_recognition.archive.archive_paths import ArchivePaths, checked, empty_directories
from field_recognition.archive.archive_progress import advance


def history_state(conn, bundles):
    """Expose retained but not yet published versions when current rows lead FIFO."""
    counts = {
        (row["entity"], row["entity_id"]): (row["total"], row["pending"])
        for row in conn.execute(
            "SELECT entity,entity_id,count(*) AS total,sum(archived_at IS NULL) AS pending "
            "FROM archive_outbox GROUP BY entity,entity_id"
        )
    }
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for bundle in bundles:
        keys = set(bundle["entity_keys"])
        group = bundle["group"] or {}
        keys.update(
            ("jobs", ident)
            for ident in group.get("job_ids", []) + group.get("superseded_job_ids", [])
        )
        keys.update(
            ("scans", source["capture_id"])
            for source in group.get("sources", [])
            if source.get("capture_id")
        )
        # New retries may have no archived version yet and therefore cannot be
        # part of prepare(). Discover their IDs without using current values as
        # archived evidence. Counts must not call a still-queued retry complete.
        captures = {ident for entity, ident in keys if entity == "scans"}
        if "jobs" in tables:
            for capture in captures:
                keys.update(
                    ("jobs", row["id"])
                    for row in conn.execute(
                        "SELECT id FROM jobs WHERE json_extract(document,'$.capture_id')=?",
                        (capture,),
                    )
                )
        if group.get("measurement_id") and "experiment_records" in tables:
            keys.update(
                ("experiment_records", row["id"])
                for row in conn.execute(
                    "SELECT id FROM experiment_records WHERE json_extract(document,'$.measurement_id')=?",
                    (group["measurement_id"],),
                )
            )
        total = sum(counts.get(key, (0, 0))[0] for key in keys)
        pending = sum(counts.get(key, (0, 0))[1] for key in keys)
        bundle["archive_history"] = {
            "total_receipts": total,
            "published_receipts": total - pending,
            "pending_receipts": pending,
            "complete": pending == 0,
            "scope": "event_entity_receipt_versions_not_background_integrity_scan",
        }


def publish_events(archive, rows=None, *, sequences=None):
    """Deliver changed business files before scanning historical NAS content."""
    with archive.core["db"]() as conn:
        pending = {
            row["seq"]
            for row in conn.execute(
                "SELECT seq FROM archive_outbox WHERE archived_at IS NOT NULL "
                "AND published_at IS NOT NULL AND readable_at IS NULL"
            )
        }
        confirmation = conn.execute(
            "SELECT value FROM archive_meta WHERE key='event_confirmation_pending'"
        ).fetchone()
    confirmations = set(json.loads(confirmation[0])) if confirmation else set()
    pending |= confirmations
    if sequences is not None:
        # Event-level retry isolation must also isolate timing echo failures.
        # Other events' durable obligations stay in the metadata unchanged.
        pending &= set(sequences)
    if not pending:
        return
    started = time.monotonic()
    stages, selected, delivered, receipts = {}, [], set(), {}

    def measured(name, action):
        began = time.monotonic()
        try:
            return action()
        finally:
            stages[name] = round((time.monotonic() - began) * 1000, 2)

    def business_views(views):
        return {
            name: raw
            for name, raw in views.items()
            if name != "Readme.html" and "/DailyReport/" not in name
        }

    result = "failed"
    try:
        root = archive._root()
        config = archive.core.get("config")
        paths = measured(
            "path_ledger_load_ms",
            lambda: ArchivePaths(
                root,
                layout=config.archive_layout if config else "events",
                source_roots=config.saved_photo_roots if config else (),
            ),
        )

        def closure():
            with archive.core["db"]() as conn:
                return event_rows(conn, pending, paths)

        rows = measured("event_closure_ms", closure)
        by_sequence = {row["seq"]: row for row in rows}
        bundles, _ = measured("event_prepare_ms", lambda: prepare(rows, paths))
        selected = [b for b in bundles if any(v["sequence"] in pending for v in b["history"])]
        with archive.core["db"]() as conn:
            history_state(conn, selected)

        def verify_receipts():
            for bundle in selected:
                for version in bundle["history"]:
                    seq = version["sequence"]
                    if seq in receipts:
                        continue
                    path = checked(root, paths.name(version["receipt"]))
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != path.stem.split("-", 1)[-1]:
                        raise ValueError("Cannot publish corrupt receipt")
                    receipt = json.loads(raw)
                    row = by_sequence[seq]
                    if (
                        receipt["sequence"],
                        receipt["entity"],
                        receipt["entity_id"],
                        receipt["document"],
                    ) != (seq, row["entity"], row["entity_id"], json.loads(row["document"])):
                        raise ValueError("Cannot publish mismatched receipt")
                    receipts[seq] = receipt

        measured("receipt_validation_ms", verify_receipts)
        if paths.layout == "metadata":
            from field_recognition.archive.archive_recovery import recover_missing

            recover_missing(paths, receipts.values(), archive.core["DATA"])
        # Same operation only: every dependency is byte-validated in this pass.
        placed_assets = {}
        views = measured(
            "asset_placement_and_view_build_ms",
            lambda: make_views(
                selected,
                paths,
                receipts,
                preserve_asset_locations=True,
                placed_assets=placed_assets,
            ),
        )
        measured(
            "event_files_write_ms", lambda: paths.retire_views(business_views(views), partial=True)
        )
        delivered = pending & {v["sequence"] for b in selected for v in b["history"]}
        if delivered:
            timestamp = archive.core["now"]()
            with archive.core["db"]() as conn:
                conn.executemany(
                    "UPDATE archive_outbox SET readable_at=?,readability_basis=? WHERE seq=? AND readable_at IS NULL",
                    [(timestamp, "archive_writer_event_files_published", seq) for seq in delivered],
                )
                # Commit the echo obligation with the confirmation so retries
                # do not depend on whole-history navigation succeeding.
                conn.execute(
                    "INSERT OR REPLACE INTO archive_meta VALUES('event_confirmation_pending',?)",
                    (json.dumps(sorted(confirmations | delivered)),),
                )
                refreshed = [
                    conn.execute(
                        "SELECT * FROM archive_outbox WHERE seq=?", (row["seq"],)
                    ).fetchone()
                    for row in rows
                ]
            keys = {b["key"] for b in selected}
            bundles, _ = measured("confirmation_prepare_ms", lambda: prepare(refreshed, paths))
            selected = [b for b in bundles if b["key"] in keys]
            with archive.core["db"]() as conn:
                history_state(conn, selected)
            views = measured(
                "confirmation_view_build_ms",
                lambda: make_views(
                    selected,
                    paths,
                    receipts,
                    preserve_asset_locations=True,
                    placed_assets=placed_assets,
                ),
            )
            measured(
                "confirmation_files_write_ms",
                lambda: paths.retire_views(business_views(views), partial=True),
            )
            with archive.core["db"]() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO archive_meta VALUES('event_confirmation_pending',?)",
                    (json.dumps(sorted(confirmations - delivered)),),
                )
        result = "published"
        advance("business_files_published", len(selected), len(selected))
    finally:
        measurement = {
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "clock_basis": "process_monotonic",
            "scope": "changed_event_files_and_timing_confirmation",
            "status": result,
            "events": len(selected),
            "receipts": len(delivered),
            "sequences": sorted(pending),
            "recorded_at": archive.core["now"](),
            "durations_ms": stages,
        }
        with archive.lock:
            archive.status["last_event_publication"] = measurement
        # Keep per-pass measurements in local SQLite; NAS telemetry publication
        # must not add another echo to the latency being measured.
        key = (
            "event_publication:" + hashlib.sha256(json.dumps(sorted(pending)).encode()).hexdigest()
        )
        try:
            with archive.core["db"]() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO archive_meta VALUES(?,?)",
                    (key, json.dumps(measurement)),
                )
        except (sqlite3.Error, OSError) as exc:
            # Diagnostic persistence must neither replace the real publication
            # failure nor interrupt a delivery whose confirmation is committed.
            with archive.lock:
                archive.status["event_timing_persistence_error"] = type(exc).__name__
        else:
            with archive.lock:
                archive.status.pop("event_timing_persistence_error", None)


def retire_legacy(root, paths):
    # Recognize the old generated schema before removing a redundant view.
    for path in (root / "Records").glob("**/Record.json"):
        data = json.loads(path.read_text())
        relative = path.relative_to(root).as_posix()
        if (
            data.get("schema") != "field-recognition-record/2"
            or not data.get("derived_view")
            or relative not in paths.data["aliases"]
        ):
            continue
        path.unlink()
        companion = path.with_name("Readme.html")
        if companion.is_file() and "完整记录 JSON" in companion.read_text():
            companion.unlink()
    for path in (root / "Browse").glob("**/Readme.html"):
        text = path.read_text()
        if (
            "<title>" in text
            and ("归档首页" in text or "归档入口" in text)
            and ('id="records"' in text or 'meta http-equiv="refresh"' in text)
        ):
            paths.alias(path.relative_to(root).as_posix(), "Readme.html")
            paths.save()
            path.unlink()
    catalog = root / "Browse/Catalog.json"
    if (
        catalog.is_file()
        and json.loads(catalog.read_text()).get("schema") == "field-recognition-folders/2"
    ):
        paths.alias("Browse/Catalog.json", ".System/Index.json")
        paths.save()
        catalog.unlink()
    for name, marker in [
        ("Index.json", "field-recognition-index/"),
        ("Audit.html", "ARCHIVE"),
        ("00_归档导航.html", "归档入口"),
    ]:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text()
        if marker in text or name == "Audit.html" and "现场识别" in text:
            paths.alias(name, "Readme.html" if name.startswith("00_") else ".System/" + name)
            paths.save()
            path.unlink()
    old = root / "LegacyLinks.json"
    if old.is_file():
        mappings = json.loads(old.read_text())
        if isinstance(mappings, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in mappings.items()
        ):
            for source, target in mappings.items():
                paths.data["aliases"].setdefault(source, target)
            paths.save()
            old.unlink()


def rebuild(archive, rows):
    from field_recognition.archive.archive_catalog import build_index, render_index
    from field_recognition.archive.archive_store import canonical

    root = archive._root()
    config = archive.core.get("config")
    paths = ArchivePaths(
        root,
        layout=config.archive_layout if config else "events",
        source_roots=config.saved_photo_roots if config else (),
    )
    with archive.core["db"]() as conn:
        old_links = conn.execute(
            "SELECT value FROM archive_meta WHERE key='legacy_links'"
        ).fetchone()
    if old_links:
        for source, target in json.loads(old_links[0]).items():
            paths.data["aliases"].setdefault(source, target)
    # Original receipts and prior integrity reports must remain byte-identical.
    for folder in ("Receipts", "Integrity"):
        for source in sorted((root / folder).glob("**/*.json")):
            relative = source.relative_to(root).as_posix()
            paths.move(relative, ".System/" + relative)
    receipts = {}
    cache = getattr(archive, "receipt_cache", {})
    archive.receipt_cache = cache
    for number, row in enumerate(rows, 1):
        path = checked(root, paths.name(row["receipt_path"]))
        receipt = cache.get(path.name)
        if receipt is None:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != path.stem.split("-", 1)[-1]:
                raise ValueError("Cannot migrate corrupt receipt")
            receipt = json.loads(raw)
            cache[path.name] = receipt
        if (receipt["sequence"], receipt["entity"], receipt["entity_id"], receipt["document"]) != (
            row["seq"],
            row["entity"],
            row["entity_id"],
            json.loads(row["document"]),
        ):
            raise ValueError("Cannot migrate mismatched receipt")
        receipts[row["seq"]] = receipt
        advance("checking_receipts", number, len(rows))
    archive.receipt_cache = dict(list(cache.items())[-4096:])
    if paths.layout == "metadata":
        from field_recognition.archive.archive_recovery import recover_missing

        recover_missing(paths, receipts.values(), archive.core["DATA"])
    bundles, _ = prepare(rows, paths)
    with archive.core["db"]() as conn:
        history_state(conn, bundles)
    paths.save()  # Persist stable event directories before moving their photos.
    previous_integrity, _ = archive.integrity.latest()
    known_missing = frozenset(
        issue["path"]
        for issue in (previous_integrity or {}).get("issues", [])
        if issue.get("kind") == "object" and issue.get("code") == "missing"
    )
    views = make_views(bundles, paths, receipts, known_missing_assets=known_missing)
    advance("building_navigation")
    # Retain orphaned historical evidence without placing it among valid measurements.
    for source in (root / "Objects").glob("*/*"):
        if not source.is_file() or len(source.stem) != 64:
            continue
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.stem:
            raise ValueError("Unreferenced archive object is corrupt")
        relative = source.relative_to(root).as_posix()
        stored = paths.data["assets"].get(source.stem)
        target = stored["path"] if stored else ".System/UnassignedAssets/" + source.name
        paths.move(relative, target)
    index = build_index(rows, archive.instance, archive.core["now"](), archive.integrity.snapshot())
    for item in index["items"]:
        for key in ("image", "receipt"):
            if item.get(key):
                item[key] = paths.name(item[key])
    if index["integrity"].get("report_path"):
        index["integrity"]["report_path"] = paths.name(index["integrity"]["report_path"])
    index["archive_base"] = "../"
    views[".System/Index.json"] = canonical(index)
    views[".System/Audit.html"] = render_index(index)
    for source, target in [
        ("Index.json", ".System/Index.json"),
        ("Audit.html", ".System/Audit.html"),
        ("00_归档导航.html", "Readme.html"),
        ("Browse/Readme.html", "Readme.html"),
    ]:
        paths.alias(source, target)
    paths.retire_views(views)
    retire_legacy(root, paths)
    paths.save()
    # Input folders now share the archive root; they belong to the uploader.
    if paths.layout != "metadata":
        empty_directories(root)
    with archive.core["db"]() as conn:
        for key, value in [
            ("readable_files", json.dumps(sorted(views))),
            ("index_count", str(len(rows))),
            ("index_version", archive.index_version),
            ("index_integrity", archive.integrity.snapshot()["report_path"] or ""),
        ]:
            conn.execute("INSERT OR REPLACE INTO archive_meta VALUES(?,?)", (key, value))
    return {
        "events": len(bundles),
        "images": len(paths.data["assets"]),
        "receipts": len(receipts),
        "visible_days": len({b["day"] for b in bundles if b["kind"] != "SourceMaterials"}),
    }

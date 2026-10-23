"""Run the model-free, bounded historical archive format migration.

Supply a private site configuration and a bounded local plan directory. Public
defaults never discover NAS, start a model, or enable this operation.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil
import signal
from threading import Event
import time

from visioncortex.application.archive_unification import (
    ArchiveUnifier,
    RECIPE,
    file_identity,
    read_source_projection,
)
from visioncortex.application.clip_delivery import VERSION
from visioncortex.build_identity import identity
from visioncortex.config import load_config
from visioncortex.device_day import exclusive
from visioncortex.device_day_contract import (
    atomic_json,
    digest,
    read_json,
    safe_child,
    validate_archive_name,
)
from visioncortex.device_day_io import slot
from visioncortex.device_day_recovery_worker import verify_storage
from visioncortex.device_day_write_guard import storage_guard, guarded_mkdir
from visioncortex.receipt_projection import _json_events
from visioncortex.runtime_control import check_yield, execution_context, resource_slot


def visible(parent):
    return (
        {p.name for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")}
        if parent.is_dir()
        else set()
    )


def run(config, archive, plan_root, stop, *, apply, public_video_inventory=None):
    from visioncortex.device_day_contract import DeviceDayLayout
    from visioncortex.device_day_timeline import day_bounds
    from visioncortex.application import clip_delivery
    from visioncortex.application.archive_documents import retire_day_html
    from visioncortex.device_day_content_paths import migrate_display_layout

    start, _ = day_bounds(archive[:10])
    layout = DeviceDayLayout(
        Path(config["storage"]["archive_root"]),
        archive[11:],
        start,
        read_only_originals=True,
    )
    runtime = Path(config["storage"]["local_runtime_root"]) / "device-day"
    with (
        storage_guard(config),
        execution_context(
            job_id="unify-archive-" + archive,
            source="retention_recovery",
            priority=20,
            background_fair=True,
            stop=stop,
            yield_signal=stop,
        ),
        resource_slot(config, "cpu"),
        slot(config),
        ExitStack() as locks,
    ):
        locks.enter_context(exclusive(runtime / "locks" / f"{archive}.index.lock"))
        plan_path = plan_root / (archive + ".json")
        index = read_json(plan_path) if plan_path.is_file() else None
        source = layout.index
        before = None
        if source.is_file():
            before = file_identity(source)
            if index is None or index.get("migration_source_identity") != before:
                index = read_source_projection(
                    source,
                    events=lambda f: _json_events(f, streaming=True),
                    checkpoint=lambda: check_yield(checkpoint=False),
                )
                index["migration_source_identity"] = before
                atomic_json(plan_path, index)
        else:
            index = {"archive": archive, "recordings": [], "segments": []}
        if index["archive"] != archive:
            raise ValueError(
                "Canonical archive identity differs from requested device day"
            )
        for record in sorted(index["recordings"], key=lambda r: r["recording_id"]):
            for stage in (
                "all",
                "retention",
                "vision",
                "stt",
                "understanding",
                "report",
            ):
                locks.enter_context(
                    exclusive(
                        runtime / "locks" / f"{record['recording_id']}.{stage}.lock"
                    )
                )
        for row in index["segments"]:
            try:
                from visioncortex.device_day_content_paths import clip_name

                clip_name(row)
            except ValueError:
                row["historical_json_path"] = row.get("json_path")
                row["json_path"] = None
        unit_catalog = clip_delivery.original_units(layout, index, [], directory_bindings=(
            config.get('collection_ingest', {}).get('directory_camera_bindings') or ()))
        a, b = visible(layout.processed / "Clips"), visible(layout.understanding)
        headers = []
        if a == b and len(a) == len(unit_catalog):
            for name in a:
                p = layout.processed / "Clips" / name / "Index.json"
                q = layout.understanding / name / "Understanding.json"
                if not p.is_file() or not q.is_file():
                    break
                h = read_json(p)
                m = read_json(q)
                if (h.get("schema_version") != VERSION
                        or m.get("schema_version") != VERSION
                        or m.get("source_unit") != h.get("source_unit")
                        or h.get("source_folder") != name
                        or m.get("source_folder") != name):
                    break
                if h.get("source_unit") not in unit_catalog:
                    break
                if not re.fullmatch(
                    r"(?:\d{2}-\d{2}-\d{2}|TimePending)_(?:\d{2}-\d{2}-\d{2}|TimePending)_(?:ExperimentActivity|ValuableNoExperimentActivity|NoValueNoExperimentActivity|NoExperimentActivity)_[a-f0-9]{8}",
                    name,
                ):
                    break
                u = unit_catalog[h["source_unit"]]
                if (h.get("original_start_us"), h.get("original_end_us")) != (
                    u["start_us"],
                    u["end_us"],
                ):
                    break
                if h.get("intervals") != m.get("intervals") or any(
                    not safe_child(layout.root, row[field]).is_file()
                    for row in h.get("intervals", [])
                    for field in ("metadata", "content")
                ):
                    break
                headers.append(h)
        current = (
            len(headers) == len(unit_catalog)
            and a == b
            and len(a) == len(unit_catalog)
            and {h["source_unit"] for h in headers} == set(unit_catalog)
        )
        checkpoint_path = plan_root / (archive + ".discovery.json")
        if checkpoint_path.is_file() and not read_json(checkpoint_path).get(
            "completed"
        ):
            current = False
        acceptance = layout.processed / ".Evidence/ArchiveUnification/Acceptance.json"
        if not acceptance.is_file() or read_json(acceptance).get("recipe") != RECIPE:
            # A grouped view alone does not prove all saved older generations
            # were included. Recover their native journals on the first pass.
            current = False
        inventory_digest = (
            digest(public_video_inventory)
            if public_video_inventory is not None
            else None
        )
        if inventory_digest and (
            not acceptance.is_file()
            or read_json(acceptance).get("public_video_inventory_digest")
            != inventory_digest
        ):
            current = False
        result = {
            "archive": archive,
            "recipe": RECIPE,
            "originals": len(unit_catalog),
            "before_clips": len(a),
            "before_understanding": len(b),
            "model_invoked": False,
            "canonical_index_changed": False,
            "original_media_changed": False,
            "already_grouped": current,
            "public_video_inventory_digest": inventory_digest,
        }
        if not apply:
            return result | {"status": "audited"}
        for directory in (
            layout.processed,
            layout.processed / "Clips",
            layout.understanding,
            layout.comments,
            layout.reports,
        ):
            guarded_mkdir(directory)
        if not current:
            unifier = ArchiveUnifier(
                config, archive, index, public_video_inventory=public_video_inventory
            )
            known = {r["recording_id"] for r in index["recordings"]}
            if checkpoint_path.is_file() and not read_json(checkpoint_path).get(
                "completed"
            ):
                unifier.restore(checkpoint_path)
            result["discovery"] = unifier.discover()
            # Earlier presentation generations have their own durable stage
            # identities; acquire their locks as well before moving a byte.
            for rid in sorted(set(unifier.records) - known):
                for stage in (
                    "all",
                    "retention",
                    "vision",
                    "stt",
                    "understanding",
                    "report",
                ):
                    locks.enter_context(
                        exclusive(runtime / "locks" / f"{rid}.{stage}.lock")
                    )
            unifier.checkpoint(checkpoint_path)
            for batch in range(256):
                check_yield(checkpoint=False)
                progress = unifier.publish()
                result["batches"] = batch + 1
                atomic_json(
                    plan_root / (archive + ".progress.json"),
                    result | {"publication": progress, "status": "publishing"},
                )
                if not progress["pending"]:
                    break
            else:
                raise ValueError(
                    "Historical video publication exceeded its bounded batch count"
                )
            unit_catalog = unifier.units
        # Existing migration owners preserve technical containers, old day
        # summaries, and transcript internals with verified copies and aliases.
        migrations = []
        for _ in range(256):
            check_yield(checkpoint=False)
            moved = migrate_display_layout(None, layout, bounded_batch=True)
            migrations.append(
                {
                    "moves": len(moved["moved"]),
                    "files": moved["files"],
                    "bytes": moved["verified_bytes"],
                }
            )
            if not moved["pending"]:
                break
            if not moved["moved"]:
                raise ValueError(
                    "Supporting evidence migration cannot progress within its cap"
                )
        else:
            raise ValueError("Supporting evidence exceeded the bounded batch count")
        retire_day_html(layout)
        from visioncortex.application.archive_unification import refresh_group_index

        result["readable_index"] = refresh_group_index(layout)
        after_a, after_b = (
            visible(layout.processed / "Clips"),
            visible(layout.understanding),
        )
        expected = {u["name"] for u in unit_catalog.values()} if not current else a
        result.update(
            status="completed",
            after_clips=len(after_a),
            after_understanding=len(after_b),
            sets_match=after_a == after_b == expected,
            supporting_migrations=migrations,
            canonical_index_changed=source.is_file()
            and file_identity(source) != before,
        )
        if not result["sets_match"] or result["canonical_index_changed"]:
            raise ValueError(
                "Post-migration verification failed: "
                + json.dumps(
                    {
                        "only_clips": sorted(after_a - after_b),
                        "only_understanding": sorted(after_b - after_a),
                        "missing": sorted(expected - after_a),
                        "extra": sorted(after_a - expected),
                        "canonical_changed": result["canonical_index_changed"],
                    },
                    ensure_ascii=False,
                )
            )
        for name in expected:
            h = read_json(layout.processed / "Clips" / name / "Index.json")
            m = read_json(layout.understanding / name / "Understanding.json")
            if (
                h["source_unit"] != m["source_unit"]
                or h["source_folder"] != name
                or m["source_folder"] != name
            ):
                raise ValueError("Paired collection ownership differs")
            if h.get("intervals") != m.get("intervals"):
                raise ValueError("Paired interval indexes differ")
            for entry in h.get("intervals", []):
                for field in ("metadata", "content"):
                    if not safe_child(layout.root, entry[field]).is_file():
                        raise ValueError("A published interval document is missing")
                if entry.get("video"):
                    from visioncortex.device_day_verification import (
                        verify_artifact_cached,
                    )

                    if not verify_artifact_cached(layout.root, entry["video"]):
                        raise ValueError(
                            "A published interval video failed byte verification"
                        )
        atomic_json(
            layout.processed / ".Evidence/ArchiveUnification/Acceptance.json",
            result | {"build": identity(), "verified_at": time.time()},
        )
        if checkpoint_path.is_file():
            atomic_json(
                checkpoint_path, read_json(checkpoint_path) | {"completed": True}
            )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument(
        "--public-video-inventory",
        type=Path,
        help="Frozen initial public-video inventory; retired hidden videos remain private",
    )
    parser.add_argument("--archive", action="append")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    from visioncortex.ai_settings import apply_active

    config = apply_active(config)
    settings = config.get("device_day") or {}
    if "understanding" not in settings.get("paused_stages", []):
        raise ValueError(
            "Paid understanding must remain explicitly paused for this format-only migration"
        )
    if not clip_enabled(config) or settings.get("archive_html_enabled", True):
        raise ValueError(
            "Migration requires the approved source-recording layout without HTML"
        )
    verify_storage(config)
    from visioncortex.local_storage import require_local_path

    require_local_path(args.plan_root)
    inventories = {}
    if args.public_video_inventory:
        from visioncortex.application.archive_unification import bounded_document

        require_local_path(args.public_video_inventory)
        inventories = {
            row["archive"]: row["files"]
            for row in bounded_document(args.public_video_inventory)["archives"]
        }
    local_root = Path(config["storage"]["local_runtime_root"]).resolve()
    if not args.plan_root.resolve().is_relative_to(local_root):
        raise ValueError(
            "Migration plans must remain inside the configured local runtime root"
        )
    args.plan_root.mkdir(parents=True, exist_ok=True)
    if (
        shutil.disk_usage("/").free < 50 * 1024**3
        or shutil.disk_usage(local_root).free < 80 * 1024**3
    ):
        raise ValueError("Local storage headroom is below the authorized floor")
    if (
        sum(p.stat().st_size for p in args.plan_root.rglob("*") if p.is_file())
        > 512 * 1024**2
    ):
        raise ValueError("Bounded local migration cache is full")
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    root = Path(config["storage"]["archive_root"])
    if args.archive:
        names = sorted(set(map(validate_archive_name, args.archive)), reverse=True)
    else:
        with storage_guard(config):
            names = []
            for path in root.iterdir():
                if path.is_dir() and not path.is_symlink():
                    try:
                        names.append(validate_archive_name(path.name))
                    except ValueError:
                        continue
            names.sort(reverse=True)
    progress = {
        "recipe": RECIPE,
        "started_at": time.time(),
        "build": identity(),
        "paid_model_calls": 0,
        "target_count": len(names),
        "archives": [],
    }
    for name in names:
        if stop.is_set():
            break
        print(
            json.dumps({"archive": name, "status": "starting"}, ensure_ascii=False),
            flush=True,
        )
        try:
            result = run(
                config,
                name,
                args.plan_root,
                stop,
                apply=args.apply,
                public_video_inventory=inventories.get(name, [])
                if args.public_video_inventory
                else None,
            )
        except BlockingIOError:
            result = {"archive": name, "status": "busy_retry", "model_invoked": False}
        except Exception as exc:
            import traceback

            traceback.print_exc()
            result = {
                "archive": name,
                "status": "verification_pending",
                "error_type": type(exc).__name__,
                "reason": str(exc)[:500],
                "model_invoked": False,
            }
        progress["archives"].append(result)
        progress["updated_at"] = time.time()
        atomic_json(args.plan_root.parent / "MigrationProgress.json", progress)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def clip_enabled(config):
    from visioncortex.application.clip_delivery import enabled

    return enabled(config)


if __name__ == "__main__":
    main()

"""Refresh mutable native main entries from committed Timeline sidecars.

No discovery, models, selection, clip export, canonical index writes or new
state store. Existing native pure builders and mutable-view writer retain output
ownership. Missing native indexes remain pending, never fabricated.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import PurePosixPath
import stat
import time

from .archive_layout import verify_archive_contract
from .domain import fingerprint
from .media import safe_path
from .native_linked_layout import module
from .native_timeline_links import (PendingCommittedAssociation, association_revision,
    attach_day, attach_segment, verified_committed_association)
from .rules import Rules
from .storage import is_nas, publication_config

MAX_JSON_BYTES = 8 * 1024**2
MAX_ENTRIES = 512
MAX_MAIN_DOCUMENTS = 32
MAX_OUTPUT_BYTES = 64 * 1024**2
MAX_RETAINED_JSON_BYTES = 32 * 1024**2


@contextmanager
def _directory(root, relative, guard):
    """Anchored read-only traversal: never mkdir on a missing NAS directory."""
    parts = PurePosixPath(relative).parts if relative else ()
    if relative and (
        PurePosixPath(relative).is_absolute()
        or "\\" in relative
        or any(p in {"", ".", ".."} for p in relative.split("/"))
    ):
        raise ValueError("Normalized native-relative path required")
    guard.validate()
    mount, path = guard.mount_for(root.joinpath(*parts))
    if mount is None and is_nas(root):
        raise OSError("Native refresh has no pinned NAS mount")
    descriptor = None
    try:
        descriptor = (
            os.dup(mount.descriptor)
            if mount
            else os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        )
        walk = path.relative_to(mount.path).parts if mount else parts
        for part in walk:
            guard.validate(volume=False)
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                guard.validate()
                yield None
                return
            os.close(descriptor)
            descriptor = child
            if mount and os.fstat(descriptor).st_dev != mount.snapshot["device"]:
                raise OSError("Native refresh crossed an unverified filesystem")
        yield descriptor
        guard.validate()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read(root, relative, guard):
    parts = PurePosixPath(relative).parts
    if not parts:
        raise ValueError("Native JSON path required")
    with _directory(root, "/".join(parts[:-1]), guard) as parent:
        if parent is None:
            return None
        guard.validate(volume=False)
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
                raise ValueError("Native refresh JSON exceeds regular-file bound")
            if before.st_dev != os.fstat(parent).st_dev:
                raise OSError("Native JSON crossed filesystem")
            raw = handle.read(MAX_JSON_BYTES + 1)
            after = os.fstat(handle.fileno())
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            len(raw) != before.st_size
            or len(raw) > MAX_JSON_BYTES
            or any(getattr(before, k) != getattr(after, k) for k in keys)
        ):
            raise OSError("Native refresh JSON changed during read")
    value = json.loads(
        raw,
        parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("Nonfinite native JSON")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("Native document must be an object")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _understandings(root, guard):
    """Only public mutable main entries; immutable execution/history excluded."""
    result = []
    with _directory(root, "MultimodalUnderstanding", guard) as parent:
        if parent is None:
            return result
        with os.scandir(parent) as entries:
            for count, entry in enumerate(entries, 1):
                if count > MAX_ENTRIES:
                    raise ValueError("Native understanding directory exceeds bound")
                if entry.name.startswith(".") or entry.name in {
                    "Analysis",
                    "ClipUnderstanding",
                    "History",
                    "Timeline",
                }:
                    continue
                if entry.is_symlink():
                    raise ValueError("Native public understanding cannot be a symlink")
                if entry.is_dir(follow_symlinks=False):
                    result.append(
                        "MultimodalUnderstanding/" + entry.name + "/Understanding.json"
                    )
    if len(result) > MAX_MAIN_DOCUMENTS:
        raise ValueError("Native mutable main count exceeds refresh bound")
    return sorted(result)


class NativeRefresh:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings

    def tick(self, limit=4):
        if (
            self.settings.processing_profile != "timeline-v1"
            or not self.settings.timeline_publish
        ):
            return dict(enabled=False, refreshed=0)
        if (
            self.settings.source_root is None
            or not self.settings.unified_archive_required
        ):
            raise ValueError(
                "Native refresh requires the explicit unified publication archive"
            )
        if type(limit) is not int or limit < 1:
            raise ValueError("Positive native refresh bound required")
        rules = Rules().identity()
        root = self.settings.incoming
        with self.core.db.transaction(readonly=True) as conn:
            scope = conn.execute(
                "SELECT data_scope FROM customer_instance WHERE id=%s",
                (self.core.customer,),
            ).fetchone()["data_scope"]
        if is_nas(root) and scope != "customer":
            raise ValueError("Fixture/development cannot refresh formal NAS")
        verify_archive_contract(root)
        destination = fingerprint(
            dict(
                root=str(root),
                customer=str(self.core.customer),
                namespace="timeline-publication/1",
                publisher_rules=rules,
            )
        )
        with self.core.db.transaction(readonly=True) as conn:
            publications = conn.execute(
                "SELECT DISTINCT ON(day) day,revision,files FROM hj_timeline_publication WHERE customer_id=%s AND destination_fingerprint=%s ORDER BY day DESC,revision DESC LIMIT %s",
                (self.core.customer, destination, min(limit, 20)),
            ).fetchall()
        config = publication_config(self.settings)
        native_runtime = self.settings.runtime_root / "native"
        if is_nas(native_runtime):
            raise ValueError("Native refresh locks must remain local")
        config["storage"].update(
            local_runtime_root=str(native_runtime),
            local_cache_root=str(native_runtime / "cache"),
        )
        config.setdefault("device_day", {}).update(archive_html_enabled=False)
        outcomes = []
        budget = MAX_OUTPUT_BYTES
        deadline = time.monotonic() + 30
        for publication in publications:
            if len(publication["files"]) > MAX_ENTRIES:
                raise ValueError("Publication member count exceeds bound")
            for entry in publication["files"]:
                parts = PurePosixPath(entry["relative_path"]).parts
                if len(parts) != 5 or parts[1:4] != (
                    "LaboratoryDailyReport",
                    "Timeline",
                    "Associations",
                ):
                    continue
                archive = module("device_day_contract").validate_archive_name(parts[0])
                if archive[:10] != str(publication["day"]):
                    raise ValueError("Publication archive/day mismatch")
                if (
                    sum(x["status"] == "refreshed" for x in outcomes) >= min(limit, 4)
                    or len(outcomes) >= 64
                    or time.monotonic() > deadline
                ):
                    break
                with self.core.db.transaction() as conn:
                    locked = conn.execute(
                        "SELECT pg_try_advisory_xact_lock(hashtextextended(%s,0)) AS acquired",
                        ("native-refresh:" + str(root) + ":" + archive,),
                    ).fetchone()["acquired"]
                    if not locked:
                        outcomes.append(dict(archive=archive, status="busy"))
                        continue
                    lock = (
                        native_runtime / "device-day/locks" / (archive + ".index.lock")
                    )
                    lock.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        (self.settings.runtime_root / "timeline-publication.lock").open(
                            "a+b"
                        ) as publication_lock,
                        lock.open("a+b") as handle,
                    ):
                        try:
                            # Share the original sidecar publisher's lock, so a
                            # late revision cannot change halfway through a main refresh.
                            fcntl.flock(publication_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            outcomes.append(dict(archive=archive, status="busy"))
                            continue
                        try:
                            with module("device_day_write_guard").storage_guard(
                                config
                            ) as guard:
                                outcome = self._refresh(
                                    root, archive, entry, guard, budget
                                )
                            budget -= outcome.pop("written_bytes", 0)
                            outcomes.append(outcome)
                        except (ValueError, OSError, KeyError, TypeError) as exc:
                            outcomes.append(
                                dict(
                                    archive=archive,
                                    status="needs_attention",
                                    error_code=type(exc).__name__,
                                )
                            )
        refreshed = sum(x["status"] == "refreshed" for x in outcomes)
        return dict(
            enabled=True,
            status="awaiting_committed_timeline_publication"
            if not publications
            else "bounded_refresh_checked",
            refreshed=refreshed,
            unchanged=sum(x["status"] == "unchanged" for x in outcomes),
            pending=sum(
                x["status"] in {"pending_native_index", "pending_committed_association"}
                for x in outcomes
            ),
            errors=sum(x["status"] == "needs_attention" for x in outcomes),
            reports=refreshed,
            outcomes=outcomes,
            model_calls=0,
            clip_exports=0,
            source_index_writes=0,
        )

    def _refresh(self, root, archive, committed, guard, budget=MAX_OUTPUT_BYTES):
        day_root = root / archive
        try:
            with verified_committed_association(day_root, archive,
                    customer_id=self.core.customer, receipt=committed) as selected:
                return self._refresh_verified(root, archive, selected[1], guard, budget)
        except PendingCommittedAssociation:
            return dict(archive=archive, status="pending_committed_association")

    def _refresh_verified(self, root, archive, reference, guard, budget):
        day_root = root / archive
        source = _read(day_root, "ProcessedClips/Index.json", guard)
        if source is None:
            return dict(archive=archive, status="pending_native_index")
        index, source_sha, _ = source
        if index.get("archive") != archive:
            raise ValueError("Native index belongs to another camera/day")
        if any(
            len(index.get(k, [])) > MAX_ENTRIES
            for k in ("recordings", "segments", "understandings")
        ):
            raise ValueError("Native index exceeds bounded refresh scope")
        contract = module("device_day_contract")
        content = module("device_day_content")
        reports = module("device_day_reports")
        mains = []
        changed = []
        written = 0
        retained_bytes = source[2]
        for relative in _understandings(day_root, guard):
            old = _read(day_root, relative, guard)
            if old is None:
                continue
            document = old[0]
            if document.get("archive") != archive:
                raise ValueError("Native main belongs to another camera/day")
            # Only mutable public projections of these explicit native schemas.
            if document.get("schema_version") not in {
                "visioncortex-readable-understanding/1",
                "visioncortex-source-recording-delivery/2",
            }:
                raise ValueError("Unknown native main schema is not refreshable")
            segment = next(
                (
                    s
                    for s in index.get("segments", [])
                    if s["segment_id"] == document.get("segment_id")
                    and s.get("recording_id") == document.get("recording_id")
                ),
                None,
            )
            if segment and (
                document.get("source_video") != segment.get("source_ref")
                or document.get("source_start_ms") != segment.get("start_ms")
                or document.get("source_end_ms") != segment.get("end_ms")
            ):
                raise ValueError("Native main and index original source/range disagree")
            updated = (
                attach_segment(document, day_root, index, segment)
                if segment
                else attach_day(document, day_root, archive)
            )
            if any(updated["关联时间线"].get(k) != v for k, v in reference.items()):
                raise OSError("Association revision changed during native planning")
            size = len(
                json.dumps(updated, ensure_ascii=False, allow_nan=False).encode()
            )
            if size > MAX_JSON_BYTES or written + size > budget:
                raise ValueError("Native refresh output budget reached")
            retained_bytes += old[2] + size
            if retained_bytes > MAX_RETAINED_JSON_BYTES:
                raise ValueError("Native retained JSON cache bound reached")
            mains.append(dict(path=relative, prior_sha256=old[1], updated=updated))
            if updated != document:
                changed.append(relative)
                written += size
        day = module("device_day_timeline").day_bounds(archive[:10])[0]
        layout = contract.DeviceDayLayout(
            root,
            archive[11:],
            day,
            self.settings.runtime_root / "native/cache",
            read_only_originals=True,
        )
        prior = _read(
            day_root, "LaboratoryDailyReport/LaboratoryDailyReport.json", guard
        )
        digest = contract.digest(index)
        day_current = bool(
            prior
            and prior[0].get("input_index_digest") == digest
            and prior[0].get("关联时间线") == reference
        )
        if day_current and not changed:
            return dict(
                archive=archive,
                status="unchanged",
                snapshot_revision=reference["snapshot_revision"],
                written_bytes=0,
            )
        if not day_current:
            # Preflight only the actual native builder; no model/synthesis call.
            readable = module("device_day_narrative").day_record(index, day_root)
            synthesis = module("device_day_synthesis").current_synthesis(layout, index)
            planned = reports.day_document(layout, index, synthesis, readable=readable)
            if planned.get("关联时间线") != reference:
                raise OSError("Association revision changed during native planning")
            size = len(
                json.dumps(planned, ensure_ascii=False, allow_nan=False).encode()
            )
            if size > MAX_JSON_BYTES or written + size > budget:
                raise ValueError("Native daily output budget reached")
            if retained_bytes + size > MAX_RETAINED_JSON_BYTES:
                raise ValueError("Native retained JSON cache bound reached")
            written += size
        # Recheck inputs before any mutable native output is refreshed.
        association_revision(day_root, archive)
        if _read(day_root, "ProcessedClips/Index.json", guard)[1] != source_sha:
            raise OSError("Native index changed before refresh")
        for main in mains:
            if _read(day_root, main["path"], guard)[1] != main["prior_sha256"]:
                raise OSError("Native main changed before refresh")
        for main in mains:
            if main["path"] in changed:
                guard.validate()
                content._write(safe_path(day_root, main["path"]), main["updated"])
        if not day_current:
            guard.validate()
            reports.render_day(layout, index, html_enabled=False)
        guard.validate()
        actual = _read(
            day_root, "LaboratoryDailyReport/LaboratoryDailyReport.json", guard
        )
        if actual is None or actual[0].get("关联时间线") != reference:
            raise OSError("Native daily refresh read-back failed")
        for main in mains:
            if _read(day_root, main["path"], guard)[0] != main["updated"]:
                raise OSError("Native understanding read-back failed")
        if _read(day_root, "ProcessedClips/Index.json", guard)[1] != source_sha:
            raise OSError("Native index changed during refresh")
        return dict(
            archive=archive,
            status="refreshed",
            snapshot_revision=reference["snapshot_revision"],
            snapshot_fingerprint=reference["snapshot_fingerprint"],
            daily_report_sha256=actual[1],
            understandings_refreshed=len(changed),
            native_entry_refresh_status="included_by_native_builder",
            written_bytes=written,
            model_calls=0,
            clip_exports=0,
            source_index_writes=0,
        )

"""Format-only migration of historical device-day results; never runs models.

The canonical producer index and recorder-owned files remain unchanged.  Old
payloads retain their byte identities in evidence while public collections use
the existing one-original layout.  The native archive implementation is an
explicit port so a migration can verify historical receipts with its frozen
runtime without declaring the new migration code to be that producer.
"""

from pathlib import Path
import json
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from ..device_day_contract import DeviceDayLayout, atomic_json, digest, safe_child
from ..device_day_content_paths import aliases, public_references, private_folder
from . import clip_delivery


RECIPE = "visioncortex-historical-format-unification/1"
RECORD_FIELDS = {
    "recording_id",
    "camera_key",
    "start_us",
    "end_us",
    "sources",
    "stages",
    "capture_complete",
    "capture_issues",
    "clock_ref",
}
SEGMENT_FIELDS = {
    "segment_id",
    "recording_id",
    "candidate_segment_id",
    "source_ref",
    "json_path",
    "start_us",
    "end_us",
    "start_ms",
    "end_ms",
    "activity",
    "video",
}


def file_identity(path):
    stat = path.stat()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Migration requires regular files")
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def read_source_projection(
    path,
    *,
    events,
    checkpoint=lambda: None,
    maximum_bytes=2 * 1024**3,
    maximum_memory=64 * 1024**2,
):
    """Read only ownership/clocks, not dense frames or model diagnostics.

    This intentionally incomplete object is a migration plan, never a producer
    index or replacement for one. Unknown canonical fields stay in the original.
    """
    from ijson.common import ObjectBuilder

    path = Path(path)
    before = file_identity(path)
    if before[2] > maximum_bytes:
        raise ValueError("Canonical index exceeds the bounded migration input")

    def include(parts):
        if not parts:
            return True
        if parts[0] not in {"archive", "schema_version", "recordings", "segments"}:
            return False
        if len(parts) < 3:
            return True
        return parts[2] in (
            RECORD_FIELDS if parts[0] == "recordings" else SEGMENT_FIELDS
        )

    builder, stack, memory = ObjectBuilder(), [], 0
    with path.open("rb") as handle:
        for count, (event, value) in enumerate(events(handle)):
            if count % 4096 == 0:
                checkpoint()
            parent = stack[-1] if stack else None
            child = (
                ()
                if parent is None
                else parent["path"] + ((parent["key"],) if parent["map"] else (None,))
            )
            if event == "map_key":
                parent["key"] = value
                keep = not parent["omit"] and include(parent["path"] + (value,))
            elif event in {"start_map", "start_array"}:
                skip = bool(parent and parent["omit"]) or not include(child)
                stack.append(
                    {
                        "path": child,
                        "map": event == "start_map",
                        "key": None,
                        "omit": skip,
                    }
                )
                keep = not skip
            elif event in {"end_map", "end_array"}:
                keep = not stack.pop()["omit"]
            else:
                keep = not (parent and parent["omit"]) and include(child)
            if keep:
                memory += 2 * (sys.getsizeof(value) + 64)
                if memory > maximum_memory:
                    raise ValueError(
                        "Migration ownership projection exceeds its memory bound"
                    )
                builder.event(event, value)
    if file_identity(path) != before:
        raise ValueError("Canonical index changed during migration planning")
    if not isinstance(builder.value, dict):
        raise ValueError("Canonical index is not an object")
    value = builder.value
    value.setdefault("recordings", [])
    value.setdefault("segments", [])
    return value


def bounded_document(path, maximum_bytes=32 * 1024**2):
    path = Path(path)
    before = file_identity(path)
    if before[2] > maximum_bytes:
        raise ValueError(
            "Historical sidecar exceeds the bounded migration document size"
        )
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if file_identity(path) != before or not isinstance(value, dict):
        raise ValueError("Historical sidecar changed or is not an object")
    return value


def _time(value):
    return datetime.fromtimestamp(value / 1e6, ZoneInfo("Asia/Shanghai")).isoformat(
        timespec="microseconds"
    )


class ArchiveUnifier:
    """Compose the existing publication owner with saved historical documents.

    There is no model backend, queue admission, selection inference or original
    file writer in this service. A historical video keeps its old selection;
    its presence is never presented as newly proven value or physical action.
    """

    def __init__(self, config, name, index):
        from ..device_day_timeline import day_bounds

        self.config, self.index = config, index
        start, _ = day_bounds(name[:10])
        self.layout = DeviceDayLayout(
            Path(config["storage"]["archive_root"]),
            name[11:],
            start,
            read_only_originals=True,
        )
        self.rows = {}
        self.by_id = {s["segment_id"]: s for s in index["segments"]}
        self.records = {r["recording_id"]: r for r in index["recordings"]}
        self.current_groups = set()
        self.legacy_folders = set()

    def checkpoint(self, path):
        """Persist the complete source-bound plan before retiring any folder."""
        atomic_json(
            path,
            {
                "recipe": RECIPE,
                "archive": self.layout.name,
                "index": self.index,
                "rows": [
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"record", "candidate", "unit"}
                    }
                    for row in self.values
                ],
                "legacy_folders": sorted(
                    self.layout.relative(p) for p in self.legacy_folders
                ),
            },
        )

    def restore(self, path):
        value = bounded_document(path, maximum_bytes=64 * 1024**2)
        if value.get("recipe") != RECIPE or value.get("archive") != self.layout.name:
            raise ValueError(
                "Historical migration checkpoint belongs to another archive"
            )
        for saved in value.get("rows", []):
            row = self._row(saved["segment"])
            if (
                row is None
                or row["key"] != saved["key"]
                or saved["category"] not in clip_delivery.LABELS
            ):
                raise ValueError(
                    "Historical migration checkpoint has an invalid source binding"
                )
            for key in ("metadata_document", "understanding_document"):
                relative = saved["meaning"].get(key)
                if relative:
                    safe_child(self.layout.root, relative)
            row.update(
                {
                    k: v
                    for k, v in saved.items()
                    if k not in {"record", "candidate", "unit"}
                }
            )
            row.update(
                record=self.records[row["segment"]["recording_id"]],
                candidate=row["segment"],
            )
        for relative in value.get("legacy_folders", []):
            parts = Path(relative).parts
            if not (
                parts[:2] == ("ProcessedClips", "Clips")
                and len(parts) == 3
                or parts[:1] == ("MultimodalUnderstanding",)
                and len(parts) == 2
            ):
                raise ValueError("Checkpoint contains an unowned retirement path")
            self.legacy_folders.add(self.layout.root / relative)

    def _recover_relocated(self):
        """Import only owned sidecars preserved by durable native move intents."""
        self._recover_aliases()
        for directory in ("ProcessedClips", "MultimodalUnderstanding"):
            journals = self.layout.root / directory / ".Evidence/ClipRelocations"
            for journal in sorted(journals.glob("*.json")):
                intent = bounded_document(journal)
                old, target = intent.get("from", ""), intent.get("to", "")
                parts = Path(old).parts
                owned = (
                    directory == "ProcessedClips"
                    and parts[:2] == (directory, "Clips")
                    and len(parts) == 3
                    or directory == "MultimodalUnderstanding"
                    and parts[:1] == (directory,)
                    and len(parts) == 2
                )
                if (
                    intent.get("schema_version") != clip_delivery.VERSION
                    or not owned
                    or intent.get("operation") != "preserve_owned_derived_directory"
                    or not target.startswith(
                        directory + "/.Evidence/LegacyPresentation/"
                    )
                ):
                    raise ValueError(
                        "Historical relocation intent lacks owned provenance"
                    )
                folder = safe_child(self.layout.root, target)
                if not folder.is_dir():
                    continue
                if directory == "ProcessedClips":
                    for path in clip_delivery._metadata_files(folder):
                        if path.is_file():
                            self._read_candidate(path)
                else:
                    path = folder / "Understanding.json"
                    if path.is_file():
                        doc = bounded_document(path)
                        if doc.get("schema_version") != clip_delivery.VERSION:
                            self._read_understanding(path)

    def _recover_aliases(self):
        """Repair absent derived aliases from durable owned move intents.

        Completed video publication can reclaim an owned redundant copy. Bind
        such old addresses only to a currently verified identical artifact.
        Never reconstruct a model receipt or change a recorder-owned path.
        """
        from ..device_day_content_paths import relocated
        from ..device_day_verification import verify_artifact_cached

        for directory in ("ProcessedClips", "MultimodalUnderstanding"):
            evidence = self.layout.root / directory / ".Evidence"

            def physical(relative):
                if not relative.startswith(directory + "/"):
                    raise ValueError(
                        "Alias reconstruction crossed its owning directory"
                    )
                # Inspect physical destinations without consulting the broken
                # alias file which this operation is repairing.
                return safe_child(
                    self.layout.root / directory, relative.removeprefix(directory + "/")
                )

            def verified(reference):
                path = physical(reference["path"])
                return verify_artifact_cached(
                    path.parent, reference | {"path": path.name}
                )

            existing = {
                k: v
                for k, v in aliases(self.layout.root).items()
                if k.startswith(directory + "/")
            }
            mappings, preserved, groups, videos = dict(existing), [], {}, {}
            parent = (
                self.layout.processed / "Clips"
                if directory == "ProcessedClips"
                else self.layout.understanding
            )
            if parent.is_dir():
                for folder in parent.iterdir():
                    if not folder.is_dir() or folder.name.startswith("."):
                        continue
                    path = folder / (
                        "Index.json"
                        if directory == "ProcessedClips"
                        else "Understanding.json"
                    )
                    header = bounded_document(path) if path.is_file() else {}
                    if header.get("schema_version") != clip_delivery.VERSION:
                        continue
                    groups[header["source_unit"]] = self.layout.relative(folder)
                    for row in header.get("intervals", []):
                        ref = row.get("video")
                        if ref:
                            videos.setdefault(
                                (ref["sha256"], ref["size_bytes"]), []
                            ).append(ref)
            for path in sorted((evidence / "ClipRelocations").glob("*.json")):
                intent = bounded_document(path)
                old, target = intent.get("from", ""), intent.get("to", "")
                prefix = directory + (
                    "/Clips/" if directory == "ProcessedClips" else "/"
                )
                if (
                    intent.get("schema_version") != clip_delivery.VERSION
                    or intent.get("operation") != "preserve_owned_derived_directory"
                    or not old.startswith(prefix)
                    or len(Path(old).parts) != len(Path(prefix).parts) + 1
                    or not target.startswith(
                        directory + "/.Evidence/LegacyPresentation/"
                    )
                ):
                    raise ValueError(
                        "Cannot reconstruct aliases from an unowned relocation"
                    )
                folder = physical(target)
                if folder.is_dir():
                    mappings.setdefault(old, target)
                    preserved.append(folder)
            for path in sorted((evidence / "OriginalGroupRenames").glob("*.json")):
                intent = bounded_document(path)
                target = groups.get(intent.get("source_unit"))
                if target and intent.get("directory") == directory:
                    for old in (intent["from"], intent["to"]):
                        physical(old)
                        if old != target:
                            mappings.setdefault(old, target)
            for path in sorted((evidence / "ClipFileRelocations").glob("*.json")):
                intent = bounded_document(path)
                old, target = intent.get("from", ""), intent.get("to", "")
                if not old.startswith(directory + "/") or not target.startswith(
                    directory + "/.Evidence/LegacyPresentation/"
                ):
                    raise ValueError(
                        "Cannot reconstruct aliases outside owned derived evidence"
                    )
                actual = physical(target)
                if verify_artifact_cached(actual.parent, intent["file"]):
                    mappings.setdefault(old, target)
            # A returning category can have the same public name as a retired
            # generation. Its actual current group must remain a physical
            # destination, never an alias back into its own history.
            for current_group in groups.values():
                mappings.pop(current_group, None)
            if directory == "ProcessedClips":
                for folder in preserved:
                    for path in clip_delivery._metadata_files(folder):
                        if not path.is_file():
                            continue
                        document = bounded_document(path)
                        ref = document.get("video")
                        if not ref or not ref.get("path", "").startswith(
                            "ProcessedClips/"
                        ):
                            continue
                        resolved = relocated(ref["path"], mappings)
                        if verified(ref | {"path": resolved}):
                            continue
                        matches = videos.get(
                            (ref.get("sha256"), ref.get("size_bytes")), []
                        )
                        verified_matches = [
                            candidate for candidate in matches if verified(candidate)
                        ]
                        if verified_matches:
                            target = verified_matches[0]["path"]
                            for old in (ref["path"], resolved):
                                if old != target:
                                    mappings[old] = target
            if mappings != existing:
                # Validate every chain before publishing the derived reader map.
                for old in mappings:
                    physical(relocated(old, mappings))
                clip_delivery._publish_aliases(self.layout, directory, mappings)

    def _row(self, value):
        reference = value.get("source_ref") or value.get("source_video")
        if not isinstance(reference, dict) or not reference.get("path"):
            return None
        clip_delivery.source_unit(reference)
        if len(str(reference.get("sha256", ""))) != 64:
            raise ValueError("Historical interval lacks a source content identity")
        rid = value.get("recording_id")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", rid):
            raise ValueError(
                "Historical recording identity is not a safe archive identifier"
            )
        fields = {key: value.get(key) for key in SEGMENT_FIELDS}
        fields.update(
            source_ref=reference,
            start_ms=value.get("start_ms", value.get("source_start_ms")),
            end_ms=value.get("end_ms", value.get("source_end_ms")),
        )
        if not rid or any(
            type(fields[k]) not in (int, float)
            for k in ("start_us", "end_us", "start_ms", "end_ms")
        ):
            raise ValueError("Historical interval lacks an exact source/time binding")
        if not (
            fields["start_us"] < fields["end_us"]
            and fields["start_ms"] < fields["end_ms"]
        ):
            raise ValueError("Historical interval has invalid bounds")
        if rid not in self.records:
            # Old presentation generations may retain an earlier producer ID.
            # Preserve it; this descriptor is not admitted as a new recording.
            record = {
                "recording_id": rid,
                "camera_key": self.layout.name[11:],
                "start_us": fields["start_us"],
                "end_us": fields["end_us"],
                "sources": [{"kind": "video", "retained": reference}],
                "stages": {},
                "historical_format_descriptor_only": True,
            }
            self.records[rid] = record
            self.index["recordings"].append(record)
        times = [fields[k] for k in ("start_ms", "end_ms", "start_us", "end_us")]
        times = [int(v) if v == int(v) else v for v in times]
        key = digest([clip_delivery.VERSION, rid, reference, *times])
        fields["segment_id"] = fields.get("segment_id") or "historical-" + key[:24]
        fields["recording_id"] = rid
        fields["activity"] = fields.get("activity") or "inactive"
        existing = self.rows.get(key)
        if existing is not None:
            return existing

        def stamp(v):
            return datetime.fromtimestamp(v / 1e6, ZoneInfo("Asia/Shanghai")).strftime(
                "%H-%M-%S"
            )

        row = {
            "key": key,
            "segment": fields,
            "record": self.records[rid],
            "candidate": fields,
            "meaning": {},
            "category": "PendingValueReview",
            "label": clip_delivery.LABELS["PendingValueReview"],
            "source_segment_ids": [fields["segment_id"]],
            "interval": {
                **{
                    k: fields[k]
                    for k in ("start_ms", "end_ms", "start_us", "end_us", "activity")
                },
                "video_value": "pending",
                "reason": "historical_value_not_revalidated",
                "historical_format_migration_only": True,
            },
            "_stamp": f"{stamp(fields['start_us'])}_{stamp(fields['end_us'])}",
        }
        self.rows[key] = row
        return row

    def _read_candidate(self, path):
        document = bounded_document(path)
        row = self._row(document)
        if row is None:
            return
        row["meaning"].setdefault("metadata_document", self.layout.relative(path))
        row["segment"]["json_path"] = self.layout.relative(path)
        retention = document.get("retention") or {}
        separate = path.parent / "Retention.json"
        if (
            not retention
            and separate.is_file()
            and path.parent not in self.current_groups
        ):
            prior = bounded_document(separate)
            if all(
                prior.get(k) == document.get(k)
                for k in ("recording_id", "source_ref", "start_us", "end_us")
            ):
                retention = prior
        selection = (
            document.get("selection_interval")
            or retention.get("selection_interval")
            or {}
        )
        if (
            selection.get("video_value") == "confirmed_no_value"
            or retention.get("no_value_proven") is True
        ):
            row["category"] = "NoValueNoExperimentActivity"
            row["interval"].update(
                video_value="confirmed_no_value",
                reason="preserved_historical_no_value_decision",
            )
        video = document.get("video") or selection.get("historical_existing_video")
        folder = path.parent
        if not video:
            candidate = next(iter(folder.glob("*.mp4")), None)
            if candidate is not None and folder not in self.current_groups:
                from ..device_day_contract import artifact

                video = artifact(self.layout.root, candidate)
                row["meaning"]["historical_video_had_no_sidecar_reference"] = True
        if video and row["category"] != "NoValueNoExperimentActivity":
            from ..device_day_verification import verify_artifact_cached

            if not verify_artifact_cached(self.layout.root, video):
                raise ValueError(
                    "Existing historical video differs from its recorded checksum"
                )
            row["segment"]["video"] = video
            row["interval"].update(
                video_value="selected",
                reason="existing_historical_materialization_preserved",
                selection_revalidated=False,
                historical_existing_video=video,
            )
            row["meaning"]["historical_selection_only"] = True
        prior = document.get("display_category")
        row["meaning"]["prior_display_category"] = prior
        return row

    def _read_understanding(self, path):
        document = bounded_document(path)
        row = self._row(document)
        if row is None:
            return
        previous = row["meaning"].get("understanding_document")
        if previous:
            old = bounded_document(safe_child(self.layout.root, previous))
            rank = {"completed": 3, "partial": 2, "submitted_unknown": 1}
            if rank.get(old.get("status"), 0) > rank.get(document.get("status"), 0):
                return
        row["meaning"]["understanding_document"] = self.layout.relative(path)
        # Preserve a saved model's classification only when there is actual
        # returned content. CV activity without a returned observation remains
        # pending and never becomes an experiment claim during migration.
        prior = document.get("display_category")
        observed = document.get("understanding") or document.get("experiment_steps")
        if row["category"] == "NoValueNoExperimentActivity":
            return
        if observed and prior in {"ExperimentActivity", "ValuableNoExperimentActivity"}:
            row["category"] = prior
        elif observed and row["meaning"].get("prior_display_category") in {
            "ExperimentActivity",
            "ValuableNoExperimentActivity",
        }:
            row["category"] = row["meaning"]["prior_display_category"]

    def discover(self):
        """Read public source-bound sidecars before any relocation happens."""
        self._recover_relocated()
        root = self.layout.root
        parent = self.layout.processed / "Clips"
        for folder in sorted(parent.iterdir()) if parent.is_dir() else []:
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            group = folder / "Index.json"
            header = bounded_document(group) if group.is_file() else {}
            if header.get("schema_version") == clip_delivery.VERSION:
                self.current_groups.add(folder)
                for entry in header.get("intervals", []):
                    relative = entry.get("metadata")
                    if relative and safe_child(root, relative).is_file():
                        self._read_candidate(safe_child(root, relative))
                continue
            paths = [
                folder / (kind + ".json")
                for kind in (*clip_delivery.LABELS, "NoExperimentActivity", "Clip")
            ]
            paths = [p for p in paths if p.is_file()]
            if not paths:
                segments = [
                    s
                    for s in self.index["segments"]
                    if s.get("json_path", "").startswith(
                        self.layout.relative(folder) + "/"
                    )
                ]
                for segment in segments:
                    row = self._row(segment)
                    if row:
                        row["segment"]["json_path"] = segment.get("json_path")
            owned = False
            for path in paths:
                owned = self._read_candidate(path) is not None or owned
            if owned:
                self.legacy_folders.add(folder)
        for folder in (
            sorted(self.layout.understanding.iterdir())
            if self.layout.understanding.is_dir()
            else []
        ):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            path = folder / "Understanding.json"
            if not path.is_file():
                continue
            value = bounded_document(path)
            if value.get("schema_version") == clip_delivery.VERSION:
                for entry in value.get("intervals", []):
                    relative = entry.get("content")
                    if relative and safe_child(root, relative).is_file():
                        self._read_understanding(safe_child(root, relative))
            else:
                self._read_understanding(path)
                self.legacy_folders.add(folder)
        for row in self.rows.values():
            kind = row["category"]
            row.update(
                name=f"{row['_stamp']}_{clip_delivery._display_category(kind)}_{row['key'][:8]}",
                label=clip_delivery.LABELS[kind],
            )
        values = sorted(
            self.rows.values(), key=lambda r: (r["segment"]["start_us"], r["key"])
        )
        units = clip_delivery.original_units(self.layout, self.index, values)
        for row in values:
            key, _ = clip_delivery.source_unit(row["segment"]["source_ref"])
            if key not in units:
                raise ValueError("Historical result has no owning original collection")
        self.values, self.units = values, units
        return {
            "originals": len(units),
            "intervals": len(values),
            "legacy_folders": len(self.legacy_folders),
        }

    def _document(self, index, root, segment, saved, interval, record, audio_entries):
        relative = saved.get("understanding_document")
        value = bounded_document(safe_child(root, relative)) if relative else {}
        value = public_references(value, aliases(root))
        metadata = saved.get("metadata_document")
        if metadata:
            original = public_references(
                bounded_document(safe_child(root, metadata)), aliases(root)
            )
            for key in (
                "key_frames",
                "scene_frames",
                "clock_ref",
                "sampling_coverage",
                "visibility",
                "uncertainties",
            ):
                if key in original:
                    segment[key] = original[key]
            if segment["activity"] == "inactive":
                segment["key_frames"] = []
                segment["scene_frames"] = [
                    dict(frame, frame_kind="scene_sample")
                    for frame in segment.get("scene_frames", [])
                ]
        status = value.get("status") or "not_processed"
        narrative = {
            key: val
            for key, val in (value.get("内容记录") or {}).items()
            if key not in {"实际模型输入", "模型原始响应", "frames", "windows"}
        }
        # The group header is a bounded reading index. Full old payloads remain
        # in each interval's JSON, never expanded once more into every header.
        for key in list(narrative):
            if len(json.dumps(narrative[key], ensure_ascii=False, default=str)) > 8192:
                narrative[key] = "详见本区间的详细记录，原有内容完整保留。"
        narrative.update(
            开始时间=_time(segment["start_us"]),
            结束时间=_time(segment["end_us"]),
            状态说明=value.get("message") or status,
            阶段状态={"历史理解": status},
            处理原因="仅统一历史文件格式，未重新调用模型",
            原有步骤数量=len(value.get("experiment_steps") or []),
        )
        value.update(
            schema_version="visioncortex-readable-understanding/1",
            archive=self.layout.name,
            device=self.layout.name[11:],
            date=self.layout.name[:10],
            timezone="Asia/Shanghai",
            path_base="device_day",
            timestamp_unit="unix_microseconds",
            physical_action_confirmed=False,
            recording_id=segment["recording_id"],
            segment_id=segment["segment_id"],
            start_us=segment["start_us"],
            end_us=segment["end_us"],
            source_start_ms=segment["start_ms"],
            source_end_ms=segment["end_ms"],
            source_video=segment["source_ref"],
            video=segment.get("video"),
            activity=segment["activity"],
            status=status,
            video_status="available" if segment.get("video") else "not_materialized",
            record_scope="retention_interval",
            candidate_segment_id=segment.get("candidate_segment_id")
            or segment["segment_id"],
        )
        value["内容记录"] = narrative
        value["historical_format_migration"] = {
            "recipe": RECIPE,
            "model_invoked": False,
            "prior_document": relative,
            "prior_metadata": metadata,
            "classification_revalidated": False,
            "existing_video_selection_revalidated": False,
            "saved_content_preserved": bool(relative),
        }
        retention = {
            "schema_version": "visioncortex-derived-video-disposition/1",
            "source_ref": segment["source_ref"],
            "start_us": segment["start_us"],
            "end_us": segment["end_us"],
            "selection_interval": interval,
            "video": segment.get("video"),
            "no_value_proven": interval["video_value"] == "confirmed_no_value",
            "historical_format_migration": value["historical_format_migration"],
        }
        return value, retention

    @staticmethod
    def _entry(value, relative):
        return {
            **{
                k: value[k]
                for k in (
                    "recording_id",
                    "segment_id",
                    "start_us",
                    "end_us",
                    "activity",
                    "status",
                    "candidate_segment_id",
                    "record_scope",
                    "video_status",
                )
            },
            "开始时间": value["内容记录"]["开始时间"],
            "结束时间": value["内容记录"]["结束时间"],
        }

    def publish(self, *, maximum_new_videos=128, maximum_video_bytes=8 * 1024**3):
        from ..device_day import copy_verified
        from ..device_day_content import content_index

        # Discovery and its durable checkpoint precede this call. The native
        # retirement owner retains exact bytes and aliases, including sidecars
        # larger than its ordinary publication fast-path inspection cap.
        for folder in sorted(self.legacy_folders):
            if folder.is_dir():
                clip_delivery._retire(self.layout, folder)
        index_path = self.layout.understanding / "Index.json"
        if index_path.is_file():
            from ..device_day_contract import file_hash

            checksum = file_hash(index_path)
            backup = private_folder(
                self.layout,
                "MultimodalUnderstanding",
                "ArchiveUnification",
                "PriorIndexes",
                checksum + ".json",
            )
            if not backup.is_file():
                copy_verified(index_path, backup)
        entries, progress = clip_delivery.publish(
            self.config,
            self.index,
            [],
            document_builder=self._document,
            writer=atomic_json,
            entry_builder=self._entry,
            source_views=self.values,
            source_units=self.units,
            maximum_retired=20000,
            maximum_new_videos=maximum_new_videos,
            maximum_video_bytes=maximum_video_bytes,
        )
        atomic_json(
            index_path,
            content_index(self.layout.name, "MultimodalUnderstanding", entries),
        )
        return progress


def refresh_group_index(layout):
    """Rebuild derived counts from saved groups, without decoding media."""
    from ..device_day_content import content_index

    entries = []
    for folder in layout.understanding.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        header = bounded_document(folder / "Understanding.json")
        if header.get("schema_version") != clip_delivery.VERSION:
            raise ValueError("Ungrouped understanding remains during index refresh")
        for source in header.get("intervals", []):
            entry = dict(source)
            entry["candidate_segment_id"] = (
                entry.get("candidate_segment_id") or entry["segment_id"]
            )
            entries.append(entry)
    value = content_index(layout.name, "MultimodalUnderstanding", entries)
    path = layout.understanding / "Index.json"
    if not path.is_file() or bounded_document(path) != value:
        if path.is_file():
            from ..device_day import copy_verified
            from ..device_day_contract import file_hash

            backup = private_folder(
                layout,
                "MultimodalUnderstanding",
                "ArchiveUnification",
                "PriorIndexes",
                file_hash(path) + ".json",
            )
            if not backup.is_file():
                copy_verified(path, backup)
        atomic_json(path, value)
    return {
        "intervals": len(entries),
        "candidate_count": len({e["candidate_segment_id"] for e in entries}),
    }

"""Durable publication using the existing device layouts and format owners.

Originals remain at their producer paths; a bounded hash cache is never an archive.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

from psycopg.types.json import Jsonb

from .components import enable_components, pipeline_module
from .domain import Conflict, Missing, clean, ident
from .media import hash_file, safe_path
from .rules import Rules
from .storage import publication_config
from .archive_layout import day_directory


def encoded(value):
    return (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


class Retention:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings
        self._planning = None

    def enqueue(self, limit=100):
        with self.core.db.transaction() as conn:
            conn.execute("INSERT INTO archive_checkpoint(customer_id) VALUES(%s) ON CONFLICT DO NOTHING", (self.core.customer,))
            checkpoint = conn.execute("SELECT cursor FROM archive_checkpoint WHERE customer_id=%s FOR UPDATE", (self.core.customer,)).fetchone()
            events = conn.execute("SELECT seq,kind FROM outbox WHERE customer_id=%s AND seq>%s ORDER BY seq LIMIT %s", (self.core.customer, checkpoint["cursor"], limit)).fetchall()
            for event in events:
                conn.execute("INSERT INTO archive_publication(customer_id,outbox_seq,status) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING", (self.core.customer, event["seq"], "pending" if event["kind"] in {"capture", "observation"} else "not_applicable"))
            if events:
                conn.execute("UPDATE archive_checkpoint SET cursor=%s WHERE customer_id=%s", (events[-1]["seq"], self.core.customer))
        return len(events)

    def _write(self, relative, raw, *, immutable=False):
        if self._planning is not None:
            self._planning.append(dict(path=relative, text=raw.decode("utf-8")))
            if sum(len(p["text"].encode()) for p in self._planning) > 8 * 1024**2:
                raise ValueError("Publication snapshot exceeds the bounded JSON plan size")
            return dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        root = self.settings.incoming  # re-verifies any NAS mount/host marker
        parts = Path(relative).parts
        if "MetaData" in parts or "MetaVideo" in parts:
            raise ValueError("Publication cannot mutate producer-owned originals")
        if not parts or parts[0] != ".System" and (len(parts) < 3 or parts[1] not in {"ProcessedClips", "MultimodalUnderstanding", "LaboratoryDailyReport", "Comment", "VoicePhotoReadings", "VideoReadings", "PhotoReadings", "Bindings", "DailyReport"}):
            raise ValueError("Path is outside the frozen result directories")
        path = safe_path(root, relative, must_exist=False)
        if path.exists():
            before = path.read_bytes()
            if before == raw:
                return dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
            if immutable:
                raise Conflict("Immutable archive receipt differs from its original bytes")
        if not str(root).startswith("/mnt/"):
            total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and "MetaData" not in p.parts and "MetaVideo" not in p.parts)
            if total + len(raw) > self.settings.archive_max_bytes or shutil.disk_usage(root).free < 80 * 1024**3:
                raise Conflict("Bounded local retention capacity reached; publication remains pending")
        guard = pipeline_module("device_day_write_guard")
        if guard.guarded_atomic(path, [raw]):
            digest, size = hash_file(path, max(self.settings.max_capture_bytes, len(raw)))
            if digest != hashlib.sha256(raw).hexdigest() or size != len(raw):
                raise Conflict("Published NAS artifact failed read-back verification")
            return dict(path=relative, sha256=digest, size_bytes=size)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + str(ident()) + ".partial")
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            # Recheck before rename so an unmounted NAS cannot become a local output root.
            self.settings.incoming
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        digest, size = hash_file(path, max(self.settings.max_capture_bytes, len(raw)))
        if digest != hashlib.sha256(raw).hexdigest() or size != len(raw):
            raise Conflict("Published archive artifact failed read-back verification")
        return dict(path=relative, sha256=digest, size_bytes=size)

    def _source(self, conn, event):
        if event["kind"] == "capture":
            capture_id = event["record_id"]
        else:
            job = conn.execute("SELECT capture_id FROM processing_job WHERE customer_id=%s AND id=%s", (self.core.customer, event["payload"].get("job_id"))).fetchone() if event["payload"].get("job_id") else None
            if not job:
                return None
            capture_id = job["capture_id"]
        return conn.execute("SELECT c.*,i.payload AS receipt,a.id AS media_id,a.verified_sha256,a.byte_size FROM capture c JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id WHERE c.customer_id=%s AND c.id=%s", (self.core.customer, capture_id)).fetchone()

    def _photo(self, conn, source, event):
        enable_components()
        from field_recognition.archive.archive_browse import token
        from field_recognition.measurements.measurement_records import RULE_VERSION
        from field_recognition.archive.archive_events import standard_records, photo_candidates
        from field_recognition.measurements.result_reprocessing import preferred_jobs
        receipt = source["receipt"]
        camera = receipt.get("camera_id") or receipt["source_instance"]
        category = 'VideoReadings' if receipt.get('video_observation') else 'VoicePhotoReadings'
        key = "photo:" + str(source["source_id"]) + ":" + (receipt.get("session_id") or receipt["capture_id"])
        # A producer's explicit burst is the only grouping authority.
        members = conn.execute("SELECT c.*,i.payload AS receipt FROM capture c JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision WHERE c.customer_id=%s AND c.source_id=%s AND (c.id=%s OR (%s::uuid IS NOT NULL AND c.session_id=%s)) ORDER BY c.received_at,c.id", (self.core.customer, source["source_id"], source["id"], source["session_id"], source["session_id"])).fetchall()
        at = min((m["occurred_at"] for m in members if m["occurred_at"]), default=None)
        moment = (at or event["created_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
        stamp = moment.strftime("%H-%M-%S.%f")[:-3] if at else "UnknownTime"
        prior = conn.execute("SELECT * FROM archive_event WHERE customer_id=%s AND event_key=%s FOR UPDATE", (self.core.customer, key)).fetchone()
        directory = prior['directory'] if prior else day_directory(camera, at or event['created_at']) + '/Comment/OCR/' + category + '/' + stamp + '_' + token(key)
        slots = prior["result_files"] if prior else {}
        ids = [m["id"] for m in members]
        jobs = conn.execute("SELECT * FROM processing_job WHERE customer_id=%s AND capture_id=ANY(%s) ORDER BY created_at,id", (self.core.customer, ids)).fetchall()
        observations = [dict((j["result"] or {}).get("field_document", {}), job_id=str(j["id"]), capture_id=str(j["capture_id"]), status=j["status"], model_revision=j["model_revision"], processing_rules=j["rule_identity"], raw_processing_result=j["result"]) for j in jobs]
        files, measurement_files = [], []
        # Raw unbound OCR observations cannot become an attributed instrument Result.
        # The fixed original formatter owns the six-field contract and conservative null rules.
        documents = [{"seq": n, "entity_id": str(job["id"]), "doc": (job["result"] or {})["field_document"]} for n, job in enumerate(jobs) if (job["result"] or {}).get("field_document")]
        for entry in standard_records(documents, None):
                iid = entry["instrument_id"]
                self.core._entity(conn, iid, "instrument")
                if iid not in slots:
                    slots[iid] = directory + ("/Result.json" if not slots else f"/Result{len(slots)+1:02d}.json")
                item = self._write(slots[iid], encoded(entry["record"]))
                files.append(item)
                measurement_files.append(dict(instrument_id=iid, result=item["path"], sha256=item["sha256"], evidence=entry["evidence"]))
        originals = []
        for member in members:
            row = conn.execute("SELECT id,verified_sha256,byte_size FROM media_asset WHERE customer_id=%s AND capture_id=%s", (self.core.customer, member["id"])).fetchone()
            path = safe_path(self.settings.incoming, member["receipt"]["relative_path"])
            digest, size = hash_file(path, self.settings.max_capture_bytes)
            if digest != row["verified_sha256"] or size != row["byte_size"]:
                raise Conflict("Producer photograph changed before retention")
            original = dict(path=member["receipt"]["relative_path"], sha256=digest, size_bytes=size, source_owned=True)
            originals.append(dict(capture_id=str(member["id"]), media_id=str(row["id"]), **original, original=original, image=original, captured_at=clean(member["occurred_at"]), capture_time_basis="source_capture_time" if member["occurred_at"] else None, received_at=clean(member["received_at"]), external_photo={"captured_at": clean(member["occurred_at"]), "source_ref": member["receipt"]["relative_path"]}, camera_id=camera, operator=None, qr_matches=[], scene_matches=[]))
        provenance = dict(rule_version=RULE_VERSION, processing_rules=Rules().identity(), source_kind=source["source_kind"], database_customer_id=str(self.core.customer))
        current_jobs = preferred_jobs(observations)
        candidates = photo_candidates(current_jobs, originals, unbound_only=False)
        reading_keys = ("capture_id", "captured_at", "capture_time_basis", "observed_at", "job_id", "reading_id", "instrument_id", "binding_id", "identity_basis", "measurement_name", "value", "display_value", "unit", "quality_issues", "result_stage", "result_revision", "parallel_review")
        readings = {"schema": "field-numeric-observations/1", "event_id": key, "camera_id": camera, "record_scope": "field_observation", "human_verified": False, "aggregation": "per_capture_not_merged", "readings": [{k: r.get(k) for k in reading_keys} for r in candidates], "evidence": "Evidence.json", **provenance}
        files.append(self._write(directory + "/Readings.json", encoded(readings)))
        history = conn.execute("SELECT outbox_seq AS sequence,receipt_path AS receipt FROM archive_publication WHERE customer_id=%s AND event_key=%s AND status='readable' ORDER BY outbox_seq", (self.core.customer, key)).fetchall()
        evidence = {"schema": "field-recognition-event/2", "derived_view": True, "event_id": key, "event_at": clean(at), "time_basis": "source_capture_time" if at else None, "folder_time_basis": "source_capture_time" if at else "receipt_date_capture_time_unknown", "camera_id": camera, "operator": None, "category": category, "status": "processing" if any(j["status"] in {"queued", "running"} for j in jobs) else "observations_retained", "sources": originals, "photos": {r["sha256"]: r["original"] for r in originals}, "measurement_files": measurement_files, "observations": observations, "record_scope": "candidate", "decision": None, "receipt_versions": history, "archive_sequence": event["seq"], **provenance}
        expected = max((m['receipt'].get('group_expected_photos', 1) or 1 for m in members), default=1)
        evidence['photo_collection'] = dict(expected=expected, received=len(members), status='complete' if len(members)>=expected else 'partial')
        files.append(self._write(directory + "/Evidence.json", encoded(evidence)))
        if not measurement_files:
            files.append(self._write(directory + "/UnboundReadings.json", encoded({"schema": "field-unbound-readings/1", "derived_view": True, "title": "未绑定照片读数候选（待核实，非正式测量结果）", "event_id": key, "camera_id": camera, "instrument_id": None, "qr_hash": None, "human_verified": False, "record_scope": "unbound_candidates", "status": "needs_review", "aggregation": "per_photo_candidates_not_merged", "readings": photo_candidates(current_jobs, originals), "evidence": directory + "/Evidence.json", "receipt_versions": history, **provenance})))
        conn.execute("INSERT INTO archive_event(customer_id,event_key,directory,result_files,last_seq) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(customer_id,event_key) DO UPDATE SET result_files=EXCLUDED.result_files,last_seq=EXCLUDED.last_seq", (self.core.customer, key, directory, Jsonb(slots), event["seq"]))
        return key, directory, files

    def _native(self, conn, source, event):
        receipt = source["receipt"]
        record = receipt.get("native_record")
        if not record:
            return None  # no native identity/time/layout may be invented for generic media
        contract = pipeline_module("device_day_contract")
        layout = contract.DeviceDayLayout(self.settings.incoming, record["camera_key"], record["recording_start_us"], self.settings.cache / "native-receipts", raw_directory=next(p for p in Path(receipt["relative_path"]).parts if p in {"MetaData", "MetaVideo"}), read_only_originals=True)
        retained = pipeline_module("device_day").retained_recording(layout, record, {"collection_ingest": {"source_root": str(self.settings.incoming), "settle_seconds": .5}, "device_day": {"raw_originals_read_only": True}})
        key = "native:" + str(source["id"])
        directory = layout.name + "/ProcessedClips/.Evidence/Retention/" + record["recording_id"]
        files = [self._write(directory + "/Retention.json", encoded(retained))]
        if source["kind"] == "audio":
            job = conn.execute("SELECT result FROM processing_job WHERE customer_id=%s AND capture_id=%s AND worker_kind='asr' AND status='completed' ORDER BY processed_at DESC,id DESC LIMIT 1", (self.core.customer, source["id"])).fetchone()
            if job and source["occurred_at"]:
                content = pipeline_module("device_day_content")
                result = job["result"]
                start = int(source["occurred_at"].timestamp() * 1000000)
                end = start + round(float(receipt["end_seconds"]) * 1000000)
                comments = [dict(comment_id=f"{source['id']}:{n}", text=s["text"], start_us=start + round(float(s["start_seconds"]) * 1000000), end_us=start + round(float(s["end_seconds"]) * 1000000)) for n, s in enumerate(result.get("segments", []))]
                record_view = dict(recording_id=record["recording_id"], start_us=start, end_us=end, audio=retained["audio"], transcription=dict(status="completed", outcome="transcribed" if comments else "no_transcript", comments=comments, model_invocation="PROVEN" if result.get("execution_receipts") else "NOT_PROVEN"))
                value, text = content.transcript_documents(layout.name, record_view)
                folder = layout.name + "/Comment/Audio/" + content.time_folder(start, end)
                files.extend([self._write(folder + "/Transcript.json", encoded(value)), self._write(folder + "/Transcript.txt", text)])
        # Screening and value selection own clip export. Retention cannot create MP4s.
        return key, directory, files

    def tick(self):
        guard = pipeline_module("device_day_write_guard")
        try:
            with guard.storage_guard(publication_config(self.settings)):
                return self._tick()
        except OSError as exc:
            return dict(enqueued=0, published=0, status="needs_attention", error_code=type(exc).__name__)

    def _tick(self):
        enqueued = self.enqueue()
        lock_path = self.settings.runtime_root / "archive-publication.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(enqueued=enqueued, published=0, status="running_elsewhere")
            row = None
            try:
                # Commit a frozen plan before filesystem writes. A retry/restart replays exact bytes.
                with self.core.db.transaction() as conn:
                    row = conn.execute("SELECT p.*,o.kind,o.record_id,o.payload,o.created_at,o.seq FROM archive_publication p JOIN outbox o ON o.seq=p.outbox_seq WHERE p.customer_id=%s AND p.status IN ('pending','needs_attention') AND p.next_attempt_at<=now() ORDER BY p.outbox_seq FOR UPDATE OF p SKIP LOCKED LIMIT 1", (self.core.customer,)).fetchone()
                    if not row:
                        return dict(enqueued=enqueued, published=0)
                    plan = row["plan"]
                    if not plan:
                        source = self._source(conn, row)
                        self._planning = []
                        try:
                            result = self._photo(conn, source, row) if source and source["kind"] == "photo" else self._native(conn, source, row) if source else None
                            documents = self._planning
                        finally:
                            self._planning = None
                        if not result:
                            conn.execute("UPDATE archive_publication SET status='not_applicable' WHERE customer_id=%s AND outbox_seq=%s", (self.core.customer, row["seq"]))
                            return dict(enqueued=enqueued, published=0, status="no_native_archive_contract")
                        key, directory, files = result
                        plan = clean(dict(key=key, directory=directory, files=files, documents=documents, source_kind=source["source_kind"], camera=source["receipt"].get("camera_id") or source["receipt"]["source_instance"], rules=Rules().identity()))
                        conn.execute("UPDATE archive_publication SET plan=%s,event_key=%s,directory=%s WHERE customer_id=%s AND outbox_seq=%s", (Jsonb(plan), key, directory, self.core.customer, row["seq"]))
                # Each version gets immutable JSON evidence; latest views may be repaired from it.
                artifacts = {}
                for n, document in enumerate(plan["documents"]):
                    raw = document["text"].encode()
                    digest = hashlib.sha256(raw).hexdigest()
                    version = f".System/Versions/{self.core.customer}/{row['seq']:012d}/{digest}.json"
                    artifacts[str(n)] = self._write(version, raw, immutable=True)
                with self.core.db.transaction(readonly=True) as conn:
                    newer = conn.execute("SELECT max(outbox_seq) AS seq FROM archive_publication WHERE customer_id=%s AND event_key=%s AND status='readable'", (self.core.customer, plan["key"])).fetchone()["seq"]
                if newer is None or newer <= row["seq"]:
                    for document in plan["documents"]:
                        self._write(document["path"], document["text"].encode())
                enable_components()
                from field_recognition.archive.archive_store import canonical
                from field_recognition.archive.archive_browse import token
                body = dict(schema="field-recognition-receipt/1" if plan["key"].startswith("photo:") else "vc-lab-native-publication-receipt/1", source_instance=str(self.core.customer), sequence=row["seq"], entity="vc_lab_" + row["kind"], entity_id=str(row["record_id"]), recorded_at=clean(row["created_at"]), document=dict(directory=plan["directory"], views=plan["files"], processing_rules=plan["rules"], source_kind=plan["source_kind"]), artifacts=artifacts, physical_action_confirmed=False)
                raw = canonical(body)
                digest = hashlib.sha256(raw).hexdigest()
                path = f".System/Receipts/{token(plan['camera'])}/{row['created_at']:%Y-%m-%d}/{self.core.customer}/{row['seq']:012d}-{digest}.json"
                self._write(path, raw, immutable=True)
                with self.core.db.transaction() as conn:
                    conn.execute("UPDATE archive_publication SET status='readable',attempts=attempts+1,receipt_path=%s,files=%s,error_code=NULL,published_at=now(),readable_at=now() WHERE customer_id=%s AND outbox_seq=%s", (path, Jsonb(plan["files"]), self.core.customer, row["seq"]))
                return dict(enqueued=enqueued, published=1, directory=plan["directory"], sequence=row["seq"])
            except (OSError, ValueError, Missing) as exc:
                if row:
                    with self.core.db.transaction() as conn:
                        conn.execute("UPDATE archive_publication SET status='needs_attention',attempts=attempts+1,error_code=%s,next_attempt_at=now()+interval '30 seconds' WHERE customer_id=%s AND outbox_seq=%s", (type(exc).__name__, self.core.customer, row["seq"]))
                return dict(enqueued=enqueued, published=0, status="needs_attention", error_code=type(exc).__name__, reason=str(exc)[:200])

    def status(self):
        with self.core.db.transaction(readonly=True) as conn:
            rows = conn.execute("SELECT status,count(*) AS count FROM archive_publication WHERE customer_id=%s GROUP BY status", (self.core.customer,)).fetchall()
        return clean(rows)

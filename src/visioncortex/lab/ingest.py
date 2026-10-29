from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from psycopg.types.json import Jsonb

from .components import ROOT, inspect_recorder, enable_components
from .domain import Conflict, clean, fingerprint, ident, instant
from .media import Media, safe_path
from .rules import Rules
from .subprocesses import run_bounded
from .input_layout import photo_location, require_finished_upload, upload_markers


def read_json(path, limit=2 * 1024**2):
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Receipt is a symlink or exceeds its size limit")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Receipt must be an object")
    return value


class Ingest:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings
        self.media = Media(core, settings)

    def source(self, conn, name, revision, kind):
        row = conn.execute("INSERT INTO source_instance(customer_id,id,name,revision,kind) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(customer_id,name) DO NOTHING RETURNING *", (self.core.customer, ident(), name, revision, kind)).fetchone()
        return row or conn.execute("SELECT * FROM source_instance WHERE customer_id=%s AND name=%s", (self.core.customer, name)).fetchone()

    def accept(self, receipt, *, native_original=False, precomputed_ocr=None):
        required = {"capture_id", "source_instance", "source_revision", "kind", "relative_path", "time_basis", "source_kind"}
        if not required.issubset(receipt):
            raise ValueError("Capture receipt is missing required identities or provenance")
        kind = receipt["kind"]
        if kind not in {"photo", "video", "audio"}:
            raise ValueError("Unsupported capture kind")
        source_kind = receipt["source_kind"]
        if source_kind not in {"customer_capture", "fixture", "derived_replay", "synthetic_silence"}:
            raise ValueError("Explicit source category required")
        with self.core.db.transaction(readonly=True) as conn:
            scope = conn.execute("SELECT data_scope FROM customer_instance WHERE id=%s", (self.core.customer,)).fetchone()["data_scope"]
        if scope == "customer" and source_kind != "customer_capture":
            raise Conflict("Fixture/replay data requires its own explicit non-customer namespace")
        root = self.settings.incoming
        require_finished_upload(root, safe_path(root, receipt['relative_path']))
        if native_original and (not receipt.get('native_record') or not any(p in {'MetaData', 'MetaVideo'} for p in Path(receipt['relative_path']).parts)):
            raise ValueError('In-place reading requires a finalized native recording')
        copied = self.media.original_file(receipt['relative_path'], receipt.get('sha256')) if native_original else self.media.cache_file(receipt["relative_path"], receipt.get("sha256"))
        digest = fingerprint(receipt)
        at = instant(receipt["captured_at"]) if receipt.get("captured_at") else None
        event = str(receipt["capture_id"])
        revision = str(receipt["source_revision"])
        rule_identity = Rules().identity()
        with self.core.db.transaction() as conn:
            source = self.source(conn, receipt["source_instance"], revision, source_kind)
            if kind == 'photo':
                conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (f"photo:{self.core.customer}:{source['id']}:{receipt['relative_path']}",))
                prior_photo = conn.execute("SELECT c.source_capture_id,c.source_revision FROM capture c JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id JOIN media_location l ON l.customer_id=a.customer_id AND l.media_id=a.id WHERE c.customer_id=%s AND c.source_id=%s AND c.kind='photo' AND l.storage='incoming' AND l.relative_path=%s", (self.core.customer, source['id'], receipt['relative_path'])).fetchone()
                if prior_photo and (prior_photo['source_capture_id'], prior_photo['source_revision']) != (event, revision):
                    raise Conflict('Existing photograph cannot be re-imported with a new capture identity')
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (f"capture:{self.core.customer}:{source['id']}:{event}:{revision}",))
            old = conn.execute("SELECT * FROM inbox WHERE customer_id=%s AND source_id=%s AND event_id=%s AND revision=%s", (self.core.customer, source["id"], event, revision)).fetchone()
            if old:
                # A different physical photo path may refer to the same stable
                # capture. Keep all other provenance/metadata immutable.
                old_identity = {k: v for k, v in old['payload'].items() if k != 'relative_path'}
                new_identity = {k: v for k, v in receipt.items() if k != 'relative_path'}
                same_photo = kind == 'photo' and fingerprint(old_identity) == fingerprint(new_identity)
                if old["fingerprint"] != digest and not same_photo:
                    raise Conflict("Same source capture revision has different content/metadata")
                capture = conn.execute("SELECT * FROM capture WHERE customer_id=%s AND source_id=%s AND source_capture_id=%s AND source_revision=%s", (self.core.customer, source["id"], event, revision)).fetchone()
                media = conn.execute('SELECT id,verified_sha256,byte_size FROM media_asset WHERE customer_id=%s AND capture_id=%s', (self.core.customer, capture['id'])).fetchone()
                if (media['verified_sha256'], media['byte_size']) != (copied['sha256'], copied['byte_size']):
                    raise Conflict('Same source capture revision now contains different media bytes')
                if same_photo:
                    conn.execute("INSERT INTO media_location(customer_id,id,media_id,storage,relative_path,availability) VALUES(%s,%s,%s,'incoming',%s,'available') ON CONFLICT DO NOTHING", (self.core.customer, ident(), media['id'], receipt['relative_path']))
                return clean(dict(capture_id=capture["id"], duplicate=True))
            session_id = None
            if receipt.get("session_id"):
                sid = str(receipt["session_id"])
                session = conn.execute("INSERT INTO capture_session(customer_id,id,source_id,source_session_id,metadata) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(customer_id,source_id,source_session_id) DO UPDATE SET source_session_id=EXCLUDED.source_session_id RETURNING id", (self.core.customer, ident(), source["id"], sid, Jsonb({"camera_id": receipt.get("camera_id"), 'expected_photos': receipt.get('group_expected_photos'), 'batch_basis': receipt.get('batch_basis')}))).fetchone()
                session_id = session["id"]
            capture_id, media_id, evidence_id = ident(), ident(), ident()
            conn.execute("INSERT INTO capture(customer_id,id,source_id,source_capture_id,source_revision,session_id,kind,occurred_at,captured_at,time_basis,source_kind) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (self.core.customer, capture_id, source["id"], event, revision, session_id, kind, at, at, Jsonb(clean(receipt["time_basis"])), source_kind))
            conn.execute("INSERT INTO media_asset(customer_id,id,capture_id,kind,declared_sha256,verified_sha256,byte_size,evidence_status,verified_at) VALUES(%s,%s,%s,%s,%s,%s,%s,'available',now())", (self.core.customer, media_id, capture_id, kind, receipt.get("sha256"), copied["sha256"], copied["byte_size"]))
            if copied['storage'] == 'cache':
                conn.execute("INSERT INTO media_location(customer_id,id,media_id,storage,relative_path,availability) VALUES(%s,%s,%s,'cache',%s,'available')", (self.core.customer, ident(), media_id, copied["relative_path"]))
            conn.execute("INSERT INTO evidence_ref(customer_id,id,media_id,media_version,start_seconds,end_seconds) VALUES(%s,%s,%s,1,%s,%s)", (self.core.customer, evidence_id, media_id, receipt.get("start_seconds"), receipt.get("end_seconds")))
            jobs = []
            workers={"photo": ["qr", "ocr"], "video": ["vision"], "audio": ["asr"]}[kind]
            if self.settings.processing_profile == 'timeline-v1':
                workers = ['qr'] if kind == 'photo' else []
            if self.settings.processing_profile == 'native' and kind=='video' and receipt.get('native_record',{}).get('embedded_audio'):
                workers.append('asr')
            for worker in workers:
                jid = ident()
                model_revision = "model-free-qr/1" if worker == "qr" else "customer-model-unconfigured"
                if worker!='qr':
                    from .processing import ModelRegistry, ModelUnavailable
                    try:
                        model_revision=ModelRegistry(self.settings).descriptor(worker)['revision']
                    except ModelUnavailable:
                        pass
                conn.execute("INSERT INTO processing_job(customer_id,id,capture_id,worker_kind,model_revision,rule_identity,rule_fingerprint) VALUES(%s,%s,%s,%s,%s,%s,%s)", (self.core.customer, jid, capture_id, worker, model_revision, Jsonb(rule_identity), rule_identity["sha256"]))
                if worker == 'ocr' and precomputed_ocr is not None:
                    if precomputed_ocr['revision'] != model_revision or precomputed_ocr['image_sha256'] != copied['sha256'] or precomputed_ocr['rule_fingerprint'] != rule_identity['sha256']:
                        raise Conflict('Resident OCR identity differs from the admitted frame/model/rules')
                    result = clean(precomputed_ocr['result'])
                    result['field_document'].update(job_id=str(jid), capture_id=str(capture_id))
                    enable_components()
                    from field_recognition.measurements.reading_results import build_readings
                    result['field_document']['readings'] = build_readings(result['field_document'])
                    result['readings'] = result['field_document']['readings']
                    conn.execute('INSERT INTO processing_precomputed(customer_id,job_id,model_identity,rule_fingerprint,image_sha256,result) VALUES(%s,%s,%s,%s,%s,%s)', (self.core.customer, jid, precomputed_ocr['model_identity'], rule_identity['sha256'], copied['sha256'], Jsonb(result)))
                jobs.append(jid)
            conn.execute("INSERT INTO inbox(customer_id,source_id,event_id,revision,fingerprint,payload) VALUES(%s,%s,%s,%s,%s,%s)", (self.core.customer, source["id"], event, revision, digest, Jsonb(clean(receipt))))
            conn.execute("INSERT INTO media_location(customer_id,id,media_id,storage,relative_path,availability) VALUES(%s,%s,%s,'incoming',%s,'available')", (self.core.customer, ident(), media_id, receipt["relative_path"]))
            result = dict(capture_id=capture_id, media_id=media_id, evidence_id=evidence_id, job_ids=jobs, source_kind=source_kind, duplicate=False)
            if self.settings.processing_profile == 'timeline-v1':
                from .timeline import Timeline
                admitted = Timeline(self.core).admit(conn, capture_id, media_id, receipt)
                from .capture_equipment_intake import admit as admit_equipment_qr
                original_path = safe_path(self.settings.incoming if copied['storage'] == 'incoming' else self.settings.cache,
                                          copied['relative_path'])
                authorization = admit_equipment_qr(conn, self.core, self.settings, admitted, receipt,
                                                   verified_path=original_path)
                if authorization is not None:
                    result['automatic_equipment_qr'] = authorization
                result['processing_profile'] = 'timeline-v1'
            self.core._emit(conn, "capture", capture_id, 1, result)
            return clean(result)

    def photo_receipt(self, path):
        safe_path(self.settings.incoming, path.relative_to(self.settings.incoming).as_posix())
        require_finished_upload(self.settings.incoming, path)
        data = read_json(path)
        if data.get("schema_version") != "field-photo-receipt/1":
            raise ValueError("Only explicit PhotoReceipt groups are admitted")
        photos = data.get("photos", [])
        measurement = data.get("measurement", {})
        if not photos or len(photos) > 16 or measurement.get("expected_photos") != len(photos):
            raise ValueError("PhotoReceipt membership/count mismatch")
        results = []
        for photo in photos:
            filename = photo["filename"]
            if Path(filename).name != filename:
                raise ValueError("Photo receipt member must be in its own capture directory")
            relative = (path.parent / filename).relative_to(self.settings.incoming).as_posix()
            group = dict(group_expected_photos=measurement['expected_photos'], batch_basis='producer_photo_receipt') if self.settings.processing_profile == 'timeline-v1' else {}
            results.append(self.accept(dict(capture_id=photo["capture_id"], source_instance="photo:" + data["camera_id"], source_revision=str(photo.get("revision", "1")), camera_id=data["camera_id"], session_id=measurement.get("burst_id"), kind="photo", relative_path=relative, captured_at=photo["captured_at"], sha256=photo["sha256"], time_basis={"basis": "producer_photo_receipt", "raw": photo["captured_at"]}, source_kind=data.get("source_kind", self.settings.source_kind), **group)))
        return results

    def native_receipt(self, path):
        require_finished_upload(self.settings.incoming, path)
        video = path.with_name(path.name.replace("recording_ready.json", "rgb.mp4"))
        safe_path(self.settings.incoming, video.relative_to(self.settings.incoming).as_posix())
        record = inspect_recorder(self.settings.incoming, video, time.time(), settle=0.5)
        if not record["processable"]:
            raise Conflict("Recorder input is not finalized: " + ";".join(record["issues"]))
        if (record.get('audio') or {}).get('status')!='provided':
            from .components import pipeline_module
            record['embedded_audio']=pipeline_module('speech').probe_audio(video) is not None
        receipt = dict(capture_id=record["recording_id"], source_instance="recorder:" + record["camera_key"], source_revision=record["source_signature"], camera_id=record["camera_key"], session_id=record["recording_session_id"], kind="video", relative_path=record["relative_path"], captured_at=record["recording_start_time"], time_basis={"basis": "recorder_native", "raw_start": record["recording_start_us"], "raw_end": record["recording_end_us"], "unit": "epoch_us", "capture_complete": record["capture_complete"]}, start_seconds="0", end_seconds=str(record["duration_seconds"]), source_kind=self.settings.source_kind)
        receipt["native_record"] = record
        results = [self.accept(receipt, native_original=True)]
        audio = record.get("audio", {})
        if audio.get("status") == "provided" and audio.get("capture_complete") and audio.get("start_us") and audio.get("end_us"):
            audio_path = next(Path(f["path"]) for f in audio["files"] if f["kind"] == "audio")
            results.append(self.accept(dict(receipt, capture_id=record["recording_id"] + ":audio", source_revision=audio["source_signature"], kind="audio", relative_path=audio_path.relative_to(self.settings.incoming).as_posix(), captured_at=datetime.fromtimestamp(audio["start_us"] / 1e6, timezone.utc).isoformat(), end_seconds=str((audio["end_us"] - audio["start_us"]) / 1e6), source_kind=audio.get("artifact_kind", self.settings.source_kind), time_basis={"basis": audio["time_basis"], "raw_start": audio["start_us"], "unit": "epoch_us"}), native_original=True))
        return results

    def directory_photo(self, path):
        from .capture_equipment_intake import policy
        configured = policy(self.settings)
        automatic = self.settings.processing_profile == 'timeline-v1' and configured.get('enabled') and configured.get('photo')
        if not automatic and self.settings.photo_batch_mode != 'directory_triplet':
            raise ValueError('Bare directories require an explicit directory_triplet configuration')
        require_finished_upload(self.settings.incoming, path)
        location = photo_location(path.relative_to(self.settings.incoming).as_posix())
        if not automatic and location.camera not in self.settings.photo_cameras:
            raise ValueError('Photo camera is not explicitly registered for directory intake')
        env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LD_LIBRARY_PATH', 'VC_LAB_NAS_MARKER_SHA256', 'VC_LAB_STORAGE_MANIFEST'}}
        env['FIELD_DEMO_DATA'] = str(self.settings.runtime_root / 'field-context')
        request = dict(root=str(self.settings.incoming), path=str(path), camera=location.camera, zone=self.settings.photo_timezone, max_bytes=min(self.settings.max_capture_bytes, 16*1024**2), batch_mode='directory_triplet')
        # Natural capture mode reuses FR's original stable reader/path grammar.
        # A directory is not proof of a planned three-shot burst. More than three
        # members are read individually with no invented grouping/context.
        members = [p for p in path.parent.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}] if automatic else []
        if automatic and len(members) > 3 and not (path.parent / 'PhotoReceipt.json').exists():
            if time.time() - path.stat().st_mtime < .5:
                raise Conflict('Bare photograph is still within the original FR stable-write wait')
            request.update(batch_mode='receipt_only', require_receipt=False)
        loaded = json.loads(run_bounded([sys.executable, '-B', str(ROOT / 'src/visioncortex/lab/field_source.py')], json.dumps(request), env=env, timeout=8))
        if loaded.get('error'):
            raise Conflict('Original FR directory admission: ' + loaded['error'] + ': ' + loaded.get('reason', ''))
        group = loaded['receipt']
        if automatic and loaded.get('receipt_basis') != 'nas_photo_receipt':
            if group is None:
                from .media import hash_file
                sha, _ = hash_file(path, min(self.settings.max_capture_bytes, 16*1024**2))
                stamp = datetime.strptime(Path(location.filename).stem[:15], '%Y%m%d_%H%M%S')
                from zoneinfo import ZoneInfo
                group = dict(photos=[dict(filename=path.name, capture_id=location.identity,
                    captured_at=stamp.replace(tzinfo=ZoneInfo(self.settings.photo_timezone)).isoformat(), sha256=sha)])
            results = []
            for photo in group['photos']:
                receipt = dict(capture_id=photo['capture_id'], source_instance='photo:' + location.camera,
                    source_revision='1', camera_id=location.camera, session_id=None, kind='photo',
                    relative_path=(path.parent / photo['filename']).relative_to(self.settings.incoming).as_posix(),
                    captured_at=photo['captured_at'], sha256=photo['sha256'], source_kind=self.settings.source_kind,
                    group_expected_photos=None, batch_basis='directory_membership_unknown_without_producer_receipt',
                    time_basis=dict(basis='nas_filename_local_time_not_hardware_verified', raw=photo['captured_at'],
                                    clock_sync_valid=False))
                with self.core.db.transaction(readonly=True) as conn:
                    old = conn.execute('''SELECT i.payload FROM inbox i JOIN source_instance s
                        ON s.customer_id=i.customer_id AND s.id=i.source_id
                        WHERE i.customer_id=%s AND s.name=%s AND i.event_id=%s AND i.revision='1' ''',
                        (self.core.customer, receipt['source_instance'], receipt['capture_id'])).fetchone()
                if old:
                    receipt = dict(old['payload'], relative_path=receipt['relative_path'])
                results.append(self.accept(receipt))
            return results
        if loaded.get('receipt_basis') != 'configured_directory_triplet':
            return self.photo_receipt(path.parent / 'PhotoReceipt.json')
        results = []
        for photo in group['photos']:
            results.append(self.accept(dict(capture_id=photo['capture_id'], source_instance='photo:' + location.camera,
                source_revision='1', camera_id=location.camera, session_id=group['measurement']['burst_id'],
                kind='photo', relative_path=(path.parent / photo['filename']).relative_to(self.settings.incoming).as_posix(),
                captured_at=photo['captured_at'], sha256=photo['sha256'], source_kind=self.settings.source_kind,
                group_expected_photos=3, batch_basis='configured_directory_triplet',
                time_basis={'basis':'nas_filename_local_time_not_hardware_verified', 'raw': photo['captured_at']})))
        return results

    def discover_once(self, limit=100):
        files = []
        scanned = 0
        truncated = False
        excluded = {'示例_留存格式', '示例_真实设备一天', '.System', 'ProcessedClips', 'MultimodalUnderstanding', 'LaboratoryDailyReport', 'Comment', 'VoicePhotoReadings', 'VideoReadings', 'PhotoReadings', 'Bindings', 'DailyReport'}
        root = self.settings.incoming
        from .capture_equipment_intake import newest_candidates, policy
        automatic = self.settings.processing_profile == 'timeline-v1' and policy(self.settings).get('enabled')
        if automatic:
            files, truncated = newest_candidates(root, excluded, include_bare_photos=policy(self.settings)['photo'])
        for directory, dirs, names in ([] if automatic else os.walk(root, followlinks=False)):
            dirs[:] = sorted(d for d in dirs if d not in excluded and not Path(directory, d).is_symlink())
            for name in sorted(names):
                path = Path(directory, name)
                candidate = name in {'CaptureReceipt.json','PhotoReceipt.json'} or name.endswith('recording_ready.json')
                if not candidate and self.settings.photo_batch_mode == 'directory_triplet' and path.suffix.lower() in {'.png','.jpg','.jpeg'} and 'PhotoReceipt.json' not in names:
                    try:
                        candidate = photo_location(path.relative_to(root).as_posix()).camera in self.settings.photo_cameras
                    except ValueError:
                        pass
                if candidate:
                    files.append(path)
                    scanned += 1
                    if scanned >= 2000:
                        truncated = True
                        break
            if truncated:
                break
        if automatic:
            # NAS receive mtime is a scheduling hint, never an event clock; keep
            # the validated native day/path order ahead of older backfill.
            files.sort(key=lambda p: (p.relative_to(root).parts[0], p.parent.as_posix(), p.name), reverse=True)
        else:
            files.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        summary = {"processed": 0, "skipped": 0, "errors": [], "coverage": "partial" if truncated or len(files) > limit else "unknown"}
        for path in files:
            if summary['processed'] + len(summary['errors']) >= limit:
                summary['coverage'] = 'partial'
                break
            relative = path.relative_to(self.settings.incoming).as_posix()
            signature = fingerprint({"receipt": path.stat().st_mtime_ns, "members": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.parent.iterdir()) if p.is_file() and not p.is_symlink()], 'uploading': upload_markers(root, path)})
            with self.core.db.transaction(readonly=True) as conn:
                prior = conn.execute("SELECT * FROM intake_item WHERE customer_id=%s AND relative_path=%s", (self.core.customer, relative)).fetchone()
            if prior and prior["signature"] == signature and prior["status"] == "accepted":
                summary["skipped"] += 1
                continue
            if prior and prior['signature'] == signature and prior['status'] == 'needs_attention' and (datetime.now(timezone.utc) - prior['checked_at']).total_seconds() < 30:
                continue
            try:
                safe_path(self.settings.incoming, relative)
                require_finished_upload(root, path)
                result = self.photo_receipt(path) if path.name == "PhotoReceipt.json" else self.native_receipt(path) if path.name.endswith("recording_ready.json") else self.accept(read_json(path)) if path.name == 'CaptureReceipt.json' else self.directory_photo(path)
                status, error = "accepted", None
                summary["processed"] += 1
            except Exception as exc:
                result, status, error = None, "needs_attention", type(exc).__name__
                summary["errors"].append({"relative_path": relative, "reason": error})
            with self.core.db.transaction() as conn:
                conn.execute("INSERT INTO intake_item(customer_id,relative_path,signature,status,error_code,result) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(customer_id,relative_path) DO UPDATE SET signature=EXCLUDED.signature,status=EXCLUDED.status,error_code=EXCLUDED.error_code,result=EXCLUDED.result,checked_at=now()", (self.core.customer, relative, signature, status, error, Jsonb(clean(result))))
        return summary

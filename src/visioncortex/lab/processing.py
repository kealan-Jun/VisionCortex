from __future__ import annotations

import json
import os
import re
import threading
import fcntl
from datetime import datetime, timezone, timedelta
from pathlib import Path

from psycopg.types.json import Jsonb

from .components import ROOT, decode_qr
from .domain import Conflict, Missing, clean, fingerprint, ident, instant
from .media import Media, hash_file, safe_path
from .speech_candidates import parse_usage
from .rules import Rules, RuleMismatch
from .subprocesses import run_bounded
from .storage import is_nas, _mount_info, _REMOTE_FS


class ModelUnavailable(RuntimeError):
    pass


class NativeStagePending(RuntimeError):
    def __init__(self,result):
        self.result=result
        self.status='needs_review' if result.get('native_status') in {'waiting_for_plan_review','submitted_unknown','failed','waiting_for_provider'} else 'queued'
        super().__init__('Native stage retained with status '+str(result.get('native_status')))


class ModelRegistry:
    def __init__(self, settings):
        self.settings = settings

    def descriptor(self, kind):
        path = self.settings.model_manifest
        if path is None or not path.is_file():
            raise ModelUnavailable("Customer model manifest is not configured")
        data = json.loads(path.read_text())
        if data.get("schema_version") != "vc-lab-models/1" or data.get("customer_code") != self.settings.customer_code:
            raise ModelUnavailable("Model manifest does not belong to this customer")
        model = data.get("models", {}).get(kind)
        if not model or not model.get("revision") or not model.get("python") or (kind not in {'asr','understanding'} and not model.get("files")):
            raise ModelUnavailable("Customer model/environment/files are incomplete")
        return model

    @staticmethod
    def _local_filesystem(path):
        # Reused current weights must remain local; an unverifiable mount is not
        # permission to read a possibly remote model tree.
        if is_nas(path):
            raise ModelUnavailable("Reused model assets must be on local non-NAS storage")
        mounts = _mount_info(path)
        if (len(mounts) != 1 or not mounts[0].get('fstype') or
                mounts[0]['fstype'] in _REMOTE_FS or
                str(mounts[0].get('source', '')).startswith('//')):
            raise ModelUnavailable("Local model filesystem identity is unavailable")

    def _root(self, kind, model):
        if 'model_root' not in model and 'reuse_source' not in model:
            return self.settings.runtime_root / "models", False
        raw, origin = model.get('model_root'), model.get('reuse_source')
        if kind not in {'vision', 'ocr'} or origin not in {'visioncortex', 'field_recognition'} or type(raw) is not str:
            raise ModelUnavailable("Local model reuse requires an explicit model_root and VC/FR reuse_source")
        root = Path(raw)
        if not root.is_absolute() or str(root) != raw or '..' in root.parts:
            raise ModelUnavailable("Reused model root must be an absolute canonical local directory")
        try:
            self._local_filesystem(root)
            if root.resolve(strict=True) != root or not root.is_dir():
                raise ModelUnavailable("Reused model root must be an existing canonical local directory")
        except ModelUnavailable:
            raise
        except Exception as exc:
            raise ModelUnavailable("Reused model root or local filesystem identity is unavailable") from exc
        return root, True

    def verified(self, kind):
        model = self.descriptor(kind)
        root, reused = self._root(kind, model)
        if kind in {'asr','understanding'}:
            connection = model.get("connection", {})
            env_name = connection.get("api_key_env", "")
            expected_engine = 'existing_aliyun_qwen' if kind=='asr' else 'existing_mllm'
            if model.get("engine") != expected_engine or not env_name.startswith("VC_LAB_") or connection.get("credential_ref"):
                raise ModelUnavailable("Explicit customer inference must use the frozen provider and a project credential variable")
            if not self.settings.allow_external_inference or not os.environ.get(env_name):
                raise ModelUnavailable("Customer external inference is not configured/authorized")
            if not Path(model['python']).is_file():
                raise ModelUnavailable('Customer worker environment is unavailable')
            if kind=='understanding' and (not connection.get('base_url') or set(connection)-{'api_key_env','base_url'}):
                raise ModelUnavailable('Understanding requires only an explicit customer endpoint and credential variable')
            return dict(model, model_root=str(root))
        if not isinstance(model['files'], dict):
            raise ModelUnavailable("Complete model file SHA256/byte-size manifest required")
        for relative, expected in model["files"].items():
            if (type(relative) is not str or not isinstance(expected, dict) or
                    type(expected.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}', expected['sha256']) or
                    type(expected.get('bytes')) is not int or expected['bytes'] < 0):
                raise ModelUnavailable("Every model file requires an exact SHA256 and integer byte size")
            try:
                path = safe_path(root, relative)
                if reused:
                    self._local_filesystem(path)
                digest, size = hash_file(path, 8 * 1024**3)
            except ModelUnavailable:
                raise
            except Exception as exc:
                raise ModelUnavailable("Verified local model file is unavailable or changed") from exc
            if digest != expected["sha256"] or size != expected["bytes"]:
                raise ModelUnavailable("Customer model file fingerprint mismatch")
        if kind == "vision":
            references = [model.get("weights")]
        elif kind == "ocr":
            references = [f"{model.get(directory, '')}/{name}" for directory in ("det_directory", "rec_directory") for name in ("inference.json", "inference.pdiparams", "inference.yml")]
            if model.get("panel_manifest"):
                references.append(model["panel_manifest"])
        elif kind == "asr":
            directory = model.get("directory", "")
            references = [directory + "/model.bin", directory + "/config.json"]
        else:
            raise ModelUnavailable("Unsupported customer model kind")
        if any(not r or r not in model["files"] for r in references):
            raise ModelUnavailable("Every runtime model reference must be listed in the customer manifest")
        if kind == "ocr" and model.get("panel_manifest"):
            panel = json.loads(safe_path(root, model["panel_manifest"]).read_text())
            weights = Path(panel["weights"])
            if not weights.is_absolute():
                raise ModelUnavailable("Panel manifest weights must use the explicit customer model root")
            try:
                relative = weights.relative_to(root).as_posix()
            except ValueError as exc:
                raise ModelUnavailable("Panel weights escaped the customer model root") from exc
            if relative not in model["files"] or panel.get("weights_sha256") != model["files"][relative]["sha256"]:
                raise ModelUnavailable("Panel model must share the verified customer weight identity")
        if not Path(model["python"]).is_file():
            raise ModelUnavailable("Customer worker environment is unavailable")
        return dict(model, model_root=str(root))


def insert_observation(core, conn, *, event_id, revision, kind, payload, source_kind, occurred_at=None, evidence_ids=None, job_id=None):
    ev = core._evidence(conn, evidence_ids or [])
    old = conn.execute("SELECT * FROM observation WHERE customer_id=%s AND source_event_id=%s AND source_revision=%s", (core.customer, event_id, revision)).fetchone()
    if old:
        if fingerprint(old["payload"]) != fingerprint(payload):
            raise Conflict("Observation revision has different immutable content")
        return old["id"]
    oid = ident()
    conn.execute("INSERT INTO observation(customer_id,id,job_id,source_event_id,source_revision,kind,occurred_at,payload,evidence_ids,source_kind) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (core.customer, oid, ident(job_id) if job_id else None, event_id, revision, kind, instant(occurred_at) if occurred_at else None, Jsonb(clean(payload)), ev, source_kind))
    # Only explicit standardized fields can be candidates. Raw model text is never parsed here.
    if payload.get("analysis_outcome") not in {"no_numeric_readout", "no_transcript", "unsupported", "unreadable"}:
        for field in payload.get("normalized_fields", []):
            if field.get("value") is None or not field.get("valid", False):
                continue
            association = field.get("association", {})
            for key, entity_kind in (("instrument_id", "instrument"), ("material_id", "material_unit"), ("batch_id", "process_batch")):
                if association.get(key):
                    core._entity(conn, association[key], entity_kind)
            conn.execute("INSERT INTO field_candidate(customer_id,id,observation_id,field_key,value,unit,semantics,association,conflicts) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)", (core.customer, ident(), oid, field["field_key"], Jsonb(clean(field["value"])), field.get("unit"), field.get("semantics", "unknown"), Jsonb(clean(association)), Jsonb(clean(field.get("conflicts", [])))))
    core._emit(conn, "observation", oid, 1, dict(observation_id=oid, job_id=job_id, kind=kind, occurred_at=occurred_at, payload=payload, source_kind=source_kind, evidence_ids=ev))
    return oid


class Worker:
    def __init__(self, core, settings):
        self.core, self.settings = core, settings
        self.media = Media(core, settings)
        self.registry = ModelRegistry(settings)

    def claim(self):
        with (self.settings.cache / '.admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._claim()

    def _claim(self):
        with self.core.db.transaction() as conn:
            expired = conn.execute("SELECT * FROM processing_job WHERE customer_id=%s AND status='running' AND lease_until<now() AND (%s='native' OR worker_kind='qr') FOR UPDATE SKIP LOCKED LIMIT 20", (self.core.customer, self.settings.processing_profile)).fetchall()
            for job in expired:
                conn.execute("UPDATE processing_attempt SET finished_at=now(),outcome='lease_expired' WHERE customer_id=%s AND job_id=%s AND lease_token=%s AND finished_at IS NULL", (self.core.customer, job["id"], job["lease_token"]))
                conn.execute("UPDATE processing_job SET status=%s,error_code='lease_expired',lease_token=NULL,lease_until=NULL WHERE customer_id=%s AND id=%s", ("failed" if job["attempts"] >= 3 else "queued", self.core.customer, job["id"]))
            job = conn.execute("SELECT j.* FROM processing_job j WHERE j.customer_id=%s AND j.status='queued' AND j.next_attempt_at<=now() AND (%s='native' OR j.worker_kind='qr') AND NOT EXISTS(SELECT 1 FROM processing_dependency d JOIN processing_job p ON p.customer_id=d.customer_id AND p.id=d.parent_id WHERE d.customer_id=j.customer_id AND d.job_id=j.id AND p.status<>'completed') ORDER BY CASE j.worker_kind WHEN 'qr' THEN 0 WHEN 'ocr' THEN 1 WHEN 'asr' THEN 2 WHEN 'vision' THEN 3 ELSE 4 END,j.created_at DESC,j.id FOR UPDATE OF j SKIP LOCKED LIMIT 1", (self.core.customer, self.settings.processing_profile)).fetchone()
            if not job:
                return None
            token = ident()
            row = conn.execute("UPDATE processing_job SET status='running',attempts=attempts+1,lease_token=%s,lease_until=now()+(%s*interval '1 second') WHERE customer_id=%s AND id=%s RETURNING *", (token, self.settings.lease_seconds, self.core.customer, job["id"])).fetchone()
            conn.execute("INSERT INTO processing_attempt(customer_id,id,job_id,attempt,lease_token) VALUES(%s,%s,%s,%s,%s)", (self.core.customer, ident(), job["id"], row["attempts"], token))
            return row

    def _renew(self, job, stop):
        while not stop.wait(max(1, self.settings.lease_seconds / 3)):
            with self.core.db.transaction() as conn:
                changed = conn.execute("UPDATE processing_job SET lease_until=now()+(%s*interval '1 second') WHERE customer_id=%s AND id=%s AND lease_token=%s AND status='running'", (self.settings.lease_seconds, self.core.customer, job["id"], job["lease_token"])).rowcount
                if not changed:
                    return

    def run_once(self):
        job = self.claim()
        if not job:
            return None
        stop = threading.Event()
        thread = threading.Thread(target=self._renew, args=(job, stop), daemon=True)
        thread.start()
        try:
            rules = Rules().identity()
            if job["rule_fingerprint"] not in {rules["sha256"], "legacy-unrecorded"}:
                raise RuleMismatch("Queued processing rules differ from the installed release; schedule a new revision")
            with self.core.db.transaction(readonly=True) as conn:
                source = conn.execute("SELECT a.id AS media_id,c.occurred_at,c.source_kind,e.id AS evidence_id,e.start_seconds,e.end_seconds FROM media_asset a JOIN capture c ON c.customer_id=a.customer_id AND c.id=a.capture_id JOIN evidence_ref e ON e.customer_id=a.customer_id AND e.media_id=a.id WHERE a.customer_id=%s AND a.capture_id=%s", (self.core.customer, job["capture_id"])).fetchone()
            if not source:
                raise Missing("Capture evidence is not available")
            location = self.media.locate(source["media_id"])
            if location["status"] != "available":
                raise Conflict("Media evidence is " + location["status"])
            if job["worker_kind"] == "qr":
                import cv2
                frame = cv2.imread(str(location["path"]))
                if frame is None:
                    raise ValueError("Photo is not decodable")
                values, polygons, diagnostics = decode_qr(frame)
                result = dict(analysis_outcome="observations" if values else "no_qr", decoded_values=values, polygons=polygons, diagnostics=diagnostics, model_invoked=False, normalized_fields=[])
                result["tag_associations"] = []
                for value in values:
                    if source["occurred_at"]:
                        if self.settings.processing_profile == 'timeline-v1':
                            from .timeline import Timeline
                            result["tag_associations"].append(Timeline(self.core).resolve_qr(value, source['occurred_at']))
                        else:
                            result["tag_associations"].append(self.core.resolve_qr(value, source["occurred_at"]))
                identity = {"component": "fieldrecognition", "source_commit": "9611eba0f1c4abb20cfa4268641fef2ac8745be8", "engine": "qr_decode"}
            else:
                model = self.registry.verified(job["worker_kind"])
                if job["model_revision"] != model["revision"]:
                    raise ModelUnavailable("Queue revision differs from the selected customer model; schedule a new revision")
                result = self.execute_model(job, source, location["path"], model)
                identity = dict(component=job["worker_kind"], model_revision=model["revision"], manifest_fingerprint=fingerprint(model))
            identity["processing_rules"] = rules
            result["processing_identity"] = identity
            return self.finish(job, result, source, identity)
        except Exception as exc:
            status = exc.status if isinstance(exc,NativeStagePending) else "blocked_model" if isinstance(exc, ModelUnavailable) else "failed" if job["attempts"] >= 3 else "queued"
            # Retry is local bounded processing only. No provider calls exist here.
            with self.core.db.transaction() as conn:
                conn.execute("UPDATE processing_job SET status=%s,error_code=%s,result=COALESCE(%s,result),lease_token=NULL,lease_until=NULL,next_attempt_at=now()+interval '5 seconds' WHERE customer_id=%s AND id=%s AND lease_token=%s", (status, type(exc).__name__, Jsonb(clean(exc.result)) if isinstance(exc,NativeStagePending) else None, self.core.customer, job["id"], job["lease_token"]))
                conn.execute("UPDATE processing_attempt SET finished_at=now(),outcome=%s WHERE customer_id=%s AND job_id=%s AND lease_token=%s", (status + ":" + type(exc).__name__, self.core.customer, job["id"], job["lease_token"]))
            return dict(job_id=str(job["id"]), status=status, error_code=type(exc).__name__, reason=str(exc)[:300])
        finally:
            stop.set()
            thread.join(timeout=2)

    def finish(self, job, result, source, identity):
        with self.core.db.transaction() as conn:
            row = conn.execute("SELECT * FROM processing_job WHERE customer_id=%s AND id=%s FOR UPDATE", (self.core.customer, job["id"])).fetchone()
            if row["status"] != "running" or row["lease_token"] != job["lease_token"] or row["lease_until"] <= datetime.now(timezone.utc):
                raise Conflict("Worker lease expired or was superseded")
            oid = insert_observation(self.core, conn, event_id="job:" + str(job["id"]), revision=job["model_revision"], kind=job["worker_kind"], payload=result, source_kind=source["source_kind"], occurred_at=source["occurred_at"], evidence_ids=[source["evidence_id"]], job_id=job["id"])
            if job["worker_kind"] == "asr":
                for index, segment in enumerate(result.get("segments", [])):
                    start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
                    if not 0 <= start < end or (source["end_seconds"] is not None and end > float(source["end_seconds"]) + 0.05):
                        raise ValueError("ASR segment escaped its source audio interval")
                    eid = ident()
                    conn.execute("INSERT INTO evidence_ref(customer_id,id,media_id,media_version,start_seconds,end_seconds) VALUES(%s,%s,%s,1,%s,%s)", (self.core.customer, eid, source["media_id"], str(start), str(end)))
                    segment_id = str(job["id"]) + ":segment:" + str(index)
                    fields = parse_usage(segment["text"], final=True, segment_id=segment_id)
                    at = source["occurred_at"] + timedelta(seconds=start) if source["occurred_at"] else None
                    insert_observation(self.core, conn, event_id=segment_id, revision=job["model_revision"], kind="asr_segment", payload=dict(analysis_outcome="observations", text=segment["text"], segment_id=segment_id, normalized_fields=fields), source_kind=source["source_kind"], occurred_at=at, evidence_ids=[eid], job_id=job["id"])
            conn.execute("UPDATE processing_job SET status='completed',analysis_outcome=%s,association_status=%s,result=%s,processed_at=now(),lease_token=NULL,lease_until=NULL WHERE customer_id=%s AND id=%s", (result["analysis_outcome"], result.get("association_status", "unknown"), Jsonb(clean(result)), self.core.customer, job["id"]))
            conn.execute("UPDATE processing_attempt SET finished_at=now(),outcome='completed',component_identity=%s WHERE customer_id=%s AND job_id=%s AND lease_token=%s", (Jsonb(identity), self.core.customer, job["id"], job["lease_token"]))
            return dict(job_id=str(job["id"]), observation_id=str(oid), status="completed", analysis_outcome=result["analysis_outcome"])

    def execute_model(self, job, source, path, model):
        request = dict(kind=job["worker_kind"], media_path=str(path), model=model, start_seconds=str(source["start_seconds"] or 0), end_seconds=str(source["end_seconds"]) if source["end_seconds"] is not None else None, occurred_at=clean(source["occurred_at"]))
        request.update(job_id=str(job["id"]), capture_id=str(job["capture_id"]))
        if job["worker_kind"] in {'asr','understanding'}:
            request.update(external_calls_authorized=self.settings.allow_external_inference, runtime_root=str(self.settings.runtime_root))
        if job["worker_kind"] == "ocr":
            with self.core.db.transaction(readonly=True) as conn:
                supplied = conn.execute('SELECT * FROM processing_precomputed WHERE customer_id=%s AND job_id=%s', (self.core.customer, job['id'])).fetchone()
                receipt = conn.execute('SELECT i.payload FROM capture c JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision WHERE c.customer_id=%s AND c.id=%s', (self.core.customer, job['capture_id'])).fetchone()['payload']
            if supplied:
                if supplied['model_identity'] != fingerprint(model) or supplied['rule_fingerprint'] != job['rule_fingerprint'] or hash_file(path,self.settings.max_capture_bytes)[0] != supplied['image_sha256']:
                    raise Conflict('Resident OCR proof differs from the current frame/model/rules')
                return supplied['result']
            camera = receipt.get('camera_id') or receipt['source_instance']
            request.update(camera_id=camera, video_observation=receipt.get('video_observation'), frame_metadata=receipt.get('frame_metadata', {}))
            request['instrument_bindings'] = receipt.get('instrument_bindings', []) if receipt.get('video_observation') else self.photo_bindings(path, source['occurred_at'], camera)
        if job["worker_kind"] in {'vision','understanding'}:
            # Use the recorder's original clocks and seal, not a hash-cache path without sidecars.
            with self.core.db.transaction(readonly=True) as conn:
                receipt = conn.execute("SELECT i.payload FROM capture c JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision WHERE c.customer_id=%s AND c.id=%s", (self.core.customer, job["capture_id"])).fetchone()
            if not receipt or not receipt["payload"].get("native_record"):
                raise ModelUnavailable("Full existing vision screening requires the finalized native recorder receipt and clocks")
            # Binding checks precede any inference subprocess or output creation.
            vision_model = model if job['worker_kind']=='vision' else self.registry.verified('vision')
            Rules().vision_configuration(vision_model, self.settings.runtime_root)
            request.update(native_record=receipt["payload"]["native_record"], source_root=str(self.settings.incoming), runtime_root=str(self.settings.runtime_root), job_id=str(job["id"]), unified_archive_required=self.settings.unified_archive_required)
            if job['worker_kind']=='understanding':
                with self.core.db.transaction(readonly=True) as conn:
                    parents = conn.execute('SELECT p.id,p.worker_kind,p.model_revision,p.result,p.rule_fingerprint FROM processing_dependency d JOIN processing_job p ON p.customer_id=d.customer_id AND p.id=d.parent_id WHERE d.customer_id=%s AND d.job_id=%s AND p.status=\'completed\'', (self.core.customer,job['id'])).fetchall()
                vision = next((p for p in parents if p['worker_kind']=='vision'),None)
                if not vision or vision['model_revision']!=vision_model['revision'] or any(p['rule_fingerprint']!=job['rule_fingerprint'] for p in parents):
                    raise ModelUnavailable('Native delivery parents differ from the selected customer models/rules')
                request.update(vision_model=vision_model,parents=clean(parents))
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LD_LIBRARY_PATH", "VC_LAB_STORAGE_MANIFEST", "VC_LAB_DECODER_PYTHON", "VC_LAB_LEGACY_UNC_ROOT", "VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST"}}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", VC_LAB_WORKER_CACHE=str(self.settings.cache / "worker"))
        env.update(XDG_STATE_HOME=str(self.settings.cache / "state"), XDG_CACHE_HOME=str(self.settings.cache / "packages"), YOLO_CONFIG_DIR=str(self.settings.cache / "ultralytics"), VISIONCORTEX_AI_SETTINGS_DIR=str(self.settings.runtime_root / "customer-ai-settings"))
        if job["worker_kind"] in {'asr','understanding'} and self.settings.allow_external_inference:
            name = model["connection"]["api_key_env"]
            env[name] = os.environ[name]
        if self.settings.nas_marker_sha256:
            env["VC_LAB_NAS_MARKER_SHA256"] = self.settings.nas_marker_sha256
        command = [model["python"], "-B", str(ROOT / "src/visioncortex/lab/engine_process.py")]
        result = json.loads(run_bounded(command, json.dumps(request), env=env, timeout=min(int(model.get("timeout_seconds", 120)), 600),max_bytes=12*1024**2))
        if result.get('analysis_outcome')=='native_stage_pending':
            raise NativeStagePending(result)
        if result.get("model_revision") != model["revision"]:
            raise Conflict("Model worker revision mismatch")
        return result

    def photo_bindings(self, path, occurred_at, camera_code=None):
        if not occurred_at:
            return []
        import cv2
        import hashlib
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("Source photograph cannot be decoded")
        decoded, _, _ = decode_qr(frame)
        result = []
        for value in decoded:
            resolved = self.core.resolve_qr(value, occurred_at)
            if resolved.get("association_status") != "explicit":
                continue
            binding = resolved["assignments"][0]
            with self.core.db.transaction(readonly=True) as conn:
                target = self.core._entity(conn, binding["target_id"])
            if target["kind"] != "instrument":
                continue
            instrument = dict(target["metadata"].get("field_recognition", {}), id=str(target["id"]))
            wearer = self.core.camera_context(camera_code, occurred_at) if camera_code else {}
            result.append(dict(binding_id=str(binding["id"]), instrument=instrument, started_at=clean(binding["valid_from"]), ended_at=clean(binding["valid_to"]), qr_hash=hashlib.sha256(value.encode()).hexdigest(), attribution_status="equipment_identity_only", wearer_id=wearer.get('wearer_id'), wearer_binding_id=wearer.get('wearer_binding_id')))
        return result

    def schedule_models(self):
        if self.settings.processing_profile == 'timeline-v1':
            return 0
        created = 0
        rules = Rules().identity()
        for kind in ("qr", "vision", "ocr", "asr"):
            try:
                model = {"revision": "model-free-qr/1"} if kind == "qr" else self.registry.descriptor(kind)
            except ModelUnavailable:
                continue
            with self.core.db.transaction() as conn:
                captures = conn.execute("SELECT DISTINCT capture_id FROM processing_job WHERE customer_id=%s AND worker_kind=%s", (self.core.customer, kind)).fetchall()
                for capture in captures:
                    created += conn.execute("INSERT INTO processing_job(customer_id,id,capture_id,worker_kind,model_revision,rule_identity,rule_fingerprint) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", (self.core.customer, ident(), capture["capture_id"], kind, model["revision"], Jsonb(rules), rules["sha256"])).rowcount
        return created

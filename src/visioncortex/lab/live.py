"""Customer Receiver adapter: reuse original FR event selection, persist to PostgreSQL."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from .components import ROOT, enable_components, pipeline_module
from .domain import Conflict, clean, fingerprint
from .ingest import Ingest
from .media import safe_path
from .processing import ModelRegistry, ModelUnavailable
from .retention import encoded
from .rules import Rules
from .storage import publication_config
from .archive_layout import day_directory


def load_sources(core, path):
    if path is None:
        return []
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ValueError('Live source manifest exceeds its bound or is a symbolic link')
    document = json.loads(path.read_text())
    if document.get('schema_version') != 'vc-lab-live-sources/1':
        raise ValueError('Unsupported live source manifest')
    sources = document.get('sources', [])
    if not isinstance(sources, list) or len(sources) > 7:
        raise ValueError('At most five wearable and two explicitly registered fixed sources')
    seen, wearable = set(), 0
    with core.db.transaction(readonly=True) as conn:
        customer = conn.execute('SELECT code FROM customer_instance WHERE id=%s', (core.customer,)).fetchone()
        if document.get('customer_code') != customer['code']:
            raise Conflict('Live source manifest is outside this customer')
        for source in sources:
            camera = source.get('camera_id', '')
            endpoint = urlsplit(source.get('receiver_url', ''))
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', camera) or camera in seen:
                raise ValueError('Live camera IDs must be explicit and unique')
            if endpoint.scheme not in {'http', 'https'} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError('Receiver requires an explicit endpoint without embedded credentials')
            row = conn.execute("SELECT metadata FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s", (core.customer, camera)).fetchone()
            if not row or row['metadata'].get('view_role') not in {'first_person', 'third_person'}:
                raise Conflict('Register each customer capture device and its view role before enabling it')
            source['view_role'] = row['metadata']['view_role']
            wearable += source['view_role'] == 'first_person'
            seen.add(camera)
    if wearable > 5 or len(sources)-wearable > 2:
        raise ValueError('Live source manifest exceeds the current software validation scope')
    return sources


class _Rows:
    def __init__(self, rows):
        self.rows = rows
    def fetchone(self):
        return self.rows[0] if self.rows else None
    def __iter__(self):
        return iter(self.rows)


class ReceiverService:
    def __init__(self, core, settings, sources, model, *, camera_factory=None, runtime_factory=None):
        self.core, self.settings, self.sources, self.model = core, settings, sources, model
        self.rules = Rules().identity()
        self.model_identity = fingerprint(model)
        self.pending = {}
        self.lock = threading.RLock()
        self.devices, self.readers = [], []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vc-lab-live-ocr')
        enable_components()
        from .native_ocr import make_runtime
        from field_recognition.capture.video_ocr import VideoOcr
        from field_recognition.recognition.config import RecognitionConfig
        from field_recognition.recognition.service import predict_panel, predict_readout
        if camera_factory is None:
            from field_recognition.capture.receiver import ReceiverCamera
            camera_factory = ReceiverCamera
        self.runtime = (runtime_factory or make_runtime)(model, [])
        self.runtime['get_instrument'] = self.instrument
        config = SimpleNamespace(video_ocr_enabled=True, recognition=RecognitionConfig(enabled=False, no_digits_seconds=Rules().profile('ocr')['no_digits_seconds']))
        for source in sources:
            camera = camera_factory(source['receiver_url'], source['camera_id'])
            runtime = dict(self.runtime, receiver_camera=camera, config=config, ocr_pool=self.pool, ocr_submit_lock=self.lock,
                           automatic_runner=SimpleNamespace(settings=lambda s=source: self.operator(s)),
                           saved_photo_watcher=SimpleNamespace(snapshot=lambda: {}), queue_ocr_warmup=lambda: None,
                           db=self.database, current_panel_bindings=lambda code: self.panel_bindings(code),
                           scan_image=self.scan_image, enqueue_ocr=self.enqueue, OcrRequest=lambda **kwargs: SimpleNamespace(**kwargs),
                           qr_matches=self.qr_matches)
            runtime['predict_panel'] = lambda *a, _r=runtime, **kw: predict_panel(_r,*a,**kw)
            runtime['predict_readout'] = lambda *a, _r=runtime, **kw: predict_readout(_r,*a,**kw)
            self.devices.append(camera)
            self.readers.append(VideoOcr(runtime))

    def instrument(self, iid):
        with self.core.db.transaction(readonly=True) as conn:
            row = self.core._entity(conn, iid, 'instrument')
        return dict(row['metadata'].get('field_recognition', {}), id=str(row['id']), name=row['metadata'].get('name', row['code']))

    def operator(self, source):
        context = self.core.camera_context(source['camera_id'], datetime.now(timezone.utc))
        operator = context if context.get('wearer_id') else None
        if source['view_role'] == 'third_person':
            operator = dict(wearer_id=None, association_status='fixed_camera_has_no_wearer')
        return dict(video_ocr_enabled=True, operator=operator)

    def panel_bindings(self, code, at=None, *, known_clock=True):
        when = at or datetime.now(timezone.utc)
        context = self.core.camera_context(code, at) if known_clock else {}
        with self.core.db.transaction(readonly=True) as conn:
            rows = conn.execute("SELECT e.*,t.code AS qr,b.id AS binding_id,b.valid_from,b.valid_to FROM entity e JOIN tag_assignment b ON b.customer_id=e.customer_id AND b.target_id=e.id JOIN entity t ON t.customer_id=b.customer_id AND t.id=b.tag_id WHERE e.customer_id=%s AND e.kind='instrument' AND b.valid_from<=%s AND (b.valid_to IS NULL OR b.valid_to>%s) AND e.metadata->'field_recognition'->'camera_ids' ? %s", (self.core.customer,when,when,code)).fetchall()
        return [dict(binding_id=str(r['binding_id']), instrument=dict(r['metadata']['field_recognition'], id=str(r['id']), name=r['metadata'].get('name',r['code'])), qr_hash=hashlib.sha256(r['qr'].encode()).hexdigest(), started_at=clean(r['valid_from']), ended_at=clean(r['valid_to']), attribution_status='explicit_camera_panel_mapping', wearer_id=context.get('wearer_id'), wearer_binding_id=context.get('wearer_binding_id')) for r in rows]

    def qr_matches(self, values, polygons):
        found = []
        for value in values:
            resolved = self.core.resolve_qr(value, datetime.now(timezone.utc))
            if resolved.get('association_status') == 'explicit':
                for binding in resolved['assignments']:
                    with self.core.db.transaction(readonly=True) as conn:
                        row = self.core._entity(conn, binding['target_id'])
                    if row['kind'] == 'instrument':
                        found.append(dict(id=str(row['id']), name=row['metadata'].get('name',row['code'])))
        return dict(matches=found)

    @contextlib.contextmanager
    def database(self):
        # Translate only the four legacy VideoOcr calls; never open the old SQLite.
        service = self
        class Facade:
            def execute(self, query, args=()):
                if query.startswith('UPDATE scans SET document='):
                    service.pending[args[1]]['capture'] = json.loads(args[0])
                    return _Rows([])
                if query.startswith('SELECT document FROM bindings WHERE camera='):
                    return _Rows([{'document':json.dumps(b)} for b in service.panel_bindings(args[0])])
                with service.core.db.transaction(readonly=True) as conn:
                    if query.startswith('SELECT count(*) FROM jobs'):
                        row = conn.execute("SELECT count(*) AS n FROM processing_job WHERE customer_id=%s AND worker_kind='ocr' AND status IN ('queued','running')", (service.core.customer,)).fetchone()
                        return _Rows([(row['n'],)])
                    if query.startswith('SELECT 1 FROM jobs'):
                        row = conn.execute("SELECT 1 FROM processing_job j JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision WHERE j.customer_id=%s AND j.worker_kind='ocr' AND j.status IN ('queued','running') AND NOT i.payload ? 'video_observation' LIMIT 1", (service.core.customer,)).fetchone()
                        return _Rows([row] if row else [])
                raise ValueError('Unexpected legacy live-OCR database operation')
        yield Facade()

    def scan_image(self, raw, source, camera, *, decoded, metadata, operator):
        from field_recognition.measurements.measurement_records import video_clock
        clock = video_clock(metadata)
        observed = datetime.now(timezone.utc)
        at = clock['captured_at']
        device = next(d for d in self.devices if d.target == camera)
        session = getattr(device,'snapshot',lambda:{})().get('service_status',{})
        metadata = dict(metadata, **{k:session[k] for k in ('media_session_id','device_boot_id') if session.get(k) is not None})
        stable = {k:metadata.get(k) for k in ('sender_id','camera_id','sequence','timestamp_us','global_timestamp_us','media_session_id','device_boot_id')}
        capture = fingerprint(dict(camera=camera, frame=stable, sha256=hashlib.sha256(raw).hexdigest()))
        directory = day_directory(camera, at or observed)
        relative = f'{directory}/MetaData/VideoEvidence/{capture}/Frame.png'
        if len(raw)>self.settings.max_capture_bytes or shutil.disk_usage(self.settings.runtime_root).free<80*1024**3:
            raise Conflict('Live evidence exceeded its local admission bound')
        root = self.settings.incoming
        if not str(root).startswith('/mnt/'):
            used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if used+len(raw)>self.settings.archive_max_bytes:
                raise Conflict('Local live evidence capacity reached')
        guard = pipeline_module('device_day_write_guard')
        path = safe_path(root,relative,must_exist=False)
        with guard.storage_guard(publication_config(self.settings)):
            self.write(path,raw,guard)
        document = dict(capture_id=capture, camera_id=camera, frame_metadata=metadata, operator=operator)
        self.pending[capture] = dict(capture=document, relative=relative, raw_sha256=hashlib.sha256(raw).hexdigest(), at=at)
        if len(self.pending)>8:
            del self.pending[next(iter(self.pending))]
        return document

    @staticmethod
    def write(path, raw, guard):
        if path.exists():
            if path.read_bytes()!=raw:
                raise Conflict('Immutable Receiver evidence changed')
            return
        if guard.guarded_atomic(path,[raw]):
            return
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.partial')
        try:
            with temporary.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)

    def enqueue(self, body, *, trigger, precomputed_local):
        from .native_ocr import format_result
        item = self.pending[body.capture_id]
        document = item['capture']
        bindings = self.panel_bindings(document['camera_id'],item['at'],known_clock=bool(item['at']))
        source = dict(capture_id=body.capture_id, source_instance=document['camera_id'], source_revision='1', kind='photo',
                      relative_path=item['relative'], sha256=item['raw_sha256'], captured_at=item['at'],
                      time_basis={'basis':'synchronized_frame_timestamp' if item['at'] else 'unknown'}, source_kind=self.settings.source_kind,
                      camera_id=document['camera_id'], video_observation=document['video_observation'], frame_metadata=document['frame_metadata'], instrument_bindings=bindings)
        # A durable source receipt lets ordinary ingest recover a crash before DB commit.
        receipt_path = safe_path(self.settings.incoming,str(Path(item['relative']).parent/'CaptureReceipt.json'),must_exist=False)
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous['capture_id'] != body.capture_id or previous['sha256'] != item['raw_sha256'] or previous['camera_id'] != document['camera_id']:
                raise Conflict('Existing Receiver frame receipt differs from this physical source')
            accepted = Ingest(self.core,self.settings).accept(previous)
            self.pending.pop(body.capture_id,None)
            with self.core.db.transaction(readonly=True) as conn:
                job = conn.execute("SELECT id FROM processing_job WHERE customer_id=%s AND capture_id=%s AND worker_kind='ocr' ORDER BY created_at,id LIMIT 1", (self.core.customer,accepted['capture_id'])).fetchone()
            return dict(job_id=str(job['id']))
        guard = pipeline_module('device_day_write_guard')
        with guard.storage_guard(publication_config(self.settings)):
            self.write(receipt_path,encoded(source),guard)
        request = dict(job_id='pending',capture_id=body.capture_id, image_sha256=item['raw_sha256'], media_path=str(self.settings.incoming/item['relative']), occurred_at=item['at'], instrument_bindings=bindings, camera_id=document['camera_id'], video_observation=document['video_observation'], frame_metadata=document['frame_metadata'])
        result = format_result(precomputed_local,request)
        result.update(model_revision=self.model['revision'],model_invoked=True, resident_frame_ocr_reused=True)
        proof = dict(result=result,revision=self.model['revision'],image_sha256=item['raw_sha256'],model_identity=self.model_identity,rule_fingerprint=self.rules['sha256'])
        accepted = Ingest(self.core,self.settings).accept(source,precomputed_ocr=proof)
        self.pending.pop(body.capture_id,None)
        with self.core.db.transaction(readonly=True) as conn:
            job = conn.execute("SELECT id FROM processing_job WHERE customer_id=%s AND capture_id=%s AND worker_kind='ocr'", (self.core.customer,accepted['capture_id'])).fetchone()
        return dict(job_id=str(job['id']))

    def start(self):
        for camera in self.devices:
            camera.start()
        for reader in self.readers:
            reader.start()

    def close(self):
        for reader in self.readers:
            reader.close()
        for camera in self.devices:
            camera.close()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.runtime['panel_detector'].close()

    def snapshot(self):
        return dict(status='running', model_revision=self.model['revision'], processing_rules=self.rules, sources=[dict(camera_id=c.target,**r.snapshot()) for c,r in zip(self.devices,self.readers)])


class ReceiverSupervisor:
    def __init__(self,core,settings):
        self.core,self.settings=core,settings
        self.process,self.signature=None,None
        self.retry_at=0
        self.stopping=threading.Event()
        self.process_lock=threading.RLock()

    def tick(self):
        if self.stopping.is_set():
            return dict(status='stopped')
        sources=load_sources(self.core,self.settings.live_sources_manifest)
        if not sources:
            self._halt()
            return dict(status='unconfigured',sources=0)
        try:
            model=ModelRegistry(self.settings).descriptor('ocr')
        except ModelUnavailable:
            self._halt()
            return dict(status='blocked_model',sources=len(sources))
        signature=fingerprint(dict(sources=sources,model=model,rules=Rules().identity()))
        if self.process and self.process.poll() is None and signature==self.signature:
            return dict(status='running',sources=len(sources))
        self._halt()
        if time.monotonic()<self.retry_at:
            return dict(status='retry_wait')
        try:
            model=ModelRegistry(self.settings).verified('ocr')
        except ModelUnavailable:
            return dict(status='blocked_model',sources=len(sources))
        allowed={'VC_LAB_STORAGE_MANIFEST','VC_LAB_DECODER_PYTHON','PATH','LANG','LD_LIBRARY_PATH','VC_LAB_DATABASE_URL','VC_LAB_RUNTIME_ROOT','VC_LAB_CUSTOMER_CODE','VC_LAB_SOURCE_ROOT','VC_LAB_NAS_MARKER_SHA256','VC_LAB_MODEL_MANIFEST','VC_LAB_LIVE_SOURCES_MANIFEST','VC_LAB_UNIFIED_ARCHIVE_REQUIRED'}
        env={k:v for k,v in os.environ.items() if k in allowed}
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',FIELD_DEMO_DATA=str(self.settings.runtime_root/'field-context'))
        with self.process_lock:
            if self.stopping.is_set():
                return dict(status='stopped')
            self.process=subprocess.Popen([model['python'],'-B',str(ROOT/'src/visioncortex/lab/live_process.py')],env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        self.signature,self.retry_at=signature,time.monotonic()+30
        return dict(status='starting',sources=len(sources))

    def close(self):
        self.stopping.set()
        self._halt()

    def _halt(self):
        with self.process_lock:
            process,self.process=self.process,None
        if process:
            try:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()

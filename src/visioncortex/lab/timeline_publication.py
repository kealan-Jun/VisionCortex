"""Publish immutable timeline views inside the existing device/day categories."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from uuid import uuid4

from psycopg.types.json import Jsonb

from .archive_layout import day_directory, verify_archive_contract
from .components import pipeline_module
from .domain import Conflict, clean, fingerprint, instant
from .media import hash_file, safe_path
from .native_timeline_links import association_path, build_sidecar
from .rules import Rules
from .storage import is_nas, publication_config
from .timeline import day_bounds
from .timeline_clock import source_clock_sql, source_clock_supported
from .timeline_report import render_report
from .timeline_timing import PublicationTiming
from .timeline_native_artifacts import artifact_documents


class TimelinePublication:
    def __init__(self, runtime, settings, *, not_before=None):
        self.runtime, self.settings = runtime, settings
        self.core, self.customer = runtime.timeline.core, runtime.timeline.customer
        # A rollout may opt out of re-publishing older snapshots under its new
        # rule identity. This is a publication clock, never an event-time claim.
        self.not_before = instant(not_before) if not_before is not None else None

    def tick(self):
        if self.settings.processing_profile != 'timeline-v1' or not self.settings.timeline_publish:
            return dict(enabled=False)
        if self.settings.source_root is None or not self.settings.unified_archive_required:
            raise ValueError('Timeline publication requires an explicit verified unified archive')
        root = self.settings.incoming
        with self.core.db.transaction(readonly=True) as conn:
            scope = conn.execute('SELECT data_scope FROM customer_instance WHERE id=%s', (self.customer,)).fetchone()['data_scope']
        if is_nas(root) and scope != 'customer':
            raise ValueError('Development/fixture reports cannot publish to the formal customer archive')
        verify_archive_contract(root)
        rules = Rules().identity()
        destination = fingerprint(dict(root=str(root), customer=str(self.customer), namespace='timeline-publication/1', publisher_rules=rules))
        with (self.settings.runtime_root / 'timeline-publication.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.core.db.transaction(readonly=True) as conn:
                snapshot = conn.execute('SELECT r.* FROM hj_daily_report r WHERE r.customer_id=%s AND (%s::timestamptz IS NULL OR r.created_at>=%s) AND NOT EXISTS(SELECT 1 FROM hj_timeline_publication p WHERE p.customer_id=r.customer_id AND p.day=r.day AND p.revision=r.revision AND p.destination_fingerprint=%s) ORDER BY r.day DESC,r.revision DESC LIMIT 1', (self.customer, self.not_before, self.not_before, destination)).fetchone()
                if not snapshot:
                    return None
                start, end = day_bounds(snapshot['day'])
                # The report owns its source set. Do not let a newly committed
                # capture change bytes after an interrupted immutable publish.
                pinned_media = snapshot['document'].get('publication_media_ids')
                if pinned_media is not None:
                    if type(pinned_media) is not list or len(pinned_media) > 4096:
                        raise ValueError('Snapshot publication source identities exceed the bounded list contract')
                    source_filter, source_value = 'm.media_id=ANY(%s::uuid[])', pinned_media
                    source_set_basis = 'snapshot_publication_media_ids'
                else:
                    source_filter, source_value = 'c.received_at<=%s', snapshot['created_at']
                    source_set_basis = 'legacy_snapshot_created_at_cutoff_not_exact_source_set'
                clock = source_clock_sql()
                media = conn.execute(f'''SELECT m.*,a.verified_sha256,a.byte_size,c.received_at AS capture_received_at,
                    i.payload AS capture_receipt FROM hj_timeline_media m
                    JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
                    JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
                    JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                        AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                    WHERE m.customer_id=%s AND {source_filter}
                        AND ((m.start_at IS NOT NULL AND COALESCE({clock},false) AND m.start_at<%s AND m.end_at>=%s)
                        OR ((m.start_at IS NULL OR NOT COALESCE({clock},false))
                            AND c.received_at>=%s AND c.received_at<%s))
                    ORDER BY m.stream_key,m.start_at,m.media_id''', (self.customer, source_value, end, start, start, end)).fetchall()
            if not media:
                return dict(enabled=True, status='no_media_in_archive_partition', day=str(snapshot['day']))
            # Recorder/photo/voice source IDs are provenance, not archive camera keys.
            # Reuse the explicit producer camera_id; never strip prefixes or hash names.
            cameras_by_media = {str(m['media_id']): m['capture_receipt'].get('camera_id') or m['stream_key'] for m in media}
            cameras = sorted(set(cameras_by_media.values()))
            with self.core.db.transaction(readonly=True) as conn:
                camera_entities=conn.execute("SELECT id,code FROM entity WHERE customer_id=%s AND kind='capture_device' AND (code=ANY(%s) OR id::text=ANY(%s)) LIMIT 1025",
                    (self.customer,cameras,cameras)).fetchall()
            if len(camera_entities)>1024:raise ValueError('Native sidecar camera identity query exceeds bound')
            camera_entity_ids={camera:{str(e['id']) for e in camera_entities if camera in {e['code'],str(e['id'])}} for camera in cameras}
            # Daily views use report day; source media keep their original capture day.
            roots = {camera: day_directory(camera, start) for camera in cameras}
            report = snapshot['document']
            suffix = f'r{snapshot["revision"]:06d}-{snapshot["fingerprint"][:16]}-{rules["sha256"][:12]}'
            canonical = roots[cameras[0]] + '/LaboratoryDailyReport/Timeline/' + suffix
            files = {canonical + '.json': self._json(dict(schema_version='vc-lab-published-daily-report/1', customer_id=str(self.customer),
                     report_scope='customer_all_streams', day=str(snapshot['day']), revision=snapshot['revision'], fingerprint=snapshot['fingerprint'], publication_rules=rules, data_scope=scope, document=report)),
                     canonical + '.txt': render_report(report).encode()}
            activities = {a['id']: a for e in report['experiments'] for a in e['activities']}
            activities.update({a['id']: a for a in report['unassigned_activities']})
            activities.update({a['id']: a for a in report.get('unknown_time_temporal_activities',{}).get('activities',[])})
            activities.update({a['id']: a for a in report.get('unknown_time_activities',{}).get('activities',[])})
            projection=report.get('person_timelines') or {}
            declared_camera_by_activity={}
            for person in projection.get('persons',[]):
                for event in person.get('events',[]):
                    for role in event.get('person_attribution',[]):
                        if role.get('basis')=='direct_operator_declaration' and role.get('scope_camera_id'):
                            declared_camera_by_activity.setdefault(event['activity_id'],set()).add(role['scope_camera_id'])
            latest = {}
            for camera in cameras:
                base = roots[camera]
                scoped = [a for a in activities.values() if any(cameras_by_media.get(e['media_id']) == camera for e in a['evidence'])]
                explicit_scoped=[a for a in activities.values() if declared_camera_by_activity.get(a['id'],set()) & camera_entity_ids[camera]]
                sidecar_activities={a['id']:a for a in scoped+explicit_scoped}
                sources={str(m['media_id']):clean(dict(source_relative_path=m['capture_receipt'].get('relative_path'),
                    source_sha256=m['verified_sha256'],byte_size=m['byte_size'],kind=m['kind'],
                    capture_id=m['capture_id'],stream_key=m['stream_key'],
                    session_key=m['session_key'],source_kind=m['capture_receipt'].get('source_kind'),time_basis=m['time_basis'],
                    source_reported_at=m['capture_receipt'].get('captured_at'),capture_received_at=m['capture_received_at'],
                    capture_time_status='supported_source_clock' if m['start_at'] is not None and source_clock_supported(m['time_basis'],m['capture_receipt'].get('source_kind')) else 'unknown',
                    archive_day_basis='source_reported_capture_day' if m['start_at'] is not None and source_clock_supported(m['time_basis'],m['capture_receipt'].get('source_kind')) else 'capture_received_at_partition_only',
                    receipt_day_is_not_physical_event_time=m['start_at'] is None or not source_clock_supported(m['time_basis'],m['capture_receipt'].get('source_kind'))))
                    for m in media if cameras_by_media[str(m['media_id'])]==camera}
                planned, artifact_refs=artifact_documents(self.core,archive=base,camera=camera,
                    activities=list(sidecar_activities.values()),media_sources=sources,rules=rules)
                for relative,value in planned.items():
                    files[relative]=value if isinstance(value,bytes) else self._json(value)
                phases=dict(report.get('operation_segments') or {})
                phases['segments']=[p for p in phases.get('segments',[]) if str(p.get('camera_id')) in camera_entity_ids[camera]]
                phases['unpaired_anchors']=[p for p in phases.get('unpaired_anchors',[]) if p.get('activity_id') in sidecar_activities]
                sidecar=build_sidecar(archive=base,customer_id=self.customer,revision=snapshot['revision'],
                    snapshot_fingerprint=snapshot['fingerprint'],rules=rules,canonical_report=canonical+'.json',
                    activities=list(sidecar_activities.values()),person_projection=projection,
                    media_sources=sources,artifact_refs=artifact_refs,operation_projection=phases,
                    bindings_latest_path=base+'/Comment/Bindings.json')
                weighing=(report.get('context_evidence') or {}).get('weighings') or {}
                episodes=[w for w in weighing.get('episodes',[]) if str(w['context']['camera_id']) in camera_entity_ids[camera]]
                pending=[p for p in weighing.get('pending',[]) if p['activity_id'] in sidecar_activities]
                spoken=[s for s in (report.get('context_evidence') or {}).get('spoken_material_usage',{}).get('entries',[])
                        if s['activity_id'] in sidecar_activities and str(s['camera_id']) in camera_entity_ids[camera]]
                history=association_path(base,snapshot['revision'],snapshot['fingerprint'],rules['sha256'])
                files[history]=self._json(sidecar)
                latest[base+'/Comment/Bindings.json']=dict(用料记录=[w['阅读摘要'] for w in episodes]+[v for s in spoken for v in s['阅读摘要']],
                    用料待处理=[dict(读数活动=p['activity_id'],原因=p['result']['reason'],已扣库存=False) for p in pending],
                    schema_version='vc-lab-camera-bindings/1',
                    archive=base,camera_key=camera,customer_id=str(self.customer),snapshot_revision=snapshot['revision'],
                    snapshot_fingerprint=snapshot['fingerprint'],processing_rules=rules,
                    source_set_basis=source_set_basis,
                    immutable_associations=dict(relative_path=history,sha256=hashlib.sha256(files[history]).hexdigest()),
                    canonical_report=canonical+'.json',observations=sidecar['observations'],media_sources=sources,
                    operation_segments=phases,comment_artifacts=artifact_refs,
                    weighings=episodes,spoken_material_usage=spoken,
                    material_quantity_written=any(w['operation_id'] for w in episodes) or any(v['已扣库存'] for s in spoken for v in s['阅读摘要']),
                    physical_panel_or_speaker_inferred=False,unknowns_are_candidates=True)
                envelope = dict(schema_version='vc-lab-timeline-published-view/1', report_day=str(snapshot['day']), customer_id=str(self.customer),
                                snapshot_revision=snapshot['revision'], snapshot_fingerprint=snapshot['fingerprint'],
                                camera_key=camera, stream_keys=sorted({m['stream_key'] for m in media if cameras_by_media[str(m['media_id'])] == camera}),
                                canonical_customer_report=canonical+'.json', raw_originals_moved=False,
                                processing_rules=rules, data_scope=scope)
                for source_type, category in [('video', 'MultimodalUnderstanding'), ('speech', 'Comment/Audio'), ('ocr', 'Comment/OCR')]:
                    files[base + '/' + category + '/Timeline/' + suffix + '.json'] = self._json(dict(envelope, observations=[a for a in scoped if a['source_type'] == source_type]))
                clips = [dict(activity_id=a['id'], kind=a['kind'], priority=a['priority'], evidence=e, media_url='/internal/media/'+e['media_id'])
                         for a in scoped for e in a['evidence'] if e['kind'] == 'video' and cameras_by_media.get(e['media_id']) == camera]
                files[base+'/ProcessedClips/Timeline/'+suffix+'.json'] = self._json(dict(envelope, clips=clips, physical_clip_generation=False))
                if camera != cameras[0]:
                    files[base+'/LaboratoryDailyReport/Timeline/'+suffix+'.json'] = self._json(envelope)
            guard = pipeline_module('device_day_write_guard')
            receipts = []
            with PublicationTiming(self.core,snapshot['day'],snapshot['revision']) as timing:
                with self.core.db.transaction() as conn:
                    # One cross-runtime writer for this customer/day, not merely
                    # one local runtime directory. Hold through the DB receipt.
                    conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                        (f'timeline-comment-publication:{self.customer}:{snapshot["day"]}',))
                    with timing.phase('write_and_full_readback'),guard.storage_guard(publication_config(self.settings)):
                        for relative, raw in files.items():
                            receipts.append(self._write(root, relative, raw, guard))
                        for relative, document in latest.items():
                            receipts.extend(self._write_latest(conn,root,relative,document,guard))
                    conn.execute('INSERT INTO hj_timeline_publication(customer_id,day,revision,destination_fingerprint,files) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING', (self.customer, snapshot['day'], snapshot['revision'], destination, Jsonb(receipts)))
            return dict(enabled=True, day=str(snapshot['day']), revision=snapshot['revision'], files=receipts)

    @staticmethod
    def _json(value):
        encoded = json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        if isinstance(value, dict) and '中文摘要' in value:
            # Keep the readable candidate view first. Canonicalize all nested
            # keys exactly as before, without changing other document types.
            document = json.loads(encoded)
            summary = document.pop('中文摘要')
            encoded = json.dumps({'中文摘要': summary, **document}, ensure_ascii=False,
                                 indent=2, allow_nan=False)
        return (encoded+'\n').encode()

    @staticmethod
    def _read_bytes(root, relative, guard):
        """Bounded stable bytes; NAS reads stay on the current pinned mount."""
        path=safe_path(root,relative,must_exist=False)
        parent=None
        if is_nas(path):
            pinned=guard.CURRENT.get()
            if pinned is None:raise OSError('Latest view read requires its pinned NAS guard')
            pinned.validate()
            mount,canonical=pinned.mount_for(path)
            parent=pinned.directory(mount,canonical.relative_to(mount.path).parts[:-1])
            name=canonical.name
        else:
            pinned=None;name=path
        try:
            try:fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
            except FileNotFoundError:return None
            with os.fdopen(fd,'rb') as handle:
                before=os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size>16*1024**2:raise Conflict('Latest view exceeds regular-file read bound')
                raw=handle.read(16*1024**2+1);after=os.fstat(handle.fileno())
                fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
                if len(raw)!=before.st_size or tuple(getattr(before,k) for k in fields)!=tuple(getattr(after,k) for k in fields):
                    raise Conflict('Latest view changed during stable read')
            if pinned is not None:pinned.validate()
            return raw
        finally:
            if parent is not None:os.close(parent)

    def _write_latest(self,conn,root,relative,content,guard):
        """Revision guard: immutable equality + DB high water, then atomic CAS.

        Caller holds the customer/day DB advisory lock through filesystem writes
        and the publication receipt transaction. A failed post-write transaction
        can resume a valid newer immutable copy, without rolling its view back.
        """
        if relative!=content['archive']+'/Comment/Bindings.json':
            raise ValueError('Only the explicit Comment Bindings latest path is mutable')
        root_fingerprint=fingerprint(str(root))
        water=conn.execute('''SELECT f FROM hj_timeline_publication p,
            LATERAL jsonb_array_elements(p.files) f WHERE p.customer_id=%s
            AND f->>'relative_path'=%s AND f->>'root_fingerprint'=%s
            AND f->>'mutable_latest'='true'
            ORDER BY (f->>'publication_revision')::bigint DESC LIMIT 1''',
            (self.customer,relative,root_fingerprint)).fetchone()
        water=water['f'] if water else None
        previous_raw=self._read_bytes(root,relative,guard)
        previous=None
        if previous_raw is not None:
            from .native_timeline_links import _unique
            previous=json.loads(previous_raw,object_pairs_hook=_unique)
            if (previous.get('schema_version')!='vc-lab-camera-bindings/1'
                    or previous.get('archive')!=content['archive'] or previous.get('customer_id')!=str(self.customer)
                    or previous.get('camera_key')!=content['camera_key']
                    or type(previous.get('revision')) is not int or previous['revision']<1):
                raise Conflict('Existing Comment Bindings is foreign or has no valid version; not overwritten')
            history=(previous.get('immutable_history') or {}).get('relative_path','')
            prefix=content['archive']+'/LaboratoryDailyReport/Timeline/Associations/Bindings/'
            if not history.startswith(prefix+f'r{previous["revision"]:06d}-'):
                raise Conflict('Latest Bindings has no bounded immutable history reference')
            if self._read_bytes(root,history,guard)!=previous_raw:
                raise Conflict('Latest Bindings differs from its immutable history bytes')
            body={k:v for k,v in previous.items() if k not in {'revision','view_fingerprint','immutable_history'}}
            if fingerprint(body)!=previous.get('view_fingerprint'):
                raise Conflict('Latest Bindings content fingerprint differs')
        if water and (previous is None or previous['revision']<water['publication_revision']
                or previous['revision']==water['publication_revision'] and hashlib.sha256(previous_raw).hexdigest()!=water['sha256']):
            raise Conflict('Latest Bindings rolled back or differs from published revision high water')
        if previous and (previous['snapshot_revision']>content['snapshot_revision']):
            return [dict(relative_path=relative,sha256=hashlib.sha256(previous_raw).hexdigest(),byte_size=len(previous_raw),
                         mutable_latest=False,status='superseded_snapshot_not_written',publication_revision=previous['revision'])]
        if previous and previous['snapshot_revision']==content['snapshot_revision'] and previous['snapshot_fingerprint']!=content['snapshot_fingerprint']:
            raise Conflict('Same report revision cannot identify different Bindings content')
        view_fingerprint=fingerprint(content)
        if previous and previous['view_fingerprint']==view_fingerprint:
            return [dict(relative_path=relative,sha256=hashlib.sha256(previous_raw).hexdigest(),byte_size=len(previous_raw),
                         mutable_latest=True,publication_revision=previous['revision'],root_fingerprint=root_fingerprint)]
        revision=max(previous['revision'] if previous else 0,water['publication_revision'] if water else 0)+1
        history=content['archive']+f'/LaboratoryDailyReport/Timeline/Associations/Bindings/r{revision:06d}-{view_fingerprint[:16]}-{content["processing_rules"]["sha256"][:12]}.json'
        document=dict(content,revision=revision,view_fingerprint=view_fingerprint,
                      immutable_history=dict(relative_path=history))
        raw=self._json(document)
        if len(raw)>2*1024**2:raise ValueError('Latest Bindings exceeds bounded readable view')
        immutable=self._write(root,history,raw,guard)
        expected=(hashlib.sha256(previous_raw).hexdigest(),len(previous_raw)) if previous_raw is not None else None
        published=self._write(root,relative,raw,guard,replace_expected=expected)
        published.update(mutable_latest=True,publication_revision=revision,root_fingerprint=root_fingerprint,
                         immutable_history=immutable,snapshot_revision=content['snapshot_revision'])
        return [immutable,published]

    @staticmethod
    def _write(root, relative, raw, guard, *, replace_expected=None):
        if len(raw) > 16 * 1024**2:
            raise ValueError('Publication document exceeds its bound')
        path = safe_path(root, relative, must_exist=False)
        digest = hashlib.sha256(raw).hexdigest()
        def existing():
            if not is_nas(path):
                return hash_file(path, 16*1024**2) if path.exists() else None
            pinned = guard.CURRENT.get()
            if pinned is None:
                raise OSError('NAS read-back requires a pinned mount guard')
            pinned.validate()
            mount, canonical = pinned.mount_for(path)
            parent = pinned.directory(mount, canonical.relative_to(mount.path).parts[:-1])
            try:
                try:
                    fd = os.open(canonical.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
                except FileNotFoundError:
                    return None
                hasher, size = hashlib.sha256(), 0
                with os.fdopen(fd, 'rb') as handle:
                    for block in iter(lambda: handle.read(1024*1024), b''):
                        size += len(block)
                        if size > 16*1024**2:
                            raise Conflict('Published output exceeds read-back bound')
                        hasher.update(block)
                pinned.validate(volume=False)
                return hasher.hexdigest(), size
            finally:
                os.close(parent)
        actual = existing()
        if actual is not None:
            if actual != (digest, len(raw)):
                if replace_expected != actual:
                    raise Conflict('Existing immutable timeline output has different bytes')
            else:
                return dict(relative_path=relative,sha256=digest,byte_size=len(raw))
        if replace_expected is not None and existing() != replace_expected:
            raise Conflict('Latest Comment view changed before atomic revision replacement')
        if not guard.guarded_atomic(path, [raw]):
            if is_nas(path):
                raise OSError('NAS publication requires a pinned write guard')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name('.'+path.name+'.'+str(uuid4())+'.partial')
            try:
                with temporary.open('xb') as handle:
                    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, path)
                fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                temporary.unlink(missing_ok=True)
        if existing() != (digest, len(raw)):
            raise Conflict('Published timeline output failed read-back verification')
        return dict(relative_path=relative, sha256=digest, byte_size=len(raw))

"""Publish immutable timeline views inside the existing device/day categories."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from uuid import uuid4

from psycopg.types.json import Jsonb

from .archive_layout import day_directory, verify_archive_contract
from .components import pipeline_module
from .domain import Conflict, clean, fingerprint
from .media import hash_file, safe_path
from .rules import Rules
from .storage import is_nas, publication_config
from .timeline import day_bounds
from .timeline_report import render_report
from .timeline_timing import PublicationTiming


class TimelinePublication:
    def __init__(self, runtime, settings):
        self.runtime, self.settings = runtime, settings
        self.core, self.customer = runtime.timeline.core, runtime.timeline.customer

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
                snapshot = conn.execute('SELECT r.* FROM hj_daily_report r WHERE r.customer_id=%s AND NOT EXISTS(SELECT 1 FROM hj_timeline_publication p WHERE p.customer_id=r.customer_id AND p.day=r.day AND p.revision=r.revision AND p.destination_fingerprint=%s) ORDER BY r.day,r.revision LIMIT 1', (self.customer, destination)).fetchone()
                if not snapshot:
                    return None
                start, end = day_bounds(snapshot['day'])
                media = conn.execute('SELECT m.*,a.verified_sha256,i.payload AS capture_receipt FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id AND i.event_id=c.source_capture_id AND i.revision=c.source_revision WHERE m.customer_id=%s AND m.start_at<%s AND m.end_at>=%s ORDER BY m.stream_key,m.start_at,m.media_id', (self.customer, end, start)).fetchall()
            if not media:
                return dict(enabled=True, status='no_known_clock_media', day=str(snapshot['day']))
            # Recorder/photo/voice source IDs are provenance, not archive camera keys.
            # Reuse the explicit producer camera_id; never strip prefixes or hash names.
            cameras_by_media = {str(m['media_id']): m['capture_receipt'].get('camera_id') or m['stream_key'] for m in media}
            cameras = sorted(set(cameras_by_media.values()))
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
            for camera in cameras:
                base = roots[camera]
                scoped = [a for a in activities.values() if any(cameras_by_media.get(e['media_id']) == camera for e in a['evidence'])]
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
                with timing.phase('write_and_full_readback'),guard.storage_guard(publication_config(self.settings)):
                    for relative, raw in files.items():
                        receipts.append(self._write(root, relative, raw, guard))
                with timing.phase('publication_commit'),self.core.db.transaction() as conn:
                    conn.execute('INSERT INTO hj_timeline_publication(customer_id,day,revision,destination_fingerprint,files) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING', (self.customer, snapshot['day'], snapshot['revision'], destination, Jsonb(receipts)))
            return dict(enabled=True, day=str(snapshot['day']), revision=snapshot['revision'], files=receipts)

    @staticmethod
    def _json(value):
        return (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()

    @staticmethod
    def _write(root, relative, raw, guard):
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
                raise Conflict('Existing immutable timeline output has different bytes')
        elif not guard.guarded_atomic(path, [raw]):
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

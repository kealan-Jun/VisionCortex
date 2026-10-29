"""Opt-in bounded Photos-only admission; producer originals remain read-only.

The existing Ingest/FR reader owns completion, byte stability and capture identity.
Only the local checkpoint and lock are written here. No video traversal or model
execution occurs; later completed photo groups are discovered on the next tick.
"""
from __future__ import annotations

from datetime import date
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import stat
import time

from .components import pipeline_module
from .domain import Conflict, fingerprint
from .ingest import Ingest, read_json
from .input_layout import photo_location, require_finished_upload, upload_markers
from .native_refresh import _directory
from .producer_ingress import _local, _pairs
from .storage import publication_config

SCHEMA = 'vc-lab-photo-ingress/1'
ENV = 'VC_LAB_PHOTO_INGRESS_MANIFEST'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
MAX_ENTRIES = 8192


def load_manifest(settings, path=None):
    path = path or os.environ.get(ENV)
    if not path:
        return None
    path = _local(path)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 65536:
        raise ValueError('Private owner-only photo ingress manifest required')
    doc = json.loads(path.read_text(), object_pairs_hook=_pairs)
    if (not isinstance(doc, dict) or doc.get('schema_version') != SCHEMA
            or type(doc.get('enabled')) is not bool or set(doc) - {
                'schema_version', 'enabled', 'revision', 'source_root', 'not_before_day',
                'max_photos_per_tick'}):
        raise ValueError('Explicit versioned Photos-only ingress manifest required')
    if not doc['enabled']:
        return None
    if (not isinstance(doc.get('revision'), str) or not 1 <= len(doc['revision']) <= 128
            or date.fromisoformat(doc['not_before_day']).isoformat() != doc['not_before_day']):
        raise ValueError('Photo ingress revision and exact lower day bound required')
    expected = settings.source_root or settings.runtime_root/'incoming'
    if doc.get('source_root') != str(expected):
        raise ValueError('Photo ingress source root differs from Settings.source_root')
    doc.setdefault('max_photos_per_tick', 4)
    if type(doc['max_photos_per_tick']) is not int or not 1 <= doc['max_photos_per_tick'] <= 4:
        raise ValueError('Photo ingress admits at most four photographs per tick')
    return doc


def _entries(root, relative, guard, budget):
    values = []
    with _directory(root, relative, guard) as directory:
        if directory is None:
            return values
        with os.scandir(directory) as entries:
            for entry in entries:
                budget[0] -= 1
                if budget[0] < 0:
                    raise Conflict('Photos-only discovery exceeds its 8192-entry bound')
                if entry.is_symlink():
                    continue
                info = entry.stat(follow_symlinks=False)
                values.append((entry.name, stat.S_ISDIR(info.st_mode), stat.S_ISREG(info.st_mode),
                    info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino))
    return sorted(values, reverse=True)


def candidates(root, doc, guard):
    """Descend only date_camera/MetaData/Photos/time, newest first."""
    budget, found = [MAX_ENTRIES], []
    contract = pipeline_module('device_day_contract')
    for row in _entries(root, '', guard, budget):
        name, directory = row[:2]
        if not directory or name[:10] < doc['not_before_day']:
            continue
        try:
            contract.validate_archive_name(name)
        except ValueError:
            continue
        camera = name[11:]
        photos = name+'/MetaData/Photos'
        for group in _entries(root, photos, guard, budget):
            if not group[1]:
                continue
            relative = photos+'/'+group[0]
            members = _entries(root, relative, guard, budget)
            images = [m for m in members if m[2] and Path(m[0]).suffix.lower() in IMAGE_SUFFIXES]
            receipt = next((m for m in members if m[0] == 'PhotoReceipt.json' and m[2]), None)
            choices = [(receipt, len(images))] if receipt else (
                [(images[0], len(images))] if 0 < len(images) <= 3 else [(m, 1) for m in images])
            for item, cost in choices:
                if item is None:
                    continue
                path = root/relative/item[0]
                state_members = members if receipt or len(images) <= 3 else [item]
                signature = fingerprint(dict(members=state_members, uploading=upload_markers(root, path)))
                found.append(dict(path=path, camera=camera, cost=cost, signature=signature))
    def latest(item):
        parts = item['path'].relative_to(root).parts
        return parts[0][:10], parts[-2], parts[-1], item['camera']
    return sorted(found, key=latest, reverse=True)


class _BudgetIngest(Ingest):
    """Keep the bound even if a producer adds group members during discovery."""
    def accept(self, receipt, **kwargs):
        if self.remaining <= 0:
            raise Conflict('Photos-only tick admission budget is exhausted')
        self.remaining -= 1
        result = super().accept(receipt, **kwargs)
        self.accepted.append(result)
        return result


class PhotoIngress:
    def __init__(self, core, settings, manifest_path=None):
        self.core, self.settings, self.manifest_path = core, settings, manifest_path

    def step(self):
        return self.tick()

    def tick(self):
        doc = load_manifest(self.settings, self.manifest_path)
        if doc is None:
            return dict(enabled=False)
        if self.settings.processing_profile != 'timeline-v1':
            raise Conflict('Photos-only ingress requires timeline-v1')
        _local(self.settings.runtime_root, directory=True)
        folder = self.settings.runtime_root/'photo-ingress'
        folder.mkdir(exist_ok=True)
        _local(folder, directory=True)
        state = folder/'Checkpoint.sqlite3'
        if state.exists() and (_local(state).stat().st_size > 16*1024**2):
            raise Conflict('Local photo ingress checkpoint exceeds 16MiB')
        lock = os.open(folder/'.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock, 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            with sqlite3.connect(state, timeout=5) as db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE IF NOT EXISTS metadata(identity TEXT NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS items(path TEXT PRIMARY KEY, signature TEXT NOT NULL, status TEXT NOT NULL, retry_after REAL NOT NULL)')
                identity = fingerprint(dict(customer_id=str(self.core.customer), root=doc['source_root'],
                    not_before_day=doc['not_before_day']))
                previous = db.execute('SELECT identity FROM metadata').fetchone()
                if previous and previous[0] != identity:
                    raise Conflict('Photo checkpoint belongs to another customer/source/day floor')
                if not previous:
                    db.execute('INSERT INTO metadata VALUES(?)', (identity,))
                    db.commit()
                return self._tick(db, doc)

    def _tick(self, db, doc):
        root = self.settings.incoming  # Existing Settings verifies all configured NAS identities.
        summary = dict(enabled=True, processed_photos=0, duplicates=0, attempted_photos=0,
            skipped=0, pending=[], errors=[], scope='MetaData/Photos_only', video_admissions=0)
        config = publication_config(self.settings)
        ingress = _BudgetIngest(self.core, self.settings)
        ingress.remaining, ingress.accepted = doc['max_photos_per_tick'], []
        with pipeline_module('device_day_write_guard').storage_guard(config) as guard:
            choices = candidates(root, doc, guard)
            for item in choices:
                if summary['attempted_photos'] >= doc['max_photos_per_tick'] or len(summary['errors']) >= 16:
                    break
                path = item['path']
                relative = path.relative_to(root).as_posix()
                old = db.execute('SELECT signature,status,retry_after FROM items WHERE path=?', (relative,)).fetchone()
                if old and old[0] == item['signature']:
                    if old[1] == 'accepted':
                        summary['skipped'] += 1
                        continue
                    if old[2] > time.time():
                        continue
                try:
                    require_finished_upload(root, path)
                    cost = item['cost']
                    if path.name == 'PhotoReceipt.json':
                        receipt = read_json(path)
                        photos = receipt.get('photos')
                        if receipt.get('camera_id') != item['camera'] or not isinstance(photos,list) or not photos:
                            raise Conflict('Photo receipt camera/membership differs from its native path')
                        cost = len(photos)
                        for photo in photos:
                            location = photo_location((path.parent/photo['filename']).relative_to(root).as_posix())
                            if location.camera != item['camera']:
                                raise Conflict('Photo receipt member escapes its original camera')
                    else:
                        photo_location(relative)
                    if cost > doc['max_photos_per_tick']-summary['attempted_photos']:
                        summary['pending'].append(dict(relative_path=relative,reason='whole_group_exceeds_remaining_photo_budget'))
                        continue
                    if time.time()-path.stat().st_mtime < .5:
                        raise Conflict('Photograph/receipt is still within the stable-write wait')
                    guard.validate()
                    result = ingress.photo_receipt(path) if path.name == 'PhotoReceipt.json' else ingress.directory_photo(path)
                    guard.validate()
                    if not isinstance(result,list) or len(result)>cost:
                        raise Conflict('Original photo admission exceeded its preflight group budget')
                    status, delay = 'accepted', 0
                except (ValueError, OSError, Conflict) as exc:
                    summary['errors'].append(dict(relative_path=relative,error_code=type(exc).__name__))
                    status, delay = 'needs_attention', time.time()+2
                summary['attempted_photos'] = doc['max_photos_per_tick']-ingress.remaining
                summary['duplicates'] = sum(bool(r.get('duplicate')) for r in ingress.accepted)
                summary['processed_photos'] = sum(not r.get('duplicate',False) for r in ingress.accepted)
                db.execute('INSERT INTO items VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET signature=excluded.signature,status=excluded.status,retry_after=excluded.retry_after',
                    (relative,item['signature'],status,delay))
                db.commit()
        return summary

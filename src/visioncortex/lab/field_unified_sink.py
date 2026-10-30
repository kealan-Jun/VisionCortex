"""Project committed original FR events into the unified Comment/OCR sink.

The original store owns SQLite, receipt acknowledgement, event membership and
Result allocation. This facade never infers, rebinds, moves evidence or copies
raw media. Each projection is an immutable, source-pinned event generation.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

from .archive_layout import verify_archive_contract
from .components import pipeline_module
from .domain import fingerprint
from .media import safe_path
from .storage import is_nas, publication_config, verify_root
from .timeline_publication import TimelinePublication

SCHEMA = 'vc-lab-fr-unified-sink/1'
MAX_BYTES = 32 * 1024**2
MEMBER = re.compile(r'(?:Evidence|Readings|UnboundReadings|Result(?:\d+)?)\.json\Z')


def load_manifest(path):
    path = Path(path)
    if (not path.is_absolute() or is_nas(path) or path.resolve(strict=True) != path
            or path.stat().st_size > 65536 or path.stat().st_mode & 0o077):
        raise ValueError('Canonical local private mode-0600 manifest required')
    value = json.loads(path.read_bytes())
    required = {'schema_version','enabled','revision','source_archive_root','destination_root',
                'runtime_root','initial_sequence','max_events_per_tick','fr_package_root'}
    if (set(value) != required or value['schema_version'] != SCHEMA or value['enabled'] is not True
            or type(value['revision']) is not str or not 1 <= len(value['revision']) <= 64
            or type(value['initial_sequence']) is not int or value['initial_sequence'] < 0
            or type(value['max_events_per_tick']) is not int or not 1 <= value['max_events_per_tick'] <= 20):
        raise ValueError('Explicit bounded FR projection manifest required')
    for key in ('runtime_root','fr_package_root'):
        p = Path(value[key])
        if not p.is_absolute() or is_nas(p) or p.resolve(strict=True) != p or not p.is_dir():
            raise ValueError('Canonical existing FR sink paths required')
    # Guard every required kernel mount/marker BEFORE resolve/is_dir touches
    # either NAS root. A cached deployment preflight is never sufficient.
    verify_root(Path(value['source_archive_root']),project_destination=False)
    verify_root(Path(value['destination_root']))
    for key in ('source_archive_root','destination_root'):
        p=Path(value[key])
        if not p.is_absolute() or p.resolve(strict=True)!=p or not p.is_dir():
            raise ValueError('Canonical existing guarded FR archive roots required')
    if value['source_archive_root'] == value['destination_root']:
        raise ValueError('Projection destination must be separate from original FR owner')
    return value


class UnifiedFRSink:
    def __init__(self, store, manifest, *, publisher=TimelinePublication):
        self.store, self.manifest = store, dict(manifest)
        self.publisher = publisher
        self.source = Path(manifest['source_archive_root'])
        self.destination = Path(manifest['destination_root'])
        self.runtime = Path(manifest['runtime_root'])
        if Path(store.configured_root()).absolute() != self.source:
            raise ValueError('Original FR store must retain its declared source archive root')
        self.identity = fingerprint(manifest)
        self.cursor = self.runtime / 'fr-unified-projection-cursor.json'
        self.last_projection = None

    def __getattr__(self, key):
        return getattr(self.store, key)

    def _config(self):
        verify_root(self.source, project_destination=False)
        verify_root(self.destination)
        verify_archive_contract(self.destination)
        return publication_config(SimpleNamespace(incoming=self.destination,nas_marker_sha256=None,
            runtime_root=self.runtime,cache=self.runtime/'cache'))

    def step(self, batch_size=20):
        # Check the sink before asking the original owner to process its queue.
        self._config()
        result = self.store.step(batch_size)
        self.last_projection = self.project_once()
        return result  # Preserve the original store's exact return contract.

    def snapshot(self):
        value=self.store.snapshot()
        if self.last_projection is not None:value=dict(value,unified_projection=self.last_projection)
        return value

    def project_once(self, *, sequence=None):
        config = self._config()
        with (self.runtime/'fr-unified-projection.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            cursor = self.manifest['initial_sequence']
            if self.cursor.exists():
                saved = json.loads(self.cursor.read_bytes())
                if saved['manifest_fingerprint'] != self.identity:
                    raise ValueError('Existing projection cursor belongs to another private manifest')
                cursor = saved['sequence']
            with self.store.core['db']() as conn:
                rows = conn.execute('''SELECT * FROM archive_outbox WHERE archived_at IS NOT NULL
                    AND published_at IS NOT NULL AND readable_at IS NOT NULL
                    AND seq>? AND (? IS NULL OR seq=?) ORDER BY seq LIMIT ?''',
                    (-1 if sequence is not None else cursor,sequence,sequence,self.manifest['max_events_per_tick'])).fetchall()
            if sequence is not None and not rows:
                raise ValueError('Requested original FR receipt is not committed and readable')
            outcomes = []
            guard = pipeline_module('device_day_write_guard')
            with guard.storage_guard(config):
                for row in rows:
                    outcomes.append(self._project(row,guard))
                    if sequence is None:
                        value = dict(schema_version=SCHEMA,manifest_fingerprint=self.identity,sequence=row['seq'])
                        temporary = self.cursor.with_suffix('.partial')
                        with temporary.open('wb') as handle:
                            handle.write(TimelinePublication._json(value));handle.flush();os.fsync(handle.fileno())
                        os.replace(temporary,self.cursor)
                        fd=os.open(self.runtime,os.O_RDONLY|os.O_DIRECTORY)
                        try:os.fsync(fd)
                        finally:os.close(fd)
            return dict(processed=len(rows),events=outcomes,model_calls=0,raw_media_copied=False)

    def _read(self, relative, guard):
        raw = self.publisher._read_bytes(self.source,relative,guard)
        if raw is None:
            raise FileNotFoundError('Original FR committed artifact is unavailable')
        return raw

    def _project(self, row, guard):
        receipt_raw = self._read(row['receipt_path'],guard)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        if Path(row['receipt_path']).stem.split('-',1)[-1] != receipt_sha:
            raise ValueError('Original FR immutable receipt hash differs')
        receipt = json.loads(receipt_raw)
        if (receipt['sequence'],receipt['entity'],receipt['entity_id'],receipt['document']) != (
                row['seq'],row['entity'],row['entity_id'],json.loads(row['document'])):
            raise ValueError('Original FR receipt differs from its SQLite committed snapshot')
        ledger_raw = self._read('.System/MigrationMap.json',guard)
        ledger = json.loads(ledger_raw)
        from field_recognition.archive.archive_browse import token
        alias = 'Records/'+''.join(w.title() for w in row['entity'].split('_'))+'/'+token(row['entity_id'])+'/Record.json'
        seen = set()
        relative = alias
        while relative in ledger['aliases']:
            if relative in seen:raise ValueError('Original FR alias cycle')
            seen.add(relative);relative=ledger['aliases'][relative]
        parts = PurePosixPath(relative).parts
        if len(parts)<4 or parts[1] not in {'VoicePhotoReadings','VideoReadings'}:
            return dict(sequence=row['seq'],status='not_an_ocr_reading_event',source_receipt_sha256=receipt_sha)
        source_folder = PurePosixPath(relative).parent.as_posix()
        evidence_raw = self._read(source_folder+'/Evidence.json',guard)
        evidence = json.loads(evidence_raw)
        if (evidence.get('schema') != 'field-recognition-event/2' or evidence.get('category') != parts[1]
                or evidence.get('camera_id') != parts[0][11:]
                or row['seq'] not in {v['sequence'] for v in evidence.get('receipt_versions',[])}):
            raise ValueError('Original FR event membership/camera/schema differs')
        evidence_sha=hashlib.sha256(evidence_raw).hexdigest()
        folder=parts[0]+'/Comment/OCR/'+parts[1]+'/'+parts[-2]+'-fr-'+evidence_sha[:16]
        member_paths={source_folder+'/Evidence.json':evidence_raw}
        required={'Readings.json'}
        required.update(Path(m['result']).name for m in evidence.get('measurement_files',[]))
        if evidence.get('unbound_readings'):required.add('UnboundReadings.json')
        for name in sorted(required):
            if not MEMBER.fullmatch(name):raise ValueError('Original FR event member is outside schema')
            member_paths[source_folder+'/'+name]=self._read(source_folder+'/'+name,guard)
        for entry in evidence.get('measurement_files',[]):
            if entry['result'] not in member_paths or hashlib.sha256(member_paths[entry['result']]).hexdigest()!=entry['sha256']:
                raise ValueError('Original FR Result member hash differs')
        if sum(map(len,member_paths.values()))>MAX_BYTES:raise ValueError('FR projection exceeds bounded JSON bytes')
        mapping={p:folder+'/'+Path(p).name for p in member_paths}

        def rewrite(value):
            if isinstance(value,dict):
                result={k:rewrite(v) for k,v in value.items()}
                # Raw photos stay at the original owner. Keep the original path
                # and SHA, with an explicit openable URI/base rather than a fake
                # relative reference under the new archive.
                if isinstance(value.get('path'),str) and value['path'] not in mapping and value.get('sha256'):
                    original=safe_path(self.source,value['path'],must_exist=False)
                    result.update(source_uri=original.as_uri(),path_base='original_fr_archive',
                        source_archive_uri=self.source.as_uri(),raw_bytes_copied=False)
                return result
            if isinstance(value,list):return [rewrite(v) for v in value]
            return mapping.get(value,value) if isinstance(value,str) else value

        outputs={}
        for path,raw in member_paths.items():
            if Path(path).name=='Evidence.json':continue
            outputs[path]=raw if Path(path).name.startswith('Result') else self.publisher._json(rewrite(json.loads(raw)))
        planned={}
        for path,raw in member_paths.items():
            original=json.loads(raw)
            # Exact six-field Results and numeric ledgers retain original bytes;
            # only Evidence navigation gains explicit source/projection refs.
            if Path(path).name=='Evidence.json':
                document=rewrite(original)
                for key in ('readings','unbound_readings'):
                    item=original.get(key)
                    if isinstance(item,dict) and item.get('path') in outputs:
                        document[key]['sha256']=hashlib.sha256(outputs[item['path']]).hexdigest()
                document['unified_projection']=dict(schema_version=SCHEMA,revision=self.manifest['revision'],
                    source_archive_uri=self.source.as_uri(),source_event=relative,source_evidence_sha256=evidence_sha,
                    source_receipt_versions=evidence.get('receipt_versions',[]),original_model_and_membership_unchanged=True,
                    source_members=[dict(path=p,sha256=hashlib.sha256(v).hexdigest(),byte_size=len(v)) for p,v in member_paths.items()])
                published=self.publisher._json(document)
            else:published=outputs[path]
            planned[path]=published
        # Source views may change while another owner publishes; never advance
        # our cursor based on a mixed generation. Existing immutable partials
        # are reusable after exact source stability is established on retry.
        if self._read(source_folder+'/Evidence.json',guard)!=evidence_raw:
            raise OSError('Original FR event changed during projection')
        for path,raw in member_paths.items():
            if self._read(path,guard)!=raw:raise OSError('Original FR member changed during projection')
        files=[]
        for path in sorted(planned,key=lambda p:(Path(p).name=='Evidence.json',p)):
            if Path(path).name=='Evidence.json':
                for member,raw in member_paths.items():
                    if self._read(member,guard)!=raw:raise OSError('Original FR member changed before Evidence commit')
            files.append(self.publisher._write(self.destination,mapping[path],planned[path],guard))
        return dict(sequence=row['seq'],status='projected',directory=folder,files=files,
            source_evidence_sha256=evidence_sha,model_calls=0,raw_media_copied=False)

"""Reuse completed native ASR in Timeline without inference or new Audio folders.

The native transcript remains its owner's file. Imports pin the original audio,
unaltered relative sentence offsets, actual retained model and immutable text.
No recorded prose is executed as a reviewed experiment command.
"""
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import PurePosixPath

from .components import pipeline_module
from .domain import Conflict, clean, fingerprint, ident
from .rules import Rules
from psycopg.types.json import Jsonb
from .producer_ingress import _read, _relative, load_manifest, inventory_snapshot
from .storage import publication_config
from .timeline import Timeline

PREFIX = 'native-comment-import:'
SCHEMA = 'vc-lab-native-comment-import/1'


def inventory_alias(source, row, prefix):
    """An explicit inventory-root alias, never a time/name proximity match."""
    native=source['receipt']['native_record']
    expected=(prefix+'/' if prefix else '')+native['relative_path']
    if (row.get('relative_path')!=expected or row.get('processable') is not True
            or row.get('camera_key')!=native.get('camera_key')
            or any(str(row.get(k))!=str(native.get(k)) for k in
                   ('recording_session_id','recording_start_us','recording_end_us'))):
        raise Conflict('Original inventory alias differs from exact recorder source')
    if not row.get('recording_id') or not row.get('source_signature') or not row.get('audio',{}).get('source_signature'):
        raise Conflict('Original inventory alias lacks source signatures')
    return dict(recording_id=row['recording_id'],audio_signature=row['audio']['source_signature'],
                inventory_source_signature=row['source_signature'],inventory_relative_path=expected)


def prepare(document, source, transcript_path):
    receipt = source['receipt']
    native = receipt.get('native_record') or {}
    aliases=source.get('recording_aliases',[])
    recording_id = document.get('recording_id')
    allowed={native.get('recording_id'),*[a['recording_id'] for a in aliases]}
    archive = receipt['relative_path'].split('/MetaData/')[0].split('/MetaVideo/')[0]
    _relative(transcript_path)
    if (not transcript_path.startswith(archive+'/Comment/Audio/')
            or PurePosixPath(transcript_path).name != 'Transcript.json'
            or recording_id not in allowed
            or document.get('device') != receipt.get('camera_id')
            or document.get('status') != 'completed' or document.get('outcome') != 'transcribed'):
        raise Conflict('Native transcript status, camera, recording or canonical Audio path differs')
    audio = document.get('audio') or {}
    if (audio.get('status') != 'provided' or audio.get('capture_complete') is not True
            or audio.get('source_signature') not in {native.get('audio',{}).get('source_signature'),
                *[a['audio_signature'] for a in aliases if a['recording_id']==recording_id]}):
        raise Conflict('Native transcript audio signature differs from admitted source')
    refs = [r for r in audio.get('artifacts',[]) if r.get('sha256') == source['verified_sha256']]
    if (len(refs) != 1 or archive+'/'+refs[0].get('path','') != receipt['relative_path']
            or source['media_version'] != source['version'] or source['evidence_status'] != 'available'):
        raise Conflict('Native transcript audio bytes/version do not match the admitted original')
    sentences = document.get('sentences') or []
    if not 1 <= len(sentences) <= 512:
        raise Conflict('Bounded native sentence list required')
    units, models, unaligned, texts = [], [], [], []
    duration = Decimal(source['duration_seconds'])
    for sentence in sentences:
        text = sentence.get('text')
        if not isinstance(text,str) or not text.strip() or len(text)>20000:
            raise Conflict('Native transcript sentence is empty or exceeds bound')
        texts.append(text);models.append(sentence.get('model'))
        ref = sentence.get('audio_ref') or {}
        if (ref.get('sha256') != source['verified_sha256']
                or archive+'/'+ref.get('path','') != receipt['relative_path']):
            raise Conflict('Native sentence refers to different audio')
        try:
            lo,hi = Decimal(str(sentence['audio_start_seconds'])),Decimal(str(sentence['audio_end_seconds']))
        except (InvalidOperation,KeyError) as exc:
            raise Conflict('Native sentence needs original source-relative alignment') from exc
        if not lo.is_finite() or not hi.is_finite() or not 0 <= lo < hi <= duration:
            unaligned.append(dict(text=text,reported_start_seconds=str(lo),reported_end_seconds=str(hi),
                                  status='source_bounds_conflict_not_clamped'))
            continue
        units.append(dict(text=text,start_seconds=str(lo),end_seconds=str(hi)))
    if not units:
        raise Conflict('Native sentence alignment escaped the original audio')
    text='\n'.join(texts)
    if len(text)>20000:
        raise Conflict('Native transcript text exceeds one bounded Timeline activity')
    original=document.get('original_transcript') or {}
    if not original.get('path','').startswith('Comment/.Evidence/Recognition/'+str(recording_id)+'/'):
        raise Conflict('Native transcript requires its retained recognition text reference')
    _relative(original['path'])
    if len(original.get('sha256',''))!=64:
        raise Conflict('Native original transcript digest required')
    ids=sorted({str(m.get('provider'))+'/'+str(m.get('model')) for m in models if isinstance(m,dict)})
    return clean(dict(schema_version=SCHEMA,media_id=source['media_id'],media_version=source['version'],
        audio_sha256=source['verified_sha256'],recording_id=recording_id,source_archive=archive,
        native_recording_id=native.get('recording_id'),recording_aliases=aliases,
        camera_code=receipt['camera_id'],text=text,aligned_units=units,unaligned_units=unaligned,models=models,
        alignment_status='partial_original_sentence_bounds' if unaligned else 'original_source_relative_sentences',
        model_revision=ids[0] if len(ids)==1 else 'retained-native-asr/mixed',
        source_start_seconds='0' if unaligned else str(min(Decimal(u['start_seconds']) for u in units)),
        source_end_seconds=str(duration) if unaligned else str(max(Decimal(u['end_seconds']) for u in units)),
        reference=dict(transcript=transcript_path,text=transcript_path[:-4]+'txt',
            original_transcript=archive+'/'+original['path'],original_transcript_sha256=original['sha256']),
        physical_clock_verified=False,human_verified=False,new_model_calls=0))


class NativeCommentIntake:
    def __init__(self,core,settings):
        self.core,self.settings=core,settings

    def source(self,media_id):
        with self.core.db.transaction(readonly=True) as conn:
            row=conn.execute('''SELECT m.media_id,m.kind,m.duration_seconds,a.version,a.version AS media_version,
                a.verified_sha256,a.evidence_status,i.payload AS receipt FROM hj_timeline_media m
                JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
                JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                    AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                WHERE m.customer_id=%s AND m.media_id=%s''',(self.core.customer,ident(media_id))).fetchone()
        if not row or row['kind']!='audio' or not row['receipt'].get('native_record'):
            raise Conflict('Existing admitted recorder-native audio required')
        doc=load_manifest(self.settings)
        if doc:
            records,_=inventory_snapshot(doc)
            prefix=doc.get('inventory_relative_prefix','')
            expected=(prefix+'/' if prefix else '')+row['receipt']['native_record']['relative_path']
            matches=[r for r in records if r.get('relative_path')==expected]
            if len(matches)>1:
                raise Conflict('Conflicting recorder inventory aliases')
            row['recording_aliases']=[inventory_alias(row,r,prefix) for r in matches]
        return row

    def consume(self,media_id):
        source=self.source(media_id)
        archive=source['receipt']['relative_path'].split('/MetaData/')[0].split('/MetaVideo/')[0]
        recording_ids={source['receipt']['native_record']['recording_id'],
                       *[a['recording_id'] for a in source.get('recording_aliases',[])]}
        root=self.settings.incoming
        guard_module=pipeline_module('device_day_write_guard')
        plans=[]
        with guard_module.storage_guard(publication_config(self.settings)) as guard:
            raw=_read(root,archive+'/Comment/Audio/Index.json',guard,8*1024**2)
            index=json.loads(raw) if raw is not None else {}
            entries=[e for e in index.get('entries',[]) if e.get('recording_id') in recording_ids]
            # Native publication can retain a completed transcript before its
            # mutable index catches up. Resolve only the exact recorder audio
            # envelope using the existing owner's naming function; never scan
            # folders or join nearby timestamps. prepare() still checks the
            # original recording ID, audio signature, hash and sentence refs.
            audio=source['receipt']['native_record'].get('audio') or {}
            lo,hi=audio.get('start_us'),audio.get('end_us')
            fallback=None
            if type(lo) is int and type(hi) is int and 0<lo<hi:
                folder=pipeline_module('device_day_content').time_folder(lo,hi)
                fallback='Comment/Audio/'+folder+'/Transcript.json'
                if not any(e.get('content')==fallback for e in entries):
                    entries.append(dict(content=fallback))
            if len(entries)>16:
                raise Conflict('Native audio import exceeds 16 original transcript spans')
            seen=set()
            for entry in entries:
                content=entry.get('content','')
                if not content.startswith('Comment/Audio/') or content in seen:
                    continue
                seen.add(content)
                relative=archive+'/'+_relative(content)
                raw=_read(root,relative,guard,4*1024**2)
                if raw is None:
                    continue
                document=json.loads(raw)
                if document.get('status')!='completed' or document.get('outcome')!='transcribed':
                    continue
                if content==fallback and (document.get('start_us'),document.get('end_us'))!=(lo,hi):
                    raise Conflict('Native transcript differs from its exact recorder audio envelope')
                plan=prepare(document,source,relative)
                backend=self.settings.native_comment_backend_root
                if backend is not None:
                    # The native owner's configured Comment evidence store is
                    # a capability, never a path selected by transcript text.
                    # Retain logical references; do not relocate or regenerate
                    # original text. A configured backend fails closed instead
                    # of silently using an older archive copy.
                    from visioncortex.comment_evidence import NAMESPACE
                    original=_read(backend,NAMESPACE+'/'+plan['reference']['original_transcript'],guard,4*1024**2)
                else:
                    original=_read(root,plan['reference']['original_transcript'],guard,4*1024**2)
                if original is None or hashlib.sha256(original).hexdigest()!=plan['reference']['original_transcript_sha256']:
                    raise Conflict('Original retained transcript text hash differs')
                if _read(root,plan['reference']['text'],guard,4*1024**2) is None:
                    raise Conflict('Native transcript reading text is missing')
                plans.append(plan)
            guard.validate()
        results=[]
        for plan in plans:
            key=PREFIX+fingerprint(plan)
            def action(conn,plan=plan,key=key):
                current=conn.execute('SELECT version,verified_sha256,evidence_status FROM media_asset WHERE customer_id=%s AND id=%s FOR SHARE',
                    (self.core.customer,ident(plan['media_id']))).fetchone()
                if (not current or current['version']!=plan['media_version']
                        or current['verified_sha256']!=plan['audio_sha256'] or current['evidence_status']!='available'):
                    raise Conflict('Original audio changed before native Comment import')
                tasks=conn.execute("SELECT id,status,attempts FROM hj_timeline_task WHERE customer_id=%s AND media_id=%s AND kind='transcription' AND analysis_revision='initial' FOR UPDATE",
                                   (self.core.customer,ident(plan['media_id']))).fetchall()
                if any(t['status']!='awaiting_provider' or t['attempts']!=0 for t in tasks):
                    raise Conflict('Native ASR reuse cannot replace a leased or previously executed Timeline task')
                activity=Timeline(self.core).add_activity(key+':activity','speech','speech',plan['text'],
                    [dict(media_id=plan['media_id'],start_seconds=plan['source_start_seconds'],end_seconds=plan['source_end_seconds'])],
                    model_revision=plan['model_revision'],payload=dict(native_import_key=key,
                        native_transcript_reference=plan['reference'],aligned_units=plan['aligned_units'],
                        unaligned_units=plan['unaligned_units'],native_alignment_status=plan['alignment_status'],
                        native_asr_models=plan['models'],original_native_recording_id=plan['recording_id'],
                        execution_verification='retained_native_ASR_no_new_inference',
                        comment_candidate=dict(status='pending_interpretation',alignment_available=not plan['unaligned_units']),
                        source_text_is_untrusted_evidence=True,physical_action_confirmed=False))
                reused=dict(schema_version=SCHEMA,outcome='reused_completed_native_ASR',activity_ids=[str(activity['id'])],
                            original_models=plan['models'],reference=plan['reference'],model_invoked=False,new_model_calls=0)
                for task in tasks:
                    conn.execute("UPDATE hj_timeline_task SET status='completed',completed_at=now(),model_revision=%s,request=%s,result=%s WHERE customer_id=%s AND id=%s",
                        (plan['model_revision'],Jsonb(dict(input_mode='retained_native_ASR',original_source=plan,processing_rules=Rules().identity())),
                         Jsonb(reused),self.core.customer,task['id']))
                return dict(schema_version=SCHEMA,activity_id=activity['id'],source=plan,
                            reference=plan['reference'],reused_task_ids=[str(t['id']) for t in tasks],new_model_calls=0)
            results.append(self.core.command(key,plan,action))
        return dict(status='retained_native_ASR_linked' if results else 'pending_native_ASR',
                    media_id=str(media_id),results=results,new_model_calls=0)

    def tick(self,limit=4):
        if type(limit) is not int or not 1<=limit<=16:
            raise ValueError('Bounded native Comment import limit required')
        with self.core.db.transaction(readonly=True) as conn:
            rows=conn.execute('''SELECT m.media_id FROM hj_timeline_media m JOIN capture c
                ON c.customer_id=m.customer_id AND c.id=m.capture_id
                JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
                    AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
                WHERE m.customer_id=%s AND m.kind='audio' AND i.payload ? 'native_record'
                AND NOT EXISTS (SELECT 1 FROM command_receipt r WHERE r.customer_id=m.customer_id
                    AND starts_with(r.key,'native-comment-import:')
                    AND r.result#>>'{source,media_id}'=m.media_id::text)
                ORDER BY c.received_at DESC,m.media_id LIMIT %s''',(self.core.customer,limit)).fetchall()
        return dict(results=[self.consume(r['media_id']) for r in rows],new_model_calls=0)


def retained_reference(core,activity,sources,camera):
    key=activity['payload'].get('native_import_key')
    with core.db.transaction(readonly=True) as conn:
        row=conn.execute('SELECT result FROM command_receipt WHERE customer_id=%s AND key=%s',
                         (core.customer,key)).fetchone()
    if not row or not isinstance(key,str) or not key.startswith(PREFIX):
        raise Conflict('Retained native ASR reference requires its committed import receipt')
    result=row['result'];source=result['source']
    if (str(result['activity_id'])!=activity['id'] or source['camera_code']!=camera
            or source['media_id'] not in sources
            or sources[source['media_id']]['source_sha256']!=source['audio_sha256']
            or result['reference']!=activity['payload'].get('native_transcript_reference')):
        raise Conflict('Native ASR reference scope or source differs')
    return dict(activity_id=activity['id'],source_type='speech',**result['reference'],
                alignment_status='retained_original_native_source_alignment',native_owner_files_reused=True,
                new_transcript_files_generated=False,new_model_calls=0)

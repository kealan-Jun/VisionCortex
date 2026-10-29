"""Immutable selected-clip provenance and original-media time mapping.

No detector, invented experiment boundaries or overwrite of producer indexes.
"""
from decimal import Decimal

from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, fingerprint, ident
from .media import Media
from .timeline import offset_time, seconds

SCHEMA='vc-lab-retained-clip/1'


def original_spans(document, start, end):
    start,end=seconds(start),seconds(end)
    output=[]
    for part in document['sources']:
        child=seconds(part['clip_start_seconds'])
        lo,hi=seconds(part['start_seconds']),seconds(part['end_seconds'])
        a,b=max(start,child),min(end,child+hi-lo)
        if a<b:
            output.append(dict(media_id=part['media_id'],start_seconds=str(lo+a-child),end_seconds=str(lo+b-child)))
    return output


class RetainedClips:
    def __init__(self,timeline,settings):
        self.timeline,self.settings=timeline,settings
        self.core,self.customer=timeline.core,timeline.customer

    def register(self,key,media_id,sources,selection_key,*,run_ids=None,actor,reason):
        if not selection_key or not actor or not reason or not isinstance(sources,list) or not 1<=len(sources)<=32:
            raise ValueError('Explicit selected clip, source intervals, actor and reason required')
        ids={ident(media_id),*(ident(s['media_id']) for s in sources)}
        if ident(media_id) in {ident(s['media_id']) for s in sources}:
            raise ValueError('Clip cannot be its own source')
        # Verify current files before registration; hashes/versions remain pinned
        # in the descriptor. This does not alone prove content derivation.
        locator=Media(self.core,self.settings)
        for mid in ids:
            if locator.locate(mid)['status']!='available':
                raise Conflict('Clip/source file is unavailable or has changed')
        def action(conn):
            rows=conn.execute('SELECT m.*,a.version,a.verified_sha256 FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id WHERE m.customer_id=%s AND m.media_id=ANY(%s)',(self.customer,list(ids))).fetchall()
            by_id={str(r['media_id']):r for r in rows}
            clip=by_id.get(str(ident(media_id)))
            if not clip or len(rows)!=len(ids):raise Missing('Clip/source outside customer scope')
            if clip['kind']!='video' or not clip['start_at']:raise ValueError('Timed video clip required')
            resolved=[];cursor=Decimal(0);clock=clip['start_at']
            for part in sources:
                parent=by_id[str(ident(part['media_id']))]
                lo,hi=seconds(part['start_seconds']),seconds(part['end_seconds'])
                if parent['kind']!='video' or parent['stream_key']!=clip['stream_key'] or not 0<=lo<hi<=parent['duration_seconds']:
                    raise ValueError('Clip sources must be valid same-camera video intervals')
                if offset_time(parent['start_at'],lo)!=clock:
                    raise ValueError('Clip cannot concatenate unrecorded gaps, overlaps or unknown clocks')
                resolved.append(dict(media_id=str(parent['media_id']),start_seconds=str(lo),end_seconds=str(hi),
                    clip_start_seconds=str(cursor),sha256=parent['verified_sha256'],media_version=parent['version']))
                cursor+=hi-lo;clock=offset_time(clock,hi-lo)
            if cursor!=clip['duration_seconds']:
                raise ValueError('Selected clip duration must exactly cover its declared source intervals')
            for rid in run_ids or []:self.timeline._run(conn,rid)
            document=clean(dict(schema_version=SCHEMA,selection_key=selection_key,media_id=clip['media_id'],
                sha256=clip['verified_sha256'],media_version=clip['version'],sources=resolved,
                run_ids=sorted({str(ident(r)) for r in run_ids or []}),actor=actor,reason=reason,
                derivation_verification='supplied_mapping_not_independent_pixel_proof'))
            old=self.get(conn,media_id)
            if old and old!=document:raise Conflict('Existing clip mapping is immutable')
            # Prevent cycles even when a previously retained clip is a source.
            for parent in {ident(s['media_id']) for s in sources}:
                cycle=conn.execute('WITH RECURSIVE ancestors(id) AS (SELECT %s::uuid UNION SELECT d.parent_id FROM media_derivation d JOIN ancestors a ON a.id=d.child_id WHERE d.customer_id=%s) SELECT 1 FROM ancestors WHERE id=%s',(parent,self.customer,ident(media_id))).fetchone()
                if cycle:raise Conflict('Clip derivation cycle')
                existing=conn.execute('SELECT transform FROM media_derivation WHERE customer_id=%s AND parent_id=%s AND child_id=%s',(self.customer,parent,ident(media_id))).fetchone()
                if existing and existing['transform']!=document:raise Conflict('Existing media derivation differs')
                conn.execute('INSERT INTO media_derivation(customer_id,parent_id,child_id,transform) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING',(self.customer,parent,ident(media_id),Jsonb(document)))
            self.timeline._dirty(conn,clip['start_at'],clip['end_at'])
            return document
        return self.timeline._command(key,'register_retained_clip',dict(media_id=media_id,sources=sources,selection_key=selection_key,run_ids=run_ids,actor=actor,reason=reason),action)

    def get(self,conn,media_id):
        rows=conn.execute("SELECT transform FROM media_derivation WHERE customer_id=%s AND child_id=%s AND transform->>'schema_version'=%s",(self.customer,ident(media_id),SCHEMA)).fetchall()
        if not rows:return None
        docs={fingerprint(r['transform']):r['transform'] for r in rows}
        if len(docs)!=1:raise Conflict('Retained clip has inconsistent source mappings')
        return next(iter(docs.values()))

    @staticmethod
    def validate(conn,customer,document):
        for ref in [document,*document['sources']]:
            row=conn.execute('SELECT version,verified_sha256,evidence_status FROM media_asset WHERE customer_id=%s AND id=%s',(customer,ident(ref['media_id']))).fetchone()
            if not row or (row['version'],row['verified_sha256'],row['evidence_status'])!=(ref['media_version'],ref['sha256'],'available'):
                raise Conflict('Retained clip provenance is unavailable or superseded')

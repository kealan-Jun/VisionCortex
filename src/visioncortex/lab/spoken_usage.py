"""Oral material/sample use and remaining-stock observations for any equipment.

Candidates, explicit actual-use declarations and remaining observations stay
separate. Existing Core confirmations/ledger own posting and corrections. An
explicit existing-operation link combines speech/photo evidence without debit.
"""
from datetime import timedelta
from decimal import Decimal
from uuid import uuid5

from .domain import Conflict, Missing, amount, clean, fingerprint, ident, instant
from .operator_scan import OperatorScans
from .processing import insert_observation
from .speech_candidates import RULE_VERSION, parse_usage_statements
from .timeline import Timeline
from .timeline_clock import source_clock_supported
from .timeline_policy import TimelinePolicies
from .weighing import Weighing

PREFIX='spoken-usage:'


class SpokenUsage:
    def __init__(self,core,settings):
        self.core,self.settings=core,settings

    def _source(self,conn,activity_id):
        a=conn.execute("SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s AND source_type='speech'",
                       (self.core.customer,ident(activity_id))).fetchone()
        if not a:
            raise Missing('Original speech activity outside this customer')
        if not a['model_revision'] or a['payload'].get('contains_context'):
            raise Conflict('Source-aligned speech and its actual ASR revision are required')
        rows=conn.execute('''SELECT e.*,m.kind,m.start_at,m.duration_seconds,m.time_basis,a.version,
            a.verified_sha256,a.evidence_status,i.payload AS receipt
            FROM evidence_ref e JOIN hj_timeline_media m USING(customer_id,media_id)
            JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id
            JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
            JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
              AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
            WHERE e.customer_id=%s AND e.id=ANY(%s)''',(self.core.customer,a['evidence_ids'])).fetchall()
        if len(rows)!=1:
            raise Conflict('One source-aligned speech channel required')
        source=rows[0]
        if (source['evidence_status']!='available' or source['version']!=source['media_version']
                or source['receipt'].get('sha256') not in {None,source['verified_sha256']}
                or source['kind'] not in {'audio','video'}):
            raise Conflict('Original speech source hash/version is unavailable')
        if source['receipt'].get('source_kind')!='fixture':
            native_key=a['payload'].get('native_import_key')
            native=conn.execute("SELECT result FROM command_receipt WHERE customer_id=%s AND key=%s AND starts_with(key,'native-comment-import:')",
                (self.core.customer,native_key)).fetchone() if native_key else None
            task=conn.execute("SELECT result,model_revision,status FROM hj_timeline_task WHERE customer_id=%s AND id=%s AND kind='transcription'",
                (self.core.customer,ident(a['payload']['task_id']))).fetchone() if a['payload'].get('task_id') else None
            if not ((native and native['result'].get('activity_id')==str(a['id'])
                     and native['result']['source']['audio_sha256']==source['verified_sha256'])
                    or (task and task['status']=='completed' and task['model_revision']==a['model_revision']
                        and str(a['id']) in task['result'].get('activity_ids',[]))):
                raise Conflict('Automatic oral usage requires its completed original ASR execution/import receipt')
        if source['kind']=='video' and not source['receipt'].get('native_record',{}).get('embedded_audio'):
            raise Conflict('Video speech requires registered embedded audio')
        camera=conn.execute("SELECT id,code FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s",
                            (self.core.customer,source['receipt'].get('camera_id'))).fetchone()
        if not camera:
            raise Conflict('Original capture camera required; no person identity is used')
        return a,source,camera

    def _context(self,conn,camera,at,field):
        reasons=[]
        if at is None:
            return {},['source_clock_unknown_no_current_label_fallback']
        context=OperatorScans(self.core)._context(conn,at,None,camera['id'])
        values=context['materials']
        if not values or not context['batch_id']:
            return {},['material_and_target_batch_required']
        newest=max(v['occurred_at'] for v in values)
        selected=[v for v in values if v['occurred_at']==newest]
        if len({v['resolution']['target_id'] for v in selected})!=1:
            return {},['ambiguous_material_labels']
        selected=sorted(selected,key=lambda v:v['event_id'])[-1]
        resolved=selected['resolution'];material=resolved['material_unit']
        batch=self.core._entity(conn,context['batch_id'],'process_batch')
        supplied=field['value']
        type_id=material['metadata'].get('material_type_id')
        type_code=self.core._entity(conn,type_id,'material_type')['code'] if type_id else material['code']
        if supplied.get('material_code') and supplied['material_code'] not in {type_code,material['code']}:
            reasons.append('spoken_material_conflicts_with_QR_batch')
        if supplied.get('batch_code') and supplied['batch_code']!=batch['code']:
            reasons.append('spoken_target_conflicts_with_registered_target')
        return clean(dict(camera_id=camera['id'],camera_code=camera['code'],instrument_id=context['equipment_id'],
            instrument_code=self.core._entity(conn,context['equipment_id'],'instrument')['code'] if context['equipment_id'] else '未关联',
            material_id=resolved['target_id'],material_code=material['code'],source_lot_id=resolved.get('source_lot_id'),
            source_lot=resolved.get('source_lot'),batch_id=context['batch_id'],batch_code=batch['code'],
            run_id=context['run_id'],material_scan_event_id=selected['event_id'],label_snapshot=resolved)),reasons

    def post(self,key,candidate_id,*,expected_revision,expected_material_revision,actor,reason,existing_operation_id=None):
        request=dict(candidate_id=str(candidate_id),expected_revision=expected_revision,
            expected_material_revision=expected_material_revision,actor=actor,reason=reason,
            existing_operation_id=existing_operation_id)
        def action(conn):
            row=conn.execute('SELECT association,value FROM field_candidate WHERE customer_id=%s AND id=%s',
                             (self.core.customer,ident(candidate_id))).fetchone()
            if not row or not row['association'].get('usage_scope'):
                raise Conflict('A source-pinned oral use candidate with resolved QR context is required')
            scope=row['association']['usage_scope']
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                ('material-use:'+str(self.core.customer)+':'+scope['camera_id']+':'+scope['material_scan_event_id'],))
            if not existing_operation_id and scope.get('instrument_id') and Weighing(self.core).active(conn,scope['camera_id'],scope['instrument_id']):
                raise Conflict('Existing weighing requires an explicit existing use occurrence link')
            operation_id=existing_operation_id or str(uuid5(self.core.customer,'spoken-use:'+row['value']['source_segment_id']))
            result=self.core.confirm_usage(key+':confirm',candidate_id,expected_revision=expected_revision,
                material_id=scope['material_id'],batch_id=scope['batch_id'],expected_material_revision=expected_material_revision,
                actor=actor,operation_id=operation_id,reason=reason,reuse_existing=existing_operation_id is not None)
            Timeline(self.core)._dirty(conn,instant(scope['occurred_at']))
            return result
        return self.core.command(key,request,action)

    def consume(self,activity_id,config):
        with self.core.db.transaction(readonly=True) as conn:
            activity,source,camera=self._source(conn,activity_id)
        payload=activity['payload'];units=payload.get('aligned_units') or []
        if payload.get('asr_alignment',{}).get('selected_unit')=='words':
            units=[dict(text=activity['text_content'],start_seconds=str(source['start_seconds']),end_seconds=str(source['end_seconds']))]
        if not units:
            units=[dict(text=activity['text_content'],start_seconds=str(source['start_seconds']),end_seconds=str(source['end_seconds']))]
        if len(units)>512:
            raise Conflict('Speech material intake exceeds bounded alignment units')
        key=PREFIX+fingerprint(dict(activity_id=str(activity_id),parser=RULE_VERSION,policy=config))
        def action(conn):
            results=[]
            current=conn.execute('SELECT version,verified_sha256,evidence_status FROM media_asset WHERE customer_id=%s AND id=%s',
                                 (self.core.customer,source['media_id'])).fetchone()
            if current['version']!=source['version'] or current['verified_sha256']!=source['verified_sha256'] or current['evidence_status']!='available':
                raise Conflict('Speech source changed before material intake')
            for unit_index,unit in enumerate(units):
                lo,hi=Decimal(unit['start_seconds']),Decimal(unit['end_seconds'])
                if not source['start_seconds']<=lo<hi<=source['end_seconds']:
                    raise Conflict('Speech material statement escaped original aligned evidence')
                at=(source['start_at']+timedelta(microseconds=int(lo*1000000)) if source['start_at']
                    and source_clock_supported(source['time_basis'],source['receipt'].get('source_kind')) else None)
                fields=parse_usage_statements(unit['text'],segment_id=str(activity_id)+':'+str(unit_index))
                prepared=[]
                for field in fields:
                    scope,reasons=self._context(conn,camera,at,field)
                    anchor=fingerprint(dict(source_sha256=source['verified_sha256'],media_version=source['version'],
                        start_seconds=str(lo),end_seconds=str(hi),statement_index=field['statement_index']))
                    field['value']['source_segment_id']=anchor
                    reasons+=field['conflicts']
                    if not field['conflicts']:
                        try:
                            q,_=amount(field['value']['quantity'],field['unit'])
                            if field['semantics']=='actual_increment' and q==0:
                                reasons.append('zero_quantity_is_record_only')
                        except ValueError:
                            reasons.append('invalid_quantity_or_precision')
                    if not field['actual_use_asserted'] and field['semantics']=='actual_increment':
                        reasons.append('actual_use_not_explicitly_asserted')
                    if field['retrospective_reference']:
                        reasons.append('retrospective_use_requires_existing_occurrence_link')
                    if scope:
                        scope.update(source_audio_sha256=source['verified_sha256'],source_start_seconds=str(lo),
                            source_end_seconds=str(hi),source_statement_index=field['statement_index'],
                            source_activity_id=str(activity_id),occurred_at=clean(at))
                    field['association']=dict(usage_scope=scope)
                    for name in ('instrument_id','material_id','batch_id'):
                        if scope.get(name):field['association'][name]=scope[name]
                    field['conflicts']=sorted(set(reasons))
                    prepared.append((field,scope,anchor))
                # A reported remainder checks the same statement's arithmetic;
                # it never creates another debit or silently resets inventory.
                arithmetic_conflicts=set()
                for field,scope,anchor in prepared:
                    if not scope:continue
                    if field['conflicts']:continue
                    state=conn.execute('SELECT * FROM material_state WHERE customer_id=%s AND material_id=%s',
                                       (self.core.customer,ident(scope['material_id']))).fetchone()
                    if not state or state['unit']!=amount(field['value']['quantity'],field['unit'])[1]:
                        field['conflicts'].append('quantity_dimension_conflicts_with_material_stock')
                        continue
                    actual=[amount(f['value']['quantity'],f['unit'])[0] for f,c,_ in prepared
                            if c.get('material_id')==scope['material_id'] and f['semantics']=='actual_increment' and not f['conflicts']]
                    remaining=[amount(f['value']['quantity'],f['unit'])[0] for f,c,_ in prepared
                               if c.get('material_id')==scope['material_id'] and f['semantics']=='balance' and not f['conflicts']]
                    if state['balance'] is not None and remaining and any(r!=state['balance']-sum(actual,Decimal(0)) for r in remaining):
                        arithmetic_conflicts.add(scope['material_id'])
                for field,scope,anchor in prepared:
                    if scope.get('material_id') in arithmetic_conflicts:
                        field['conflicts'].append('spoken_remaining_conflicts_with_inventory_arithmetic')
                for field,scope,anchor in prepared:
                    oid=insert_observation(self.core,conn,event_id='spoken-use:'+anchor,
                        revision=activity['model_revision']+':'+RULE_VERSION,kind='asr_segment',
                        source_kind=source['receipt']['source_kind'],occurred_at=at,evidence_ids=activity['evidence_ids'],
                        payload=dict(analysis_outcome='observations',text=unit['text'],normalized_fields=[field],
                                     source_activity_id=str(activity_id),original_audio_sha256=source['verified_sha256'],
                                     source_start_seconds=str(lo),source_end_seconds=str(hi),human_verified=False))
                    candidate=conn.execute('SELECT * FROM field_candidate WHERE customer_id=%s AND observation_id=%s',
                                           (self.core.customer,oid)).fetchone()
                    result=dict(activity_id=str(activity_id),candidate_id=str(candidate['id']),occurrence_id=anchor,
                        source_text=unit['text'],semantics=field['semantics'],quantity=field['value']['quantity'],
                        unit=field['unit'],context=scope,reasons=field['conflicts'],status='recorded_candidate',
                        quantity_ledger_written=False,source_path=source['receipt']['relative_path'])
                    if (config['post_actual_declarations'] and field['semantics']=='actual_increment'
                            and not field['conflicts'] and scope):
                        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                            ('material-use:'+str(self.core.customer)+':'+scope['camera_id']+':'+scope['material_scan_event_id'],))
                        existing=conn.execute("SELECT 1 FROM operation WHERE customer_id=%s AND metadata->>'candidate_id'=%s LIMIT 1",
                                              (self.core.customer,str(candidate['id']))).fetchone()
                        overlap=conn.execute("""SELECT metadata->'usage_scope' AS scope FROM operation
                            WHERE customer_id=%s AND metadata#>>'{usage_scope,source_audio_sha256}'=%s
                            AND metadata#>>'{usage_scope,material_id}'=%s
                            AND (metadata#>>'{usage_scope,source_start_seconds}')::numeric<%s
                            AND (metadata#>>'{usage_scope,source_end_seconds}')::numeric>%s LIMIT 100""",
                            (self.core.customer,source['verified_sha256'],scope['material_id'],hi,lo)).fetchall()
                        ambiguous_overlap=any(not (
                            Decimal(o['scope']['source_start_seconds'])==lo and Decimal(o['scope']['source_end_seconds'])==hi
                            and o['scope'].get('source_statement_index')!=field['statement_index']) for o in overlap)
                        active=Weighing(self.core).active(conn,camera['id'],scope['instrument_id']) if scope['instrument_id'] else None
                        if active:
                            result['reasons']=['existing_weighing_requires_explicit_occurrence_link']
                        elif existing:
                            result['status']='already_posted'
                        elif ambiguous_overlap:
                            result['reasons']=['overlapping_audio_requires_existing_occurrence_link']
                        else:
                            state=conn.execute('SELECT revision FROM material_state WHERE customer_id=%s AND material_id=%s',
                                               (self.core.customer,ident(scope['material_id']))).fetchone()
                            try:
                                with conn.transaction():
                                    posted=self.post(key+':post:'+anchor,candidate['id'],expected_revision=candidate['revision'],
                                        expected_material_revision=state['revision'],actor='automatic explicit oral-use policy',
                                        reason='source-pinned actual-use declaration; '+config['revision'])
                            except (Conflict,ValueError) as exc:
                                result['reasons']=['stock_post_pending:'+str(exc)[:200]]
                            else:
                                result.update(status='declared_use_posted',operation=posted,quantity_ledger_written=True)
                    if scope:
                        result['inventory']=clean(conn.execute('SELECT balance,used,unit,revision FROM material_state WHERE customer_id=%s AND material_id=%s',
                            (self.core.customer,ident(scope['material_id']))).fetchone())
                    result['occurred_at']=clean(at)
                    results.append(result)
            return clean(dict(schema_version='vc-lab-spoken-usage/1',activity_id=str(activity_id),camera_id=camera['id'],statements=results,
                              new_model_calls=0,person_identity_used=False))
        return self.core.command(key,dict(activity_id=str(activity_id),parser=RULE_VERSION,policy=config),action)

    def tick(self):
        config=TimelinePolicies.load(self.settings.timeline_policy_path).document.get('spoken_material_usage',{})
        if not config.get('enabled'):
            return dict(enabled=False)
        with self.core.db.transaction(readonly=True) as conn:
            rows=conn.execute("""SELECT a.id FROM hj_activity a WHERE a.customer_id=%s AND a.source_type='speech'
                AND NOT EXISTS(SELECT 1 FROM command_receipt r WHERE r.customer_id=a.customer_id AND starts_with(r.key,%s)
                  AND r.result->>'activity_id'=a.id::text)
                ORDER BY a.start_at DESC NULLS LAST,a.created_at DESC LIMIT %s""",
                (self.core.customer,PREFIX,config['max_activities_per_tick'])).fetchall()
        results=[]
        for row in rows:
            try:results.append(self.consume(row['id'],config))
            except (Conflict,Missing,ValueError) as exc:results.append(dict(activity_id=str(row['id']),status='pending',reason=str(exc),quantity_ledger_written=False))
        return dict(enabled=True,results=results,new_model_calls=0)

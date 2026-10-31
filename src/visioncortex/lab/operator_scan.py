"""Explicit operator declarations, distinct from background QR pixel observations.

Append-only events share Core's transaction, outbox and timeline lock. Context is
an event-time replay of one exact person/camera scope, never wearer or speaker
inference, experiment completion or a material quantity operation.
"""
from __future__ import annotations

from datetime import timedelta
from itertools import groupby
import json
from uuid import uuid4

from psycopg.types.json import Jsonb

from .context_views import mark_context_dirty
from .domain import Conflict, clean, fingerprint, ident, instant
from .timeline import Timeline
from .rules import Rules

SCHEMA = 'vc-lab-operator-scan/1'
CAMERA_INTENT = 'camera_equipment_presence'
CAMERA_MATERIAL_INTENT = 'camera_material_presence'


def _text(value,name,limit=512):
    if type(value) is not str or not value or len(value.encode())>limit:
        raise ValueError('Explicit bounded '+name+' required')


def declaration_activity(conn,core,*,text,at,evidence_ids,payload,run_id=None,
                         person_id=None,instrument_id=None,sample_id=None,kind='settings'):
    """One existing activity/outbox path; supports explicit scanner events without media."""
    timeline=Timeline(core)
    aid=uuid4()
    inputs=clean(dict(sample_id=sample_id,run_ids=[run_id] if run_id else []))
    stored=clean(dict(payload,_association_input=inputs,clock_uncertainty_us=0,
                     temporal_world_clock_status='verified' if at else 'uncalibrated'))
    if len(json.dumps(stored).encode())>1024**2:raise Conflict('Declaration activity exceeds its byte budget')
    conn.execute("INSERT INTO hj_activity(customer_id,id,kind,source_type,text_content,start_at,end_at,sample_id,instrument_id,person_id,tag_id,label_resolution,evidence_ids,payload,model_revision,association_status,priority) VALUES(%s,%s,%s,'reviewed_note',%s,%s,%s,%s,%s,%s,NULL,'{}',%s,%s,NULL,%s,'high')",
      (core.customer,aid,kind,text,at,at,ident(sample_id) if sample_id else None,
       ident(instrument_id) if instrument_id else None,ident(person_id) if person_id else None,
       evidence_ids,Jsonb(stored),'explicit' if run_id else 'unassigned'))
    if run_id:
        conn.execute("INSERT INTO hj_activity_run(customer_id,activity_id,run_id,basis) VALUES(%s,%s,%s,'explicit_operator_declaration')",(core.customer,aid,ident(run_id)))
        timeline._touch(conn,run_id,at,at)
    timeline._dirty(conn,at)
    core._emit(conn,'experiment_activity',aid,1,dict(activity_id=str(aid),occurred_at=clean(at),payload=stored,run_ids=[run_id] if run_id else []))
    return str(aid)


class OperatorScans:
    def __init__(self,core,*,max_events=2048):
        if type(max_events) is not int or not 1<=max_events<=10000:
            raise ValueError('Bounded operator event budget required')
        self.core,self.customer=core,core.customer
        self.timeline=Timeline(core)
        self.max_events=max_events

    def _scope(self,conn,person_id,camera_id):
        if person_id is None and camera_id is None:
            raise ValueError('Explicit person or camera scope required')
        for value,kind in ((person_id,'person'),(camera_id,'capture_device')):
            if value is not None:self.core._entity(conn,value,kind)

    def _resolve(self,conn,value,kind,at):
        result=dict(status='unknown_time' if at is None else 'unregistered_label',
                    tag_id=None,target_id=None,assignment=None)
        tag=conn.execute("SELECT * FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s",(self.customer,value)).fetchone()
        if tag is None or at is None:return result
        result['tag_id']=str(tag['id'])
        label=self.timeline._resolve_label(conn,tag['id'],at)
        result.update(status=label['status'],assignment=clean(label))
        if label.get('target_id') is None:return result
        target=self.core._entity(conn,label['target_id'])
        assignment=conn.execute('SELECT * FROM tag_assignment WHERE customer_id=%s AND id=%s',(self.customer,ident(label['assignment_id']))).fetchone()
        if assignment:
            result['assignment_revision']=dict(table='tag_assignment',id=str(assignment['id']),version=assignment['version'],valid_from=clean(assignment['valid_from']),valid_to=clean(assignment['valid_to']))
        else:
            result['assignment_revision']=dict(table='hj_label_event',id=str(label['assignment_id']),occurred_at=clean(label['valid_from']))
        if kind=='equipment':
            if target['kind']!='instrument':return dict(result,status='wrong_identity_kind')
            device=conn.execute('SELECT * FROM equipment_identity WHERE customer_id=%s AND device_id=%s',(self.customer,target['id'])).fetchone()
            if (device is None or device['verified_at'] is None or at<device['verified_at']
                    or device['retired_at'] is not None and at>=device['retired_at']):
                return dict(result,status='equipment_not_active_at_event')
            return dict(result,status='declared_equipment',target_id=str(target['id']),equipment=clean(target),
                        device_code=device['device_code'],active_from=clean(device['verified_at']),retired_at=clean(device['retired_at']))
        if target['kind']=='container':
            bindings=conn.execute('SELECT * FROM container_content_binding WHERE customer_id=%s AND container_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s)',(self.customer,target['id'],at,at)).fetchall()
            if len(bindings)!=1:return dict(result,status='unknown_container_contents')
            result['container_id']=str(target['id']);result['content_binding']=clean(bindings[0])
            target=self.core._entity(conn,bindings[0]['material_id'],'material_unit')
        if target['kind']!='material_unit':return dict(result,status='wrong_identity_kind')
        source=target['metadata'].get('source_lot_id')
        lot=self.core._entity(conn,source,'source_lot') if source else None
        return dict(result,status='declared_material',target_id=str(target['id']),material_unit=clean(target),
                    source_lot_id=str(lot['id']) if lot else None,source_lot=clean(lot),
                    provenance_status='explicit_registered_source_lot' if lot else 'unknown_source_lot')

    def _camera_source(self,conn,job_id,camera_id,qr_value,at,evidence,scan_kind='equipment'):
        """Machine authority requires a completed, independently decoded source.

        This also guards direct command/API callers: supplying the new intent
        string alone cannot replace a persisted source job or its source clock.
        """
        from .scan_intake import verified_clock

        if job_id is None or camera_id is None or at is None:
            raise Conflict('Camera equipment presence requires saved QR source and verified event clock')
        from .capture_equipment_intake import source_proof
        generic=source_proof(conn,self.core,job_id,camera_id,qr_value,at,evidence,scan_kind=scan_kind)
        if generic is not None:return generic
        if scan_kind != 'equipment':
            raise Conflict('Camera material declarations require a pinned generic material QR policy')
        job=conn.execute('''SELECT j.status,j.result,j.rule_identity,j.rule_fingerprint,
            a.verified_sha256,a.version,c.time_basis,c.occurred_at,i.payload AS receipt,
            d.code AS camera_code,o.evidence_ids
            FROM processing_job j JOIN capture c ON c.customer_id=j.customer_id AND c.id=j.capture_id
            JOIN media_asset a ON a.customer_id=c.customer_id AND a.capture_id=c.id
            JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
              AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
            JOIN observation o ON o.customer_id=j.customer_id AND o.job_id=j.id AND o.kind='qr'
            JOIN entity d ON d.customer_id=j.customer_id AND d.id=%s AND d.kind='capture_device'
            WHERE j.customer_id=%s AND j.id=%s AND j.worker_kind='qr' AND c.kind='photo' ''',
            (ident(camera_id),self.customer,ident(job_id))).fetchall()
        if len(job)!=1:raise Conflict('One persisted camera QR evidence job required')
        j=job[0];receipt=j['receipt'];source=receipt.get('scan_source',{});declared=receipt.get('scan_intent',{})
        result=j['result'] or {};rules=result.get('processing_identity',{}).get('processing_rules')
        if (j['status']!='completed' or rules!=j['rule_identity'] or not rules or rules.get('sha256')!=j['rule_fingerprint']
                or source.get('schema_version')!='vc-lab-scan-source/2' or source.get('mode')!='camera_equipment'
                or source.get('intent_authority')!='configured_camera_equipment_presence_not_manual_scan'
                or source.get('manual_session_required') is not False
                or not isinstance(source.get('config_sha256'),str) or len(source['config_sha256'])!=64
                or source.get('png_sha256')!=j['verified_sha256'] or receipt.get('sha256')!=j['verified_sha256']
                or declared.get('intent')!=CAMERA_INTENT or declared.get('scan_kind')!='equipment'
                or any(declared.get(k) is not None for k in ('person_id','run_id','batch_id'))
                or receipt.get('camera_id')!=j['camera_code'] or not verified_clock(j['time_basis'])
                or j['occurred_at']!=at or set(map(str,j['evidence_ids']))!=set(map(str,evidence))
                or not evidence or sorted(set(result.get('decoded_values',[])))!=[qr_value]):
            raise Conflict('Camera equipment source, independent QR result, evidence or clock does not match')
        return dict(qr_job_id=str(ident(job_id)),source_sha256=j['verified_sha256'],media_version=j['version'],
                    original_time_basis=clean(j['time_basis']),result_sha256=fingerprint(result),processing_rules=rules)

    def _camera_resolution(self,conn,value,at):
        result=self._resolve(conn,value,'equipment',at)
        if at is None or result['status']=='declared_equipment':return result
        try:
            uid=ident(value)
        except (ValueError,TypeError,AttributeError):
            return result
        device=conn.execute('SELECT * FROM equipment_identity WHERE customer_id=%s AND device_id=%s',
                            (self.customer,uid)).fetchone()
        if device and (device['verified_at'] is None or at<device['verified_at']
                       or device['retired_at'] is not None and at>=device['retired_at']):
            # A reservation has no active instrument/tag assignment yet. Report
            # the known lifecycle gap without manufacturing either relationship.
            return dict(result,status='equipment_not_active_at_event',reserved_device_id=str(uid),
                        device_code=device['device_code'],active_from=clean(device['verified_at']),retired_at=clean(device['retired_at']))
        return result

    def record(self,key,*,intent,scan_kind,qr_value,occurred_at,evidence_ids,actor,reason,
               person_id=None,camera_id=None,run_id=None,batch_id=None,source_job_id=None):
        camera_presence=intent in {CAMERA_INTENT,CAMERA_MATERIAL_INTENT}
        equipment_presence=intent==CAMERA_INTENT
        if intent not in {'operator_scan',CAMERA_INTENT,CAMERA_MATERIAL_INTENT} or scan_kind not in {'equipment','material'}:
            raise ValueError('Only explicit operator_scan equipment/material declarations are accepted')
        if camera_presence and (scan_kind!=('equipment' if equipment_presence else 'material') or person_id is not None or run_id is not None or batch_id is not None):
            raise ValueError('Natural camera presence declares only its QR kind in camera scope; no inferred person/run/batch')
        if not camera_presence and source_job_id is not None:
            raise ValueError('Source job authority applies only to camera equipment presence')
        _text(qr_value,'QR text',4096);_text(actor,'operator declaration actor');_text(reason,'declaration reason')
        if not isinstance(evidence_ids,list) or len(evidence_ids)>64:
            raise ValueError('Explicit bounded evidence list required')
        at=instant(occurred_at) if occurred_at is not None else None
        request=dict(intent=intent,scan_kind=scan_kind,qr_value=qr_value,occurred_at=occurred_at,
                     person_id=person_id,camera_id=camera_id,run_id=run_id,batch_id=batch_id,
                     evidence_ids=evidence_ids,actor=actor,reason=reason)
        if camera_presence:request['source_job_id']=source_job_id
        def action(conn):
            self.timeline.lock(conn);self._scope(conn,person_id,camera_id)
            evidence=self.core._evidence(conn,evidence_ids)
            source=self._camera_source(conn,source_job_id,camera_id,qr_value,at,evidence,scan_kind) if camera_presence else None
            if camera_presence:
                old=conn.execute("SELECT id,payload FROM hj_operator_scan_event WHERE customer_id=%s AND COALESCE(payload->>'source_observation_id',payload->>'source_qr_job_id')=%s AND scan_kind=%s AND payload->>'qr_value'=%s",(self.customer,str(ident(source_job_id)),scan_kind,qr_value)).fetchone()
                if old:
                    activity=conn.execute("SELECT id FROM hj_activity WHERE customer_id=%s AND payload->>'operator_scan_event'=%s",(self.customer,str(old['id']))).fetchone()
                    return dict(event_id=str(old['id']),activity_id=str(activity['id']),status=old['payload']['status'],
                                snapshot=old['payload'],context=self._context(conn,at,None,camera_id))
            run=self.timeline._run(conn,run_id) if run_id else None
            if batch_id:self.core._entity(conn,batch_id,'process_batch')
            if run and run['batch_id'] and batch_id and ident(batch_id)!=run['batch_id']:
                raise Conflict('Declared run and target process batch disagree')
            effective_run=run_id
            declared_batch=batch_id or (str(run['batch_id']) if run and run['batch_id'] else None)
            target_basis=dict(kind='explicit_target' if run_id or batch_id else 'no_explicit_prior_target',
                              source_scan_event_ids=[],inherited_as_of=None)
            if scan_kind=='material' and run_id is None and batch_id is None and at is not None:
                # Strictly earlier event time: simultaneous declarations have no reliable order.
                before=at-timedelta(microseconds=1)
                prior=self._context(conn,before,person_id,camera_id)
                effective_run,declared_batch=prior['run_id'],prior['batch_id']
                if effective_run or declared_batch:
                    target_basis=dict(kind='inherited_exact_scope_event_time_target',
                        source_scan_event_ids=prior['target_declaration_event_ids'],inherited_as_of=clean(before),
                        fixed_at_recording=True,late_history_requires_explicit_revision=True)
            resolution=self._camera_resolution(conn,qr_value,at) if equipment_presence else self._resolve(conn,qr_value,scan_kind,at)
            status=resolution['status']
            if camera_presence and status!=('declared_equipment' if equipment_presence else 'declared_material'):
                raise Conflict('Natural camera presence requires a unique registered identity valid at the event')
            if scan_kind=='material' and status=='declared_material' and not (effective_run or declared_batch):
                status='needs_material_context'
            payload=clean(dict(schema_version=SCHEMA,processing_rules=Rules().identity(),intent=intent,scan_kind=scan_kind,qr_value=qr_value,
                occurred_at=at,person_id=person_id,camera_id=camera_id,run_id=effective_run,batch_id=declared_batch,
                target_selection=target_basis,
                run_explicit=run_id is not None,batch_explicit=batch_id is not None,
                resolution=resolution,status=status,evidence_ids=evidence,actor=actor,reason=reason,
                business_quantity_written=False,experiment_end_implied=False,speaker_attribution='not_inferred',
                authority='explicit_operator_declaration_not_background_pixel_decode'))
            if camera_presence:
                previous=self._context(conn,at-timedelta(microseconds=1),None,camera_id)['equipment_id']
                payload.update(source_qr_job_id=str(ident(source_job_id)),source_evidence=source,
                    authority='configured_camera_equipment_presence_not_manual_scan',
                    event_role=('equipment_presence' if previous==resolution['target_id'] else
                        'operation_start' if previous is None else 'operation_phase_switch') if equipment_presence else 'material_association',
                    event_role_basis='event_time_context_known_at_recording_replay_required_for_late_history',
                    origin='camera_qr_machine_evidence',human_review_performed=False,
                    source_type_presentation='machine_equipment_presence' if equipment_presence else 'machine_material_presence',panel_geometry_verified=False)
                if source.get('source_observation_id'):
                    payload.update(source_observation_id=str(source['source_observation_id']),
                        source_qr_job_id=str(source['qr_job_id']),source_frame_event_id=source.get('source_frame_event_id'),
                        source_authority=source['authority'])
            if len(json.dumps(payload).encode())>128*1024:raise Conflict('Operator scan snapshot exceeds its byte budget')
            event=uuid4()
            conn.execute('INSERT INTO hj_operator_scan_event(customer_id,id,event_key,person_id,camera_id,occurred_at,scan_kind,payload,payload_sha256,evidence_ids) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                         (self.customer,event,key,ident(person_id) if person_id else None,ident(camera_id) if camera_id else None,at,scan_kind,Jsonb(payload),fingerprint(payload),evidence))
            self.core._emit(conn,'operator_scan',event,1,dict(event_id=str(event),**payload))
            mark_context_dirty(conn,self.customer,at)
            if at is not None:
                from .operation_segments import mark_operation_reports_dirty
                mark_operation_reports_dirty(conn,self.customer,at)
            context=self._context(conn,at,person_id,camera_id) if at else self._unknown(person_id,camera_id)
            associated_run=context['run_id'] if scan_kind=='equipment' or status=='declared_material' else None
            summary={k:context.get(k) for k in ('status','equipment_id','run_id','batch_id','materials','pending_materials')}
            summary['switch']=context.get('switches',[])[-1] if scan_kind=='equipment' and context.get('switches') and payload.get('event_role')!='equipment_presence' else None
            activity=declaration_activity(conn,self.core,text=('相机自然看码设备观察：' if camera_presence else '实验员主动扫码声明：')+scan_kind+'；'+reason,at=at,
                evidence_ids=evidence,payload=dict(operator_scan_event=str(event),operator_scan=payload,
                operator_context=summary,why=reason,evidence_status='original_media_references' if evidence else 'explicit_scanner_declaration_without_media',
                panel_or_speaker_attribution='not_inferred',**(dict(origin='camera_qr_machine_evidence',human_review_performed=False,
                    source_type_presentation='machine_equipment_presence') if camera_presence else {})),run_id=associated_run,person_id=person_id,
                instrument_id=context.get('equipment_id') if scan_kind=='equipment' else None)
            return dict(event_id=str(event),activity_id=activity,status=status,snapshot=payload,context=context)
        return self.core.command(key,dict(schema_version=SCHEMA,**request),action)

    def _unknown(self,person_id,camera_id):
        return dict(schema_version=SCHEMA,status='unknown_time',as_of=None,person_id=person_id,camera_id=camera_id,
                    equipment_id=None,run_id=None,batch_id=None,materials=[],pending_materials=[],
                    wearer_or_speaker_inferred=False,quantity_ledger_written=False,experiment_end_implied=False)

    def context(self,*,as_of,person_id=None,camera_id=None):
        at=instant(as_of) if as_of is not None else None
        with self.core.db.transaction(readonly=True) as conn:
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            self._scope(conn,person_id,camera_id)
            return clean(self._context(conn,at,person_id,camera_id) if at else self._unknown(person_id,camera_id))

    def _context(self,conn,at,person_id,camera_id):
        rows=conn.execute('SELECT * FROM hj_operator_scan_event WHERE customer_id=%s AND person_id IS NOT DISTINCT FROM %s::uuid AND camera_id IS NOT DISTINCT FROM %s::uuid AND occurred_at<=%s ORDER BY occurred_at,id LIMIT %s',
                          (self.customer,ident(person_id) if person_id else None,ident(camera_id) if camera_id else None,at,self.max_events+1)).fetchall()
        if len(rows)>self.max_events:raise Conflict('Operator scope history exceeds bounded replay; no partial context returned')
        equipment=None;run=None;batch=None;identity_status='no_operator_equipment_declaration';switches=[];materials=[];pending=[];run_basis=[];batch_basis=[];presences=[]
        for _,group in groupby(rows,key=lambda r:r['occurred_at']):
            events=list(group)
            if any(fingerprint(e['payload'])!=e['payload_sha256'] for e in events):raise Conflict('Operator event immutable snapshot hash differs')
            declarations=[e for e in events if e['scan_kind']=='equipment']
            targets={e['payload']['resolution'].get('target_id') for e in declarations}
            if declarations:
                previous=equipment
                equipment=next(iter(targets)) if len(targets)==1 and None not in targets else None
                identity_status='declared_equipment' if equipment else 'ambiguous_or_unresolved_operator_scan'
                camera_only=all(e['payload'].get('intent')==CAMERA_INTENT for e in declarations)
                if camera_only:
                    presences.extend(dict(event_id=str(e['id']),occurred_at=clean(e['occurred_at']),
                        equipment_id=e['payload']['resolution'].get('target_id'),source_qr_job_id=e['payload'].get('source_qr_job_id')) for e in declarations)
                if not camera_only or previous!=equipment:
                    switches.append(dict(occurred_at=clean(events[0]['occurred_at']),from_equipment_id=previous,to_equipment_id=equipment,
                        reason='configured_camera_equipment_presence' if camera_only and equipment else
                            'explicit_operator_device_scan' if equipment else 'conflicting_or_unresolved_explicit_scans',
                        event_ids=[str(e['id']) for e in declarations],ends_previous_experiment=False))
            declared_runs={e['payload']['run_id'] for e in events if e['payload'].get('run_explicit')}
            declared_batches={e['payload']['batch_id'] for e in events
                              if e['payload'].get('batch_explicit') or e['payload'].get('run_explicit') and e['payload'].get('batch_id') is not None}
            if len(declared_runs)>1 or len(declared_batches)>1:
                run=batch=None;materials=[];run_basis=[];batch_basis=[]
            elif declared_runs or declared_batches:
                next_run=next(iter(declared_runs)) if declared_runs else run
                next_batch=next(iter(declared_batches)) if declared_batches else None if declared_runs else batch
                if not declared_runs and next_batch!=batch:next_run=None
                if declared_runs:
                    run_basis=[str(e['id']) for e in events if e['payload'].get('run_explicit')]
                elif next_run is None:run_basis=[]
                if declared_batches:
                    batch_basis=[str(e['id']) for e in events if e['payload'].get('batch_explicit') or
                                 e['payload'].get('run_explicit') and e['payload'].get('batch_id') is not None]
                elif declared_runs:batch_basis=[]
                if (next_run,next_batch)!=(run,batch):materials=[]
                run,batch=next_run,next_batch
            for event in events:
                p=event['payload']
                if p['scan_kind']!='material':continue
                item=dict(event_id=str(event['id']),occurred_at=clean(event['occurred_at']),run_id=p['run_id'],batch_id=p['batch_id'],
                          resolution=p['resolution'],status=p['status'],declared_only_no_usage=True)
                (materials if p['status']=='declared_material' else pending).append(item)
        selected=[m for m in materials if (m['run_id'] is not None and m['run_id']==run) or
                  (m['run_id'] is None and batch is not None and m['batch_id']==batch)]
        return clean(dict(schema_version=SCHEMA,status='ready' if equipment and (run or batch) else 'needs_material_context' if equipment else 'needs_equipment_context',
            as_of=at,person_id=person_id,camera_id=camera_id,equipment_id=equipment,identity_status=identity_status,
            run_id=run,batch_id=batch,materials=selected,pending_materials=pending,switches=switches,equipment_presences=presences,
            target_declaration_event_ids=sorted(set(run_basis+batch_basis)),
            replayed_event_ids=[str(r['id']) for r in rows],knowledge_basis='event_time_replay_of_immutable_declarations',
            scope='exact_person_camera_pair_no_cross_scope_inheritance',global_batch_materials='use_existing_process_batch_summary',
            wearer_or_speaker_inferred=False,quantity_ledger_written=False,experiment_end_implied=False,
            panel_ocr_attribution='not_supplied_by_operator_current_context'))

"""One source-scoped QR context and occurrence guard for oral/OCR material use."""
from .domain import Conflict, clean, ident
from .operator_scan import OperatorScans
from .timeline import Timeline


def resolve(core, conn, camera_id, at):
    if at is None:
        return {}, ['source_clock_unknown_no_current_label_fallback']
    camera=core._entity(conn,camera_id,'capture_device')
    context=OperatorScans(core)._context(conn,at,None,camera_id)
    values=context['materials']
    if not values or not context['batch_id']:
        return {}, ['material_and_target_batch_required']
    newest=max(v['occurred_at'] for v in values)
    selected=[v for v in values if v['occurred_at']==newest]
    if len({v['resolution']['target_id'] for v in selected})!=1:
        return {}, ['ambiguous_material_labels']
    selected=sorted(selected,key=lambda v:v['event_id'])[-1]
    resolved=selected['resolution'];material=resolved['material_unit']
    instrument=core._entity(conn,context['equipment_id'],'instrument') if context['equipment_id'] else None
    batch=core._entity(conn,context['batch_id'],'process_batch')
    run=Timeline(core)._run(conn,context['run_id']) if context['run_id'] else None
    samples=[core._entity(conn,sid,'material_unit') for sid in run['sample_ids']] if run else []
    return clean(dict(camera_id=camera_id,camera_code=camera['code'],instrument_id=context['equipment_id'],
        instrument_code=instrument['code'] if instrument else '未关联',instrument_snapshot=instrument,
        material_id=resolved['target_id'],material_code=material['code'],source_lot_id=resolved.get('source_lot_id'),
        source_lot=resolved.get('source_lot'),batch_id=context['batch_id'],batch_code=batch['code'],
        target_batch_snapshot=batch,target_samples=samples,run_id=context['run_id'],
        material_scan_event_id=selected['event_id'],target_scan_event_ids=context['target_declaration_event_ids'],
        label_snapshot=resolved)), []


def lock(core, conn, context):
    conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
        ('material-use:'+str(core.customer)+':'+context['camera_id']+':'+context['material_scan_event_id'],))


def other_channel(core, conn, context, channel, *, baseline_at=None):
    """Flag ambiguity, never deduplicate quantities or infer occurrence equality."""
    if channel not in {'speech','ocr'}:
        raise ValueError('Known material evidence channel required')
    path='context_snapshot' if channel=='speech' else 'usage_scope'
    # path is selected only from two fixed internal JSON keys.
    query="""SELECT id,occurred_at,metadata FROM operation WHERE customer_id=%s
        AND status='confirmed' AND coalesce(metadata->>'effective_kind',kind)='consume'
        AND metadata#>>'{"""+path+""",camera_id}'=%s
        AND metadata#>>'{"""+path+""",material_scan_event_id}'=%s
        AND metadata#>>'{"""+path+""",material_id}'=%s
        AND batch_id=%s"""
    args=[core.customer,context['camera_id'],context['material_scan_event_id'],context['material_id'],ident(context['batch_id'])]
    if baseline_at is not None:
        query+=' AND occurred_at>=%s';args.append(baseline_at)
    rows=conn.execute(query+' LIMIT 2',args).fetchall()
    return rows


def reading_base(context, *, at, kind, quantity, unit, operation_id, debited):
    return dict(时间=at,记录类型=kind,数量=quantity,单位=unit,设备=context.get('instrument_code','未关联'),
        材料=context.get('material_code','未关联'),来源批次=(context.get('source_lot') or {}).get('code','未登记'),
        目标批次=context.get('batch_code','未关联'),账本操作=operation_id,已扣库存=debited,人工核实=False)


def accounting(conn, customer, operation_id, material_id):
    if not operation_id:return None
    row=conn.execute('SELECT * FROM operation WHERE customer_id=%s AND id=%s',
                     (customer,ident(operation_id))).fetchone()
    for _ in range(32):
        if not row:raise Conflict('Referenced material operation is missing')
        if row['status']=='confirmed':break
        children=conn.execute('SELECT * FROM operation WHERE customer_id=%s AND supersedes_id=%s LIMIT 2',
                              (customer,row['id'])).fetchall()
        if len(children)!=1:raise Conflict('Material correction chain is unresolved')
        row=children[0]
    else:raise Conflict('Material correction chain exceeds bounded projection')
    lines=conn.execute("SELECT quantity,unit FROM operation_line WHERE customer_id=%s AND operation_id=%s AND material_id=%s AND direction='input'",
                       (customer,row['id'],ident(material_id))).fetchall()
    state=conn.execute('SELECT balance,unit,revision FROM material_state WHERE customer_id=%s AND material_id=%s',
                       (customer,ident(material_id))).fetchone()
    cancelled=row['metadata'].get('consumption_cancelled') is True
    if not cancelled and len(lines)!=1:raise Conflict('Material use has no unique accounting quantity')
    return clean(dict(operation_id=row['id'],operation_revision=row['revision'],
        quantity='0' if cancelled else lines[0]['quantity'],unit=state['unit'] if cancelled else lines[0]['unit'],
        current_stock=state['balance'],stock_revision=state['revision'],corrected=str(row['id'])!=str(operation_id),cancelled=cancelled))

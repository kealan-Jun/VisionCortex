"""Bounded reading views of existing weighing command and ledger receipts."""
from .domain import Conflict, clean, instant
from .weighing import OPEN, EVENT


def readable(value):
    context, reading = value['context'], value['reading']
    material = context['label_snapshot']
    ledger = value.get('ledger_result', {}).get('balances', {}).get(context['material_id'], {})
    return dict(时间=reading['occurred_at'], 设备=context['instrument_snapshot']['code'],
        设备名称=context['instrument_snapshot']['metadata'].get('name'),
        材料=material['material_unit']['code'], 来源批次=(material.get('source_lot') or {}).get('code','未登记'),
        目标批次=context['target_batch_snapshot']['code'],
        目标样品=[s['code'] for s in context.get('target_samples', [])],
        原始读数=reading['displayed_value'], 基准读数=value['baseline']['displayed_value'],
        累计用量=value['quantity'], 单位=value['unit'], 本条新增扣量=value['delta_used'],
        本笔账后库存=ledger.get('balance'), 库存状态='未知' if ledger.get('balance') is None else '已知',
        状态=value['status'], 同一次称量=value['session_id'], 原图=reading['source_path'],
        读数活动=reading['activity_id'], 账本操作=value['operation_id'], 人工核实=False)


def projection(conn, customer, start, end):
    heads = conn.execute("""SELECT h.result,h.created_at FROM command_receipt h WHERE customer_id=%s
        AND starts_with(h.key,%s) AND (h.result#>>'{baseline,occurred_at}')::timestamptz<%s
        AND ((h.result#>>'{baseline,occurred_at}')::timestamptz>=%s OR EXISTS (
          SELECT 1 FROM command_receipt e WHERE e.customer_id=h.customer_id
          AND starts_with(e.key,%s||(h.result->>'session_id')||':')
          AND e.result->>'schema_version'='vc-lab-weighing/1'
          AND (e.result#>>'{reading,occurred_at}')::timestamptz>=%s
          AND (e.result#>>'{reading,occurred_at}')::timestamptz<%s))
        ORDER BY h.created_at DESC LIMIT 101""", (customer,OPEN,end,start,EVENT,start,end)).fetchall()
    if len(heads) > 100:
        raise Conflict('Weighing report exceeds 100-episode budget')
    values = []
    for row in heads:
        head = row['result']
        event = conn.execute("SELECT result FROM command_receipt WHERE customer_id=%s AND starts_with(key,%s) AND result->>'schema_version'='vc-lab-weighing/1' AND (result#>>'{reading,occurred_at}')::timestamptz<%s ORDER BY (result->>'sequence')::integer DESC LIMIT 1",
                             (customer, EVENT+head['session_id']+':', end)).fetchone()
        value = event['result'] if event else head
        if instant(value['reading']['occurred_at']) < start:
            continue
        values.append(dict(value, 阅读摘要=readable(value)))
    pending = conn.execute("""WITH latest AS (SELECT DISTINCT ON (a.id) a.id AS activity_id,a.start_at,r.result
        FROM command_receipt r JOIN hj_activity a ON a.customer_id=r.customer_id
          AND a.id::text=r.result->>'activity_id'
        WHERE r.customer_id=%s AND starts_with(r.key,'weighing-intake:')
          AND (a.start_at IS NULL OR (a.start_at>=%s AND a.start_at<%s))
        ORDER BY a.id,r.created_at DESC)
        SELECT * FROM latest WHERE result->>'status'='pending' LIMIT 101""", (customer,start,end)).fetchall()
    return clean(dict(episodes=values, pending=pending[:100], pending_truncated=len(pending)>100,
                      ledger_authority='Core.operate / Core.correct_operation', human_verified=False))

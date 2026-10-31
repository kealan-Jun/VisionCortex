"""Shallow reading projection of source-pinned oral use; no new ledger writer."""
from .domain import clean
from .spoken_usage import PREFIX

NAMES={'actual_increment':'实际取用声明','balance':'口述剩余量','planned':'计划用量',
       'cumulative':'累计用量声明','correction':'更正声明','unknown':'含数量的口述'}


def readable(statement):
    context=statement['context'];stock=statement.get('inventory') or {}
    return dict(时间=statement['occurred_at'],记录类型=NAMES[statement['semantics']],
        原话=statement['source_text'],数量=statement['quantity'],单位=statement['unit'],
        设备=context.get('instrument_code','未关联'),材料=context.get('material_code','未关联'),
        来源批次=(context.get('source_lot') or {}).get('code','未登记'),
        目标批次=context.get('batch_code','未关联'),账后库存=stock.get('balance'),
        已扣库存=statement['quantity_ledger_written'],
        状态='已按实际取用声明记账' if statement['quantity_ledger_written'] else '已记录；未扣库存',
        待核对原因=statement['reasons'],原始录音=statement['source_path'],
        录音内开始秒=context.get('source_start_seconds'),录音内结束秒=context.get('source_end_seconds'),
        账本操作=(statement.get('operation') or {}).get('operation_id'),人工核实=False)


def projection(conn,customer,start,end):
    rows=conn.execute("""SELECT r.result FROM command_receipt r JOIN hj_activity a
        ON a.customer_id=r.customer_id AND a.id::text=r.result->>'activity_id'
        WHERE r.customer_id=%s AND starts_with(r.key,%s)
          AND r.result->>'schema_version'='vc-lab-spoken-usage/1'
          AND (a.start_at IS NULL OR (a.start_at<%s AND a.end_at>=%s))
        ORDER BY r.created_at DESC LIMIT 101""",(customer,PREFIX,end,start)).fetchall()
    entries=[]
    for row in rows[:100]:
        item=dict(row['result'])
        item['阅读摘要']=[readable(s) for s in item['statements']]
        entries.append(item)
    return clean(dict(entries=entries,truncated=len(rows)>100,ledger_authority='Core.confirm_usage',human_verified=False))

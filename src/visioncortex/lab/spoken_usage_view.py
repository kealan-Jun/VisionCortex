"""Shallow reading projection of source-pinned oral use; no new ledger writer."""
from .domain import clean, ident
from .material_use_context import reading_base, accounting
from .spoken_usage import PREFIX

NAMES={'actual_increment':'实际取用声明','balance':'口述剩余量','planned':'计划用量',
       'cumulative':'累计用量声明','correction':'更正声明','unknown':'含数量的口述'}


def readable(statement):
    context=statement['context'];stock=statement.get('inventory') or {}
    ledger=statement.get('accounting')
    operation_id=ledger['operation_id'] if ledger else (statement.get('operation') or {}).get('operation_id')
    result=reading_base(context,at=statement['occurred_at'],kind=NAMES[statement['semantics']],
        quantity=statement['quantity'],unit=statement['unit'],operation_id=operation_id,debited=bool(operation_id))
    result.update(原话=statement['source_text'],账后库存=stock.get('balance'),库存单位=stock.get('unit'),
        状态='已关联用料账；同一操作不重复扣量' if ledger else
            '已按实际取用声明记账' if statement['quantity_ledger_written'] else '已记录；未扣库存',
        待核对原因=[] if ledger else statement['reasons'],原始录音=statement['source_path'],
        录音内开始秒=context.get('source_start_seconds'),录音内结束秒=context.get('source_end_seconds'))
    if ledger:
        result.update(账本用量=ledger['quantity'],账本单位=ledger['unit'],库存单位=ledger['unit'],当前库存=ledger['current_stock'],账本已更正=ledger['corrected'])
    return result


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
        item['statements']=[dict(s) for s in item['statements']]
        for statement in item['statements']:
            fact=conn.execute("SELECT value FROM fact_revision WHERE customer_id=%s AND candidate_id=%s AND scope='material_consumption' AND status='confirmed' ORDER BY revision DESC LIMIT 1",
                (customer,ident(statement['candidate_id']))).fetchone()
            if fact:
                statement['accounting']=accounting(conn,customer,fact['value']['operation_id'],fact['value']['material_id'])
        item['阅读摘要']=[readable(s) for s in item['statements']]
        entries.append(item)
    return clean(dict(entries=entries,truncated=len(rows)>100,ledger_authority='Core.confirm_usage',human_verified=False))

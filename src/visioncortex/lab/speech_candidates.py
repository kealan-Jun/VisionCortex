"""Conservative, versioned candidates. No text/time based business deduplication."""
from __future__ import annotations

import re

RULE_VERSION = "zh-lab-usage/2"


def parse_usage(text, *, final=True, segment_id=None):
    if not final:
        return []
    intent = "correction" if re.search(r"更正|改成|改为|不是.*是|说错", text) else "planned" if re.search(r"准备|计划|打算|将要|预计|还需|需要|要取", text) else "cumulative" if re.search(r"累计|总共|合计|一共", text) else "balance" if re.search(r"剩余|还剩|余额", text) else "actual_increment" if re.search(r"使用|用了|用掉|加入|加了|投入|投了|消耗|取了|取出|取用|称取|拿了", text) else "unknown"
    matches = re.findall(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)|[零〇一二两三四五六七八九十百千万点]+)\s*(毫克|千克|克|毫升|升|件|个|mg|kg|g|mL|ml|L|pcs)(?![A-Za-z])", text)
    batch = re.findall(r"(?<![A-Za-z0-9])PB[-_]?\d+(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
    materials = re.findall(r"(?<![A-Za-z0-9])[A-Z]\d{3,}(?![A-Za-z0-9])", text)
    conflicts = []
    if len(matches) != 1:
        conflicts.append("multiple_or_missing_quantities")
    if len(set(batch)) > 1 or len(set(materials)) > 1:
        conflicts.append("multiple_materials_or_batches")
    if re.search(r"各|两次|三次|每次", text):
        conflicts.append("multiple_operations_require_explicit_split")
    if re.search(r'没有|没取|没用|未取|未用|不要|取消|不是取|不是用', text):
        conflicts.append('negated_or_cancelled_statement')
    if re.search(r'大约|大概|可能|估计|差不多|好像|约\s*[\d零一二两三四五六七八九十]', text):
        conflicts.append('approximate_or_uncertain_quantity')
    if re.search(r'[?？]|吗[，。！!\s]*$',text):
        conflicts.append('question_is_not_usage_declaration')
    if not matches:
        return []
    raw, raw_unit = matches[-1]
    try:
        value = chinese_number(raw) if re.fullmatch('[零〇一二两三四五六七八九十百千万点]+',raw) else raw
    except ValueError:
        value = raw
        conflicts.append('ambiguous_spoken_number')
    if value.startswith("-"):
        conflicts.append("negative_usage_requires_correction")
    units = {"毫克": "mg", "千克": "kg", "克": "g", "毫升": "mL", "升": "L", "ml": "mL", "件":"pcs", "个":"pcs"}
    unit = units.get(raw_unit, raw_unit)
    return [dict(field_key="material_usage", value=dict(quantity=value, intent=intent, batch_code=batch[0].upper().replace("_", "-") if len(set(batch)) == 1 else None, material_code=materials[0] if len(set(materials)) == 1 else None, original_text=text, source_segment_id=segment_id, parser_version=RULE_VERSION), unit=unit, semantics=intent, valid=True, conflicts=conflicts, association={})]


def chinese_number(text):
    digits={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    if text.count('点')>1:
        raise ValueError('Ambiguous Chinese decimal')
    integer, separator, fraction=text.partition('点')
    if separator and (not fraction or any(c not in digits for c in fraction)):
        raise ValueError('Ambiguous Chinese fractional digits')
    if not integer:
        integer='零'
    if all(c in digits for c in integer):
        value=''.join(str(digits[c]) for c in integer)
    else:
        total=section=number=0
        previous_scale=10000
        last_scale=None
        zero=False
        pending_digit=False
        for c in integer:
            if c in digits:
                if pending_digit and not zero:
                    raise ValueError('Ambiguous adjacent Chinese digits')
                number=digits[c];zero=number==0;pending_digit=not zero
            elif c=='万':
                if total or not section+number:
                    raise ValueError('Ambiguous Chinese scale')
                total=(section+number)*10000;section=number=0
                previous_scale=10000;last_scale=10000;pending_digit=False;zero=False
            elif c in {'十','百','千'}:
                scale={'十':10,'百':100,'千':1000}[c]
                if scale>=previous_scale or (not pending_digit and not (c=='十' and section==0)):
                    raise ValueError('Ambiguous Chinese scale ordering')
                section+=(number or 1)*scale;number=0
                previous_scale=scale;last_scale=scale;pending_digit=False;zero=False
            else:raise ValueError('Unsupported Chinese quantity')
        if pending_digit and last_scale and last_scale>10 and not any(c in integer for c in ('零','〇')):
            raise ValueError('Colloquial omitted Chinese place value is ambiguous')
        value=str(total+section+number)
    return value+('.'+''.join(str(digits[c]) for c in fraction) if separator else '')


def parse_usage_statements(text, *, segment_id=None):
    """Keep actual use, plan, cumulative and remaining clauses separate.

    Character spans point into the unchanged ASR text. These are candidates,
    not independent truth, equipment attribution or stock-write authorization.
    """
    if not isinstance(text,str) or len(text)>20000:
        raise ValueError('Bounded source transcript required')
    statements=[]
    all_batch=set(re.findall(r'(?<![A-Za-z0-9])PB[-_]?\d+(?![A-Za-z0-9])',text,flags=re.I))
    all_material=set(re.findall(r'(?<![A-Za-z0-9])[A-Z]\d{3,}(?![A-Za-z0-9])',text))
    for match in re.finditer(r'[^，,。；;！!\n]+[？?]?',text):
        clause=match.group()
        fields=parse_usage(clause,segment_id=segment_id)
        for field in fields:
            value=field['value']
            if value['batch_code'] is None and len(all_batch)==1:
                value['batch_code']=next(iter(all_batch)).upper().replace('_','-')
            if value['material_code'] is None and len(all_material)==1:
                value['material_code']=next(iter(all_material))
            field.update(source_character_span=[match.start(),match.end()],statement_index=len(statements),
                         actual_use_asserted=field['semantics']=='actual_increment' and bool(re.search(
                             r'取了|取出|用了|用掉了|加入了|加了|投入了|投了|消耗了|拿了|已经.*(?:取|用|加|投入|消耗)',clause)),
                         retrospective_reference=bool(re.search(r'刚才|之前|先前|取过|用过|再说|重复说',clause)),
                         human_verified=False)
            prefix=text[:match.start()]
            if re.search(r'如果|假如|假设|要是|例如|比如',prefix):
                field['conflicts'].append('conditional_or_example_not_actual_use')
            if re.search(r'更正|改成|改为|说错|准备|计划|打算|将要|预计|没有|取消',prefix):
                field['conflicts'].append('earlier_clause_changes_usage_meaning')
            statements.append(field)
    return statements

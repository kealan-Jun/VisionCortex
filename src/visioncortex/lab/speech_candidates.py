"""Conservative, versioned candidates. No text/time based business deduplication."""
from __future__ import annotations

import re

RULE_VERSION = "zh-lab-usage/1"
SMALL_NUMBERS = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}


def parse_usage(text, *, final=True, segment_id=None):
    if not final:
        return []
    intent = "correction" if re.search(r"更正|改成|改为|不是.*是", text) else "planned" if re.search(r"准备|计划|打算|将要|预计", text) else "cumulative" if re.search(r"累计|总共|合计|一共", text) else "balance" if re.search(r"剩余|还剩|余额", text) else "actual_increment" if re.search(r"使用|用了|用掉|加入|加了|投入|投了|消耗", text) else "unknown"
    matches = re.findall(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+)|[零一二两三四五六七八九十])\s*(毫克|千克|克|毫升|升|mg|kg|g|mL|ml|L)(?![A-Za-z])", text)
    batch = re.findall(r"(?<![A-Za-z0-9])PB[-_]?\d+(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
    materials = re.findall(r"(?<![A-Za-z0-9])[A-Z]\d{3,}(?![A-Za-z0-9])", text)
    conflicts = []
    if len(matches) != 1:
        conflicts.append("multiple_or_missing_quantities")
    if len(set(batch)) > 1 or len(set(materials)) > 1:
        conflicts.append("multiple_materials_or_batches")
    if re.search(r"各|两次|三次|每次", text):
        conflicts.append("multiple_operations_require_explicit_split")
    if not matches:
        return []
    raw, raw_unit = matches[-1]
    value = SMALL_NUMBERS.get(raw, raw)
    if value.startswith("-"):
        conflicts.append("negative_usage_requires_correction")
    units = {"毫克": "mg", "千克": "kg", "克": "g", "毫升": "mL", "升": "L", "ml": "mL"}
    unit = units.get(raw_unit, raw_unit)
    return [dict(field_key="material_usage", value=dict(quantity=value, intent=intent, batch_code=batch[0].upper().replace("_", "-") if len(set(batch)) == 1 else None, material_code=materials[0] if len(set(materials)) == 1 else None, original_text=text, source_segment_id=segment_id, parser_version=RULE_VERSION), unit=unit, semantics=intent, valid=True, conflicts=conflicts, association={})]

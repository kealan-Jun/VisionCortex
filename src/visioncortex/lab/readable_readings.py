"""Chinese readable views of existing OCR/FR candidates; no new numeric rules."""
from __future__ import annotations

REGISTERED = {'verified_source_registration','automatic_registered_layout','temporal_registered_video'}
REASONS = {
    'unmapped_raw_token_not_a_confirmed_numeric_field':'原始文字尚未对应到字段，可能是编号、标签、时间或其他文字',
    'decimal_uncertain':'原识别链标记小数点不确定，保留原文等待核对',
    'decimal_scale_conflict':'原识别链的两次结果存在小数位差异',
    'sign_conflict':'原识别链的正负号结果存在冲突',
    'display_precision_mismatch':'原识别结果与已有显示精度规则不一致',
    'integer_or_missing_decimal_unresolved':'上游无法区分整数与漏点，保留其待核对状态',
    'visible_decimal_missing_in_ocr':'上游图像有小数点候选，但识别文本没有',
    'verified_display_precision_mismatch':'上游识别位数与已有显示精度规则不符',
    'low_ocr_score':'原识别链标记置信度不足',
    'outside_verified_range':'原识别链标记超出已核验范围',
    'unit_mismatch_or_unverified':'原识别链标记单位不一致或未核验',
}


def _raw(reading, field):
    tokens = field.get('raw_tokens') or []
    if tokens:return ' | '.join(str(t.get('text','')) for t in tokens)
    for value in (field.get('raw_text'),reading.get('text'),reading.get('raw_text'),reading.get('display_value')):
        if value is not None:return str(value)
    return ''


def project_readings(readings, *, field_mapping=None):
    """Keep every English row/value unchanged; explain only upstream provenance."""
    if len(readings)>4224:raise ValueError('Readable OCR candidate count exceeds bound')
    chinese=[]
    for index,row in enumerate(readings):
        field=row.get('raw_field') or {}
        registered=bool(field) and field_mapping in REGISTERED
        issues=list(dict.fromkeys((row.get('quality_issues') or [])+(field.get('issues') or [])))
        normalization=field.get('normalization') or row.get('normalization') or {}
        rule=field.get('display_rule') or row.get('display_rule') or {}
        if normalization.get('method') in {'registered_fixed_decimal_v1','registered_fixed_decimal_v2'}:
            decimal_basis='沿用原FR已确认的固定小数位显示规则；保留原字形与规范化记录'
        elif any('decimal' in i or 'precision' in i for i in issues):
            decimal_basis='沿用原识别链的小数点/精度待核对状态，展示层不补点或改变候选'
        else:
            decimal_basis='展示层不新增小数规则；是否适用原FR规则见原始规则/规范化记录'
        reasons=[REASONS.get(i,'原识别链待核对标记：'+i) for i in issues]
        if row.get('instrument_id') is None:reasons.append('设备归属未知，识别原文和候选仍保留')
        if not registered:reasons.append('未提供注册字段映射，不把编号或时间指定为秤值')
        if not reasons:reasons.append('机器识别候选尚未通过内容核验')
        chinese.append(dict(序号=index+1,
            字段=(field.get('field_key') or row.get('measurement_name')) if registered else '未映射文字（不猜读数字段）',
            识别原文=_raw(row,field),候选值=row.get('value'),
            原候选值=field.get('raw_value',row.get('raw_value',field.get('tentative_value'))),
            单位=row.get('unit'),设备归属=row.get('instrument_id') or '未绑定或未核验',
            小数点依据=decimal_basis,原显示规则=rule or None,原规范化记录=normalization or None,
            待核对原因=reasons,状态='识别候选；非业务确认'))
    return list(readings),dict(说明='保留原识别数据和已有FR规则，不因未绑定设备而丢弃识别结果。空值不是零。',
        候选条数=len(chinese),已确认业务读数=0,识别结果=chinese)

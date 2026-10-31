"""Presentation of existing Comment evidence, without recognition or inference."""
from copy import deepcopy
from datetime import datetime
import json
import math
import re
from zoneinfo import ZoneInfo


def _clock(value):
    if type(value) is not int:
        return None
    return datetime.fromtimestamp(value / 1_000_000, ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S.%f')[:-3]


def transcript_text(document):
    """Render the saved sentences; never infer a speaker, silence or an action."""
    audio = document.get('audio') or {}
    sentences = document.get('sentences') or []
    lines = [f"{document.get('date', '日期未确认')}｜采集设备：{document.get('device', '未确认')}",
             '以下为自动转写原文，未经人工核对。']
    basis = audio.get('time_basis')
    unknown = (document.get('capture_time_status') == 'unknown_or_uncalibrated'
               or basis in ('unknown', 'uncalibrated', 'unknown_or_uncalibrated')
               or isinstance(basis, dict) and basis.get('clock_sync_valid') is False
               or document.get('outcome') == 'transcribed_candidate_source_clock_unknown')
    if sentences:
        lines += ['时间依据：原录音记录时间，未独立校准；无法对时的句子显示录音内偏移。', '']
        for sentence in sentences:
            start, end = _clock(sentence.get('start_us')), _clock(sentence.get('end_us'))
            if start and end and not unknown:
                stamp = start + '—' + end
            else:
                left = sentence.get('source_start_seconds', sentence.get('audio_start_seconds'))
                right = sentence.get('source_end_seconds', sentence.get('audio_end_seconds'))
                stamp = f'录音内 {left}—{right} 秒（实际时刻未确认）' if left is not None and right is not None else '时间未确认'
            lines.append(f'{stamp}  {sentence.get("text", "")}')
    else:
        status, outcome = document.get('status'), document.get('outcome')
        if outcome == 'no_audio' or audio.get('status') in ('no_input', 'missing'):
            message = '未取得音频，暂无转写。不能据此判断实验员没有说话。'
        elif status in ('failed', 'needs_attention') or outcome in ('failed', 'error'):
            message = '转写失败，暂无可读文字。原音和失败记录保留。'
        elif status in ('pending', 'queued', 'running', 'waiting_for_capture'):
            message = '转写尚未完成，暂无文字。'
        else:
            message = '转写结果为空，不能据此判断录音没有语音。'
        lines += ['', message]
    return ('\n'.join(lines) + '\n').encode('utf-8')


_NUMBER = re.compile(
    r'(?:(?:PV|SV)\s*[:：]?\s*)?([+\-−]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))'
    r'\s*(mg|kg|g|℃|°C|℉|°F|rpm|r/min|MPa|kPa|Pa|%RH|%|V|A)?')
_DECIMAL_BLOCKERS = {'decimal_uncertain', 'decimal_scale_conflict',
    'visible_decimal_missing_in_ocr', 'display_precision_mismatch',
    'verified_display_precision_mismatch', 'integer_or_missing_decimal_unresolved'}


def _number(row):
    """Read one saved numeric token, without joining digits or placing a dot.

    Signs, units and field roles do not gate this reading view. The signed
    machine value, raw evidence and their original decisions remain untouched.
    """
    field = row.get('raw_field') or {}
    if field.get('data_type') in {'text', 'clock'}:
        return None
    issues = set(row.get('quality_issues') or []) | set(field.get('issues') or [])
    if issues & _DECIMAL_BLOCKERS:
        return None
    tokens = field.get('raw_tokens') or ([row['raw_token']] if row.get('raw_token') else [])
    if not tokens:
        value = row.get('value')
        if value is None:
            return None
        tokens = [dict(text=str(value), score=1)]
    numbers = []
    for token in tokens:
        match = _NUMBER.fullmatch(str(token.get('text', '')).strip())
        if not match:
            continue
        if not field and row.get('value') is None and '.' not in match[1] and not match[2]:
            return None  # An unmapped whole-photo integer may be a clock or ID.
        try:
            score = float(token.get('score', 0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score) or not .8 <= score <= 1:
            return None
        numbers.append(match[1].lstrip('+-−'))
    if len(numbers) != 1:
        return None  # Never concatenate two tokens into an invented reading.
    normalization = field.get('normalization') or {}
    if (normalization.get('method') in {'registered_fixed_decimal_v1', 'registered_fixed_decimal_v2'}
            and field.get('value') is not None):
        normalized = _NUMBER.fullmatch(str(field['value']))
        if normalized:
            return normalized[1].lstrip('+-−')
    return numbers[0]


def _same_region(left, right):
    a = (left.get('raw_field') or {}).get('source_pixel_box')
    b = (right.get('raw_field') or {}).get('source_pixel_box')
    if not a or not b:
        return False
    area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area > 0 and overlap / area > .65


def readings_document(document):
    """Show time, recorded numbers and source; preserve all machine members."""
    if not isinstance(document, dict):
        return document
    if document.get('schema') not in ('field-numeric-observations/1', 'field-unbound-readings/1'):
        return document
    rows = document.get('readings') or []
    if len(rows) > 4224:
        raise ValueError('Readable OCR record count exceeds bound')
    sources = list((document.get('source_refs') or {}).values())
    times = list(dict.fromkeys(str(s['source_reported_at']) for s in sources if s.get('source_reported_at')))
    if not times:
        times=list(dict.fromkeys(str(r.get('captured_at') or r.get('observed_at'))
                                for r in rows if r.get('captured_at') or r.get('observed_at')))
    if document.get('capture_time_status') == 'verified' and document.get('event_at'):
        at = str(document['event_at'])
    elif times:
        at = '；'.join(times) + '（采集端时间，尚未校准）'
    else:
        at = '时间未确认'
    paths = list(dict.fromkeys(s['source_relative_path'] for s in sources if s.get('source_relative_path')))
    mapped_tokens = {str(t.get('text', '')).strip() for row in rows
                     for t in (row.get('raw_field') or {}).get('raw_tokens', [])}
    numbers = [None if row.get('raw_token') and str(row['raw_token'].get('text', '')).strip() in mapped_tokens
               else _number(row) for row in rows]
    # Conflicting crop/whole-frame predictions remain unreadable. A sign or
    # unit difference alone need not suppress an identical digit sequence.
    for index, row in enumerate(rows):
        if 'ocr_view_conflict' not in (row.get('raw_field') or {}).get('issues', []):
            continue
        peers = [numbers[i] for i, other in enumerate(rows) if i != index and _same_region(row, other)]
        if not peers or any(number != numbers[index] for number in peers):
            numbers[index] = None
    # This is a per-source reading summary, not a field/operation count.
    headline = dict(时间=at, 数值=list(dict.fromkeys(number for number in numbers if number is not None)),
                    原图或视频=paths or ['Evidence.json'])
    if not headline['数值']:
        headline['识别结果'] = '未读清'
    value = deepcopy(document)
    value['中文摘要'] = headline
    return {'中文摘要': value.pop('中文摘要'), **value}


def readings_text(document):
    """Keep the concise headline before the existing detailed candidate rows."""
    return (json.dumps(readings_document(document), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def field_context_document(document):
    """Readable historical QR/FR context, keeping canonical source facts intact."""
    if document.get('schema_version')!='visioncortex-field-archive-context/1':
        return document
    events=document.get('events') or []
    if len(events)>10000:raise ValueError('Historical Comment context exceeds display bound')
    labels={'instrument_binding':'历史关联记录','scene_binding':'场景关联记录','ocr_observation':'屏幕识别记录'}
    rows=[]
    for event in events:
        payload=event.get('payload') or {};media=event.get('media') or {}
        rows.append(dict(时间=_clock(event.get('start_us')) or '未确认',
            记录类型=labels.get(event.get('type'),event.get('type')),
            原设备标识=event.get('instrument_id'),原记录状态=payload.get('status'),
            原记录读数=payload.get('normalized_values') or payload.get('reading_observations') or [],
            原关联依据=payload.get('identity_basis'),原有效起止=[payload.get('valid_from'),payload.get('valid_to')],
            原图=(media.get('photo') or {}).get('path'),原结果文件=media.get('result_file'),
            记录编号=event.get('event_id')))
    summary=dict(说明='历史二维码关联与屏幕记录；沿用当时的设备和读数，不据此重新绑定、确认操作或扣库存。',
                 时间依据='显示原记录时间；校准状态与来源见原 clock 和 evidence_refs。',
                 原记录状态=document.get('status'),记录条数=len(rows),
                 缺失或冲突项=len(document.get('errors') or []),
                 记录=rows if rows else ['此日期没有已留存的FR关联记录；不代表没有发生操作。'])
    return {'中文摘要':summary,**{k:v for k,v in document.items() if k!='中文摘要'}}

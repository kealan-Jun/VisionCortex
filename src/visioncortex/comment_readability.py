"""Presentation of existing Comment evidence, without recognition or inference."""
from copy import deepcopy
from datetime import datetime
import json
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
             '说话人：未确认｜以下为自动转写原文，未经人工核对。']
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


def readings_document(document):
    """Only improve an existing Chinese summary; all other fields stay equal."""
    if not isinstance(document, dict):
        return document
    if document.get('schema') not in ('field-numeric-observations/1', 'field-unbound-readings/1'):
        return document
    summary = document.get('中文摘要')
    historical = not isinstance(summary, dict) or str(summary.get('说明','')).startswith(('历史FR识别结果','历史识别结果'))
    if not isinstance(summary, dict):
        # Historical FR rows already have their own identity/quality decisions.
        # Reflect them verbatim instead of applying a new numeric/confirmation rule.
        rows=[]
        for index, row in enumerate(document.get('readings') or []):
            rows.append(dict(序号=index+1,字段=row.get('measurement_name') or '未映射文字',
                识别原文=row.get('raw_text',row.get('text',row.get('display_value',''))),
                候选值=row.get('value'),单位=row.get('unit'),设备归属=row.get('instrument_id') or '未绑定或未核验',
                原记录时间=row.get('captured_at') or row.get('observed_at'),
                原记录状态=row.get('result_stage') or row.get('status'),
                原确认标记=row.get('human_verified',document.get('human_verified')),
                原质量标记=row.get('quality_issues') or [],状态='沿用原记录状态；本次未重新核验'))
        summary=dict(说明='历史识别结果；字段、设备、批次与确认状态沿用原记录，展示层不重新绑定或确认。',
                     候选条数=len(rows),已确认业务读数=None,识别结果=rows)
    rows = summary.get('识别结果') or []
    sources = list((document.get('source_refs') or {}).values())
    times = list(dict.fromkeys(str(s['source_reported_at']) for s in sources if s.get('source_reported_at')))
    if not times:
        times=list(dict.fromkeys(str(r['原记录时间']) for r in rows if r.get('原记录时间')))
    if document.get('capture_time_status') == 'verified' and document.get('event_at'):
        at = str(document['event_at']) + '（有源时钟依据，详见证据）'
    elif times:
        at = '；'.join(times) + '（来源标记时间，尚未校准）'
    else:
        at = '未确认；目录日期不代表实际发生时刻'
    devices = list(dict.fromkeys(str(r['设备归属']) for r in rows if r.get('设备归属') not in (None, '未绑定或未核验')))
    raw = list(dict.fromkeys(r['识别原文'] for r in rows if r.get('识别原文')))
    reasons = []
    if any(str(r.get('字段', '')).startswith('未映射') for r in rows):
        reasons.append('尚未对应到具体屏幕字段')
    if rows and any(r.get('单位') is None for r in rows):
        reasons.append('单位尚未确定')
    if not devices:
        reasons.append('设备归属未确认')
    reasons.append('沿用历史状态，本次未重新核验' if historical else '识别内容尚未核对，具体问题见下方逐条结果')
    paths = list(dict.fromkeys(s['source_relative_path'] for s in sources if s.get('source_relative_path')))
    confirmed = summary.get('已确认业务读数', 0)
    candidates = [dict(字段=r.get('字段'),候选读数=r.get('候选值'),单位=r.get('单位'),状态=r.get('状态'))
                  for r in rows if r.get('候选值') is not None and not str(r.get('字段','')).startswith('未映射')]
    overview = (f'历史字段读数 {len(candidates)} 条；原状态保留，本次未重新核验' if historical
                else f'已确认业务读数 {confirmed} 条；字段候选 {len(candidates)} 条') if candidates else '尚未形成可靠读数'
    headline = dict(时间=at, 设备归属='、'.join(devices) if devices else '未关联或未核验',
                    读数概况=overview,
                    字段读数=candidates,
                    识别原文=raw, 待核对原因=reasons or ['暂无可核实的字段读数'],
                    原图或视频=paths or ['来源位置见 Evidence.json'])
    value = deepcopy(document)
    value['中文摘要'] = {**headline, **{k: v for k, v in summary.items() if k not in headline}}
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
    labels={'instrument_binding':'设备或人员关联记录','scene_binding':'场景关联记录','ocr_observation':'屏幕识别记录'}
    rows=[]
    for event in events:
        payload=event.get('payload') or {};media=event.get('media') or {}
        rows.append(dict(时间=_clock(event.get('start_us')) or '未确认',
            记录类型=labels.get(event.get('type'),event.get('type')),
            原设备标识=event.get('instrument_id'),佩戴人标识=event.get('wearer_id'),
            原操作人标识=event.get('operator_id'),原记录状态=payload.get('status'),
            原记录读数=payload.get('normalized_values') or payload.get('reading_observations') or [],
            原关联依据=payload.get('identity_basis'),原有效起止=[payload.get('valid_from'),payload.get('valid_to')],
            原图=(media.get('photo') or {}).get('path'),原结果文件=media.get('result_file'),
            记录编号=event.get('event_id')))
    summary=dict(说明='历史二维码关联与屏幕记录；沿用当时的设备、人员和读数，不据此重新绑定、确认操作或扣库存。',
                 时间依据='显示原记录时间；校准状态与来源见原 clock 和 evidence_refs。',
                 原记录状态=document.get('status'),记录条数=len(rows),
                 缺失或冲突项=len(document.get('errors') or []),
                 记录=rows if rows else ['此日期没有已留存的FR关联记录；不代表没有发生操作。'])
    return {'中文摘要':summary,**{k:v for k,v in document.items() if k!='中文摘要'}}

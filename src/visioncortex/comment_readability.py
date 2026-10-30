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
    if not isinstance(document, dict) or not isinstance(document.get('中文摘要'), dict):
        return document
    if document.get('schema') not in ('field-numeric-observations/1', 'field-unbound-readings/1'):
        return document
    summary = document['中文摘要']
    rows = summary.get('识别结果') or []
    sources = list((document.get('source_refs') or {}).values())
    times = list(dict.fromkeys(str(s['source_reported_at']) for s in sources if s.get('source_reported_at')))
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
    reasons.append('识别内容尚未核对，具体问题见下方逐条结果')
    paths = list(dict.fromkeys(s['source_relative_path'] for s in sources if s.get('source_relative_path')))
    confirmed = summary.get('已确认业务读数', 0)
    candidates = [dict(字段=r.get('字段'),候选读数=r.get('候选值'),单位=r.get('单位'),状态=r.get('状态'))
                  for r in rows if r.get('候选值') is not None and not str(r.get('字段','')).startswith('未映射')]
    headline = dict(时间=at, 设备归属='、'.join(devices) if devices else '未关联或未核验',
                    读数概况=f'已确认业务读数 {confirmed} 条；字段候选 {len(candidates)} 条' if candidates else '尚未形成可靠读数',
                    字段读数=candidates,
                    识别原文=raw, 待核对原因=reasons or ['暂无可核实的字段读数'],
                    原图或视频=paths or ['来源位置见 Evidence.json'])
    value = deepcopy(document)
    value['中文摘要'] = {**headline, **{k: v for k, v in summary.items() if k not in headline}}
    return {'中文摘要': value.pop('中文摘要'), **value}


def readings_text(document):
    """Keep the concise headline before the existing detailed candidate rows."""
    return (json.dumps(readings_document(document), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')

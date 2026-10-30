"""Readable existing result files; a reversible view over saved machine data.

No model, discovery, filesystem join, clock calibration or business write occurs
here. Machine readers validate and remove this one derived field. Public JSON
keeps every original key/value and begins with the same reproducible summary.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KEY = '阅读摘要'
VERSION = 'visioncortex-result-reading/2'
LEGACY_VERSION = 'visioncortex-result-reading/1'


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def kind(path):
    """Only existing mutable result documents, never indexes or .Evidence."""
    parts = Path(path).parts
    if any(p in {'.Evidence', 'Timeline', 'Analysis', 'History', 'ClipUnderstanding', 'RequestPlans'} for p in parts):
        return None
    if len(parts) >= 4 and parts[-4:-2] == ('ProcessedClips', 'Clips') and parts[-1] == 'Retention.json':
        return 'retention'
    if len(parts) >= 3 and parts[-3] == 'MultimodalUnderstanding' and parts[-1].endswith('.json') and parts[-1] != 'Index.json':
        return 'understanding'
    if parts[-2:] == ('LaboratoryDailyReport', 'LaboratoryDailyReport.json'):
        return 'day'
    return None


def _time(value):
    if type(value) is not int:
        return None
    try:
        return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')
    except (ValueError, OverflowError, OSError):
        return None


def _range(value):
    return {'开始': _time(value.get('start_us', value.get('original_start_us'))),
            '结束': _time(value.get('end_us', value.get('original_end_us'))),
            '时间说明': '沿用来源记录时间；未新增时钟校准，区间不证明连续操作。'}


def _association(value):
    ref = value.get('关联时间线') or {}
    return {'状态': '有已提交关联证据引用，具体归属需查看原记录' if ref else '本文件未载入已验证关联',
            '实验设备': None, '面板与字段': None, '实验或样品': None, '材料来源批次': None,
            '目标制备批次': None, '引用': ref or None,
            '已有记录入口': 'Comment/Bindings.json（仅在该档案已有此文件时读取）',
            '说明': '相邻时间、当前主设备和文件夹名称不能代替设备、面板、样品及批次归属；未据此扣量。'}


def _retention(value):
    interval = value.get('selection_interval') or {}
    disposition = value.get('disposition') or value.get('derived_video_disposition') or {}
    category = value.get('display_category') or interval.get('activity_category')
    decision = interval.get('video_value') or disposition.get('video_value')
    if decision == 'confirmed_no_value':
        label = '原记录判为无价值，保留原片依据'
    elif decision == 'selected':
        label = '有价值实验片段' if category in {'ExperimentActivity', 'experiment', 'experiment_activity'} else '有价值片段，具体类别见依据'
    else:
        label = '价值待判断'
    intervals = value.get('intervals') or []
    grouped = value.get('folder_scope') == 'complete_original_recording_collection'
    rows = []
    for row in intervals:
        if not isinstance(row, dict):
            rows.append({'旧版区间原文': row, '留存结论': '价值待判断'})
            continue
        selected = row.get('selection_interval') or {}
        state = row.get('video_value') or selected.get('video_value')
        rows.append({**_range(row), '原分类': row.get('display_label') or row.get('video_value'),
            '留存结论': {'pending': '价值待判断', 'selected': '已选中有价值区间',
                       'confirmed_no_value': '原记录判为无价值'}.get(state, '价值待判断'),
            '原决定': row.get('decision') or row.get('video_value'),
            '依据': _reason(row.get('reason') or selected.get('reason')),
            '视频': row.get('video'), '理解记录': row.get('content'),
            '状态': row.get('video_status') or row.get('processing_status')})
    return {**_range(value), '记录范围': '原片汇总；各区间分别判断' if grouped else '原片中的一个区间',
        '留存结论': ('本原片尚无已发布判断区间' if not rows else
                    '价值待判断' if all(r['留存结论'] == '价值待判断' for r in rows) else
                    '见下方各区间的实际分类') if grouped else label,
        '为什么保留': _reason(interval.get('reason') or disposition.get('reason')),
        '可播放视频': value.get('video'),
        '播放状态': ('原记录有视频引用，完整性以现有索引校验为准' if value.get('video') else
                   f"原记录可用视频数：{value.get('available_video_count', 0)}" if grouped else '未生成选中视频；原片保持可用'),
        '原片': value.get('source_ref') or value.get('original_source'), '区间': rows,
        '关联': _association(value)}


def _reason(value):
    return {'value_not_determined': '尚未完成价值判断，保留原片等待处理。',
            'evidence_unverified': '价值依据尚未通过校验，不能确认选中。',
            'continuous_inactivity_proven': '原记录有连续无价值判断依据，详见区间证据。'}.get(value,
            value or '判断尚未完成；文件夹旧名称不作为无价值证据。')


def _understanding(value):
    original_narrative = value.get('内容记录')
    narrative = original_narrative if isinstance(original_narrative, dict) else {}
    grouped = value.get('folder_scope') == 'complete_original_recording_collection'
    rows = value.get('片段记录') or []
    windows = narrative.get('分窗口观察') or []
    return {**_range(value), **({'旧版内容原文': original_narrative} if isinstance(original_narrative, str) else {}),
        '记录范围': '原片汇总；各区间分别查看' if grouped else '一个有界视频区间',
        '理解状态': narrative.get('状态说明') or ('旧版文字保留；未据此确认结构化理解已完成' if isinstance(original_narrative, str) else
            '各区间分别处理，不能称整段已理解' if grouped else
            '模型理解尚未完成' if not value.get('understanding') else '保留已有模型观察，准确性未据此确认'),
        '实际观察': [({'开始': row.get('开始时间'), '结束': row.get('结束时间'),
                     '内容': row.get('内容'), '不确定': row.get('不确定性'), '回执': row.get('模型回执')}
                     if isinstance(row, dict) else {'旧版观察原文': row}) for row in windows],
        '实验步骤': narrative.get('实验步骤') or [], '其他有价值活动': narrative.get('非实验工作') or [],
        '当时说的话': narrative.get('录音记录') or [], '仪器与读数': narrative.get('仪器与现场记录') or [],
        '视频': value.get('video'), '视频状态': value.get('video_status') or ('分区间查看' if grouped else '未生成选中视频'),
        '区间': [({'开始': row.get('开始时间'), '结束': row.get('结束时间'), '原分类': row.get('类别'),
                  '状态': (row.get('内容记录') or {}).get('状态说明'), '详细记录': row.get('详细记录'),
                  '视频': row.get('视频')} if isinstance(row, dict) and isinstance(row.get('内容记录') or {}, dict)
                 else {'旧版区间原文': row}) for row in rows],
        '关联': _association(value), '溯源': narrative.get('溯源') or value.get('original_source'),
        '说明': '原话、读数候选、模型观察和业务事实分别保留；没有观察结果时不补写实验过程。'}


def _day_v1(value):
    day = value.get('全天记录') or {}
    rows = []
    for row in day.get('全天时间线') or []:
        rows.append({'开始': row.get('开始时间'), '结束': row.get('结束时间'), '类型': '视频记录',
            '状态': row.get('状态说明'), '实验员': row.get('实验员'),
            '实验步骤': row.get('实验步骤'), '其他有价值活动': row.get('非实验工作'),
            '原话': row.get('录音记录'), '仪器与读数': row.get('仪器与现场记录'),
            '留存判断': row.get('留存判断'), '证据': row.get('溯源')})
    for row in day.get('录音时间线') or []:
        rows.append({'开始': row.get('开始时间'), '结束': row.get('结束时间'), '类型': '自动录音转写',
                     '原话': row.get('原话'), '证据': row.get('转写来源'),
                     '说明': '原转写未经人工核对；不自动解释为开始、结束或材料用量指令。'})
    for row in day.get('现场事件时间线') or []:
        rows.append({'开始': row.get('记录时间'), '结束': None, '类型': row.get('类型'),
                     '内容': row.get('内容'), '证据': row.get('来源'), '时钟': row.get('时钟'),
                     '说明': '来源事件的时间与归属沿用原记录，未新增操作确认。'})
    rows.sort(key=lambda row: (row.get('开始') is None, row.get('开始') or '', row.get('类型') or ''))
    return {'日期': day.get('日期') or str(value.get('archive', ''))[:10], '采集设备': day.get('设备'),
        '实验员': day.get('实验员') or '未建立有效人员归属', '范围': day.get('记录范围') or [],
        '完整一天已验收': False, '采集间隙': day.get('采集间隙') or [],
        '视频留存': value.get('视频留存统计') or day.get('视频留存统计'),
        '按时间查看': rows, '独立录音状态': day.get('独立录音覆盖') or [],
        '跨模态关联': _association(value),
        '综合理解状态': (value.get('全天综合理解') or {}).get('status'),
        '说明': '只展示现有证据覆盖。15分钟发送边界不表示实验结束，未决间隔不拼成连续实验。'}


def _day(value):
    """Also read older reports whose only timeline is the original entries."""
    result = _day_v1(value)
    rows = result['按时间查看']
    if not value.get('全天记录'):
        rows = []
        for number, entry in enumerate(value.get('entries') or []):
            meaning = entry.get('understanding') or {}
            windows = meaning.get('windows') or [] if isinstance(meaning, dict) else []
            transcription = entry.get('transcription') or {}
            speech = transcription.get('comments') or [] if isinstance(transcription, dict) else []
            rows.append({'开始': _time(entry.get('start_us')), '结束': _time(entry.get('end_us')),
                '类型': '旧版视频记录', '原标签': entry.get('label'),
                '理解状态': entry.get('understanding_status') or '未提供',
                '模型原概述': [w.get('summary') for w in windows if w.get('summary')],
                '原话': [{'开始': _time(s.get('start_us')), '结束': _time(s.get('end_us')),
                          '内容': s.get('text'), '来源': s.get('transcript_path')} for s in speech],
                '证据': {'原片': entry.get('source_ref'), '片段索引': entry.get('segment_json')},
                '完整原记录': '/entries/' + str(number),
                '说明': '沿用旧版标签与模型原文，未重新核验分类、时钟和物理动作。'})
        rows.sort(key=lambda row: (row.get('开始') is None, row.get('开始') or ''))
        result['旧版记录范围'] = '从原 entries 读取；没有新结构不表示当天没有活动。'
    # A quick reading view does not replicate dense technical frame/audit trees.
    stats = result['视频留存'] or {}
    result['视频留存'] = {key: stats[key] for key in (
        '原始视频数', '选中片段数', '实际可读视频数', '实际视频总时长秒',
        '待判定分区数', '确认无价值分区数', 'whole_day_capture_proven') if key in stats}
    result['时间线记录总数'] = len(rows)
    result['按时间查看'] = rows[:256]
    result['未在摘要展开的记录数'] = max(0, len(rows) - 256)
    result['完整时间线位置'] = '/全天记录/全天时间线' if value.get('全天记录') else '/entries'
    return result


def present(value, category):
    core = canonical(value)
    if not isinstance(core, dict) or category not in {'retention', 'understanding', 'day'}:
        return core
    body = {'retention': _retention, 'understanding': _understanding, 'day': _day}[category](core)
    summary = {'展示版本': VERSION if category == 'day' else LEGACY_VERSION,
               '原数据摘要': _sha(core), '类别': category, **body}
    return {KEY: summary, **core}


def canonical(value):
    if not isinstance(value, dict) or KEY not in value:
        return value
    core = {key: item for key, item in value.items() if key != KEY}
    summary = value[KEY]
    if not isinstance(summary, dict) or summary.get('展示版本') not in {VERSION, LEGACY_VERSION} or summary.get('原数据摘要') != _sha(core):
        raise ValueError('Readable result does not match its saved machine data')
    category = summary.get('类别')
    if category not in {'retention', 'understanding', 'day'}:
        raise ValueError('Unknown readable result category')
    builder = {'retention': _retention, 'understanding': _understanding,
               'day': _day_v1 if summary['展示版本'] == LEGACY_VERSION else _day}[category]
    expected = {'展示版本': summary['展示版本'], '原数据摘要': _sha(core), '类别': category, **builder(core)}
    if summary != expected:
        raise ValueError('Readable summary differs from its source projection')
    return core

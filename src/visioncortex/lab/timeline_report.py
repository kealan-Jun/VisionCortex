"""Human-readable, evidence-first daily report; no invented narrative or SOP judgement."""
from .timeline_policy import ZONE
from .domain import instant


def time_text(value):
    return instant(value).astimezone(ZONE).strftime('%H:%M:%S') if value else '时间未知'


def render_report(report):
    lines = [f'VC Lab 实验室记录日报　{report["day"]}', '时区：Asia/Shanghai',
             f'已关联实验记录：{report["experiment_count"]}；证据观察：{report["activity_count"]}条。',
             '观察条数不代表独立物理动作次数。归档仅代表记录收尾，不能证明设备实际完成。', '']
    for experiment in report['experiments']:
        record = experiment['record']
        lines += [f'实验：{record["code"]}　类型：{record["experiment_type"]}',
                  f'样品：{", ".join(map(str, record["sample_ids"]))}；人员：{record.get("person_id") or "未明确"}',
                  f'记录状态：{record["record_status"]}；收尾依据：{record.get("closure_reason") or "尚未收尾"}',
                  '设备实际结束时间：未知；设备电脑结果文件：未采集。']
        for binding in experiment['bindings']:
            lines.append(f'  {time_text(binding["occurred_at"])} 绑定设备 {binding["instrument_id"]} 与样品 {binding["sample_id"]}；语义：{binding["semantics"]}。')
        for activity in experiment['activities']:
            lines.extend(activity_lines(activity))
        lines.append('')
    lines.append('尚未归属具体实验的观察：')
    for activity in report['unassigned_activities']:
        lines.extend(activity_lines(activity))
    if not report['unassigned_activities']:
        lines.append('  当前无已存储的未归属观察；不代表数据完整。')
    pending_time = report.get('unknown_time_temporal_activities', {})
    if pending_time.get('activities'):
        lines += ['', '时间未校准的时序读数：以下仅为客户待定时记录，不归入本日或上述实验。']
        for activity in pending_time['activities']:
            lines.extend(activity_lines(activity))
        if pending_time.get('truncated'):
            lines.append('  仅列出最近100条，尚有未列记录。')
    context = report.get('context_evidence', {})
    if context.get('dynamic_contexts'):
        lines += ['', '设备关注上下文：关注候选不代表已经确认操作，切换不结束设备实验。']
        for item in context['dynamic_contexts']:
            lines.append(f'  媒体 {item["media_id"]} / 会话 {item["source_session_key"]}；修订 {item["revision"]}；{item["range_assignment"]}。')
            previous = '未确定'
            for interval in item.get('focus_intervals', []):
                target = interval.get('instrument_id') or '未确定'
                lines.append(f'    源偏移 {interval.get("start_seconds")}—{interval.get("end_seconds") or "开放"} 秒：'
                             f'{previous} → {target}；{interval.get("focus_status")}；结束/切换依据：{interval.get("reason_ended") or "尚未观察到退出"}。')
                previous = target
    if context.get('material_observations'):
        lines += ['', '材料证据候选：来源批次与目标制备批次分别记录，以下观察未扣库存。']
        for item in context['material_observations']:
            snap = item['snapshot']
            lines.append(f'  动作 {item["occurrence_key"]} / 修订 {item["revision"]} / {item["relation"]}；{item["range_assignment"]}。')
            for resolved in snap.get('resolutions', []):
                lines.append(f'    标签 {resolved.get("qr_value")}：材料 {resolved.get("material_id") or "未知"}；'
                             f'来源批次 {resolved.get("source_lot_id") or "未知"}；容器 {resolved.get("container_id") or "未知"}；{resolved.get("association_status")}。')
            event = snap.get('event_context', {})
            lines.append(f'    用于目标批次 {event.get("target_batch_id") or "未知"}；实验 {event.get("run_id") or "未知"}；待核验项：{", ".join(snap.get("reasons", [])) or "候选仍未确认为使用事实"}。')
    lines += ['', '三连拍/读数记录：']
    for readout in report['readouts']:
        lines.append(f'  批次 {readout["source_burst_id"]}：收到 {readout["received_photos"]} 张，状态 {readout["burst_status"]}。')
        for field in readout['fields']:
            values = ', '.join(str(v.get('value')) for v in field['candidates'])
            lines.append(f'    {field["field_key"]} / {field.get("role", "未定角色")} / {field.get("channel", "")}：{values} {field.get("unit") or "单位未明确"}；{field["status"]}。')
    lines += ['', '处理状态：']
    for job in report['analysis_jobs']:
        lines.append(f'  {job["kind"]}：{job["status"]}，{job["count"]} 项。')
    lines += ['完整性：仅覆盖当前已入库证据，不能证明全天记录完整。',
              '识别质量：尚未通过客户真实数据验收。OCR为候选读数，未执行确认、飞书填表或材料记账。',
              '每条观察附原媒体ID、片段偏移和证据ID，可通过配套JSON追溯原始录音、视频或照片。']
    return '\n'.join(lines) + '\n'


def activity_lines(activity):
    source = {'video': '视频观察', 'speech': '录音转写', 'ocr': '面板识别', 'reviewed_note': '人工复核'}[activity['source_type']]
    lines = [f'  {time_text(activity["start_at"])}—{time_text(activity["end_at"])} [{source}/{activity["kind"]}] {activity["text_content"]}']
    for field in activity.get('payload', {}).get('fields', []):
        temporal = field.get('temporal_display')
        if not isinstance(temporal, dict):
            continue
        status = field.get('status')
        value = temporal.get('value_text') if status == 'confirmed' else None
        label = '视觉支持候选' if status == 'confirmed' and value is not None else '未确认'
        lines.append(f'    时序字段 {field.get("field_key", "未知字段")} / {field.get("role", "未定角色")}：'
                     f'{value if value is not None else "未知"} {field.get("unit") or "单位未明确"}；{status}（{label}，业务未确认）。')
        lo, hi = temporal.get('observation_time_start'), temporal.get('observation_time_end')
        lines.append(f'    支持范围：源媒体PTS {lo if lo is not None else "未知"}—{hi if hi is not None else "未知"}；'
                     f'方法 {temporal.get("method", "未知")}；不将窗口末端视为读数发生瞬时。')
        reasons = temporal.get('rejection_reasons') or temporal.get('unresolved_regions')
        if reasons:
            lines.append('    未通过项：' + ', '.join(map(str, reasons)))
    for evidence in activity['evidence']:
        lines.append(f'    证据 {evidence["id"]}；媒体 {evidence["media_id"]}；偏移 {evidence["start_seconds"]}—{evidence["end_seconds"]} 秒。')
    return lines

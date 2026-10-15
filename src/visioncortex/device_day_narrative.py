"""Human-readable evidence views; temporal overlap never proves causation."""
from datetime import datetime
from zoneinfo import ZoneInfo


def timestamp(value):
    return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')


def overlaps(left, right, start, end):
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and left < end and (right > start or left == right and start <= left))


def experimenter(context, camera, start, end):
    """Name the effective device binding; do not assign a camera for all time."""
    from .evidence.field_events import instant
    bindings, covering = [], []
    for event in (context or {}).get('events', []):
        payload = event.get('payload') or {}
        if (event.get('type') != 'instrument_binding' or event.get('capture_device_id') != camera
                or payload.get('role') != 'wearer' or not event.get('wearer_id')):
            continue
        left, right = instant(payload.get('valid_from')), instant(payload.get('valid_to'))
        if left is None:
            continue
        left = round(left.timestamp() * 1e6)
        right = round(right.timestamp() * 1e6) if right else None
        if (left > start if start == end else left >= end) or right is not None and right <= start:
            continue
        name = payload.get('wearer_name')
        if not name and event.get('operator_id') in (None, event['wearer_id']):
            name = payload.get('operator_label')
        row = {'人员ID': event['wearer_id'], '姓名': name, '生效时间': timestamp(left),
               '结束时间': timestamp(right) if right else None, '来源': event.get('evidence_refs'),
               '身份依据': payload.get('identity_basis'), '绑定编号': payload.get('binding_id')}
        bindings.append(row)
        if left <= start and (right is None or end <= right):
            covering.append(row)
    people = {r['人员ID'] for r in bindings}
    resolved = len(people) == 1 and bool(covering)
    selected = covering[0] if resolved else {}
    unresolved = ('绑定冲突' if len(people) > 1 else '绑定仅覆盖部分时段' if bindings else '未找到有效绑定')
    return {'姓名': selected.get('姓名') or ('姓名未提供' if resolved else '未关联实验员'),
            '人员ID': selected.get('人员ID'), '状态': '有效设备绑定' if resolved else unresolved,
            '关联方式': 'FieldRecognition设备自动绑定', '绑定记录': bindings,
            '说明': '按同设备有效时间自动关联实验员，无需再次人工确认；绑定不覆盖的时段不归给该人员。'}


def recognition_payload(payload):
    """Source workflow annotations stay in evidence, not an approval task."""
    return {key: value for key, value in (payload or {}).items()
            if key not in {'review_status', 'human_reviewed'}}


def segment_record(segment, meaning, recording, context=None, *, segments=None):
    """Readable content first, technical identities at the evidence boundary.

    Observations remain per model window. Overlapping windows are not silently
    merged into completed actions or counted as independent experiments.
    """
    from .device_day_coverage import value_assessment
    context = context or {}
    start, end = segment['start_us'], segment['end_us']
    windows = (meaning or {}).get('windows', [])
    steps, work, observations, receipts = [], [], [], []
    for window in windows:
        observations.append({'内容': window.get('summary'), '活动类别': window.get('activity_kind', '旧版观察'),
                             '可见性': window.get('observability', 'unknown'),
                             '不确定性': window.get('uncertainties', []), '模型回执': window.get('model_receipt')})
        for field, output in [('experiment_steps', steps), ('work_activities', work)]:
            for step in window.get(field, []):
                origin = start - round(segment['start_ms'] * 1000)
                left = step.get('start_us', origin + round(step['start_ms'] * 1000))
                right = step.get('end_us', origin + round(step['end_ms'] * 1000))
                output.append({'开始时间': timestamp(left), '结束时间': timestamp(right),
                    '动作': step['action'], '操作对象': step['objects'], '内容': step['description'],
                    '操作前': step['before_state'], '操作后': step['after_state'],
                    '可见依据': step['visible_evidence'], '结果状态': step['outcome'],
                    '工作类别': step.get('category'), '判断依据': step['basis'],
                    '画面引用': step['frame_ids'], '模型输入': window.get('input'),
                    '模型回执': window.get('model_receipt')})
        receipts.append({'输入': window.get('input'), '输出': window.get('model_receipt'),
                         '覆盖': window.get('coverage')})
    speech = [row for row in ((recording.get('transcription') or {}).get('comments') or [])
              if overlaps(row.get('start_us'), row.get('end_us'), start, end)]
    events = [row for row in context.get('events', [])
              if overlaps(row.get('start_us'), row.get('end_us'), start, end)
              and row.get('capture_device_id', recording.get('camera_key')) == recording.get('camera_key')]
    assessment = value_assessment(segment, segments or [segment], context | {
        'comments': [*(context.get('comments') or []), *speech], 'events': events})
    protected = assessment['protection_signals']
    # Sparse model pictures do not establish continuous inactivity. Even a
    # unanimous static result remains a candidate and cannot delete media.
    static = bool(windows) and all(w.get('activity_kind') == 'static' for w in windows)
    decision = ('retain_valuable_activity' if steps or work else
                'retain_protected_context' if protected else
                'static_candidate_requires_continuous_review' if static else 'retain_uncertain')
    return {'开始时间': timestamp(start), '结束时间': timestamp(end),
        '实验员': experimenter(context, recording.get('camera_key'), start, end),
        '内容状态': (meaning or {}).get('status', 'pending'),
        '实验步骤': steps, '非实验工作': work, '分窗口观察': observations,
        '录音记录': [{'开始时间': timestamp(s['start_us']), '结束时间': timestamp(s['end_us']),
                      '原话': s.get('text'), '转写来源': s.get('transcript_path'), '识别方式': '自动转写'} for s in speech],
        '仪器与现场记录': [{'记录时间': timestamp(e['start_us']), '类型': e['type'],
            '仪器': e.get('instrument_id'), '识别内容': recognition_payload(e.get('payload')),
            '原始照片与媒体': e.get('media'), '来源': e.get('evidence_refs'),
            '时钟': e.get('clock'), '关联依据': '同设备与时间重叠，尚未证明属于同一步操作'} for e in events],
        '留存判断': {'决定': decision, '保护原因': protected, '原视频允许删除': False,
                     '边界说明': '动作前后上下文、实验等待、读数、录音和不可见区间需保留；静止候选须连续复核'},
        '溯源': {'原视频': segment['source_ref'], '保留视频': segment.get('video'),
                 '原视频开始毫秒': segment['start_ms'], '原视频结束毫秒': segment['end_ms'],
                 '片段索引': segment.get('json_path'), '执行记录': receipts,
                 '原录音': recording.get('audio'), '片段编号': segment['segment_id']},
        '质量说明': '自动模型记录，保留可见依据与不确定性；抽样覆盖不证明完整动作召回。'}


def day_record(index):
    recordings = {r['recording_id']: r | {'camera_key': r.get('camera_key') or index['archive'][11:]}
                  for r in index.get('recordings', [])}
    meanings = {m['segment_id']: m for m in index.get('understandings', [])}
    segments = sorted(index.get('segments', []), key=lambda row: (row['start_us'], row['segment_id']))
    timeline = [segment_record(s, meanings.get(s['segment_id']), recordings[s['recording_id']],
                index.get('context'), segments=[x for x in segments if x['recording_id'] == s['recording_id']]) for s in segments]
    intervals = sorted((r['start_us'], r['end_us']) for r in recordings.values()
                       if r.get('start_us') is not None and r.get('end_us') is not None)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return {'日期': index['archive'][:10], '设备': index['archive'][11:], '时区': 'Asia/Shanghai',
        '实验员': sorted({r['实验员']['姓名'] for r in timeline if r['实验员']['人员ID']}),
        '记录范围': [{'开始时间': timestamp(a), '结束时间': timestamp(b)} for a, b in merged],
        '采集间隙': [{'开始时间': timestamp(a[1]), '结束时间': timestamp(b[0]), '说明': '没有已索引视频，原因未知'}
                     for a, b in zip(merged, merged[1:], strict=False)],
        '全天时间线': timeline,
        '录音时间线': [{'开始时间': timestamp(s['start_us']), '结束时间': timestamp(s['end_us']),
                         '实验员': experimenter(index.get('context'), r['camera_key'], s['start_us'], s['end_us']),
                         '原话': s.get('text'), '转写来源': s.get('transcript_path'),
                         '原录音': r.get('audio'), '识别方式': '自动转写'}
                        for r in recordings.values() for s in ((r.get('transcription') or {}).get('comments') or [])],
        '现场事件时间线': [{'记录时间': timestamp(e['start_us']), '类型': e['type'],
                             '设备': e.get('capture_device_id'), '内容': recognition_payload(e.get('payload')),
                             '照片与媒体': e.get('media'), '来源': e.get('evidence_refs'), '时钟': e.get('clock')}
                            for e in (index.get('context') or {}).get('events', [])],
        '现场事件接收状态': (index.get('context') or {}).get('field_event_status', 'not_received'),
        '人员关联': '设备视图；需有效人员绑定才能组成实验员全天记录',
        '覆盖说明': '仅列已索引来源；日期不表示完整24小时，片段也不表示独立实验次数。'}


def attach_context(config, index):
    """Late OCR/protocol changes update views without repeating paid video work."""
    from .application.field_event_consumer import context_events
    from .device_day import load_day_context
    from .device_day_contract import DeviceDayLayout
    from pathlib import Path
    records = [r for r in index.get('recordings', []) if r.get('start_us') is not None and r.get('end_us') is not None]
    if not records:
        return index
    layout = DeviceDayLayout(Path(config['storage']['archive_root']), index['archive'][11:], min(r['start_us'] for r in records))
    events = context_events(config, {'camera_key': index['archive'][11:],
        'recording_start_us': min(r['start_us'] for r in records), 'recording_end_us': max(r['end_us'] for r in records)})
    index['context'] = load_day_context(layout) | {'events': events['events'], 'field_event_status': events['status']}
    return index

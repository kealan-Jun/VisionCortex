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


def understanding_state(meaning, recording):
    """Keep returned content and the owning stage's unresolved state distinct."""
    stage = (recording.get('stages') or {}).get('understanding') or {}
    status = ((meaning.get('status') or 'completed') if meaning else stage.get('status') or 'pending')
    labels = {
        'completed': '已返回模型观察，具体步骤和可见依据见下文。',
        'partial': '仅部分窗口已返回结果；已展示的观察不代表全片理解完成。',
        'submitted_unknown': '请求已提交，本地尚无可确认的返回结果；当前没有可展示的模型理解。为避免重复提交，等待原请求结果确认。',
        'pending': '模型理解尚未完成，当前没有已确认的观察结果。',
        'queued': '等待模型理解处理。',
        'running': '模型理解正在处理，尚未完成。',
        'blocked': '模型理解已暂停，需先解决处理原因。',
        'failed': '模型理解处理失败，当前没有完整结果。',
        'waiting_for_plan_review': '采样计划尚未就绪，模型理解未完成。',
        'scheduled_checkpoint': '已保存处理进度，仍有窗口待完成。',
    }
    reason = stage.get('reason') or stage.get('message')
    return {'status': status, '说明': labels.get(status, '模型理解尚未确认完成，详见处理状态。'),
            '阶段状态': stage.get('status', 'pending'), '处理原因': reason}


def effective_meaning(meaning):
    """Exclude explicit placeholder rows, preserving original response evidence.

    Uncertain work and experimental waiting remain useful observations. This
    only rejects a row which calls itself a placeholder AND supplies neither a
    known state nor an observed outcome; static labels alone never reject work.
    """
    if not meaning:
        return meaning
    windows, changed = [], False
    for window in meaning.get('windows', []):
        rejected = []
        for row in window.get('work_activities', []):
            text = str(row.get('description') or '')
            explicit = (any(term in text.lower() for term in (
                            'placeholder only', 'only a placeholder', 'just a placeholder', 'placeholder entry')) or '占位' in text
                        and any(term in text for term in ('不承载动作', '无任何可观察', '仅为占位', '仅作占位')))
            unknown = (None, '', '未知', 'unknown')
            if (explicit and row.get('basis') == 'uncertain'
                    and row.get('outcome') == 'incomplete_or_uncertain'
                    and row.get('before_state') in unknown and row.get('after_state') in unknown):
                rejected.append(row)
        if not rejected:
            windows.append(window)
            continue
        changed = True
        reason = '模型将明确占位说明写入非实验工作数组，且没有可确认状态或结果；不作为实际动作或视频价值依据。'
        # device_day_steps stores the same observation twice: the full work
        # record and its SceneStep projection. Match that projection rather
        # than comparing dictionaries with different sets of fields.
        fields = ('start_ms', 'end_ms', 'description', 'frame_ids', 'comment_ids', 'basis')
        rejected_steps = [{key: row.get(key) for key in fields} for row in rejected]
        windows.append(window | {
            'work_activities': [row for row in window.get('work_activities', []) if row not in rejected],
            'steps': [row for row in window.get('steps', [])
                      if {key: row.get(key) for key in fields} not in rejected_steps],
            'invalid_structured_content': [*(window.get('invalid_structured_content') or []),
                *({'collection': 'work_activities', 'reason': reason, 'row': row,
                   'model_receipt': window.get('model_receipt'), 'input': window.get('input')} for row in rejected)],
            'uncertainties': [*(window.get('uncertainties') or []), reason]})
    return meaning | {'windows': windows} if changed else meaning


def segment_record(segment, meaning, recording, context=None, *, segments=None):
    """Readable content first, technical identities at the evidence boundary.

    Observations remain per model window. Overlapping windows are not silently
    merged into completed actions or counted as independent experiments.
    """
    from .device_day_coverage import value_assessment
    meaning = effective_meaning(meaning)
    context = context or {}
    start, end = segment['start_us'], segment['end_us']
    windows = (meaning or {}).get('windows', [])
    steps, work, observations, receipts = [], [], [], []
    for window in windows:
        observations.append({'内容': window.get('summary'), '活动类别': window.get('activity_kind', '旧版观察'),
                             '可见性': window.get('observability', 'unknown'),
                             '不确定性': window.get('uncertainties', []), '模型回执': window.get('model_receipt'),
                             '无效结构化内容': window.get('invalid_structured_content', [])})
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
    selected_context = [signal for signal in protected if signal in {
        'experiment_candidate', 'selected_action_frames', 'experiment_waiting',
        'ordinary_valuable_activity', 'experiment_context', 'between_experiment_candidates',
        'comments', 'audio_events'}]
    # Sparse model pictures do not establish continuous inactivity. Even a
    # unanimous static result remains a candidate and cannot delete media.
    static = bool(windows) and all(w.get('activity_kind') == 'static' for w in windows)
    decision = ('retain_valuable_activity' if steps or work else
                'retain_protected_context' if selected_context else
                'static_candidate_requires_continuous_review' if static else 'pending_value_review')
    state = understanding_state(meaning, recording)
    return {'开始时间': timestamp(start), '结束时间': timestamp(end),
        '实验员': experimenter(context, recording.get('camera_key'), start, end),
        '内容状态': state['status'], '状态说明': state['说明'],
        '阶段状态': state['阶段状态'], '处理原因': state['处理原因'],
        '实验步骤': steps, '非实验工作': work, '分窗口观察': observations,
        '录音记录': [{'开始时间': timestamp(s['start_us']), '结束时间': timestamp(s['end_us']),
                      '原话': s.get('text'), '转写来源': s.get('transcript_path'), '识别方式': '自动转写'} for s in speech],
        '仪器与现场记录': [{'记录时间': timestamp(e['start_us']), '类型': e['type'],
            '仪器': e.get('instrument_id'), '识别内容': recognition_payload(e.get('payload')),
            '原始照片与媒体': e.get('media'), '来源': e.get('evidence_refs'),
            '时钟': e.get('clock'), '关联依据': '同设备与时间重叠，尚未证明属于同一步操作'} for e in events],
        '留存判断': {'决定': decision, '保护原因': protected, '价值依据': selected_context, '原视频允许删除': False,
                     '留存单位': '有证据的具体原片区间；预处理候选和活动分类不决定整段留存',
                     '区间记录': segment.get('retention_intervals', []),
                     '边界说明': '保护原片不等于选中处理视频；只为有价值内容另存片段，判断未完成时保留原片与待筛选状态。'},
        '溯源': {'原视频': segment['source_ref'], '保留视频': segment.get('video'),
                 '原视频开始毫秒': segment['start_ms'], '原视频结束毫秒': segment['end_ms'],
                 '片段索引': segment.get('json_path'), '执行记录': receipts,
                 '原录音': recording.get('audio'), '片段编号': segment['segment_id']},
        '质量说明': '自动模型记录，保留可见依据与不确定性；抽样覆盖不证明完整动作召回。'}


def report_candidates(index):
    """A derived interval is a sibling view, never a second daily process."""
    return [row for row in index.get('segments', []) if row.get('record_scope') != 'retention_interval'
            and not (row.get('candidate_segment_id') and row['candidate_segment_id'] != row['segment_id'])]


def _finite_number(value):
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _merged_ranges(rows):
    merged = []
    for left, right in sorted(rows):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def _range_gaps(left, right, covered):
    gaps, cursor = [], left
    for start, end in _merged_ranges(covered):
        start, end = max(left, start), min(right, end)
        if start > cursor:
            gaps.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < right:
        gaps.append([cursor, right])
    return gaps


def _verified_interval_video(root, candidate, row):
    """Check sealed export duration and actual bytes, without ffprobing history."""
    from pathlib import Path
    from .device_day_contract import read_json, safe_child
    from .device_day_verification import verify_artifact_cached
    if root is None:
        return {'status': 'NOT_PROVEN', 'reason': 'actual_files_not_checked', 'duration_ms': None}
    root = Path(root)
    try:
        video, reference = row.get('video'), row.get('receipt')
        if not video or not verify_artifact_cached(root, video):
            raise ValueError('selected_video_missing_or_corrupt')
        if not reference or not verify_artifact_cached(root, reference):
            raise ValueError('exact_export_receipt_missing_or_corrupt')
        receipt = read_json(safe_child(root, reference['path']))
        expected = {'source_ref': candidate['source_ref'], 'candidate_segment_id': candidate['segment_id'],
            'segment_id': row['segment_id'], 'source_start_ms': row['start_ms'], 'source_end_ms': row['end_ms'],
            'start_us': row['start_us'], 'end_us': row['end_us'], 'video': video}
        if (receipt.get('decision') != 'retain_derivative' or receipt.get('video_value') != 'selected'
                or receipt.get('selection_policy') != 'visioncortex-exact-value-intervals/1'
                or any(receipt.get(key) != value for key, value in expected.items())):
            raise ValueError('exact_export_receipt_source_or_interval_mismatch')
        mapping = receipt.get('source_mapping') or {}
        mapped = {'source_start_ms': row['start_ms'], 'source_end_ms': row['end_ms'],
            'source_start_us': row['start_us'], 'source_end_us': row['end_us'], 'derived_start_ms': 0}
        duration, tolerance = mapping.get('derived_end_ms'), mapping.get('duration_tolerance_ms')
        if (any(mapping.get(key) != value for key, value in mapped.items())
                or row.get('source_mapping', mapping) != mapping
                or not _finite_number(duration) or duration <= 0
                or not _finite_number(tolerance) or not 0 <= tolerance <= 100
                or abs(duration - (row['end_ms'] - row['start_ms'])) > tolerance):
            raise ValueError('sealed_export_duration_or_source_mapping_unverified')
        if not verify_artifact_cached(root, reference) or not verify_artifact_cached(root, video):
            raise ValueError('exact_export_changed_during_report_verification')
        # The immutable receipt binds the measured export duration to these
        # exact current bytes; a bare duration in a mutable index is insufficient.
        return {'status': 'PROVEN', 'reason': 'exact_receipt_duration_and_video_bytes_verified',
            'duration_ms': duration, 'receipt': reference, 'video': video,
            'duration_basis': 'sealed_export_probe_bound_to_current_artifact_hash'}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {'status': 'NOT_PROVEN', 'reason': str(exc) or type(exc).__name__, 'duration_ms': None}


def retention_summary(index, root=None):
    """Count parent sources once and actual exact exports separately from plans.

    With no archive root this is a planned index view only (NOT_PROVEN).
    With a root, only hash-verified videos with an exact sealed export receipt
    contribute duration. Historical whole-candidate videos never contribute.
    """
    from .device_day_contract import digest
    candidates = report_candidates(index)
    recordings = {row['recording_id']: row for row in index.get('recordings', [])}
    meanings = {row['segment_id']: row for row in index.get('understandings', [])}
    sources, partitions, coverage, seen, duplicate_rows = set(), [], [], set(), 0
    for recording in recordings.values():
        for source in recording.get('sources', []):
            reference = source.get('retained') or source.get('source_ref')
            if source.get('kind') == 'video' and reference:
                sources.add(digest([recording['recording_id'], reference.get('path')]))
    for candidate in candidates:
        if candidate.get('source_ref'):
            sources.add(digest([candidate['recording_id'], candidate['source_ref'].get('path')]))
        left, right = candidate.get('start_ms'), candidate.get('end_ms')
        valid = _finite_number(left) and _finite_number(right) and left < right
        rows = candidate.get('retention_intervals') or []
        partition_available = bool(rows)
        ranges, model_ranges = [], []
        for window in (meanings.get(candidate['segment_id']) or {}).get('windows', []):
            start, end = window.get('start_ms'), window.get('end_ms')
            if valid and _finite_number(start) and _finite_number(end) and start < end:
                if max(left, start) < min(right, end):
                    model_ranges.append([max(left, start), min(right, end)])
        if not rows:
            rows = [{'start_ms': left, 'end_ms': right, 'start_us': candidate.get('start_us'),
                'end_us': candidate.get('end_us'), 'video_value': 'pending', 'activity_category': 'unknown',
                'reason': 'retention_partition_not_available', 'processing_status': 'waiting_for_value_review',
                'evidence': [], 'context_basis': []}]
        for row in rows:
            identity = digest([candidate['segment_id'], candidate.get('source_ref'),
                *[row.get(key) for key in ('start_ms', 'end_ms', 'start_us', 'end_us', 'activity')]])
            if identity in seen:
                duplicate_rows += 1
                continue
            seen.add(identity)
            start, end = row.get('start_ms'), row.get('end_ms')
            bounded = (valid and _finite_number(start) and _finite_number(end) and left <= start < end <= right)
            if bounded:
                ranges.append([start, end])
            value = row.get('video_value') if bounded and row.get('video_value') in {
                'selected', 'pending', 'confirmed_no_value'} else 'pending'
            detail = {key: row.get(key) for key in ('segment_id', 'start_ms', 'end_ms', 'start_us', 'end_us',
                'activity', 'activity_category', 'reason', 'processing_status', 'evidence', 'context_basis',
                'core_action_ranges', 'context_subranges',
                'json_path', 'understanding_path', 'video', 'receipt', 'error', 'retry_at')}
            detail.update(candidate_segment_id=candidate['segment_id'], recording_id=candidate['recording_id'],
                source_ref=candidate.get('source_ref'), video_value=value,
                duration_ms=end-start if bounded else None, bounds_verified_against_candidate=bounded,
                original_video_unchanged=True)
            if value == 'selected':
                detail['actual_video_verification'] = _verified_interval_video(root, candidate, row)
            partitions.append(detail)
        merged = _merged_ranges(ranges)
        coverage.append({'candidate_segment_id': candidate['segment_id'], 'recording_id': candidate['recording_id'],
            'source_ref': candidate.get('source_ref'), 'source_start_ms': left, 'source_end_ms': right,
            'partition_ranges_ms': merged, 'partition_gaps_ms': _range_gaps(left, right, ranges) if valid else [],
            'partition_available': partition_available,
            'partition_overlap_detected': sum(b-a for a, b in ranges) > sum(b-a for a, b in merged),
            'source_bounds_status': 'indexed_media_bounds' if valid else 'NOT_PROVEN',
            'model_window_ranges_ms': _merged_ranges(model_ranges),
            'model_window_gaps_ms': _range_gaps(left, right, model_ranges) if valid else [],
            'model_window_scope': 'returned_sample_windows_not_continuous_visual_coverage'})
    selected = [row for row in partitions if row['video_value'] == 'selected']
    available = [row for row in selected if row['actual_video_verification']['status'] == 'PROVEN']
    actual = sum(row['actual_video_verification']['duration_ms'] for row in available) if root is not None else None
    counts = {value: sum(row['video_value'] == value for row in partitions)
              for value in ('selected', 'pending', 'confirmed_no_value')}
    return {'original_recording_count': len(recordings), 'original_video_count': len(sources),
        'original_count_basis': 'unique_indexed_source_identity_not_full_original_byte_scan',
        'preprocessing_candidate_count': len(candidates), 'partition_count': len(partitions),
        'selected_interval_count': counts['selected'],
        'available_selected_interval_count': len(available) if root is not None else None,
        'unverified_selected_interval_count': len(selected)-len(available) if root is not None else len(selected),
        'pending_interval_count': counts['pending'], 'confirmed_no_value_interval_count': counts['confirmed_no_value'],
        'planned_selected_duration_ms': sum(row['duration_ms'] or 0 for row in selected),
        'pending_duration_ms': sum(row['duration_ms'] or 0 for row in partitions if row['video_value'] == 'pending'),
        'confirmed_no_value_duration_ms': sum(row['duration_ms'] or 0 for row in partitions if row['video_value'] == 'confirmed_no_value'),
        'actual_video_duration_ms': actual, 'actual_video_duration_seconds': actual/1000 if actual is not None else None,
        '原始视频数': len(sources), '选中片段数': counts['selected'],
        '实际可读视频数': len(available) if root is not None else None,
        '实际视频总时长秒': actual/1000 if actual is not None else None,
        '待判定分区数': counts['pending'], '确认无价值分区数': counts['confirmed_no_value'],
        'partition_gap_duration_ms': sum(b-a for row in coverage for a, b in row['partition_gaps_ms']),
        'actual_duration_scope': 'sum_of_verified_available_exact_export_files_only',
        'evidence_status': ('PROVEN' if len(available) == len(selected) else
                            'PARTIAL_EVIDENCE' if available else 'NOT_PROVEN') if root is not None else 'NOT_PROVEN',
        'actual_files_checked': root is not None, 'duplicate_partition_rows_ignored': duplicate_rows,
        'evidence_scope': 'indexed_counts_and_exact_export_integrity_not_model_or_physical_action_accuracy',
        'partitions': partitions, 'source_coverage': coverage,
        'recordings_without_candidates': [rid for rid in recordings
            if not any(row['recording_id'] == rid for row in candidates)],
        'whole_day_capture_proven': False,
        '说明': '原始录像按采集身份计数；分析候选和历史整段视频不计为选中片段。选中计划与实际可读视频分开，时长只累计已验证导出回执对应的实际文件；模型窗口覆盖不等于连续观察。'}


def day_record(index, root=None):
    recordings = {r['recording_id']: r | {'camera_key': r.get('camera_key') or index['archive'][11:]}
                  for r in index.get('recordings', [])}
    meanings = {m['segment_id']: m for m in index.get('understandings', [])}
    segments = sorted(report_candidates(index), key=lambda row: (row['start_us'], row['segment_id']))
    timeline = [segment_record(s, meanings.get(s['segment_id']), recordings[s['recording_id']],
                index.get('context'), segments=[x for x in segments if x['recording_id'] == s['recording_id']]) for s in segments]
    intervals = sorted((r['start_us'], r['end_us']) for r in recordings.values()
                       if r.get('start_us') is not None and r.get('end_us') is not None
                       and r.get('video_processing_complete') is not False)
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
        '全天时间线': timeline, '视频留存统计': retention_summary(index, root),
        '独立录音覆盖': [{'开始时间': timestamp(r['audio']['start_us']) if r['audio'].get('start_us') else None,
                        '结束时间': timestamp(r['audio']['end_us']) if r['audio'].get('end_us') else None,
                        '录音状态': (r.get('audio') or {}).get('status'),
                        '识别状态': (r.get('transcription') or {}).get('status'),
                        '识别结果': (r.get('transcription') or {}).get('outcome'),
                        '原录音': r.get('audio'), '录像处理完成': r.get('video_processing_complete')}
                       for r in recordings.values() if r.get('audio')],
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
    from .device_day_timeline import day_bounds
    start, end = day_bounds(index['archive'][:10])
    layout = DeviceDayLayout(Path(config['storage']['archive_root']), index['archive'][11:], start)
    events = context_events(config, {'camera_key': index['archive'][11:],
        'recording_start_us': start, 'recording_end_us': end})
    index['context'] = load_day_context(layout) | {'events': events['events'], 'field_event_status': events['status'],
        'field_event_errors': events.get('errors', []), 'field_source_references': events.get('source_references', [])}
    return index

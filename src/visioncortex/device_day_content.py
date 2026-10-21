"""Readable time folders, projected from receipts without running models."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .device_day_contract import atomic_bytes, atomic_json, digest, read_json, safe_child
from .device_day_schedule import in_processing_scope


def time_folder(start, end):
    def label(value):
        return datetime.fromtimestamp(value / 1e6, ZoneInfo('Asia/Shanghai')).strftime('%H-%M-%S.%f')
    return f'{label(start)}_{label(end)}'


def _write(path, value):
    if path.is_file() and read_json(path) == value:
        return
    atomic_json(path, value)


def window_files(config, index):
    """Only the current receipt generation, never arbitrary historical results."""
    backend = config['storage'].get('local_cache_root')
    if not backend:
        return []
    root = Path(config['storage']['archive_root']) / index['archive']
    receipts = Path(backend) / 'device-day-receipts' / index['archive']
    records = {r['recording_id']: r for r in index.get('recordings', []) if in_processing_scope(
        config.get('device_day', {}), {'recording_start_us': r.get('start_us'), 'recording_end_us': r.get('end_us')})}
    result = []
    for segment in index.get('segments', []):
        if segment['recording_id'] not in records:
            continue
        receipt = receipts / segment['recording_id'] / 'understanding.json'
        if not receipt.is_file():
            continue
        stage = read_json(receipt)
        key = stage.get('key')
        if not isinstance(key, str) or len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            continue
        from .device_day_content_paths import analysis_folder
        from types import SimpleNamespace
        folder = analysis_folder(SimpleNamespace(root=root), segment, key)
        if not folder.is_dir():
            folder = safe_child(root, 'MultimodalUnderstanding/' + time_folder(segment['start_us'], segment['end_us'])
                                + f'/Analysis/{segment["segment_id"]}/{key}')
        if not folder.is_dir():
            folder = safe_child(root, f'MultimodalUnderstanding/ClipUnderstanding/{segment["segment_id"]}/{key}')
        for path in sorted(folder.glob('*/Result.json')):
            request = path.with_name('Input.json')
            if request.is_file():
                result.append((segment, stage, request, path))
    return result


def content_revision(config, index):
    backend = config['storage'].get('local_cache_root')
    receipts = Path(backend) / 'device-day-receipts' / index['archive'] if backend else None
    root = Path(config['storage']['archive_root']) / index['archive']
    paths = []
    for _, _, request, result in window_files(config, index):
        paths.extend((request, result))
        try:
            frames = read_json(request).get('metadata', {}).get('frames', [])
            for frame in frames:
                paths.append(safe_child(root, frame['path']))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass  # The window reader isolates malformed requests individually.
    for segment in index.get('segments', []):
        paths.extend((root/'MultimodalUnderstanding/.Evidence/Analysis'/segment['segment_id']/'StepReview.json',
                      root/'MultimodalUnderstanding'/time_folder(segment['start_us'], segment['end_us'])/'Analysis/StepReview.json'))
    paths.extend(root/'Comment'/name for name in ('Protocol.json', 'Comment.jsonl'))
    runtime = config['storage'].get('local_runtime_root')
    if runtime:
        paths.extend(Path(runtime)/'device-day'/name for name in ('field-events.sqlite3', 'field-events.sqlite3-wal'))
    if receipts:
        paths.extend(receipts / r['recording_id'] / 'understanding.json' for r in index.get('recordings', []))
    revisions = []
    for path in dict.fromkeys(paths):
        try:
            stat = path.stat()
            revisions.append((str(path), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_ino))
        except OSError as exc:
            revisions.append((str(path), type(exc).__name__))
    return tuple(revisions)


def readable_source_index(index):
    """Copy only source fields consumed by the readable/time projections.

    This is never a replacement for the canonical ProcessedClips/Index.json.
    It can be cached without retaining a second set of dense CV diagnostics.
    """
    # Dense CV candidate/audit arrays can make a day index tens of MB. The
    # readable projection needs media references and clocks, not another copy
    # of those unchanged diagnostics. Preserve the canonical index on disk.
    value = {k: v for k, v in index.items() if k not in ('recordings', 'segments', 'understandings')}
    value['recordings'] = [deepcopy({**r, 'processing': {k: v for k, v in (r.get('processing') or {}).items()
                            if k not in ('batches', 'scan_reports', 'audit_artifacts')}})
                           for r in index.get('recordings', [])]
    value['segments'] = [deepcopy({k: v for k, v in s.items() if k != 'activity_audit'})
                         for s in index.get('segments', [])]
    value['understandings'] = deepcopy(index.get('understandings', []))
    return value


def with_partial_understandings(config, index):
    """Expose valid completed windows even when a later window is pending/failed."""
    from .device_day_models import validate_understanding
    from .media_time import capture_us
    from .device_day_verification import verify_artifact_cached
    value = readable_source_index(index)
    root = Path(config['storage']['archive_root']) / index['archive']
    from .device_day_content_paths import aliases, public_references
    mapping = aliases(root)
    value['understandings'] = public_references(value['understandings'], mapping)
    value['segments'] = public_references(value['segments'], mapping)
    for row in value['recordings']:
        row['transcription'] = public_references(row.get('transcription'), mapping)
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    backend = config['storage'].get('local_cache_root')
    if backend:
        for row in value.get('recordings', []):
            receipt = Path(backend) / 'device-day-receipts' / index['archive'] / row['recording_id'] / 'understanding.json'
            if receipt.is_file():
                stage = read_json(receipt)
                row.setdefault('stages', {})['understanding'] = {k: stage.get(k) for k in ('status', 'key', 'message', 'reason')}
    complete = {u['segment_id'] for u in value.get('understandings', []) if u.get('status') != 'partial'}
    partial = {}
    for segment, stage, request_path, result_path in window_files(config, index):
        sid = segment['segment_id']
        if sid in complete:
            continue
        try:
            request, result = read_json(request_path), read_json(result_path)
            metadata, raw = request.get('metadata', {}), result.get('model_result', {})
            if (not request.get('input_key') or request['input_key'] != result.get('input_key')
                    or raw.get('status') != 'completed' or metadata.get('source_ref') != segment['source_ref']
                    or metadata.get('segment_id') != sid):
                continue
            left, right = metadata.get('start_ms'), metadata.get('end_ms')
            if left is None or right is None or not segment['start_ms'] <= left < right <= segment['end_ms'] + 1:
                continue
            frames = metadata.get('frames')
            if (not isinstance(frames, list) or not frames
                    or not all(verify_artifact_cached(root, frame) for frame in frames)):
                raise ValueError('partial_window_frame_artifact_integrity_failed')
            identity_integrity = 'unavailable_legacy_prompt_missing'
            if 'prompt' in request:
                if not isinstance(request['prompt'], str) or request['input_key'] != digest([metadata, request['prompt']]):
                    raise ValueError('partial_window_request_identity_changed')
                identity_integrity = 'verified_exact_metadata_and_prompt'
            receipt_integrity = {}
            for name, path in (('input', request_path), ('result', result_path)):
                refs = [ref for ref in stage.get('artifacts') or []
                        if public_references(ref.get('path'), mapping) == path.relative_to(root).as_posix()]
                if refs and not all(verify_artifact_cached(root, ref) for ref in refs):
                    raise ValueError('partial_window_' + name + '_receipt_identity_changed')
                receipt_integrity[name] = 'verified_against_stage_artifact' if refs else 'unavailable_stage_artifact_reference'
            record = records[segment['recording_id']]
            clock = (record.get('processing') or {}).get('clock_mapping') or {'origin_us': record['start_us']}
            if 'experiment_steps' in raw:
                from .device_day_steps import validate
                parsed = validate(raw, metadata['frames'], left, right, record['start_us'], clock)
            else:
                parsed = validate_understanding(raw, metadata['frames'], metadata.get('comments', []),
                    left, right, record['start_us'], clock)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            value.setdefault('partial_window_integrity_errors', []).append({'segment_id': sid,
                'input': request_path.relative_to(root).as_posix(),
                'model_receipt': result_path.relative_to(root).as_posix(), 'error_type': type(exc).__name__,
                'reason': str(exc)[:200], 'automatic_action': 'isolate_window_and_recheck_original_evidence'})
            continue  # Invalid model output remains in its receipt, not report prose.
        if identity_integrity.startswith('unavailable') or any(state.startswith('unavailable') for state in receipt_integrity.values()):
            parsed['uncertainties'].append('历史窗口缺少完整的请求或阶段产物摘要，当前仅展示已核验帧字节与结构的部分观察；不能视为整阶段完成。')
        for step in parsed['steps']:
            step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
        item = partial.setdefault(sid, {'segment_id': sid, 'activity': segment['activity'],
            'status': 'partial', 'stage_status': stage.get('status'), 'mode': 'sampled_frames',
            'evidence_status': 'PARTIAL_EVIDENCE', 'physical_action_confirmed': False, 'windows': []})
        item['windows'].append({'start_ms': left, 'end_ms': right, **parsed,
            'source_frames': metadata['frames'], 'coverage': metadata.get('coverage'),
            'frame_artifact_integrity': 'verified', 'request_identity_integrity': identity_integrity,
            'receipt_integrity': receipt_integrity,
            'content_source': metadata.get('content_source', 'multimodal_context'),
            'model_receipt': result_path.relative_to(root).as_posix(),
            'input': request_path.relative_to(root).as_posix(), 'usage': raw.get('usage'),
            'response_cache_reused': result.get('response_cache_reused', False)})
    # An earlier canonical checkpoint is not completion. Replace its view
    # with the current byte-verified windows, without duplicating the segment.
    value['understandings'] = [*[u for u in value.get('understandings', [])
                                if u['segment_id'] not in partial], *partial.values()]
    from .device_day_steps import reviewed_window
    meanings = {u['segment_id']: u for u in value['understandings']}
    for segment in value['segments']:
        if not all(k in segment for k in ('start_us', 'end_us')):
            continue
        record = records[segment['recording_id']]
        clock = (record.get('processing') or {}).get('clock_mapping') or {'origin_us': record['start_us']}
        reviewed = reviewed_window(root, segment, clock)
        if reviewed is not None:
            previous = meanings.get(segment['segment_id'], {})
            meanings[segment['segment_id']] = {**previous, 'segment_id': segment['segment_id'],
                'activity': segment['activity'], 'mode': 'historical_sampled_step_review',
                'content_source': 'video_only',
                'status': 'completed', 'evidence_status': 'PARTIAL_EVIDENCE',
                'physical_action_confirmed': False, 'windows': [reviewed],
                'prior_window_receipts': [w.get('model_receipt') for w in previous.get('windows', [])]}
    value['understandings'] = public_references(list(meanings.values()), mapping)
    return value


def transcript_interval(record):
    """New independent seals use their own fixed audio interval; old paths stay."""
    span = record.get('transcript_span')
    if span is None:
        return record['start_us'], record['end_us']
    audio = record.get('audio') or {}
    start, end = span.get('start_us'), span.get('end_us')
    if (span.get('schema_version') != 'visioncortex-sealed-audio-span/1'
            or span.get('recording_id') != record['recording_id']
            or span.get('time_basis') != 'sealed_recorder_audio'
            or type(start) is not int or type(end) is not int or not 0 < start < end
            or audio.get('status') != 'provided'
            or (start, end) != (audio.get('start_us'), audio.get('end_us'))
            or span.get('audio_source_signature') != audio.get('source_signature')
            or not isinstance(span.get('audio_identity_sha256'), str)
            or len(span['audio_identity_sha256']) != 64):
        raise ValueError('Independent transcript publication has no bound sealed audio interval')
    return start, end


def transcript_documents(archive, record, context=None):
    """Pure native transcript projection shared by publication and proof."""
    common = {'archive': archive, 'device': archive[11:], 'date': archive[:10],
              'timezone': 'Asia/Shanghai', 'timestamp_unit': 'unix_microseconds',
              'path_base': 'device_day', 'physical_action_confirmed': False}
    start, end = transcript_interval(record)
    stt = record.get('transcription') or {}
    value = {**common, 'schema_version': 'visioncortex-readable-transcript/1',
        'recording_id': record['recording_id'], 'start_us': start, 'end_us': end,
        **({'transcript_span': record['transcript_span']} if record.get('transcript_span') else {}),
        'audio': record.get('audio'), 'status': stt.get('status'), 'outcome': stt.get('outcome'),
        'content_source': 'machine_transcribed_speech', 'model_invocation': stt.get('model_invocation'),
        'processing_method': 'automatic_stt', 'accuracy': 'NOT_PROVEN', 'sentences': stt.get('comments') or [],
        'execution_windows': stt.get('chunks') or [], 'original_transcript': stt.get('transcript_file')}
    from .device_day_narrative import experimenter, timestamp
    value['录音时间线'] = [{'开始时间': timestamp(s['start_us']), '结束时间': timestamp(s['end_us']),
        '原话': s.get('text'), '实验员': experimenter(context, common['device'], s['start_us'], s['end_us']),
        '转写来源': s.get('transcript_path')} for s in value['sentences']]
    lines = [f"设备：{common['device']}  日期：{common['date']}",
             f"录音识别状态：{value['status']} / {value['outcome']}",
             '自动语音转写。识别为空不证明录音没有语音。', '']
    for row in value['sentences']:
        stamp = datetime.fromtimestamp(row['start_us']/1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')
        lines.append(f"[{stamp}] {row['text']}")
    return value, ('\n'.join(lines) + '\n').encode()


def publish_transcript(config, archive, record, context=None):
    """Publish one completed recording without walking or rewriting the day."""
    root = Path(config['storage']['archive_root']) / archive
    value, data = transcript_documents(archive, record, context)
    relative = 'Comment/' + time_folder(value['start_us'], value['end_us'])
    _write(safe_child(root, relative + '/Transcript.json'), value)
    target = safe_child(root, relative + '/Transcript.txt')
    if not target.is_file() or target.read_bytes() != data:
        atomic_bytes(target, data)
    return ({k: value[k] for k in ('recording_id', 'start_us', 'end_us', 'status', 'outcome')} |
            {'content': relative + '/Transcript.json', 'text': relative + '/Transcript.txt'})


def segment_documents(index, root, segment, source_meaning, interval, record, audio_entries):
    """Pure readable segment/retention documents; source receipts stay intact."""
    common = {'archive': index['archive'], 'device': index['archive'][11:], 'date': index['archive'][:10],
              'timezone': 'Asia/Shanghai', 'timestamp_unit': 'unix_microseconds',
              'path_base': 'device_day', 'physical_action_confirmed': False}
    from .device_day_steps import readable_meaning
    meaning, experiment_steps, step_status = readable_meaning(source_meaning)
    from .device_day_narrative import effective_meaning
    meaning = effective_meaning(meaning)
    from .device_day_content_paths import LAYOUT_VERSION
    if segment.get('camera_key', index['archive'][11:]) != index['archive'][11:]:
        raise ValueError('Understanding belongs to another device/camera')
    record = record | {'camera_key': index['archive'][11:]}
    stage = (record.get('stages') or {}).get('understanding') or {}
    value = {**common, 'schema_version': 'visioncortex-readable-understanding/1',
        'recording_id': segment['recording_id'], 'segment_id': segment['segment_id'],
        'start_us': segment['start_us'], 'end_us': segment['end_us'],
        'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
        'source_video': segment['source_ref'], 'video': segment.get('video'),
        'record_scope': 'retention_interval' if interval else 'preprocessing_candidate',
        'candidate_segment_id': segment.get('candidate_segment_id', segment['segment_id']),
        'retention_intervals': segment.get('retention_intervals', []),
        'selection_interval': interval,
        'activity': segment['activity'], 'status': (meaning or {}).get('status', 'completed' if meaning else stage.get('status', 'pending')),
        'message': stage.get('message'), 'understanding': meaning,
        'experiment_steps': experiment_steps, 'step_understanding_status': step_status,
        'step_accuracy': 'NOT_PROVEN',
        'transcript_content': next((a['content'] for a in audio_entries if a['recording_id'] == segment['recording_id']), None)}
    from .device_day_narrative import segment_record, understanding_state
    state = understanding_state(meaning, record)
    value['status'] = state['status']
    value = {'内容记录': segment_record(segment, meaning, record, index.get('context'),
              segments=[s for s in index['segments'] if s['recording_id'] == segment['recording_id']]), **value}
    if interval is not None:
        value['内容记录']['区间价值依据'] = interval.get('evidence', [])
        value['内容记录']['动作原始区间'] = interval.get('core_action_ranges', [])
        value['内容记录']['必要上下文区间'] = interval.get('context_subranges', [])
        value['内容记录']['连贯性说明'] = ('有来源的必要上下文仅帮助观看相近操作，'
            '各动作仍按原始范围记录；不证明衔接时间内动作持续发生。'
            if interval.get('context_subranges') else '按已有价值与边界证据记录，未自动增加上下文。')
        value['内容记录']['原模型关联观察'] = [
            {'动作': row['original_observation'].get('action'),
             '内容': row['original_observation'].get('description'),
             '对象': row['original_observation'].get('objects'),
             '原片开始毫秒': row['original_observation'].get('start_ms'),
             '原片结束毫秒': row['original_observation'].get('end_ms'),
             '说明': '原观察跨越当前区间或引用区间外画面；这里仅保留关联，不声称当前视频呈现完整动作。'}
            for window in (meaning or {}).get('windows', [])
            for row in window.get('overlapping_parent_observations', [])
            if row['collection'] in {'experiment_steps', 'work_activities'}]
    from .media.retained_intervals import disposition
    decision = disposition(segment, meaning, record, index.get('context'),
                segments=[s for s in index['segments'] if s['recording_id'] == segment['recording_id']])
    if interval is None and segment.get('retention_intervals'):
        rows = segment['retention_intervals']
        selected_rows = [row for row in rows if row['video_value'] == 'selected']
        no_value = all(row['video_value'] == 'confirmed_no_value' for row in rows)
        decision.update(proposed_partition_identity=(decision.get('partition') or {}).get('partition_identity'),
            partition={'schema_version': 'visioncortex-retention-interval-partition/1',
                'candidate_segment_id': segment['segment_id'], 'source_ref': segment['source_ref'],
                'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
                'intervals': rows, 'actual_publication_evidence_applied': True,
                'candidate_is_export_unit': False, 'default_padding_ms': 0, 'physical_action_confirmed': False},
            selected_duration_ms=sum(row['end_ms']-row['start_ms'] for row in selected_rows),
            pending_duration_ms=sum(row['end_ms']-row['start_ms'] for row in rows if row['video_value'] == 'pending'),
            video_value='selected' if selected_rows else 'confirmed_no_value' if no_value else 'pending',
            no_value_confirmed=no_value,
            decision='retain_derivative' if selected_rows else 'omit_static_derivative' if no_value else 'pending_value_review',
            value_signals=sorted({row['reason'] for row in selected_rows}),
            reason='只导出已通过证据校验的精确选中区间；其他区间按各自原因保留记录与原片映射。')
    if interval is not None:
        # Presentation of the parent's verified partition. Do not infer a
        # second selection from cropped model observations.
        decision.update({key: interval[key] for key in ('video_value', 'reason', 'evidence',
            'context_basis', 'activity_category') if key in interval})
        decision['decision'] = {'selected': 'retain_derivative',
            'confirmed_no_value': 'omit_static_derivative', 'pending': 'pending_value_review'}[interval['video_value']]
        decision['no_value_confirmed'] = interval['video_value'] == 'confirmed_no_value'
        decision['source_start_ms'], decision['source_end_ms'] = segment['start_ms'], segment['end_ms']
        decision['partition'] = {'schema_version': 'visioncortex-retention-interval-partition/1',
            'candidate_segment_id': segment['candidate_segment_id'], 'source_ref': segment['source_ref'],
            'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
            'intervals': [interval], 'projection_from_verified_candidate_partition': True,
            'candidate_is_export_unit': False, 'default_padding_ms': 0, 'physical_action_confirmed': False}
        duration = segment['end_ms']-segment['start_ms']
        decision['selected_duration_ms'] = duration if interval['video_value'] == 'selected' else 0
        decision['pending_duration_ms'] = duration if interval['video_value'] == 'pending' else 0
        decision['value_signals'] = [interval['reason']] if interval['video_value'] == 'selected' else []
    video = segment.get('video')
    from .device_day_verification import verify_artifact_cached
    present = bool(video and verify_artifact_cached(root, video))
    value['video_status'] = ('available' if present else 'missing' if video else
                             'omitted_by_selection' if decision['decision'] == 'omit_static_derivative' else
                             'pending_selection' if decision['decision'] == 'pending_value_review' else 'pending_retention')
    if interval is None and segment.get('retention_intervals'):
        value['video_status'] = 'historical_video_preserved' if present else 'see_exact_retention_intervals'
    value['处理视频留存'] = {**decision,
                            'previous_execution': segment.get('derived_video_disposition'),
                            'derived_video_present': present, 'status': value['video_status']}
    execution = segment.get('derived_video_disposition') or {}
    if interval is not None:
        execution = execution | {key: interval[key] for key in ('processing_status', 'error', 'retry_at') if key in interval}
    value['处理视频留存']['processing_status'] = ('completed' if present else
        'failed' if execution.get('processing_status') == 'failed' else
        'completed' if decision['decision'] == 'omit_static_derivative' else
        'waiting_for_value' if decision['decision'] == 'pending_value_review' else 'queued')
    if interval is not None and not present and execution.get('processing_status') == 'waiting_for_evidence':
        value['video_status'] = value['处理视频留存']['status'] = 'waiting_for_evidence'
        value['处理视频留存']['processing_status'] = 'waiting_for_evidence'
    if interval is None and segment.get('retention_intervals'):
        rows = segment['retention_intervals']
        selected = [row for row in rows if row['video_value'] == 'selected']
        available = sum(bool(row.get('video') and verify_artifact_cached(root, row['video'])) for row in selected)
        undecided = sum(row['video_value'] == 'pending' for row in rows)
        value['处理视频留存'].update(
            candidate_is_export_unit=False, historical_scope=segment.get('historical_scope'),
            selected_interval_count=len(selected), available_selected_interval_count=available,
            pending_value_interval_count=undecided,
            processing_status=('failed' if any(row.get('processing_status') == 'failed' for row in selected)
                else 'partial' if available < len(selected) or undecided else 'completed'),
            derived_video_present=bool(available), historical_video_present=present)
    if execution.get('processing_status') == 'failed' and not present:
        value['处理视频留存'].update(error=execution.get('error'), retry_at=execution.get('retry_at'))
    value['layout_version'] = LAYOUT_VERSION
    retention = {**common, 'schema_version': 'visioncortex-readable-retention/1',
        'recording_id': segment['recording_id'], 'segment_id': segment['segment_id'],
        'start_us': segment['start_us'], 'end_us': segment['end_us'],
        'source_ref': segment['source_ref'], 'video': video,
        'folder_retained': True, 'no_value_proven': bool(decision.get('no_value_confirmed')),
        'record_scope': value['record_scope'], 'candidate_segment_id': value['candidate_segment_id'],
        'retention_intervals': value['retention_intervals'], 'selection_interval': interval,
        'selection_status': decision['decision'],
        **value['处理视频留存']}
    return value, retention


def visual_entry(value, relative):
    """The readable index repeats only this document's actual state."""
    narrative = value['内容记录']
    return {'开始时间': narrative['开始时间'], '结束时间': narrative['结束时间'],
        '状态说明': narrative['状态说明'], '阶段状态': narrative['阶段状态'], '处理原因': narrative['处理原因'],
        **{key: value[key] for key in ('recording_id', 'segment_id', 'start_us', 'end_us', 'activity', 'status')},
        'content': relative + '/Understanding.json', 'video_status': value['video_status'],
        'record_scope': value['record_scope'], 'candidate_segment_id': value['candidate_segment_id']}


def content_index(archive, directory, entries):
    """Pure index metadata and understanding counts, shared with proof."""
    summary = {}
    if directory == 'MultimodalUnderstanding':
        from collections import Counter
        candidates = [row for row in entries if row.get('record_scope') != 'retention_interval']
        counts = Counter(row['status'] for row in candidates)
        summary = {'理解进度': {'片段总数': len(candidates),
            '留存区间记录数': len(entries)-len(candidates), '已返回结果': counts['completed'],
            '部分返回': counts['partial'], '已提交待确认': counts['submitted_unknown'], '各状态片段数': dict(counts),
            '说明': '按下方 content 路径打开对应片段的 Understanding.json；只有实际返回的模型观察才展示步骤内容。'}}
    return {**summary, 'archive': archive, 'device': archive[11:], 'date': archive[:10],
        'timezone': 'Asia/Shanghai', 'timestamp_unit': 'unix_microseconds', 'path_base': 'device_day',
        'physical_action_confirmed': False, 'schema_version': 'visioncortex-readable-content-index/1',
        'entries': sorted(entries, key=lambda row: row['start_us'])}


def publish_content(config, index):
    """Mutable readable projections; immutable execution paths stay resolvable."""
    root = Path(config['storage']['archive_root']) / index['archive']
    common = {'archive': index['archive'], 'device': index['archive'][11:], 'date': index['archive'][:10],
              'timezone': 'Asia/Shanghai', 'timestamp_unit': 'unix_microseconds',
              'path_base': 'device_day', 'physical_action_confirmed': False}
    audio_entries, visual_entries, records = [], [], {}
    for record in index.get('recordings', []):
        if not in_processing_scope(config.get('device_day', {}),
                {'recording_start_us': record.get('start_us'), 'recording_end_us': record.get('end_us')}):
            continue
        records[record['recording_id']] = record
        if not record.get('start_us') or not record.get('end_us'):
            continue
        audio_entries.append(publish_transcript(config, index['archive'], record, index.get('context')))
    from .device_day_retention_projection import interval_views
    for segment, source_meaning, interval in interval_views(index):
        if segment['recording_id'] not in records:
            continue
        from .device_day_content_paths import clip_name, understanding_relative
        relative = understanding_relative(segment)
        value, retention = segment_documents(index, root, segment, source_meaning, interval,
                                              records[segment['recording_id']], audio_entries)
        _write(safe_child(root, relative + '/Understanding.json'), value)
        _write(safe_child(root, 'ProcessedClips/Clips/' + clip_name(segment) + '/Retention.json'), retention)
        if interval is not None:
            from .device_day_contract import VERSION
            _write(safe_child(root, segment['json_path']), {
                **common, **segment, 'schema_version': VERSION,
                'selection_interval': interval, 'projection_from_candidate': True,
                'model_invoked': False, 'physical_action_confirmed': False})
        visual_entries.append(visual_entry(value, relative))
    for directory, entries in [('Comment', audio_entries), ('MultimodalUnderstanding', visual_entries)]:
        _write(root / directory / 'Index.json', content_index(index['archive'], directory, entries))
    return {'audio': audio_entries, 'multimodal': visual_entries, 'digest': digest([audio_entries, visual_entries])}

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
    paths = [p for _, _, request, result in window_files(config, index) for p in (request, result)]
    root = Path(config['storage']['archive_root']) / index['archive']
    paths.extend(root/'MultimodalUnderstanding'/time_folder(s['start_us'], s['end_us'])/'Analysis/StepReview.json'
                 for s in index.get('segments', []))
    if receipts:
        paths.extend(receipts / r['recording_id'] / 'understanding.json' for r in index.get('recordings', []))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.is_file())


def with_partial_understandings(config, index):
    """Expose valid completed windows even when a later window is pending/failed."""
    from .device_day_models import validate_understanding
    from .media_time import capture_us
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
    root = Path(config['storage']['archive_root']) / index['archive']
    from .device_day_content_paths import aliases, public_references
    mapping = aliases(root)
    value['understandings'] = public_references(value['understandings'], mapping)
    for row in value['recordings']:
        row['transcription'] = public_references(row.get('transcription'), mapping)
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    backend = config['storage'].get('local_cache_root')
    if backend:
        for row in value.get('recordings', []):
            receipt = Path(backend) / 'device-day-receipts' / index['archive'] / row['recording_id'] / 'understanding.json'
            if receipt.is_file():
                stage = read_json(receipt)
                row.setdefault('stages', {})['understanding'] = {k: stage.get(k) for k in ('status', 'key', 'message')}
    complete = {u['segment_id'] for u in value.get('understandings', [])}
    partial = {}
    for segment, stage, request_path, result_path in window_files(config, index):
        sid = segment['segment_id']
        if sid in complete:
            continue
        request, result = read_json(request_path), read_json(result_path)
        metadata, raw = request.get('metadata', {}), result.get('model_result', {})
        if (not request.get('input_key') or request['input_key'] != result.get('input_key')
                or raw.get('status') != 'completed' or metadata.get('source_ref') != segment['source_ref']
                or metadata.get('segment_id') != sid):
            continue
        left, right = metadata.get('start_ms'), metadata.get('end_ms')
        if left is None or right is None or not segment['start_ms'] <= left < right <= segment['end_ms'] + 1:
            continue
        record = records[segment['recording_id']]
        clock = (record.get('processing') or {}).get('clock_mapping') or {'origin_us': record['start_us']}
        try:
            if 'experiment_steps' in raw:
                from .device_day_steps import validate
                parsed = validate(raw, metadata['frames'], left, right, record['start_us'], clock)
            else:
                parsed = validate_understanding(raw, metadata['frames'], metadata.get('comments', []),
                    left, right, record['start_us'], clock)
        except (ValueError, KeyError):
            continue  # Invalid model output remains in its receipt, not report prose.
        for step in parsed['steps']:
            step.update(start_us=capture_us(clock, step['start_ms']), end_us=capture_us(clock, step['end_ms']))
        item = partial.setdefault(sid, {'segment_id': sid, 'activity': segment['activity'],
            'status': 'partial', 'stage_status': stage.get('status'), 'mode': 'sampled_frames',
            'evidence_status': 'PARTIAL_EVIDENCE', 'physical_action_confirmed': False, 'windows': []})
        item['windows'].append({'start_ms': left, 'end_ms': right, **parsed,
            'source_frames': metadata['frames'], 'coverage': metadata.get('coverage'),
            'content_source': metadata.get('content_source', 'multimodal_context'),
            'model_receipt': result_path.relative_to(root).as_posix(),
            'input': request_path.relative_to(root).as_posix(), 'usage': raw.get('usage'),
            'response_cache_reused': result.get('response_cache_reused', False)})
    value['understandings'] = [*value.get('understandings', []), *partial.values()]
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


def publish_transcript(config, archive, record):
    """Publish one completed recording without walking or rewriting the day."""
    root = Path(config['storage']['archive_root']) / archive
    common = {'archive': archive, 'device': archive[11:], 'date': archive[:10],
              'timezone': 'Asia/Shanghai', 'timestamp_unit': 'unix_microseconds',
              'path_base': 'device_day', 'physical_action_confirmed': False}
    relative = 'Comment/' + time_folder(record['start_us'], record['end_us'])
    stt = record.get('transcription') or {}
    value = {**common, 'schema_version': 'visioncortex-readable-transcript/1',
        'recording_id': record['recording_id'], 'start_us': record['start_us'], 'end_us': record['end_us'],
        'audio': record.get('audio'), 'status': stt.get('status'), 'outcome': stt.get('outcome'),
        'content_source': 'machine_transcribed_speech', 'model_invocation': stt.get('model_invocation'),
        'human_reviewed': False, 'accuracy': 'NOT_PROVEN', 'sentences': stt.get('comments') or [],
        'execution_windows': stt.get('chunks') or [], 'original_transcript': stt.get('transcript_file')}
    _write(safe_child(root, relative + '/Transcript.json'), value)
    lines = [f"设备：{common['device']}  日期：{common['date']}",
             f"录音识别状态：{value['status']} / {value['outcome']}",
             '机器转写，未经人工复核。识别为空不证明录音没有语音。', '']
    for row in value['sentences']:
        stamp = datetime.fromtimestamp(row['start_us']/1e6, ZoneInfo('Asia/Shanghai')).isoformat(timespec='milliseconds')
        lines.append(f"[{stamp}] {row['text']}")
    target = safe_child(root, relative + '/Transcript.txt')
    data = ('\n'.join(lines) + '\n').encode()
    if not target.is_file() or target.read_bytes() != data:
        atomic_bytes(target, data)
    return ({k: value[k] for k in ('recording_id', 'start_us', 'end_us', 'status', 'outcome')} |
            {'content': relative + '/Transcript.json', 'text': relative + '/Transcript.txt'})


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
        audio_entries.append(publish_transcript(config, index['archive'], record))
    meanings = {u['segment_id']: u for u in index.get('understandings', [])}
    for segment in index.get('segments', []):
        if segment['recording_id'] not in records:
            continue
        from .device_day_steps import readable_meaning
        meaning, experiment_steps, step_status = readable_meaning(meanings.get(segment['segment_id']))
        relative = 'MultimodalUnderstanding/' + time_folder(segment['start_us'], segment['end_us'])
        record = records[segment['recording_id']]
        stage = (record.get('stages') or {}).get('understanding') or {}
        value = {**common, 'schema_version': 'visioncortex-readable-understanding/1',
            'recording_id': segment['recording_id'], 'segment_id': segment['segment_id'],
            'start_us': segment['start_us'], 'end_us': segment['end_us'],
            'source_start_ms': segment['start_ms'], 'source_end_ms': segment['end_ms'],
            'source_video': segment['source_ref'], 'video': segment.get('video'),
            'activity': segment['activity'], 'status': (meaning or {}).get('status', 'completed' if meaning else stage.get('status', 'pending')),
            'message': stage.get('message'), 'understanding': meaning,
            'experiment_steps': experiment_steps, 'step_understanding_status': step_status,
            'step_accuracy': 'NOT_PROVEN',
            'transcript_content': next((a['content'] for a in audio_entries if a['recording_id'] == segment['recording_id']), None)}
        _write(safe_child(root, relative + '/Understanding.json'), value)
        visual_entries.append({k: value[k] for k in ('recording_id', 'segment_id', 'start_us', 'end_us', 'activity', 'status')} |
                              {'content': relative + '/Understanding.json'})
    for directory, entries in [('Comment', audio_entries), ('MultimodalUnderstanding', visual_entries)]:
        _write(root / directory / 'Index.json', {**common, 'schema_version': 'visioncortex-readable-content-index/1',
            'entries': sorted(entries, key=lambda r: r['start_us'])})
    return {'audio': audio_entries, 'multimodal': visual_entries, 'digest': digest([audio_entries, visual_entries])}

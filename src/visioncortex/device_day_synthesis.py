"""A separate, source-cited day model pass over saved observations.

Video windows remain immutable observations. The daily report owns contextual
synthesis, its exact inputs, paid-call receipts and completeness limits.
Rendering never invokes this paid pass implicitly.
"""
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from .device_day_contract import artifact, digest, read_json, safe_child, verify_artifact
from .device_day_narrative import experimenter, timestamp
from .media_time import capture_us

from .device_day_call_templates import DAY_SYNTHESIS

PROMPT = DAY_SYNTHESIS.prompt


def _device_bundle(index):
    """Include every available window and independent stream, without pixels."""
    records = {r['recording_id']: r for r in index.get('recordings', [])}
    meanings = {m['segment_id']: m for m in index.get('understandings', [])}
    camera, context = index['archive'][11:], index.get('context') or {}
    rows = []

    def add(kind, value, source):
        row = {'kind': kind, 'device': camera, **value, 'source': source}
        row['evidence_id'] = kind + '-' + digest(row)[:24]
        rows.append(row)

    missing = []
    for segment in sorted(index.get('segments', []), key=lambda s: (s['start_us'], s['segment_id'])):
        record = records[segment['recording_id']]
        clock = (record.get('processing') or {}).get('clock_mapping') or {'origin_us': record['start_us']}
        meaning = meanings.get(segment['segment_id']) or {}
        if meaning.get('status') != 'completed':
            missing.append({'segment_id': segment['segment_id'], 'start_us': segment['start_us'],
                            'end_us': segment['end_us'], 'status': meaning.get('status', 'pending')})
        for window in meaning.get('windows', []):
            if window.get('start_ms') is None or window.get('end_ms') is None:
                missing.append({'segment_id': segment['segment_id'], 'status': 'legacy_window_time_missing'})
                continue
            left = capture_us(clock, window['start_ms'])
            right = capture_us(clock, window['end_ms'])
            add('video', {'start_us': left, 'end_us': right, 'start': timestamp(left), 'end': timestamp(right),
                'summary': window.get('summary'), 'activity_kind': window.get('activity_kind'),
                'observability': window.get('observability'),
                'experimenter': experimenter(context, camera, left, right),
                'experiment_steps': [{k: deepcopy(v) for k, v in s.items() if k not in ('frame_ids', 'comment_ids')}
                                     for s in window.get('experiment_steps', [])],
                'work_activities': [{k: deepcopy(v) for k, v in s.items() if k not in ('frame_ids', 'comment_ids')}
                                    for s in window.get('work_activities', [])],
                'uncertainties': window.get('uncertainties', []), 'human_reviewed': False},
                {'segment_id': segment['segment_id'], 'input': window.get('input'),
                 'model_receipt': window.get('model_receipt'), 'original_video': segment['source_ref'],
                 'clock': {k: v for k, v in clock.items() if k != 'points'},
                 'physical_action_confirmed': False})
    for record in records.values():
        for sentence in (record.get('transcription') or {}).get('comments') or []:
            add('speech', {'start_us': sentence['start_us'], 'end_us': sentence['end_us'],
                'start': timestamp(sentence['start_us']), 'end': timestamp(sentence['end_us']),
                'text': sentence.get('text'), 'time_basis': sentence.get('time_basis'),
                'time_alignment': sentence.get('time_alignment'), 'human_reviewed': False,
                'experimenter': experimenter(context, camera, sentence['start_us'], sentence['end_us'])},
                {'transcript': sentence.get('transcript_path'), 'audio': record.get('audio')})
    for event in context.get('events', []):
        add('field', {'start_us': event.get('start_us'), 'end_us': event.get('end_us'),
            'type': event.get('type'), 'capture_device_id': event.get('capture_device_id'),
            'payload': event.get('payload'), 'clock': event.get('clock'),
            'wearer_id': event.get('wearer_id'), 'instrument_id': event.get('instrument_id')},
            {'event_id': event.get('event_id'), 'revision': event.get('revision'),
             'evidence_refs': event.get('evidence_refs'), 'media': event.get('media')})
    if context.get('protocol'):
        add('protocol', {'role': 'planned_not_observed', 'protocol': context['protocol']},
            {'file': 'Comment/Protocol.json'})
    rows.sort(key=lambda r: (r.get('start_us') or 0, r['evidence_id']))
    completeness = {'missing_or_partial_segments': missing,
        'unprocessed_recordings': [{'recording_id': r['recording_id'], 'start_us': r.get('start_us'),
                                   'end_us': r.get('end_us')}
                                  for r in records.values() if r.get('source_original')
                                  and r['recording_id'] not in {s['recording_id'] for s in index.get('segments', [])}],
        'unresolved_index_records': [{'recording_id': r.get('recording_id'),
                                     'start_us': r.get('start_us'), 'end_us': r.get('end_us')}
                                    for r in index.get('unresolved_index_records', [])],
        'requested_recordings': index.get('requested_recordings'),
        'indexed_recordings': len(records), 'video_window_count': sum(r['kind'] == 'video' for r in rows),
        'field_event_status': context.get('field_event_status', 'not_received'),
        'whole_day_capture_proven': False, 'semantic_accuracy': 'NOT_PROVEN'}
    if index.get('processing_scope') == 'selected_processed_clips_only':
        completeness.update(processing_scope=index['processing_scope'],
            selected_clips_expected=index.get('selected_clip_count'),
            pending_value_review_intervals=[{k: row.get(k) for k in
                ('start_us', 'end_us', 'status', 'no_value_proven', 'original_available')}
                for row in index.get('pending_value_review_intervals', [])])
    return {'schema_version': 'visioncortex-day-context/1', 'archive': index['archive'],
            'timezone': 'Asia/Shanghai', 'evidence': rows, 'completeness': completeness}


def evidence_bundle(index, context_indexes=()):
    indexes = {index['archive']: index}
    for other in context_indexes:
        if other['archive'][:10] != index['archive'][:10]:
            raise ValueError('Day context cannot substitute another date')
        if other['archive'] in indexes and other != indexes[other['archive']]:
            raise ValueError('Conflicting device-day context revisions')
        indexes[other['archive']] = other
    bundles = [_device_bundle(row) for _, row in sorted(indexes.items())]
    bundle = bundles[0] if len(bundles) == 1 else {
        'schema_version': 'visioncortex-laboratory-day-context/1', 'date': index['archive'][:10],
        'timezone': 'Asia/Shanghai', 'archives': sorted(indexes),
        'evidence': sorted([e for b in bundles for e in b['evidence']],
                           key=lambda e: (e.get('start_us') or 0, e['evidence_id'])),
        'completeness': {b['archive']: b['completeness'] for b in bundles},
        'cross_device_note': '独立设备与时钟来源；时间接近不能自动确认同一人、同一动作或同一次实验。'}
    coverage_id = 'coverage-' + digest(bundle['completeness'])[:24]
    bundle['coverage_evidence_id'] = coverage_id
    bundle['evidence'].append({'evidence_id': coverage_id, 'kind': 'coverage',
        'completeness': bundle['completeness'], 'source': {'archives': sorted(indexes),
        'files': [name+'/ProcessedClips/Index.json' for name in sorted(indexes)],
        'scope': 'saved_index_metadata_not_physical_observation'}})
    return bundle


def validate_findings(raw, evidence_ids):
    if raw.get('status') != 'completed' or not isinstance(raw.get('findings'), list) or len(raw['findings']) > 16:
        raise ValueError('Day model must return a completed bounded findings array')
    findings = deepcopy(raw['findings'])
    for row in findings:
        refs = row.get('evidence_refs')
        if (not isinstance(refs, list) or not refs or any(not isinstance(r, str) for r in refs)
                or len(refs) != len(set(refs)) or not set(refs) <= evidence_ids):
            raise ValueError('Day finding cites missing, duplicate or unknown evidence')
        if (row.get('category') not in {'experiment', 'support', 'uncertainty'}
                or row.get('basis') not in {'source_observation', 'contextual_inference'}
                or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('title', 'text'))
                or not isinstance(row.get('uncertainties'), list)
                or any(not isinstance(s, str) for s in row['uncertainties'])):
            raise ValueError('Invalid day finding content')
        if row['basis'] == 'contextual_inference' and len(refs) < 2:
            raise ValueError('Contextual inference requires multiple source references')
    return findings


def _batches(rows, maximum=48000, *, measure=None):
    """Bound each model request; never truncate the end of a day."""
    batches, batch, size = [], [], 0
    for row in rows:
        width = (measure(row) if measure else len(json.dumps(row, ensure_ascii=False, separators=(',', ':'))))
        if width > maximum:
            raise ValueError('A day evidence item exceeds the bounded model context; split it explicitly')
        if batch and size + width > maximum:
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += width
    if batch:
        batches.append(batch)
    return batches


def synthesize(config, layout, index, analyzer, *, context_indexes=(), retry_output_tokens=None, analyzer_factory=None):
    """Explicit paid operation; same-input retries reuse the existing ledger."""
    if retry_output_tokens is not None and (type(retry_output_tokens) is not int
            or not int(config.get('mllm', {}).get('max_output_tokens', 4096)) < retry_output_tokens <= 65536):
        raise ValueError('Explicit day synthesis output allowance must increase the base limit and remain bounded')
    from contextlib import ExitStack
    with ExitStack() as resources:
        expanded = None
        if retry_output_tokens is not None:
            from .mllm import ArkAnalyzer
            expanded_config = config | {'mllm': config['mllm'] | {'max_output_tokens': retry_output_tokens}}
            expanded = (analyzer_factory or ArkAnalyzer)(expanded_config)
            resources.callback(expanded.close)
        return _synthesize(config, layout, index, analyzer, context_indexes=context_indexes,
                           retry_output_tokens=retry_output_tokens, expanded_analyzer=expanded)


def _synthesize(config, layout, index, analyzer, *, context_indexes, retry_output_tokens, expanded_analyzer):
    from .scene_requests import window
    context_indexes = list(context_indexes)
    bundle = evidence_bundle(index, context_indexes)
    if not bundle['evidence']:
        raise ValueError('No saved evidence for day synthesis')
    key = digest(bundle)
    folder = layout.reports/'Analysis'/'DayContext'/key
    rows, receipts = bundle['evidence'], []
    for level in range(8):
        from .scene_transport import compact
        def width(row):
            if not config.get('mllm', {}).get('compact_scene_metadata', False):
                return len(json.dumps(row, ensure_ascii=False, separators=(',', ':')))
            _, _, receipt = compact({'schema_version': 'visioncortex-day-context-input/1', 'frames': [],
                'evidence': [row], 'coverage_evidence_id': bundle['coverage_evidence_id']}, [])
            return receipt['metadata_characters_after']
        batches = _batches(rows, 24000, measure=width)
        collected = []
        def one(number, batch):
            metadata = {'schema_version': 'visioncortex-day-context-input/1', 'activity': 'all_day_context',
                'source_ref': {'sha256': key}, 'start_ms': 0, 'end_ms': 86400000,
                'frames': [], 'comments': [], 'protocol': None, 'archive': index['archive'],
                'completeness': bundle['completeness'], 'evidence': batch,
                'synthesis_level': level, 'scope': 'whole_available_day' if len(batches) == 1 else 'bounded_day_part'}
            directory = folder/f'Level{level}'/f'Part{number:04d}'
            available = ({r['evidence_id'] for r in batch} if level == 0 else
                         {ref for r in batch for ref in r['evidence_refs']})
            available.add(bundle['coverage_evidence_id'])
            metadata['coverage_evidence_id'] = bundle['coverage_evidence_id']
            request_config = config
            input_path, result_path = directory/'Input.json', directory/'Result.json'
            increase = retry_output_tokens is not None and level > 0
            if retry_output_tokens is not None and input_path.is_file() and result_path.is_file():
                from .scene_requests import request_identity
                previous, received = read_json(input_path), read_json(result_path)
                paired_input = (previous.get('prompt') == PROMPT and previous.get('metadata') == metadata
                    and previous.get('input_key') == received.get('input_key') == digest([metadata, PROMPT]))
                paired = paired_input and received.get('request_policy') == request_identity(config, PROMPT, metadata)['request_policy']
                raw_previous = received.get('model_result') or {}
                if paired and raw_previous.get('status') == 'completed':
                    increase = False  # Keep the successful original generation policy, without repaying.
                elif paired and raw_previous.get('status') == 'failed' and '截断' in str(raw_previous.get('error', '')):
                    increase = True
                expanded_config = config | {'mllm': config['mllm'] | {'max_output_tokens': retry_output_tokens}}
                if paired_input and received.get('request_policy') == request_identity(expanded_config, PROMPT, metadata)['request_policy']:
                    increase = True
            if increase:
                from .multimodal_usage import UsageLedger, SubmittedUnknown
                from .scene_requests import request_key
                if UsageLedger(config, readonly=True).unresolved(request_key(config, PROMPT, metadata)):
                    raise SubmittedUnknown('submitted_unknown: reconcile the original allowance before changing output limits')
                request_config = config | {'mllm': config['mllm'] | {'max_output_tokens': retry_output_tokens}}
            raw, findings, reused = window(request_config, expanded_analyzer if increase else analyzer, PROMPT, metadata, [],
                directory/'Input.json', directory/'Result.json', lambda raw: validate_findings(raw, available))
            receipt = {'input': artifact(layout.root, directory/'Input.json'),
                'result': artifact(layout.root, directory/'Result.json'), 'response_cache_reused': reused,
                'usage': raw.get('usage')}
            return findings, receipt
        workers = min(len(batches), 4, max(1, int(config.get('mllm', {}).get('group_workers', 2))))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(copy_context().run, one, number, batch) for number, batch in enumerate(batches)]
            results = [future.result() for future in futures]
        for findings, receipt in results:
            receipts.append(receipt)
            collected.extend(findings)
        if len(batches) == 1:
            return {'schema_version': 'visioncortex-day-synthesis/1', 'status': 'completed_for_available_inputs',
                'scope': 'available_saved_evidence_only', 'evidence_status': 'PARTIAL_EVIDENCE',
                'input_digest': key, 'findings': collected, 'evidence': bundle['evidence'],
                'context_archives': sorted({i['archive'] for i in [index, *context_indexes]}),
                'completeness': bundle['completeness'], 'model_receipts': receipts,
                'human_reviewed': False, 'physical_action_confirmed': False}
        if not collected:
            raise ValueError('Intermediate day synthesis omitted all findings')
        rows = collected
    raise ValueError('Day synthesis exceeds bounded hierarchy; saved requests remain reusable')


def current_synthesis(layout, index):
    """Render the saved synthesis only for its exact current evidence bundle."""
    path = layout.reports/'LaboratoryDailyReport.json'
    value = read_json(path).get('全天综合理解') if path.is_file() else None
    others = []
    for name in (value or {}).get('context_archives', []):
        if name != index['archive']:
            other = safe_child(layout.root.parent, name + '/ProcessedClips/Index.json')
            if not other.is_file():
                return {'status': 'stale_inputs_changed', '说明': '全天上下文来源暂不可读，需要核验后重新综合。'}
            others.append(read_json(other))
    if value and isinstance(value.get('input_digest'), str) and value['input_digest'] == digest(evidence_bundle(index, others)):
        for ref in value.get('model_receipts', []):
            # Readable prose must still have its original successful response.
            if not all(verify_artifact(layout.root, ref[name]) for name in ('input', 'result')):
                raise ValueError('Saved day synthesis receipt identity changed')
            result = read_json(safe_child(layout.root, ref['result']['path']))
            if result.get('model_result', {}).get('status') != 'completed':
                raise ValueError('Saved day synthesis response is not completed')
        if not value.get('model_receipts') or result['model_result'].get('findings') != value.get('findings'):
            raise ValueError('Saved day synthesis differs from its model response')
        return value
    return {'status': 'pending' if not value else 'stale_inputs_changed',
            '说明': '等待模型结合当前全天证据综合理解；已有片段理解继续保留。'}


def main():
    import argparse
    from pathlib import Path
    from .config import load_config
    from .device_day_contract import DeviceDayLayout, atomic_json
    from .device_day_reports import render_day, preserve_prior_synthesis
    from .device_day_write_guard import storage_guard
    from .mllm import ArkAnalyzer
    parser = argparse.ArgumentParser(description='Explicit source-cited whole-day model synthesis')
    parser.add_argument('--config', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--context-archive', action='append', default=[])
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--retry-output-tokens', type=int)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    root = Path(config['storage']['archive_root'])
    index = read_json(safe_child(root, args.archive + '/ProcessedClips/Index.json'))
    from datetime import datetime
    from zoneinfo import ZoneInfo
    start = round(datetime.fromisoformat(index['archive'][:10]).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()*1e6)
    layout = DeviceDayLayout(root, index['archive'][11:], start)
    others = [read_json(safe_child(root, name + '/ProcessedClips/Index.json')) for name in args.context_archive]
    all_indexes = [index, *others]
    incomplete = any(_device_bundle(i)['completeness']['missing_or_partial_segments']
                     or _device_bundle(i)['completeness']['unprocessed_recordings']
                     or i.get('unresolved_index_records') or i.get('pending_value_review_intervals') for i in all_indexes)
    if incomplete and not args.allow_partial:
        parser.error('Incomplete day inputs; --allow-partial explicitly permits a labelled interim synthesis')
    with storage_guard(config):
        analyzer = ArkAnalyzer(config)
        try:
            result = synthesize(config, layout, index, analyzer, context_indexes=others,
                                retry_output_tokens=args.retry_output_tokens)
        finally:
            analyzer.close()
        path = layout.reports/'LaboratoryDailyReport.json'
        previous = read_json(path) if path.is_file() else {}
        preserve_prior_synthesis(layout, result)
        atomic_json(path, previous | {'全天综合理解': result})
        render_day(layout, index)


if __name__ == '__main__':
    main()

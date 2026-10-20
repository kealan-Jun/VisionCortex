"""Partition visible, valuable video spans without changing recorder-owned data."""
from copy import deepcopy
import math

from ..device_day_contract import atomic_bytes, atomic_json, digest
from ..device_day_narrative import effective_meaning
from ..media_time import capture_us

POLICY = 'visioncortex-value-only-derivatives/3'


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bounds(row, left, right):
    start, end = row.get('start_ms'), row.get('end_ms')
    return (_number(start) and _number(end) and left <= start < end <= right)


def _refs(window):
    return {key: deepcopy(window[key]) for key in ('input', 'model_receipt', 'coverage') if key in window}


def _frame_reference(frame):
    return (isinstance(frame.get('path'), str) and bool(frame['path'])
            and type(frame.get('size_bytes')) is int and frame['size_bytes'] > 0
            and isinstance(frame.get('sha256'), str) and len(frame['sha256']) == 64)


def _clock(segment, recording):
    clock = (recording.get('processing') or {}).get('clock_mapping')
    clock = clock or {'origin_us': segment['start_us'] - round(segment['start_ms'] * 1000)}
    try:
        valid = (capture_us(clock, segment['start_ms']) == segment['start_us']
                 and capture_us(clock, segment['end_ms']) == segment['end_us'])
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    return clock, valid


def _actions(segment, windows, clock, valid_clock):
    selected, uncertain, static = [], [], []
    left, right = segment['start_ms'], segment['end_ms']
    for window in windows:
        if not isinstance(window, dict):
            uncertain.append({'start_ms': left, 'end_ms': right,
                'evidence': {'invalid_window': repr(window)}, 'reason': 'boundary_uncertain'})
            continue
        if not _bounds(window, left, right):
            uncertain.append({'start_ms': left, 'end_ms': right,
                'evidence': _refs(window) | {'invalid_window_bounds': True}, 'reason': 'boundary_uncertain'})
            continue
        refs = _refs(window)
        frames = window.get('source_frames') or []
        frames = frames if isinstance(frames, list) else []
        frame_map = {f['frame_id']: f for f in frames if isinstance(f, dict)
                     and isinstance(f.get('frame_id'), str) and _number(f.get('local_ms'))}
        completed = window.get('status', 'completed') == 'completed'
        window_identity = (completed and valid_clock and bool(window.get('input'))
                           and bool(window.get('model_receipt'))
                           and window.get('source_ref', segment['source_ref']) == segment['source_ref'])
        for field, category in (('experiment_steps', 'experiment'), ('work_activities', 'non_experiment')):
            for number, step in enumerate(window.get(field) or []):
                if not isinstance(step, dict):
                    continue
                evidence = refs | {'collection': field, 'step_index': number,
                    'description': step.get('description'), 'action': step.get('action'),
                    'objects': deepcopy(step.get('objects') or []), 'work_category': step.get('category'),
                    'before_state': step.get('before_state'), 'after_state': step.get('after_state'),
                    'model_claimed_interval': {k: (step.get(k) if step.get(k) is None or _number(step.get(k))
                        else repr(step.get(k))) for k in ('start_ms', 'end_ms', 'start_us', 'end_us')},
                    'basis': step.get('basis'), 'outcome': step.get('outcome'),
                    'visible_evidence': step.get('visible_evidence'),
                    'frame_ids': deepcopy(step.get('frame_ids') or [])}
                bounded = _bounds(step, window['start_ms'], window['end_ms'])
                time_matches = valid_clock and all(
                    key not in step or step[key] == capture_us(clock, step[ms])
                    for key, ms in (('start_us', 'start_ms'), ('end_us', 'end_ms'))
                ) if bounded else False
                ids = step.get('frame_ids') or []
                cited = [frame_map[identifier] for identifier in ids if identifier in frame_map]
                trusted = (window_identity and bounded and time_matches and step.get('basis') == 'observed'
                           and step.get('boundary_status') not in {'uncertain', 'boundary_uncertain'}
                           and step.get('boundary_uncertain') is not True
                           and step.get('source_ref', segment['source_ref']) == segment['source_ref']
                           and bool(step.get('visible_evidence')) and len(cited) == len(ids)
                           and len({f['local_ms'] for f in cited}) >= 2
                           and all(_frame_reference(f) for f in cited)
                           and all(step['start_ms'] <= f['local_ms'] <= step['end_ms'] for f in cited))
                if trusted:
                    start = max(step['start_ms'], min(f['local_ms'] for f in cited))
                    end = min(step['end_ms'], max(f['local_ms'] for f in cited))
                    evidence.update(source_frames=deepcopy(cited), boundary_basis='observed_source_frame_extent',
                        boundary_uncertainty='仅保存所引实际画面之间的可见内容，不证明完整物理动作或未观察到的边界。',
                        physical_action_confirmed=False)
                    selected.append({'start_ms': start, 'end_ms': end, 'activity_category': category,
                        'reason': 'observed_experiment_step' if category == 'experiment' else 'observed_non_experiment_work',
                        'evidence': evidence, 'context_basis': []})
                else:
                    # Keep rejected/uncertain observations, including their original
                    # claimed extent; never turn a model window into a selected action.
                    start, end = ((step['start_ms'], step['end_ms']) if bounded
                                  else (window['start_ms'], window['end_ms']))
                    uncertain.append({'start_ms': start, 'end_ms': end, 'evidence': evidence,
                                      'reason': 'boundary_uncertain'})
        coverage = window.get('coverage') or {}
        continuous = (window_identity and window.get('activity_kind') == 'static'
            and window.get('observability') == 'clear'
            and not window.get('experiment_steps') and not window.get('work_activities')
            and not window.get('steps')
            and not window.get('invalid_structured_content')
            and coverage.get('all_decoded_frames_submitted') is True
            and coverage.get('native_pts_to_pixels_verified') is True
            and coverage.get('input_frames') == len(frames) and len(frame_map) == len(frames) >= 2
            and all(_frame_reference(f) for f in frames)
            and all(window['start_ms'] <= f['local_ms'] < window['end_ms'] for f in frames))
        if continuous:
            static.append({'start_ms': window['start_ms'], 'end_ms': window['end_ms'],
                           'evidence': refs | {'source_frames': deepcopy(frames)}})
    return selected, uncertain, static


def _evidence_rows(item):
    value = item['evidence']
    return value if isinstance(value, list) else [value]


def _known_state(value):
    return (isinstance(value, str) and bool(value.strip()) and not any(
        term in value.casefold() for term in ('未知', 'unknown', '无法确认', '无法判断', '不清楚', '不确定', 'uncertain')))


def _object_names(values):
    return sorted(row.strip() for row in values) if values and all(
        isinstance(row, str) and row.strip() for row in values) else []


def _object_absent(text, objects):
    compact = ''.join(text.split()).casefold()
    for item in objects:
        obj = ''.join(str(item).split()).casefold()
        if obj and (any(prefix + obj in compact for prefix in ('未见', '没有', '未看到', '看不到', '不见'))
                    or any(obj + suffix in compact for suffix in (
                        '不可见', '不再可见', '不在画面', '已移出', '移出画面', '已离开', '被遮挡'))):
            return True
    return False


def _explicit_other_activity(text, category):
    """A narrow context veto, never an activity classifier or new fact."""
    import re
    for other, phrases in {'phone': ('操作手机', '使用手机'), 'cleaning': ('擦拭', '清洁'),
                           'experiment': ('移液',)}.items():
        if other == category:
            continue
        for phrase in phrases:
            for match in re.finditer(re.escape(phrase), text):
                prefix = re.split(r'[，,。；;！!？?]', text[:match.start()])[-1]
                if not any(term in prefix for term in ('未见', '没有', '未观察到', '未看到', '未发生',
                    '并未', '不是', '不在', '不曾', '禁止', '不要', '无需', '无法确认', '不确定', '有无', '是否')):
                    return True
    return False


def _context_uncertain(text):
    return any(term in text.casefold() for term in (
        '未知', 'unknown', 'uncertain', '不确定', '看不清', '不可见', '遮挡', '模糊', '无法', '不明', '疑似'))


def _observed_context(window, start, end, objects, category, *, include_start=False, include_end=False):
    """Real readable object observations, not an inference from time proximity."""
    observations = {row['frame_id']: row for row in window.get('frame_observations') or []
                    if isinstance(row, dict) and isinstance(row.get('text'), str) and row['text'].strip()}
    found = []
    for frame in window.get('source_frames') or []:
        if (not isinstance(frame, dict) or not _frame_reference(frame) or not _number(frame.get('local_ms'))
                or not (start <= frame['local_ms'] if include_start else start < frame['local_ms'])
                or not (frame['local_ms'] <= end if include_end else frame['local_ms'] < end)
                or frame.get('frame_id') not in observations):
            continue
        observation = observations[frame['frame_id']]
        words = observation['text'].casefold()
        if (_object_absent(observation['text'], objects) or _explicit_other_activity(observation['text'], category)
                or _context_uncertain(observation['text'])):
            # An explicit contrary observation cannot be outweighed by another
            # matching object name elsewhere in this same proposed gap.
            return []
        if (objects and any(str(obj).casefold() in words for obj in objects)
                and not any(term in words for term in ('无关活动', '其他活动', '另一活动', '看不清', '不可见'))):
            found.append((frame, observation))
    return sorted(found, key=lambda row: (row[0]['local_ms'], row[0]['frame_id']))


def _static_scene_context(window, start, end, objects, category):
    """All actual gap samples explicitly describe static scenery, never action.

    This is only a bounded viewing connection between anchored same-object
    operations. It does not establish continuous stillness or object presence
    between samples, and it never supplies short-work pre/post padding.
    """
    observations = {row['frame_id']: row for row in window.get('frame_observations') or []
                    if isinstance(row, dict) and isinstance(row.get('text'), str)}
    frames = [frame for frame in window.get('source_frames') or [] if isinstance(frame, dict)
              and _number(frame.get('local_ms')) and start < frame['local_ms'] < end]
    if not frames or len({frame.get('frame_id') for frame in frames}) != len(frames):
        return []
    found = []
    for frame in frames:
        observation = observations.get(frame.get('frame_id'))
        if not _frame_reference(frame) or not observation:
            return []
        text = observation['text'].strip()
        if (not text or _context_uncertain(text) or _object_absent(text, objects)
                or _explicit_other_activity(text, category)
                or any(term in text for term in ('其他活动', '无关活动', '另一活动', '走动', '行走', '移动', '拿起', '放下'))
                or not any(term in text.casefold() for term in ('静止', '无人', '无变化', '没有变化', 'unchanged', 'static'))
                or any(term in text for term in ('非静止', '不再静止', '不是静止', '并非静止'))):
            return []
        found.append((frame, observation))
    return sorted(found, key=lambda row: (row[0]['local_ms'], row[0]['frame_id']))


def _other_activity(windows, start, end, category, objects):
    for window in windows:
        if not isinstance(window, dict):
            continue
        for field in ('experiment_steps', 'work_activities'):
            for row in window.get(field) or []:
                if (not isinstance(row, dict) or not _number(row.get('start_ms'))
                        or not _number(row.get('end_ms')) or row['start_ms'] > row['end_ms']):
                    continue
                overlaps = (row['start_ms'] < end and (row['end_ms'] > start
                            or row['start_ms'] == row['end_ms'] and row['start_ms'] >= start))
                if overlaps and (field == 'experiment_steps' or row.get('category') != category
                                 or _object_names(row.get('objects') or []) != objects):
                    return True
    return False


def _context_basis(window, kind, start, end, observed, **extra):
    return _refs(window) | {'kind': kind, 'basis': 'observed_context_only',
        'start_ms': start, 'end_ms': end,
        'source_frames': [deepcopy(frame) for frame, _ in observed],
        'frame_observations': [deepcopy(row) for _, row in observed],
        'reason': '仅为连贯观看同对象的相近操作提供已观察上下文，不证明间隙持续动作、状态相同或完整物理过程。',
        'physical_action_confirmed': False, **extra}


def _coherent_work(selected, windows, segment):
    """Join only source-evidenced viewing context; keep every action separate."""
    by_window = {(w.get('input'), w.get('model_receipt')): w for w in windows if isinstance(w, dict)}
    groups = []
    for item in sorted(selected, key=lambda row: (row['start_ms'], row['end_ms'])):
        evidence = _evidence_rows(item)[0]
        core = {key: deepcopy(evidence.get(key)) for key in ('work_category', 'objects', 'frame_ids', 'input', 'model_receipt')}
        core.update(start_ms=item['start_ms'], end_ms=item['end_ms'], activity_category=item['activity_category'])
        current = item | {'evidence': _evidence_rows(item), 'core_action_ranges': [core], 'context_subranges': []}
        previous = groups[-1] if groups else None
        prior = previous['evidence'][-1] if previous else {}
        objects = _object_names(evidence.get('objects') or [])
        window = by_window.get((evidence.get('input'), evidence.get('model_receipt')))
        compatible = (previous and item['activity_category'] == previous['activity_category'] == 'non_experiment'
            and evidence.get('work_category') == prior.get('work_category')
            and evidence.get('work_category') in {'computer', 'phone', 'organizing', 'cleaning', 'walking', 'waiting', 'other'}
            and bool(objects) and _object_names(prior.get('objects') or []) == objects
            and (evidence.get('input'), evidence.get('model_receipt')) == (prior.get('input'), prior.get('model_receipt'))
            and window is not None and 0 <= item['start_ms'] - previous['end_ms'] <= 15000
            and _known_state(prior.get('after_state')) and _known_state(evidence.get('before_state')))
        observed = _observed_context(window, previous['end_ms'], item['start_ms'], objects,
                                     evidence.get('work_category')) if compatible else []
        context_kind = 'continuity_context'
        if compatible and not observed:
            observed = _static_scene_context(window, previous['end_ms'], item['start_ms'], objects,
                                             evidence.get('work_category'))
            context_kind = 'anchored_static_scene_context'
        if (compatible and observed and not _other_activity(windows, previous['end_ms'], item['start_ms'],
                                                           evidence.get('work_category'), objects)):
            basis = _context_basis(window, context_kind, previous['end_ms'], item['start_ms'], observed,
                objects=objects, work_category=evidence.get('work_category'),
                preceding_after_state=prior.get('after_state'), following_before_state=evidence.get('before_state'))
            if context_kind == 'anchored_static_scene_context':
                basis.update(sampled_static_context_only=True, object_presence_during_gap='NOT_PROVEN',
                    continuous_motion='NOT_PROVEN', actual_input_gap_frame_count=len(observed),
                    endpoint_action_refs=[{key: deepcopy(row.get(key)) for key in (
                        'input', 'model_receipt', 'collection', 'step_index', 'model_claimed_interval', 'frame_ids')}
                        for row in (prior, evidence)],
                    reason='仅为观看两端同对象的已观察操作保留有实际采样静止记录的短衔接；不证明间隙逐帧同对象存在、持续动作或状态相同。')
            previous['end_ms'] = item['end_ms']
            previous['reason'] = 'observed_non_experiment_work_with_continuity_context'
            previous['evidence'].extend(current['evidence'])
            previous['core_action_ranges'].extend(current['core_action_ranges'])
            previous['context_basis'].append(basis)
            previous['context_subranges'].append(deepcopy(basis))
        else:
            groups.append(current)
    for group in groups:
        if group['activity_category'] != 'non_experiment':
            continue
        first, last = group['evidence'][0], group['evidence'][-1]
        window = by_window.get((first.get('input'), first.get('model_receipt')))
        if window is None:
            continue
        objects = _object_names(first.get('objects') or [])
        if not objects:
            continue
        for side, evidence in (('before', first), ('after', last)):
            core = group['core_action_ranges'][0 if side == 'before' else -1]
            if core['end_ms'] - core['start_ms'] > 2000 or not _known_state(evidence.get('before_state' if side == 'before' else 'after_state')):
                continue
            edge = group['start_ms' if side == 'before' else 'end_ms']
            start, end = ((max(window['start_ms'], edge-2000), edge) if side == 'before'
                          else (edge, min(window['end_ms'], edge+2000)))
            observed = _observed_context(window, start, end, objects, evidence.get('work_category'),
                                         include_start=side == 'before', include_end=side == 'after')
            if not observed:
                continue
            nearest = observed[-1:] if side == 'before' else observed[:1]
            stamp = nearest[0][0]['local_ms']
            left, right = (stamp, edge) if side == 'before' else (edge, stamp)
            if (_other_activity(windows, left, right, evidence.get('work_category'), objects)
                    or min(group['start_ms'], left) <= segment['start_ms']
                    and max(group['end_ms'], right) >= segment['end_ms']):
                continue
            basis = _context_basis(window, 'readability_context', left, right, nearest, side=side,
                objects=objects, work_category=evidence.get('work_category'),
                adjacent_state=evidence.get('before_state' if side == 'before' else 'after_state'))
            group['start_ms' if side == 'before' else 'end_ms'] = stamp
            group['context_basis'].append(basis)
            group['context_subranges'].append(deepcopy(basis))
            group['reason'] = 'observed_non_experiment_work_with_readable_context'
    return groups


def _context_intervals(segment, context, clock, valid_clock):
    """Explicit bounded context backed by previously verified owned evidence.

    This internal port has no implicit producer: its caller must verify the
    cited evidence artifacts and their source/time binding before supplying a
    row. Speech/OCR overlap, a nearby event, or an unbounded reason is never a
    substitute and never adds padding to the visible action span.
    """
    result = []
    for row in (context or {}).get('video_context_intervals') or []:
        if (not isinstance(row, dict) or not valid_clock or row.get('source_ref') != segment['source_ref']
                or row.get('basis') != 'observed' or not row.get('reason') or not row.get('evidence')
                or not _bounds(row, segment['start_ms'], segment['end_ms'])
                or any(key in row and row[key] != capture_us(clock, row[ms])
                       for key, ms in (('start_us', 'start_ms'), ('end_us', 'end_ms')))):
            continue
        result.append({'start_ms': row['start_ms'], 'end_ms': row['end_ms'],
            'activity_category': row.get('activity_category') if row.get('activity_category') in
                {'experiment', 'non_experiment', 'unknown'} else 'unknown',
            'reason': 'explicit_bounded_necessary_context', 'evidence': deepcopy(row['evidence']),
            'context_basis': [deepcopy(row)]})
    return result


def _continuous_selected_groups(intervals):
    """One export for a connected positive union; no unselected gap is added.

    Atomic evidence membership is a provenance boundary, not a video boundary.
    Retain every original atomic row and action while combining only touching
    selected rows of the same category/activity in this one source candidate.
    """
    groups = []
    for row in intervals:
        previous = groups[-1] if groups else None
        boundary = row['start_ms']
        # Context connections retain their separate, stricter owner above.
        # Two context extensions touching must not establish a new cross-window
        # or different-work bridge. This join only removes atomic boundaries
        # crossed by actual positive action coverage on both sides.
        positive_boundary = (previous
            and any(core['start_ms'] < boundary <= core['end_ms']
                    for core in previous.get('core_action_ranges', []))
            and any(core['start_ms'] <= boundary < core['end_ms']
                    for core in row.get('core_action_ranges', [])))
        compatible = (positive_boundary and row['video_value'] == previous['video_value'] == 'selected'
            and row['start_ms'] == previous['end_ms'] and row['start_us'] == previous['end_us']
            and all(row.get(key) == previous.get(key) for key in
                    ('activity', 'activity_category', 'activity_categories')))
        if not compatible:
            groups.append(row)
            continue
        if 'selected_subranges' not in previous:
            previous['selected_subranges'] = [deepcopy(previous)]
        previous['selected_subranges'].append(deepcopy(row))
        previous.update(end_ms=row['end_ms'], end_us=row['end_us'],
            export_group_basis='contiguous_selected_union',
            export_group_limitation='连续选中内容作为一个视频保存；各原始观察范围独立保留，不宣称同一或完整物理动作。')
        for key in ('evidence', 'context_basis', 'core_action_ranges', 'context_subranges'):
            values = previous.setdefault(key, [])
            for item in row.get(key, []):
                if item not in values:
                    values.append(deepcopy(item))
        previous['reason'] = '+'.join(sorted({atom['reason'] for atom in previous['selected_subranges']}))
    return groups


def partition(segment, meaning, recording, context=None, *, segments=None):
    """Full half-open partition in original media milliseconds and recorder time.

    The candidate is an analysis unit, never an export decision. Select only
    observed action spans narrowed to cited source-frame extent, or explicitly
    bounded necessary context. No default padding or activity-label selection.
    This pure policy consumes evidence already verified by publication ports;
    materialization must also verify actual source and generated video bytes.
    """
    left, right = segment['start_ms'], segment['end_ms']
    if not _bounds(segment, 0, float('inf')):
        raise ValueError('Candidate requires finite positive original-media bounds')
    clock, valid_clock = _clock(segment, recording)
    meaning = effective_meaning(meaning) or {}
    identity_matches = (meaning.get('segment_id', segment['segment_id']) == segment['segment_id']
                        and recording.get('recording_id', segment.get('recording_id')) == segment.get('recording_id'))
    windows = (meaning.get('windows') or []) if identity_matches else []
    chosen, uncertain, static = _actions(segment, windows, clock, valid_clock)
    chosen = _coherent_work(chosen, windows, segment)
    chosen.extend(_context_intervals(segment, context, clock, valid_clock))
    from ..device_day_coverage import value_assessment
    speech = ((recording.get('transcription') or {}).get('comments') or [])
    ctx = (context or {}) | {'comments': [*((context or {}).get('comments') or []), *speech]}
    protected = value_assessment(segment, segments or [segment], ctx)['protection_signals']
    audio_pending = ((recording.get('audio') or {}).get('status') == 'provided'
                     and (recording.get('transcription') or {}).get('status') != 'completed')
    static_allowed = meaning.get('status', 'completed') == 'completed' and not protected and not audio_pending
    boundaries = sorted({left, right, *(item[key] for item in [*chosen, *uncertain, *static]
                                      for key in ('start_ms', 'end_ms'))}) if valid_clock else [left, right]
    intervals = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        selected = [r for r in chosen if r['start_ms'] <= start and r['end_ms'] >= end]
        unknown = [r for r in uncertain if r['start_ms'] <= start and r['end_ms'] >= end] if valid_clock else uncertain
        idle = [r for r in static if r['start_ms'] <= start and r['end_ms'] >= end]
        categories = sorted({r['activity_category'] for r in selected})
        value = 'selected' if selected else 'confirmed_no_value' if idle and static_allowed and not unknown else 'pending'
        reason = ('+'.join(sorted({r['reason'] for r in selected})) if selected else
                  'continuous_static_review' if value == 'confirmed_no_value' else
                  'clock_mapping_mismatch' if not valid_clock else 'boundary_uncertain' if unknown else
                  'source_identity_mismatch' if not identity_matches else
                  'input_or_context_pending' if audio_pending or protected else 'value_not_determined')
        row = {'start_ms': start, 'end_ms': end,
            'start_us': capture_us(clock, start) if valid_clock else segment['start_us'],
            'end_us': capture_us(clock, end) if valid_clock else segment['end_us'],
            'video_value': value, 'activity': ('active' if 'experiment' in categories else 'inactive') if selected
                else segment.get('activity', 'inactive'),
            'activity_category': categories[0] if len(categories) == 1 else 'unknown',
            'reason': reason, 'evidence': [deepcopy(e) for r in selected or unknown or idle for e in _evidence_rows(r)],
            'context_basis': [deepcopy(b) for r in selected for b in r['context_basis']],
            'boundary_status': 'observed_extent' if selected else 'boundary_uncertain' if unknown else 'unassessed',
            'physical_action_confirmed': False}
        if selected:
            row['core_action_ranges'] = [deepcopy(core) for r in selected for core in r.get('core_action_ranges', [])]
            row['context_subranges'] = [deepcopy(part) for r in selected for part in r.get('context_subranges', [])]
        if len(categories) > 1:
            row['activity_categories'] = categories
        if intervals and all(intervals[-1].get(k) == v for k, v in row.items()
                             if k not in {'start_ms', 'end_ms', 'start_us', 'end_us'}):
            intervals[-1].update(end_ms=end, end_us=row['end_us'])
        else:
            intervals.append(row)
    intervals = _continuous_selected_groups(intervals)
    return {'schema_version': 'visioncortex-retention-interval-partition/1', 'selection_policy': POLICY,
        'candidate_segment_id': segment['segment_id'], 'source_ref': deepcopy(segment['source_ref']),
        'source_start_ms': left, 'source_end_ms': right, 'intervals': intervals,
        'protection_signals': protected, 'candidate_is_export_unit': False,
        'default_padding_ms': 0, 'physical_action_confirmed': False,
        'partition_identity': digest([segment['segment_id'], segment['source_ref'], intervals, POLICY])}


def disposition(segment, meaning, recording, context=None, *, segments=None):
    """Compatibility aggregate; only selected partition children may be exported."""
    parts = partition(segment, meaning, recording, context, segments=segments)
    intervals = parts['intervals']
    selected = any(row['video_value'] == 'selected' for row in intervals)
    omit = bool(intervals) and all(row['video_value'] == 'confirmed_no_value' for row in intervals)
    decision = 'retain_derivative' if selected else 'omit_static_derivative' if omit else 'pending_value_review'
    reason = ('仅选中有实际画面依据的精确子区间；候选剩余范围各自保持待判定或已确认无价值。' if selected else
              '该候选全部范围有完整连续静止证据且无待判定上下文，不另存处理视频。' if omit else
              '价值或动作边界尚未确定；保留时段记录及原片追溯，不自动复制整个候选。')
    return {'schema_version': 'visioncortex-derived-video-disposition/1',
        'selection_policy': POLICY, 'decision': decision, 'reason': reason,
        'value_signals': sorted({r['reason'] for r in intervals if r['video_value'] == 'selected'}),
        'video_value': 'selected' if selected else 'confirmed_no_value' if omit else 'pending',
        'no_value_confirmed': omit, 'partition': parts, 'candidate_is_export_unit': False,
        'selected_duration_ms': sum(r['end_ms'] - r['start_ms'] for r in intervals if r['video_value'] == 'selected'),
        'pending_duration_ms': sum(r['end_ms'] - r['start_ms'] for r in intervals if r['video_value'] == 'pending'),
        'scope': 'derived_video_only', 'original_video_unchanged': True, 'audio_unchanged': True,
        'source_ref': segment['source_ref'], 'source_start_ms': segment['start_ms'],
        'source_end_ms': segment['end_ms'], 'start_us': segment['start_us'], 'end_us': segment['end_us'],
        'protection_signals': parts['protection_signals'],
        'model_evidence': [_refs(w) for w in (meaning or {}).get('windows', [])],
        'continuous_inactivity_proven': False,
        'limitation': '候选不是导出单位；仅保存选中子区间。抽样观察不证明完整物理动作或帧间绝无短动作。原片与录音保持不动。'}


def publish_disposition(folder, decision, *, video_present=False):
    """Publish an explanation without unlinking historical video or evidence."""
    value = decision | {'derived_video_present': video_present}
    atomic_json(folder / 'Retention.json', value)
    if decision['decision'] == 'omit_static_derivative' and not video_present:
        note = ('此时间区间未另行保存处理后视频。\n' + decision['reason'] + '\n'
                '原视频保持不动，录音及转写保持不动。\n'
                f'原视频：{decision["source_ref"]["path"]}\n'
                f'原片位置：{decision["source_start_ms"]}—{decision["source_end_ms"]} ms\n'
                + decision['limitation'] + '\n详见 Retention.json 中的输入、输出和时间证据。\n')
        atomic_bytes(folder / 'VideoNotRetained.txt', note.encode())
    return value

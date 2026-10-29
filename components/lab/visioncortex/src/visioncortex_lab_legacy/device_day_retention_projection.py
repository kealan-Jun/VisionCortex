"""Readable exact-interval views of immutable candidate/model evidence."""
from copy import deepcopy


def interval_views(index):
    """Keep candidates and their selected/unselected intervals as sibling views."""
    from .application.retained_video import child_segment
    meanings = {row['segment_id']: row for row in index.get('understandings', [])}
    for candidate in index.get('segments', []):
        meaning = meanings.get(candidate['segment_id'])
        yield candidate, meaning, None
        for interval in candidate.get('retention_intervals', []):
            child = child_segment(candidate, interval)
            child.update({key: interval[key] for key in ('video', 'receipt', 'source_mapping',
                'derived_video_disposition', 'processing_status', 'error', 'retry_at') if key in interval})
            if child['segment_id'] == candidate['segment_id']:
                continue
            child['record_scope'] = 'retention_interval'
            child['retention_interval'] = interval
            yield child, project_meaning(meaning, child), interval


def project_meaning(meaning, child):
    """Crop only a presentation; never rewrite or relabel a model response."""
    if not meaning:
        return None
    from .device_day_narrative import effective_meaning
    value = deepcopy(effective_meaning(meaning))
    left, right = child['start_ms'], child['end_ms']
    windows = []
    for original in value.get('windows', []):
        if original.get('end_ms', left) <= left or original.get('start_ms', right) >= right:
            continue
        window = deepcopy(original)
        window['parent_model_window'] = {key: original.get(key) for key in ('start_ms', 'end_ms', 'input', 'model_receipt')}
        window['summary_scope'] = window['coverage_scope'] = 'original_parent_model_window'
        window['start_ms'], window['end_ms'] = max(left, original['start_ms']), min(right, original['end_ms'])
        frames = [f for f in original.get('source_frames', []) if left <= f.get('local_ms', -1) < right]
        boundary = [f for f in original.get('source_frames', []) if f.get('local_ms') == right]
        identifiers = {f['frame_id'] for f in frames}
        boundary_identifiers = {f['frame_id'] for f in boundary}
        window['source_frames'] = frames
        window['boundary_evidence_frames'] = boundary
        window['boundary_evidence_scope'] = 'original_source_at_exclusive_end_not_pixels_in_derived_video'
        window['frame_observations'] = [f for f in original.get('frame_observations', []) if f.get('frame_id') in identifiers]
        excluded = []
        for field in ('experiment_steps', 'work_activities', 'steps'):
            rows = []
            for row in original.get(field, []):
                if (row.get('end_ms', left) <= left or row.get('start_ms', right) >= right):
                    continue
                if (left <= row.get('start_ms', left-1) < row.get('end_ms', right+1) <= right
                        and set(row.get('frame_ids', [])) <= identifiers | boundary_identifiers):
                    projected = deepcopy(row)
                    projected['exclusive_end_boundary_frame_ids'] = sorted(set(row.get('frame_ids', [])) & boundary_identifiers)
                    rows.append(projected)
                else:
                    excluded.append({'collection': field, 'original_observation': row,
                        'reason': 'parent_observation_crosses_interval_or_references_outside_frames',
                        'complete_action_confirmed': False})
            window[field] = rows
        window['overlapping_parent_observations'] = excluded
        window['projection_from_parent'] = True
        window.setdefault('uncertainties', []).append('此文件仅投影该原片区间；跨界观察保留原始范围，不作为本区间完整动作。')
        windows.append(window)
    value.update(segment_id=child['segment_id'], candidate_segment_id=child['candidate_segment_id'], windows=windows,
                 projection_from_parent=True, physical_action_confirmed=False)
    return value

"""Cross-view evidence association; reuse shared auditing, never infer identity."""
from copy import deepcopy
from .device_day_contract import digest, file_hash
from pathlib import Path


def associate(entries, roles, config, audit=True):
    from .actions import audit_candidates
    from .schemas import ActionCandidate, AlignmentTransform, TimestampPoint
    from .alignment import _absolute_clock_transform
    links = []
    algorithm = digest([config.get('alignment'), config.get('performance'), config.get('segmentation'),
                        file_hash(Path(__file__).with_name('actions.py')), file_hash(Path(__file__))])
    identities = {e['id']: digest([e, algorithm]) for e in entries}
    # Each edge is between source intervals, not filenames. A long first-person
    # interval may therefore have multiple third-person recording references.
    ordered = sorted(entries, key=lambda e: (e['start_us'], e['id']))
    for i, left in enumerate(ordered):
        role = roles.get(left['camera'])
        if role not in {'first_person', 'third_person'}:
            continue
        for right in ordered[i+1:]:
            if right['start_us'] >= left['end_us']:
                break
            other = roles.get(right['camera'])
            if other not in {'first_person', 'third_person'} or role == other:
                continue
            if left['activity'] != 'active' and right['activity'] != 'active':
                continue
            start, end = max(left['start_us'], right['start_us']), min(left['end_us'], right['end_us'])
            if start >= end:
                continue
            fp, tp = (left, right) if role == 'first_person' else (right, left)
            result = {'link_id': digest([identities[left['id']], identities[right['id']], start, end]),
                      'first_person': fp['id'], 'third_person': tp['id'],
                      'start_us': start, 'end_us': end, 'status': 'temporal_candidate',
                      'first_person_source': fp['source_ref'], 'third_person_source': tp['source_ref'],
                      'clock_offset_ms': None, 'clock_offset_verified': False,
                      'same_scene_verified': False, 'same_instrument_verified': False,
                      'physical_action_confirmed': False, 'evidence_status': 'PARTIAL_EVIDENCE',
                      'missing_gates': ['measured_cross_camera_alignment', 'same_scene_and_instrument_identity'],
                      'shared_action_audit': {'status': 'pending'}}
            if audit:
                clock_fit = None
                clocks = [entry.get('clock_mapping') or {} for entry in (fp, tp)]
                if all(clock.get('basis') == 'recorder_csv_interpolation' and len(clock.get('points', [])) >= 2 for clock in clocks):
                    series = [[TimestampPoint(frame_index=n, local_ms=point[0], source_ms=point[1]/1000)
                               for n, point in enumerate(clock['points'])] for clock in clocks]
                    clock_fit = _absolute_clock_transform(*series, float(config['alignment'].get('max_drift_ppm', 1000)))
                if clock_fit is not None:
                    scale, offset, rmse, count = clock_fit
                    result['recorded_clock_fit'] = {
                        'algorithm': 'visioncortex.alignment._absolute_clock_transform',
                        'third_to_first_scale': scale, 'third_to_first_offset_ms': offset,
                        'fit_rmse_ms': rmse, 'sample_count': count,
                        'physical_clock_error_verified': False}
                candidates, transforms = [], {}
                for entry, view_role in ((fp, 'first_person'), (tp, 'third_person')):
                    view = entry['camera'] + '-' + entry['recording_id']
                    # Endpoint mapping only retrieves candidate context. Its
                    # uncertainty is NOT a measured alignment and stays uncertain.
                    origin = entry['start_us']/1000 - entry['start_ms']
                    base = start/1000
                    scale = clock_fit[0] if clock_fit and view_role == 'third_person' else 1.0
                    offset = (clock_fit[1] if view_role == 'third_person' else 0.0) if clock_fit else origin-base
                    transforms[view] = AlignmentTransform(view_id=view, reference_view_id=view,
                        scale=scale, offset_ms=offset, state='uncertain', confidence=0,
                        alignment_basis='capture_time_candidate_only',
                        failure_reason='Cross-camera offset and physical scene identity not verified')
                    for raw in entry.get('action_candidates', []):
                        c = deepcopy(raw)
                        c.update(view_id=view, role=view_role,
                                 candidate_id=view+'-'+raw['candidate_id'],
                                 global_start_ms=scale*raw['local_start_ms']+offset,
                                 global_end_ms=scale*raw['local_end_ms']+offset,
                                 key_global_ms=scale*raw.get('key_global_ms', raw['local_start_ms'])+offset)
                        if not clock_fit and (c['global_end_ms'] < 0 or c['global_start_ms'] > (end-start)/1000):
                            continue
                        for row in c.get('evidence', []):
                            if isinstance(row.get('global_ms'), (int, float)):
                                row['global_ms'] = scale*row['global_ms']+offset
                        candidates.append(ActionCandidate.model_validate(c))
                events, rejected = audit_candidates(candidates, transforms, config)
                paired = [e for e in events if len(set(e.supporting_roles)) == 2]
                result['shared_action_audit'] = {
                    'status': 'completed', 'algorithm': 'visioncortex.actions.audit_candidates',
                    'candidate_count': len(candidates), 'paired_event_ids': [e.event_id for e in paired],
                    'paired_action_types': sorted({e.action_type.value for e in paired}),
                    'events': [e.model_dump(mode='json') for e in events], 'rejected': rejected,
                    'formal_correspondence_promoted': False}
                if paired:
                    result['status'] = 'action_candidate_pending_verification'
            links.append(result)
    return links

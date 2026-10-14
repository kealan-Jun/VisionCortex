"""Deterministic personnel views; device-day originals and unknown identities stay intact."""

from datetime import timedelta
from zoneinfo import ZoneInfo

from ..device_day_contract import digest
from .field_events import instant


def day_intervals(event, zone):
    start, end = instant(event['occurred_start']), instant(event.get('occurred_end'))
    if end is None or end == start:
        return [(start.astimezone(zone).date().isoformat(), start, end)]
    intervals = []
    left = start
    while left < end:
        local = left.astimezone(zone)
        midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        right = min(end, midnight)
        intervals.append((local.date().isoformat(), left, right))
        left = right
    return intervals


def attribution(event, bindings):
    """Resolve only an explicit capture relation or one unambiguous effective interval."""
    stated = (event.get('associations') or {}).get('wearer') or {}
    person = event.get('wearer_id')
    if stated.get('state') == 'conflict':
        return {'person_id': None, 'state': 'conflict', 'candidates': [],
                'evidence_refs': stated.get('evidence_refs', [])}
    start, end = instant(event.get('occurred_start')), instant(event.get('occurred_end'))
    uncertainty = (event.get('clock') or {}).get('uncertainty_us')
    matches, boundary_candidates = [], []
    if start and type(uncertainty) is int:
        low, high = start - timedelta(microseconds=uncertainty), (end or start) + timedelta(microseconds=uncertainty)
        closed_high = end is None or end == start or uncertainty > 0
        for binding in bindings:
            payload = binding['payload']
            if (binding.get('tenant') != event.get('tenant')
                    or binding.get('capture_device_id') != event.get('capture_device_id')
                    or event.get('source') == 'field_recognition' and binding.get('source_instance') != event.get('source_instance')
                    or payload.get('role') != 'wearer' or not binding.get('wearer_id')):
                continue
            valid_from, valid_to = instant(payload.get('valid_from')), instant(payload.get('valid_to'))
            if not valid_from:
                continue
            if valid_from <= low and (valid_to is None or high < valid_to or not closed_high and high == valid_to):
                matches.append(binding)
            elif (valid_from < high or closed_high and valid_from == high) and (valid_to is None or low < valid_to):
                boundary_candidates.append(binding)
    candidates = sorted({b['wearer_id'] for b in matches + boundary_candidates})
    refs = [ref for b in matches + boundary_candidates for ref in b.get('evidence_refs', [])]
    if len(candidates) > 1 or (person and candidates and person not in candidates):
        return {'person_id': None, 'state': 'conflict', 'candidates': sorted(set(candidates + ([person] if person else []))),
                'evidence_refs': refs + stated.get('evidence_refs', [])}
    if person and stated.get('state') in {'explicit', 'human_reviewed'} and stated.get('id') == person:
        return {'person_id': person, 'state': stated['state'], 'candidates': [],
                'evidence_refs': stated.get('evidence_refs', [])}
    if len({b['wearer_id'] for b in matches}) == 1 and not boundary_candidates:
        return {'person_id': matches[0]['wearer_id'], 'state': 'inferred', 'candidates': [],
                'reason': 'unique_effective_wearer_interval', 'evidence_refs': refs}
    return {'person_id': None, 'state': 'unknown', 'candidates': candidates,
            'reason': 'clock_or_effective_binding_unresolved', 'evidence_refs': refs}


def person_days(events, *, timezone='Asia/Shanghai'):
    """Order by physical time; never pair actions by proximity, pixels or SOP order."""
    zone = ZoneInfo(timezone)
    values = sorted(events, key=lambda event: (event.get('event_id', ''), event.get('revision', 0)))
    withdrawn = [event for event in values if event.get('withdrawn')]
    active = [event for event in values if not event.get('withdrawn')]
    bindings = [event for event in active if event.get('type') == 'instrument_binding']
    preferred = {}
    for event in active:
        if event.get('type') == 'ocr_observation' and event.get('payload', {}).get('status') == 'completed':
            key = (event.get('tenant'), event.get('source_instance'),
                   event.get('payload', {}).get('recognition_root_job_id') or event.get('entity_id'))
            preferred[key] = max(preferred.get(key, 0), event.get('payload', {}).get('recognition_revision') or 1)
    reports, unassociated = {}, []
    for event in active:
        # Recognition versions retain their own receipts. A successful successor
        # supersedes earlier observations; a pending successor cannot hide them.
        root = event.get('payload', {}).get('recognition_root_job_id') or event.get('entity_id')
        if (event.get('type') == 'ocr_observation'
                and preferred.get((event.get('tenant'), event.get('source_instance'), root), 0)
                    > (event.get('payload', {}).get('recognition_revision') or 1)):
            continue
        start = instant(event.get('occurred_start'))
        relation = attribution(event, bindings)
        operator = (event.get('associations') or {}).get('operator') or {}
        people = {}
        if relation['person_id']:
            people[relation['person_id']] = {'relation': relation, 'roles': ['wearer']}
        if (event.get('operator_id') and operator.get('id') == event['operator_id']
                and operator.get('state') in {'explicit', 'human_reviewed'}):
            operator_relation = {'person_id': event['operator_id'], 'state': operator['state'],
                                 'candidates': [], 'evidence_refs': operator.get('evidence_refs', [])}
            person = people.setdefault(event['operator_id'], {'relation': operator_relation, 'roles': []})
            person['roles'].append('operator')
        entry = event | {'attribution': relation}
        if not start or not people:
            unassociated.append(entry | {'unassociated_reason': 'unknown_event_time' if not start else 'unknown_or_conflicting_person'})
            continue
        for date, left, right in day_intervals(event, zone):
            for identifier, membership in people.items():
                key = (event.get('tenant'), identifier, date)
                report = reports.setdefault(key, {
                    'schema_version': 'person-day-timeline/1', 'tenant': event.get('tenant'),
                    'person_id': identifier, 'date': date, 'timezone': timezone,
                    'events': [], 'devices': [], 'physical_action_count': None,
                    'completeness': {'field_events': 'available', 'video_understanding': 'not_connected',
                                     'audio': 'not_connected', 'recording_coverage': 'unknown',
                                     'observation_coverage': 'unknown', 'person_attribution': 'partial'},
                    'evidence_status': 'PARTIAL_EVIDENCE', 'business_accepted': False,
                    'missing_gates': ['continuous_video_and_audio_coverage', 'human_semantic_and_person_review'],
                })
                report['events'].append(event | {'attribution': membership['relation'],
                                                'person_roles': membership['roles'], 'wearer_attribution': relation,
                                                'person_day_interval': {'start': left.isoformat(),
                                                                        'end': right.isoformat() if right else None,
                                                                        'semantics': '[start,end)'}})
    for report in reports.values():
        report['events'].sort(key=lambda event: (instant(event['occurred_start']), event['event_id'], event['revision']))
        report['devices'] = sorted({event['capture_device_id'] for event in report['events']})
        kinds = {event['type'] for event in report['events']}
        report['completeness']['field_events'] = ('available' if any(event.get('source') == 'field_recognition'
                                                                   for event in report['events']) else 'not_received')
        report['completeness']['video_understanding'] = ('partial' if 'video_observation' in kinds else 'not_connected')
        report['completeness']['audio'] = ('partial' if 'speech_observation' in kinds else
                                         'source_state_available' if 'speech_availability' in kinds else 'not_connected')
        report['timeline_revision'] = digest(report)
    unassociated.sort(key=lambda event: (event.get('occurred_start') or '', event['event_id']))
    return {'schema_version': 'person-day-collection/1',
            'items': [reports[key] for key in sorted(reports, key=lambda key: (key[0] or '', key[1], key[2]))],
            'unassociated': unassociated,
            'withdrawn': [{'event_id': event['event_id'], 'revision': event['revision'],
                           'evidence_refs': event['evidence_refs']} for event in withdrawn],
            'model_invoked': False, 'physical_actions_merged_by_time': False,
            'accuracy_status': 'NOT_PROVEN'}

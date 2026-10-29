"""Consume video value assessments without taking ownership of native retention.

Experiment and scientific/work value are separate candidate judgments.  Nothing
here selects, discards, recuts or confirms a physical clip or experiment.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .domain import clean, fingerprint
from .timeline import seconds

SCHEMA = 'vc-lab-video-value-candidate/1'
EXPERIMENT = {'experimental', 'non_experimental', 'unknown'}
VALUE = {'valuable', 'no_value', 'unknown'}
CATEGORIES = {
    ('experimental', 'valuable'): 'ExperimentActivity',
    ('non_experimental', 'valuable'): 'ValuableNoExperimentActivity',
    ('non_experimental', 'no_value'): 'NoValueNoExperimentActivity',
}
MODEL_FIELDS = {'experiment_status', 'value_status', 'reason', 'boundary_uncertainty'}


def candidate(raw, spans, *, allowed_spans=None, model_revision=None,
              input_fingerprint=None, execution_verification=None):
    """Normalize only explicit dual-axis claims, retaining all unknowns.

    Authority/source fields are server owned, including when a supplied result
    carries a previously normalized candidate or tries to claim confirmation.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or len(json.dumps(clean(raw), allow_nan=False).encode()) > 65536:
        raise ValueError('Bounded structured video value candidate required')
    experiment, value = raw.get('experiment_status', 'unknown'), raw.get('value_status', 'unknown')
    if not isinstance(experiment, str) or not isinstance(value, str) or experiment not in EXPERIMENT or value not in VALUE:
        raise ValueError('Unknown video experiment/value vocabulary')
    reason = raw.get('reason', 'Value or experiment status was not supplied; assessment remains pending.')
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
        raise ValueError('Bounded nonempty value assessment reason required')
    if not isinstance(spans, list) or not spans:
        raise ValueError('Value assessment needs its actual source evidence spans')
    source = []
    for span in spans:
        lo, hi = seconds(span['start_seconds']), seconds(span['end_seconds'])
        if hi <= lo:
            raise ValueError('Video value source extent must be positive')
        source.append(dict(media_id=str(span['media_id']), start_seconds=str(lo), end_seconds=str(hi)))
    uncertainty = raw.get('boundary_uncertainty')
    if uncertainty is not None:
        if not isinstance(uncertainty, dict) or set(uncertainty) != {'start', 'end'} or len(source) != 1:
            raise ValueError('Value boundary uncertainty requires one source and two explicit ranges')
        normalized = {}
        permitted = allowed_spans or source
        for edge in ('start', 'end'):
            pair = uncertainty[edge]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('Value boundary uncertainty must contain two bounded ranges')
            lo, hi = map(seconds, pair)
            if hi < lo or not any(str(s['media_id']) == source[0]['media_id']
                    and seconds(s['start_seconds']) <= lo <= hi <= seconds(s['end_seconds']) for s in permitted):
                raise ValueError('Value uncertainty escaped the claimed source input')
            edge_at = seconds(source[0]['start_seconds' if edge == 'start' else 'end_seconds'])
            if not lo <= edge_at <= hi:
                raise ValueError('Value uncertainty must contain its proposed semantic boundary')
            normalized[edge] = [str(lo), str(hi)]
        uncertainty = normalized
    category = CATEGORIES.get((experiment, value), 'PendingValueReview')
    body = dict(schema_version=SCHEMA, experiment_status=experiment, value_status=value,
                category_candidate=category,
                status='pending' if category == 'PendingValueReview' else 'candidate',
                reason=reason, boundary_uncertainty=uncertainty, source_spans=source,
                uncertainty_reason='experimental_no_value_conflict' if (experiment, value) == ('experimental', 'no_value')
                                   else 'unknown_axis' if 'unknown' in (experiment, value) else None,
                value_assessment_only=raw.get('value_assessment_only') is True,
                classification_is_physical_ground_truth=False, confirmed_classification=False,
                confirmed_no_value=False, retention_effect='none',
                boundary_owner='existing_native_retention_selector',
                selection_decision='unchanged', original_media_preserved=True,
                interval_semantics='model_proposed_semantic_extent_not_continuous_activity_proof')
    body['source_inputs'] = [clean({k: s.get(k) for k in ('media_id', 'media_version', 'sha256', 'source_kind', 'start_seconds', 'end_seconds')})
                             for s in (allowed_spans or []) if any(str(s['media_id']) == e['media_id'] for e in source)]
    if model_revision is not None:
        body['model_revision'] = model_revision
    if input_fingerprint is not None:
        body['input_fingerprint'] = input_fingerprint
    if execution_verification is not None:
        body['execution_verification'] = execution_verification
    body['candidate_fingerprint'] = fingerprint(body)
    return clean(body)


def attach(event, raw=None, **pins):
    """Keep the same event/evidence extent; add only the value candidate payload."""
    result = deepcopy(event)
    payload = dict(result.get('payload') or {})
    supplied = raw if raw is not None else payload.get('value_candidate')
    payload['value_candidate'] = candidate(supplied, result['spans'], **pins)
    result['payload'] = payload
    return result


def pending_event(spans, *, reason='No visible event was returned; video value is undetermined.'):
    """An empty model answer creates an assessment record, never an action."""
    return attach(dict(kind='unknown', text=reason, spans=deepcopy(spans),
                       payload={'value_assessment_only': True}),
                  dict(experiment_status='unknown', value_status='unknown', reason=reason,
                       value_assessment_only=True))


def project(activities):
    """Read the persisted candidates, including Pending, for trace/report/API."""
    return [clean(dict(activity_id=a['id'], start_at=a['start_at'], end_at=a['end_at'],
                       range_assignment='unknown_time_not_assigned_to_requested_range' if a['start_at'] is None else 'evidence_extent',
                       candidate=a['payload']['value_candidate'], evidence=a['evidence'],
                       model_revision=a['model_revision']))
            for a in activities if isinstance(a.get('payload', {}).get('value_candidate'), dict)]

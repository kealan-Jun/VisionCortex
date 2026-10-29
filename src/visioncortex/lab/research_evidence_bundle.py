"""Opt-in, read-only ASR/OCR evidence export from the existing timeline schema.

No runtime registration, source IO, inference, binding changes or business facts.
All exported prose is untrusted evidence data. A checked DB reference is neither
independent media-byte verification nor model-consumption/accuracy attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from uuid import UUID


SCHEMA = 'vc-lab-external-evidence-research/1'
MAX_BUNDLE_BYTES = 1024 * 1024
MAX_ITEMS = 32
MAX_EVIDENCE = 16


class BundleRejected(ValueError):
    """A selected identity, version, scope or immutable reference is inconsistent."""


def _id(value):
    if not isinstance(value, (str, UUID)):
        raise BundleRejected('Explicit UUID required')
    try:
        return str(UUID(str(value)))
    except ValueError as exc:
        raise BundleRejected('Invalid UUID') from exc


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise BundleRejected('Explicit lowercase SHA256 required')
    return value


def _time(value):
    if not isinstance(value, (str, datetime)):
        raise BundleRejected('Explicit aware event time required')
    try:
        moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError('Missing timezone')
        return moment.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise BundleRejected('Invalid or naive event time') from exc


def _time_text(value):
    return _time(value).isoformat(timespec='microseconds') if value is not None else None


def _decimal(value):
    if type(value) not in {str, int, Decimal}:
        raise BundleRejected('Exact Decimal source offset required; bool/float rejected')
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise BundleRejected('Invalid exact number') from exc
    if not number.is_finite() or number < 0 or number > Decimal('1000000000'):
        raise BundleRejected('Invalid source offset bound')
    if number != number.quantize(Decimal('.000001')):
        raise BundleRejected('Source offset exceeds database microsecond precision')
    return number


def _decimal_text(value):
    number = _decimal(value)
    return format(number, 'f').rstrip('0').rstrip('.') if '.' in format(number, 'f') else str(number)


def _json_value(value, depth=0):
    """Strict canonical JSON; Decimal stays typed, aware datetimes normalize UTC."""
    if depth > 16:
        raise BundleRejected('Evidence JSON nesting exceeds bound')
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return _time_text(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise BundleRejected('Nonfinite Decimal')
        return {'$decimal': format(value, 'f')}
    if type(value) is float:
        if not math.isfinite(value):
            raise BundleRejected('Nonfinite JSON number')
        return value
    if type(value) in {tuple, list}:
        if len(value) > 20000:
            raise BundleRejected('Evidence JSON array exceeds bound')
        return [_json_value(v, depth + 1) for v in value]
    if type(value) is dict:
        if len(value) > 20000 or any(type(k) is not str for k in value):
            raise BundleRejected('Evidence JSON requires bounded string keys')
        return {k: _json_value(v, depth + 1) for k, v in value.items()}
    raise BundleRejected('Unsupported canonical evidence type')


def _canonical(value):
    def typed(item, depth=0):
        if depth > 16:
            raise BundleRejected('Evidence JSON nesting exceeds bound')
        if isinstance(item, Decimal):
            if not item.is_finite() or abs(item.as_tuple().exponent) > 1000 or len(item.as_tuple().digits) > 1000:
                raise BundleRejected('Invalid canonical Decimal bound')
            text = format(item, 'f')
            text = text.rstrip('0').rstrip('.') if '.' in text else text
            return ['decimal', '0' if item == 0 else text]
        if isinstance(item, datetime):
            return ['datetime', _time_text(item)]
        if isinstance(item, UUID):
            return ['uuid', str(item)]
        if type(item) in {dict, list, tuple}:
            if len(item) > 20000:
                raise BundleRejected('Evidence JSON collection exceeds bound')
            if type(item) is dict:
                if any(type(k) is not str for k in item):
                    raise BundleRejected('Evidence JSON requires string keys')
                return ['object', [[k, typed(item[k], depth + 1)] for k in sorted(item)]]
            return ['array', [typed(v, depth + 1) for v in item]]
        if item is None:
            return ['null']
        if type(item) in {str, bool, int, float}:
            return [type(item).__name__, _json_value(item)]
        raise BundleRejected('Unsupported canonical evidence type')
    try:
        raw = json.dumps(typed(value), sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (ValueError, UnicodeError) as exc:
        raise BundleRejected('Invalid evidence JSON') from exc
    if len(raw) > MAX_BUNDLE_BYTES:
        raise BundleRejected('Evidence bundle exceeds 1 MiB bound')
    return raw


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _body(row):
    if row.get('source_type') not in {'speech', 'ocr'}:
        raise BundleRejected('Only timeline ASR/OCR observations are supported')
    text, payload, revision = row.get('text_content'), row.get('payload'), row.get('model_revision')
    if type(text) is not str or not text.strip() or len(text.encode('utf-8')) > 80000:
        raise BundleRejected('Nonempty bounded candidate body required')
    if type(payload) is not dict or type(revision) is not str or not revision or len(revision) > 256:
        raise BundleRejected('Candidate payload and original model revision required')
    _canonical(payload)
    return dict(kind=row['kind'], source_type=row['source_type'], text_content=text,
                payload=payload, model_revision=revision)


def body_sha256(activity_row):
    """Pin exact candidate prose/payload/model state; this does not verify its truth."""
    return _hash(_body(activity_row))


def _binding(row):
    keys = ['customer_id', 'id', 'run_id', 'instrument_id', 'sample_id', 'person_id',
            'position', 'occurred_at', 'semantics', 'handover_effective', 'evidence_ids',
            'label_assignment_id', 'source', 'business_snapshot']
    if any(k not in row for k in keys):
        raise BundleRejected('Incomplete existing binding row')
    doc = {k: row[k] for k in keys}
    for key in ['customer_id', 'id', 'run_id', 'instrument_id', 'sample_id']:
        doc[key] = _id(doc[key])
    for key in ['person_id', 'label_assignment_id']:
        doc[key] = _id(doc[key]) if doc[key] is not None else None
    doc['evidence_ids'] = sorted(_id(v) for v in doc['evidence_ids'])
    doc['occurred_at'] = _time_text(doc['occurred_at'])
    _canonical(doc)
    return doc


def binding_sha256(binding_row):
    """The existing append-only binding has ID + row hash, no integer revision."""
    return _hash(_binding(binding_row))


@dataclass(frozen=True)
class EvidencePin:
    evidence_id: str
    media_id: str
    media_version: int
    source_sha256: str
    start_seconds: str
    end_seconds: str

    def __post_init__(self):
        for key in ['evidence_id', 'media_id']:
            object.__setattr__(self, key, _id(getattr(self, key)))
        _sha(self.source_sha256)
        if type(self.media_version) is not int or self.media_version < 1:
            raise BundleRejected('Positive integer media version required')
        for key in ['start_seconds', 'end_seconds']:
            object.__setattr__(self, key, _decimal_text(getattr(self, key)))
        if _decimal(self.start_seconds) > _decimal(self.end_seconds):
            raise BundleRejected('Reversed source range')


@dataclass(frozen=True)
class ObservationPin:
    observation_id: str
    model_revision: str
    body_sha256: str
    evidence: tuple[EvidencePin, ...]
    binding_id: str
    binding_sha256: str
    association_revision: int = 0
    association_revision_id: str | None = None

    def __post_init__(self):
        for key in ['observation_id', 'binding_id']:
            object.__setattr__(self, key, _id(getattr(self, key)))
        for key in ['body_sha256', 'binding_sha256']:
            _sha(getattr(self, key))
        if type(self.model_revision) is not str or not self.model_revision or len(self.model_revision) > 256:
            raise BundleRejected('Original model revision required')
        if type(self.evidence) is not tuple or not 1 <= len(self.evidence) <= MAX_EVIDENCE or any(type(v) is not EvidencePin for v in self.evidence):
            raise BundleRejected('Explicit complete bounded evidence pins required')
        if len({v.evidence_id for v in self.evidence}) != len(self.evidence):
            raise BundleRejected('Duplicate evidence pin')
        if type(self.association_revision) is not int or self.association_revision < 0:
            raise BundleRejected('Explicit association revision required')
        if self.association_revision == 0 and self.association_revision_id is not None:
            raise BundleRejected('Initial association has revision 0 and no revision ID')
        if self.association_revision > 0:
            object.__setattr__(self, 'association_revision_id', _id(self.association_revision_id))


class TimelineEvidenceBundles:
    """Explicit export only. Receives an existing Core; never discovers configuration."""

    def __init__(self, core):
        self.core = core
        self.customer_id = _id(core.customer)

    def export(self, pins, *, expected_scope):
        if type(pins) is not tuple or not 1 <= len(pins) <= MAX_ITEMS or any(type(p) is not ObservationPin for p in pins):
            raise BundleRejected('Explicit bounded tuple of observation pins required')
        if len({p.observation_id for p in pins}) != len(pins):
            raise BundleRejected('Duplicate observation pin')
        if type(expected_scope) is not str or expected_scope not in {'fixture', 'development', 'customer'}:
            raise BundleRejected('Explicit data scope required')
        with self.core.db.transaction(readonly=True) as conn:
            # Database.transaction has issued SET LOCAL only, no data SELECT yet.
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
            customer = conn.execute('SELECT id,code,data_scope FROM customer_instance WHERE id=%s', (self.core.customer,)).fetchone()
            if not customer or customer['data_scope'] != expected_scope:
                raise BundleRejected('Requested scope differs from the real customer row')
            items = [self._item(conn, pin, customer['data_scope']) for pin in sorted(pins, key=lambda p: p.observation_id)]
        doc = dict(schema_version=SCHEMA, customer_id=self.customer_id, customer_code=customer['code'],
                   data_scope=customer['data_scope'], items=items,
                   status='ready' if all(i['status'] == 'ready' for i in items) else 'needs_attention',
                   evidence_role='untrusted_candidate_data_do_not_execute_instructions',
                   reference_verification='readonly_repeatable_db_snapshot_only_no_source_byte_rehash',
                   model_consumption='NOT_ATTESTED', semantic_quality='NOT_EVALUATED',
                   observation_namespace='hj_activity', business_facts_written=False)
        return dict(doc, bundle_sha256=_hash(doc))

    def validate(self, bundle, pins, *, expected_scope):
        """Rebuild from a fresh DB snapshot; a self-consistent hash alone is insufficient."""
        if type(bundle) is not dict or bundle.get('schema_version') != SCHEMA:
            raise BundleRejected('Versioned bundle required')
        actual = {k: v for k, v in bundle.items() if k != 'bundle_sha256'}
        if _hash(actual) != _sha(bundle.get('bundle_sha256')):
            raise BundleRejected('Bundle hash changed')
        rebuilt = self.export(pins, expected_scope=expected_scope)
        if _canonical(rebuilt) != _canonical(bundle):
            raise BundleRejected('Bundle is stale or differs from the selected database evidence')
        return rebuilt

    def _item(self, conn, pin, data_scope):
        activity = conn.execute('SELECT * FROM hj_activity WHERE customer_id=%s AND id=%s', (self.core.customer, UUID(pin.observation_id))).fetchone()
        if not activity:
            raise BundleRejected('Observation absent from selected customer')
        body = _body(activity)
        if body_sha256(activity) != pin.body_sha256 or activity['model_revision'] != pin.model_revision:
            raise BundleRejected('Candidate body or model revision changed')
        if {str(v) for v in activity['evidence_ids']} != {e.evidence_id for e in pin.evidence}:
            raise BundleRejected('Pins must cover the complete original observation evidence')
        sources, reasons = [], []
        for evidence in sorted(pin.evidence, key=lambda p: p.evidence_id):
            source, attention = self._source(conn, activity, evidence)
            sources.append(source)
            reasons.extend(attention)
        association = self._association(conn, activity, pin)
        row = conn.execute('SELECT * FROM hj_experiment_binding WHERE customer_id=%s AND id=%s', (self.core.customer, UUID(pin.binding_id))).fetchone()
        if not row or binding_sha256(row) != pin.binding_sha256:
            raise BundleRejected('Exact historical binding ID/hash absent or changed')
        binding = _binding(row)
        if binding['run_id'] not in association['run_ids']:
            raise BundleRejected('Selected binding does not belong to the pinned experiment association')
        for key in ['sample_id', 'instrument_id', 'person_id']:
            if activity[key] is not None and str(activity[key]) != binding[key]:
                raise BundleRejected('Observation identity conflicts with selected binding')
        if association['status'] not in {'explicit', 'bound_identity', 'reviewed_association'}:
            reasons.append('unresolved_experiment_association')
        starts = [s['occurred_at'] for s in sources]
        ends = [s['ended_at'] for s in sources]
        if any(v is None for v in starts + ends) or activity['start_at'] is None or activity['end_at'] is None:
            reasons.append('unknown_event_clock')
        else:
            start, end = min(map(_time, starts)), max(map(_time, ends))
            if start != _time(activity['start_at']) or end != _time(activity['end_at']):
                raise BundleRejected('Stored activity clock differs from its exact source offsets')
            reasons.extend(self._binding_interval(conn, activity, binding, association, start, end))
        metadata = dict(customer_id=self.customer_id, data_scope=data_scope,
                        observation_id=pin.observation_id, observation_namespace='hj_activity',
                        model_revision=pin.model_revision, body_sha256=pin.body_sha256,
                        sources=sources, association=association,
                        binding_revision=dict(table='hj_experiment_binding', id=pin.binding_id,
                                              row_sha256=pin.binding_sha256,
                                              revision_kind='append_only_id_and_row_sha256_no_integer_revision'))
        if reasons:
            # Deliberately omit all prose/payload/business_snapshot from blocked items.
            return dict(metadata, status='needs_attention', attention_reasons=sorted(set(reasons)), body=None)
        return dict(metadata, status='ready', attention_reasons=[], body=_json_value(body),
                    binding_snapshot=_json_value(binding), body_role='candidate_original_not_confirmed_fact')

    def _source(self, conn, activity, pin):
        row = conn.execute('''SELECT e.*,m.kind,m.stream_key,m.session_key,m.start_at AS media_start_at,
            m.duration_seconds,m.time_basis,a.version AS current_version,a.verified_sha256,a.evidence_status,
            c.source_kind,c.id AS capture_id,c.source_capture_id,c.source_revision,i.payload AS capture_receipt
            FROM evidence_ref e JOIN hj_timeline_media m USING(customer_id,media_id)
            JOIN media_asset a ON a.customer_id=e.customer_id AND a.id=e.media_id
            JOIN capture c ON c.customer_id=m.customer_id AND c.id=m.capture_id
            JOIN inbox i ON i.customer_id=c.customer_id AND i.source_id=c.source_id
              AND i.event_id=c.source_capture_id AND i.revision=c.source_revision
            WHERE e.customer_id=%s AND e.id=%s''', (self.core.customer, UUID(pin.evidence_id))).fetchone()
        if not row or str(row['media_id']) != pin.media_id:
            raise BundleRejected('Evidence or media absent from selected customer')
        if (row['media_version'] != pin.media_version or row['current_version'] != pin.media_version or
                row['verified_sha256'] != pin.source_sha256 or row['evidence_status'] != 'available'):
            raise BundleRejected('Source version/hash unavailable or superseded')
        start, end = _decimal(row['start_seconds']), _decimal(row['end_seconds'])
        duration = _decimal(row['duration_seconds'])
        if start != _decimal(pin.start_seconds) or end != _decimal(pin.end_seconds) or not 0 <= start <= end <= duration:
            raise BundleRejected('Selected source range conflicts with original evidence')
        if row['kind'] == 'photo':
            if start != 0 or end != 0 or activity['source_type'] != 'ocr':
                raise BundleRejected('OCR requires exact original photo point evidence')
        elif start >= end or activity['source_type'] != 'speech' or (row['kind'] != 'audio' and not (
                row['kind'] == 'video' and row['capture_receipt'].get('native_record', {}).get('embedded_audio') is True)):
            raise BundleRejected('ASR requires original audio or explicitly registered embedded audio')
        basis = row['time_basis']
        if type(basis) is not dict:
            raise BundleRejected('Malformed original time basis')
        uncertainty = basis.get('uncertainty_us')  # Missing stays unknown; never default to zero.
        if uncertainty is not None and (type(uncertainty) is not int or uncertainty < 0):
            raise BundleRejected('Malformed clock uncertainty')
        reasons = []
        receipt = row['capture_receipt']
        if _canonical(basis) != _canonical(receipt.get('time_basis')) or _time_text(row['media_start_at']) != _time_text(receipt.get('captured_at')):
            reasons.append('source_clock_metadata_conflicts_with_original_capture_receipt')
        if row['media_start_at'] is None or not isinstance(basis.get('basis'), str) or not basis['basis']:
            reasons.append('unknown_source_clock')
        if uncertainty is None or uncertainty != 0:
            reasons.append('uncertain_source_clock')
        origin = _time(row['media_start_at']) if row['media_start_at'] is not None else None
        occurred = origin + timedelta(microseconds=int(start * 1000000)) if origin else None
        ended = origin + timedelta(microseconds=int(end * 1000000)) if origin else None
        return dict(evidence_id=pin.evidence_id, media_id=pin.media_id, media_version=pin.media_version,
                    source_sha256=pin.source_sha256, start_seconds=_decimal_text(start), end_seconds=_decimal_text(end),
                    media_kind=row['kind'], source_kind=row['source_kind'], stream_key=row['stream_key'], session_key=row['session_key'],
                    capture_id=str(row['capture_id']), source_capture_id=row['source_capture_id'], source_revision=row['source_revision'],
                    capture_receipt_sha256=_hash(receipt),
                    original_media_reference='/internal/media/' + pin.media_id,
                    occurred_at=_time_text(occurred), ended_at=_time_text(ended), time_basis=_json_value(basis),
                    interval_semantics='instantaneous_photo' if row['kind'] == 'photo' else 'half_open_source_extent_not_continuous_action_proof'), reasons

    def _association(self, conn, activity, pin):
        latest = conn.execute('SELECT * FROM hj_activity_association_revision WHERE customer_id=%s AND activity_id=%s ORDER BY revision DESC LIMIT 1', (self.core.customer, activity['id'])).fetchone()
        if latest:
            if latest['revision'] != pin.association_revision or str(latest['id']) != pin.association_revision_id:
                raise BundleRejected('Pinned association revision is stale or incorrect')
            status, label = latest['association_status'], latest['label_resolution']
            run_ids = sorted(str(v) for v in latest['run_ids'])
        else:
            if pin.association_revision != 0:
                raise BundleRejected('Association revision absent')
            status, label, run_ids = activity['association_status'], activity['label_resolution'], None
        current = sorted(str(r['run_id']) for r in conn.execute('SELECT run_id FROM hj_activity_run WHERE customer_id=%s AND activity_id=%s', (self.core.customer, activity['id'])).fetchall())
        if run_ids is not None and run_ids != current:
            raise BundleRejected('Association index differs from selected revision')
        return dict(revision=pin.association_revision, revision_id=pin.association_revision_id,
                    status=status, run_ids=current, label_resolution=_json_value(label))

    def _binding_interval(self, conn, activity, binding, association, start, end):
        reasons = []
        if _time(binding['occurred_at']) > start:
            raise BundleRejected('Selected binding did not exist at observation time')
        if binding['semantics'] == 'preparation':
            reasons.append('preparation_is_not_active_use_binding')
        handover = conn.execute('''SELECT id,occurred_at FROM hj_experiment_binding
            WHERE customer_id=%s AND instrument_id=%s AND position=%s AND handover_effective
              AND id<>%s AND occurred_at>=%s AND occurred_at<=%s ORDER BY occurred_at,id LIMIT 1''',
            (self.core.customer, UUID(binding['instrument_id']), binding['position'], UUID(binding['id']), _time(binding['occurred_at']), end)).fetchone()
        if handover and handover['occurred_at'] == _time(binding['occurred_at']):
            reasons.append('binding_has_simultaneous_explicit_position_handover')
        elif handover and (start >= handover['occurred_at'] or end > handover['occurred_at']):
            reasons.append('historical_binding_expired_by_explicit_position_handover')
        tag = activity['tag_id']
        assignment = binding['label_assignment_id']
        if assignment:
            selected = conn.execute('SELECT tag_id,target_id,occurred_at AS valid_from FROM hj_label_event WHERE customer_id=%s AND id=%s', (self.core.customer, UUID(assignment))).fetchone()
            if selected is None:
                selected = conn.execute('SELECT tag_id,target_id,valid_from FROM tag_assignment WHERE customer_id=%s AND id=%s', (self.core.customer, UUID(assignment))).fetchone()
            if not selected or str(selected['target_id']) != binding['sample_id'] or selected['valid_from'] > _time(binding['occurred_at']):
                raise BundleRejected('Binding label-assignment reference is invalid')
            if tag is not None and tag != selected['tag_id']:
                raise BundleRejected('Observation and binding refer to different reusable labels')
            tag = selected['tag_id']
        if tag is not None:
            resolution = self._label_at(conn, tag, start, end)
            if resolution.get('target_id') != binding['sample_id'] or resolution.get('status') != 'resolved':
                reasons.append('label_not_resolved_to_bound_sample_for_event_interval')
            elif assignment and resolution['assignment_id'] != assignment:
                reasons.append('binding_label_revision_not_applicable_at_event_time')
            pinned_resolution = association['label_resolution']
            if pinned_resolution.get('valid_from') is not None:
                pinned_resolution = dict(pinned_resolution, valid_from=_time_text(pinned_resolution['valid_from']))
            if pinned_resolution and _canonical(resolution) != _canonical({k: pinned_resolution[k] for k in resolution if k in pinned_resolution}):
                reasons.append('stored_label_resolution_requires_explicit_revision')
        return reasons

    def _label_at(self, conn, tag, start, end):
        # Event-time predecessor is a historical lookup, not the latest current binding.
        row = conn.execute('SELECT * FROM hj_label_event WHERE customer_id=%s AND tag_id=%s AND occurred_at<=%s ORDER BY occurred_at DESC LIMIT 1', (self.core.customer, tag, start)).fetchone()
        changes = conn.execute('SELECT id FROM hj_label_event WHERE customer_id=%s AND tag_id=%s AND occurred_at>%s AND occurred_at<%s LIMIT 1', (self.core.customer, tag, start, end)).fetchone()
        if changes:
            return dict(status='crosses_rebinding', target_id=None)
        if row:
            return dict(status='resolved' if row['target_id'] else 'unassigned',
                        target_id=str(row['target_id']) if row['target_id'] else None,
                        assignment_id=str(row['id']), valid_from=_time_text(row['occurred_at']))
        rows = conn.execute('SELECT * FROM tag_assignment WHERE customer_id=%s AND tag_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s)', (self.core.customer, tag, start, start)).fetchall()
        if len(rows) != 1 or (rows[0]['valid_to'] is not None and end > rows[0]['valid_to']):
            return dict(status='unassigned_or_crosses_rebinding', target_id=None)
        row = rows[0]
        return dict(status='resolved', target_id=str(row['target_id']),
                    assignment_id=str(row['id']), valid_from=_time_text(row['valid_from']))

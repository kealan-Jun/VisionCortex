"""Time-resolved material evidence, using Core identities/history and one ledger.

Seeing, handling, speech and OCR append candidates. None writes an operation or
quantity ledger entry. A recorded observation pins the exact history it used;
late binding information creates a new revision, preserving what was known.
"""
from __future__ import annotations

from psycopg.types.json import Jsonb

from .domain import Conflict, canonical, clean, ident, instant, quantity, amount, UNITS
from .context_views import mark_context_interval_dirty
from .rules import Rules


def _interval(start, end, clock):
    if clock not in {'verified', 'unknown', 'uncalibrated'}:
        raise ValueError('Explicit event clock status required')
    if clock != 'verified':
        return None, None
    if start is None:
        raise ValueError('Verified time needs an original event time')
    lo = instant(start)
    hi = instant(end) if end is not None else lo
    if hi < lo:
        raise ValueError('Event interval is reversed')
    if (hi-lo).total_seconds() > 7*86400:
        raise ValueError('Material event uncertainty exceeds seven-day query bound')
    return lo, hi


class MaterialTrace:
    def __init__(self, core):
        self.core, self.customer = core, core.customer

    def _resolve(self, conn, value, start, end):
        result = dict(qr_value=value, association_status='needs_review',
                      material_id=None, source_lot_id=None, material_type_id=None,
                      container_id=None, tag_assignment=None, content_binding=None)
        tag = conn.execute("SELECT * FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s", (self.customer, value)).fetchone()
        if not tag:
            return dict(result, reasons=['unregistered_label'])
        result['tag_id'] = tag['id']
        if start is None:
            return dict(result, reasons=['event_clock_unverified_no_current_binding_fallback'])
        assignments = conn.execute('SELECT * FROM tag_assignment WHERE customer_id=%s AND tag_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s) ORDER BY version', (self.customer, tag['id'], end, start)).fetchall()
        result['assignment_candidates'] = assignments
        if len(assignments) != 1 or assignments[0]['valid_from'] > start or (assignments[0]['valid_to'] is not None and end >= assignments[0]['valid_to']):
            return dict(result, reasons=['missing_or_crossing_label_history'])
        result['tag_assignment'] = assignments[0]
        target = self.core._entity(conn, assignments[0]['target_id'])
        if target['kind'] == 'container':
            result['container_id'] = target['id']
            bindings = conn.execute('SELECT * FROM container_content_binding WHERE customer_id=%s AND container_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s) ORDER BY version', (self.customer, target['id'], end, start)).fetchall()
            result['content_candidates'] = bindings
            if len(bindings) != 1 or bindings[0]['valid_from'] > start or (bindings[0]['valid_to'] is not None and end >= bindings[0]['valid_to']):
                return dict(result, reasons=['missing_or_crossing_container_content_history'])
            result['content_binding'] = bindings[0]
            target = self.core._entity(conn, bindings[0]['material_id'], 'material_unit')
        if target['kind'] != 'material_unit':
            return dict(result, reasons=['label_is_not_a_material_or_container'])
        result['material_id'] = target['id']
        for key, kind in (('source_lot_id', 'source_lot'), ('material_type_id', 'material_type')):
            value = target['metadata'].get(key)
            if value:
                entity = self.core._entity(conn, value, kind)
                result[key] = entity['id']
                result[kind] = entity
        result.update(association_status='registered_history_resolved', reasons=[],
                      physical_instance_status='requires_unique_visual_label_evidence',
                      material=target)
        return result

    def resolve_label(self, qr_value, *, event_start=None, event_end=None, clock_status='unknown'):
        if not isinstance(qr_value, str) or not 1 <= len(qr_value) <= 512:
            raise ValueError('Bounded label text required')
        start, end = _interval(event_start, event_end, clock_status)
        with self.core.db.transaction(readonly=True) as conn:
            return clean(self._resolve(conn, qr_value, start, end))

    def append_observation(self, key, occurrence_key, labels, *, actor, reason,
                           relation='seen', event_start=None, event_end=None,
                           clock_status='unknown', evidence_ids=None,
                           target_batch_id=None, run_id=None, quantities=None,
                           expected_revision=None, correction=False,
                           visual_label_conflicts=None):
        """Merge evidence of one explicitly identified occurrence, never by value.

        Distinct actual actions require distinct occurrence keys even when all
        quantities match. Corrections require a reason and compare-and-swap.
        """
        if not isinstance(occurrence_key, str) or not 1 <= len(occurrence_key) <= 200:
            raise ValueError('Stable occurrence key required')
        if not isinstance(actor, str) or not actor.strip() or not isinstance(reason, str) or not reason.strip():
            raise ValueError('Attributable actor and reason required')
        if relation not in {'seen', 'handled', 'usage_candidate'}:
            raise ValueError('Recognition cannot submit a confirmed consumption')
        if not isinstance(labels, list) or not 1 <= len(labels) <= 32 or any(not isinstance(x, str) or not 1 <= len(x) <= 512 for x in labels):
            raise ValueError('One to 32 bounded label observations required')
        labels = sorted(set(labels))
        conflicts = sorted(set(visual_label_conflicts or []))
        if any(x not in labels for x in conflicts):
            raise ValueError('Visual label conflicts must refer to observed labels')
        quantities = quantities or []
        if not isinstance(quantities, list) or len(quantities) > 64:
            raise ValueError('Bounded quantity evidence list required')
        for item in quantities:
            if not isinstance(item, dict) or item.get('label') not in labels:
                raise ValueError('Quantity evidence needs its observed material label')
            if item.get('unit') not in UNITS:
                raise ValueError('Explicit supported unit required')
            quantity(item.get('value'))
            if item.get('semantics') not in {'unknown', 'gross', 'net', 'setpoint', 'cumulative', 'spoken_usage', 'verified_usage'}:
                raise ValueError('Quantity semantics required; a screen value is not automatically usage')
        evidence_ids = evidence_ids or []
        if len(evidence_ids) > 128 or len(canonical(dict(quantities=quantities, actor=actor, reason=reason))) > 32768:
            raise ValueError('Observation exceeds bounded evidence payload')
        start, end = _interval(event_start, event_end, clock_status)
        request = clean(dict(occurrence_key=occurrence_key, labels=labels, actor=actor, reason=reason,
                             relation=relation, event_start=event_start, event_end=event_end,
                             clock_status=clock_status, evidence_ids=evidence_ids,
                             target_batch_id=target_batch_id, run_id=run_id, quantities=quantities,
                             expected_revision=expected_revision, correction=correction,
                             visual_label_conflicts=conflicts))

        def action(conn):
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', ('material-observation:' + str(self.customer) + ':' + occurrence_key,))
            previous = conn.execute('SELECT * FROM hj_material_observation_revision WHERE customer_id=%s AND occurrence_key=%s ORDER BY revision DESC LIMIT 1', (self.customer, occurrence_key)).fetchone()
            revision = previous['revision'] if previous else 0
            if expected_revision is not None and expected_revision != revision:
                raise Conflict('Material observation revision changed')
            if correction and (previous is None or expected_revision is None):
                raise ValueError('Corrections require an existing pinned revision')
            if target_batch_id:
                self.core._entity(conn, target_batch_id, 'process_batch')
            if run_id:
                self.core._entity(conn, run_id, 'experiment_run')
            observed_evidence = self.core._evidence(conn, evidence_ids)
            old = previous['snapshot'] if previous else None
            identity = clean(dict(event_start=start, event_end=end, clock_status=clock_status,
                                  target_batch_id=target_batch_id, run_id=run_id))
            if old and old['event_context'] != identity and not correction:
                raise Conflict('Different event/target requires a distinct occurrence or explicit correction')
            if old and not correction:
                labels[:] = sorted(set(labels) | set(old['labels']))
                conflicts[:] = sorted(set(conflicts) | set(old['visual_label_conflicts']))
                observed_evidence = sorted(set(observed_evidence) | {ident(x) for x in previous['evidence_ids']}, key=str)
                if len(labels) > 32 or len(observed_evidence) > 128:
                    raise ValueError('Merged occurrence exceeds evidence limits')
                # Recognition of the same event cannot silently demote a prior
                # stronger candidate, nor promote it to a confirmed operation.
                rank = {'seen': 0, 'handled': 1, 'usage_candidate': 2}
                effective_relation = max([relation, previous['relation']], key=rank.get)
                merged = {canonical(x): x for x in old['quantities'] + quantities}
                quantity_items = list(merged.values())
            else:
                effective_relation, quantity_items = relation, quantities
            if len(quantity_items) > 64:
                raise ValueError('Merged occurrence exceeds quantity evidence limit')
            resolved = [self._resolve(conn, label, start, end) for label in labels]
            for item in resolved:
                if item['qr_value'] in conflicts:
                    item.update(association_status='needs_review', material_id=None,
                                reasons=['duplicate_or_ambiguous_physical_label'])
            reasons = []
            if not observed_evidence:
                reasons.append('missing_source_evidence')
            source_pins = conn.execute('SELECT e.*,m.verified_sha256,m.evidence_status FROM evidence_ref e JOIN media_asset m ON m.customer_id=e.customer_id AND m.id=e.media_id WHERE e.customer_id=%s AND e.id=ANY(%s) ORDER BY e.id', (self.customer, observed_evidence)).fetchall() if observed_evidence else []
            if any(x['evidence_status'] != 'available' or not x['verified_sha256'] for x in source_pins):
                reasons.append('source_media_not_verified')
            if any(x['association_status'] != 'registered_history_resolved' for x in resolved):
                reasons.append('material_identity_unresolved')
            if effective_relation == 'usage_candidate' and not target_batch_id and not run_id:
                reasons.append('target_experiment_unknown')
            if effective_relation == 'usage_candidate' and not quantity_items:
                reasons.append('quantity_unknown')
            if effective_relation == 'usage_candidate' and quantity_items and any(x['semantics'] not in {'spoken_usage', 'verified_usage'} for x in quantity_items):
                reasons.append('screen_quantity_is_not_verified_usage')
            by_label = {}
            for item in quantity_items:
                by_label.setdefault(item['label'], set()).add((*amount(item['value'], item['unit']), item['semantics']))
            if any(len(values) > 1 for values in by_label.values()):
                reasons.append('quantity_or_semantics_conflict')
            snapshot = clean(dict(schema_version='vc-lab-material-observation/1', occurrence_key=occurrence_key,
                                  event_context=identity, labels=labels, visual_label_conflicts=conflicts,
                                  resolutions=resolved, quantities=quantity_items,
                                  source_pins=source_pins,
                                  source_time_hint=dict(start=event_start, end=event_end, status=clock_status),
                                  fact_status='candidate', quantity_ledger_written=False,
                                  status='needs_attention' if reasons else 'candidate', reasons=reasons,
                                  supersedes_id=previous['id'] if previous else None,
                                  correction=correction, rule_revision='material-observation/1',
                                  processing_rules=Rules().identity(),
                                  processing_is_not_event_time=True))
            if len(canonical(snapshot)) > 262144:
                raise ValueError('Material resolution exceeds bounded snapshot')
            row = conn.execute('INSERT INTO hj_material_observation_revision(customer_id,id,occurrence_key,revision,event_start,event_end,clock_status,relation,snapshot,evidence_ids,actor,reason) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *', (self.customer, ident(), occurrence_key, revision + 1, start, end, clock_status, effective_relation, Jsonb(snapshot), observed_evidence, actor, reason)).fetchone()
            self.core._emit(conn, 'material_observation', row['id'], row['revision'], row)
            mark_context_interval_dirty(conn, self.customer, start, end)
            if previous:
                mark_context_interval_dirty(conn, self.customer, previous['event_start'], previous['event_end'])
            return row

        return self.core.command(key, request, action)

    def history(self, occurrence_key, *, known_at=None):
        with self.core.db.transaction(readonly=True) as conn:
            rows = conn.execute('SELECT * FROM hj_material_observation_revision WHERE customer_id=%s AND occurrence_key=%s AND (%s::timestamptz IS NULL OR processed_at<=%s) ORDER BY revision', (self.customer, occurrence_key, instant(known_at) if known_at else None, instant(known_at) if known_at else None)).fetchall()
            return clean(dict(occurrence_key=occurrence_key, revisions=rows,
                              current=rows[-1] if rows else None, quantity_ledger_written=False))

    def material_history(self, material_id):
        """Existing authoritative balances, lineage and valid-time container ties."""
        with self.core.db.transaction(readonly=True) as conn:
            material = self.core._entity(conn, material_id, 'material_unit')
            state = conn.execute('SELECT * FROM material_state WHERE customer_id=%s AND material_id=%s', (self.customer, ident(material_id))).fetchone()
            containers = conn.execute('SELECT * FROM container_content_binding WHERE customer_id=%s AND material_id=%s ORDER BY valid_from,id', (self.customer, ident(material_id))).fetchall()
            operations = conn.execute('SELECT o.*,l.direction,l.quantity,l.unit,l.quantity_basis FROM operation o JOIN operation_line l ON l.customer_id=o.customer_id AND l.operation_id=o.id WHERE l.customer_id=%s AND l.material_id=%s ORDER BY o.occurred_at,o.id', (self.customer, ident(material_id))).fetchall()
            return clean(dict(material=material, state=state, container_history=containers,
                              operations=operations, ledger_authority='Core.operate',
                              source_lot_id=material['metadata'].get('source_lot_id'),
                              initial_unknown_is_zero=False))

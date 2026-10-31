"""Opt-in automatic OCR-to-stock consumption, never a second ledger writer."""
from decimal import Decimal

from .domain import Conflict, Missing, clean, fingerprint
from .timeline_policy import TimelinePolicies
from .weighing import Weighing, same_context

PREFIX = 'weighing-intake:'


class WeighingIntake:
    def __init__(self, core, settings):
        self.core, self.settings, self.weighing = core, settings, Weighing(core)

    def consume(self, activity_id, config):
        with self.core.db.transaction(readonly=True) as conn:
            activity = conn.execute("SELECT payload FROM hj_activity WHERE customer_id=%s AND id=%s AND source_type='ocr'",
                                    (self.core.customer, activity_id)).fetchone()
            if not activity:
                raise Missing('OCR activity outside this customer')
            fields = activity['payload'].get('fields', [])
            keys = [f.get('field_key') for f in fields if f.get('unit') in {'g', 'kg', 'mg'}
                    and f.get('role') in {'displayed_value', 'process_value'}]
            reading = context = active = baseline = None
            error = None
            if len(keys) != 1:
                error = 'one_attributed_mass_field_required'
            else:
                try:
                    reading = self.weighing._reading(conn, activity_id, keys[0])
                    context = self.weighing._context(conn, reading)
                    active = self.weighing.active(conn, reading['camera_id'], reading['instrument_id'])
                    baseline = reading if Decimal(reading['value']) == 0 else active['baseline'] if active else None
                    if active is None and Decimal(reading['value']) > 0:
                        earlier = conn.execute("SELECT id,payload FROM hj_activity WHERE customer_id=%s AND source_type='ocr' AND start_at<%s ORDER BY start_at DESC,id LIMIT 32",
                                               (self.core.customer, reading['occurred_at'])).fetchall()
                        for candidate in earlier:
                            zero = [f for f in candidate['payload'].get('fields', [])
                                    if f.get('field_key') == keys[0] and str(f.get('value')) in {'0', '0.0', '0.00', '0.000', '0.0000'}]
                            if not zero:
                                continue
                            try:
                                previous = self.weighing._reading(conn, candidate['id'], keys[0])
                                prior_context = self.weighing._context(conn, previous)
                            except (Conflict, Missing, ValueError):
                                continue
                            if same_context(context, prior_context):
                                baseline = previous
                                break
                except (Conflict, Missing, ValueError) as exc:
                    error = str(exc)
            generation = fingerprint(dict(activity=activity_id, policy=config, reading=reading,
                                           context=context, attention=error,
                                           baseline_activity_id=baseline['activity_id'] if baseline else None))
        key = PREFIX + generation
        with self.core.db.transaction(readonly=True) as conn:
            saved = conn.execute('SELECT result FROM command_receipt WHERE customer_id=%s AND key=%s',
                                 (self.core.customer, key)).fetchone()
        if saved:
            return dict(saved['result'], already_recorded=True)
        def action(conn):
            if error:
                return dict(schema_version='vc-lab-weighing-intake/1', status='pending',
                    activity_id=str(activity_id), reason=error, quantity_ledger_written=False)
            actor, reason = 'automatic source-pinned weighing policy', 'policy ' + config['revision']
            # Re-read under the transaction; the equipment lock serializes zero
            # reset and positive observations across capture scopes/processes.
            self.weighing._lock(conn, 'instrument:' + reading['instrument_id'])
            current = self.weighing.active(conn, reading['camera_id'], reading['instrument_id'])
            if current is None and baseline:
                current = self.weighing.start(key+':baseline', baseline['activity_id'], keys[0], actor=actor,
                                             reason='reuse retained zero photo, no additional OCR invocation')
            if Decimal(reading['value']) == 0:
                if current and current['baseline']['source_key'] == reading['source_key']:
                    result = dict(current, quantity_ledger_written=False, status='baseline_already_recorded')
                else:
                    if current:
                        if current['state'] == 'needs_attention':
                            return dict(schema_version='vc-lab-weighing-intake/1', status='pending',
                                activity_id=str(activity_id), reason='episode_requires_correction_before_reset',
                                quantity_ledger_written=False)
                        self.weighing.close(key+':close', current['session_id'], expected_sequence=current['sequence'],
                                            actor=actor, reason='observed scale zero; close preceding weighing only')
                    result = self.weighing.start(key+':start', activity_id, keys[0], actor=actor, reason=reason)
            elif current is None or not same_context(current['context'], context):
                return dict(schema_version='vc-lab-weighing-intake/1', status='pending',
                    activity_id=str(activity_id), reason='evidenced_baseline_required_for_this_material_and_target',
                    quantity_ledger_written=False)
            else:
                result = self.weighing.observe(key+':observe', current['session_id'], activity_id, actor=actor, reason=reason)
            return dict(schema_version='vc-lab-weighing-intake/1', status='processed',
                        activity_id=str(activity_id), weighing=result,
                        quantity_ledger_written=result['quantity_ledger_written'])
        return self.core.command(key, dict(activity_id=str(activity_id), generation=generation, policy=config), action)

    def tick(self):
        config = TimelinePolicies.load(self.settings.timeline_policy_path).document.get('weighing', {})
        if not config.get('enabled'):
            return dict(enabled=False)
        with self.core.db.transaction(readonly=True) as conn:
            # Old pending observations remain discoverable when late QR/history
            # arrives. A persisted generation prevents repeat stock writes.
            rows = conn.execute("SELECT id FROM hj_activity WHERE customer_id=%s AND source_type='ocr' ORDER BY start_at DESC NULLS LAST,created_at DESC,id LIMIT 128",
                                (self.core.customer,)).fetchall()
        results = []
        for row in rows:
            try:
                result = self.consume(row['id'], config)
                if not result.get('already_recorded'):
                    results.append(result)
            except (Conflict, Missing, ValueError) as exc:
                results.append(dict(activity_id=str(row['id']), status='pending', reason=str(exc), quantity_ledger_written=False))
            if len(results) >= config['max_activities_per_tick']:
                break
        return clean(dict(enabled=True, processed=len(results), results=results))

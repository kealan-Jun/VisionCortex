"""Customer supplied permanent device UUIDs, separate from visible EQ codes."""
from __future__ import annotations

import re
from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, ident, instant


def validate_catalogue(document):
    if document.get('schema_version') != 'vc-lab-equipment-catalogue/1':
        raise ValueError('Unsupported equipment catalogue')
    source = document.get('source', {})
    if not re.fullmatch(r'[a-f0-9]{64}', source.get('sha256', '')):
        raise ValueError('Equipment catalogue requires its source fingerprint')
    rows = document.get('equipment', [])
    if not rows or len(rows) > 10000:
        raise ValueError('Explicit bounded equipment catalogue required')
    ids, codes = set(), set()
    for row in rows:
        device_id = str(ident(row['device_id']))
        if device_id == '00000000-0000-0000-0000-000000000000' or row.get('qr_payload') != device_id:
            raise ValueError('Device QR must encode the exact permanent UUID')
        code = row.get('device_code', '')
        if not re.fullmatch(r'EQ-[0-9]{6}', code) or code == 'EQ-000000':
            raise ValueError('Visible equipment code must be EQ followed by six digits')
        if device_id in ids or code in codes:
            raise Conflict('Equipment catalogue repeats an ID or visible code')
        ids.add(device_id); codes.add(code)
    return rows


def reserve_catalogue(core, key, catalogue):
    rows = validate_catalogue(catalogue)
    def action(conn):
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', ('equipment:' + str(core.customer),))
        for item in rows:
            device_id, code = ident(item['device_id']), item['device_code']
            found = conn.execute('SELECT * FROM equipment_identity WHERE device_id=%s OR (customer_id=%s AND device_code=%s) FOR UPDATE', (device_id, core.customer, code)).fetchall()
            if found:
                if len(found) != 1 or found[0]['customer_id'] != core.customer or found[0]['device_id'] != device_id or found[0]['device_code'] != code:
                    raise Conflict('A permanent device ID/code is already reserved for a different object')
                if found[0]['provenance']['source_sha256'] != catalogue['source']['sha256']:
                    raise Conflict('Existing device provenance requires an explicit reconciliation')
                continue
            metadata = {k: item[k] for k in ('name', 'source_location', 'current_location', 'ordinal', 'group_count') if k in item}
            conn.execute('INSERT INTO equipment_identity(customer_id,device_id,device_code,metadata,provenance) VALUES(%s,%s,%s,%s,%s)',
                         (core.customer, device_id, code, Jsonb(metadata), Jsonb(dict(source_sha256=catalogue['source']['sha256'], pdf_page=item.get('pdf_page'), label_position=item.get('label_position'), qr_image_sha256=item.get('qr_image_sha256')))))
        conn.execute("INSERT INTO identity_counter(customer_id,kind,last_number) VALUES(%s,'equipment_uuid',%s) ON CONFLICT(customer_id,kind) DO UPDATE SET last_number=GREATEST(identity_counter.last_number,EXCLUDED.last_number)",
                     (core.customer, max(int(r['device_code'][3:]) for r in rows)))
        result = dict(reserved=len(rows), activated=0, source_sha256=catalogue['source']['sha256'], equipment_ids=[r['device_id'] for r in rows])
        return result
    return core.command(key, dict(catalogue=catalogue), action)


def issue_equipment(core, key, *, metadata=None, actor):
    """Reserve a new UUID/code; activation remains a distinct physical check."""
    def action(conn):
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', ('equipment:' + str(core.customer),))
        current = conn.execute("SELECT last_number FROM identity_counter WHERE customer_id=%s AND kind='equipment_uuid' FOR UPDATE", (core.customer,)).fetchone()
        if current and current['last_number'] >= 999999:
            raise Conflict('Equipment code space exhausted')
        number = conn.execute("INSERT INTO identity_counter(customer_id,kind,last_number) VALUES(%s,'equipment_uuid',1) ON CONFLICT(customer_id,kind) DO UPDATE SET last_number=identity_counter.last_number+1 RETURNING last_number", (core.customer,)).fetchone()['last_number']
        device_id, code = ident(), f'EQ-{number:06d}'
        conn.execute('INSERT INTO equipment_identity(customer_id,device_id,device_code,metadata,provenance,actor) VALUES(%s,%s,%s,%s,%s,%s)',
                     (core.customer, device_id, code, Jsonb(clean(metadata or {})), Jsonb(dict(issuance='new_equipment', source_sha256=None)), actor))
        return dict(device_id=device_id, device_code=code, qr_payload=str(device_id), state='reserved', physical_activation_performed=False)
    return core.command(key, dict(action='issue_equipment', metadata=metadata, actor=actor), action)


def activate_equipment(core, key, device_id, *, verified_at, actor, evidence, metadata=None):
    at = instant(verified_at)
    if not actor or not isinstance(evidence, dict) or not evidence:
        raise ValueError('Physical verification actor and evidence are required')
    def action(conn):
        row = conn.execute('SELECT * FROM equipment_identity WHERE customer_id=%s AND device_id=%s FOR UPDATE', (core.customer, ident(device_id))).fetchone()
        if not row:
            raise Missing('Device UUID is not reserved for this customer')
        if row['state'] != 'reserved':
            raise Conflict('Device already activated or retired; preserve its identity/history')
        values = dict(row['metadata'], **(metadata or {}), device_id=str(row['device_id']), device_code=row['device_code'], identity_scheme='device-uuid/1', physical_verification=dict(at=clean(at), actor=actor, evidence=evidence))
        existing = conn.execute('SELECT id FROM entity WHERE customer_id=%s AND kind=\'instrument\' AND code=%s', (core.customer, row['device_code'])).fetchone()
        if existing and existing['id'] != row['device_id']:
            raise Conflict('Visible equipment code already identifies another UUID')
        target = core._register(conn, 'instrument', row['device_code'], values, entity_id=row['device_id'])
        tag = core.register(key + ':label', 'tag', str(row['device_id']), {'mode': 'permanent', 'identity_scheme': 'device-uuid/1'})
        core.assign_tag(key + ':binding', tag['id'], target['id'], at, actor=actor, reason='verified permanent device UUID')
        conn.execute("UPDATE equipment_identity SET state='active',verified_at=%s,actor=%s,metadata=%s WHERE customer_id=%s AND device_id=%s", (at, actor, Jsonb(clean(values)), core.customer, row['device_id']))
        core._emit(conn, 'entity', target['id'], target['revision'], target)
        return dict(device_id=target['id'], device_code=row['device_code'], qr_payload=str(target['id']), state='active', label_id=tag['id'])
    return core.command(key, dict(action='activate_equipment', device_id=device_id, verified_at=verified_at, actor=actor, evidence=evidence, metadata=metadata), action)


def retire_equipment(core, key, device_id, *, at, actor, reason):
    when = instant(at)
    if not actor or not reason:
        raise ValueError('Retirement requires actor and reason')
    def action(conn):
        row = conn.execute('SELECT * FROM equipment_identity WHERE customer_id=%s AND device_id=%s FOR UPDATE', (core.customer, ident(device_id))).fetchone()
        if not row or row['state'] != 'active' or when <= row['verified_at']:
            raise Conflict('Only an active verified device can be retired at a later time')
        conn.execute("UPDATE equipment_identity SET state='retired',retired_at=%s WHERE customer_id=%s AND device_id=%s", (when, core.customer, row['device_id']))
        conn.execute('UPDATE tag_assignment SET valid_to=%s WHERE customer_id=%s AND target_id=%s AND valid_to IS NULL', (when, core.customer, row['device_id']))
        result = dict(device_id=device_id, device_code=row['device_code'], state='retired', retired_at=when, actor=actor, reason=reason)
        core._emit(conn, 'equipment_retirement', row['device_id'], 1, result)
        return result
    return core.command(key, dict(action='retire_equipment', device_id=device_id, at=at, actor=actor, reason=reason), action)


def resolve_qr(core, value, at):
    with core.db.transaction(readonly=True) as conn:
        device = None
        try:
            uid = ident(value)
        except (ValueError, AttributeError):
            uid = None
        if uid:
            device = conn.execute('SELECT * FROM equipment_identity WHERE customer_id=%s AND device_id=%s', (core.customer, uid)).fetchone()
        tag = conn.execute("SELECT id FROM entity WHERE customer_id=%s AND kind='tag' AND code=%s", (core.customer, value)).fetchone()
    result = dict(decoded_value=value, association_status='needs_review', target_id=None, assignments=[])
    if device:
        result.update(device_id=str(device['device_id']), device_code=device['device_code'], device_state=device['state'])
        if device['state'] == 'reserved':
            return dict(result, reason='physical_verification_pending')
    if tag:
        result.update(core.resolve_tag(tag['id'], at))
    return result

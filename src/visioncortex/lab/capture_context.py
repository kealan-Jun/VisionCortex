"""Temporal wearer identity; a capture device is never a laboratory instrument."""
from .domain import Conflict, clean, ident, instant


def assign_camera_person(core, key, camera_id, person_id, valid_from, *, actor, reason):
    if not actor or not reason:
        raise ValueError('Wearer changes require an actor and reason')
    when = instant(valid_from)
    def action(conn):
        camera = core._entity(conn, camera_id, 'capture_device', lock=True)
        if camera['metadata'].get('view_role') != 'first_person':
            raise Conflict('Only a registered first-person capture device has a wearer')
        if person_id:
            core._entity(conn, person_id, 'person')
        rows = conn.execute('SELECT * FROM camera_person_binding WHERE customer_id=%s AND camera_id=%s ORDER BY valid_from DESC FOR UPDATE', (core.customer, ident(camera_id))).fetchall()
        if rows and when <= rows[0]['valid_from']:
            raise Conflict('Wearer changes must append to temporal history')
        for row in rows:
            if row['valid_to'] is None:
                conn.execute('UPDATE camera_person_binding SET valid_to=%s WHERE customer_id=%s AND id=%s', (when, core.customer, row['id']))
            elif when < row['valid_to']:
                raise Conflict('Wearer changes cannot overwrite historical intervals')
        binding = None
        if person_id:
            binding = conn.execute('INSERT INTO camera_person_binding(customer_id,id,camera_id,person_id,valid_from,actor,reason) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *', (core.customer, ident(), ident(camera_id), ident(person_id), when, actor, reason)).fetchone()
        core._emit(conn, 'camera_person', camera['id'], len(rows)+1, dict(camera_id=camera_id, person_id=person_id, valid_from=when, actor=actor, reason=reason, binding=binding))
        return dict(camera_id=camera_id, person_id=person_id, valid_from=when, binding=binding)
    return core.command(key, dict(camera_id=camera_id, person_id=person_id, valid_from=when, actor=actor, reason=reason), action)


def camera_context(core, camera_code, at):
    result = dict(camera_id=camera_code, wearer_id=None, wearer_binding_id=None, association_status='unknown')
    with core.db.transaction(readonly=True) as conn:
        camera = conn.execute("SELECT * FROM entity WHERE customer_id=%s AND kind='capture_device' AND code=%s", (core.customer, camera_code)).fetchone()
        if not camera:
            return result
        result.update(capture_device_id=str(camera['id']), view_role=camera['metadata'].get('view_role'))
        if result['view_role'] != 'first_person' or not at:
            return result
        binding = conn.execute('SELECT * FROM camera_person_binding WHERE customer_id=%s AND camera_id=%s AND valid_from<=%s AND (valid_to IS NULL OR valid_to>%s)', (core.customer, camera['id'], instant(at), instant(at))).fetchone()
        if binding:
            result.update(wearer_id=str(binding['person_id']), wearer_binding_id=str(binding['id']), association_status='explicit_at_capture', wearer_interval=clean(dict(valid_from=binding['valid_from'], valid_to=binding['valid_to'])))
    return result

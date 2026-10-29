"""Source-pinned, versioned panel registration. Automatic vision can supply this later."""
from psycopg.types.json import Jsonb

from .domain import Conflict, Missing, clean, ident
from .ocr_fields import validate_registration


class TimelineReadouts:
    def __init__(self, timeline):
        self.timeline = timeline
        self.core, self.customer = timeline.core, timeline.customer

    def register_frame(self, key, media_id, registration, *, actor, reason):
        if not actor or not reason:
            raise ValueError('Registration author and reason required')
        def action(conn):
            media = conn.execute("SELECT m.kind,a.verified_sha256 FROM hj_timeline_media m JOIN media_asset a ON a.customer_id=m.customer_id AND a.id=m.media_id WHERE m.customer_id=%s AND m.media_id=%s", (self.customer, ident(media_id))).fetchone()
            if not media:
                raise Missing('OCR photograph outside customer scope')
            if media['kind'] != 'photo':
                raise ValueError('OCR frame registration requires a photograph')
            validate_registration(registration, media['verified_sha256'])
            self.core._entity(conn, registration['instrument_id'], 'instrument')
            for gauge in registration.get('gauges', []):
                if gauge['calibration']['instrument_id'] != registration['instrument_id']:
                    raise Conflict('Gauge calibration belongs to a different instrument')
            revision = conn.execute('SELECT COALESCE(MAX(revision),0)+1 AS n FROM hj_ocr_frame_registration WHERE customer_id=%s AND media_id=%s', (self.customer, ident(media_id))).fetchone()['n']
            conn.execute('INSERT INTO hj_ocr_frame_registration(customer_id,media_id,revision,source_sha256,registration,actor,reason) VALUES(%s,%s,%s,%s,%s,%s,%s)', (self.customer, ident(media_id), revision, media['verified_sha256'], Jsonb(registration), actor, reason))
            task_id = ident()
            conn.execute("INSERT INTO hj_timeline_task(customer_id,id,media_id,kind,analysis_revision) VALUES(%s,%s,%s,'ocr',%s)", (self.customer, task_id, ident(media_id), 'frame-registration:' + str(revision)))
            return clean(dict(media_id=media_id, revision=revision, task_id=task_id, physical_identity_basis='explicit_source_frame_registration'))
        return self.timeline._command(key, 'register_ocr_frame', dict(media_id=media_id, registration=registration, actor=actor, reason=reason), action)

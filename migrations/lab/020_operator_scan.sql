-- Explicit human declarations; background pixel observations cannot become these events.
CREATE TABLE hj_operator_scan_event (
 customer_id uuid NOT NULL, id uuid NOT NULL, event_key text NOT NULL,
 person_id uuid, camera_id uuid, occurred_at timestamptz,
 scan_kind text NOT NULL CHECK(scan_kind IN ('equipment','material')),
 payload jsonb NOT NULL, payload_sha256 text NOT NULL, evidence_ids uuid[] NOT NULL,
 processed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,event_key),
 CHECK(person_id IS NOT NULL OR camera_id IS NOT NULL),
 FOREIGN KEY(customer_id,person_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,camera_id) REFERENCES entity(customer_id,id)
);
CREATE INDEX hj_operator_scan_scope ON hj_operator_scan_event(customer_id,person_id,camera_id,occurred_at);
CREATE FUNCTION vc_lab_operator_scan_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'Operator scan events are append-only'; END;
$$;
CREATE TRIGGER vc_lab_operator_scan_immutable BEFORE UPDATE OR DELETE ON hj_operator_scan_event
 FOR EACH ROW EXECUTE FUNCTION vc_lab_operator_scan_immutable();

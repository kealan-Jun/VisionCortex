-- Candidate evidence only. Authoritative quantities remain Core.operate/ledger.
CREATE TABLE hj_material_observation_revision (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), id uuid NOT NULL,
 occurrence_key text NOT NULL, revision integer NOT NULL,
 event_start timestamptz, event_end timestamptz, clock_status text NOT NULL,
 relation text NOT NULL CHECK(relation IN ('seen','handled','usage_candidate')),
 snapshot jsonb NOT NULL, evidence_ids uuid[] NOT NULL,
 actor text NOT NULL, reason text NOT NULL,
 processed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,occurrence_key,revision),
 CHECK(event_end IS NULL OR (event_start IS NOT NULL AND event_end>=event_start))
);
CREATE INDEX hj_material_observation_time ON hj_material_observation_revision(customer_id,event_start,event_end);

-- Additive dynamic visual context. Does not end experiments/material relations.
CREATE TABLE hj_dynamic_panel_link (
 customer_id uuid NOT NULL, id uuid NOT NULL, instrument_id uuid NOT NULL,
 panel_id text NOT NULL, layout_revision text NOT NULL, revision integer NOT NULL,
 geometry jsonb NOT NULL, evidence_ids uuid[] NOT NULL,
 actor text NOT NULL, reason text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,instrument_id,panel_id,revision),
 FOREIGN KEY(customer_id,instrument_id) REFERENCES entity(customer_id,id)
);
CREATE TABLE hj_dynamic_frame_event (
 customer_id uuid NOT NULL, id uuid NOT NULL, media_id uuid NOT NULL,
 source_session_key text NOT NULL, frame_key text NOT NULL,
 source_seconds numeric(28,12) NOT NULL CHECK(source_seconds>=0),
 payload jsonb NOT NULL, payload_sha256 text NOT NULL, supersedes_frame_key text,
 event_at timestamptz, observed_at timestamptz,
 processed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,media_id,source_session_key,frame_key),
 FOREIGN KEY(customer_id,media_id) REFERENCES hj_timeline_media(customer_id,media_id)
);
CREATE INDEX hj_dynamic_frame_order ON hj_dynamic_frame_event(customer_id,media_id,source_session_key,source_seconds);
CREATE TABLE hj_dynamic_context_revision (
 customer_id uuid NOT NULL, id uuid NOT NULL, media_id uuid NOT NULL,
 source_session_key text NOT NULL, revision integer NOT NULL,
 input_fingerprint text NOT NULL, policy_fingerprint text NOT NULL,
 snapshot jsonb NOT NULL, processed_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,id), UNIQUE(customer_id,media_id,source_session_key,revision),
 FOREIGN KEY(customer_id,media_id) REFERENCES hj_timeline_media(customer_id,media_id)
);

CREATE TABLE camera_person_binding (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), id uuid NOT NULL,
 camera_id uuid NOT NULL, person_id uuid NOT NULL,
 valid_from timestamptz NOT NULL, valid_to timestamptz,
 actor text NOT NULL, reason text NOT NULL,
 PRIMARY KEY(customer_id,id),
 FOREIGN KEY(customer_id,camera_id) REFERENCES entity(customer_id,id),
 FOREIGN KEY(customer_id,person_id) REFERENCES entity(customer_id,id),
 CHECK(valid_to IS NULL OR valid_to>valid_from)
);
CREATE UNIQUE INDEX camera_one_current_person ON camera_person_binding(customer_id,camera_id) WHERE valid_to IS NULL;
CREATE INDEX camera_person_at ON camera_person_binding(customer_id,camera_id,valid_from,valid_to);
CREATE TABLE processing_precomputed (
 customer_id uuid NOT NULL, job_id uuid NOT NULL,
 model_identity text NOT NULL, rule_fingerprint text NOT NULL, image_sha256 text NOT NULL,
 result jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,job_id),
 FOREIGN KEY(customer_id,job_id) REFERENCES processing_job(customer_id,id)
);

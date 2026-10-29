CREATE TABLE intake_item (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), relative_path text NOT NULL,
 signature text NOT NULL, status text NOT NULL, error_code text,
 result jsonb, checked_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(customer_id,relative_path)
);
ALTER TABLE operation_line ADD CONSTRAINT operation_direction_unique UNIQUE(customer_id,operation_id,material_id,direction);

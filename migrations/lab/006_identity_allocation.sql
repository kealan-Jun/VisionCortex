CREATE TABLE identity_namespace (
 customer_id uuid PRIMARY KEY REFERENCES customer_instance(id), prefix text NOT NULL UNIQUE,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE identity_counter (
 customer_id uuid NOT NULL REFERENCES customer_instance(id), kind text NOT NULL,
 last_number integer NOT NULL CHECK(last_number BETWEEN 1 AND 999999),
 PRIMARY KEY(customer_id,kind)
);

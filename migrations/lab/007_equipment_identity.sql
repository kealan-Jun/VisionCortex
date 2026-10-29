-- External permanent device UUIDs are reserved before a physical verification.
-- A reservation is not an activated physical asset or an inventory opening.
CREATE TABLE equipment_identity (
 customer_id uuid NOT NULL REFERENCES customer_instance(id),
 device_id uuid NOT NULL UNIQUE, device_code text NOT NULL,
 state text NOT NULL DEFAULT 'reserved' CHECK(state IN ('reserved','active','retired')),
 metadata jsonb NOT NULL DEFAULT '{}', provenance jsonb NOT NULL,
 verified_at timestamptz, retired_at timestamptz, actor text,
 PRIMARY KEY(customer_id,device_id), UNIQUE(customer_id,device_code),
 CHECK(device_code ~ '^EQ-[0-9]{6}$'),
 CHECK(state='reserved' OR verified_at IS NOT NULL),
 CHECK(state<>'retired' OR retired_at>=verified_at)
);

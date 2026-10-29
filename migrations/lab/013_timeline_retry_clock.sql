-- Use a finite due time: psycopg intentionally rejects PostgreSQL infinity dates.
ALTER TABLE hj_timeline_task ALTER COLUMN next_attempt_at SET DEFAULT '1970-01-01T00:00:00Z';
UPDATE hj_timeline_task SET next_attempt_at='1970-01-01T00:00:00Z' WHERE next_attempt_at='-infinity';

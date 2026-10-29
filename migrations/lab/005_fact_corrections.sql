ALTER TABLE fact_revision ADD COLUMN correction_metadata jsonb NOT NULL DEFAULT '{}';

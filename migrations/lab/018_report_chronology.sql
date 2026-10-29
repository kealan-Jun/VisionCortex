-- A correction can restore earlier content. Preserve a new chronological
-- revision rather than leaving a now-stale highest revision for publishers.
ALTER TABLE hj_daily_report DROP CONSTRAINT hj_daily_report_customer_id_day_fingerprint_key;
CREATE INDEX hj_daily_report_content_history ON hj_daily_report(customer_id,day,fingerprint);

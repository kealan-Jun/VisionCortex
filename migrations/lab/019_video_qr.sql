-- Independent model-free video QR tasks reuse the existing lease/checkpoint queue.
ALTER TABLE hj_timeline_task DROP CONSTRAINT hj_timeline_task_kind_check;
ALTER TABLE hj_timeline_task ADD CONSTRAINT hj_timeline_task_kind_check
 CHECK(kind IN ('video_understanding','transcription','ocr','qr'));

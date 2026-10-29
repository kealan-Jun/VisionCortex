"""Resume new durable photo jobs after a process interruption, preserving each attempt."""

import copy
import json

from field_recognition.measurements.photo_measurements import versions, refresh


def recover(core):
    recovered = []
    with core["ocr_submit_lock"], core["db"]() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT document FROM jobs WHERE status='interrupted' AND json_extract(document,'$.resume_pending')=1 ORDER BY rowid"
        ).fetchall()
        for row in rows:
            job = json.loads(row["document"])
            if job.get("request_trigger") == "video_stream":
                continue
            history = job.get("attempt_history", [])
            previous = copy.deepcopy(job)
            previous.pop("attempt_history", None)
            history.append(previous | {"status": "interrupted", "recorded_at": core["now"]()})
            job["attempt_history"] = history
            for key in (
                "local_ocr",
                "fallback",
                "lines",
                "readings",
                "measurement_records",
                "panel_regions",
                "panel_detection",
                "finished_at",
                "error",
                "outcome",
                "recognition_skipped",
                "skip_reason",
            ):
                job.pop(key, None)
            job.update(
                status="queued",
                phase="recovered_after_restart",
                recognition_versions=versions(),
                resumed_at=core["now"](),
            )
            conn.execute(
                "UPDATE jobs SET status=?,document=? WHERE id=?",
                ("queued", json.dumps(job), job["job_id"]),
            )
            if job.get("measurement_id"):
                refresh(core, conn, job["measurement_id"])
            recovered.append(job)
    for job in recovered:
        core["readout_pool"].submit(core["run_ocr"], job)
    return [j["job_id"] for j in recovered]

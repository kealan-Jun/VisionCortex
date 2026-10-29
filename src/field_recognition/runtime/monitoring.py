"""Health checks, metrics, and monitoring for production deployment."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse

from field_recognition.runtime.database import Database


class MonitoringService:
    """Centralized monitoring and health check service."""

    def __init__(self, db: Database, core: dict):
        self.db = db
        self.core = core
        self.start_time = time.time()

    def health_check(self) -> tuple[dict, int]:
        """
        Comprehensive health check for readiness probes.

        Returns:
            Tuple of (health_status_dict, http_status_code)
        """
        checks = {}
        degraded_services = []

        # Database check
        try:
            with self.db.connection() as conn:
                conn.execute("SELECT 1").fetchone()
            checks["database"] = {"status": "healthy"}
        except Exception as e:
            checks["database"] = {"status": "unhealthy", "error": str(e)}
            degraded_services.append("database")

        # OCR model check
        ocr_status = self.core.get("ocr_state", {}).get("status", "not_loaded")
        if ocr_status == "ready":
            checks["ocr"] = {"status": "healthy", "resident": True}
        elif ocr_status in ["not_loaded", "loading"]:
            checks["ocr"] = {"status": "degraded", "loading": True}
        else:
            checks["ocr"] = {"status": "unhealthy", "reason": ocr_status}
            degraded_services.append("ocr")

        # Archive check
        archive_store = self.core.get("archive_store")
        if archive_store and archive_store.enabled():
            archive_status = archive_store.status.get("status")
            if archive_status == "ready":
                checks["archive"] = {"status": "healthy"}
            elif archive_status == "retrying":
                checks["archive"] = {"status": "degraded", "retrying": True}
            else:
                checks["archive"] = {"status": "degraded", "reason": archive_status}
        else:
            checks["archive"] = {"status": "disabled"}

        # Camera receiver check
        receiver_camera = self.core.get("receiver_camera")
        if receiver_camera:
            try:
                snapshot = receiver_camera.snapshot()
                if snapshot.get("status") == "streaming":
                    checks["receiver"] = {"status": "healthy"}
                else:
                    checks["receiver"] = {"status": "degraded", "reason": snapshot.get("status")}
            except Exception as e:
                checks["receiver"] = {"status": "unhealthy", "error": str(e)}
                degraded_services.append("receiver")
        else:
            checks["receiver"] = {"status": "disabled"}

        # Overall status
        if degraded_services:
            overall_status = "unhealthy"
            http_status = 503
        elif any(c["status"] == "degraded" for c in checks.values()):
            overall_status = "degraded"
            http_status = 200  # Still accepting traffic
        else:
            overall_status = "healthy"
            http_status = 200

        result = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self.start_time),
            "checks": checks,
        }

        return result, http_status

    def liveness_check(self) -> tuple[dict, int]:
        """Simple liveness check (should always return 200 unless process is dying)."""
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self.start_time),
        }, 200

    def get_metrics(self) -> str:
        """
        Prometheus-format metrics.

        Returns:
            Metrics in Prometheus text format
        """
        lines = []

        # Uptime
        uptime = int(time.time() - self.start_time)
        lines.append(f"# HELP field_uptime_seconds Service uptime in seconds")
        lines.append(f"# TYPE field_uptime_seconds gauge")
        lines.append(f"field_uptime_seconds {uptime}")

        # Database stats
        try:
            with self.db.connection() as conn:
                # Job queue stats
                for status in ["queued", "running", "completed", "failed", "interrupted"]:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status=?", (status,)
                    ).fetchone()[0]
                    lines.append(f'field_jobs_total{{status="{status}"}} {count}')

                # Binding stats
                active_bindings = conn.execute(
                    "SELECT COUNT(*) FROM bindings WHERE ended IS NULL"
                ).fetchone()[0]
                lines.append(f"field_bindings_active {active_bindings}")

                total_bindings = conn.execute("SELECT COUNT(*) FROM bindings").fetchone()[0]
                lines.append(f"field_bindings_total {total_bindings}")

                # Archive stats
                pending_archive = conn.execute(
                    "SELECT COUNT(*) FROM archive_outbox WHERE archived_at IS NULL"
                ).fetchone()[0]
                lines.append(f"field_archive_pending {pending_archive}")

                archived_total = conn.execute(
                    "SELECT COUNT(*) FROM archive_outbox WHERE archived_at IS NOT NULL"
                ).fetchone()[0]
                lines.append(f"field_archive_completed {archived_total}")

                # Measurement stats
                try:
                    measurement_states = conn.execute("""
                        SELECT measurement_state, COUNT(*) as count
                        FROM photo_measurements
                        GROUP BY measurement_state
                    """).fetchall()
                    for row in measurement_states:
                        lines.append(f'field_measurements_total{{state="{row[0]}"}} {row[1]}')
                except Exception:
                    pass  # Table might not exist yet

        except Exception as e:
            lines.append(f"# ERROR: Database query failed: {e}")

        # OCR status
        ocr_state = self.core.get("ocr_state", {})
        ocr_ready = 1 if ocr_state.get("status") == "ready" else 0
        lines.append(f"field_ocr_ready {ocr_ready}")

        if ocr_state.get("load_count"):
            lines.append(f"field_ocr_load_count {ocr_state['load_count']}")

        return "\n".join(lines) + "\n"

    def get_detailed_status(self) -> dict:
        """Get detailed system status for admin dashboard."""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self.start_time),
            "environment": {
                "record_mode": os.environ.get("FIELD_RECORD_MODE", "test"),
                "ocr_device": os.environ.get("FIELD_OCR_DEVICE", "cpu"),
                "archive_enabled": os.environ.get("FIELD_ARCHIVE_ENABLED") == "1",
                "photo_watch_enabled": os.environ.get("FIELD_SAVED_PHOTO_WATCH_ENABLED") == "1",
                "video_ocr_enabled": os.environ.get("FIELD_VIDEO_OCR_ENABLED") == "1",
            },
        }

        # Database info
        try:
            with self.db.connection() as conn:
                db_size = Path(self.db.path).stat().st_size
                status["database"] = {
                    "path": str(self.db.path),
                    "size_mb": round(db_size / 1024 / 1024, 2),
                    "tables": {},
                }

                for table in [
                    "users",
                    "bindings",
                    "jobs",
                    "instruments",
                    "scans",
                    "archive_outbox",
                    "photo_measurements",
                ]:
                    try:
                        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        status["database"]["tables"][table] = count
                    except Exception:
                        pass

        except Exception as e:
            status["database"] = {"error": str(e)}

        # OCR info
        status["ocr"] = self.core.get("ocr_state", {})

        # Archive info
        archive_store = self.core.get("archive_store")
        if archive_store and archive_store.enabled():
            status["archive"] = archive_store.status
        else:
            status["archive"] = {"enabled": False}

        # Queue stats
        try:
            from field_recognition.runtime.task_queue import TaskQueue

            queue = TaskQueue(self.db)
            status["queue"] = queue.get_queue_stats()
        except Exception as e:
            status["queue"] = {"error": str(e)}

        return status


def setup_monitoring_routes(app: FastAPI, monitoring: MonitoringService):
    """Add monitoring endpoints to FastAPI app."""

    @app.get("/health")
    def health():
        """Kubernetes readiness probe."""
        result, status_code = monitoring.health_check()
        return JSONResponse(content=result, status_code=status_code)

    @app.get("/health/live")
    def liveness():
        """Kubernetes liveness probe."""
        result, status_code = monitoring.liveness_check()
        return JSONResponse(content=result, status_code=status_code)

    @app.get("/metrics")
    def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            content=monitoring.get_metrics(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/api/admin/status")
    def admin_status():
        """Detailed status for admin dashboard (requires auth in production)."""
        return monitoring.get_detailed_status()

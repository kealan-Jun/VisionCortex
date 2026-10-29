"""Integration adapter for production components with legacy app.py."""

import json
from typing import Optional

from field_recognition.runtime.database import Database
from field_recognition.runtime.auth import AuthService, User
from field_recognition.runtime.task_queue import TaskQueue
from field_recognition.bindings.camera_registry import CameraRegistry
from field_recognition.measurements.measurement_state import MeasurementStateMachine
from field_recognition.bindings.instrument_config import InstrumentConfig


class LegacyAdapter:
    """
    Adapter to integrate production components with legacy app.py code.

    Provides backward-compatible interfaces while enabling new production features.
    """

    def __init__(self, production_core):
        """
        Initialize adapter with production core.

        Args:
            production_core: ProductionCore instance
        """
        self.core = production_core
        self.db = production_core.db
        self.auth_service = production_core.auth_service
        self.camera_registry = production_core.camera_registry
        self.measurement_state = production_core.measurement_state
        self.instrument_config = production_core.instrument_config

        # Legacy compatibility
        self._legacy_db_cache = {}

    def get_legacy_db_function(self):
        """
        Return a db() function compatible with legacy code.

        Returns:
            Function that returns database connection context manager
        """

        def db():
            """Legacy db() context manager."""
            return self.db.connection()

        return db

    def get_legacy_db_transaction(self, mode="DEFERRED"):
        """
        Get legacy-style transaction context manager.

        Args:
            mode: Transaction mode (DEFERRED, IMMEDIATE, EXCLUSIVE)

        Returns:
            Database transaction context manager
        """
        return self.db.transaction(mode)

    def get_camera_context(self, camera_id: str):
        """
        Get camera context, creating if needed for legacy compatibility.

        Args:
            camera_id: Camera identifier

        Returns:
            Camera context
        """
        return self.camera_registry.get_or_register_legacy(camera_id)

    def verify_user_or_test(self, session_id: Optional[str]) -> dict:
        """
        Verify user session or return test user in test mode.

        Args:
            session_id: Optional session ID from header

        Returns:
            User dict with id, username, display_name, role
        """
        # If auth is disabled, return test user
        if not self.auth_service:
            return {
                "id": "test-user",
                "username": "test_operator",
                "display_name": "Test Operator",
                "role": "operator",
            }

        # In test mode, allow missing session
        if self.core.config.record_mode == "test" and not session_id:
            return self.core.get_or_create_test_user()

        # Production mode requires valid session
        user = self.auth_service.verify_session(session_id)
        return user.model_dump()

    def enrich_job_document(self, job: dict, user: Optional[dict] = None) -> dict:
        """
        Enrich job document with production metadata.

        Args:
            job: Job document
            user: Optional user info

        Returns:
            Enriched job document
        """
        if user:
            job["operator_user_id"] = user.get("id")
            job["operator_username"] = user.get("username")

        job["record_mode"] = self.core.config.record_mode

        return job

    def create_measurement_draft(
        self, measurement_id: str, context: dict, camera_id: str, operator: str, job_ids: list[str]
    ) -> dict:
        """
        Create measurement draft using production state machine.

        Args:
            measurement_id: Measurement ID
            context: Measurement context
            camera_id: Camera ID
            operator: Operator name
            job_ids: Job IDs

        Returns:
            Created measurement document
        """
        return self.measurement_state.create_measurement(
            measurement_id=measurement_id,
            burst_id=context.get("burst_id"),
            camera_id=camera_id,
            operator=operator,
            job_ids=job_ids,
            context=context,
        )

    def get_task_queue(self, worker_id: str) -> TaskQueue:
        """
        Get task queue for worker.

        Args:
            worker_id: Worker identifier

        Returns:
            TaskQueue instance
        """
        return self.core.task_queue_factory(worker_id)

    def check_instrument_ownership(self, instrument_id: str, user_id: str) -> bool:
        """
        Check if user has access to instrument.

        Args:
            instrument_id: Instrument ID
            user_id: User ID

        Returns:
            True if user has access
        """
        # In test mode, allow all access
        if self.core.config.record_mode == "test":
            return True

        # Check if instrument is bound to this user
        with self.db.connection() as conn:
            row = conn.execute(
                """
                SELECT id FROM bindings
                WHERE instrument_id=? AND ended IS NULL
            """,
                (instrument_id,),
            ).fetchone()

            if not row:
                return True  # Unbound instrument

            binding_doc = json.loads(
                conn.execute("SELECT document FROM bindings WHERE id=?", (row["id"],)).fetchone()[
                    "document"
                ]
            )

            return binding_doc.get("operator_user_id") == user_id

    def apply_production_constraints(self, operation: str, **kwargs) -> dict:
        """
        Apply production-specific constraints to operations.

        Args:
            operation: Operation name
            **kwargs: Operation parameters

        Returns:
            Validation result with 'allowed' bool and optional 'reason'
        """
        if self.core.config.record_mode == "test":
            return {"allowed": True}

        # Production-specific constraints
        if operation == "auto_write_measurement":
            return {"allowed": False, "reason": "生产模式下自动写入测量结果必须经过确认流程"}

        if operation == "modify_confirmed_measurement":
            return {"allowed": False, "reason": "已确认的测量记录不能修改"}

        return {"allowed": True}

    def get_operator_from_user(self, user: dict) -> str:
        """
        Get operator name from user dict (for legacy compatibility).

        Args:
            user: User dict

        Returns:
            Operator name string
        """
        return user.get("display_name") or user.get("username", "unknown")


def integrate_with_legacy_app(app_globals: dict, production_core):
    """
    Integrate production components into legacy app globals.

    Modifies app_globals dict in-place to add production services
    while maintaining backward compatibility.

    Args:
        app_globals: Dictionary of app global variables
        production_core: ProductionCore instance
    """
    # Create adapter
    adapter = LegacyAdapter(production_core)

    # Replace db() function with adapter version
    app_globals["db"] = adapter.get_legacy_db_function()
    app_globals["Database"] = production_core.db

    # Add production services
    app_globals["production_core"] = production_core
    app_globals["adapter"] = adapter
    app_globals["auth_service"] = production_core.auth_service
    app_globals["camera_registry"] = production_core.camera_registry
    app_globals["measurement_state"] = production_core.measurement_state
    app_globals["instrument_config"] = production_core.instrument_config

    # Add config
    app_globals["PRODUCTION_CONFIG"] = production_core.config
    app_globals["IS_PRODUCTION"] = production_core.config.record_mode == "production"

    return adapter

"""Production-ready application bootstrap and integration layer."""

import sys
from pathlib import Path

from field_recognition.runtime.database import Database
from field_recognition.runtime.auth import AuthService
from field_recognition.runtime.task_queue import TaskQueue
from field_recognition.bindings.camera_registry import CameraRegistry
from field_recognition.runtime.monitoring import MonitoringService
from field_recognition.bindings.instrument_config import InstrumentConfig
from field_recognition.measurements.measurement_state import MeasurementStateMachine
from field_recognition.runtime.production_config import (
    ProductionConfig,
    validate_production_deployment,
)


class ProductionCore:
    """
    Production-ready core services integrating authentication, multi-camera,
    state management, and monitoring.
    """

    def __init__(self, config: ProductionConfig):
        """
        Initialize production core.

        Args:
            config: Production configuration
        """
        self.config = config

        # Initialize database
        db_path = Path(config.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = Database(db_path)

        # Initialize services
        self.auth_service = (
            AuthService(self.db, session_duration_hours=config.session_duration_hours)
            if config.auth_enabled
            else None
        )
        self.camera_registry = CameraRegistry(self.db)
        self.instrument_config = InstrumentConfig(self.db)
        self.measurement_state = None  # Installed with the evidence-backed application workflow.

        # Task queue (will be initialized per worker)
        self.task_queue_factory = lambda worker_id: TaskQueue(self.db, worker_id)

        # Monitoring (initialized after app starts)
        self.monitoring = None

        # Legacy compatibility: load default camera if configured
        self._init_legacy_camera()

    def _init_legacy_camera(self):
        """Initialize legacy single-camera setup for backward compatibility."""
        if self.config.camera_id:
            self.camera_registry.get_or_register_legacy(
                camera_id=self.config.camera_id, display_name=f"Camera {self.config.camera_id}"
            )

    def init_monitoring(self, core_dict: dict):
        """
        Initialize monitoring service with access to app core.

        Args:
            core_dict: Dictionary with app core objects (ocr_state, archive_store, etc.)
        """
        self.monitoring = MonitoringService(self.db, core_dict)
        self.measurement_state = MeasurementStateMachine(core_dict)

    def create_initial_admin_if_needed(
        self, username: str = None, display_name: str = None, password: str = None
    ) -> dict | None:
        """
        Create initial admin user if no users exist.

        Args:
            username: Admin username (prompt if not provided)
            display_name: Display name (prompt if not provided)
            password: Password (prompt if not provided)

        Returns:
            Created user dict or None if users already exist
        """
        if not self.auth_service:
            return None

        with self.db.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if count > 0:
                return None

        # Interactive prompts if not provided
        if not username:
            username = input("Enter initial admin username: ").strip()
        if not display_name:
            display_name = input("Enter admin display name: ").strip()
        if not password:
            import getpass

            password = getpass.getpass("Enter admin password: ")

        if not username or not password:
            raise ValueError("Username and password required")

        user = self.auth_service.create_initial_admin(username, display_name, password)
        print(f"✓ Created initial admin user: {user.username}")
        return user.model_dump()

    def get_or_create_test_user(self, username: str = "test_user") -> dict:
        """
        Get or create a test user for development/testing.

        Only works in test mode. Returns user info.
        """
        if self.config.record_mode == "production":
            raise RuntimeError("Cannot create test users in production mode")

        if not self.auth_service:
            # Auth disabled, return fake user
            return {
                "id": "test-user-id",
                "username": username,
                "display_name": "Test User",
                "role": "operator",
            }

        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

            if row:
                return {
                    key: row[key]
                    for key in ("id", "username", "display_name", "role", "created_at", "disabled")
                }

        # Create test user
        import uuid
        from datetime import datetime, timezone

        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO users VALUES(?,?,?,?,?,?,?)",
                (user_id, username, "Test User", "operator", None, now, 0),
            )

        return {
            "id": user_id,
            "username": username,
            "display_name": "Test User",
            "role": "operator",
        }

    def validate_deployment(self) -> bool:
        """
        Validate deployment is ready to start.

        Returns:
            True if ready, False otherwise
        """
        # Check migrations
        from field_recognition.runtime.database import get_migration_status

        status = get_migration_status(self.db)
        if status["pending"]:
            print(f"❌ {len(status['pending'])} pending database migrations", file=sys.stderr)
            print("Run: python migrate_db.py", file=sys.stderr)
            return False

        # Production mode validations
        if self.config.record_mode == "production":
            if not self.config.auth_enabled:
                print("❌ Authentication must be enabled in production mode", file=sys.stderr)
                return False

            # Verify at least one user exists
            with self.db.connection() as conn:
                user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                if user_count == 0:
                    print("❌ No users exist. Create initial admin:", file=sys.stderr)
                    print("   python production_bootstrap.py --create-admin", file=sys.stderr)
                    return False

        return True

    def cleanup(self):
        """Cleanup resources on shutdown."""
        if hasattr(self, "db"):
            self.db.close()


def create_production_core(validate: bool = True) -> ProductionCore:
    """
    Factory to create production core with configuration validation.

    Args:
        validate: Whether to validate configuration (default True)

    Returns:
        Initialized ProductionCore

    Raises:
        SystemExit: If validation fails in production mode
    """
    # Load configuration
    config = ProductionConfig.from_environment()

    # Validate if requested
    if validate:
        if not validate_production_deployment():
            if config.record_mode == "production":
                print("❌ Production deployment validation failed", file=sys.stderr)
                sys.exit(1)

    # Create and validate core
    core = ProductionCore(config)

    if validate and not core.validate_deployment():
        if config.record_mode == "production":
            print("❌ Deployment validation failed", file=sys.stderr)
            sys.exit(1)

    return core


if __name__ == "__main__":
    """Bootstrap script for production deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="Production deployment bootstrap")
    parser.add_argument(
        "--create-admin", action="store_true", help="Create initial admin user interactively"
    )
    parser.add_argument("--validate", action="store_true", help="Validate deployment configuration")
    parser.add_argument("--username", help="Admin username (for --create-admin)")
    parser.add_argument("--display-name", help="Admin display name")
    parser.add_argument("--password", help="Admin password (use with caution)")

    args = parser.parse_args()

    if args.validate:
        valid = validate_production_deployment()
        sys.exit(0 if valid else 1)

    if args.create_admin:
        core = create_production_core(validate=False)

        if not core.auth_service:
            print("❌ Authentication not enabled. Set FIELD_AUTH_ENABLED=1", file=sys.stderr)
            sys.exit(1)

        try:
            user = core.create_initial_admin_if_needed(
                username=args.username, display_name=args.display_name, password=args.password
            )

            if not user:
                print("ℹ️  Admin user already exists")
            else:
                print(f"✓ Admin user created: {user['username']}")

            sys.exit(0)

        except Exception as e:
            print(f"❌ Failed to create admin: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

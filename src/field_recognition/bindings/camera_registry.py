"""Multi-camera registry and per-camera context management."""

import json
import re
from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel

from field_recognition.runtime.database import Database


class CameraConfig(BaseModel):
    """Camera configuration model."""

    camera_id: str
    display_name: str
    device_credential_id: Optional[str] = None
    receiver_url: Optional[str] = None
    nas_photo_root: Optional[str] = None
    enabled: bool = True
    registered_at: str
    metadata: Optional[dict] = None


class CameraContext:
    """Runtime context for a single camera instance."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera_id = config.camera_id
        self.current_session_id: Optional[str] = None
        self.active_bindings: Dict[str, dict] = {}
        self.photo_watch_cursor: Optional[str] = None
        self.last_activity: Optional[str] = None

    def update_session(self, session_id: str):
        """Update current device service session."""
        self.current_session_id = session_id
        self.last_activity = datetime.now(timezone.utc).isoformat()

    def add_binding(self, binding_id: str, binding_data: dict):
        """Add active binding."""
        self.active_bindings[binding_id] = binding_data
        self.last_activity = datetime.now(timezone.utc).isoformat()

    def remove_binding(self, binding_id: str):
        """Remove binding."""
        self.active_bindings.pop(binding_id, None)
        self.last_activity = datetime.now(timezone.utc).isoformat()

    def get_binding_for_instrument(self, instrument_id: str) -> Optional[dict]:
        """Find active binding for an instrument."""
        for binding in self.active_bindings.values():
            if binding.get("instrument_id") == instrument_id:
                return binding
        return None


class CameraRegistry:
    """
    Multi-camera registry with independent contexts.

    Manages multiple cameras, each with its own:
    - Device service session
    - Active bindings
    - Photo monitoring state
    - Task quotas
    """

    def __init__(self, db: Database):
        self.db = db
        self.contexts: Dict[str, CameraContext] = {}
        self._load_cameras()

    def _load_cameras(self):
        """Load registered cameras from database."""
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM camera_registry WHERE enabled=1").fetchall()

            for row in rows:
                config = CameraConfig(
                    camera_id=row["camera_id"],
                    display_name=row["display_name"],
                    device_credential_id=row["device_credential_id"],
                    receiver_url=row["receiver_url"],
                    nas_photo_root=row["nas_photo_root"],
                    enabled=bool(row["enabled"]),
                    registered_at=row["registered_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                )
                self.contexts[config.camera_id] = CameraContext(config)

    def register_camera(
        self,
        camera_id: str,
        display_name: str,
        receiver_url: Optional[str] = None,
        nas_photo_root: Optional[str] = None,
        device_credential_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CameraConfig:
        """
        Register a new camera.

        Args:
            camera_id: Unique camera identifier
            display_name: Human-readable name
            receiver_url: Optional receiver URL
            nas_photo_root: Optional NAS photo directory
            device_credential_id: Optional device credential reference
            metadata: Optional metadata dict

        Returns:
            Camera configuration

        Raises:
            ValueError: If camera already registered
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", camera_id):
            raise ValueError("Invalid camera ID")
        with self.db.transaction("IMMEDIATE") as conn:
            existing = conn.execute(
                "SELECT camera_id FROM camera_registry WHERE camera_id=?", (camera_id,)
            ).fetchone()

            if existing:
                raise ValueError(f"相机 {camera_id} 已注册")

            now = datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            conn.execute(
                """INSERT INTO camera_registry
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    camera_id,
                    display_name,
                    device_credential_id,
                    receiver_url,
                    nas_photo_root,
                    1,
                    now,
                    metadata_json,
                ),
            )

            config = CameraConfig(
                camera_id=camera_id,
                display_name=display_name,
                device_credential_id=device_credential_id,
                receiver_url=receiver_url,
                nas_photo_root=nas_photo_root,
                enabled=True,
                registered_at=now,
                metadata=metadata,
            )

            self.contexts[camera_id] = CameraContext(config)
            return config

    def get_camera(self, camera_id: str) -> Optional[CameraContext]:
        """Get camera context by ID."""
        return self.contexts.get(camera_id)

    def get_or_register_legacy(
        self, camera_id: str, display_name: Optional[str] = None
    ) -> CameraContext:
        """
        Get camera or auto-register (for migration from single-camera setup).

        Args:
            camera_id: Camera identifier
            display_name: Optional display name

        Returns:
            Camera context
        """
        context = self.contexts.get(camera_id)
        if context:
            return context

        # Auto-register for backward compatibility
        config = self.register_camera(
            camera_id=camera_id,
            display_name=display_name or camera_id,
            metadata={"auto_registered": True},
        )
        return self.contexts[camera_id]

    def list_cameras(self) -> list[CameraConfig]:
        """List all registered cameras."""
        return [ctx.config for ctx in self.contexts.values()]

    def disable_camera(self, camera_id: str):
        """Disable a camera (stops accepting new tasks)."""
        with self.db.transaction() as conn:
            conn.execute("UPDATE camera_registry SET enabled=0 WHERE camera_id=?", (camera_id,))

        if camera_id in self.contexts:
            self.contexts[camera_id].config.enabled = False

    def enable_camera(self, camera_id: str):
        """Re-enable a disabled camera."""
        with self.db.transaction() as conn:
            conn.execute("UPDATE camera_registry SET enabled=1 WHERE camera_id=?", (camera_id,))

        if camera_id in self.contexts:
            self.contexts[camera_id].config.enabled = True

    def update_camera_session(self, camera_id: str, session_id: str):
        """Update camera's current device service session."""
        context = self.get_or_register_legacy(camera_id)
        context.update_session(session_id)

    def get_camera_bindings(self, camera_id: str) -> list[dict]:
        """Get all active bindings for a camera."""
        context = self.contexts.get(camera_id)
        if not context:
            return []
        return list(context.active_bindings.values())

    def refresh_bindings_from_db(self):
        """Refresh active bindings from database (call on startup)."""
        for context in self.contexts.values():
            context.active_bindings.clear()
        with self.db.connection() as conn:
            rows = conn.execute("""
                SELECT id, camera, document
                FROM bindings
                WHERE ended IS NULL
            """).fetchall()

            for row in rows:
                camera_id = row["camera"]
                context = self.contexts.get(camera_id)
                if context:
                    binding_data = json.loads(row["document"])
                    context.add_binding(row["id"], binding_data)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_cameras": len(self.contexts),
            "enabled_cameras": sum(1 for c in self.contexts.values() if c.config.enabled),
            "cameras_with_bindings": sum(1 for c in self.contexts.values() if c.active_bindings),
            "total_active_bindings": sum(len(c.active_bindings) for c in self.contexts.values()),
        }

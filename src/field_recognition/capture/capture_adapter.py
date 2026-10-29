"""Unified capture event abstraction for different sources."""

import hashlib
from pathlib import Path
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, AwareDatetime


class CaptureSource(str, Enum):
    """Capture source types."""

    RECEIVER = "receiver"  # Live receiver stream
    NAS_PHOTO = "nas_photo"  # NAS voice photo
    MANUAL_UPLOAD = "manual_upload"  # Web upload
    AGENT = "agent"  # Agent photo


class CaptureEvent(BaseModel):
    """Unified capture event from any source."""

    # Identity
    capture_id: str = Field(description="Unique capture identifier")
    camera_id: str = Field(description="Camera that captured this")
    source: CaptureSource = Field(description="Source of this capture")

    # Session context
    session_id: Optional[str] = Field(default=None, description="Device service session")
    burst_id: Optional[str] = Field(default=None, description="Burst/连拍 identifier")

    # Timing
    captured_at: AwareDatetime = Field(description="When photo was captured")
    received_at: AwareDatetime = Field(description="When system received it")

    # File reference
    source_ref: str = Field(description="Source file path or reference")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    content_hash: Optional[str] = Field(default=None, description="SHA256 hash")

    # Completion status
    write_completed: bool = Field(default=True, description="Write transaction completed")
    verified: bool = Field(default=False, description="Content verified")

    # Metadata
    metadata: dict = Field(default_factory=dict, description="Source-specific metadata")


class CaptureAdapter:
    """Base adapter for capture sources."""

    def should_process(self, event: CaptureEvent) -> bool:
        """Check if this event should be processed."""
        return event.write_completed

    def get_image_data(self, event: CaptureEvent) -> bytes:
        """Get image data from source reference."""
        raise NotImplementedError

    def mark_processed(self, event: CaptureEvent):
        """Mark event as processed (optional)."""
        pass


class NASPhotoAdapter(CaptureAdapter):
    """Adapter for NAS voice photos."""

    def __init__(self, nas_root: str):
        if not nas_root:
            raise ValueError("NAS source root is required")
        self.nas_root = nas_root

    def path(self, event):
        from field_recognition.api.security import require_camera

        require_camera(event.camera_id)
        root = (Path(self.nas_root) / event.camera_id).resolve()
        path = Path(event.source_ref)
        if not path.is_absolute():
            path = Path(self.nas_root) / path
        path = path.resolve()
        path.relative_to(root)
        return path

    def get_image_data(self, event: CaptureEvent) -> bytes:
        """Read image from NAS path."""

        path = self.path(event)
        before = path.stat()
        if before.st_size > 16 * 1024 * 1024:
            raise ValueError("Photo exceeds limit")
        data = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Photo changed while reading")
        if event.content_hash and hashlib.sha256(data).hexdigest() != event.content_hash:
            raise ValueError("Photo SHA-256 mismatch")
        return data

    def should_process(self, event: CaptureEvent) -> bool:
        """Check if file is stable and complete."""
        if not event.write_completed:
            return False

        import time

        path = self.path(event)

        # Check file hasn't changed recently (stability check)
        if path.exists():
            mtime = path.stat().st_mtime
            age = time.time() - mtime
            return age > 0.5  # 500ms stability

        return False


class ReceiverAdapter(CaptureAdapter):
    """Adapter for receiver stream captures."""

    def __init__(self, receiver_url: str):
        self.receiver_url = receiver_url

    def get_image_data(self, event: CaptureEvent) -> bytes:
        """Image data already in metadata."""
        data = event.metadata.get("image_data") or event.metadata.get("image_base64")
        if isinstance(data, str):
            import base64

            data = base64.b64decode(data, validate=True)
        if data is None:
            raise ValueError("Receiver capture missing image_data in metadata")
        return data

    def should_process(self, event: CaptureEvent) -> bool:
        """Receiver frames are immediately ready."""
        return True


class ManualUploadAdapter(CaptureAdapter):
    """Adapter for web uploads."""

    def get_image_data(self, event: CaptureEvent) -> bytes:
        """Image data in metadata or read from temp file."""
        data = event.metadata.get("image_data") or event.metadata.get("image_base64")
        if isinstance(data, str):
            import base64

            data = base64.b64decode(data, validate=True)
        if data:
            return data

        raise ValueError(
            "Uploads must provide image_data; arbitrary filesystem reads are forbidden"
        )


def create_adapter(source: CaptureSource, **kwargs) -> CaptureAdapter:
    """Factory to create appropriate adapter."""
    if source == CaptureSource.NAS_PHOTO:
        return NASPhotoAdapter(kwargs.get("nas_root"))
    elif source == CaptureSource.RECEIVER:
        return ReceiverAdapter(kwargs.get("receiver_url"))
    elif source == CaptureSource.MANUAL_UPLOAD:
        return ManualUploadAdapter()
    elif source == CaptureSource.AGENT:
        return ManualUploadAdapter()  # Same as manual upload
    else:
        raise ValueError(f"Unknown source: {source}")


def ingest_event(core, event):
    """Feed a normalized event into the existing idempotent photo/OCR path."""
    import base64
    from field_recognition.api.security import require_camera

    require_camera(event.camera_id)
    adapter = create_adapter(
        event.source, nas_root=__import__("os").environ.get("FIELD_SAVED_PHOTO_ROOT")
    )
    if not adapter.should_process(event):
        raise ValueError("Capture is not complete")
    raw = adapter.get_image_data(event)
    digest = hashlib.sha256(raw).hexdigest()
    if event.content_hash and digest != event.content_hash:
        raise ValueError("Capture SHA-256 mismatch")
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("Capture exceeds size limit")
    photo = {
        "capture_id": event.capture_id,
        "camera_id": event.camera_id,
        "captured_at": event.captured_at.isoformat(),
        "source_ref": event.source_ref,
        "image_base64": base64.b64encode(raw).decode(),
        "sha256": digest,
    }
    measurement = dict(event.metadata.get("measurement") or {})
    if event.burst_id:
        measurement["burst_id"] = event.burst_id
    request = core["SavedPhotoRequest"](photo=photo, measurement=measurement)
    return core["read_saved_panel"](request, trigger="capture_event")

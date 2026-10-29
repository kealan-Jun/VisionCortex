from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path


def load_env(path: Path) -> None:
    """Read a local configuration file as data, never shell code."""
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.replace("_", "").isalnum():
            raise ValueError("Invalid environment configuration line")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    runtime_root: Path
    customer_code: str
    token: str = field(default="", repr=False)
    poll_seconds: float = 1.0
    cache_max_bytes: int = 2 * 1024**3
    max_capture_bytes: int = 512 * 1024**2
    lease_seconds: int = 300
    model_manifest: Path | None = None
    source_kind: str = "customer_capture"
    source_root: Path | None = None
    nas_marker_sha256: str | None = None
    archive_max_bytes: int = 2 * 1024**3
    allow_external_inference: bool = False
    cache_ttl_seconds: int = 24 * 3600
    max_native_bytes: int = 16 * 1024**3
    photo_batch_mode: str = 'receipt_only'
    photo_cameras: tuple[str, ...] = ()
    photo_timezone: str = 'Asia/Shanghai'
    live_sources_manifest: Path | None = None
    unified_archive_required: bool = False
    processing_profile: str = 'native'
    timeline_policy_path: Path | None = None
    timeline_providers_path: Path | None = None
    timeline_publish: bool = False

    def __post_init__(self):
        if self.processing_profile not in {'native', 'timeline-v1'}:
            raise ValueError('Unknown VC Lab processing profile')

    @classmethod
    def from_env(cls) -> "Settings":
        dsn = os.environ.get("VC_LAB_DATABASE_URL", "")
        if not dsn.startswith(("postgresql://", "postgres://")):
            raise ValueError("VC_LAB_DATABASE_URL must identify the dedicated PostgreSQL database")
        code = os.environ.get("VC_LAB_CUSTOMER_CODE", "").strip()
        if not code:
            raise ValueError("Explicit VC_LAB_CUSTOMER_CODE is required")
        root = Path(os.environ.get("VC_LAB_RUNTIME_ROOT", ".runtime")).expanduser().absolute()
        # SMB roots are not accepted for live runtime state.
        if str(root).startswith(("/mnt/", "/media/")):
            raise ValueError("Active runtime state must be on local storage")
        manifest = os.environ.get("VC_LAB_MODEL_MANIFEST")
        source = os.environ.get("VC_LAB_SOURCE_ROOT")
        live = os.environ.get('VC_LAB_LIVE_SOURCES_MANIFEST')
        return cls(dsn, root, code, os.environ.get("VC_LAB_API_TOKEN", ""),
                   model_manifest=Path(manifest).absolute() if manifest else None,
                   source_root=Path(source).absolute() if source else None,
                   nas_marker_sha256=os.environ.get("VC_LAB_NAS_MARKER_SHA256"),
                   allow_external_inference=os.environ.get("VC_LAB_ALLOW_EXTERNAL_INFERENCE") == "1",
                   photo_batch_mode=os.environ.get('VC_LAB_PHOTO_BATCH_MODE', 'receipt_only'),
                   photo_cameras=tuple(json.loads(os.environ.get('VC_LAB_PHOTO_CAMERAS', '[]'))),
                   live_sources_manifest=Path(live).absolute() if live else None,
                   unified_archive_required=os.environ.get('VC_LAB_UNIFIED_ARCHIVE_REQUIRED') == '1',
                   processing_profile=os.environ.get('VC_LAB_PROCESSING_PROFILE', 'native'),
                   timeline_policy_path=Path(os.environ['VC_LAB_TIMELINE_POLICY']).absolute() if os.environ.get('VC_LAB_TIMELINE_POLICY') else None,
                   timeline_providers_path=Path(os.environ['VC_LAB_TIMELINE_PROVIDERS']).absolute() if os.environ.get('VC_LAB_TIMELINE_PROVIDERS') else None,
                   timeline_publish=os.environ.get('VC_LAB_TIMELINE_PUBLISH') == '1')

    @property
    def incoming(self) -> Path:
        from .storage import verify_root, is_nas
        root = verify_root(self.source_root, self.nas_marker_sha256) if self.source_root else self.runtime_root / "incoming"
        if not is_nas(root) and not root.is_relative_to(self.runtime_root.absolute()):
            raise ValueError('Local input/archive roots must stay inside the bounded runtime root')
        if self.unified_archive_required:
            from .archive_layout import verify_archive_contract
            verify_archive_contract(root)
        return root

    @property
    def cache(self) -> Path:
        return self.runtime_root / "cache"

    def prepare(self) -> None:
        from .storage import is_nas
        if is_nas(self.runtime_root):
            raise ValueError('Active runtime state must use local storage')
        for path in (self.runtime_root, self.incoming, self.cache):
            path.mkdir(parents=True, exist_ok=True)

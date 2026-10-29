"""Versioned, optional ports around the unchanged original LiveScanner.

Use ``app_factory`` only for an explicitly configured source process, or attach
to a not-yet-started local runtime. Existing FR launchers remain unchanged.
No class monkeypatch, duplicate scanning algorithm, or new HTTP protocol is used.
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from types import ModuleType


def _resource_factory(assets):
    """Three isolated import ports over verified, unchanged original sources."""
    from .components import ROOT, verify_components
    from .media import safe_path
    from field_recognition.runtime.paths import source_path as packaged_path

    verify_components()

    def source_path(name):
        if name.startswith("static/"):
            path = safe_path(assets, name)
        elif name.startswith("field_recognition/"):
            path = safe_path(ROOT / "src", name)
        else:
            path = packaged_path(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024**2:
            raise ValueError(
                "Pinned scanner source/static resource unavailable or oversized"
            )
        return path

    def isolated(relative, import_line, ports):
        path = ROOT / "src/field_recognition" / relative
        source = path.read_text()
        if source.count(import_line) != 1:
            raise ValueError("Frozen scanner builder import port no longer matches")
        module = ModuleType("visioncortex_lab_scan_" + path.stem)
        module.__file__ = str(path)
        module.__dict__.update(ports)
        source = source.replace(
            import_line, "# Explicit isolated scanner builder dependency port."
        )
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
        return module

    archive = isolated(
        "archive/archive_store.py",
        "from field_recognition.runtime.paths import source_path",
        dict(source_path=source_path),
    )
    components = isolated(
        "runtime/components.py",
        "from field_recognition.archive.archive_store import ArchiveStore",
        dict(ArchiveStore=archive.ArchiveStore),
    )
    factory = isolated(
        "runtime/factory.py",
        "from field_recognition.runtime.components import install_components",
        dict(install_components=components.install_components),
    )
    return factory.create_runtime


class _FramePort:
    def __init__(self, camera):
        self.camera = camera
        self.local = threading.local()

    def __getattr__(self, name):
        return getattr(self.camera, name)

    def frame(self):
        # Original preview and scanner threads can share a camera. Only the
        # calling thread's frame may accompany that thread's decoded QR values.
        self.local.current = None
        frame, metadata = self.camera.frame()
        self.local.current = (frame, dict(metadata), time.monotonic())
        return frame, metadata


def attach_to_scanner(scanner, publisher):
    from field_recognition.capture.live_scan import ACTIVE, LiveScanner

    if not isinstance(scanner, LiveScanner):
        raise ValueError(
            "Scan source requires the owning local LiveScanner, not an RPC proxy"
        )
    with scanner.lock:
        if scanner.session and scanner.session.get("status") in ACTIVE:
            raise ValueError("Attach scan source before starting a manual session")
        if getattr(scanner, "_scan_source_attached", False):
            raise ValueError("Scan source is already attached")
        core = dict(scanner.core)
        camera = core.get("receiver_camera")
        if camera is None:
            raise ValueError("Explicit source camera required for scan publisher")
        port = _FramePort(camera)
        original_matches = core["qr_matches"]

        def matches(values, points):
            current = getattr(port.local, "current", None)
            port.local.current = None
            with scanner.lock:
                session = scanner.session
                if (
                    current is not None
                    and session
                    and session.get("owner") not in {None, "", "automation"}
                    and scanner._running(session["session_id"])
                ):
                    frame, metadata, acquired = current
                    age = metadata.get("frame_age_ms", 0)
                    if type(age) not in {int, float} or age < 0:
                        raise ValueError("Invalid original frame age")
                    if time.monotonic() - acquired + age / 1000 <= 1:
                        source = dict(metadata, source_camera_key=camera.target)
                        result = publisher.publish(
                            frame, source, copy.deepcopy(session), (values, points, {})
                        )
                        session["source_publication"] = result
            return original_matches(values, points)

        core.update(receiver_camera=port, qr_matches=matches)
        scanner.core = core
        scanner._scan_source_attached = True
    return scanner


def attach_to_runtime(runtime):
    from .scan_source import load_publisher_from_environment

    publisher = load_publisher_from_environment()
    if publisher is not None:
        attach_to_scanner(runtime["live_scanner"], publisher)
    return runtime


def create_runtime(*args, **kwargs):
    """Explicit source factory; construct original runtime, then attach ports."""
    from field_recognition.runtime.factory import create_runtime as original

    runtime = original(*args, **kwargs)
    try:
        return attach_to_runtime(runtime)
    except BaseException:
        runtime.close()
        raise


def app_factory():
    """Explicit private assets/state; disabled sources never inherit FIELD_* env."""
    from field_recognition.api.application import create_app
    from field_recognition.runtime.production_config import ProductionConfig
    from .existing_archive_binding import private_json
    from .scan_source import _local

    name = os.environ.get("VC_LAB_SCAN_LAUNCHER_MANIFEST")
    if not name:
        raise ValueError(
            "Explicit private VC_LAB_SCAN_LAUNCHER_MANIFEST required for original FR assets"
        )
    document = private_json(name)
    if (
        set(document)
        != {
            "schema_version",
            "asset_root",
            "asset_index_sha256",
            "runtime_root",
            "max_runtime_bytes",
            "source_enabled",
            "field_config",
        }
        or document.get("schema_version") != "vc-lab-scan-launcher/1"
        or type(document["source_enabled"]) is not bool
        or type(document["max_runtime_bytes"]) is not int
        or not 1024 <= document["max_runtime_bytes"] <= 64 * 1024**2
    ):
        raise ValueError("Explicit bounded scan launcher descriptor required")
    assets, data = _local(document["asset_root"]), _local(document["runtime_root"])
    if data == assets or data.is_relative_to(assets):
        raise ValueError(
            "Scanner state must be independent of the original asset/source root"
        )
    index = assets / "static/index.html"
    if (
        index.is_symlink()
        or index.resolve() != index
        or not index.is_file()
        or index.stat().st_size > 8 * 1024**2
    ):
        raise ValueError("Original local static index is unavailable or unsafe")
    if hashlib.sha256(index.read_bytes()).hexdigest() != document["asset_index_sha256"]:
        raise ValueError("Original static index differs from the private launcher pin")
    options = document["field_config"]
    allowed = {
        "receiver_url",
        "camera_id",
        "auth_enabled",
        "event_tenant",
        "service_role",
        "production_enabled",
        "record_mode",
        "session_duration_hours",
    }
    if type(options) is not dict or set(options) - allowed:
        raise ValueError("Unsupported private source runtime option")
    enabled = document["source_enabled"]
    if not enabled and options:
        raise ValueError("Disabled-source launcher requires an empty field_config")
    if enabled and (
        not options.get("receiver_url")
        or not options.get("camera_id")
        or not os.environ.get("VC_LAB_SCAN_SOURCE_MANIFEST")
    ):
        raise ValueError(
            "Explicit receiver/camera and publisher descriptor required before enabling source"
        )
    if enabled and options.get("service_role", "capture") not in {
        "capture",
        "combined",
    }:
        raise ValueError("Enabled source launcher must own its local scanner")
    if data.exists():
        used, count = 0, 0
        for path in data.rglob("*"):
            count += 1
            if path.is_symlink() or count > 1024:
                raise ValueError(
                    "Scanner runtime contains links or exceeds its bounded inventory"
                )
            if path.is_file():
                used += path.stat().st_size
            if used > document["max_runtime_bytes"]:
                raise ValueError("Scanner runtime exceeded its explicit capacity")
    defaults = dict(
        service_role="capture" if enabled else "api",
        registry_policy="none",
        data_dir=str(data),
        database_path=str(data / "Scanner.sqlite3"),
        ocr_device="cpu",
    )
    defaults.update(options)
    config = ProductionConfig(**defaults)
    runtime = _resource_factory(assets)(config, project_root=assets, data_dir=data)
    try:
        if enabled:
            attach_to_runtime(runtime)
        return create_app(runtime)
    except BaseException:
        runtime.close()
        raise

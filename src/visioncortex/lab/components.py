"""Pinned historical implementations in isolated namespaces; never shadow the host."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAB_COMPONENTS = ROOT / "components/lab"
LAB_CONTRACTS = ROOT / "contracts/lab"
LAB_MIGRATIONS = ROOT / "migrations/lab"
_LOCK = threading.RLock()


def verify_components():
    manifest = json.loads((LAB_COMPONENTS / "manifest.json").read_text())
    for component in manifest["components"]:
        base = ROOT / component.get("root", "components/lab/" + component["name"])
        if not base.resolve().is_relative_to(ROOT):
            raise ValueError("Component root escaped repository resources")
        for item in component["files"]:
            path = base / item["path"]
            if not path.is_file() or not path.resolve().is_relative_to(base.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Component fingerprint mismatch: " + str(path))
    return manifest


def _package(namespace, component):
    with _LOCK:
        if namespace not in sys.modules:
            path = LAB_COMPONENTS / component / "src" / namespace
            spec = importlib.util.spec_from_file_location(namespace, path / "__init__.py", submodule_search_locations=[str(path)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(namespace, None)
                raise
    return namespace


def enable_components():
    # All other imports are normal package imports from this distribution. In
    # particular, no component directory is ever added to sys.path.
    _package("visioncortex_lab_legacy", "visioncortex")
    _package("visioncortex_lab_pipeline", "visioncortex_pipeline")


def legacy_module(name):
    return importlib.import_module(_package("visioncortex_lab_legacy", "visioncortex") + "." + name)


def pipeline_module(name):
    return importlib.import_module(_package("visioncortex_lab_pipeline", "visioncortex_pipeline") + "." + name)


def decode_qr(image):
    return importlib.import_module("field_recognition.capture.qr_decode").decode_qr(image)


def inspect_recorder(root: Path, video: Path, now: float, settle: float):
    return pipeline_module("nas_recordings")._inspect(root, video, now, settle, allow_plain=False)

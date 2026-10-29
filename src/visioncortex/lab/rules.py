"""Verified, versioned existing processing rules. No ambient production configuration."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .components import ROOT, verify_components


class RuleMismatch(ValueError):
    pass


class Rules:
    def __init__(self, root: Path = ROOT):
        self.root = root

    def identity(self):
        raw = (self.root / "contracts/lab/rules.lock.json").read_bytes()
        lock = json.loads(raw)
        manifest = (self.root / "components/lab/manifest.json").read_bytes()
        if hashlib.sha256(manifest).hexdigest() != lock["components_sha256"]:
            raise RuleMismatch("Component release differs from frozen processing rules")
        for item in lock["files"]:
            content = (self.root / "contracts/lab" / item["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise RuleMismatch("Frozen rule profile changed: " + item["path"])
        for item in lock.get("adapter_files", []):
            if hashlib.sha256((self.root / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise RuleMismatch("Processing adapter changed without a release revision: " + item["path"])
        if self.root == ROOT:
            verify_components()
        return {"revision": lock["revision"], "sha256": hashlib.sha256(raw).hexdigest(), "components_sha256": lock["components_sha256"]}

    def profile(self, kind):
        self.identity()
        name = {"archive": "archive-layout-v2.json", "vision": "vision-rules.json", "pipeline": "pipeline-rules.json", "native_pipeline": "native-pipeline-rules.json", "qr": "field-rules.json", "ocr": "field-rules.json", "asr": "pipeline-rules.json", "understanding": "pipeline-rules.json"}[kind]
        return json.loads((self.root / "contracts/lab" / name).read_text())

    def vision_configuration(self, model, runtime_root):
        """Bind assets/hardware explicitly, preserving all semantic screening settings."""
        profile = self.profile("vision")
        supplied = model.get("runtime_bindings", {})
        missing = set(profile["customer_bindings"]) - set(supplied)
        if missing:
            raise RuleMismatch("Customer vision asset bindings are incomplete: " + ", ".join(sorted(missing)))
        root = Path(model["model_root"])
        for key, spec in profile["customer_bindings"].items():
            value = supplied[key]
            if spec["kind"] == "path":
                path = Path(value)
                if not path.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
                    raise RuleMismatch("Vision asset binding escaped the customer model root: " + key)
                relative = path.relative_to(root).as_posix()
                if path.is_file():
                    if relative not in model["files"]:
                        raise RuleMismatch("Vision file is not in the verified model manifest: " + key)
                elif path.is_dir():
                    actual = {p.relative_to(root).as_posix() for p in path.rglob("*") if p.is_file()}
                    if not actual or not actual.issubset(model["files"]):
                        raise RuleMismatch("Vision directory contains unverified model files: " + key)
                else:
                    raise RuleMismatch("Customer asset is unavailable: " + key)
        def bind(value):
            if isinstance(value, dict):
                if set(value) == {"customer_binding"}:
                    return supplied[value["customer_binding"]]
                return {k: bind(v) for k, v in value.items()}
            if isinstance(value, list):
                return [bind(v) for v in value]
            return value
        config = bind(copy.deepcopy(profile["settings"]))
        # Deployment adaptation is limited to hardware knobs; semantic overrides need a new bundle.
        allowed = {"device", "half", "tensor_rt", "batch_size", "coarse_batch_size", "fine_batch_size", "workers", "decode_workers", "ffmpeg_hwaccel", "ffmpeg_video_encoder", "coarse_decode_lanes", "fine_decode_lanes"}
        hardware = model.get("hardware", {})
        if set(hardware) - allowed:
            raise RuleMismatch("Semantic screening overrides require an explicit rule bundle revision")
        config["performance"].update(hardware)
        config["storage"] = {"local_runtime_root": str(runtime_root), "local_cache_root": str(runtime_root / "cache")}
        config["runtime"] = {"local_only": True}
        config["device_day"].update(delete_capture_sources=False, raw_originals_read_only=True, archive_html_enabled=False)
        return config

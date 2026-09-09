"""Dependency-aware execution with data-only, identity-bound recovery points.

Checkpoints are an execution optimization, never a quality acceptance receipt.
Source identity follows the existing sealed-input/stat contract; this module does
not claim a second full-video hash or silently reuse a historical legacy run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from pydantic import BaseModel

from . import schemas
from .archive import write_json
from .candidate_index import CoarseFrameIndex, FineFrameIndex


# Order is for scheduling, dependencies are the actual failure boundary. In
# particular, clip encoding and key material analysis are independent siblings.
DEPENDENCIES = {
    "preflight": (),
    "capture_quality": ("preflight",),
    "alignment": ("preflight",),
    "speech": ("preflight", "alignment"),
    "motion_probe": ("preflight", "alignment"),
    "candidate_coarse": ("motion_probe",),
    "candidate_fine": ("candidate_coarse",),
    "candidate_audit": ("candidate_fine",),
    "experiment_understanding": ("candidate_audit",),
    "experiment_clips": ("experiment_understanding",),
    "key_materials": ("experiment_understanding",),
    "mllm": ("key_materials",),
    "material_refinement": ("mllm",),
    "semantic_refinement": ("material_refinement",),
    "package": ("semantic_refinement", "experiment_clips"),
    "daily_report": ("package",),
    "professional_pdf": ("daily_report",),
    "finalizing": ("daily_report", "professional_pdf"),
}

# Only explicit cross-stage data is retained. No Python frames, closures,
# credentials, model runtimes, executable pickle or imported classes are saved.
OUTPUTS = {
    "preflight": "infos disk_report",
    "capture_quality": "",
    "alignment": "transforms",
    "speech": "",
    "motion_probe": "coarse_frame_index coarse_index_report coarse_full_timeline motion_candidates motion_paths motion_probe_views motion_windows preselected_coarse_views shared_coarse_motion_scan",
    "candidate_coarse": "coarse_frame_index coarse_index_report boundary_candidates fine_view_report progressive_enabled fine_views",
    "candidate_fine": "detection_paths exhaustive_full_timeline fine_runtime_index progressive_report scanned_fine_ids candidates fine_views fine_view_report",
    "candidate_audit": "events rejected segments groups key_events observability_path semantic_review_plan_path state_machine_path boundary_precheck",
    "experiment_understanding": "groups segments events key_events group_understanding_path boundary_precheck",
    "experiment_clips": "groups segments",
    "key_materials": "key_events groups",
    "mllm": "key_events reviewed_key_events groups",
    "material_refinement": "key_events reviewed_key_events groups segments semantic_curation",
    "semantic_refinement": "groups key_events reviewed_key_events segments",
    "package": "summary quality_attention",
    "daily_report": "",
    "professional_pdf": "",
    "finalizing": "summary",
}
MODEL_TYPES = {name: getattr(schemas, name) for name in (
    "RunManifest", "ViewInput", "VideoInfo", "AlignmentTransform", "ActionCandidate",
    "EvidenceEvent", "ExperimentSegment", "ExperimentGroup", "RunSummary",
)}
INDEX_TYPES = {"CoarseFrameIndex": CoarseFrameIndex, "FineFrameIndex": FineFrameIndex}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def call_identity(call):
    fields = ("provider", "model", "request_id") if call.get("request_id") else ("provider", "model", "input_fingerprint", "stage", "event_id", "group_id", "usage")
    return digest({k: call.get(k) for k in fields})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise ValueError("Invalid recovery artifact path")
    path.resolve().relative_to(root.resolve())
    return path


def encode(value, seen=None):
    """Preserve shared event identities without executing serialized code."""
    seen = {} if seen is None else seen
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, (CoarseFrameIndex, FineFrameIndex)):
        return {"kind": "index", "name": type(value).__name__, "path": str(value.path)}
    if id(value) in seen:
        return {"ref": seen[id(value)]}
    key = len(seen)
    seen[id(value)] = key
    if isinstance(value, BaseModel) and type(value).__name__ in MODEL_TYPES:
        return {"id": key, "kind": "model", "name": type(value).__name__, "value": value.model_dump(mode="json")}
    if isinstance(value, dict):
        return {"id": key, "kind": "dict", "value": [[encode(k, seen), encode(v, seen)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {"id": key, "kind": type(value).__name__, "value": [encode(v, seen) for v in value]}
    raise ValueError(f"Unsupported recovery data: {type(value).__name__}")


def decode(value, seen=None):
    seen = {} if seen is None else seen
    if not isinstance(value, dict):
        return value
    if "ref" in value:
        return seen[value["ref"]]
    kind = value["kind"]
    if kind == "path":
        return Path(value["value"])
    if kind == "index":
        return INDEX_TYPES[value["name"]](Path(value["path"]))
    if kind == "model":
        result = MODEL_TYPES[value["name"]].model_validate(value["value"])
    elif kind == "dict":
        result = {decode(k, seen): decode(v, seen) for k, v in value["value"]}
    elif kind in {"list", "tuple", "set"}:
        result = {"list": list, "tuple": tuple, "set": set}[kind](decode(v, seen) for v in value["value"])
    else:
        raise ValueError("Unknown recovery data type")
    seen[value["id"]] = result
    return result


def index(root: Path) -> dict:
    try:
        value = json.loads((root / "JSON-Config-Files/Recovery/index.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and value.get("schema_version") == "visioncortex-recovery/1" else {}
    except (OSError, ValueError):
        return {}


class StageRunner:
    def __init__(self, pipeline, layout, manifest, cache_identity):
        self.pipeline, self.layout, self.manifest = pipeline, layout, manifest
        self.context = SimpleNamespace(early_result=None, quality_attention=False)
        self.outcomes = {}
        self.errors = []
        self.request_keys = {}
        self.reused_call_ids = set()
        self.saved = index(layout.root).get("stages", {})
        self.resume = bool(pipeline.config.get("project", {}).get("resume_stages"))
        self.identity = cache_identity
        self.code = digest({p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
                            if p.name not in {"api.py", "cli.py"}})
        self._hashes = {}
        self.initial_config = deepcopy(pipeline.config)
        self.initial_config.get("project", {}).pop("resume_stages", None)
        self.initial_config.get("project", {}).pop("semantic_recovery_attempt", None)
        self.initial_config.get("project", {}).pop("cache_mode", None)
        self.initial_config.get("project", {}).pop("semantic_cache_mode", None)

    def request_key(self, stage):
        config = deepcopy(self.initial_config)
        # No secret values are persisted; only the irreversible identity digest.
        if stage in {"preflight", "alignment", "motion_probe", "candidate_coarse", "candidate_fine", "candidate_audit"}:
            for name in ("mllm", "speech_recognition", "daily_report", "capture_quality"):
                config.pop(name, None)
        optional = ("speech", "capture_quality") if stage == "experiment_understanding" else ()
        deps = {name: [self.request_keys.get(name), self.outcomes.get(name)] for name in (*DEPENDENCIES[stage], *optional)}
        return digest({"schema": 1, "stage": stage, "code": self.code,
                       "config": config, "manifest": self.manifest.model_dump(mode="json"),
                       "inputs": self.identity.get("inputs"), "models": self.identity.get("models"),
                       "dependencies": deps})

    def _state(self, stage, context):
        names = set(OUTPUTS[stage].split())
        names.update("events segments groups key_events reviewed_key_events".split())
        fields = {name: getattr(context, name) for name in sorted(names) if hasattr(context, name)}
        return {"context": fields, "performance": self.pipeline.config.get("performance", {}),
                "preprocessing_seconds": self.pipeline._preprocessing_completed_seconds,
                "certification": self.pipeline._model_certification_audit,
                "speech_understanding": self.pipeline._speech_understanding}

    def _version(self, receipt):
        path = safe_path(self.layout.root, receipt["version_manifest"])
        value = json.loads(path.read_text(encoding="utf-8"))
        for item in value["files"]:
            if item.get("snapshot"):
                if sha256(safe_path(self.layout.root, item["snapshot"])) != item["sha256"]:
                    raise ValueError("Recovery metadata was modified")
            else:
                stat = safe_path(self.layout.root, item["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
                    raise ValueError("Retained media changed or is missing")
        return value

    def _external_records(self, state):
        # These local ledgers/indexes are needed by downstream code and are not
        # media. Bind their complete content; don't accept mere path existence.
        found = {}
        def visit(value):
            if isinstance(value, (CoarseFrameIndex, FineFrameIndex)):
                visit(value.path)
            elif isinstance(value, Path) and value.is_file():
                # The runtime ledgers may live in .work inside the archive as
                # well as in the external cache. Bind both, without hashing video.
                if value.suffix.lower() in {".json", ".jsonl", ".sqlite", ".sqlite3", ".db", ".csv"}:
                    stat = value.stat()
                    key = (str(value), stat.st_size, stat.st_mtime_ns)
                    if key not in self._hashes:
                        self._hashes[key] = sha256(value)
                    found[str(value)] = self._hashes[key]
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
        visit(state["context"])
        return found

    def reuse(self, stage, key):
        item = self.saved.get(stage, {})
        if not self.resume or stage in {"preflight", "package", "finalizing"} or item.get("request_key") != key:
            return False
        try:
            path = safe_path(self.layout.root, item["path"])
            if sha256(path) != item["sha256"]:
                return False
            payload = json.loads(path.read_text(encoding="utf-8"))
            for raw, expected in payload["external_records"].items():
                file = Path(raw)
                allowed = [self.layout.root.resolve()] + [Path(self.pipeline.config["storage"][name]).resolve()
                           for name in ("local_cache_root", "local_runtime_root")]
                if not any(file.resolve().is_relative_to(root) for root in allowed) or sha256(file) != expected:
                    return False
            version_path = safe_path(self.layout.root, payload["receipt"]["version_manifest"])
            if sha256(version_path) != payload["version_sha256"]:
                return False
            version = self._version(payload["receipt"])
            state = decode(payload["state"])
            # All dependencies were verified before replacing current metadata.
            for record in version["files"]:
                if record.get("snapshot") and Path(record["path"]).name not in {"cache_identity.json", "run_manifest.json", "input_manifest.yaml"}:
                    destination = safe_path(self.layout.root, record["path"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(safe_path(self.layout.root, record["snapshot"]), destination)
            vars(self.context).update(state["context"])
            self.pipeline.config["performance"] = state["performance"]
            self.pipeline._preprocessing_completed_seconds = state["preprocessing_seconds"]
            self.pipeline._model_certification_audit = state["certification"]
            if stage == "speech":
                self.pipeline._speech_understanding = state["speech_understanding"]
            receipt = dict(payload["receipt"], reused=True,
                           original_completed_at=payload["receipt"]["completed_at"],
                           completed_at=datetime.now(timezone.utc).isoformat())
            write_json(self.layout.json_config / "Stage-Receipts" / f"{stage}.json", receipt)
            self.pipeline._stage_outcomes[stage] = {"status": receipt["status"], "reason": "已校验并复用此前完成的产出", "reused": True}
            self.outcomes[stage] = receipt["status"]
            for call in self.pipeline._metrics(getattr(self.context, "events", []), getattr(self.context, "groups", [])).get("mllm_calls", []):
                self.reused_call_ids.add(call_identity(call))
            if stage == "candidate_audit":
                self.pipeline._preprocessing_completed_seconds = time.perf_counter() - self.pipeline._run_started_perf
            if self.pipeline._publisher is not None:
                for artifact in receipt["artifacts"]:
                    source = safe_path(self.layout.root, artifact)
                    if source.is_dir():
                        self.pipeline._publisher.publish_directory(artifact)
                    else:
                        self.pipeline._publisher.publish_file(source)
                self.pipeline._publisher.publish_file(self.layout.json_config / "Stage-Receipts" / f"{stage}.json")
            self._inventory()
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def save(self, stage, key, context):
        receipt_path = self.layout.json_config / "Stage-Receipts" / f"{stage}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] not in {"completed", "skipped"}:
            return
        state = self._state(stage, context)
        path = self.layout.json_config / "Recovery" / f"{uuid4().hex[:16]}.json"
        write_json(path, {"schema_version": "visioncortex-stage-checkpoint/1", "stage": stage,
                          "request_key": key, "receipt": receipt, "state": encode(state),
                          "version_sha256": sha256(safe_path(self.layout.root, receipt["version_manifest"])),
                          "external_records": self._external_records(state), "formal_release": False})
        self.saved[stage] = {"request_key": key, "path": path.relative_to(self.layout.root).as_posix(), "sha256": sha256(path)}
        write_json(path.parent / "index.json", {"schema_version": "visioncortex-recovery/1", "stages": self.saved})

    def _inventory(self):
        items = []
        for name in self.outcomes:
            path = self.layout.json_config / "Stage-Receipts" / f"{name}.json"
            if path.is_file():
                items.append(json.loads(path.read_text(encoding="utf-8")))
        inventory = self.layout.root / "阶段产出清单.json"
        write_json(inventory, {"schema_version": "visioncortex-stage-outputs/1", "formal_release": False, "stages": items})
        if self.pipeline._publisher is not None:
            self.pipeline._publisher.publish_file(inventory)

    def flush_archive(self):
        publisher = self.pipeline._publisher
        if not hasattr(publisher, "flush"):
            return True
        if not publisher.flush():
            self.pipeline._complete_stage(self.layout, "archive_sync", [], status="failed",
                                          reason="本地产物已保留，指定 NAS 目录暂时无法完成归档")
            return False
        # Immutable stage versions retain the original delivery state. Only the
        # current receipt changes after all queued artifacts were delivered.
        for path in (self.layout.json_config / "Stage-Receipts").glob("*.json"):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("archive_status") == "pending":
                receipt["archive_status"] = "saved"
                write_json(path, receipt)
                publisher.publish_file(path)
                self.pipeline._stage_outcomes.get(receipt["stage"], {})["archive_status"] = "saved"
        self._inventory()
        if not publisher.flush():
            return False
        if (self.layout.json_config / "Stage-Receipts/archive_sync.json").exists():
            self.pipeline._complete_stage(self.layout, "archive_sync", [], reason="待归档产物已同步至指定 NAS 目录")
        return not publisher.pending

    def run(self):
        for number, (stage, dependencies) in enumerate(DEPENDENCIES.items()):
            blocked = [d for d in dependencies if self.outcomes.get(d) not in {"completed", "skipped"}]
            if stage == "package" and self.pipeline.config.get("speech_recognition", {}).get("required") and self.outcomes.get("speech") == "failed":
                blocked.append("speech")
            if stage in {"daily_report", "professional_pdf", "finalizing"} and self.context.quality_attention:
                blocked.append("quality_acceptance")
            if stage == "finalizing" and not self.flush_archive():
                blocked.append("archive_sync")
                self.errors.append(("archive_sync", OSError("计算产物已保存，NAS 归档仍待恢复；无需重新运行已完成计算")))
            if blocked:
                self.outcomes[stage] = "blocked"
                self.pipeline._complete_stage(self.layout, stage, [], status="blocked", reason="等待必要环节：" + ", ".join(blocked))
                continue
            key = self.request_key(stage)
            self.request_keys[stage] = key
            # An invalidated parent invalidates its children even if its inputs
            # are unchanged (e.g. a missing frame that had to be regenerated).
            if any(self.pipeline._stage_outcomes.get(d, {}).get("recomputed") for d in dependencies if d != "preflight"):
                self.saved.pop(stage, None)
            if self.reuse(stage, key):
                continue
            self.pipeline._status(self.layout, stage, number / len(DEPENDENCIES), "正在处理当前环节")
            working = deepcopy(self.context)
            speech_config = deepcopy(self.pipeline.config.get("speech_recognition", {}))
            if stage in {"experiment_understanding", "mllm", "material_refinement", "semantic_refinement"} and self.outcomes.get("speech") in {"failed", "skipped", "blocked"}:
                self.pipeline.config.setdefault("speech_recognition", {})["enabled"] = False
            try:
                getattr(self.pipeline, "_stage_" + stage)(working, self.layout, self.manifest)
                self.context = working
                outcome = self.pipeline._stage_outcomes.get(stage, {}).get("status", "completed")
                self.outcomes[stage] = outcome
                if working.early_result:
                    return working.early_result
                if stage == "package" and working.quality_attention:
                    self.pipeline._complete_stage(self.layout, stage, [self.layout.json_config / "quality_acceptance.json"], status="completed", reason="质量检查已完成，证据不足；完整报告未发布")
                if outcome in {"completed", "skipped"}:
                    try:
                        self.save(stage, key, working)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        write_json(self.layout.json_config / "recovery_warning.json", {"stage": stage, "error_type": type(exc).__name__, "message": "本环节产出保留，但恢复点未保存；再次恢复时须重算此环节"})
                self.pipeline._stage_outcomes.setdefault(stage, {})["recomputed"] = True
            except Exception as exc:
                if stage != "capture_quality" and (stage != "speech" or self.pipeline.config.get("speech_recognition", {}).get("required")):
                    self.errors.append((stage, exc))
                self.outcomes[stage] = "failed"
                self.pipeline._complete_stage(self.layout, stage, [], status="failed", reason=f"{type(exc).__name__}: {exc}")
                if stage in {"experiment_understanding", "mllm", "material_refinement", "semantic_refinement"}:
                    self.pipeline._failed_stage_metrics = self.pipeline._metrics(getattr(working, "events", []), getattr(working, "groups", []))
                    self.pipeline._retained_failed_calls = self.pipeline._failed_stage_metrics.get("mllm_calls", [])
            finally:
                self.pipeline.config["speech_recognition"] = speech_config
            self._inventory()
        if self.errors:
            try:
                from .partial_delivery import write_partial_delivery
                metrics = self.pipeline._metrics(getattr(self.context, "events", []), getattr(self.context, "groups", []))
                write_partial_delivery(self.layout.root, metrics)
                self.pipeline._complete_stage(self.layout, "stage_report", [self.layout.root / "Partial-Results", self.layout.json_config / "partial_delivery.json"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.pipeline._complete_stage(self.layout, "stage_report", [], status="failed", reason=f"{type(exc).__name__}: {exc}")
        if self.errors:
            self.pipeline._active_stage = self.errors[0][0]
            raise self.errors[0][1]
        if self.context.quality_attention:
            return self.pipeline._finish_quality_attention(self.layout, getattr(self.context, "events", []), getattr(self.context, "groups", []))
        if not self.flush_archive():
            raise OSError("阶段产物已保存，NAS 归档仍待恢复")
        self.pipeline._status(self.layout, "completed", 1.0, "处理完成并通过自动发布前验收")
        self.pipeline._complete_stage(self.layout, "completed", [self.layout.json_config])
        if not self.pipeline.config["archive"].get("keep_debug_candidates") and self.layout.work.exists():
            shutil.rmtree(self.layout.work)
        return self.layout.root

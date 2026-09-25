"""Durable, independently resumable device/day stages.

The capture deletion function is deliberately disconnected from the runner.
Production execution never enables deletion while the user's suspension is active.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .device_day_contract import (
    DEPENDENCIES, DIRECTORIES, STAGES, VERSION, DeviceDayLayout,
    atomic_bytes, atomic_json, digest, file_hash, read_json, safe_child,
    validate_archive_name, validate_config, validate_segment, verify_artifact,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def visual_input(retention):
    record = retention.get("recording") or {}
    return {"recording": {k: record.get(k) for k in ("recording_id", "camera_key", "configured_role",
                                                   "recording_start_us", "recording_end_us", "capture_complete")},
            "sources": [s["retained"] for s in retention.get("sources", []) if s["kind"] in {"video", "clock"}]}


@contextmanager
def exclusive(path: Path):
    """OS-owned nonblocking lock; process death releases it on Linux and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            yield
        finally:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def copy_verified(source: Path, target: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise ValueError("Only completed regular capture files may be retained")
    before = source.stat()
    identity = (before.st_size, before.st_mtime_ns)
    checksum = None
    if target.exists():
        checksum = file_hash(source)
        if target.is_symlink() or target.stat().st_size != before.st_size or file_hash(target) != checksum:
            raise ValueError("Immutable retained source collision")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
        try:
            with source.open("rb") as original, temporary.open("xb") as output:
                # Hash the bytes while copying: do not reread the entire NAS
                # source before copying it. The independent target hash below
                # still proves durable byte equality before publication.
                import hashlib
                source_digest = hashlib.sha256()
                while block := original.read(1024 * 1024):
                    source_digest.update(block)
                    output.write(block)
                checksum = source_digest.hexdigest()
                output.flush()
                os.fsync(output.fileno())
            if file_hash(temporary) != checksum:
                raise ValueError("Retained source checksum mismatch")
            copied_source = source.stat()
            if identity != (copied_source.st_size, copied_source.st_mtime_ns):
                raise ValueError("Capture source changed during retention")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    after = source.stat()
    if identity != (after.st_size, after.st_mtime_ns):
        raise ValueError("Capture source changed during retention")
    return {"original_path": str(source), "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "sha256": checksum}


def delete_capture_video(archive: Path, retention: dict, vision: dict, *, enabled=False, backend_root: Path | None = None) -> dict:
    """Prepared cleanup operation. Runtime never passes enabled=True.

    Tests may exercise this only with owned temporary files. No recursive removal,
    metadata/audio deletion, or existence-only acceptance is permitted.
    """
    if not enabled:
        return {"status": "disabled_by_user", "deleted": False}
    if retention.get("status") != "completed" or vision.get("status") != "completed":
        raise ValueError("Retention and vision must both be complete")
    if vision.get("retention_digest") != digest(retention):
        raise ValueError("Vision receipt does not bind this retained source")
    references = [*retention.get("artifacts", []), *vision.get("artifacts", [])]
    if not references or not vision.get("artifacts") or not all(verify_artifact(archive, r) for r in references):
        raise ValueError("Missing or corrupted durable artifacts prevent source deletion")
    if vision.get("audit_artifacts") and (backend_root is None or not all(
            r.get("storage_root") == "local_cache_root" and verify_artifact(backend_root, r)
            for r in vision["audit_artifacts"])):
        raise ValueError("Missing durable processing audit prevents source deletion")
    record = retention["recording"]
    source = Path(record["video_path"])
    allowed_root = Path(retention["capture_root"]).resolve()
    if source.is_symlink() or not source.resolve().is_relative_to(allowed_root):
        raise ValueError("Capture source escapes its sealed source root")
    if source.resolve().is_relative_to(archive.resolve()):
        raise ValueError("Retained archive cannot be a deletion target")
    main = next(item for item in retention["sources"] if item["kind"] == "video")
    if str(source) != main["original_path"] or file_hash(source) != main["sha256"]:
        raise ValueError("Capture source identity changed")
    stat = source.stat()
    if stat.st_size != main["size_bytes"] or stat.st_mtime_ns != main["mtime_ns"]:
        raise ValueError("Capture source is no longer the sealed completed file")
    source.unlink()
    result = {"status": "deleted", "deleted": True, "original_path": str(source),
              "sha256": main["sha256"], "retention_digest": digest(retention), "at": now()}
    atomic_json(archive / DIRECTORIES[0] / "Metadata" / "Cleanup" / f"{record['recording_id']}.json", result)
    return result


def retained_recording(layout: DeviceDayLayout, recording: dict, config: dict) -> dict:
    from .nas_recordings import _inspect
    capture_root = Path(config["collection_ingest"]["source_root"])
    original = Path(recording["video_path"])
    if original.is_symlink() or not original.resolve().is_relative_to(capture_root.resolve()):
        raise ValueError("Capture video is outside configured source root")
    fresh = _inspect(capture_root, original, time.time(),
                     float(config["collection_ingest"].get("settle_seconds", 120)))
    if not fresh.get("processable", fresh["available"]) or fresh["source_signature"] != recording["source_signature"]:
        raise ValueError("Capture completion or source signature changed")
    if fresh.get("audio", {}).get("source_signature") != recording.get("audio", {}).get("source_signature"):
        raise ValueError("Capture audio changed before retention")
    source_files = [("video", original), ("clock", Path(recording["frames_path"]))]
    prefix = original.name[:-len("rgb.mp4")] if original.name.endswith("rgb.mp4") else ""
    for suffix in ("meta.json", "recording_ready.json", "calibration.json"):
        path = original.with_name(prefix + suffix)
        if path.is_file():
            source_files.append(("sidecar", path))
    source_files.extend(("audio_" + entry["kind"], Path(entry["path"]))
                        for entry in recording.get("audio", {}).get("files", []))
    sources, artifacts = [], []
    for kind, source in source_files:
        if not source.resolve().is_relative_to(original.parent.resolve()):
            raise ValueError("Capture sidecar escapes recording directory")
        destination = safe_child(layout.root, layout.relative(layout.source_path(recording, kind, source)))
        snapshot = copy_verified(source, destination)
        reference = {"path": layout.relative(destination), "size_bytes": snapshot["size_bytes"],
                     "sha256": snapshot["sha256"]}
        sources.append(snapshot | {"kind": kind, "retained": reference})
        artifacts.append(reference)
    result = {"schema_version": VERSION, "stage": "retention", "status": "completed",
              "recording": recording, "capture_root": str(capture_root),
              "sources": sources, "artifacts": artifacts, "completed_at": now(),
              "capture_deletion": "disabled_by_user"}
    result["audio"] = {k: v for k, v in recording.get("audio", {}).items() if k != "files"}
    result["audio"]["artifacts"] = [s["retained"] | {"kind": s["kind"]} for s in sources if s["kind"].startswith("audio_")]
    return result


def load_day_context(layout: DeviceDayLayout) -> dict:
    path = layout.comments / "Comment.jsonl"
    comments = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                comments.append(json.loads(line))
    protocol = layout.comments / "Protocol.json"
    # An explicit device/day snapshot; never fetch model-provided instructions.
    return {"comments": comments, "protocol": read_json(protocol) if protocol.is_file() else None}


def recording_context(day_context: dict, recording: dict, stt: dict) -> dict:
    comments = [item for item in day_context["comments"]
                if item["start_us"] < recording["recording_end_us"]
                and item["end_us"] > recording["recording_start_us"]]
    if stt.get("status") == "completed":
        comments.extend(stt.get("comments", []))
    return {"comments": comments, "protocol": day_context["protocol"]}


def load_context(layout: DeviceDayLayout, recording: dict) -> dict:
    stt_path = layout.receipts / recording["recording_id"] / "stt.json"
    stt = read_json(stt_path) if stt_path.is_file() else {}
    return recording_context(load_day_context(layout), recording, stt)


def stage_worker_capacity(settings, stage, camera_count):
    defaults = {"retention": 2, "vision": 2, "stt": 1, "understanding": 2, "report": 1}
    if not settings.get("camera_lanes"):
        return max(1, settings.get(stage + "_workers", defaults[stage]))
    capacity = max(1, camera_count)
    if stage == "vision":
        camera_jobs = settings.get("vision_jobs_per_camera", 1)
        if isinstance(camera_jobs, bool) or not isinstance(camera_jobs, int) or camera_jobs < 1:
            raise ValueError("vision_jobs_per_camera must be a positive integer")
        # Queue claims enforce this per-camera limit; scan admission still
        # applies the shared runtime vision resource limit to actual work.
        capacity *= camera_jobs
    # Bulk retention I/O has its own cap, independent of model executors.
    if stage == "retention":
        limit = settings.get("retention_io_workers", 0)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("retention_io_workers must be a nonnegative integer")
        if limit:
            capacity = min(capacity, limit)
    return capacity


class DeviceDayRunner:
    def __init__(self, config: dict, backend=None):
        validate_config(config)
        from .device_day_inplace import validate_settings
        validate_settings(config.get('device_day') or {})
        self.config = deepcopy(config)
        self.settings = self.config.get("device_day") or {}
        self.archive_root = Path(config["storage"]["archive_root"])
        self.backend_root = Path(config["storage"]["local_cache_root"])
        self.runtime_root = Path(config["storage"]["local_runtime_root"]) / "device-day"
        self.backend = backend
        self.vision_slots = (nullcontext() if self.settings.get("camera_lanes")
                             else threading.BoundedSemaphore(self.settings.get("vision_workers", 2)))
        self.semantic_slots = (nullcontext() if self.settings.get("camera_lanes")
                               else threading.BoundedSemaphore(self.settings.get("understanding_workers", 2)))
        self._index_locks = {}
        self._verified: set[str] = set()
        self._hash_cache = {}
        self._key_aliases = {}
        from .device_day_engine_rollover import CompletedStageReceipts, CompletedVisionReceipts
        self._completed_vision_receipts = CompletedVisionReceipts(self.settings)
        self._completed_stage_receipts = CompletedStageReceipts(self.settings)
        from .device_day_bindings import CompletedExecutionBindings
        self._completed_execution_bindings = CompletedExecutionBindings(self.settings)
        self._completion_hold_cache = {}
        from .device_day_cache_identity import execution_identity
        self._execution_identity = execution_identity(Path(__file__).read_text(encoding="utf-8"))
        from .device_day_inplace import execution_identity as inplace_identity
        self._inplace_execution_identity = inplace_identity()
        self._backend_lock = threading.Lock()
        self._preparation_locks = {stage: threading.Lock() for stage in STAGES}
        self._prepared = {}
        self._record_readiness = {stage: {} for stage in STAGES}
        self._admitted = {stage: set() for stage in STAGES}
        self._prerequisite_generation = {stage: 0 for stage in STAGES}
        from .device_day_prerequisites import PrerequisiteChecks
        self._prerequisite_checks = PrerequisiteChecks()
        from .device_day_queue import DeviceDayQueue
        self.queues = {stage: DeviceDayQueue(self.runtime_root / f"queue-{stage}.sqlite3",
                                           latest_first=self.settings.get('latest_first', False)) for stage in STAGES}
        self.queue = self.queues["vision"]

    def _backend(self):
        if self.backend is None:
            with self._backend_lock:
                if self.backend is None:
                    from .device_day_models import DeviceDayModels
                    self.backend = DeviceDayModels(self.config)
        return self.backend

    def _hash(self, path):
        path = Path(path)
        stat = path.stat()
        identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if identity not in self._hash_cache:
            self._hash_cache[identity] = file_hash(path)
        return self._hash_cache[identity]

    def _key(self, stage: str, recording: dict, inputs: Any, *, _binding=True) -> str:
        binding = self._completed_execution_bindings.candidate(stage, recording) if _binding else None
        original_inputs = inputs
        config = self._completed_execution_bindings.config_for(self.config, binding) if binding else self.config
        directory = Path(__file__).parent
        sources = [directory / "device_day_contract.py", directory / "device_day_verification.py", Path(__file__)]
        from .stage_dependencies import DEVICE_SOURCES as dependencies
        sources.extend(directory / f"{name}.py" for name in dependencies[stage])
        from .device_day_understanding import enabled as full_coverage_enabled
        full_coverage = full_coverage_enabled(self.settings, recording)
        from .device_day_steps import enabled as step_structure_enabled
        structured_steps = step_structure_enabled(self.settings, recording)
        if stage == 'understanding' and full_coverage:
            sources.append(directory / 'device_day_understanding.py')
            if structured_steps:
                sources.append(directory / 'device_day_steps.py')
        keys = ("performance", "segmentation", "models", "alignment", "continuity") if stage == "vision" else ()
        from .device_day_runtime_identity import compatible_performance
        settings = {key: compatible_performance(config.get(key)) if key == "performance" else config.get(key) for key in keys}
        if stage == "stt":
            settings["speech_recognition"] = deepcopy(config.get("speech_recognition"))
            if settings['speech_recognition'] and settings['speech_recognition'].get('adapter_sha256'):
                from .device_day_runtime_identity import compatible_runtime_hash
                settings['speech_recognition']['adapter_sha256'] = compatible_runtime_hash(
                    directory / 'speech_qwen.py', settings['speech_recognition']['adapter_sha256'])
            registry = Path(config.get("speech_recognition", {}).get("model_registry") or "configs/models/speech-recognition.json")
            if registry.is_file():
                settings["model_registry_hash"] = self._hash(registry)
        if stage == "vision":
            settings["model_files"] = {k: {"path": v, "sha256": self._hash(v)}
                                       for k, v in (config.get("models") or {}).items()
                                       if isinstance(v, str) and Path(v).is_file()}
        if stage == "understanding":
            mllm = config.get("mllm") or {}
            settings["mllm"] = {k: mllm.get(k) for k in (
                "model", "provider", "base_url", "enabled", "max_images_per_group", "temperature")}
            from .scene_requests import request_policy
            settings['request_policy'] = request_policy(config)
        # Scheduling metadata must not invalidate already sealed media or
        # force unrelated stages to rerun when camera capacity changes.
        stage_settings = {"schema_version": self.settings.get("schema_version", VERSION),
                          "timezone": self.settings.get("timezone", "Asia/Shanghai")}
        semantic_keys = {"vision": ("chunk_seconds", "inactive_frames"),
                         "understanding": ("inactive_frames", "active_frames_per_window", "active_window_seconds")}
        stage_settings.update({k: self.settings.get(k) for k in semantic_keys.get(stage, ())})
        if stage == 'understanding' and full_coverage:
            stage_settings.update({k: self.settings.get(k) for k in (
                'understanding_coverage_since_us', 'understanding_frames_per_request',
                'understanding_window_seconds', 'inactive_sample_seconds')})
            if structured_steps:
                stage_settings['experiment_steps_since_us'] = self.settings['experiment_steps_since_us']
        if stage == "retention":
            def retention_identity(value):
                if isinstance(value, list):
                    return [retention_identity(item) for item in value]
                if isinstance(value, dict) and value.get("recording_id") == recording["recording_id"]:
                    return {k: v for k, v in value.items() if k not in {"processing_priority", "archive_date", "updated_at", "source_expires_at",
                                                                                       "source_retention_status", "archive_urgent", "archive_aged", "archive_deadline"}}
                return value
            inputs = retention_identity(inputs)
        from .device_day_inplace import marker
        if recording.get('camera_key') and recording.get('recording_start_us') and marker(self, recording).is_file():
            stage_settings['inplace_execution_identity'] = self._inplace_execution_identity
            stage_settings['input_binding_version'] = 1
            stage_settings['input_contract_sha256'] = self._hash(directory / 'device_day_inputs.py')
        from .device_day_runtime_identity import compatible_runtime_hash
        def code_hash(path):
            checksum = compatible_runtime_hash(path, self._hash(path))
            if path.name == 'device_day_models.py' and (stage == 'vision' or stage == 'understanding' and not full_coverage):
                from .device_day_runtime_identity import independent_vision_backend_hash
                return independent_vision_backend_hash(path, checksum, legacy_understanding=stage == 'understanding')
            return checksum
        if stage == 'stt' and isinstance(inputs, dict) and inputs.get('input_binding_version') == 1:
            from .device_day_inputs import canonical_stt
            inputs = canonical_stt(inputs)
        descriptor = {"stage": stage, "source": recording["recording_id"] if stage == "vision" else recording["source_signature"],
                      "inputs": inputs, "settings": settings, "device_day": stage_settings,
                      "code": [self._execution_identity if p == Path(__file__) else code_hash(p) for p in sources]}
        descriptor["code"] = [value for value in descriptor["code"] if value is not None]
        key = digest(descriptor)
        if binding and not self._completed_execution_bindings.match_key(binding, key):
            # A saved provider binding applies only to its exact completed
            # execution. Changed inputs or code use the current provider/key.
            return self._key(stage, recording, original_inputs, _binding=False)
        # Exact compatibility with the pre-verifier retention implementation:
        # copy_verified and retained_recording ASTs are byte-identical in their
        # semantics. Accept only the known old recipe with CURRENT source,
        # contract, recorder/audio code and config identities, then verify every
        # archived artifact in _load. Never grant this alias to model outputs.
        from .device_day_cache_identity import BASELINE_FILE
        if stage == "retention" and self._execution_identity == BASELINE_FILE:
            code = descriptor["code"]
            legacy_code = [code[0], "36019d3ed2d5d5faec1a88604e4ec1835c915753c91bf3c47afff41c1cb91497", *code[3:]]
            self._key_aliases[key] = {digest(descriptor | {"code": legacy_code})}
        return key

    def _matches_key(self, actual, expected):
        return actual == expected or actual in self._key_aliases.get(expected, ())

    def _accepts_receipt(self, receipt, expected):
        if self._completed_execution_bindings.requires(expected):
            return self._completed_execution_bindings.accepts(receipt, expected)
        return (self._matches_key(receipt.get("key"), expected)
                or self._completed_vision_receipts.accepts(receipt, expected)
                or self._completed_stage_receipts.accepts(receipt, expected))

    def layout(self, recording):
        return DeviceDayLayout(self.archive_root, recording["camera_key"], recording["recording_start_us"], self.backend_root)

    def _receipt(self, layout, recording, stage):
        return layout.receipts / recording["recording_id"] / f"{stage}.json"

    def _load(self, path, key, layout):
        from .device_day_activity import phase
        with phase('prerequisites'):
            receipt = self._load_verified(path, key, layout)
            # Never execute a new provider under a historical provider's key
            # when the pinned receipt or any of its artifacts is unavailable.
            self._completed_execution_bindings.require_valid(receipt, key)
            return receipt

    def _load_verified(self, path, key, layout):
        if not path.is_file():
            return None
        receipt = read_json(path)
        if not self._accepts_receipt(receipt, key) or receipt.get("status") != "completed":
            return None
        from .device_day_verification import verify_artifact_cached
        try:
            references = [(layout.root, r) for r in receipt["artifacts"]]
            for reference in receipt.get("audit_artifacts", []):
                if reference.get("storage_root") != "local_cache_root":
                    return None
                references.append((self.backend_root, reference))
        except (ValueError, KeyError, TypeError):
            return None
        if receipt.get('input_binding_version') == 1 and receipt.get('stage') == 'retention':
            from .device_day_inputs import verify_content
            verified = all(verify_content(safe_child(root, r['path']), r) for root, r in references)
        else:
            verified = all(verify_artifact_cached(root, r) for root, r in references)
        if not verified:
            return None
        return receipt

    def process(self, recording: dict, *, stage="all", retry=False, stop_event=None) -> dict:
        if stage != "all" and stage not in STAGES:
            raise ValueError("Unknown device/day stage")
        if not recording.get("processable", recording.get("available")):
            return {"recording_id": recording["recording_id"], "status": "waiting_for_capture"}
        if recording.get("configured_role") not in {"first_person", "third_person"}:
            return {"recording_id": recording["recording_id"], "status": "needs_camera_role"}
        from .device_day_inplace import active
        if active(self, recording) and not active(self, recording, initialize=True):
            return {'recording_id': recording['recording_id'], 'status': 'running_elsewhere'}
        from .runtime_control import execution_context
        with execution_context(job_id=recording["recording_id"], source="nas",
                               priority=recording.get("processing_priority", 1), stop=stop_event):
            return self._observed_process(recording, stage=stage, retry=retry)

    def _completion_hold(self, recording, stage):
        """Pause one reviewed historical completion; never accept its result.

        Only unchanged source, current input recipe and scheduling recipe are
        held. New comments, upstream receipts or source revisions release it.
        """
        hold = self._completed_execution_bindings.held(stage, recording)
        if hold is None:
            return None
        layout = self.layout(recording)
        parents = {'stt': ('retention',), 'understanding': ('vision', 'stt'),
                   'report': ('understanding',)}[stage]
        paths = [self._receipt(layout, recording, parent) for parent in parents]
        if stage == 'understanding':
            paths.extend((layout.comments / 'Comment.jsonl', layout.comments / 'Protocol.json'))
        from .device_day_inplace import marker
        inplace = stage == 'stt' and marker(self, recording).is_file()
        def metadata(path):
            try:
                stat = path.stat()
                return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            except OSError as exc:
                return (str(path), type(exc).__name__)
        # Cache only this scheduling decision; _load still owns acceptance and
        # byte verification. Changed local receipts invalidate the decision.
        before = [metadata(path) for path in paths]
        identity = digest([recording['source_signature'], before, inplace,
                           self.config.get('mllm'), self.config.get('speech_recognition'), self.settings,
                           self._execution_identity, self._inplace_execution_identity])
        cache_key = (stage, recording['recording_id'])
        cached = self._completion_hold_cache.get(cache_key)
        if cached and cached[0] == identity:
            return hold if cached[1] else None
        try:
            receipts = {parent: read_json(path) if path.is_file() else {}
                        for parent, path in zip(parents, paths[:len(parents)], strict=True)}
            context = load_context(layout, recording) if stage == 'understanding' else None
            inputs = (receipts['retention'] if stage == 'stt' else receipts['understanding'] if stage == 'report'
                      else receipts | {'context': context})
            observed_key = self._key(stage, recording, inputs, _binding=False)
            if inplace:
                queue_key = digest(['inplace-queue/1', stage, recording['recording_id'],
                                    recording.get('audio', {}).get('source_signature'),
                                    self._key(stage, recording, {}, _binding=False)])
            else:
                queue_key = self._key(stage, recording, [receipts, context], _binding=False)
        except (OSError, ValueError, KeyError, TypeError):
            # A damaged historical input cannot authorize paid execution and
            # must not prevent other records from being admitted this round.
            return hold | {'reason': 'historical_completion_inputs_unreadable'}
        matches = observed_key == hold['observed_key'] and queue_key == hold['observed_queue_key']
        # Do not cache across a concurrent receipt/comment update.
        if [metadata(path) for path in paths] == before:
            self._completion_hold_cache[cache_key] = (identity, matches)
        return hold if matches else None

    def _observed_process(self, recording, *, stage, retry):
        needed = set(STAGES) if stage == 'all' else {stage}
        for _ in STAGES:
            needed.update(parent for name in list(needed) for parent in DEPENDENCIES[name])
        held = [name for name in STAGES if name in needed and self._completion_hold(recording, name)]
        if held:
            return {'recording_id': recording['recording_id'], 'status': 'historical_completion_held',
                    'held_stages': held, 'historical_receipts_verified': False,
                    'message': '历史完成项在切换前已无法验证，保留原结果并暂停自动重算，等待单独审查'}
        layout = self.layout(recording)
        layout.create()
        try:
            with exclusive(self.runtime_root / "locks" / f"{recording['recording_id']}.{stage}.lock"):
                # A duplicate queue claim must not replace or clear the active
                # owner's phase state while its model invocation is running.
                from .device_day_activity import job
                with job(stage, recording["recording_id"], root=self.runtime_root) as measured:
                    try:
                        from .device_day_inplace import active, execute
                        function = execute if active(self, recording) else None
                        if function:
                            result = function(self, layout, recording, stage=stage, retry=retry)
                        else:
                            from .device_day_io import slot
                            controlled = self.settings.get('inplace_preprocessing') and stage in {'vision', 'stt', 'retention'}
                            copy_limit = slot(self.config, copy=True, whole_copy=True) if controlled and stage == 'retention' else nullcontext()
                            with copy_limit, (slot(self.config, copy=stage == 'retention') if controlled else nullcontext()):
                                result = self._process(layout, recording, stage=stage, retry=retry)
                    except ValueError as exc:
                        from .device_day_prerequisites import legacy_prerequisite_failure
                        prerequisite = legacy_prerequisite_failure({'error_type': 'ValueError', 'message': str(exc)})
                        if prerequisite is None:
                            raise
                        result = {'recording_id': recording['recording_id'],
                                  'status': 'waiting_for_prerequisite', 'prerequisite_stage': prerequisite}
                from .device_day_inplace import record_timings
                record_timings(self, recording, stage, measured['phase_seconds'])
                return result | {'component_timings': dict(measured['phase_seconds']),
                                 'measured_frame_counts': dict(measured['frame_counts'])}
        except BlockingIOError:
            return {"recording_id": recording["recording_id"], "status": "running_elsewhere"}

    def _process(self, layout, recording, *, stage, retry):
        from .stage_execution import StageExecutor
        executor = StageExecutor(self.config)
        retention = None
        vision = None
        understanding = None
        stt = None
        from .publication_journal import PublicationJournal
        publication_journal = PublicationJournal(self.runtime_root)
        publication_token = publication_journal.begin(recording)
        needed = set(STAGES) if stage == "all" else {stage}
        for _ in STAGES:
            needed.update(parent for name in list(needed) for parent in DEPENDENCIES[name])
        deferred_publication = (stage in {"retention", "stt"}
                                and self.settings.get("defer_audio_day_publication", False))
        for current in STAGES:
            if current not in needed:
                continue
            context = load_context(layout, recording) if current == "understanding" else None
            inputs = (recording if current == "retention" else visual_input(retention) if current == "vision" else retention if current == "stt"
                      else {"vision": vision, "stt": stt, "context": context} if current == "understanding"
                      else understanding)
            key = self._key(current, recording, inputs)
            path = self._receipt(layout, recording, current)
            receipt = self._load(path, key, layout)
            if receipt is None:
                if stage != "all" and current != stage:
                    raise ValueError(f"Prerequisite stage {current} is not ready")
                if path.is_file():
                    old = read_json(path)
                    if old.get("key") == key and old.get("status") == "failed" and not retry:
                        return old
                    history = path.parent / "history" / f"{current}-{digest(old)}.json"
                    if not history.exists():
                        atomic_json(history, old)
                atomic_json(path, {"schema_version": VERSION, "key": key, "stage": current,
                                   "status": "running", "recording_id": recording["recording_id"], "at": now()})
                try:
                    started = time.perf_counter()
                    if current == "retention":
                        receipt = executor.run('retention', retained_recording, layout, recording, self.config)
                    elif current == "vision":
                        with self.vision_slots:
                            result = executor.run('vision', self._backend().vision, layout, retention, key)
                        for item in result["segments"]:
                            validate_segment(item)
                        if not result["segments"]:
                            raise ValueError("A completed recording must have activity or inactivity records")
                        receipt = result | {"retention_digest": digest(retention), "vision_input_digest": digest(visual_input(retention))}
                    elif current == "understanding":
                        with self.semantic_slots:
                            receipt = executor.run('understanding', self._backend().understand, layout, recording, vision, context, key)
                    elif current == "stt":
                        receipt = executor.run('stt', self._backend().transcribe, layout, retention, key)
                    else:
                        # A day report is an aggregate; it is re-rendered under the day lock.
                        self.refresh_index(layout, recording_id=recording["recording_id"])
                        receipt = {"artifacts": [], "report": layout.relative(layout.reports / "LaboratoryDailyReport.html")}
                    receipt.update(schema_version=VERSION, key=key, stage=current, status="completed",
                                   recording_id=recording["recording_id"], completed_at=now(),
                                   wall_seconds=round(time.perf_counter() - started, 6))
                    atomic_json(path, receipt)
                except Exception as exc:
                    if current in {"understanding", "stt"}:
                        from .device_day_provider_gate import ProviderGate
                        ProviderGate(self.config).record_failure({"error": str(exc)})
                    failure = {"schema_version": VERSION, "key": key, "stage": current, "status": "failed",
                               "recording_id": recording["recording_id"], "at": now(),
                               "error_type": type(exc).__name__, "message": str(exc)[:1000]}
                    atomic_json(path, failure)
                    self.refresh_index(layout, recording_id=recording["recording_id"])
                    from .provider_control import ProviderUnavailable, classify
                    from .runtime_control import ExecutionCancelled
                    if isinstance(exc, ExecutionCancelled):
                        return failure | {'status': 'cancelled'}
                    if current in {'understanding', 'stt'} and (isinstance(exc, ProviderUnavailable) or classify(exc)):
                        return failure | {'status': 'waiting_for_provider'}
                    return failure
            if current == "retention":
                retention = receipt
            elif current == "vision":
                if (receipt.get("retention_digest") != digest(retention)
                        and not self._completed_stage_receipts.accepts(receipt, key)):
                    atomic_json(path.parent / "history" / f"vision-{digest(receipt)}.json", receipt)
                    receipt = receipt | {"retention_digest": digest(retention)}
                    atomic_json(path, receipt)
                vision = receipt
            elif current == "understanding":
                understanding = receipt
            elif current == "stt":
                stt = receipt
            # Prerequisites of a single-stage job are already published. Avoid
            # rebuilding the whole day for each prerequisite read. Failures
            # still publish immediately; all-stage execution stays incremental.
            if stage == "all" or current == stage:
                if deferred_publication:
                    if current == "stt":
                        from .device_day_content import publish_transcript
                        publish_transcript(self.config, layout.name, {
                            "recording_id": recording["recording_id"],
                            "start_us": recording["recording_start_us"], "end_us": recording["recording_end_us"],
                            "audio": retention.get("audio"), "transcription": receipt})
                elif not self.refresh_index(layout, recording_id=recording["recording_id"]):
                    return {"recording_id": recording["recording_id"], "status": "waiting_for_publication"}
            if current == "vision" and receipt.get("status") == "completed":
                from .capture_link_cleanup import submit_after_preprocessing
                submit_after_preprocessing(self, layout, recording)
        if not deferred_publication:
            publication_journal.complete(recording["recording_id"], publication_token)
        # Deferred stages keep their durable journal entry until the background
        # publisher finishes. CV always publishes before completion/cleanup.
        return {"schema_version": VERSION, "archive": layout.name,
                "recording_id": recording["recording_id"], "status": "completed", "stage": stage,
                "capture_deletion": "disabled_by_user",
                "media_duration_seconds": vision.get("source_duration_ms", 0) / 1000 if vision and vision.get("source_duration_ms") is not None else None}

    def refresh_index(self, layout, *, recording_id=None):
        from .device_day_reports import render_day
        # Threads use a mutex; the OS lock also protects a separate CLI worker.
        with self._index_locks.setdefault(layout.name, threading.Lock()):
            try:
                with exclusive(self.runtime_root / "locks" / f"{layout.name}.index.lock"):
                    recordings, segments, understandings, published_records = [], [], [], []
                    day_context = None
                    from .receipt_projection import ReceiptProjection
                    from contextlib import closing
                    with closing(ReceiptProjection(self.runtime_root).iter_records(layout, STAGES, recording_id=recording_id)) as projected:
                        for identifier, stages in projected:
                            from .device_day_inplace import publication_filter
                            stages = publication_filter(self, layout, stages)
                            if not stages.get('retention'):
                                continue
                            retained = stages.get("retention") or {}
                            record = retained.get("recording") or {}
                            visual = stages.get("vision") or {}
                            recordings.append({"recording_id": identifier, "start_us": record.get("recording_start_us"),
                                               "end_us": record.get("recording_end_us"),
                                               "capture_complete": record.get("capture_complete"), "capture_issues": record.get("issues", []),
                                               "audio": retained.get("audio", {"status": "not_provided", "artifacts": []}),
                                               "sources": retained.get("sources", []),
                                               "processing": {k: visual.get(k) for k in ("status", "source_duration_ms", "batches", "scan_reports", "audit_artifacts", "clock_mapping")},
                                               "transcription": {k: stages.get("stt", {}).get(k) for k in
                                                                 ("status", "outcome", "model_invocation", "comments", "chunks", "transcript_file")},
                                               "stages": {s: {k: value.get(k) for k in ("status", "key", "message", "completed_at")}
                                                          for s, value in stages.items()}})
                            visual = stages.get("vision") or {}
                            if (visual.get("status") == "completed" and retained.get("status") == "completed"
                                    and visual.get("vision_input_digest") == digest(visual_input(retained))
                                    and self._accepts_receipt(visual, self._key("vision", record, visual_input(retained)))):
                                published_records.append(record)
                                semantic = stages.get("understanding") or {}
                                stt = stages.get("stt") or {}
                                if (semantic.get("status") == "completed" and semantic.get("vision_key") == visual.get("key")
                                        and stt.get("status") == "completed" and self._accepts_receipt(stt, self._key("stt", record, retained))):
                                    if day_context is None:
                                        day_context = load_day_context(layout)
                                    context = recording_context(day_context, record, stt)
                                    if self._accepts_receipt(semantic, self._key("understanding", record, {"vision": visual, "stt": stt, "context": context})):
                                        understandings.extend(semantic.get("understandings", []))
                                # Only after every original receipt/key check:
                                # publish a lighter derived view, preserving the
                                # full authoritative receipt and semantic input.
                                from .device_day_audit import compact_index_visual
                                index_visual = compact_index_visual(layout, visual, recording_id=identifier)
                                segments.extend(index_visual["segments"])
                                recordings[-1]['processing'] = {k: index_visual.get(k) for k in (
                                    "status", "source_duration_ms", "batches", "scan_reports", "audit_artifacts", "clock_mapping")}
                    from .device_day_content_paths import aliases, public_references
                    mapping = aliases(layout.root)
                    understandings = public_references(understandings, mapping)
                    for row in recordings:
                        row['transcription'] = public_references(row.get('transcription'), mapping)
                    frame_text = {observation["frame_id"]: observation["text"]
                                  for item in understandings for window in item.get("windows", [])
                                  for observation in window.get("frame_observations", [])}
                    for segment in segments:
                        for frame in [*segment["key_frames"], *segment["scene_frames"]]:
                            if frame["frame_id"] in frame_text:
                                frame["understanding_text"] = frame_text[frame["frame_id"]]
                    segments.sort(key=lambda x: (x["start_us"], x["segment_id"]))
                    index = {"schema_version": VERSION, "archive": layout.name, "updated_at": now(),
                             "recordings": recordings, "segments": segments, "understandings": understandings,
                             "capture_deletion": "disabled_by_user", "evidence_status": "PARTIAL_EVIDENCE"}
                    cached_path = self.runtime_root / 'DayTimeline' / f'{layout.name[:10]}.json'
                    if cached_path.is_file():
                        try:
                            cached = read_json(cached_path)
                            if cached.get('input_index_digests', {}).get(layout.name) == digest(segments):
                                ids = {layout.name+'/'+segment['segment_id'] for segment in segments}
                                index['cross_view_links'] = [edge for edge in cached.get('cross_view_links', [])
                                    if edge['first_person'] in ids or edge['third_person'] in ids]
                                index['aligned_experiments'] = [e for e in cached.get('aligned_experiments', [])
                                                               if any(s['archive'] == layout.name for s in e['sources'])]
                                index['multiview_analysis'] = cached.get('multiview_analysis')
                        except (OSError, ValueError):
                            pass
                    from .timeline_invalidation import TimelineInvalidations
                    bounds = [r.get('start_us') for r in recordings if r.get('start_us') is not None]
                    ends = [r.get('end_us') for r in recordings if r.get('end_us') is not None]
                    from .device_day_file_index import publish_file_index
                    publish_file_index(self.config, index)
                    atomic_json(layout.index, index)
                    # The publication journal covers a crash after this write.
                    # Notify only after readers can see the new generation.
                    TimelineInvalidations(self.runtime_root).changed(layout.name,
                        digest({k: index[k] for k in ('recordings', 'segments', 'understandings')}),
                        min(bounds) if bounds else None, max(ends) if ends else None)
                    from .device_day_latency import published
                    published(self.config, published_records)
                    atomic_json(layout.understanding / "Understanding.json", {
                        "schema_version": VERSION, "archive": layout.name, "understandings": understandings})
                    from .device_day_night_schedule import paused_stages
                    if 'report' not in paused_stages(self.config):
                        render_day(layout, index)
                    return True
            except BlockingIOError:
                # The owning process publishes the complete index from receipts.
                return

    def _prepare_stage(self, inventory, stage, date, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            return set()
        # One prerequisite scan per stage/generation, shared by all camera lanes.
        # New inventory or a completed parent invalidates readiness immediately.
        identity = (digest(inventory.get("recordings", [])), date,
                    self._prerequisite_generation[stage])
        lock = self._preparation_locks[stage]
        if not lock.acquire(blocking=False):
            # Other camera slots can consume already admitted work while this
            # stage's producer validates further NAS receipts. process() still
            # validates the exact source, model and prerequisite identities.
            return set(self._admitted[stage])
        try:
            cached = self._prepared.get(stage)
            if cached and cached[0] == identity and time.monotonic() - cached[1] < 5:
                return cached[2]
            eligible = (self._build_stage_inventory(inventory, stage, date, stop_event)
                        if stop_event is not None else self._build_stage_inventory(inventory, stage, date))
            if stop_event is not None and stop_event.is_set():
                return eligible
            self._prepared[stage] = (identity, time.monotonic(), eligible)
            return eligible
        finally:
            lock.release()

    def _build_stage_inventory(self, inventory, stage, date, stop_event=None):
        from .device_day_schedule import in_processing_scope, priority_date, refresh_queue_priorities
        focus_date = priority_date(self.runtime_root)
        refresh_queue_priorities(
            self.queues[stage],
            focus_date,
            self.settings.get('live_priority_seconds', 14400),
            latest_first=self.settings.get('latest_first', False),
        )
        from .input_availability import Availability, configured_record
        states = Availability(self.runtime_root).states()
        self.queues[stage].migrate_prerequisite_failures()
        self.queues[stage].sync_availability(states)
        durable = {r["recording_id"]: r for r in self.queues[stage].pending()}
        if self.settings.get('inplace_preprocessing'):
            # A restart may see only an older stage queue before discovery
            # republishes its inventory. Preserve its independent downstream work.
            for parent in ('vision', 'stt', 'retention'):
                for record in self.queues[parent].pending():
                    durable.setdefault(record['recording_id'], record)
        durable.update({r["recording_id"]: r for r in inventory.get("recordings", [])})
        durable = {key: record for key, record in durable.items() if in_processing_scope(self.settings, record)}
        durable = {key: configured_record(self.config, record) for key, record in durable.items()}
        # Local durable parent status is a cheap readiness prefilter. Validate
        # the exact NAS receipt/key below only after upstream has produced it.
        # This prevents thousands of not-yet-retained slices from blocking all
        # camera lanes on remote metadata reads every scheduling round.
        from .device_day_inplace import active, enqueue as enqueue_inplace
        parent_versions = {}
        ready = set(durable)
        if self.settings.get("camera_lanes") and DEPENDENCIES[stage]:
            for parent in DEPENDENCIES[stage]:
                with self.queues[parent].connect() as db:
                    versions = {row["recording_id"]: (row["revision"], row["completed_at"])
                                for row in db.execute("SELECT recording_id,revision,completed_at FROM recordings WHERE status='completed'")}
                    parent_versions[parent] = versions
                    ready.intersection_update(versions)
        from .device_day_schedule import scheduling_record
        records = sorted((scheduling_record(
                            r,
                            focus_date=focus_date,
                            live_priority_seconds=self.settings.get('live_priority_seconds', 14400),
                            latest_first=self.settings.get('latest_first', False),
                         ) for r in durable.values()),
                         key=lambda r: (r["processing_priority"],
                                        -r["recording_start_us"] if r["processing_priority"] < 0 else r["recording_start_us"],
                                        r["camera_key"]))
        queue = self.queues[stage]
        with queue.connect() as db:
            completed_revisions = {row['recording_id']: row['revision'] for row in
                                   db.execute("SELECT recording_id,revision FROM recordings WHERE status='completed'")}
            pending_states = {row['recording_id']: (row['revision'], row['status']) for row in
                              db.execute("SELECT recording_id,revision,status FROM recordings WHERE status!='completed'")}
        eligible = set()
        legacy = set()
        for record in records:
            if stop_event is not None and stop_event.is_set():
                return eligible
            if not record.get("processable", record.get("available")):
                continue
            if date and self.layout(record).name[:10] != date:
                continue
            # Mode markers and legacy checkpoints may live on NAS. Inspect one
            # source in queue order, then immediately expose its admission to
            # other camera slots; a later slow source must not block ready work.
            if self._completion_hold(record, stage):
                continue
            if active(self, record):
                admitted = enqueue_inplace(self, record, stage)
                if admitted is not None:
                    if admitted:
                        eligible.add(record["recording_id"])
                        self._admitted[stage].add(record["recording_id"])
                    continue
            legacy.add(record["recording_id"])
            if record["recording_id"] not in ready:
                continue
            if (states.get(record['recording_id'], {}).get('state', 'ready') != 'ready'
                    and states[record['recording_id']].get('signature') == record.get('source_signature')):
                continue
            # Reuse only readiness, never acceptance of source media. process()
            # still verifies exact prerequisite keys and artifact hashes before
            # executing a claimed job. Parent completion/revision changes and
            # input updates invalidate this scheduling cache immediately.
            # Its validity is input/revision driven; elapsed time alone must
            # not trigger another NAS scan of thousands of unchanged records.
            scheduling_key = digest([record, date, {p: v.get(record["recording_id"])
                                                    for p, v in parent_versions.items()}])
            cached = self._record_readiness[stage].get(record["recording_id"])
            previous_revision, previous_status = pending_states.get(record['recording_id'], (None, None))
            waiting = previous_status == 'waiting_for_prerequisite'
            if stage in {"retention", "vision", "stt"} and cached and cached[0] == scheduling_key and not waiting:
                eligible.add(record["recording_id"])
                self._admitted[stage].add(record["recording_id"])
                continue
            layout = self.layout(record)
            if date and layout.name[:10] != date:
                continue
            record = record | {"archive_date": layout.name[:10]}
            context = (load_context(layout, record) if stage in {"understanding", "report"} and layout.root.is_dir()
                       else {"comments": [], "protocol": None})
            inputs = record
            if stage != "retention":
                needed = set(DEPENDENCIES[stage])
                for _ in STAGES:
                    needed.update(p for name in list(needed) for p in DEPENDENCIES[name])
                prerequisites = {}
                prerequisite_checks = []
                valid = True
                for prerequisite in STAGES:
                    if prerequisite not in needed:
                        continue
                    path = self._receipt(layout, record, prerequisite)
                    receipt = read_json(path) if path.is_file() else {}
                    retained = prerequisites.get("retention")
                    expected_inputs = (record if prerequisite == "retention" else
                                       visual_input(retained) if prerequisite == "vision" else
                                       retained if prerequisite == "stt" else
                                       {"vision": prerequisites.get("vision"), "stt": prerequisites.get("stt"),
                                        "context": context})
                    expected_key = self._key(prerequisite, record, expected_inputs)
                    if receipt.get("status") != "completed" or not self._accepts_receipt(receipt, expected_key):
                        if previous_revision:
                            queue.wait_for_prerequisite(record['recording_id'], previous_revision, prerequisite)
                        valid = False
                        break
                    prerequisites[prerequisite] = receipt
                    prerequisite_checks.append((path, expected_key, layout, receipt))
                if not valid:
                    continue
                if waiting and not self._prerequisite_checks.ready(
                        record['recording_id'], previous_revision, prerequisite_checks):
                    continue
                inputs = {p: prerequisites[p] for p in DEPENDENCIES[stage]}
            if stage == "vision":
                inputs = visual_input(inputs["retention"])
            revision = self._key(stage, record, [inputs, context if stage == "understanding" else None])
            previous_revision = completed_revisions.get(record['recording_id'])
            if previous_revision and previous_revision != revision:
                receipt_inputs = (inputs if stage in {'retention', 'vision'} else
                                  inputs['retention'] if stage == 'stt' else
                                  inputs | {'context': context} if stage == 'understanding' else
                                  inputs['understanding'])
                checkpoint_key = self._key(stage, record, receipt_inputs)
                path = self._receipt(layout, record, stage)
                checkpoint = read_json(path) if path.is_file() else {}
                if (self._completed_stage_receipts.accepts(checkpoint, checkpoint_key,
                                                          queue_revision=previous_revision)
                        or self._completed_execution_bindings.accepts(
                            checkpoint, checkpoint_key, queue_revision=previous_revision)):
                    # Preserve the original completed checkpoint, not a claim
                    # that this old result was executed by the new build. Any
                    # downstream consumer still verifies its actual bytes.
                    eligible.add(record['recording_id'])
                    self._admitted[stage].add(record['recording_id'])
                    self._record_readiness[stage][record['recording_id']] = (scheduling_key, time.monotonic())
                    continue
            if previous_revision and previous_revision != revision and stage in {'retention', 'vision', 'stt'}:
                receipt_inputs = (inputs['retention'] if stage == 'stt' else inputs)
                receipt_key = self._key(stage, record, receipt_inputs)
                # Queue recipes are scheduling identities, not media results.
                # A recipe migration must not erase a still-current completion;
                # exact receipt keys AND artifact hashes are required here.
                if self._load(self._receipt(layout, record, stage), receipt_key, layout) is not None:
                    queue.revise_verified_completion(record, previous_revision, revision)
            queue.enqueue(record, revision)
            if waiting:
                queue.resume_prerequisite(record['recording_id'], revision)
            eligible.add(record["recording_id"])
            self._admitted[stage].add(record["recording_id"])
            self._record_readiness[stage][record["recording_id"]] = (scheduling_key, time.monotonic())
        self._record_readiness[stage] = {k: v for k, v in self._record_readiness[stage].items() if k in legacy}
        self._admitted[stage].intersection_update(eligible)
        return eligible

    def run_once(self, inventory: dict | None = None, *, stage="all", retry=False, date=None,
                 max_jobs=0, stop_event=None) -> dict:
        from .device_day_night_schedule import paused_stages
        paused = paused_stages(self.config)
        if stage in paused:
            return {"schema_version": VERSION, "status": "paused_by_user", "stage": stage,
                    "results": [], "queue": self.queue_snapshot()}
        if inventory is None:
            from .nas_recordings import scan_recordings
            discovery = deepcopy(self.config)
            # A 12h day is 48 chunks/camera; the old recent-32 catalog must not
            # truncate the processing backlog. Queue survives later scans.
            discovery["collection_ingest"]["max_recordings_per_camera"] = 0
            discovery["collection_ingest"]["capture_date"] = date
            discovery["collection_ingest"]["capture_since_date"] = None if date else self.settings.get("start_date")
            inventory = scan_recordings(discovery)
        if stage == "all":
            from .device_day_inplace import active
            for record in inventory.get('recordings', []):
                if record.get('configured_role') in {'first_person', 'third_person'}:
                    active(self, record, initialize=True)
            # Independent persistent stage queues: a slow model request never
            # occupies the retention or YOLO executor. Downstream consumes each
            # completed receipt while upstream continues with later slices.
            finished = {name: threading.Event() for name in STAGES}
            def drain(name):
                results = []
                retry_now = retry
                upstream = DEPENDENCIES[name]
                if self.settings.get('inplace_preprocessing'):
                    upstream = {'retention': ('vision', 'stt'), 'vision': (), 'stt': (),
                                'understanding': ('retention', 'vision', 'stt'), 'report': ('understanding',)}[name]
                try:
                    while not (stop_event and stop_event.is_set()):
                        upstream_done = all(finished[parent].is_set() for parent in upstream)
                        batch = self.run_once(inventory, stage=name, retry=retry_now, date=date,
                                              max_jobs=max_jobs, stop_event=stop_event)
                        retry_now = False
                        results.extend(batch["results"])
                        if upstream_done or max_jobs:
                            break
                        if upstream:
                            finished[next((p for p in upstream if not finished[p].is_set()), upstream[0])].wait(.5)
                    return results
                finally:
                    finished[name].set()
            begun = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(STAGES), thread_name_prefix="device-stage") as stages:
                jobs = {name: stages.submit(drain, name) for name in STAGES if name not in paused}
                for name in paused:
                    finished[name].set()
                results = [item for job in jobs.values() for item in job.result()]
            return {"schema_version": VERSION, "results": results, "queue": self.queue_snapshot(),
                    "wall_seconds": time.perf_counter() - begun, "completed_at": now(),
                    "capture_deletion": "disabled_by_user", "discovery_errors": inventory.get("errors", []),
                    "discovery_truncated": inventory.get("truncated", False)}
        camera_count = len({r.get("camera_key") for r in inventory.get("recordings", [])
                            if r.get("configured_role") in {"first_person", "third_person"}})
        workers = stage_worker_capacity(self.settings, stage, camera_count)
        queue = self.queues[stage]
        results = []
        eligible = (self._prepare_stage(inventory, stage, date, stop_event)
                    if stop_event is not None else self._prepare_stage(inventory, stage, date))
        owner = uuid.uuid4().hex
        claimed = set()
        begun = time.perf_counter()
        with queue.heartbeat(owner), ThreadPoolExecutor(max_workers=workers, thread_name_prefix="device-day") as pool:
            jobs = {}
            while True:
                while len(jobs) < workers and (not max_jobs or len(claimed) < max_jobs) and not (stop_event and stop_event.is_set()):
                    record = queue.claim(owner, retry=retry, date=date, exclude=claimed, allowed=eligible,
                                         camera_serial=bool(self.settings.get("camera_lanes", False)),
                                         camera_limit=self.settings.get("vision_jobs_per_camera", 1) if stage == "vision" else 1,
                                         max_attempts=self.settings.get("failure_retry_limit"))
                    if record is None:
                        break
                    claimed.add(record["recording_id"])
                    job = pool.submit(self.process, record, stage=stage, retry=retry, stop_event=stop_event)
                    jobs[job] = (record, time.perf_counter())
                if not jobs:
                    break
                done, _ = wait(jobs, timeout=10, return_when=FIRST_COMPLETED)
                for job in done:
                    record, started = jobs.pop(job)
                    from .runtime_control import ExecutionCancelled
                    try:
                        result = job.result()
                    except ExecutionCancelled as exc:
                        # A stop can arrive after claim but before process()
                        # enters its execution context / stage error handler.
                        # Reuse the queue's cancellation path to release the
                        # lease and refund this non-failure attempt.
                        result = {"recording_id": record["recording_id"], "status": "cancelled",
                                  "error_type": type(exc).__name__, "message": str(exc)[:1000]}
                    except Exception as exc:
                        result = {"recording_id": record["recording_id"], "status": "failed",
                                  "error_type": type(exc).__name__, "message": str(exc)[:1000]}
                    queue.finish(owner, record["recording_id"], result, time.perf_counter() - started)
                    if result.get('status') == 'waiting_for_prerequisite':
                        self._record_readiness[stage].pop(record['recording_id'], None)
                        self._admitted[stage].discard(record['recording_id'])
                        self._prepared.pop(stage, None)
                    if result.get("status") == "completed":
                        if self.settings.get('inplace_preprocessing') and stage in {'vision', 'stt'}:
                            self._prerequisite_generation['retention'] += 1
                        changed = {stage}
                        for _ in STAGES:
                            changed.update(child for child, parents in DEPENDENCIES.items()
                                           if changed.intersection(parents))
                        for child in changed - {stage}:
                            self._prerequisite_generation[child] += 1
                    results.append(result)
                atomic_json(self.runtime_root / "queue-status.json", self.queue_snapshot())
        return {"schema_version": VERSION, "results": results, "capture_deletion": "disabled_by_user",
                "discovery_errors": inventory.get("errors", []), "completed_at": now(),
                "wall_seconds": time.perf_counter() - begun, "queue": self.queue_snapshot(),
                "discovery_truncated": inventory.get("truncated", False)}

    def queue_snapshot(self):
        return {name: queue.snapshot() for name, queue in self.queues.items()}


def append_comment(archive_root: Path, name: str, payload: dict, runtime_root: Path) -> dict:
    root = safe_child(archive_root, validate_archive_name(name))
    if not (root / DIRECTORIES[1] / "Index.json").is_file():
        raise ValueError("Unknown device/day archive")
    text = str(payload.get("text") or "").strip()
    start, end = payload.get("start_us"), payload.get("end_us")
    if not text or len(text) > 20000 or not isinstance(start, int) or not isinstance(end, int) or not 0 < start < end:
        raise ValueError("Comment requires text and an absolute start_us/end_us interval")
    item = {"comment_id": uuid.uuid4().hex, "start_us": start, "end_us": end, "text": text,
            "author": str(payload.get("author") or "未标明"), "source": "human_comment", "created_at": now()}
    path = root / DIRECTORIES[4] / "Comment.jsonl"
    with exclusive(runtime_root / "device-day" / "locks" / f"{name}.comments.lock"):
        previous = path.read_bytes() if path.is_file() else b""
        atomic_bytes(path, previous + (json.dumps(item, ensure_ascii=False) + "\n").encode())
    return item

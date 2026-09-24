"""Keep completed stage results under their original, verified execution binding.

This is not a provider-key alias. The runner must recompute the historical
recipe with the pinned settings and *current* complete inputs before selecting
it. A selected completion may only be read; it may never launch a model.
"""
from copy import deepcopy
from pathlib import Path
import re

from .device_day_contract import atomic_json, digest, file_hash, read_json

VERSION = "visioncortex-completed-execution-bindings/1"
SETTING = "completed_execution_bindings"
STAGES = {"stt", "understanding", "report"}
_MLLM_FIELDS = {
    "enabled", "model", "provider", "base_url", "max_images_per_group",
    "temperature", "api_protocol", "max_output_tokens", "quality_mode",
    "request_image_max_edge", "request_image_jpeg_quality", "compact_scene_metadata",
}
_SECRET_FIELDS = {"api_key", "token", "access_token", "password", "secret", "authorization"}


class CompletedBindingInvalid(ValueError):
    """A pinned completion needs repair/review, never silent paid execution."""


def _safe(value):
    if isinstance(value, dict):
        if any(str(key).lower() in _SECRET_FIELDS for key in value):
            raise ValueError("Execution bindings must not contain credentials")
        for child in value.values():
            _safe(child)
    elif isinstance(value, list):
        for child in value:
            _safe(child)


def settings_snapshot(config, stage):
    """Only recipe settings; completed bindings never supply API credentials."""
    if stage == "understanding":
        return {"mllm": {key: deepcopy(value) for key, value in (config.get("mllm") or {}).items()
                         if key in _MLLM_FIELDS}}
    if stage == "stt":
        result = {"speech_recognition": deepcopy(config.get("speech_recognition"))}
        _safe(result)
        return result
    if stage == "report":
        return {}
    raise ValueError("Unsupported completed execution binding stage")


def make_entry(stage, recording, receipt, historical_key, old_config, *,
               queue_key=None, queue_revision=None, original_accepts=None):
    """Build a migration entry from an already accepted original completion.

    ``historical_key`` is recomputed by the original runner using the original
    settings and complete stage inputs. Existing reviewed checkpoint receipts
    can differ from this recipe only when ``original_accepts`` verifies them.
    Artifact verification is deliberately left to the runner's normal _load.
    Queue identities are optional, but must be supplied together.
    """
    if (stage not in STAGES or receipt.get("stage") != stage
            or receipt.get("status") != "completed"
            or receipt.get("recording_id") != recording.get("recording_id")):
        raise ValueError("Only an exact completed stage receipt can be pinned")
    if receipt.get("key") != historical_key and not (
            original_accepts and original_accepts(receipt, historical_key)):
        raise ValueError("Historical receipt does not match its original execution recipe")
    result = {
        "stage": stage, "recording_id": recording["recording_id"],
        "source_signature": recording["source_signature"],
        "settings": settings_snapshot(old_config, stage),
        "expected_key": historical_key, "receipt_key": receipt["key"],
        "receipt_digest": digest(receipt),
    }
    if queue_key is not None or queue_revision is not None:
        result.update(queue_key=queue_key, queue_revision=queue_revision)
    _validate_entry(result)
    return result


def _validate_entry(entry):
    stage = entry.get("stage")
    if stage not in STAGES:
        raise ValueError("Unsupported completed execution binding stage")
    for field in ("recording_id", "source_signature"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise ValueError("Invalid completed execution binding source")
    hashes = ["expected_key", "receipt_key", "receipt_digest"]
    if "queue_key" in entry or "queue_revision" in entry:
        hashes.extend(("queue_key", "queue_revision"))
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field, ""))) for field in hashes):
        raise ValueError("Invalid completed execution binding identity")
    expected = {"understanding": {"mllm"}, "stt": {"speech_recognition"}, "report": set()}[stage]
    settings = entry.get("settings")
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError("Invalid completed execution binding settings")
    if stage == "understanding" and (not isinstance(settings["mllm"], dict)
                                      or set(settings["mllm"]) - _MLLM_FIELDS):
        raise ValueError("Invalid completed provider settings")
    _safe(settings)


def make_hold(stage, recording, queue_revision, *, status, observed_key, observed_queue_key,
              reason="pre_switch_completion_unverifiable"):
    """Quarantine a pre-existing completion that cannot pass the old recipe.

    This does not validate or relabel its results. It only prevents a provider
    switch from turning old completed queue rows into new paid requests. The
    observed recipes bind current complete inputs, including comments/protocol.
    ``observed_queue_key`` is the actual queue revision, including the outer
    inplace queue digest (unlike a binding entry's inner ``queue_key``).
    Rebuild this explicit migration manifest before any later provider switch;
    it does not implement a general future "new tasks only" activation policy.
    """
    hold = {"stage": stage, "recording_id": recording["recording_id"],
            "source_signature": recording["source_signature"],
            "queue_revision": queue_revision, "status": status, "reason": reason,
            "observed_key": observed_key, "observed_queue_key": observed_queue_key}
    _validate_hold(hold)
    return hold


def _validate_hold(hold):
    if (hold.get("stage") not in STAGES or hold.get("status") != "completed"
            or hold.get("reason") != "pre_switch_completion_unverifiable"):
        raise ValueError("Only pre-existing unverifiable completions can be held")
    if any(not isinstance(hold.get(field), str) or not hold[field]
           for field in ("recording_id", "source_signature")):
        raise ValueError("Invalid completed hold source")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(hold.get(field, "")))
           for field in ("queue_revision", "observed_key", "observed_queue_key")):
        raise ValueError("Invalid completed hold queue identity")


def write_manifest(path, entries, *, holds=()):
    """Write an owned migration manifest; return its exact config reference."""
    manifest = {"schema_version": VERSION, "entries": list(entries), "holds": list(holds)}
    _entries(manifest)
    atomic_json(Path(path), manifest)
    return {"path": str(Path(path).resolve()), "sha256": file_hash(Path(path))}


def _entries(manifest):
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unknown completed execution binding manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) > 100_000:
        raise ValueError("Invalid completed execution binding manifest size")
    seen, keys = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid completed execution binding entry")
        _validate_entry(entry)
        identity = (entry["stage"], entry["recording_id"])
        if identity in seen or entry["expected_key"] in keys:
            raise ValueError("Duplicate completed execution binding")
        seen.add(identity)
        keys.add(entry["expected_key"])
    holds = manifest.get("holds", [])
    if not isinstance(holds, list) or len(holds) > 100_000:
        raise ValueError("Invalid completed hold manifest size")
    for hold in holds:
        if not isinstance(hold, dict):
            raise ValueError("Invalid completed hold entry")
        _validate_hold(hold)
        identity = (hold["stage"], hold["recording_id"])
        if identity in seen:
            raise ValueError("Duplicate or conflicting completed hold")
        seen.add(identity)
    return entries


class CompletedExecutionBindings:
    def __init__(self, settings):
        self.entries, self.by_key, self.holds = {}, {}, {}
        reference = settings.get(SETTING)
        if not reference:
            return
        path = Path(reference["path"])
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Completed execution binding manifest exceeds size limit")
        if file_hash(path) != reference.get("sha256"):
            raise ValueError("Completed execution binding manifest checksum mismatch")
        manifest = read_json(path)
        for entry in _entries(manifest):
            self.entries[(entry["stage"], entry["recording_id"])] = entry
            self.by_key[entry["expected_key"]] = entry
        for hold in manifest.get("holds", []):
            self.holds[(hold["stage"], hold["recording_id"])] = hold

    def held(self, stage, recording, *, key=None):
        hold = self.holds.get((stage, recording.get("recording_id")))
        if not hold or hold["source_signature"] != recording.get("source_signature"):
            return None
        return hold if key is None or key in {hold["observed_key"], hold["observed_queue_key"]} else None

    def candidate(self, stage, recording):
        entry = self.entries.get((stage, recording.get("recording_id")))
        return entry if entry and entry["source_signature"] == recording.get("source_signature") else None

    @staticmethod
    def config_for(config, entry):
        # Never mutate shared runner configuration across simultaneous stages.
        return {**config, **deepcopy(entry["settings"])}

    @staticmethod
    def match_key(entry, computed_key):
        return computed_key in {entry["expected_key"], entry.get("queue_key")}

    def requires(self, expected_key):
        return expected_key in self.by_key

    def accepts(self, receipt, expected_key, *, queue_revision=None):
        entry = self.by_key.get(expected_key)
        return bool(entry and isinstance(receipt, dict)
                    and receipt.get("status") == "completed"
                    and receipt.get("stage") == entry["stage"]
                    and receipt.get("recording_id") == entry["recording_id"]
                    and receipt.get("key") == entry["receipt_key"]
                    and (queue_revision is None or queue_revision == entry.get("queue_revision"))
                    and digest(receipt) == entry["receipt_digest"])

    def require_valid(self, receipt, expected_key):
        """Call after the normal source/artifact validator, including on None."""
        if self.requires(expected_key) and not self.accepts(receipt, expected_key):
            raise CompletedBindingInvalid(
                "Pinned completed stage is missing or invalid; quarantine it for review, "
                "do not execute the current provider under its historical key")
        return receipt

"""Explicit, receipt-specific continuity across a reviewed engine deployment.

An old completed result keeps its old execution key and model provenance. It is
never relabelled as a result of the new engine. New inputs, changed recipes,
changed receipts and damaged artifacts must use the normal execution path.
"""
from pathlib import Path
import re

from .device_day_contract import digest, file_hash, read_json


class CompletedVisionReceipts:
    def __init__(self, settings):
        self.entries = {}
        setting = settings.get("completed_vision_receipts")
        if not setting:
            return
        path = Path(setting["path"])
        if file_hash(path) != setting.get("sha256"):
            raise ValueError("Completed vision receipt manifest checksum mismatch")
        manifest = read_json(path)
        if manifest.get("schema_version") != "visioncortex-engine-rollover/1":
            raise ValueError("Unknown completed vision receipt manifest")
        for entry in manifest["entries"]:
            for field in ("expected_key", "receipt_key", "receipt_digest"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field, ""))):
                    raise ValueError("Invalid completed vision receipt identity")
            key = entry["expected_key"]
            if key in self.entries:
                raise ValueError("Duplicate completed vision receipt identity")
            self.entries[key] = entry

    def accepts(self, receipt, expected_key):
        entry = self.entries.get(expected_key)
        return bool(entry and receipt.get("stage") == "vision"
                    and receipt.get("status") == "completed"
                    and receipt.get("key") == entry["receipt_key"]
                    and receipt.get("recording_id") == entry["recording_id"]
                    and digest(receipt) == entry["receipt_digest"])


class CompletedStageReceipts:
    """Explicit historical checkpoints, not equivalence to a new algorithm.

    The manifest pins each receipt to one current input/config/code key. Queue
    completions retain their original revision. Actual downstream use must
    still pass DeviceDayRunner._load's artifact checksum verification.
    """
    def __init__(self, settings):
        self.entries = {}
        setting = settings.get("completed_stage_receipts")
        if not setting:
            return
        path = Path(setting["path"])
        if file_hash(path) != setting.get("sha256"):
            raise ValueError("Completed stage receipt manifest checksum mismatch")
        manifest = read_json(path)
        if manifest.get("schema_version") != "visioncortex-resume-checkpoints/1":
            raise ValueError("Unknown completed stage receipt manifest")
        for entry in manifest["entries"]:
            if entry.get("stage") not in {"retention", "vision", "stt", "understanding", "report"}:
                raise ValueError("Invalid checkpoint stage")
            if not isinstance(entry.get("recording_id"), str) or not entry["recording_id"]:
                raise ValueError("Invalid checkpoint recording")
            for field in ("expected_key", "receipt_key", "receipt_digest", "queue_revision"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field, ""))):
                    raise ValueError("Invalid checkpoint identity")
            key = (entry["stage"], entry["expected_key"])
            if key in self.entries:
                raise ValueError("Duplicate checkpoint identity")
            self.entries[key] = entry

    def accepts(self, receipt, expected_key, *, queue_revision=None):
        entry = self.entries.get((receipt.get("stage"), expected_key))
        return bool(entry and receipt.get("status") == "completed"
                    and receipt.get("recording_id") == entry["recording_id"]
                    and receipt.get("key") == entry["receipt_key"]
                    and (queue_revision is None or queue_revision == entry["queue_revision"])
                    and digest(receipt) == entry["receipt_digest"])

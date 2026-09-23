"""Prerequisite waits are scheduling state, not failed model invocations."""
import stat
import threading
import time

from .device_day_contract import STAGES, digest, safe_child


def legacy_prerequisite_failure(result):
    if result.get('error_type') != 'ValueError':
        return None
    return next((stage for stage in STAGES
                 if result.get('message') == f'Prerequisite stage {stage} is not ready'), None)


def artifact_metadata_identity(layout, backend_root, receipt):
    """Cheap rejection only; acceptance still requires _load's byte checks."""
    try:
        references = [(layout.root, item) for item in receipt['artifacts']]
        for item in receipt.get('audit_artifacts', []):
            if item.get('storage_root') != 'local_cache_root':
                return None
            references.append((backend_root, item))
        identities = []
        for root, item in references:
            path = safe_child(root, item['path'])
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != item['size_bytes']:
                return None
            identities.append((str(path), info.st_dev, info.st_ino, info.st_size,
                               info.st_mtime_ns, info.st_ctime_ns))
        return identities
    except (OSError, ValueError, KeyError, TypeError):
        return None


def verified_prerequisite(runner, path, key, layout, receipt):
    identities = artifact_metadata_identity(layout, runner.backend_root, receipt)
    if identities is None:
        return False
    # Cache rejection only while every receipt and file identity is unchanged.
    # This avoids repeatedly hashing a corrupt movie while it awaits repair.
    rejected = getattr(runner, '_rejected_prerequisites', None)
    if rejected is None:
        rejected = runner._rejected_prerequisites = {}
    identity = digest([key, receipt, identities])
    if identity in rejected:
        return False
    if runner._load(path, key, layout) is not None:
        return True
    if identities == artifact_metadata_identity(layout, runner.backend_root, receipt):
        rejected[identity] = True
        while len(rejected) > 4096:
            rejected.pop(next(iter(rejected)))
    return False


class PrerequisiteChecks:
    """One mailbox consumed by the existing recovery worker, never by admission.

    Grants describe one exact local receipt snapshot and are consumed once.
    Execution still verifies all artifacts after claim. A blocked NAS read can
    occupy only the recovery worker, not a stage's admission lock or cameras.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.pending = None
        self.running = None
        self.outcomes = {}

    def ready(self, recording_id, revision, checks):
        identity = digest([recording_id, revision, [(str(path), key, receipt) for path, key, _, receipt in checks]])
        with self.lock:
            outcome = self.outcomes.get(identity)
            if outcome:
                if outcome[1]:
                    self.outcomes.pop(identity)
                    return True
                if time.monotonic() < outcome[2]:
                    return False
                self.outcomes.pop(identity)
            if self.pending is None and self.running is None:
                self.pending = (identity, recording_id, checks)
        return False

    def invalidate_failed(self, recording_id):
        with self.lock:
            self.outcomes = {key: value for key, value in self.outcomes.items()
                             if value[0] != recording_id or value[1]}

    def verify_pending(self, runner):
        from .device_day_contract import read_json
        with self.lock:
            if self.pending is None or self.running is not None:
                return None
            identity, recording_id, checks = self.pending
            self.pending, self.running = None, identity
        valid = False
        try:
            valid = all(read_json(path) == receipt and verified_prerequisite(runner, path, key, layout, receipt)
                        and read_json(path) == receipt for path, key, layout, receipt in checks)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        finally:
            with self.lock:
                self.running = None
                self.outcomes[identity] = (recording_id, valid, time.monotonic() + 30)
                while len(self.outcomes) > 4096:
                    self.outcomes.pop(next(iter(self.outcomes)))
        return {'status': 'prerequisite_restored' if valid else 'waiting_for_prerequisite',
                'recording_id': recording_id, 'prerequisite_artifacts_verified': valid}

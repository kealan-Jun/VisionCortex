"""One immutable full activity audit per batch, with backward-readable summaries."""
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from .device_day_contract import atomic_json, digest, read_json, safe_child
from .device_day_verification import verify_artifact_cached


_WRITTEN = OrderedDict()
_WRITTEN_LOCK = Lock()


def persist(layout, recording_id, key, ordinal, audit):
    """Persist every audit field once; summaries never replace its evidence."""
    checksum = digest(audit)
    path = layout.receipts / recording_id / 'YOLO' / key / f'{ordinal:04d}' / f'ActivityAudit-{checksum}.json'
    safe_child(layout.backend_root, path.relative_to(layout.backend_root).as_posix())
    with _WRITTEN_LOCK:
        reference = _WRITTEN.get(str(path))
    # Only our own fully written and hashed artifacts enter this cache. A
    # matching filename or saved reference alone never grants content trust.
    if reference is None or not verify_artifact_cached(layout.backend_root, reference):
        atomic_json(path, audit)
        reference = layout.backend_artifact(path)
    with _WRITTEN_LOCK:
        _WRITTEN[str(path)] = reference
        _WRITTEN.move_to_end(str(path))
        while len(_WRITTEN) > 2048:
            _WRITTEN.popitem(last=False)
    summary = {name: audit[name] for name in (
        'status', 'scope', 'experiment_grouping', 'coarse_only_can_label_activity',
        'key_selection_summary', 'selected_key_events', 'activity_algorithm',
        'cross_camera_alignment_verified', 'physical_action_confirmed') if name in audit}
    summary.update(detail_storage='verified_activity_audit_reference',
                   event_count=len(audit.get('events', [])),
                   rejected_count=len(audit.get('rejected', [])))
    return summary, reference


def compact_index_visual(layout, visual, *, recording_id):
    """Project historical output without altering its receipt or execution key.

    Call only after receipt/key checks. Original full objects remain the input
    to semantic-cache validation; this separate view is for the public index.
    """
    if visual.get('status') != 'completed' or not visual.get('key'):
        return visual
    projected = dict(visual)
    known, references = [], []

    def compact(item):
        if item.get('activity_audit_ref') or not item.get('activity_audit'):
            return item
        audit = item['activity_audit']
        # Historical JSON expands one shared batch audit into each segment.
        # Compare the complete values before reusing a reference; no truncated
        # signature, event-count heuristic or time-window guess grants trust.
        for original, summary, reference in known:
            if audit is original or audit == original:
                return item | {'activity_audit': summary, 'activity_audit_ref': reference}
        summary, reference = persist(layout, recording_id, visual['key'], len(known), audit)
        known.append((audit, summary, reference))
        references.append(reference)
        return item | {'activity_audit': summary, 'activity_audit_ref': reference}

    for field in ('batches', 'segments'):
        if field in visual:
            projected[field] = [compact(item) for item in visual[field]]
    if references:
        projected['audit_artifacts'] = [*visual.get('audit_artifacts', []), *references]
    return projected


class AuditReader:
    """Resolve new references or historical inline audits; cache one verified batch."""

    def __init__(self, config):
        self.root = (config or {}).get('storage', {}).get('local_cache_root')
        self.cached = None

    def read(self, item):
        reference = item.get('activity_audit_ref')
        if reference is None:
            value = item.get('activity_audit') or {}
            if value.get('detail_storage') == 'verified_activity_audit_reference':
                raise ValueError('Activity audit reference is missing')
            return value
        if not isinstance(reference, dict) or not self.root or reference.get('storage_root') != 'local_cache_root':
            raise ValueError('Unknown activity audit storage root')
        root = Path(self.root)
        path = safe_child(root, reference['path'])
        before = path.stat()
        if not verify_artifact_cached(root, reference):
            raise ValueError('Activity audit failed content verification')
        identity = (str(path), before.st_dev, before.st_ino, before.st_size,
                    before.st_mtime_ns, before.st_ctime_ns, reference['sha256'])
        value = self.cached[1] if self.cached and self.cached[0] == identity else read_json(path)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Activity audit changed during reading')
        if not isinstance(value, dict):
            raise ValueError('Activity audit must be an object')
        self.cached = identity, value
        return value

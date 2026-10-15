"""Compact storage references, retaining every image and semantic input field."""
import json
import math
import re
from copy import deepcopy

VERSION = 'visioncortex-scene-wire/1'
DAY_CONTEXT_VERSION = 'visioncortex-day-context-wire/2'
STORAGE_FIELDS = frozenset({'path', 'source_path', 'size_bytes', 'sha256'})
FRAME_LEDGER_VERSION = 'visioncortex-scene-frame-ledger/2'
_FRAME_IDENTITY = ('sha256', 'size_bytes', 'local_ms', 'source_path')
_OPTIONAL_IDENTITY = ('capture_us', 'source_native_index', 'source_frame_index',
                      'source_frame_index_estimate', 'source_frame_verification')
_FRAME_PLACEMENT = frozenset({'path', *_FRAME_IDENTITY, *_OPTIONAL_IDENTITY, 'frame_id', 'frame_associations'})


def _frame_identity(frame):
    """Parent artifact verification owns bytes; this merge cannot weaken it."""
    checksum, size, stamp, source = (frame.get(name) for name in _FRAME_IDENTITY)
    if (not isinstance(checksum, str) or re.fullmatch('[0-9a-f]{64}', checksum) is None
            or type(size) is not int or size <= 0
            or type(stamp) not in {int, float} or not math.isfinite(stamp) or stamp < 0
            or not isinstance(source, str) or not source):
        raise ValueError('Scene frame identity is incomplete')
    return checksum, size, stamp, source


def _frame_associations(frame):
    # Keep every semantic extension/selected event while the canonical image
    # retains its original kind. A scene sample must not become an action frame.
    return [deepcopy({key: value for key, value in frame.items() if key not in _FRAME_PLACEMENT}),
            *deepcopy(frame.get('frame_associations', []))]


def merge_frame_ledger(frames):
    """One image per content/time identity, with every semantic association.

    Preserve first-seen order and placement. Unique ledgers remain identical,
    allowing their existing paid request receipts to pass the same validators.
    """
    result, seen = [], {}
    for frame in frames:
        identifier = frame.get('frame_id')
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Scene frame identifier is missing')
        if identifier not in seen:
            seen[identifier] = len(result)
            result.append(deepcopy(frame))
            continue
        existing = result[seen[identifier]]
        if (_frame_identity(existing) != _frame_identity(frame)
                or any((name in existing) != (name in frame) or existing.get(name) != frame.get(name)
                       for name in _OPTIONAL_IDENTITY)):
            raise ValueError('Duplicate scene frame identity has conflicting content or time')
        associations = existing.get('frame_associations')
        if associations is None:
            associations = _frame_associations(existing)
            existing['frame_associations'] = associations
        for association in _frame_associations(frame):
            if association not in associations:
                associations.append(association)
    return result


def prepare_frame_ledger(metadata, image_paths):
    """Keep the caller's validator, request and receipt on the same frame list."""
    original = metadata['frames']
    if [label for label, _ in image_paths] != [f['frame_id'] for f in original]:
        raise ValueError('Scene image labels/order differ from the supplied frame ledger')
    frames = merge_frame_ledger(original)
    if len(frames) == len(original):
        return image_paths
    paths = {}
    for label, path in image_paths:
        paths.setdefault(label, path)
    # An explicit in-memory ledger argument is updated only after every
    # collision passes. Parents and archived inputs are not changed; copies
    # preserve all event associations for this new request recipe.
    original[:] = frames
    metadata['frame_ledger_version'] = FRAME_LEDGER_VERSION
    return [(frame['frame_id'], paths[frame['frame_id']]) for frame in frames]


def project(metadata, *, aliases=False):
    value = deepcopy(metadata)
    frames = value['frames']
    ids = [frame['frame_id'] for frame in frames]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate scene frame identifiers')
    mapping = {f'f{i+1}': identifier for i, identifier in enumerate(ids)} if aliases else {}
    reverse = {full: short for short, full in mapping.items()}
    value['frames'] = [{k: reverse.get(v, v) if k == 'frame_id' else v
                        for k, v in frame.items() if k not in STORAGE_FIELDS} for frame in frames]
    # Keep any semantic extensions to the source reference, omitting only
    # filesystem placement and integrity fields which remain in Input.json.
    value['source_ref'] = {k: v for k, v in value.get('source_ref', {}).items() if k not in STORAGE_FIELDS}
    # A validation correction carries the rejected response as untrusted data.
    # Its frame labels must use the same wire aliases as the actual images.
    previous = (value.get('validation_repair') or {}).get('previous_observations')
    if reverse and isinstance(previous, dict):
        observations = previous.get('frame_observations')
        for observation in observations if isinstance(observations, list) else []:
            if isinstance(observation, dict) and isinstance(observation.get('frame_id'), str):
                observation['frame_id'] = reverse.get(observation['frame_id'], observation['frame_id'])
        for field in ('experiment_steps', 'work_activities'):
            steps = previous.get(field)
            for step in steps if isinstance(steps, list) else []:
                if isinstance(step, dict) and isinstance(step.get('frame_ids'), list):
                    step['frame_ids'] = [reverse.get(identifier, identifier) if isinstance(identifier, str) else identifier
                                         for identifier in step['frame_ids']]
    return value, mapping


def _day_completeness(value):
    """Keep gap times and counts in the wire; identities remain on disk."""
    from .device_day_narrative import timestamp
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = _day_completeness(item)
        elif key in {'missing_or_partial_segments', 'unprocessed_recordings', 'unresolved_index_records',
                     'pending_value_review_intervals'}:
            intervals = sorted((r['start_us'], r['end_us']) for r in item
                               if isinstance(r.get('start_us'), (int, float))
                               and isinstance(r.get('end_us'), (int, float)) and r['start_us'] < r['end_us'])
            merged = []
            for left, right in intervals:
                if merged and left <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], right)
                else:
                    merged.append([left, right])
            result[key] = {'record_count': len(item), 'untimed_record_count': len(item)-len(intervals),
                           'ranges': [{'start': timestamp(a), 'end': timestamp(b)} for a, b in merged]}
        else:
            result[key] = item
    return result


def compact(metadata, image_paths):
    wire, mapping = project(metadata, aliases=True)
    expected = [f['frame_id'] for f in metadata['frames']]
    if [label for label, _ in image_paths] != expected:
        raise ValueError('Scene image labels/order differ from the supplied frame ledger')
    reverse = {full: short for short, full in mapping.items()}
    day_aliases = None
    if metadata.get('schema_version') == 'visioncortex-day-context-input/1':
        from .device_day_narrative import timestamp
        wire['completeness'] = _day_completeness(wire.get('completeness', {}))
        coverage_id = wire.get('coverage_evidence_id')
        ids = dict.fromkeys(identifier for row in wire['evidence']
                            for identifier in ([row['evidence_id']] if 'evidence_id' in row else row['evidence_refs']))
        ids.pop(coverage_id, None)
        day_aliases = {f'e{i+1}': identifier for i, identifier in enumerate(ids)}
        if coverage_id:
            day_aliases['coverage'] = coverage_id
            wire['coverage_evidence_id'] = 'coverage'
            wire['citation_rule'] = 'evidence_refs只引用evidence_id或coverage；覆盖缺口引用coverage，不引用相机名或completeness路径。'
        day_reverse = {full: short for short, full in day_aliases.items()}
        for row in wire['evidence']:
            # Storage provenance stays in Input.json, rather than consuming
            # model context repeatedly. Observations, clocks and limits stay.
            row.pop('source', None)
            if row.get('kind') == 'coverage':
                row['completeness'] = _day_completeness(row['completeness'])
            if row.get('start') and row.get('end'):
                row.pop('start_us', None)
                row.pop('end_us', None)
            for field in ('experiment_steps', 'work_activities'):
                for step in row.get(field, []):
                    for prefix in ('start', 'end'):
                        if step.get(prefix+'_us') is not None:
                            step[prefix] = timestamp(step.pop(prefix+'_us'))
                            step.pop(prefix+'_ms', None)
            if 'evidence_id' in row:
                row['evidence_id'] = day_reverse[row['evidence_id']]
            if 'evidence_refs' in row:
                row['evidence_refs'] = [day_reverse[r] for r in row['evidence_refs']]
            person = row.get('experimenter')
            if person:
                row['experimenter'] = {k: v for k, v in person.items() if k not in ('绑定记录', '说明')}
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    receipt = {'version': VERSION, 'frame_aliases': mapping,
               'metadata_characters_before': size(metadata), 'metadata_characters_after': size(wire),
               'image_count': len(image_paths), 'images_changed': False,
               'timestamps_changed': False, 'quality_equivalence': 'NOT_PROVEN'}
    if day_aliases is not None:
        receipt.update(day_context_version=DAY_CONTEXT_VERSION, day_evidence_aliases=day_aliases)
    return wire, [(reverse[label], path) for label, path in image_paths], receipt


def expand(result, receipt):
    value = deepcopy(result)
    mapping = receipt['frame_aliases']
    def original(identifier):
        if not isinstance(identifier, str) or identifier not in mapping:
            raise ValueError('Model invented a compact scene frame reference')
        return mapping[identifier]
    for row in value.get('frame_observations', []):
        row['frame_id'] = original(row['frame_id'])
    for field in ('steps', 'experiment_steps', 'work_activities'):
        for step in value.get(field, []):
            step['frame_ids'] = [original(identifier) for identifier in step['frame_ids']]
    if 'day_evidence_aliases' in receipt:
        aliases = receipt['day_evidence_aliases']
        for finding in value.get('findings', []):
            refs = finding.get('evidence_refs')
            if not isinstance(refs, list) or any(not isinstance(r, str) or r not in aliases for r in refs):
                raise ValueError('Model invented a day context source reference')
            finding['evidence_refs'] = [aliases[r] for r in refs]
    value['scene_input_transport'] = receipt
    return value

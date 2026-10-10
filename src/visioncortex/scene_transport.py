"""Compact storage references, retaining every image and semantic input field."""
from copy import deepcopy
import json

VERSION = 'visioncortex-scene-wire/1'
STORAGE_FIELDS = frozenset({'path', 'source_path', 'size_bytes', 'sha256'})


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
    return value, mapping


def compact(metadata, image_paths):
    wire, mapping = project(metadata, aliases=True)
    expected = [f['frame_id'] for f in metadata['frames']]
    if [label for label, _ in image_paths] != expected:
        raise ValueError('Scene image labels/order differ from the supplied frame ledger')
    reverse = {full: short for short, full in mapping.items()}
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    receipt = {'version': VERSION, 'frame_aliases': mapping,
               'metadata_characters_before': size(metadata), 'metadata_characters_after': size(wire),
               'image_count': len(image_paths), 'images_changed': False,
               'timestamps_changed': False, 'quality_equivalence': 'NOT_PROVEN'}
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
    for field in ('steps', 'experiment_steps'):
        for step in value.get(field, []):
            step['frame_ids'] = [original(identifier) for identifier in step['frame_ids']]
    value['scene_input_transport'] = receipt
    return value

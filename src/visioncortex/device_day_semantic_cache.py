"""Reuse a model response only when its visual/textual input is unchanged."""
from __future__ import annotations

from pathlib import Path

from .device_day_contract import artifact, atomic_json, digest, read_json, verify_artifact


def identity(config, prompt, metadata):
    provider = config.get('mllm') or {}
    return {'schema_version': 'visioncortex-scene-response/1', 'prompt': prompt,
            'provider': {k: provider.get(k) for k in ('provider', 'model', 'base_url', 'temperature', 'quality_mode', 'api_protocol')},
            'source_sha256': metadata['source_ref']['sha256'],
            'activity': metadata['activity'], 'start_ms': metadata['start_ms'], 'end_ms': metadata['end_ms'],
            'frames': [{k: frame[k] for k in ('frame_id', 'local_ms', 'sha256')} for frame in metadata['frames']],
            'comments': metadata['comments'], 'protocol': metadata['protocol']}


def location(config, value):
    return Path(config['storage']['local_cache_root']) / 'device-day-understanding' / digest(value)


def save(config, value, model_result, *, provenance=None):
    root = location(config, value)
    path = root / 'model-result.json'
    atomic_json(path, model_result)
    atomic_json(root / 'receipt.json', {'identity': value, 'result': artifact(root, path),
                                       'provenance': provenance or {'actual_call': True}})


def load(config, value):
    root = location(config, value)
    path = root / 'receipt.json'
    if not path.is_file():
        return None
    receipt = read_json(path)
    if digest(receipt.get('identity')) != digest(value) or not verify_artifact(root, receipt['result']):
        raise ValueError('Cached scene model evidence was modified')
    result = read_json(root / receipt['result']['path'])
    return result if result.get('status') == 'completed' else None


def reject(config, value):
    """Preserve invalid model evidence without reusing it on an explicit retry."""
    import uuid
    root = location(config, value)
    destination = root.parent / 'invalid' / (root.name + '-' + uuid.uuid4().hex)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root.rename(destination)

"""One validated window cache and accounting boundary for device-day vision calls."""
from pathlib import Path

from . import device_day_semantic_cache as cache
from .device_day_contract import atomic_json, digest, read_json
from .mllm_provider import vision_request_identity
from .multimodal_usage import UsageLedger
from .scene_transport import VERSION, compact, expand, project


def request_policy(config):
    return {'generation': vision_request_identity(config['mllm']),
            'scene_transport': VERSION if config['mllm'].get('compact_scene_metadata', False) else 'original'}


def _legacy_policy_known(config, raw):
    """Old receipts did not bind the output budget; accept only its old default."""
    from .mllm import SCENE_TIME_REFERENCE_RULES
    settings = config['mllm']
    return (not settings.get('compact_scene_metadata', False)
            and int(settings.get('max_output_tokens', 4096)) == 4096
            and raw.get('model') == settings.get('model')
            and raw.get('provider') == settings.get('provider', 'volcengine')
            and raw.get('api_protocol') == settings.get('api_protocol', 'ark_responses')
            and raw.get('scene_timing_contract') == {
                'version': 'visioncortex-scene-time-references/1',
                'instructions': SCENE_TIME_REFERENCE_RULES, 'validation_relaxed': False}
            and raw.get('request_image_transport') == {
                'maximum_edge_pixels': int(settings.get('request_image_max_edge', 0)),
                'jpeg_quality': int(settings.get('request_image_jpeg_quality', 85)),
                'source_artifacts_mutated': False})


def _legacy_candidate(config, legacy, prompt, semantic, request_path):
    """Recover paid windows across stage-key folders, with full input evidence."""
    raw = cache.load(config, legacy)
    if not raw or not _legacy_policy_known(config, raw):
        return None
    # Only sibling stage revisions of this segment/window, never a day-wide
    # search or a relaxed comparison of the legacy cache's incomplete key.
    paths = {request_path, *request_path.parent.parent.parent.glob(f'*/{request_path.parent.name}/Input.json')}
    for path in sorted(paths):
        result_path = path.with_name('Result.json')
        if not path.is_file() or not result_path.is_file():
            continue
        previous, result = read_json(path), read_json(result_path)
        old_metadata = previous.get('metadata')
        if (not isinstance(old_metadata, dict) or previous.get('prompt') != prompt
                or result.get('model_result') != raw or result.get('request_policy') is not None):
            continue
        if (previous.get('input_key') == result.get('input_key') == digest([old_metadata, prompt])
                and project(old_metadata)[0] == semantic):
            return raw
    return None


def window(config, analyzer, prompt, metadata, image_paths, request_path, result_path, validator):
    from .device_day import exclusive
    legacy = cache.identity(config, prompt, metadata)
    policy = request_policy(config)
    semantic, _ = project(metadata)
    identity = legacy | {'schema_version': 'visioncortex-scene-response/2',
                         'request_policy': policy, 'semantic_metadata': semantic}
    input_key = digest([metadata, prompt])
    # Independent callers/queue retries cannot both miss and pay for this
    # exact window. The existing stage handles lock contention as queued work.
    lock = Path(config['storage']['local_runtime_root'])/'device-day'/'locks'/f'scene-{digest(identity)}.lock'
    with exclusive(lock):
        candidates = []
        local_candidate = None
        reservation = None
        if request_path.is_file() and result_path.is_file():
            previous, result = read_json(request_path), read_json(result_path)
            if previous.get('input_key') == input_key == result.get('input_key'):
                candidate = result.get('model_result') or {}
                if result.get('request_policy') == policy:
                    candidates.append(candidate)
                    local_candidate = candidate
                    reservation = result.get('usage_reservation')
        saved = cache.load(config, identity)
        if saved:
            candidates.append(saved)
        if not candidates:
            old = _legacy_candidate(config, legacy, prompt, semantic, request_path)
            if old:
                candidates.append(old)
        raw = parsed = None
        for candidate in candidates:
            if candidate.get('status') != 'completed':
                continue
            try:
                parsed = validator(candidate)
            except (ValueError, KeyError, TypeError):
                continue
            raw = candidate
            break
        reused = raw is not None
        if raw is not local_candidate:
            reservation = None
        wire_receipt = wire_response = None
        if raw is None:
            wire_metadata, wire_images = metadata, image_paths
            if config['mllm'].get('compact_scene_metadata', False):
                wire_metadata, wire_images, wire_receipt = compact(metadata, image_paths)
            ledger = UsageLedger(config)
            if result_path.is_file():
                previous = {'input': read_json(request_path) if request_path.is_file() else None,
                            'result': read_json(result_path)}
                atomic_json(result_path.parent/'History'/f'{digest(previous)}.json', previous)
            # Input provenance exists before any paid call; the usage ledger
            # stays pending if the process dies before a receipt is available.
            atomic_json(request_path, {'input_key': input_key, 'prompt': prompt, 'metadata': metadata,
                                      'request_policy': policy, 'scene_input_transport': wire_receipt})
            reservation = ledger.reserve(digest(identity))
            wire_response = analyzer._call(prompt, wire_metadata, wire_images, max_images=len(wire_images))
            raw = wire_response
            if wire_receipt and raw.get('status') == 'completed':
                try:
                    raw = expand(raw, wire_receipt)
                except (ValueError, KeyError, TypeError) as exc:
                    raw = {**raw, 'status': 'failed', 'error': 'Invalid compact frame references',
                           'validation_error_type': type(exc).__name__}
        # Preserve failed attempts, billed usage and previous successful
        # receipts before replacing the current window pointer.
        if reused and result_path.is_file():
            previous = {'input': read_json(request_path) if request_path.is_file() else None,
                        'result': read_json(result_path)}
            atomic_json(result_path.parent/'History'/f'{digest(previous)}.json', previous)
        atomic_json(request_path, {'input_key': input_key, 'prompt': prompt, 'metadata': metadata,
                                  'request_policy': policy, 'scene_input_transport': raw.get('scene_input_transport')})
        atomic_json(result_path, {'input_key': input_key, 'request_policy': policy, 'model_result': raw,
                                 'response_cache_reused': reused,
                                 'usage_reservation': reservation,
                                 **({'wire_response': wire_response} if wire_receipt else {})})
        # Publish the paid response first. A restart can finish its pending
        # ledger reservation from this durable receipt without paying again.
        if reservation:
            UsageLedger(config).finish(reservation, raw)
        if raw.get('status') != 'completed':
            from .device_day_provider_gate import ProviderGate
            ProviderGate(config).record_failure(raw)
            raise ValueError('Multimodal window failed; validated preceding windows remain reusable')
        parsed = parsed if parsed is not None else validator(raw)
        # Also migrate validated legacy/local results into the stronger cache
        # identity without invoking the model or rewriting historical receipts.
        if saved != raw:
            if saved:
                cache.reject(config, identity)
            cache.save(config, identity, raw, provenance={'actual_call': not reused})
        return raw, parsed, reused


def additional_usage(raw, reused):
    fields = ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')
    return ({**dict.fromkeys(fields, 0), 'actual_call': False} if reused
            else {**(raw.get('usage') or {}), 'actual_call': True})

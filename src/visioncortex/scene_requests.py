"""One validated window cache and accounting boundary for device-day vision calls."""
from pathlib import Path

from . import device_day_semantic_cache as cache
from .device_day_contract import atomic_json, digest, read_json
from .mllm_provider import vision_request_identity
from .multimodal_usage import SubmittedUnknown, UsageLedger
from .scene_transport import VERSION, compact, expand, prepare_frame_ledger, project


def request_policy(config):
    return {'generation': vision_request_identity(config['mllm']),
            'scene_transport': VERSION if config['mllm'].get('compact_scene_metadata', False) else 'original'}


def request_identity(config, prompt, metadata):
    semantic, _ = project(metadata)
    policy = request_policy(config)
    if (config['mllm'].get('compact_scene_metadata', False)
            and metadata.get('schema_version') == 'visioncortex-day-context-input/1'):
        from .scene_transport import DAY_CONTEXT_VERSION
        policy['day_context_transport'] = DAY_CONTEXT_VERSION
    return cache.identity(config, prompt, metadata) | {
        'schema_version': 'visioncortex-scene-response/2',
        'request_policy': policy, 'semantic_metadata': semantic}


def request_key(config, prompt, metadata):
    """The same exact input identity used by caches and the paid-call ledger."""
    return digest(request_identity(config, prompt, metadata))


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


def _exact_sibling_candidate(config, identity, prompt, request_path):
    """Reuse a received response across parser revisions, never relabel inputs.

    A conservative parser fix may make a previously rejected response usable.
    It must not resubmit an identical paid request merely because its output
    directory has a new execution identity. The original receipt stays intact.
    """
    def stage_key(value):
        return len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if not stage_key(request_path.parent.parent.name):
        return None
    for path in sorted(request_path.parent.parent.parent.glob(f'*/{request_path.parent.name}/Input.json')):
        result_path = path.with_name('Result.json')
        if path == request_path or not stage_key(path.parent.parent.name) or not result_path.is_file():
            continue
        previous, result = read_json(path), read_json(result_path)
        metadata = previous.get('metadata')
        if (previous.get('prompt') != prompt or not isinstance(metadata, dict)
                or previous.get('input_key') != result.get('input_key')
                or previous.get('input_key') != digest([metadata, prompt])
                or result.get('request_policy') != identity['request_policy']):
            continue
        if request_identity(config, prompt, metadata) == identity:
            return result.get('model_result')
    return None


def window(config, analyzer, prompt, metadata, image_paths, request_path, result_path, validator):
    from .device_day_write_guard import storage_guard
    with storage_guard(config):
        return _window(config, analyzer, prompt, metadata, image_paths, request_path, result_path, validator)


def _window(config, analyzer, prompt, metadata, image_paths, request_path, result_path, validator):
    from .device_day import exclusive
    from contextlib import ExitStack
    from .runtime_control import CURRENT, check_yield, defer_yield, resource_slot, sync_files
    check_yield(checkpoint=False)
    image_paths = prepare_frame_ledger(metadata, image_paths)
    legacy = cache.identity(config, prompt, metadata)
    identity = request_identity(config, prompt, metadata)
    policy, semantic = identity['request_policy'], identity['semantic_metadata']
    paid_key = digest(identity)
    input_key = digest([metadata, prompt])
    # Independent callers/queue retries cannot both miss and pay for this
    # exact window. The existing stage handles lock contention as queued work.
    lock = Path(config['storage']['local_runtime_root'])/'device-day'/'locks'/f'scene-{paid_key}.lock'
    with exclusive(lock), ExitStack() as unit:
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
        sibling = _exact_sibling_candidate(config, identity, prompt, request_path)
        if sibling:
            candidates.append(sibling)
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
        ledger = None
        if raw is None:
            ledger = UsageLedger(config)
            recovered = ledger.recover(paid_key)
            if recovered:
                reservation, raw = recovered
                reused = True
        if raw is None:
            if CURRENT.get().source == 'device_day_backfill' or CURRENT.get().yield_signal is not None:
                # Admission precedes the budget reservation. Nested adapter
                # requests reuse this same-thread slot through receipt save.
                unit.enter_context(resource_slot(config, 'cloud'))
                unit.enter_context(defer_yield())
            wire_metadata, wire_images = metadata, image_paths
            if config['mllm'].get('compact_scene_metadata', False):
                wire_metadata, wire_images, wire_receipt = compact(metadata, image_paths)
            ledger = ledger or UsageLedger(config)
            if result_path.is_file():
                previous = {'input': read_json(request_path) if request_path.is_file() else None,
                            'result': read_json(result_path)}
                atomic_json(result_path.parent/'History'/f'{digest(previous)}.json', previous)
            # Input provenance exists before any paid call; the usage ledger
            # stays pending if the process dies before a receipt is available.
            atomic_json(request_path, {'input_key': input_key, 'prompt': prompt, 'metadata': metadata,
                                      'request_policy': policy, 'scene_input_transport': wire_receipt})
            from .device_day_write_guard import verify_before_submission
            verify_before_submission()
            reservation = ledger.reserve(paid_key)
            try:
                wire_response = analyzer._call(prompt, wire_metadata, wire_images, max_images=len(wire_images))
            except BaseException as exc:
                if getattr(exc, 'external_submission_started', None) is False:
                    not_submitted = {'status': 'not_submitted', 'attempts': 0,
                        'reason': 'provider_admission_yielded_before_http_submission',
                        'usage': dict.fromkeys(('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens'), 0)}
                    ledger.received(reservation, not_submitted)
                    atomic_json(result_path.parent/'Admission.json', {
                        'input_key': input_key, 'usage_reservation': reservation, 'receipt': not_submitted})
                    ledger.finish(reservation, not_submitted)
                    ledger.published(reservation)
                else:
                    ledger.submitted_unknown(reservation)
                raise
            raw = wire_response
            if wire_receipt and raw.get('status') == 'completed':
                try:
                    raw = expand(raw, wire_receipt)
                except (ValueError, KeyError, TypeError) as exc:
                    raw = {**raw, 'status': 'failed', 'error': 'Invalid compact frame references',
                           'validation_error_type': type(exc).__name__}
            # This bounded SSD receipt survives NAS loss after a paid response.
            # A restart replays publication rather than submitting again.
            ledger.received(reservation, raw)
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
            ledger = ledger or UsageLedger(config)
            ledger.finish(reservation, raw)
            ledger.published(reservation)
        if raw.get('status') != 'completed':
            from .device_day_provider_gate import ProviderGate
            ProviderGate(config).record_failure(raw)
            if raw.get('status') == 'submitted_unknown':
                raise SubmittedUnknown('submitted_unknown: received an unresolved external request outcome; reconcile before retry')
            raise ValueError('Multimodal window failed; validated preceding windows remain reusable')
        parsed = parsed if parsed is not None else validator(raw)
        # Also migrate validated legacy/local results into the stronger cache
        # identity without invoking the model or rewriting historical receipts.
        if saved != raw:
            if saved:
                cache.reject(config, identity)
            cache.save(config, identity, raw, provenance={'actual_call': not reused})
    # A new live arrival cannot discard a paid response or leave its usage
    # reservation pending. Resume will validate and reuse this saved unit.
    response_cache = cache.location(config, identity)
    check_yield(checkpoint=not reused, persist=lambda: sync_files(request_path, result_path,
                response_cache/'model-result.json', response_cache/'receipt.json'))
    return raw, parsed, reused


def additional_usage(raw, reused):
    fields = ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens')
    return ({**dict.fromkeys(fields, 0), 'actual_call': False} if reused
            else {**(raw.get('usage') or {}), 'actual_call': True})

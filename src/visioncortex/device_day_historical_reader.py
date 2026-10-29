"""Explicit receipt-only readers; native producer identities never become aliases.

The reviewed R2 audit proves a completed producer, not its missing environment
or permission to run it again. Reader descriptors bind today's exact inputs
and a narrowly reviewed association-only delta. No function launches a model.
"""
from copy import deepcopy
from pathlib import Path
import json
import re
import sqlite3
import time

from .device_day_bindings import CompletedBindingInvalid, _safe
from .device_day_contract import archive_name, digest, file_hash, read_json

KIND = 'receipt_only_historical'
VERSION = 'visioncortex-historical-receipt-reader/1'
PRODUCER = 'visioncortex-native-r2-producer-proof/1'


def _document(path, reference=None):
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or not path.is_file() or before.st_size > 64 * 1024 * 1024:
        raise ValueError('Historical reader document is not bounded regular evidence')
    checksum = file_hash(path)
    value = read_json(path)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError('Historical reader evidence changed during read')
    if reference and (before.st_size != reference['size_bytes'] or checksum != reference['sha256']):
        raise ValueError('Historical producer proof checksum differs')
    return value, checksum


def reader_identity(config):
    """Actual source bytes and non-secret reader configuration, not a model key."""
    from .device_day_bindings import settings_snapshot
    from .source_identity import SOURCE_RECIPE, source_manifest
    settings = {key: value for key, value in (config.get('device_day') or {}).items()
                if key not in {'completed_execution_bindings', 'completed_stage_receipts', 'completed_vision_receipts'}}
    return digest({'recipe': VERSION, 'source_recipe': SOURCE_RECIPE, 'sources': source_manifest(),
                   'settings': settings, 'mllm': settings_snapshot(config, 'understanding'),
                   'storage': {key: (config.get('storage') or {}).get(key) for key in
                               ('archive_root', 'local_cache_root', 'local_runtime_root')},
                   'capture_root': (config.get('collection_ingest') or {}).get('source_root')})


def descriptor_key(entry):
    return digest({'recipe': VERSION, 'reader_descriptor': {key: value for key, value in entry.items()
                                                          if key != 'reader_key'}})


def _profile_identities(reader):
    """An optional audited list preserves every complete runtime identity.

    The baseline identity remains the publication proof's stable identity.
    Names describe reviewed service profiles; no name or partial setting can
    grant access without the exact full source/configuration identity.
    """
    if 'runtime_profiles' not in reader:
        return {reader.get('identity')}
    profiles = reader['runtime_profiles']
    if not isinstance(profiles, list) or not 1 <= len(profiles) <= 16:
        raise ValueError('Historical reader requires a bounded explicit runtime profile list')
    names, identities = [], set()
    for profile in profiles:
        if (not isinstance(profile, dict) or set(profile) != {'name', 'identity'}
                or not isinstance(profile['name'], str)
                or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,63}', profile['name'])
                or not re.fullmatch(r'[0-9a-f]{64}', str(profile['identity']))):
            raise ValueError('Invalid historical reader runtime profile')
        names.append(profile['name'])
        identities.add(profile['identity'])
    if names != sorted(names) or len(names) != len(set(names)) or reader.get('identity') not in identities:
        raise ValueError('Historical reader profiles must include the baseline with unique ordered names')
    return identities


def validate_entry(entry):
    if entry.get('kind') != KIND or entry.get('stage') != 'understanding':
        raise ValueError('Unsupported historical receipt-only reader')
    if any(not isinstance(entry.get(key), str) or not entry[key] for key in
           ('recording_id', 'source_signature', 'archive')):
        raise ValueError('Historical reader requires an exact source')
    producer, reader = entry.get('producer') or {}, entry.get('reader') or {}
    _profile_identities(reader)
    hashes = [entry.get(key) for key in ('reader_key', 'receipt_key', 'receipt_digest', 'queue_revision')]
    hashes += [producer.get(key) for key in ('native_key', 'native_owner_sha256', 'inputs_digest', 'context_digest')]
    hashes += [reader.get(key) for key in ('identity', 'record_digest', 'producer_record_digest', 'context_digest')]
    hashes += list((reader.get('parents') or {}).values())
    if (producer.get('recipe') != PRODUCER or reader.get('recipe') != VERSION
            or not re.fullmatch(r'[0-9a-f]{40}', str(producer.get('source_sha', '')))
            or any(not re.fullmatch(r'[0-9a-f]{64}', str(value or '')) for value in hashes)
            or set(reader.get('parents') or {}) != {'retention', 'vision', 'stt'}
            or not isinstance(reader.get('historical_context'), dict)
            or reader.get('additions') != {'events': [], 'field_event_status': 'no_archive_for_camera_day'}
            or any(key in reader['historical_context'] for key in reader['additions'])
            or digest(reader['historical_context']) != producer['context_digest']
            or digest(reader['historical_context'] | reader['additions']) != reader['context_digest']
            or entry['receipt_key'] != producer['native_key'] or descriptor_key(entry) != entry['reader_key']):
        raise ValueError('Invalid historical reader descriptor or association-only delta')
    for field in ('audit', 'binding_audit'):
        reference = producer.get(field) or {}
        if (not isinstance(reference.get('path'), str) or not Path(reference['path']).is_absolute()
                or type(reference.get('size_bytes')) is not int or not 0 < reference['size_bytes'] <= 64 * 1024 * 1024
                or not re.fullmatch(r'[0-9a-f]{64}', str(reference.get('sha256', '')))):
            raise ValueError('Historical reader requires exact local producer proofs')
    _safe(entry)


def _producer(entry, receipt, parents):
    """Check the reviewed native proof separately from the current reader."""
    producer = entry['producer']
    audit, _ = _document(producer['audit']['path'], producer['audit'])
    binding, _ = _document(producer['binding_audit']['path'], producer['binding_audit'])
    native = binding.get('historical_native_execution') or {}
    provider = (producer.get('settings') or {}).get('mllm') or {}
    paid = audit.get('completed_paid_evidence') or {}
    window_count = sum(len(value.get('windows', [])) for value in receipt.get('understandings', []))
    recipe = next((row for row in audit.get('native_recipe_reproduction', [])
                   if row.get('stage') == 'understanding'), {})
    if (audit.get('schema') != 'visioncortex-readonly-historical-producer-audit/1'
            or binding.get('schema') != 'visioncortex-readonly-historical-binding-assessment/1'
            or audit.get('recording_id') != entry['recording_id'] or binding.get('recording_id') != entry['recording_id']
            or (audit.get('source') or {}).get('git_sha') != producer['source_sha']
            or (audit.get('source') or {}).get('native_owner_sha256') != producer['native_owner_sha256']
            or (audit.get('source') or {}).get('tracked_and_untracked_worktree_clean') is not True
            or (binding.get('producer_audit') or {}).get('sha256') != producer['audit']['sha256']
            or (audit.get('configuration_proof') or {}).get('provider') != provider.get('provider')
            or (audit.get('configuration_proof') or {}).get('model') != provider.get('model')
            or native.get('source_sha') != producer['source_sha'] or native.get('stage') != entry['stage']
            or native.get('native_key') != producer['native_key'] or native.get('settings_snapshot') != producer['settings']
            or recipe.get('native_expected_key') != producer['native_key']
            or recipe.get('receipt_key') != producer['native_key'] or recipe.get('equal_full_64_characters') is not True
            or recipe.get('inputs_digest') != producer['inputs_digest']
            or paid.get('status') != 'PROVEN' or paid.get('receipt_all_artifacts_verified') is not True
            or paid.get('ledger_unknown_attempts') != 0 or paid.get('complete_receipt_and_queue_payload_stable') is not True
            or not isinstance(receipt.get('artifacts'), list) or not receipt['artifacts']
            or len(receipt['artifacts']) != paid.get('artifact_count')
            or sum(value['size_bytes'] for value in receipt['artifacts']) != paid.get('artifact_bytes')
            or not window_count or paid.get('window_count') != window_count
            or type(paid.get('ledger_attempts')) is not int or paid['ledger_attempts'] < window_count
            or (audit.get('input_proof') or {}).get('context_digest') != producer['context_digest']
            or (audit.get('input_proof') or {}).get('saved_context_digest') != producer['context_digest']
            or (audit.get('queue_proof') or {}).get('revision') != entry['queue_revision']
            or (audit.get('source_files_proof') or {}).get('all_original_and_retained_bytes_match') is not True
            or digest({'vision': parents['vision'], 'stt': parents['stt'],
                       'context': entry['reader']['historical_context']}) != producer['inputs_digest']):
        raise ValueError('Historical native producer proof does not match the exact saved execution')
    fingerprints = (audit.get('configuration_proof') or {}).get('matching_historical_binding_fingerprints') or {}
    if any(native.get(key) != fingerprints.get(key) or not native.get(key) for key in
           ('connection_sha256', 'request_policy_sha256', 'credential_revision_sha256', 'verification_sha256')):
        raise ValueError('Historical provider proof fingerprints differ')
    for stage, value in (parents | {'understanding': receipt}).items():
        proof = next((row for row in audit.get('receipt_proof', []) if row.get('stage') == stage), {})
        if (value.get('stage') != stage or value.get('recording_id') != entry['recording_id']
                or value.get('status') != 'completed' or proof.get('status') != 'completed'
                or proof.get('receipt_key') != value.get('key') or proof.get('receipt_digest') != digest(value)):
            raise ValueError('Historical producer parent or completion differs')
    if (receipt.get('key') != entry['receipt_key'] or digest(receipt) != entry['receipt_digest']
            or receipt.get('vision_key') != parents['vision']['key']
            or receipt.get('context_digest') != producer['context_digest']):
        raise ValueError('Historical completion source/context differs')
    return audit


def _queue_snapshot(config, entry):
    """An existing exact queue generation is evidence, never a write target."""
    from .sqlite_store import connection
    path = Path(config['storage']['local_runtime_root'])/'device-day/queue-understanding.sqlite3'
    with connection(path, readonly=True) as db:
        row = db.execute('SELECT * FROM recordings WHERE recording_id=?', (entry['recording_id'],)).fetchone()
    value = dict(row) if row else {}
    result = json.loads(value.get('result') or '{}')
    if (value.get('revision') != entry['queue_revision']
            or digest(json.loads(value.get('payload') or '{}')) != entry['reader']['record_digest']
            or value.get('status') == 'running' and (value.get('lease_until') or 0) > time.time()
            or value.get('blocked_reason') == 'submitted_unknown'
            or result.get('reason') == 'submitted_unknown' or result.get('status') == 'submitted_unknown'):
        raise ValueError('Historical reader queue generation changed or contains a new unknown request')
    return value


def make_entry(config, recording, parents, receipt, *, producer_audit, binding_audit,
               historical_context, current_context, queue_revision, profiles=None):
    """Prepare an explicit reviewed descriptor; never grants execution or writes."""
    audit, _ = _document(producer_audit['path'], producer_audit)
    binding, _ = _document(binding_audit['path'], binding_audit)
    native = binding['historical_native_execution']
    recipe = next(row for row in audit['native_recipe_reproduction'] if row['stage'] == 'understanding')
    additions = {key: value for key, value in current_context.items() if key not in historical_context}
    if historical_context | additions != current_context:
        raise ValueError('Historical model inputs changed; not an association-only addition')
    entry = {'kind': KIND, 'stage': 'understanding', 'recording_id': recording['recording_id'],
             'source_signature': recording['source_signature'],
             'archive': archive_name(recording['camera_key'], recording['recording_start_us']),
             'receipt_key': receipt['key'], 'receipt_digest': digest(receipt), 'queue_revision': queue_revision,
             'producer': {'recipe': PRODUCER, 'audit': deepcopy(producer_audit), 'binding_audit': deepcopy(binding_audit),
                 'source_sha': audit['source']['git_sha'], 'native_owner_sha256': audit['source']['native_owner_sha256'],
                 'native_key': native['native_key'], 'settings': deepcopy(native['settings_snapshot']),
                 'inputs_digest': recipe['inputs_digest'], 'context_digest': digest(historical_context)},
             'reader': {'recipe': VERSION, 'identity': reader_identity(config), 'record_digest': digest(recording),
                 'producer_record_digest': digest(parents['retention']['recording']),
                 'parents': {stage: digest(value) for stage, value in parents.items()},
                 'historical_context': deepcopy(historical_context), 'additions': deepcopy(additions),
                 'context_digest': digest(current_context)}}
    if profiles is not None:
        if (not isinstance(profiles, dict) or not 1 <= len(profiles) <= 16
                or any(not isinstance(name, str) or not isinstance(profile, dict)
                       for name, profile in profiles.items())):
            raise ValueError('Historical reader profiles require explicit named effective configurations')
        entry['reader']['runtime_profiles'] = [
            {'name': name, 'identity': reader_identity(profile)} for name, profile in sorted(profiles.items())]
    entry['reader_key'] = descriptor_key(entry)
    validate_entry(entry)
    _producer(entry, receipt, parents)
    return entry


def current_key(config, entry, recording, inputs):
    """Select a read capability before any provider-recipe fallback."""
    try:
        validate_entry(entry)
        if isinstance(inputs, list) and len(inputs) == 2:
            parents, context = inputs
        else:
            parents, context = inputs, inputs['context']
        if (recording['recording_id'] != entry['recording_id'] or recording['source_signature'] != entry['source_signature']
                or digest(recording) not in {entry['reader']['record_digest'], entry['reader']['producer_record_digest']}
                or archive_name(recording['camera_key'], recording['recording_start_us']) != entry['archive']
                or reader_identity(config) not in _profile_identities(entry['reader'])
                or digest(context) != entry['reader']['context_digest']
                or 'retention' in parents and digest(parents['retention']) != entry['reader']['parents']['retention']
                or any(digest(parents.get(stage)) != entry['reader']['parents'][stage] for stage in ('vision', 'stt'))):
            raise ValueError('Current record/source/parents/context or reader identity drifted')
        return entry['reader_key']
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CompletedBindingInvalid('Historical receipt-only reader is quarantined; no provider fallback: ' + str(exc)) from exc


def read_bundle(config, entry, layout, *, context_reader, recording=None):
    """Read exact source/parent/output bytes; a failure can never request inference."""
    from .device_day_verification import verify_artifact_cached
    try:
        if layout.name != entry['archive']:
            raise ValueError('Historical reader crossed its exact archive')
        queue_before = _queue_snapshot(config, entry)
        parents, hashes = {}, {}
        for stage in ('retention', 'vision', 'stt', 'understanding'):
            value, checksum = _document(layout.receipts/entry['recording_id']/(stage + '.json'))
            hashes[stage] = checksum
            if stage == 'understanding':
                receipt = value
            else:
                parents[stage] = value
        audit = _producer(entry, receipt, parents)
        current = recording if recording is not None else parents['retention']['recording']
        current_key(config, entry, current, parents | {'context': context_reader(current)})
        if any(hashes[row['stage']] != row['sha256'] for row in audit['receipt_proof'] if row['stage'] in hashes):
            raise ValueError('Original receipt bytes changed')
        if any(digest(value) != entry['reader']['parents'][stage] for stage, value in parents.items()):
            raise ValueError('Current historical parent differs')
        if digest(parents['retention']['recording']) != entry['reader']['producer_record_digest']:
            raise ValueError('Original producer recording differs')
        for value in [*parents.values(), receipt]:
            if not isinstance(value.get('artifacts'), list):
                raise ValueError('Historical artifact manifest is unavailable')
            for reference in value['artifacts']:
                if not verify_artifact_cached(layout.root, reference, backend_root=layout.backend_root):
                    raise ValueError('Historical archive artifact is missing or changed')
            for reference in value.get('audit_artifacts', []):
                if reference.get('storage_root') != 'local_cache_root' or not verify_artifact_cached(layout.backend_root, reference):
                    raise ValueError('Historical local audit artifact is missing or changed')
        capture_root = Path(config['collection_ingest']['source_root'])
        sources = parents['retention'].get('sources') or []
        if not sources or not any(row.get('kind') == 'video' for row in sources):
            raise ValueError('Historical exact source manifest is unavailable')
        for source in sources:
            original = Path(source['original_path'])
            relative = original.relative_to(capture_root).as_posix()
            reference = {key: source[key] for key in ('size_bytes', 'sha256')}
            if (not verify_artifact_cached(capture_root, reference | {'path': relative})
                    or not verify_artifact_cached(layout.root, source['retained'])):
                raise ValueError('Historical original/retained source bytes changed')
        if _queue_snapshot(config, entry) != queue_before:
            raise ValueError('Historical reader queue changed during artifact validation')
        return parents | {'understanding': receipt}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration, sqlite3.Error) as exc:
        raise CompletedBindingInvalid('Historical receipt-only completion is unavailable; no provider fallback: ' + str(exc)) from exc


def verify(config, entry, layout, *, context_reader, recording=None):
    """Compatibility port returning the unchanged original understanding receipt."""
    return read_bundle(config, entry, layout, context_reader=context_reader, recording=recording)['understanding']


def provenance(entry, receipt=None):
    """Public reading provenance; old receipt/window keys and references stay intact."""
    return {'reader_recipe': VERSION, 'reader_key': entry['reader_key'], 'read_only': True, 'new_model_invoked': False,
            'producer_recipe': PRODUCER, 'producer_source_sha': entry['producer']['source_sha'],
            'producer_native_key': entry['producer']['native_key'], 'producer_settings': deepcopy(entry['producer']['settings']),
            'producer_context_digest': entry['producer']['context_digest'], 'receipt_digest': entry['receipt_digest'],
            'original_model_windows': [{key: window.get(key) for key in ('start_ms', 'end_ms', 'input', 'model_receipt')}
                for meaning in (receipt or {}).get('understandings', []) for window in meaning.get('windows', [])],
            'association_overlay': {'scope': 'current_association_only_not_original_model_inputs',
                                    'context_digest': entry['reader']['context_digest'],
                                    'added_fields': deepcopy(entry['reader']['additions'])},
            'full_historical_environment': 'NOT_PROVEN'}


def local_completion(config, entry, recording):
    """Local metadata observation only, not current artifact/source readability."""
    if (recording.get('source_signature') != entry['source_signature']
            or digest(recording) != entry['reader']['record_digest']):
        return False
    root = Path(config['storage']['local_cache_root'])/'device-day-receipts'/entry['archive']/entry['recording_id']
    try:
        parents = {stage: _document(root/(stage+'.json'))[0] for stage in ('retention', 'vision', 'stt')}
        receipt, _ = _document(root/'understanding.json')
        _producer(entry, receipt, parents)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
        return False

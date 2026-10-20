"""Synthetic owned-file proofs exercise readers, never historical model quality."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout
from visioncortex.device_day import DeviceDayRunner, load_context
from visioncortex.device_day_bindings import (
    CompletedBindingInvalid, CompletedExecutionBindings, settings_snapshot, write_manifest,
)
from visioncortex.device_day_contract import artifact, atomic_json, digest, file_hash, read_json, safe_child
from visioncortex.device_day_historical_reader import (
    current_key, descriptor_key, make_entry, provenance, reader_identity, validate_entry, verify,
)

__all__ = ['device_config']


class NeverModels:
    def __getattr__(self, name):
        raise AssertionError('A receipt-only reader must not request any model: ' + name)


@pytest.fixture
def historical_case(device_config, tmp_path, request):
    old_config = deepcopy(device_config)
    old_config['mllm'].update(provider='nianfeng', model='gpt-fixture-original')
    capture(old_config)
    record, layout = item_and_layout(old_config)
    old = DeviceDayRunner(old_config, FakeModels())
    assert old.process(record)['status'] == 'completed'
    stages = {stage: read_json(layout.receipts/record['recording_id']/(stage+'.json'))
              for stage in ('retention', 'vision', 'stt', 'understanding')}
    selected = getattr(request, 'param', None) == 'selected'
    if selected:
        from test_readable_all_day import add_work_evidence
        stages['understanding']['artifacts'] = add_work_evidence(layout,
            stages['understanding']['understandings'][0], 'historical-native-structured', stages['vision']['segments'][0])
    # Emulate an explicitly audited different historical source recipe. These
    # native keys are deliberately not accepted by today's parent recipes.
    for stage, character in zip(stages, '6789', strict=True):
        stages[stage]['key'] = character*64
    stages['vision'].pop('vision_input_digest', None)  # Legacy producer field shape.
    stages['vision']['retention_digest'] = digest(stages['retention'])
    stages['understanding']['vision_key'] = stages['vision']['key']
    for stage, value in stages.items():
        atomic_json(layout.receipts/record['recording_id']/(stage+'.json'), value)
    parents = {key: stages[key] for key in ('retention', 'vision', 'stt')}
    receipt = stages['understanding']
    historical_context = load_context(layout, record)
    current_context = historical_context | {'events': [], 'field_event_status': 'no_archive_for_camera_day'}
    fingerprints = {key: digest(key) for key in (
        'connection_sha256', 'request_policy_sha256', 'credential_revision_sha256', 'verification_sha256')}
    audit = {'schema': 'visioncortex-readonly-historical-producer-audit/1', 'recording_id': record['recording_id'],
        'source': {'git_sha': '1'*40, 'native_owner_sha256': '2'*64, 'tracked_and_untracked_worktree_clean': True},
        'native_recipe_reproduction': [{'stage': 'understanding', 'native_expected_key': receipt['key'],
            'receipt_key': receipt['key'], 'equal_full_64_characters': True,
            'inputs_digest': digest({'vision': parents['vision'], 'stt': parents['stt'], 'context': historical_context})}],
        'completed_paid_evidence': {'status': 'PROVEN', 'receipt_all_artifacts_verified': True,
            'artifact_count': len(receipt['artifacts']), 'ledger_unknown_attempts': 0,
            'artifact_bytes': sum(row['size_bytes'] for row in receipt['artifacts']),
            'window_count': sum(len(row.get('windows', [])) for row in receipt['understandings']), 'ledger_attempts': 1,
            'complete_receipt_and_queue_payload_stable': True},
        'input_proof': {'context_digest': digest(historical_context), 'saved_context_digest': digest(historical_context)},
        'queue_proof': {'revision': 'd'*64},
        'source_files_proof': {'all_original_and_retained_bytes_match': True},
        'configuration_proof': {'matching_historical_binding_fingerprints': fingerprints,
                                'provider': old_config['mllm']['provider'], 'model': old_config['mllm']['model']},
        'receipt_proof': [{'stage': stage, 'status': 'completed', 'receipt_key': value['key'],
            'receipt_digest': digest(value), 'sha256': file_hash(layout.receipts/record['recording_id']/(stage+'.json'))}
            for stage, value in stages.items()]}
    native = {'source_sha': '1'*40, 'stage': 'understanding', 'native_key': receipt['key'],
              'settings_snapshot': settings_snapshot(old_config, 'understanding'), **fingerprints}
    audit_path, binding_path = tmp_path/'ProducerProof.json', tmp_path/'BindingProof.json'
    atomic_json(audit_path, audit)
    atomic_json(binding_path, {'schema': 'visioncortex-readonly-historical-binding-assessment/1',
                              'recording_id': record['recording_id'], 'historical_native_execution': native,
                              'producer_audit': {'path': str(audit_path), 'sha256': file_hash(audit_path)}})
    config = deepcopy(old_config)
    config['mllm']['model'] = 'glm-fixture-current'
    fr_root = tmp_path/'FieldRecognition'
    fr_root.mkdir()
    config['device_day'].update(readable_content_enabled=True, derived_video_retention=selected, field_recognition={
        'enabled': False, 'archive_context_enabled': True, 'archive_root': str(fr_root)})
    entry = make_entry(config, record, parents, receipt,
        producer_audit={'path': str(audit_path), **{k: v for k, v in artifact(tmp_path, audit_path).items() if k != 'path'}},
        binding_audit={'path': str(binding_path), **{k: v for k, v in artifact(tmp_path, binding_path).items() if k != 'path'}},
        historical_context=historical_context, current_context=current_context, queue_revision='d'*64)
    config['device_day']['completed_execution_bindings'] = write_manifest(tmp_path/'Readers.json', [entry])
    runner = DeviceDayRunner(config, NeverModels())
    runner._context = lambda *_args: deepcopy(current_context)
    runner.queues['understanding'].enqueue(record, entry['queue_revision'])
    return SimpleNamespace(config=config, old=old, record=record, layout=layout, receipt=receipt, parents=parents,
        historical_context=historical_context, current_context=current_context, entry=entry, runner=runner,
        audit_path=audit_path, binding_path=binding_path, selected=selected)


def enable_runtime_profiles(case):
    """Eight explicitly reviewed full configs emulate the split production roles."""
    profiles = {}
    for offset, name in enumerate(('Discovery', 'Downstream', 'Integrity', 'Primary', 'Progress',
                                   'Publication', 'Retry', 'Web'), 1):
        config = deepcopy(case.config)
        if name != 'Publication':
            config['device_day'].update(admission_min_available_gib=offset / 1000,
                                         defer_primary_publication=True)
        if name in {'Integrity', 'Web'}:
            config['device_day']['input_integrity'] = {'enabled': True, 'max_records': 5}
        if name == 'Retry':
            config['device_day']['retry_repair'] = {'enabled': True, 'max_records': 3}
        profiles[name] = config
    entry = make_entry(profiles['Publication'], case.record, case.parents, case.receipt,
        producer_audit=case.entry['producer']['audit'], binding_audit=case.entry['producer']['binding_audit'],
        historical_context=case.historical_context, current_context=case.current_context,
        queue_revision=case.entry['queue_revision'], profiles=profiles)
    reference = write_manifest(case.audit_path.parent/'RuntimeProfileReaders.json', [entry])
    for config in profiles.values():
        config['device_day']['completed_execution_bindings'] = reference
    case.config, case.entry = profiles['Publication'], entry
    case.runner = DeviceDayRunner(case.config, NeverModels())
    case.runner._context = lambda *_args: deepcopy(case.current_context)
    return profiles


def test_explicit_runtime_profiles_preserve_native_proof_and_exact_full_configuration(historical_case):
    case = historical_case
    original = {stage: (case.layout.receipts/case.record['recording_id']/(stage+'.json')).read_bytes()
                for stage in (*case.parents, 'understanding')}
    profiles = enable_runtime_profiles(case)
    expected = [{'name': name, 'identity': reader_identity(config)} for name, config in profiles.items()]
    assert case.entry['reader']['runtime_profiles'] == expected
    assert len({row['identity'] for row in expected}) == 8
    assert case.entry['reader']['identity'] == reader_identity(profiles['Publication'])
    for config in profiles.values():
        inputs = case.parents | {'context': case.current_context}
        assert current_key(config, case.entry, case.record, inputs) == case.entry['reader_key']
        assert verify(config, case.entry, case.layout, context_reader=case.runner._context) == case.receipt
    for stage, value in original.items():
        assert (case.layout.receipts/case.record['recording_id']/(stage+'.json')).read_bytes() == value


@pytest.mark.parametrize('change', ['unknown_profile', 'same_role_changed_setting', 'source_identity'])
def test_unreviewed_runtime_identity_never_falls_back_to_any_provider(historical_case, monkeypatch, change):
    case = historical_case
    profiles = enable_runtime_profiles(case)
    config = deepcopy(profiles['Web'])
    if change == 'unknown_profile':
        config['device_day']['unreviewed_runtime_setting'] = 'not approved'
    elif change == 'same_role_changed_setting':
        config['device_day']['input_integrity']['max_records'] += 1
    else:
        monkeypatch.setattr('visioncortex.device_day_historical_reader.reader_identity', lambda _: 'f'*64)
    with pytest.raises(CompletedBindingInvalid, match='reader identity drifted'):
        current_key(config, case.entry, case.record, case.parents | {'context': case.current_context})


@pytest.mark.parametrize('damage', ['empty', 'null', 'duplicate_name', 'unknown_field', 'invalid_hash',
                                   'missing_baseline', 'unordered', 'unbound_profile'])
def test_runtime_profile_descriptor_is_explicit_bounded_and_cryptographically_bound(historical_case, damage):
    case = historical_case
    enable_runtime_profiles(case)
    entry = deepcopy(case.entry)
    profiles = entry['reader']['runtime_profiles']
    if damage == 'empty':
        entry['reader']['runtime_profiles'] = []
    elif damage == 'null':
        entry['reader']['runtime_profiles'] = None
    elif damage == 'duplicate_name':
        profiles[1]['name'] = profiles[0]['name']
    elif damage == 'unknown_field':
        profiles[0]['settings_to_ignore'] = ['input_integrity']
    elif damage == 'invalid_hash':
        profiles[0]['identity'] = '*'
    elif damage == 'missing_baseline':
        profiles[:] = [row for row in profiles if row['identity'] != entry['reader']['identity']]
    elif damage == 'unordered':
        profiles.reverse()
    else:
        profiles[0]['identity'] = 'f'*64
    if damage != 'unbound_profile':
        entry['reader_key'] = descriptor_key(entry)
    with pytest.raises(ValueError, match='[Pp]rofile|descriptor'):
        validate_entry(entry)


def test_profile_creation_requires_explicit_full_configs_and_baseline_membership(historical_case):
    case = historical_case
    options = dict(producer_audit=case.entry['producer']['audit'], binding_audit=case.entry['producer']['binding_audit'],
        historical_context=case.historical_context, current_context=case.current_context,
        queue_revision=case.entry['queue_revision'])
    assert 'runtime_profiles' not in case.entry['reader']  # Existing single-identity reader remains valid.
    assert verify(case.config, case.entry, case.layout, context_reader=case.runner._context) == case.receipt
    for profiles in ({}, [], {'Publication': None}):
        with pytest.raises(ValueError, match='explicit named'):
            make_entry(case.config, case.record, case.parents, case.receipt, **options, profiles=profiles)
    other = deepcopy(case.config)
    other['device_day']['defer_primary_publication'] = 'unreviewed other profile'
    with pytest.raises(ValueError, match='include the baseline'):
        make_entry(case.config, case.record, case.parents, case.receipt, **options, profiles={'Other': other})


def test_native_producer_key_is_distinct_from_current_model_and_read_descriptor(historical_case):
    case = historical_case
    inputs = case.parents | {'context': case.current_context}
    key = case.runner._key('understanding', case.record, inputs)
    assert key == case.entry['reader_key']
    assert key != case.receipt['key']
    assert case.runner._key('understanding', case.record, inputs, _binding=False) not in {key, case.receipt['key']}
    before = (case.layout.receipts/case.record['recording_id']/'understanding.json').read_bytes()
    loaded = case.runner._load(case.layout.receipts/case.record['recording_id']/'understanding.json', key, case.layout)
    assert loaded == case.receipt and loaded['key'] == case.receipt['key']
    assert (case.layout.receipts/case.record['recording_id']/'understanding.json').read_bytes() == before
    info = provenance(case.entry)
    assert info['producer_settings']['mllm']['model'] == 'gpt-fixture-original'
    assert info['producer_context_digest'] == digest(case.historical_context)
    assert info['association_overlay']['scope'] == 'current_association_only_not_original_model_inputs'
    assert not info['new_model_invoked']


def test_native_or_current_execution_key_cannot_bypass_the_selected_read_only_guard(historical_case):
    case = historical_case
    current = case.runner._key('understanding', case.record, case.parents | {'context': case.current_context}, _binding=False)
    for key in (case.receipt['key'], current):
        assert not case.runner._accepts_receipt(case.receipt, key)
        with pytest.raises(CompletedBindingInvalid, match='explicit read descriptor'):
            case.runner._load(case.layout.receipts/case.record['recording_id']/'understanding.json', key, case.layout)


def test_existing_publication_preserves_producer_provenance_without_model_or_queue_changes(historical_case):
    case = historical_case
    from visioncortex.device_day import visual_input
    assert 'vision_input_digest' not in case.parents['vision']
    for name, inputs in (('retention', case.record), ('vision', visual_input(case.parents['retention'])),
                         ('stt', case.parents['retention'])):
        assert not case.runner._accepts_receipt(case.parents[name], case.runner._key(name, case.record, inputs))
    queue = case.runner.queues['understanding']
    with queue.connect() as db:
        before_queue = dict(db.execute('SELECT * FROM recordings').fetchone())
    receipt_path = case.layout.receipts/case.record['recording_id']/'understanding.json'
    original_bytes = receipt_path.read_bytes()
    all_original_bytes = {path: path.read_bytes() for path in receipt_path.parent.glob('*.json')}
    case.runner.refresh_index(case.layout, recording_id=case.record['recording_id'])
    public = read_json(case.layout.index)['understandings'][0]['historical_execution_read']
    assert public['producer_native_key'] == case.receipt['key']
    assert public['reader_key'] == case.entry['reader_key'] and public['read_only']
    assert public['producer_settings']['mllm']['model'] == 'gpt-fixture-original'
    assert receipt_path.read_bytes() == original_bytes
    assert {path: path.read_bytes() for path in receipt_path.parent.glob('*.json')} == all_original_bytes
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM recordings').fetchone()) == before_queue


@pytest.mark.parametrize('damage', ['missing_receipt', 'changed_receipt', 'missing_artifact', 'changed_artifact',
                                     'missing_parent', 'changed_parent', 'missing_source', 'changed_source', 'proof'])
def test_selected_reader_damage_fails_closed_without_either_provider(historical_case, damage):
    case = historical_case
    receipt_path = case.layout.receipts/case.record['recording_id']/'understanding.json'
    before_receipt = receipt_path.read_bytes()
    parent_path = receipt_path.with_name('stt.json')
    media = safe_child(case.layout.root, case.receipt['artifacts'][0]['path'])
    source = safe_child(case.layout.root, case.parents['retention']['sources'][0]['retained']['path'])
    original = case.parents['retention']['sources'][0]['original_path']
    from pathlib import Path
    targets = {'missing_receipt': receipt_path, 'changed_receipt': receipt_path,
               'missing_artifact': media, 'changed_artifact': media,
               'missing_parent': parent_path, 'changed_parent': parent_path,
               'missing_source': Path(original), 'changed_source': source, 'proof': case.audit_path}
    path = targets[damage]
    if damage.startswith('missing'):
        path.unlink()
    else:
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])  # Same-size damage is still rejected.
    with pytest.raises(CompletedBindingInvalid, match='no provider fallback'):
        case.runner.process(case.record, stage='understanding')
    if damage not in {'missing_receipt', 'changed_receipt'}:
        assert receipt_path.read_bytes() == before_receipt


@pytest.mark.parametrize('change', ['record', 'source', 'parent', 'retention_parent', 'context', 'model', 'reader_source'])
def test_exact_current_inputs_cannot_fall_through_to_active_recipe(historical_case, monkeypatch, change):
    case = historical_case
    config, record, parents = deepcopy(case.config), deepcopy(case.record), deepcopy(case.parents)
    context = deepcopy(case.current_context)
    if change == 'record':
        record['recording_end_us'] += 1
    elif change == 'source':
        record['source_signature'] = 'different-source'
    elif change == 'parent':
        parents['vision']['key'] = 'e'*64
    elif change == 'retention_parent':
        parents['retention']['key'] = 'e'*64
    elif change == 'context':
        context['events'] = [{'type': 'ocr_observation', 'new_fact': True}]
    elif change == 'model':
        config['mllm']['model'] = 'different-current-model'
    else:
        monkeypatch.setattr('visioncortex.device_day_historical_reader.reader_identity', lambda _config: 'f'*64)
    with pytest.raises(CompletedBindingInvalid, match='no provider fallback'):
        current_key(config, case.entry, record, parents | {'context': context})


def test_v1_manifest_cannot_silently_enable_a_historical_reader(historical_case, tmp_path):
    case = historical_case
    path = tmp_path/'NotV2.json'
    atomic_json(path, {'schema_version': 'visioncortex-completed-execution-bindings/1', 'entries': [case.entry]})
    with pytest.raises(ValueError, match='explicit v2'):
        CompletedExecutionBindings({'completed_execution_bindings': {'path': str(path), 'sha256': file_hash(path)}})


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt', 'understanding', 'report'])
def test_inventory_never_reenqueues_or_revises_reader_only_completion(historical_case, monkeypatch, stage):
    case = historical_case
    queue = case.runner.queues[stage]
    queue.enqueue(case.record, case.entry['queue_revision'])
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='failed',result=?",
                   ('{"error_type":"ValueError","message":"Prerequisite stage retention is not ready"}',))
        before = dict(db.execute('SELECT * FROM recordings').fetchone())
    from visioncortex.input_availability import Availability
    monkeypatch.setattr(Availability, 'states_for_stage', lambda *_: {
        case.record['recording_id']: {'state': 'missing', 'signature': case.record['source_signature']}})
    assert not case.runner._build_stage_inventory({'recordings': [case.record]}, stage, None)
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM recordings').fetchone()) == before


@pytest.mark.parametrize('stage', ['all', 'retention', 'vision', 'stt', 'understanding', 'report'])
def test_reserved_producer_graph_never_enters_any_execution_stage(historical_case, monkeypatch, stage):
    case = historical_case
    receipts = {path: path.read_bytes() for path in case.layout.receipts.rglob('*.json')}
    before = {}
    for name, queue in case.runner.queues.items():
        with queue.connect() as db:
            before[name] = [dict(row) for row in db.execute('SELECT * FROM recordings')]
    def forbidden(*_args, **_kwargs):
        pytest.fail('Historical publication must not enter producer/prerequisite execution')
    monkeypatch.setattr(case.runner, '_process', forbidden)
    monkeypatch.setattr('visioncortex.device_day_inplace.active', forbidden)
    monkeypatch.setattr(case.runner, '_backend', forbidden)
    for process in (case.runner.process, case.runner._guarded_process):
        result = process(case.record, stage=stage, retry=True)
        assert result['status'] == 'historical_read_only_publication_required'
        assert not result['publication_verified'] and not result['new_model_invoked']
    assert {path: path.read_bytes() for path in case.layout.receipts.rglob('*.json')} == receipts
    for name, queue in case.runner.queues.items():
        with queue.connect() as db:
            assert [dict(row) for row in db.execute('SELECT * FROM recordings')] == before[name]


@pytest.mark.parametrize('stage', ['retention', 'vision', 'stt', 'understanding', 'report'])
def test_background_worker_cannot_admit_or_reconcile_reserved_original_producer(historical_case, monkeypatch, stage):
    from visioncortex.device_day_retention_worker import RetentionWorker
    case = historical_case
    queue = case.runner.queues[stage]
    queue.enqueue(case.record, case.entry['queue_revision'])
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='running',lease_until=0,lease_owner='old-producer'")
        before = dict(db.execute('SELECT * FROM recordings').fetchone())
    def forbidden(*_args, **_kwargs):
        pytest.fail('Reserved producer must not enter background admission or recovery')
    monkeypatch.setattr('visioncortex.device_day_inplace.active', forbidden)
    worker = RetentionWorker(stage=stage)
    monkeypatch.setattr(worker, '_expired_completion', forbidden)
    assert not worker.admit(case.runner, case.record)
    assert worker.recover_expired(case.runner) == {'completed': 0, 'result_unknown': 0, 'busy': 0}
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM recordings').fetchone()) == before


def test_current_association_context_is_checked_again_when_reading_not_only_selecting(historical_case):
    case = historical_case
    with pytest.raises(CompletedBindingInvalid, match='no provider fallback'):
        verify(case.config, case.entry, case.layout,
               context_reader=lambda _record: case.current_context | {'events': [{'new': 'fact'}]})


@pytest.mark.parametrize('change', ['revision', 'payload', 'submitted_unknown', 'blocked_unknown', 'active_lease', 'concurrent'])
def test_reader_never_borrows_another_queue_generation_or_hides_unknown_submission(historical_case, monkeypatch, change):
    case = historical_case
    queue = case.runner.queues['understanding']
    if change == 'concurrent':
        from visioncortex import device_day_historical_reader as reader
        original = reader._queue_snapshot
        reads = []
        def changed(config, entry):
            result = original(config, entry)
            reads.append(result)
            return result if len(reads) == 1 else result | {'attempts': result['attempts'] + 1}
        monkeypatch.setattr(reader, '_queue_snapshot', changed)
    else:
        with queue.connect() as db:
            if change == 'revision':
                db.execute('UPDATE recordings SET revision=?', ('e'*64,))
            elif change == 'payload':
                import json
                db.execute('UPDATE recordings SET payload=?', (json.dumps(case.record | {'recording_end_us': 1}),))
            elif change == 'active_lease':
                import time
                db.execute("UPDATE recordings SET status='running',lease_until=?", (time.time()+60,))
            elif change == 'blocked_unknown':
                db.execute("UPDATE recordings SET blocked_reason='submitted_unknown',result='{}'")
            else:
                db.execute('UPDATE recordings SET result=?', ('{"status":"submitted_unknown"}',))
    with pytest.raises(CompletedBindingInvalid, match='no provider fallback'):
        case.runner.process(case.record, stage='understanding')


def test_progress_separates_known_local_completion_without_changing_queue_or_claiming_bytes(historical_case):
    from visioncortex.device_day_progress import processing_health
    case = historical_case
    queue = case.runner.queues['understanding']
    queue.enqueue(case.record, case.entry['queue_revision'])
    with queue.connect() as db:
        before = dict(db.execute('SELECT * FROM recordings').fetchone())
    result = processing_health(case.config)
    stage = result['stages']['understanding']
    assert stage['categories'] == {'local_completed_unpublished': 1}
    assert stage['workload']['publication_pending'] == 1
    assert stage['workload']['result_unknown'] == stage['workload']['processing_backlog'] == 0
    assert result['publication']['artifact_validation_required']
    assert 'not_current_artifact_or_source_readability' in result['publication']['historical_completion_scope']
    with queue.connect() as db:
        assert dict(db.execute('SELECT * FROM recordings').fetchone()) == before
        db.execute('UPDATE recordings SET result=?', ('{"status":"submitted_unknown"}',))
    unknown = processing_health(case.config)['stages']['understanding']
    assert unknown['categories'] == {'awaiting_request_reconciliation': 1}
    assert unknown['workload']['result_unknown'] == 1 and unknown['workload']['publication_pending'] == 0
    with queue.connect() as db:
        db.execute("UPDATE recordings SET blocked_reason='submitted_unknown',result='{}'")
    unknown = processing_health(case.config)['stages']['understanding']
    assert unknown['categories'] == {'awaiting_request_reconciliation': 1}
    assert unknown['workload']['publication_pending'] == 0
    import time
    with queue.connect() as db:
        db.execute("UPDATE recordings SET status='running',lease_until=?,blocked_reason=NULL,result='{}'", (time.time()+60,))
    active = processing_health(case.config)['stages']['understanding']
    assert active['categories'] == {'running': 1}
    assert active['workload']['publication_pending'] == 0


def test_reader_owner_is_in_formal_understanding_dependency_inventory():
    from visioncortex.stage_dependencies import DEVICE_SOURCES, impact, stage_source_manifest
    assert {'device_day_historical_reader', 'device_day_historical_publication', 'device_day_bindings'} <= set(DEVICE_SOURCES['understanding'])
    assert impact(['device_day_historical_reader.py'])['changed_stages'] == ['understanding']
    assert 'device_day_historical_reader.py' in stage_source_manifest('understanding')
    assert 'device_day_historical_publication.py' in stage_source_manifest('understanding')
    assert impact(['device_day_historical_publication.py'])['changed_stages'] == ['report', 'understanding']

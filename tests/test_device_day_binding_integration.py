"""Completed providers remain historical; new work uses the active provider."""
from copy import deepcopy

import pytest

from test_device_day import FakeModels, capture, device_config, item_and_layout  # noqa: F401
from visioncortex.device_day import DeviceDayRunner, load_context
from visioncortex.device_day_bindings import CompletedBindingInvalid, make_entry, write_manifest
from visioncortex.device_day_contract import STAGES, atomic_json, read_json, safe_child
from visioncortex.nas_recordings import scan_recordings


class ProviderModels(FakeModels):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider
        self.stt_calls = 0

    def understand(self, *args):
        return super().understand(*args) | {'executed_provider': self.provider}

    def transcribe(self, *args):
        self.stt_calls += 1
        return super().transcribe(*args) | {'executed_provider': 'aliyun_qwen'}


def migration(config, tmp_path):
    config['mllm'].update(provider='aliyun_qwen', model='qwen-fixture',
                          api_protocol='chat_completions', base_url='https://example.test/aliyun/v1')
    capture(config)
    record, layout = item_and_layout(config)
    old = DeviceDayRunner(config, ProviderModels('aliyun_qwen'))
    assert old.process(record)['status'] == 'completed'
    receipts = {stage: read_json(old._receipt(layout, record, stage)) for stage in STAGES}
    for stage in STAGES:
        old.queues[stage].enqueue(record, '1' * 64)
        old.queues[stage].claim('fixture')
        old.queues[stage].finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    context = load_context(layout, record)
    inputs = {'stt': receipts['retention'], 'report': receipts['understanding'],
              'understanding': {'vision': receipts['vision'], 'stt': receipts['stt'], 'context': context}}
    queued = {'stt': [{'retention': receipts['retention']}, None],
              'report': [{'understanding': receipts['understanding']}, None],
              'understanding': [{'vision': receipts['vision'], 'stt': receipts['stt']}, context]}
    entries = [make_entry(stage, record, receipts[stage], old._key(stage, record, inputs[stage]), config,
                          queue_key=old._key(stage, record, queued[stage]), queue_revision='1' * 64)
               for stage in ('stt', 'understanding', 'report')]
    updated = deepcopy(config)
    updated['mllm'].update(provider='nianfeng', model='gpt-6-sol', base_url='https://example.test/nianfeng/v1')
    updated['speech_recognition']['connection'] = {
        'provider': 'aliyun_qwen', 'model': 'qwen-fixture', 'api_protocol': 'chat_completions',
        'base_url': 'https://example.test/aliyun/v1', 'credential_ref': 'fixture-reference-only'}
    updated['device_day']['completed_execution_bindings'] = write_manifest(tmp_path/'bindings.json', entries)
    backend = ProviderModels('nianfeng')
    current = DeviceDayRunner(updated, backend)
    return current, backend, record, layout, receipts, inputs


def test_provider_switch_preserves_completions_queues_and_reports_and_only_runs_new_work(device_config, tmp_path):  # noqa: F811
    current, backend, record, layout, receipts, inputs = migration(device_config, tmp_path)
    for stage in ('stt', 'understanding', 'report'):
        key = current._key(stage, record, inputs[stage])
        assert key == receipts[stage]['key']
        assert current._load(current._receipt(layout, record, stage), key, layout) == receipts[stage]
        current._build_stage_inventory({'recordings': [record]}, stage, None)
        with current.queues[stage].connect() as db:
            row = db.execute('SELECT status,revision FROM recordings').fetchone()
            assert (row['status'], row['revision']) == ('completed', '1' * 64)
    assert current.process(record)['status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == backend.stt_calls == 0
    assert {stage: read_json(current._receipt(layout, record, stage)) for stage in STAGES} == receipts
    assert read_json(layout.index)['understandings']
    assert (layout.reports/'LaboratoryDailyReport.html').is_file()
    capture(current.config, start=record['recording_start_us'] + 900_000_000, label='101500')
    following = next(item for item in scan_recordings(current.config)['recordings']
                     if item['recording_id'] != record['recording_id'])
    assert current.process(following)['status'] == 'completed'
    assert backend.vision_calls == backend.semantic_calls == backend.stt_calls == 1
    result = read_json(current._receipt(current.layout(following), following, 'understanding'))
    assert result['executed_provider'] == 'nianfeng'
    assert receipts['understanding']['executed_provider'] == 'aliyun_qwen'


def test_changed_inputs_or_source_or_code_never_borrow_historical_provider_key(device_config, tmp_path):  # noqa: F811
    current, _, record, _, receipts, inputs = migration(device_config, tmp_path)
    changed = inputs['understanding'] | {'context': {'comments': [{'text': 'new instruction'}], 'protocol': None}}
    key = current._key('understanding', record, changed)
    assert key == current._key('understanding', record, changed, _binding=False)
    assert key != receipts['understanding']['key']
    revised = record | {'source_signature': 'new-source-identity'}
    assert current._key('understanding', revised, inputs['understanding']) == current._key(
        'understanding', revised, inputs['understanding'], _binding=False)
    current._execution_identity = 'a' * 64
    assert current._key('understanding', record, inputs['understanding']) == current._key(
        'understanding', record, inputs['understanding'], _binding=False)


@pytest.mark.parametrize('damage', ['missing_receipt', 'tampered_receipt', 'damaged_artifact'])
def test_damaged_pinned_completion_cannot_execute_new_provider_under_old_key(device_config, tmp_path, damage):  # noqa: F811
    current, backend, record, layout, receipts, _ = migration(device_config, tmp_path)
    path = current._receipt(layout, record, 'understanding')
    if damage == 'missing_receipt':
        path.unlink()
    elif damage == 'tampered_receipt':
        atomic_json(path, receipts['understanding'] | {'executed_provider': 'forged'})
    else:
        safe_child(layout.root, receipts['understanding']['artifacts'][0]['path']).write_bytes(b'damaged')
    with pytest.raises(CompletedBindingInvalid):
        current.process(record, stage='understanding')
    assert backend.vision_calls == backend.semantic_calls == backend.stt_calls == 0


def test_inplace_stt_queue_and_receipt_remain_completed_after_explicit_asr_binding(device_config, tmp_path):  # noqa: F811
    from visioncortex.device_day_contract import digest
    device_config['device_day']['inplace_preprocessing'] = True
    capture(device_config)
    record, layout = item_and_layout(device_config)
    old = DeviceDayRunner(device_config, ProviderModels('legacy'))
    assert old.process(record)['status'] == 'completed'
    retained = read_json(old._receipt(layout, record, 'retention'))
    completed = read_json(old._receipt(layout, record, 'stt'))
    queue_key = old._key('stt', record, {})
    revision = digest(['inplace-queue/1', 'stt', record['recording_id'],
                       record.get('audio', {}).get('source_signature'), queue_key])
    old.queues['stt'].enqueue(record, revision)
    old.queues['stt'].claim('fixture')
    old.queues['stt'].finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    entry = make_entry('stt', record, completed, old._key('stt', record, retained), device_config,
                       queue_key=queue_key, queue_revision=revision)
    updated = deepcopy(device_config)
    updated['speech_recognition']['connection'] = {'provider': 'aliyun_qwen', 'credential_ref': 'reference-only'}
    updated['device_day']['completed_execution_bindings'] = write_manifest(tmp_path/'inplace-bindings.json', [entry])
    backend = ProviderModels('new')
    current = DeviceDayRunner(updated, backend)
    current._build_stage_inventory({'recordings': [record]}, 'stt', None)
    with current.queues['stt'].connect() as db:
        row = db.execute('SELECT status,revision FROM recordings').fetchone()
        assert (row['status'], row['revision']) == ('completed', revision)
    assert current.process(record, stage='stt')['status'] == 'completed'
    assert backend.stt_calls == 0
    assert read_json(current._receipt(layout, record, 'stt')) == completed


@pytest.mark.parametrize(('stage', 'inplace'), [('understanding', False), ('stt', False), ('stt', True)])
def test_hold_preserves_unverifiable_completion_without_accepting_it_and_releases_new_inputs(device_config, tmp_path, stage, inplace, monkeypatch):  # noqa: F811
    from visioncortex.device_day_bindings import make_hold
    from visioncortex.device_day_contract import digest
    device_config['device_day']['inplace_preprocessing'] = inplace
    capture(device_config)
    record, layout = item_and_layout(device_config)
    old = DeviceDayRunner(device_config, ProviderModels('legacy'))
    assert old.process(record)['status'] == 'completed'
    receipts = {name: read_json(old._receipt(layout, record, name)) for name in STAGES}
    original = receipts[stage] | {'key': 'b' * 64}
    path = old._receipt(layout, record, stage)
    atomic_json(path, original)
    old.queues[stage].enqueue(record, '1' * 64)
    old.queues[stage].claim('fixture')
    old.queues[stage].finish('fixture', record['recording_id'], {'status': 'completed'}, 1)
    updated = deepcopy(device_config)
    updated['mllm'].update(provider='nianfeng', model='gpt-6-sol')
    updated['speech_recognition']['connection'] = {'provider': 'aliyun', 'credential_ref': 'reference-only'}
    preview = DeviceDayRunner(updated, ProviderModels('new'))
    context = load_context(layout, record) if stage == 'understanding' else None
    queued = ({'retention': receipts['retention']} if stage == 'stt' else
              {'vision': receipts['vision'], 'stt': receipts['stt']})
    inputs = queued['retention'] if stage == 'stt' else queued | {'context': context}
    observed_key = preview._key(stage, record, inputs, _binding=False)
    observed_queue_key = (digest(['inplace-queue/1', stage, record['recording_id'],
                                  record.get('audio', {}).get('source_signature'),
                                  preview._key(stage, record, {}, _binding=False)]) if inplace else
                          preview._key(stage, record, [queued, context], _binding=False))
    hold = make_hold(stage, record, '1' * 64, status='completed', observed_key=observed_key,
                     observed_queue_key=observed_queue_key)
    updated['device_day']['completed_execution_bindings'] = write_manifest(tmp_path/'holds.json', [], holds=[hold])
    backend = ProviderModels('new')
    current = DeviceDayRunner(updated, backend)
    assert current._completion_hold(record, stage) == hold
    assert current._completion_hold(record, stage) == hold  # Metadata cache path.
    assert not current._accepts_receipt(original, observed_key)
    before = path.read_bytes()
    result = current.process(record, stage=stage)
    assert result['status'] == 'historical_completion_held'
    assert result['historical_receipts_verified'] is False
    assert current._build_stage_inventory({'recordings': [record]}, stage, None) == set()
    with current.queues[stage].connect() as db:
        row = db.execute('SELECT status,revision FROM recordings').fetchone()
        assert (row['status'], row['revision']) == ('completed', '1' * 64)
    assert path.read_bytes() == before
    assert backend.vision_calls == backend.semantic_calls == backend.stt_calls == 0
    assert current._completion_hold(record | {'source_signature': 'new-source'}, stage) is None
    # One unreadable held record cannot abort admission for the whole stage.
    input_path = current._receipt(layout, record, 'vision' if stage == 'understanding' else 'retention')
    input_bytes = input_path.read_bytes()
    input_path.write_text('{broken-json')
    assert current._completion_hold(record, stage)['reason'] == 'historical_completion_inputs_unreadable'
    assert current._build_stage_inventory({'recordings': [record]}, stage, None) == set()
    input_path.write_bytes(input_bytes)
    import visioncortex.device_day as module
    reader = module.read_json
    def denied(path):
        if path == input_path:
            raise PermissionError('owned test input unavailable')
        return reader(path)
    current._completion_hold_cache.clear()
    with monkeypatch.context() as patch:
        patch.setattr(module, 'read_json', denied)
        assert current._completion_hold(record, stage)['reason'] == 'historical_completion_inputs_unreadable'
        assert current._build_stage_inventory({'recordings': [record]}, stage, None) == set()
    if stage == 'understanding':
        atomic_json(layout.comments/'Protocol.json', {'text': 'new experiment instructions'})
    else:
        retained_path = current._receipt(layout, record, 'retention')
        atomic_json(retained_path, receipts['retention'] | {
            'audio': receipts['retention'].get('audio', {}) | {'duration_us': 123}})
    assert current._completion_hold(record, stage) is None

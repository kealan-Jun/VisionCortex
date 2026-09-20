from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from threading import Event
from zoneinfo import ZoneInfo
import json

import cv2
import httpx
import numpy as np
import pytest

from visioncortex import scene_requests, scene_transport
from visioncortex.device_day_contract import atomic_json, digest, file_hash, read_json
from visioncortex.device_day_models import SCENE_PROMPT, validate_understanding
from visioncortex.mllm import ArkAnalyzer, SCENE_TIME_REFERENCE_RULES
from visioncortex.multimodal_usage import UsageLedger, validate
from visioncortex.provider_control import ProviderUnavailable


@pytest.fixture
def scene(default_config, tmp_path):
    config = deepcopy(default_config)
    config['storage'].update(local_runtime_root=str(tmp_path/'runtime'), local_cache_root=str(tmp_path/'cache'))
    config['mllm'].update(provider='aliyun', model='synthetic-vision', base_url='https://example.test/v1',
                          api_protocol='chat_completions', quality_mode='quality', max_retries=2,
                          request_image_max_edge=768, request_image_jpeg_quality=82)
    frames, paths = [], []
    for i, stamp in enumerate([1250.123, 1750.456]):
        path = tmp_path/f'image-{i}.png'
        assert cv2.imwrite(str(path), np.full((24, 32, 3), 50 + i*100, np.uint8))
        frame_id = 'frame-' + file_hash(path)[:24]
        frames.append({'frame_id': frame_id, 'local_ms': stamp, 'capture_us': 1789005600000000 + round(stamp*1000),
                       'path': str(path), 'source_path': 'MetaVideo/Source.mp4', 'sha256': file_hash(path),
                       'size_bytes': path.stat().st_size, 'source_frame_index': i, 'frame_kind': 'scene_sample',
                       'source_time_basis': 'test-unverified'})
        paths.append((frame_id, path))
    metadata = {'schema_version': 'visioncortex-device-day/1', 'segment_id': 'test-segment',
                'source_ref': {'sha256': 'a'*64, 'path': 'MetaVideo/Source.mp4', 'size_bytes': 1000, 'clock': 'unverified'},
                'activity': 'inactive', 'start_ms': 1000, 'end_ms': 2000, 'frames': frames, 'comments': [],
                'protocol': None, 'physical_action_confirmed': False, 'content_source': 'video_only',
                'coverage': {'mode': 'interval_sampling', 'model_attention_to_every_frame_verified': False}}
    return config, metadata, paths, tmp_path/'window'


def response(metadata):
    return {'status': 'completed', 'summary': 'synthetic evidence only', 'activity_observed': 'inactive',
            'scene_timing_contract': {'version': 'visioncortex-scene-time-references/1',
                                     'instructions': SCENE_TIME_REFERENCE_RULES, 'validation_relaxed': False},
            'steps': [], 'frame_observations': [{'frame_id': f['frame_id'], 'text': 'visible test scene'} for f in metadata['frames']],
            'uncertainties': ['Not real-video quality evidence'], 'attempts': 1,
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30, 'cached_input_tokens': 0}}


class Analyzer:
    def __init__(self, results=()):
        self.results = list(results)
        self.calls = 0

    def _call(self, prompt, metadata, images, **kwargs):
        self.calls += 1
        return deepcopy(self.results.pop(0)) if self.results else response(metadata)


def run(scene, analyzer):
    config, metadata, paths, folder = scene
    def validator(raw):
        return validate_understanding(raw, metadata['frames'], metadata['comments'], 1000, 2000, 1789005600000000)
    return scene_requests.window(config, analyzer, SCENE_PROMPT, metadata, paths,
                                 folder/'Input.json', folder/'Result.json', validator)


def today():
    return datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()


def test_compaction_preserves_semantics_and_images_and_reverses_only_references(scene):
    _, metadata, paths, _ = scene
    before = deepcopy(metadata)
    wire, images, receipt = scene_transport.compact(metadata, paths)
    assert metadata == before
    assert [p.read_bytes() for _, p in images] == [p.read_bytes() for _, p in paths]
    assert [label for label, _ in images] == ['f1', 'f2']
    restored = deepcopy(wire)
    for original, projected in zip(metadata['frames'], restored['frames'], strict=True):
        projected['frame_id'] = receipt['frame_aliases'][projected['frame_id']]
        assert projected == {k: v for k, v in original.items() if k not in scene_transport.STORAGE_FIELDS}
    restored['frames'] = metadata['frames']
    restored['source_ref'] = metadata['source_ref']
    assert restored == metadata
    raw = response(wire)
    raw['summary'] = 'f1 remains free text, not a structured reference'
    expanded = scene_transport.expand(raw, receipt)
    assert expanded['summary'] == raw['summary']
    assert [f['frame_id'] for f in expanded['frame_observations']] == [f['frame_id'] for f in metadata['frames']]
    assert receipt['metadata_characters_after'] < receipt['metadata_characters_before']
    raw['frame_observations'][0]['frame_id'] = 'f999'
    with pytest.raises(ValueError, match='invented'):
        scene_transport.expand(raw, receipt)
    with pytest.raises(ValueError, match='order'):
        scene_transport.compact(metadata, list(reversed(paths)))
    metadata['frames'][1]['frame_id'] = metadata['frames'][0]['frame_id']
    with pytest.raises(ValueError, match='Duplicate'):
        scene_transport.compact(metadata, paths)


def test_real_adapter_keeps_prompt_images_thinking_and_output_budget(scene, monkeypatch):
    config, metadata, _, folder = scene
    monkeypatch.setattr('visioncortex.mllm.model_api_key', lambda *a, **kw: 'synthetic-only')
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        supplied = json.loads(body['messages'][1]['content'][0]['text'])
        chunks = [{'id': 'test-request', 'choices': [{'delta': {'content': json.dumps(response(supplied))}, 'finish_reason': 'stop'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}]
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n')
    for compact in (False, True):
        config['mllm']['compact_scene_metadata'] = compact
        analyzer = ArkAnalyzer(config)
        analyzer.client.close()
        analyzer.client = httpx.Client(transport=httpx.MockTransport(handler))
        try:
            raw, parsed, reused = run(scene, analyzer)
        finally:
            analyzer.close()
        assert not reused and len(parsed['frame_observations']) == 2
        assert read_json(folder/'Input.json')['metadata'] == metadata
        if compact:
            assert raw['scene_input_transport']['quality_equivalence'] == 'NOT_PROVEN'
            assert read_json(folder/'Result.json')['wire_response']['frame_observations'][0]['frame_id'] == 'f1'
    first, second = requests
    assert first['messages'][0] == second['messages'][0]
    assert first['max_tokens'] == second['max_tokens'] == 8192
    assert first['enable_thinking'] is second['enable_thinking'] is True
    assert first['model'] == second['model']
    def images(body):
        return [x for x in body['messages'][1]['content'] if x['type'] == 'image_url']
    assert images(first) == images(second)
    assert UsageLedger(config).summary(today())['known_tokens']['total_tokens'] == 60


def test_restart_reuses_successful_windows_and_retries_only_failed_window(scene):
    config, metadata, paths, folder = scene
    analyzer = Analyzer()
    run(scene, analyzer)
    second = (config, metadata, paths, folder.parent/'second')
    # Different semantic input prevents treating these two windows as equal.
    second = (config, {**metadata, 'segment_id': 'second'}, paths, second[3])
    failed = {'status': 'failed', 'attempts': 2, 'usage': {'total_tokens': 15, 'unknown_attempt_count': 1}}
    with pytest.raises(ValueError, match='failed'):
        run(second, Analyzer([failed]))
    restarted = Analyzer()
    raw, _, reused = run(scene, restarted)
    assert reused and restarted.calls == 0
    assert scene_requests.additional_usage(raw, reused)['total_tokens'] == 0
    run(second, restarted)
    assert restarted.calls == 1
    history = list((second[3]/'History').glob('*.json'))
    assert any(read_json(p)['result']['model_result']['status'] == 'failed' for p in history)
    stats = UsageLedger(config).summary(today())
    assert stats['calls'] == 3 and stats['attempts_or_reservations'] == 4
    assert stats['known_tokens']['total_tokens'] == 75 and stats['unknown_attempts'] == 1


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'invented', 'out_of_window'])
def test_completed_but_invalid_response_is_never_reused(scene, problem):
    config, metadata, _, _ = scene
    bad = response(metadata)
    if problem == 'missing':
        bad['frame_observations'].pop()
    elif problem == 'duplicate':
        bad['frame_observations'][1] = bad['frame_observations'][0]
    elif problem == 'invented':
        bad['frame_observations'][0]['frame_id'] = 'invented'
    else:
        bad['steps'] = [{'start_ms': 9999, 'end_ms': 10000, 'description': 'bad time',
                         'frame_ids': [metadata['frames'][0]['frame_id']], 'basis': 'uncertain'}]
    with pytest.raises(ValueError):
        run(scene, Analyzer([bad]))
    good = Analyzer()
    assert run(scene, good)[2] is False and good.calls == 1
    assert UsageLedger(config).summary(today())['calls'] == 2


@pytest.mark.parametrize('change', [{'model': 'different'}, {'request_image_max_edge': 512},
                                    {'request_image_jpeg_quality': 70}, {'max_output_tokens': 16384},
                                    {'quality_mode': 'balanced'}, {'compact_scene_metadata': True}])
def test_cache_binds_generation_and_transport_policy(scene, change):
    run(scene, Analyzer())
    scene[0]['mllm'].update(change)
    analyzer = Analyzer()
    assert not run(scene, analyzer)[2] and analyzer.calls == 1


def test_cache_relocates_storage_but_binds_semantics(scene):
    run(scene, Analyzer())
    config, metadata, paths, folder = scene
    relocated = deepcopy(metadata)
    relocated['frames'][0]['path'] = 'relocated/reference.png'
    relocated['source_ref']['path'] = 'relocated/video.mp4'
    another = (config, relocated, paths, folder.parent/'relocated')
    analyzer = Analyzer()
    assert run(another, analyzer)[2] and analyzer.calls == 0
    for change in ({'coverage': {'mode': 'different'}}, {'protocol': {'text': 'new'}},
                   {'comments': [{'comment_id': 'new', 'text': 'new'}]}):
        assert not run((config, {**relocated, **change}, paths, another[3]), analyzer)[2]
    relocated['frames'][0]['capture_us'] += 1
    assert not run(another, analyzer)[2]


def test_legacy_reuse_needs_complete_input_verified_cache_and_matching_model(scene):
    from visioncortex import device_day_semantic_cache as cache
    config, metadata, paths, folder = scene
    raw = response(metadata) | {'model': config['mllm']['model'], 'provider': 'aliyun', 'api_protocol': 'chat_completions',
                               'request_image_transport': {'maximum_edge_pixels': 768, 'jpeg_quality': 82, 'source_artifacts_mutated': False}}
    input_key = digest([metadata, SCENE_PROMPT])
    atomic_json(folder/'Input.json', {'input_key': input_key, 'prompt': SCENE_PROMPT, 'metadata': metadata})
    atomic_json(folder/'Result.json', {'input_key': input_key, 'model_result': raw})
    cache.save(config, cache.identity(config, SCENE_PROMPT, metadata), raw)
    analyzer = Analyzer()
    assert run(scene, analyzer)[2] and analyzer.calls == 0
    assert UsageLedger(config).summary(today())['calls'] == 0
    # Legacy v1 by itself cannot prove unchanged coverage or capture time.
    changed = deepcopy(metadata)
    changed['frames'][0]['capture_us'] += 1
    assert not run((config, changed, paths, folder.parent/'legacy-only'), analyzer)[2]


def test_legacy_success_is_reused_across_stage_revision_folders(scene):
    from visioncortex import device_day_semantic_cache as cache
    config, metadata, paths, folder = scene
    old_folder = folder/'segment'/'old-key'/'00-01_00-02'
    new_folder = folder/'segment'/'new-key'/'00-01_00-02'
    old_metadata = deepcopy(metadata)
    old_metadata['frames'][0]['path'] = 'old-key/frame.png'
    raw = response(metadata) | {'model': config['mllm']['model'], 'provider': 'aliyun', 'api_protocol': 'chat_completions',
                               'request_image_transport': {'maximum_edge_pixels': 768, 'jpeg_quality': 82, 'source_artifacts_mutated': False}}
    key = digest([old_metadata, SCENE_PROMPT])
    atomic_json(old_folder/'Input.json', {'input_key': key, 'prompt': SCENE_PROMPT, 'metadata': old_metadata})
    atomic_json(old_folder/'Result.json', {'input_key': key, 'model_result': raw})
    cache.save(config, cache.identity(config, SCENE_PROMPT, metadata), raw)
    analyzer = Analyzer()
    assert run((config, metadata, paths, new_folder), analyzer)[2] and analyzer.calls == 0
    assert read_json(old_folder/'Input.json')['metadata'] == old_metadata


def test_compact_structured_steps_restore_full_references(scene):
    from visioncortex.device_day_steps import validate
    _, metadata, paths, _ = scene
    wire, _, receipt = scene_transport.compact(metadata, paths)
    raw = response(wire)
    raw.pop('steps')
    raw['activity_observed'] = 'active'
    raw['experiment_steps'] = [{'start_ms': 1250, 'end_ms': 1751, 'description': 'test-only movement',
        'action': 'moving', 'objects': ['test object'], 'before_state': 'unknown', 'after_state': 'unknown',
        'visible_evidence': 'synthetic frames', 'outcome': 'incomplete_or_uncertain', 'frame_ids': ['f1', 'f2'],
        'comment_ids': [], 'basis': 'uncertain'}]
    expanded = scene_transport.expand(raw, receipt)
    parsed = validate(expanded, metadata['frames'], 1000, 2000, 1789005600000000, {'origin_us': 1789005600000000})
    assert parsed['experiment_steps'][0]['frame_ids'] == [f['frame_id'] for f in metadata['frames']]
    raw['experiment_steps'][0]['frame_ids'] = ['unknown']
    with pytest.raises(ValueError, match='invented'):
        scene_transport.expand(raw, receipt)


def test_concurrent_identical_window_has_one_actual_submission(scene):
    entered, release = Event(), Event()
    class Blocking(Analyzer):
        def _call(self, *args, **kwargs):
            entered.set()
            assert release.wait(10)
            return super()._call(*args, **kwargs)
    first, second = Blocking(), Analyzer()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(run, scene, first)
        try:
            assert entered.wait(10)
            with pytest.raises(BlockingIOError):
                run(scene, second)
        finally:
            release.set()
        pending.result()
    assert run(scene, second)[2] and second.calls == 0 and first.calls == 1


def test_atomic_budget_survives_restart_and_does_not_touch_speech_gate(scene):
    from visioncortex.device_day_provider_gate import ProviderGate
    config = scene[0]
    config['mllm']['scene_daily_attempt_limit'] = 2
    ledger = UsageLedger(config)
    def reserve(_):
        try:
            return ledger.reserve('same-day')
        except ProviderUnavailable:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        reservations = [x for x in pool.map(reserve, range(4)) if x]
    assert len(reservations) == 1
    assert UsageLedger(config).summary(today())['pending_calls'] == 1
    assert not ProviderGate(config).blocks('stt')
    with pytest.raises(ProviderUnavailable):
        UsageLedger(config).reserve('restart')
    ledger.finish(reservations[0], response(scene[1]))
    assert ledger.summary(today())['attempts_or_reservations'] == 1
    config['mllm']['max_retries'] = 1
    assert ledger.reserve('remaining-attempt')


def test_token_threshold_blocks_pending_unknown_and_reached_usage(scene):
    config = scene[0]
    config['mllm']['scene_daily_token_limit'] = 25
    ledger = UsageLedger(config)
    pending = ledger.reserve('one')
    with pytest.raises(ProviderUnavailable):
        ledger.reserve('pending-usage')
    ledger.finish(pending, {'status': 'completed', 'attempts': 2, 'usage': {'total_tokens': 10},
                            'attempt_receipts': [{'usage': {'total_tokens': 10}}]})
    assert ledger.summary(today())['unknown_attempts'] == 1
    with pytest.raises(ProviderUnavailable):
        ledger.reserve('unknown-usage')
    # A separate ledger proves one call can exceed the threshold; no false hard-cap claim.
    config['storage']['local_runtime_root'] += '-threshold'
    ledger = UsageLedger(config)
    pending = ledger.reserve('one')
    ledger.finish(pending, response(scene[1]))
    assert ledger.summary(today())['known_tokens']['total_tokens'] == 30
    with pytest.raises(ProviderUnavailable):
        ledger.reserve('threshold-reached')


def test_readonly_summary_and_unsubmitted_calls(scene):
    config = scene[0]
    ledger = UsageLedger(config, readonly=True)
    assert ledger.summary(today())['calls'] == 0
    assert not ledger.path.exists()
    ledger = UsageLedger(config)
    reservation = ledger.reserve('not-submitted')
    ledger.finish(reservation, {'status': 'skipped_missing_api_key'})
    assert ledger.summary(today())['attempts_or_reservations'] == 0
    assert ledger.summary(today())['unknown_attempts'] == 0


def test_cache_hit_is_free_even_after_budget_reached(scene):
    scene[0]['mllm'].update(max_retries=1, scene_daily_attempt_limit=1)
    run(scene, Analyzer())
    another = Analyzer()
    assert run(scene, another)[2] and another.calls == 0
    scene[1]['segment_id'] = 'new-window'
    with pytest.raises(ProviderUnavailable):
        run(scene, another)
    assert another.calls == 0


def test_durable_response_recovers_usage_commit_failure_without_new_call(scene, monkeypatch):
    original = UsageLedger.finish
    def fail_once(*args):
        raise OSError('synthetic ledger commit interruption')
    monkeypatch.setattr(UsageLedger, 'finish', fail_once)
    analyzer = Analyzer()
    with pytest.raises(OSError, match='commit interruption'):
        run(scene, analyzer)
    assert analyzer.calls == 1
    assert UsageLedger(scene[0]).summary(today())['pending_calls'] == 1
    monkeypatch.setattr(UsageLedger, 'finish', original)
    assert run(scene, analyzer)[2] and analyzer.calls == 1
    stats = UsageLedger(scene[0]).summary(today())
    assert stats['pending_calls'] == 0 and stats['known_tokens']['total_tokens'] == 30


def test_daily_budget_uses_reservation_day_and_resets_next_day(scene, monkeypatch):
    import visioncortex.multimodal_usage as usage
    day = [datetime(2026, 9, 20, 23, 59, tzinfo=ZoneInfo('Asia/Shanghai'))]
    class Clock:
        @staticmethod
        def now(tz):
            return day[0]
    monkeypatch.setattr(usage, 'datetime', Clock)
    scene[0]['mllm'].update(max_retries=1, scene_daily_attempt_limit=1)
    ledger = UsageLedger(scene[0])
    pending = ledger.reserve('before-midnight')
    day[0] = datetime(2026, 9, 21, 0, 1, tzinfo=ZoneInfo('Asia/Shanghai'))
    ledger.finish(pending, response(scene[1]))
    assert ledger.summary('2026-09-20')['known_tokens']['total_tokens'] == 30
    assert ledger.summary('2026-09-21')['calls'] == 0
    assert ledger.reserve('new-day')


@pytest.mark.parametrize('settings', [{'compact_scene_metadata': 'true'}, {'scene_daily_token_limit': -1},
                                      {'scene_daily_attempt_limit': True}, {'scene_daily_attempt_limit': 0}])
def test_budget_configuration_rejects_ambiguous_values(settings):
    with pytest.raises(ValueError):
        validate(settings)

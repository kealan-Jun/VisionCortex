"""JoyAI port boundaries; fake model transport, no GPU/model startup."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

from .test_model_services import load, setup


def config():
    return dict(engine='joyai-hf', device='cuda:0', do_resize=False, do_sample_frames=False,
        attention='flash_attention_2', cache_implementation='offloaded', served_model_name='fixture-joyai',
        max_frames=60, max_duration_seconds=2, max_media_bytes=64*1024**2,
        max_request_bytes=96*1024**2, max_decoded_bytes=1024**3, max_frame_pixels=16_000_000,
        max_model_len=131072, max_new_tokens=256, request_seconds=900,
        gpu_memory_limit_bytes=42*1024**3, gpu_reserve_bytes=4*1024**3)


def body(text='原先验：{"historical_prior":[],"contemporaneous_evidence":[]}'):
    return dict(model='fixture-joyai', messages=[dict(role='user', content=[
        dict(type='text', text='视频原偏移0至2秒。'),
        dict(type='video_url', video_url=dict(url='data:video/mp4;base64,'+base64.b64encode(b'fixture-video').decode())),
        dict(type='text', text=text)])], temperature=0, max_tokens=128,
        response_format={'type': 'json_object'}, chat_template_kwargs={'enable_thinking': False})


def test_native_settings_require_explicit_memory_frame_and_processor_limits():
    module = load('joyai_vlm')
    assert module.settings(config())['gpu_memory_limit_bytes'] == 42*1024**3
    for change in [dict(engine='vllm'), dict(do_resize=True), dict(do_sample_frames=True),
                   dict(max_frames=True), dict(gpu_reserve_bytes=0), dict(gpu_memory_limit_bytes=43*1024**3),
                   dict(max_duration_seconds=11), dict(max_request_bytes=97*1024**2)]:
        with pytest.raises(ValueError):
            module.settings(dict(config(), **change))


def test_processor_rope_inputs_are_preserved_or_explicitly_rejected():
    module = load('joyai_vlm')

    class CurrentModel:
        def forward(self, input_ids=None, mm_token_type_ids=None):
            pass

    class LegacyModel:
        def forward(self, input_ids=None, **kwargs):
            pass

    raw = dict(input_ids=object(), mm_token_type_ids=object())
    assert module.validate_model_inputs(raw, CurrentModel()) is raw
    assert raw['mm_token_type_ids'] is not None
    with pytest.raises(ValueError, match='mm_token_type_ids'):
        module.validate_model_inputs(raw, LegacyModel())


def test_full_ordered_prior_text_survives_without_media_fetch_or_silent_drop():
    module = load('joyai_vlm')
    request = body()
    messages, raw, texts, maximum = module.parse_request(request, config())
    assert raw == b'fixture-video' and maximum == 128
    assert texts == [p['text'] for p in request['messages'][0]['content'] if p['type'] == 'text']
    assert messages[0]['content'] == [dict(type='text', text=texts[0]), dict(type='video'), dict(type='text', text=texts[1])]
    assert request['messages'][0]['content'][1]['type'] == 'video_url'
    for change in ['remote_url', 'extra_image', 'two_videos', 'system_role', 'large_output']:
        invalid = deepcopy(request)
        parts = invalid['messages'][0]['content']
        if change == 'remote_url':
            parts[1]['video_url']['url'] = 'https://example.invalid/video.mp4'
        elif change == 'extra_image':
            parts.append(dict(type='image_url', image_url={'url': 'data:image/png;base64,QQ=='}))
        elif change == 'two_videos':
            parts.append(parts[1])
        elif change == 'system_role':
            invalid['messages'][0]['role'] = 'system'
        else:
            invalid['max_tokens'] = 257
        with pytest.raises(ValueError):
            module.parse_request(invalid, config())


def test_service_auth_actual_body_receipt_and_content_addressed_cache(tmp_path):
    module = load('joyai_vlm')
    calls = []

    def infer(request, receipt):
        calls.append(request)
        _, _, texts, _ = module.parse_request(request, config())
        receipt.update(input_texts=texts, input_texts_sha256=[module.sha(t.encode()) for t in texts],
            model_forward_entries=1, vision_forward_entries=1, model_consumed_full_prompt=True)
        return '{"events":[]}', 6, 'stop'

    app = module.application(infer, config(), 'fixture-token', 'fixture-revision', 'a'*64, tmp_path, {'fixture': True})
    with TestClient(app) as client:
        assert client.get('/health').json()['loaded']['fixture']
        assert client.post('/v1/chat/completions', json=body()).status_code == 401
        headers = {'Authorization': 'Bearer fixture-token'}
        assert client.get('/v1/models', headers=headers).json()['data'][0]['id'] == 'fixture-joyai'
        first = client.post('/v1/chat/completions', json=body(), headers=headers)
        assert first.status_code == 200 and first.json()['vc_lab_execution']['model_consumed_full_prompt']
        again = client.post('/v1/chat/completions', json=body(), headers=headers)
        assert again.json()['vc_lab_cache_hit'] and len(calls) == 1
        assert client.post('/v1/chat/completions', json=body('changed real prior'), headers=headers).status_code == 200
        assert len(calls) == 2
    saved = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert all(p['actual_model_invocation'] and p['status'] == 'completed' for p in saved)
    assert any('contemporaneous_evidence' in p['input_texts'][1] for p in saved)


def test_failed_request_is_preserved_and_not_identically_reinvoked(tmp_path):
    module = load('joyai_vlm')
    calls = []

    def infer(request, receipt):
        calls.append(request)
        raise module.CapacityError('fixture resource limit')

    app = module.application(infer, config(), 'fixture-token', 'fixture-revision', 'a'*64, tmp_path, {})
    with TestClient(app) as client:
        headers = {'Authorization': 'Bearer fixture-token'}
        assert client.post('/v1/chat/completions', json=body(), headers=headers).status_code == 422
        assert client.post('/v1/chat/completions', json=body(), headers=headers).status_code == 409
    saved = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert saved['status'] == 'failed' and not saved['actual_model_invocation'] and len(calls) == 1


def test_explicit_joyai_dispatch_uses_repo_service_and_identity_includes_helper(tmp_path):
    manager_module = load('lab_model_services')
    root, _, _, _, document, path = setup(tmp_path)
    document['services']['vlm'].update(config())
    path.write_text(json.dumps(document))
    manager = manager_module.ModelServices(root, path)
    command, _ = manager.command('vlm', probe_port=False)
    assert command[:2] == [sys.executable, '-u']
    assert Path(command[2]).name == 'lab_model_service.py'
    assert command[command.index('--kind')+1] == 'vlm' and 'vllm.entrypoints.cli.main' not in command
    source = load('lab_model_service')
    verified = {'fixture-vlm': {'manifest_sha256': 'a'*64, 'files_sha256': 'b'*64}}
    sys.path.insert(0, str(Path(command[2]).parent))
    try:
        first = source.inference_identity('vlm', document, verified)
        document['services']['vlm']['max_new_tokens'] -= 1
        assert source.inference_identity('vlm', document, verified) != first
    finally:
        sys.path.pop(0)

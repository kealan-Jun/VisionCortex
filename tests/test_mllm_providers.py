from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from visioncortex.mllm import ArkAnalyzer, _usage
from visioncortex.mllm_provider import build_vision_request, connection_identity, normalize_connection, vision_request_identity
from visioncortex.provider_connection import adapter_identity, connection_health, verification_matches, verify_connection


CATALOG = json.loads((Path(__file__).resolve().parents[1] / 'configs/mllm-providers.json').read_text())


def selection(provider='aliyun', **overrides):
    return normalize_connection({**CATALOG[provider], "provider": provider, "model": CATALOG[provider].get("model") or "synthetic-vision", **overrides})


@pytest.mark.parametrize('configured,effective', [(90, 180), (180, 180), (300, 300)])
def test_pipeline_keeps_verified_provider_timeout_for_retained_jobs(default_config, configured, effective):
    from copy import deepcopy
    from visioncortex.pipeline import EvidencePipeline

    config = deepcopy(default_config)
    config['mllm'].update(selection('aliyun'), timeout_seconds=configured)
    original = deepcopy(config)
    pipeline = EvidencePipeline(config)
    assert config == original, 'The immutable queued settings must not be rewritten'
    assert pipeline.config['mllm']['timeout_seconds'] == effective
    assert connection_identity(pipeline.config['mllm']) == connection_identity(config['mllm'])
    assert vision_request_identity(pipeline.config['mllm']) == vision_request_identity(config['mllm'])
    assert pipeline._mllm_timeout_policy['effective_timeout_seconds'] == effective
    assert pipeline._mllm_timeout_policy['request_content_changed'] is False


def test_catalog_recommends_vision_candidates_and_keeps_custom_choice():
    assert {'volcengine', 'aliyun', 'zhipu', 'custom', 'google', 'openrouter', 'siliconflow'} <= set(CATALOG)
    for provider in ('volcengine', 'aliyun', 'zhipu'):
        entry = CATALOG[provider]
        ids = {item['id'] for item in entry['models']}
        assert set(entry['recommendations'].values()) <= ids
        assert all(item['source'].startswith('https://') for item in entry['models'])
        assert entry['real_experiment_ranking'] == 'NOT_PROVEN'
    assert 'glm-5.3-flash' in {item['id'] for item in CATALOG['zhipu']['models']}
    assert 'glm-5.3' not in {item['id'] for item in CATALOG['zhipu']['models']}


@pytest.mark.parametrize('provider', ['volcengine', 'aliyun', 'zhipu', 'custom', 'google', 'openrouter', 'siliconflow'])
def test_production_request_keeps_all_images_and_uses_selected_protocol(provider):
    settings = dict(selection(provider) if provider != 'custom' else selection(provider, base_url='https://example.test/v1', model='my-vision'), max_output_tokens=1024)
    content = [{'type': 'input_text', 'text': 'metadata'}]
    for index in range(12):
        content.extend([{'type': 'input_text', 'text': f'image-{index}'}, {'type': 'input_image', 'image_url': 'data:image/png;base64,aW1hZ2U='}])
    url, request = build_vision_request(settings, 'evidence rules', content)
    assert request['model'] == settings['model']
    if provider == 'volcengine':
        assert url.endswith('/responses')
        assert len(request['input'][0]['content']) == 25
        assert request['thinking']['type'] == 'enabled'
    else:
        assert url.endswith('/chat/completions')
        assert len(request['messages'][1]['content']) == 25
        assert 'input' not in request and 'store' not in request
        assert request['stream'] is (provider == 'aliyun')
    assert request.get('max_output_tokens', request.get('max_tokens')) == 8192


def test_model_specific_zhipu_image_and_thinking_parameters():
    content = [{'type': 'input_image', 'image_url': 'data:image/png;base64,aW1hZ2U='}]
    _, current = build_vision_request(selection('zhipu', quality_mode='balanced'), 'rules', content)
    assert current['thinking'] == {'type': 'enabled'}
    assert current['messages'][1]['content'][0]['image_url']['url'].startswith('data:')
    _, previous = build_vision_request(selection('zhipu', model='glm-4.6v', quality_mode='balanced'), 'rules', content)
    assert previous['thinking'] == {'type': 'disabled'}
    assert previous['messages'][1]['content'][0]['image_url']['url'] == 'aW1hZ2U='


@pytest.mark.parametrize('url', ['http://example.test/v1', 'https://user:secret@example.test/v1', 'https://example.test/v1?api_key=secret', 'https://example.test/v1#fragment', 'https://example.test/v1/chat/completions'])
def test_connection_rejects_credential_urls_and_wrong_endpoint_shape(url):
    with pytest.raises(ValueError):
        selection('aliyun', base_url=url)


def test_usage_retains_server_reported_zero_tokens():
    assert _usage({'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}})['input_tokens'] == 0


def test_semantic_cache_binds_effective_model_policy_without_credentials(tmp_path):
    from visioncortex.archive import _semantic_fingerprint
    selected = selection('aliyun')
    identity = vision_request_identity(selected)
    assert identity == vision_request_identity(selected | {'api_key_env': 'DIFFERENT_SECRET_STORE'})
    baseline = _semantic_fingerprint('event', {'mllm': selected}, 'same prompt', {}, [])
    for change in ({'quality_mode': 'balanced'}, {'model': 'other-vision'}, {'base_url': 'https://other.test/v1'}, {'max_output_tokens': 16384}, {'api_protocol': 'ark_responses'}):
        updated = selected | change
        assert vision_request_identity(updated) != identity
        assert _semantic_fingerprint('event', {'mllm': updated}, 'same prompt', {}, []) != baseline


def factory_for(handler):
    def factory(config):
        analyzer = ArkAnalyzer(config)
        analyzer.client.close()
        analyzer.client = httpx.Client(transport=httpx.MockTransport(handler))
        return analyzer
    return factory


def streamed_response(content, *, done=True, finish="stop", usage=True):
    chunks = [
        {"id": "stream-test", "model": "synthetic-vision", "choices": [{"index": 0, "delta": {"reasoning_content": "private provider reasoning; not the answer"}}]},
        {"choices": [{"index": 0, "delta": {"content": content[:3]}}]},
        {"choices": [{"index": 0, "delta": {"content": content[3:]}, "finish_reason": finish}]},
    ]
    if usage:
        chunks.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
    body = "".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\r\n\r\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def test_streamed_probe_uses_final_answer_and_trailing_usage(tmp_path, monkeypatch):
    expected = fixed_challenge(monkeypatch)
    def handler(request):
        data = json.loads(request.content)
        assert data["stream"] and data["stream_options"]["include_usage"]
        assert request.extensions["timeout"]["read"] == 30
        return streamed_response(json.dumps({"images": expected}))
    receipt = verify_connection(selection(), "synthetic-probe-credential", tmp_path, analyzer_factory=factory_for(handler))
    assert receipt["status"] == "verified"
    assert receipt["usage"]["total_tokens"] == 30
    assert receipt["transport"]["mode"] == "stream"
    assert receipt["request_id"] == "stream-test"
    assert "private provider reasoning" not in json.dumps(receipt)


@pytest.mark.parametrize("done,finish", [(False, "stop"), (True, None), (True, "length"), (True, "content_filter")])
def test_partial_or_truncated_stream_cannot_pass_connection(done, finish, tmp_path, monkeypatch):
    expected = fixed_challenge(monkeypatch)
    receipt = verify_connection(selection(), "synthetic-probe-credential", tmp_path,
        analyzer_factory=factory_for(lambda request: streamed_response(json.dumps({"images": expected}), done=done, finish=finish)))
    assert receipt["status"] == "failed"


def test_stream_deadline_and_size_are_bounded():
    from visioncortex.mllm import _read_streamed_chat
    response = streamed_response('{"ok":true}')
    response.request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    with pytest.raises(httpx.ReadTimeout):
        _read_streamed_chat(response, deadline=0)
    with pytest.raises(ValueError, match="接收大小"):
        _read_streamed_chat(response, deadline=float("inf"), max_bytes=20)


@pytest.mark.parametrize("event", [[], {"choices": "invalid"}, {"choices": [1]}, {"choices": [{"delta": "invalid"}]}, {"usage": 123}])
def test_malformed_stream_is_a_reported_failure_not_an_uncaught_crash(event, tmp_path):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")
    receipt = verify_connection(selection(), "synthetic-probe-credential", tmp_path, analyzer_factory=factory_for(handler))
    assert receipt["status"] == "failed"


def test_stream_handles_split_utf8_frames_and_preserves_unknown_usage(tmp_path, monkeypatch):
    expected = fixed_challenge(monkeypatch)
    wire = streamed_response(json.dumps({"images": expected}, ensure_ascii=False), usage=False).content
    class SplitStream(httpx.SyncByteStream):
        def __iter__(self):
            for offset in range(0, len(wire), 7):
                yield wire[offset:offset + 7]
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SplitStream())
    receipt = verify_connection(selection(), "synthetic-probe-credential", tmp_path, analyzer_factory=factory_for(handler))
    assert receipt["status"] == "verified"
    assert receipt["usage"]["total_tokens"] is None
    assert receipt["usage"]["unknown_attempt_count"] == 1


def test_stream_framing_keeps_existing_validated_semantic_cache_identity(monkeypatch):
    import visioncortex.mllm_provider as provider
    selected = selection()
    identity = vision_request_identity(selected)
    original = provider.build_vision_request
    def legacy(*args):
        url, request = original(*args)
        request["stream"] = False
        request.pop("stream_options", None)
        return url, request
    monkeypatch.setattr(provider, "build_vision_request", legacy)
    assert vision_request_identity(selected) == identity


def fixed_challenge(monkeypatch):
    monkeypatch.setattr('visioncortex.provider_connection.secrets.randbelow', lambda _: 2345)
    monkeypatch.setattr('visioncortex.provider_connection.secrets.choice', lambda values: values[0])
    return [{'code': '3345', 'color': 'red', 'shape': 'circle'}] * 2


@pytest.mark.parametrize('provider', ['volcengine', 'aliyun', 'zhipu', 'google', 'openrouter', 'siliconflow'])
def test_connection_probe_uses_production_transport_and_binds_saved_model(provider, tmp_path, monkeypatch):
    expected = fixed_challenge(monkeypatch)
    selected = selection(provider)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers['Authorization'] == 'Bearer synthetic-probe-credential'
        body = json.loads(request.content)
        assert body['model'] == selected['model']
        content = body['input'][0]['content'] if provider == 'volcengine' else body['messages'][1]['content']
        assert len([item for item in content if item['type'] in {'input_image', 'image_url'}]) == 2
        assert '3345' not in ''.join(item.get('text', '') for item in content)
        return httpx.Response(200, json={'id': 'mock-request-1', 'model': selected['model'], 'choices': [{'message': {'content': json.dumps({'images': expected})}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 800, 'completion_tokens': 70, 'total_tokens': 870}})

    receipt = verify_connection(selected, 'synthetic-probe-credential', tmp_path, analyzer_factory=factory_for(handler))
    assert len(requests) == 1
    assert receipt['status'] == 'verified'
    assert receipt['request_id'] == 'mock-request-1'
    assert receipt['usage']['total_tokens'] == 870
    assert receipt['real_video_quality'] == 'NOT_PROVEN'
    assert verification_matches(selected, receipt)
    assert not verification_matches(dict(selected, model='other-model'), receipt)
    assert not verification_matches(dict(selected, quality_mode='balanced'), receipt)
    assert not verification_matches(dict(selected, base_url='https://elsewhere.test/v1'), receipt)
    assert connection_health(selected | {'connection_verification': receipt}, False)['status'] == 'not_verified'
    assert connection_health(selected | {'connection_verification': receipt}, True)['status'] == 'verified'
    assert 'synthetic-probe-credential' not in json.dumps(receipt)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('status', [400, 401, 403, 404, 429, 503])
def test_failed_call_never_becomes_verified_even_if_provider_echoes_key(status, tmp_path):
    def handler(_request):
        return httpx.Response(status, json={'error': {'message': 'synthetic-probe-credential'}})
    result = verify_connection(selection(), 'synthetic-probe-credential', tmp_path, analyzer_factory=factory_for(handler))
    assert result['status'] == 'failed'
    assert result['http_status'] == status
    assert not verification_matches(selection(), result)
    assert 'synthetic-probe-credential' not in json.dumps(result)


@pytest.mark.parametrize('content,finish', [('{}', 'stop'), ('{"images":[]}', 'stop'), ('not JSON', 'stop'), ('{"images":[]}', 'length')])
def test_http_200_without_correct_vision_answer_does_not_pass(content, finish, tmp_path):
    def handler(_request):
        return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': finish}]})
    result = verify_connection(selection(), 'synthetic-probe-credential', tmp_path, analyzer_factory=factory_for(handler))
    assert result['status'] == 'failed'
    assert not verification_matches(selection(), result)


def test_timeout_is_reported_without_retrying_billable_probe(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout('connection timeout')
    result = verify_connection(selection(), 'synthetic-probe-credential', tmp_path, analyzer_factory=factory_for(handler))
    assert result['status'] == 'failed'
    assert len(calls) == 1


def test_desktop_uses_verified_selection_without_secret_in_config_or_engine_identity(tmp_path, monkeypatch):
    import importlib.util
    import shutil
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('portable', root / 'tools/rtx4050_portable.py')
    portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portable)
    monkeypatch.setattr(portable, 'DESKTOP_MODE', True)
    monkeypatch.setattr(portable.os, 'environ', dict(portable.os.environ))
    shutil.copytree(root / 'configs', tmp_path / 'configs')
    (tmp_path / 'SHA256SUMS.json').write_text('{}')
    engines = []
    for provider in ('aliyun', 'zhipu'):
        selected = selection(provider)
        receipt = {'status': 'verified', 'model_invocation': 'PROVEN', 'connection_sha256': connection_identity(selected), 'adapter_sha256': adapter_identity()}
        monkeypatch.setenv('MLLM_API_KEY', 'synthetic-desktop-secret')
        monkeypatch.setenv('VISIONCORTEX_DESKTOP_CONNECTION', json.dumps({'connection': selected, 'verification': receipt}))
        portable.configure_environment(tmp_path)
        path, config = portable.effective_config(tmp_path, {'gpu_uuid': 'test'})
        assert config['mllm']['api_key_env'] == 'MLLM_API_KEY'
        assert config['mllm']['provider'] == provider
        assert config['mllm']['model'] == selected['model']
        assert 'synthetic-desktop-secret' not in path.read_text()
        engines.append(config['models']['first_person_engine'])
    assert engines[0] == engines[1]
    monkeypatch.setenv('VISIONCORTEX_DESKTOP_CONNECTION', '{}')
    with pytest.raises(ValueError):
        portable.effective_config(tmp_path, {'gpu_uuid': 'test'})


def test_unlisted_provider_uses_protocol_adapter_without_name_allowlist():
    connection = normalize_connection({'provider': 'example-vendor', 'base_url': 'https://example.test/v1',
                                       'model': 'vision', 'api_protocol': 'chat_completions'})
    url, body = build_vision_request(connection, 'rules', [{'type': 'input_image', 'image_url': 'data:image/png;base64,eA=='}])
    assert url == 'https://example.test/v1/chat/completions'
    assert body['messages'][1]['content'][0]['image_url']['url'] == 'data:image/png;base64,eA=='
    for change in ({'provider': '../vendor'}, {'provider': 'a' * 65}, {'api_protocol': 'unknown'}):
        with pytest.raises(ValueError):
            normalize_connection(connection | change)

"""Offline request contracts; provider availability requires live verification."""
import pytest

from visioncortex.mllm_provider import build_vision_request


@pytest.mark.parametrize('model', ['glm-5.3-flash', 'glm-5.3-flashx'])
def test_nianfeng_glm_uses_selected_model_and_required_thinking(model):
    settings = {'provider': 'nianfeng', 'model': model, 'base_url': 'https://provider.invalid/v1',
                'api_protocol': 'chat_completions', 'quality_mode': 'balanced', 'max_output_tokens': 512}
    url, request = build_vision_request(settings, 'system', [
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,owned-fixture'},
        {'type': 'input_text', 'text': 'input'}])
    assert url == 'https://provider.invalid/v1/chat/completions'
    assert request['model'] == model
    assert request['thinking'] == {'type': 'enabled'}
    assert request['max_tokens'] == 8192
    assert request['messages'][1]['content'][0]['image_url']['url'].startswith('data:')


def test_other_nianfeng_models_do_not_get_glm_parameters():
    _, request = build_vision_request({'provider': 'nianfeng', 'model': 'gpt-6-sol',
                                      'base_url': 'https://provider.invalid/v1',
                                      'api_protocol': 'chat_completions', 'max_output_tokens': 512}, '', [])
    assert 'thinking' not in request
    assert request['max_tokens'] == 512

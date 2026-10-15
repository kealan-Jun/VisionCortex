"""Template provenance at the real request/cache boundary; no external calls."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from visioncortex.device_day_call_templates import CLIP_UNDERSTANDING, DAY_SYNTHESIS, identify
from visioncortex.device_day_contract import atomic_json, read_json
from visioncortex.scene_requests import request_key, window
from visioncortex.stage_dependencies import impact


@pytest.mark.parametrize('template', [CLIP_UNDERSTANDING, DAY_SYNTHESIS])
def test_dates_and_devices_share_template_but_keep_distinct_evidence(default_config, tmp_path, template):
    config = deepcopy(default_config)
    config['storage'].update(local_runtime_root=str(tmp_path/'runtime'), local_cache_root=str(tmp_path/'cache'))
    calls = []

    class Analyzer:
        def _call(self, prompt, metadata, images, **kwargs):
            calls.append((prompt, deepcopy(metadata)))
            return {'status': 'completed', 'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}

    keys = []
    for archive, start in [('2026-01-01_camera-a', 0), ('2026-02-02_camera-b', 60000)]:
        metadata = {'schema_version': template.input_contract.split(':')[0], 'archive': archive,
                    'source_ref': {'sha256': 'a'*64}, 'activity': 'uncertain',
                    'start_ms': start, 'end_ms': start+1000, 'frames': [], 'comments': [], 'protocol': None}
        directory = tmp_path/archive
        input_path, result_path = directory/'Input.json', directory/'Result.json'
        keys.append(request_key(config, template.prompt, metadata))
        _, _, reused = window(config, Analyzer(), template.prompt, metadata, [], input_path, result_path, lambda raw: raw)
        assert not reused
        for path in (input_path, result_path):
            assert read_json(path)['call_template'] == template.receipt()
        # Old receipts lack a template label; exact paid inputs still reuse.
        for path in (input_path, result_path):
            old = read_json(path)
            old.pop('call_template')
            atomic_json(path, old)
        _, _, reused = window(config, Analyzer(), template.prompt, metadata, [], input_path, result_path, lambda raw: raw)
        assert reused
    assert len(calls) == 2 and keys[0] != keys[1]
    assert calls[0][0] == calls[1][0] == template.prompt
    assert calls[0][1]['archive'] != calls[1][1]['archive']


def test_custom_or_modified_prompts_cannot_claim_a_registered_template():
    assert identify('custom prompt') is None
    assert identify(CLIP_UNDERSTANDING.prompt+'changed instructions') is None
    assert identify(DAY_SYNTHESIS.prompt)['destination'] == 'LaboratoryDailyReport'
    with pytest.raises(FrozenInstanceError):
        CLIP_UNDERSTANDING.version = 2


def test_template_source_changes_invalidate_its_consumers():
    assert impact(['device_day_call_templates.py'])['changed_stages'] == ['report', 'understanding']
    assert impact(['device_day_steps.py'])['changed_stages'] == ['understanding']

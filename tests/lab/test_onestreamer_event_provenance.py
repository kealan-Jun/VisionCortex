from ._support import research_helpers
import importlib.util
import sys

import pytest


SCRIPTS = research_helpers()
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('one_event_provenance', SCRIPTS/'derive_onestreamer_event_provenance.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generation_time_visual_history_and_model_claim_remain_distinct():
    rows = [dict(start=i, end=i+1, raw=f'raw{i}', source='native.mp4',
        sampled_timestamps=[i+j/4 for j in range(4)]) for i in range(40)]
    case = dict(sources=[dict(relative='native.mp4', sha256='a'*64,
        stream=dict(avg_frame_rate='30/1'), offset_seconds=0)])
    event = dict(start=0, end=40, raw='raw39', kind='summary', text='[0s-3s] Model timing claim')
    result = MODULE.visual_provenance(event, rows, case)
    assert result['native_event_time']['start'] == 0  # Native summary bookkeeping stays immutable.
    assert result['generation_update'] == dict(start=39, end=40, index=39)
    history = result['actual_visual_history']
    assert history['successful_updates'] == 32
    assert history['sampled_frames'] == 128
    assert history['first_actual_source_time'] == 8
    assert history['last_actual_source_time'] == 39+22/30
    assert history['source_frame_groups'][0]['source_frame_indices'][1] == 247  # Native floor, not round.
    assert result['model_embedded_time_claims'] == [dict(start=0, end=3, accuracy='UNVERIFIED_MODEL_TEXT')]
    assert result['automatic_action_evidence_acceptance'] is False


def test_event_must_match_an_actual_successful_generation_update():
    with pytest.raises(ValueError, match='successful generation'):
        MODULE.visual_provenance(dict(start=0, end=1, raw='not consumed'),
            [dict(start=0, end=1, raw='actual')], dict(sources=[]))

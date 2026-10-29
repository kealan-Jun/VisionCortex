"""Do not turn a model's idle claim into a confirmed idle interval or action."""
from ._support import research_helpers
import sys

import pytest

sys.path.insert(0, str(research_helpers()))
from plan_measured_coarse_readouts import classify_readout
from visioncortex.lab.coarse_fine_research import Hint, Packet, Policy, Span, build_plan


@pytest.mark.parametrize('text,expected', [
    ('未见新动作。', ('unknown', False, True)),
    ('无法确定，动作未知', ('unknown', True, False)),
    ('未见新动作，但手移向天平', ('action', False, False)),
])
def test_real_readout_classification_preserves_uncertainty(text, expected):
    kind, uncertain, no_new_action = classify_readout(text)
    assert (kind, uncertain, no_new_action) == expected
    packet = Packet('p', 'a' * 64, 'camera', 'source', '2026-10-21T12:59:00+08:00',
                    60000000, 'filename_label_not_independently_calibrated', None)
    hint = Hint('h', Span('p', 'a' * 64, 0, 30000000), kind, text, 'pinned/1',
                'b' * 64, uncertain=uncertain)
    plan = build_plan([packet], [hint], policy=Policy(
        audit_period_us=60000000, audit_length_us=2000000, audit_seed='fixed'))
    action_tasks = [t for t in plan.tasks if t.hint_ids]
    assert bool(action_tasks) == (not no_new_action)
    if uncertain:
        assert action_tasks[0].trigger_kinds == ('uncertain',)
    assert any('blind_audit' in t.trigger_kinds for t in plan.tasks)
    assert all(c['raw_evidence_retained'] and c['semantic_completeness'] == 'NOT_PROVEN'
               and c['clock_uncertainty_us'] is None for c in plan.coverage)

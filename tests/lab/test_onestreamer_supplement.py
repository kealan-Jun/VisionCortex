"""Endpoint/control questions are separate from the unchanged primary44."""
from ._support import research_helpers
import importlib.util
import json
import sys

import pytest


SCRIPTS = research_helpers()
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('one_supplement', SCRIPTS/'eval_onestreamer_retention_supplement.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_supplement_preserves_exact_questions_and_separate_modes():
    identities = [f'C_e{i}_near' for i in range(1, 5)]+['B_e3_60m', 'C_e2_60m', 'C_e4_60m']
    questions = dict(queries=[dict(id=identity, question='fixed '+identity, question_time=100,
        evidence_end=99, case=identity[0], event_id=identity.split('_')[0]) for identity in identities])
    original = json.dumps(questions, sort_keys=True)
    values = MODULE.questions_for_supplement(questions)
    assert len(values) == 7
    assert all(v['question_time'] == 7200 for v in values[:4])
    assert [v['question'] for v in values] == [q['question'] for q in questions['queries']]
    assert [v['diagnostic_mode'] for v in values[4:]] == ['native_recent_only_empty_memory']*3
    assert json.dumps(questions, sort_keys=True) == original


def cache_fixture(tmp_path, cold=False, partial=False):
    main, legacy = tmp_path/'main', tmp_path/'legacy'
    case = dict(id='C', sources=[dict(offset_seconds=i*900, duration_seconds=900,
        relative=f'{i}.mp4', sha256=str(i)*64) for i in range(8)])
    folder = main/'C'
    folder.mkdir(parents=True)
    (legacy/'C').mkdir(parents=True)
    rows = []
    for t in range(7199 if partial else 7200):
        source = case['sources'][t//900]
        rows.append(dict(start=t, end=t+1, source=source['relative'], source_sha256=source['sha256'],
            sampled_timestamps=[t+i/4 for i in range(4)]))
    mode = 'independent_cold_replay_after_failed_recovery' if cold else 'reused_prefix_plus_validated_extension'
    (folder/'ExecutionMode.json').write_text(json.dumps(dict(mode=mode)))
    (legacy/'C'/'CaptionUpdates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:3600]))
    (folder/'CaptionUpdatesExtension.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in (rows if cold else rows[3600:])))
    return main, legacy, case


@pytest.mark.parametrize('cold', [False, True])
def test_endpoint_loads_all_real_successful_updates_without_future_summary(tmp_path, cold):
    main, legacy, case = cache_fixture(tmp_path, cold)
    rows, receipts = MODULE.load_full_c_prefix(main, legacy, case)
    assert len(rows) == 7200
    assert rows[0]['start'] == 0 and rows[-1]['end'] == 7200
    assert len(receipts) == (1 if cold else 2)


def test_endpoint_rejects_partial_materialization(tmp_path):
    main, legacy, case = cache_fixture(tmp_path, partial=True)
    with pytest.raises(RuntimeError, match='all 7200'):
        MODULE.load_full_c_prefix(main, legacy, case)
